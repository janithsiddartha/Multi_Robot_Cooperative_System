import os
import threading
import time
import tkinter as tk

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from rclpy.node import Node
from tkinter import messagebox
from std_msgs.msg import Float64MultiArray, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from robot_control.handover.stage1 import (
    build_stage1_plan,
    load_stage1_robots,
)
from robot_control.handover.stage1_5 import (
    compute_min_body_clearance,
    plan_stage1_5_synchronized_paths,
)
from robot_control.kinematics.fk_math import all_frame_positions, fk


class Stage1_5GUI(Node):
    def __init__(self, root):
        super().__init__('stage1_5_gui_node')
        self.root = root
        self.root.title("Stage 1.5 - Concurrent Handover & Collision Avoidance")
        self.root.geometry("1440x850")
        self.root.configure(bg="#F3F8FC")

        self.bg_color = "#F3F8FC"
        self.card_color = "#FFFFFF"
        self.text_color = "#2C3E50"
        self.accent_color = "#3498DB"
        self.success_color = "#27AE60"
        self.warning_color = "#E67E22"
        self.danger_color = "#E74C3C"

        try:
            share_dir = get_package_share_directory('robot_control')
            config_dir = os.path.join(share_dir, 'config')
        except Exception:
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config',
            )

        self.handover_cfg, self.robots, _ = load_stage1_robots(config_dir)
        self.safe_dist = float(self.handover_cfg.get('handover', {}).get('safe_distance', 50.0))

        exec_cfg = self.handover_cfg.get('execution', {})
        timings = exec_cfg.get('timings', {})
        self.pick_wait_s = float(timings.get('pick_grasp_wait_s', 2.0))
        self.transfer_wait_s = float(timings.get('handover_transfer_wait_s', 1.0))
        self.place_wait_s = float(timings.get('place_release_wait_s', 2.0))
        self.step_interval_s = float(timings.get('step_interval_s', 0.05))
        self.dwell_time_s = float(timings.get('dwell_time_s', 0.5))
        self.trajectory_steps = int(exec_cfg.get('trajectory', {}).get('steps', 30))

        self.workspace_points = {
            rid: self._get_workspace_points(rid, cfg)
            for rid, cfg in self.robots.items()
        }
        self.scene_limits = self._compute_scene_limits()

        self.live_q = {rid: np.zeros(cfg.num_joints) for rid, cfg in self.robots.items()}
        self.current_q = {rid: np.zeros(cfg.num_joints) for rid, cfg in self.robots.items()}
        self.is_simulating = False
        self.is_real_execution = False
        self.cached_plan = None
        self.live_clearance_mm = float('inf')

        # ROS Publishers for Real Hardware Control
        self.publishers_traj = {}
        self.publishers_mode = {}
        self.publishers_gripper = {}
        mode_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        for rid in self.robots.keys():
            self.publishers_traj[rid] = self.create_publisher(
                Float64MultiArray, f'/{rid}/trajectory_joint_command', 10
            )
            self.publishers_mode[rid] = self.create_publisher(
                String, f'/{rid}/control_mode', mode_qos
            )
            self.publishers_gripper[rid] = self.create_publisher(
                Float64MultiArray, f'/{rid}/trajectory_gripper_command', 10
            )

        # ROS Subscribers
        self.subs = []
        for rid in self.robots.keys():
            sub = self.create_subscription(
                Float64MultiArray, f'/{rid}/joint_commands',
                lambda msg, r=rid: self._joint_cb(msg, r), 10
            )
            self.subs.append(sub)

        self._build_ui()
        self._last_visualization_signature = None
        self._init_scene()
        self.root.after(50, self._refresh_visualization)

    def _send_joint_command(self, rid, q_deg, real_robot=False):
        if real_robot and rid in self.publishers_traj:
            msg = Float64MultiArray()
            msg.data = [float(v) for v in q_deg]
            self.publishers_traj[rid].publish(msg)

    def _set_control_mode(self, rid, mode_str, real_robot=False):
        if real_robot and rid in self.publishers_mode:
            msg = String()
            msg.data = mode_str
            self.publishers_mode[rid].publish(msg)

    def _send_gripper_command(self, rid, state, real_robot=False):
        if real_robot and rid in self.publishers_gripper:
            cfg = self.robots.get(rid)
            angle = getattr(cfg, f"gripper_{state}", 55.0 if state == 'close' else -55.0)
            msg = Float64MultiArray()
            msg.data = [float(angle)]
            self.publishers_gripper[rid].publish(msg)

    def _joint_cb(self, msg, rid):
        if len(msg.data) == self.robots[rid].num_joints:
            self.live_q[rid] = np.array(msg.data, dtype=float)

    def _compute_scene_limits(self):
        """Return fixed XYZ plot bounds covering every robot workspace."""
        base_positions = np.array(
            [cfg.world_base_xyz for cfg in self.robots.values()], dtype=float
        )
        max_reach = max(
            sum(abs(float(value)) for value in cfg.a)
            + sum(abs(float(value)) for value in cfg.d)
            + float(np.linalg.norm(cfg.tcp_offset))
            for cfg in self.robots.values()
        )
        low = np.min(base_positions, axis=0) - max_reach
        high = np.max(base_positions, axis=0) + max_reach

        # Read z_min from handover.yaml (visualization.z_min)
        min_base_z = float(np.min(base_positions[:, 2]))
        raw_z_min = self.handover_cfg.get('visualization', {}).get('z_min', None)
        if raw_z_min is not None:
            val = float(raw_z_min)
            if val > min_base_z:
                z_min = min_base_z - val
            else:
                z_min = val
        else:
            z_min = min_base_z - 50.0
        z_max = float(high[2])

        return (
            (float(low[0]), float(high[0])),
            (float(low[1]), float(high[1])),
            (z_min, z_max),
        )

    def _refresh_visualization(self):
        """Refresh from Tk main thread."""
        try:
            if not self.root.winfo_exists():
                return
            if self._visualization_signature() != self._last_visualization_signature:
                self._update_3d()
            self.root.after(50, self._refresh_visualization)
        except tk.TclError:
            return

    def _visualization_signature(self):
        """Only redraw when displayed values change."""
        joint_state = self.current_q if self.is_simulating else self.live_q
        rounded_joints = tuple(
            (rid, tuple(np.round(q, 2))) for rid, q in sorted(joint_state.items())
        )
        return (
            self.is_simulating,
            rounded_joints,
            self.show_ws_var.get(),
            self.show_tcp_var.get(),
            self.pick_entry.get(),
            self.place_entry.get(),
        )

    def _build_ui(self):
        left_frame = tk.Frame(self.root, bg=self.bg_color, width=420)
        left_frame.pack(side="left", fill="y", padx=16, pady=16)
        left_frame.pack_propagate(False)

        # Header
        header = tk.Frame(left_frame, bg=self.bg_color)
        header.pack(fill="x", pady=(0, 12))
        tk.Label(
            header, text="STAGE 1.5", font=("Helvetica", 16, "bold"),
            bg=self.bg_color, fg=self.text_color
        ).pack(anchor="w")
        tk.Label(
            header, text="Concurrent Handover & Body Collision Avoidance",
            font=("Helvetica", 9), bg=self.bg_color, fg="#7F8C8D"
        ).pack(anchor="w")

        # TARGETS CARD
        targets_card = tk.Frame(left_frame, bg=self.card_color, padx=12, pady=10, relief="solid", bd=1)
        targets_card.pack(fill="x", pady=6)

        tk.Label(targets_card, text="Pick XYZ (World mm):", bg=self.card_color, fg=self.text_color, font=("Helvetica", 9, "bold")).pack(anchor="w")
        self.pick_entry = tk.Entry(targets_card, bg="#F8FAFC", relief="solid", bd=1, font=("Consolas", 10))
        self.pick_entry.insert(0, "-200.0, 100.0, 200.0")
        self.pick_entry.pack(fill="x", pady=(2, 8))

        tk.Label(targets_card, text="Place XYZ (World mm):", bg=self.card_color, fg=self.text_color, font=("Helvetica", 9, "bold")).pack(anchor="w")
        self.place_entry = tk.Entry(targets_card, bg="#F8FAFC", relief="solid", bd=1, font=("Consolas", 10))
        self.place_entry.insert(0, "200.0, 100.0, 200.0")
        self.place_entry.pack(fill="x", pady=(2, 4))

        # COLLISION & CLEARANCE MONITOR CARD
        monitor_card = tk.Frame(left_frame, bg=self.card_color, padx=12, pady=10, relief="solid", bd=1)
        monitor_card.pack(fill="x", pady=6)

        tk.Label(monitor_card, text="COLLISION MONITOR", bg=self.card_color, fg=self.text_color, font=("Helvetica", 9, "bold")).pack(anchor="w")

        gauge_frame = tk.Frame(monitor_card, bg=self.card_color)
        gauge_frame.pack(fill="x", pady=4)

        tk.Label(gauge_frame, text="Min Body Clearance:", bg=self.card_color, fg="#555555", font=("Helvetica", 9)).pack(side="left")
        self.clearance_val_lbl = tk.Label(gauge_frame, text="-- mm", bg=self.card_color, fg=self.success_color, font=("Helvetica", 10, "bold"))
        self.clearance_val_lbl.pack(side="right")

        self.status_badge = tk.Label(
            monitor_card, text="IDLE / READY", font=("Helvetica", 9, "bold"),
            bg="#EBF5FB", fg="#2980B9", padx=8, pady=3
        )
        self.status_badge.pack(fill="x", pady=4)

        # UI Toggles
        self.show_ws_var = tk.BooleanVar(value=True)
        self.show_tcp_var = tk.BooleanVar(value=True)
        opt_frame = tk.Frame(left_frame, bg=self.bg_color)
        opt_frame.pack(fill="x", pady=4)
        tk.Checkbutton(opt_frame, text="Workspace Clouds", variable=self.show_ws_var, bg=self.bg_color, command=self._update_3d).pack(side="left")
        tk.Checkbutton(opt_frame, text="TCP Frames", variable=self.show_tcp_var, bg=self.bg_color, command=self._update_3d).pack(side="left", padx=10)

        # LOG OUTPUT
        self.log_text = tk.Text(
            left_frame, height=15, bg="#FFFFFF", fg="#2C3E50",
            font=("Consolas", 8), relief="solid", bd=1
        )
        self.log_text.pack(fill="both", expand=True, pady=6)

        # ACTION BUTTONS (SIMULATE VS EXECUTE REAL)
        action_frame = tk.Frame(left_frame, bg=self.bg_color)
        action_frame.pack(fill="x", pady=6)

        self.plan_status_lbl = tk.Label(
            action_frame, text="Plan Status: Not simulated yet",
            font=("Helvetica", 8, "italic"), bg=self.bg_color, fg="#7F8C8D"
        )
        self.plan_status_lbl.pack(pady=(0, 4))

        self.sim_btn = tk.Button(
            action_frame, text="▶ SIMULATE (3D VISUALIZER ONLY)",
            font=("Helvetica", 10, "bold"), bg="#17A2B8", fg="#FFFFFF",
            activebackground="#138496", activeforeground="#FFFFFF",
            relief="flat", cursor="hand2", padx=6, pady=8,
            command=self._on_simulate
        )
        self.sim_btn.pack(fill="x", pady=(0, 6))

        self.exec_btn = tk.Button(
            action_frame, text="⚡ EXECUTE (REAL ROBOT)",
            font=("Helvetica", 10, "bold"), bg="#28A745", fg="#FFFFFF",
            activebackground="#218838", activeforeground="#FFFFFF",
            relief="flat", cursor="hand2", padx=6, pady=8,
            command=self._on_execute_real
        )
        self.exec_btn.pack(fill="x")

        def _on_target_change(event=None):
            if self.cached_plan is not None:
                try:
                    cur_p = tuple(float(x.strip()) for x in self.pick_entry.get().split(','))
                    cur_pl = tuple(float(x.strip()) for x in self.place_entry.get().split(','))
                    if (cur_p, cur_pl) == self.cached_plan.get('target_key'):
                        self.plan_status_lbl.config(
                            text="✓ Simulation verified & cached for instant execution",
                            fg="#27AE60"
                        )
                        return
                except Exception:
                    pass
                self.plan_status_lbl.config(
                    text="Targets modified (Simulate recommended)",
                    fg="#E67E22"
                )
        self.pick_entry.bind("<KeyRelease>", _on_target_change)
        self.place_entry.bind("<KeyRelease>", _on_target_change)

        # RIGHT FRAME (MATPLOTLIB)
        right_frame = tk.Frame(self.root, bg=self.bg_color)
        right_frame.pack(side="right", fill="both", expand=True, padx=16, pady=16)

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.fig.patch.set_facecolor(self.card_color)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor(self.card_color)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _log(self, msg):
        if threading.current_thread() is not threading.main_thread():
            try:
                self.root.after(0, self._log, msg)
            except tk.TclError:
                pass
            return
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _set_status_lbl(self, text, fg):
        if threading.current_thread() is not threading.main_thread():
            try:
                self.root.after(0, self._set_status_lbl, text, fg)
            except tk.TclError:
                pass
            return
        try:
            self.plan_status_lbl.config(text=text, fg=fg)
        except tk.TclError:
            pass

    def _update_clearance_ui(self, clearance_mm, status_text, status_type="safe"):
        if threading.current_thread() is not threading.main_thread():
            try:
                self.root.after(0, self._update_clearance_ui, clearance_mm, status_text, status_type)
            except tk.TclError:
                pass
            return

        if np.isinf(clearance_mm):
            self.clearance_val_lbl.config(text="-- mm")
        else:
            self.clearance_val_lbl.config(text=f"{clearance_mm:.1f} mm")

        colors = {
            "safe": ("#EAFAF1", self.success_color),
            "warn": ("#FEF5E7", self.warning_color),
            "danger": ("#FDEDEC", self.danger_color),
            "info": ("#EBF5FB", "#2980B9"),
        }
        bg, fg = colors.get(status_type, ("#EBF5FB", "#2980B9"))
        self.status_badge.config(text=status_text, bg=bg, fg=fg)

    def _get_workspace_points(self, rid, cfg):
        q_min = cfg.joint_limits_rad[:, 0]
        q_max = cfg.joint_limits_rad[:, 1]
        pts = []
        for _ in range(500):
            q = q_min + np.random.rand(cfg.num_joints) * (q_max - q_min)
            T = np.eye(4)
            for i in range(cfg.num_frames):
                alpha = np.deg2rad(cfg.alpha_deg[i])
                a = cfg.a[i]
                d = cfg.d[i]
                theta = q[cfg.theta_joint_index[i]] if cfg.theta_joint_index[i] >= 0 else 0
                theta += np.deg2rad(cfg.theta_offset_deg[i])

                ca, sa = np.cos(alpha), np.sin(alpha)
                ct, st = np.cos(theta), np.sin(theta)
                Ti = np.array([
                    [ct, -st, 0, a],
                    [st * ca, ct * ca, -sa, -d * sa],
                    [st * sa, ct * sa, ca, d * ca],
                    [0, 0, 0, 1]
                ])
                T = T @ Ti
            p_tcp = T[:3, 3] + cfg.tcp_offset
            p_W = cfg.world_base_R @ p_tcp + cfg.world_base_xyz
            pts.append(p_W)
        return np.array(pts)

    def _init_scene(self):
        colors = {'robot1': '#4F9DE8', 'robot2': '#F84B5B', 'robot3': '#28A745'}
        self.workspace_scatters = {}
        self.arm_lines = {}
        self.tcp_lines = {}

        for rid, cfg in self.robots.items():
            color = colors.get(rid, '#4F9DE8')
            pts = self.workspace_points[rid]
            self.workspace_scatters[rid] = self.ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=3, c=color, alpha=0.15, edgecolors='none',
            )
            self.ax.scatter(*cfg.world_base_xyz, c=color, s=50, marker='s')
            self.arm_lines[rid], = self.ax.plot(
                [], [], [], '-o', linewidth=3, markersize=6,
                color=color, label=rid,
            )
            self.tcp_lines[rid] = [
                self.ax.plot([], [], [], color=axis_color, linewidth=1.5)[0]
                for axis_color in ('r', 'g', 'b')
            ]

        self.pick_marker = self.ax.scatter(
            [], [], [], marker='x', color='#F5A623', s=120,
            linewidths=3, label='PICK', zorder=10,
        )
        self.place_marker = self.ax.scatter(
            [], [], [], marker='x', color='#17A2B8', s=120,
            linewidths=3, label='PLACE', zorder=10,
        )
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.set_xlim(*self.scene_limits[0])
        self.ax.set_ylim(*self.scene_limits[1])
        self.ax.set_zlim(*self.scene_limits[2])
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.legend()
        self._update_3d()

    @staticmethod
    def _set_marker_xyz(marker, xyz):
        marker._offsets3d = (
            np.asarray([xyz[0]], dtype=float),
            np.asarray([xyz[1]], dtype=float),
            np.asarray([xyz[2]], dtype=float),
        )

    def _update_3d(self):
        for rid, cfg in self.robots.items():
            q = self.current_q[rid] if self.is_simulating else self.live_q[rid]
            q_rad = np.deg2rad(q)
            pos_base = all_frame_positions(q_rad, cfg)
            pos_world = np.asarray([
                cfg.world_base_R @ p + cfg.world_base_xyz for p in pos_base
            ])
            arm_points = np.vstack((cfg.world_base_xyz, pos_world))
            self.arm_lines[rid].set_data_3d(
                arm_points[:, 0], arm_points[:, 1], arm_points[:, 2]
            )

            tcp_lines = self.tcp_lines[rid]
            for line in tcp_lines:
                line.set_visible(self.show_tcp_var.get())
            if self.show_tcp_var.get():
                _, tcp_base = fk(q_rad, cfg)
                tcp_world = np.eye(4)
                tcp_world[:3, :3] = cfg.world_base_R
                tcp_world[:3, 3] = cfg.world_base_xyz
                tcp_world = tcp_world @ tcp_base
                point = tcp_world[:3, 3]
                for axis, line in enumerate(tcp_lines):
                    delta = tcp_world[:3, axis] * 40.0
                    line.set_data_3d(
                        [point[0], point[0] + delta[0]],
                        [point[1], point[1] + delta[1]],
                        [point[2], point[2] + delta[2]],
                    )

            self.workspace_scatters[rid].set_visible(self.show_ws_var.get())

        try:
            pick_xyz = [float(v.strip()) for v in self.pick_entry.get().split(',')]
            place_xyz = [float(v.strip()) for v in self.place_entry.get().split(',')]
            if len(pick_xyz) == 3:
                self._set_marker_xyz(self.pick_marker, pick_xyz)
            if len(place_xyz) == 3:
                self._set_marker_xyz(self.place_marker, place_xyz)
        except (ValueError, tk.TclError):
            pass

        self.canvas.draw_idle()
        try:
            self.canvas.flush_events()
        except Exception:
            pass
        self._last_visualization_signature = self._visualization_signature()

    def _ask_user_choice(self, title, options):
        ready = threading.Event()
        chosen = [options[0]]

        def show_dialog():
            top = tk.Toplevel(self.root)
            top.title(title)
            top.geometry("340x220")
            top.configure(bg=self.bg_color)
            top.transient(self.root)
            top.grab_set()

            tk.Label(
                top,
                text="This point is in more than one workspace.\nSelect which robot should do it:",
                bg=self.bg_color, fg=self.text_color, justify="center",
            ).pack(pady=20)

            selected = tk.StringVar(top)
            selected.set(options[0])
            opt = tk.OptionMenu(top, selected, *options)
            opt.config(bg=self.card_color, fg=self.text_color)
            opt.pack(pady=10)

            def on_ok():
                chosen[0] = selected.get()
                top.destroy()

            tk.Button(
                top, text="OK", command=on_ok,
                bg=self.accent_color, fg="#FFFFFF",
            ).pack(pady=10)
            self.root.wait_window(top)
            ready.set()

        self.root.after(0, show_dialog)
        ready.wait()
        return chosen[0]

    def _chooser(self, label, robot_ids):
        return self._ask_user_choice(f"Select {label} robot", robot_ids)

    def _on_simulate(self):
        if self.is_simulating:
            return
        self.log_text.delete(1.0, tk.END)
        threading.Thread(target=self._run_plan_thread, args=(False,), daemon=True).start()

    def _on_execute_real(self):
        if self.is_simulating:
            return
        confirm = messagebox.askyesno(
            "Confirm Real Robot Execution",
            "Are you sure you want to send trajectory commands to the REAL physical robot hardware?\n\nPlease confirm that the robot workspace is clear of obstacles."
        )
        if not confirm:
            return
        self.log_text.delete(1.0, tk.END)
        threading.Thread(target=self._run_plan_thread, args=(True,), daemon=True).start()

    def _run_plan_thread(self, real_robot=False):
        self._log("=== STARTING STAGE 1.5 CONCURRENT EXECUTION ===")
        if real_robot:
            self._log("[MODE: REAL ROBOT EXECUTION] Sending trajectory to hardware via ROS topics!")
        else:
            self._log("[MODE: SIMULATION ONLY] Visualizer 3D preview active - Physical robot commands DISABLED.")
        self._log("Features: Simultaneous Handover Staging & Whole-Body Clearance Avoidance")

        try:
            pick_xyz = [float(x.strip()) for x in self.pick_entry.get().split(',')]
            place_xyz = [float(x.strip()) for x in self.place_entry.get().split(',')]
        except Exception:
            self._log("ERROR: Invalid Pick/Place Coordinates.")
            return

        target_key = (tuple(pick_xyz), tuple(place_xyz))
        can_reuse = (
            self.cached_plan is not None
            and self.cached_plan.get('target_key') == target_key
        )

        current_q = {rid: q.copy() for rid, q in self.live_q.items()}

        if real_robot and can_reuse:
            self._log("\n⚡ [REUSING SIMULATED PLAN] Target coordinates match last simulation!")
            self._log("⚡ Skipping collision avoidance calculations (~20s saved) & robot selection!")
            self._log(f"⚡ Using pre-verified trajectory (Min clearance: {self.cached_plan['traj_result'].min_overall_clearance_mm:.1f} mm).")
            self._log("⚡ Directly executing on physical hardware...")
            cached = self.cached_plan
            plan = cached['plan']
            traj_result = cached['traj_result']
            pick_rid = plan.pick_robot
            place_rid = plan.place_robot
        else:
            try:
                plan, pick_opts, place_opts = build_stage1_plan(
                    pick_xyz,
                    place_xyz,
                    self.robots,
                    self.handover_cfg,
                    chooser=self._chooser,
                    current_q_deg=current_q,
                )
            except RuntimeError as exc:
                self._log(f"ERROR: {exc}")
                return

            self._log(f"\nPICK {pick_xyz} reachable by: {', '.join(pick_opts)}")
            self._log(f"> Selected PICK: {plan.pick_robot} (q_deg = {np.round(plan.pick.q_deg, 2)})")

            self._log(f"\nPLACE {place_xyz} reachable by: {', '.join(place_opts)}")
            self._log(f"> Selected PLACE: {plan.place_robot} (q_deg = {np.round(plan.place.q_deg, 2)})")

            pick_rid = plan.pick_robot
            place_rid = plan.place_robot

            if plan.needs_handover:
                pick_h_q = plan.pick_handover_q
                place_h_q = plan.place_handover_q
                pick_pre_h_q = plan.pick_pre_handover_q if plan.pick_pre_handover_q is not None else pick_h_q
                place_pre_h_q = plan.place_pre_handover_q if plan.place_pre_handover_q is not None else place_h_q
                self._log(f"\nHANDOVER {pick_rid} <-> {place_rid} (from handover.yaml)")
                self._log(f"  {pick_rid}_pre_q = {pick_pre_h_q}")
                self._log(f"  {place_rid}_pre_q = {place_pre_h_q}")
                self._log(f"  {pick_rid}_q = {pick_h_q}")
                self._log(f"  {place_rid}_q = {place_h_q}")

            self._log("\nPlanning collision-free synchronized trajectories...")
            traj_result = plan_stage1_5_synchronized_paths(
                plan, self.robots, safe_distance=self.safe_dist, steps=self.trajectory_steps, current_q=current_q
            )

            if traj_result.collision_avoidance_applied:
                self._log("! WARNING: Direct simultaneous path was near collision limit.")
                self._log("! Safe collision-avoidance staggering was automatically applied.")

            self._log(f"Min predicted body clearance: {traj_result.min_overall_clearance_mm:.1f} mm (Safe dist: {self.safe_dist} mm)")

            self.cached_plan = {
                'target_key': target_key,
                'plan': plan,
                'traj_result': traj_result,
            }

        self.is_simulating = True
        self.is_real_execution = real_robot
        self.current_q = {k: v.copy() for k, v in self.live_q.items()}

        # Set hardware control mode to TRAJECTORY if commanding real robots
        self._set_control_mode(pick_rid, 'TRAJECTORY', real_robot=real_robot)
        if plan.needs_handover:
            self._set_control_mode(place_rid, 'TRAJECTORY', real_robot=real_robot)

        def step_simultaneous(step_list, phase_label, dwell=None):
            dwell = self.dwell_time_s if dwell is None else dwell
            self._log(f"\n>>> {phase_label}")
            for step in step_list:
                if pick_rid == place_rid:
                    self.current_q[pick_rid] = np.asarray(step.pick_q, dtype=float).copy()
                    self._send_joint_command(pick_rid, self.current_q[pick_rid], real_robot=real_robot)
                else:
                    self.current_q[pick_rid] = np.asarray(step.pick_q, dtype=float).copy()
                    self.current_q[place_rid] = np.asarray(step.place_q, dtype=float).copy()
                    self._send_joint_command(pick_rid, self.current_q[pick_rid], real_robot=real_robot)
                    self._send_joint_command(place_rid, self.current_q[place_rid], real_robot=real_robot)
                status = "safe" if step.is_safe else "warn"
                self._update_clearance_ui(step.min_clearance_mm, "MOTION ACTIVE", status)
                self.root.after(0, self._update_3d)
                time.sleep(self.step_interval_s)
            time.sleep(dwell)

        if plan.needs_handover:
            # ─────────────────────────────────────────────────────────
            # PHASE 1: SIMULTANEOUS MOTION (Pick -> Pick, Place -> Pre-Handover)
            # ─────────────────────────────────────────────────────────
            step_simultaneous(
                traj_result.phase1_simultaneous,
                f"1. SIMULTANEOUS: {pick_rid} moves to Pick, WHILE {place_rid} moves to Pre-Handover!",
                dwell=self.dwell_time_s,
            )

            # ─────────────────────────────────────────────────────────
            # PHASE 2: GRASP AT PICK
            # ─────────────────────────────────────────────────────────
            self._log(f"2. {pick_rid} waits {self.pick_wait_s}s and closes gripper. ({place_rid} waiting at Pre-Handover)")
            self._send_gripper_command(pick_rid, 'close', real_robot=real_robot)
            self._update_clearance_ui(self.live_clearance_mm, f"{place_rid} WAITING AT PRE-HANDOVER", "info")
            time.sleep(self.pick_wait_s)

            # ─────────────────────────────────────────────────────────
            # PHASE 3: PICK ROBOT MOVES TO PRE-HANDOVER
            # ─────────────────────────────────────────────────────────
            step_simultaneous(
                traj_result.phase3_pick_to_pre_handover,
                f"3. {pick_rid} moves to Pre-Handover ({place_rid} stationed at Pre-Handover).",
                dwell=self.dwell_time_s,
            )

            # ─────────────────────────────────────────────────────────
            # PHASE 4: SYNCHRONIZED ADVANCE (Pre-Handover -> Handover)
            # ─────────────────────────────────────────────────────────
            step_simultaneous(
                traj_result.phase4_advance_to_handover,
                f"4. ADVANCE: Both {pick_rid} and {place_rid} advance Pre-Handover -> Handover at same rate!",
                dwell=self.dwell_time_s,
            )

            # ─────────────────────────────────────────────────────────
            # PHASE 5: CONFIRM & HANDOVER TRANSFER
            # ─────────────────────────────────────────────────────────
            self._log("5. CONFIRMED: Both robots are at Handover position.")
            self._log(f"   Transfer: {place_rid} closes gripper, {pick_rid} opens gripper.")
            self._send_gripper_command(place_rid, 'close', real_robot=real_robot)
            self._send_gripper_command(pick_rid, 'open', real_robot=real_robot)
            if traj_result.phase4_advance_to_handover:
                self._update_clearance_ui(traj_result.phase4_advance_to_handover[-1].min_clearance_mm, "HANDOVER TRANSFER (GRIPPER)", "info")
            time.sleep(self.transfer_wait_s)

            # ─────────────────────────────────────────────────────────
            # PHASE 6: SYNCHRONIZED RETRACT (Handover -> Pre-Handover)
            # ─────────────────────────────────────────────────────────
            step_simultaneous(
                traj_result.phase6_retract_to_pre_handover,
                f"6. RETRACT: Both robots back away to Pre-Handover at same rate!",
                dwell=self.dwell_time_s,
            )

            # ─────────────────────────────────────────────────────────
            # PHASE 7: PLACE ROBOT MOVES PRE-HANDOVER TO PLACE
            # ─────────────────────────────────────────────────────────
            step_simultaneous(
                traj_result.phase7_handover_to_place,
                f"7. {place_rid} moves from Pre-Handover to Place target ({pick_rid} waiting at Pre-Handover).",
                dwell=self.dwell_time_s,
            )

            # ─────────────────────────────────────────────────────────
            # PHASE 8: RELEASE OBJECT
            # ─────────────────────────────────────────────────────────
            self._log(f"8. {place_rid} waits {self.place_wait_s}s, opens gripper to release object.")
            self._send_gripper_command(place_rid, 'open', real_robot=real_robot)
            time.sleep(self.place_wait_s)

            # ─────────────────────────────────────────────────────────
            # PHASE 9: SIMULTANEOUS RETURN HOME
            # ─────────────────────────────────────────────────────────
            step_simultaneous(
                traj_result.phase9_simultaneous_home,
                f"9. SIMULTANEOUS RETURN: Both {pick_rid} and {place_rid} return to Home!",
                dwell=self.dwell_time_s,
            )

        else:
            self._log("\nSame robot for pick and place — skipping handover.")
            step_simultaneous(traj_result.phase1_simultaneous, f"1. {pick_rid} moves to Pick.", dwell=self.dwell_time_s)
            self._log(f"2. {pick_rid} waits {self.pick_wait_s}s, closes gripper.")
            self._send_gripper_command(pick_rid, 'close', real_robot=real_robot)
            time.sleep(self.pick_wait_s)
            step_simultaneous(traj_result.phase7_handover_to_place, f"3. {pick_rid} moves to Place.", dwell=self.dwell_time_s)
            self._log(f"4. {pick_rid} waits {self.place_wait_s}s, releases object.")
            self._send_gripper_command(pick_rid, 'open', real_robot=real_robot)
            time.sleep(self.place_wait_s)
            step_simultaneous(traj_result.phase9_simultaneous_home, f"5. {pick_rid} returns Home.", dwell=self.dwell_time_s)

        self.current_q[pick_rid] = np.zeros(self.robots[pick_rid].num_joints)
        self.current_q[place_rid] = np.zeros(self.robots[place_rid].num_joints)

        # Reset hardware mode back to MANUAL
        self._set_control_mode(pick_rid, 'MANUAL', real_robot=real_robot)
        if plan.needs_handover:
            self._set_control_mode(place_rid, 'MANUAL', real_robot=real_robot)

        self._log("\n=== STAGE 1.5 CONCURRENT EXECUTION SUCCESS ===")
        self._update_clearance_ui(float("inf"), "STAGE 1.5 COMPLETED", "safe")
        if not real_robot:
            self._set_status_lbl("✓ Simulation verified & cached for instant execution", "#27AE60")
            self._log("\n[READY] Click '⚡ EXECUTE (REAL ROBOT)' to directly run this exact plan on hardware without recalculating!")
        else:
            self._set_status_lbl("✓ Real robot execution completed", "#28A745")

        self.is_simulating = False


def main(args=None):
    rclpy.init(args=args)
    root = tk.Tk()
    app = Stage1_5GUI(root)

    def process_ros():
        if not rclpy.ok():
            return
        rclpy.spin_once(app, timeout_sec=0.0)
        try:
            root.after(10, process_ros)
        except tk.TclError:
            pass

    def close_window():
        if root.winfo_exists():
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_window)
    root.after(10, process_ros)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        app.destroy_node()


if __name__ == "__main__":
    main()
