import os
import sys
import yaml
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from ament_index_python.packages import get_package_share_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handover.task_ik import solve_task_ik, TaskIKCandidate
from handover.geometry import euler_to_matrix
from handover.collision import CollisionChecker
from kinematics.fk_math import all_frame_positions, fk

class MockKinematicsConfig:
    def __init__(self, yaml_data, world_base_xyz, world_base_rpy_deg):
        params = yaml_data['/**']['ros__parameters']
        kin = params['kinematics']
        self.alpha_deg = kin['dh']['alpha_deg']
        self.a = kin['dh']['a']
        self.d = kin['dh']['d']
        self.theta_offset_deg = kin['dh']['theta_offset_deg']
        self.theta_joint_index = kin['dh']['theta_joint_index']
        self.theta_joint_multiplier = kin['dh']['theta_joint_multiplier']
        self.tcp_offset = np.array(kin['tcp']['offset'], dtype=float)
        limits_min = kin['joints']['limits_deg_min']
        limits_max = kin['joints']['limits_deg_max']
        self.home_deg = kin['joints']['home_deg']
        self.joint_limits_deg = np.array(list(zip(limits_min, limits_max)))
        self.joint_limits_rad = np.deg2rad(self.joint_limits_deg)
        self.num_frames = len(self.alpha_deg)
        self.num_joints = len(self.home_deg)
        gripper = params.get('gripper', {})
        self.gripper_open = float(gripper.get('open_angle', -55.0))
        self.gripper_close = float(gripper.get('close_angle', 55.0))
        self.world_base_xyz = np.array(world_base_xyz)
        self.world_base_R = euler_to_matrix(np.deg2rad(world_base_rpy_deg))

def transform_world_to_base(p_W, config):
    return config.world_base_R.T @ (np.array(p_W) - config.world_base_xyz)

class Stage2GUI(Node):
    def __init__(self, root):
        super().__init__('stage2_gui_node')
        self.root = root
        self.root.title("Stage 2 - Task IK Handover Execution")
        self.root.geometry("1400x800")
        self.root.configure(bg="#F3F8FC")
        
        self.bg_color = "#F3F8FC"
        self.card_color = "#FFFFFF"
        self.text_color = "#333333"
        self.accent_color = "#4A90E2"
        
        # Load configs
        try:
            share_dir = get_package_share_directory('robot_control')
            config_dir = os.path.join(share_dir, 'config')
        except Exception:
            # Fallback for direct script execution without ROS sourcing
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_dir = os.path.join(base_dir, 'config')
            
        with open(os.path.join(config_dir, 'handover.yaml'), 'r') as f:
            self.handover_cfg = yaml.safe_load(f)
            
        self.safe_dist = self.handover_cfg.get('handover', {}).get('safe_distance', 50.0)

        exec_cfg = self.handover_cfg.get('execution', {})
        timings = exec_cfg.get('timings', {})
        self.pick_wait_s = float(timings.get('pick_grasp_wait_s', 2.0))
        self.transfer_wait_s = float(timings.get('handover_transfer_wait_s', 1.0))
        self.place_wait_s = float(timings.get('place_release_wait_s', 2.0))
        self.step_interval_s = float(timings.get('step_interval_s', 0.05))
        self.dwell_time_s = float(timings.get('dwell_time_s', 0.5))
        self.trajectory_steps = int(exec_cfg.get('trajectory', {}).get('steps', 30))

        self.robots = {}
        for rid in ['robot1', 'robot2', 'robot3']:
            r_file = os.path.join(config_dir, f'{rid}.yaml')
            if os.path.exists(r_file):
                with open(r_file, 'r') as f:
                    y_data = yaml.safe_load(f)
                r_base = self.handover_cfg['robots'][rid]
                self.robots[rid] = MockKinematicsConfig(
                    y_data, r_base['world_base_xyz'], r_base['world_base_rpy_deg']
                )

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
                lambda msg, r=rid: self._joint_cb(msg, r), 10)
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

        # Read z_min from handover.yaml (visualization.z_min), defaulting to base_z - 50.0.
        # Supports values like -50.0, 0.0, or 50 (margin below base).
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
        """Refresh from Tk's main thread, like the robot test GUIs."""
        try:
            if not self.root.winfo_exists():
                return
            if self._visualization_signature() != self._last_visualization_signature:
                self._update_3d()
            self.root.after(50, self._refresh_visualization)
        except tk.TclError:
            return

    def _visualization_signature(self):
        """Only redraw when a displayed value has changed."""
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
        left_frame = tk.Frame(self.root, bg=self.bg_color, width=400)
        left_frame.pack(side="left", fill="y", padx=20, pady=20)
        left_frame.pack_propagate(False)
        
        tk.Label(left_frame, text="STAGE 2 CONFIG", font=("Helvetica", 16, "bold"), bg=self.bg_color, fg=self.text_color).pack(pady=(0, 20))
        
        # PICK TARGET
        pick_frame = tk.Frame(left_frame, bg=self.card_color, padx=10, pady=10)
        pick_frame.pack(fill="x", pady=10)
        tk.Label(pick_frame, text="Pick XYZ (World):", bg=self.card_color, fg=self.text_color).pack(anchor="w")
        self.pick_entry = tk.Entry(pick_frame, bg="#FFFFFF")
        self.pick_entry.insert(0, "100.0, 100.0, 200.0")
        self.pick_entry.pack(fill="x", pady=5)
        
        # PLACE TARGET
        place_frame = tk.Frame(left_frame, bg=self.card_color, padx=10, pady=10)
        place_frame.pack(fill="x", pady=10)
        tk.Label(place_frame, text="Place XYZ (World):", bg=self.card_color, fg=self.text_color).pack(anchor="w")
        self.place_entry = tk.Entry(place_frame, bg="#FFFFFF")
        self.place_entry.insert(0, "0.0, 150.0, 150.0")
        self.place_entry.pack(fill="x", pady=5)
        
        # UI Toggles
        self.show_ws_var = tk.BooleanVar(value=True)
        self.show_tcp_var = tk.BooleanVar(value=True)
        opt_frame = tk.Frame(left_frame, bg=self.bg_color)
        opt_frame.pack(fill="x", pady=5)
        tk.Checkbutton(opt_frame, text="Workspace", variable=self.show_ws_var, bg=self.bg_color, command=self._update_3d).pack(side="left")
        tk.Checkbutton(opt_frame, text="TCP Frames", variable=self.show_tcp_var, bg=self.bg_color, command=self._update_3d).pack(side="left")
        
        # LOG OUTPUT
        self.log_text = tk.Text(left_frame, height=20, bg="#FFFFFF", fg="#333333", font=("Consolas", 9), relief="solid", bd=1)
        self.log_text.pack(fill="both", expand=True, pady=10)
        
        # ACTION BUTTONS (SIMULATE VS EXECUTE REAL)
        action_frame = tk.Frame(left_frame, bg=self.bg_color)
        action_frame.pack(fill="x", pady=8)

        self.plan_status_lbl = tk.Label(
            action_frame, text="Plan Status: Not simulated yet",
            font=("Helvetica", 8, "italic"), bg=self.bg_color, fg="#7F8C8D"
        )
        self.plan_status_lbl.pack(pady=(0, 6))

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
        right_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        
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

    def _get_workspace_points(self, rid, cfg):
        # Sparse approximation of workspace
        q_min = cfg.joint_limits_rad[:, 0]
        q_max = cfg.joint_limits_rad[:, 1]
        pts = []
        for _ in range(500):
            q = q_min + np.random.rand(cfg.num_joints) * (q_max - q_min)
            # pseudo FK
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
                    [st*ca, ct*ca, -sa, -d*sa],
                    [st*sa, ct*sa, ca, d*ca],
                    [0, 0, 0, 1]
                ])
                T = T @ Ti
            p_tcp = T[:3, 3] + cfg.tcp_offset
            # Transform to world
            p_W = cfg.world_base_R @ p_tcp + cfg.world_base_xyz
            pts.append(p_W)
        return np.array(pts)

    def _init_scene(self):
        """Create the static scene once and retain artists for live updates."""
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
        """Update robot artists without clearing or rebuilding the workspace."""
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
        self._last_visualization_signature = self._visualization_signature()

    def _ask_user_choice(self, title, options):
        """Create the robot chooser on Tk's main thread."""
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
                bg=self.bg_color,
                fg=self.text_color,
                justify="center",
            ).pack(pady=20)

            selected = tk.StringVar(top, value=options[0])
            tk.OptionMenu(top, selected, *options).pack(pady=10)

            def on_ok():
                chosen[0] = selected.get()
                top.destroy()

            tk.Button(
                top, text="OK", command=on_ok,
                bg=self.accent_color, fg=self.text_color,
            ).pack(pady=10)
            self.root.wait_window(top)
            ready.set()

        self.root.after(0, show_dialog)
        ready.wait()
        return chosen[0]

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
        self._log("=== STARTING STAGE 2 ===")
        if real_robot:
            self._log("[MODE: REAL ROBOT EXECUTION] Sending trajectory to hardware via ROS topics!")
        else:
            self._log("[MODE: SIMULATION ONLY] Visualizer 3D preview active - Physical robot commands DISABLED.")
        
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
            self._log("⚡ Skipping Task-IK orientation search & robot selection dialog.")
            self._log("⚡ Directly executing pre-computed verified trajectory on physical hardware...")
            cached = self.cached_plan
            pick_rid = cached['pick_rid']
            place_rid = cached['place_rid']
            pick_cand = cached['pick_cand']
            place_cand = cached['place_cand']
            needs_handover = cached['needs_handover']
            pick_path = cached['pick_path']
            place_path = cached['place_path']
            pick_home_path = cached['pick_home_path']
            place_home_path = cached.get('place_home_path')
            if needs_handover:
                pick_to_pre_path = cached['pick_to_pre_path']
                place_to_pre_path = cached['place_to_pre_path']
                advance_pick = cached['advance_pick']
                advance_place = cached['advance_place']
                retract_pick = cached['retract_pick']
                retract_place = cached['retract_place']
        else:
            self._log(f"Evaluating PICK at {pick_xyz}")
            pick_opts = {}
            for rid, cfg in self.robots.items():
                cands = solve_task_ik(rid, transform_world_to_base(pick_xyz, cfg), cfg)
                if cands:
                    pick_opts[rid] = cands[0]
                    self._log(f"  {rid} can pick ({len(cands)} solutions)")
                    
            if not pick_opts:
                self._log("ERROR: No robot can reach Pick XYZ.")
                return
                
            pick_rid = list(pick_opts.keys())[0]
            if len(pick_opts) > 1:
                pick_rid = self._ask_user_choice("Select Pick Robot", list(pick_opts.keys()))
            pick_cand = pick_opts[pick_rid]
            pick_q_deg = np.rad2deg(pick_cand.joints)
            self._log(f"> Selected PICK: {pick_rid} (Approach: {pick_cand.approach_direction}, error = {pick_cand.position_error:.3f} mm)")
            self._log(f"  q_deg = {np.round(pick_q_deg, 2)}")

            self._log(f"\nEvaluating PLACE at {place_xyz}")
            place_opts = {}
            for rid, cfg in self.robots.items():
                cands = solve_task_ik(rid, transform_world_to_base(place_xyz, cfg), cfg)
                if cands:
                    place_opts[rid] = cands[0]
                    self._log(f"  {rid} can place ({len(cands)} solutions)")

            if not place_opts:
                self._log("ERROR: No robot can reach Place XYZ.")
                return

            place_rid = list(place_opts.keys())[0]
            if len(place_opts) > 1:
                place_rid = self._ask_user_choice("Select Place Robot", list(place_opts.keys()))
            place_cand = place_opts[place_rid]
            place_q_deg = np.rad2deg(place_cand.joints)
            self._log(f"> Selected PLACE: {place_rid} (Approach: {place_cand.approach_direction}, error = {place_cand.position_error:.3f} mm)")
            self._log(f"  q_deg = {np.round(place_q_deg, 2)}")

            needs_handover = pick_rid != place_rid

            if needs_handover:
                # Get Manual Handover Configs
                pair_key_1 = f"{pick_rid}_{place_rid}"
                pair_key_2 = f"{place_rid}_{pick_rid}"
                h_cfg = self.handover_cfg.get('manual_handover', {}).get(pair_key_1)
                if not h_cfg:
                    h_cfg = self.handover_cfg.get('manual_handover', {}).get(pair_key_2)

                if not h_cfg:
                    self._log(f"ERROR: No manual handover defined for {pick_rid} <-> {place_rid} in handover.yaml")
                    return

                pick_h_q = np.array(h_cfg[f"{pick_rid}_q"], dtype=float)
                place_h_q = np.array(h_cfg[f"{place_rid}_q"], dtype=float)
                pick_pre_h_q = np.array(h_cfg.get(f"{pick_rid}_pre_q", pick_h_q), dtype=float)
                place_pre_h_q = np.array(h_cfg.get(f"{place_rid}_pre_q", place_h_q), dtype=float)

                self._log(f"\nHANDOVER {pick_rid} <-> {place_rid} (from handover.yaml)")
                self._log(f"  {pick_rid}_pre_q = {pick_pre_h_q}")
                self._log(f"  {place_rid}_pre_q = {place_pre_h_q}")
                self._log(f"  {pick_rid}_q = {pick_h_q}")
                self._log(f"  {place_rid}_q = {place_h_q}")

            self._log(f"\nPreparing motion paths...")
            start_pick_q = current_q[pick_rid].copy()
            start_place_q = current_q[place_rid].copy()

            pick_path = [start_pick_q + s * (pick_q_deg - start_pick_q) for s in np.linspace(0, 1, 30)]

            if needs_handover:
                pick_to_pre_path = [pick_q_deg + s * (pick_pre_h_q - pick_q_deg) for s in np.linspace(0, 1, 25)]
                place_to_pre_path = [start_place_q + s * (place_pre_h_q - start_place_q) for s in np.linspace(0, 1, 25)]
                advance_pick = [pick_pre_h_q + s * (pick_h_q - pick_pre_h_q) for s in np.linspace(0, 1, 20)]
                advance_place = [place_pre_h_q + s * (place_h_q - place_pre_h_q) for s in np.linspace(0, 1, 20)]
                retract_pick = [pick_h_q + s * (pick_pre_h_q - pick_h_q) for s in np.linspace(0, 1, 20)]
                retract_place = [place_h_q + s * (place_pre_h_q - place_h_q) for s in np.linspace(0, 1, 20)]
                place_path = [place_pre_h_q + s * (place_q_deg - place_pre_h_q) for s in np.linspace(0, 1, 30)]
                pick_home_path = [pick_pre_h_q + s * (np.zeros(self.robots[pick_rid].num_joints) - pick_pre_h_q) for s in np.linspace(0, 1, 20)]
                place_home_path = [place_q_deg + s * (np.zeros(self.robots[place_rid].num_joints) - place_q_deg) for s in np.linspace(0, 1, 20)]

                self.cached_plan = {
                    'target_key': target_key,
                    'pick_rid': pick_rid,
                    'place_rid': place_rid,
                    'pick_cand': pick_cand,
                    'place_cand': place_cand,
                    'needs_handover': needs_handover,
                    'pick_path': pick_path,
                    'pick_to_pre_path': pick_to_pre_path,
                    'place_to_pre_path': place_to_pre_path,
                    'advance_pick': advance_pick,
                    'advance_place': advance_place,
                    'retract_pick': retract_pick,
                    'retract_place': retract_place,
                    'place_path': place_path,
                    'pick_home_path': pick_home_path,
                    'place_home_path': place_home_path,
                }
            else:
                self._log("\nSame robot for pick and place — skipping handover.")
                place_path = [pick_q_deg + s * (place_q_deg - pick_q_deg) for s in np.linspace(0, 1, 30)]
                pick_home_path = [place_q_deg + s * (np.zeros(self.robots[pick_rid].num_joints) - place_q_deg) for s in np.linspace(0, 1, 20)]
                place_home_path = None

                self.cached_plan = {
                    'target_key': target_key,
                    'pick_rid': pick_rid,
                    'place_rid': place_rid,
                    'pick_cand': pick_cand,
                    'place_cand': place_cand,
                    'needs_handover': needs_handover,
                    'pick_path': pick_path,
                    'place_path': place_path,
                    'pick_home_path': pick_home_path,
                    'place_home_path': None,
                }

        self.is_simulating = True
        self.is_real_execution = real_robot
        self.current_q = {k: v.copy() for k, v in self.live_q.items()}

        # Set hardware control mode to TRAJECTORY if commanding real robots
        self._set_control_mode(pick_rid, 'TRAJECTORY', real_robot=real_robot)
        if needs_handover:
            self._set_control_mode(place_rid, 'TRAJECTORY', real_robot=real_robot)

        self._log("\n=== EXECUTING PLAN ===")

        def animate_path(rid, path, label, wait_time=None):
            wait_time = self.dwell_time_s if wait_time is None else wait_time
            self._log(label)
            for q in path:
                self.current_q[rid] = q
                self._send_joint_command(rid, q, real_robot=real_robot)
                self.root.after(0, self._update_3d)
                time.sleep(self.step_interval_s)
            time.sleep(wait_time)

        def animate_sync_path(path_a, path_b, label, wait_time=None):
            wait_time = self.dwell_time_s if wait_time is None else wait_time
            self._log(label)
            for qa, qb in zip(path_a, path_b):
                self.current_q[pick_rid] = qa
                self.current_q[place_rid] = qb
                self._send_joint_command(pick_rid, qa, real_robot=real_robot)
                self._send_joint_command(place_rid, qb, real_robot=real_robot)
                self.root.after(0, self._update_3d)
                time.sleep(self.step_interval_s)
            time.sleep(wait_time)

        animate_path(pick_rid, pick_path, f"1. {pick_rid} moves to Pick...", self.dwell_time_s)
        self._log(f"2. {pick_rid} waits {self.pick_wait_s}s, closes gripper.")
        self._send_gripper_command(pick_rid, 'close', real_robot=real_robot)
        time.sleep(self.pick_wait_s)

        if needs_handover:
            animate_path(pick_rid, pick_to_pre_path, f"3. {pick_rid} moves to Pre-Handover.", self.dwell_time_s)
            animate_path(place_rid, place_to_pre_path, f"4. {place_rid} moves to Pre-Handover.", self.dwell_time_s)
            animate_sync_path(
                advance_pick, advance_place,
                f"5. Advance: Both robots move Pre-Handover -> Handover at same rate.", self.dwell_time_s
            )
            self._log(f"6. Transfer: {place_rid} closes gripper, {pick_rid} opens gripper.")
            self._send_gripper_command(place_rid, 'close', real_robot=real_robot)
            self._send_gripper_command(pick_rid, 'open', real_robot=real_robot)
            time.sleep(self.transfer_wait_s)
            animate_sync_path(
                retract_pick, retract_place,
                f"7. Retract: Both robots back away to Pre-Handover at same rate.", self.dwell_time_s
            )
            animate_path(place_rid, place_path, f"8. {place_rid} moves to Place...", self.dwell_time_s)
            self._log(f"9. {place_rid} waits {self.place_wait_s}s, releases object.")
            self._send_gripper_command(place_rid, 'open', real_robot=real_robot)
            time.sleep(self.place_wait_s)
            animate_path(pick_rid, pick_home_path, f"10. {pick_rid} returns Home.", 0.2)
            animate_path(place_rid, place_home_path, f"11. {place_rid} returns Home.", self.dwell_time_s)
        else:
            animate_path(pick_rid, place_path, f"3. {pick_rid} moves to Place...", self.dwell_time_s)
            self._log(f"4. {pick_rid} waits {self.place_wait_s}s, releases object.")
            self._send_gripper_command(pick_rid, 'open', real_robot=real_robot)
            time.sleep(self.place_wait_s)
            animate_path(pick_rid, pick_home_path, f"5. {pick_rid} returns Home.", self.dwell_time_s)

        # Reset to home
        self.current_q[pick_rid] = np.zeros(self.robots[pick_rid].num_joints)
        self.current_q[place_rid] = np.zeros(self.robots[place_rid].num_joints)

        # Reset hardware mode back to MANUAL
        self._set_control_mode(pick_rid, 'MANUAL', real_robot=real_robot)
        if needs_handover:
            self._set_control_mode(place_rid, 'MANUAL', real_robot=real_robot)

        self._log("=== STAGE 2 SUCCESS ===")
        if not real_robot:
            self._set_status_lbl("✓ Simulation verified & cached for instant execution", "#27AE60")
            self._log("\n[READY] Click '⚡ EXECUTE (REAL ROBOT)' to directly run this exact plan on hardware without recalculating!")
        else:
            self._set_status_lbl("✓ Real robot execution completed", "#28A745")

        self.is_simulating = False

def main(args=None):
    rclpy.init(args=args)
    root = tk.Tk()
    app = Stage2GUI(root)

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
