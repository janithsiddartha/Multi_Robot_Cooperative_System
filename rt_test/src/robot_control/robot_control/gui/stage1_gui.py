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
from std_msgs.msg import Float64MultiArray

from robot_control.handover.stage1 import (
    build_stage1_plan,
    generate_interpolated_path,
    load_stage1_robots,
)
from robot_control.kinematics.fk_math import all_frame_positions, fk

class Stage1GUI(Node):
    def __init__(self, root):
        super().__init__('stage1_gui_node')
        self.root = root
        self.root.title("Stage 1 - Handover Execution")
        self.root.geometry("1400x800")
        self.root.configure(bg="#F3F8FC")
        
        self.bg_color = "#F3F8FC"
        self.card_color = "#FFFFFF"
        self.text_color = "#333333"
        self.accent_color = "#4A90E2"
        
        try:
            share_dir = get_package_share_directory('robot_control')
            config_dir = os.path.join(share_dir, 'config')
        except Exception:
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config',
            )

        self.handover_cfg, self.robots, _ = load_stage1_robots(config_dir)
        self.safe_dist = self.handover_cfg.get('handover', {}).get('safe_distance', 50.0)

        self.workspace_points = {
            rid: self._get_workspace_points(rid, cfg)
            for rid, cfg in self.robots.items()
        }
        self.scene_limits = self._compute_scene_limits()

        self.live_q = {rid: np.zeros(cfg.num_joints) for rid, cfg in self.robots.items()}
        self.current_q = {rid: np.zeros(cfg.num_joints) for rid, cfg in self.robots.items()}
        self.is_simulating = False
        
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
        return tuple((float(low[i]), float(high[i])) for i in range(3))

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
        
        tk.Label(left_frame, text="STAGE 1 CONFIG", font=("Helvetica", 16, "bold"), bg=self.bg_color, fg=self.text_color).pack(pady=(0, 20))
        
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
        
        # EXECUTE BUTTON
        btn = tk.Button(left_frame, text="EXECUTE STAGE 1", font=("Helvetica", 12, "bold"), bg=self.accent_color, fg=self.text_color, command=self._on_execute)
        btn.pack(fill="x", pady=10)
        
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
        """Ask which robot to use when a point sits in an intersection."""
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
                bg=self.accent_color, fg=self.text_color,
            ).pack(pady=10)
            self.root.wait_window(top)
            ready.set()

        self.root.after(0, show_dialog)
        ready.wait()
        return chosen[0]

    def _on_execute(self):
        self.log_text.delete(1.0, tk.END)
        threading.Thread(target=self._run_plan_thread, daemon=True).start()

    def _chooser(self, label, robot_ids):
        return self._ask_user_choice(f"Select {label} robot", robot_ids)

    def _run_plan_thread(self):
        self._log("=== STARTING STAGE 1 ===")
        self._log("IK: stage1_ik (XYZ only, same as ik_solver)")

        try:
            pick_xyz = [float(x.strip()) for x in self.pick_entry.get().split(',')]
            place_xyz = [float(x.strip()) for x in self.place_entry.get().split(',')]
        except Exception:
            self._log("ERROR: Invalid Pick/Place Coordinates.")
            return

        current_q = {rid: q.copy() for rid, q in self.live_q.items()}

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

        self._log(f"\nPICK {pick_xyz} is in workspace of: {', '.join(pick_opts)}")
        for rid, sol in pick_opts.items():
            self._log(f"  {rid}: reachable, one XYZ pose selected (error={sol.error_mm:.3f} mm)")
        if len(pick_opts) == 1:
            self._log(f"> Auto-selected PICK: {plan.pick_robot}")
        else:
            self._log(f"> Selected PICK: {plan.pick_robot}")
        self._log(f"  q_deg = {np.round(plan.pick.q_deg, 2)}")

        self._log(f"\nPLACE {place_xyz} is in workspace of: {', '.join(place_opts)}")
        for rid, sol in place_opts.items():
            self._log(f"  {rid}: reachable, one XYZ pose selected (error={sol.error_mm:.3f} mm)")
        if len(place_opts) == 1:
            self._log(f"> Auto-selected PLACE: {plan.place_robot}")
        else:
            self._log(f"> Selected PLACE: {plan.place_robot}")
        self._log(f"  q_deg = {np.round(plan.place.q_deg, 2)}")

        pick_rid = plan.pick_robot
        place_rid = plan.place_robot
        pick_q = plan.pick.q_deg
        place_q = plan.place.q_deg

        self._log("\nPreparing motion paths...")
        self.is_simulating = True
        self.current_q = {k: v.copy() for k, v in self.live_q.items()}

        start_pick_q = self.current_q[pick_rid].copy()
        start_place_q = self.current_q[place_rid].copy()
        pick_path = generate_interpolated_path(start_pick_q, pick_q, steps=30)

        if plan.needs_handover:
            pick_h_q = plan.pick_handover_q
            place_h_q = plan.place_handover_q
            self._log(f"\nHANDOVER {pick_rid} <-> {place_rid} (from handover.yaml)")
            self._log(f"  {pick_rid}_q = {pick_h_q}")
            self._log(f"  {place_rid}_q = {place_h_q}")
            pick_handover_path = generate_interpolated_path(pick_q, pick_h_q, steps=30)
            place_handover_path = generate_interpolated_path(start_place_q, place_h_q, steps=30)
            place_path = generate_interpolated_path(place_h_q, place_q, steps=30)
        else:
            self._log("\nSame robot for pick and place — skipping handover.")
            pick_handover_path = None
            place_handover_path = None
            place_path = generate_interpolated_path(pick_q, place_q, steps=30)
        self._log("\n=== EXECUTING SEQUENTIAL PLAN ===")

        def animate_path(rid, path, label, wait_time=0.5):
            self._log(label)
            for q in path:
                self.current_q[rid] = q
                time.sleep(0.05)
            time.sleep(wait_time)

        animate_path(pick_rid, pick_path, f"1. {pick_rid} moves to Pick...", 0.5)
        self._log(f"2. {pick_rid} waits 2s, closes gripper.")
        time.sleep(2)

        if plan.needs_handover:
            animate_path(pick_rid, pick_handover_path, f"3. {pick_rid} moves to handover.", 0.5)
            animate_path(place_rid, place_handover_path, f"4. {place_rid} moves to handover.", 0.5)
            self._log(f"5. Transfer: {place_rid} closes, {pick_rid} opens.")
            time.sleep(1)
            animate_path(place_rid, place_path, f"6. {place_rid} moves to Place...", 0.5)
            self._log(f"7. {place_rid} waits 2s, releases object.")
        else:
            animate_path(pick_rid, place_path, f"3. {pick_rid} moves to Place...", 0.5)
            self._log(f"4. {pick_rid} waits 2s, releases object.")
        time.sleep(2)

        self.current_q[pick_rid] = np.zeros(self.robots[pick_rid].num_joints)
        self.current_q[place_rid] = np.zeros(self.robots[place_rid].num_joints)

        self._log("=== STAGE 1 SUCCESS ===")
        self.is_simulating = False

def main(args=None):
    rclpy.init(args=args)
    root = tk.Tk()
    app = Stage1GUI(root)

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
