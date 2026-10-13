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
from std_msgs.msg import Float64MultiArray
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
        self.world_base_xyz = np.array(world_base_xyz)
        self.world_base_R = euler_to_matrix(np.deg2rad(world_base_rpy_deg))

def transform_world_to_base(p_W, config):
    return config.world_base_R.T @ (np.array(p_W) - config.world_base_xyz)

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
        self.timer = self.create_timer(0.05, self._update_3d)
        
    def _joint_cb(self, msg, rid):
        if len(msg.data) == self.robots[rid].num_joints:
            self.live_q[rid] = np.array(msg.data, dtype=float)
        
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
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update()

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

    def _update_3d(self):
        self.ax.clear()
        
        # Colors for workspace and robots
        colors = {'robot1': '#4F9DE8', 'robot2': '#F84B5B', 'robot3': '#28A745'}
        
        # Draw Workspace (Sparse)
        for rid, cfg in self.robots.items():
            if self.show_ws_var.get():
                pts = self._get_workspace_points(rid, cfg)
                self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, c=colors[rid], alpha=0.15, edgecolors='none')
                
            self.ax.scatter(*cfg.world_base_xyz, c=colors[rid], s=50, marker='s')
            
            # Draw actual arm
            q = self.current_q[rid] if self.is_simulating else self.live_q[rid]
            q_rad = np.deg2rad(q)
            pos_B = all_frame_positions(q_rad, cfg)
            
            # Transform to World
            pos_W = [cfg.world_base_R @ p + np.array(cfg.world_base_xyz) for p in pos_B]
            
            # Add base frame
            arm_pts = [cfg.world_base_xyz] + pos_W
            arm_pts = np.array(arm_pts)
            
            self.ax.plot(arm_pts[:, 0], arm_pts[:, 1], arm_pts[:, 2],
                         '-o', linewidth=3, markersize=6,
                         color=colors[rid], label=rid)
            
            # Draw TCP Frame
            if self.show_tcp_var.get():
                try:
                    _, T_tcp_B = fk(q_rad, cfg) # Full 4x4 transform
                    T_W_B = np.eye(4)
                    T_W_B[:3, :3] = cfg.world_base_R
                    T_W_B[:3, 3] = cfg.world_base_xyz
                    T_tcp_W = T_W_B @ T_tcp_B
                    
                    p = T_tcp_W[:3, 3]
                    for ax_idx, c in enumerate(['r', 'g', 'b']):
                        d = T_tcp_W[:3, ax_idx] * 40
                        self.ax.plot(
                            [p[0], p[0]+d[0]],
                            [p[1], p[1]+d[1]],
                            [p[2], p[2]+d[2]],
                            color=c, linewidth=1.5)
                except Exception:
                    pass
            
        try:
            pick_xyz = [float(x.strip()) for x in self.pick_entry.get().split(',')]
            place_xyz = [float(x.strip()) for x in self.place_entry.get().split(',')]
            self.ax.scatter(*pick_xyz, marker='x', color='#F5A623', s=120, linewidths=3, label='PICK', zorder=10)
            self.ax.scatter(*place_xyz, marker='x', color='#17A2B8', s=120, linewidths=3, label='PLACE', zorder=10)
        except Exception:
            pass
            
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.legend()
        self.canvas.draw()

    def _ask_user_choice(self, title, options):
        # A simple blocking custom dialog
        top = tk.Toplevel(self.root)
        top.title(title)
        top.geometry("300x200")
        top.configure(bg=self.bg_color)
        top.transient(self.root)
        top.grab_set()
        
        tk.Label(top, text="Multiple feasible robots found.\nPlease select one:", bg=self.bg_color, fg=self.text_color).pack(pady=20)
        
        selected = tk.StringVar(top)
        selected.set(options[0])
        
        opt = tk.OptionMenu(top, selected, *options)
        opt.config(bg=self.card_color, fg=self.text_color)
        opt.pack(pady=10)
        
        result = [None]
        def on_ok():
            result[0] = selected.get()
            top.destroy()
            
        tk.Button(top, text="OK", command=on_ok, bg=self.accent_color, fg=self.text_color).pack(pady=10)
        self.root.wait_window(top)
        return result[0]

    def _on_execute(self):
        threading.Thread(target=self._run_plan_thread, daemon=True).start()
        
    def _run_plan_thread(self):
        self.log_text.delete(1.0, tk.END)
        self._log("=== STARTING STAGE 1 ===")
        
        try:
            pick_xyz = [float(x.strip()) for x in self.pick_entry.get().split(',')]
            place_xyz = [float(x.strip()) for x in self.place_entry.get().split(',')]
        except Exception:
            self._log("ERROR: Invalid Pick/Place Coordinates.")
            return
            
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
        self._log(f"> Selected PICK: {pick_rid} (Approach: {pick_cand.approach_direction})")
        
        self._log(f"\nEvaluating PLACE at {place_xyz}")
        place_opts = {}
        for rid, cfg in self.robots.items():
            if rid == pick_rid: continue # Different robot for place usually, but could be same. We allow it if feasible.
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
        self._log(f"> Selected PLACE: {place_rid} (Approach: {place_cand.approach_direction})")
        
        # Get Manual Handover Configs
        pair_key_1 = f"{pick_rid}_{place_rid}"
        pair_key_2 = f"{place_rid}_{pick_rid}"
        h_cfg = self.handover_cfg.get('manual_handover', {}).get(pair_key_1)
        if not h_cfg:
            h_cfg = self.handover_cfg.get('manual_handover', {}).get(pair_key_2)
            
        if not h_cfg:
            self._log(f"ERROR: No manual handover defined for {pick_rid} <-> {place_rid} in handover.yaml")
            return
            
        pick_h_q = np.array(h_cfg[f"{pick_rid}_q"])
        place_h_q = np.array(h_cfg[f"{place_rid}_q"])
        
        self._log(f"\nVerifying paths...")
        self.is_simulating = True
        self.current_q = {k: v.copy() for k, v in self.live_q.items()}
        
        start_pick_q = self.current_q[pick_rid].copy()
        start_place_q = self.current_q[place_rid].copy()
        
        pick_path = [start_pick_q + s * (pick_cand.joints - start_pick_q) for s in np.linspace(0, 1, 30)]
        pick_handover_path = [pick_cand.joints + s * (pick_h_q - pick_cand.joints) for s in np.linspace(0, 1, 30)]
        place_handover_path = [start_place_q + s * (place_h_q - start_place_q) for s in np.linspace(0, 1, 30)]
        place_path = [place_h_q + s * (place_cand.joints - place_h_q) for s in np.linspace(0, 1, 30)]
        
        def cc(path, cfg):
            for step_idx, q in enumerate(path):
                q_rad = np.deg2rad(q)
                pos = all_frame_positions(q_rad, cfg)
                n = len(pos)
                for i in range(n):
                    for j in range(i+2, n-1):
                        dist = np.linalg.norm(pos[i]-pos[j])
                        if dist < self.safe_dist:
                            return False, f"Self-collision (Link {i} <-> Link {j}, dist={dist:.1f} < {self.safe_dist}) at path step {step_idx}"
            return True, "Valid path"
            
        pick_app_ok, pick_app_msg = cc(pick_path, self.robots[pick_rid])
        pick_h_ok, pick_h_msg = cc(pick_handover_path, self.robots[pick_rid])
        place_h_ok, place_h_msg = cc(place_handover_path, self.robots[place_rid])
        place_app_ok, place_app_msg = cc(place_path, self.robots[place_rid])
        
        self._log("  Pick Approach: " + ("PASS" if pick_app_ok else f"FAIL - {pick_app_msg}"))
        self._log("  Pick->Handover: " + ("PASS" if pick_h_ok else f"FAIL - {pick_h_msg}"))
        self._log("  Place Handover: " + ("PASS" if place_h_ok else f"FAIL - {place_h_msg}"))
        self._log("  Handover->Place: " + ("PASS" if place_app_ok else f"FAIL - {place_app_msg}"))
        
        if not (pick_app_ok and pick_h_ok and place_h_ok and place_app_ok):
            self._log("\n=== ABORTED ===")
            self._log("Cannot execute Stage 1 sequence. Please adjust your target XYZ or the manual_handover joint angles in handover.yaml to ensure a collision-free trajectory.")
            self.is_simulating = False
            return
            
        self._log("\n=== EXECUTING SEQUENTIAL PLAN ===")
        
        def animate_path(rid, path, label, wait_time=1.0):
            self._log(label)
            for q in path:
                self.current_q[rid] = q
                self._update_3d()
            time.sleep(wait_time)
            
        animate_path(pick_rid, pick_path, f"1. {pick_rid} moves to Pick...", 0.5)
        self._log(f"2. {pick_rid} waits 2s, closes gripper.")
        time.sleep(2)
        animate_path(pick_rid, pick_handover_path, f"3. {pick_rid} moves to Manual Handover.", 0.5)
        animate_path(place_rid, place_handover_path, f"4. {place_rid} moves to Manual Handover.", 0.5)
        self._log(f"5. Transfer: {place_rid} closes, {pick_rid} opens.")
        time.sleep(1)
        animate_path(place_rid, place_path, f"6. {place_rid} moves to Place...", 0.5)
        self._log(f"7. {place_rid} waits 2s, releases object.")
        time.sleep(2)
        
        # Reset to home
        self.current_q[pick_rid] = np.zeros(self.robots[pick_rid].num_joints)
        self.current_q[place_rid] = np.zeros(self.robots[place_rid].num_joints)
        self._update_3d()
        
        self._log("=== STAGE 1 SUCCESS ===")
        self.is_simulating = False

def ros_spin_thread(node):
    rclpy.spin(node)

def main(args=None):
    rclpy.init(args=args)
    root = tk.Tk()
    app = Stage1GUI(root)
    
    thread = threading.Thread(target=ros_spin_thread, args=(app,), daemon=True)
    thread.start()
    
    root.mainloop()
    app.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
