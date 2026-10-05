#!/usr/bin/env python3
"""Multi-Robot Handover Control Station.

Provides full-pose pick/place planning with detailed diagnostics,
3D visualization, and handover control.  Does NOT replace existing
single-robot GUIs — they remain intact.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray, String
import yaml
import threading
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from robot_control.kinematics.fk_math import (
    all_frame_positions, mdh_matrix, dh_joint_angles, fk
)
from robot_control.handover.planner import (
    HandoverPlanner, euler_to_matrix, matrix_to_euler, PlanAnalysis
)
from robot_control.handover.trajectory import smoothstep7
from robot_control.nodes.handover_planner_node import YamlKinematicsConfig


# ============================================================
# STYLED BUTTON
# ============================================================

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command=None,
                 bg_parent="#F3F8FC", bg="#4F9DE8", fg="#FFFFFF",
                 hover_bg="#6FB7E9", radius=10, height=34,
                 font=("Arial", 10, "bold"), **kwargs):
        super().__init__(parent, height=height, bg=bg_parent,
                         highlightthickness=0, **kwargs)
        self.command = command
        self.text = text
        self.bg_color = bg
        self.fg_color = fg
        self.hover_bg = hover_bg
        self.radius = radius
        self.font = font
        self.rect_id = None
        self.text_id = None
        self.bind("<Configure>", self.draw)
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def draw(self, event=None):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or h < 10:
            return
        self.rect_id = self._rounded_rect(
            1, 1, w - 3, h - 3, self.radius,
            fill=self.bg_color, outline="")
        self.text_id = self.create_text(
            w / 2, (h - 2) / 2, text=self.text,
            fill=self.fg_color, font=self.font, justify="center")

    def _rounded_rect(self, x1, y1, x2, y2, r, **kw):
        pts = (x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1,
               x2, y1, x2, y1+r, x2, y1+r, x2, y2-r,
               x2, y2-r, x2, y2, x2-r, y2, x2-r, y2,
               x1+r, y2, x1+r, y2, x1, y2, x1, y2-r,
               x1, y2-r, x1, y1+r, x1, y1+r, x1, y1)
        return self.create_polygon(pts, smooth=True, **kw)

    def on_press(self, e):
        if self.rect_id:
            self.itemconfig(self.rect_id, fill=self.hover_bg)

    def on_release(self, e):
        if self.rect_id:
            self.itemconfig(self.rect_id, fill=self.hover_bg)
        if self.command:
            self.command()

    def on_enter(self, e):
        if self.rect_id:
            self.itemconfig(self.rect_id, fill=self.hover_bg)
            self.config(cursor="hand2")

    def on_leave(self, e):
        if self.rect_id:
            self.itemconfig(self.rect_id, fill=self.bg_color)
            self.config(cursor="")


# ============================================================
# MAIN GUI
# ============================================================

class HandoverGUI(Node):
    """Multi-robot handover control station (Part 12)."""

    def __init__(self, root):
        super().__init__('multi_robot_handover_gui_node')
        self.root = root
        self.root.title("MULTI-ROBOT HANDOVER CONTROL STATION")
        self.root.geometry("1500x950")
        self.bg_main = "#F3F8FC"
        self.bg_card = "#FFFFFF"
        self.root.configure(bg=self.bg_main)

        # Load configs
        cfg_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', 'config')
        cfg_dir = os.path.normpath(cfg_dir)
        # Fallback to known path
        if not os.path.isdir(cfg_dir):
            cfg_dir = '/home/iit/rt_test/src/robot_control/config'

        handover_yaml = os.path.join(cfg_dir, 'handover.yaml')
        with open(handover_yaml, 'r') as f:
            self.handover_config = yaml.safe_load(f)

        self.robots = {}
        for i in [1, 2, 3]:
            path = os.path.join(cfg_dir, f'robot{i}.yaml')
            self.robots[i] = YamlKinematicsConfig(path)

        self.planner = HandoverPlanner(
            self.handover_config, self.robots,
            self.handover_config.get('robots', {}))

        # State
        self.q_deg = {
            1: np.zeros(self.robots[1].num_joints),
            2: np.zeros(self.robots[2].num_joints),
            3: np.zeros(self.robots[3].num_joints),
        }
        self.sim_q_deg = {1: None, 2: None, 3: None}
        self.is_simulating = False
        self.planned_result = None
        self.plan_analysis = None
        self.plan_validated = False
        self.traces = {1: [], 2: [], 3: []}
        self.robot_roles = {1: "NONE", 2: "NONE", 3: "NONE"}

        # Visualization toggles
        self.show_traces = tk.BooleanVar(value=True)
        self.show_position_workspace = tk.BooleanVar(value=False)
        self.show_tcp_frames = tk.BooleanVar(value=True)
        self.show_handover_candidates = tk.BooleanVar(value=False)

        # ROS publishers / subscribers
        self.publishers_traj = {}
        self.publishers_mode = {}
        mode_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.subs = []
        for i in [1, 2, 3]:
            sub = self.create_subscription(
                Float64MultiArray, f'/robot{i}/joint_commands',
                lambda msg, rid=i: self._joint_cb(msg, rid), 10)
            self.subs.append(sub)
            self.publishers_traj[i] = self.create_publisher(
                Float64MultiArray,
                f'/robot{i}/trajectory_joint_command', 10)
            self.publishers_mode[i] = self.create_publisher(
                String, f'/robot{i}/control_mode', mode_qos)

        self._build_ui()
        self.timer = self.create_timer(0.05, self._update_3d)
        self.status_timer = self.create_timer(0.1, self._update_status)
        
        # Proper clean shutdown handling (Part 23)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def _on_closing(self):
        """Cleanly shutdown Matplotlib and Tkinter to prevent TclError."""
        self.is_simulating = False
        try:
            if hasattr(self, 'timer'):
                self.timer.cancel()
            if hasattr(self, 'status_timer'):
                self.status_timer.cancel()
            
            plt.close('all')
        except Exception:
            pass
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    # ── ROS callback ──

    def _joint_cb(self, msg, robot_id):
        if len(msg.data) == self.robots[robot_id].num_joints:
            self.q_deg[robot_id] = np.array(msg.data, dtype=float)

    # ── World-to-base helper ──

    def get_world_to_base(self, robot_id):
        info = self.handover_config.get('robots', {}).get(
            f'robot{robot_id}', {})
        xyz = info.get('world_base_xyz', [0, 0, 0])
        rpy = np.deg2rad(info.get('world_base_rpy_deg', [0, 0, 0]))
        R = euler_to_matrix(rpy)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = xyz
        return T

    # ══════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════

    def _build_ui(self):
        main = tk.Frame(self.root, bg=self.bg_main)
        main.pack(fill='both', expand=True, padx=8, pady=8)

        # LEFT PANEL (controls + diagnostics)
        left = tk.Frame(main, bg=self.bg_main, width=400)
        left.pack(side='left', fill='y', padx=(0, 8))
        left.pack_propagate(False)

        # Scrollable left panel
        canvas_frame = tk.Canvas(left, bg=self.bg_main,
                                 highlightthickness=0)
        scrollbar = ttk.Scrollbar(left, orient='vertical',
                                  command=canvas_frame.yview)
        self.scroll_frame = tk.Frame(canvas_frame, bg=self.bg_main)
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas_frame.configure(
                scrollregion=canvas_frame.bbox("all")))
        canvas_frame.create_window(
            (0, 0), window=self.scroll_frame, anchor="nw")
        canvas_frame.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        canvas_frame.pack(side='left', fill='both', expand=True)

        sf = self.scroll_frame  # shorthand

        # ── Title ──
        tk.Label(sf, text="MULTI-ROBOT\nHANDOVER STATION",
                 fg='#4F9DE8', bg=self.bg_main,
                 font=("Arial", 14, "bold")).pack(pady=(10, 5))

        # ── A. ROBOT STATUS (Part 12A) ──
        self.status_labels = {}
        self.role_labels = {}
        for i in [1, 2, 3]:
            card = tk.Frame(sf, bg=self.bg_card, pady=4, padx=6,
                            highlightbackground="#D9E7F2",
                            highlightthickness=1)
            card.pack(fill='x', padx=8, pady=3)
            hdr = tk.Frame(card, bg=self.bg_card)
            hdr.pack(fill='x')
            tk.Label(hdr, text=f"ROBOT {i}", fg='#333',
                     bg=self.bg_card,
                     font=("Arial", 10, "bold")).pack(side='left')
            role_lbl = tk.Label(hdr, text="NONE", fg='#888',
                                bg=self.bg_card,
                                font=("Arial", 9))
            role_lbl.pack(side='right')
            self.role_labels[i] = role_lbl

            info_lbl = tk.Label(
                card, text="TCP: --\nJoints: --", fg='#555',
                bg=self.bg_card, justify='left',
                font=("Consolas", 8))
            info_lbl.pack(anchor='w')
            self.status_labels[i] = info_lbl

        # ── B. PICK POSE (Part 12B) ──
        self._build_pose_section(sf, "PICK POSE", "pick")

        # ── C. PLACE POSE (Part 12C) ──
        self._build_pose_section(sf, "PLACE POSE", "place")

        # ── D. STATIC HANDOVER MAP ──
        map_card = tk.Frame(sf, bg=self.bg_card, pady=6, padx=8,
                            highlightbackground="#D9E7F2", highlightthickness=1)
        map_card.pack(fill='x', padx=8, pady=5)
        tk.Label(map_card, text="STATIC HANDOVER MAP", fg='#333',
                 bg=self.bg_card, font=("Arial", 10, "bold")).pack(anchor='w')
                 
        self.map_lbls = {}
        for pair in [(1,2), (2,3), (3,1)]:
            lbl = tk.Label(map_card, text=f"R{pair[0]} ↔ R{pair[1]}: --", 
                           fg='#555', bg=self.bg_card, font=("Consolas", 8))
            lbl.pack(anchor='w')
            self.map_lbls[pair] = lbl
            
        self.map_hash_lbl = tk.Label(map_card, text="Hash: --", fg='#888', bg=self.bg_card, font=("Consolas", 7))
        self.map_hash_lbl.pack(anchor='w')
        
        RoundedButton(map_card, "REBUILD MAP", command=self._rebuild_map,
                      bg="#E89F4F", hover_bg="#E8B072").pack(fill='x', pady=5)
                      
        self._update_map_ui()

        # ── Toggles ──
        tog = tk.Frame(sf, bg=self.bg_main)
        tog.pack(fill='x', padx=8, pady=5)
        tk.Checkbutton(tog, text="Traces",
                       variable=self.show_traces,
                       bg=self.bg_main, font=("Arial", 8)).pack(
                           side='left')
        tk.Checkbutton(tog, text="Position WS",
                       variable=self.show_position_workspace,
                       bg=self.bg_main, font=("Arial", 8)).pack(
                           side='left')
        tk.Checkbutton(tog, text="TCP Frames",
                       variable=self.show_tcp_frames,
                       bg=self.bg_main, font=("Arial", 8)).pack(
                           side='left')

        # ── G. PLANNING STATUS (Part 12G) ──
        self.plan_state_label = tk.Label(
            sf, text="Status: IDLE", fg='#F5A623', bg=self.bg_main,
            font=("Arial", 10, "bold"))
        self.plan_state_label.pack(pady=5)

        # ── D. DIAGNOSTICS (Part 12D) ──
        self.diag_text = scrolledtext.ScrolledText(
            sf, height=12, bg=self.bg_card, fg='#333',
            font=("Consolas", 8), relief='flat',
            highlightbackground="#D9E7F2", highlightthickness=1,
            wrap='word')
        self.diag_text.pack(fill='x', padx=8, pady=4)

        # ── H. BUTTONS (Part 12H) ──
        btn = tk.Frame(sf, bg=self.bg_main)
        btn.pack(fill='x', padx=8, pady=8)
        RoundedButton(btn, "PLAN", bg_parent=self.bg_main,
                      bg="#4F9DE8", hover_bg="#6FB7E9",
                      command=self._run_plan).pack(fill='x', pady=3)
        RoundedButton(btn, "SIMULATE", bg_parent=self.bg_main,
                      bg="#F5A623", hover_bg="#F7B74E",
                      command=self._run_simulate).pack(fill='x', pady=3)
        self.exec_btn = RoundedButton(
            btn, "EXECUTE", bg_parent=self.bg_main,
            bg="#999999", hover_bg="#AAAAAA",
            command=self._run_execute)
        self.exec_btn.pack(fill='x', pady=3)
        RoundedButton(btn, "STOP", bg_parent=self.bg_main,
                      bg="#DC3545", hover_bg="#F84B5B",
                      command=self._run_stop).pack(fill='x', pady=3)

        btn2 = tk.Frame(sf, bg=self.bg_main)
        btn2.pack(fill='x', padx=8)
        RoundedButton(btn2, "RESET", bg_parent=self.bg_main,
                      bg="#6C757D", hover_bg="#868E96",
                      command=self._reset).pack(fill='x', pady=3)
        RoundedButton(btn2, "CLEAR PLAN", bg_parent=self.bg_main,
                      bg="#6C757D", hover_bg="#868E96",
                      command=self._clear_plan).pack(fill='x', pady=3)

        # RIGHT PANEL (3D view)
        right = tk.Frame(main, bg=self.bg_card,
                         highlightbackground="#D9E7F2",
                         highlightthickness=1)
        right.pack(side='left', fill='both', expand=True)

        self.fig = plt.figure(facecolor=self.bg_card)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor(self.bg_card)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    # ── Pose section builder ──

    def _build_pose_section(self, parent, title, prefix):
        frame = tk.LabelFrame(parent, text=title, fg='#333',
                              bg=self.bg_main,
                              font=("Arial", 9, "bold"))
        frame.pack(fill='x', padx=8, pady=4)

        entries = {}
        # Position inputs
        row_pos = tk.Frame(frame, bg=self.bg_main)
        row_pos.pack(fill='x', padx=4, pady=1)
        for i, (lbl, default) in enumerate([("X", "0"), ("Y", "0"), ("Z", "200")]):
            tk.Label(row_pos, text=lbl, fg='#555', bg=self.bg_main, width=2, anchor='w', font=("Arial", 8)).pack(side='left')
            e = tk.Entry(row_pos, width=6, bg='#FFF', fg='#333', font=("Consolas", 9))
            e.pack(side='left', padx=2)
            e.insert(0, default)
            entries[lbl] = e

        # Orientation mode
        mode_var = tk.StringVar(value="AUTO")
        row_mode = tk.Frame(frame, bg=self.bg_main)
        row_mode.pack(fill='x', padx=4, pady=2)
        tk.Radiobutton(row_mode, text="AUTO Orientation", variable=mode_var, value="AUTO", bg=self.bg_main).pack(side='left')
        tk.Radiobutton(row_mode, text="MANUAL", variable=mode_var, value="MANUAL", bg=self.bg_main).pack(side='left')
        setattr(self, f'{prefix}_mode', mode_var)

        # Manual Orientation inputs
        row_rot = tk.Frame(frame, bg=self.bg_main)
        row_rot.pack(fill='x', padx=4, pady=1)
        for i, (lbl, default) in enumerate([("Roll", "0"), ("Pitch", "90"), ("Yaw", "0")]):
            tk.Label(row_rot, text=lbl, fg='#555', bg=self.bg_main, width=4, anchor='w', font=("Arial", 8)).pack(side='left')
            e = tk.Entry(row_rot, width=5, bg='#FFF', fg='#333', font=("Consolas", 9))
            e.pack(side='left', padx=2)
            e.insert(0, default)
            entries[lbl] = e

        setattr(self, f'{prefix}_entries', entries)

    # ── Read pose from entries ──

    def _read_pose(self, prefix):
        ent = getattr(self, f'{prefix}_entries')
        mode = getattr(self, f'{prefix}_mode').get()
        x = float(ent["X"].get())
        y = float(ent["Y"].get())
        z = float(ent["Z"].get())
        pos = np.array([x, y, z])
        if mode == "AUTO":
            rot = "AUTO"
        else:
            roll = np.deg2rad(float(ent["Roll"].get()))
            pitch = np.deg2rad(float(ent["Pitch"].get()))
            yaw = np.deg2rad(float(ent["Yaw"].get()))
            rot = euler_to_matrix([roll, pitch, yaw])
        return pos, rot

    # ══════════════════════════════════════════════
    # STATUS UPDATES
    # ══════════════════════════════════════════════

    def _update_status(self):
        for i in [1, 2, 3]:
            q_rad = np.deg2rad(self.q_deg[i])
            try:
                p_tcp, T_fk = fk(q_rad, self.robots[i])
                T_W_B = self.get_world_to_base(i)
                tcp_world = (T_W_B @ np.append(p_tcp, 1.0))[:3]
                js = ", ".join(f"{x:.1f}" for x in self.q_deg[i])
                self.status_labels[i].config(
                    text=f"TCP: {tcp_world[0]:.0f}, "
                         f"{tcp_world[1]:.0f}, {tcp_world[2]:.0f}\n"
                         f"Joints: [{js}]")
            except Exception:
                pass
            self.role_labels[i].config(
                text=self.robot_roles[i],
                fg={'PICKER': '#28A745', 'RECEIVER': '#4F9DE8',
                    'NONE': '#888'}.get(self.robot_roles[i], '#888'))

    # ══════════════════════════════════════════════
    # PLANNING  (Part 12G state machine)
    # ══════════════════════════════════════════════

    def _set_state(self, state, color='#F5A623'):
        self.plan_state_label.config(text=f"Status: {state}", fg=color)

    def _run_plan(self):
        self._set_state("PLANNING...", "#4F9DE8")
        self.diag_text.delete("1.0", tk.END)
        self.plan_validated = False
        self._update_execute_btn()
        self.root.update()

        def thread():
            try:
                pick_pos, pick_rot = self._read_pose("pick")
                place_pos, place_rot = self._read_pose("place")

                current_qs = {
                    i: np.deg2rad(self.q_deg[i]) for i in [1, 2, 3]}

                t0 = time.time()
                analysis = self.planner.plan_with_analysis(
                    pick_pos, place_pos, current_qs,
                    pick_rot=pick_rot, place_rot=place_rot)
                t_plan = time.time() - t0
                analysis.total_plan_time = t_plan

                self.plan_analysis = analysis

                # Also store legacy dict for simulate/execute
                if analysis.plan_type != "FAILED":
                    if analysis.plan_type == "DIRECT":
                        self.planned_result = {
                            "status": "SUCCESS", "type": "DIRECT",
                            "robot": analysis.pick_robot,
                            "pick_q": analysis.pick_joints,
                            "place_q": analysis.place_joints,
                        }
                    else:
                        self.planned_result = {
                            "status": "SUCCESS", "type": "HANDOVER",
                            "candidate": analysis.handover_candidate,
                            "q_pick_init": analysis.pick_joints,
                            "q_place_final": analysis.place_joints,
                        }
                    self.plan_validated = True
                else:
                    self.planned_result = None
                    self.plan_validated = False

                # Update roles
                self.robot_roles = {1: "NONE", 2: "NONE", 3: "NONE"}
                if analysis.pick_robot:
                    self.robot_roles[analysis.pick_robot] = "PICKER"
                if analysis.place_robot:
                    self.robot_roles[analysis.place_robot] = "RECEIVER"
                if (analysis.pick_robot and analysis.place_robot
                        and analysis.pick_robot == analysis.place_robot):
                    self.robot_roles[analysis.pick_robot] = "PICKER"

                def update_ui():
                    self.diag_text.delete("1.0", tk.END)
                    self.diag_text.insert(tk.END,
                                          analysis.format_report())
                    if analysis.plan_type == "FAILED":
                        self._set_state("FAILED", "#DC3545")
                    else:
                        self._set_state(
                            f"PLAN READY ({analysis.plan_type})",
                            "#28A745")
                    self.traces = {1: [], 2: [], 3: []}
                    self._update_execute_btn()

                self.root.after(0, update_ui)

            except Exception as e:
                def show_err():
                    self._set_state("ERROR", "#DC3545")
                    self.diag_text.delete("1.0", tk.END)
                    self.diag_text.insert(tk.END, f"Error: {e}")
                    self.plan_validated = False
                    self._update_execute_btn()
                self.root.after(0, show_err)

        threading.Thread(target=thread, daemon=True).start()

    def _update_execute_btn(self):
        if self.plan_validated:
            self.exec_btn.bg_color = "#28A745"
            self.exec_btn.hover_bg = "#34CE57"
        else:
            self.exec_btn.bg_color = "#999999"
            self.exec_btn.hover_bg = "#AAAAAA"
        self.exec_btn.draw()

    # ── SIMULATE ──

    def _run_simulate(self):
        if not self.planned_result:
            return
        self.is_simulating = True
        self.traces = {1: [], 2: [], 3: []}

        def thread():
            try:
                if self.planned_result.get('type') == 'HANDOVER':
                    cand = self.planned_result['candidate']
                    r1 = cand.picker_robot
                    r2 = cand.receiver_robot
                    q_pick = self.planned_result['q_pick_init']
                    q_r1_H = cand.picker_joints
                    q_r2_H = cand.receiver_joints
                    q_place = self.planned_result['q_place_final']

                    T = max(cand.trajectory_time, 1.5)
                    times = np.linspace(0, T, int(T * 20))

                    # Phase 1: current → pick (picker) + current → H approach (receiver)
                    q_r1_start = np.deg2rad(self.q_deg[r1])
                    q_r2_start = np.deg2rad(self.q_deg[r2])
                    for t in times:
                        if not self.is_simulating:
                            return
                        qp, _, _, _ = smoothstep7(
                            q_r1_start, q_pick, t, T)
                        qr, _, _, _ = smoothstep7(
                            q_r2_start, q_r2_H, t, T)
                        self.sim_q_deg[r1] = np.rad2deg(qp)
                        self.sim_q_deg[r2] = np.rad2deg(qr)
                        time.sleep(0.05)

                    # Phase 2: PICK → H (picker)
                    for t in times:
                        if not self.is_simulating:
                            return
                        qp, _, _, _ = smoothstep7(
                            q_pick, q_r1_H, t, T)
                        self.sim_q_deg[r1] = np.rad2deg(qp)
                        time.sleep(0.05)

                    # Phase 3: H → PLACE (receiver)
                    for t in times:
                        if not self.is_simulating:
                            return
                        qr, _, _, _ = smoothstep7(
                            q_r2_H, q_place, t, T)
                        self.sim_q_deg[r2] = np.rad2deg(qr)
                        time.sleep(0.05)

                elif self.planned_result.get('type') == 'DIRECT':
                    r = self.planned_result['robot']
                    q0 = np.deg2rad(self.q_deg[r])
                    q_pick = self.planned_result['pick_q']
                    q_place = self.planned_result['place_q']
                    T = 2.0
                    times = np.linspace(0, T, int(T * 20))
                    for t in times:
                        if not self.is_simulating:
                            return
                        q, _, _, _ = smoothstep7(q0, q_pick, t, T)
                        self.sim_q_deg[r] = np.rad2deg(q)
                        time.sleep(0.05)
                    for t in times:
                        if not self.is_simulating:
                            return
                        q, _, _, _ = smoothstep7(
                            q_pick, q_place, t, T)
                        self.sim_q_deg[r] = np.rad2deg(q)
                        time.sleep(0.05)

            finally:
                self.is_simulating = False
                self.sim_q_deg = {1: None, 2: None, 3: None}

        threading.Thread(target=thread, daemon=True).start()

    def _rebuild_map(self):
        def thread():
            self._set_state("REBUILDING MAP...", "#E89F4F")
            t0 = time.time()
            self.planner.static_map.load_or_generate(force_rebuild=True)
            self._set_state(f"Map rebuilt in {time.time()-t0:.1f}s", "#28A745")
            self._update_map_ui()
        threading.Thread(target=thread, daemon=True).start()

    def _update_map_ui(self):
        sm = self.planner.static_map
        self.map_hash_lbl.config(text=f"Hash: {sm.current_hash[:8]}...")
        for pair, lbl in self.map_lbls.items():
            if pair in sm.maps:
                lbl.config(text=f"R{pair[0]} ↔ R{pair[1]}: LOADED ({len(sm.maps[pair])} poses)")
            else:
                lbl.config(text=f"R{pair[0]} ↔ R{pair[1]}: --")

    # ── EXECUTE (guarded — Part 12H) ──

    def _run_execute(self):
        if not self.plan_validated or not self.planned_result:
            self._set_state("No validated plan — cannot execute",
                            "#DC3545")
            return
        self.traces = {1: [], 2: [], 3: []}

        def thread():
            try:
                if self.planned_result.get('type') == 'HANDOVER':
                    cand = self.planned_result['candidate']
                    r1 = cand.picker_robot
                    r2 = cand.receiver_robot
                    q_pick = self.planned_result['q_pick_init']
                    q_r1_H = cand.picker_joints
                    q_r2_H = cand.receiver_joints
                    q_place = self.planned_result['q_place_final']

                    msg = String()
                    msg.data = 'TRAJECTORY'
                    self.publishers_mode[r1].publish(msg)
                    self.publishers_mode[r2].publish(msg)
                    T = max(cand.trajectory_time, 1.5)
                    times = np.linspace(0, T, int(T * 20))

                    # Phase 1: current → pick + current → H approach
                    q_r1_s = np.deg2rad(self.q_deg[r1])
                    q_r2_s = np.deg2rad(self.q_deg[r2])
                    for t in times:
                        qp, _, _, _ = smoothstep7(q_r1_s, q_pick, t, T)
                        qr, _, _, _ = smoothstep7(q_r2_s, q_r2_H, t, T)
                        m1 = Float64MultiArray()
                        m1.data = np.rad2deg(qp).tolist()
                        m2 = Float64MultiArray()
                        m2.data = np.rad2deg(qr).tolist()
                        self.publishers_traj[r1].publish(m1)
                        self.publishers_traj[r2].publish(m2)
                        time.sleep(0.05)

                    # Phase 2: PICK → H
                    for t in times:
                        qp, _, _, _ = smoothstep7(
                            q_pick, q_r1_H, t, T)
                        m1 = Float64MultiArray()
                        m1.data = np.rad2deg(qp).tolist()
                        self.publishers_traj[r1].publish(m1)
                        time.sleep(0.05)

                    # Phase 3: H → PLACE
                    for t in times:
                        qr, _, _, _ = smoothstep7(
                            q_r2_H, q_place, t, T)
                        m2 = Float64MultiArray()
                        m2.data = np.rad2deg(qr).tolist()
                        self.publishers_traj[r2].publish(m2)
                        time.sleep(0.05)

                    msg.data = 'MANUAL'
                    self.publishers_mode[r1].publish(msg)
                    self.publishers_mode[r2].publish(msg)

                elif self.planned_result.get('type') == 'DIRECT':
                    r = self.planned_result['robot']
                    msg = String()
                    msg.data = 'TRAJECTORY'
                    self.publishers_mode[r].publish(msg)
                    q0 = np.deg2rad(self.q_deg[r])
                    q_pick = self.planned_result['pick_q']
                    q_place = self.planned_result['place_q']
                    T = 2.0
                    times = np.linspace(0, T, int(T * 20))
                    for t in times:
                        q, _, _, _ = smoothstep7(q0, q_pick, t, T)
                        m = Float64MultiArray()
                        m.data = np.rad2deg(q).tolist()
                        self.publishers_traj[r].publish(m)
                        time.sleep(0.05)
                    for t in times:
                        q, _, _, _ = smoothstep7(
                            q_pick, q_place, t, T)
                        m = Float64MultiArray()
                        m.data = np.rad2deg(q).tolist()
                        self.publishers_traj[r].publish(m)
                        time.sleep(0.05)
                    msg.data = 'MANUAL'
                    self.publishers_mode[r].publish(msg)

            except Exception as e:
                def err():
                    self._set_state(f"EXEC ERROR: {e}", "#DC3545")
                self.root.after(0, err)

        threading.Thread(target=thread, daemon=True).start()

    # ── STOP / RESET / CLEAR ──

    def _run_stop(self):
        for i in [1, 2, 3]:
            msg = String()
            msg.data = 'MANUAL'
            self.publishers_mode[i].publish(msg)
        self.is_simulating = False
        self.sim_q_deg = {1: None, 2: None, 3: None}

    def _reset(self):
        self._clear_plan()
        self.q_deg = {
            i: np.zeros(self.robots[i].num_joints) for i in [1, 2, 3]}

    def _clear_plan(self):
        self.planned_result = None
        self.plan_analysis = None
        self.plan_validated = False
        self.robot_roles = {1: "NONE", 2: "NONE", 3: "NONE"}
        self.traces = {1: [], 2: [], 3: []}
        self._set_state("IDLE", "#F5A623")
        self.diag_text.delete("1.0", tk.END)
        self._update_execute_btn()

    # ══════════════════════════════════════════════
    # 3D VISUALIZATION  (Part 12F + Part 13)
    # ══════════════════════════════════════════════

    def _get_frame_transforms(self, q_mdh, config, T_W_B):
        T = np.copy(T_W_B)
        transforms = [T.copy()]
        for i in range(config.num_frames):
            Ti = mdh_matrix(
                config.a[i],
                np.deg2rad(config.alpha_deg[i]),
                q_mdh[i], config.d[i])
            T = T @ Ti
            transforms.append(T.copy())
        return transforms

    def _update_3d(self):
        self.ax.clear()
        colors = {1: '#4F9DE8', 2: '#F84B5B', 3: '#28A745'}

        for i in [1, 2, 3]:
            q_deg = (self.sim_q_deg[i]
                     if (self.is_simulating
                         and self.sim_q_deg[i] is not None)
                     else self.q_deg[i])
            q_rad = np.deg2rad(q_deg)
            cfg = self.robots[i]
            q_mdh = dh_joint_angles(q_rad, cfg)
            T_W_B = self.get_world_to_base(i)
            transforms = self._get_frame_transforms(
                q_mdh, cfg, T_W_B)

            pts = np.array([T[:3, 3] for T in transforms])
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                         '-o', linewidth=3, markersize=6,
                         color=colors[i], label=f'R{i}')

            # TCP frames
            if self.show_tcp_frames.get():
                T_tcp = transforms[-1]
                p = T_tcp[:3, 3]
                for ax_idx, c in enumerate(['r', 'g', 'b']):
                    d = T_tcp[:3, ax_idx] * 40
                    self.ax.plot(
                        [p[0], p[0]+d[0]],
                        [p[1], p[1]+d[1]],
                        [p[2], p[2]+d[2]],
                        color=c, linewidth=1.5)

            # Trace
            if self.show_traces.get() and self.is_simulating:
                self.traces[i].append(pts[-1].tolist())
            if self.show_traces.get() and len(self.traces[i]) > 0:
                tr = np.array(self.traces[i])
                self.ax.plot(tr[:, 0], tr[:, 1], tr[:, 2],
                             ':', color=colors[i], alpha=0.6,
                             linewidth=1.5)

        # PICK / PLACE markers
        try:
            pick_pos, _ = self._read_pose("pick")
            place_pos, _ = self._read_pose("place")
            self.ax.scatter(*pick_pos, marker='x', color='#F5A623',
                            s=120, linewidths=3,
                            label='PICK', zorder=10)
            self.ax.scatter(*place_pos, marker='x', color='#17A2B8',
                            s=120, linewidths=3,
                            label='PLACE', zorder=10)
        except Exception:
            pass

        # Handover point
        if (self.plan_analysis
                and self.plan_analysis.handover_candidate):
            hp = self.plan_analysis.handover_candidate.position
            self.ax.scatter(*hp, marker='D', color='#FF00FF',
                            s=100, linewidths=2,
                            label='Handover H*', zorder=10)

        # Position workspace (Part 13 — clearly labeled)
        if self.show_position_workspace.get():
            for i in [1, 2, 3]:
                ws = self._get_position_workspace(i)
                self.ax.scatter(
                    ws[:, 0], ws[:, 1], ws[:, 2],
                    s=3, color=colors[i], alpha=0.15,
                    edgecolors='none')

        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.set_xlim(-800, 800)
        self.ax.set_ylim(-800, 800)
        self.ax.set_zlim(0, 800)
        self.ax.set_box_aspect([1, 1, 0.5])
        self.ax.legend(
            facecolor=self.bg_card, edgecolor='#D9E7F2',
            fontsize=7, loc='upper right')
        self.canvas.draw_idle()
        self.canvas.flush_events()

    def _get_position_workspace(self, r_id):
        """Generate position-reachability cloud (NOT pose workspace).

        This shows where the TCP can physically reach in XYZ,
        but does NOT imply any specific orientation is achievable.
        """
        if not hasattr(self, '_ws_cache'):
            self._ws_cache = {}
        if r_id not in self._ws_cache:
            cfg = self.robots[r_id]
            q_min = cfg.joint_limits_rad[:, 0]
            q_max = cfg.joint_limits_rad[:, 1]
            n = 2000
            pts = np.zeros((n, 3))
            T_W_B = self.get_world_to_base(r_id)
            for k in range(n):
                q = q_min + (q_max - q_min) * np.random.rand(
                    cfg.num_joints)
                p, T_fk = fk(q, cfg)
                p_w = (T_W_B @ np.append(p, 1.0))[:3]
                pts[k] = p_w
            self._ws_cache[r_id] = pts
        return self._ws_cache[r_id]


# ============================================================
# ENTRY POINT
# ============================================================

def ros_spin_thread(node):
    rclpy.spin(node)


def main(args=None):
    rclpy.init(args=args)
    root = tk.Tk()
    app = HandoverGUI(root)

    thread = threading.Thread(
        target=ros_spin_thread, args=(app,), daemon=True)
    thread.start()

    root.mainloop()
    app.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
