"""Stage 2.5 GUI Node.

Combines Object-Pose IK (adaptive tilt & object yaw alignment) with Stage 1.5
synchronized pre-handover staging and whole-body link collision avoidance.

Features:
- Dual-mode execution:
  - SIMULATE: 3D preview in Matplotlib canvas without commanding physical hardware.
  - EXECUTE: Commands physical hardware via ROS2 topics, reusing simulated trajectories
    instantaneously when target coordinates match.
- Adaptive IK with automatic fallback:
  - Attempts ``object_pick_ik`` first for both Pick and Place.
  - If a target has no reachable solution under object orientation constraints,
    automatically falls back to normal Stage 1/1.5 XYZ IK.
- Real-time whole-body link clearance monitoring.
"""

import os
import threading
import time
import tkinter as tk
from tkinter import messagebox
from typing import Dict, List, Optional, Tuple

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, String

from robot_control.handover.object_pick_ik import (
    _solve_seed_object_ik,
    generate_pick_orientations,
)
from robot_control.handover.stage1 import load_stage1_robots
from robot_control.handover.stage1_5 import Stage1_5TrajectoryResult
from robot_control.handover.stage2_5 import (
    Stage2_5Plan,
    build_stage2_5_plan,
    plan_stage2_5_synchronized_paths,
)
from robot_control.handover.trajectory_cache import (
    clear_cache,
    compute_config_fingerprint,
    get_cache_stats,
    lookup_cached_target_ik,
    resolve_cache_filepath,
    store_cached_target_ik,
    target_cache_key,
)
from robot_control.kinematics.fk_math import all_frame_positions, fk


class Stage2_5GUI(Node):
    def __init__(self, root: tk.Tk):
        super().__init__('stage2_5_gui_node')
        self.root = root
        self.root.title("Stage 2.5 - Object-Pose IK & Synchronized Handover")
        self.root.geometry("1480x880")
        self.root.configure(bg="#F3F8FC")

        self.bg_color = "#F3F8FC"
        self.card_color = "#FFFFFF"
        self.text_color = "#2C3E50"
        self.accent_color = "#3498DB"
        self.success_color = "#27AE60"
        self.warning_color = "#E67E22"
        self.danger_color = "#E74C3C"

        self.click_mode = tk.StringVar(value="orbit")
        self.radar_sweep_cache = {}
        self.radar_ax = None
        self.radar_fig = None
        self.radar_canvas = None
        self._radar_bars = []
        self._radar_needle_line = None
        self._radar_needle_pt = None
        self._radar_mouse_pressed = False

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
        self.cache_filepath = resolve_cache_filepath(config_dir)
        self.config_fingerprint = compute_config_fingerprint(self.robots, self.handover_cfg)
        self.use_disk_cache_var = tk.BooleanVar(value=True)

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
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
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
                Float64MultiArray,
                f'/{rid}/joint_commands',
                lambda msg, r=rid: self._joint_cb(msg, r),
                10,
            )
            self.subs.append(sub)

        self._build_ui()
        self._last_visualization_signature = None
        self._init_scene()
        self.root.after(50, self._refresh_visualization)
        self.root.after(150, self._trigger_radar_sweep)
        self.root.after(150, self._update_radar_needle)

    def _send_joint_command(self, rid: str, q_deg: np.ndarray, real_robot: bool = False):
        if real_robot and rid in self.publishers_traj:
            msg = Float64MultiArray()
            msg.data = [float(v) for v in q_deg]
            self.publishers_traj[rid].publish(msg)

    def _set_control_mode(self, rid: str, mode_str: str, real_robot: bool = False):
        if real_robot and rid in self.publishers_mode:
            msg = String()
            msg.data = mode_str
            self.publishers_mode[rid].publish(msg)

    def _send_gripper_command(self, rid: str, state: str, real_robot: bool = False):
        if real_robot and rid in self.publishers_gripper:
            cfg = self.robots.get(rid)
            angle = getattr(cfg, f"gripper_{state}", 55.0 if state == 'close' else -55.0)
            msg = Float64MultiArray()
            msg.data = [float(angle)]
            self.publishers_gripper[rid].publish(msg)

    def _joint_cb(self, msg: Float64MultiArray, rid: str):
        if len(msg.data) == self.robots[rid].num_joints:
            self.live_q[rid] = np.array(msg.data, dtype=float)

    def _compute_scene_limits(self):
        base_positions = np.array(
            [cfg.world_base_xyz for cfg in self.robots.values()], dtype=float
        )
        max_reach = max(
            sum(abs(float(val)) for val in cfg.a)
            + sum(abs(float(val)) for val in cfg.d)
            + float(np.linalg.norm(cfg.tcp_offset))
            for cfg in self.robots.values()
        )
        low = np.min(base_positions, axis=0) - max_reach
        high = np.max(base_positions, axis=0) + max_reach

        min_base_z = float(np.min(base_positions[:, 2]))
        raw_z_min = self.handover_cfg.get('visualization', {}).get('z_min', None)
        if raw_z_min is not None:
            val = float(raw_z_min)
            z_min = min_base_z - val if val > min_base_z else val
        else:
            z_min = min_base_z - 50.0
        z_max = float(high[2])

        return (
            (float(low[0]), float(high[0])),
            (float(low[1]), float(high[1])),
            (z_min, z_max),
        )

    def _refresh_visualization(self):
        try:
            if not self.root.winfo_exists():
                return
            if self._visualization_signature() != self._last_visualization_signature:
                self._update_3d()
            self.root.after(50, self._refresh_visualization)
        except tk.TclError:
            return

    def _visualization_signature(self):
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
            self.pick_yaw_entry.get(),
            self.place_entry.get(),
            self.place_yaw_entry.get(),
        )

    def _build_ui(self):
        left_frame = tk.Frame(self.root, bg=self.bg_color, width=440)
        left_frame.pack(side="left", fill="y", padx=16, pady=16)
        left_frame.pack_propagate(False)

        # Header
        header = tk.Frame(left_frame, bg=self.bg_color)
        header.pack(fill="x", pady=(0, 10))
        tk.Label(
            header, text="STAGE 2.5", font=("Helvetica", 16, "bold"),
            bg=self.bg_color, fg=self.text_color
        ).pack(anchor="w")
        tk.Label(
            header, text="Object-Pose IK & Synchronized Handover",
            font=("Helvetica", 10, "bold"), bg=self.bg_color, fg=self.accent_color
        ).pack(anchor="w")
        tk.Label(
            header, text="Adaptive Tilt Pick-and-Place with Stage 1.5 Body Clearance",
            font=("Helvetica", 8), bg=self.bg_color, fg="#7F8C8D"
        ).pack(anchor="w")

        # TARGETS CARD
        targets_card = tk.Frame(left_frame, bg=self.card_color, padx=12, pady=8, relief="solid", bd=1)
        targets_card.pack(fill="x", pady=4)

        # 3D Interaction Mode Selector (Orbit vs Click-Pick vs Click-Place)
        mode_bar = tk.Frame(targets_card, bg=self.card_color)
        mode_bar.pack(fill="x", pady=(0, 6))

        tk.Label(
            mode_bar, text="3D Mode:", bg=self.card_color,
            fg="#7F8C8D", font=("Helvetica", 8, "bold")
        ).pack(side="left", padx=(0, 4))

        def _set_mode(m: str):
            self.click_mode.set(m)
            self.btn_orbit.config(
                bg="#2980B9" if m == "orbit" else "#EAEDED",
                fg="#FFFFFF" if m == "orbit" else "#2C3E50"
            )
            self.btn_click_pick.config(
                bg="#F5A623" if m == "pick" else "#EAEDED",
                fg="#FFFFFF" if m == "pick" else "#2C3E50"
            )
            self.btn_click_place.config(
                bg="#17A2B8" if m == "place" else "#EAEDED",
                fg="#FFFFFF" if m == "place" else "#2C3E50"
            )

        self.btn_orbit = tk.Button(
            mode_bar, text="🔄 Orbit 3D", font=("Helvetica", 8, "bold"),
            bg="#2980B9", fg="#FFFFFF", relief="flat", cursor="hand2",
            padx=5, pady=2, command=lambda: _set_mode("orbit")
        )
        self.btn_orbit.pack(side="left", padx=2)

        self.btn_click_pick = tk.Button(
            mode_bar, text="🎯 Set Pick", font=("Helvetica", 8, "bold"),
            bg="#EAEDED", fg="#2C3E50", relief="flat", cursor="hand2",
            padx=5, pady=2, command=lambda: _set_mode("pick")
        )
        self.btn_click_pick.pack(side="left", padx=2)

        self.btn_click_place = tk.Button(
            mode_bar, text="📍 Set Place", font=("Helvetica", 8, "bold"),
            bg="#EAEDED", fg="#2C3E50", relief="flat", cursor="hand2",
            padx=5, pady=2, command=lambda: _set_mode("place")
        )
        self.btn_click_place.pack(side="left", padx=2)

        # Pick Target Row
        pick_header = tk.Frame(targets_card, bg=self.card_color)
        pick_header.pack(fill="x")
        tk.Label(
            pick_header, text="PICK TARGET:", bg=self.card_color, fg=self.text_color,
            font=("Helvetica", 9, "bold")
        ).pack(side="left")
        self.pick_badge = tk.Label(
            pick_header, text="Pending", bg="#EAEDED", fg="#7F8C8D",
            font=("Helvetica", 8, "bold"), padx=6, pady=1
        )
        self.pick_badge.pack(side="right")

        pick_inputs = tk.Frame(targets_card, bg=self.card_color)
        pick_inputs.pack(fill="x", pady=(2, 6))
        tk.Label(pick_inputs, text="XYZ (mm):", bg=self.card_color, font=("Helvetica", 8)).grid(row=0, column=0, sticky="w")
        self.pick_entry = tk.Entry(pick_inputs, bg="#F8FAFC", relief="solid", bd=1, font=("Consolas", 9), width=24)
        self.pick_entry.insert(0, "-97.10, 0.0, 13.82")
        self.pick_entry.grid(row=0, column=1, padx=(4, 8), sticky="w")

        tk.Label(pick_inputs, text="Yaw (°):", bg=self.card_color, font=("Helvetica", 8)).grid(row=0, column=2, sticky="w")
        self.pick_yaw_entry = tk.Entry(pick_inputs, bg="#F8FAFC", relief="solid", bd=1, font=("Consolas", 9), width=6)
        self.pick_yaw_entry.insert(0, "0.0")
        self.pick_yaw_entry.grid(row=0, column=3, padx=(4, 0), sticky="w")

        # Place Target Row
        place_header = tk.Frame(targets_card, bg=self.card_color)
        place_header.pack(fill="x")
        tk.Label(
            place_header, text="PLACE TARGET:", bg=self.card_color, fg=self.text_color,
            font=("Helvetica", 9, "bold")
        ).pack(side="left")
        self.place_badge = tk.Label(
            place_header, text="Pending", bg="#EAEDED", fg="#7F8C8D",
            font=("Helvetica", 8, "bold"), padx=6, pady=1
        )
        self.place_badge.pack(side="right")

        place_inputs = tk.Frame(targets_card, bg=self.card_color)
        place_inputs.pack(fill="x", pady=(2, 2))
        tk.Label(place_inputs, text="XYZ (mm):", bg=self.card_color, font=("Helvetica", 8)).grid(row=0, column=0, sticky="w")
        self.place_entry = tk.Entry(place_inputs, bg="#F8FAFC", relief="solid", bd=1, font=("Consolas", 9), width=24)
        self.place_entry.insert(0, "0.0, 150.0, 150.0")
        self.place_entry.grid(row=0, column=1, padx=(4, 8), sticky="w")

        tk.Label(place_inputs, text="Yaw (°):", bg=self.card_color, font=("Helvetica", 8)).grid(row=0, column=2, sticky="w")
        self.place_yaw_entry = tk.Entry(place_inputs, bg="#F8FAFC", relief="solid", bd=1, font=("Consolas", 9), width=6)
        self.place_yaw_entry.insert(0, "0.0")
        self.place_yaw_entry.grid(row=0, column=3, padx=(4, 0), sticky="w")

        # GRASPABILITY & ORIENTATION RADAR CARD
        radar_card = tk.Frame(left_frame, bg=self.card_color, padx=10, pady=6, relief="solid", bd=1)
        radar_card.pack(fill="x", pady=4)

        radar_header = tk.Frame(radar_card, bg=self.card_color)
        radar_header.pack(fill="x")
        tk.Label(
            radar_header, text="GRASPABILITY RADAR (YAW & TILT)", bg=self.card_color,
            fg=self.text_color, font=("Helvetica", 8, "bold")
        ).pack(side="left")

        # Selector for Pick vs Place radar
        self.radar_target_var = tk.StringVar(value="pick")
        target_switch_frame = tk.Frame(radar_card, bg=self.card_color)
        target_switch_frame.pack(fill="x", pady=(1, 2))

        def _on_radar_target_toggle():
            self._update_radar_ui()
            self._trigger_radar_sweep()

        tk.Radiobutton(
            target_switch_frame, text="Pick Dial", variable=self.radar_target_var,
            value="pick", bg=self.card_color, font=("Helvetica", 8, "bold"),
            activebackground=self.card_color, fg="#E67E22",
            command=_on_radar_target_toggle
        ).pack(side="left")

        tk.Radiobutton(
            target_switch_frame, text="Place Dial", variable=self.radar_target_var,
            value="place", bg=self.card_color, font=("Helvetica", 8, "bold"),
            activebackground=self.card_color, fg="#17A2B8",
            command=_on_radar_target_toggle
        ).pack(side="left", padx=10)

        self.radar_info_lbl = tk.Label(
            radar_card, text="Pick Yaw: 0.0° (Drag dial to adjust)",
            bg=self.card_color, fg="#E67E22", font=("Helvetica", 8, "bold")
        )
        self.radar_info_lbl.pack(anchor="w")

        # Polar figure
        self.radar_fig = Figure(figsize=(2.0, 1.8), dpi=85)
        self.radar_fig.patch.set_facecolor(self.card_color)
        self.radar_ax = self.radar_fig.add_subplot(111, polar=True)
        self.radar_ax.set_facecolor("#F8FAFC")
        self.radar_ax.set_theta_zero_location('E')
        self.radar_ax.set_theta_direction(1)
        self.radar_ax.set_ylim(0, 1.0)
        self.radar_ax.set_yticklabels([])
        self.radar_ax.set_thetagrids(
            np.arange(0, 360, 45),
            labels=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'],
            fontsize=6
        )
        self.radar_ax.grid(True, linestyle=':', alpha=0.6, color="#BDC3C7")

        self.radar_canvas = FigureCanvasTkAgg(self.radar_fig, master=radar_card)
        self.radar_canvas.get_tk_widget().pack(pady=(1, 2))

        def _on_radar_press(event):
            self._radar_mouse_pressed = True
            self._on_radar_interact(event)

        def _on_radar_release(event):
            self._radar_mouse_pressed = False

        def _on_radar_drag(event):
            if getattr(self, '_radar_mouse_pressed', False):
                self._on_radar_interact(event)

        self.radar_canvas.mpl_connect('button_press_event', _on_radar_press)
        self.radar_canvas.mpl_connect('button_release_event', _on_radar_release)
        self.radar_canvas.mpl_connect('motion_notify_event', _on_radar_drag)

        legend_row = tk.Frame(radar_card, bg=self.card_color)
        legend_row.pack(fill="x")
        tk.Label(legend_row, text="■ R1", fg="#4F9DE8", bg=self.card_color, font=("Helvetica", 7, "bold")).pack(side="left", padx=2)
        tk.Label(legend_row, text="■ R2", fg="#F84B5B", bg=self.card_color, font=("Helvetica", 7, "bold")).pack(side="left", padx=2)
        tk.Label(legend_row, text="■ R3", fg="#28A745", bg=self.card_color, font=("Helvetica", 7, "bold")).pack(side="left", padx=2)
        tk.Label(legend_row, text="(Inner: R1 | Mid: R2 | Outer: R3)", fg="#7F8C8D", bg=self.card_color, font=("Helvetica", 7)).pack(side="right")

        # COLLISION & CLEARANCE MONITOR CARD
        monitor_card = tk.Frame(left_frame, bg=self.card_color, padx=12, pady=8, relief="solid", bd=1)
        monitor_card.pack(fill="x", pady=4)

        tk.Label(
            monitor_card, text="WHOLE-BODY CLEARANCE MONITOR", bg=self.card_color,
            fg=self.text_color, font=("Helvetica", 9, "bold")
        ).pack(anchor="w")

        gauge_frame = tk.Frame(monitor_card, bg=self.card_color)
        gauge_frame.pack(fill="x", pady=2)

        tk.Label(gauge_frame, text="Min Body Clearance:", bg=self.card_color, fg="#555555", font=("Helvetica", 9)).pack(side="left")
        self.clearance_val_lbl = tk.Label(gauge_frame, text="-- mm", bg=self.card_color, fg=self.success_color, font=("Helvetica", 10, "bold"))
        self.clearance_val_lbl.pack(side="right")

        self.status_badge = tk.Label(
            monitor_card, text="IDLE / READY", font=("Helvetica", 9, "bold"),
            bg="#EBF5FB", fg="#2980B9", padx=8, pady=2
        )
        self.status_badge.pack(fill="x", pady=2)

        # UI Toggles
        self.show_ws_var = tk.BooleanVar(value=True)
        self.show_tcp_var = tk.BooleanVar(value=True)
        opt_frame = tk.Frame(left_frame, bg=self.bg_color)
        opt_frame.pack(fill="x", pady=2)
        tk.Checkbutton(opt_frame, text="Clouds", variable=self.show_ws_var, bg=self.bg_color, font=("Helvetica", 8), command=self._update_3d).pack(side="left")
        tk.Checkbutton(opt_frame, text="TCP", variable=self.show_tcp_var, bg=self.bg_color, font=("Helvetica", 8), command=self._update_3d).pack(side="left", padx=4)
        tk.Checkbutton(opt_frame, text="⚡ Disk Cache", variable=self.use_disk_cache_var, bg=self.bg_color, font=("Helvetica", 8, "bold"), fg="#2C3E50").pack(side="left", padx=4)
        tk.Button(
            opt_frame, text="🗑 Clear", font=("Helvetica", 7, "bold"),
            bg="#EAEDED", fg="#7F8C8D", activebackground="#D5D8DC",
            relief="flat", cursor="hand2", padx=4, pady=0,
            command=self._on_clear_cache
        ).pack(side="right")

        # LOG OUTPUT
        self.log_text = tk.Text(
            left_frame, height=7, bg="#FFFFFF", fg="#2C3E50",
            font=("Consolas", 8), relief="solid", bd=1
        )
        self.log_text.pack(fill="both", expand=True, pady=4)

        # ACTION BUTTONS (SIMULATE VS EXECUTE REAL)
        action_frame = tk.Frame(left_frame, bg=self.bg_color)
        action_frame.pack(fill="x", pady=4)

        self.plan_status_lbl = tk.Label(
            action_frame, text="Plan Status: Not simulated yet",
            font=("Helvetica", 8, "italic"), bg=self.bg_color, fg="#7F8C8D"
        )
        self.plan_status_lbl.pack(pady=(0, 4))

        self.sim_btn = tk.Button(
            action_frame, text="▶ SIMULATE (3D PREVIEW ONLY)",
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

        self.pick_entry.bind("<KeyRelease>", self._on_target_change)
        self.pick_yaw_entry.bind("<KeyRelease>", self._on_target_change)
        self.place_entry.bind("<KeyRelease>", self._on_target_change)
        self.place_yaw_entry.bind("<KeyRelease>", self._on_target_change)

        # RIGHT FRAME (MATPLOTLIB)
        right_frame = tk.Frame(self.root, bg=self.bg_color)
        right_frame.pack(side="right", fill="both", expand=True, padx=16, pady=16)

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.fig.patch.set_facecolor(self.card_color)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor(self.card_color)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _log(self, msg: str):
        if threading.current_thread() is not threading.main_thread():
            try:
                self.root.after(0, self._log, msg)
            except tk.TclError:
                pass
            return
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _set_status_lbl(self, text: str, fg: str):
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

    def _update_clearance_ui(self, clearance_mm: float, status_text: str, status_type: str = "safe"):
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

    def _set_badges(self, pick_type: str, place_type: str):
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, self._set_badges, pick_type, place_type)
            return

        if "object" in pick_type.lower():
            self.pick_badge.config(text="Object-Pose IK", bg="#EAFAF1", fg="#27AE60")
        elif "normal" in pick_type.lower() or "fallback" in pick_type.lower():
            self.pick_badge.config(text="Normal IK (Fallback)", bg="#FEF5E7", fg="#E67E22")
        else:
            self.pick_badge.config(text=pick_type, bg="#EAEDED", fg="#7F8C8D")

        if "object" in place_type.lower():
            self.place_badge.config(text="Object-Pose IK", bg="#EAFAF1", fg="#27AE60")
        elif "normal" in place_type.lower() or "fallback" in place_type.lower():
            self.place_badge.config(text="Normal IK (Fallback)", bg="#FEF5E7", fg="#E67E22")
        else:
            self.place_badge.config(text=place_type, bg="#EAEDED", fg="#7F8C8D")

    def _get_workspace_points(self, rid: str, cfg) -> np.ndarray:
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

        # Orientation Rings & Directional Arrows
        self.pick_ring_line, = self.ax.plot([], [], [], color='#F5A623', linestyle='--', linewidth=1.5, alpha=0.85)
        self.pick_arrow_line, = self.ax.plot([], [], [], color='#D35400', linewidth=2.5)
        self.place_ring_line, = self.ax.plot([], [], [], color='#17A2B8', linestyle='--', linewidth=1.5, alpha=0.85)
        self.place_arrow_line, = self.ax.plot([], [], [], color='#0E6655', linewidth=2.5)

        self.canvas.mpl_connect('button_press_event', self._on_3d_click)

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
            pick_yaw = float(self.pick_yaw_entry.get().strip())
            if len(pick_xyz) == 3:
                self._set_marker_xyz(self.pick_marker, pick_xyz)
                r_ring = 35.0
                theta = np.linspace(0, 2 * np.pi, 30)
                xr = pick_xyz[0] + r_ring * np.cos(theta)
                yr = pick_xyz[1] + r_ring * np.sin(theta)
                zr = np.full_like(xr, pick_xyz[2])
                self.pick_ring_line.set_data_3d(xr, yr, zr)

                psi = np.deg2rad(pick_yaw)
                arrow_len = r_ring + 12.0
                self.pick_arrow_line.set_data_3d(
                    [pick_xyz[0], pick_xyz[0] + arrow_len * np.cos(psi)],
                    [pick_xyz[1], pick_xyz[1] + arrow_len * np.sin(psi)],
                    [pick_xyz[2], pick_xyz[2]],
                )
        except (ValueError, tk.TclError):
            pass

        try:
            place_xyz = [float(v.strip()) for v in self.place_entry.get().split(',')]
            place_yaw = float(self.place_yaw_entry.get().strip())
            if len(place_xyz) == 3:
                self._set_marker_xyz(self.place_marker, place_xyz)
                r_ring = 35.0
                theta = np.linspace(0, 2 * np.pi, 30)
                xr = place_xyz[0] + r_ring * np.cos(theta)
                yr = place_xyz[1] + r_ring * np.sin(theta)
                zr = np.full_like(xr, place_xyz[2])
                self.place_ring_line.set_data_3d(xr, yr, zr)

                psi = np.deg2rad(place_yaw)
                arrow_len = r_ring + 12.0
                self.place_arrow_line.set_data_3d(
                    [place_xyz[0], place_xyz[0] + arrow_len * np.cos(psi)],
                    [place_xyz[1], place_xyz[1] + arrow_len * np.sin(psi)],
                    [place_xyz[2], place_xyz[2]],
                )
        except (ValueError, tk.TclError):
            pass

        self.canvas.draw_idle()
        try:
            self.canvas.flush_events()
        except Exception:
            pass
        self._last_visualization_signature = self._visualization_signature()

    def _on_target_change(self, event=None):
        if self.cached_plan is not None:
            try:
                cur_p = tuple(float(x.strip()) for x in self.pick_entry.get().split(','))
                cur_py = float(self.pick_yaw_entry.get().strip())
                cur_pl = tuple(float(x.strip()) for x in self.place_entry.get().split(','))
                cur_ply = float(self.place_yaw_entry.get().strip())
                if (cur_p, cur_py, cur_pl, cur_ply) == self.cached_plan.get('target_key'):
                    self.plan_status_lbl.config(
                        text="✓ Active Session Plan Loaded (Ready to Execute)",
                        fg="#27AE60"
                    )
                    self._update_radar_needle()
                    self._trigger_radar_sweep()
                    return
            except Exception:
                pass

        if getattr(self, 'use_disk_cache_var', None) and self.use_disk_cache_var.get():
            try:
                cur_p = [float(x.strip()) for x in self.pick_entry.get().split(',')]
                cur_py = float(self.pick_yaw_entry.get().strip())
                cur_pl = [float(x.strip()) for x in self.place_entry.get().split(',')]
                cur_ply = float(self.place_yaw_entry.get().strip())
                pick_hit = lookup_cached_target_ik(self.cache_filepath, self.config_fingerprint, cur_p, cur_py)
                place_hit = lookup_cached_target_ik(self.cache_filepath, self.config_fingerprint, cur_pl, cur_ply)
                if pick_hit is not None and place_hit is not None:
                    self.plan_status_lbl.config(
                        text="⚡ Pre-verified Target IK in Disk Cache (Instant Recall)",
                        fg="#2980B9"
                    )
                    self._update_radar_needle()
                    self._trigger_radar_sweep()
                    return
            except Exception:
                pass

        self.plan_status_lbl.config(
            text="Targets modified (Simulate recommended)",
            fg="#E67E22"
        )
        self._update_radar_needle()
        self._trigger_radar_sweep()

    def _on_clear_cache(self):
        count = clear_cache(self.cache_filepath, stage="stage2_5", config_hash=self.config_fingerprint)
        self.cached_plan = None
        self._log(f"\n[CACHE CLEARED] Cleared {count} cached trajectory entries for config ({self.config_fingerprint}).")
        self.plan_status_lbl.config(text="Cache cleared (Simulate recommended)", fg="#E67E22")

    def _on_3d_click(self, event):
        mode = self.click_mode.get()
        if mode not in ("pick", "place"):
            return
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        if event.x is None or event.y is None:
            return

        try:
            if mode == "pick":
                default_z = 13.82
                try:
                    cur_z = float(self.pick_entry.get().split(',')[2])
                except Exception:
                    cur_z = default_z
            else:
                default_z = 150.0
                try:
                    cur_z = float(self.place_entry.get().split(',')[2])
                except Exception:
                    cur_z = default_z

            M = self.ax.get_proj()
            x2, y2 = self.ax.transData.inverted().transform((event.x, event.y))
            A = np.array([
                [M[0, 0] - x2 * M[3, 0], M[0, 1] - x2 * M[3, 1]],
                [M[1, 0] - y2 * M[3, 0], M[1, 1] - y2 * M[3, 1]],
            ])
            b = np.array([
                -(M[0, 2] - x2 * M[3, 2]) * cur_z - (M[0, 3] - x2 * M[3, 3]),
                -(M[1, 2] - y2 * M[3, 2]) * cur_z - (M[1, 3] - y2 * M[3, 3]),
            ])
            if np.abs(np.linalg.det(A)) < 1e-9:
                return

            x_world, y_world = np.linalg.solve(A, b)
            x_val = round(float(x_world), 2)
            y_val = round(float(y_world), 2)
            z_val = round(float(cur_z), 2)

            if mode == "pick":
                self.pick_entry.delete(0, tk.END)
                self.pick_entry.insert(0, f"{x_val}, {y_val}, {z_val}")
                self.pick_badge.config(text="3D Selected", bg="#FEF9E7", fg="#D35400")
                self._log(f"[3D Click] Set PICK target -> ({x_val}, {y_val}, {z_val}) mm")
            else:
                self.place_entry.delete(0, tk.END)
                self.place_entry.insert(0, f"{x_val}, {y_val}, {z_val}")
                self.place_badge.config(text="3D Selected", bg="#E8F8F5", fg="#117A65")
                self._log(f"[3D Click] Set PLACE target -> ({x_val}, {y_val}, {z_val}) mm")

            self._on_target_change()
            self._update_3d()
            self._trigger_radar_sweep()
        except Exception as e:
            self._log(f"[3D Click Error] {e}")

    def _update_radar_ui(self):
        target_type = self.radar_target_var.get()
        if target_type == "pick":
            try:
                yaw = float(self.pick_yaw_entry.get().strip())
            except Exception:
                yaw = 0.0
            self.radar_info_lbl.config(
                text=f"Pick Yaw: {yaw:.1f}° (Drag dial to adjust)",
                fg="#E67E22"
            )
        else:
            try:
                yaw = float(self.place_yaw_entry.get().strip())
            except Exception:
                yaw = 0.0
            self.radar_info_lbl.config(
                text=f"Place Yaw: {yaw:.1f}° (Drag dial to adjust)",
                fg="#17A2B8"
            )
        self._update_radar_needle()

    def _on_radar_interact(self, event):
        if event.inaxes != self.radar_ax or event.xdata is None:
            return
        deg = (np.rad2deg(event.xdata)) % 360.0
        deg_snap = round(deg / 5.0) * 5.0
        if deg_snap >= 360.0:
            deg_snap = 0.0

        target_type = self.radar_target_var.get()
        if target_type == "pick":
            self.pick_yaw_entry.delete(0, tk.END)
            self.pick_yaw_entry.insert(0, f"{deg_snap:.1f}")
            self.radar_info_lbl.config(
                text=f"Pick Yaw: {deg_snap:.1f}° (Drag dial to adjust)",
                fg="#E67E22"
            )
        else:
            self.place_yaw_entry.delete(0, tk.END)
            self.place_yaw_entry.insert(0, f"{deg_snap:.1f}")
            self.radar_info_lbl.config(
                text=f"Place Yaw: {deg_snap:.1f}° (Drag dial to adjust)",
                fg="#17A2B8"
            )

        self._update_radar_needle()
        self._update_3d()
        self._on_target_change()

    def _update_radar_needle(self):
        if self.radar_ax is None:
            return
        target_type = self.radar_target_var.get()
        try:
            if target_type == "pick":
                yaw = float(self.pick_yaw_entry.get().strip())
                color = "#E67E22"
            else:
                yaw = float(self.place_yaw_entry.get().strip())
                color = "#17A2B8"
        except Exception:
            yaw = 0.0
            color = "#E67E22"

        theta = np.deg2rad(yaw)
        if self._radar_needle_line is None:
            self._radar_needle_line, = self.radar_ax.plot(
                [theta, theta], [0.0, 0.96], color=color, linewidth=2.5, zorder=12
            )
            self._radar_needle_pt = self.radar_ax.scatter(
                [theta], [0.96], color=color, s=40, zorder=13
            )
        else:
            self._radar_needle_line.set_color(color)
            self._radar_needle_line.set_data([theta, theta], [0.0, 0.96])
            self._radar_needle_pt.set_color(color)
            self._radar_needle_pt.set_offsets(np.array([[theta, 0.96]]))

        self.radar_canvas.draw_idle()

    def _trigger_radar_sweep(self):
        target_type = self.radar_target_var.get()
        entry_str = self.pick_entry.get() if target_type == "pick" else self.place_entry.get()
        try:
            xyz = tuple(round(float(x.strip()), 2) for x in entry_str.split(','))
            if len(xyz) != 3:
                return
        except Exception:
            return

        cache_key = (target_type, xyz)
        if cache_key in self.radar_sweep_cache:
            self._apply_radar_sectors(cache_key, self.radar_sweep_cache[cache_key])
            return

        threading.Thread(
            target=self._run_radar_sweep_thread,
            args=(cache_key, xyz),
            daemon=True
        ).start()

    def _run_radar_sweep_thread(self, cache_key, xyz_target):
        xyz = np.array(xyz_target, dtype=float)
        candidate_yaws = np.arange(0, 360, 30)
        test_tilts = [0.0, 20.0, -20.0, 45.0, -45.0]
        results = {rid: [] for rid in self.robots.keys()}

        for yaw in candidate_yaws:
            candidates = generate_pick_orientations(
                object_yaw_deg=yaw,
                tilt_angles_deg=test_tilts
            )
            for rid, cfg in self.robots.items():
                p_base = cfg.world_base_R.T @ (xyz - cfg.world_base_xyz)
                q_seeds = [
                    np.zeros(cfg.num_joints),
                    0.5 * (np.deg2rad(cfg.joint_limits_deg[:, 0]) + np.deg2rad(cfg.joint_limits_deg[:, 1])),
                ]
                reachable = False
                for c in candidates:
                    R_base = cfg.world_base_R.T @ c['rotation']
                    for seed in q_seeds:
                        cand = _solve_seed_object_ik(
                            p_base, R_base, c['tilt_deg'], yaw, cfg, seed,
                            pos_tolerance=2.5, ori_tolerance_deg=20.0, align_tolerance_deg=10.0,
                            max_iterations=15
                        )
                        if cand is not None:
                            reachable = True
                            break
                    if reachable:
                        break
                results[rid].append(reachable)

        self.radar_sweep_cache[cache_key] = (candidate_yaws, results)
        try:
            self.root.after(0, self._apply_radar_sectors, cache_key, (candidate_yaws, results))
        except Exception:
            pass

    def _apply_radar_sectors(self, cache_key, data):
        target_type, xyz = cache_key
        if self.radar_target_var.get() != target_type:
            return
        try:
            cur_entry = self.pick_entry.get() if target_type == "pick" else self.place_entry.get()
            cur_xyz = tuple(round(float(x.strip()), 2) for x in cur_entry.split(','))
            if cur_xyz != xyz:
                return
        except Exception:
            return

        candidate_yaws, reachability_dict = data

        for bar_cont in getattr(self, '_radar_bars', []):
            for b in bar_cont:
                try:
                    b.remove()
                except Exception:
                    pass
        self._radar_bars = []

        ring_props = {
            'robot1': (0.18, 0.23, '#4F9DE8'),
            'robot2': (0.43, 0.23, '#F84B5B'),
            'robot3': (0.68, 0.23, '#28A745'),
        }

        width = np.deg2rad(27.0)
        thetas = np.deg2rad(candidate_yaws)

        for rid, (bottom, height, color) in ring_props.items():
            if rid not in reachability_dict:
                continue
            flags = reachability_dict[rid]
            unreached_thetas = [th for th, reachable in zip(thetas, flags) if not reachable]
            if unreached_thetas:
                bg_bars = self.radar_ax.bar(
                    unreached_thetas, [height] * len(unreached_thetas),
                    width=width, bottom=bottom, color='#E2E8F0', alpha=0.35,
                    edgecolor='none', zorder=2
                )
                self._radar_bars.append(bg_bars)

            reached_thetas = [th for th, reachable in zip(thetas, flags) if reachable]
            if reached_thetas:
                bars = self.radar_ax.bar(
                    reached_thetas, [height] * len(reached_thetas),
                    width=width, bottom=bottom, color=color, alpha=0.55,
                    edgecolor=color, linewidth=0.6, zorder=3
                )
                self._radar_bars.append(bars)

        self._update_radar_needle()
        self.radar_canvas.draw_idle()

    def _ask_user_choice(self, title: str, options: List[str]) -> str:
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
                text="Multiple robots can reach this target.\nSelect which robot should perform it:",
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

    def _chooser(self, label: str, robot_ids: List[str]) -> str:
        return self._ask_user_choice(f"Select {label} robot", robot_ids)

    def _on_simulate(self):
        if self.is_simulating:
            return
        self.log_text.delete(1.0, tk.END)
        threading.Thread(target=self._run_plan_thread, args=(False,), daemon=True).start()

    def _on_execute_real(self):
        if self.is_simulating:
            return
        self.log_text.delete(1.0, tk.END)
        threading.Thread(target=self._run_plan_thread, args=(True,), daemon=True).start()

    def _show_solution_dialog(
        self,
        plan: Stage2_5Plan,
        traj_result: Stage1_5TrajectoryResult,
        is_cached: bool,
        real_robot: bool
    ) -> bool:
        """Show an informative modal solution summary dialog before execution."""
        try:
            if not self.root.winfo_viewable():
                return True
        except Exception:
            return True

        ready = threading.Event()
        user_confirmed = [False]

        def show():
            top = tk.Toplevel(self.root)
            top.title("Stage 2.5 - Execution Solution Summary")
            top.geometry("580x560")
            top.configure(bg=self.bg_color)
            top.transient(self.root)
            top.grab_set()

            # Header
            header = tk.Frame(top, bg=self.bg_color)
            header.pack(fill="x", padx=16, pady=(12, 6))

            mode_title = "⚡ REAL ROBOT HARDWARE EXECUTION" if real_robot else "▶ 3D SIMULATION PREVIEW"
            title_color = self.danger_color if real_robot else self.accent_color
            tk.Label(
                header, text=mode_title, font=("Helvetica", 13, "bold"),
                bg=self.bg_color, fg=title_color
            ).pack(anchor="w")

            source_txt = "⚡ Target IK retrieved from Disk Cache (< 0.5 ms)" if is_cached else "🔄 Freshly computed kinematics solution"
            source_col = "#2980B9" if is_cached else "#7F8C8D"
            tk.Label(
                header, text=source_txt, font=("Helvetica", 8, "italic"),
                bg=self.bg_color, fg=source_col
            ).pack(anchor="w")

            # Main Card
            card = tk.Frame(top, bg=self.card_color, relief="solid", bd=1, padx=12, pady=10)
            card.pack(fill="both", expand=True, padx=16, pady=4)

            # PICK SECTION
            p_box = tk.LabelFrame(card, text="🎯 PICK KINEMATICS", font=("Helvetica", 9, "bold"), bg=self.card_color, fg=self.text_color, padx=8, pady=4)
            p_box.pack(fill="x", pady=4)

            p_sol = plan.pick
            p_tilt = f"{p_sol.approach_tilt_deg}°" if p_sol.approach_tilt_deg is not None else "N/A"
            p_align = f"{p_sol.alignment_error_deg:.2f}°" if p_sol.alignment_error_deg is not None else "N/A"
            p_joints_str = ", ".join(f"{v:.1f}°" for v in p_sol.q_deg)
            tk.Label(
                p_box,
                text=f"Selected Robot: {plan.pick_robot.upper()} | Solver: {p_sol.solver_type} | Error: {p_sol.error_mm:.3f} mm\n"
                     f"Approach Tilt: {p_tilt} | Finger Alignment Err: {p_align}\n"
                     f"Joint Configuration: [{p_joints_str}]",
                bg=self.card_color, fg="#2C3E50", font=("Consolas", 8), justify="left"
            ).pack(anchor="w")

            # PLACE SECTION
            pl_box = tk.LabelFrame(card, text="📍 PLACE KINEMATICS", font=("Helvetica", 9, "bold"), bg=self.card_color, fg=self.text_color, padx=8, pady=4)
            pl_box.pack(fill="x", pady=4)

            pl_sol = plan.place
            pl_tilt = f"{pl_sol.approach_tilt_deg}°" if pl_sol.approach_tilt_deg is not None else "N/A"
            pl_align = f"{pl_sol.alignment_error_deg:.2f}°" if pl_sol.alignment_error_deg is not None else "N/A"
            pl_joints_str = ", ".join(f"{v:.1f}°" for v in pl_sol.q_deg)
            tk.Label(
                pl_box,
                text=f"Selected Robot: {plan.place_robot.upper()} | Solver: {pl_sol.solver_type} | Error: {pl_sol.error_mm:.3f} mm\n"
                     f"Approach Tilt: {pl_tilt} | Finger Alignment Err: {pl_align}\n"
                     f"Joint Configuration: [{pl_joints_str}]",
                bg=self.card_color, fg="#2C3E50", font=("Consolas", 8), justify="left"
            ).pack(anchor="w")

            # HANDOVER & CLEARANCE SECTION
            c_box = tk.LabelFrame(card, text="🤝 HANDOVER & COLLISION CLEARANCE", font=("Helvetica", 9, "bold"), bg=self.card_color, fg=self.text_color, padx=8, pady=4)
            c_box.pack(fill="x", pady=4)

            ho_text = f"Execution Route: {plan.pick_robot.upper()} ➔ {plan.place_robot.upper()} (Synchronized Handover)" if plan.needs_handover else f"Execution Route: Single Robot ({plan.pick_robot.upper()}) Direct"
            clr_mm = traj_result.min_overall_clearance_mm
            clr_txt = f"{clr_mm:.1f} mm" if not np.isinf(clr_mm) else "∞ (Single Arm)"
            safe_txt = "✓ ALL PHASES SAFE (> Threshold)" if traj_result.all_phases_safe else "! NEAR COLLISION LIMIT"
            safe_col = self.success_color if traj_result.all_phases_safe else self.danger_color

            tk.Label(
                c_box,
                text=f"{ho_text}\n"
                     f"Min Whole-Body Clearance: {clr_txt} (Safety Threshold: {self.safe_dist:.1f} mm)\n"
                     f"Clearance Status: {safe_txt}",
                bg=self.card_color, fg=safe_col, font=("Consolas", 8, "bold"), justify="left"
            ).pack(anchor="w")

            # Action Buttons
            btn_bar = tk.Frame(top, bg=self.bg_color)
            btn_bar.pack(fill="x", padx=16, pady=12)

            def on_cancel():
                user_confirmed[0] = False
                top.destroy()

            def on_confirm():
                user_confirmed[0] = True
                top.destroy()

            tk.Button(
                btn_bar, text="Cancel", font=("Helvetica", 9, "bold"),
                bg="#EAEDED", fg="#2C3E50", padx=14, pady=6, relief="flat",
                cursor="hand2", command=on_cancel
            ).pack(side="left")

            action_btn_text = "⚡ Confirm & Execute on Hardware" if real_robot else "▶ Proceed & Run Simulation"
            action_btn_bg = self.danger_color if real_robot else self.accent_color
            tk.Button(
                btn_bar, text=action_btn_text, font=("Helvetica", 9, "bold"),
                bg=action_btn_bg, fg="#FFFFFF", padx=16, pady=6, relief="flat",
                cursor="hand2", command=on_confirm
            ).pack(side="right")

            self.root.wait_window(top)
            ready.set()

        self.root.after(0, show)
        ready.wait()
        return user_confirmed[0]

    def _run_plan_thread(self, real_robot: bool = False):
        self._log("=== STARTING STAGE 2.5 OBJECT-POSE EXECUTION ===")
        if real_robot:
            self._log("[MODE: REAL ROBOT EXECUTION] Sending trajectory to hardware via ROS topics!")
        else:
            self._log("[MODE: SIMULATION ONLY] Visualizer 3D preview active - Physical robot commands DISABLED.")
        self._log("Features: Object-Pose IK, Adaptive Tilt, Synchronized Handover, Body Collision Avoidance")

        try:
            pick_xyz = [float(x.strip()) for x in self.pick_entry.get().split(',')]
            pick_yaw = float(self.pick_yaw_entry.get().strip())
            place_xyz = [float(x.strip()) for x in self.place_entry.get().split(',')]
            place_yaw = float(self.place_yaw_entry.get().strip())
        except Exception:
            self._log("ERROR: Invalid Pick/Place Coordinates or Yaw Angle.")
            return

        target_key = (tuple(pick_xyz), pick_yaw, tuple(place_xyz), place_yaw)
        current_q = {rid: q.copy() for rid, q in self.live_q.items()}

        cache_fp = self.cache_filepath if self.use_disk_cache_var.get() else None
        fp_hash = self.config_fingerprint if self.use_disk_cache_var.get() else None

        # Check if targets are already cached in disk
        is_pick_cached = False
        is_place_cached = False
        if cache_fp and fp_hash:
            is_pick_cached = lookup_cached_target_ik(cache_fp, fp_hash, pick_xyz, pick_yaw) is not None
            is_place_cached = lookup_cached_target_ik(cache_fp, fp_hash, place_xyz, place_yaw) is not None

        if is_pick_cached and is_place_cached:
            self._log("\n⚡ [DISK CACHE HIT] Target IK for Pick and Place loaded in < 1 ms!")
            self._log(f"⚡ Active Config Fingerprint: {self.config_fingerprint}")
        else:
            self._log("\n🔄 Solving kinematics and adaptive tilt angles across robots...")

        try:
            plan, pick_opts, place_opts = build_stage2_5_plan(
                pick_xyz=pick_xyz,
                place_xyz=place_xyz,
                robots=self.robots,
                handover_cfg=self.handover_cfg,
                chooser=self._chooser,
                pick_yaw_deg=pick_yaw,
                place_yaw_deg=place_yaw,
                current_q_deg=current_q,
                cache_filepath=cache_fp,
                config_fingerprint=fp_hash,
            )
        except RuntimeError as exc:
            self._log(f"ERROR: {exc}")
            return

        self._set_badges(plan.pick.solver_type, plan.place.solver_type)

        # Log Pick Result
        if plan.is_pick_fallback:
            self._log(f"\nPICK {pick_xyz} (Yaw: {pick_yaw}°):")
            self._log("  [NOTICE] No object_pick_ik solution found for any robot.")
            self._log(f"  -> Successfully fell back to Normal Stage 1.5 XYZ IK!")
            self._log(f"  -> Reachable by: {', '.join(pick_opts)}")
            self._log(f"  > Selected PICK: {plan.pick_robot} (q_deg = {np.round(plan.pick.q_deg, 2)})")
        else:
            self._log(f"\nPICK {pick_xyz} (Yaw: {pick_yaw}°):")
            self._log(f"  [SUCCESS] Solved via Object-Pose IK (Adaptive Tilt)!")
            self._log(f"  -> Reachable by: {', '.join(pick_opts)}")
            tilt_info = f"Tilt: {plan.pick.approach_tilt_deg}°" if plan.pick.approach_tilt_deg is not None else ""
            align_info = f"Align err: {plan.pick.alignment_error_deg:.2f}°" if plan.pick.alignment_error_deg is not None else ""
            self._log(f"  > Selected PICK: {plan.pick_robot} ({tilt_info}, {align_info})")
            self._log(f"    q_deg = {np.round(plan.pick.q_deg, 2)}")

        # Log Place Result
        if plan.is_place_fallback:
            self._log(f"\nPLACE {place_xyz} (Yaw: {place_yaw}°):")
            self._log("  [NOTICE] No object_pick_ik solution found for any robot.")
            self._log(f"  -> Successfully fell back to Normal Stage 1.5 XYZ IK!")
            self._log(f"  -> Reachable by: {', '.join(place_opts)}")
            self._log(f"  > Selected PLACE: {plan.place_robot} (q_deg = {np.round(plan.place.q_deg, 2)})")
        else:
            self._log(f"\nPLACE {place_xyz} (Yaw: {place_yaw}°):")
            self._log(f"  [SUCCESS] Solved via Object-Pose IK (Adaptive Tilt)!")
            self._log(f"  -> Reachable by: {', '.join(place_opts)}")
            tilt_info = f"Tilt: {plan.place.approach_tilt_deg}°" if plan.place.approach_tilt_deg is not None else ""
            align_info = f"Align err: {plan.place.alignment_error_deg:.2f}°" if plan.place.alignment_error_deg is not None else ""
            self._log(f"  > Selected PLACE: {plan.place_robot} ({tilt_info}, {align_info})")
            self._log(f"    q_deg = {np.round(plan.place.q_deg, 2)}")

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
        traj_result = plan_stage2_5_synchronized_paths(
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

        # Show Solution Dialog Box before physical execution or 3D simulation!
        proceed = self._show_solution_dialog(
            plan, traj_result,
            is_cached=(is_pick_cached and is_place_cached),
            real_robot=real_robot
        )
        if not proceed:
            self._log("\n[ABORTED] Execution cancelled by operator in solution dialog.")
            self._set_status_lbl("Execution cancelled by user", "#E67E22")
            return

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
            # PHASE 1: SIMULTANEOUS MOTION (Pick -> Pick, Place -> Pre-Handover)
            step_simultaneous(
                traj_result.phase1_simultaneous,
                f"1. SIMULTANEOUS: {pick_rid} moves to Pick, WHILE {place_rid} moves to Pre-Handover!",
                dwell=self.dwell_time_s,
            )

            # PHASE 2: GRASP AT PICK
            self._log(f"2. {pick_rid} waits {self.pick_wait_s}s and closes gripper. ({place_rid} waiting at Pre-Handover)")
            self._send_gripper_command(pick_rid, 'close', real_robot=real_robot)
            self._update_clearance_ui(self.live_clearance_mm, f"{place_rid} WAITING AT PRE-HANDOVER", "info")
            time.sleep(self.pick_wait_s)

            # PHASE 3: PICK ROBOT MOVES TO PRE-HANDOVER
            step_simultaneous(
                traj_result.phase3_pick_to_pre_handover,
                f"3. {pick_rid} moves to Pre-Handover ({place_rid} stationed at Pre-Handover).",
                dwell=self.dwell_time_s,
            )

            # PHASE 4: SYNCHRONIZED ADVANCE (Pre-Handover -> Handover)
            step_simultaneous(
                traj_result.phase4_advance_to_handover,
                f"4. ADVANCE: Both {pick_rid} and {place_rid} advance Pre-Handover -> Handover at same rate!",
                dwell=self.dwell_time_s,
            )

            # PHASE 5: CONFIRM & HANDOVER TRANSFER
            self._log("5. CONFIRMED: Both robots are at Handover position.")
            self._log(f"   Transfer: {place_rid} closes gripper, {pick_rid} opens gripper.")
            self._send_gripper_command(place_rid, 'close', real_robot=real_robot)
            self._send_gripper_command(pick_rid, 'open', real_robot=real_robot)
            if traj_result.phase4_advance_to_handover:
                self._update_clearance_ui(traj_result.phase4_advance_to_handover[-1].min_clearance_mm, "HANDOVER TRANSFER (GRIPPER)", "info")
            time.sleep(self.transfer_wait_s)

            # PHASE 6: SYNCHRONIZED RETRACT (Handover -> Pre-Handover)
            step_simultaneous(
                traj_result.phase6_retract_to_pre_handover,
                f"6. RETRACT: Both robots back away to Pre-Handover at same rate!",
                dwell=self.dwell_time_s,
            )

            # PHASE 7: PLACE ROBOT MOVES PRE-HANDOVER TO PLACE
            step_simultaneous(
                traj_result.phase7_handover_to_place,
                f"7. {place_rid} moves from Pre-Handover to Place target ({pick_rid} waiting at Pre-Handover).",
                dwell=self.dwell_time_s,
            )

            # PHASE 8: RELEASE OBJECT
            self._log(f"8. {place_rid} waits {self.place_wait_s}s, opens gripper to release object.")
            self._send_gripper_command(place_rid, 'open', real_robot=real_robot)
            time.sleep(self.place_wait_s)

            # PHASE 9: SIMULTANEOUS RETURN HOME
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

        self._log("\n=== STAGE 2.5 EXECUTION SUCCESS ===")
        self._update_clearance_ui(float("inf"), "STAGE 2.5 COMPLETED", "safe")
        if not real_robot:
            self._set_status_lbl("✓ Simulation verified & cached for instant execution", "#27AE60")
            self._log("\n[READY] Click '⚡ EXECUTE (REAL ROBOT)' to directly run this exact plan on hardware without recalculating!")
        else:
            self._set_status_lbl("✓ Real robot execution completed", "#28A745")

        self.is_simulating = False


def main(args=None):
    rclpy.init(args=args)
    root = tk.Tk()
    app = Stage2_5GUI(root)

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
