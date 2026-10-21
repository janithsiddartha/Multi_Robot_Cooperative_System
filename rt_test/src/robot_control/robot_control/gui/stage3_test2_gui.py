"""Stage 3 Test 2 GUI - Streamlined Multi-Robot Handover Control Station.

Key Enhancements over Stage 3 Test:
1. Strategy is permanently and transparently locked to TOP_OF_WORKSPACE (descending Z).
   - Candidate strategy selector is completely removed from the UI.
2. Graspability Radar is completely removed:
   - Yaw angles are directly specified in the Pick & Place coordinate cards.
   - All polar matplotlib dial overhead and background sweep threads are eliminated.
3. System Log & Telemetry Dialog Box is greatly enlarged:
   - Reclaims the full ~240px vertical space previously consumed by the radar.
   - Includes dedicated smooth vertical scrolling for comprehensive real-time telemetry.
4. Uses dedicated `auto_handover_poses_test2.json` cache and `stage3_test2_solver`.
5. Retains the full interactive robot chooser, 3D visualizer, and non-blocking collision monitoring.
"""

import os
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, List, Optional
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from ament_index_python.packages import get_package_share_directory
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from robot_control.gui.stage3_gui import Stage3GUI
from robot_control.handover.stage1_5 import validate_execution_safety_gate
from robot_control.handover.stage3_test2 import (
    Stage3Test2Plan,
    build_stage3_test2_plan,
    plan_stage3_test2_paths,
)
from robot_control.handover.stage3_test2_solver import (
    AutoHandoverTest2Pose,
    compute_all_test2_handover_poses,
    load_test2_handover_cache,
    resolve_test2_handover_cache_path,
)
from robot_control.handover.stage1 import load_stage1_robots
from robot_control.handover.trajectory_cache import (
    compute_config_fingerprint,
    lookup_cached_target_ik,
    resolve_cache_filepath,
)


class Stage3Test2GUI(Stage3GUI):
    """Stage 3 Test 2 Streamlined Control Station GUI Node."""

    def __init__(self, root: tk.Tk):
        # We do NOT run Stage3GUI.__init__ directly because it calls Stage3GUI._build_ui()
        # which builds the radar dial. Instead, we initialize the Node and data attributes
        # and invoke our streamlined _build_ui().
        Node.__init__(self, 'stage3_test2_gui_node')
        self.root = root

        # Styles and Color Tokens (matching Stage 3 design system)
        self.bg_color = "#F4F6F9"
        self.card_color = "#FFFFFF"
        self.text_color = "#2C3E50"
        self.accent_color = "#2980B9"
        self.success_color = "#27AE60"
        self.warning_color = "#F39C12"
        self.danger_color = "#E74C3C"
        self.purple_color = "#8E44AD"

        self.root.title("Stage 3 Test 2 - Multi-Robot Handover Control Station (Top-of-Workspace Only)")
        self.root.geometry("1480x950")
        self.root.configure(bg=self.bg_color)

        self.dwell_time_s = 0.5
        self.pick_wait_s = 1.0
        self.step_interval_s = 0.05
        self.safe_dist = 50.0
        self.trajectory_steps = 25

        # Initialize configurations and paths
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
        self.auto_handover_cache_path = resolve_test2_handover_cache_path()
        self.config_fingerprint = compute_config_fingerprint(self.robots, self.handover_cfg)

        self.use_disk_cache_var = tk.BooleanVar(value=True)
        self.handover_mode_var = tk.StringVar(value="auto")  # 'auto' or 'manual'
        self.click_mode = tk.StringVar(value="orbit")
        self.selected_ho_pair = tk.StringVar(value="robot1_robot2")

        exec_cfg = self.handover_cfg.get('execution', {})
        timings = exec_cfg.get('timings', {})
        self.pick_wait_s = float(timings.get('pick_grasp_wait_s', 2.0))
        self.transfer_wait_s = float(timings.get('handover_transfer_wait_s', 1.0))
        self.place_wait_s = float(timings.get('place_release_wait_s', 2.0))
        self.step_interval_s = float(timings.get('step_interval_s', 0.05))
        self.dwell_time_s = float(timings.get('dwell_time_s', 0.5))
        self.trajectory_steps = int(exec_cfg.get('trajectory', {}).get('steps', 30))

        # Workspace Points & Scene Limits
        self.workspace_points = {
            rid: self._get_workspace_points(rid, cfg)
            for rid, cfg in self.robots.items()
        }
        self.scene_limits = self._compute_scene_limits()

        # Auto-handover poses
        self.auto_handover_poses: Dict[str, AutoHandoverTest2Pose] = {}
        self.is_recomputing_handover = False
        self.has_live_q = {rid: False for rid in self.robots.keys()}
        self.is_previewing_handover = False
        self.preview_q = {}
        self._load_initial_auto_handover_poses()

        self.live_q = {rid: np.zeros(cfg.num_joints) for rid, cfg in self.robots.items()}
        self.current_q = {rid: np.zeros(cfg.num_joints) for rid, cfg in self.robots.items()}
        self.is_simulating = False
        self.is_real_execution = False
        self.cached_plan = None
        self.live_clearance_mm = float('inf')

        # Dummy placeholders for radar variables so inherited methods never throw AttributeError
        self.radar_sweep_cache = {}
        self.radar_target_var = tk.StringVar(value="pick")
        self.radar_ax = None
        self.radar_fig = None
        self.radar_canvas = None
        self._radar_bars = []
        self._radar_needle_line = None
        self._radar_needle_pt = None
        self._radar_mouse_pressed = False

        # ROS Publishers & Subscribers
        self._init_ros_communication()

        # Build Streamlined UI (No Radar, Enormous Dialog Box, No Candidate Strategy Selector)
        self._build_ui()
        self._last_visualization_signature = None
        self._init_scene()

        self.root.after(50, self._refresh_visualization)
        self._update_handover_status_card()

    def _init_ros_communication(self):
        """Set up ROS 2 publishers and joint subscriptions."""
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

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

        self.subs = []
        for rid in self.robots.keys():
            sub = self.create_subscription(
                Float64MultiArray,
                f'/{rid}/joint_commands',
                lambda msg, r=rid: self._joint_cb(msg, r),
                10,
            )
            self.subs.append(sub)

    def _load_initial_auto_handover_poses(self):
        """Load test2 auto-handover poses from JSON cache without blocking UI."""
        poses = None
        try:
            if os.path.exists(self.auto_handover_cache_path):
                poses = load_test2_handover_cache(self.auto_handover_cache_path, self.config_fingerprint)
        except Exception as e:
            self.get_logger().warn(f"Initial test2 auto-handover load: {e}")

        # Fallback to test cache if test2 cache not found
        if not poses:
            try:
                from robot_control.handover.stage3_test_solver import (
                    load_test_handover_cache,
                    resolve_test_handover_cache_path,
                )
                t_cache = resolve_test_handover_cache_path()
                if os.path.exists(t_cache):
                    t_poses = load_test_handover_cache(t_cache, self.config_fingerprint)
                    if t_poses:
                        poses = {}
                        for k, p in t_poses.items():
                            poses[k] = AutoHandoverTest2Pose(
                                pair_name=p.pair_name,
                                handover_position_world=p.handover_position_world,
                                standoff_distance_mm=p.standoff_distance_mm,
                                robot_a_id=p.robot_a_id,
                                robot_b_id=p.robot_b_id,
                                robot_a_handover_q_deg=p.robot_a_handover_q_deg,
                                robot_b_handover_q_deg=p.robot_b_handover_q_deg,
                                robot_a_pre_handover_q_deg=p.robot_a_pre_handover_q_deg,
                                robot_b_pre_handover_q_deg=p.robot_b_pre_handover_q_deg,
                                handover_clearance_mm=p.handover_clearance_mm,
                                pre_handover_clearance_mm=p.pre_handover_clearance_mm,
                                error_a_mm=p.error_a_mm,
                                error_b_mm=p.error_b_mm,
                                dot_z=p.dot_z,
                                dot_x=p.dot_x,
                                approach_angle_dev_deg=p.approach_angle_dev_deg,
                                approach_mode=p.approach_mode,
                                candidate_origin="TOP_OF_WORKSPACE",
                                corridor_a_valid=p.corridor_a_valid,
                                corridor_b_valid=p.corridor_b_valid,
                            )
                        self.get_logger().info(f"Loaded {len(poses)} baseline poses from test cache.")
            except Exception as e:
                self.get_logger().warn(f"Fallback to test cache: {e}")

        self.auto_handover_poses = poses if poses is not None else {}
        if self.auto_handover_poses:
            self.get_logger().info(f"Active test2 auto-handover pairs: {list(self.auto_handover_poses.keys())}")

    # =========================================================================
    # STREAMLINED UI SETUP (NO RADAR, EXPANDED LOG TEXT BOX)
    # =========================================================================

    def _build_ui(self):
        left_frame = tk.Frame(self.root, bg=self.bg_color, width=450)
        left_frame.pack(side="left", fill="y", padx=16, pady=16)
        left_frame.pack_propagate(False)

        # Header
        header = tk.Frame(left_frame, bg=self.bg_color)
        header.pack(fill="x", pady=(0, 4))

        title_row = tk.Frame(header, bg=self.bg_color)
        title_row.pack(fill="x")
        tk.Label(
            title_row, text="STAGE 3 TEST 2", font=("Helvetica", 15, "bold"),
            bg=self.bg_color, fg=self.text_color
        ).pack(side="left")
        self.stage_badge = tk.Label(
            title_row, text="TOP-ONLY AUTO HANDOVER", font=("Helvetica", 8, "bold"),
            bg="#27AE60", fg="#FFFFFF", padx=5, pady=1
        )
        self.stage_badge.pack(side="right")

        tk.Label(
            header, text="Top-of-Workspace Kinematics & Streamlined Console",
            font=("Helvetica", 9, "bold"), bg=self.bg_color, fg=self.accent_color
        ).pack(anchor="w")

        # 1. TARGETS CARD (Pick & Place Coordinates + Direct Yaw Entries)
        targets_card = tk.Frame(left_frame, bg=self.card_color, padx=10, pady=5, relief="solid", bd=1)
        targets_card.pack(fill="x", pady=2)

        mode_bar = tk.Frame(targets_card, bg=self.card_color)
        mode_bar.pack(fill="x", pady=(0, 3))

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
            padx=5, pady=1, command=lambda: _set_mode("orbit")
        )
        self.btn_orbit.pack(side="left", padx=2)

        self.btn_click_pick = tk.Button(
            mode_bar, text="🎯 Set Pick", font=("Helvetica", 8, "bold"),
            bg="#EAEDED", fg="#2C3E50", relief="flat", cursor="hand2",
            padx=5, pady=1, command=lambda: _set_mode("pick")
        )
        self.btn_click_pick.pack(side="left", padx=2)

        self.btn_click_place = tk.Button(
            mode_bar, text="📍 Set Place", font=("Helvetica", 8, "bold"),
            bg="#EAEDED", fg="#2C3E50", relief="flat", cursor="hand2",
            padx=5, pady=1, command=lambda: _set_mode("place")
        )
        self.btn_click_place.pack(side="left", padx=2)

        # Pick Target Row
        pick_header = tk.Frame(targets_card, bg=self.card_color)
        pick_header.pack(fill="x")
        tk.Label(
            pick_header, text="PICK TARGET:", bg=self.card_color, fg=self.text_color,
            font=("Helvetica", 8, "bold")
        ).pack(side="left")
        self.pick_badge = tk.Label(
            pick_header, text="Pending", bg="#EAEDED", fg="#7F8C8D",
            font=("Helvetica", 7, "bold"), padx=5, pady=1
        )
        self.pick_badge.pack(side="right")

        pick_inputs = tk.Frame(targets_card, bg=self.card_color)
        pick_inputs.pack(fill="x", pady=(1, 2))
        tk.Label(pick_inputs, text="XYZ (mm):", bg=self.card_color, font=("Helvetica", 8)).grid(row=0, column=0, sticky="w")
        self.pick_entry = tk.Entry(pick_inputs, bg="#F8FAFC", relief="solid", bd=1, font=("Consolas", 8), width=22)
        self.pick_entry.insert(0, "100.0, 100.0, 200.0")
        self.pick_entry.grid(row=0, column=1, padx=(4, 6), sticky="w")

        tk.Label(pick_inputs, text="Yaw (°):", bg=self.card_color, font=("Helvetica", 8)).grid(row=0, column=2, sticky="w")
        self.pick_yaw_entry = tk.Entry(pick_inputs, bg="#F8FAFC", relief="solid", bd=1, font=("Consolas", 8), width=5)
        self.pick_yaw_entry.insert(0, "0.0")
        self.pick_yaw_entry.grid(row=0, column=3, padx=(4, 0), sticky="w")

        # Place Target Row
        place_header = tk.Frame(targets_card, bg=self.card_color)
        place_header.pack(fill="x")
        tk.Label(
            place_header, text="PLACE TARGET:", bg=self.card_color, fg=self.text_color,
            font=("Helvetica", 8, "bold")
        ).pack(side="left")
        self.place_badge = tk.Label(
            place_header, text="Pending", bg="#EAEDED", fg="#7F8C8D",
            font=("Helvetica", 7, "bold"), padx=5, pady=1
        )
        self.place_badge.pack(side="right")

        place_inputs = tk.Frame(targets_card, bg=self.card_color)
        place_inputs.pack(fill="x", pady=(1, 1))
        tk.Label(place_inputs, text="XYZ (mm):", bg=self.card_color, font=("Helvetica", 8)).grid(row=0, column=0, sticky="w")
        self.place_entry = tk.Entry(place_inputs, bg="#F8FAFC", relief="solid", bd=1, font=("Consolas", 8), width=22)
        self.place_entry.insert(0, "0.0, 150.0, 150.0")
        self.place_entry.grid(row=0, column=1, padx=(4, 6), sticky="w")

        tk.Label(place_inputs, text="Yaw (°):", bg=self.card_color, font=("Helvetica", 8)).grid(row=0, column=2, sticky="w")
        self.place_yaw_entry = tk.Entry(place_inputs, bg="#F8FAFC", relief="solid", bd=1, font=("Consolas", 8), width=5)
        self.place_yaw_entry.insert(0, "0.0")
        self.place_yaw_entry.grid(row=0, column=3, padx=(4, 0), sticky="w")

        # (NOTE: GRASPABILITY RADAR IS INTENTIONALLY OMITTED TO RECLAIM SPACE)

        # 2. HANDOVER CONFIGURATION & INSPECTION CARD
        ho_card = tk.Frame(left_frame, bg=self.card_color, padx=10, pady=5, relief="solid", bd=1)
        ho_card.pack(fill="x", pady=2)

        ho_hdr = tk.Frame(ho_card, bg=self.card_color)
        ho_hdr.pack(fill="x")
        tk.Label(
            ho_hdr, text="HANDOVER PAIR SOLUTIONS", bg=self.card_color,
            fg=self.text_color, font=("Helvetica", 8, "bold")
        ).pack(side="left")
        self.ho_status_badge = tk.Label(
            ho_hdr, text="Loading...", bg="#EAEDED", fg="#7F8C8D",
            font=("Helvetica", 7, "bold"), padx=5, pady=1
        )
        self.ho_status_badge.pack(side="right")

        # Mode switch
        mode_radios = tk.Frame(ho_card, bg=self.card_color)
        mode_radios.pack(fill="x", pady=1)

        tk.Radiobutton(
            mode_radios, text="Auto Kinematics (Top-of-WS)", variable=self.handover_mode_var,
            value="auto", bg=self.card_color, font=("Helvetica", 8, "bold"),
            fg=self.success_color, activebackground=self.card_color,
            command=self._on_handover_mode_changed
        ).pack(side="left")

        tk.Radiobutton(
            mode_radios, text="Manual Override", variable=self.handover_mode_var,
            value="manual", bg=self.card_color, font=("Helvetica", 8),
            fg="#7F8C8D", activebackground=self.card_color,
            command=self._on_handover_mode_changed
        ).pack(side="left", padx=8)

        # Approach Mode Badge
        mode_frame = tk.Frame(ho_card, bg=self.card_color)
        mode_frame.pack(fill="x", pady=(1, 2))
        tk.Label(
            mode_frame, text="Approach Mode:", bg=self.card_color,
            fg="#555555", font=("Helvetica", 8, "bold")
        ).pack(side="left")
        self.test_approach_mode_lbl = tk.Label(
            mode_frame, text="DUAL_SYNCHRONIZED", bg="#EAF2F8", fg="#2980B9",
            font=("Consolas", 8, "bold"), padx=5, pady=1
        )
        self.test_approach_mode_lbl.pack(side="left", padx=4)

        # Pair Selector Dropdown
        pair_row = tk.Frame(ho_card, bg=self.card_color)
        pair_row.pack(fill="x", pady=(1, 2))
        tk.Label(
            pair_row, text="Select Pair:", bg=self.card_color,
            fg="#555555", font=("Helvetica", 8, "bold")
        ).pack(side="left", padx=(0, 4))

        self.pair_combobox = ttk.Combobox(
            pair_row,
            textvariable=self.selected_ho_pair,
            values=["robot1_robot2", "robot2_robot3", "robot3_robot1"],
            state="readonly",
            width=18,
            font=("Consolas", 8, "bold")
        )
        self.pair_combobox.pack(side="left", padx=(0, 6))
        self.pair_combobox.bind("<<ComboboxSelected>>", self._on_handover_pair_selected)

        self.btn_reload_ho = tk.Button(
            pair_row, text="🔄 Reload", font=("Helvetica", 7, "bold"),
            bg="#EAEDED", fg="#2C3E50", relief="flat", padx=4, pady=1, cursor="hand2",
            command=self._on_reload_handover_cache
        )
        self.btn_reload_ho.pack(side="left")

        # Shared Target and Mating Info
        self.ho_target_lbl = tk.Label(
            ho_card, text="Shared Target: [-12.5, 87.5, 212.5] mm",
            bg=self.card_color, fg="#2C3E50", font=("Consolas", 8, "bold"), anchor="w"
        )
        self.ho_target_lbl.pack(fill="x")

        self.ho_metrics_lbl = tk.Label(
            ho_card, text="Standoff: 50.0 mm | Clr: 238.1 mm | Dev: 12.4°",
            bg=self.card_color, fg="#555555", font=("Consolas", 7), anchor="w"
        )
        self.ho_metrics_lbl.pack(fill="x")

        # Robot A & Robot B summaries
        self.ho_summary_lbl = tk.Label(
            ho_card, text="R1: q_H=[...]\nR2: q_H=[...]",
            bg="#F8FAFC", fg="#2C3E50", font=("Consolas", 7), justify="left", relief="solid", bd=1, padx=4, pady=2
        )
        self.ho_summary_lbl.pack(fill="x", pady=(1, 2))

        # Action Buttons Row 1: Full Inspector & Recompute
        action_row1 = tk.Frame(ho_card, bg=self.card_color)
        action_row1.pack(fill="x", pady=1)

        self.btn_inspect_ho = tk.Button(
            action_row1, text="🔍 Inspect Pose Details...", font=("Helvetica", 7, "bold"),
            bg="#2980B9", fg="#FFFFFF", relief="flat", padx=5, pady=2, cursor="hand2",
            command=self._show_handover_inspector_dialog
        )
        self.btn_inspect_ho.pack(side="left", padx=(0, 4))

        self.btn_recompute_ho = tk.Button(
            action_row1, text="⚡ Recompute (Top-WS)", font=("Helvetica", 7, "bold"),
            bg="#EAF2F8", fg="#2980B9", relief="flat", padx=4, pady=2, cursor="hand2",
            command=self._on_recompute_test2_handover_clicked
        )
        self.btn_recompute_ho.pack(side="left")

        self.ho_recomp_msg = tk.Label(
            action_row1, text="✓ Cached", bg=self.card_color, fg=self.success_color, font=("Helvetica", 7, "italic")
        )
        self.ho_recomp_msg.pack(side="right")

        # Action Buttons Row 2: 3D Preview Handover / Pre-Handover / Reset
        action_row2 = tk.Frame(ho_card, bg=self.card_color)
        action_row2.pack(fill="x", pady=(1, 1))

        self.btn_preview_ho = tk.Button(
            action_row2, text="👁️ 3D Preview Handover", font=("Helvetica", 7, "bold"),
            bg="#EAEDED", fg="#2C3E50", relief="flat", padx=4, pady=1, cursor="hand2",
            command=lambda: self._preview_handover_in_3d('handover')
        )
        self.btn_preview_ho.pack(side="left", padx=(0, 2))

        self.btn_preview_pre = tk.Button(
            action_row2, text="👁️ Pre-H", font=("Helvetica", 7, "bold"),
            bg="#EAEDED", fg="#2C3E50", relief="flat", padx=4, pady=1, cursor="hand2",
            command=lambda: self._preview_handover_in_3d('pre_handover')
        )
        self.btn_preview_pre.pack(side="left", padx=2)

        self.btn_reset_preview = tk.Button(
            action_row2, text="🏠 Reset", font=("Helvetica", 7, "bold"),
            bg="#EAEDED", fg="#7F8C8D", relief="flat", padx=4, pady=1, cursor="hand2",
            command=self._reset_3d_preview
        )
        self.btn_reset_preview.pack(side="right")

        # 3. COLLISION & CLEARANCE MONITOR CARD
        monitor_card = tk.Frame(left_frame, bg=self.card_color, padx=10, pady=4, relief="solid", bd=1)
        monitor_card.pack(fill="x", pady=2)

        gauge_frame = tk.Frame(monitor_card, bg=self.card_color)
        gauge_frame.pack(fill="x")
        tk.Label(gauge_frame, text="Min Body Clearance:", bg=self.card_color, fg="#555555", font=("Helvetica", 8)).pack(side="left")
        self.clearance_val_lbl = tk.Label(gauge_frame, text="-- mm", bg=self.card_color, fg=self.success_color, font=("Helvetica", 9, "bold"))
        self.clearance_val_lbl.pack(side="right")

        self.status_badge = tk.Label(
            monitor_card, text="IDLE / READY", font=("Helvetica", 8, "bold"),
            bg="#EBF5FB", fg="#2980B9", padx=6, pady=1
        )
        self.status_badge.pack(fill="x", pady=1)

        # 4. UI TOGGLES (Clouds, TCP, Disk Cache, Clear)
        self.show_ws_var = tk.BooleanVar(value=True)
        self.show_tcp_var = tk.BooleanVar(value=True)
        opt_frame = tk.Frame(left_frame, bg=self.bg_color)
        opt_frame.pack(fill="x", pady=1)
        tk.Checkbutton(opt_frame, text="Clouds", variable=self.show_ws_var, bg=self.bg_color, font=("Helvetica", 8), command=self._update_3d).pack(side="left")
        tk.Checkbutton(opt_frame, text="TCP", variable=self.show_tcp_var, bg=self.bg_color, font=("Helvetica", 8), command=self._update_3d).pack(side="left", padx=4)
        tk.Checkbutton(opt_frame, text="⚡ Disk Cache", variable=self.use_disk_cache_var, bg=self.bg_color, font=("Helvetica", 8, "bold"), fg="#2C3E50").pack(side="left", padx=4)
        tk.Button(
            opt_frame, text="🗑 Clear", font=("Helvetica", 7, "bold"),
            bg="#EAEDED", fg="#7F8C8D", activebackground="#D5D8DC",
            relief="flat", cursor="hand2", padx=4, pady=0,
            command=self._on_clear_cache
        ).pack(side="right")

        # 5. ACTION BUTTONS (SIMULATE VS EXECUTE REAL)
        action_frame = tk.Frame(left_frame, bg=self.bg_color)
        action_frame.pack(fill="x", pady=2)

        self.plan_status_lbl = tk.Label(
            action_frame, text="Plan Status: Ready to plan",
            font=("Helvetica", 8, "italic"), bg=self.bg_color, fg="#7F8C8D"
        )
        self.plan_status_lbl.pack(pady=(0, 2))

        self.sim_btn = tk.Button(
            action_frame, text="▶ SIMULATE (3D PREVIEW ONLY)",
            font=("Helvetica", 9, "bold"), bg="#17A2B8", fg="#FFFFFF",
            activebackground="#138496", activeforeground="#FFFFFF",
            relief="flat", cursor="hand2", padx=6, pady=5,
            command=self._on_simulate
        )
        self.sim_btn.pack(fill="x", pady=(0, 3))

        self.exec_btn = tk.Button(
            action_frame, text="⚡ EXECUTE (REAL ROBOT)",
            font=("Helvetica", 9, "bold"), bg="#28A745", fg="#FFFFFF",
            activebackground="#218838", activeforeground="#FFFFFF",
            relief="flat", cursor="hand2", padx=6, pady=5,
            command=self._on_execute_real
        )
        self.exec_btn.pack(fill="x")

        # 6. ENLARGED SYSTEM LOG & TELEMETRY DIALOG BOX
        log_header = tk.Frame(left_frame, bg=self.bg_color)
        log_header.pack(fill="x", pady=(4, 1))
        tk.Label(
            log_header, text="SYSTEM LOG & TELEMETRY:", font=("Helvetica", 8, "bold"),
            bg=self.bg_color, fg="#555555"
        ).pack(side="left")

        log_container = tk.Frame(left_frame, bg="#FFFFFF", relief="solid", bd=1)
        log_container.pack(fill="both", expand=True, pady=(1, 0))

        log_scroll = ttk.Scrollbar(log_container, orient="vertical")
        log_scroll.pack(side="right", fill="y")

        self.log_text = tk.Text(
            log_container, bg="#FFFFFF", fg="#2C3E50",
            font=("Consolas", 8), relief="flat", wrap="word",
            yscrollcommand=log_scroll.set
        )
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.config(command=self.log_text.yview)

        # Bind Entry Changes
        self.pick_entry.bind("<KeyRelease>", self._on_target_change)
        self.pick_yaw_entry.bind("<KeyRelease>", self._on_target_change)
        self.place_entry.bind("<KeyRelease>", self._on_target_change)
        self.place_yaw_entry.bind("<KeyRelease>", self._on_target_change)

        # RIGHT PANEL (FULL 3D MATPLOTLIB VISUALIZER)
        right_frame = tk.Frame(self.root, bg=self.bg_color)
        right_frame.pack(side="right", fill="both", expand=True, padx=16, pady=16)

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.fig.patch.set_facecolor(self.card_color)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor(self.card_color)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # =========================================================================
    # OVERRIDDEN RADAR NO-OPS (Eliminates background sweep threads & overhead)
    # =========================================================================

    def _trigger_radar_sweep(self):
        pass

    def _update_radar_needle(self):
        pass

    def _update_radar_ui(self):
        pass

    def _on_radar_interact(self, event):
        pass

    def _apply_radar_sectors(self, cache_key, data):
        pass

    def _run_radar_sweep_thread(self, cache_key, xyz_target):
        pass

    def _on_target_change(self, event=None):
        """Update 3D markers when coordinates or yaw change."""
        self.cached_plan = None
        self._update_3d()

    # =========================================================================
    # HANDOVER STATUS & 3D PREVIEW
    # =========================================================================

    def _update_handover_status_card(self):
        """Update Handover card UI with approach mode and telemetry."""
        mode = self.handover_mode_var.get()
        pair_name = self.selected_ho_pair.get() if hasattr(self, 'selected_ho_pair') else "robot1_robot2"
        parts = pair_name.split('_') if '_' in pair_name else ["robot1", "robot2"]
        r_a, r_b = parts[0], parts[1]
        inv_key = f"{r_b}_{r_a}"

        if mode == "manual":
            super()._update_handover_status_card()
            if hasattr(self, 'test_approach_mode_lbl'):
                self.test_approach_mode_lbl.config(
                    text="MANUAL (handover.yaml)", bg="#FCF3CF", fg="#B7950B"
                )
            return

        pose = None
        if hasattr(self, 'auto_handover_poses') and self.auto_handover_poses:
            pose = self.auto_handover_poses.get(pair_name) or self.auto_handover_poses.get(inv_key)

        if pose is not None:
            super()._update_handover_status_card()
            mode_str = getattr(pose, 'approach_mode', 'DUAL_SYNCHRONIZED')

            if hasattr(self, 'test_approach_mode_lbl'):
                if mode_str == "DUAL_SYNCHRONIZED":
                    self.test_approach_mode_lbl.config(
                        text="DUAL_SYNCHRONIZED", bg="#E8F8F5", fg="#1E8449"
                    )
                else:
                    self.test_approach_mode_lbl.config(
                        text=mode_str, bg="#FEF9E7", fg="#B7950B"
                    )

            if hasattr(self, 'ho_metrics_lbl'):
                self.ho_metrics_lbl.config(
                    text=f"Standoff: {pose.standoff_distance_mm:.1f} mm | Clr: {pose.handover_clearance_mm:.1f} mm | Dev: {pose.approach_angle_dev_deg:.1f}° | Mode: {mode_str}"
                )
        else:
            super()._update_handover_status_card()
            if hasattr(self, 'test_approach_mode_lbl'):
                self.test_approach_mode_lbl.config(
                    text="NO AUTO SOLUTION", bg="#FADBD8", fg="#922B21"
                )

    def _preview_handover_in_3d(self, which='handover'):
        """Visualize Handover or Pre-Handover stance in the 3D scene."""
        mode = self.handover_mode_var.get()
        pair_key = self.selected_ho_pair.get() if hasattr(self, 'selected_ho_pair') else "robot1_robot2"
        parts = pair_key.split('_') if '_' in pair_key else ["robot1", "robot2"]
        r_a, r_b = parts[0], parts[1]
        inv_key = f"{r_b}_{r_a}"

        if mode == "manual":
            from robot_control.handover.stage1 import lookup_manual_handover_with_pre
            res = lookup_manual_handover_with_pre(self.handover_cfg, r_a, r_b)
            if res is None:
                self._log(f"⚠️ Cannot preview 3D: No manual handover found for {pair_key} in handover.yaml.")
                return
            q_a, q_b, pre_q_a, pre_q_b = res
            self.is_previewing_handover = True
            if which == 'pre_handover':
                self.preview_q = {
                    r_a: np.asarray(pre_q_a, dtype=float).copy(),
                    r_b: np.asarray(pre_q_b, dtype=float).copy(),
                }
                self._log(f"👁️ [3D PREVIEW] Showing MANUAL PRE-HANDOVER stance for {r_a.upper()} and {r_b.upper()}.")
            else:
                self.preview_q = {
                    r_a: np.asarray(q_a, dtype=float).copy(),
                    r_b: np.asarray(q_b, dtype=float).copy(),
                }
                self._log(f"👁️ [3D PREVIEW] Showing MANUAL HANDOVER stance for {r_a.upper()} and {r_b.upper()}.")
            self._update_3d()
            return

        p = self.auto_handover_poses.get(pair_key) or self.auto_handover_poses.get(inv_key)
        if not p:
            self._load_initial_auto_handover_poses()
            p = self.auto_handover_poses.get(pair_key) or self.auto_handover_poses.get(inv_key)

        if not p:
            self._log(f"⚠️ Cannot preview 3D: No solution available for {pair_key}. Click '⚡ Recompute' below.")
            return

        self.is_previewing_handover = True
        if which == 'pre_handover':
            self.preview_q = {
                p.robot_a_id: np.asarray(p.pre_handover_a_joint_angles_deg, dtype=float).copy(),
                p.robot_b_id: np.asarray(p.pre_handover_b_joint_angles_deg, dtype=float).copy(),
            }
            self._log(f"👁️ [3D PREVIEW] Showing PRE-HANDOVER stance for {p.robot_a_id.upper()} and {p.robot_b_id.upper()}. (No robot motion commanded)")
        else:
            self.preview_q = {
                p.robot_a_id: np.asarray(p.robot_a_joint_angles_deg, dtype=float).copy(),
                p.robot_b_id: np.asarray(p.robot_b_joint_angles_deg, dtype=float).copy(),
            }
            self._log(f"👁️ [3D PREVIEW] Showing HANDOVER stance for {p.robot_a_id.upper()} and {p.robot_b_id.upper()}. (No robot motion commanded)")

        self._update_3d()

    def _on_recompute_test2_handover_clicked(self):
        """Recompute test2 handover poses using Top-of-Workspace strategy only."""
        if getattr(self, 'is_recomputing_handover', False):
            return

        self.is_recomputing_handover = True
        self.btn_recompute_ho.config(state="disabled", text="⏳ Solving...")
        self.ho_recomp_msg.config(text="Computing...", fg=self.accent_color)
        self._log("\n⚡ Starting Stage 3 Test 2 Handover Recomputation (Top-of-Workspace Only)...")

        def progress_cb(msg: str):
            self.root.after(0, lambda m=msg: self._log(f"  [Solver] {m}"))

        def worker():
            try:
                poses = compute_all_test2_handover_poses(
                    self.robots,
                    self.handover_cfg,
                    cache_path=self.auto_handover_cache_path,
                    force_recompute=True,
                    progress_callback=progress_cb,
                )

                def on_done():
                    self.is_recomputing_handover = False
                    self.auto_handover_poses = poses
                    self.btn_recompute_ho.config(state="normal", text="⚡ Recompute (Top-WS)")
                    self.ho_recomp_msg.config(text="✓ Saved", fg=self.success_color)
                    self._update_handover_status_card()
                    self._update_3d()
                    self._log(f"✓ Top-of-Workspace recomputation complete! {len(poses)} pairs saved to cache.")

                self.root.after(0, on_done)
            except Exception as e:
                def on_err(err=str(e)):
                    self.is_recomputing_handover = False
                    self.btn_recompute_ho.config(state="normal", text="⚡ Recompute (Top-WS)")
                    self.ho_recomp_msg.config(text="Error", fg=self.danger_color)
                    self._log(f"ERROR recomputing test2 handover poses: {err}")

                self.root.after(0, on_err)

        threading.Thread(target=worker, daemon=True).start()

    # =========================================================================
    # PLANNING & EXECUTION
    # =========================================================================

    def _run_plan_thread(self, real_robot: bool = False):
        """Execute Stage 3 Test 2 planning and execution with instant cache reuse."""
        self._log("=== STARTING STAGE 3 TEST 2 TASK EXECUTION ===")
        mode_str = "REAL ROBOT HARDWARE" if real_robot else "3D SIMULATION PREVIEW"
        self._log(f"[MODE: {mode_str}] Strategy: TOP_OF_WORKSPACE")

        try:
            pick_xyz = [float(x.strip()) for x in self.pick_entry.get().split(',')]
            pick_yaw = float(self.pick_yaw_entry.get().strip())
            place_xyz = [float(x.strip()) for x in self.place_entry.get().split(',')]
            place_yaw = float(self.place_yaw_entry.get().strip())
        except Exception:
            self._log("ERROR: Invalid Pick/Place Coordinates or Yaw Angle.")
            return

        current_q = {rid: q.copy() for rid, q in self.live_q.items()}
        use_auto = (self.handover_mode_var.get() == "auto")
        target_key = (tuple(pick_xyz), pick_yaw, tuple(place_xyz), place_yaw, "TOP_OF_WORKSPACE", use_auto)

        cache_fp = self.cache_filepath if (hasattr(self, 'use_disk_cache_var') and self.use_disk_cache_var.get()) else None
        fp_hash = self.config_fingerprint if (hasattr(self, 'use_disk_cache_var') and self.use_disk_cache_var.get()) else None

        is_pick_cached = False
        is_place_cached = False
        if cache_fp and fp_hash:
            is_pick_cached = lookup_cached_target_ik(cache_fp, fp_hash, pick_xyz, pick_yaw) is not None
            is_place_cached = lookup_cached_target_ik(cache_fp, fp_hash, place_xyz, place_yaw) is not None

        if is_pick_cached and is_place_cached:
            self._log("\n⚡ [DISK CACHE HIT] Target IK loaded from disk in < 1 ms!")
        else:
            self._log("\n🔄 Solving kinematics and adaptive tilt angles across robots...")

        try:
            plan, pick_opts, place_opts = build_stage3_test2_plan(
                pick_xyz=pick_xyz,
                place_xyz=place_xyz,
                robots=self.robots,
                handover_cfg=self.handover_cfg,
                chooser=self._chooser,
                pick_yaw_deg=pick_yaw,
                place_yaw_deg=place_yaw,
                use_auto_handover=use_auto,
                cache_filepath=cache_fp,
                config_fingerprint=fp_hash,
                test_poses=self.auto_handover_poses,
            )
        except RuntimeError as exc:
            self._log(f"ERROR: {exc}")
            return

        self._set_badges(plan.pick.solver_type, plan.place.solver_type)

        self._log(f"\nPICK: {plan.pick_robot.upper()} (Solver: {plan.pick.solver_type})")
        self._log(f"PLACE: {plan.place_robot.upper()} (Solver: {plan.place.solver_type})")

        if plan.needs_handover:
            info = plan.auto_handover_info
            if info:
                self._log(f"HANDOVER: AUTO {plan.pick_robot} <-> {plan.place_robot}")
                self._log(f"  Handover Position: {info.handover_position_world} mm (Top-of-WS)")
                self._log(f"  Standoff: {info.standoff_distance_mm:.1f} mm | Clr: {info.handover_clearance_mm:.1f} mm")
                self._log(f"  Approach Mode: {info.approach_mode}")
            else:
                self._log(f"HANDOVER: MANUAL OVERRIDE (handover.yaml)")

        self._log("\nPlanning collision-free synchronized trajectories...")
        traj_result = plan_stage3_test2_paths(
            plan, self.robots, safe_distance=self.safe_dist, steps=self.trajectory_steps
        )

        if traj_result.collision_avoidance_applied:
            self._log("! WARNING: Direct simultaneous path was near collision limit.")
            self._log("! Safe collision-avoidance staggering was automatically applied.")

        self._log(f"Min predicted body clearance: {traj_result.min_overall_clearance_mm:.1f} mm (Safe dist: {self.safe_dist:.1f} mm)")

        self.cached_plan = {
            'target_key': target_key,
            'plan': plan,
            'traj_result': traj_result,
        }
        is_cached = (is_pick_cached and is_place_cached)

        proceed = self._show_solution_dialog(
            plan, traj_result,
            is_cached=is_cached,
            real_robot=real_robot,
        )
        if not proceed:
            self._log("\n[ABORTED] Execution cancelled by operator in solution dialog.")
            self._set_status_lbl("Execution cancelled by user", self.warning_color)
            return

        self.is_simulating = True
        self.is_real_execution = real_robot
        self.current_q = {k: v.copy() for k, v in self.live_q.items()}

        pick_rid = plan.pick_robot
        place_rid = plan.place_robot

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
            # PHASE 1: SIMULTANEOUS MOTION
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

            # PHASE 4: BOTH ADVANCE INTO HANDOVER
            step_simultaneous(
                traj_result.phase4_advance_to_handover,
                f"4. ADVANCE: Both robots enter Handover mating corridor.",
                dwell=self.dwell_time_s,
            )

            # PHASE 5: TRANSFER
            self._log(f"5. HANDOVER TRANSFER: {place_rid} closes gripper, {pick_rid} opens gripper.")
            self._send_gripper_command(place_rid, 'close', real_robot=real_robot)
            time.sleep(0.5)
            self._send_gripper_command(pick_rid, 'open', real_robot=real_robot)
            time.sleep(0.5)

            # PHASE 6: RETRACT
            step_simultaneous(
                traj_result.phase6_retract_to_pre_handover,
                f"6. RETRACT: Both robots retreat safely back to Pre-Handover.",
                dwell=self.dwell_time_s,
            )

            # PHASE 7: PLACE MOTION
            step_simultaneous(
                traj_result.phase7_handover_to_place,
                f"7. {place_rid} moves to Place Target ({pick_rid} waiting at Pre-Handover).",
                dwell=self.dwell_time_s,
            )

            # PHASE 8: RELEASE
            self._log(f"8. {place_rid} waits {self.pick_wait_s}s and opens gripper at Place Target.")
            self._send_gripper_command(place_rid, 'open', real_robot=real_robot)
            time.sleep(self.pick_wait_s)

            # PHASE 9: RETURN HOME
            step_simultaneous(
                traj_result.phase9_simultaneous_home,
                f"9. SIMULTANEOUS RETURN: Both {pick_rid} and {place_rid} return to Home!",
                dwell=self.dwell_time_s,
            )
        else:
            # SINGLE ROBOT
            step_simultaneous(traj_result.phase1_simultaneous, f"1. {pick_rid} moves to Pick Target.")
            self._send_gripper_command(pick_rid, 'close', real_robot=real_robot)
            time.sleep(self.pick_wait_s)
            step_simultaneous(traj_result.phase7_handover_to_place, f"2. {pick_rid} moves to Place Target.")
            self._send_gripper_command(pick_rid, 'open', real_robot=real_robot)
            time.sleep(self.pick_wait_s)
            step_simultaneous(traj_result.phase9_simultaneous_home, f"3. {pick_rid} returns home.")

        self._set_control_mode(pick_rid, 'POSITION', real_robot=real_robot)
        if plan.needs_handover:
            self._set_control_mode(place_rid, 'POSITION', real_robot=real_robot)

        self.is_simulating = False
        self.is_real_execution = False
        self._update_clearance_ui(self.live_clearance_mm, "IDLE / READY", "safe")
        self._set_status_lbl("Task completed successfully", self.success_color)
        self._log("\n✓ Task completed successfully!")


def main(args=None):
    rclpy.init(args=args)
    root = tk.Tk()
    gui_node = Stage3Test2GUI(root)

    def ros_spin():
        try:
            rclpy.spin(gui_node)
        except Exception:
            pass

    t = threading.Thread(target=ros_spin, daemon=True)
    t.start()

    try:
        root.mainloop()
    finally:
        gui_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
