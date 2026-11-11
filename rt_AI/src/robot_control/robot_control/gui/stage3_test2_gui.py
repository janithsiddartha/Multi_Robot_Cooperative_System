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
from std_msgs.msg import Float64MultiArray, String, Int32
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
    compute_single_test2_handover_pose,
    clear_test2_handover_cache,
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
        self.root.after(200, self._scan_system_serial_ports)

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

        # Speed publishers for all robots
        self.publishers_speed = {}
        self.publishers_gripper_speed = {}
        for rid in self.robots.keys():
            self.publishers_speed[rid] = self.create_publisher(
                Int32, f'/{rid}/robot_speed', 10
            )
            self.publishers_gripper_speed[rid] = self.create_publisher(
                Int32, f'/{rid}/gripper_speed', 10
            )

        # Hardware serial command publishers & subscriptions
        self.publishers_command = {}
        for rid in self.robots.keys():
            self.publishers_command[rid] = self.create_publisher(
                String, f'/{rid}/robot_command', 10
            )
        self.global_command_pub = self.create_publisher(
            String, 'robot_command', 10
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

            sub_port = self.create_subscription(
                String,
                f'/{rid}/available_ports',
                lambda msg, r=rid: self._on_available_ports_cb(msg, r),
                10,
            )
            self.subs.append(sub_port)

            sub_conn = self.create_subscription(
                String,
                f'/{rid}/connected_port',
                lambda msg, r=rid: self._on_connected_port_cb(msg, r),
                10,
            )
            self.subs.append(sub_conn)

            sub_cmd = self.create_subscription(
                String,
                f'/{rid}/robot_command',
                lambda msg, r=rid: self._on_robot_command_cb(msg, r),
                10,
            )
            self.subs.append(sub_cmd)

            sub_spd = self.create_subscription(
                Int32,
                f'/{rid}/robot_speed',
                lambda msg, r=rid: self._on_robot_speed_cb(msg, r),
                10,
            )
            self.subs.append(sub_spd)

        sub_global_ports = self.create_subscription(
            String,
            'available_ports',
            self._on_global_available_ports_cb,
            10,
        )
        self.subs.append(sub_global_ports)

        # Hardware port tracking
        self.hw_available_ports: List[str] = []
        self.hw_connected = {r: False for r in self.robots.keys()}
        self.hw_assigned_port = {r: "" for r in self.robots.keys()}
        self.hw_port_combo = {}
        self.hw_conn_badge = {}

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

        if not hasattr(self, 'auto_handover_poses'):
            self.auto_handover_poses = {}

        if poses:
            self.auto_handover_poses.update(poses)
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
            width=14,
            font=("Consolas", 8, "bold")
        )
        self.pair_combobox.pack(side="left", padx=(0, 4))
        self.pair_combobox.bind("<<ComboboxSelected>>", self._on_handover_pair_selected)

        self.btn_reload_ho = tk.Button(
            pair_row, text="🔄 Reload", font=("Helvetica", 7, "bold"),
            bg="#EAEDED", fg="#2C3E50", relief="flat", padx=3, pady=1, cursor="hand2",
            command=self._on_reload_handover_cache
        )
        self.btn_reload_ho.pack(side="left", padx=(0, 2))

        self.btn_clear_ho = tk.Button(
            pair_row, text="🗑️ Clear", font=("Helvetica", 7, "bold"),
            bg="#FDEDEC", fg="#C0392B", relief="flat", padx=3, pady=1, cursor="hand2",
            command=self._on_clear_handover_cache
        )
        self.btn_clear_ho.pack(side="left")

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

        # Recompute Row: Recompute 1-2, 2-3, 3-1, and ALL
        recomp_row = tk.Frame(ho_card, bg=self.card_color)
        recomp_row.pack(fill="x", pady=(2, 1))

        tk.Label(
            recomp_row, text="⚡ Recompute:", font=("Helvetica", 7, "bold"),
            bg=self.card_color, fg="#2980B9"
        ).pack(side="left", padx=(0, 3))

        self.btn_recomp_12 = tk.Button(
            recomp_row, text="1-2", font=("Helvetica", 7, "bold"),
            bg="#EAF2F8", fg="#2980B9", relief="flat", padx=4, pady=1, cursor="hand2",
            command=lambda: self._on_recompute_single_pair("robot1_robot2")
        )
        self.btn_recomp_12.pack(side="left", padx=1)

        self.btn_recomp_23 = tk.Button(
            recomp_row, text="2-3", font=("Helvetica", 7, "bold"),
            bg="#EAF2F8", fg="#2980B9", relief="flat", padx=4, pady=1, cursor="hand2",
            command=lambda: self._on_recompute_single_pair("robot2_robot3")
        )
        self.btn_recomp_23.pack(side="left", padx=1)

        self.btn_recomp_31 = tk.Button(
            recomp_row, text="3-1", font=("Helvetica", 7, "bold"),
            bg="#EAF2F8", fg="#2980B9", relief="flat", padx=4, pady=1, cursor="hand2",
            command=lambda: self._on_recompute_single_pair("robot3_robot1")
        )
        self.btn_recomp_31.pack(side="left", padx=1)

        self.btn_recompute_ho = tk.Button(
            recomp_row, text="ALL", font=("Helvetica", 7, "bold"),
            bg="#2980B9", fg="#FFFFFF", relief="flat", padx=5, pady=1, cursor="hand2",
            command=self._on_recompute_test2_handover_clicked
        )
        self.btn_recompute_ho.pack(side="left", padx=(2, 0))

        self.ho_recomp_msg = tk.Label(
            recomp_row, text="✓ Cached", bg=self.card_color, fg=self.success_color, font=("Helvetica", 7, "italic")
        )
        self.ho_recomp_msg.pack(side="right")

        # Action Buttons Row 2: 3D Preview Handover / Pre-Handover / Inspect / Reset (Home All)
        action_row2 = tk.Frame(ho_card, bg=self.card_color)
        action_row2.pack(fill="x", pady=(2, 1))

        self.btn_preview_ho = tk.Button(
            action_row2, text="👁️ Preview", font=("Helvetica", 7, "bold"),
            bg="#EAEDED", fg="#2C3E50", relief="flat", padx=4, pady=1, cursor="hand2",
            command=lambda: self._preview_handover_in_3d('handover')
        )
        self.btn_preview_ho.pack(side="left", padx=(0, 2))

        self.btn_preview_pre = tk.Button(
            action_row2, text="👁️ Pre-H", font=("Helvetica", 7, "bold"),
            bg="#EAEDED", fg="#2C3E50", relief="flat", padx=4, pady=1, cursor="hand2",
            command=lambda: self._preview_handover_in_3d('pre_handover')
        )
        self.btn_preview_pre.pack(side="left", padx=(0, 2))

        self.btn_inspect_ho = tk.Button(
            action_row2, text="🔍 Inspect...", font=("Helvetica", 7, "bold"),
            bg="#EAEDED", fg="#2C3E50", relief="flat", padx=4, pady=1, cursor="hand2",
            command=self._show_handover_inspector_dialog
        )
        self.btn_inspect_ho.pack(side="left", padx=(0, 2))

        self.btn_reset_preview = tk.Button(
            action_row2, text="🏠 Reset / Home All", font=("Helvetica", 7, "bold"),
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

        # 3.5. PHYSICAL ROBOT PORTS (UART) CARD
        hw_card = tk.Frame(left_frame, bg=self.card_color, padx=10, pady=4, relief="solid", bd=1)
        hw_card.pack(fill="x", pady=2)

        hw_hdr = tk.Frame(hw_card, bg=self.card_color)
        hw_hdr.pack(fill="x", pady=(0, 2))
        tk.Label(
            hw_hdr, text="PHYSICAL ROBOT PORTS (UART)",
            font=("Helvetica", 8, "bold"), fg="#2C3E50", bg=self.card_color
        ).pack(side="left")
        tk.Button(
            hw_hdr, text="🔄 Scan Ports", font=("Helvetica", 7, "bold"),
            bg="#EAEDED", fg="#2C3E50", activebackground="#D5D8DC",
            relief="flat", cursor="hand2", padx=4, pady=0,
            command=self._scan_system_serial_ports
        ).pack(side="right")

        robot_color_map = {
            "robot1": ("R1", "#2980B9"),
            "robot2": ("R2", "#E74C3C"),
            "robot3": ("R3", "#27AE60"),
        }

        for rid in ["robot1", "robot2", "robot3"]:
            if rid not in self.robots:
                continue
            lbl_name, col = robot_color_map.get(rid, (rid.upper(), "#34495E"))
            r_row = tk.Frame(hw_card, bg=self.card_color)
            r_row.pack(fill="x", pady=1)

            tk.Label(
                r_row, text=lbl_name, font=("Helvetica", 8, "bold"),
                fg=col, bg=self.card_color, width=3, anchor="w"
            ).pack(side="left")

            combo = ttk.Combobox(r_row, width=15, state="readonly", font=("Helvetica", 8))
            combo.pack(side="left", padx=(0, 3))
            self.hw_port_combo[rid] = combo

            btn_set = tk.Button(
                r_row, text="SET", font=("Helvetica", 7, "bold"),
                bg="#2980B9", fg="#FFFFFF", activebackground="#1B4F72", activeforeground="#FFFFFF",
                relief="flat", cursor="hand2", padx=4, pady=0,
                command=lambda r=rid: self._on_set_robot_port(r)
            )
            btn_set.pack(side="left", padx=(0, 2))

            btn_disc = tk.Button(
                r_row, text="DISC", font=("Helvetica", 7, "bold"),
                bg="#EAEDED", fg="#7F8C8D", activebackground="#D5D8DC",
                relief="flat", cursor="hand2", padx=3, pady=0,
                command=lambda r=rid: self._on_disconnect_robot_port(r)
            )
            btn_disc.pack(side="left", padx=(0, 3))

            badge = tk.Label(
                r_row, text="DISCONNECTED", font=("Helvetica", 7, "bold"),
                bg="#FDEDEC", fg="#C0392B", padx=3, pady=0, width=13, anchor="center"
            )
            badge.pack(side="right")
            self.hw_conn_badge[rid] = badge

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

        # Speed Control Panel (1 to 10 matching Individual GUI & Web Twin)
        speed_frame = tk.Frame(action_frame, bg=self.bg_color)
        speed_frame.pack(fill="x", pady=(1, 3))
        self.speed = 3
        self.speed_lbl = tk.Label(
            speed_frame, text=f"MOTION SPEED: LEVEL {self.speed} / 10",
            font=("Helvetica", 8, "bold"), bg=self.bg_color, fg="#2C3E50"
        )
        self.speed_lbl.pack(side="top", anchor="w")
        self.speed_slider = ttk.Scale(
            speed_frame, from_=1, to=10, orient="horizontal",
            command=self._on_speed_slider_change
        )
        self.speed_slider.set(self.speed)
        self.speed_slider.pack(fill="x", pady=(1, 2))

        self.gripper_speed = 5
        self.gripper_speed_lbl = tk.Label(
            speed_frame, text=f"GRIPPER SPEED: LEVEL {self.gripper_speed} / 10",
            font=("Helvetica", 8, "bold"), bg=self.bg_color, fg="#10B981"
        )
        self.gripper_speed_lbl.pack(side="top", anchor="w", pady=(3, 0))
        self.gripper_speed_slider = ttk.Scale(
            speed_frame, from_=1, to=10, orient="horizontal",
            command=self._on_gripper_speed_slider_change
        )
        self.gripper_speed_slider.set(self.gripper_speed)
        self.gripper_speed_slider.pack(fill="x", pady=(1, 2))

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

    def _reset_3d_preview(self):
        """Reset 3D visualizer to mechanical Home positions without clearing handover cache."""
        self.is_previewing_handover = False
        self.preview_q.clear()
        for rid, cfg in self.robots.items():
            self.live_q[rid] = np.asarray(cfg.home_deg, dtype=float).copy()
            self.current_q[rid] = np.asarray(cfg.home_deg, dtype=float).copy()
        self._log("🏠 [3D PREVIEW] Reset / Home All: Arms returned to mechanical Home positions.")
        self._update_handover_status_card()
        self._update_3d()

    def _on_reload_handover_cache(self):
        """Reload test2 handover cache from disk matching current configuration fingerprint."""
        self._log(f"[CACHE RELOAD] Reloading test2 handover cache from {self.auto_handover_cache_path}...")
        try:
            poses = load_test2_handover_cache(self.auto_handover_cache_path, self.config_fingerprint)
            if poses:
                self.auto_handover_poses = poses
                self.ho_recomp_msg.config(text=f"✓ Loaded ({len(poses)})", fg=self.success_color)
                self._log(f"✓ Loaded {len(poses)} verified auto-handover pose pairs from test2 cache.")
            else:
                self.ho_recomp_msg.config(text="⚠️ Empty/Stale", fg=self.warning_color)
                self._log("⚠️ No valid test2 cache found matching configuration fingerprint.")
        except Exception as e:
            self.ho_recomp_msg.config(text="Error", fg=self.danger_color)
            self._log(f"ERROR reloading test2 cache: {e}")
        self._update_handover_status_card()
        self._update_3d()

    def _on_clear_handover_cache(self):
        """Clear cached handover poses for the active configuration fingerprint."""
        if not messagebox.askyesno("Clear Handover Cache", "Are you sure you want to clear the cached auto-handover poses for this configuration?"):
            return
        clear_test2_handover_cache(self.auto_handover_cache_path, self.config_fingerprint)
        self.auto_handover_poses = {}
        self.ho_recomp_msg.config(text="Cache Cleared", fg=self.warning_color)
        self._log("🗑️ Handover cache cleared for active configuration fingerprint.")
        self._update_handover_status_card()
        self._update_3d()

    def _set_recompute_buttons_state(self, state: str):
        for btn_name in ('btn_recomp_12', 'btn_recomp_23', 'btn_recomp_31', 'btn_recompute_ho'):
            if hasattr(self, btn_name):
                getattr(self, btn_name).config(state=state)

    def _on_recompute_single_pair(self, pair_name: str):
        """Recompute test2 handover pose for a single pair (Top-of-Workspace only)."""
        if getattr(self, 'is_recomputing_handover', False):
            return

        self.is_recomputing_handover = True
        self.selected_ho_pair.set(pair_name)
        self._set_recompute_buttons_state("disabled")
        self.ho_recomp_msg.config(text=f"Solving {pair_name}...", fg=self.accent_color)
        self._log(f"\n⚡ Starting Handover Recomputation for {pair_name} (Top-of-Workspace Only)...")

        def progress_cb(msg: str):
            self.root.after(0, lambda m=msg: self._log(f"  [Solver] {m}"))

        def worker():
            try:
                pose = compute_single_test2_handover_pose(
                    pair_name=pair_name,
                    robots=self.robots,
                    handover_cfg=self.handover_cfg,
                    cache_path=self.auto_handover_cache_path,
                    progress_callback=progress_cb,
                )

                def on_done():
                    self.is_recomputing_handover = False
                    self._set_recompute_buttons_state("normal")
                    all_poses = load_test2_handover_cache(self.auto_handover_cache_path, self.config_fingerprint) or {}
                    if pose is not None:
                        all_poses[pair_name] = pose
                        self.auto_handover_poses = all_poses
                        self.ho_recomp_msg.config(text=f"✓ {pair_name} Saved", fg=self.success_color)
                        self._log(f"✓ Recomputation complete for {pair_name}! Saved to cache.")
                    else:
                        self.auto_handover_poses = all_poses
                        self.ho_recomp_msg.config(text="⚠️ No solution", fg=self.warning_color)
                        self._log(f"⚠️ No valid Top-of-Workspace handover solution found for {pair_name}.")
                    self._update_handover_status_card()
                    self._update_3d()

                self.root.after(0, on_done)
            except Exception as e:
                def on_err(err=str(e)):
                    self.is_recomputing_handover = False
                    self._set_recompute_buttons_state("normal")
                    self.ho_recomp_msg.config(text="Error", fg=self.danger_color)
                    self._log(f"ERROR recomputing handover pose for {pair_name}: {err}")

                self.root.after(0, on_err)

        threading.Thread(target=worker, daemon=True).start()

    def _on_recompute_test2_handover_clicked(self):
        """Recompute test2 handover poses using Top-of-Workspace strategy only for all pairs."""
        if getattr(self, 'is_recomputing_handover', False):
            return

        self.is_recomputing_handover = True
        self._set_recompute_buttons_state("disabled")
        self.btn_recompute_ho.config(text="⏳ Solving...")
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
                    self._set_recompute_buttons_state("normal")
                    self.btn_recompute_ho.config(text="ALL")
                    all_poses = load_test2_handover_cache(self.auto_handover_cache_path, self.config_fingerprint) or poses or {}
                    self.auto_handover_poses = all_poses
                    self.ho_recomp_msg.config(text=f"✓ Saved ({len(all_poses)})", fg=self.success_color)
                    self._update_handover_status_card()
                    self._update_3d()
                    self._log(f"✓ Top-of-Workspace recomputation complete! {len(all_poses)} pairs saved to cache.")

                self.root.after(0, on_done)
            except Exception as e:
                def on_err(err=str(e)):
                    self.is_recomputing_handover = False
                    self._set_recompute_buttons_state("normal")
                    self.btn_recompute_ho.config(text="ALL")
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
                effective_dt = max(0.015, 0.08 - (getattr(self, 'speed', 3) * 0.006))
                time.sleep(effective_dt)
            time.sleep(dwell)

        if plan.needs_handover:
            # PHASE 0: PROACTIVE OPEN GRIPPER FOR APPROACH
            self._send_gripper_command(pick_rid, 'open', real_robot=real_robot)

            # PHASE 1: SIMULTANEOUS MOTION (Pre-Pick approach + Cartesian descent to Pick)
            step_simultaneous(
                traj_result.phase1_simultaneous,
                f"1. SIMULTANEOUS: {pick_rid} approaches Pre-Pick (+50mm) & Cartesian descends (-Z) to Pick, WHILE {place_rid} moves to Pre-Handover!",
                dwell=self.dwell_time_s,
            )

            # PHASE 2: GRASP AT PICK
            self._log(f"2. {pick_rid} dwells {self.pick_wait_s}s, closes gripper (grasp), and dwells {self.pick_wait_s}s. ({place_rid} waiting at Pre-Handover)")
            time.sleep(self.dwell_time_s)
            self._send_gripper_command(pick_rid, 'close', real_robot=real_robot)
            self._update_clearance_ui(self.live_clearance_mm, f"{place_rid} WAITING AT PRE-HANDOVER", "info")
            time.sleep(self.pick_wait_s)

            # PHASE 3: CARTESIAN LIFT TO PRE-PICK & TRANSIT TO PRE-HANDOVER
            step_simultaneous(
                traj_result.phase3_pick_to_pre_handover,
                f"3. {pick_rid} Cartesian lifts (+50mm) to Pre-Pick and transfers to Pre-Handover ({place_rid} stationed at Pre-Handover).",
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
            time.sleep(self.transfer_wait_s)
            self._send_gripper_command(pick_rid, 'open', real_robot=real_robot)
            time.sleep(self.transfer_wait_s)

            # PHASE 6: RETRACT
            step_simultaneous(
                traj_result.phase6_retract_to_pre_handover,
                f"6. RETRACT: Both robots retreat safely back to Pre-Handover along reverse corridors.",
                dwell=self.dwell_time_s,
            )

            # PHASE 7: PLACE MOTION (Pre-Place approach + Cartesian descent to Place)
            step_simultaneous(
                traj_result.phase7_handover_to_place,
                f"7. {place_rid} transfers to Pre-Place (+50mm) & Cartesian descends (-Z) to Place Target ({pick_rid} waiting safely).",
                dwell=self.dwell_time_s,
            )

            # PHASE 8: RELEASE
            self._log(f"8. {place_rid} dwells {self.place_wait_s}s, opens gripper at Place Target, and dwells {self.place_wait_s}s.")
            time.sleep(self.dwell_time_s)
            self._send_gripper_command(place_rid, 'open', real_robot=real_robot)
            time.sleep(self.place_wait_s)

            # PHASE 9: CARTESIAN RETRACT TO PRE-PLACE & RETURN HOME
            step_simultaneous(
                traj_result.phase9_simultaneous_home,
                f"9. {place_rid} Cartesian retracts (+50mm) to Pre-Place & SIMULTANEOUS RETURN: Both {pick_rid} and {place_rid} return to Home!",
                dwell=self.dwell_time_s,
            )
        else:
            # SINGLE ROBOT
            self._send_gripper_command(pick_rid, 'open', real_robot=real_robot)
            step_simultaneous(traj_result.phase1_simultaneous, f"1. {pick_rid} approaches Pre-Pick (+50mm) & Cartesian descends (-Z) to Pick Target.")
            time.sleep(self.dwell_time_s)
            self._send_gripper_command(pick_rid, 'close', real_robot=real_robot)
            time.sleep(self.pick_wait_s)
            step_simultaneous(traj_result.phase7_handover_to_place, f"2. {pick_rid} Cartesian lifts (+50mm), transfers to Pre-Place, & Cartesian descends (-Z) to Place Target.")
            time.sleep(self.dwell_time_s)
            self._send_gripper_command(pick_rid, 'open', real_robot=real_robot)
            time.sleep(self.place_wait_s)
            step_simultaneous(traj_result.phase9_simultaneous_home, f"3. {pick_rid} Cartesian retracts (+50mm) to Pre-Place and returns home.")

        self._set_control_mode(pick_rid, 'POSITION', real_robot=real_robot)
        if plan.needs_handover:
            self._set_control_mode(place_rid, 'POSITION', real_robot=real_robot)

        self.is_simulating = False
        self.is_real_execution = False
        self._update_clearance_ui(self.live_clearance_mm, "IDLE / READY", "safe")
        self._set_status_lbl("Task completed successfully", self.success_color)
        self._log("\n✓ Task completed successfully!")

    # =========================================================================
    # PHYSICAL ROBOT HARDWARE SERIAL PORTS (UART)
    # =========================================================================

    def _scan_system_serial_ports(self):
        """Scan real physical serial ports and merge with ROS available_ports topics."""
        scanned = set(self.hw_available_ports)
        try:
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                if any(k in p.device for k in ["USB", "ACM", "COM", "robot"]):
                    scanned.add(p.device)
        except Exception as e:
            self.get_logger().warn(f"Port scan error: {e}")

        if os.path.isdir("/dev"):
            for item in os.listdir("/dev"):
                if item.startswith(("ttyUSB", "ttyACM", "robot")):
                    scanned.add(f"/dev/{item}")

        ports = sorted(list(scanned))
        self.hw_available_ports = ports
        for rid, combo in self.hw_port_combo.items():
            r_ports = getattr(self, 'hw_robot_ports', {}).get(rid, [])
            vals = r_ports if r_ports else ports
            combo["values"] = vals if vals else ["No devices"]
            if vals and (combo.get() not in vals or combo.get() == "No devices"):
                combo.set(vals[0])
            elif not vals:
                combo.set("No devices")

        if ports:
            self._log(f"✓ Ports available ({len(ports)}): {', '.join(ports)}")
        else:
            self._log("ℹ️ Scanning for active robot serial ports on ROS /available_ports...")

    def _update_robot_port_dropdown(self, rid: str, ports: List[str]):
        """Update combobox for a specific robot when available_ports arrives from ROS."""
        if not hasattr(self, 'hw_robot_ports'):
            self.hw_robot_ports = {r: [] for r in self.robots.keys()}
        self.hw_robot_ports[rid] = ports

        # Merge with overall ports
        all_known = set(self.hw_available_ports)
        for p in ports:
            all_known.add(p)
        self.hw_available_ports = sorted(list(all_known))

        combo = self.hw_port_combo.get(rid)
        if not combo:
            return

        current_val = combo.get()
        new_values = ports if ports else (self.hw_available_ports if self.hw_available_ports else ["No devices"])
        if list(combo["values"]) != new_values:
            combo["values"] = new_values

        if ports:
            if current_val in ports:
                combo.set(current_val)
            elif self.hw_assigned_port.get(rid) in ports:
                combo.set(self.hw_assigned_port[rid])
            else:
                combo.set(ports[0])
        elif current_val not in new_values:
            combo.set(new_values[0])

    def _joint_cb(self, msg: Float64MultiArray, rid: str):
        if rid in self.robots and len(msg.data) == self.robots[rid].num_joints:
            new_q = np.array(msg.data, dtype=float)
            if self.is_previewing_handover and np.linalg.norm(new_q - self.live_q[rid]) > 0.5:
                self.is_previewing_handover = False
                self.preview_q.clear()
            self.live_q[rid] = new_q
            self.has_live_q[rid] = True

    def _on_speed_slider_change(self, val):
        s = int(float(val))
        if s != getattr(self, 'speed', 3):
            self.speed = s
            if hasattr(self, 'speed_lbl'):
                self.speed_lbl.config(text=f"MOTION SPEED: LEVEL {self.speed} / 10")
            msg = Int32()
            msg.data = self.speed
            for rid, pub in getattr(self, 'publishers_speed', {}).items():
                pub.publish(msg)

    def _on_gripper_speed_slider_change(self, val):
        s = int(float(val))
        if s != getattr(self, 'gripper_speed', 5):
            self.gripper_speed = s
            if hasattr(self, 'gripper_speed_lbl'):
                self.gripper_speed_lbl.config(text=f"GRIPPER SPEED: LEVEL {self.gripper_speed} / 10")
            msg = Int32()
            msg.data = self.gripper_speed
            for rid, pub in getattr(self, 'publishers_gripper_speed', {}).items():
                pub.publish(msg)

    def _on_robot_speed_cb(self, msg, rid: str):
        s = max(1, min(10, int(msg.data)))
        if s != getattr(self, 'speed', 3):
            self.speed = s
            self.root.after(0, self._apply_speed_ui)

    def _apply_speed_ui(self):
        if hasattr(self, 'speed_slider'):
            self.speed_slider.set(self.speed)
        if hasattr(self, 'speed_lbl'):
            self.speed_lbl.config(text=f"MOTION SPEED: LEVEL {self.speed} / 10")

    def _on_connected_port_cb(self, msg, rid: str):
        port = msg.data.strip()
        connected = bool(port and port.upper() != "NONE")
        self.root.after(0, lambda r=rid, c=connected, p=(port if connected else ""): self._apply_robot_port_state(r, c, p))

    def _on_robot_command_cb(self, msg, rid: str):
        cmd = msg.data.strip()
        if cmd.upper().startswith("PORT,"):
            p = cmd.split(",", 1)[1].strip()
            connected = bool(p and p.upper() != "NONE")
            self.root.after(0, lambda r=rid, c=connected, port=(p if connected else ""): self._apply_robot_port_state(r, c, port))

    def _apply_robot_port_state(self, rid: str, connected: bool, port: str):
        self.hw_connected[rid] = connected
        self.hw_assigned_port[rid] = port
        combo = self.hw_port_combo.get(rid)
        if combo and port:
            current_vals = list(combo["values"])
            if port not in current_vals:
                current_vals.append(port)
                combo["values"] = current_vals
            combo.set(port)
        self._set_connection_badge(rid, connected, port)

    def _on_available_ports_cb(self, msg, rid: str):
        raw = msg.data.strip()
        ports = [p.strip() for p in raw.split(",") if p.strip()] if (raw and raw != "NONE") else []
        self.root.after(0, lambda r=rid, p=ports: self._update_robot_port_dropdown(r, p))

    def _on_global_available_ports_cb(self, msg):
        raw = msg.data.strip()
        ports = [p.strip() for p in raw.split(",") if p.strip()] if (raw and raw != "NONE") else []
        if ports:
            self.root.after(0, lambda p=ports: self._apply_global_available_ports(p))

    def _apply_global_available_ports(self, ports: List[str]):
        all_known = set(self.hw_available_ports)
        for p in ports:
            all_known.add(p)
        self.hw_available_ports = sorted(list(all_known))
        for rid, combo in self.hw_port_combo.items():
            r_ports = getattr(self, 'hw_robot_ports', {}).get(rid, [])
            vals = r_ports if r_ports else self.hw_available_ports
            combo["values"] = vals if vals else ["No devices"]
            if vals and (combo.get() not in vals or combo.get() == "No devices"):
                combo.set(vals[0])

    def _on_set_robot_port(self, rid: str):
        combo = self.hw_port_combo.get(rid)
        if not combo:
            return
        selected = combo.get().strip()
        if not selected or selected == "No devices":
            self._log(f"⚠️ Cannot connect {rid.upper()}: No valid serial port selected.")
            return

        msg = String()
        msg.data = f"PORT,{selected}"
        if rid in self.publishers_command:
            self.publishers_command[rid].publish(msg)
        if hasattr(self, "global_command_pub"):
            self.global_command_pub.publish(msg)

        self.hw_connected[rid] = True
        self.hw_assigned_port[rid] = selected
        self._set_connection_badge(rid, True, selected)
        self._log(f"🔌 [{rid.upper()}] Bound to serial port: {selected} (CONNECTED)")

    def _on_disconnect_robot_port(self, rid: str):
        msg = String()
        msg.data = "PORT,NONE"
        if rid in self.publishers_command:
            self.publishers_command[rid].publish(msg)
        if hasattr(self, "global_command_pub"):
            self.global_command_pub.publish(msg)

        self.hw_connected[rid] = False
        self.hw_assigned_port[rid] = ""
        self._set_connection_badge(rid, False)
        self._log(f"🔌 [{rid.upper()}] Port disconnected.")

    def _set_connection_badge(self, rid: str, connected: bool, port: str = ""):
        badge = self.hw_conn_badge.get(rid)
        if not badge:
            return
        if connected:
            dev_name = os.path.basename(port) if port else ""
            txt = f"CONN ({dev_name})" if dev_name else "CONNECTED"
            badge.config(text=txt, bg="#E8F8F5", fg="#27AE60")
        else:
            badge.config(text="DISCONNECTED", bg="#FDEDEC", fg="#C0392B")


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
