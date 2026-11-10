#!/usr/bin/env python3
"""web_twin/server.py

High-Performance 3D Web Digital Twin Server for rt_AI.
Provides:
  1. Static file hosting (HTML5, Three.js, CSS, JS).
  2. High-speed WebSocket streaming (/ws) for 60 FPS Digital Twin telemetry.
  3. Direct bridge to Stage 3 Test 2 kinematics & trajectory engine without
     modifying any existing ROS 2 files.
  4. Real-time 3D laser clearance calculation, live IK reachability checks,
     and workpiece physical attachment state machine.
"""

import asyncio
import json
import math
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import WSMsgType, web
import numpy as np

# Ensure src/robot_control is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_ROBOT_CONTROL = os.path.join(PROJECT_ROOT, "src", "robot_control")
if SRC_ROBOT_CONTROL not in sys.path:
    sys.path.insert(0, SRC_ROBOT_CONTROL)

from robot_control.handover.geometry import matrix_to_euler
from robot_control.handover.stage1 import load_stage1_robots, lookup_manual_handover_with_pre
from robot_control.handover.stage1_ik import solve_stage1_ik
from robot_control.handover.stage1_5 import get_robot_link_segments, compute_cartesian_linear_path
from robot_control.handover.stage2_5 import resolve_stage2_5_target
from robot_control.handover.stage3_test2 import (
    Stage3Test2Plan,
    build_stage3_test2_plan,
    plan_stage3_test2_paths,
)
from robot_control.handover.stage3_test2_solver import (
    compute_all_test2_handover_poses,
    compute_single_test2_handover_pose,
    clear_test2_handover_cache,
    load_test2_handover_cache,
    resolve_test2_handover_cache_path,
)
from robot_control.handover.trajectory_cache import (
    compute_config_fingerprint,
    resolve_cache_filepath,
    lookup_cached_target_ik,
    store_cached_target_ik,
)
from robot_control.kinematics.fk_math import all_frame_positions, fk, dh_joint_angles

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from web_twin.ros_bridge import WebTwinRosBridge


def dist_point_to_segment_pt(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray]:
    ab = b - a
    d_ab = float(np.dot(ab, ab))
    if d_ab < 1e-8:
        return float(np.linalg.norm(p - a)), a.copy()
    t = float(np.clip(np.dot(p - a, ab) / d_ab, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj)), proj


def closest_segment_points(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calculate exact closest points between 3D line segment p1-p2 and p3-p4."""
    u = p2 - p1
    v = p4 - p3
    w = p1 - p3
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    D = a * c - b * b
    if D < 1e-8:
        sc = 0.0
        tc = float(np.clip(e / c, 0.0, 1.0)) if c > 1e-8 else 0.0
    else:
        sc = float(np.clip((b * e - c * d) / D, 0.0, 1.0))
        tc = float(np.clip((a * e - b * d) / D, 0.0, 1.0))
    pt1 = p1 + sc * u
    pt2 = p3 + tc * v
    d_direct = float(np.linalg.norm(pt1 - pt2))
    d_sc, pt_sc2 = dist_point_to_segment_pt(pt1, p3, p4)
    d_tc, pt_tc1 = dist_point_to_segment_pt(pt2, p1, p2)
    m = min(d_direct, d_sc, d_tc)
    if m == d_direct:
        return d_direct, pt1, pt2
    elif m == d_sc:
        return d_sc, pt1, pt_sc2
    else:
        return d_tc, pt_tc1, pt2


def compute_laser_endpoints(
    q1_deg: np.ndarray, cfg1: object, q2_deg: np.ndarray, cfg2: object
) -> Tuple[float, List[float], List[float]]:
    """Compute minimum distance and exact 3D endpoints on Robot 1 & 2 for the laser beam."""
    seg1 = get_robot_link_segments(q1_deg, cfg1, exclude_tcp=True)
    seg2 = get_robot_link_segments(q2_deg, cfg2, exclude_tcp=True)

    min_d = float("inf")
    closest_a = np.zeros(3)
    closest_b = np.zeros(3)

    for s1 in seg1:
        for s2 in seg2:
            d, pt_a, pt_b = closest_segment_points(s1[0], s1[1], s2[0], s2[1])
            if d < min_d:
                min_d = d
                closest_a = pt_a
                closest_b = pt_b

    return (
        float(min_d),
        [round(float(v), 2) for v in closest_a],
        [round(float(v), 2) for v in closest_b],
    )


class WebTwinEngine:
    def __init__(self):
        print("[WebTwin] Loading robot kinematics configurations...")
        self.handover_cfg, self.robots, _ = load_stage1_robots()
        self.fingerprint = compute_config_fingerprint(self.robots, self.handover_cfg)
        print(f"[WebTwin] Config fingerprint: {self.fingerprint}")

        # Load Stage 3 Test 2 auto-handover poses
        self.cache_path = resolve_test2_handover_cache_path()
        self.auto_poses = {}
        if os.path.exists(self.cache_path):
            try:
                self.auto_poses = load_test2_handover_cache(self.cache_path, self.fingerprint) or {}
                if not self.auto_poses:
                    print(f"[WebTwin] No cached auto-handover poses found for fingerprint {self.fingerprint}.")
                else:
                    print(f"[WebTwin] Loaded {len(self.auto_poses)} auto-handover pair poses from cache for {self.fingerprint}.")
            except Exception as e:
                print(f"[WebTwin] Note on loading cache: {e}")

        # Target IK cache (Always bind to current configuration fingerprint)
        self.cache_filepath = resolve_cache_filepath()
        self.target_cache_fp = self.fingerprint

        # Live state
        self.current_q = {
            "robot1": [float(v) for v in self.robots["robot1"].home_deg],
            "robot2": [float(v) for v in self.robots["robot2"].home_deg],
            "robot3": [float(v) for v in self.robots["robot3"].home_deg],
        }
        self.gripper_state = {
            "robot1": "open",
            "robot2": "open",
            "robot3": "open",
        }
        # Default targets identical to Stage 3 Test 2 Tkinter GUI
        self.default_pick = [100.0, 100.0, 200.0]
        self.default_place = [0.0, 150.0, 150.0]
        self.reachability_cache = {}

    def get_initial_payload(self) -> dict:
        robot_data = {}
        for rid, cfg in self.robots.items():
            q_rad = np.deg2rad(cfg.home_deg)
            pos_base = all_frame_positions(q_rad, cfg)
            pos_world = [
                [round(float(x), 2) for x in (cfg.world_base_R @ p + cfg.world_base_xyz)]
                for p in pos_base
            ]
            tcp_base, R_flange = fk(q_rad, cfg)
            tcp_world = [
                round(float(x), 2) for x in (cfg.world_base_R @ tcp_base + cfg.world_base_xyz)
            ]
            R_tcp_world = cfg.world_base_R @ R_flange[:3, :3]
            rpy_world = [round(float(v), 1) for v in np.rad2deg(matrix_to_euler(R_tcp_world))]

            # Build DH parameters table for single-robot inspector (testGUI 1, 2, 3)
            dh_table = []
            for j in range(cfg.num_joints):
                dh_table.append({
                    "joint": f"J{j+1}",
                    "theta_offset_deg": round(float(cfg.theta_offset_deg[j]), 1) if j < len(cfg.theta_offset_deg) else 0.0,
                    "d_mm": round(float(cfg.d[j]), 1) if j < len(cfg.d) else 0.0,
                    "a_mm": round(float(cfg.a[j]), 1) if j < len(cfg.a) else 0.0,
                    "alpha_deg": round(float(cfg.alpha_deg[j]), 1) if j < len(cfg.alpha_deg) else 0.0,
                    "limit_min_deg": round(float(cfg.joint_limits_deg[j][0]), 1),
                    "limit_max_deg": round(float(cfg.joint_limits_deg[j][1]), 1),
                })

            robot_data[rid] = {
                "id": rid,
                "dof": len(cfg.home_deg),
                "home_deg": [float(v) for v in cfg.home_deg],
                "joint_limits_deg": [
                    [float(r[0]), float(r[1])] for r in cfg.joint_limits_deg
                ],
                "base_xyz": [round(float(v), 2) for v in cfg.world_base_xyz],
                "base_rpy_deg": self.handover_cfg["robots"][rid]["world_base_rpy_deg"],
                "frames_world": pos_world,
                "tcp_world": tcp_world,
                "tcp_rpy_deg": rpy_world,
                "tcp_R": [
                    [round(float(v), 4) for v in row] for row in R_tcp_world
                ],
                "finger_axis": cfg.gripper_finger_axis,
                "open_angle": float(cfg.gripper_open),
                "close_angle": float(cfg.gripper_close),
                "dh_table": dh_table,
            }

        # Handover pair solutions (both Auto Kinematics and Manual Override)
        handover_pairs = self.get_handover_data()
        handover_summaries = {}
        for p_k, p_dat in handover_pairs.items():
            if p_dat["auto"]["available"]:
                handover_summaries[p_k] = p_dat["auto"]

        min_d, pt_a, pt_b = compute_laser_endpoints(
            self.current_q["robot1"],
            self.robots["robot1"],
            self.current_q["robot2"],
            self.robots["robot2"],
        )

        initial_inspection = self.get_kinematics_inspection("robot1")
        ports = ros_bridge.get_available_ports() if "ros_bridge" in globals() and ros_bridge else []
        hw_all = ros_bridge.get_all_robot_states() if "ros_bridge" in globals() and ros_bridge else {}
        hw_states = hw_all.get("robots", hw_all)

        return {
            "type": "initial_state",
            "robots": robot_data,
            "handover_pairs": handover_pairs,
            "handover_poses": handover_summaries,
            "default_pick": self.default_pick,
            "default_place": self.default_place,
            "current_q": self.current_q,
            "clearance": {
                "min_distance_mm": round(min_d, 2),
                "endpoint_a": pt_a,
                "endpoint_b": pt_b,
                "pair": "robot1_robot2",
            },
            "kinematics_inspection": initial_inspection,
            "available_ports": ports,
            "robot_hardware": hw_states,
        }

    def get_handover_data(self) -> dict:
        """Get structured auto and manual handover data for all robot pairs."""
        pairs = ["robot1_robot2", "robot2_robot3", "robot3_robot1"]
        result = {}

        for pair_key in pairs:
            parts = pair_key.split("_")
            rA, rB = parts[0], parts[1]
            inv_key = f"{rB}_{rA}"

            # 1. Auto Kinematics pose (Top-of-Workspace)
            p = self.auto_poses.get(pair_key) or self.auto_poses.get(inv_key)
            if p is not None:
                auto_info = {
                    "available": True,
                    "pair_name": p.pair_name,
                    "handover_position_world": [round(float(v), 2) for v in p.handover_position_world],
                    "standoff_distance_mm": round(float(p.standoff_distance_mm), 1),
                    "robot_a_id": p.robot_a_id,
                    "robot_b_id": p.robot_b_id,
                    "handover_clearance_mm": round(float(p.handover_clearance_mm), 1),
                    "pre_handover_clearance_mm": round(float(p.pre_handover_clearance_mm), 1),
                    "approach_mode": p.approach_mode,
                    "candidate_origin": p.candidate_origin,
                    "dot_z": round(float(p.dot_z), 4),
                    "dot_x": round(float(p.dot_x), 4),
                    "approach_angle_dev_deg": round(float(p.approach_angle_dev_deg), 2),
                }
            else:
                auto_info = {
                    "available": False,
                    "pair_name": pair_key,
                    "approach_mode": "NO AUTO SOLUTION",
                    "error": "Cache empty or invalid configuration fingerprint.",
                }

            # 2. Manual Override pose (from handover.yaml)
            res = lookup_manual_handover_with_pre(self.handover_cfg, rA, rB)
            if res is not None:
                qA, qB, pre_qA, pre_qB = res
                tcpa_b, _ = fk(np.deg2rad(qA), self.robots[rA])
                tcpa_w = self.robots[rA].world_base_R @ tcpa_b + self.robots[rA].world_base_xyz
                tcpb_b, _ = fk(np.deg2rad(qB), self.robots[rB])
                tcpb_w = self.robots[rB].world_base_R @ tcpb_b + self.robots[rB].world_base_xyz
                midpoint = [round(float(x), 1) for x in (tcpa_w + tcpb_w) / 2.0]
                dist_mm, _, _ = compute_laser_endpoints(
                    np.array(qA), self.robots[rA],
                    np.array(qB), self.robots[rB],
                )
                manual_info = {
                    "available": True,
                    "pair_name": pair_key,
                    "handover_position_world": midpoint,
                    "standoff_distance_mm": 50.0,
                    "handover_clearance_mm": round(float(dist_mm), 1),
                    "approach_mode": "MANUAL (handover.yaml)",
                    "robot_a_id": rA,
                    "robot_b_id": rB,
                    "robot_a_q": [round(float(v), 1) for v in qA],
                    "robot_b_q": [round(float(v), 1) for v in qB],
                    "robot_a_pre_q": [round(float(v), 1) for v in pre_qA],
                    "robot_b_pre_q": [round(float(v), 1) for v in pre_qB],
                }
            else:
                manual_info = {
                    "available": False,
                    "pair_name": pair_key,
                    "approach_mode": "MANUAL UNAVAILABLE",
                }

            result[pair_key] = {
                "auto": auto_info,
                "manual": manual_info,
            }

        return result

    def reload_handover_cache(self) -> dict:
        """Reload auto-handover poses from disk cache."""
        self.fingerprint = compute_config_fingerprint(self.robots, self.handover_cfg)
        self.auto_poses = {}
        if os.path.exists(self.cache_path):
            try:
                self.auto_poses = load_test2_handover_cache(self.cache_path, self.fingerprint) or {}
                print(f"[WebTwin] Reloaded {len(self.auto_poses)} auto-handover poses from cache.")
            except Exception as e:
                print(f"[WebTwin] Error reloading cache: {e}")
        return self.get_handover_data()

    def preview_handover(self, pair: str, which: str = "handover", mode: str = "auto") -> dict:
        """Preview either the mated handover or pre-handover stance in 3D (Auto or Manual)."""
        parts = pair.split("_") if "_" in pair else ["robot1", "robot2"]
        rA, rB = parts[0], parts[1]
        inv_key = f"{rB}_{rA}"

        if mode == "manual":
            res = lookup_manual_handover_with_pre(self.handover_cfg, rA, rB)
            if not res:
                return {
                    "cmd": "preview_handover_result",
                    "success": False,
                    "error": f"No manual handover postures defined for {pair} in handover.yaml.",
                    "pair": pair,
                    "mode": "manual",
                }

            qA, qB, pre_qA, pre_qB = res
            sel_qA = pre_qA if which == "pre_handover" else qA
            sel_qB = pre_qB if which == "pre_handover" else qB

            self.current_q[rA] = [float(v) for v in sel_qA]
            self.current_q[rB] = [float(v) for v in sel_qB]
            other = [r for r in ["robot1", "robot2", "robot3"] if r not in (rA, rB)][0]
            self.current_q[other] = [float(v) for v in self.robots[other].home_deg]

            frame_data = self.compute_frames(self.current_q)
            dist_mm, pt_a, pt_b = compute_laser_endpoints(
                np.array(self.current_q[rA]), self.robots[rA],
                np.array(self.current_q[rB]), self.robots[rB],
            )
            tcpa_w = np.array(frame_data[rA]["tcp_world"])
            tcpb_w = np.array(frame_data[rB]["tcp_world"])
            midpoint = [round(float(x), 1) for x in (tcpa_w + tcpb_w) / 2.0]

            return {
                "success": True,
                "cmd": "preview_handover_result",
                "pair": pair,
                "which": which,
                "mode": "manual",
                "joint_state": {k: list(v) for k, v in self.current_q.items()},
                "frames": frame_data,
                "clearance": {
                    "min_distance_mm": round(dist_mm, 2),
                    "endpoint_a": pt_a,
                    "endpoint_b": pt_b,
                    "pair": pair,
                },
                "standoff_mm": 50.0,
                "approach_mode": "MANUAL (handover.yaml)",
                "handover_xyz": midpoint,
            }

        # Auto Kinematics mode
        pose_obj = self.auto_poses.get(pair) or self.auto_poses.get(inv_key)
        if not pose_obj:
            return {
                "cmd": "preview_handover_result",
                "success": False,
                "error": f"No auto-solution available for {pair}. Cache empty or invalid configuration fingerprint. Click 'Recompute (Top-WS)' or select Manual Override.",
                "pair": pair,
                "mode": "auto",
                "no_auto": True,
            }

        rA = pose_obj.robot_a_id
        rB = pose_obj.robot_b_id

        if which == "handover":
            qA = [float(v) for v in pose_obj.robot_a_handover_q_deg]
            qB = [float(v) for v in pose_obj.robot_b_handover_q_deg]
        else:
            qA = [float(v) for v in pose_obj.robot_a_pre_handover_q_deg]
            qB = [float(v) for v in pose_obj.robot_b_pre_handover_q_deg]

        self.current_q[rA] = qA
        self.current_q[rB] = qB
        other = [r for r in ["robot1", "robot2", "robot3"] if r not in (rA, rB)][0]
        self.current_q[other] = [float(v) for v in self.robots[other].home_deg]

        frame_data = self.compute_frames(self.current_q)
        dist_mm, pt_a, pt_b = compute_laser_endpoints(
            np.array(self.current_q[rA]), self.robots[rA],
            np.array(self.current_q[rB]), self.robots[rB],
        )

        return {
            "success": True,
            "cmd": "preview_handover_result",
            "pair": pair,
            "which": which,
            "mode": "auto",
            "joint_state": {k: list(v) for k, v in self.current_q.items()},
            "frames": frame_data,
            "clearance": {
                "min_distance_mm": round(dist_mm, 2),
                "endpoint_a": pt_a,
                "endpoint_b": pt_b,
                "pair": pair,
            },
            "standoff_mm": round(float(pose_obj.standoff_distance_mm), 1),
            "approach_mode": pose_obj.approach_mode,
            "handover_xyz": [round(float(x), 1) for x in pose_obj.handover_position_world],
        }

    def manual_jog_joint(self, robot_id: str, joint_idx: int, angle_deg: float) -> dict:
        """Jog a single joint angle in degrees with live FK and clearance calculation."""
        if robot_id in self.current_q:
            if 0 <= joint_idx < len(self.current_q[robot_id]):
                limits = self.robots[robot_id].joint_limits_deg[joint_idx]
                angle_deg = float(np.clip(angle_deg, limits[0], limits[1]))
                self.current_q[robot_id][joint_idx] = angle_deg

        frame_data = self.compute_frames(self.current_q)
        min_dist = float('inf')
        best_a, best_b, best_pair = [0, 0, 0], [0, 0, 0], "robot1_robot2"
        for ra, rb in [("robot1", "robot2"), ("robot2", "robot3"), ("robot3", "robot1")]:
            d, pa, pb = compute_laser_endpoints(
                np.array(self.current_q[ra]), self.robots[ra],
                np.array(self.current_q[rb]), self.robots[rb],
            )
            if d < min_dist:
                min_dist = d
                best_a, best_b, best_pair = pa, pb, f"{ra}_{rb}"

        return {
            "success": True,
            "cmd": "manual_jog_result",
            "robot_id": robot_id,
            "joint_state": {k: list(v) for k, v in self.current_q.items()},
            "frames": frame_data,
            "clearance": {
                "min_distance_mm": round(min_dist, 2),
                "endpoint_a": best_a,
                "endpoint_b": best_b,
                "pair": best_pair,
            },
        }

    def toggle_gripper(self, robot_id: str, state: str) -> dict:
        """Open or close gripper for a specific robot."""
        self.gripper_state[robot_id] = state
        return {
            "success": True,
            "cmd": "gripper_result",
            "robot_id": robot_id,
            "state": state,
            "grippers": dict(self.gripper_state),
        }

    def compute_frames(self, q_dict: Dict[str, List[float]]) -> Dict[str, Any]:
        """Compute all 3D joint frame positions for given joint angles."""
        res = {}
        for rid, q_deg in q_dict.items():
            cfg = self.robots[rid]
            q_rad = np.deg2rad(q_deg)
            pos_base = all_frame_positions(q_rad, cfg)
            pos_world = [
                [round(float(x), 2) for x in (cfg.world_base_R @ p + cfg.world_base_xyz)]
                for p in pos_base
            ]
            tcp_base, R_flange = fk(q_rad, cfg)
            tcp_world = [
                round(float(x), 2) for x in (cfg.world_base_R @ tcp_base + cfg.world_base_xyz)
            ]
            R_tcp_world = cfg.world_base_R @ R_flange[:3, :3]
            rpy_world = [round(float(v), 1) for v in np.rad2deg(matrix_to_euler(R_tcp_world))]

            res[rid] = {
                "frames_world": pos_world,
                "tcp_world": tcp_world,
                "tcp_rpy_deg": rpy_world,
                "tcp_R": [
                    [round(float(v), 4) for v in row] for row in R_tcp_world
                ],
            }
        return res

    def get_kinematics_inspection(self, robot_id: str, q_deg: Optional[List[float]] = None) -> dict:
        """Calculate live 4x4 Transformation Matrix T0n, dynamic DH table, and system geometry."""
        if robot_id not in self.robots:
            return {"success": False, "error": f"Unknown robot {robot_id}"}

        cfg = self.robots[robot_id]
        if q_deg is None:
            q_deg = list(self.current_q.get(robot_id, cfg.home_deg))

        q_rad = np.deg2rad(q_deg)
        p_tcp, T_flange = fk(q_rad, cfg)

        p_world = cfg.world_base_R @ p_tcp + cfg.world_base_xyz
        T_world = np.eye(4)
        T_world[:3, :3] = cfg.world_base_R @ T_flange[:3, :3]
        T_world[:3, 3] = p_world

        theta_dh = dh_joint_angles(q_rad, cfg)
        dh_rows = []
        for i in range(cfg.num_frames):
            th_deg = float(np.rad2deg(theta_dh[i]))
            dh_rows.append({
                "frame": f"F{i+1}",
                "joint": f"J{cfg.theta_joint_index[i]+1}" if cfg.theta_joint_index[i] >= 0 else "-",
                "a_mm": round(float(cfg.a[i]), 2) if i < len(cfg.a) else 0.0,
                "d_mm": round(float(cfg.d[i]), 2) if i < len(cfg.d) else 0.0,
                "alpha_deg": round(float(cfg.alpha_deg[i]), 1) if i < len(cfg.alpha_deg) else 0.0,
                "theta_deg": round(th_deg, 1),
                "limit_min_deg": round(float(cfg.joint_limits_deg[i][0]), 1) if i < len(cfg.joint_limits_deg) else -180.0,
                "limit_max_deg": round(float(cfg.joint_limits_deg[i][1]), 1) if i < len(cfg.joint_limits_deg) else 180.0,
            })

        t_matrix = [[round(float(val), 3) for val in row] for row in T_world]
        t_base_matrix = [[round(float(val), 3) for val in row] for row in T_flange]
        rpy_world = [round(float(v), 1) for v in np.rad2deg(matrix_to_euler(T_world[:3, :3]))]

        geometry = {
            "base_height_mm": round(float(cfg.d[0]), 2) if len(cfg.d) > 0 else 0.0,
            "link1_mm": round(float(cfg.a[2]), 2) if len(cfg.a) > 2 else 0.0,
            "link2_mm": round(float(cfg.d[3]), 2) if len(cfg.d) > 3 else 0.0,
            "link3_mm": round(float(cfg.a[5]), 2) if len(cfg.a) > 5 else 0.0,
            "link4_mm": round(float(cfg.d[6]), 2) if len(cfg.d) > 6 else 0.0,
            "tool_length_mm": 0.0,
            "reach_radius_mm": 370.0,
            "dof": cfg.num_joints,
        }

        return {
            "success": True,
            "robot_id": robot_id,
            "q_deg": [round(float(v), 2) for v in q_deg],
            "tcp_world": [round(float(v), 2) for v in p_world],
            "tcp_rpy_deg": rpy_world,
            "tcp_base": [round(float(v), 2) for v in p_tcp],
            "transformation_matrix": t_matrix,
            "transformation_matrix_base": t_base_matrix,
            "dh_table": dh_rows,
            "geometry": geometry,
        }

    def solve_single_ik(self, robot_id: str, target_xyz_world: List[float], target_yaw: float = 0.0) -> dict:
        """Direct XYZ + Yaw Inverse Kinematics solver for a single robot (matching individual GUIs)."""
        if robot_id not in self.robots:
            return {
                "cmd": "solve_single_ik_result",
                "success": False,
                "robot_id": robot_id,
                "target_xyz": target_xyz_world,
                "target_yaw": target_yaw,
                "error": f"Unknown robot {robot_id}",
            }
        try:
            cfg = self.robots[robot_id]
            target_world = np.array(target_xyz_world, dtype=float)
            current_q_deg = np.array(self.current_q[robot_id], dtype=float)

            res = solve_stage1_ik(
                robot_id=robot_id,
                target_xyz_world=target_world,
                config=cfg,
                current_q_deg=current_q_deg,
                num_random_starts=8,
            )
            if res is None:
                # Fallback: check if coordinates were given in robot's local base frame (matching robot_gui.py)
                target_world_alt = cfg.world_base_R @ target_world + cfg.world_base_xyz
                res = solve_stage1_ik(
                    robot_id=robot_id,
                    target_xyz_world=target_world_alt,
                    config=cfg,
                    current_q_deg=current_q_deg,
                    num_random_starts=8,
                )

            if res is None:
                from robot_control.handover.stage1_ik import world_to_base
                from robot_control.kinematics.ik_solver import ik_single
                target_base = world_to_base(target_world, cfg)
                current_q_rad = np.deg2rad(current_q_deg)
                single_res = ik_single(target_base, cfg, q_init=current_q_rad, max_iterations=100)
                err_mm = float(single_res.get("error", 999.0))
                return {
                    "cmd": "solve_single_ik_result",
                    "success": False,
                    "robot_id": robot_id,
                    "target_xyz": target_xyz_world,
                    "target_yaw": target_yaw,
                    "error_mm": round(err_mm, 3),
                    "error": f"IK FAILED | High error = {err_mm:.3f} mm (Target [{target_xyz_world[0]}, {target_xyz_world[1]}, {target_xyz_world[2]}] mm unreachable by {robot_id.upper()})",
                }

            self.current_q[robot_id] = [round(float(v), 2) for v in res.q_deg]
            frame_data = self.compute_frames(self.current_q)
            inspection = self.get_kinematics_inspection(robot_id, self.current_q[robot_id])

            min_dist = float("inf")
            best_a, best_b = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            for other_id in ["robot1", "robot2", "robot3"]:
                if other_id != robot_id:
                    d, pa, pb = compute_laser_endpoints(
                        np.array(self.current_q[robot_id]),
                        cfg,
                        np.array(self.current_q[other_id]),
                        self.robots[other_id],
                    )
                    if d < min_dist:
                        min_dist = d
                        best_a, best_b = pa, pb

            return {
                "cmd": "solve_single_ik_result",
                "success": True,
                "robot_id": robot_id,
                "target_xyz": target_xyz_world,
                "target_yaw": target_yaw,
                "error_mm": round(float(res.error_mm), 3),
                "joint_angles": self.current_q[robot_id],
                "joint_state": self.current_q,
                "frames": frame_data,
                "kinematics_inspection": inspection,
                "clearance": {
                    "min_distance_mm": round(min_dist, 2),
                    "endpoint_a": best_a,
                    "endpoint_b": best_b,
                },
            }
        except Exception as e:
            return {
                "cmd": "solve_single_ik_result",
                "success": False,
                "robot_id": robot_id,
                "target_xyz": target_xyz_world,
                "target_yaw": target_yaw,
                "error": f"IK Solver error: {str(e)}",
            }

    def plan_single_pick_place(
        self,
        robot_id: str,
        pick_xyz: List[float],
        pick_yaw: float,
        place_xyz: List[float],
        place_yaw: float,
    ) -> dict:
        """Autonomous single-robot Pick & Place trajectory generator (matching individual GUIs)."""
        if robot_id not in self.robots:
            return {"success": False, "error": f"Unknown robot {robot_id}"}

        cfg = self.robots[robot_id]
        current_q = np.array(self.current_q[robot_id], dtype=float)

        pick_arr = np.array(pick_xyz, dtype=float)
        pick_res = solve_stage1_ik(robot_id, pick_arr, cfg, current_q, num_random_starts=8)
        if pick_res is None:
            pick_world_alt = cfg.world_base_R @ pick_arr + cfg.world_base_xyz
            pick_res = solve_stage1_ik(robot_id, pick_world_alt, cfg, current_q, num_random_starts=8)

        if pick_res is None:
            return {
                "success": False,
                "error": f"Pick position [{pick_xyz[0]}, {pick_xyz[1]}, {pick_xyz[2]}] mm is outside reach envelope of {robot_id.upper()}."
            }

        place_arr = np.array(place_xyz, dtype=float)
        place_res = solve_stage1_ik(robot_id, place_arr, cfg, pick_res.q_deg, num_random_starts=8)
        if place_res is None:
            place_world_alt = cfg.world_base_R @ place_arr + cfg.world_base_xyz
            place_res = solve_stage1_ik(robot_id, place_world_alt, cfg, pick_res.q_deg, num_random_starts=8)

        if place_res is None:
            return {
                "success": False,
                "error": f"Place position [{place_xyz[0]}, {place_xyz[1]}, {place_xyz[2]}] mm is outside reach envelope of {robot_id.upper()}."
            }

        home_deg = [float(v) for v in cfg.home_deg]
        pick_deg = [round(float(v), 2) for v in pick_res.q_deg]
        place_deg = [round(float(v), 2) for v in place_res.q_deg]

        # Pre-pick and Pre-place Cartesian heights (Z + 50 mm)
        pre_pick_xyz = [pick_xyz[0], pick_xyz[1], pick_xyz[2] + 50.0]
        pre_place_xyz = [place_xyz[0], place_xyz[1], place_xyz[2] + 50.0]

        pre_pick_res = solve_stage1_ik(robot_id, pre_pick_xyz, cfg, current_q, num_random_starts=8)
        pre_pick_deg = [round(float(v), 2) for v in pre_pick_res.q_deg] if pre_pick_res else pick_deg

        pre_place_res = solve_stage1_ik(robot_id, pre_place_xyz, cfg, pick_res.q_deg, num_random_starts=8)
        pre_place_deg = [round(float(v), 2) for v in pre_place_res.q_deg] if pre_place_res else place_deg

        steps = []
        def add_interp(q_start, q_end, num_steps, phase_name, grip_state):
            for s in range(num_steps):
                alpha = (s + 1) / float(num_steps)
                q_interp = [float(q_start[j] + alpha * (q_end[j] - q_start[j])) for j in range(len(q_start))]
                full_q = {r: list(self.current_q[r]) for r in ["robot1", "robot2", "robot3"]}
                full_q[robot_id] = q_interp
                steps.append({
                    "step_idx": len(steps),
                    "phase": phase_name,
                    "joint_state": full_q,
                    "frames": self.compute_frames(full_q),
                    "gripper_state": {robot_id: grip_state},
                })

        # Phase 0 & 1a: Proactive open gripper + Free-space transit to Pre-Pick (Z + 50mm)
        add_interp(self.current_q[robot_id], pre_pick_deg, 12, "Phase 1a: Move to Pre-Pick (+50mm)", "open")

        # Phase 1b: Linear Vertical Descent to Pick (Cartesian straight line -Z)
        cart_descent = compute_cartesian_linear_path(pre_pick_deg, pick_xyz, cfg, num_steps=8, frame="world")
        if cart_descent is not None:
            for q_step in cart_descent:
                full_q = {r: list(self.current_q[r]) for r in ["robot1", "robot2", "robot3"]}
                full_q[robot_id] = [round(float(v), 2) for v in q_step]
                steps.append({
                    "step_idx": len(steps),
                    "phase": "Phase 1b: Cartesian Descent to Pick (-Z)",
                    "joint_state": full_q,
                    "frames": self.compute_frames(full_q),
                    "gripper_state": {robot_id: "open"},
                })
        else:
            add_interp(pre_pick_deg, pick_deg, 8, "Phase 1b: Cartesian Descent to Pick", "open")

        # Phase 2: Grasp Object with settling dwell
        for _ in range(4):
            full_q = {r: list(self.current_q[r]) for r in ["robot1", "robot2", "robot3"]}
            full_q[robot_id] = pick_deg
            steps.append({
                "step_idx": len(steps),
                "phase": "Phase 2: Grasp & Dwell (Gripper Closed)",
                "joint_state": full_q,
                "frames": self.compute_frames(full_q),
                "gripper_state": {robot_id: "close"},
            })

        # Phase 3: Linear Vertical Lift to Pre-Pick (Cartesian straight line +50mm)
        cart_lift = compute_cartesian_linear_path(pick_deg, pre_pick_xyz, cfg, num_steps=8, frame="world")
        if cart_lift is not None:
            for q_step in cart_lift:
                full_q = {r: list(self.current_q[r]) for r in ["robot1", "robot2", "robot3"]}
                full_q[robot_id] = [round(float(v), 2) for v in q_step]
                steps.append({
                    "step_idx": len(steps),
                    "phase": "Phase 3: Cartesian Lift (+50mm)",
                    "joint_state": full_q,
                    "frames": self.compute_frames(full_q),
                    "gripper_state": {robot_id: "close"},
                })
        else:
            add_interp(pick_deg, pre_pick_deg, 8, "Phase 3: Cartesian Lift (+50mm)", "close")

        # Phase 4a: Free-space transit to Pre-Place (+50mm)
        add_interp(pre_pick_deg, pre_place_deg, 12, "Phase 4a: Transit to Pre-Place (+50mm)", "close")

        # Phase 4b: Linear Vertical Descent to Place (Cartesian straight line -Z)
        cart_descent_pl = compute_cartesian_linear_path(pre_place_deg, place_xyz, cfg, num_steps=8, frame="world")
        if cart_descent_pl is not None:
            for q_step in cart_descent_pl:
                full_q = {r: list(self.current_q[r]) for r in ["robot1", "robot2", "robot3"]}
                full_q[robot_id] = [round(float(v), 2) for v in q_step]
                steps.append({
                    "step_idx": len(steps),
                    "phase": "Phase 4b: Cartesian Descent to Place (-Z)",
                    "joint_state": full_q,
                    "frames": self.compute_frames(full_q),
                    "gripper_state": {robot_id: "close"},
                })
        else:
            add_interp(pre_place_deg, place_deg, 8, "Phase 4b: Cartesian Descent to Place", "close")

        # Phase 5: Release Object with settling dwell
        for _ in range(4):
            full_q = {r: list(self.current_q[r]) for r in ["robot1", "robot2", "robot3"]}
            full_q[robot_id] = place_deg
            steps.append({
                "step_idx": len(steps),
                "phase": "Phase 5: Release & Dwell (Gripper Open)",
                "joint_state": full_q,
                "frames": self.compute_frames(full_q),
                "gripper_state": {robot_id: "open"},
            })

        # Phase 6: Linear Vertical Retraction to Pre-Place (Cartesian straight line +50mm)
        cart_retract_pl = compute_cartesian_linear_path(place_deg, pre_place_xyz, cfg, num_steps=8, frame="world")
        if cart_retract_pl is not None:
            for q_step in cart_retract_pl:
                full_q = {r: list(self.current_q[r]) for r in ["robot1", "robot2", "robot3"]}
                full_q[robot_id] = [round(float(v), 2) for v in q_step]
                steps.append({
                    "step_idx": len(steps),
                    "phase": "Phase 6: Cartesian Retraction (+50mm)",
                    "joint_state": full_q,
                    "frames": self.compute_frames(full_q),
                    "gripper_state": {robot_id: "open"},
                })
        else:
            add_interp(place_deg, pre_place_deg, 8, "Phase 6: Cartesian Retraction (+50mm)", "open")

        # Phase 7: Return to Mechanical Home
        add_interp(pre_place_deg, home_deg, 10, "Phase 7: Return to Mechanical Home", "open")

        return {
            "success": True,
            "cmd": "single_pick_place_planned",
            "robot_id": robot_id,
            "pick_xyz": pick_xyz,
            "pick_yaw": pick_yaw,
            "place_xyz": place_xyz,
            "place_yaw": place_yaw,
            "pick_q_deg": pick_deg,
            "place_q_deg": place_deg,
            "steps": steps,
            "total_steps": len(steps),
        }


    def check_reachability(self, target_xyz: List[float], target_yaw: float = 0.0) -> dict:
        """Query reachability across all robots using Stage 2.5 object IK and fallback."""
        cache_key = (
            round(float(target_xyz[0]), 1),
            round(float(target_xyz[1]), 1),
            round(float(target_xyz[2]), 1),
            round(float(target_yaw), 1),
        )
        if cache_key in self.reachability_cache:
            return dict(self.reachability_cache[cache_key])

        try:
            cached = lookup_cached_target_ik(self.cache_filepath, self.target_cache_fp, target_xyz, target_yaw)
            if cached is not None:
                opts, is_fallback = cached
            else:
                opts, is_fallback = resolve_stage2_5_target(
                    target_xyz_world=target_xyz,
                    robots=self.robots,
                    target_yaw_deg=target_yaw,
                )
                try:
                    store_cached_target_ik(self.cache_filepath, self.target_cache_fp, target_xyz, target_yaw, opts, is_fallback)
                except Exception:
                    pass
            reachable = {}
            for rid, sol in opts.items():
                reachable[rid] = {
                    "robot_id": rid,
                    "solver_type": sol.solver_type,
                    "is_fallback": sol.is_fallback,
                    "error_mm": round(float(sol.error_mm), 3),
                    "q_deg": [round(float(v), 2) for v in sol.q_deg],
                    "approach_tilt_deg": round(float(sol.approach_tilt_deg), 1)
                    if sol.approach_tilt_deg is not None
                    else None,
                }
            result = {
                "success": len(reachable) > 0,
                "reachable_robots": reachable,
                "is_fallback": is_fallback,
                "count": len(reachable),
            }
            self.reachability_cache[cache_key] = result
            return result
        except Exception as e:
            return {"success": False, "error": str(e), "reachable_robots": {}}

    def plan_trajectory(
        self,
        pick_xyz: List[float],
        pick_yaw: float,
        place_xyz: List[float],
        place_yaw: float,
        chosen_pick_robot: Optional[str] = None,
        chosen_place_robot: Optional[str] = None,
        handover_mode: str = "auto",
    ) -> dict:
        """Generate Stage 3 Test 2 plan and discretized 9-phase trajectory."""
        print(f"[WebTwin] Planning trajectory: Pick {pick_xyz} (yaw {pick_yaw}°), Place {place_xyz} (yaw {place_yaw}°), Mode: {handover_mode}")

        # Step 1: Check options to see if interactive choice is required
        cached_pick = lookup_cached_target_ik(self.cache_filepath, self.target_cache_fp, pick_xyz, pick_yaw)
        if cached_pick is not None:
            pick_opts, _ = cached_pick
        else:
            pick_opts, _ = resolve_stage2_5_target(pick_xyz, self.robots, pick_yaw, label="pick")
            try:
                store_cached_target_ik(self.cache_filepath, self.target_cache_fp, pick_xyz, pick_yaw, pick_opts, False)
            except Exception:
                pass

        cached_place = lookup_cached_target_ik(self.cache_filepath, self.target_cache_fp, place_xyz, place_yaw)
        if cached_place is not None:
            place_opts, _ = cached_place
        else:
            place_opts, _ = resolve_stage2_5_target(place_xyz, self.robots, place_yaw, label="place")
            try:
                store_cached_target_ik(self.cache_filepath, self.target_cache_fp, place_xyz, place_yaw, place_opts, False)
            except Exception:
                pass

        if not pick_opts:
            return {"success": False, "error": f"Target Pick {pick_xyz} is out of reach for all robots."}
        if not place_opts:
            return {"success": False, "error": f"Target Place {place_xyz} is out of reach for all robots."}

        pick_candidates = list(pick_opts.keys())
        place_candidates = list(place_opts.keys())

        # If user hasn't explicitly chosen and multiple options exist, ask client to show chooser modal
        if (len(pick_candidates) > 1 and chosen_pick_robot is None) or (
            len(place_candidates) > 1 and chosen_place_robot is None
        ):
            return {
                "success": True,
                "needs_choice": True,
                "pick_candidates": pick_candidates,
                "place_candidates": place_candidates,
                "pick_details": {
                    rid: {
                        "solver": pick_opts[rid].solver_type,
                        "tilt": pick_opts[rid].approach_tilt_deg,
                        "error_mm": round(float(pick_opts[rid].error_mm), 2),
                    }
                    for rid in pick_candidates
                },
                "place_details": {
                    rid: {
                        "solver": place_opts[rid].solver_type,
                        "tilt": place_opts[rid].approach_tilt_deg,
                        "error_mm": round(float(place_opts[rid].error_mm), 2),
                    }
                    for rid in place_candidates
                },
            }

        # Assign robots
        def chooser(label, choices):
            if label == "pick" and chosen_pick_robot and chosen_pick_robot in choices:
                return chosen_pick_robot
            if label == "place" and chosen_place_robot and chosen_place_robot in choices:
                return chosen_place_robot
            if label == "pick" and len(choices) > 1:
                # Prefer handover pair if place candidates are distinct
                other_choices = [c for c in choices if c not in place_candidates]
                if other_choices:
                    return other_choices[0]
            return choices[0]

        use_auto = (handover_mode == "auto")
        try:
            plan, _, _ = build_stage3_test2_plan(
                pick_xyz=pick_xyz,
                place_xyz=place_xyz,
                robots=self.robots,
                handover_cfg=self.handover_cfg,
                chooser=chooser,
                pick_yaw_deg=pick_yaw,
                place_yaw_deg=place_yaw,
                use_auto_handover=use_auto,
                cache_filepath=self.cache_filepath,
                config_fingerprint=self.target_cache_fp,
                test_poses=self.auto_poses if use_auto else None,
            )

            steps_per_phase = 16
            safe_dist = 45.0
            traj_result = plan_stage3_test2_paths(
                plan=plan,
                robots=self.robots,
                safe_distance=safe_dist,
                steps=steps_per_phase,
            )
        except Exception as exc:
            print(f"[WebTwin] Planning error: {exc}")
            return {"success": False, "error": str(exc)}

        # Discretize and package all trajectory steps for smooth 60 FPS animation
        steps_payload = []
        phase_definitions = [
            (1, "Phase 1: Approach Pre-Pick (+50mm) & Cartesian Descent (-Z)", traj_result.phase1_simultaneous),
            (2, "Phase 2: Settle & Gripper Grasp (Dwell)", None),  # Dwell 2.0s
            (3, "Phase 3: Cartesian Lift (+50mm) & Transit to Pre-Handover", traj_result.phase3_pick_to_pre_handover),
            (4, "Phase 4: Synchronized Advance along Handover Corridors", traj_result.phase4_advance_to_handover),
            (5, "Phase 5: Gripper Handover Transfer (Dwell)", None),  # Dwell 1.0s
            (6, "Phase 6: Synchronized Retraction along Reverse Corridors", traj_result.phase6_retract_to_pre_handover),
            (7, "Phase 7: Transfer to Pre-Place (+50mm) & Cartesian Descent (-Z)", traj_result.phase7_handover_to_place),
            (8, "Phase 8: Settle & Gripper Release (Dwell)", None),  # Dwell 2.0s
            (9, "Phase 9: Cartesian Retract (+50mm) & Simultaneous Return Home", traj_result.phase9_simultaneous_home),
        ]

        pick_r = plan.pick_robot
        place_r = plan.place_robot
        remaining = [r for r in ["robot1", "robot2", "robot3"] if r not in (pick_r, place_r)]
        other_r = remaining[0] if remaining else ("robot2" if pick_r != "robot2" else "robot1")
        laser_peer = place_r if pick_r != place_r else other_r

        current_joint_state = {
            r: [float(v) for v in self.robots[r].home_deg]
            for r in ["robot1", "robot2", "robot3"]
        }

        # Workpiece tracking
        workpiece_state = {
            "attached_to": None,
            "position": [float(pick_xyz[0]), float(pick_xyz[1]), 20.0],
            "rotation": [0.0, 0.0, float(pick_yaw)],
        }

        for phase_num, phase_title, step_list in phase_definitions:
            if step_list is not None:
                for step_idx, step in enumerate(step_list):
                    current_joint_state[pick_r] = [round(float(v), 2) for v in step.pick_q]
                    current_joint_state[place_r] = [round(float(v), 2) for v in step.place_q]

                    # Compute 3D frames
                    frame_data = self.compute_frames(current_joint_state)

                    # Determine gripper states & workpiece attachment
                    if pick_r == place_r:
                        g_act = "close" if phase_num in (2, 3, 4, 5, 6, 7) else "open"
                        g_pick = g_act
                        g_place = g_act
                        if 2 <= phase_num < 8:
                            workpiece_state = {
                                "attached_to": pick_r,
                                "position": frame_data[pick_r]["tcp_world"],
                                "rotation": [0.0, 0.0, float(pick_yaw)],
                            }
                        elif phase_num >= 8:
                            workpiece_state = {
                                "attached_to": None,
                                "position": [float(place_xyz[0]), float(place_xyz[1]), 20.0],
                                "rotation": [0.0, 0.0, float(place_yaw)],
                            }
                    else:
                        g_pick = "close" if 2 <= phase_num < 5 else "open"
                        g_place = "close" if 5 <= phase_num < 8 else "open"
                        if phase_num in (2, 3, 4):
                            workpiece_state = {
                                "attached_to": pick_r,
                                "position": frame_data[pick_r]["tcp_world"],
                                "rotation": [0.0, 0.0, float(pick_yaw)],
                            }
                        elif phase_num in (5, 6, 7):
                            workpiece_state = {
                                "attached_to": place_r,
                                "position": frame_data[place_r]["tcp_world"],
                                "rotation": [0.0, 0.0, float(place_yaw)],
                            }
                        elif phase_num >= 8:
                            workpiece_state = {
                                "attached_to": None,
                                "position": [float(place_xyz[0]), float(place_xyz[1]), 20.0],
                                "rotation": [0.0, 0.0, float(place_yaw)],
                            }

                    # Clearance & laser endpoints between active robots
                    dist_mm, pt_a, pt_b = compute_laser_endpoints(
                        np.array(current_joint_state[pick_r]),
                        self.robots[pick_r],
                        np.array(current_joint_state[laser_peer]),
                        self.robots[laser_peer],
                    )

                    steps_payload.append({
                        "phase_number": phase_num,
                        "phase_title": phase_title,
                        "step_index": len(steps_payload),
                        "joint_state": {k: list(v) for k, v in current_joint_state.items()},
                        "frames": frame_data,
                        "grippers": {
                            pick_r: g_pick,
                            place_r: g_place,
                            other_r: "open",
                        },
                        "workpiece": dict(workpiece_state),
                        "clearance": {
                            "min_distance_mm": round(dist_mm, 2),
                            "endpoint_a": pt_a,
                            "endpoint_b": pt_b,
                            "pair": f"{pick_r}_{laser_peer}",
                            "is_safe": dist_mm >= safe_dist,
                        },
                    })
            else:
                # Dwell step (Phases 2, 5, 8)
                if pick_r == place_r:
                    g_act = "close" if phase_num in (2, 5) else "open"
                    g_pick = g_act
                    g_place = g_act
                else:
                    g_pick = "close" if phase_num in (2, 5) else "open"
                    g_place = "close" if phase_num in (5, 8) else "open"
                frame_data = self.compute_frames(current_joint_state)
                dist_mm, pt_a, pt_b = compute_laser_endpoints(
                    np.array(current_joint_state[pick_r]),
                    self.robots[pick_r],
                    np.array(current_joint_state[laser_peer]),
                    self.robots[laser_peer],
                )
                steps_payload.append({
                    "phase_number": phase_num,
                    "phase_title": phase_title,
                    "step_index": len(steps_payload),
                    "joint_state": {k: list(v) for k, v in current_joint_state.items()},
                    "frames": frame_data,
                    "grippers": {
                        pick_r: g_pick,
                        place_r: g_place,
                        other_r: "open",
                    },
                    "workpiece": dict(workpiece_state),
                    "clearance": {
                        "min_distance_mm": round(dist_mm, 2),
                        "endpoint_a": pt_a,
                        "endpoint_b": pt_b,
                        "pair": f"{pick_r}_{laser_peer}",
                        "is_safe": dist_mm >= safe_dist,
                    },
                })

        min_overall_clr = (
            round(float(traj_result.min_overall_clearance_mm), 1)
            if not math.isinf(traj_result.min_overall_clearance_mm)
            else 999.0
        )
        clr_display_str = (
            f"{min_overall_clr} mm"
            if not math.isinf(traj_result.min_overall_clearance_mm)
            else "N/A (Single Robot)"
        )

        ho_mode_str = (
            "AUTO KINEMATICS (Top-of-WS)" if (plan.needs_handover and plan.use_auto_handover and plan.auto_handover_info)
            else ("DIRECT (Same Robot - No Handover)" if not plan.needs_handover else "MANUAL OVERRIDE (handover.yaml)")
        )

        return {
            "success": True,
            "needs_choice": False,
            "plan_summary": {
                "pick_robot": pick_r,
                "place_robot": place_r,
                "pick_solver": plan.pick.solver_type,
                "place_solver": plan.place.solver_type,
                "pick_tilt": round(float(plan.pick.approach_tilt_deg or 0.0), 1),
                "place_tilt": round(float(plan.place.approach_tilt_deg or 0.0), 1),
                "needs_handover": plan.needs_handover,
                "handover_pair": f"{pick_r}_{place_r}",
                "handover_mode": "auto" if plan.use_auto_handover else "manual",
                "handover_desc": ho_mode_str,
                "min_overall_clearance_mm": min_overall_clr,
                "all_phases_safe": traj_result.all_phases_safe,
                "total_steps": len(steps_payload),
                "approach_mode": getattr(plan, "approach_mode", "DUAL_SYNCHRONIZED"),
                "handover_xyz": [round(float(x), 1) for x in plan.auto_handover_info.handover_position_world]
                if plan.auto_handover_info else None,
                "standoff_mm": round(float(plan.auto_handover_info.standoff_distance_mm), 1)
                if plan.auto_handover_info else None,
                "handover_clearance_mm": round(float(plan.auto_handover_info.handover_clearance_mm), 1)
                if plan.auto_handover_info else None,
            },
            "planning_logs": [
                "=== STARTING STAGE 3 TEST 2 TASK EXECUTION ===",
                "[MODE: 3D SIMULATION PREVIEW] Strategy: TOP_OF_WORKSPACE",
                f"PICK: {pick_r.upper()} (Solver: {plan.pick.solver_type})",
                f"PLACE: {place_r.upper()} (Solver: {plan.place.solver_type})",
                f"HANDOVER: {ho_mode_str}",
                "Planning collision-free synchronized trajectories...",
                f"Min predicted body clearance: {clr_display_str} (Safe dist: {safe_dist:.1f} mm)",
                f">>> 1. SIMULTANEOUS: {pick_r} approaches Pre-Pick (+50mm) & Cartesian descends (-Z) to Pick. WHILE {place_r if plan.needs_handover else 'other'} moves to Pre-Handover!",
                f">>> 2. {pick_r} dwells 2.0s, closes gripper (grasp), and dwells 2.0s. ({place_r if plan.needs_handover else 'other'} waiting at Pre-Handover)",
                f">>> 3. {pick_r} Cartesian lifts (+50mm) to Pre-Pick and transfers to Pre-Handover.",
                f">>> 4. ADVANCE: Both robots enter Handover mating corridor." if plan.needs_handover else ">>> 4. Transit directly to Pre-Place.",
                f">>> 5. HANDOVER TRANSFER: {place_r} closes gripper, {pick_r} opens gripper." if plan.needs_handover else ">>> 5. Continuous transfer.",
                f">>> 6. RETRACT: Both robots retreat safely back to Pre-Handover along reverse corridors." if plan.needs_handover else ">>> 6. Settle.",
                f">>> 7. {place_r} transfers to Pre-Place (+50mm) & Cartesian descends (-Z) to Place Target.",
                f">>> 8. {place_r} dwells 2.0s, opens gripper at Place Target, and dwells 2.0s.",
                f">>> 9. {place_r} Cartesian retracts (+50mm) to Pre-Place & SIMULTANEOUS RETURN: Both robots return to Home!",
                "✓ Task completed successfully!",
            ],
            "steps": steps_payload,
        }


# Global engine & hardware bridge instances
twin_engine = WebTwinEngine()
ros_bridge = WebTwinRosBridge(twin_engine)

ACTIVE_WEBSOCKETS: set = set()


async def telemetry_broadcaster(app: web.Application):
    """High-speed 10 Hz broadcaster pushing live ROS 2 telemetry to all connected web clients."""
    print("[WebTwin] Started ROS 2 telemetry broadcast daemon (10 Hz).")
    while True:
        try:
            await asyncio.sleep(0.1)
            if ACTIVE_WEBSOCKETS and ros_bridge:
                hw_state = ros_bridge.get_all_robot_states()
                frame_data = twin_engine.compute_frames(twin_engine.current_q)
                min_d, pt_a, pt_b = compute_laser_endpoints(
                    np.array(twin_engine.current_q["robot1"]),
                    twin_engine.robots["robot1"],
                    np.array(twin_engine.current_q["robot2"]),
                    twin_engine.robots["robot2"],
                )
                payload = json.dumps({
                    "cmd": "telemetry_broadcast",
                    "hardware": hw_state,
                    "joint_state": twin_engine.current_q,
                    "frames": frame_data,
                    "clearance": {
                        "min_distance_mm": round(min_d, 2),
                        "endpoint_a": pt_a,
                        "endpoint_b": pt_b,
                    },
                    "grippers": twin_engine.gripper_state,
                })
                dead = set()
                for ws in list(ACTIVE_WEBSOCKETS):
                    if ws.closed:
                        dead.add(ws)
                    else:
                        try:
                            await ws.send_str(payload)
                        except Exception:
                            dead.add(ws)
                if dead:
                    ACTIVE_WEBSOCKETS.difference_update(dead)
        except asyncio.CancelledError:
            break
        except Exception as e:
            await asyncio.sleep(0.5)


async def handle_websocket(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=25.0)
    await ws.prepare(request)
    loop = asyncio.get_running_loop()
    ACTIVE_WEBSOCKETS.add(ws)
    print(f"[WebTwin] New client connected from {request.remote} (Total clients: {len(ACTIVE_WEBSOCKETS)})")

    # Send initial state immediately
    await ws.send_str(json.dumps(twin_engine.get_initial_payload()))

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    cmd = data.get("cmd")

                    if cmd == "get_initial_state":
                        await ws.send_str(json.dumps(twin_engine.get_initial_payload()))

                    elif cmd == "get_ports":
                        ports = ros_bridge.refresh_available_ports()
                        await ws.send_str(json.dumps({
                            "cmd": "ports_list",
                            "available_ports": ports,
                        }))

                    elif cmd == "set_robot_port":
                        robot_id = data.get("robot_id", "robot1")
                        port = data.get("port", "/dev/robot1")
                        res = ros_bridge.set_robot_port(robot_id, port)
                        await ws.send_str(json.dumps({"cmd": "set_port_result", **res}))

                    elif cmd == "disconnect_robot_port":
                        robot_id = data.get("robot_id", "robot1")
                        res = ros_bridge.disconnect_robot_port(robot_id)
                        await ws.send_str(json.dumps({"cmd": "disconnect_port_result", **res}))

                    elif cmd == "emergency_stop":
                        target = data.get("robot_id", "all")
                        res = ros_bridge.emergency_stop(target)
                        await ws.send_str(json.dumps({"cmd": "emergency_stop_result", **res}))

                    elif cmd == "resume_manual":
                        target = data.get("robot_id", "all")
                        res = ros_bridge.resume_manual(target)
                        await ws.send_str(json.dumps({"cmd": "resume_manual_result", **res}))

                    elif cmd == "home_robot":
                        target = data.get("robot_id", "robot1")
                        res = ros_bridge.home_robot(target)
                        frame_data = twin_engine.compute_frames(twin_engine.current_q)
                        await ws.send_str(json.dumps({
                            "cmd": "home_result",
                            "joint_state": twin_engine.current_q,
                            "frames": frame_data,
                            **res,
                        }))

                    elif cmd in ("set_robot_speed", "set_speed"):
                        robot_id = data.get("robot_id", "robot1")
                        speed = int(data.get("speed", 5))
                        res = ros_bridge.set_speed(robot_id, speed)
                        await ws.send_str(json.dumps({"cmd": "speed_result", **res}))

                    elif cmd == "set_gripper_limits":
                        robot_id = data.get("robot_id", "robot1")
                        open_d = float(data.get("open_deg", 90.0))
                        close_d = float(data.get("close_deg", 0.0))
                        res = ros_bridge.set_gripper_limits(robot_id, open_d, close_d)
                        await ws.send_str(json.dumps({"cmd": "gripper_limits_result", **res}))

                    elif cmd == "set_gripper_angle":
                        robot_id = data.get("robot_id", "robot1")
                        angle = float(data.get("angle", 90.0))
                        state = data.get("state", "CUSTOM")
                        res = ros_bridge.set_gripper(robot_id, angle, state)
                        twin_engine.gripper_state[robot_id] = "open" if state == "OPEN" else ("close" if state == "CLOSED" else state)
                        await ws.send_str(json.dumps({"cmd": "gripper_angle_result", **res}))

                    elif cmd == "get_kinematics_inspection":
                        robot_id = data.get("robot_id", "robot1")
                        q_deg = data.get("q_deg")
                        res = twin_engine.get_kinematics_inspection(robot_id, q_deg)
                        res["cmd"] = "kinematics_inspection_result"
                        await ws.send_str(json.dumps(res))

                    elif cmd == "plan_single_pick_place":
                        robot_id = data.get("robot_id", "robot1")
                        pick_xyz = data.get("pick_xyz", [200.0, 100.0, 150.0])
                        pick_yaw = float(data.get("pick_yaw", 0.0))
                        place_xyz = data.get("place_xyz", [150.0, -100.0, 180.0])
                        place_yaw = float(data.get("place_yaw", 0.0))
                        res = await loop.run_in_executor(
                            None,
                            twin_engine.plan_single_pick_place,
                            robot_id,
                            pick_xyz,
                            pick_yaw,
                            place_xyz,
                            place_yaw,
                        )
                        await ws.send_str(json.dumps(res))

                    elif cmd == "execute_single_pick_place":
                        robot_id = data.get("robot_id", "robot1")
                        pick_xyz = data.get("pick_xyz", [200.0, 100.0, 150.0])
                        pick_yaw = float(data.get("pick_yaw", 0.0))
                        place_xyz = data.get("place_xyz", [150.0, -100.0, 180.0])
                        place_yaw = float(data.get("place_yaw", 0.0))
                        res = ros_bridge.execute_single_pick_place(
                            robot_id, pick_xyz, pick_yaw, place_xyz, place_yaw
                        )
                        await ws.send_str(json.dumps({"cmd": "execute_single_result", **res}))

                    elif cmd == "execute_hardware_trajectory":
                        steps = data.get("steps", [])
                        delay = float(data.get("step_delay", 0.08))
                        res = ros_bridge.execute_multi_robot_trajectory(steps, step_delay=delay)
                        await ws.send_str(json.dumps({"cmd": "execute_hardware_result", **res}))

                    elif cmd == "check_reachability":
                        target_xyz = data.get("target_xyz", [0.0, 0.0, 200.0])
                        target_yaw = float(data.get("target_yaw", 0.0))
                        res = await loop.run_in_executor(
                            None, twin_engine.check_reachability, target_xyz, target_yaw
                        )
                        res["cmd"] = "reachability_result"
                        res["target_type"] = data.get("target_type", "pick")
                        await ws.send_str(json.dumps(res))

                    elif cmd == "manual_fk":
                        joint_state = data.get("joint_state", {})
                        frame_data = twin_engine.compute_frames(joint_state)
                        q1 = np.array(joint_state.get("robot1", twin_engine.current_q["robot1"]))
                        q2 = np.array(joint_state.get("robot2", twin_engine.current_q["robot2"]))
                        dist_mm, pt_a, pt_b = compute_laser_endpoints(
                            q1, twin_engine.robots["robot1"], q2, twin_engine.robots["robot2"]
                        )
                        await ws.send_str(json.dumps({
                            "cmd": "manual_fk_result",
                            "frames": frame_data,
                            "clearance": {
                                "min_distance_mm": round(dist_mm, 2),
                                "endpoint_a": pt_a,
                                "endpoint_b": pt_b,
                                "pair": "robot1_robot2",
                            },
                        }))

                    elif cmd == "plan_trajectory":
                        pick_xyz = data.get("pick_xyz", twin_engine.default_pick)
                        pick_yaw = float(data.get("pick_yaw", 0.0))
                        place_xyz = data.get("place_xyz", twin_engine.default_place)
                        place_yaw = float(data.get("place_yaw", 0.0))
                        chosen_pick = data.get("chosen_pick_robot")
                        chosen_place = data.get("chosen_place_robot")
                        handover_mode = data.get("handover_mode", "auto")

                        plan_res = await loop.run_in_executor(
                            None,
                            lambda: twin_engine.plan_trajectory(
                                pick_xyz=pick_xyz,
                                pick_yaw=pick_yaw,
                                place_xyz=place_xyz,
                                place_yaw=place_yaw,
                                chosen_pick_robot=chosen_pick,
                                chosen_place_robot=chosen_place,
                                handover_mode=handover_mode,
                            ),
                        )
                        plan_res["cmd"] = "trajectory_planned"
                        if not ws.closed:
                            await ws.send_str(json.dumps(plan_res))

                    elif cmd == "preview_handover":
                        pair = data.get("pair", "robot1_robot2")
                        which = data.get("which", "handover")
                        mode = data.get("mode", "auto")
                        res = twin_engine.preview_handover(pair, which, mode)
                        if not ws.closed:
                            await ws.send_str(json.dumps(res))

                    elif cmd == "reload_handover_cache":
                        ho_data = twin_engine.reload_handover_cache()
                        res = {
                            "cmd": "handover_cache_reloaded",
                            "success": True,
                            "handover_pairs": ho_data,
                            "count": len(twin_engine.auto_poses),
                        }
                        if not ws.closed:
                            await ws.send_str(json.dumps(res))

                    elif cmd == "clear_handover_cache":
                        print("[WebTwin] Clearing test2 handover cache for active configuration...")
                        clear_test2_handover_cache(twin_engine.cache_path, twin_engine.fingerprint)
                        twin_engine.auto_poses = {}
                        res = {
                            "cmd": "handover_cache_cleared",
                            "success": True,
                            "handover_pairs": twin_engine.get_handover_data(),
                        }
                        if not ws.closed:
                            await ws.send_str(json.dumps(res))

                    elif cmd == "recompute_handover":
                        pair = data.get("pair")
                        if pair:
                            print(f"[WebTwin] Recomputing Stage 3 Test 2 handover pose for pair {pair} (Top-of-Workspace)...")
                            pose = await loop.run_in_executor(
                                None,
                                lambda: compute_single_test2_handover_pose(
                                    pair,
                                    twin_engine.robots,
                                    twin_engine.handover_cfg,
                                    cache_path=twin_engine.cache_path,
                                ),
                            )
                            all_poses = load_test2_handover_cache(twin_engine.cache_path, twin_engine.fingerprint) or {}
                            if pose is not None:
                                all_poses[pair] = pose
                            twin_engine.auto_poses = all_poses
                            print(f"[WebTwin] Recomputation of {pair} complete. Pose found: {pose is not None}.")
                        else:
                            print("[WebTwin] Recomputing Stage 3 Test 2 handover poses for all pairs (Top-of-Workspace)...")
                            poses = await loop.run_in_executor(
                                None,
                                lambda: compute_all_test2_handover_poses(
                                    twin_engine.robots,
                                    twin_engine.handover_cfg,
                                    cache_path=twin_engine.cache_path,
                                    force_recompute=True,
                                ),
                            )
                            all_poses = load_test2_handover_cache(twin_engine.cache_path, twin_engine.fingerprint) or poses or {}
                            twin_engine.auto_poses = all_poses
                            print(f"[WebTwin] Recomputation complete. {len(twin_engine.auto_poses)} pairs in cache.")

                        res = {
                            "cmd": "handover_recomputed",
                            "success": True,
                            "count": len(twin_engine.auto_poses),
                            "handover_pairs": twin_engine.get_handover_data(),
                        }
                        if not ws.closed:
                            await ws.send_str(json.dumps(res))

                    elif cmd == "manual_jog_joint":
                        robot_id = data.get("robot_id", "robot1")
                        joint_idx = int(data.get("joint_idx", 0))
                        angle_deg = float(data.get("angle_deg", 0.0))
                        res = twin_engine.manual_jog_joint(robot_id, joint_idx, angle_deg)
                        # Also stream to hardware
                        ros_bridge.publish_manual_joints(robot_id, twin_engine.current_q[robot_id])
                        # Attach kinematics inspection
                        res["kinematics_inspection"] = twin_engine.get_kinematics_inspection(robot_id, twin_engine.current_q[robot_id])
                        if not ws.closed:
                            await ws.send_str(json.dumps(res))

                    elif cmd == "toggle_gripper":
                        robot_id = data.get("robot_id", "robot1")
                        state = data.get("state", "open")
                        res = twin_engine.toggle_gripper(robot_id, state)
                        # Publish to hardware
                        ang = ros_bridge.robot_states[robot_id]["gripper_open_deg"] if state == "open" else ros_bridge.robot_states[robot_id]["gripper_close_deg"]
                        ros_bridge.set_gripper(robot_id, ang, state.upper())
                        if not ws.closed:
                            await ws.send_str(json.dumps(res))

                    elif cmd == "solve_single_ik":
                        robot_id = data.get("robot_id", "robot1")
                        target_xyz = data.get("target_xyz", [200.0, 100.0, 150.0])
                        target_yaw = float(data.get("target_yaw", 0.0))
                        res = await loop.run_in_executor(
                            None, twin_engine.solve_single_ik, robot_id, target_xyz, target_yaw
                        )
                        if res.get("success"):
                            ros_bridge.publish_manual_joints(robot_id, twin_engine.current_q[robot_id])
                        if not ws.closed:
                            await ws.send_str(json.dumps(res))

                    elif cmd == "home_all":
                        ros_bridge.home_robot("all")
                        twin_engine.current_q = {
                            "robot1": [float(v) for v in twin_engine.robots["robot1"].home_deg],
                            "robot2": [float(v) for v in twin_engine.robots["robot2"].home_deg],
                            "robot3": [float(v) for v in twin_engine.robots["robot3"].home_deg],
                        }
                        frame_data = twin_engine.compute_frames(twin_engine.current_q)
                        min_d, pt_a, pt_b = compute_laser_endpoints(
                            np.array(twin_engine.current_q["robot1"]),
                            twin_engine.robots["robot1"],
                            np.array(twin_engine.current_q["robot2"]),
                            twin_engine.robots["robot2"],
                        )
                        if not ws.closed:
                            await ws.send_str(json.dumps({
                                "cmd": "home_result",
                                "joint_state": twin_engine.current_q,
                                "frames": frame_data,
                                "clearance": {
                                    "min_distance_mm": round(min_d, 2),
                                    "endpoint_a": pt_a,
                                    "endpoint_b": pt_b,
                                    "pair": "robot1_robot2",
                                },
                            }))

                    elif cmd == "home_arm":
                        robot_id = data.get("robot_id", "robot1")
                        ros_bridge.home_robot(robot_id)
                        if robot_id in twin_engine.robots:
                            twin_engine.current_q[robot_id] = [float(v) for v in twin_engine.robots[robot_id].home_deg]
                            frame_data = twin_engine.compute_frames(twin_engine.current_q)
                            q1 = np.array(twin_engine.current_q["robot1"])
                            q2 = np.array(twin_engine.current_q["robot2"])
                            dist_mm, pt_a, pt_b = compute_laser_endpoints(
                                q1, twin_engine.robots["robot1"], q2, twin_engine.robots["robot2"]
                            )
                            res = {
                                "cmd": "manual_jog_result",
                                "robot_id": robot_id,
                                "joint_state": twin_engine.current_q,
                                "frames": frame_data,
                                "kinematics_inspection": twin_engine.get_kinematics_inspection(robot_id, twin_engine.current_q[robot_id]),
                                "clearance": {
                                    "min_distance_mm": round(dist_mm, 2),
                                    "endpoint_a": pt_a,
                                    "endpoint_b": pt_b,
                                    "pair": "robot1_robot2",
                                },
                            }
                            if not ws.closed:
                                await ws.send_str(json.dumps(res))

                except Exception as e:
                    traceback.print_exc()
                    if not ws.closed:
                        await ws.send_str(json.dumps({"cmd": "error", "message": str(e)}))

            elif msg.type == WSMsgType.ERROR:
                print(f"[WebTwin] WebSocket error: {ws.exception()}")

    finally:
        ACTIVE_WEBSOCKETS.discard(ws)
        print(f"[WebTwin] Client disconnected from {request.remote} (Remaining clients: {len(ACTIVE_WEBSOCKETS)})")

    return ws


@web.middleware
async def no_cache_middleware(request: web.Request, handler):
    resp = await handler(request)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


async def handle_index(request: web.Request) -> web.Response:
    index_file = os.path.join(SCRIPT_DIR, "static", "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        return web.Response(text=f.read(), content_type="text/html")


async def on_app_startup(app: web.Application):
    """Start the ROS 2 background bridge and telemetry broadcaster."""
    ros_bridge.start()
    app["telemetry_broadcaster_task"] = asyncio.create_task(telemetry_broadcaster(app))


async def on_app_cleanup(app: web.Application):
    """Clean up background tasks and ROS 2 node."""
    if "telemetry_broadcaster_task" in app:
        app["telemetry_broadcaster_task"].cancel()
    ros_bridge.stop()


def create_app() -> web.Application:
    app = web.Application(middlewares=[no_cache_middleware])
    app.on_startup.append(on_app_startup)
    app.on_cleanup.append(on_app_cleanup)
    app.router.add_get("/", handle_index)
    app.router.add_get("/ws", handle_websocket)
    static_dir = os.path.join(SCRIPT_DIR, "static")
    app.router.add_static("/static/", static_dir)
    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="rt_AI GPU-Accelerated 3D Web Digital Twin Server")
    parser.add_argument("--port", type=int, default=8080, help="HTTP/WS port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    args = parser.parse_args()

    print("=" * 70)
    print("  🚀 rt_AI GPU-ACCELERATED 3D WEB DIGITAL TWIN")
    print(f"  Server URL: http://localhost:{args.port}")
    print(f"  WebSocket:  ws://localhost:{args.port}/ws")
    print("  Stage:      Stage 3 Test 2 Production Kinematics + ROS 2 Hardware Bridge")
    print("=" * 70)

    app = create_app()
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
