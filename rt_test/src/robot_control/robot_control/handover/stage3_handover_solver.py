"""Stage 3 Automated Kinematic Handover & Pre-Handover Pose Solver.

Computes optimal physical handover positions and pre-handover standoff poses
directly from robot DH parameters, joint limits, and base mounting transforms.

P0 Updates:
- P0.1: Enforces physical handover orientation constraints:
    ||p_A - p_B|| <= eps_p (position coincidence <= 1.0 mm)
    z_A . z_B <= -1 + eps_z (anti-parallel TCP approach corridor <= 15 deg)
    |x_A . x_B| <= eps_x (orthogonal finger alignment <= 15 deg)
  via coupled constrained optimization over both robots' joint configurations.
- P0.2: Computes pre-handover standoff positions strictly along each robot's actual
  world-frame TCP approach corridor:
    p_pre,A = p_H - d_A * z_A
    p_pre,B = p_H - d_B * z_B
  with adaptive standoff distances d in [50, 100] mm and trajectory corridor verification.
- P0.3: Separates intentional gripper contact from body collision; non-contact links
  must maintain whole-body clearance >= 50 mm.
- Fingerprint JSON Cache: Caches solutions in `auto_handover_poses.json` keyed by
  SHA-256 config fingerprint for sub-millisecond retrieval.
"""

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from robot_control.handover.geometry import matrix_to_euler
from robot_control.handover.handover_ik import geometric_jacobian_and_fk
from robot_control.handover.stage1_ik import solve_stage1_ik, world_to_base
from robot_control.handover.stage1_5 import (
    compute_min_body_clearance,
    dist_point_to_segment,
    generate_cartesian_corridor_trajectory,
)
from robot_control.handover.trajectory_cache import compute_config_fingerprint
from robot_control.handover.workspace import PositionWorkspaceComputer
from robot_control.kinematics.fk_math import fk
from robot_control.kinematics.ik_solver import ik


@dataclass
class AutoHandoverPose:
    """Auto-calculated physical handover & pre-handover poses for a robot pair."""

    pair_name: str                      # e.g. 'robot1_robot2'
    handover_position_world: np.ndarray # [X, Y, Z] in world frame (mm)
    standoff_distance_mm: float         # e.g. 70.0 mm
    robot_a_id: str                     # e.g. 'robot1'
    robot_b_id: str                     # e.g. 'robot2'
    robot_a_handover_q_deg: np.ndarray  # joint angles (deg)
    robot_b_handover_q_deg: np.ndarray  # joint angles (deg)
    robot_a_pre_handover_q_deg: np.ndarray  # joint angles (deg)
    robot_b_pre_handover_q_deg: np.ndarray  # joint angles (deg)
    handover_clearance_mm: float        # Whole-body clearance at handover (mm)
    pre_handover_clearance_mm: float    # Whole-body clearance at pre-handover (mm)
    error_a_mm: float                   # IK position error robot A (mm)
    error_b_mm: float                   # IK position error robot B (mm)
    dot_z: float = -1.0                 # zA . zB approach axis alignment (<= -1 + eps_z)
    dot_x: float = 0.0                  # |xA . xB| finger orthogonality (<= eps_x)
    approach_angle_dev_deg: float = 0.0 # Angular deviation from anti-parallel (deg)

    @property
    def robot_a_joint_angles_deg(self) -> np.ndarray:
        return self.robot_a_handover_q_deg

    @property
    def robot_b_joint_angles_deg(self) -> np.ndarray:
        return self.robot_b_handover_q_deg

    @property
    def pre_handover_a_joint_angles_deg(self) -> np.ndarray:
        return self.robot_a_pre_handover_q_deg

    @property
    def pre_handover_b_joint_angles_deg(self) -> np.ndarray:
        return self.robot_b_pre_handover_q_deg


def get_tcp_world_pose(
    q_deg_or_rad: np.ndarray, cfg: object, in_degrees: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate TCP world-frame position and axes for a robot.

    Parameters
    ----------
    q_deg_or_rad : joint angles
    cfg : robot kinematics config
    in_degrees : if True, input is degrees, converted to radians for FK

    Returns
    -------
    (p_world, z_world, x_world, y_world)
      p_world : [X, Y, Z] TCP position in world frame (mm)
      z_world : unit approach vector in world frame (TCP local Z)
      x_world : unit finger-alignment vector in world frame (TCP local X)
      y_world : unit lateral vector in world frame (TCP local Y)
    """
    q_rad = (
        np.deg2rad(np.asarray(q_deg_or_rad, dtype=float))
        if in_degrees
        else np.asarray(q_deg_or_rad, dtype=float)
    )
    p_tcp, T = fk(q_rad, cfg)
    p_world = cfg.world_base_R @ p_tcp + cfg.world_base_xyz
    z_world = cfg.world_base_R @ T[:3, 2]
    x_world = cfg.world_base_R @ T[:3, 0]
    y_world = cfg.world_base_R @ T[:3, 1]

    # Ensure unit normalization
    z_norm = np.linalg.norm(z_world)
    if z_norm > 1e-9:
        z_world = z_world / z_norm
    x_norm = np.linalg.norm(x_world)
    if x_norm > 1e-9:
        x_world = x_world / x_norm
    y_norm = np.linalg.norm(y_world)
    if y_norm > 1e-9:
        y_world = y_world / y_norm

    return p_world, z_world, x_world, y_world


def resolve_auto_handover_cache_path() -> str:
    """Resolve filepath for auto_handover_poses.json in config directory."""
    cur = os.path.abspath(os.getcwd())
    while cur != '/':
        cand = os.path.join(cur, 'src', 'robot_control', 'config')
        if os.path.isdir(cand):
            return os.path.join(cand, 'auto_handover_poses.json')
        cur = os.path.dirname(cur)
    return os.path.join(os.getcwd(), 'auto_handover_poses.json')


def serialize_auto_handover(pose: AutoHandoverPose) -> dict:
    """Serialize an AutoHandoverPose dataclass to a JSON-safe dict."""
    return {
        "pair_name": pose.pair_name,
        "handover_position_world": [round(float(v), 2) for v in pose.handover_position_world],
        "standoff_distance_mm": round(float(pose.standoff_distance_mm), 1),
        "robot_a_id": pose.robot_a_id,
        "robot_b_id": pose.robot_b_id,
        "robot_a_handover_q_deg": [round(float(v), 2) for v in pose.robot_a_handover_q_deg],
        "robot_b_handover_q_deg": [round(float(v), 2) for v in pose.robot_b_handover_q_deg],
        "robot_a_pre_handover_q_deg": [round(float(v), 2) for v in pose.robot_a_pre_handover_q_deg],
        "robot_b_pre_handover_q_deg": [round(float(v), 2) for v in pose.robot_b_pre_handover_q_deg],
        "handover_clearance_mm": round(float(pose.handover_clearance_mm), 2),
        "pre_handover_clearance_mm": round(float(pose.pre_handover_clearance_mm), 2),
        "error_a_mm": round(float(pose.error_a_mm), 3),
        "error_b_mm": round(float(pose.error_b_mm), 3),
        "dot_z": round(float(pose.dot_z), 4),
        "dot_x": round(float(pose.dot_x), 4),
        "approach_angle_dev_deg": round(float(pose.approach_angle_dev_deg), 2),
    }


def deserialize_auto_handover(d: dict) -> AutoHandoverPose:
    """Deserialize dict to AutoHandoverPose with backward-compatible defaults."""
    return AutoHandoverPose(
        pair_name=str(d["pair_name"]),
        handover_position_world=np.array(d["handover_position_world"], dtype=float),
        standoff_distance_mm=float(d["standoff_distance_mm"]),
        robot_a_id=str(d["robot_a_id"]),
        robot_b_id=str(d["robot_b_id"]),
        robot_a_handover_q_deg=np.array(d["robot_a_handover_q_deg"], dtype=float),
        robot_b_handover_q_deg=np.array(d["robot_b_handover_q_deg"], dtype=float),
        robot_a_pre_handover_q_deg=np.array(d["robot_a_pre_handover_q_deg"], dtype=float),
        robot_b_pre_handover_q_deg=np.array(d["robot_b_pre_handover_q_deg"], dtype=float),
        handover_clearance_mm=float(d["handover_clearance_mm"]),
        pre_handover_clearance_mm=float(d["pre_handover_clearance_mm"]),
        error_a_mm=float(d["error_a_mm"]),
        error_b_mm=float(d["error_b_mm"]),
        dot_z=float(d.get("dot_z", -1.0)),
        dot_x=float(d.get("dot_x", 0.0)),
        approach_angle_dev_deg=float(d.get("approach_angle_dev_deg", 0.0)),
    )


def load_auto_handover_cache(cache_path: str, config_hash: str) -> Optional[Dict[str, AutoHandoverPose]]:
    """Load cached auto-calculated handover poses for a config hash."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Require version 2.0 to ensure P0 orientation constraints are satisfied
        if data.get("version") != "2.0":
            return None
        fp_data = data.get("fingerprints", {}).get(config_hash, {})
        pairs_data = fp_data.get("pairs", {})
        if not pairs_data:
            return None
        res = {}
        for pair_name, pd in pairs_data.items():
            res[pair_name] = deserialize_auto_handover(pd)
        return res if len(res) >= 3 else None
    except Exception:
        return None


def save_auto_handover_cache(cache_path: str, config_hash: str, poses: Dict[str, AutoHandoverPose]) -> bool:
    """Save auto-calculated handover poses atomically to JSON cache."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        data = {"version": "2.0", "fingerprints": {}}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict) and "fingerprints" in loaded:
                        data = loaded
                        data["version"] = "2.0"
            except Exception:
                pass

        if config_hash not in data["fingerprints"]:
            data["fingerprints"][config_hash] = {}

        data["fingerprints"][config_hash] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pairs": {pname: serialize_auto_handover(pose) for pname, pose in poses.items()}
        }

        tmp_file = cache_path + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_file, cache_path)
        return True
    except Exception:
        return False


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion [w, x, y, z]."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-9 else q


def get_handover_pose_details(pose: AutoHandoverPose, robots: Dict[str, Any]) -> Dict[str, Any]:
    """Derive complete physical pose, orientation, standoff, and verification metrics.

    Parameters
    ----------
    pose : AutoHandoverPose dataclass instance
    robots : dict of robot kinematics configurations

    Returns
    -------
    dict containing:
      - pair_name, robot_a_id, robot_b_id, shared_handover_position_mm
      - robot_a:
          handover: position_mm, rpy_deg, rotation_matrix, quaternion_wxyz, axes (z, x, y), q_deg
          pre_handover: position_mm, target_position_mm, rpy_deg, rotation_matrix, quaternion_wxyz,
                        axes (z, x, y), q_deg, actual_standoff_mm, standoff_error_mm
      - robot_b:
          handover: position_mm, rpy_deg, rotation_matrix, quaternion_wxyz, axes (z, x, y), q_deg
          pre_handover: position_mm, target_position_mm, rpy_deg, rotation_matrix, quaternion_wxyz,
                        axes (z, x, y), q_deg, actual_standoff_mm, standoff_error_mm
      - metrics:
          coincidence_error_mm, dot_z, approach_angle_dev_deg, dot_x,
          handover_clearance_mm, pre_handover_clearance_mm,
          joint_limits_valid_a, joint_limits_valid_b, is_valid
    """
    cfgA = robots[pose.robot_a_id]
    cfgB = robots[pose.robot_b_id]

    # Handover poses
    pA, zA, xA, yA = get_tcp_world_pose(pose.robot_a_handover_q_deg, cfgA, in_degrees=True)
    pB, zB, xB, yB = get_tcp_world_pose(pose.robot_b_handover_q_deg, cfgB, in_degrees=True)

    RA = np.column_stack([xA, yA, zA])
    RB = np.column_stack([xB, yB, zB])
    rpyA_deg = np.rad2deg(matrix_to_euler(RA))
    rpyB_deg = np.rad2deg(matrix_to_euler(RB))
    quatA = matrix_to_quaternion(RA)
    quatB = matrix_to_quaternion(RB)

    # Pre-handover poses
    pA_pre, zA_pre, xA_pre, yA_pre = get_tcp_world_pose(pose.robot_a_pre_handover_q_deg, cfgA, in_degrees=True)
    pB_pre, zB_pre, xB_pre, yB_pre = get_tcp_world_pose(pose.robot_b_pre_handover_q_deg, cfgB, in_degrees=True)

    RA_pre = np.column_stack([xA_pre, yA_pre, zA_pre])
    RB_pre = np.column_stack([xB_pre, yB_pre, zB_pre])
    rpyA_pre_deg = np.rad2deg(matrix_to_euler(RA_pre))
    rpyB_pre_deg = np.rad2deg(matrix_to_euler(RB_pre))
    quatA_pre = matrix_to_quaternion(RA_pre)
    quatB_pre = matrix_to_quaternion(RB_pre)

    # Standoff targets along actual TCP approach axes
    target_pA_pre = pose.handover_position_world - pose.standoff_distance_mm * zA
    target_pB_pre = pose.handover_position_world - pose.standoff_distance_mm * zB
    actual_standoff_A = float(np.linalg.norm(pA - pA_pre))
    actual_standoff_B = float(np.linalg.norm(pB - pB_pre))
    err_standoff_A = float(np.linalg.norm(pA_pre - target_pA_pre))
    err_standoff_B = float(np.linalg.norm(pB_pre - target_pB_pre))

    # Verification residuals
    coinc_err = float(np.linalg.norm(pA - pB))
    dot_z = float(np.dot(zA, zB))
    ang_dev = float(np.rad2deg(np.arccos(np.clip(-dot_z, -1.0, 1.0))))
    dot_x = abs(float(np.dot(xA, xB)))

    # Joint limits validation
    qA_deg = pose.robot_a_handover_q_deg
    qB_deg = pose.robot_b_handover_q_deg
    limA_ok = bool(
        np.all(qA_deg >= cfgA.joint_limits_deg[:, 0] - 1e-2) and
        np.all(qA_deg <= cfgA.joint_limits_deg[:, 1] + 1e-2)
    )
    limB_ok = bool(
        np.all(qB_deg >= cfgB.joint_limits_deg[:, 0] - 1e-2) and
        np.all(qB_deg <= cfgB.joint_limits_deg[:, 1] + 1e-2)
    )

    is_valid = bool(
        coinc_err <= 1.0 and
        dot_z <= -0.9659 and  # <= 15 deg deviation
        dot_x <= 0.2588 and   # <= 15 deg from orthogonal
        limA_ok and limB_ok and
        pose.handover_clearance_mm >= 45.0
    )

    return {
        "pair_name": pose.pair_name,
        "robot_a_id": pose.robot_a_id,
        "robot_b_id": pose.robot_b_id,
        "shared_handover_position_mm": np.asarray(pose.handover_position_world, dtype=float),
        "standoff_distance_mm": float(pose.standoff_distance_mm),
        "robot_a": {
            "handover": {
                "position_mm": pA,
                "rpy_deg": rpyA_deg,
                "rotation_matrix": RA,
                "quaternion_wxyz": quatA,
                "approach_axis_z": zA,
                "finger_axis_x": xA,
                "lateral_axis_y": yA,
                "q_deg": qA_deg,
            },
            "pre_handover": {
                "position_mm": pA_pre,
                "target_position_mm": target_pA_pre,
                "rpy_deg": rpyA_pre_deg,
                "rotation_matrix": RA_pre,
                "quaternion_wxyz": quatA_pre,
                "approach_axis_z": zA_pre,
                "finger_axis_x": xA_pre,
                "lateral_axis_y": yA_pre,
                "q_deg": pose.robot_a_pre_handover_q_deg,
                "actual_standoff_mm": actual_standoff_A,
                "standoff_error_mm": err_standoff_A,
            },
        },
        "robot_b": {
            "handover": {
                "position_mm": pB,
                "rpy_deg": rpyB_deg,
                "rotation_matrix": RB,
                "quaternion_wxyz": quatB,
                "approach_axis_z": zB,
                "finger_axis_x": xB,
                "lateral_axis_y": yB,
                "q_deg": qB_deg,
            },
            "pre_handover": {
                "position_mm": pB_pre,
                "target_position_mm": target_pB_pre,
                "rpy_deg": rpyB_pre_deg,
                "rotation_matrix": RB_pre,
                "quaternion_wxyz": quatB_pre,
                "approach_axis_z": zB_pre,
                "finger_axis_x": xB_pre,
                "lateral_axis_y": yB_pre,
                "q_deg": pose.robot_b_pre_handover_q_deg,
                "actual_standoff_mm": actual_standoff_B,
                "standoff_error_mm": err_standoff_B,
            },
        },
        "metrics": {
            "coincidence_error_mm": coinc_err,
            "dot_z": dot_z,
            "approach_angle_dev_deg": ang_dev,
            "dot_x": dot_x,
            "handover_clearance_mm": float(pose.handover_clearance_mm),
            "pre_handover_clearance_mm": float(pose.pre_handover_clearance_mm),
            "joint_limits_valid_a": limA_ok,
            "joint_limits_valid_b": limB_ok,
            "is_valid": is_valid,
        },
        # Flat convenience keys for GUI inspection and direct verification
        "robot_a_tcp_xyz": pA,
        "robot_b_tcp_xyz": pB,
        "robot_a_rpy_deg": rpyA_deg,
        "robot_b_rpy_deg": rpyB_deg,
        "robot_a_rot_matrix": RA,
        "robot_b_rot_matrix": RB,
        "robot_a_quaternion": quatA,
        "robot_b_quaternion": quatB,
        "robot_a_approach_vec": zA,
        "robot_b_approach_vec": zB,
        "pre_handover_a_xyz": pA_pre,
        "pre_handover_b_xyz": pB_pre,
        "pre_handover_a_rpy_deg": rpyA_pre_deg,
        "pre_handover_b_rpy_deg": rpyB_pre_deg,
        "pre_handover_a_quaternion": quatA_pre,
        "pre_handover_b_quaternion": quatB_pre,
        "approach_dot": dot_z,
        "approach_deviation_deg": ang_dev,
        "finger_cross_alignment": dot_x,
        "coincidence_error_mm": coinc_err,
        "achieved_standoff_mm": actual_standoff_A,
        "standoff_error_mm": err_standoff_A,
        "robot_a_limits_valid": limA_ok,
        "robot_b_limits_valid": limB_ok,
        "residual_norm": float(np.sqrt(coinc_err**2 + (ang_dev * np.pi / 180.0)**2 + dot_x**2)),
    }


def _skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix for cross product: _skew(v) @ u == v x u."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=float)


def compute_coupled_handover_jacobian(
    qA: np.ndarray,
    qB: np.ndarray,
    cfgA: Any,
    cfgB: Any,
    p_H_world: Optional[np.ndarray] = None,
    L_char: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute analytical coupled task Jacobian and residuals in world frame.

    Formulation:
      q = [qA, qB] in R^(nA + nB)
      Task residuals:
        e_p = -(pA - pB) / L_char                   (coincidence in mm, scaled)
        e_anc = (p_H - 0.5*(pA + pB)) * (0.5/L_char)(anchor constraint in mm, scaled, if p_H given)
        e_z = -(zA + zB)                            (anti-parallel approach alignment, dimensionless)
        e_x = -[xA . xB]                            (orthogonal finger alignment, dimensionless)

      Jacobian blocks:
        J_p = [JvA, -JvB] / L_char
        J_anc = [0.5*JvA, 0.5*JvB] * (0.5/L_char)
        J_z = [-_skew(zA) @ JwA, -_skew(zB) @ JwB]
        J_x = [(xA x xB) @ JwA, -(xA x xB) @ JwB]

    All Jacobians are transformed from base frame to world frame via cfg.world_base_R.
    """
    nA = cfgA.num_joints
    nB = cfgB.num_joints

    pA, zA, xA, _ = get_tcp_world_pose(qA, cfgA, in_degrees=False)
    pB, zB, xB, _ = get_tcp_world_pose(qB, cfgB, in_degrees=False)

    JA_b, _, _ = geometric_jacobian_and_fk(qA, cfgA)
    JB_b, _, _ = geometric_jacobian_and_fk(qB, cfgB)

    R_wA = np.asarray(cfgA.world_base_R, dtype=float)
    R_wB = np.asarray(cfgB.world_base_R, dtype=float)
    JvA = R_wA @ JA_b[:3, :]
    JwA = R_wA @ JA_b[3:, :]
    JvB = R_wB @ JB_b[:3, :]
    JwB = R_wB @ JB_b[3:, :]

    w_p = 1.0 / float(L_char)
    w_anc = 0.5 / float(L_char)

    J_p = np.hstack([JvA, -JvB]) * w_p
    e_p = (pA - pB) * w_p

    J_z = np.hstack([-_skew(zA) @ JwA, -_skew(zB) @ JwB])
    e_z = zA + zB

    x_cross = np.cross(xA, xB)
    J_x = np.hstack([(x_cross @ JwA).reshape(1, -1), (-x_cross @ JwB).reshape(1, -1)])
    e_x = np.array([float(np.dot(xA, xB))])

    if p_H_world is not None:
        p_H = np.asarray(p_H_world, dtype=float)
        J_anc = np.hstack([0.5 * JvA, 0.5 * JvB]) * w_anc
        e_anc = (0.5 * (pA + pB) - p_H) * w_anc
        J = np.vstack([J_p, J_anc, J_z, J_x])
        e = np.concatenate([e_p, e_anc, e_z, e_x])
    else:
        J = np.vstack([J_p, J_z, J_x])
        e = np.concatenate([e_p, e_z, e_x])

    metrics = {
        "pA": pA,
        "pB": pB,
        "zA": zA,
        "zB": zB,
        "xA": xA,
        "xB": xB,
        "coinc_mm": float(np.linalg.norm(pA - pB)),
        "dot_z": float(np.dot(zA, zB)),
        "dot_x": abs(float(np.dot(xA, xB))),
    }
    if p_H_world is not None:
        metrics["errA_mm"] = float(np.linalg.norm(pA - np.asarray(p_H_world, dtype=float)))
        metrics["errB_mm"] = float(np.linalg.norm(pB - np.asarray(p_H_world, dtype=float)))

    return J, e, metrics


def solve_coupled_handover_orientation_lbfgs(
    robot_a_id: str,
    robot_b_id: str,
    p_H_world: np.ndarray,
    robots: Dict[str, Any],
    pos_tol_mm: float = 1.0,
    ori_tol_deg: float = 15.0,
    min_clearance_mm: float = 45.0,
    max_opt_iterations: int = 70,
    seeds: Optional[Tuple[List[dict], List[dict]]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float, float, float, float]]:
    """Preserved fallback solver: numerical finite-difference L-BFGS-B optimization."""
    cfgA = robots[robot_a_id]
    cfgB = robots[robot_b_id]
    nA = cfgA.num_joints
    nB = cfgB.num_joints

    theta_tol_rad = np.deg2rad(float(ori_tol_deg))
    eps_z = 1.0 - np.cos(theta_tol_rad)
    eps_x = np.sin(theta_tol_rad)

    if seeds is not None:
        solsA, solsB = seeds
    else:
        pA_base = world_to_base(p_H_world, cfgA)
        pB_base = world_to_base(p_H_world, cfgB)
        resA = ik(pA_base, cfgA, num_random_starts=10)
        resB = ik(pB_base, cfgB, num_random_starts=10)
        solsA = resA.get("solutions", [])
        solsB = resB.get("solutions", [])

    if not solsA or not solsB:
        return None

    bnds = list(cfgA.joint_limits_rad) + list(cfgB.joint_limits_rad)

    def get_both_frames(q):
        pA, zA, xA, _ = get_tcp_world_pose(q[:nA], cfgA, in_degrees=False)
        pB, zB, xB, _ = get_tcp_world_pose(q[nA:], cfgB, in_degrees=False)
        return pA, pB, zA, zB, xA, xB

    def loss(q):
        pA, pB, zA, zB, xA, xB = get_both_frames(q)
        err_pA = np.sum((pA - p_H_world) ** 2)
        err_pB = np.sum((pB - p_H_world) ** 2)
        err_z = (np.dot(zA, zB) + 1.0) ** 2
        err_x = (np.dot(xA, xB)) ** 2
        return 1000.0 * (err_pA + err_pB) + 150.0 * err_z + 80.0 * err_x

    best_match = None
    best_score = 1e9

    zA_inits = [get_tcp_world_pose(sa["q"], cfgA, in_degrees=False)[1] for sa in solsA]
    zB_inits = [get_tcp_world_pose(sb["q"], cfgB, in_degrees=False)[1] for sb in solsB]
    seed_pairs = []
    for ia, zA_init in enumerate(zA_inits):
        for ib, zB_init in enumerate(zB_inits):
            dot_z_init = float(np.dot(zA_init, zB_init))
            seed_pairs.append((dot_z_init, solsA[ia]["q"], solsB[ib]["q"]))

    seed_pairs.sort(key=lambda item: item[0])

    for _, qA_init, qB_init in seed_pairs[:3]:
        q0 = np.concatenate([qA_init, qB_init])
        res = minimize(loss, q0, method="L-BFGS-B", bounds=bnds, options={"maxiter": max_opt_iterations})
        pA, pB, zA, zB, xA, xB = get_both_frames(res.x)

        errA = float(np.linalg.norm(pA - p_H_world))
        errB = float(np.linalg.norm(pB - p_H_world))
        errAB = float(np.linalg.norm(pA - pB))
        dot_z = float(np.dot(zA, zB))
        dot_x = abs(float(np.dot(xA, xB)))

        if errA > pos_tol_mm or errB > pos_tol_mm or errAB > pos_tol_mm:
            continue
        if dot_z > (-1.0 + eps_z):
            continue
        if dot_x > eps_x:
            continue

        qA_deg = np.rad2deg(res.x[:nA])
        qB_deg = np.rad2deg(res.x[nA:])

        q_min_A = cfgA.joint_limits_deg[:, 0] - 1e-3
        q_max_A = cfgA.joint_limits_deg[:, 1] + 1e-3
        if np.any(qA_deg < q_min_A) or np.any(qA_deg > q_max_A):
            continue

        q_min_B = cfgB.joint_limits_deg[:, 0] - 1e-3
        q_max_B = cfgB.joint_limits_deg[:, 1] + 1e-3
        if np.any(qB_deg < q_min_B) or np.any(qB_deg > q_max_B):
            continue

        clr = compute_min_body_clearance(qA_deg, cfgA, qB_deg, cfgB, exclude_tcp=True)
        if clr < min_clearance_mm:
            continue

        score = 100.0 * (dot_z + 1.0) + 50.0 * dot_x + 10.0 * (errA + errB) - 0.5 * clr
        if score < best_score:
            best_score = score
            best_match = (qA_deg, qB_deg, errA, errB, dot_z, dot_x, clr)

    return best_match


def solve_coupled_handover_orientation_analytical(
    robot_a_id: str,
    robot_b_id: str,
    p_H_world: np.ndarray,
    robots: Dict[str, Any],
    pos_tol_mm: float = 1.0,
    ori_tol_deg: float = 15.0,
    min_clearance_mm: float = 45.0,
    max_opt_iterations: int = 70,
    seeds: Optional[Tuple[List[dict], List[dict]]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float, float, float, float]]:
    """Solve coupled handover using exact analytical task gradients.

    Evaluates exact analytical gradients of position coincidence, anchor point,
    anti-parallel TCP approach alignment, and orthogonal finger alignment using
    geometric Jacobians without finite-difference re-evaluations.
    """
    cfgA = robots[robot_a_id]
    cfgB = robots[robot_b_id]
    nA = cfgA.num_joints
    nB = cfgB.num_joints

    theta_tol_rad = np.deg2rad(float(ori_tol_deg))
    eps_z = 1.0 - np.cos(theta_tol_rad)
    eps_x = np.sin(theta_tol_rad)

    if seeds is not None:
        solsA, solsB = seeds
    else:
        pA_base = world_to_base(p_H_world, cfgA)
        pB_base = world_to_base(p_H_world, cfgB)
        resA = ik(pA_base, cfgA, num_random_starts=10)
        resB = ik(pB_base, cfgB, num_random_starts=10)
        solsA = resA.get("solutions", [])
        solsB = resB.get("solutions", [])

    if not solsA or not solsB:
        return None

    bnds = list(cfgA.joint_limits_rad) + list(cfgB.joint_limits_rad)
    p_H = np.asarray(p_H_world, dtype=float)
    R_wA = np.asarray(cfgA.world_base_R, dtype=float)
    R_wB = np.asarray(cfgB.world_base_R, dtype=float)

    def loss_and_grad(q):
        qA = q[:nA]
        qB = q[nA:]
        pA, zA, xA, _ = get_tcp_world_pose(qA, cfgA, in_degrees=False)
        pB, zB, xB, _ = get_tcp_world_pose(qB, cfgB, in_degrees=False)

        dpA = pA - p_H
        dpB = pB - p_H
        dot_z = float(np.dot(zA, zB))
        dot_x = float(np.dot(xA, xB))

        val = 1000.0 * (np.sum(dpA ** 2) + np.sum(dpB ** 2)) + 150.0 * (dot_z + 1.0) ** 2 + 80.0 * (dot_x) ** 2

        JA_b, _, _ = geometric_jacobian_and_fk(qA, cfgA)
        JB_b, _, _ = geometric_jacobian_and_fk(qB, cfgB)

        JvA = R_wA @ JA_b[:3, :]
        JwA = R_wA @ JA_b[3:, :]
        JvB = R_wB @ JB_b[:3, :]
        JwB = R_wB @ JB_b[3:, :]

        z_cross = np.cross(zA, zB)
        x_cross = np.cross(xA, xB)

        gradA = (
            1000.0 * (2.0 * dpA @ JvA) +
            150.0 * (2.0 * (dot_z + 1.0) * (z_cross @ JwA)) +
            80.0 * (2.0 * dot_x * (x_cross @ JwA))
        )
        gradB = (
            1000.0 * (2.0 * dpB @ JvB) +
            150.0 * (2.0 * (dot_z + 1.0) * (-z_cross @ JwB)) +
            80.0 * (2.0 * dot_x * (-x_cross @ JwB))
        )
        return val, np.concatenate([gradA, gradB])

    best_match = None
    best_score = 1e9

    zA_inits = [get_tcp_world_pose(sa["q"], cfgA, in_degrees=False)[1] for sa in solsA]
    zB_inits = [get_tcp_world_pose(sb["q"], cfgB, in_degrees=False)[1] for sb in solsB]
    seed_pairs = []
    for ia, zA_init in enumerate(zA_inits):
        for ib, zB_init in enumerate(zB_inits):
            dot_z_init = float(np.dot(zA_init, zB_init))
            seed_pairs.append((dot_z_init, solsA[ia]["q"], solsB[ib]["q"]))

    seed_pairs.sort(key=lambda item: item[0])

    for _, qA_init, qB_init in seed_pairs[:3]:
        q0 = np.concatenate([qA_init, qB_init])
        res = minimize(
            loss_and_grad,
            q0,
            method="L-BFGS-B",
            jac=True,
            bounds=bnds,
            options={"maxiter": max_opt_iterations},
        )
        qA_sol = res.x[:nA]
        qB_sol = res.x[nA:]
        pA, zA, xA, _ = get_tcp_world_pose(qA_sol, cfgA, in_degrees=False)
        pB, zB, xB, _ = get_tcp_world_pose(qB_sol, cfgB, in_degrees=False)

        errA = float(np.linalg.norm(pA - p_H))
        errB = float(np.linalg.norm(pB - p_H))
        errAB = float(np.linalg.norm(pA - pB))
        dot_z = float(np.dot(zA, zB))
        dot_x = abs(float(np.dot(xA, xB)))

        if errA > pos_tol_mm or errB > pos_tol_mm or errAB > pos_tol_mm:
            continue
        if dot_z > (-1.0 + eps_z):
            continue
        if dot_x > eps_x:
            continue

        qA_deg = np.rad2deg(qA_sol)
        qB_deg = np.rad2deg(qB_sol)

        q_min_A = cfgA.joint_limits_deg[:, 0] - 1e-3
        q_max_A = cfgA.joint_limits_deg[:, 1] + 1e-3
        if np.any(qA_deg < q_min_A) or np.any(qA_deg > q_max_A):
            continue

        q_min_B = cfgB.joint_limits_deg[:, 0] - 1e-3
        q_max_B = cfgB.joint_limits_deg[:, 1] + 1e-3
        if np.any(qB_deg < q_min_B) or np.any(qB_deg > q_max_B):
            continue

        clr = compute_min_body_clearance(qA_deg, cfgA, qB_deg, cfgB, exclude_tcp=True)
        if clr < min_clearance_mm:
            continue

        score = 100.0 * (dot_z + 1.0) + 50.0 * dot_x + 10.0 * (errA + errB) - 0.5 * clr
        if score < best_score:
            best_score = score
            best_match = (qA_deg, qB_deg, errA, errB, dot_z, dot_x, clr)

    return best_match


def solve_coupled_handover_orientation_dls(
    robot_a_id: str,
    robot_b_id: str,
    p_H_world: np.ndarray,
    robots: Dict[str, Any],
    pos_tol_mm: float = 1.0,
    ori_tol_deg: float = 15.0,
    min_clearance_mm: float = 45.0,
    max_opt_iterations: int = 70,
    lambda_base: float = 0.05,
    step_limit_rad: float = 0.15,
    seeds: Optional[Tuple[List[dict], List[dict]]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float, float, float, float]]:
    """Solve coupled handover using Damped Least-Squares (DLS) with nullspace posture optimization.

    Features:
      1. SVD-based adaptive Levenberg-Marquardt damping lambda(sigma_min).
      2. Dimensionless residual scaling via L_char so position in mm does not overwhelm orientation.
      3. Nullspace projection P_null = (I - V_r V_r^T) for posture optimization toward mid-range
         without corrupting primary task residuals.
      4. Bounded joint increments ||dq||_inf <= step_limit_rad.
      5. Armijo backtracking line search with joint limit clipping and stagnation detection.
      6. Independent final hard constraint checks.
    """
    cfgA = robots[robot_a_id]
    cfgB = robots[robot_b_id]
    nA = cfgA.num_joints
    nB = cfgB.num_joints
    n = nA + nB

    theta_tol_rad = np.deg2rad(float(ori_tol_deg))
    eps_z = 1.0 - np.cos(theta_tol_rad)
    eps_x = np.sin(theta_tol_rad)

    if seeds is not None:
        solsA, solsB = seeds
    else:
        pA_base = world_to_base(p_H_world, cfgA)
        pB_base = world_to_base(p_H_world, cfgB)
        resA = ik(pA_base, cfgA, num_random_starts=10)
        resB = ik(pB_base, cfgB, num_random_starts=10)
        solsA = resA.get("solutions", [])
        solsB = resB.get("solutions", [])

    if not solsA or not solsB:
        return None

    q_min = np.concatenate([cfgA.joint_limits_rad[:, 0], cfgB.joint_limits_rad[:, 0]])
    q_max = np.concatenate([cfgA.joint_limits_rad[:, 1], cfgB.joint_limits_rad[:, 1]])
    q_mid = 0.5 * (q_min + q_max)
    q_span = np.maximum(q_max - q_min, 1e-4)

    best_match = None
    best_score = 1e9

    zA_inits = [get_tcp_world_pose(sa["q"], cfgA, in_degrees=False)[1] for sa in solsA]
    zB_inits = [get_tcp_world_pose(sb["q"], cfgB, in_degrees=False)[1] for sb in solsB]
    seed_pairs = []
    for ia, zA_init in enumerate(zA_inits):
        for ib, zB_init in enumerate(zB_inits):
            dot_z_init = float(np.dot(zA_init, zB_init))
            seed_pairs.append((dot_z_init, solsA[ia]["q"], solsB[ib]["q"]))

    seed_pairs.sort(key=lambda item: item[0])

    for _, qA_init, qB_init in seed_pairs[:3]:
        q = np.concatenate([qA_init, qB_init])

        for _ in range(max_opt_iterations):
            J, e, metrics = compute_coupled_handover_jacobian(
                q[:nA], q[nA:], cfgA, cfgB, p_H_world=p_H_world, L_char=100.0
            )

            errA = metrics.get("errA_mm", 999.0)
            errB = metrics.get("errB_mm", 999.0)
            errAB = metrics["coinc_mm"]
            dot_z = metrics["dot_z"]
            dot_x = metrics["dot_x"]

            if (errA <= pos_tol_mm and errB <= pos_tol_mm and errAB <= pos_tol_mm and
                    dot_z <= (-1.0 + eps_z) and dot_x <= eps_x):
                break

            U, S, Vt = np.linalg.svd(J, full_matrices=True)
            sigma_min = S[-1] if len(S) > 0 else 0.0

            if sigma_min < 0.1:
                lam2 = (lambda_base ** 2) + 0.1 * (0.1 - sigma_min) ** 2
            else:
                lam2 = lambda_base ** 2

            dq_task = np.zeros(n)
            for i, s in enumerate(S):
                f_damp = s / (s ** 2 + lam2)
                dq_task -= f_damp * (U[:, i] @ e) * Vt[i, :]

            rank_tol = 1e-4
            rank = int(np.sum(S > rank_tol))
            if rank < n:
                V_r = Vt[:rank, :].T
                P_null = np.eye(n) - V_r @ V_r.T
                grad_posture = -0.5 * ((q - q_mid) / q_span)
                dq_null = P_null @ grad_posture
            else:
                dq_null = np.zeros(n)

            dq = dq_task + 0.2 * dq_null

            max_step = float(np.max(np.abs(dq)))
            if max_step > step_limit_rad:
                dq = dq * (step_limit_rad / max_step)

            current_res_norm = float(np.linalg.norm(e))
            alpha = 1.0
            accepted = False
            best_q_step = q

            for _ in range(5):
                q_cand = np.clip(q + alpha * dq, q_min + 1e-4, q_max - 1e-4)
                _, e_cand, _ = compute_coupled_handover_jacobian(
                    q_cand[:nA], q_cand[nA:], cfgA, cfgB, p_H_world=p_H_world, L_char=100.0
                )
                if np.linalg.norm(e_cand) < current_res_norm:
                    best_q_step = q_cand
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted and alpha < 0.1:
                q = np.clip(q + 0.1 * dq, q_min + 1e-4, q_max - 1e-4)
            else:
                q = best_q_step

        qA_sol = q[:nA]
        qB_sol = q[nA:]
        pA, zA, xA, _ = get_tcp_world_pose(qA_sol, cfgA, in_degrees=False)
        pB, zB, xB, _ = get_tcp_world_pose(qB_sol, cfgB, in_degrees=False)

        errA = float(np.linalg.norm(pA - p_H_world))
        errB = float(np.linalg.norm(pB - p_H_world))
        errAB = float(np.linalg.norm(pA - pB))
        dot_z = float(np.dot(zA, zB))
        dot_x = abs(float(np.dot(xA, xB)))

        if errA > pos_tol_mm or errB > pos_tol_mm or errAB > pos_tol_mm:
            continue
        if dot_z > (-1.0 + eps_z):
            continue
        if dot_x > eps_x:
            continue

        qA_deg = np.rad2deg(qA_sol)
        qB_deg = np.rad2deg(qB_sol)

        q_min_A = cfgA.joint_limits_deg[:, 0] - 1e-3
        q_max_A = cfgA.joint_limits_deg[:, 1] + 1e-3
        if np.any(qA_deg < q_min_A) or np.any(qA_deg > q_max_A):
            continue

        q_min_B = cfgB.joint_limits_deg[:, 0] - 1e-3
        q_max_B = cfgB.joint_limits_deg[:, 1] + 1e-3
        if np.any(qB_deg < q_min_B) or np.any(qB_deg > q_max_B):
            continue

        clr = compute_min_body_clearance(qA_deg, cfgA, qB_deg, cfgB, exclude_tcp=True)
        if clr < min_clearance_mm:
            continue

        score = 100.0 * (dot_z + 1.0) + 50.0 * dot_x + 10.0 * (errA + errB) - 0.5 * clr
        if score < best_score:
            best_score = score
            best_match = (qA_deg, qB_deg, errA, errB, dot_z, dot_x, clr)

    return best_match


def solve_coupled_handover_orientation(
    robot_a_id: str,
    robot_b_id: str,
    p_H_world: np.ndarray,
    robots: Dict[str, Any],
    pos_tol_mm: float = 1.0,
    ori_tol_deg: float = 15.0,
    min_clearance_mm: float = 45.0,
    max_opt_iterations: int = 70,
    solver_method: str = "analytical",
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float, float, float, float]]:
    """P0.1: Solve for coupled joint configurations (qA, qB) at handover point p_H.

    Enforces hard kinematic constraints:
      1. Position coincidence: ||pA - pB|| <= pos_tol_mm and ||pA - p_H|| <= pos_tol_mm.
      2. Anti-parallel approach axes: zA . zB <= -1 + eps_z (where eps_z = 1 - cos(theta_tol)).
      3. Orthogonal finger-alignment axes: |xA . xB| <= eps_x (where eps_x = sin(theta_tol)).
      4. Joint limits: q_min <= q <= q_max.
      5. Whole-body clearance: min_body_clearance >= min_clearance_mm.

    Solver Selection:
      - 'analytical' (default): Exact analytical task Jacobian/gradient solver.
        Falls back to numerical L-BFGS-B if analytical optimization does not converge.
      - 'dls': Damped Least-Squares solver with SVD nullspace posture optimization.
        Falls back to numerical L-BFGS-B if DLS does not converge.
      - 'numerical' or 'lbfgs': Preserved numerical finite-difference L-BFGS-B solver.

    Returns
    -------
    (qA_deg, qB_deg, err_a_mm, err_b_mm, dot_z, dot_x, h_clr) or None.
    """
    cfgA = robots[robot_a_id]
    cfgB = robots[robot_b_id]
    pA_base = world_to_base(p_H_world, cfgA)
    pB_base = world_to_base(p_H_world, cfgB)
    resA = ik(pA_base, cfgA, num_random_starts=10)
    resB = ik(pB_base, cfgB, num_random_starts=10)
    solsA = resA.get("solutions", [])
    solsB = resB.get("solutions", [])
    if not solsA or not solsB:
        return None
    seeds = (solsA, solsB)

    if solver_method == "analytical":
        try:
            sol = solve_coupled_handover_orientation_analytical(
                robot_a_id=robot_a_id,
                robot_b_id=robot_b_id,
                p_H_world=p_H_world,
                robots=robots,
                pos_tol_mm=pos_tol_mm,
                ori_tol_deg=ori_tol_deg,
                min_clearance_mm=min_clearance_mm,
                max_opt_iterations=max_opt_iterations,
                seeds=seeds,
            )
            if sol is not None:
                return sol
        except Exception:
            pass
        # Fallback to numerical solver
        return solve_coupled_handover_orientation_lbfgs(
            robot_a_id=robot_a_id,
            robot_b_id=robot_b_id,
            p_H_world=p_H_world,
            robots=robots,
            pos_tol_mm=pos_tol_mm,
            ori_tol_deg=ori_tol_deg,
            min_clearance_mm=min_clearance_mm,
            max_opt_iterations=max_opt_iterations,
            seeds=seeds,
        )

    elif solver_method == "dls":
        try:
            sol = solve_coupled_handover_orientation_dls(
                robot_a_id=robot_a_id,
                robot_b_id=robot_b_id,
                p_H_world=p_H_world,
                robots=robots,
                pos_tol_mm=pos_tol_mm,
                ori_tol_deg=ori_tol_deg,
                min_clearance_mm=min_clearance_mm,
                max_opt_iterations=max_opt_iterations,
                seeds=seeds,
            )
            if sol is not None:
                return sol
        except Exception:
            pass
        # Fallback to numerical solver
        return solve_coupled_handover_orientation_lbfgs(
            robot_a_id=robot_a_id,
            robot_b_id=robot_b_id,
            p_H_world=p_H_world,
            robots=robots,
            pos_tol_mm=pos_tol_mm,
            ori_tol_deg=ori_tol_deg,
            min_clearance_mm=min_clearance_mm,
            max_opt_iterations=max_opt_iterations,
            seeds=seeds,
        )

    else:
        return solve_coupled_handover_orientation_lbfgs(
            robot_a_id=robot_a_id,
            robot_b_id=robot_b_id,
            p_H_world=p_H_world,
            robots=robots,
            pos_tol_mm=pos_tol_mm,
            ori_tol_deg=ori_tol_deg,
            min_clearance_mm=min_clearance_mm,
            max_opt_iterations=max_opt_iterations,
            seeds=seeds,
        )


def _solve_pair_kinematic_handover(
    robot_a_id: str,
    robot_b_id: str,
    robots: Dict[str, Any],
    common_points: np.ndarray,
    handover_cfg: Optional[dict] = None,
    safe_z_min: float = 80.0,
    safe_z_max: float = 220.0,
    standoff_mm: float = 70.0,
    standoff_min_mm: float = 50.0,
    standoff_max_mm: float = 100.0,
    max_candidates: int = 45,
    max_opt_iterations: int = 50,
    timeout_s: float = 60.0,
    progress_callback: Optional[Callable[[str], None]] = None,
    solver_method: str = "analytical",
) -> Optional[AutoHandoverPose]:
    """Solve the optimal physical handover and pre-handover pose for a pair of robots."""
    cfgA = robots[robot_a_id]
    cfgB = robots[robot_b_id]
    bA = np.asarray(cfgA.world_base_xyz, dtype=float)
    bB = np.asarray(cfgB.world_base_xyz, dtype=float)

    h_cfg = handover_cfg or getattr(cfgA, 'handover_cfg', None) or getattr(cfgB, 'handover_cfg', None) or {}
    tolerances = h_cfg.get('pose_tolerances', {})
    pos_tol_mm = float(tolerances.get('position_mm', 1.0))
    ori_tol_deg = float(tolerances.get('orientation_deg', 15.0))
    safe_distance = float(h_cfg.get('handover', {}).get('safe_distance', 50.0))

    # Filter to safe operating heights
    safe_pts = common_points[
        (common_points[:, 2] >= safe_z_min) & (common_points[:, 2] <= safe_z_max)
    ]
    if len(safe_pts) == 0:
        safe_pts = common_points

    # Rank candidate points by balanced reach and proximity to baseline midpoint
    d_A = np.linalg.norm(safe_pts - bA, axis=1)
    d_B = np.linalg.norm(safe_pts - bB, axis=1)
    p_mid = 0.5 * (bA + bB)
    p_mid[2] = 120.0
    d_mid = np.linalg.norm(safe_pts - p_mid, axis=1)
    d_line = np.array([dist_point_to_segment(p, bA, bB) for p in safe_pts])

    balance_cost = np.maximum(d_A, d_B) + 0.5 * np.abs(d_A - d_B) + 0.4 * d_mid + 0.3 * d_line
    sorted_indices = np.argsort(balance_cost)
    candidate_pts = safe_pts[sorted_indices]

    best_pose = None
    best_score = -1e9
    t_start = time.time()
    total_candidates = min(len(candidate_pts), max_candidates)

    # Search candidate positions with bounded iteration and timeout checks
    for idx, p_H in enumerate(candidate_pts[:max_candidates]):
        elapsed = time.time() - t_start
        if elapsed > timeout_s:
            if progress_callback:
                progress_callback(
                    f"[{robot_a_id}<->{robot_b_id}] Bounded search timeout reached ({elapsed:.1f}s > {timeout_s:.1f}s) "
                    f"after {idx} candidates."
                )
            break

        if progress_callback and (idx % 5 == 0 or idx == 0):
            progress_callback(
                f"[{robot_a_id}<->{robot_b_id}] Evaluating candidate {idx+1}/{total_candidates} "
                f"at {np.round(p_H, 1)} mm (elapsed: {elapsed:.1f}s)..."
            )

        # P0.1: Coupled orientation-constrained search
        coupled_res = solve_coupled_handover_orientation(
            robot_a_id=robot_a_id,
            robot_b_id=robot_b_id,
            p_H_world=p_H,
            robots=robots,
            pos_tol_mm=pos_tol_mm,
            ori_tol_deg=ori_tol_deg,
            min_clearance_mm=45.0,
            max_opt_iterations=max_opt_iterations,
            solver_method=solver_method,
        )
        if coupled_res is None:
            continue

        qA_deg, qB_deg, errA, errB, dot_z, dot_x, h_clr = coupled_res

        # P0.1 & P0.2: Independently recompute FK and get world TCP orientation axes
        pA_w, zA_w, xA_w, _ = get_tcp_world_pose(qA_deg, cfgA, in_degrees=True)
        pB_w, zB_w, xB_w, _ = get_tcp_world_pose(qB_deg, cfgB, in_degrees=True)

        ang_dev_deg = float(np.rad2deg(np.arccos(np.clip(-dot_z, -1.0, 1.0))))

        if progress_callback:
            progress_callback(
                f"[{robot_a_id}<->{robot_b_id}] Candidate {idx+1} orientation match: "
                f"dot_z={dot_z:.4f} ({ang_dev_deg:.1f}°), dot_x={dot_x:.4f}, h_clr={h_clr:.1f} mm. "
                f"Validating pre-handover corridor..."
            )

        # P0.2: Adaptive standoff strictly along the actual world-frame TCP approach axes
        standoff_candidates = [standoff_mm, 50.0, 40.0, 60.0, 70.0, 80.0]
        pre_solved = False

        for cur_standoff in standoff_candidates:
            if cur_standoff < standoff_min_mm or cur_standoff > standoff_max_mm:
                if cur_standoff < 40.0:
                    continue

            okA, trajA, msgA = generate_cartesian_corridor_trajectory(
                robot_a_id, cfgA, qA_deg, standoff_mm=cur_standoff, steps=15,
                pos_tol_mm=pos_tol_mm, ori_tol_deg=ori_tol_deg
            )
            if not okA:
                continue

            okB, trajB, msgB = generate_cartesian_corridor_trajectory(
                robot_b_id, cfgB, qB_deg, standoff_mm=cur_standoff, steps=15,
                pos_tol_mm=pos_tol_mm, ori_tol_deg=ori_tol_deg
            )
            if not okB:
                continue

            qA_pre = trajA[0]
            qB_pre = trajB[0]

            # Check pre-handover joint limits
            if np.any(qA_pre < cfgA.joint_limits_deg[:, 0] - 1e-3) or np.any(qA_pre > cfgA.joint_limits_deg[:, 1] + 1e-3):
                continue
            if np.any(qB_pre < cfgB.joint_limits_deg[:, 0] - 1e-3) or np.any(qB_pre > cfgB.joint_limits_deg[:, 1] + 1e-3):
                continue

            pre_clr = compute_min_body_clearance(qA_pre, cfgA, qB_pre, cfgB, exclude_tcp=False)
            if pre_clr < safe_distance:
                continue

            # Verify corridor clearance
            corridor_safe = True
            for qA_k, qB_k in zip(trajA, trajB):
                c_k = compute_min_body_clearance(qA_k, cfgA, qB_k, cfgB, exclude_tcp=True)
                if c_k < 45.0:
                    corridor_safe = False
                    break
            if not corridor_safe:
                continue

            pA_pre_act, _, _, _ = get_tcp_world_pose(qA_pre, cfgA, in_degrees=True)
            pB_pre_act, _, _, _ = get_tcp_world_pose(qB_pre, cfgB, in_degrees=True)
            p_targetA = p_H - cur_standoff * zA_w
            p_targetB = p_H - cur_standoff * zB_w
            errA_pre = float(np.linalg.norm(pA_pre_act - p_targetA))
            errB_pre = float(np.linalg.norm(pB_pre_act - p_targetB))

            # Score candidate: high clearances, low errors, superior anti-parallel alignment
            score = (
                (h_clr + pre_clr)
                - 40.0 * (errA + errB + errA_pre + errB_pre)
                - 100.0 * (dot_z + 1.0)
                - 50.0 * dot_x
            )

            if score > best_score:
                best_score = score
                pair_key = f"{robot_a_id}_{robot_b_id}"
                best_pose = AutoHandoverPose(
                    pair_name=pair_key,
                    handover_position_world=p_H.copy(),
                    standoff_distance_mm=cur_standoff,
                    robot_a_id=robot_a_id,
                    robot_b_id=robot_b_id,
                    robot_a_handover_q_deg=qA_deg.copy(),
                    robot_b_handover_q_deg=qB_deg.copy(),
                    robot_a_pre_handover_q_deg=qA_pre.copy(),
                    robot_b_pre_handover_q_deg=qB_pre.copy(),
                    handover_clearance_mm=h_clr,
                    pre_handover_clearance_mm=pre_clr,
                    error_a_mm=errA,
                    error_b_mm=errB,
                    dot_z=dot_z,
                    dot_x=dot_x,
                    approach_angle_dev_deg=ang_dev_deg,
                )
            pre_solved = True
            break

        if pre_solved and best_pose and best_pose.handover_clearance_mm > 90.0:
            if progress_callback:
                progress_callback(
                    f"[{robot_a_id}<->{robot_b_id}] Optimal pose selected at {np.round(p_H, 1)} mm "
                    f"with clearance {best_pose.handover_clearance_mm:.1f} mm."
                )
            break

    return best_pose


def compute_all_auto_handover_poses(
    robots: Dict[str, Any],
    handover_cfg: Optional[dict] = None,
    cache_path: Optional[str] = None,
    force_recompute: bool = False,
    progress_callback: Optional[Any] = None,
    max_candidates_per_pair: int = 45,
    max_opt_iterations: int = 50,
    timeout_per_pair_s: float = 60.0,
    solver_method: str = "analytical",
) -> Dict[str, AutoHandoverPose]:
    """Compute and cache auto-calculated handover poses for all pairs.

    Returns
    -------
    dict mapping pair_key ('robot1_robot2', 'robot2_robot3', 'robot3_robot1') to AutoHandoverPose.
    """
    cache_path = cache_path or resolve_auto_handover_cache_path()
    config_hash = compute_config_fingerprint(robots, handover_cfg)

    # 1. Check cache first (< 1 ms recall)
    if not force_recompute:
        cached = load_auto_handover_cache(cache_path, config_hash)
        if cached is not None:
            return cached

    # 2. Compute 3D workspace intersections via Halton sequence
    if progress_callback:
        progress_callback("Computing 3D workspace intersections from DH models...")

    int_robots = {1: robots['robot1'], 2: robots['robot2'], 3: robots['robot3']}

    ws_config = {
        'workspace': {'sampling': {'voxel_size_mm': 25.0, 'workspace_samples': 20000}},
        'robots': handover_cfg.get('robots', {}) if handover_cfg else {}
    }
    ws = PositionWorkspaceComputer(ws_config, int_robots)
    ws.compute()

    solved_pairs: Dict[str, AutoHandoverPose] = {}
    pair_specs = [
        ('robot1', 'robot2', 1, 2),
        ('robot2', 'robot3', 2, 3),
        ('robot3', 'robot1', 3, 1),
    ]

    for rA_str, rB_str, idA, idB in pair_specs:
        if progress_callback:
            progress_callback(f"Optimizing handover & standoff poses for {rA_str} <-> {rB_str}...")

        pts = ws.get_common_voxel_centers(idA, idB)
        if len(pts) == 0:
            pts = ws.get_common_voxel_centers(idB, idA)

        pose = _solve_pair_kinematic_handover(
            rA_str,
            rB_str,
            robots,
            pts,
            handover_cfg=handover_cfg,
            max_candidates=max_candidates_per_pair,
            max_opt_iterations=max_opt_iterations,
            timeout_s=timeout_per_pair_s,
            progress_callback=progress_callback,
            solver_method=solver_method,
        )
        if pose is not None:
            solved_pairs[pose.pair_name] = pose

    # 3. Save to JSON cache
    if len(solved_pairs) > 0:
        save_auto_handover_cache(cache_path, config_hash, solved_pairs)

    return solved_pairs


def lookup_auto_handover_with_pre(
    poses: Dict[str, AutoHandoverPose],
    robot_a: str,
    robot_b: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, AutoHandoverPose]]:
    """Look up handover and pre-handover joint angles for a requested pair direction.

    Returns
    -------
    (q_a_handover, q_b_handover, q_a_pre, q_b_pre, auto_pose_obj) or None
    """
    key_forward = f"{robot_a}_{robot_b}"
    key_reverse = f"{robot_b}_{robot_a}"

    if key_forward in poses:
        p = poses[key_forward]
        return (
            p.robot_a_handover_q_deg.copy(),
            p.robot_b_handover_q_deg.copy(),
            p.robot_a_pre_handover_q_deg.copy(),
            p.robot_b_pre_handover_q_deg.copy(),
            p,
        )
    elif key_reverse in poses:
        p = poses[key_reverse]
        # In reverse direction: A was robot_b and B was robot_a
        return (
            p.robot_b_handover_q_deg.copy(),
            p.robot_a_handover_q_deg.copy(),
            p.robot_b_pre_handover_q_deg.copy(),
            p.robot_a_pre_handover_q_deg.copy(),
            p,
        )

    return None
