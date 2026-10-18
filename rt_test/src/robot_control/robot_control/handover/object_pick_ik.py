"""Object-Pose IK for Pick-and-Place in Multi-Robot Systems.

This module solves for robot joint configurations given an object target
position (x, y, z) and optional object yaw angle. It aligns the gripper
finger axis (TCP local X) with the object yaw while searching adaptive
approach tilt angles (TCP local Z) to maximize reachable workspace.

Features:
- Compatible with arbitrary robot joint counts via dynamic config.
- 5-DOF: Task-priority nullspace formulation (XYZ position as hard
  primary task; finger-axis yaw alignment as secondary task projected in
  the position nullspace; approach tilt as weak objective).
- 6-DOF: Full 6D pose inverse kinematics (position + orientation).
- Adaptive approach tilt search: downward approach preferred.
- Lexicographic filtering of hard constraints and normalized scoring.
- Multi-start deterministic seeds with priority for current joint position.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from robot_control.handover.handover_ik import (
    adaptive_damping,
    geometric_jacobian_and_fk,
    rotation_error_vector,
    rotation_matrix_to_quaternion,
    singularity_metrics,
)
from robot_control.kinematics.fk_math import fk


def rotation_x(angle_rad: float) -> np.ndarray:
    """Rotation about local X axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def rotation_y(angle_rad: float) -> np.ndarray:
    """Rotation about local Y axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def rotation_z(angle_rad: float) -> np.ndarray:
    """Rotation about world/base Z axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def generate_pick_orientations(
    object_yaw_deg: float = 0.0,
    tilt_angles_deg: Optional[List[float]] = None,
    finger_axis: str = 'x',
) -> List[dict]:
    """Generate candidate TCP orientations for pick-and-place.

    Assumptions:
      - World/Base Z points upward.
      - TCP local Z is the gripper approach axis.
      - TCP local X or Y is the gripper finger-alignment axis (configurable).
      - A downward approach is the zero-tilt orientation.

    Returns:
      List of dicts containing:
        rotation: 3x3 rotation matrix
        yaw_deg: normalized yaw in [0, 360)
        tilt_deg: approach tilt in degrees
        approach_direction: unit 3D vector of TCP local Z
    """
    if tilt_angles_deg is None:
        # Search downward first, then progressively tilted approaches.
        tilt_angles_deg = [
            0.0, 10.0, -10.0, 20.0, -20.0,
            30.0, -30.0, 45.0, -45.0,
            60.0, -60.0, 75.0, -75.0, 90.0, -90.0,
        ]

    yaw = np.deg2rad(float(object_yaw_deg))
    r_yaw = rotation_z(yaw)

    use_y = str(finger_axis).strip().lower() == 'y'

    candidates = []
    for tilt_deg in tilt_angles_deg:
        tilt = np.deg2rad(float(tilt_deg))

        if use_y:
            # Gripper fingers are along local Y.
            # At zero tilt: local Y = [cos(yaw), sin(yaw), 0],
            # local Z = [0, 0, -1] (downward),
            # local X = Y x Z = [-sin(yaw), cos(yaw), 0].
            # Tilting about the finger axis (local Y):
            r_base = r_yaw @ np.array([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ])
            r_tcp = r_base @ rotation_y(tilt)
        else:
            # Gripper fingers are along local X (default).
            # At zero tilt: local X = [cos(yaw), sin(yaw), 0],
            # local Z = [0, 0, -1] (downward).
            # Tilting about the finger axis (local X):
            r_tcp = r_yaw @ rotation_x(np.pi + tilt)

        # Direction in which the TCP's local Z axis points.
        approach = r_tcp[:, 2].copy()
        norm = np.linalg.norm(approach)
        if norm > 1e-9:
            approach /= norm

        candidates.append({
            'rotation': r_tcp,
            'yaw_deg': float(object_yaw_deg) % 360.0,
            'tilt_deg': float(tilt_deg),
            'approach_direction': approach,
        })

    return candidates


def parse_object_pick_target(
    x: float,
    y: float,
    z: float,
    yaw_deg: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Normalize an object target.

    Position units must match existing FK/IK implementation (currently mm).
    Missing yaw defaults to 0 degrees. Explicit yaw is normalized to [0, 360).
    """
    position = np.asarray([x, y, z], dtype=float)

    if not np.all(np.isfinite(position)):
        raise ValueError('Target X, Y, Z must be finite numbers.')

    yaw = 0.0 if yaw_deg is None else float(yaw_deg)
    if not np.isfinite(yaw):
        raise ValueError('Object yaw must be a finite number.')

    return position, float(yaw % 360.0)


def compute_achieved_tilt_deg(
    r_tcp: np.ndarray,
    object_yaw_deg: float,
    finger_axis: str = 'x',
) -> float:
    """Compute signed approach tilt in degrees from achieved TCP rotation.

    At tilt = 0, approach vector is [0, 0, -1] (downward).
    Positive/negative tilt rotates the approach vector about the finger axis.
    """
    yaw_rad = np.deg2rad(object_yaw_deg)
    # Approach axis is TCP local Z
    z_c = r_tcp[:, 2]
    # In-plane component perpendicular to finger axis
    s = -z_c[0] * np.sin(yaw_rad) + z_c[1] * np.cos(yaw_rad)
    c = -z_c[2]
    tilt_rad = np.arctan2(s, c)
    return float(np.rad2deg(tilt_rad))


def compute_finger_alignment_error_deg(
    r_tcp: np.ndarray,
    object_yaw_deg: float,
    finger_axis: str = 'x',
) -> float:
    """Compute angle error between finger axis and object yaw vector.

    Evaluates both 3D vector alignment and in-plane yaw alignment in the
    object/table plane.
    """
    yaw_rad = np.deg2rad(object_yaw_deg)
    target_axis = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
    finger_col = 1 if str(finger_axis).strip().lower() == 'y' else 0
    actual_axis = r_tcp[:, finger_col]
    cos_angle = np.clip(np.dot(actual_axis, target_axis), -1.0, 1.0)
    angle_3d = float(np.rad2deg(np.arccos(cos_angle)))

    yaw_actual = float(np.rad2deg(
        np.arctan2(actual_axis[1], actual_axis[0])
    ) % 360.0)
    yaw_diff = abs(yaw_actual - (object_yaw_deg % 360.0))
    if yaw_diff > 180.0:
        yaw_diff = 360.0 - yaw_diff

    return float(min(angle_3d, yaw_diff))


@dataclass
class ObjectPickIKCandidate:
    """A candidate joint solution for object pick."""

    robot_id: str
    q: np.ndarray
    q_deg: np.ndarray
    target_position: np.ndarray
    achieved_position: np.ndarray
    position_error: float
    requested_yaw_deg: float
    requested_tilt_deg: float
    achieved_tilt_deg: float
    rotation: np.ndarray
    quaternion: np.ndarray
    alignment_error_deg: float
    orientation_error_deg: float
    joint_limit_margin_deg: float
    manipulability: float
    cost: float
    is_valid: bool
    finger_axis: str = 'x'


@dataclass
class ObjectPickIKResult:
    """Result of object pick IK solve."""

    success: bool
    robot_id: str
    q: Optional[np.ndarray] = None
    q_deg: Optional[np.ndarray] = None
    target_position: Optional[np.ndarray] = None
    achieved_position: Optional[np.ndarray] = None
    position_error: Optional[float] = None
    requested_yaw_deg: Optional[float] = None
    selected_tilt_deg: Optional[float] = None
    achieved_tilt_deg: Optional[float] = None
    rotation: Optional[np.ndarray] = None
    quaternion: Optional[np.ndarray] = None
    alignment_error_deg: Optional[float] = None
    orientation_error_deg: Optional[float] = None
    joint_limit_margin_deg: Optional[float] = None
    manipulability: Optional[float] = None
    candidate_count: int = 0
    candidates: List[ObjectPickIKCandidate] = field(default_factory=list)
    failure_reason: Optional[str] = None
    cost: Optional[float] = None
    finger_axis: str = 'x'


def _solve_seed_object_ik(
    target_position: np.ndarray,
    r_desired: np.ndarray,
    requested_tilt_deg: float,
    object_yaw_deg: float,
    config,
    q_init: np.ndarray,
    current_q_rad: Optional[np.ndarray] = None,
    pos_tolerance: float = 1.0,
    ori_tolerance_deg: float = 15.0,
    align_tolerance_deg: float = 5.0,
    min_margin_deg: float = 0.5,
    min_manipulability: float = 1e-4,
    max_iterations: int = 100,
    tilt_weight: float = 0.05,
    cost_weights: Optional[dict] = None,
    finger_axis: str = 'x',
) -> Optional[ObjectPickIKCandidate]:
    """Solve IK for a single seed and return a candidate if valid."""
    q_min = config.joint_limits_rad[:, 0]
    q_max = config.joint_limits_rad[:, 1]
    num_joints = config.num_joints

    q = np.clip(np.asarray(q_init, dtype=float).copy(), q_min, q_max)

    for _ in range(max_iterations):
        j_mat, p_current, r_current = geometric_jacobian_and_fk(q, config)
        e_p = target_position - p_current
        e_r = rotation_error_vector(r_current, r_desired)

        pos_err = float(np.linalg.norm(e_p))
        align_err_deg = compute_finger_alignment_error_deg(
            r_current, object_yaw_deg, finger_axis=finger_axis
        )

        if num_joints < 6:
            # 5-DOF Task-Priority Nullspace formulation
            if (
                pos_err <= pos_tolerance
                and align_err_deg <= align_tolerance_deg
            ):
                break

            j_v = j_mat[:3, :]
            j_w = j_mat[3:, :]

            lam_p = 1e-4
            j_v_pinv = j_v.T @ np.linalg.inv(
                j_v @ j_v.T + (lam_p ** 2) * np.eye(3)
            )
            dq_p = j_v_pinv @ e_p

            n_pos = np.eye(num_joints) - j_v_pinv @ j_v

            finger_col = 1 if str(finger_axis).strip().lower() == 'y' else 0
            finger_axis_current = r_current[:, finger_col]
            e_r_par = np.dot(e_r, finger_axis_current) * finger_axis_current
            e_r_perp = e_r - e_r_par

            jn_w = j_w @ n_pos
            lam_w = 1e-2
            jn_w_pinv = jn_w.T @ np.linalg.inv(
                jn_w @ jn_w.T + (lam_w ** 2) * np.eye(3)
            )
            dq_w = jn_w_pinv @ (e_r_perp + tilt_weight * e_r_par)

            if pos_err > 0.8 * pos_tolerance:
                dq = 0.8 * dq_p
            else:
                dq = 0.8 * dq_p + 0.5 * (n_pos @ dq_w)

            step_norm = np.linalg.norm(dq)
            if step_norm > 0.35:
                dq = dq * (0.35 / step_norm)

            q_next = np.clip(q + dq, q_min, q_max)
            if np.linalg.norm(q_next - q) < 1e-6:
                q = q_next
                break
            q = q_next
        else:
            # 6-DOF: Full 6D pose IK with balanced units
            r_err_mat = r_desired.T @ r_current
            tr = np.trace(r_err_mat)
            ori_err_deg = float(np.rad2deg(
                np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
            ))

            if pos_err <= pos_tolerance and ori_err_deg <= ori_tolerance_deg:
                break

            w_p = 1.0 / 100.0
            j_scaled = np.vstack([w_p * j_mat[:3, :], j_mat[3:, :]])
            e_scaled = np.concatenate([w_p * e_p, e_r])

            sigma_min, _ = singularity_metrics(j_mat)
            lam = adaptive_damping(sigma_min)
            h_mat = j_scaled.T @ j_scaled + (lam ** 2) * np.eye(num_joints)
            g_vec = j_scaled.T @ e_scaled
            try:
                dq = np.linalg.solve(h_mat, g_vec)
            except np.linalg.LinAlgError:
                break

            step_norm = np.linalg.norm(dq)
            if step_norm > 0.4:
                dq = dq * (0.4 / step_norm)

            q_next = np.clip(q + 0.6 * dq, q_min, q_max)
            if np.linalg.norm(q_next - q) < 1e-6:
                q = q_next
                break
            q = q_next

    # Final independent FK verification
    achieved_p, t_fk = fk(q, config)
    r_fk = t_fk[:3, :3]
    final_pos_err = float(np.linalg.norm(target_position - achieved_p))

    final_align_err_deg = compute_finger_alignment_error_deg(
        r_fk, object_yaw_deg, finger_axis=finger_axis
    )
    r_err_final = r_desired.T @ r_fk
    tr_final = np.trace(r_err_final)
    final_ori_err_deg = float(np.rad2deg(
        np.arccos(np.clip((tr_final - 1.0) / 2.0, -1.0, 1.0))
    ))
    achieved_tilt_deg = compute_achieved_tilt_deg(
        r_fk, object_yaw_deg, finger_axis=finger_axis
    )

    # Joint limit margins
    margin_low = q - q_min
    margin_high = q_max - q
    min_margin_rad = float(np.min(np.minimum(margin_low, margin_high)))
    margin_deg = float(np.rad2deg(min_margin_rad))

    j_final, _, _ = geometric_jacobian_and_fk(q, config)
    final_sigma, _ = singularity_metrics(j_final)

    # Check hard constraints
    is_valid = True
    if final_pos_err > pos_tolerance:
        is_valid = False
    if num_joints < 6:
        if final_align_err_deg > align_tolerance_deg:
            is_valid = False
    else:
        if final_ori_err_deg > ori_tolerance_deg:
            is_valid = False
    if margin_deg < min_margin_deg:
        is_valid = False
    if final_sigma < min_manipulability:
        is_valid = False

    # Normalized multi-objective cost calculation
    s_tilt = abs(achieved_tilt_deg) / 90.0
    s_pos = min(final_pos_err / max(pos_tolerance, 1e-6), 5.0)
    s_align = min(final_align_err_deg / max(align_tolerance_deg, 1e-6), 5.0)
    s_margin = 1.0 / (1.0 + max(margin_deg, 0.0))
    s_sigma = 1.0 / (1.0 + 100.0 * final_sigma)

    s_dq = 0.0
    if current_q_rad is not None:
        dq_norm = float(np.linalg.norm(q - current_q_rad))
        s_dq = dq_norm / (np.pi * np.sqrt(num_joints))

    w_tilt = 30.0
    w_pos = 20.0
    w_align = 20.0
    w_margin = 10.0
    w_sigma = 10.0
    w_dq = 35.0
    if cost_weights:
        w_tilt = float(cost_weights.get('tilt', 0.40)) * 75.0
        w_margin = float(cost_weights.get('joint_margin', 0.15)) * 66.6
        w_sigma = float(cost_weights.get('manipulability', 0.10)) * 100.0
        w_align = float(cost_weights.get('finger_alignment', 0.10)) * 200.0
        w_dq = float(cost_weights.get('current_q', 0.25)) * 140.0

    cost = (
        w_tilt * s_tilt
        + w_pos * s_pos
        + w_align * s_align
        + w_margin * s_margin
        + w_sigma * s_sigma
        + (w_dq * s_dq if current_q_rad is not None else 0.0)
    )

    return ObjectPickIKCandidate(
        robot_id=getattr(config, 'name', 'unknown'),
        q=q,
        q_deg=np.rad2deg(q),
        target_position=target_position,
        achieved_position=achieved_p,
        position_error=final_pos_err,
        requested_yaw_deg=object_yaw_deg,
        requested_tilt_deg=requested_tilt_deg,
        achieved_tilt_deg=achieved_tilt_deg,
        rotation=r_fk,
        quaternion=rotation_matrix_to_quaternion(r_fk),
        alignment_error_deg=final_align_err_deg,
        orientation_error_deg=final_ori_err_deg,
        joint_limit_margin_deg=margin_deg,
        manipulability=final_sigma,
        cost=cost,
        is_valid=is_valid,
        finger_axis=finger_axis,
    )


def solve_object_pick_ik(
    robot_id: str,
    x: float,
    y: float,
    z: float,
    object_yaw_deg: Optional[float] = None,
    config=None,
    current_q: Optional[Union[np.ndarray, List[float]]] = None,
    tilt_angles_deg: Optional[List[float]] = None,
    pos_tolerance: Optional[float] = None,
    ori_tolerance_deg: Optional[float] = None,
    align_tolerance_deg: Optional[float] = None,
    min_margin_deg: Optional[float] = None,
    min_manipulability: Optional[float] = None,
    frame: str = 'base',
    early_stop_on_tilt: bool = True,
    max_iterations: Optional[int] = None,
    handover_cfg: Optional[dict] = None,
    finger_axis: Optional[str] = None,
) -> ObjectPickIKResult:
    """Solve object-target pick IK for any robot with adaptive approach tilt.

    Parameters
    ----------
    robot_id : str
        Robot identifier, e.g. 'robot1', 'robot2', 'robot3'.
    x, y, z : float
        Target position (in mm).
    object_yaw_deg : float, optional
        Target object yaw in degrees. Defaults to 0.0. Normalized to [0, 360).
    config : optional
        Kinematics configuration. If None, loaded via `load_stage1_robots()`.
    current_q : array-like, optional
        Current robot joint angles (in radians or degrees). Used as primary
        seed and ranking tie-breaker.
    tilt_angles_deg : list of float, optional
        Ordered candidate tilt angles. Defaults to downward first (0), then
        expanding tilts.
    pos_tolerance : float
        Maximum allowed Cartesian position error in mm (default 1.0 mm).
    ori_tolerance_deg : float
        Maximum allowed orientation error for >= 6-DOF robots in degrees
        (default 15.0 deg).
    align_tolerance_deg : float
        Maximum allowed finger-axis alignment error for 5-DOF robots
        (default 5.0 deg).
    min_margin_deg : float
        Minimum required margin from all joint limits in degrees
        (default 0.5 deg).
    min_manipulability : float
        Minimum acceptable smallest singular value (default 1e-4).
    frame : str
        Reference frame of input position: 'base' (default) or 'world'.
    early_stop_on_tilt : bool
        If True, stops search when a tilt angle yields valid solutions
        (default True).
    max_iterations : int
        Maximum DLS solver iterations per seed.
    handover_cfg : dict, optional
        Handover configuration dictionary.
    finger_axis : str, optional
        Gripper finger axis to align: 'x' (default) or 'y'. If None, loaded
        from robot YAML or handover YAML.

    Returns
    -------
    ObjectPickIKResult
        Structured result containing solution status, joints, errors, and
        metadata.
    """
    target_pos, normalized_yaw = parse_object_pick_target(
        x, y, z, object_yaw_deg
    )

    if config is None:
        from robot_control.handover.stage1 import load_stage1_robots
        _, robots, _ = load_stage1_robots()
        if robot_id not in robots:
            return ObjectPickIKResult(
                success=False,
                robot_id=robot_id,
                target_position=target_pos,
                requested_yaw_deg=normalized_yaw,
                candidate_count=0,
                finger_axis=str(finger_axis or 'x').lower(),
                failure_reason=(
                    f"Unknown robot_id '{robot_id}'. "
                    f'Available: {list(robots.keys())}'
                ),
            )
        config = robots[robot_id]

    # Handle frame conversion
    target_base = target_pos.copy()
    base_yaw_deg = normalized_yaw
    if (
        frame == 'world'
        and hasattr(config, 'world_base_R')
        and hasattr(config, 'world_base_xyz')
    ):
        origin = np.asarray(config.world_base_xyz, dtype=float)
        target_base = config.world_base_R.T @ (target_pos - origin)
        # Transform yaw direction vector into base frame
        yaw_rad = np.deg2rad(normalized_yaw)
        v_world = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
        v_base = config.world_base_R.T @ v_world
        base_yaw_deg = float(
            np.rad2deg(np.arctan2(v_base[1], v_base[0])) % 360.0
        )

    # Resolve parameters from config / handover_cfg if not explicitly provided
    cfg_src = handover_cfg or getattr(config, 'handover_cfg', None) or {}
    ik_cfg = cfg_src.get('object_pick_ik', {})
    tols_cfg = ik_cfg.get('tolerances', {})
    solver_cfg = ik_cfg.get('solver', {})

    eff_pos_tol = float(tols_cfg.get('position_mm', 1.0)) if pos_tolerance is None else float(pos_tolerance)
    eff_ori_tol = float(tols_cfg.get('orientation_deg', 15.0)) if ori_tolerance_deg is None else float(ori_tolerance_deg)
    eff_align_tol = float(tols_cfg.get('finger_alignment_deg', 5.0)) if align_tolerance_deg is None else float(align_tolerance_deg)
    eff_margin_tol = float(tols_cfg.get('joint_margin_deg', 0.5)) if min_margin_deg is None else float(min_margin_deg)
    eff_manip_tol = float(tols_cfg.get('min_manipulability', 1e-4)) if min_manipulability is None else float(min_manipulability)
    eff_max_iter = int(solver_cfg.get('max_iterations', 100)) if max_iterations is None else int(max_iterations)

    # Resolve finger_axis: arg -> robot config -> handover config -> 'x'
    eff_finger_axis = 'x'
    if finger_axis is not None:
        eff_finger_axis = str(finger_axis).strip().lower()
    elif hasattr(config, 'gripper_finger_axis') and config.gripper_finger_axis:
        eff_finger_axis = str(config.gripper_finger_axis).strip().lower()
    elif 'finger_axis' in ik_cfg:
        eff_finger_axis = str(ik_cfg['finger_axis']).strip().lower()
    if eff_finger_axis not in ('x', 'y'):
        eff_finger_axis = 'x'

    if tilt_angles_deg is None and 'candidate_tilts_deg' in ik_cfg:
        tilt_angles_deg = [float(v) for v in ik_cfg['candidate_tilts_deg']]

    num_joints = config.num_joints
    q_min = config.joint_limits_rad[:, 0]
    q_max = config.joint_limits_rad[:, 1]

    # Parse current_q
    current_q_rad = None
    if current_q is not None:
        raw_q = np.asarray(current_q, dtype=float)
        if len(raw_q) != num_joints:
            return ObjectPickIKResult(
                success=False,
                robot_id=robot_id,
                target_position=target_base,
                requested_yaw_deg=base_yaw_deg,
                candidate_count=0,
                finger_axis=eff_finger_axis,
                failure_reason=(
                    f'current_q length {len(raw_q)} does not match '
                    f'robot num_joints {num_joints}'
                ),
            )
        # Auto-detect degrees vs radians
        if (
            np.any(np.abs(raw_q) > 2 * np.pi + 1e-3)
            or np.any(raw_q < q_min - 0.1)
            or np.any(raw_q > q_max + 0.1)
        ):
            deg_rad = np.deg2rad(raw_q)
            fits_limits = (
                np.all(deg_rad >= q_min - 0.1)
                and np.all(deg_rad <= q_max + 0.1)
            )
            if fits_limits:
                current_q_rad = deg_rad
            else:
                current_q_rad = np.clip(deg_rad, q_min, q_max)
        else:
            current_q_rad = np.clip(raw_q, q_min, q_max)

    # Generate orientation candidates
    ori_candidates = generate_pick_orientations(
        object_yaw_deg=base_yaw_deg,
        tilt_angles_deg=tilt_angles_deg,
        finger_axis=eff_finger_axis,
    )

    # Multi-start initial seeds
    seeds = []
    if current_q_rad is not None:
        seeds.append(current_q_rad.copy())
    if hasattr(config, 'home_deg'):
        seeds.append(np.deg2rad(np.asarray(config.home_deg, dtype=float)))
    seeds.append(0.5 * (q_min + q_max))
    seeds.append(q_min + 0.2 * (q_max - q_min))
    seeds.append(q_max - 0.2 * (q_max - q_min))

    # Downward posture seeds for articulated manipulators
    if num_joints == 5:
        seeds.append(np.clip(
            np.deg2rad([0.0, -30.0, -60.0, 80.0, 0.0]), q_min, q_max
        ))
        seeds.append(np.clip(
            np.deg2rad([45.0, -30.0, -60.0, 80.0, 0.0]), q_min, q_max
        ))
        seeds.append(np.clip(
            np.deg2rad([-45.0, -30.0, -60.0, 80.0, 0.0]), q_min, q_max
        ))

    rng = np.random.RandomState(42)
    for _ in range(12):
        seeds.append(q_min + (q_max - q_min) * rng.rand(num_joints))

    valid_candidates: List[ObjectPickIKCandidate] = []
    all_evaluated_count = 0

    for ori in ori_candidates:
        r_d = ori['rotation']
        tilt_deg = ori['tilt_deg']
        tilt_valid_candidates: List[ObjectPickIKCandidate] = []

        for q_seed in seeds:
            all_evaluated_count += 1
            cand = _solve_seed_object_ik(
                target_position=target_base,
                r_desired=r_d,
                requested_tilt_deg=tilt_deg,
                object_yaw_deg=base_yaw_deg,
                config=config,
                q_init=q_seed,
                current_q_rad=current_q_rad,
                pos_tolerance=eff_pos_tol,
                ori_tolerance_deg=eff_ori_tol,
                align_tolerance_deg=eff_align_tol,
                min_margin_deg=eff_margin_tol,
                min_manipulability=eff_manip_tol,
                max_iterations=eff_max_iter,
                cost_weights=ik_cfg.get('cost_weights', None),
                finger_axis=eff_finger_axis,
            )

            if cand is not None and cand.is_valid:
                # Deduplicate joint solutions
                is_duplicate = False
                for existing in valid_candidates + tilt_valid_candidates:
                    if np.allclose(existing.q, cand.q, atol=1e-3, rtol=1e-3):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    tilt_valid_candidates.append(cand)

        if tilt_valid_candidates:
            valid_candidates.extend(tilt_valid_candidates)
            if early_stop_on_tilt:
                break

    if not valid_candidates:
        return ObjectPickIKResult(
            success=False,
            robot_id=robot_id,
            target_position=target_base,
            requested_yaw_deg=base_yaw_deg,
            candidate_count=0,
            candidates=[],
            finger_axis=eff_finger_axis,
            failure_reason=(
                'No feasible solution found satisfying hard constraints '
                f'within tolerance (evaluated {all_evaluated_count} candidate '
                f'seeds across {len(ori_candidates)} tilts).'
            ),
        )

    # Sort valid candidates by normalized multi-objective cost
    valid_candidates.sort(key=lambda c: c.cost)
    best = valid_candidates[0]

    return ObjectPickIKResult(
        success=True,
        robot_id=robot_id,
        q=best.q,
        q_deg=best.q_deg,
        target_position=best.target_position,
        achieved_position=best.achieved_position,
        position_error=best.position_error,
        requested_yaw_deg=best.requested_yaw_deg,
        selected_tilt_deg=best.requested_tilt_deg,
        achieved_tilt_deg=best.achieved_tilt_deg,
        rotation=best.rotation,
        quaternion=best.quaternion,
        alignment_error_deg=best.alignment_error_deg,
        orientation_error_deg=best.orientation_error_deg,
        joint_limit_margin_deg=best.joint_limit_margin_deg,
        manipulability=best.manipulability,
        candidate_count=len(valid_candidates),
        candidates=valid_candidates,
        finger_axis=eff_finger_axis,
        failure_reason=None,
        cost=best.cost,
    )
