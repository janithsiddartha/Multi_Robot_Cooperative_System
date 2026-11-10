"""Stage-1.5 Concurrent Handover & Body-to-Body Collision-Avoidance Planner.

Extends Stage 1 with:
  1. Simultaneous pre-positioning: Place robot moves to Handover simultaneously
     while Pick robot moves to Pick, saving execution time.
  2. Full-body link-to-link clearance checking (capsule/segment distance) to ensure
     that the entire robot bodies never collide during simultaneous or approach motions.
  3. Collision recovery / safe avoidance if an interpolated path would violate safe distance.
  4. Simultaneous return-to-home after place execution.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from robot_control.handover.stage1 import (
    Stage1Plan,
    build_stage1_plan,
    generate_interpolated_path,
    load_stage1_robots,
)
from robot_control.handover.handover_ik import (
    geometric_jacobian_and_fk,
    rotation_error_vector,
)
from robot_control.handover.object_pick_ik import solve_object_pick_ik
from robot_control.handover.stage1_ik import solve_stage1_ik, world_to_base
from robot_control.kinematics.fk_math import all_frame_positions


def dist_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from 3D point p to line segment a-b."""
    ab = b - a
    d_ab = float(np.dot(ab, ab))
    if d_ab < 1e-8:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / d_ab, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def dist_segment_to_segment(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> float:
    """Exact minimum Euclidean distance between segment p1-p2 and segment p3-p4."""
    d12 = float(np.linalg.norm(p2 - p1))
    d34 = float(np.linalg.norm(p4 - p3))
    if d12 < 1e-6 and d34 < 1e-6:
        return float(np.linalg.norm(p1 - p3))
    if d12 < 1e-6:
        return dist_point_to_segment(p1, p3, p4)
    if d34 < 1e-6:
        return dist_point_to_segment(p3, p1, p2)

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

    p_sc = p1 + sc * u
    p_tc = p3 + tc * v
    d_direct = float(np.linalg.norm(p_sc - p_tc))
    d_sc = dist_point_to_segment(p_sc, p3, p4)
    d_tc = dist_point_to_segment(p_tc, p1, p2)
    return min(d_direct, d_sc, d_tc)


def get_robot_link_segments(
    q_deg: np.ndarray, cfg: object, exclude_tcp: bool = False
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return all link segments of a robot arm in the common World frame.
    
    When exclude_tcp is True, excludes only the final segment representing
    the tool flange to TCP (fingertip contact geometry), while preserving
    all robot body links (wrist, forearm, elbow, upper arm, base).
    """
    q_rad = np.deg2rad(np.asarray(q_deg, dtype=float))
    pos_base = all_frame_positions(q_rad, cfg)
    pos_world = np.asarray([cfg.world_base_R @ p + cfg.world_base_xyz for p in pos_base])

    segments = []
    n = len(pos_world)
    limit = n - 2 if exclude_tcp else n - 1
    for i in range(limit):
        p_start, p_end = pos_world[i], pos_world[i + 1]
        if np.linalg.norm(p_end - p_start) > 1e-3:
            segments.append((p_start, p_end))
    return segments


def compute_min_body_clearance(
    q1_deg: np.ndarray,
    cfg1: object,
    q2_deg: np.ndarray,
    cfg2: object,
    exclude_tcp: bool = False,
) -> float:
    """Calculate the minimum physical distance between any link of Robot 1 and Robot 2."""
    seg1 = get_robot_link_segments(q1_deg, cfg1, exclude_tcp=exclude_tcp)
    seg2 = get_robot_link_segments(q2_deg, cfg2, exclude_tcp=exclude_tcp)

    min_dist = float("inf")
    for s1 in seg1:
        for s2 in seg2:
            d = dist_segment_to_segment(s1[0], s1[1], s2[0], s2[1])
            if d < min_dist:
                min_dist = d
    return min_dist


@dataclass
class Stage1_5Step:
    name: str
    pick_q: np.ndarray
    place_q: np.ndarray
    min_clearance_mm: float
    is_safe: bool
    description: str


@dataclass
class Stage1_5TrajectoryResult:
    needs_handover: bool
    pick_robot: str
    place_robot: str
    phase1_simultaneous: List[Stage1_5Step]
    phase3_pick_to_pre_handover: List[Stage1_5Step]
    phase4_advance_to_handover: List[Stage1_5Step]
    phase6_retract_to_pre_handover: List[Stage1_5Step]
    phase7_handover_to_place: List[Stage1_5Step]
    phase9_simultaneous_home: List[Stage1_5Step]
    min_overall_clearance_mm: float
    all_phases_safe: bool
    collision_avoidance_applied: bool

    @property
    def phase3_pick_to_handover(self) -> List[Stage1_5Step]:
        return self.phase3_pick_to_pre_handover

    @property
    def phase5_handover_to_place(self) -> List[Stage1_5Step]:
        return self.phase7_handover_to_place

    @property
    def phase7_simultaneous_home(self) -> List[Stage1_5Step]:
        return self.phase9_simultaneous_home

    @property
    def phase7_place_and_home(self) -> List[Stage1_5Step]:
        return self.phase7_handover_to_place

    @property
    def phase9_place_to_home(self) -> List[Stage1_5Step]:
        return self.phase9_simultaneous_home


def get_tcp_world_pose(
    q_deg_or_rad: np.ndarray, cfg: object, in_degrees: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate TCP world-frame position and axes for a robot."""
    from robot_control.kinematics.fk_math import fk
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


def solve_continuous_ik_waypoint(
    robot_id: str,
    cfg: object,
    p_target_w: np.ndarray,
    q_seed_deg: np.ndarray,
    z_target_w: Optional[np.ndarray] = None,
    ori_weight: float = 5.0,
) -> Tuple[np.ndarray, float, float]:
    """Solve IK for a Cartesian waypoint with posture continuity and orientation regularizer."""
    from scipy.optimize import minimize
    from robot_control.kinematics.fk_math import fk
    from robot_control.handover.stage1_ik import world_to_base

    q_seed_rad = np.deg2rad(np.asarray(q_seed_deg, dtype=float))
    p_target_b = world_to_base(p_target_w, cfg)
    q_bnds = cfg.joint_limits_rad
    Rw_b = cfg.world_base_R

    if z_target_w is not None:
        z_target_b = Rw_b.T @ z_target_w
    else:
        z_target_b = None

    def obj(q):
        p_tcp, T = fk(q, cfg)
        pos_err = np.sum((p_tcp - p_target_b) ** 2)
        reg = 0.05 * np.sum((q - q_seed_rad) ** 2)
        if z_target_b is not None:
            z_cur = T[:3, 2]
            dot_z = np.dot(z_cur, z_target_b)
            ori_err = (1.0 - dot_z) ** 2
        else:
            ori_err = 0.0
        return pos_err + ori_weight * ori_err + reg

    res = minimize(
        obj,
        q_seed_rad,
        bounds=q_bnds,
        method="L-BFGS-B",
        options={"maxiter": 120, "ftol": 1e-7},
    )
    q_deg = np.rad2deg(res.x)
    p_act, z_act, _, _ = get_tcp_world_pose(q_deg, cfg, in_degrees=True)
    pos_err_mm = float(np.linalg.norm(p_act - p_target_w))
    if z_target_w is not None:
        z_dev_deg = float(np.rad2deg(np.arccos(np.clip(np.dot(z_act, z_target_w), -1.0, 1.0))))
    else:
        z_dev_deg = 0.0
    return q_deg, pos_err_mm, z_dev_deg


def generate_cartesian_corridor_trajectory(
    robot_id: str,
    cfg: object,
    q_H_deg: np.ndarray,
    q_pre_deg: Optional[np.ndarray] = None,
    standoff_mm: float = 50.0,
    steps: int = 30,
    pos_tol_mm: float = 1.0,
    ori_tol_deg: float = 15.0,
    max_joint_step_deg: float = 15.0,
) -> Tuple[bool, List[np.ndarray], str]:
    """Generate Cartesian approach corridor waypoints from pre-handover to handover.
    
    Path:
        p(s) = p_pre + s * standoff * z_H
        R(s) = R_H
    for s in [0, 1].
    """
    p_H, z_H, _, _ = get_tcp_world_pose(q_H_deg, cfg, in_degrees=True)
    p_pre = p_H - standoff_mm * z_H

    # Check if pre-handover pose is continuous with handover pose
    has_continuous_pre = False
    if q_pre_deg is not None:
        p_act_pre, z_act_pre, _, _ = get_tcp_world_pose(q_pre_deg, cfg, in_degrees=True)
        pe_pre = np.linalg.norm(p_act_pre - p_pre)
        ze_pre = np.rad2deg(np.arccos(np.clip(np.dot(z_act_pre, z_H), -1.0, 1.0)))
        max_delta = np.max(np.abs(q_pre_deg - q_H_deg))
        if pe_pre <= pos_tol_mm and ze_pre <= ori_tol_deg and max_delta < 35.0:
            has_continuous_pre = True

    s_vals = np.linspace(0.0, 1.0, steps)

    if has_continuous_pre:
        qs_deg = []
        for s in s_vals:
            p_s = p_pre + s * standoff_mm * z_H
            q_seed = (1.0 - s) * q_pre_deg + s * q_H_deg
            q_sol, pe, ze = solve_continuous_ik_waypoint(robot_id, cfg, p_s, q_seed, z_target_w=z_H)
            if pe > pos_tol_mm or ze > ori_tol_deg:
                has_continuous_pre = False
                break
            qs_deg.append(q_sol)

        if has_continuous_pre:
            qs_deg[0] = q_pre_deg.copy()
            qs_deg[-1] = q_H_deg.copy()
            max_step = max(np.max(np.abs(qs_deg[i] - qs_deg[i - 1])) for i in range(1, steps))
            if max_step <= max_joint_step_deg:
                return True, qs_deg, "Success (seeded from continuous pre-handover)"

    # Solve backwards from q_H to p_pre
    q_cur = q_H_deg.copy()
    qs_backward = [q_cur.copy()]

    for s in reversed(s_vals[:-1]):
        p_s = p_pre + s * standoff_mm * z_H
        q_sol, pe, ze = solve_continuous_ik_waypoint(robot_id, cfg, p_s, q_cur, z_target_w=z_H)
        if pe > pos_tol_mm:
            return False, [], f"IK position error {pe:.2f} mm > {pos_tol_mm} mm at s={s:.2f}"
        if ze > ori_tol_deg:
            return False, [], f"IK orientation error {ze:.1f}° > {ori_tol_deg}° at s={s:.2f}"

        step_d = np.max(np.abs(q_sol - q_cur))
        if step_d > max_joint_step_deg:
            return False, [], f"Discontinuous joint step {step_d:.1f}° > {max_joint_step_deg}° at s={s:.2f}"

        q_cur = q_sol.copy()
        qs_backward.append(q_cur.copy())

    approach_qs_deg = list(reversed(qs_backward))
    if q_pre_deg is not None and np.max(np.abs(q_pre_deg - approach_qs_deg[0])) < 5.0:
        approach_qs_deg[0] = q_pre_deg.copy()
    approach_qs_deg[-1] = q_H_deg.copy()
    return True, approach_qs_deg, "Success (backwards continuous corridor)"


def compute_cartesian_linear_path(
    q_start_deg: np.ndarray,
    target_xyz: np.ndarray,
    config: Any,
    num_steps: int = 15,
    frame: str = "base",
) -> Optional[List[np.ndarray]]:
    """Straight-line Cartesian approach/depart using Jacobian IK tracking with locked orientation.

    At every interpolated step the TCP position is moved linearly from the
    current FK position to *target_xyz* while the TCP orientation captured
    at q_start_deg is kept constant (damped-least-squares Jacobian update).

    Returns a list of joint-angle arrays in DEGREES, or None if IK fails.
    """
    q = np.deg2rad(np.asarray(q_start_deg, dtype=float))
    q_min = config.joint_limits_rad[:, 0]
    q_max = config.joint_limits_rad[:, 1]

    # Convert to base frame if specified in world frame
    p_target = np.asarray(target_xyz, dtype=float)
    if frame == "world":
        p_target = world_to_base(p_target, config)

    # Capture starting TCP pose (position + orientation to freeze)
    _, p_start, R_start = geometric_jacobian_and_fk(q, config)

    # Use 5-DOF weight: heavily weight position, lightly weight orientation
    is_5dof = config.num_joints == 5
    if is_5dof:
        W = np.diag([1.0, 1.0, 1.0, 0.01, 0.01, 0.01])
    else:
        W = np.eye(6)

    path_deg = []
    for step in range(1, num_steps + 1):
        t = step / num_steps
        p_interp = p_start + t * (p_target - p_start)   # linear XYZ interpolation

        for _ in range(30):
            J, p_cur, R_cur = geometric_jacobian_and_fk(q, config)
            e_p = p_interp - p_cur
            e_R = rotation_error_vector(R_cur, R_start)
            e = np.concatenate([e_p, e_R])

            U, S, Vt = np.linalg.svd(J, full_matrices=False)
            sigma_min = S[-1]
            lam = 0.01 if sigma_min >= 0.01 else min(0.2, 0.01 + 0.1 * (1.0 - sigma_min / 0.01) ** 2)

            if is_5dof:
                H = J.T @ W @ J + (lam ** 2) * np.eye(config.num_joints)
                dq = np.linalg.solve(H, J.T @ W @ e)
            else:
                H = J.T @ J + (lam ** 2) * np.eye(config.num_joints)
                dq = np.linalg.solve(H, J.T @ e)

            q = np.clip(q + 0.6 * dq, q_min, q_max)
            if np.linalg.norm(e_p) < 0.5:
                break

        path_deg.append(np.rad2deg(q).copy())

    _, p_final, _ = geometric_jacobian_and_fk(q, config)
    if np.linalg.norm(p_final - p_target) > 5.0:
        return None
    return path_deg


def resolve_target_pre_pose(
    robot_id: str,
    cfg: Any,
    target_xyz_world: np.ndarray,
    yaw_deg: float = 0.0,
    approach_height_mm: float = 50.0,
    seed_q_deg: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve Pre-Target XYZ and joint angles (+50mm approach height).
    
    Returns (pre_xyz_world, pre_xyz_base, pre_q_deg).
    """
    pre_xyz_w = np.array(target_xyz_world, dtype=float).copy()
    pre_xyz_w[2] += approach_height_mm
    pre_xyz_b = world_to_base(pre_xyz_w, cfg)
    
    seed = seed_q_deg if seed_q_deg is not None else np.zeros(cfg.num_joints)
    
    # 1. Try object_pick_ik with yaw
    try:
        res = solve_object_pick_ik(
            robot_id,
            pre_xyz_b[0], pre_xyz_b[1], pre_xyz_b[2],
            object_yaw_deg=yaw_deg,
            config=cfg,
            current_q=seed,
            frame='base'
        )
        if res.success:
            return pre_xyz_w, pre_xyz_b, res.q_deg.copy()
    except Exception:
        pass
        
    # 2. Fallback to stage1_ik
    try:
        fb = solve_stage1_ik(robot_id, pre_xyz_w, cfg, current_q_deg=seed)
        if fb is not None:
            return pre_xyz_w, pre_xyz_b, fb.q_deg.copy()
    except Exception:
        pass
        
    # 3. Default fallback to seed
    return pre_xyz_w, pre_xyz_b, seed.copy()


def schedule_collision_aware_motion(
    start_pick_q: np.ndarray,
    target_pick_q: np.ndarray,
    start_place_q: np.ndarray,
    target_place_q: np.ndarray,
    cfg_pick: object,
    cfg_place: object,
    safe_distance: float,
    steps: int,
    pick_rid: str,
    place_rid: str,
    phase_name: str,
    desc_concurrent: str,
    desc_serialized_pick_first: str,
    desc_serialized_place_first: str,
) -> Tuple[List[Stage1_5Step], float, bool]:
    """Schedule independent robot motion with collision awareness.
    
    1. If concurrent independent motion is safe across all steps, execute concurrently.
    2. If concurrent motion conflicts (< safe_distance), automatically serialize with safe waiting.
    """
    raw_pick = generate_interpolated_path(start_pick_q, target_pick_q, steps=steps)
    raw_place = generate_interpolated_path(start_place_q, target_place_q, steps=steps)

    # 1. Evaluate concurrent execution
    concurrent_steps = []
    concurrent_min_c = float("inf")
    concurrent_safe = True

    for q_p, q_pl in zip(raw_pick, raw_place):
        c = compute_min_body_clearance(q_p, cfg_pick, q_pl, cfg_place, exclude_tcp=False)
        if c < concurrent_min_c:
            concurrent_min_c = c
        is_safe = (c >= safe_distance)
        if not is_safe:
            concurrent_safe = False
        concurrent_steps.append(
            Stage1_5Step(
                name=phase_name,
                pick_q=q_p,
                place_q=q_pl,
                min_clearance_mm=c,
                is_safe=is_safe,
                description=desc_concurrent,
            )
        )

    if concurrent_safe:
        return concurrent_steps, concurrent_min_c, False

    # 2. Concurrent is conflicting (< safe_distance). Serialize!
    # Option A: Pick robot moves first while Place robot waits at start_place_q.
    #           Then Place robot moves while Pick robot waits at target_pick_q.
    steps_A = []
    min_c_A = float("inf")
    safe_A = True
    for q_p in raw_pick:
        c = compute_min_body_clearance(q_p, cfg_pick, start_place_q, cfg_place, exclude_tcp=False)
        if c < min_c_A:
            min_c_A = c
        is_s = (c >= safe_distance)
        if not is_s:
            safe_A = False
        steps_A.append(
            Stage1_5Step(
                name=phase_name,
                pick_q=q_p,
                place_q=start_place_q,
                min_clearance_mm=c,
                is_safe=is_s,
                description=f"{desc_serialized_pick_first}: {pick_rid} moving ({place_rid} waiting)",
            )
        )
    for q_pl in raw_place:
        c = compute_min_body_clearance(target_pick_q, cfg_pick, q_pl, cfg_place, exclude_tcp=False)
        if c < min_c_A:
            min_c_A = c
        is_s = (c >= safe_distance)
        if not is_s:
            safe_A = False
        steps_A.append(
            Stage1_5Step(
                name=phase_name,
                pick_q=target_pick_q,
                place_q=q_pl,
                min_clearance_mm=c,
                is_safe=is_s,
                description=f"{desc_serialized_pick_first}: {place_rid} moving ({pick_rid} waiting)",
            )
        )

    # Option B: Place robot moves first while Pick robot waits at start_pick_q.
    #           Then Pick robot moves while Place robot waits at target_place_q.
    steps_B = []
    min_c_B = float("inf")
    safe_B = True
    for q_pl in raw_place:
        c = compute_min_body_clearance(start_pick_q, cfg_pick, q_pl, cfg_place, exclude_tcp=False)
        if c < min_c_B:
            min_c_B = c
        is_s = (c >= safe_distance)
        if not is_s:
            safe_B = False
        steps_B.append(
            Stage1_5Step(
                name=phase_name,
                pick_q=start_pick_q,
                place_q=q_pl,
                min_clearance_mm=c,
                is_safe=is_s,
                description=f"{desc_serialized_place_first}: {place_rid} moving ({pick_rid} waiting)",
            )
        )
    for q_p in raw_pick:
        c = compute_min_body_clearance(q_p, cfg_pick, target_place_q, cfg_place, exclude_tcp=False)
        if c < min_c_B:
            min_c_B = c
        is_s = (c >= safe_distance)
        if not is_s:
            safe_B = False
        steps_B.append(
            Stage1_5Step(
                name=phase_name,
                pick_q=q_p,
                place_q=target_place_q,
                min_clearance_mm=c,
                is_safe=is_s,
                description=f"{desc_serialized_place_first}: {pick_rid} moving ({place_rid} waiting)",
            )
        )

    if safe_A and (not safe_B or min_c_A >= min_c_B):
        return steps_A, min_c_A, True
    elif safe_B:
        return steps_B, min_c_B, True
    else:
        if min_c_A >= min_c_B:
            return steps_A, min_c_A, True
        else:
            return steps_B, min_c_B, True


def plan_stage1_5_synchronized_paths(
    plan: Stage1Plan,
    robots: Dict[str, object],
    safe_distance: Optional[float] = None,
    steps: Optional[int] = None,
    current_q: Optional[Dict[str, np.ndarray]] = None,
    handover_cfg: Optional[dict] = None,
) -> Stage1_5TrajectoryResult:
    """Compute synchronized simultaneous trajectories with link collision avoidance."""
    pick_rid = plan.pick_robot
    place_rid = plan.place_robot
    cfg_pick = robots[pick_rid]
    cfg_place = robots[place_rid]

    h_cfg = handover_cfg or getattr(cfg_pick, 'handover_cfg', None) or getattr(cfg_place, 'handover_cfg', None) or {}
    if safe_distance is None:
        safe_distance = float(h_cfg.get('handover', {}).get('safe_distance', 50.0))
    if steps is None:
        steps = int(h_cfg.get('execution', {}).get('trajectory', {}).get('steps', 30))

    current_q = current_q or {}
    start_pick_q = current_q.get(pick_rid, np.zeros(cfg_pick.num_joints)).copy()
    start_place_q = current_q.get(place_rid, np.zeros(cfg_place.num_joints)).copy()

    # -------------------------------------------------------------
    # INDUSTRIAL PRE-TARGET & CARTESIAN STRAIGHT-LINE SETUP (+50mm)
    # -------------------------------------------------------------
    pick_xyz_w = np.asarray(getattr(plan.pick, 'xyz_world', None) if getattr(plan.pick, 'xyz_world', None) is not None else [100.0, 100.0, 200.0], dtype=float)
    pick_xyz_b = np.asarray(getattr(plan.pick, 'xyz_base', None) if getattr(plan.pick, 'xyz_base', None) is not None else world_to_base(pick_xyz_w, cfg_pick), dtype=float)
    pick_yaw = float(getattr(plan.pick, 'requested_yaw_deg', 0.0) or 0.0)

    place_xyz_w = np.asarray(getattr(plan.place, 'xyz_world', None) if getattr(plan.place, 'xyz_world', None) is not None else [0.0, 150.0, 150.0], dtype=float)
    place_xyz_b = np.asarray(getattr(plan.place, 'xyz_base', None) if getattr(plan.place, 'xyz_base', None) is not None else world_to_base(place_xyz_w, cfg_place), dtype=float)
    place_yaw = float(getattr(plan.place, 'requested_yaw_deg', 0.0) or 0.0)

    app_h_pick = float(getattr(cfg_pick, 'approach_height', 50.0))
    app_h_place = float(getattr(cfg_place, 'approach_height', 50.0))
    num_cart_steps = max(8, steps // 2)

    # 1. Pre-Pick & Cartesian Descent/Lift
    pre_pick_w, pre_pick_b, pre_pick_q = resolve_target_pre_pose(
        pick_rid, cfg_pick, pick_xyz_w, yaw_deg=pick_yaw, approach_height_mm=app_h_pick, seed_q_deg=start_pick_q
    )
    cart_descent_pick = compute_cartesian_linear_path(pre_pick_q, pick_xyz_b, cfg_pick, num_steps=num_cart_steps)
    if cart_descent_pick is None:
        cart_descent_pick = generate_interpolated_path(pre_pick_q, plan.pick.q_deg, steps=num_cart_steps)
    actual_pick_q = cart_descent_pick[-1].copy()

    cart_lift_pick = compute_cartesian_linear_path(actual_pick_q, pre_pick_b, cfg_pick, num_steps=num_cart_steps)
    if cart_lift_pick is None:
        cart_lift_pick = generate_interpolated_path(actual_pick_q, pre_pick_q, steps=num_cart_steps)

    # 2. Pre-Place & Cartesian Descent/Retract
    place_pre_h_q_temp = getattr(plan, 'place_pre_handover_q', None)
    if place_pre_h_q_temp is None:
        place_pre_h_q_temp = getattr(plan, 'place_handover_q', None)
    seed_place = place_pre_h_q_temp if plan.needs_handover else actual_pick_q

    pre_place_w, pre_place_b, pre_place_q = resolve_target_pre_pose(
        place_rid, cfg_place, place_xyz_w, yaw_deg=place_yaw, approach_height_mm=app_h_place, seed_q_deg=seed_place
    )
    cart_descent_place = compute_cartesian_linear_path(pre_place_q, place_xyz_b, cfg_place, num_steps=num_cart_steps)
    if cart_descent_place is None:
        cart_descent_place = generate_interpolated_path(pre_place_q, plan.place.q_deg, steps=num_cart_steps)
    actual_place_q = cart_descent_place[-1].copy()

    cart_retract_place = compute_cartesian_linear_path(actual_place_q, pre_place_b, cfg_place, num_steps=num_cart_steps)
    if cart_retract_place is None:
        cart_retract_place = generate_interpolated_path(actual_place_q, pre_place_q, steps=num_cart_steps)

    if not plan.needs_handover:
        home_q = np.array(getattr(cfg_pick, 'home_deg', [0.0]*cfg_pick.num_joints), dtype=float)

        # Phase 1: Transit to Pre-Pick + Cartesian Straight-Line Descent to Pick
        path_approach_pre = generate_interpolated_path(start_pick_q, pre_pick_q, steps=steps)
        p1_steps = []
        for q in path_approach_pre:
            p1_steps.append(Stage1_5Step("single_pre_pick", q.copy(), q.copy(), float("inf"), True, f"{pick_rid} approaching Pre-Pick (+{app_h_pick:.0f}mm)"))
        for q in cart_descent_pick:
            p1_steps.append(Stage1_5Step("single_cartesian_descent", q.copy(), q.copy(), float("inf"), True, f"{pick_rid} Cartesian vertical descent to Pick (-Z)"))

        # Phase 7: Cartesian Lift to Pre-Pick + Transfer to Pre-Place + Cartesian Descent to Place
        path_transfer = generate_interpolated_path(pre_pick_q, pre_place_q, steps=steps)
        p7_steps = []
        for q in cart_lift_pick:
            p7_steps.append(Stage1_5Step("single_cartesian_lift", q.copy(), q.copy(), float("inf"), True, f"{pick_rid} Cartesian vertical lift (+{app_h_pick:.0f}mm)"))
        for q in path_transfer:
            p7_steps.append(Stage1_5Step("single_transfer_pre_place", q.copy(), q.copy(), float("inf"), True, f"{pick_rid} transferring to Pre-Place"))
        for q in cart_descent_place:
            p7_steps.append(Stage1_5Step("single_cartesian_descent_place", q.copy(), q.copy(), float("inf"), True, f"{pick_rid} Cartesian vertical descent to Place (-Z)"))

        # Phase 9: Cartesian Retract to Pre-Place + Return Home
        path_home = generate_interpolated_path(pre_place_q, home_q, steps=steps)
        p9_steps = []
        for q in cart_retract_place:
            p9_steps.append(Stage1_5Step("single_cartesian_retract", q.copy(), q.copy(), float("inf"), True, f"{pick_rid} Cartesian vertical retraction (+{app_h_place:.0f}mm)"))
        for q in path_home:
            p9_steps.append(Stage1_5Step("single_home", q.copy(), q.copy(), float("inf"), True, f"{pick_rid} returning Home"))

        return Stage1_5TrajectoryResult(
            needs_handover=False,
            pick_robot=pick_rid,
            place_robot=place_rid,
            phase1_simultaneous=p1_steps,
            phase3_pick_to_pre_handover=[],
            phase4_advance_to_handover=[],
            phase6_retract_to_pre_handover=[],
            phase7_handover_to_place=p7_steps,
            phase9_simultaneous_home=p9_steps,
            min_overall_clearance_mm=float("inf"),
            all_phases_safe=True,
            collision_avoidance_applied=False,
        )

    # Multi-Robot Handover
    pick_h_q = plan.pick_handover_q
    place_h_q = plan.place_handover_q
    pick_pre_h_q = plan.pick_pre_handover_q if plan.pick_pre_handover_q is not None else pick_h_q
    place_pre_h_q = plan.place_pre_handover_q if plan.place_pre_handover_q is not None else place_h_q
    pick_target_q = actual_pick_q
    place_target_q = actual_place_q

    # ─────────────────────────────────────────────────────────────
    # PHASE 1: INDEPENDENT TASK MOTION WITH COLLISION-AWARE SCHEDULING
    # Pick robot -> Pre-Pick & Cartesian Descent, Place robot -> Pre-Handover
    # ─────────────────────────────────────────────────────────────
    collision_avoidance_applied = False
    phase1_pre_steps, min_p1_clearance, p1_avoided = schedule_collision_aware_motion(
        start_pick_q=start_pick_q,
        target_pick_q=pre_pick_q,
        start_place_q=start_place_q,
        target_place_q=place_pre_h_q,
        cfg_pick=cfg_pick,
        cfg_place=cfg_place,
        safe_distance=safe_distance,
        steps=steps,
        pick_rid=pick_rid,
        place_rid=place_rid,
        phase_name="phase1_simultaneous",
        desc_concurrent=f"Concurrent: {pick_rid}->Pre-Pick, {place_rid}->Pre-Handover",
        desc_serialized_pick_first="Serialized (Pre-Pick first)",
        desc_serialized_place_first="Serialized (Pre-Handover first)",
    )
    if p1_avoided:
        collision_avoidance_applied = True

    phase1_steps = list(phase1_pre_steps)
    for q_p in cart_descent_pick:
        c = compute_min_body_clearance(q_p, cfg_pick, place_pre_h_q, cfg_place, exclude_tcp=False)
        if c < min_p1_clearance:
            min_p1_clearance = c
        phase1_steps.append(
            Stage1_5Step(
                name="phase1_cartesian_descent",
                pick_q=q_p,
                place_q=place_pre_h_q,
                min_clearance_mm=c,
                is_safe=c >= safe_distance,
                description=f"{pick_rid} Cartesian vertical descent to Pick ({place_rid} waiting at Pre-Handover)",
            )
        )

    # ─────────────────────────────────────────────────────────────
    # PHASE 3: CARTESIAN LIFT TO PRE-PICK & TRANSIT TO PRE-HANDOVER
    # Robot 1: actual_pick_q -> pre_pick_q (Cartesian Lift) -> pick_pre_h_q
    # Robot 2: waiting stationed at place_pre_h_q
    # ─────────────────────────────────────────────────────────────
    phase3_steps = []
    min_p3_clearance = float("inf")

    # 1. Cartesian Lift
    for q_p in cart_lift_pick:
        c = compute_min_body_clearance(q_p, cfg_pick, place_pre_h_q, cfg_place, exclude_tcp=False)
        if c < min_p3_clearance:
            min_p3_clearance = c
        phase3_steps.append(
            Stage1_5Step(
                name="phase3_cartesian_lift",
                pick_q=q_p,
                place_q=place_pre_h_q,
                min_clearance_mm=c,
                is_safe=c >= safe_distance,
                description=f"{pick_rid} Cartesian vertical lift (+{app_h_pick:.0f}mm)",
            )
        )

    # 2. Joint space transit from Pre-Pick to Pre-Handover
    raw_p3_pick = generate_interpolated_path(pre_pick_q, pick_pre_h_q, steps=steps)
    for q_p in raw_p3_pick:
        c = compute_min_body_clearance(q_p, cfg_pick, place_pre_h_q, cfg_place, exclude_tcp=False)
        if c < min_p3_clearance:
            min_p3_clearance = c
        phase3_steps.append(
            Stage1_5Step(
                name="phase3_pick_to_pre_handover",
                pick_q=q_p,
                place_q=place_pre_h_q,
                min_clearance_mm=c,
                is_safe=c >= safe_distance,
                description=f"{pick_rid} moving to Pre-Handover ({place_rid} waiting at Pre-Handover)",
            )
        )

    # ─────────────────────────────────────────────────────────────
    # PHASE 4: COORDINATED ADVANCE ALONG CARTESIAN TCP Z CORRIDORS
    # Robot 1: pick_pre_h_q -> pick_h_q
    # Robot 2: place_pre_h_q -> place_h_q
    # ─────────────────────────────────────────────────────────────
    p_pick_H, z_pick_H, _, _ = get_tcp_world_pose(pick_h_q, cfg_pick, in_degrees=True)
    p_pick_pre, _, _, _ = get_tcp_world_pose(pick_pre_h_q, cfg_pick, in_degrees=True)
    standoff_pick = float(np.linalg.norm(p_pick_H - p_pick_pre))
    if standoff_pick < 1.0:
        standoff_pick = float(h_cfg.get('handover', {}).get('standoff_distance_mm', 50.0))

    p_place_H, z_place_H, _, _ = get_tcp_world_pose(place_h_q, cfg_place, in_degrees=True)
    p_place_pre, _, _, _ = get_tcp_world_pose(place_pre_h_q, cfg_place, in_degrees=True)
    standoff_place = float(np.linalg.norm(p_place_H - p_place_pre))
    if standoff_place < 1.0:
        standoff_place = float(h_cfg.get('handover', {}).get('standoff_distance_mm', 50.0))

    corridor_required = getattr(plan, "use_auto_handover", None) is not None
    if corridor_required:
        ok_pick, raw_p4_pick, msg_pick = generate_cartesian_corridor_trajectory(
            pick_rid, cfg_pick, pick_h_q, q_pre_deg=pick_pre_h_q, standoff_mm=standoff_pick, steps=steps
        )
        ok_place, raw_p4_place, msg_place = generate_cartesian_corridor_trajectory(
            place_rid, cfg_place, place_h_q, q_pre_deg=place_pre_h_q, standoff_mm=standoff_place, steps=steps
        )
        if not ok_pick:
            raw_p4_pick = generate_interpolated_path(pick_pre_h_q, pick_h_q, steps=steps)
        if not ok_place:
            raw_p4_place = generate_interpolated_path(place_pre_h_q, place_h_q, steps=steps)
        corridor_valid = ok_pick and ok_place
    else:
        raw_p4_pick = generate_interpolated_path(pick_pre_h_q, pick_h_q, steps=steps)
        raw_p4_place = generate_interpolated_path(place_pre_h_q, place_h_q, steps=steps)
        corridor_valid = True
    phase4_steps = []
    min_p4_clearance = float("inf")

    for i, (q_p, q_pl) in enumerate(zip(raw_p4_pick, raw_p4_place)):
        c = compute_min_body_clearance(q_p, cfg_pick, q_pl, cfg_place, exclude_tcp=True)
        if c < min_p4_clearance:
            min_p4_clearance = c
        is_step_safe = (c >= safe_distance) and corridor_valid
        phase4_steps.append(
            Stage1_5Step(
                name="phase4_advance_to_handover",
                pick_q=q_p,
                place_q=q_pl,
                min_clearance_mm=c,
                is_safe=is_step_safe,
                description=f"Advance: {pick_rid} & {place_rid} along Cartesian TCP Z corridors",
            )
        )

    # ─────────────────────────────────────────────────────────────
    # PHASE 6: COORDINATED RETRACT ALONG REVERSE CARTESIAN CORRIDORS
    # Robot 1: pick_h_q -> pick_pre_h_q
    # Robot 2: place_h_q -> place_pre_h_q
    # ─────────────────────────────────────────────────────────────
    raw_p6_pick = list(reversed(raw_p4_pick))
    raw_p6_place = list(reversed(raw_p4_place))
    phase6_steps = []
    min_p6_clearance = float("inf")

    for i, (q_p, q_pl) in enumerate(zip(raw_p6_pick, raw_p6_place)):
        c = compute_min_body_clearance(q_p, cfg_pick, q_pl, cfg_place, exclude_tcp=True)
        if c < min_p6_clearance:
            min_p6_clearance = c
        is_step_safe = (c >= safe_distance) and corridor_valid
        phase6_steps.append(
            Stage1_5Step(
                name="phase6_retract_to_pre_handover",
                pick_q=q_p,
                place_q=q_pl,
                min_clearance_mm=c,
                is_safe=is_step_safe,
                description=f"Retract: {pick_rid} & {place_rid} backing away along reverse Cartesian corridors",
            )
        )

    # ─────────────────────────────────────────────────────────────
    # PHASE 7: PRE-HANDOVER TO PRE-PLACE & CARTESIAN DESCENT TO PLACE TARGET
    # Robot 2: place_pre_h_q -> pre_place_q -> actual_place_q
    # Robot 1: waiting stationed at pick_pre_h_q (or designated safe waiting pose)
    # ─────────────────────────────────────────────────────────────
    raw_p7_place = generate_interpolated_path(place_pre_h_q, pre_place_q, steps=steps)
    phase7_steps = []
    min_p7_clearance = float("inf")

    # Check if keeping pick_robot waiting at pick_pre_h_q is safe
    p7_stationed_safe = True
    for q_pl in raw_p7_place:
        c = compute_min_body_clearance(pick_pre_h_q, cfg_pick, q_pl, cfg_place, exclude_tcp=False)
        if c < safe_distance:
            p7_stationed_safe = False
            break

    home_pick = np.zeros(cfg_pick.num_joints)
    if p7_stationed_safe:
        pick_end_q_p7 = pick_pre_h_q
        for q_pl in raw_p7_place:
            c = compute_min_body_clearance(pick_pre_h_q, cfg_pick, q_pl, cfg_place, exclude_tcp=False)
            if c < min_p7_clearance:
                min_p7_clearance = c
            phase7_steps.append(
                Stage1_5Step(
                    name="phase7_handover_to_pre_place",
                    pick_q=pick_pre_h_q,
                    place_q=q_pl,
                    min_clearance_mm=c,
                    is_safe=c >= safe_distance,
                    description=f"{place_rid} moving from Pre-Handover to Pre-Place ({pick_rid} waiting at Pre-Handover)",
                )
            )
    else:
        # Move pick_robot to validated safe waiting pose (home_pick) first
        raw_p7_pick_retire = generate_interpolated_path(pick_pre_h_q, home_pick, steps=steps)
        collision_avoidance_applied = True
        pick_end_q_p7 = home_pick

        # Step 1: pick_robot retires to safe waiting pose while place_robot waits at pre-handover
        for q_p in raw_p7_pick_retire:
            c = compute_min_body_clearance(q_p, cfg_pick, place_pre_h_q, cfg_place, exclude_tcp=False)
            if c < min_p7_clearance:
                min_p7_clearance = c
            phase7_steps.append(
                Stage1_5Step(
                    name="phase7_pick_retire",
                    pick_q=q_p,
                    place_q=place_pre_h_q,
                    min_clearance_mm=c,
                    is_safe=c >= safe_distance,
                    description=f"{pick_rid} moving to safe waiting pose ({place_rid} waiting at Pre-Handover)",
                )
            )

        # Step 2: place_robot moves to pre-place while pick_robot waits at safe waiting pose
        for q_pl in raw_p7_place:
            c = compute_min_body_clearance(home_pick, cfg_pick, q_pl, cfg_place, exclude_tcp=False)
            if c < min_p7_clearance:
                min_p7_clearance = c
            phase7_steps.append(
                Stage1_5Step(
                    name="phase7_handover_to_pre_place",
                    pick_q=home_pick,
                    place_q=q_pl,
                    min_clearance_mm=c,
                    is_safe=c >= safe_distance,
                    description=f"{place_rid} moving to Pre-Place ({pick_rid} waiting at safe pose)",
                )
            )

    # Cartesian Descent to Place Target
    for q_pl in cart_descent_place:
        c = compute_min_body_clearance(pick_end_q_p7, cfg_pick, q_pl, cfg_place, exclude_tcp=False)
        if c < min_p7_clearance:
            min_p7_clearance = c
        phase7_steps.append(
            Stage1_5Step(
                name="phase7_cartesian_descent_place",
                pick_q=pick_end_q_p7,
                place_q=q_pl,
                min_clearance_mm=c,
                is_safe=c >= safe_distance,
                description=f"{place_rid} Cartesian vertical descent to Place (-Z)",
            )
        )

    # ─────────────────────────────────────────────────────────────
    # PHASE 9: CARTESIAN RETRACT TO PRE-PLACE & RETURN HOME
    # Robot 1: pick_end_q_p7 -> home_pick
    # Robot 2: actual_place_q -> pre_place_q -> home_place
    # ─────────────────────────────────────────────────────────────
    phase9_steps = []
    min_p9_clearance = float("inf")

    # 1. Cartesian Retract
    for q_pl in cart_retract_place:
        c = compute_min_body_clearance(pick_end_q_p7, cfg_pick, q_pl, cfg_place, exclude_tcp=False)
        if c < min_p9_clearance:
            min_p9_clearance = c
        phase9_steps.append(
            Stage1_5Step(
                name="phase9_cartesian_retract",
                pick_q=pick_end_q_p7,
                place_q=q_pl,
                min_clearance_mm=c,
                is_safe=c >= safe_distance,
                description=f"{place_rid} Cartesian vertical retract (+{app_h_place:.0f}mm)",
            )
        )

    # 2. Both robots return to Home
    home_place = np.zeros(cfg_place.num_joints)
    p9_ret_steps, min_p9_ret_c, p9_avoided = schedule_collision_aware_motion(
        start_pick_q=pick_end_q_p7,
        target_pick_q=home_pick,
        start_place_q=pre_place_q,
        target_place_q=home_place,
        cfg_pick=cfg_pick,
        cfg_place=cfg_place,
        safe_distance=safe_distance,
        steps=steps,
        pick_rid=pick_rid,
        place_rid=place_rid,
        phase_name="phase9_simultaneous_home",
        desc_concurrent=f"Concurrent return home: {pick_rid} & {place_rid}",
        desc_serialized_pick_first="Serialized return home (Pick first)",
        desc_serialized_place_first="Serialized return home (Place first)",
    )
    if p9_avoided:
        collision_avoidance_applied = True
    if min_p9_ret_c < min_p9_clearance:
        min_p9_clearance = min_p9_ret_c
    phase9_steps.extend(p9_ret_steps)

    overall_min = min(
        min_p1_clearance,
        min_p3_clearance,
        min_p4_clearance,
        min_p6_clearance,
        min_p7_clearance,
        min_p9_clearance,
    )
    all_safe = all(
        s.is_safe
        for s in (
            phase1_steps
            + phase3_steps
            + phase4_steps
            + phase6_steps
            + phase7_steps
            + phase9_steps
        )
    )

    return Stage1_5TrajectoryResult(
        needs_handover=True,
        pick_robot=pick_rid,
        place_robot=place_rid,
        phase1_simultaneous=phase1_steps,
        phase3_pick_to_pre_handover=phase3_steps,
        phase4_advance_to_handover=phase4_steps,
        phase6_retract_to_pre_handover=phase6_steps,
        phase7_handover_to_place=phase7_steps,
        phase9_simultaneous_home=phase9_steps,
        min_overall_clearance_mm=overall_min,
        all_phases_safe=all_safe,
        collision_avoidance_applied=collision_avoidance_applied,
    )


def validate_execution_safety_gate(
    traj_result: Optional[Stage1_5TrajectoryResult],
    robots: Dict[str, Any],
    plan: Optional[Any] = None,
) -> Tuple[bool, str]:
    """P0.4: Mandatory programmatic execution safety gate.
    
    Verifies that:
      1. Trajectory result is present and valid.
      2. If handover plan is required, trajectory result indicates needs_handover=True.
      3. traj_result.all_phases_safe is True.
      4. Trajectory contains valid, non-empty required phases.
      5. Every single trajectory step in all phases has is_safe=True.
      6. All joint configurations have expected dimensions and satisfy joint limits.
      
    Returns
    -------
    (is_safe, message): tuple of boolean pass/fail and descriptive status/failure string.
    """
    if traj_result is None:
        return False, "Execution rejected: trajectory result is None."

    if plan is not None and getattr(plan, "needs_handover", False):
        if not traj_result.needs_handover:
            return False, "Execution rejected: plan requires handover but trajectory needs_handover is False."

    if not traj_result.all_phases_safe:
        return False, f"Execution rejected: all_phases_safe is False (min clearance: {traj_result.min_overall_clearance_mm:.1f} mm)."

    pick_rid = traj_result.pick_robot
    place_rid = traj_result.place_robot

    if pick_rid not in robots or place_rid not in robots:
        return False, f"Execution rejected: robot configs for {pick_rid} or {place_rid} not found."

    cfg_pick = robots[pick_rid]
    cfg_place = robots[place_rid]

    # Required phase definitions
    if traj_result.needs_handover:
        phases = [
            ("phase1_simultaneous", traj_result.phase1_simultaneous),
            ("phase3_pick_to_pre_handover", traj_result.phase3_pick_to_pre_handover),
            ("phase4_advance_to_handover", traj_result.phase4_advance_to_handover),
            ("phase6_retract_to_pre_handover", traj_result.phase6_retract_to_pre_handover),
            ("phase7_handover_to_place", traj_result.phase7_handover_to_place),
            ("phase9_simultaneous_home", traj_result.phase9_simultaneous_home),
        ]
    else:
        phases = [
            ("phase1_simultaneous", traj_result.phase1_simultaneous),
            ("phase7_handover_to_place", traj_result.phase7_handover_to_place),
            ("phase9_simultaneous_home", traj_result.phase9_simultaneous_home),
        ]

    for phase_name, steps in phases:
        if not steps or len(steps) == 0:
            return False, f"Execution rejected: required phase '{phase_name}' is empty."

        for step_idx, step in enumerate(steps):
            if not step.is_safe:
                return (
                    False,
                    f"Execution rejected: unsafe step at {phase_name}[{step_idx}] "
                    f"(clearance={step.min_clearance_mm:.1f} mm, desc='{step.description}').",
                )

            # Check joint dimensions
            if len(step.pick_q) != cfg_pick.num_joints:
                return (
                    False,
                    f"Execution rejected: {pick_rid} joint dimension mismatch in {phase_name}[{step_idx}] "
                    f"(expected {cfg_pick.num_joints}, got {len(step.pick_q)}).",
                )
            if len(step.place_q) != cfg_place.num_joints:
                return (
                    False,
                    f"Execution rejected: {place_rid} joint dimension mismatch in {phase_name}[{step_idx}] "
                    f"(expected {cfg_place.num_joints}, got {len(step.place_q)}).",
                )

            # Check joint limits (with small 1e-3 deg tolerance for numerical precision)
            q_min_pick = cfg_pick.joint_limits_deg[:, 0] - 1e-3
            q_max_pick = cfg_pick.joint_limits_deg[:, 1] + 1e-3
            if np.any(step.pick_q < q_min_pick) or np.any(step.pick_q > q_max_pick):
                return (
                    False,
                    f"Execution rejected: {pick_rid} joint limit violation in {phase_name}[{step_idx}] "
                    f"q={np.round(step.pick_q, 2)} deg.",
                )

            q_min_place = cfg_place.joint_limits_deg[:, 0] - 1e-3
            q_max_place = cfg_place.joint_limits_deg[:, 1] + 1e-3
            if np.any(step.place_q < q_min_place) or np.any(step.place_q > q_max_place):
                return (
                    False,
                    f"Execution rejected: {place_rid} joint limit violation in {phase_name}[{step_idx}] "
                    f"q={np.round(step.place_q, 2)} deg.",
                )

    return True, f"Safety gate passed: all phases safe, verified clearances (min: {traj_result.min_overall_clearance_mm:.1f} mm)."
