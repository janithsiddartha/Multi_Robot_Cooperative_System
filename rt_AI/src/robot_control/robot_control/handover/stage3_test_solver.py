"""Stage 3 Test - Experimental Kinematic Handover & Asymmetric Corridor Solver.

Testing Solver Variants:
1. Candidate Selection:
   - Primary: TOP_OF_WORKSPACE (sorted descending by Z, no midpoint balance filtering).
   - Fallback: CENTER_OF_WORKSPACE (closest to centroid of common 3D reachability).
2. Orientation Optimization:
   - Position Coincidence: ||pA - pH|| <= 1.0 mm, ||pB - pH|| <= 1.0 mm.
   - Anti-Parallel Approach: zA . zB <= -1 + eps_z (<= 15 deg deviation).
   - Orthogonal Finger Alignment: |xA . xB| <= eps_x.
     Robot 1 and Robot 3 have a wrist rotation joint before the gripper;
     when pairing with Robot 2 (5-DOF), Robot 1 or 3's wrist joint is
     aligned to match Robot 2's finger orientation perpendicularly.
3. Pre-Handover Standoff:
   - p_pre = pH - d * z (standoff d in [50, 70] mm along local TCP Z).
4. Approach & Retraction Classification:
   - DUAL_SYNCHRONIZED: Both robots can follow straight Cartesian TCP Z corridors.
   - ASYMMETRIC_A_FIRST: Robot A cannot follow straight corridor; it moves to pH FIRST
     while Robot B waits at pre-handover. Then Robot B advances along its straight corridor.
     Retract: Robot B retreats along its corridor FIRST, then Robot A departs.
   - ASYMMETRIC_B_FIRST: Robot B cannot follow straight corridor; it moves to pH FIRST
     while Robot A waits at pre-handover. Then Robot A advances along its straight corridor.
     Retract: Robot A retreats along its corridor FIRST, then Robot B departs.
   - REJECTED: Neither robot can follow straight corridor; triggers fallback to CENTER_OF_WORKSPACE.
5. Caching:
   - Caches to `auto_handover_poses_test.json` so production cache is unaffected.
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
from robot_control.handover.stage3_handover_solver import (
    get_tcp_world_pose,
    matrix_to_quaternion,
    solve_coupled_handover_orientation,
)
from robot_control.handover.trajectory_cache import compute_config_fingerprint
from robot_control.handover.workspace import PositionWorkspaceComputer
from robot_control.kinematics.fk_math import fk
from robot_control.kinematics.ik_solver import ik


@dataclass
class AutoHandoverTestPose:
    """Auto-calculated experimental handover & pre-handover pose."""

    pair_name: str                      # e.g. 'robot1_robot2'
    handover_position_world: np.ndarray # [X, Y, Z] in world frame (mm)
    standoff_distance_mm: float         # e.g. 50.0 mm
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
    dot_z: float = -1.0                 # zA . zB approach axis alignment
    dot_x: float = 0.0                  # |xA . xB| finger orthogonality
    approach_angle_dev_deg: float = 0.0 # Angular deviation from anti-parallel (deg)
    approach_mode: str = "DUAL_SYNCHRONIZED" # DUAL_SYNCHRONIZED, ASYMMETRIC_A_FIRST, ASYMMETRIC_B_FIRST
    candidate_origin: str = "TOP_OF_WORKSPACE" # TOP_OF_WORKSPACE, CENTER_OF_WORKSPACE
    corridor_a_valid: bool = True
    corridor_b_valid: bool = True

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


def resolve_test_handover_cache_path() -> str:
    """Resolve filepath for auto_handover_poses_test.json in config directory."""
    cur = os.path.abspath(os.getcwd())
    while cur != '/':
        cand = os.path.join(cur, 'src', 'robot_control', 'config')
        if os.path.isdir(cand):
            return os.path.join(cand, 'auto_handover_poses_test.json')
        cur = os.path.dirname(cur)
    return os.path.join(os.getcwd(), 'auto_handover_poses_test.json')


def serialize_test_auto_handover(pose: AutoHandoverTestPose) -> dict:
    """Serialize an AutoHandoverTestPose dataclass to a JSON dict."""
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
        "approach_mode": pose.approach_mode,
        "candidate_origin": pose.candidate_origin,
        "corridor_a_valid": pose.corridor_a_valid,
        "corridor_b_valid": pose.corridor_b_valid,
    }


def deserialize_test_auto_handover(d: dict) -> AutoHandoverTestPose:
    """Deserialize dict to AutoHandoverTestPose."""
    return AutoHandoverTestPose(
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
        approach_mode=str(d.get("approach_mode", "DUAL_SYNCHRONIZED")),
        candidate_origin=str(d.get("candidate_origin", "TOP_OF_WORKSPACE")),
        corridor_a_valid=bool(d.get("corridor_a_valid", True)),
        corridor_b_valid=bool(d.get("corridor_b_valid", True)),
    )


def load_test_handover_cache(cache_path: str, config_hash: str) -> Optional[Dict[str, AutoHandoverTestPose]]:
    """Load cached test handover poses for a config hash."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") not in ("2.0-test", "2.0"):
            return None
        fp_data = data.get("fingerprints", {}).get(config_hash, {})
        pairs_data = fp_data.get("pairs", {})
        if not pairs_data:
            return None
        res = {}
        for pair_name, pd in pairs_data.items():
            res[pair_name] = deserialize_test_auto_handover(pd)
        return res if len(res) >= 1 else None
    except Exception:
        return None


def save_test_handover_cache(cache_path: str, config_hash: str, poses: Dict[str, AutoHandoverTestPose]) -> bool:
    """Save test handover poses atomically to JSON cache."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        data = {"version": "2.0-test", "fingerprints": {}}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict) and "fingerprints" in loaded:
                        data = loaded
                        data["version"] = "2.0-test"
            except Exception:
                pass

        if config_hash not in data["fingerprints"]:
            data["fingerprints"][config_hash] = {}

        data["fingerprints"][config_hash] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pairs": {pname: serialize_test_auto_handover(pose) for pname, pose in poses.items()}
        }

        tmp_file = cache_path + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_file, cache_path)
        return True
    except Exception:
        return False


def get_test_candidate_points(
    common_points: np.ndarray,
    strategy: str = "TOP_OF_WORKSPACE",
    cfgA: Optional[object] = None,
    cfgB: Optional[object] = None,
) -> np.ndarray:
    """Get sorted candidate points based on strategy:
    
    1. 'TOP_OF_WORKSPACE': Sorted strictly descending by Z (high Z first).
       No midpoint balance filtering or low-height cutoffs.
    2. 'CENTER_OF_WORKSPACE': Sorted strictly by proximity to common workspace centroid.
    """
    if len(common_points) == 0:
        return common_points

    pts = common_points
    if cfgA is not None and cfgB is not None:
        bA = np.asarray(cfgA.world_base_xyz, dtype=float)
        bB = np.asarray(cfgB.world_base_xyz, dtype=float)
        reachA = (sum(abs(float(v)) for v in cfgA.a) + sum(abs(float(v)) for v in cfgA.d)) * 0.90
        reachB = (sum(abs(float(v)) for v in cfgB.a) + sum(abs(float(v)) for v in cfgB.d)) * 0.90
        dA = np.linalg.norm(pts - bA, axis=1)
        dB = np.linalg.norm(pts - bB, axis=1)
        mask = (dA <= reachA) & (dB <= reachB)
        if np.any(mask):
            pts = pts[mask]

    if strategy == "TOP_OF_WORKSPACE":
        # Sort descending by Z
        order = np.argsort(-pts[:, 2])
        return pts[order]
    elif strategy == "CENTER_OF_WORKSPACE":
        centroid = np.mean(pts, axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        order = np.argsort(dists)
        return pts[order]
    else:
        return pts


def optimize_wrist_for_orthogonal_fingers(
    robot_id: str,
    cfg: object,
    q_deg: np.ndarray,
    z_target: np.ndarray,
    x_other: np.ndarray,
) -> np.ndarray:
    """Adjust the wrist roll joint of Robot 1 or Robot 3 to make fingers orthogonal to the other robot.
    
    Robot 1 has joint index 4 as wrist roll before gripper.
    Robot 3 has joint index 5 as wrist roll before gripper.
    """
    wrist_joint_idx = None
    if robot_id == "robot1" and cfg.num_joints >= 5:
        wrist_joint_idx = 4
    elif robot_id == "robot3" and cfg.num_joints >= 6:
        wrist_joint_idx = 5

    if wrist_joint_idx is None:
        return q_deg

    q_best = q_deg.copy()
    min_dot_x = 1e9

    # Sweep wrist angles within joint limits
    min_limit, max_limit = cfg.joint_limits_deg[wrist_joint_idx]
    candidate_angles = np.linspace(min_limit, max_limit, 37)

    for ang in candidate_angles:
        q_cand = q_best.copy()
        q_cand[wrist_joint_idx] = ang
        p_w, z_w, x_w, _ = get_tcp_world_pose(q_cand, cfg, in_degrees=True)
        dot_x = abs(float(np.dot(x_w, x_other)))
        if dot_x < min_dot_x:
            min_dot_x = dot_x
            q_best = q_cand.copy()

    return q_best


def solve_pair_test_handover(
    robot_a_id: str,
    robot_b_id: str,
    robots: Dict[str, Any],
    common_points: np.ndarray,
    handover_cfg: Optional[dict] = None,
    preferred_strategy: str = "TOP_OF_WORKSPACE",
    standoff_mm: float = 50.0,
    max_candidates: int = 50,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Optional[AutoHandoverTestPose]:
    """Solve optimal handover and pre-handover pose for testing pipeline.
    
    Sequence:
    1. Try preferred_strategy (default: TOP_OF_WORKSPACE).
    2. For each candidate point:
       - Solve coupled IK for position coincidence and opposite approach.
       - Adjust wrist joint on Robot 1/3 to match finger orthogonality for Robot 2.
       - Check Cartesian corridors for Robot A and Robot B.
       - If both OK -> DUAL_SYNCHRONIZED.
       - If one OK -> ASYMMETRIC_A_FIRST or ASYMMETRIC_B_FIRST.
       - If neither -> Reject candidate.
    3. If all candidates in TOP_OF_WORKSPACE fail -> fallback to CENTER_OF_WORKSPACE.
    """
    cfgA = robots[robot_a_id]
    cfgB = robots[robot_b_id]

    h_cfg = handover_cfg or getattr(cfgA, 'handover_cfg', None) or getattr(cfgB, 'handover_cfg', None) or {}
    tolerances = h_cfg.get('pose_tolerances', {})
    pos_tol_mm = float(tolerances.get('position_mm', 1.0))
    ori_tol_deg = float(tolerances.get('orientation_deg', 15.0))
    safe_distance = float(h_cfg.get('handover', {}).get('safe_distance', 50.0))

    strategies_to_try = [preferred_strategy]
    if preferred_strategy == "TOP_OF_WORKSPACE":
        strategies_to_try.append("CENTER_OF_WORKSPACE")

    for current_strategy in strategies_to_try:
        candidates = get_test_candidate_points(common_points, strategy=current_strategy, cfgA=cfgA, cfgB=cfgB)
        total_eval = min(len(candidates), max_candidates)

        if progress_callback:
            progress_callback(
                f"[{robot_a_id}<->{robot_b_id}] Searching {total_eval} candidate points using strategy '{current_strategy}'..."
            )

        for idx, p_H in enumerate(candidates[:total_eval]):
            if progress_callback and idx % 5 == 0:
                progress_callback(
                    f"[{robot_a_id}<->{robot_b_id}] [{current_strategy}] Evaluating candidate {idx+1}/{total_eval} at Z={p_H[2]:.1f} mm..."
                )

            # Solve coupled orientation
            coupled_res = solve_coupled_handover_orientation(
                robot_a_id=robot_a_id,
                robot_b_id=robot_b_id,
                p_H_world=p_H,
                robots=robots,
                pos_tol_mm=pos_tol_mm,
                ori_tol_deg=ori_tol_deg,
                min_clearance_mm=40.0,
                max_opt_iterations=45,
                solver_method="analytical",
            )
            if coupled_res is None:
                continue

            qA_deg, qB_deg, errA, errB, dot_z, dot_x, h_clr = coupled_res

            # Recompute world pose
            pA_w, zA_w, xA_w, _ = get_tcp_world_pose(qA_deg, cfgA, in_degrees=True)
            pB_w, zB_w, xB_w, _ = get_tcp_world_pose(qB_deg, cfgB, in_degrees=True)

            # If Robot 2 is involved, adjust wrist roll of Robot 1 or Robot 3 to ensure orthogonal fingers
            if robot_a_id == "robot2" and robot_b_id in ["robot1", "robot3"]:
                qB_deg = optimize_wrist_for_orthogonal_fingers(robot_b_id, cfgB, qB_deg, zB_w, xA_w)
                pB_w, zB_w, xB_w, _ = get_tcp_world_pose(qB_deg, cfgB, in_degrees=True)
                dot_x = abs(float(np.dot(xA_w, xB_w)))
            elif robot_b_id == "robot2" and robot_a_id in ["robot1", "robot3"]:
                qA_deg = optimize_wrist_for_orthogonal_fingers(robot_a_id, cfgA, qA_deg, zA_w, xB_w)
                pA_w, zA_w, xA_w, _ = get_tcp_world_pose(qA_deg, cfgA, in_degrees=True)
                dot_x = abs(float(np.dot(xA_w, xB_w)))

            # Check pre-handover corridors along actual TCP Z axes
            standoff = standoff_mm
            okA, trajA, msgA = generate_cartesian_corridor_trajectory(
                robot_a_id, cfgA, qA_deg, standoff_mm=standoff, steps=15,
                pos_tol_mm=pos_tol_mm, ori_tol_deg=ori_tol_deg
            )
            okB, trajB, msgB = generate_cartesian_corridor_trajectory(
                robot_b_id, cfgB, qB_deg, standoff_mm=standoff, steps=15,
                pos_tol_mm=pos_tol_mm, ori_tol_deg=ori_tol_deg
            )

            # Pre-handover joint angles
            if okA and len(trajA) > 0:
                qA_pre = trajA[0].copy()
            else:
                p_pre_A = p_H - standoff * zA_w
                res_pre_A = ik(world_to_base(p_pre_A, cfgA), cfgA, current_q=np.deg2rad(qA_deg), num_random_starts=5)
                qA_pre = np.rad2deg(res_pre_A['q']) if res_pre_A['success'] else qA_deg.copy()

            if okB and len(trajB) > 0:
                qB_pre = trajB[0].copy()
            else:
                p_pre_B = p_H - standoff * zB_w
                res_pre_B = ik(world_to_base(p_pre_B, cfgB), cfgB, current_q=np.deg2rad(qB_deg), num_random_starts=5)
                qB_pre = np.rad2deg(res_pre_B['q']) if res_pre_B['success'] else qB_deg.copy()

            # Check joint limits
            if np.any(qA_pre < cfgA.joint_limits_deg[:, 0] - 1e-3) or np.any(qA_pre > cfgA.joint_limits_deg[:, 1] + 1e-3):
                continue
            if np.any(qB_pre < cfgB.joint_limits_deg[:, 0] - 1e-3) or np.any(qB_pre > cfgB.joint_limits_deg[:, 1] + 1e-3):
                continue

            pre_clr = compute_min_body_clearance(qA_pre, cfgA, qB_pre, cfgB, exclude_tcp=False)
            if pre_clr < safe_distance:
                continue

            # Classify approach strategy
            if okA and okB:
                approach_mode = "DUAL_SYNCHRONIZED"
            elif (not okA) and okB:
                approach_mode = "ASYMMETRIC_A_FIRST"
            elif okA and (not okB):
                approach_mode = "ASYMMETRIC_B_FIRST"
            else:
                # Both failed corridor - reject this candidate
                continue

            ang_dev_deg = float(np.rad2deg(np.arccos(np.clip(-dot_z, -1.0, 1.0))))
            pair_key = f"{robot_a_id}_{robot_b_id}"

            pose = AutoHandoverTestPose(
                pair_name=pair_key,
                handover_position_world=p_H.copy(),
                standoff_distance_mm=standoff,
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
                approach_mode=approach_mode,
                candidate_origin=current_strategy,
                corridor_a_valid=okA,
                corridor_b_valid=okB,
            )

            if progress_callback:
                progress_callback(
                    f"[{robot_a_id}<->{robot_b_id}] Found valid solution! Strategy: {current_strategy}, "
                    f"Mode: {approach_mode}, Z={p_H[2]:.1f} mm, dot_z={dot_z:.4f}, clr={h_clr:.1f} mm."
                )

            return pose

    return None


def compute_all_test_handover_poses(
    robots: Dict[str, Any],
    handover_cfg: Optional[dict] = None,
    cache_path: Optional[str] = None,
    force_recompute: bool = False,
    preferred_strategy: str = "TOP_OF_WORKSPACE",
    progress_callback: Optional[Any] = None,
) -> Dict[str, AutoHandoverTestPose]:
    """Compute and cache test handover poses for all 3 pairs."""
    cache_path = cache_path or resolve_test_handover_cache_path()
    config_hash = compute_config_fingerprint(robots, handover_cfg)

    if not force_recompute:
        cached = load_test_handover_cache(cache_path, config_hash)
        if cached is not None:
            return cached

    if progress_callback:
        progress_callback("Computing 3D workspace intersections from DH models...")

    int_robots = {1: robots['robot1'], 2: robots['robot2'], 3: robots['robot3']}
    ws_config = {
        'workspace': {'sampling': {'voxel_size_mm': 25.0, 'workspace_samples': 20000}},
        'robots': handover_cfg.get('robots', {}) if handover_cfg else {}
    }
    ws = PositionWorkspaceComputer(ws_config, int_robots)
    ws.compute()

    solved_pairs: Dict[str, AutoHandoverTestPose] = {}
    pair_specs = [
        ('robot1', 'robot2', 1, 2),
        ('robot2', 'robot3', 2, 3),
        ('robot3', 'robot1', 3, 1),
    ]

    for rA_str, rB_str, idA, idB in pair_specs:
        pts = ws.get_common_voxel_centers(idA, idB)
        if len(pts) == 0:
            pts = ws.get_common_voxel_centers(idB, idA)

        pose = solve_pair_test_handover(
            rA_str,
            rB_str,
            robots,
            pts,
            handover_cfg=handover_cfg,
            preferred_strategy=preferred_strategy,
            progress_callback=progress_callback,
        )
        if pose is not None:
            solved_pairs[f"{rA_str}_{rB_str}"] = pose

    if len(solved_pairs) >= 3:
        save_test_handover_cache(cache_path, config_hash, solved_pairs)

    return solved_pairs
