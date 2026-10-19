"""Stage 3 Test Planning & Trajectory Engine.

Supports both:
1. DUAL_SYNCHRONIZED: Full simultaneous Cartesian approach & retract corridors.
2. ASYMMETRIC_SEQUENTIAL: Non-corridor robot moves to handover first while other waits
   at pre-handover, then corridor robot advances along its straight Cartesian corridor.
   Retract strictly reverses this order to guarantee safe clearance and no base spins.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from robot_control.handover.stage1 import (
    MockKinematicsConfig,
    Stage1Plan,
    load_stage1_robots,
    generate_interpolated_path,
    select_robot,
    lookup_manual_handover_with_pre,
)
from robot_control.handover.stage1_5 import (
    Stage1_5Step,
    Stage1_5TrajectoryResult,
    compute_min_body_clearance,
    get_tcp_world_pose,
    generate_cartesian_corridor_trajectory,
    schedule_collision_aware_motion,
    validate_execution_safety_gate,
    plan_stage1_5_synchronized_paths,
)
from robot_control.handover.stage2_5 import resolve_stage2_5_target
from robot_control.handover.stage3_test_solver import (
    AutoHandoverTestPose,
    compute_all_test_handover_poses,
    load_test_handover_cache,
    resolve_test_handover_cache_path,
)
from robot_control.handover.trajectory_cache import (
    compute_config_fingerprint,
    lookup_cached_target_ik,
    store_cached_target_ik,
)


@dataclass
class Stage3TestPlan:
    """Task plan for Stage 3 Testing Environment."""

    pick_robot: str
    place_robot: str
    pick: Any
    place: Any
    needs_handover: bool
    pick_handover_q: Optional[np.ndarray] = None
    place_handover_q: Optional[np.ndarray] = None
    pick_pre_handover_q: Optional[np.ndarray] = None
    place_pre_handover_q: Optional[np.ndarray] = None
    use_auto_handover: bool = True
    auto_handover_info: Optional[AutoHandoverTestPose] = None
    approach_mode: str = "DUAL_SYNCHRONIZED"
    candidate_origin: str = "TOP_OF_WORKSPACE"

    def to_stage1_plan(self) -> Stage1Plan:
        """Convert to Stage1Plan format for downstream trajectory planners (identical to Stage 2.5)."""
        from robot_control.handover.stage1 import Stage1IKResult, Stage1Plan
        pick_q = _extract_q_deg(self.pick)
        place_q = _extract_q_deg(self.place)
        pick_ik = Stage1IKResult(
            robot_id=self.pick_robot,
            xyz_world=getattr(self.pick, "xyz_world", np.zeros(3)),
            xyz_base=getattr(self.pick, "xyz_base", np.zeros(3)),
            q_deg=pick_q,
            q_rad=np.deg2rad(pick_q),
            error_mm=getattr(self.pick, "error_mm", 0.0),
            solution_count=getattr(self.pick, "solution_count", 1),
        )
        place_ik = Stage1IKResult(
            robot_id=self.place_robot,
            xyz_world=getattr(self.place, "xyz_world", np.zeros(3)),
            xyz_base=getattr(self.place, "xyz_base", np.zeros(3)),
            q_deg=place_q,
            q_rad=np.deg2rad(place_q),
            error_mm=getattr(self.place, "error_mm", 0.0),
            solution_count=getattr(self.place, "solution_count", 1),
        )
        return Stage1Plan(
            pick_robot=self.pick_robot,
            place_robot=self.place_robot,
            pick=pick_ik,
            place=place_ik,
            needs_handover=self.needs_handover,
            pick_handover_q=self.pick_handover_q,
            place_handover_q=self.place_handover_q,
            pick_pre_handover_q=self.pick_pre_handover_q,
            place_pre_handover_q=self.place_pre_handover_q,
        )


def _extract_q_deg(target: Any) -> np.ndarray:
    """Extract joint angles in degrees from target object or array."""
    if hasattr(target, "q_deg"):
        return np.asarray(target.q_deg, dtype=float)
    if hasattr(target, "joint_angles_deg"):
        return np.asarray(target.joint_angles_deg, dtype=float)
    return np.asarray(target, dtype=float)


def build_stage3_test_plan(
    pick_xyz: List[float],
    place_xyz: List[float],
    robots: Dict[str, Any],
    handover_cfg: Optional[dict] = None,
    chooser: Optional[Callable[[str, List[str]], str]] = None,
    pick_yaw_deg: float = 0.0,
    place_yaw_deg: float = 0.0,
    use_auto_handover: bool = True,
    preferred_strategy: str = "TOP_OF_WORKSPACE",
    cache_filepath: Optional[str] = None,
    config_fingerprint: Optional[str] = None,
    test_poses: Optional[Dict[str, AutoHandoverTestPose]] = None,
) -> Tuple[Stage3TestPlan, List[str], List[str]]:
    """Build task plan for Stage 3 Test environment."""
    handover_cfg = handover_cfg or {}
    pick_target_xyz = np.array(pick_xyz, dtype=float)
    place_target_xyz = np.array(place_xyz, dtype=float)

    # 1. Resolve pick options (check disk cache first)
    pick_opts = None
    is_pick_fallback = False
    if cache_filepath and config_fingerprint:
        try:
            cached_pick = lookup_cached_target_ik(cache_filepath, config_fingerprint, pick_xyz, pick_yaw_deg)
            if cached_pick is not None:
                pick_opts, is_pick_fallback = cached_pick
        except Exception:
            pass

    if pick_opts is None:
        pick_opts, is_pick_fallback = resolve_stage2_5_target(
            pick_xyz,
            robots,
            target_yaw_deg=pick_yaw_deg,
            label="pick",
        )
        if cache_filepath and config_fingerprint and pick_opts:
            try:
                store_cached_target_ik(
                    cache_filepath, config_fingerprint, pick_xyz, pick_yaw_deg,
                    pick_opts, is_pick_fallback
                )
            except Exception:
                pass

    if not pick_opts:
        raise ValueError(f"No robot can reach pick target {pick_xyz}.")

    # 2. Resolve place options (check disk cache first)
    place_opts = None
    is_place_fallback = False
    if cache_filepath and config_fingerprint:
        try:
            cached_place = lookup_cached_target_ik(cache_filepath, config_fingerprint, place_xyz, place_yaw_deg)
            if cached_place is not None:
                place_opts, is_place_fallback = cached_place
        except Exception:
            pass

    if place_opts is None:
        place_opts, is_place_fallback = resolve_stage2_5_target(
            place_xyz,
            robots,
            target_yaw_deg=place_yaw_deg,
            label="place",
        )
        if cache_filepath and config_fingerprint and place_opts:
            try:
                store_cached_target_ik(
                    cache_filepath, config_fingerprint, place_xyz, place_yaw_deg,
                    place_opts, is_place_fallback
                )
            except Exception:
                pass

    if not place_opts:
        raise ValueError(f"No robot can reach place target {place_xyz}.")

    # Robot selection
    if chooser is not None:
        pick_rid = select_robot(pick_opts, chooser, "pick")
        place_rid = select_robot(place_opts, chooser, "place")
    else:
        pick_rid = list(pick_opts.keys())[0]
        place_cands = [r for r in place_opts.keys() if r != pick_rid]
        place_rid = place_cands[0] if place_cands else list(place_opts.keys())[0]

    if pick_rid is None or pick_rid not in pick_opts:
        pick_rid = list(pick_opts.keys())[0]
    if place_rid is None or place_rid not in place_opts:
        place_cands = [r for r in place_opts.keys() if r != pick_rid]
        place_rid = place_cands[0] if place_cands else list(place_opts.keys())[0]

    pick_target = pick_opts[pick_rid]
    place_target = place_opts[place_rid]

    needs_handover = (pick_rid != place_rid)

    pick_h_q = None
    place_h_q = None
    pick_pre_q = None
    place_pre_q = None
    auto_info = None
    approach_mode = "DUAL_SYNCHRONIZED"
    candidate_origin = preferred_strategy

    if needs_handover:
        if use_auto_handover:
            # Load or compute test poses
            if test_poses is None:
                cache_path = cache_filepath or resolve_test_handover_cache_path()
                test_poses = compute_all_test_handover_poses(
                    robots, handover_cfg, cache_path=cache_path, preferred_strategy=preferred_strategy
                )

            pair_key = f"{pick_rid}_{place_rid}"
            inv_key = f"{place_rid}_{pick_rid}"

            if pair_key in test_poses:
                p = test_poses[pair_key]
                pick_h_q = p.robot_a_handover_q_deg
                place_h_q = p.robot_b_handover_q_deg
                pick_pre_q = p.robot_a_pre_handover_q_deg
                place_pre_q = p.robot_b_pre_handover_q_deg
                auto_info = p
                candidate_origin = p.candidate_origin
            elif inv_key in test_poses:
                p = test_poses[inv_key]
                pick_h_q = p.robot_b_handover_q_deg
                place_h_q = p.robot_a_handover_q_deg
                pick_pre_q = p.robot_b_pre_handover_q_deg
                place_pre_q = p.robot_a_pre_handover_q_deg
                auto_info = p
                candidate_origin = p.candidate_origin
            else:
                pick_h_q = np.zeros(robots[pick_rid].num_joints)
                place_h_q = np.zeros(robots[place_rid].num_joints)
                pick_pre_q = pick_h_q.copy()
                place_pre_q = place_h_q.copy()
        else:
            # Manual handover from handover.yaml (like Stage 2.5)
            manual_h = lookup_manual_handover_with_pre(handover_cfg, pick_rid, place_rid)
            if manual_h is not None:
                pick_h_q, place_h_q, pick_pre_q, place_pre_q = manual_h
            else:
                pick_h_q = np.zeros(robots[pick_rid].num_joints)
                place_h_q = np.zeros(robots[place_rid].num_joints)
                pick_pre_q = pick_h_q.copy()
                place_pre_q = place_h_q.copy()

    plan = Stage3TestPlan(
        pick_robot=pick_rid,
        place_robot=place_rid,
        pick=pick_target,
        place=place_target,
        needs_handover=needs_handover,
        pick_handover_q=pick_h_q,
        place_handover_q=place_h_q,
        pick_pre_handover_q=pick_pre_q,
        place_pre_handover_q=place_pre_q,
        use_auto_handover=use_auto_handover,
        auto_handover_info=auto_info,
        approach_mode=approach_mode,
        candidate_origin=candidate_origin,
    )

    return plan, list(pick_opts.keys()), list(place_opts.keys())


def plan_stage3_test_paths(
    plan: Stage3TestPlan,
    robots: Dict[str, Any],
    safe_distance: Optional[float] = None,
    steps: Optional[int] = None,
    current_q: Optional[Dict[str, np.ndarray]] = None,
    handover_cfg: Optional[dict] = None,
) -> Stage1_5TrajectoryResult:
    """Plan collision-free synchronized trajectories using Stage 1.5 trajectory engine (identical to Stage 2.5)."""
    stage1_plan = plan.to_stage1_plan()
    return plan_stage1_5_synchronized_paths(
        plan=stage1_plan,
        robots=robots,
        safe_distance=safe_distance,
        steps=steps,
        current_q=current_q,
        handover_cfg=handover_cfg,
    )
