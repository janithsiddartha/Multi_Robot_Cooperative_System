"""Stage 3 Test 2 Planning & Trajectory Engine (Top-of-Workspace Only).

Differences from Stage 3 Test:
1. Strategy is permanently and transparently locked to TOP_OF_WORKSPACE.
2. Uses dedicated `stage3_test2_solver` and `auto_handover_poses_test2.json`.
3. Supports full dual synchronized and asymmetric sequential corridor approaches.
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
from robot_control.handover.stage3_test2_solver import (
    AutoHandoverTest2Pose,
    compute_all_test2_handover_poses,
    load_test2_handover_cache,
    resolve_test2_handover_cache_path,
)
from robot_control.handover.trajectory_cache import (
    compute_config_fingerprint,
    lookup_cached_target_ik,
    store_cached_target_ik,
)


@dataclass
class Stage3Test2Plan:
    """Task plan for Stage 3 Test 2 Environment."""

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
    auto_handover_info: Optional[AutoHandoverTest2Pose] = None
    approach_mode: str = "DUAL_SYNCHRONIZED"
    candidate_origin: str = "TOP_OF_WORKSPACE"

    def to_stage1_plan(self) -> Stage1Plan:
        """Convert to Stage1Plan format for downstream trajectory planners."""
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


def build_stage3_test2_plan(
    pick_xyz: List[float],
    place_xyz: List[float],
    robots: Dict[str, Any],
    handover_cfg: Optional[dict] = None,
    chooser: Optional[Callable[[str, List[str]], str]] = None,
    pick_yaw_deg: float = 0.0,
    place_yaw_deg: float = 0.0,
    use_auto_handover: bool = True,
    cache_filepath: Optional[str] = None,
    config_fingerprint: Optional[str] = None,
    test_poses: Optional[Dict[str, AutoHandoverTest2Pose]] = None,
) -> Tuple[Stage3Test2Plan, List[str], List[str]]:
    """Build task plan for Stage 3 Test 2 environment."""
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

    if not pick_opts:
        raise RuntimeError(f"Target Pick {pick_xyz} unreachable by any robot.")
    if not place_opts:
        raise RuntimeError(f"Target Place {place_xyz} unreachable by any robot.")

    pick_robot_candidates = list(pick_opts.keys())
    place_robot_candidates = list(place_opts.keys())

    # Interactive Chooser selection
    if chooser is not None and len(pick_robot_candidates) > 1:
        pick_robot = chooser("pick", pick_robot_candidates)
    else:
        pick_robot = pick_robot_candidates[0]

    if chooser is not None and len(place_robot_candidates) > 1:
        place_robot = chooser("place", place_robot_candidates)
    else:
        place_robot = place_robot_candidates[0]

    pick_res = pick_opts[pick_robot]
    place_res = place_opts[place_robot]

    needs_handover = (pick_robot != place_robot)
    pick_ho_q = None
    place_ho_q = None
    pick_pre_ho_q = None
    place_pre_ho_q = None
    matched_pose: Optional[AutoHandoverTest2Pose] = None
    approach_mode = "DUAL_SYNCHRONIZED"

    if needs_handover:
        pair_key = f"{pick_robot}_{place_robot}"
        inv_key = f"{place_robot}_{pick_robot}"

        if use_auto_handover:
            poses = test_poses
            if poses is None:
                cache_path = resolve_test2_handover_cache_path()
                fp = config_fingerprint or compute_config_fingerprint(robots, handover_cfg)
                poses = load_test2_handover_cache(cache_path, fp)
                if not poses:
                    # Fallback to test cache
                    try:
                        from robot_control.handover.stage3_test_solver import (
                            load_test_handover_cache,
                            resolve_test_handover_cache_path,
                        )
                        c_test = resolve_test_handover_cache_path()
                        poses = load_test_handover_cache(c_test, fp)
                    except Exception:
                        poses = None

            if poses and (pair_key in poses or inv_key in poses):
                p = poses.get(pair_key)
                if p is not None:
                    matched_pose = p
                    pick_ho_q = p.robot_a_handover_q_deg.copy()
                    place_ho_q = p.robot_b_handover_q_deg.copy()
                    pick_pre_ho_q = p.robot_a_pre_handover_q_deg.copy()
                    place_pre_ho_q = p.robot_b_pre_handover_q_deg.copy()
                    approach_mode = p.approach_mode
                else:
                    p = poses[inv_key]
                    matched_pose = p
                    pick_ho_q = p.robot_b_handover_q_deg.copy()
                    place_ho_q = p.robot_a_handover_q_deg.copy()
                    pick_pre_ho_q = p.robot_b_pre_handover_q_deg.copy()
                    place_pre_ho_q = p.robot_a_pre_handover_q_deg.copy()
                    if p.approach_mode == "ASYMMETRIC_A_FIRST":
                        approach_mode = "ASYMMETRIC_B_FIRST"
                    elif p.approach_mode == "ASYMMETRIC_B_FIRST":
                        approach_mode = "ASYMMETRIC_A_FIRST"
                    else:
                        approach_mode = p.approach_mode
            else:
                use_auto_handover = False

        if not use_auto_handover:
            res = lookup_manual_handover_with_pre(handover_cfg, pick_robot, place_robot)
            if res is not None:
                pick_ho_q, place_ho_q, pick_pre_ho_q, place_pre_ho_q = res
            else:
                raise RuntimeError(
                    f"No handover configuration available for {pair_key} in auto-handover or handover.yaml."
                )

    plan = Stage3Test2Plan(
        pick_robot=pick_robot,
        place_robot=place_robot,
        pick=pick_res,
        place=place_res,
        needs_handover=needs_handover,
        pick_handover_q=pick_ho_q,
        place_handover_q=place_ho_q,
        pick_pre_handover_q=pick_pre_ho_q,
        place_pre_handover_q=place_pre_ho_q,
        use_auto_handover=use_auto_handover,
        auto_handover_info=matched_pose,
        approach_mode=approach_mode,
        candidate_origin="TOP_OF_WORKSPACE",
    )

    return plan, pick_robot_candidates, place_robot_candidates


def plan_stage3_test2_paths(
    plan: Stage3Test2Plan,
    robots: Dict[str, Any],
    safe_distance: Optional[float] = None,
    steps: Optional[int] = None,
    current_q: Optional[Dict[str, np.ndarray]] = None,
    handover_cfg: Optional[dict] = None,
) -> Stage1_5TrajectoryResult:
    """Plan collision-free synchronized trajectories using Stage 1.5 trajectory engine."""
    stage1_plan = plan.to_stage1_plan()
    return plan_stage1_5_synchronized_paths(
        plan=stage1_plan,
        robots=robots,
        safe_distance=safe_distance,
        steps=steps,
        current_q=current_q,
        handover_cfg=handover_cfg,
    )


def main():
    """CLI runner for Stage 3 Test 2."""
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3 Test 2 Multi-Robot Planning Engine (Top-of-Workspace Only)")
    parser.add_argument("--pick", nargs=3, type=float, default=[100.0, 100.0, 200.0], help="Pick XYZ in mm")
    parser.add_argument("--place", nargs=3, type=float, default=[0.0, 150.0, 150.0], help="Place XYZ in mm")
    parser.add_argument("--pick-yaw", type=float, default=0.0, help="Pick yaw angle in degrees")
    parser.add_argument("--place-yaw", type=float, default=0.0, help="Place yaw angle in degrees")
    parser.add_argument("--recompute-handover", action="store_true", help="Force recomputation of Top-of-Workspace handover poses")
    args = parser.parse_args()

    robots, handover_cfg = load_stage1_robots()
    print("=== Stage 3 Test 2 Planning CLI (Top-of-Workspace Only) ===")

    if args.recompute_handover:
        print("Recomputing Top-of-Workspace handover poses...")
        poses = compute_all_test2_handover_poses(robots, handover_cfg, force_recompute=True)
        print(f"Solved {len(poses)} pairs.")

    plan, p_opts, pl_opts = build_stage3_test2_plan(
        pick_xyz=args.pick,
        place_xyz=args.place,
        robots=robots,
        handover_cfg=handover_cfg,
        pick_yaw_deg=args.pick_yaw,
        place_yaw_deg=args.place_yaw,
    )
    print(f"Plan built: Pick={plan.pick_robot}, Place={plan.place_robot}, Handover={plan.needs_handover}")
    traj = plan_stage3_test2_paths(plan, robots)
    print(f"Trajectory steps: {len(traj.steps)}, Min clearance: {traj.min_predicted_clearance_mm:.1f} mm")


if __name__ == "__main__":
    main()
