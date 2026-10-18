"""Stage 3 Multi-Robot Handover System with Kinematic Optimization.

Stage 3 builds on Stage 2.5:
  1. Pick & Place: Object-Pose IK with adaptive tilt angles and yaw alignment
     (with seamless fallback to normal position IK when orientation is unachievable).
  2. Handover & Pre-Handover Poses:
     - AUTO-CALCULATED MODE (Default): Handover point p_H and standoff positions
       p_pre are computed automatically from robot DH parameters and base mountings,
       maintaining continuous capsule clearance > 50 mm without manual configuration.
     - MANUAL OVERRIDE MODE: Operator can switch to legacy/manual handover poses
       from `handover.yaml`.
  3. Caching:
     - Target IK solutions are cached in `trajectory_cache.json`.
     - Auto-calculated handover poses are cached in `auto_handover_poses.json`
       keyed by config fingerprint.
  4. Synchronized 7-Phase Trajectory Execution:
     - Simultaneous movement to Pick and Pre-Handover.
     - Advance to Handover, object transfer, and retraction to Pre-Handover.
     - Simultaneous movement to Place and Home.
     - Whole-body capsule collision clearance monitoring.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from robot_control.handover.object_pick_ik import ObjectPickIKResult
from robot_control.handover.stage1 import (
    MockKinematicsConfig,
    Stage1Plan,
    load_stage1_robots,
    lookup_manual_handover_with_pre,
    select_robot,
)
from robot_control.handover.stage1_5 import (
    Stage1_5Step,
    Stage1_5TrajectoryResult,
    compute_min_body_clearance,
    plan_stage1_5_synchronized_paths,
    validate_execution_safety_gate,
)
from robot_control.handover.stage1_ik import (
    Stage1IKResult,
    solve_stage1_ik,
    world_to_base,
)
from robot_control.handover.stage2_5 import (
    Stage2_5Plan,
    Stage2_5TargetIKResult,
    resolve_stage2_5_target,
)
from robot_control.handover.stage3_handover_solver import (
    AutoHandoverPose,
    compute_all_auto_handover_poses,
    lookup_auto_handover_with_pre,
    resolve_auto_handover_cache_path,
)
from robot_control.handover.trajectory_cache import (
    compute_config_fingerprint,
    lookup_cached_target_ik,
    store_cached_target_ik,
)
from robot_control.kinematics.fk_math import fk


@dataclass
class Stage3Plan:
    """Complete Stage 3 Task Plan with Auto/Manual Handover metadata."""

    pick_robot: str
    place_robot: str
    pick: Stage2_5TargetIKResult
    place: Stage2_5TargetIKResult
    needs_handover: bool
    pick_handover_q: Optional[np.ndarray] = None
    place_handover_q: Optional[np.ndarray] = None
    pick_pre_handover_q: Optional[np.ndarray] = None
    place_pre_handover_q: Optional[np.ndarray] = None
    is_pick_fallback: bool = False
    is_place_fallback: bool = False
    use_auto_handover: bool = True
    auto_handover_info: Optional[AutoHandoverPose] = None

    def to_stage1_plan(self) -> Stage1Plan:
        """Convert to Stage1Plan for execution with the trajectory engine."""
        pick_ik = Stage1IKResult(
            robot_id=self.pick.robot_id,
            xyz_world=self.pick.xyz_world,
            xyz_base=self.pick.xyz_base,
            q_deg=self.pick.q_deg,
            q_rad=self.pick.q_rad,
            error_mm=self.pick.error_mm,
            solution_count=self.pick.solution_count,
        )
        place_ik = Stage1IKResult(
            robot_id=self.place.robot_id,
            xyz_world=self.place.xyz_world,
            xyz_base=self.place.xyz_base,
            q_deg=self.place.q_deg,
            q_rad=self.place.q_rad,
            error_mm=self.place.error_mm,
            solution_count=self.place.solution_count,
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

    def to_stage2_5_plan(self) -> Stage2_5Plan:
        """Convert to Stage2_5Plan."""
        return Stage2_5Plan(
            pick_robot=self.pick_robot,
            place_robot=self.place_robot,
            pick=self.pick,
            place=self.place,
            needs_handover=self.needs_handover,
            pick_handover_q=self.pick_handover_q,
            place_handover_q=self.place_handover_q,
            pick_pre_handover_q=self.pick_pre_handover_q,
            place_pre_handover_q=self.place_pre_handover_q,
            is_pick_fallback=self.is_pick_fallback,
            is_place_fallback=self.is_place_fallback,
        )


def build_stage3_plan(
    pick_xyz: Union[np.ndarray, List[float]],
    place_xyz: Union[np.ndarray, List[float]],
    robots: Dict[str, MockKinematicsConfig],
    handover_cfg: dict,
    chooser: Callable[[str, List[str]], str],
    pick_yaw_deg: float = 0.0,
    place_yaw_deg: float = 0.0,
    current_q_deg: Optional[Dict[str, np.ndarray]] = None,
    use_auto_handover: bool = True,
    cache_filepath: Optional[str] = None,
    config_fingerprint: Optional[str] = None,
    auto_poses: Optional[Dict[str, AutoHandoverPose]] = None,
    auto_handover_cache_path: Optional[str] = None,
) -> Tuple[Stage3Plan, Dict[str, Stage2_5TargetIKResult], Dict[str, Stage2_5TargetIKResult]]:
    """Build a complete Stage 3 plan.

    Parameters
    ----------
    pick_xyz : [X, Y, Z] world
    place_xyz : [X, Y, Z] world
    robots : dict of robot kinematics configs
    handover_cfg : parsed handover.yaml
    chooser : user callback to select robot when multiple are available
    pick_yaw_deg : object yaw at pick
    place_yaw_deg : object yaw at place
    current_q_deg : current joint angles
    use_auto_handover : bool, if True uses auto-calculated handover poses;
                        if False, falls back to manual poses in handover.yaml
    cache_filepath : path to trajectory_cache.json
    config_fingerprint : 16-char SHA-256 config hash
    auto_poses : in-memory dict of AutoHandoverPose objects
    auto_handover_cache_path : path to auto_handover_poses.json

    Returns
    -------
    (plan, pick_opts, place_opts)
    """
    current_q_deg = current_q_deg or {}

    # 1. Resolve PICK Target (checks cache first)
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
            current_q_deg=current_q_deg,
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
        raise RuntimeError("No robot workspace contains the pick XYZ (neither object IK nor normal IK found solutions).")

    pick_rid = select_robot(pick_opts, chooser, "pick")
    if pick_rid is None:
        raise RuntimeError("Pick robot was not selected.")

    # 2. Resolve PLACE Target (checks cache first)
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
            current_q_deg=current_q_deg,
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
        raise RuntimeError("No robot workspace contains the place XYZ (neither object IK nor normal IK found solutions).")

    place_rid = select_robot(place_opts, chooser, "place")
    if place_rid is None:
        raise RuntimeError("Place robot was not selected.")

    # 3. Resolve HANDOVER
    pick_h_q = None
    place_h_q = None
    pick_pre_h_q = None
    place_pre_h_q = None
    auto_pose_info = None
    needs_handover = pick_rid != place_rid

    if needs_handover:
        if use_auto_handover:
            # Mode A: Auto-Calculated Kinematic Handover
            if auto_poses is None:
                auto_poses = compute_all_auto_handover_poses(
                    robots=robots,
                    handover_cfg=handover_cfg,
                    cache_path=auto_handover_cache_path,
                    force_recompute=False,
                )

            res = lookup_auto_handover_with_pre(auto_poses, pick_rid, place_rid)
            if res is None:
                # Force recomputation once if missing
                auto_poses = compute_all_auto_handover_poses(
                    robots=robots,
                    handover_cfg=handover_cfg,
                    cache_path=auto_handover_cache_path,
                    force_recompute=True,
                )
                res = lookup_auto_handover_with_pre(auto_poses, pick_rid, place_rid)

            if res is None:
                raise RuntimeError(
                    f"Failed to find or calculate auto-handover pose for pair {pick_rid} <-> {place_rid}."
                )

            pick_h_q, place_h_q, pick_pre_h_q, place_pre_h_q, auto_pose_info = res
        else:
            # Mode B: Manual Override from handover.yaml
            manual_res = lookup_manual_handover_with_pre(handover_cfg, pick_rid, place_rid)
            if manual_res is None:
                raise RuntimeError(
                    f"No manual_handover entry for {pick_rid} <-> {place_rid} in handover.yaml"
                )
            pick_h_q, place_h_q, pick_pre_h_q, place_pre_h_q = manual_res

    plan = Stage3Plan(
        pick_robot=pick_rid,
        place_robot=place_rid,
        pick=pick_opts[pick_rid],
        place=place_opts[place_rid],
        needs_handover=needs_handover,
        pick_handover_q=pick_h_q,
        place_handover_q=place_h_q,
        pick_pre_handover_q=pick_pre_h_q,
        place_pre_handover_q=place_pre_h_q,
        is_pick_fallback=is_pick_fallback,
        is_place_fallback=is_place_fallback,
        use_auto_handover=use_auto_handover,
        auto_handover_info=auto_pose_info,
    )

    return plan, pick_opts, place_opts


def plan_stage3_synchronized_paths(
    plan: Stage3Plan,
    robots: Dict[str, MockKinematicsConfig],
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
    """CLI demonstration and verification of Stage 3 planning."""
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3 Handover Task Planning")
    parser.add_argument("--pick", nargs=3, type=float, default=[100.0, 100.0, 200.0])
    parser.add_argument("--place", nargs=3, type=float, default=[0.0, 150.0, 150.0])
    parser.add_argument("--pick-yaw", type=float, default=0.0)
    parser.add_argument("--place-yaw", type=float, default=0.0)
    parser.add_argument("--manual", action="store_true", help="Use manual handover from handover.yaml")
    args = parser.parse_args()

    handover_cfg, robots, _ = load_stage1_robots()
    print("===== STAGE 3 TASK PLANNING =====")
    print(f"Pick: {args.pick} (Yaw: {args.pick_yaw}°)")
    print(f"Place: {args.place} (Yaw: {args.place_yaw}°)")
    print(f"Handover Mode: {'Manual Override' if args.manual else 'Auto-Calculated Kinematic'}")

    def _cli_chooser(label: str, candidates: List[str]) -> str:
        print(f"Multiple robots reach {label}: {candidates}. Selecting {candidates[0]}.")
        return candidates[0]

    from robot_control.handover.trajectory_cache import resolve_cache_filepath
    cache_path = resolve_cache_filepath()
    config_hash = compute_config_fingerprint(robots, handover_cfg)

    plan, pick_opts, place_opts = build_stage3_plan(
        pick_xyz=args.pick,
        place_xyz=args.place,
        robots=robots,
        handover_cfg=handover_cfg,
        chooser=_cli_chooser,
        pick_yaw_deg=args.pick_yaw,
        place_yaw_deg=args.place_yaw,
        use_auto_handover=not args.manual,
        cache_filepath=cache_path,
        config_fingerprint=config_hash,
    )

    print("\n--- Plan Summary ---")
    print(f"Pick Robot: {plan.pick_robot} (Solver: {plan.pick.solver_type})")
    print(f"Place Robot: {plan.place_robot} (Solver: {plan.place.solver_type})")
    print(f"Needs Handover: {plan.needs_handover}")
    if plan.needs_handover:
        print(f"Handover Mode: {'AUTO-CALCULATED' if plan.use_auto_handover else 'MANUAL OVERRIDE'}")
        if plan.auto_handover_info:
            info = plan.auto_handover_info
            print(f"  Handover Position: {info.handover_position_world} mm")
            print(f"  Standoff Distance: {info.standoff_distance_mm} mm")
            print(f"  Handover Clearance: {info.handover_clearance_mm:.1f} mm")
            print(f"  Pre-Handover Clearance: {info.pre_handover_clearance_mm:.1f} mm")
        print(f"  Pick Handover Q: {np.round(plan.pick_handover_q, 2)}")
        print(f"  Place Handover Q: {np.round(plan.place_handover_q, 2)}")

    traj = plan_stage3_synchronized_paths(plan, robots)
    total_steps = (
        len(traj.phase1_simultaneous)
        + len(traj.phase3_pick_to_pre_handover)
        + len(traj.phase4_advance_to_handover)
        + len(traj.phase6_retract_to_pre_handover)
        + len(traj.phase7_handover_to_place)
        + len(traj.phase9_simultaneous_home)
    )
    print(f"\nTrajectory: {total_steps} steps planned. Min clearance: {traj.min_overall_clearance_mm:.1f} mm")

    # P0.4: Mandatory safety gate validation
    is_safe, msg = validate_execution_safety_gate(traj, robots, plan=plan)
    print(f"Execution Safety Gate: {'PASSED' if is_safe else 'BLOCKED'} - {msg}")


if __name__ == "__main__":
    main()
