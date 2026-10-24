"""Stage-2.5 Object-Pose IK with Stage 1.5 Synchronized Pre-Handover.

Stage 2.5 is an add-on to Stage 1.5:
  1. For both Pick and Place, attempts to solve via ``object_pick_ik``
     (aligning gripper finger-axis with object yaw and searching adaptive
     approach tilt angles).
  2. If ``object_pick_ik`` has no solution for a target for any robot,
     it automatically falls back to normal Stage 1 / 1.5 XYZ IK (``solve_stage1_ik``).
  3. Pick and Place fallbacks operate independently: if Pick falls back to normal IK
     while Place succeeds with ``object_pick_ik`` (or vice versa), both are utilized.
  4. Preserves Stage 1.5's synchronized staging:
     - Place robot moves to Pre-Handover simultaneously while Pick robot moves to Pick.
     - Advance to Handover, transfer object, and retract to Pre-Handover.
     - Place robot moves to Place, then simultaneous Home.
  5. Full-body link-to-link clearance checking and collision avoidance.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from robot_control.handover.object_pick_ik import (
    ObjectPickIKResult,
    solve_object_pick_ik,
)
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
)
from robot_control.handover.stage1_ik import (
    Stage1IKResult,
    solve_stage1_ik,
    world_to_base,
)
from robot_control.kinematics.fk_math import fk


@dataclass
class Stage2_5TargetIKResult:
    """IK result for Stage 2.5 target with object orientation & fallback metadata."""

    robot_id: str
    xyz_world: np.ndarray
    xyz_base: np.ndarray
    q_deg: np.ndarray
    q_rad: np.ndarray
    error_mm: float
    solver_type: str  # 'object_pick_ik' or 'normal_ik'
    is_fallback: bool
    requested_yaw_deg: float
    approach_tilt_deg: Optional[float] = None
    alignment_error_deg: Optional[float] = None
    orientation_error_deg: Optional[float] = None
    manipulability: Optional[float] = None
    score: float = 0.0
    solution_count: int = 1
    object_ik_result: Optional[ObjectPickIKResult] = None


def resolve_stage2_5_target(
    target_xyz_world: Union[np.ndarray, List[float]],
    robots: Dict[str, MockKinematicsConfig],
    target_yaw_deg: float = 0.0,
    current_q_deg: Optional[Dict[str, np.ndarray]] = None,
    label: str = "target",
) -> Tuple[Dict[str, Stage2_5TargetIKResult], bool]:
    """Resolve reachable robots for a target position and yaw.

    First attempts ``solve_object_pick_ik`` across all robots. If at least
    one robot has an object-aligned solution, those solutions are returned
    (is_fallback=False).

    If NO robot finds a solution via ``object_pick_ik``, it falls back to
    normal Stage 1/1.5 position IK (``solve_stage1_ik``) for all robots
    (is_fallback=True).

    Returns
    -------
    (reachable_dict, is_fallback):
        reachable_dict maps robot_id -> Stage2_5TargetIKResult.
        is_fallback is True if normal IK was used because object_pick_ik had no solution.
    """
    target_world = np.asarray(target_xyz_world, dtype=float)
    current_q_deg = current_q_deg or {}

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Attempt Object-Pose IK across all robots
    # ─────────────────────────────────────────────────────────────
    object_ik_solutions: Dict[str, Stage2_5TargetIKResult] = {}
    for rid, cfg in robots.items():
        q_cur = current_q_deg.get(rid)
        res: ObjectPickIKResult = solve_object_pick_ik(
            robot_id=rid,
            x=float(target_world[0]),
            y=float(target_world[1]),
            z=float(target_world[2]),
            object_yaw_deg=float(target_yaw_deg),
            config=cfg,
            current_q=q_cur,
            frame="world",
        )
        if res.success and res.q_deg is not None:
            achieved_base, _ = fk(res.q, cfg)
            target_base = world_to_base(target_world, cfg)
            error_mm = float(np.linalg.norm(target_base - achieved_base))

            object_ik_solutions[rid] = Stage2_5TargetIKResult(
                robot_id=rid,
                xyz_world=target_world.copy(),
                xyz_base=target_base,
                q_deg=np.asarray(res.q_deg, dtype=float).copy(),
                q_rad=np.asarray(res.q, dtype=float).copy(),
                error_mm=error_mm,
                solver_type="object_pick_ik",
                is_fallback=False,
                requested_yaw_deg=float(target_yaw_deg % 360.0),
                approach_tilt_deg=res.selected_tilt_deg,
                alignment_error_deg=res.alignment_error_deg,
                orientation_error_deg=res.orientation_error_deg,
                manipulability=res.manipulability,
                score=float(res.cost) if res.cost is not None else 0.0,
                solution_count=res.candidate_count,
                object_ik_result=res,
            )

    if len(object_ik_solutions) > 0:
        return object_ik_solutions, False

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Fallback to normal Stage 1/1.5 IK (Position Only)
    # ─────────────────────────────────────────────────────────────
    fallback_solutions: Dict[str, Stage2_5TargetIKResult] = {}
    for rid, cfg in robots.items():
        s1_res: Optional[Stage1IKResult] = solve_stage1_ik(
            robot_id=rid,
            target_xyz_world=target_world,
            config=cfg,
            current_q_deg=current_q_deg.get(rid),
        )
        if s1_res is not None:
            fallback_solutions[rid] = Stage2_5TargetIKResult(
                robot_id=rid,
                xyz_world=target_world.copy(),
                xyz_base=s1_res.xyz_base.copy(),
                q_deg=s1_res.q_deg.copy(),
                q_rad=s1_res.q_rad.copy(),
                error_mm=s1_res.error_mm,
                solver_type="normal_ik",
                is_fallback=True,
                requested_yaw_deg=float(target_yaw_deg % 360.0),
                approach_tilt_deg=None,
                alignment_error_deg=None,
                orientation_error_deg=None,
                manipulability=None,
                score=s1_res.error_mm,
                solution_count=1,
                object_ik_result=None,
            )

    return fallback_solutions, True


@dataclass
class Stage2_5Plan:
    """Stage 2.5 pick-and-place plan with solver metadata."""

    pick_robot: str
    place_robot: str
    pick: Stage2_5TargetIKResult
    place: Stage2_5TargetIKResult
    needs_handover: bool
    pick_handover_q: Optional[np.ndarray]
    place_handover_q: Optional[np.ndarray]
    pick_pre_handover_q: Optional[np.ndarray] = None
    place_pre_handover_q: Optional[np.ndarray] = None
    is_pick_fallback: bool = False
    is_place_fallback: bool = False

    def to_stage1_plan(self) -> Stage1Plan:
        """Convert to Stage1Plan format for downstream trajectory planners."""
        return Stage1Plan(
            pick_robot=self.pick_robot,
            place_robot=self.place_robot,
            pick=Stage1IKResult(
                robot_id=self.pick.robot_id,
                xyz_world=self.pick.xyz_world,
                xyz_base=self.pick.xyz_base,
                q_deg=self.pick.q_deg,
                q_rad=self.pick.q_rad,
                error_mm=self.pick.error_mm,
                solution_count=self.pick.solution_count,
            ),
            place=Stage1IKResult(
                robot_id=self.place.robot_id,
                xyz_world=self.place.xyz_world,
                xyz_base=self.place.xyz_base,
                q_deg=self.place.q_deg,
                q_rad=self.place.q_rad,
                error_mm=self.place.error_mm,
                solution_count=self.place.solution_count,
            ),
            needs_handover=self.needs_handover,
            pick_handover_q=self.pick_handover_q,
            place_handover_q=self.place_handover_q,
            pick_pre_handover_q=self.pick_pre_handover_q,
            place_pre_handover_q=self.place_pre_handover_q,
        )


def build_stage2_5_plan(
    pick_xyz: Union[np.ndarray, List[float]],
    place_xyz: Union[np.ndarray, List[float]],
    robots: Dict[str, MockKinematicsConfig],
    handover_cfg: dict,
    chooser: Callable[[str, List[str]], str],
    pick_yaw_deg: float = 0.0,
    place_yaw_deg: float = 0.0,
    current_q_deg: Optional[Dict[str, np.ndarray]] = None,
    cache_filepath: Optional[str] = None,
    config_fingerprint: Optional[str] = None,
) -> Tuple[Stage2_5Plan, Dict[str, Stage2_5TargetIKResult], Dict[str, Stage2_5TargetIKResult]]:
    """Build a complete Stage 2.5 plan with adaptive object IK & fallback.

    Parameters
    ----------
    pick_xyz : 3D world coordinates for pick.
    place_xyz : 3D world coordinates for place.
    robots : dict mapping robot_id to KinematicsConfig.
    handover_cfg : parsed handover.yaml config dict.
    chooser : callback ``chooser(label, robot_ids)`` when multiple robots reach a target.
    pick_yaw_deg : target object yaw at pick (deg).
    place_yaw_deg : target object yaw at place (deg).
    current_q_deg : optional current joint state dict.
    cache_filepath : optional path to persistent JSON cache file.
    config_fingerprint : optional configuration hash.

    Returns
    -------
    (plan, pick_opts, place_opts)
    """
    current_q_deg = current_q_deg or {}

    from robot_control.handover.trajectory_cache import (
        lookup_cached_target_ik,
        store_cached_target_ik,
    )

    # 1. Resolve PICK (check cache first if available)
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

    # 2. Resolve PLACE (check cache first if available)
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
    needs_handover = pick_rid != place_rid

    if needs_handover:
        handover = lookup_manual_handover_with_pre(handover_cfg, pick_rid, place_rid)
        if handover is None:
            raise RuntimeError(
                f"No manual_handover entry for {pick_rid} <-> {place_rid} in handover.yaml"
            )
        pick_h_q, place_h_q, pick_pre_h_q, place_pre_h_q = handover

    plan = Stage2_5Plan(
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
    )

    return plan, pick_opts, place_opts


def plan_stage2_5_synchronized_paths(
    plan: Stage2_5Plan,
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
