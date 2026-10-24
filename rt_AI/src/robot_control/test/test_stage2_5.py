"""Unit tests for Stage 2.5 planning and object IK fallback."""

import numpy as np
import pytest

from robot_control.handover.stage1 import load_stage1_robots
from robot_control.handover.stage2_5 import (
    Stage2_5Plan,
    build_stage2_5_plan,
    plan_stage2_5_synchronized_paths,
    resolve_stage2_5_target,
)


@pytest.fixture(scope="module")
def handover_setup():
    handover_cfg, robots, _ = load_stage1_robots()
    return handover_cfg, robots


def test_stage2_5_target_object_ik_success(handover_setup):
    """Target reachable by object_pick_ik must use object_pick_ik without fallback."""
    _, robots = handover_setup
    target_xyz = [100.0, 100.0, 200.0]
    opts, is_fallback = resolve_stage2_5_target(
        target_xyz_world=target_xyz,
        robots=robots,
        target_yaw_deg=0.0,
        label="test_pick",
    )

    assert not is_fallback
    assert len(opts) > 0
    for rid, sol in opts.items():
        assert sol.solver_type == "object_pick_ik"
        assert not sol.is_fallback
        assert sol.error_mm <= 5.0
        assert sol.approach_tilt_deg is not None


def test_stage2_5_target_fallback_to_normal_ik(handover_setup):
    """Target unreachable by object orientation constraints must fall back to normal IK."""
    _, robots = handover_setup
    target_xyz = [-200.0, 100.0, 200.0]
    opts, is_fallback = resolve_stage2_5_target(
        target_xyz_world=target_xyz,
        robots=robots,
        target_yaw_deg=0.0,
        label="test_fallback",
    )

    assert is_fallback
    assert "robot3" in opts
    sol = opts["robot3"]
    assert sol.solver_type == "normal_ik"
    assert sol.is_fallback
    assert sol.error_mm <= 1.0


def test_stage2_5_independent_pick_and_place_modes(handover_setup):
    """Pick can use fallback while Place uses object_pick_ik, or vice versa."""
    handover_cfg, robots = handover_setup

    pick_xyz = [-200.0, 100.0, 200.0]  # Falls back to normal IK (robot3)
    place_xyz = [0.0, 150.0, 150.0]     # Solves with object_pick_ik (robot2/robot1)

    plan, pick_opts, place_opts = build_stage2_5_plan(
        pick_xyz=pick_xyz,
        place_xyz=place_xyz,
        robots=robots,
        handover_cfg=handover_cfg,
        chooser=lambda label, choices: choices[0],
        pick_yaw_deg=0.0,
        place_yaw_deg=45.0,
    )

    assert isinstance(plan, Stage2_5Plan)
    assert plan.is_pick_fallback is True
    assert plan.pick.solver_type == "normal_ik"

    assert plan.is_place_fallback is False
    assert plan.place.solver_type == "object_pick_ik"
    assert plan.place.approach_tilt_deg is not None

    assert plan.needs_handover is True
    assert plan.pick_pre_handover_q is not None
    assert plan.place_pre_handover_q is not None


def test_stage2_5_synchronized_trajectory_planning(handover_setup):
    """Trajectory planning produces valid, safe synchronized phases with clearances."""
    handover_cfg, robots = handover_setup

    pick_xyz = [-200.0, 100.0, 200.0]
    place_xyz = [0.0, 150.0, 150.0]

    plan, _, _ = build_stage2_5_plan(
        pick_xyz=pick_xyz,
        place_xyz=place_xyz,
        robots=robots,
        handover_cfg=handover_cfg,
        chooser=lambda label, choices: "robot3" if "robot3" in choices else choices[0],
        pick_yaw_deg=0.0,
        place_yaw_deg=0.0,
    )

    traj_result = plan_stage2_5_synchronized_paths(
        plan=plan,
        robots=robots,
        safe_distance=50.0,
        steps=15,
    )

    assert traj_result.needs_handover is True
    assert len(traj_result.phase1_simultaneous) == 15
    assert len(traj_result.phase4_advance_to_handover) == 15
    assert len(traj_result.phase9_simultaneous_home) == 15
    assert traj_result.min_overall_clearance_mm > 0.0
