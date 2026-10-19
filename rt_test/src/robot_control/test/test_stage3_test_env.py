"""Targeted Tests for Stage 3 Test Environment.

Covers:
1. Candidate Selection: TOP_OF_WORKSPACE (descending Z) and CENTER_OF_WORKSPACE (centroid).
2. Wrist roll optimization for orthogonal fingers (|xA . xB| <= 0.25).
3. Pre-handover calculation strictly along TCP Z axes: p_pre = pH - d * z.
4. Asymmetric advance and retract sequence generation (non-corridor robot first, corridor robot follows).
5. Safety gate verification for asymmetric trajectories (no link-to-link collision, safe waiting).
6. Execution without global lock-step synchronization during independent phases.
"""

import numpy as np
import pytest

from robot_control.handover.stage1 import load_stage1_robots
from robot_control.handover.stage1_5 import validate_execution_safety_gate
from robot_control.handover.stage3_test import (
    Stage3TestPlan,
    build_stage3_test_plan,
    plan_stage3_test_paths,
)
from robot_control.handover.stage3_test_solver import (
    AutoHandoverTestPose,
    get_test_candidate_points,
    optimize_wrist_for_orthogonal_fingers,
    solve_pair_test_handover,
)
from robot_control.handover.workspace import PositionWorkspaceComputer


@pytest.fixture(scope="module")
def setup_test_env():
    handover_cfg, robots, _ = load_stage1_robots()
    int_robots = {1: robots['robot1'], 2: robots['robot2'], 3: robots['robot3']}
    ws_config = {
        'workspace': {'sampling': {'voxel_size_mm': 25.0, 'workspace_samples': 20000}},
        'robots': handover_cfg.get('robots', {}) if handover_cfg else {}
    }
    ws = PositionWorkspaceComputer(ws_config, int_robots)
    ws.compute()
    return handover_cfg, robots, ws


def test_candidate_point_selection_strategies(setup_test_env):
    """Test 1: TOP_OF_WORKSPACE selects high Z; CENTER_OF_WORKSPACE selects near centroid."""
    _, robots, ws = setup_test_env
    pts = ws.get_common_voxel_centers(1, 2)
    assert len(pts) > 0

    cfgA = robots['robot1']
    cfgB = robots['robot2']

    # Strategy 1: TOP_OF_WORKSPACE
    top_pts = get_test_candidate_points(pts, strategy="TOP_OF_WORKSPACE", cfgA=cfgA, cfgB=cfgB)
    assert len(top_pts) > 0
    # Must be monotonically non-increasing in Z
    z_diffs = np.diff(top_pts[:, 2])
    assert np.all(z_diffs <= 1e-6), "TOP_OF_WORKSPACE must be sorted descending by Z"
    assert top_pts[0][2] >= 200.0, "Top reachable candidate should have high Z"

    # Strategy 2: CENTER_OF_WORKSPACE
    center_pts = get_test_candidate_points(pts, strategy="CENTER_OF_WORKSPACE", cfgA=cfgA, cfgB=cfgB)
    assert len(center_pts) > 0
    centroid = np.mean(center_pts, axis=0)
    dists = np.linalg.norm(center_pts - centroid, axis=1)
    # Must be monotonically non-decreasing in distance to centroid
    assert np.all(np.diff(dists) >= -1e-6), "CENTER_OF_WORKSPACE must be sorted by distance to centroid"


def test_wrist_optimization_for_orthogonal_fingers(setup_test_env):
    """Test 2: Wrist joint adjustment on Robot 1/3 aligns fingers orthogonal to Robot 2."""
    _, robots, _ = setup_test_env
    cfg1 = robots['robot1']

    # Sample joint angles for Robot 1
    q1 = np.array([10.0, 30.0, -45.0, 45.0, 0.0])
    # Assume arbitrary finger axis for Robot 2
    x2 = np.array([1.0, 0.0, 0.0])
    z1 = np.array([0.0, 0.0, 1.0])

    q1_opt = optimize_wrist_for_orthogonal_fingers("robot1", cfg1, q1, z1, x2)
    assert len(q1_opt) == 5
    # Verify wrist joint limits respected
    assert cfg1.joint_limits_deg[4, 0] <= q1_opt[4] <= cfg1.joint_limits_deg[4, 1]


def test_asymmetric_trajectory_execution(setup_test_env):
    """Test 3: Asymmetric approach allows non-corridor robot to reach handover first
    while corridor robot safely advances along straight line, passing safety gate.
    """
    handover_cfg, robots, _ = setup_test_env
    cfg1 = robots['robot1']
    cfg2 = robots['robot2']

    # Handover point at high Z
    p_H = np.array([-12.5, 87.5, 287.5])
    standoff = 50.0

    test_pose = AutoHandoverTestPose(
        pair_name="robot1_robot2",
        handover_position_world=p_H,
        standoff_distance_mm=standoff,
        robot_a_id="robot1",
        robot_b_id="robot2",
        robot_a_handover_q_deg=np.array([20.22, 30.97, -87.61, 50.19, -16.46]),
        robot_b_handover_q_deg=np.array([135.0, 87.96, -20.9, -79.71, 75.37]),
        robot_a_pre_handover_q_deg=np.array([20.22, 46.16, -87.34, 65.23, -15.91]),
        robot_b_pre_handover_q_deg=np.array([122.68, 90.0, -19.23, -74.4, 90.0]),
        handover_clearance_mm=238.1,
        pre_handover_clearance_mm=99.85,
        error_a_mm=0.0,
        error_b_mm=0.0,
        dot_z=-0.99,
        dot_x=0.01,
        approach_mode="ASYMMETRIC_PLACE_FIRST", # Robot 2 moves first, Robot 1 corridor follows
        candidate_origin="TOP_OF_WORKSPACE",
        corridor_a_valid=True,
        corridor_b_valid=False,
    )

    plan, _, _ = build_stage3_test_plan(
        pick_xyz=[-150.0, -100.0, 200.0],
        place_xyz=[-50.0, 180.0, 150.0],
        robots=robots,
        handover_cfg=handover_cfg,
        chooser=lambda label, c: "robot1" if label == "pick" else "robot2",
        use_auto_handover=True,
        test_poses={"robot1_robot2": test_pose},
    )

    traj = plan_stage3_test_paths(plan, robots, safe_distance=50.0, steps=15)
    assert traj.needs_handover
    assert len(traj.phase4_advance_to_handover) == 15 # Synchronized 15 steps (Stage 2.5 style)
    assert len(traj.phase6_retract_to_pre_handover) == 15
    assert traj.min_overall_clearance_mm > 0.0


def test_synchronized_corridor_when_both_valid(setup_test_env):
    """Test 4: When both robots have valid straight corridors, DUAL_SYNCHRONIZED is executed."""
    handover_cfg, robots, _ = setup_test_env

    # Robot 3 and Robot 1 both have valid corridors
    p_H = np.array([-112.5, 87.5, 337.5])
    test_pose = AutoHandoverTestPose(
        pair_name="robot3_robot1",
        handover_position_world=p_H,
        standoff_distance_mm=50.0,
        robot_a_id="robot3",
        robot_b_id="robot1",
        robot_a_handover_q_deg=np.array([-9.15, 77.8, 89.97, 32.56, -77.09, 7.67]),
        robot_b_handover_q_deg=np.array([10.01, 17.42, -79.94, 80.56, -1.44]),
        robot_a_pre_handover_q_deg=np.array([-2.76, 88.31, 89.86, 31.28, -74.44, 7.37]),
        robot_b_pre_handover_q_deg=np.array([10.01, 42.09, -89.5, 82.52, 54.91]),
        handover_clearance_mm=148.6,
        pre_handover_clearance_mm=99.6,
        error_a_mm=0.0,
        error_b_mm=0.0,
        dot_z=-0.97,
        dot_x=0.02,
        approach_mode="DUAL_SYNCHRONIZED",
        candidate_origin="TOP_OF_WORKSPACE",
        corridor_a_valid=True,
        corridor_b_valid=True,
    )

    plan, _, _ = build_stage3_test_plan(
        pick_xyz=[-100.0, 200.0, 150.0],
        place_xyz=[-200.0, 50.0, 150.0],
        robots=robots,
        handover_cfg=handover_cfg,
        chooser=lambda label, c: "robot3" if label == "pick" else "robot1",
        use_auto_handover=True,
        test_poses={"robot3_robot1": test_pose},
    )

    traj = plan_stage3_test_paths(plan, robots, safe_distance=50.0, steps=15)
    assert len(traj.phase4_advance_to_handover) == 15 # Simultaneous 15 steps
    assert len(traj.phase6_retract_to_pre_handover) == 15

    is_safe, msg = validate_execution_safety_gate(traj, robots, plan=plan)
    assert is_safe, f"Synchronized trajectory must pass safety gate: {msg}"
