"""Unit tests for Object-Pose IK pick-and-place solver.

Covers all 12 requirements from Section 8 of the task specification.
"""

import numpy as np

from robot_control.handover.handover_ik import (
    geometric_jacobian_and_fk,
    ik_pose_single,
)
from robot_control.handover.object_pick_ik import (
    compute_achieved_tilt_deg,
    generate_pick_orientations,
    parse_object_pick_target,
    solve_object_pick_ik,
)
from robot_control.handover.stage1 import load_stage1_robots
from robot_control.handover.stage1_ik import solve_stage1_ik
from robot_control.handover.task_ik import solve_task_ik
from robot_control.kinematics.fk_math import fk


def test_missing_yaw_defaults_to_zero():
    """Verify missing yaw defaults to 0.0 degrees."""
    pos, yaw = parse_object_pick_target(150.0, -50.0, 100.0)
    assert np.allclose(pos, [150.0, -50.0, 100.0])
    assert yaw == 0.0

    _, yaw_none = parse_object_pick_target(
        150.0, -50.0, 100.0, yaw_deg=None
    )
    assert yaw_none == 0.0


def test_explicit_yaw_normalized_correctly():
    """Verify explicit yaw values are normalized to [0, 360)."""
    _, yaw1 = parse_object_pick_target(0, 0, 0, yaw_deg=450.0)
    assert yaw1 == 90.0

    _, yaw2 = parse_object_pick_target(0, 0, 0, yaw_deg=-30.0)
    assert yaw2 == 330.0

    _, yaw3 = parse_object_pick_target(0, 0, 0, yaw_deg=360.0)
    assert yaw3 == 0.0

    _, yaw4 = parse_object_pick_target(0, 0, 0, yaw_deg=720.0)
    assert yaw4 == 0.0

    _, yaw5 = parse_object_pick_target(0, 0, 0, yaw_deg=0.0)
    assert yaw5 == 0.0


def test_zero_tilt_generates_downward_approach():
    """Verify zero tilt produces strictly downward approach."""
    cands = generate_pick_orientations(
        object_yaw_deg=0.0, tilt_angles_deg=[0.0]
    )
    assert len(cands) == 1
    r_mat = cands[0]['rotation']
    approach = cands[0]['approach_direction']

    assert np.allclose(approach, [0.0, 0.0, -1.0], atol=1e-7)
    assert np.allclose(r_mat[:, 2], [0.0, 0.0, -1.0], atol=1e-7)
    assert np.allclose(r_mat[:, 0], [1.0, 0.0, 0.0], atol=1e-7)

    cands_90 = generate_pick_orientations(
        object_yaw_deg=90.0, tilt_angles_deg=[0.0]
    )
    r_90 = cands_90[0]['rotation']
    assert np.allclose(
        cands_90[0]['approach_direction'], [0.0, 0.0, -1.0], atol=1e-7
    )
    assert np.allclose(r_90[:, 0], [0.0, 1.0, 0.0], atol=1e-7)


def test_changing_tilt_preserves_finger_axis_alignment():
    """Verify varying approach tilt preserves finger-axis alignment."""
    test_yaw_deg = 45.0
    tilts = [-45.0, -20.0, 0.0, 20.0, 45.0, 75.0]
    cands = generate_pick_orientations(
        object_yaw_deg=test_yaw_deg, tilt_angles_deg=tilts
    )

    expected_finger_axis = np.array([
        np.cos(np.deg2rad(test_yaw_deg)),
        np.sin(np.deg2rad(test_yaw_deg)),
        0.0,
    ])

    for cand in cands:
        r_mat = cand['rotation']
        tilt = cand['tilt_deg']
        assert np.allclose(r_mat[:, 0], expected_finger_axis, atol=1e-7), (
            f'Finger axis diverged at tilt {tilt} deg'
        )

        approach = cand['approach_direction']
        if abs(tilt) > 1e-3:
            assert not np.allclose(approach, [0.0, 0.0, -1.0], atol=1e-2)

        achieved_tilt = compute_achieved_tilt_deg(r_mat, test_yaw_deg)
        assert abs(achieved_tilt - tilt) < 1e-5


def test_all_robot_configurations_work_without_hardcoded_dof():
    """Verify all robots work dynamically without hard-coded DOF."""
    _, robots, _ = load_stage1_robots()

    assert robots['robot1'].num_joints == 5
    assert robots['robot2'].num_joints == 5
    assert robots['robot3'].num_joints == 6

    # robot1 (5-DOF)
    q1 = np.deg2rad([-60.0, -70.0, -18.9, 75.6, 30.0])
    p1, _ = fk(q1, robots['robot1'])
    res1 = solve_object_pick_ik(
        'robot1', p1[0], p1[1], p1[2],
        object_yaw_deg=330.9, config=robots['robot1']
    )
    assert res1.success, f'robot1 solve failed: {res1.failure_reason}'
    assert len(res1.q) == 5
    assert len(res1.q_deg) == 5

    # robot2 (5-DOF)
    q2 = np.deg2rad([-49.1, 47.8, 69.2, -0.9, -70.6])
    p2, _ = fk(q2, robots['robot2'])
    res2 = solve_object_pick_ik(
        'robot2', p2[0], p2[1], p2[2],
        object_yaw_deg=131.0, config=robots['robot2']
    )
    assert res2.success, f'robot2 solve failed: {res2.failure_reason}'
    assert len(res2.q) == 5
    assert len(res2.q_deg) == 5

    # robot3 (6-DOF)
    q3 = np.deg2rad([-28.1, -57.9, 26.5, -8.6, -54.4, -0.7])
    p3, _ = fk(q3, robots['robot3'])
    res3 = solve_object_pick_ik(
        'robot3', p3[0], p3[1], p3[2],
        object_yaw_deg=337.6, config=robots['robot3']
    )
    assert res3.success, f'robot3 solve failed: {res3.failure_reason}'
    assert len(res3.q) == 6
    assert len(res3.q_deg) == 6


def test_solutions_respect_joint_limits():
    """Verify returned solutions strictly respect joint limits."""
    _, robots, _ = load_stage1_robots()

    targets = {
        'robot1': ([-60.0, -70.0, -18.9, 75.6, 30.0], 330.9),
        'robot2': ([-49.1, 47.8, 69.2, -0.9, -70.6], 131.0),
        'robot3': ([-28.1, -57.9, 26.5, -8.6, -54.4, -0.7], 337.6),
    }

    for rid, (q_seed_deg, yaw) in targets.items():
        cfg = robots[rid]
        q_seed = np.deg2rad(q_seed_deg)
        p, _ = fk(q_seed, cfg)
        res = solve_object_pick_ik(
            rid, p[0], p[1], p[2], object_yaw_deg=yaw, config=cfg
        )
        assert res.success, f'Failed on {rid}: {res.failure_reason}'

        q_min = cfg.joint_limits_rad[:, 0]
        q_max = cfg.joint_limits_rad[:, 1]
        assert np.all(res.q >= q_min - 1e-6), f'{rid} violated lower limits'
        assert np.all(res.q <= q_max + 1e-6), f'{rid} violated upper limits'
        assert res.joint_limit_margin_deg >= 0.0


def test_fk_of_successful_solution_reaches_target():
    """Verify independent FK of solution reaches target within tolerance."""
    _, robots, _ = load_stage1_robots()
    cfg = robots['robot2']
    q_seed = np.deg2rad([-49.1, 47.8, 69.2, -0.9, -70.6])
    p_target, _ = fk(q_seed, cfg)

    pos_tol = 1.0
    res = solve_object_pick_ik(
        'robot2', p_target[0], p_target[1], p_target[2],
        object_yaw_deg=131.0, pos_tolerance=pos_tol
    )
    assert res.success, f'Solve failed: {res.failure_reason}'

    achieved_p, _ = fk(res.q, cfg)
    error = float(np.linalg.norm(p_target - achieved_p))
    assert error <= pos_tol
    assert res.position_error <= pos_tol


def test_infeasible_targets_return_failure():
    """Verify unreachable targets return clear failure."""
    res = solve_object_pick_ik(
        'robot1', 2000.0, 2000.0, 2000.0, object_yaw_deg=0.0
    )
    assert not res.success
    assert res.q is None
    assert res.q_deg is None
    assert res.failure_reason is not None
    assert len(res.failure_reason) > 0


def test_failed_first_orientation_tries_subsequent_candidates():
    """Verify failed first candidate searches alternative tilts."""
    _, robots, _ = load_stage1_robots()
    cfg = robots['robot3']

    q_tilted = np.deg2rad([30.0, -10.0, -40.0, 30.0, 45.0, 10.0])
    _, p_target, r_target = geometric_jacobian_and_fk(q_tilted, cfg)
    yaw = float(
        np.rad2deg(np.arctan2(r_target[1, 0], r_target[0, 0])) % 360.0
    )

    res_zero = solve_object_pick_ik(
        'robot3', p_target[0], p_target[1], p_target[2],
        object_yaw_deg=yaw, config=cfg,
        tilt_angles_deg=[0.0]
    )
    assert not res_zero.success

    res_search = solve_object_pick_ik(
        'robot3', p_target[0], p_target[1], p_target[2],
        object_yaw_deg=yaw, config=cfg,
        tilt_angles_deg=[0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 80.0, 85.0]
    )
    assert res_search.success, (
        f'Solver should succeed by trying tilted candidates: '
        f'{res_search.failure_reason}'
    )
    assert abs(res_search.selected_tilt_deg) > 0.0


def test_current_position_seed_preferred():
    """Verify current_q seed is preferred when equally valid."""
    _, robots, _ = load_stage1_robots()
    cfg = robots['robot3']
    q_current = np.deg2rad([-28.1, -57.9, 26.5, -8.6, -54.4, -0.7])
    p_target, t_current = fk(q_current, cfg)
    yaw = float(
        np.rad2deg(np.arctan2(t_current[1, 0], t_current[0, 0])) % 360.0
    )

    res = solve_object_pick_ik(
        'robot3', p_target[0], p_target[1], p_target[2],
        object_yaw_deg=yaw, config=cfg, current_q=q_current
    )
    assert res.success, f'Solve failed: {res.failure_reason}'
    assert np.allclose(res.q, q_current, atol=0.05)


def test_five_dof_reduced_task_formulation():
    """Verify 5-DOF reduced-task solves position and finger alignment."""
    _, robots, _ = load_stage1_robots()
    cfg1 = robots['robot1']
    assert cfg1.num_joints == 5

    q_seed = np.deg2rad([-60.0, -70.0, -18.9, 75.6, 30.0])
    p_target, _ = fk(q_seed, cfg1)

    res = solve_object_pick_ik(
        'robot1', p_target[0], p_target[1], p_target[2],
        object_yaw_deg=330.9, config=cfg1,
        pos_tolerance=1.0, align_tolerance_deg=5.0
    )
    assert res.success, f'Solve failed: {res.failure_reason}'
    assert res.position_error <= 1.0
    assert res.alignment_error_deg <= 5.0


def test_existing_ik_and_handover_apis_still_work():
    """Verify existing IK and handover APIs remain functional."""
    _, robots, _ = load_stage1_robots()
    cfg = robots['robot1']

    q_home = np.deg2rad(cfg.home_deg)
    p_home, _ = fk(q_home, cfg)
    p_world = cfg.world_base_R @ p_home + cfg.world_base_xyz
    res_stage1 = solve_stage1_ik(
        'robot1', p_world, cfg, current_q_deg=cfg.home_deg
    )
    assert res_stage1 is not None
    assert res_stage1.error_mm < 1.0

    r_d = np.eye(3)
    res_single = ik_pose_single(p_home, r_d, cfg, q_home)
    assert 'success' in res_single

    candidates = solve_task_ik(
        'robot1', p_home, cfg,
        approaches=[(0, 0, 1)],
        yaw_angles_deg=[0],
        tilt_angles_deg=[0]
    )
    assert isinstance(candidates, list)


def test_configurable_finger_axis_orientations():
    """Verify generate_pick_orientations aligns finger axis along X or Y."""
    yaw = 60.0
    u_expected = np.array([
        np.cos(np.deg2rad(yaw)),
        np.sin(np.deg2rad(yaw)),
        0.0,
    ])

    # Case finger_axis='x': local X (column 0) aligns with object yaw
    cands_x = generate_pick_orientations(
        object_yaw_deg=yaw,
        tilt_angles_deg=[-30.0, 0.0, 30.0],
        finger_axis='x',
    )
    for c in cands_x:
        assert np.allclose(c['rotation'][:, 0], u_expected, atol=1e-6)

    # Case finger_axis='y': local Y (column 1) aligns with object yaw
    cands_y = generate_pick_orientations(
        object_yaw_deg=yaw,
        tilt_angles_deg=[-30.0, 0.0, 30.0],
        finger_axis='y',
    )
    for c in cands_y:
        assert np.allclose(c['rotation'][:, 1], u_expected, atol=1e-6)
        # Approach direction (local Z) at zero tilt should be downward
        if abs(c['tilt_deg']) < 1e-4:
            assert np.allclose(c['rotation'][:, 2], [0.0, 0.0, -1.0], atol=1e-6)


def test_solve_object_pick_ik_with_finger_axis_y():
    """Verify solve_object_pick_ik succeeds and aligns local Y when finger_axis='y'."""
    _, robots, _ = load_stage1_robots()

    # Test 6-DOF robot (robot3)
    cfg3 = robots['robot3']
    q_home3 = np.deg2rad(cfg3.home_deg)
    p3, t3 = fk(q_home3, cfg3)
    yaw3 = float(np.rad2deg(np.arctan2(t3[1, 1], t3[0, 1])) % 360.0)
    res3_y = solve_object_pick_ik(
        'robot3', p3[0], p3[1], p3[2],
        object_yaw_deg=yaw3, config=cfg3,
        current_q=q_home3,
        finger_axis='y',
    )
    assert res3_y.success, f'Solve robot3 with finger_axis=y failed: {res3_y.failure_reason}'
    assert res3_y.finger_axis == 'y'
    assert res3_y.alignment_error_deg <= 5.0
    # Confirm achieved TCP local Y is aligned with yaw3
    r_tcp3 = res3_y.rotation
    target_u = np.array([np.cos(np.deg2rad(yaw3)), np.sin(np.deg2rad(yaw3)), 0.0])
    dot_y = np.dot(r_tcp3[:, 1], target_u)
    assert dot_y > np.cos(np.deg2rad(5.0))

    # Test 5-DOF robot (robot1)
    cfg1 = robots['robot1']
    q_seed1 = np.deg2rad([-60.0, -70.0, -18.9, 75.6, 30.0])
    p1, _ = fk(q_seed1, cfg1)
    # Target yaw aligned with robot1's base pointing
    target_yaw = float(np.rad2deg(np.arctan2(p1[1], p1[0])) % 360.0)
    res1_y = solve_object_pick_ik(
        'robot1', p1[0], p1[1], p1[2],
        object_yaw_deg=target_yaw, config=cfg1,
        finger_axis='y',
        pos_tolerance=1.0, align_tolerance_deg=5.0,
    )
    assert res1_y.success, f'Solve robot1 with finger_axis=y failed: {res1_y.failure_reason}'
    assert res1_y.finger_axis == 'y'
    assert res1_y.position_error <= 1.0
    assert res1_y.alignment_error_deg <= 5.0


def test_robot_yaml_finger_axis_setting():
    """Verify changing finger_axis on config dynamically influences solver."""
    _, robots, _ = load_stage1_robots()
    cfg = robots['robot3']

    # Default in robot3.yaml is 'x'
    assert getattr(cfg, 'gripper_finger_axis', None) == 'x'

    # Dynamically set to 'y' as if read from YAML
    cfg.gripper_finger_axis = 'y'
    p, _ = fk(cfg.home_deg, cfg)
    res = solve_object_pick_ik(
        'robot3', p[0], p[1], p[2],
        object_yaw_deg=0.0, config=cfg,
    )
    assert res.finger_axis == 'y'

