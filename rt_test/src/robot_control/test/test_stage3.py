"""Unit tests for Stage 3 Kinematic Handover, Safety Gate, and Approach Corridors."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from robot_control.handover.stage1 import load_stage1_robots
from robot_control.handover.stage1_5 import (
    Stage1_5Step,
    Stage1_5TrajectoryResult,
    compute_min_body_clearance,
    dist_point_to_segment,
    dist_segment_to_segment,
    validate_execution_safety_gate,
)
from robot_control.handover.stage3 import (
    Stage3Plan,
    build_stage3_plan,
    plan_stage3_synchronized_paths,
)
from robot_control.handover.stage3_handover_solver import (
    AutoHandoverPose,
    compute_all_auto_handover_poses,
    compute_coupled_handover_jacobian,
    get_tcp_world_pose,
    lookup_auto_handover_with_pre,
    resolve_auto_handover_cache_path,
    solve_coupled_handover_orientation,
    solve_coupled_handover_orientation_analytical,
    solve_coupled_handover_orientation_dls,
    solve_coupled_handover_orientation_lbfgs,
)
from robot_control.handover.trajectory_cache import (
    compute_config_fingerprint,
    resolve_cache_filepath,
)
from robot_control.kinematics.fk_math import fk


@pytest.fixture(scope="module")
def stage3_setup():
    handover_cfg, robots, _ = load_stage1_robots()
    cache_path = resolve_cache_filepath()
    auto_cache_path = resolve_auto_handover_cache_path()
    config_hash = compute_config_fingerprint(robots, handover_cfg)
    auto_poses = compute_all_auto_handover_poses(robots, handover_cfg, cache_path=auto_cache_path)
    return handover_cfg, robots, cache_path, auto_cache_path, config_hash, auto_poses


# =============================================================================
# P0.1: ORIENTATION CONSTRAINT TESTS
# =============================================================================

def test_stage3_orientation_constraints_all_pairs(stage3_setup):
    """P0.1: Handover orientation constraints must hold across all 3 robot pairs:
    1. World TCP positions agree: ||p_A - p_B|| <= 1.0 mm.
    2. Approach axes anti-parallel: z_A . z_B <= -1 + eps_z (<= -cos(15 deg) ~= -0.9659).
    3. Finger axes orthogonal: |x_A . x_B| <= eps_x (<= sin(15 deg) ~= 0.2588).
    Independently recomputed via forward kinematics.
    """
    handover_cfg, robots, _, _, _, auto_poses = stage3_setup
    assert len(auto_poses) == 3

    ori_tol_deg = float(handover_cfg.get("pose_tolerances", {}).get("orientation_deg", 15.0))
    pos_tol_mm = float(handover_cfg.get("pose_tolerances", {}).get("position_mm", 1.0))
    theta_tol_rad = np.deg2rad(ori_tol_deg)
    eps_z = 1.0 - np.cos(theta_tol_rad)
    eps_x = np.sin(theta_tol_rad)

    pairs = [
        ("robot1_robot2", "robot1", "robot2"),
        ("robot2_robot3", "robot2", "robot3"),
        ("robot3_robot1", "robot3", "robot1"),
    ]

    for pair_key, rA_id, rB_id in pairs:
        assert pair_key in auto_poses, f"Missing handover pose for {pair_key}"
        pose = auto_poses[pair_key]
        cfgA = robots[rA_id]
        cfgB = robots[rB_id]

        # Independently recompute FK in world frame
        pA_w, zA_w, xA_w, _ = get_tcp_world_pose(pose.robot_a_handover_q_deg, cfgA, in_degrees=True)
        pB_w, zB_w, xB_w, _ = get_tcp_world_pose(pose.robot_b_handover_q_deg, cfgB, in_degrees=True)

        # 1. Position agreement
        err_pos = float(np.linalg.norm(pA_w - pB_w))
        assert err_pos <= pos_tol_mm, (
            f"[{pair_key}] Handover positions disagree by {err_pos:.3f} mm (> {pos_tol_mm} mm)"
        )

        # 2. Anti-parallel approach axes: zA . zB <= -1 + eps_z
        dot_z = float(np.dot(zA_w, zB_w))
        ang_dev_deg = float(np.rad2deg(np.arccos(np.clip(-dot_z, -1.0, 1.0))))
        assert dot_z <= (-1.0 + eps_z), (
            f"[{pair_key}] Approach axes not anti-parallel: zA . zB = {dot_z:.4f} "
            f"(deviation {ang_dev_deg:.1f} deg > {ori_tol_deg} deg)"
        )

        # 3. Orthogonal finger axes: |xA . xB| <= eps_x
        dot_x = abs(float(np.dot(xA_w, xB_w)))
        assert dot_x <= eps_x, (
            f"[{pair_key}] Finger axes not orthogonal: |xA . xB| = {dot_x:.4f} (> {eps_x:.4f})"
        )

        # 4. Whole-body clearance must exceed safe threshold
        h_clr = compute_min_body_clearance(
            pose.robot_a_handover_q_deg, cfgA,
            pose.robot_b_handover_q_deg, cfgB,
            exclude_tcp=True,
        )
        assert h_clr >= 45.0, f"[{pair_key}] Handover body clearance {h_clr:.1f} mm < 45 mm"


def test_stage3_position_only_ik_rejected_when_orientation_fails(stage3_setup):
    """P0.1: A candidate where position IK succeeds but orientation fails must be rejected."""
    _, robots, _, _, _, _ = stage3_setup
    # Position where independent position IK succeeds but natural orientations face skewed
    # [0.0, 0.0, 150.0] has dot_z ~ 0.49 (away from anti-parallel)
    bad_p = np.array([0.0, 0.0, 150.0])

    # Run coupled orientation solver with strict tolerances (10.0 deg)
    res = solve_coupled_handover_orientation(
        robot_a_id="robot1",
        robot_b_id="robot2",
        p_H_world=bad_p,
        robots=robots,
        pos_tol_mm=1.0,
        ori_tol_deg=10.0,
        min_clearance_mm=45.0,
        max_opt_iterations=20,
    )
    # The solver must reject or return None because hard orientation constraints cannot be met
    if res is not None:
        _, _, _, _, dot_z, dot_x, _ = res
        theta_tol_rad = np.deg2rad(10.0)
        eps_z = 1.0 - np.cos(theta_tol_rad)
        eps_x = np.sin(theta_tol_rad)
        # If it returned a solution, it MUST strictly satisfy the hard constraints
        assert dot_z <= (-1.0 + eps_z)
        assert dot_x <= eps_x


def test_stage3_solver_reports_failure_on_impossible_constraints(stage3_setup):
    """P0.1: If no compatible orientation exists, solver reports failure (None) without relaxing."""
    _, robots, _, _, _, _ = stage3_setup
    # Point outside robot reachable overlap
    impossible_p = np.array([500.0, 500.0, 500.0])
    res = solve_coupled_handover_orientation(
        robot_a_id="robot1",
        robot_b_id="robot2",
        p_H_world=impossible_p,
        robots=robots,
        pos_tol_mm=1.0,
        ori_tol_deg=15.0,
    )
    assert res is None, "Solver should return None when position/orientation is impossible."


# =============================================================================
# P0.2: STANDOFF & TCP APPROACH CORRIDOR TESTS
# =============================================================================

def test_stage3_standoff_tcp_corridor_calculation(stage3_setup):
    """P0.2: Pre-handover positions must be calculated strictly along the world TCP Z axis:
      p_pre,A = p_H - d_A * z_A
      p_pre,B = p_H - d_B * z_B
    and standoff distances must remain within configured limits [50, 100] mm.
    """
    _, robots, _, _, _, auto_poses = stage3_setup

    for pair_key, pose in auto_poses.items():
        cfgA = robots[pose.robot_a_id]
        cfgB = robots[pose.robot_b_id]

        p_H = pose.handover_position_world
        d = pose.standoff_distance_mm
        assert 50.0 <= d <= 100.0, f"[{pair_key}] Standoff {d} mm outside [50, 100] mm range"

        # Obtain world TCP approach axes
        _, zA_w, _, _ = get_tcp_world_pose(pose.robot_a_handover_q_deg, cfgA, in_degrees=True)
        _, zB_w, _, _ = get_tcp_world_pose(pose.robot_b_handover_q_deg, cfgB, in_degrees=True)

        expected_preA = p_H - d * zA_w
        expected_preB = p_H - d * zB_w

        # Check that pre-handover joint positions achieve these expected Cartesian points
        p_preA_actual, _, _, _ = get_tcp_world_pose(pose.robot_a_pre_handover_q_deg, cfgA, in_degrees=True)
        p_preB_actual, _, _, _ = get_tcp_world_pose(pose.robot_b_pre_handover_q_deg, cfgB, in_degrees=True)

        err_preA = np.linalg.norm(p_preA_actual - expected_preA)
        err_preB = np.linalg.norm(p_preB_actual - expected_preB)

        assert err_preA <= 1.5, f"[{pair_key}] Pre-handover A deviates from approach corridor by {err_preA:.2f} mm"
        assert err_preB <= 1.5, f"[{pair_key}] Pre-handover B deviates from approach corridor by {err_preB:.2f} mm"

        # Pre-handover body clearance must exceed 50 mm
        assert pose.pre_handover_clearance_mm >= 50.0


def test_stage3_standoff_trajectory_corridor_continuity(stage3_setup):
    """P0.2: Connecting joint trajectory between pre-handover and handover must maintain clearance."""
    _, robots, _, _, _, auto_poses = stage3_setup

    for pair_key, pose in auto_poses.items():
        cfgA = robots[pose.robot_a_id]
        cfgB = robots[pose.robot_b_id]

        qA_pre = pose.robot_a_pre_handover_q_deg
        qA_h = pose.robot_a_handover_q_deg
        qB_pre = pose.robot_b_pre_handover_q_deg
        qB_h = pose.robot_b_handover_q_deg

        # Sample 10 points along the connecting trajectory
        for alpha in np.linspace(0.0, 1.0, 10):
            qA_k = qA_pre + alpha * (qA_h - qA_pre)
            qB_k = qB_pre + alpha * (qB_h - qB_pre)
            clr_k = compute_min_body_clearance(qA_k, cfgA, qB_k, cfgB, exclude_tcp=True)
            assert clr_k >= 45.0, (
                f"[{pair_key}] Connecting trajectory clearance dropped to {clr_k:.1f} mm at alpha={alpha:.2f}"
            )


# =============================================================================
# P0.3: INTENDED CONTACT VS. BODY COLLISION SEPARATION
# =============================================================================

def test_stage3_no_safety_bypass_near_contact(stage3_setup):
    """P0.3: During advance/retract, clearance below safe_distance must mark step unsafe.
    No unconditional safety bypass (is_near_contact) is allowed.
    """
    handover_cfg, robots, cache_path, auto_cache_path, config_hash, auto_poses = stage3_setup

    def chooser(label, candidates):
        return "robot1" if label == "pick" else "robot2"

    plan, _, _ = build_stage3_plan(
        pick_xyz=[100.0, 100.0, 200.0],
        place_xyz=[-50.0, 180.0, 150.0],
        robots=robots,
        handover_cfg=handover_cfg,
        chooser=chooser,
        use_auto_handover=True,
        cache_filepath=cache_path,
        config_fingerprint=config_hash,
        auto_poses=auto_poses,
    )

    # Artificially set safe_distance very high (300 mm) so body clearance violates threshold
    traj_unsafe = plan_stage3_synchronized_paths(plan, robots, safe_distance=300.0, steps=15)

    # In advance and retract phases, all steps that violate safe_distance must be is_safe=False
    for step in traj_unsafe.phase4_advance_to_handover:
        if step.min_clearance_mm < 300.0:
            assert not step.is_safe, (
                f"Safety bypass detected in advance phase: step has clearance {step.min_clearance_mm:.1f} mm "
                f"< 300.0 mm but is_safe=True"
            )

    for step in traj_unsafe.phase6_retract_to_pre_handover:
        if step.min_clearance_mm < 300.0:
            assert not step.is_safe, (
                f"Safety bypass detected in retract phase: step has clearance {step.min_clearance_mm:.1f} mm "
                f"< 300.0 mm but is_safe=True"
            )

    assert not traj_unsafe.all_phases_safe


def test_stage3_segment_distance_geometries():
    """P0.3: Test exact segment distance calculations for key spatial geometries."""
    # 1. Parallel segments at distance 1.0
    d_par = dist_segment_to_segment(
        np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0])
    )
    assert abs(d_par - 1.0) < 1e-4

    # 2. Intersecting segments
    d_int = dist_segment_to_segment(
        np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])
    )
    assert abs(d_int) < 1e-4

    # 3. Skew orthogonal segments at distance 1.0
    d_skew = dist_segment_to_segment(
        np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 1.0]), np.array([0.0, 1.0, 1.0])
    )
    assert abs(d_skew - 1.0) < 1e-4

    # 4. Collinear segments separated by distance 1.0
    d_end = dist_segment_to_segment(
        np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])
    )
    assert abs(d_end - 1.0) < 1e-4

    # 5. Degenerate zero-length segment (point to segment)
    d_deg = dist_segment_to_segment(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0])
    )
    assert abs(d_deg - 1.0) < 1e-4


# =============================================================================
# P0.4: MANDATORY EXECUTION SAFETY GATE TESTS
# =============================================================================

def test_stage3_execution_safety_gate_passes_for_valid_plan(stage3_setup):
    """P0.4: A fully validated safe trajectory passes the execution safety gate."""
    handover_cfg, robots, cache_path, auto_cache_path, config_hash, auto_poses = stage3_setup

    def chooser(label, candidates):
        return "robot1" if label == "pick" else "robot2"

    plan, _, _ = build_stage3_plan(
        pick_xyz=[-150.0, -100.0, 200.0],
        place_xyz=[150.0, 100.0, 180.0],
        robots=robots,
        handover_cfg=handover_cfg,
        chooser=chooser,
        use_auto_handover=True,
        cache_filepath=cache_path,
        config_fingerprint=config_hash,
        auto_poses=auto_poses,
    )

    traj = plan_stage3_synchronized_paths(plan, robots, safe_distance=50.0, steps=15)
    is_safe, msg = validate_execution_safety_gate(traj, robots, plan=plan)
    assert is_safe, f"Valid plan failed safety gate: {msg}"
    assert "Safety gate passed" in msg


def test_stage3_execution_safety_gate_blocks_unsafe_flag(stage3_setup):
    """P0.4: A trajectory with all_phases_safe=False must be blocked by the safety gate."""
    _, robots, _, _, _, auto_poses = stage3_setup
    pose = auto_poses["robot1_robot2"]

    # Construct mock trajectory with all_phases_safe=False
    step = Stage1_5Step("step", pose.robot_a_handover_q_deg, pose.robot_b_handover_q_deg, 20.0, False, "test")
    traj = Stage1_5TrajectoryResult(
        needs_handover=True,
        pick_robot="robot1",
        place_robot="robot2",
        phase1_simultaneous=[step],
        phase3_pick_to_pre_handover=[step],
        phase4_advance_to_handover=[step],
        phase6_retract_to_pre_handover=[step],
        phase7_handover_to_place=[step],
        phase9_simultaneous_home=[step],
        min_overall_clearance_mm=20.0,
        all_phases_safe=False,
        collision_avoidance_applied=False,
    )

    is_safe, msg = validate_execution_safety_gate(traj, robots)
    assert not is_safe
    assert "all_phases_safe is False" in msg


def test_stage3_execution_safety_gate_blocks_unsafe_step_override(stage3_setup):
    """P0.4: Trajectory containing an unsafe step is blocked even if all_phases_safe=True."""
    _, robots, _, _, _, auto_poses = stage3_setup
    pose = auto_poses["robot1_robot2"]

    safe_step = Stage1_5Step("step", pose.robot_a_handover_q_deg, pose.robot_b_handover_q_deg, 60.0, True, "safe")
    unsafe_step = Stage1_5Step("step", pose.robot_a_handover_q_deg, pose.robot_b_handover_q_deg, 10.0, False, "unsafe")

    traj = Stage1_5TrajectoryResult(
        needs_handover=True,
        pick_robot="robot1",
        place_robot="robot2",
        phase1_simultaneous=[safe_step],
        phase3_pick_to_pre_handover=[safe_step],
        phase4_advance_to_handover=[safe_step, unsafe_step],  # Unsafe step here!
        phase6_retract_to_pre_handover=[safe_step],
        phase7_handover_to_place=[safe_step],
        phase9_simultaneous_home=[safe_step],
        min_overall_clearance_mm=10.0,
        all_phases_safe=True,  # Inconsistently marked True
        collision_avoidance_applied=False,
    )

    is_safe, msg = validate_execution_safety_gate(traj, robots)
    assert not is_safe
    assert "unsafe step" in msg


def test_stage3_execution_safety_gate_blocks_joint_limit_violation(stage3_setup):
    """P0.4: A trajectory with joint angles outside configured limits must be blocked."""
    _, robots, _, _, _, auto_poses = stage3_setup
    pose = auto_poses["robot1_robot2"]

    # Inject joint limit violation (+120 deg on Robot 1 joint 0 whose limits are [-90, +90])
    violating_q = pose.robot_a_handover_q_deg.copy()
    violating_q[0] = 120.0

    violating_step = Stage1_5Step("step", violating_q, pose.robot_b_handover_q_deg, 60.0, True, "limit violation")
    traj = Stage1_5TrajectoryResult(
        needs_handover=True,
        pick_robot="robot1",
        place_robot="robot2",
        phase1_simultaneous=[violating_step],
        phase3_pick_to_pre_handover=[violating_step],
        phase4_advance_to_handover=[violating_step],
        phase6_retract_to_pre_handover=[violating_step],
        phase7_handover_to_place=[violating_step],
        phase9_simultaneous_home=[violating_step],
        min_overall_clearance_mm=60.0,
        all_phases_safe=True,
        collision_avoidance_applied=False,
    )

    is_safe, msg = validate_execution_safety_gate(traj, robots)
    assert not is_safe
    assert "joint limit violation" in msg


def test_stage3_mock_command_interface_blocks_execution_on_gate_failure(stage3_setup):
    """P0.4: Mocked command interface verifies that NO robot command is issued when gate fails."""
    _, robots, _, _, _, auto_poses = stage3_setup
    pose = auto_poses["robot1_robot2"]

    mock_cmd_publisher = MagicMock()

    # Unsafe trajectory
    unsafe_step = Stage1_5Step("step", pose.robot_a_handover_q_deg, pose.robot_b_handover_q_deg, 10.0, False, "unsafe")
    traj = Stage1_5TrajectoryResult(
        needs_handover=True,
        pick_robot="robot1",
        place_robot="robot2",
        phase1_simultaneous=[unsafe_step],
        phase3_pick_to_pre_handover=[],
        phase4_advance_to_handover=[],
        phase6_retract_to_pre_handover=[],
        phase7_handover_to_place=[],
        phase9_simultaneous_home=[],
        min_overall_clearance_mm=10.0,
        all_phases_safe=False,
        collision_avoidance_applied=False,
    )

    # Simulating execution dispatcher protected by the gate
    is_safe, msg = validate_execution_safety_gate(traj, robots)
    if is_safe:
        mock_cmd_publisher.publish("command")

    # Command publisher MUST NOT have been called
    mock_cmd_publisher.publish.assert_not_called()
    assert mock_cmd_publisher.call_count == 0


# =============================================================================
# ANALYTICAL COUPLED JACOBIAN & OPTIMIZER BENCHMARK / REGRESSION TESTS
# =============================================================================

def test_analytical_jacobian_vs_central_finite_diff_all_pairs(stage3_setup):
    """Verify analytical coupled Jacobian against central finite differences.

    Compares the analytical task Jacobian blocks (coincidence, anchor, approach axis,
    and finger orthogonality) against central finite differences (step delta = 1e-6)
    across all 3 robot pairs:
      1. robot1_robot2 (5+5 DOF)
      2. robot2_robot3 (5+6 DOF)
      3. robot3_robot1 (6+5 DOF)
    """
    _, robots, _, _, _, _ = stage3_setup

    pairs = [
        ("robot1", "robot2", np.array([-12.5, 87.5, 212.5])),
        ("robot2", "robot3", np.array([187.5, 212.5, 212.5])),
        ("robot3", "robot1", np.array([-37.5, 37.5, 112.5])),
    ]

    delta = 1e-6

    for rA_id, rB_id, p_H in pairs:
        cfgA = robots[rA_id]
        cfgB = robots[rB_id]
        nA = cfgA.num_joints
        nB = cfgB.num_joints

        qA0 = np.deg2rad(cfgA.home_deg) + 0.05
        qB0 = np.deg2rad(cfgB.home_deg) - 0.05
        q0 = np.concatenate([qA0, qB0])

        J_anal, e0, _ = compute_coupled_handover_jacobian(qA0, qB0, cfgA, cfgB, p_H_world=p_H)

        m = len(e0)
        J_num = np.zeros((m, nA + nB))

        for j in range(nA + nB):
            qp = q0.copy()
            qp[j] += delta
            qm = q0.copy()
            qm[j] -= delta

            _, ep, _ = compute_coupled_handover_jacobian(qp[:nA], qp[nA:], cfgA, cfgB, p_H_world=p_H)
            _, em, _ = compute_coupled_handover_jacobian(qm[:nA], qm[nA:], cfgA, cfgB, p_H_world=p_H)

            # Exact derivative de/dq == J_anal
            J_num[:, j] = (ep - em) / (2.0 * delta)

        max_abs_err = float(np.max(np.abs(J_anal - J_num)))
        rel_denom = np.maximum(np.abs(J_anal), 1e-6)
        max_rel_err = float(np.max(np.abs(J_anal - J_num) / rel_denom))
        assert max_abs_err < 1e-7, (
            f"Analytical Jacobian mismatch for {rA_id}-{rB_id}: max_abs_err = {max_abs_err:.2e} (expected < 1e-7)"
        )
        assert max_rel_err < 1e-5, (
            f"Analytical Jacobian relative mismatch for {rA_id}-{rB_id}: max_rel_err = {max_rel_err:.2e} (expected < 1e-5)"
        )


def test_coupled_jacobian_dimensions_and_signs(stage3_setup):
    """Verify coupled Jacobian dimensions and signs for 5+5 and 5+6 DOF pairs."""
    _, robots, _, _, _, _ = stage3_setup

    # 1. 5+5 DOF pair: robot1 + robot2 -> 10 joints, 10 task rows
    cfg1 = robots["robot1"]
    cfg2 = robots["robot2"]
    q1 = np.deg2rad(cfg1.home_deg)
    q2 = np.deg2rad(cfg2.home_deg)
    p_H = np.array([0.0, 100.0, 150.0])

    J_12, e_12, _ = compute_coupled_handover_jacobian(q1, q2, cfg1, cfg2, p_H_world=p_H)
    assert J_12.shape == (10, 10), f"Expected shape (10, 10), got {J_12.shape}"
    assert len(e_12) == 10

    # 2. 5+6 DOF pair: robot2 + robot3 -> 11 joints, 10 task rows
    cfg3 = robots["robot3"]
    q3 = np.deg2rad(cfg3.home_deg)
    J_23, e_23, _ = compute_coupled_handover_jacobian(q2, q3, cfg2, cfg3, p_H_world=p_H)
    assert J_23.shape == (10, 11), f"Expected shape (10, 11), got {J_23.shape}"
    assert len(e_23) == 10

    # 3. 6+5 DOF pair: robot3 + robot1 -> 11 joints, 10 task rows
    J_31, e_31, _ = compute_coupled_handover_jacobian(q3, q1, cfg3, cfg1, p_H_world=p_H)
    assert J_31.shape == (10, 11), f"Expected shape (10, 11), got {J_31.shape}"
    assert len(e_31) == 10

    # 4. Sign check: coincidence block J_p = [JvA, -JvB] / L_char
    # Moving joints of robot A contribute positive linear velocity to pA,
    # while robot B contributes negative linear velocity to (pA - pB)
    assert np.linalg.norm(J_12[:3, :5]) > 0.1
    assert np.linalg.norm(J_12[:3, 5:]) > 0.1


def test_analytical_solver_convergence_and_joint_limits(stage3_setup):
    """Verify analytical solver converges to valid solution respecting joint limits and clearance."""
    _, robots, _, _, _, auto_poses = stage3_setup
    np.random.seed(42)
    pose12 = auto_poses["robot1_robot2"]
    p_H = pose12.handover_position_world

    res = solve_coupled_handover_orientation_analytical(
        robot_a_id="robot1",
        robot_b_id="robot2",
        p_H_world=p_H,
        robots=robots,
        pos_tol_mm=1.0,
        ori_tol_deg=15.0,
        min_clearance_mm=45.0,
        max_opt_iterations=70,
    )
    assert res is not None, "Analytical solver failed on known reachable handover candidate."

    qA_deg, qB_deg, errA, errB, dot_z, dot_x, clr = res
    theta_tol_rad = np.deg2rad(15.0)
    eps_z = 1.0 - np.cos(theta_tol_rad)
    eps_x = np.sin(theta_tol_rad)

    assert errA <= 1.0, f"Robot A position error {errA:.3f} mm > 1.0 mm"
    assert errB <= 1.0, f"Robot B position error {errB:.3f} mm > 1.0 mm"
    assert dot_z <= (-1.0 + eps_z), f"Approach axis dot_z = {dot_z:.4f} > {-1.0 + eps_z:.4f}"
    assert dot_x <= eps_x, f"Finger orthogonality dot_x = {dot_x:.4f} > {eps_x:.4f}"
    assert clr >= 45.0, f"Body clearance {clr:.1f} mm < 45.0 mm"

    # Joint limits check
    cfg1 = robots["robot1"]
    cfg2 = robots["robot2"]
    assert np.all(qA_deg >= cfg1.joint_limits_deg[:, 0] - 1e-3)
    assert np.all(qA_deg <= cfg1.joint_limits_deg[:, 1] + 1e-3)
    assert np.all(qB_deg >= cfg2.joint_limits_deg[:, 0] - 1e-3)
    assert np.all(qB_deg <= cfg2.joint_limits_deg[:, 1] + 1e-3)


def test_fallback_solver_regression_and_dispatch(stage3_setup):
    """Verify that numerical L-BFGS-B fallback works directly and is dispatched properly."""
    _, robots, _, _, _, auto_poses = stage3_setup
    np.random.seed(42)
    pose12 = auto_poses["robot1_robot2"]
    p_H = pose12.handover_position_world

    # 1. Direct call to fallback solver
    res_lbfgs = solve_coupled_handover_orientation_lbfgs(
        robot_a_id="robot1",
        robot_b_id="robot2",
        p_H_world=p_H,
        robots=robots,
        pos_tol_mm=1.0,
        ori_tol_deg=15.0,
        min_clearance_mm=45.0,
        max_opt_iterations=50,
    )
    assert res_lbfgs is not None, "Fallback numerical L-BFGS-B solver failed on valid point."

    # 2. Dispatch via main API with solver_method="numerical"
    res_num = solve_coupled_handover_orientation(
        robot_a_id="robot1",
        robot_b_id="robot2",
        p_H_world=p_H,
        robots=robots,
        solver_method="numerical",
    )
    assert res_num is not None

    # 3. Dispatch via main API with solver_method="analytical"
    res_anal = solve_coupled_handover_orientation(
        robot_a_id="robot1",
        robot_b_id="robot2",
        p_H_world=p_H,
        robots=robots,
        solver_method="analytical",
    )
    assert res_anal is not None

    # 4. Infeasible candidate returns None without converting failure to success
    bad_p = np.array([1000.0, 1000.0, 1000.0])
    res_fail = solve_coupled_handover_orientation(
        robot_a_id="robot1",
        robot_b_id="robot2",
        p_H_world=bad_p,
        robots=robots,
        solver_method="analytical",
    )
    assert res_fail is None, "Infeasible candidate must return None, not a false success."


def test_auto_handover_cache_version_and_invalidation(stage3_setup):
    """Verify cache schema version 2.0 and invalidation on fingerprint change."""
    import json
    import os
    _, robots, _, auto_cache_path, config_hash, auto_poses = stage3_setup

    assert os.path.exists(auto_cache_path), "Auto handover cache file should exist."
    with open(auto_cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("version") == "2.0", f"Expected cache version 2.0, got {data.get('version')}"
    assert config_hash in data.get("fingerprints", {}), f"Fingerprint {config_hash} not found in cache."

    # Cache lookup with non-existent fingerprint returns None
    from robot_control.handover.stage3_handover_solver import load_auto_handover_cache
    miss = load_auto_handover_cache(auto_cache_path, "invalid_hash_12345")
    assert miss is None, "Cache lookup with invalid fingerprint must return None (cache miss)."

