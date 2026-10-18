"""Stage 3 Execution Correction Targeted Tests.

Covers the 12 required test conditions from Stage 3 Execution Correction:
1. Phase 1 and phases 7–9 do not require unnecessary global synchronization.
2. Phase 4 uses Cartesian waypoints and maintains the intended TCP orientation.
3. Phase 6 retraces the validated corridor.
4. Both robots reach their pre-handover poses before coordinated approach.
5. Transfer does not begin before the required handover conditions are met.
6. Robot 2's intermediate TCP orientation remains within the configured tolerance.
7. Joint solutions remain continuous and within limits.
8. Unsafe trajectories are rejected before command publication.
9. A waiting robot cannot be moved to an unvalidated waiting pose.
10. AUTO and MANUAL modes both use the same safety requirements.
11. A 7.1 mm predicted clearance blocks the relevant phase when it is below the configured threshold.
12. A feasible trajectory can be accepted without requiring every unrelated task phase to be synchronized.
"""

import os
import pytest
import numpy as np

from robot_control.handover.stage1 import (
    load_stage1_robots,
    generate_interpolated_path,
)
from robot_control.handover.stage1_5 import (
    Stage1_5Step,
    Stage1_5TrajectoryResult,
    compute_min_body_clearance,
    get_tcp_world_pose,
    generate_cartesian_corridor_trajectory,
    schedule_collision_aware_motion,
    plan_stage1_5_synchronized_paths,
    validate_execution_safety_gate,
)
from robot_control.handover.stage3 import (
    build_stage3_plan,
    plan_stage3_synchronized_paths,
)
from robot_control.handover.stage3_handover_solver import (
    load_auto_handover_cache,
    resolve_auto_handover_cache_path,
)
from robot_control.handover.trajectory_cache import (
    compute_config_fingerprint,
    resolve_cache_filepath,
)


@pytest.fixture(scope="module")
def setup_env():
    handover_cfg, robots, _ = load_stage1_robots()
    auto_cache_path = resolve_auto_handover_cache_path()
    cache_path = resolve_cache_filepath()
    config_hash = compute_config_fingerprint(robots, handover_cfg)
    auto_poses = load_auto_handover_cache(auto_cache_path, config_hash)
    return handover_cfg, robots, cache_path, auto_cache_path, config_hash, auto_poses


# =============================================================================
# 1. Phase 1 and phases 7-9 decoupled independent scheduling
# =============================================================================
def test_phase1_and_phase9_decoupled_synchronization(setup_env):
    """Test 1: Phase 1 and Phase 9 do not force unnecessary global lockstep synchronization.
    Concurrent independent motion is supported, and conflicts serialize with safe waiting.
    """
    handover_cfg, robots, cache_path, auto_cache_path, config_hash, auto_poses = setup_env
    cfg1 = robots["robot1"]
    cfg2 = robots["robot2"]

    # Test concurrent independent scheduling when safe
    start1 = np.zeros(cfg1.num_joints)
    target1 = np.array([25.0, 30.0, -80.0, 45.0, 0.0])
    start2 = np.zeros(cfg2.num_joints)
    target2 = np.array([122.68, 90.0, -19.23, -74.40, 90.0])

    steps, min_c, serialized = schedule_collision_aware_motion(
        start1, target1, start2, target2, cfg1, cfg2,
        safe_distance=50.0, steps=20,
        pick_rid="robot1", place_rid="robot2",
        phase_name="phase1_test",
        desc_concurrent="Concurrent move",
        desc_serialized_pick_first="Serialized pick first",
        desc_serialized_place_first="Serialized place first",
    )
    assert not serialized, "Non-conflicting paths should execute concurrently without forced serialization"
    assert min_c >= 50.0, "Concurrent steps must maintain >= safe_distance"
    assert len(steps) == 20

    # Test conflicting paths: scheduler must serialize motions with waiting
    # Place target directly inside Robot 1's path
    conflict_target1 = np.array([0.0, 45.0, -45.0, 0.0, 0.0])
    conflict_target2 = np.array([0.0, 45.0, -45.0, 0.0, 0.0])

    steps_ser, min_c_ser, serialized_ser = schedule_collision_aware_motion(
        start1, conflict_target1, start2, conflict_target2, cfg1, cfg2,
        safe_distance=50.0, steps=10,
        pick_rid="robot1", place_rid="robot2",
        phase_name="phase1_conflict",
        desc_concurrent="Concurrent move",
        desc_serialized_pick_first="Serialized pick first",
        desc_serialized_place_first="Serialized place first",
    )
    assert serialized_ser, "Conflicting paths must be serialized"
    # In serialized schedule, one robot moves while the other remains stationary (waiting)
    part1 = steps_ser[:10]
    part2 = steps_ser[10:]
    # During part 1, place robot is waiting at start2
    for s in part1:
        assert np.allclose(s.place_q, start2)
    # During part 2, pick robot is waiting at target1
    for s in part2:
        assert np.allclose(s.pick_q, conflict_target1)


# =============================================================================
# 2. Phase 4 uses Cartesian waypoints and maintains intended TCP orientation
# =============================================================================
def test_phase4_cartesian_corridor_and_orientation(setup_env):
    """Test 2: Phase 4 uses intermediate Cartesian waypoints along p(s) = p_pre + s*d*z_H
    and maintains intended TCP orientation within 15 deg tolerance.
    """
    _, robots, _, _, _, auto_poses = setup_env
    p12 = auto_poses["robot1_robot2"]
    cfg1 = robots["robot1"]
    cfg2 = robots["robot2"]

    ok1, traj1, msg1 = generate_cartesian_corridor_trajectory(
        "robot1", cfg1, p12.robot_a_handover_q_deg, standoff_mm=p12.standoff_distance_mm, steps=25
    )
    ok2, traj2, msg2 = generate_cartesian_corridor_trajectory(
        "robot2", cfg2, p12.robot_b_handover_q_deg, standoff_mm=p12.standoff_distance_mm, steps=25
    )
    assert ok1, f"Robot 1 corridor failed: {msg1}"
    assert ok2, f"Robot 2 corridor failed: {msg2}"
    assert len(traj1) == 25
    assert len(traj2) == 25

    p1_H, z1_H, _, _ = get_tcp_world_pose(p12.robot_a_handover_q_deg, cfg1)
    p1_pre = p1_H - p12.standoff_distance_mm * z1_H

    for s_idx, q in enumerate(traj1):
        s = s_idx / (len(traj1) - 1)
        expected_p = p1_pre + s * p12.standoff_distance_mm * z1_H
        actual_p, actual_z, _, _ = get_tcp_world_pose(q, cfg1)
        pos_err = np.linalg.norm(actual_p - expected_p)
        z_dot = np.dot(actual_z, z1_H)
        z_dev = np.rad2deg(np.arccos(np.clip(z_dot, -1.0, 1.0)))

        assert pos_err <= 1.0, f"Step {s_idx} pos error {pos_err:.2f} mm > 1.0 mm"
        assert z_dev <= 15.0, f"Step {s_idx} orientation error {z_dev:.2f}° > 15.0°"


# =============================================================================
# 3. Phase 6 retraces the validated corridor
# =============================================================================
def test_phase6_retraces_validated_corridor(setup_env):
    """Test 3: Phase 6 retraces the exact reverse of the validated Phase 4 corridor."""
    handover_cfg, robots, cache_path, auto_cache_path, config_hash, auto_poses = setup_env

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

    traj = plan_stage3_synchronized_paths(plan, robots, safe_distance=50.0, steps=20)
    p4 = traj.phase4_advance_to_handover
    p6 = traj.phase6_retract_to_pre_handover

    assert len(p4) == len(p6), "Phase 4 and Phase 6 must have equal step counts"

    n = len(p4)
    for i in range(n):
        step4 = p4[i]
        step6 = p6[n - 1 - i]
        assert np.allclose(step4.pick_q, step6.pick_q, atol=1e-3), (
            f"Phase 6 step {n-1-i} pick_q does not match Phase 4 step {i}"
        )
        assert np.allclose(step4.place_q, step6.place_q, atol=1e-3), (
            f"Phase 6 step {n-1-i} place_q does not match Phase 4 step {i}"
        )


# =============================================================================
# 4. Both robots reach pre-handover poses before coordinated approach
# =============================================================================
def test_pre_handover_reached_before_coordinated_approach(setup_env):
    """Test 4: Both participating robots must be at their validated pre-handover poses
    at the start of Phase 4.
    """
    handover_cfg, robots, cache_path, auto_cache_path, config_hash, auto_poses = setup_env

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

    traj = plan_stage3_synchronized_paths(plan, robots, safe_distance=50.0, steps=20)

    # In Phase 1, place robot reaches place_pre_handover_q
    last_p1 = traj.phase1_simultaneous[-1]
    assert np.allclose(last_p1.place_q, plan.place_pre_handover_q, atol=1e-2), (
        "Place robot must reach pre-handover pose by end of Phase 1"
    )

    # In Phase 3, pick robot reaches pick_pre_handover_q while place robot stays at pre-handover
    last_p3 = traj.phase3_pick_to_pre_handover[-1]
    assert np.allclose(last_p3.pick_q, plan.pick_pre_handover_q, atol=1e-2), (
        "Pick robot must reach pre-handover pose by end of Phase 3"
    )
    assert np.allclose(last_p3.place_q, plan.place_pre_handover_q, atol=1e-2), (
        "Place robot must remain stationed at pre-handover pose during Phase 3"
    )

    # First step of Phase 4 must start at both pre-handover poses
    first_p4 = traj.phase4_advance_to_handover[0]
    assert np.allclose(first_p4.pick_q, plan.pick_pre_handover_q, atol=1e-2)
    assert np.allclose(first_p4.place_q, plan.place_pre_handover_q, atol=1e-2)


# =============================================================================
# 5. Transfer does not begin before required handover conditions are met
# =============================================================================
def test_transfer_requires_verified_handover_conditions(setup_env):
    """Test 5: Transfer (Phase 5) requires Phase 4 to have completed with both robots
    at their validated handover mating poses.
    """
    handover_cfg, robots, cache_path, auto_cache_path, config_hash, auto_poses = setup_env

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

    traj = plan_stage3_synchronized_paths(plan, robots, safe_distance=50.0, steps=20)
    last_p4 = traj.phase4_advance_to_handover[-1]

    # Must match plan handover joint vectors
    assert np.allclose(last_p4.pick_q, plan.pick_handover_q, atol=1e-2), (
        "Pick robot did not reach handover mating configuration at end of Phase 4"
    )
    assert np.allclose(last_p4.place_q, plan.place_handover_q, atol=1e-2), (
        "Place robot did not reach handover mating configuration at end of Phase 4"
    )

    # Check coincidence at end of Phase 4
    cfg1 = robots[plan.pick_robot]
    cfg2 = robots[plan.place_robot]
    p1, _, _, _ = get_tcp_world_pose(last_p4.pick_q, cfg1)
    p2, _, _, _ = get_tcp_world_pose(last_p4.place_q, cfg2)
    coinc = np.linalg.norm(p1 - p2)
    assert coinc <= 1.0, f"TCPs not coincident at handover point: {coinc:.3f} mm > 1.0 mm"


# =============================================================================
# 6. Robot 2's intermediate TCP orientation remains within configured tolerance
# =============================================================================
def test_robot2_intermediate_tcp_orientation_within_tolerance(setup_env):
    """Test 6: Diagnose & verify Robot 2's rotation correction:
    Robot 2 does not flip branch (q1 delta < 20 deg), and intermediate TCP approach
    axis orientation stays within 15 deg tolerance throughout approach.
    """
    _, robots, _, _, _, auto_poses = setup_env
    p12 = auto_poses["robot1_robot2"]
    cfg2 = robots["robot2"]

    q2_H = p12.robot_b_handover_q_deg
    q2_pre = p12.robot_b_pre_handover_q_deg

    # Verify no 180 deg branch flip in cached pose
    delta_q = np.abs(q2_H - q2_pre)
    assert delta_q[0] < 20.0, f"Robot 2 joint 1 flipped branch by {delta_q[0]:.1f}° (must be < 20°)"

    ok2, traj2, msg = generate_cartesian_corridor_trajectory(
        "robot2", cfg2, q2_H, q_pre_deg=q2_pre, standoff_mm=p12.standoff_distance_mm, steps=20
    )
    assert ok2, f"Corridor generation failed: {msg}"

    _, z2_H, _, _ = get_tcp_world_pose(q2_H, cfg2)

    for i, q in enumerate(traj2):
        _, z_act, _, _ = get_tcp_world_pose(q, cfg2)
        dev = np.rad2deg(np.arccos(np.clip(np.dot(z_act, z2_H), -1.0, 1.0)))
        assert dev <= 15.0, f"Robot 2 intermediate orientation deviation {dev:.2f}° exceeded 15.0° at step {i}"


# =============================================================================
# 7. Joint solutions remain continuous and within limits
# =============================================================================
def test_joint_solutions_continuous_and_within_limits(setup_env):
    """Test 7: Across all trajectory phases, joint angles strictly remain within limits
    and step-to-step joint deltas are bounded (< 15 deg).
    """
    handover_cfg, robots, cache_path, auto_cache_path, config_hash, auto_poses = setup_env

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

    traj = plan_stage3_synchronized_paths(plan, robots, safe_distance=50.0, steps=20)
    cfg_pick = robots[plan.pick_robot]
    cfg_place = robots[plan.place_robot]

    phases = [
        traj.phase1_simultaneous,
        traj.phase3_pick_to_pre_handover,
        traj.phase4_advance_to_handover,
        traj.phase6_retract_to_pre_handover,
        traj.phase7_handover_to_place,
        traj.phase9_simultaneous_home,
    ]

    for phase in phases:
        for i, step in enumerate(phase):
            # Check limits
            assert np.all(step.pick_q >= cfg_pick.joint_limits_deg[:, 0] - 1e-3)
            assert np.all(step.pick_q <= cfg_pick.joint_limits_deg[:, 1] + 1e-3)
            assert np.all(step.place_q >= cfg_place.joint_limits_deg[:, 0] - 1e-3)
            assert np.all(step.place_q <= cfg_place.joint_limits_deg[:, 1] + 1e-3)

            # Check step continuity within phase
            if i > 0:
                d_pick = np.max(np.abs(step.pick_q - phase[i-1].pick_q))
                d_place = np.max(np.abs(step.place_q - phase[i-1].place_q))
                assert d_pick <= 15.0, f"Pick joint jump {d_pick:.1f}° at step {i}"
                assert d_place <= 15.0, f"Place joint jump {d_place:.1f}° at step {i}"


# =============================================================================
# 8. Unsafe trajectories are rejected before command publication
# =============================================================================
def test_unsafe_trajectories_rejected_before_command_publication(setup_env):
    """Test 8: Programmatic execution safety gate rejects trajectories containing unsafe steps."""
    _, robots, _, _, _, _ = setup_env
    cfg1 = robots["robot1"]
    cfg2 = robots["robot2"]

    # Trajectory with one unsafe step
    step_unsafe = Stage1_5Step("phase4_advance_to_handover", np.zeros(5), np.zeros(5), 20.0, False, "collision")
    step_safe = Stage1_5Step("phase4_advance_to_handover", np.zeros(5), np.zeros(5), 60.0, True, "safe")

    traj = Stage1_5TrajectoryResult(
        needs_handover=True,
        pick_robot="robot1",
        place_robot="robot2",
        phase1_simultaneous=[step_safe],
        phase3_pick_to_pre_handover=[step_safe],
        phase4_advance_to_handover=[step_safe, step_unsafe],
        phase6_retract_to_pre_handover=[step_safe],
        phase7_handover_to_place=[step_safe],
        phase9_simultaneous_home=[step_safe],
        min_overall_clearance_mm=20.0,
        all_phases_safe=False,
        collision_avoidance_applied=False,
    )

    is_safe, msg = validate_execution_safety_gate(traj, robots)
    assert not is_safe, "Safety gate must block execution when unsafe step exists"
    assert "Execution rejected" in msg


# =============================================================================
# 9. A waiting robot cannot be moved to an unvalidated waiting pose
# =============================================================================
def test_waiting_robot_cannot_move_to_unvalidated_pose(setup_env):
    """Test 9: Waiting robot must remain strictly stationed at its validated waiting pose."""
    handover_cfg, robots, cache_path, auto_cache_path, config_hash, auto_poses = setup_env

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

    traj = plan_stage3_synchronized_paths(plan, robots, safe_distance=50.0, steps=20)

    # In Phase 3, place robot is waiting. All steps must equal place_pre_handover_q exactly.
    for step in traj.phase3_pick_to_pre_handover:
        assert np.allclose(step.place_q, plan.place_pre_handover_q), (
            "Waiting robot in Phase 3 moved from validated waiting pose"
        )

    # In Phase 7, pick robot is waiting. All steps must equal pick_pre_handover_q exactly.
    for step in traj.phase7_handover_to_place:
        assert np.allclose(step.pick_q, plan.pick_pre_handover_q), (
            "Waiting robot in Phase 7 moved from validated waiting pose"
        )


# =============================================================================
# 10. AUTO and MANUAL modes both use the same safety requirements
# =============================================================================
def test_auto_and_manual_mode_safety_gate_parity(setup_env):
    """Test 10: AUTO mode and MANUAL mode plans undergo identical safety gate validation."""
    handover_cfg, robots, cache_path, auto_cache_path, config_hash, auto_poses = setup_env

    def chooser(label, candidates):
        return "robot1" if label == "pick" else "robot2"

    # Plan with AUTO mode
    plan_auto, _, _ = build_stage3_plan(
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
    traj_auto = plan_stage3_synchronized_paths(plan_auto, robots, safe_distance=50.0, steps=15)
    safe_auto, msg_auto = validate_execution_safety_gate(traj_auto, robots, plan=plan_auto)

    # Plan with MANUAL mode
    plan_manual, _, _ = build_stage3_plan(
        pick_xyz=[100.0, 100.0, 200.0],
        place_xyz=[-50.0, 180.0, 150.0],
        robots=robots,
        handover_cfg=handover_cfg,
        chooser=chooser,
        use_auto_handover=False,
        cache_filepath=cache_path,
        config_fingerprint=config_hash,
    )
    traj_manual = plan_stage3_synchronized_paths(plan_manual, robots, safe_distance=50.0, steps=15)
    safe_manual, msg_manual = validate_execution_safety_gate(traj_manual, robots, plan=plan_manual)

    # Both must pass through validate_execution_safety_gate without bypass
    assert isinstance(safe_auto, bool)
    assert isinstance(safe_manual, bool)
    assert safe_auto, f"AUTO mode plan failed safety gate: {msg_auto}"
    assert safe_manual, f"MANUAL mode plan failed safety gate: {msg_manual}"


# =============================================================================
# 11. A 7.1 mm predicted clearance blocks the relevant phase when below threshold
# =============================================================================
def test_7_1mm_predicted_clearance_blocks_unsafe_phase(setup_env):
    """Test 11: A 7.1 mm predicted clearance correctly blocks the phase when safe_distance is 50 mm.
    Safety gate must NOT bypass checks for low clearances.
    """
    _, robots, _, _, _, _ = setup_env
    cfg1 = robots["robot1"]
    cfg2 = robots["robot2"]

    # Construct step with 7.1 mm clearance
    step_7_1mm = Stage1_5Step("phase1_simultaneous", np.zeros(5), np.zeros(5), 7.1, False, "7.1 mm clearance")
    step_safe = Stage1_5Step("phase1_simultaneous", np.zeros(5), np.zeros(5), 60.0, True, "safe step")

    traj = Stage1_5TrajectoryResult(
        needs_handover=True,
        pick_robot="robot1",
        place_robot="robot2",
        phase1_simultaneous=[step_safe, step_7_1mm],
        phase3_pick_to_pre_handover=[step_safe],
        phase4_advance_to_handover=[step_safe],
        phase6_retract_to_pre_handover=[step_safe],
        phase7_handover_to_place=[step_safe],
        phase9_simultaneous_home=[step_safe],
        min_overall_clearance_mm=7.1,
        all_phases_safe=False,
        collision_avoidance_applied=False,
    )

    is_safe, msg = validate_execution_safety_gate(traj, robots)
    assert not is_safe, "7.1 mm clearance must block execution safety gate"
    assert "7.1" in msg or "all_phases_safe is False" in msg


# =============================================================================
# 12. Feasible trajectory accepted without global synchronization
# =============================================================================
def test_feasible_trajectory_accepted_without_global_synchronization(setup_env):
    """Test 12: A feasible multi-robot task plan is accepted and passes the execution safety gate
    without requiring all unrelated phases to be locked in global synchronization.
    """
    handover_cfg, robots, cache_path, auto_cache_path, config_hash, auto_poses = setup_env

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

    traj = plan_stage3_synchronized_paths(plan, robots, safe_distance=50.0, steps=20)
    assert traj.all_phases_safe, f"Feasible trajectory flagged unsafe: min clr={traj.min_overall_clearance_mm:.1f} mm"

    is_safe, msg = validate_execution_safety_gate(traj, robots, plan=plan)
    assert is_safe, f"Feasible trajectory failed safety gate: {msg}"
    assert "Safety gate passed" in msg
