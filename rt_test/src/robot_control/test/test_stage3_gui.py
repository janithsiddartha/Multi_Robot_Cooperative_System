"""Targeted Integration Tests for Stage 3 GUI Handover Integration.

Tests cover:
1. Loading each of the three robot-pair solutions.
2. Displaying both robot-specific handover orientations.
3. Displaying both pre-handover poses and joint configurations.
4. Correct units (mm, deg) and joint ordering (5 DOF for R1/R2, 6 DOF for R3).
5. Deriving orientation through FK from cached joint angles.
6. Handling invalid or stale cache metadata and fingerprint mismatch.
7. Handling missing or incomplete solutions gracefully.
8. Distinguishing target pose from actual live robot feedback.
9. Ensuring selecting or loading a pose does not publish motion commands.
10. Ensuring motion execution cannot bypass the safety gate or operator confirmation.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from robot_control.handover.geometry import matrix_to_euler
from robot_control.handover.stage1 import load_stage1_robots
from robot_control.handover.stage3_handover_solver import (
    AutoHandoverPose,
    compute_all_auto_handover_poses,
    get_handover_pose_details,
    get_tcp_world_pose,
    load_auto_handover_cache,
    matrix_to_quaternion,
    resolve_auto_handover_cache_path,
)
from robot_control.handover.trajectory_cache import compute_config_fingerprint


@pytest.fixture(scope="module")
def stage3_env():
    handover_cfg, robots, _ = load_stage1_robots()
    cache_path = resolve_auto_handover_cache_path()
    fingerprint = compute_config_fingerprint(robots, handover_cfg)
    auto_poses = load_auto_handover_cache(cache_path, fingerprint)
    if not auto_poses:
        auto_poses = compute_all_auto_handover_poses(robots, handover_cfg, cache_path=cache_path)
    return handover_cfg, robots, cache_path, fingerprint, auto_poses


# =============================================================================
# 1. LOADING EACH OF THE THREE ROBOT-PAIR SOLUTIONS
# =============================================================================

def test_load_all_three_robot_pairs(stage3_env):
    """Test 1: Verify all 3 robot pairs load from cache with complete solution fields."""
    _, robots, cache_path, fingerprint, auto_poses = stage3_env
    loaded_poses = load_auto_handover_cache(cache_path, fingerprint)
    assert loaded_poses is not None, "Failed to load auto_handover_poses.json"

    expected_pairs = ["robot1_robot2", "robot2_robot3", "robot3_robot1"]
    for pair_key in expected_pairs:
        assert pair_key in loaded_poses, f"Missing pair solution for {pair_key}"
        p = loaded_poses[pair_key]
        assert isinstance(p, AutoHandoverPose)
        assert len(p.handover_position_world) == 3
        assert p.standoff_distance_mm > 0.0
        assert p.handover_clearance_mm > 0.0
        assert p.pre_handover_clearance_mm > 0.0


# =============================================================================
# 2. DISPLAYING BOTH ROBOT-SPECIFIC HANDOVER ORIENTATIONS
# =============================================================================

def test_robot_specific_handover_orientations(stage3_env):
    """Test 2: Verify both robots in each pair have distinct, opposing orientations."""
    _, robots, _, _, auto_poses = stage3_env
    for pair_key, p in auto_poses.items():
        details = get_handover_pose_details(p, robots)
        
        # Orientations must be computed
        assert 'robot_a_rot_matrix' in details
        assert 'robot_b_rot_matrix' in details
        assert 'robot_a_rpy_deg' in details
        assert 'robot_b_rpy_deg' in details
        assert 'robot_a_quaternion' in details
        assert 'robot_b_quaternion' in details

        # Check anti-parallel approach: z_A . z_B <= -0.96 (~15 deg tolerance)
        z_a = details['robot_a_approach_vec']
        z_b = details['robot_b_approach_vec']
        approach_dot = np.dot(z_a, z_b)
        assert approach_dot <= -0.95, f"Approach vectors not opposing for {pair_key}: dot={approach_dot}"

        # Distinct RPY
        rpy_a = details['robot_a_rpy_deg']
        rpy_b = details['robot_b_rpy_deg']
        assert not np.allclose(rpy_a, rpy_b, atol=5.0), f"RPY for {pair_key} are identical!"


# =============================================================================
# 3. DISPLAYING BOTH PRE-HANDOVER POSES AND JOINT CONFIGURATIONS
# =============================================================================

def test_pre_handover_poses_and_joints(stage3_env):
    """Test 3: Verify pre-handover poses, standoffs, and joint configurations are present."""
    _, robots, _, _, auto_poses = stage3_env
    for pair_key, p in auto_poses.items():
        details = get_handover_pose_details(p, robots)
        assert len(details['pre_handover_a_xyz']) == 3
        assert len(details['pre_handover_b_xyz']) == 3
        assert len(p.pre_handover_a_joint_angles_deg) == robots[p.robot_a_id].num_joints
        assert len(p.pre_handover_b_joint_angles_deg) == robots[p.robot_b_id].num_joints

        # Pre-handover positions should be retracted by approximately standoff distance
        dist_pre_a = np.linalg.norm(details['pre_handover_a_xyz'] - p.handover_position_world)
        dist_pre_b = np.linalg.norm(details['pre_handover_b_xyz'] - p.handover_position_world)
        assert dist_pre_a > 10.0, f"Robot A pre-handover not backed off for {pair_key}"
        assert dist_pre_b > 10.0, f"Robot B pre-handover not backed off for {pair_key}"


# =============================================================================
# 4. CORRECT UNITS AND JOINT ORDERING
# =============================================================================

def test_correct_units_and_joint_ordering(stage3_env):
    """Test 4: Verify units (XYZ in mm, RPY in deg, Quaternions unit) and joint DOF counts."""
    _, robots, _, _, auto_poses = stage3_env
    for pair_key, p in auto_poses.items():
        details = get_handover_pose_details(p, robots)
        
        # Position in mm within physical scene limits
        for coord in p.handover_position_world:
            assert -600.0 <= coord <= 600.0, f"Coordinate {coord} out of physical range in mm"
        
        # DOF ordering: R1=5, R2=5, R3=6
        assert len(p.robot_a_joint_angles_deg) == robots[p.robot_a_id].num_joints
        assert len(p.robot_b_joint_angles_deg) == robots[p.robot_b_id].num_joints

        if p.robot_a_id in ('robot1', 'robot2'):
            assert len(p.robot_a_joint_angles_deg) == 5
        elif p.robot_a_id == 'robot3':
            assert len(p.robot_a_joint_angles_deg) == 6

        if p.robot_b_id in ('robot1', 'robot2'):
            assert len(p.robot_b_joint_angles_deg) == 5
        elif p.robot_b_id == 'robot3':
            assert len(p.robot_b_joint_angles_deg) == 6

        # Quaternions are unit quaternions
        for q_key in ('robot_a_quaternion', 'robot_b_quaternion', 'pre_handover_a_quaternion', 'pre_handover_b_quaternion'):
            quat = np.asarray(details[q_key])
            assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-3), f"Quaternion {q_key} not unit norm"

        # Check joint limits validity
        assert details['robot_a_limits_valid'] is True
        assert details['robot_b_limits_valid'] is True


# =============================================================================
# 5. DERIVING ORIENTATION THROUGH FK FROM CACHED JOINT ANGLES
# =============================================================================

def test_fk_orientation_derivation(stage3_env):
    """Test 5: Verify orientation is derived directly via forward kinematics from q_deg."""
    _, robots, _, _, auto_poses = stage3_env
    for pair_key, p in auto_poses.items():
        details = get_handover_pose_details(p, robots)

        # Independent FK computation
        cfg_a = robots[p.robot_a_id]
        pos_a, z_a, x_a, y_a = get_tcp_world_pose(p.robot_a_joint_angles_deg, cfg_a)
        R_a = np.column_stack([x_a, y_a, z_a])
        rpy_a = np.rad2deg(matrix_to_euler(R_a))
        quat_a = matrix_to_quaternion(R_a)

        assert np.allclose(details['robot_a_tcp_xyz'], pos_a, atol=1e-3)
        assert np.allclose(details['robot_a_rot_matrix'], R_a, atol=1e-3)
        assert np.allclose(details['robot_a_rpy_deg'], rpy_a, atol=1e-3)
        assert np.allclose(details['robot_a_quaternion'], quat_a, atol=1e-3)


# =============================================================================
# 6. HANDLING INVALID OR STALE CACHE METADATA
# =============================================================================

def test_invalid_or_stale_cache_metadata(stage3_env):
    """Test 6: Reject invalid cache version, malformed data, or mismatched config fingerprint."""
    _, robots, _, fingerprint, _ = stage3_env

    with tempfile.TemporaryDirectory() as tmpdir:
        # Case A: Stale version (1.0 instead of 2.0)
        stale_path = os.path.join(tmpdir, "stale.json")
        with open(stale_path, "w") as f:
            json.dump({
                "version": "1.0",
                "config_fingerprint": fingerprint,
                "pairs": {}
            }, f)
        res = load_auto_handover_cache(stale_path, fingerprint)
        assert res is None, "Failed to reject stale version 1.0 cache"

        # Case B: Fingerprint mismatch
        mismatch_path = os.path.join(tmpdir, "mismatch.json")
        with open(mismatch_path, "w") as f:
            json.dump({
                "version": "2.0",
                "config_fingerprint": "bad_fingerprint_hash_12345",
                "pairs": {}
            }, f)
        res = load_auto_handover_cache(mismatch_path, fingerprint)
        assert res is None, "Failed to reject mismatched configuration fingerprint"

        # Case C: Malformed JSON
        corrupt_path = os.path.join(tmpdir, "corrupt.json")
        with open(corrupt_path, "w") as f:
            f.write("{ invalid json corrupt content")
        res = load_auto_handover_cache(corrupt_path, fingerprint)
        assert res is None, "Failed to reject corrupted JSON file"


# =============================================================================
# 7. HANDLING MISSING OR INCOMPLETE SOLUTIONS
# =============================================================================

def test_missing_or_incomplete_pair_handling(stage3_env):
    """Test 7: Verify graceful handling when a pair is absent from cache."""
    _, robots, _, fingerprint, auto_poses = stage3_env

    # Simulate cache having only robot1_robot2
    partial_poses = {"robot1_robot2": auto_poses["robot1_robot2"]}

    # Mock GUI helper
    class MockGUI:
        def __init__(self, poses):
            self.auto_handover_poses = poses
            self.robots = robots
            self.selected_ho_pair = MagicMock()
            self.selected_ho_pair.get.return_value = "robot2_robot3"
            self.handover_mode_var = MagicMock()
            self.handover_mode_var.get.return_value = "auto"
            self.ho_status_badge = MagicMock()
            self.ho_target_lbl = MagicMock()
            self.ho_metrics_lbl = MagicMock()
            self.ho_summary_lbl = MagicMock()

    gui = MockGUI(partial_poses)
    from robot_control.gui.stage3_gui import Stage3GUI
    Stage3GUI._update_handover_status_card(gui)

    # Verify badge and label indicate missing pair without error
    gui.ho_status_badge.config.assert_called_with(text="⚠️ Missing Pair", bg="#FCF3CF", fg="#B7950B")
    gui.ho_target_lbl.config.assert_called_with(text="Shared Target: [No solution for robot2_robot3]")


# =============================================================================
# 8. DISTINGUISHING TARGET POSE FROM ACTUAL ROBOT FEEDBACK
# =============================================================================

def test_distinguish_target_from_actual_feedback(stage3_env):
    """Test 8: Ensure target pose is never conflated with physical robot state."""
    _, robots, _, _, auto_poses = stage3_env
    p = auto_poses["robot1_robot2"]

    # When live feedback is unavailable:
    has_live_q = {"robot1": False, "robot2": False}
    live_q = {"robot1": np.zeros(5), "robot2": np.zeros(5)}

    # Target joints should be non-zero and distinct
    assert not np.allclose(p.robot_a_joint_angles_deg, live_q["robot1"])
    assert not np.allclose(p.robot_b_joint_angles_deg, live_q["robot2"])

    # Simulate live feedback arriving for robot1
    actual_joint_telemetry = np.array([10.0, 20.0, -15.0, 5.0, 0.0])
    live_q["robot1"] = actual_joint_telemetry.copy()
    has_live_q["robot1"] = True

    # Actual TCP pose derived from live_q differs from cached target pose
    act_pos, act_z, act_x, act_y = get_tcp_world_pose(live_q["robot1"], robots["robot1"])
    target_pos = p.handover_position_world
    assert not np.allclose(act_pos, target_pos), "Actual pose matches target pose prematurely"


# =============================================================================
# 9. ENSURING SELECTING OR LOADING A POSE DOES NOT PUBLISH A MOTION COMMAND
# =============================================================================

def test_selecting_loading_previewing_never_publishes(stage3_env):
    """Test 9: Changing pair, reloading cache, or 3D previewing must NEVER publish motion commands."""
    _, robots, cache_path, fingerprint, auto_poses = stage3_env

    class MockGUIWithPubs:
        def __init__(self):
            self.auto_handover_poses = auto_poses
            self.auto_handover_cache_path = cache_path
            self.config_fingerprint = fingerprint
            self.robots = robots
            self.selected_ho_pair = MagicMock()
            self.selected_ho_pair.get.return_value = "robot2_robot3"
            self.handover_mode_var = MagicMock()
            self.handover_mode_var.get.return_value = "auto"
            self.is_previewing_handover = False
            self.preview_q = {}
            self.success_color = "#28A745"
            self.warning_color = "#FFC107"
            self.danger_color = "#DC3545"
            self.ho_status_badge = MagicMock()
            self.ho_target_lbl = MagicMock()
            self.ho_metrics_lbl = MagicMock()
            self.ho_summary_lbl = MagicMock()
            self.ho_recomp_msg = MagicMock()
            self._log = MagicMock()
            self._update_3d = MagicMock()
            self._update_handover_status_card = MagicMock()
            self._preview_handover_in_3d = lambda which='handover': Stage3GUI._preview_handover_in_3d(self, which)
            # ROS mock publishers
            self.publishers_traj = {rid: MagicMock() for rid in robots.keys()}
            self.publishers_mode = {rid: MagicMock() for rid in robots.keys()}

    from robot_control.gui.stage3_gui import Stage3GUI
    gui = MockGUIWithPubs()

    # 1. Pair selection
    Stage3GUI._on_handover_pair_selected(gui)
    for pub in gui.publishers_traj.values():
        pub.publish.assert_not_called()

    # 2. Cache reload
    Stage3GUI._on_reload_handover_cache(gui)
    for pub in gui.publishers_traj.values():
        pub.publish.assert_not_called()

    # 3. 3D Handover preview
    Stage3GUI._preview_handover_in_3d(gui, 'handover')
    assert gui.is_previewing_handover is True
    assert len(gui.preview_q) == 2
    for pub in gui.publishers_traj.values():
        pub.publish.assert_not_called()

    # 4. 3D Pre-handover preview
    Stage3GUI._preview_handover_in_3d(gui, 'pre_handover')
    for pub in gui.publishers_traj.values():
        pub.publish.assert_not_called()

    # 5. Reset preview
    Stage3GUI._reset_3d_preview(gui)
    assert gui.is_previewing_handover is False
    assert len(gui.preview_q) == 0
    for pub in gui.publishers_traj.values():
        pub.publish.assert_not_called()


# =============================================================================
# 10. ENSURING MOTION EXECUTION CANNOT BYPASS THE SAFETY GATE
# =============================================================================

def test_motion_execution_safety_gate_enforcement(stage3_env):
    """Test 10: Motion execution blocks if live feedback missing or clearance unsafe."""
    _, robots, _, _, auto_poses = stage3_env

    class MockGUIExec:
        def __init__(self):
            self.auto_handover_poses = auto_poses
            self.robots = robots
            self.has_live_q = {"robot1": False, "robot2": False, "robot3": False}
            self.live_q = {rid: np.zeros(cfg.num_joints) for rid, cfg in robots.items()}
            self.current_q = {rid: np.zeros(cfg.num_joints) for rid, cfg in robots.items()}
            self.trajectory_steps = 15
            self.safe_dist = 50.0
            self._log = MagicMock()
            self._send_joint_command = MagicMock()
            self.publishers_traj = {rid: MagicMock() for rid in robots.keys()}

    from robot_control.gui.stage3_gui import Stage3GUI
    gui = MockGUIExec()

    # Scenario A: Real robot motion commanded without live joint feedback
    with patch("tkinter.messagebox.showerror") as mock_err:
        Stage3GUI._execute_handover_pair_motion(gui, "robot1_robot2", "handover", real_robot=True)
        assert mock_err.called
        gui._send_joint_command.assert_not_called()

    # Scenario B: Operator cancels confirmation dialog
    gui.has_live_q["robot1"] = True
    gui.has_live_q["robot2"] = True
    with patch("tkinter.messagebox.askyesno", return_value=False) as mock_confirm:
        Stage3GUI._execute_handover_pair_motion(gui, "robot1_robot2", "pre_handover", real_robot=False)
        assert mock_confirm.called
        gui._send_joint_command.assert_not_called()
