"""Targeted Tests for Stage 3 Test 2 Environment (Top-of-Workspace Only).

Covers:
1. Candidate Selection: Strictly TOP_OF_WORKSPACE (descending Z, no centroid or center strategy).
2. Wrist roll optimization for orthogonal fingers.
3. Pre-handover calculation strictly along TCP Z axes.
4. Planning & trajectory execution using build_stage3_test2_plan and plan_stage3_test2_paths.
5. Verification that radar widgets and candidate strategy selector are absent in Stage3Test2GUI.
"""

import numpy as np
import pytest

from robot_control.handover.stage1 import load_stage1_robots
from robot_control.handover.stage1_5 import validate_execution_safety_gate
from robot_control.handover.stage3_test2 import (
    Stage3Test2Plan,
    build_stage3_test2_plan,
    plan_stage3_test2_paths,
)
from robot_control.handover.stage3_test2_solver import (
    AutoHandoverTest2Pose,
    get_test2_candidate_points,
    optimize_wrist_for_orthogonal_fingers,
    solve_pair_test2_handover,
)
from robot_control.handover.workspace import PositionWorkspaceComputer


@pytest.fixture(scope="module")
def setup_test2_env():
    handover_cfg, robots, _ = load_stage1_robots()
    int_robots = {1: robots['robot1'], 2: robots['robot2'], 3: robots['robot3']}
    ws_config = {
        'workspace': {'sampling': {'voxel_size_mm': 25.0, 'workspace_samples': 20000}},
        'robots': handover_cfg.get('robots', {}) if handover_cfg else {}
    }
    ws = PositionWorkspaceComputer(ws_config, int_robots)
    ws.compute()
    return handover_cfg, robots, ws


def test_top_of_workspace_candidate_selection(setup_test2_env):
    """Test 1: Candidate points are strictly sorted descending by Z."""
    _, robots, ws = setup_test2_env
    pts = ws.get_common_voxel_centers(1, 2)
    assert len(pts) > 0

    cfgA = robots['robot1']
    cfgB = robots['robot2']

    cand_pts = get_test2_candidate_points(pts, cfgA=cfgA, cfgB=cfgB)
    assert len(cand_pts) > 0

    # Must be monotonically non-increasing in Z
    z_diffs = np.diff(cand_pts[:, 2])
    assert np.all(z_diffs <= 1e-6), "get_test2_candidate_points must be sorted strictly descending by Z"
    assert cand_pts[0][2] >= 200.0, "Top reachable candidate should have high Z"


def test_orthogonal_wrist_optimization(setup_test2_env):
    """Test 2: Wrist roll joint sweep adjusts Robot 1 fingers orthogonal to Robot 2."""
    _, robots, _ = setup_test2_env
    cfg1 = robots['robot1']

    q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    z_dummy = np.array([0.0, 0.0, -1.0])
    x_robot2 = np.array([1.0, 0.0, 0.0])

    q_opt = optimize_wrist_for_orthogonal_fingers(
        robot_id="robot1",
        cfg=cfg1,
        q_deg=q_init,
        z_target=z_dummy,
        x_other=x_robot2,
    )

    from robot_control.handover.stage3_handover_solver import get_tcp_world_pose
    _, _, x_w, _ = get_tcp_world_pose(q_opt, cfg1, in_degrees=True)
    dot_x = abs(float(np.dot(x_w, x_robot2)))
    assert dot_x <= 0.25, f"Optimized wrist finger orthogonality should be near 0, got {dot_x:.3f}"


def test_test2_handover_cache_io(setup_test2_env):
    """Test 3: Verifies loading, serialization, and deserialization for test2 cache."""
    from robot_control.handover.stage3_test2_solver import (
        resolve_test2_handover_cache_path,
        load_test2_handover_cache,
        serialize_test2_auto_handover,
        deserialize_test2_auto_handover,
    )
    from robot_control.handover.trajectory_cache import compute_config_fingerprint

    handover_cfg, robots, _ = setup_test2_env
    cache_path = resolve_test2_handover_cache_path()
    assert cache_path.endswith("auto_handover_poses_test2.json")

    fp = compute_config_fingerprint(robots, handover_cfg)
    poses = load_test2_handover_cache(cache_path, fp)
    assert poses is not None, "Failed to load poses from auto_handover_poses_test2.json"
    assert "robot1_robot2" in poses

    p = poses["robot1_robot2"]
    assert p.candidate_origin == "TOP_OF_WORKSPACE"
    assert p.handover_clearance_mm >= 40.0
    assert p.error_a_mm <= 1.0
    assert p.error_b_mm <= 1.0

    d = serialize_test2_auto_handover(p)
    assert d["candidate_origin"] == "TOP_OF_WORKSPACE"
    p2 = deserialize_test2_auto_handover(d)
    assert p2.pair_name == p.pair_name
    assert np.allclose(p2.robot_a_handover_q_deg, p.robot_a_handover_q_deg)


def test_stage3_test2_gui_streamlined_structure():
    """Test 4: Stage3Test2GUI eliminates radar widgets and candidate strategy selector."""
    import tkinter as tk
    import rclpy
    from robot_control.gui.stage3_test2_gui import Stage3Test2GUI

    if not rclpy.ok():
        rclpy.init()

    try:
        root = tk.Tk()
        root.withdraw()
        gui = Stage3Test2GUI(root)

        # 1. Radar is completely removed
        assert gui.radar_ax is None, "Radar polar axis must not be initialized"
        assert gui.radar_fig is None, "Radar figure must not be initialized"
        assert gui.radar_canvas is None, "Radar canvas must not be created"

        # 2. Strategy selector is removed from UI
        assert not hasattr(gui, 'test_strategy_var'), "Candidate strategy selector must be removed"

        # 3. Cache path is auto_handover_poses_test2.json
        assert gui.auto_handover_cache_path.endswith("auto_handover_poses_test2.json")

        # 4. Large log box exists
        assert hasattr(gui, 'log_text')
        assert gui.log_text is not None

        gui.destroy_node()
        root.destroy()
    finally:
        pass


def test_build_and_plan_stage3_test2(setup_test2_env):
    """Test 4: build_stage3_test2_plan and plan_stage3_test2_paths run end-to-end with handover."""
    handover_cfg, robots, _ = setup_test2_env

    # Handover test pose
    test_pose = AutoHandoverTest2Pose(
        pair_name="robot1_robot2",
        handover_position_world=np.array([37.5, -62.5, 362.5]),
        standoff_distance_mm=50.0,
        robot_a_id="robot1",
        robot_b_id="robot2",
        robot_a_handover_q_deg=np.array([-7.43, -33.1, 15.16, 60.88, 85.0]),
        robot_b_handover_q_deg=np.array([20.2, -13.54, -3.67, -17.49, 77.94]),
        robot_a_pre_handover_q_deg=np.array([-7.43, -21.69, 16.95, 82.02, 85.0]),
        robot_b_pre_handover_q_deg=np.array([23.56, -1.32, -5.31, -14.76, 90.0]),
        handover_clearance_mm=238.6,
        pre_handover_clearance_mm=99.41,
        error_a_mm=0.0,
        error_b_mm=0.0,
        dot_z=-0.9766,
        dot_x=0.0152,
        approach_angle_dev_deg=12.42,
        approach_mode="DUAL_SYNCHRONIZED",
        candidate_origin="TOP_OF_WORKSPACE",
        corridor_a_valid=True,
        corridor_b_valid=True,
    )

    pick_xyz = [100.0, 100.0, 200.0]
    place_xyz = [0.0, 150.0, 150.0]

    plan, pick_opts, place_opts = build_stage3_test2_plan(
        pick_xyz=pick_xyz,
        place_xyz=place_xyz,
        robots=robots,
        handover_cfg=handover_cfg,
        chooser=lambda label, c: "robot1" if label == "pick" else "robot2",
        pick_yaw_deg=0.0,
        place_yaw_deg=0.0,
        use_auto_handover=True,
        test_poses={"robot1_robot2": test_pose},
    )

    assert plan is not None
    assert plan.candidate_origin == "TOP_OF_WORKSPACE"
    assert plan.needs_handover is True
    assert plan.auto_handover_info is not None

    traj = plan_stage3_test2_paths(plan, robots, safe_distance=50.0, steps=15)
    assert traj is not None
    assert len(traj.phase4_advance_to_handover) == 15
    assert len(traj.phase6_retract_to_pre_handover) == 15
    assert traj.min_overall_clearance_mm > 0.0

    is_safe, msg = validate_execution_safety_gate(traj, robots, plan=plan)
    assert is_safe, f"Synchronized trajectory must pass safety gate: {msg}"
