import numpy as np

from robot_control.handover.stage1 import (
    build_stage1_plan,
    load_stage1_robots,
    lookup_manual_handover,
    lookup_manual_handover_with_pre,
    select_robot,
)
from robot_control.handover.stage1_5 import plan_stage1_5_synchronized_paths
from robot_control.handover.stage1_ik import solve_stage1_ik
from robot_control.kinematics.fk_math import fk


def test_stage1_ik_roundtrip_robot1():
    _, robots, _ = load_stage1_robots()
    cfg = robots["robot1"]
    q_rad = np.deg2rad(np.array(cfg.home_deg, dtype=float))
    p_base, _ = fk(q_rad, cfg)
    p_world = cfg.world_base_R @ p_base + cfg.world_base_xyz

    result = solve_stage1_ik("robot1", p_world, cfg, current_q_deg=cfg.home_deg)
    assert result is not None
    assert result.error_mm < 1.0
    assert result.q_deg.shape == (cfg.num_joints,)


def test_workspace_intersection_asks_chooser():
    called = []

    def chooser(label, robot_ids):
        called.append((label, list(robot_ids)))
        return robot_ids[0]

    single = select_robot({"robot2": object()}, chooser, "pick")
    assert single == "robot2"
    assert called == []

    chosen = select_robot({"robot1": object(), "robot2": object()}, chooser, "place")
    assert chosen == "robot1"
    assert called[0][0] == "place"
    assert set(called[0][1]) == {"robot1", "robot2"}


def test_manual_handover_pairs_exist():
    handover_cfg, robots, _ = load_stage1_robots()
    q1, q2 = lookup_manual_handover(handover_cfg, "robot1", "robot2")
    assert len(q1) == robots["robot1"].num_joints
    assert len(q2) == robots["robot2"].num_joints


def test_manual_handover_with_pre_all_pairs():
    handover_cfg, robots, _ = load_stage1_robots()
    pairs = [("robot1", "robot2"), ("robot2", "robot3"), ("robot3", "robot1")]
    for r_a, r_b in pairs:
        res = lookup_manual_handover_with_pre(handover_cfg, r_a, r_b)
        assert res is not None, f"Failed for pair {r_a} - {r_b}"
        q_a, q_b, pre_q_a, pre_q_b = res
        assert len(q_a) == robots[r_a].num_joints
        assert len(q_b) == robots[r_b].num_joints
        assert len(pre_q_a) == robots[r_a].num_joints
        assert len(pre_q_b) == robots[r_b].num_joints
        # Verify pre_q is distinct from handover q
        assert not np.allclose(q_a, pre_q_a), f"pre_q_a should differ from q_a for {r_a}"
        assert not np.allclose(q_b, pre_q_b), f"pre_q_b should differ from q_b for {r_b}"


def test_stage1_plan_has_pre_handover_and_stage1_5_planning():
    handover_cfg, robots, _ = load_stage1_robots()
    pick_xyz = [100.0, 100.0, 200.0]
    place_xyz = [0.0, 150.0, 150.0]

    def chooser(label, robot_ids):
        return "robot1" if label == "pick" else "robot2"

    plan, _, _ = build_stage1_plan(pick_xyz, place_xyz, robots, handover_cfg, chooser=chooser)
    assert plan.needs_handover
    assert plan.pick_pre_handover_q is not None
    assert plan.place_pre_handover_q is not None

    # Plan Stage 1.5 synchronized trajectories
    traj_result = plan_stage1_5_synchronized_paths(plan, robots, safe_distance=50.0, steps=20)
    assert traj_result.needs_handover
    assert len(traj_result.phase1_simultaneous) == 20
    assert len(traj_result.phase3_pick_to_pre_handover) == 20
    assert len(traj_result.phase4_advance_to_handover) == 20
    assert len(traj_result.phase6_retract_to_pre_handover) == 20
    assert len(traj_result.phase7_handover_to_place) == 20
    assert len(traj_result.phase9_simultaneous_home) == 20
    assert traj_result.all_phases_safe

