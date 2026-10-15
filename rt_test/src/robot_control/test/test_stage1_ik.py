import numpy as np

from robot_control.handover.stage1 import load_stage1_robots, lookup_manual_handover, select_robot
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
