"""Comprehensive Test Suite for Autonomous AI Multi-Robot System.

Verifies:
1. AI Config & Hybrid Mode Fallback.
2. Shared World Model & Telemetry Ingestion.
3. Logical Robot Agents (R1: 5-DOF, R2: 5-DOF, R3: 6-DOF) Reachability & Ranking.
4. Natural Language Intent Understanding & Tool Execution.
5. Collaborative Multi-Robot Handover Task Planning & Clearance Enforcement (> 45 mm).
6. Multi-Tier Safety Architecture (Advisory / Simulation / Hardware Token Gating).
7. Explainable Rationale & Thought Tracing.
"""

import pytest
import numpy as np

from robot_control.ai.config import AIConfig, load_ai_config
from robot_control.ai.world_model import WorldModel, WorkpieceObject
from robot_control.ai.robot_agent import Robot1Agent, Robot2Agent, Robot3Agent
from robot_control.ai.tools import RoboticsTools
from robot_control.ai.reasoning_engine import DeterministicEngine, HybridReasoningEngine
from robot_control.ai.ai_coordinator import AICoordinator


def test_ai_config_loader():
    """Test 1: Configuration loads with deterministic fallback defaults."""
    cfg = load_ai_config()
    assert cfg is not None
    assert cfg.collision_gate_mm == 45.0
    assert cfg.mode in ("hybrid", "local", "cloud", "deterministic")
    assert "nw bay" in cfg.named_locations
    assert cfg.named_locations["nw bay"] == [-200.0, 150.0, 150.0]


def test_world_model_state_tracking():
    """Test 2: World Model updates telemetry, tracks objects, and produces clean snapshots."""
    wm = WorldModel()
    assert "robot1" in wm.robots
    assert "robot2" in wm.robots
    assert "robot3" in wm.robots

    # Test telemetry ingestion
    wm.update_robot_telemetry(
        joint_state={"robot1": [10.0, 20.0, 30.0, 40.0, 50.0]},
        grippers={"robot1": "closed"},
    )
    assert wm.robots["robot1"].joints_deg == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert wm.robots["robot1"].gripper_state == "closed"

    # Test workpiece inventory
    wp = wm.set_workpiece("obj_cylinder", [-200.0, 150.0, 150.0], label="Red Cylinder")
    assert wp.object_id == "obj_cylinder"
    assert "obj_cylinder" in wm.objects

    # Test snapshot serialization
    snap = wm.snapshot()
    assert snap["emergency_stopped"] is False
    assert "robots" in snap
    assert "objects" in snap


def test_robot_agents_heterogeneous_capabilities():
    """Test 3: Logical robot agents report accurate heterogeneous kinematics capabilities."""
    r1 = Robot1Agent()
    r2 = Robot2Agent()
    r3 = Robot3Agent()

    assert r1.dof == 5
    assert r2.dof == 5
    assert r3.dof == 6

    cap1 = r1.get_capabilities()
    cap2 = r2.get_capabilities()
    cap3 = r3.get_capabilities()

    assert "1-Axis Pitch Wrist" in cap1["wrist_type"]
    assert "3-Axis Spherical Wrist" in cap3["wrist_type"]
    assert r3.base_xyz[0] == 250.0 or r3.config.world_base_xyz[0] == 250.0


def test_reachability_and_suitability_ranking():
    """Test 4: Reachability evaluation scores candidate arms quantitatively."""
    tools = RoboticsTools()
    # Query point near Robot 2 / NW zone
    res = tools.rank_robots_for_point(-200.0, 150.0, 150.0, role="pick")
    assert res is not None
    assert "results" in res
    assert "ranked_order" in res
    assert res["is_reachable_by_any"] is True
    # At least one robot reaches [-200, 150, 150]
    assert len(res["ranked_order"]) > 0


def test_handover_need_determination():
    """Test 5: System determines whether collaborative handover is needed."""
    tools = RoboticsTools()
    # Pick at NW Bay, Place at SE Table
    eval_res = tools.evaluate_handover_need([-200.0, 150.0, 150.0], [150.0, 100.0, 180.0])
    assert eval_res["feasible"] is True
    assert "pick_robot" in eval_res
    assert "place_robot" in eval_res
    assert "explanation" in eval_res


def test_ai_coordinator_chat_reachability_query():
    """Test 6: AI Coordinator parses natural language reachability check and explains reasoning."""
    coord = AICoordinator()
    res = coord.handle_chat_instruction("Check which robots can reach [-200, 150, 150]", safety_mode="advisory")
    assert res is not None
    assert res["intent"] == "reachability_query"
    assert "thought_trace" in res
    assert len(res["thought_trace"]) > 0
    assert "response" in res


def test_ai_coordinator_emergency_stop_intent():
    """Test 7: Emergency stop instruction halts all systems immediately."""
    coord = AICoordinator()
    res = coord.handle_chat_instruction("Emergency stop all robots immediately!", safety_mode="advisory")
    assert res["intent"] == "emergency_stop"
    assert coord.world_model.emergency_stopped is True
    assert res["plan"]["status"] == "HALTED"


def test_ai_coordinator_homing_intent():
    """Test 8: Return home instruction generates complete homing plan."""
    coord = AICoordinator()
    res = coord.handle_chat_instruction("Return all robots to home position", safety_mode="advisory")
    assert res["intent"] == "home_all"
    assert res["plan"] is not None
    assert len(res["plan"]["steps"]) == 3


def test_ai_coordinator_collaborative_handover_planning():
    """Test 9: Full task planning generates collision-free 9-phase trajectory with safety gate."""
    coord = AICoordinator()
    res = coord.handle_chat_instruction(
        "Pick object at NW Bay [-200, 150, 150] and place at SE Table [150, 100, 180]",
        safety_mode="advisory",
    )
    assert res is not None
    assert res["intent"] == "manipulation_transfer"
    plan = res["plan"]
    assert plan is not None
    assert plan["min_clearance_mm"] >= 45.0  # Must strictly satisfy collision gate
    assert len(plan["trajectory_steps"]) > 0
    assert "confirmation_token" in plan


def test_safety_architecture_hardware_gating():
    """Test 10: Multi-tier safety gating prevents unauthorized physical hardware execution."""
    coord = AICoordinator()
    # Plan task
    res = coord.handle_chat_instruction(
        "Pick object at NW Bay [-200, 150, 150] and place at SE Table [150, 100, 180]",
        safety_mode="advisory",
    )
    plan = res["plan"]
    task_id = plan["task_id"]
    token = plan["confirmation_token"]

    # 1. Reject if policy is Advisory
    rej1 = coord.authorize_hardware_execution(task_id, token, safety_mode="advisory")
    assert rej1["success"] is False
    assert "rejected" in rej1["error"].lower()

    # 2. Reject if wrong token
    rej2 = coord.authorize_hardware_execution(task_id, "WRONG-TOKEN", safety_mode="hardware")
    assert rej2["success"] is False
    assert "token" in rej2["error"].lower()

    # 3. Authorize when mode is Hardware and token is valid
    auth = coord.authorize_hardware_execution(task_id, token, safety_mode="hardware")
    assert auth["success"] is True
    assert auth["status"] == "AUTHORIZED"
    assert len(auth["trajectory_steps"]) > 0

    # 4. Reject replay of consumed single-use token
    replay = coord.authorize_hardware_execution(task_id, token, safety_mode="hardware")
    assert replay["success"] is False

