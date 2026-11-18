"""Autonomous AI Multi-Robot Orchestration Package for rt_AI.

Provides:
- Shared Heterogeneous World Model
- Logical Robot Agents (Robot 1, Robot 2, Robot 3)
- Hybrid AI Reasoning Engine (Cloud / Local Ollama / Deterministic Fallback)
- Typed Robotics Tool-Calling Interface
- Central Multi-Robot Coordinator with Explainable Rationale
- Multi-Tier Safety Architecture (Advisory / Simulation / Hardware)
"""

from robot_control.ai.config import AIConfig, load_ai_config
from robot_control.ai.world_model import WorldModel, WorkpieceObject
from robot_control.ai.robot_agent import RobotLogicalAgent, Robot1Agent, Robot2Agent, Robot3Agent
from robot_control.ai.tools import RoboticsTools
from robot_control.ai.reasoning_engine import HybridReasoningEngine, DeterministicEngine
from robot_control.ai.ai_coordinator import AICoordinator

__all__ = [
    "AIConfig",
    "load_ai_config",
    "WorldModel",
    "WorkpieceObject",
    "RobotLogicalAgent",
    "Robot1Agent",
    "Robot2Agent",
    "Robot3Agent",
    "RoboticsTools",
    "HybridReasoningEngine",
    "DeterministicEngine",
    "AICoordinator",
]

