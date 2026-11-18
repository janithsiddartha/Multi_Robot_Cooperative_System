"""Central Multi-Robot AI Coordinator for rt_AI.

Coordinates high-level task understanding, kinematic feasibility verification,
safety policy enforcement, and execution dispatch for heterogeneous arms:
Robot 1 (5-DOF), Robot 2 (5-DOF), and Robot 3 (6-DOF).
"""

import time
import uuid
from typing import Dict, List, Optional, Any

from robot_control.ai.config import AIConfig, load_ai_config
from robot_control.ai.world_model import WorldModel
from robot_control.ai.tools import RoboticsTools
from robot_control.ai.reasoning_engine import HybridReasoningEngine


class AICoordinator:
    """Master AI Coordinator orchestrating multi-robot planning and execution."""

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or load_ai_config()
        self.world_model = WorldModel()
        self.tools = RoboticsTools(self.world_model)
        self.reasoning_engine = HybridReasoningEngine(self.config, self.tools, self.world_model)

        # Active Plan & Execution Token Management
        self.active_plan: Optional[Dict[str, Any]] = None
        self.confirmation_tokens: Dict[str, float] = {}

    def handle_chat_instruction(self, message: str, safety_mode: str = "advisory") -> Dict[str, Any]:
        """Ingest natural language command and formulate reasoning, response, and trajectory."""
        res = self.reasoning_engine.process_instruction(message, safety_mode=safety_mode)

        if res.get("plan"):
            self.active_plan = res["plan"]
            self.world_model.active_plan = res["plan"]
            self.world_model.active_task_id = res["plan"].get("task_id")

            # Generate single-use authorization token for hardware execution
            token = f"AUTH-{uuid.uuid4().hex[:8].upper()}"
            self.confirmation_tokens[token] = time.time() + 300.0  # 5 min expiry
            res["plan"]["confirmation_token"] = token

        return res

    def authorize_hardware_execution(
        self,
        task_id: str,
        token: str,
        safety_mode: str = "hardware",
    ) -> Dict[str, Any]:
        """Gate hardware execution behind explicit operator confirmation."""
        # Check Safety Mode
        if safety_mode != "hardware":
            return {
                "success": False,
                "error": "Execution rejected: Policy is not set to Mode 3 (Hardware).",
            }

        # Check Active Plan
        if not self.active_plan or self.active_plan.get("task_id") != task_id:
            return {
                "success": False,
                "error": f"Execution rejected: Task '{task_id}' does not match active synthesized plan.",
            }

        # Check Confirmation Token
        exp = self.confirmation_tokens.get(token)
        if not exp or time.time() > exp:
            return {
                "success": False,
                "error": "Execution rejected: Invalid or expired operator confirmation token.",
            }

        # Check Min Clearance
        min_clr = self.active_plan.get("min_clearance_mm", 0.0)
        if min_clr < self.config.collision_gate_mm:
            return {
                "success": False,
                "error": f"Execution blocked by Safety Gate: Predicted clearance {min_clr} mm < {self.config.collision_gate_mm} mm limit.",
            }

        # Consume token
        del self.confirmation_tokens[token]

        return {
            "success": True,
            "task_id": task_id,
            "status": "AUTHORIZED",
            "message": "Safety gate passed: Hardware trajectory streaming authorized.",
            "trajectory_steps": self.active_plan.get("trajectory_steps", []),
        }

    def update_telemetry(
        self,
        joint_state: Optional[Dict[str, List[float]]] = None,
        frames: Optional[Dict[str, Any]] = None,
        grippers: Optional[Dict[str, str]] = None,
        hardware_status: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Forward telemetry into world model."""
        self.world_model.update_robot_telemetry(
            joint_state=joint_state,
            frames=frames,
            grippers=grippers,
            hardware_status=hardware_status,
        )

    def get_status(self) -> Dict[str, Any]:
        """Query complete status of coordinator and world model."""
        return {
            "ai_mode": self.config.mode,
            "model_name": self.config.model_name,
            "collision_gate_mm": self.config.collision_gate_mm,
            "active_plan_id": self.active_plan.get("task_id") if self.active_plan else None,
            "world_model": self.world_model.snapshot(),
        }

