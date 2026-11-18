"""Logical Robot Agents representing heterogeneous robotic manipulators.

Robot 1: 5-DOF articulated arm, base at [0, 0, 0], yaw 0°.
Robot 2: 5-DOF articulated arm, base at [-250, 0, 0], yaw 180°.
Robot 3: 6-DOF articulated arm, base at [0, 250, 0], yaw 90° with full spherical wrist.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from robot_control.handover.stage1 import load_stage1_robots
from robot_control.handover.stage1_ik import solve_stage1_ik, world_to_base
from robot_control.handover.stage2_5 import resolve_stage2_5_target
from robot_control.kinematics.fk_math import fk


class RobotLogicalAgent(ABC):
    """Abstract Base Class for logical heterogeneous robot agents."""

    def __init__(self, robot_id: str, dof: int, base_xyz: List[float], base_rpy_deg: List[float]):
        self.robot_id = robot_id
        self.dof = dof
        self.base_xyz = np.array(base_xyz, dtype=float)
        self.base_rpy_deg = np.array(base_rpy_deg, dtype=float)

        # Lazy load kinematics config
        self._cfg = None

    @property
    def config(self):
        if self._cfg is None:
            _, robots_cfg, _ = load_stage1_robots()
            self._cfg = robots_cfg.get(self.robot_id)
        return self._cfg

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return structural capabilities, kinematics limits, and payload ratings."""
        pass

    def evaluate_point(
        self,
        target_xyz_world: List[float],
        target_yaw_deg: float = 0.0,
        role: str = "pick",
    ) -> Dict[str, Any]:
        """Evaluate point reachability and compute a normalized suitability score [0..1]."""
        p_world = np.array(target_xyz_world, dtype=float)
        dist_from_base = float(np.linalg.norm(p_world - self.base_xyz))

        if self.config is None:
            return {
                "robot_id": self.robot_id,
                "reachable": False,
                "distance_from_base_mm": dist_from_base,
                "error_mm": 999.0,
                "suitability_score": 0.0,
                "explanation": f"Kinematics configuration for {self.robot_id} not available.",
            }

        # Solve target using Stage 2.5 solver (attempts orientation IK first, falls back to position IK)
        try:
            opts, is_fb = resolve_stage2_5_target(
                target_xyz_world=p_world,
                robots={self.robot_id: self.config},
                target_yaw_deg=target_yaw_deg,
                label=role,
            )
            sol = opts.get(self.robot_id)
        except Exception as e:
            sol = None

        if sol is None or sol.error_mm > 5.0:
            return {
                "robot_id": self.robot_id,
                "reachable": False,
                "distance_from_base_mm": dist_from_base,
                "error_mm": 999.0 if sol is None else float(sol.error_mm),
                "suitability_score": 0.0,
                "explanation": f"Target [{p_world[0]:.1f}, {p_world[1]:.1f}, {p_world[2]:.1f}] is out of reach for {self.robot_id} (dist: {dist_from_base:.1f} mm).",
            }

        # Compute suitability based on distance to reach limit, error, and manipulability
        max_reach = 420.0 if self.dof == 6 else 380.0
        reach_ratio = np.clip(dist_from_base / max_reach, 0.0, 1.0)
        # Optimal reach is typically 45-75% of maximum reach
        distance_score = float(1.0 - abs(reach_ratio - 0.6) * 1.5)
        distance_score = max(0.1, min(1.0, distance_score))

        accuracy_score = float(max(0.0, 1.0 - (sol.error_mm / 5.0)))
        solver_bonus = 0.2 if sol.solver_type == "object_pick_ik" else 0.0

        raw_score = 0.5 * distance_score + 0.3 * accuracy_score + solver_bonus
        normalized_score = round(float(np.clip(raw_score, 0.0, 1.0)), 3)

        explanation = (
            f"✓ Reachable by {self.robot_id.upper()} via {sol.solver_type} "
            f"(error: {sol.error_mm:.2f} mm, distance: {dist_from_base:.1f} mm, score: {normalized_score})."
        )

        return {
            "robot_id": self.robot_id,
            "reachable": True,
            "solver_type": sol.solver_type,
            "error_mm": float(sol.error_mm),
            "joints_deg": [round(float(q), 2) for q in sol.q_deg],
            "distance_from_base_mm": round(dist_from_base, 1),
            "suitability_score": normalized_score,
            "is_fallback": sol.is_fallback,
            "approach_tilt_deg": sol.approach_tilt_deg,
            "explanation": explanation,
        }


class Robot1Agent(RobotLogicalAgent):
    """Robot 1: 5-DOF articulated manipulator stationed at World Base [0, 0, 0]."""

    def __init__(self):
        super().__init__(
            robot_id="robot1",
            dof=5,
            base_xyz=[0.0, 0.0, 0.0],
            base_rpy_deg=[0.0, 0.0, 0.0],
        )

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "robot_id": "robot1",
            "dof": 5,
            "arm_architecture": "5-DOF Articulated Pitch-Roll Gripper",
            "wrist_type": "1-Axis Pitch Wrist",
            "base_location": "World Origin [0, 0, 0]",
            "primary_workspace": "South / South-East quadrant (Place zones, SE table)",
            "nominal_reach_radius_mm": 380.0,
            "handover_compatibility": ["robot2", "robot3"],
            "strengths": "Ideal for placing at SE Table [150, 100, 180] and dual synchronized top-of-workspace handovers.",
        }


class Robot2Agent(RobotLogicalAgent):
    """Robot 2: 5-DOF articulated manipulator stationed at [-250, 0, 0] rotated 180°."""

    def __init__(self):
        super().__init__(
            robot_id="robot2",
            dof=5,
            base_xyz=[-250.0, 0.0, 0.0],
            base_rpy_deg=[0.0, 0.0, 180.0],
        )

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "robot_id": "robot2",
            "dof": 5,
            "arm_architecture": "5-DOF Articulated Pitch-Roll Gripper",
            "wrist_type": "1-Axis Pitch Wrist",
            "base_location": "West Station [-250, 0, 0]",
            "primary_workspace": "West / South-West quadrant (Buffer zone, West table)",
            "nominal_reach_radius_mm": 380.0,
            "handover_compatibility": ["robot1", "robot3"],
            "strengths": "Covers western buffer table and center handover mating zones.",
        }


class Robot3Agent(RobotLogicalAgent):
    """Robot 3: 6-DOF articulated manipulator stationed at [0, 250, 0] rotated 90°."""

    def __init__(self):
        super().__init__(
            robot_id="robot3",
            dof=6,
            base_xyz=[0.0, 250.0, 0.0],
            base_rpy_deg=[0.0, 0.0, 90.0],
        )

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "robot_id": "robot3",
            "dof": 6,
            "arm_architecture": "6-DOF Full Spherical Wrist Manipulator",
            "wrist_type": "3-Axis Spherical Wrist (Pitch-Yaw-Roll)",
            "base_location": "North Station [0, 250, 0]",
            "primary_workspace": "North / North-West quadrant (NW Bay, North table)",
            "nominal_reach_radius_mm": 420.0,
            "handover_compatibility": ["robot1", "robot2"],
            "strengths": "Full 6-DOF flexibility with continuous wrist roll; superior top-down grasp capability at NW Bay [-200, 150, 150].",
        }
