"""Shared World Model for Heterogeneous Multi-Robot System.

Tracks live kinematics, physical attachments, workpiece locations,
named workspace regions, collision envelopes, and active execution state.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class WorkpieceObject:
    object_id: str
    label: str
    xyz: List[float]
    bounding_radius_mm: float = 25.0
    attached_to_robot: Optional[str] = None
    confidence: float = 1.0
    source: str = "user_input"
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RobotArmState:
    robot_id: str
    dof: int
    base_xyz: List[float]
    base_rpy_deg: List[float]
    joints_deg: List[float]
    tcp_world_xyz: List[float]
    tcp_rpy_deg: List[float]
    gripper_state: str = "open"  # "open" or "closed"
    gripper_angle_deg: float = 0.0
    connected: bool = False
    assigned_port: Optional[str] = None
    status: str = "IDLE"  # "IDLE", "SIMULATING", "EXECUTING", "HALTED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorldModel:
    """Central World Model tracking multi-robot fleet and manipulation environment."""

    def __init__(self):
        # Known Heterogeneous Robots
        self.robots: Dict[str, RobotArmState] = {
            "robot1": RobotArmState(
                robot_id="robot1",
                dof=5,
                base_xyz=[0.0, 0.0, 0.0],
                base_rpy_deg=[0.0, 0.0, 0.0],
                joints_deg=[0.0, 0.0, 0.0, 0.0, 0.0],
                tcp_world_xyz=[0.0, 0.0, 390.0],
                tcp_rpy_deg=[0.0, 0.0, 0.0],
                connected=False,
            ),
            "robot2": RobotArmState(
                robot_id="robot2",
                dof=5,
                base_xyz=[-250.0, 0.0, 0.0],
                base_rpy_deg=[0.0, 0.0, 180.0],
                joints_deg=[0.0, 0.0, 0.0, 0.0, 0.0],
                tcp_world_xyz=[-250.0, 0.0, 390.0],
                tcp_rpy_deg=[0.0, 0.0, 180.0],
                connected=False,
            ),
            "robot3": RobotArmState(
                robot_id="robot3",
                dof=6,
                base_xyz=[0.0, 250.0, 0.0],
                base_rpy_deg=[0.0, 0.0, 90.0],
                joints_deg=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                tcp_world_xyz=[0.0, 250.0, 450.0],
                tcp_rpy_deg=[0.0, 0.0, 90.0],
                connected=False,
            ),
        }

        # Workpiece Inventory
        self.objects: Dict[str, WorkpieceObject] = {
            "workpiece_default": WorkpieceObject(
                object_id="workpiece_default",
                label="Target Object",
                xyz=[-200.0, 150.0, 150.0],
                bounding_radius_mm=25.0,
                attached_to_robot=None,
                source="default_workspace",
            )
        }

        # Named Semantic Workspace Regions
        self.named_regions: Dict[str, List[float]] = {
            "nw bay": [-200.0, 150.0, 150.0],
            "nw table": [-200.0, 150.0, 150.0],
            "north bay": [0.0, 150.0, 180.0],
            "center buffer": [0.0, -150.0, 150.0],
            "se table": [150.0, 100.0, 180.0],
            "east bay": [150.0, 100.0, 100.0],
            "east table": [150.0, 100.0, 100.0],
            "default pick": [-200.0, 150.0, 150.0],
            "default place": [150.0, 100.0, 180.0],
        }

        # Safety & Clearance
        self.min_body_clearance_mm: float = 500.0
        self.collision_gate_mm: float = 45.0
        self.emergency_stopped: bool = False

        # Active Task & Execution State
        self.active_task_id: Optional[str] = None
        self.active_plan: Optional[Dict[str, Any]] = None
        self.execution_history: List[Dict[str, Any]] = []

    def update_robot_telemetry(
        self,
        joint_state: Optional[Dict[str, List[float]]] = None,
        frames: Optional[Dict[str, Any]] = None,
        grippers: Optional[Dict[str, str]] = None,
        hardware_status: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update robot states from incoming telemetry stream."""
        if joint_state:
            for rid, joints in joint_state.items():
                if rid in self.robots:
                    self.robots[rid].joints_deg = list(joints)

        if frames:
            for rid, f_data in frames.items():
                if rid in self.robots:
                    tcp = f_data.get("tcp_world")
                    if tcp:
                        self.robots[rid].tcp_world_xyz = [float(c) for c in tcp]
                    rpy = f_data.get("tcp_rpy_deg")
                    if rpy:
                        self.robots[rid].tcp_rpy_deg = [float(a) for a in rpy]

        if grippers:
            for rid, g_state in grippers.items():
                if rid in self.robots:
                    self.robots[rid].gripper_state = str(g_state).lower()

        if hardware_status:
            for rid, hw in hardware_status.get("robots", {}).items():
                if rid in self.robots:
                    self.robots[rid].connected = bool(hw.get("connected", False))
                    self.robots[rid].assigned_port = hw.get("assigned_port")

    def update_clearance(self, min_dist_mm: float) -> None:
        self.min_body_clearance_mm = float(min_dist_mm)

    def set_emergency_stop(self, stopped: bool = True) -> None:
        self.emergency_stopped = stopped
        for r in self.robots.values():
            r.status = "HALTED" if stopped else "IDLE"

    def set_workpiece(
        self,
        object_id: str,
        xyz: List[float],
        label: str = "Object",
        attached_to: Optional[str] = None,
        source: str = "user_input",
    ) -> WorkpieceObject:
        wp = WorkpieceObject(
            object_id=object_id,
            label=label,
            xyz=[float(c) for c in xyz],
            attached_to_robot=attached_to,
            source=source,
            updated_at=time.time(),
        )
        self.objects[object_id] = wp
        return wp

    def resolve_location(self, text: str) -> Optional[List[float]]:
        """Resolve a natural language location or coordinate description."""
        t_clean = text.lower().strip()
        for name, coords in self.named_regions.items():
            if name in t_clean:
                return list(coords)
        return None

    def snapshot(self) -> Dict[str, Any]:
        """Produce a complete JSON-serializable snapshot of the world model."""
        return {
            "timestamp": time.time(),
            "emergency_stopped": self.emergency_stopped,
            "min_body_clearance_mm": self.min_body_clearance_mm,
            "collision_gate_mm": self.collision_gate_mm,
            "robots": {rid: r.to_dict() for rid, r in self.robots.items()},
            "objects": {oid: o.to_dict() for oid, o in self.objects.items()},
            "named_regions": self.named_regions,
            "active_task_id": self.active_task_id,
            "active_plan_summary": self.active_plan.get("summary") if self.active_plan else None,
        }

