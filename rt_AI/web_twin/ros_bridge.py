#!/usr/bin/env python3
"""web_twin/ros_bridge.py

Production ROS 2 Hardware Bridge for rt_AI 3D Web Digital Twin.
Connects the Web Twin to the physical robots via ROS 2 topics:
  - Subscribes to:
      /{robot}/available_ports
      /{robot}/control_mode (TRANSIENT_LOCAL)
      /{robot}/joint_commands
      /{robot}/end_effector_position
      /{robot}/trajectory_status
      /{robot}/ik_gui_status
      /{robot}/gripper_command
  - Publishes to:
      /{robot}/robot_command (PORT,<port>, STOP, HOME, RESUME)
      /{robot}/control_mode_request (MANUAL, TRAJECTORY, HALTED)
      /{robot}/robot_speed (Int32: 1-10)
      /{robot}/manual_joint_commands (Float64MultiArray)
      /{robot}/manual_gripper_command (Float64MultiArray)
      /{robot}/ik_gui_target (Float64MultiArray)
      /{robot}/pick_coordinates (Float64MultiArray)
      /{robot}/place_coordinates (Float64MultiArray)
      /{robot}/trajectory_command (String: EXECUTE, STOP)
"""

import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from std_msgs.msg import Float64MultiArray, String as RosString, Int32 as RosInt32
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

try:
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class WebTwinRosBridge:
    def __init__(self, twin_engine=None):
        self.twin_engine = twin_engine
        self.node: Optional[Any] = None
        self.thread: Optional[threading.Thread] = None
        self.is_running = False
        self.is_executing = False
        self._execution_step = 0
        self._execution_total = 0

        self.robot_names = ["robot1", "robot2", "robot3"]
        self.available_ports: List[str] = ["/dev/robot1", "/dev/robot2", "/dev/robot3"]
        self._ports_lock = threading.Lock()

        # State tracking per robot
        self.robot_states: Dict[str, Dict[str, Any]] = {
            r: {
                "robot_id": r,
                "assigned_port": f"/dev/{r}",
                "connected": False,
                "control_mode": "MANUAL",
                "speed": 5,
                "gripper_angle": 90.0,
                "gripper_open_deg": 90.0,
                "gripper_close_deg": 0.0,
                "gripper_state": "OPEN",
                "trajectory_status": "IDLE",
                "ik_status": "IDLE",
                "hardware_joints": [0.0] * 5,
                "hardware_tcp": [0.0, 0.0, 0.0],
                "last_heartbeat": 0.0,
            }
            for r in self.robot_names
        }

        # Initialize gripper limits from twin_engine configs if available
        if twin_engine and hasattr(twin_engine, "robots"):
            for r in self.robot_names:
                if r in twin_engine.robots:
                    cfg = twin_engine.robots[r]
                    self.robot_states[r]["gripper_open_deg"] = float(cfg.gripper_open)
                    self.robot_states[r]["gripper_close_deg"] = float(cfg.gripper_close)
                    self.robot_states[r]["gripper_angle"] = float(cfg.gripper_open)
                    if hasattr(cfg, "home_deg"):
                        self.robot_states[r]["hardware_joints"] = [float(v) for v in cfg.home_deg]

        self.publishers: Dict[str, Dict[str, Any]] = {r: {} for r in self.robot_names}
        self.refresh_available_ports()

    def refresh_available_ports(self) -> List[str]:
        """Scan system serial devices and merge with preset robot symlinks."""
        discovered = set()
        # Default known udev rules symlinks
        for p in ["/dev/robot1", "/dev/robot2", "/dev/robot3"]:
            discovered.add(p)

        if SERIAL_AVAILABLE:
            try:
                for p in serial.tools.list_ports.comports():
                    dev = p.device
                    if any(term in dev for term in ["USB", "ACM", "COM", "robot"]):
                        discovered.add(dev)
            except Exception as e:
                print(f"[WebTwin ROS] Note scanning serial ports: {e}")

        # Also check /dev directly if on Linux
        if sys.platform.startswith("linux") and os.path.isdir("/dev"):
            for item in os.listdir("/dev"):
                if item.startswith("ttyUSB") or item.startswith("ttyACM") or item.startswith("robot"):
                    discovered.add(f"/dev/{item}")

        with self._ports_lock:
            self.available_ports = sorted(list(discovered))
        return list(self.available_ports)

    def get_available_ports(self) -> List[str]:
        with self._ports_lock:
            return list(self.available_ports)

    def start(self):
        """Start the ROS 2 background spin thread."""
        if not ROS2_AVAILABLE:
            print("[WebTwin ROS] rclpy is not installed or ROS 2 environment is not sourced. Running in standalone mock mode.")
            return

        if self.is_running:
            return

        def _ros_spin():
            try:
                if not rclpy.ok():
                    rclpy.init()
                self.node = Node("web_twin_ros_bridge")
                print("[WebTwin ROS] Initialized web_twin_ros_bridge ROS 2 node successfully.")

                mode_qos = QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                )

                for r in self.robot_names:
                    # Create publishers in robot namespace
                    self.publishers[r]["command"] = self.node.create_publisher(RosString, f"/{r}/robot_command", 10)
                    self.publishers[r]["control_mode_request"] = self.node.create_publisher(RosString, f"/{r}/control_mode_request", 10)
                    self.publishers[r]["speed"] = self.node.create_publisher(RosInt32, f"/{r}/robot_speed", 10)
                    self.publishers[r]["manual_joints"] = self.node.create_publisher(Float64MultiArray, f"/{r}/manual_joint_commands", 10)
                    self.publishers[r]["manual_gripper"] = self.node.create_publisher(Float64MultiArray, f"/{r}/manual_gripper_command", 10)
                    self.publishers[r]["ik_target"] = self.node.create_publisher(Float64MultiArray, f"/{r}/ik_gui_target", 10)
                    self.publishers[r]["pick_coords"] = self.node.create_publisher(Float64MultiArray, f"/{r}/pick_coordinates", 10)
                    self.publishers[r]["place_coords"] = self.node.create_publisher(Float64MultiArray, f"/{r}/place_coordinates", 10)
                    self.publishers[r]["trajectory_command"] = self.node.create_publisher(RosString, f"/{r}/trajectory_command", 10)
                    self.publishers[r]["trajectory_joints"] = self.node.create_publisher(Float64MultiArray, f"/{r}/trajectory_joint_commands", 10)

                    # Subscribers
                    self.node.create_subscription(
                        RosString, f"/{r}/available_ports",
                        lambda msg, rid=r: self._on_available_ports(rid, msg), 10
                    )
                    self.node.create_subscription(
                        RosString, f"/{r}/control_mode",
                        lambda msg, rid=r: self._on_control_mode(rid, msg), mode_qos
                    )
                    self.node.create_subscription(
                        Float64MultiArray, f"/{r}/joint_commands",
                        lambda msg, rid=r: self._on_joint_commands(rid, msg), 10
                    )
                    self.node.create_subscription(
                        Float64MultiArray, f"/{r}/end_effector_position",
                        lambda msg, rid=r: self._on_tcp_position(rid, msg), 10
                    )
                    self.node.create_subscription(
                        RosString, f"/{r}/trajectory_status",
                        lambda msg, rid=r: self._on_trajectory_status(rid, msg), 10
                    )
                    self.node.create_subscription(
                        RosString, f"/{r}/ik_gui_status",
                        lambda msg, rid=r: self._on_ik_status(rid, msg), 10
                    )
                    self.node.create_subscription(
                        Float64MultiArray, f"/{r}/gripper_command",
                        lambda msg, rid=r: self._on_gripper_command(rid, msg), 10
                    )

                self.is_running = True
                rclpy.spin(self.node)
            except Exception as e:
                print(f"[WebTwin ROS] Exception in ROS spin thread: {e}")
            finally:
                self.is_running = False

        self.thread = threading.Thread(target=_ros_spin, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the ROS 2 node cleanly."""
        self.is_running = False
        self.is_executing = False
        if self.node:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            self.node = None

    # Callbacks
    def _on_available_ports(self, robot_id: str, msg):
        raw = msg.data.strip()
        if raw and raw != "NONE":
            ports = [p.strip() for p in raw.split(",") if p.strip()]
            with self._ports_lock:
                for p in ports:
                    if p not in self.available_ports:
                        self.available_ports.append(p)
                self.available_ports.sort()

    def _on_control_mode(self, robot_id: str, msg):
        mode = msg.data.strip().upper()
        if mode in ["MANUAL", "TRAJECTORY", "HALTED"]:
            self.robot_states[robot_id]["control_mode"] = mode
            self.robot_states[robot_id]["connected"] = True
            self.robot_states[robot_id]["last_heartbeat"] = time.time()

    def _on_joint_commands(self, robot_id: str, msg):
        if msg.data:
            q_list = [round(float(v), 2) for v in msg.data]
            self.robot_states[robot_id]["hardware_joints"] = q_list
            self.robot_states[robot_id]["connected"] = True
            self.robot_states[robot_id]["last_heartbeat"] = time.time()

    def _on_tcp_position(self, robot_id: str, msg):
        if len(msg.data) >= 3:
            self.robot_states[robot_id]["hardware_tcp"] = [round(float(v), 2) for v in msg.data[:3]]

    def _on_trajectory_status(self, robot_id: str, msg):
        self.robot_states[robot_id]["trajectory_status"] = msg.data.strip().upper()

    def _on_ik_status(self, robot_id: str, msg):
        self.robot_states[robot_id]["ik_status"] = msg.data.strip().upper()

    def _on_gripper_command(self, robot_id: str, msg):
        if msg.data:
            deg = round(float(msg.data[0]), 1)
            self.robot_states[robot_id]["gripper_angle"] = deg
            open_d = self.robot_states[robot_id]["gripper_open_deg"]
            close_d = self.robot_states[robot_id]["gripper_close_deg"]
            if abs(deg - open_d) < 5.0:
                self.robot_states[robot_id]["gripper_state"] = "OPEN"
            elif abs(deg - close_d) < 5.0:
                self.robot_states[robot_id]["gripper_state"] = "CLOSED"
            else:
                self.robot_states[robot_id]["gripper_state"] = "CUSTOM"

    # Control API
    def set_robot_port(self, robot_id: str, port: str) -> dict:
        """Publish PORT command to connect robot to specified serial port."""
        if robot_id not in self.robot_states:
            return {"success": False, "error": f"Invalid robot_id: {robot_id}"}

        self.robot_states[robot_id]["assigned_port"] = port
        self.robot_states[robot_id]["connected"] = True

        if self.node and robot_id in self.publishers and "command" in self.publishers[robot_id]:
            msg = RosString()
            msg.data = f"PORT,{port}"
            self.publishers[robot_id]["command"].publish(msg)
            print(f"[WebTwin ROS] Sent PORT command: PORT,{port} -> /{robot_id}/robot_command")

        return {
            "success": True,
            "robot_id": robot_id,
            "port": port,
            "status": "CONNECTING",
            "message": f"Bound {robot_id.upper()} to physical serial port {port}.",
        }

    def emergency_stop(self, robot_id: str = "all") -> dict:
        """Immediately halts all physical motion on hardware and enters HALTED mode."""
        targets = self.robot_names if robot_id in ["all", "ALL", None] else [robot_id]
        self.is_executing = False

        for r in targets:
            if r in self.robot_states:
                self.robot_states[r]["control_mode"] = "HALTED"
            if self.node and r in self.publishers:
                if "command" in self.publishers[r]:
                    msg = RosString()
                    msg.data = "STOP"
                    self.publishers[r]["command"].publish(msg)
                if "control_mode_request" in self.publishers[r]:
                    req = RosString()
                    req.data = "HALTED"
                    self.publishers[r]["control_mode_request"].publish(req)

        print(f"[WebTwin ROS] EMERGENCY STOP triggered on {targets}")
        return {
            "success": True,
            "action": "emergency_stop",
            "status": "HALTED",
            "targets": targets,
            "message": "EMERGENCY STOP EXECUTED. Physical robots halted immediately.",
        }

    def resume_manual(self, robot_id: str = "all") -> dict:
        """Recover from HALTED / fault state and request MANUAL mode."""
        targets = self.robot_names if robot_id in ["all", "ALL", None] else [robot_id]
        for r in targets:
            if r in self.robot_states:
                self.robot_states[r]["control_mode"] = "MANUAL"
            if self.node and r in self.publishers:
                if "command" in self.publishers[r]:
                    msg = RosString()
                    msg.data = "RESUME"
                    self.publishers[r]["command"].publish(msg)
                if "control_mode_request" in self.publishers[r]:
                    req = RosString()
                    req.data = "MANUAL"
                    self.publishers[r]["control_mode_request"].publish(req)

        return {
            "success": True,
            "action": "resume_manual",
            "status": "MANUAL",
            "targets": targets,
            "message": f"Manual control authority resumed for {', '.join(targets)}.",
        }

    def home_robot(self, robot_id: str = "all") -> dict:
        """Send HOME command to hardware and reset internal target joints."""
        targets = self.robot_names if robot_id in ["all", "ALL", None] else [robot_id]
        for r in targets:
            if self.node and r in self.publishers and "command" in self.publishers[r]:
                msg = RosString()
                msg.data = "HOME"
                self.publishers[r]["command"].publish(msg)

            if self.twin_engine and r in self.twin_engine.robots:
                cfg = self.twin_engine.robots[r]
                self.twin_engine.current_q[r] = [float(v) for v in cfg.home_deg]
                self.robot_states[r]["hardware_joints"] = [float(v) for v in cfg.home_deg]

        return {
            "success": True,
            "action": "home_robot",
            "targets": targets,
            "message": f"Homed robot(s): {', '.join(targets)}",
        }

    def set_speed(self, robot_id: str, speed: int) -> dict:
        """Set motion speed level (1 to 10) for hardware."""
        speed = max(1, min(10, int(speed)))
        targets = self.robot_names if robot_id in ["all", "ALL", None] else [robot_id]
        for r in targets:
            if r in self.robot_states:
                self.robot_states[r]["speed"] = speed
            if self.node and r in self.publishers and "speed" in self.publishers[r]:
                msg = RosInt32()
                msg.data = speed
                self.publishers[r]["speed"].publish(msg)

        return {"success": True, "robot_id": robot_id, "speed": speed}

    def set_gripper_limits(self, robot_id: str, open_deg: float, close_deg: float) -> dict:
        """Update customized open and close degree limits for gripper."""
        if robot_id in self.robot_states:
            self.robot_states[robot_id]["gripper_open_deg"] = float(open_deg)
            self.robot_states[robot_id]["gripper_close_deg"] = float(close_deg)
            return {
                "success": True,
                "robot_id": robot_id,
                "open_deg": float(open_deg),
                "close_deg": float(close_deg),
            }
        return {"success": False, "error": f"Invalid robot: {robot_id}"}

    def set_gripper(self, robot_id: str, angle: float, state: str = "CUSTOM") -> dict:
        """Publish manual gripper angle command to hardware."""
        if robot_id in self.robot_states:
            self.robot_states[robot_id]["gripper_angle"] = float(angle)
            self.robot_states[robot_id]["gripper_state"] = state

        if self.node and robot_id in self.publishers and "manual_gripper" in self.publishers[robot_id]:
            msg = Float64MultiArray()
            msg.data = [float(angle)]
            self.publishers[robot_id]["manual_gripper"].publish(msg)

        return {"success": True, "robot_id": robot_id, "angle": angle, "state": state}

    def publish_manual_joints(self, robot_id: str, joint_angles: List[float]):
        """Publish manual joint angles when slider moves or typed angle is submitted."""
        if self.node and robot_id in self.publishers and "manual_joints" in self.publishers[robot_id]:
            msg = Float64MultiArray()
            msg.data = [float(a) for a in joint_angles]
            self.publishers[robot_id]["manual_joints"].publish(msg)

    def execute_single_pick_place(
        self, robot_id: str, pick_xyz: List[float], pick_yaw: float, place_xyz: List[float], place_yaw: float
    ) -> dict:
        """Command single-robot pick and place to trajectory_node on ROS backend."""
        if self.node and robot_id in self.publishers:
            p_msg = Float64MultiArray()
            p_msg.data = [float(pick_xyz[0]), float(pick_xyz[1]), float(pick_xyz[2]), float(pick_yaw)]
            self.publishers[robot_id]["pick_coords"].publish(p_msg)

            pl_msg = Float64MultiArray()
            pl_msg.data = [float(place_xyz[0]), float(place_xyz[1]), float(place_xyz[2]), float(place_yaw)]
            self.publishers[robot_id]["place_coords"].publish(pl_msg)

            cmd_msg = RosString()
            cmd_msg.data = "EXECUTE"
            self.publishers[robot_id]["trajectory_command"].publish(cmd_msg)
            self.robot_states[robot_id]["trajectory_status"] = "PLANNING"

        return {
            "success": True,
            "robot_id": robot_id,
            "action": "execute_single_pick_place",
            "message": f"Published Pick & Place task to {robot_id.upper()} ROS backend nodes.",
        }

    def execute_multi_robot_trajectory(self, steps: List[dict], step_delay: float = 0.08) -> dict:
        """Stream synchronized Stage 3 trajectory steps across all participating physical robots."""
        if self.is_executing:
            return {"success": False, "error": "Hardware execution is already running."}

        self._execution_step = 0
        self._execution_total = len(steps)
        self.is_executing = True

        def _streamer():
            try:
                for idx, step in enumerate(steps):
                    if not self.is_executing:
                        print("[WebTwin ROS] Hardware trajectory stream aborted.")
                        break

                    self._execution_step = idx + 1
                    jstate = step.get("joint_state", {})
                    grippers = step.get("gripper_state", {})

                    for r, q_vals in jstate.items():
                        if self.twin_engine:
                            self.twin_engine.current_q[r] = list(q_vals)
                        self.robot_states[r]["hardware_joints"] = list(q_vals)

                        if self.node and r in self.publishers and "manual_joints" in self.publishers[r]:
                            msg = Float64MultiArray()
                            msg.data = [float(v) for v in q_vals]
                            self.publishers[r]["manual_joints"].publish(msg)

                    for r, g_state in grippers.items():
                        if self.twin_engine:
                            self.twin_engine.gripper_state[r] = g_state
                        ang = (
                            self.robot_states[r]["gripper_close_deg"]
                            if g_state == "close"
                            else self.robot_states[r]["gripper_open_deg"]
                        )
                        self.robot_states[r]["gripper_angle"] = ang
                        self.robot_states[r]["gripper_state"] = "CLOSED" if g_state == "close" else "OPEN"

                        if self.node and r in self.publishers and "manual_gripper" in self.publishers[r]:
                            msg = Float64MultiArray()
                            msg.data = [float(ang)]
                            self.publishers[r]["manual_gripper"].publish(msg)

                    time.sleep(step_delay)
            except Exception as e:
                print(f"[WebTwin ROS] Error in hardware trajectory execution stream: {e}")
            finally:
                self.is_executing = False

        threading.Thread(target=_streamer, daemon=True).start()
        return {
            "success": True,
            "message": f"Hardware trajectory execution streaming active for {len(steps)} synchronized steps.",
            "total_steps": len(steps),
        }

    def get_all_robot_states(self) -> Dict[str, Any]:
        """Return snapshot of live telemetry across all physical robots."""
        return {
            "is_ros_active": self.is_running,
            "is_hardware_executing": self.is_executing,
            "execution_step": self._execution_step,
            "execution_total": self._execution_total,
            "available_ports": self.get_available_ports(),
            "robots": {r: dict(state) for r, state in self.robot_states.items()},
        }
