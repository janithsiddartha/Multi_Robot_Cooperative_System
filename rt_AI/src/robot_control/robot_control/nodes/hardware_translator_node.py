#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray, Int32, String
from rcl_interfaces.msg import ParameterDescriptor

class HardwareTranslatorNode(Node):
    def _get_required_param(self, name):
        self.declare_parameter(name, descriptor=ParameterDescriptor(dynamic_typing=True))
        val = self.get_parameter(name).value
        if val is None:
            err_msg = f"Required parameter '{name}' not found."
            self.get_logger().fatal(err_msg)
            raise ValueError(err_msg)
        return val

    def _get_optional_param(self, name, default):
        self.declare_parameter(name, value=default, descriptor=ParameterDescriptor(dynamic_typing=True))
        val = self.get_parameter(name).value
        return default if val is None else val

    def __init__(self):
        super().__init__('hardware_translator_node')
        
        self.control_mode = 'MANUAL'
        mode_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # Configs
        self.joint_scale = self._get_required_param('kinematics.joints.scale')
        self.num_joints = len(self.joint_scale)
        self.joint_sign = self._get_required_param('kinematics.joints.sign')
        self.joint_offset = self._get_required_param('kinematics.joints.offset')
        self.gripper_open = self._get_required_param('gripper.open_angle')
        self.gripper_close = self._get_required_param('gripper.close_angle')
        self.gripper_min = float(self._get_optional_param('gripper.min_angle', -90.0))
        self.gripper_max = float(self._get_optional_param('gripper.max_angle', 90.0))
        self.gripper_scale = self._get_required_param('gripper.scale')
        self.gripper_sign = self._get_required_param('gripper.sign')
        self.gripper_offset = self._get_required_param('gripper.offset')

        # Subscribers
        self.create_subscription(String, 'control_mode', self.mode_callback, mode_qos)
        self.create_subscription(Float64MultiArray, 'joint_commands', self.joint_callback, 10)
        self.create_subscription(Float64MultiArray, 'gripper_command', self.gripper_callback, 10)
        self.create_subscription(Int32, 'robot_speed', self.speed_callback, 10)
        self.create_subscription(String, 'robot_command', self.system_callback, 10)

        # Publisher for dumb relay
        self.serial_tx_pub = self.create_publisher(String, 'serial_tx', 10)
        
        self.get_logger().info('HARDWARE TRANSLATOR NODE READY')

    def publish_serial(self, command):
        msg = String()
        msg.data = command
        self.serial_tx_pub.publish(msg)

    def mode_callback(self, msg):
        mode = msg.data.strip().upper()
        if mode in ['MANUAL', 'TRAJECTORY', 'HALTED']:
            self.control_mode = mode

    def joint_callback(self, msg):
        if len(msg.data) != self.num_joints:
            return
        
        for channel in range(self.num_joints):
            robot_angle = float(msg.data[channel])
            scale = self.joint_scale[channel]
            sign = self.joint_sign[channel]
            offset = self.joint_offset[channel]

            servo_angle = (sign * (robot_angle / scale) + offset)
            servo_angle = int(round(max(-180.0, min(180.0, servo_angle))))
            
            self.publish_serial(f'J,{channel},{servo_angle}')

    def gripper_callback(self, msg):
        if self.control_mode == 'HALTED' or len(msg.data) != 1:
            return
        
        angle = float(msg.data[0])
        min_lim = min(self.gripper_min, self.gripper_max)
        max_lim = max(self.gripper_min, self.gripper_max)
        angle = max(min_lim, min(max_lim, angle))
        
        servo_angle = (self.gripper_sign * (angle / self.gripper_scale) + self.gripper_offset)
        servo_angle = int(round(max(-180.0, min(180.0, servo_angle))))
        
        self.publish_serial(f'J,{self.num_joints},{servo_angle}')

    def speed_callback(self, msg):
        speed = max(1, min(10, int(msg.data)))
        self.publish_serial(f'S,{speed}')

    def system_callback(self, msg):
        cmd = msg.data.strip().upper()
        if cmd == 'STOP':
            self.publish_serial('STOP')
        elif cmd == 'HOME' and self.control_mode == 'MANUAL':
            self.publish_serial('HOME')

def main(args=None):
    rclpy.init(args=args)
    node = HardwareTranslatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
