import rclpy
from rclpy.node import Node
import yaml
import os
import numpy as np
from std_msgs.msg import String, Float64MultiArray
from robot_control.handover.planner import HandoverPlanner

class YamlKinematicsConfig:
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            # ROS 2 params file usually has: /**: ros__parameters: ...
            params = data.get('/**')
            if params is None:
                params = list(data.values())[0] # Try first key
            params = params.get('ros__parameters', {})
            
            kinematics = params.get('kinematics', {})
            dh = kinematics.get('dh', {})
            self.alpha_deg = dh.get('alpha_deg')
            self.a = dh.get('a')
            self.d = dh.get('d')
            self.theta_offset_deg = dh.get('theta_offset_deg')
            self.theta_joint_index = dh.get('theta_joint_index')
            self.theta_joint_multiplier = dh.get('theta_joint_multiplier')
            
            self.tcp_offset = np.array(kinematics.get('tcp', {}).get('offset'), dtype=float)
            
            joints = kinematics.get('joints', {})
            limits_min = joints.get('limits_deg_min')
            limits_max = joints.get('limits_deg_max')
            self.home_deg = joints.get('home_deg')
            self.joint_limits_deg = np.array(list(zip(limits_min, limits_max)))
            self.joint_limits_rad = np.deg2rad(self.joint_limits_deg)
            
            self.num_frames = len(self.alpha_deg)
            self.num_joints = len(self.home_deg)

            # Dynamics limits (with defaults for backward compatibility)
            vel = joints.get('velocity_limits_deg_per_s',
                             [120.0] * self.num_joints)
            acc = joints.get('acceleration_limits_deg_per_s2',
                             [600.0] * self.num_joints)
            jrk = joints.get('jerk_limits_deg_per_s3',
                             [3000.0] * self.num_joints)
            self.velocity_limits_rad = np.deg2rad(vel)
            self.acceleration_limits_rad = np.deg2rad(acc)
            self.jerk_limits_rad = np.deg2rad(jrk)

class HandoverPlannerNode(Node):
    def __init__(self):
        super().__init__('handover_planner_node')
        
        # Load handover config
        handover_yaml = '/home/iit/rt_test/src/robot_control/config/handover.yaml'
        with open(handover_yaml, 'r') as f:
            self.handover_config = yaml.safe_load(f)
            
        # Load robot configs
        self.robots = {}
        for i in [1, 2, 3]:
            path = f'/home/iit/rt_test/src/robot_control/config/robot{i}.yaml'
            self.robots[i] = YamlKinematicsConfig(path)
            
        self.planner = HandoverPlanner(self.handover_config, self.robots, self.handover_config.get('robots', {}))
        
        self.get_logger().info('Handover Planner Node Started')
        
def main(args=None):
    rclpy.init(args=args)
    node = HandoverPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
