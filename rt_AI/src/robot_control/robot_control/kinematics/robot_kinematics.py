import numpy as np
from rcl_interfaces.msg import ParameterDescriptor

class KinematicsConfig:
    def __init__(self, node=None):
        if node is None:
            raise ValueError("KinematicsConfig requires a ROS 2 node to load configuration parameters.")
            
        # Declare and get parameters from ROS 2 Node
        self.alpha_deg = self.get_param(node, 'kinematics.dh.alpha_deg')
        self.a = self.get_param(node, 'kinematics.dh.a')
        self.d = self.get_param(node, 'kinematics.dh.d')
        self.theta_offset_deg = self.get_param(node, 'kinematics.dh.theta_offset_deg')
        self.theta_joint_index = self.get_param(node, 'kinematics.dh.theta_joint_index')
        self.theta_joint_multiplier = self.get_param(node, 'kinematics.dh.theta_joint_multiplier')

        self.tcp_offset = np.array(self.get_param(node, 'kinematics.tcp.offset'), dtype=float)
        
        limits_min = self.get_param(node, 'kinematics.joints.limits_deg_min')
        limits_max = self.get_param(node, 'kinematics.joints.limits_deg_max')
        self.home_deg = self.get_param(node, 'kinematics.joints.home_deg')
        self.joint_limits_deg = np.array(list(zip(limits_min, limits_max)))
        self.joint_limits_rad = np.deg2rad(self.joint_limits_deg)

        self.num_frames = len(self.alpha_deg)
        self.num_joints = len(self.home_deg)

        self.gripper_open_angle = self.get_param(node, 'gripper.open_angle')
        self.gripper_close_angle = self.get_param(node, 'gripper.close_angle')

        self.trajectory_dt = self.get_param(node, 'trajectory.dt')
        self.approach_height = self.get_param(node, 'trajectory.approach_height')
        # GUI text fields (with safe defaults if missing)
        try:
            self.gui_title = self.get_param(node, 'gui.title')
            self.gui_overview = self.get_param(node, 'gui.overview')
            self.gui_system_info = self.get_param(node, 'gui.system_info')
        except ValueError:
            self.gui_title = f"{node.get_name().upper()} - Control Station"
            self.gui_overview = "Welcome to the Robot Engineering Workstation."
            self.gui_system_info = "DOF: 5\nRobot Type: Articulated Serial Manipulator"
            
        self.segment_time = self.get_param(node, 'trajectory.segment_time')

    def get_param(self, node, name):
        node.declare_parameter(name, descriptor=ParameterDescriptor(dynamic_typing=True))
        val = node.get_parameter(name).value
        if val is None:
            raise ValueError(f"Required parameter '{name}' not found in configuration file. (Did you forget to pass the YAML file using --ros-args --params-file ?)")
        return val
