"""
robot_bringup.launch.py

Brings up the core 5-DOF robot control stack with multi-robot support:

    control_mode_manager_node - sole publisher of control_mode
    command_mux_node          - arbitrates joint commands
    gripper_mux_node          - arbitrates gripper commands
    fk_node                   - forward kinematics
    ik_node                   - inverse kinematics
    trajectory_node           - pick & place motion planning / execution
    uart_node                 - serial bridge to the ESP32

Does NOT launch robot_gui or visualizer_node - add them separately.

Usage:
    ros2 launch robot_control robot_bringup.launch.py robot_name:=robot1 config:=robot1.yaml
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    pkg_share = get_package_share_directory('robot_control')

    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='robot1',
        description='Name of the robot (used for namespace)'
    )

    config_arg = DeclareLaunchArgument(
        'config',
        default_value='robot1.yaml',
        description='YAML configuration file for the robot'
    )

    robot_name = LaunchConfiguration('robot_name')
    config_file = LaunchConfiguration('config')

    config_path = PathJoinSubstitution([pkg_share, 'config', config_file])

    # Push the namespace for all nodes
    namespace_action = PushRosNamespace(namespace=robot_name)

    nodes = [
        'control_mode_manager_node',
        'command_mux_node',
        'gripper_mux_node',
        'fk_node',
        'ik_node',
        'trajectory_node',
        'uart_node'
    ]

    launch_actions = [robot_name_arg, config_arg, namespace_action]

    for node_name in nodes:
        launch_actions.append(
            Node(
                package='robot_control',
                executable=node_name,
                name=node_name,
                output='screen',
                parameters=[config_path]
            )
        )

    return LaunchDescription(launch_actions)