"""
robot_bringup.launch.py

Brings up the core 5-DOF robot control stack:

    control_mode_manager_node - sole publisher of /control_mode
    command_mux_node          - arbitrates joint commands
    gripper_mux_node          - arbitrates gripper commands
    fk_node                   - forward kinematics
    ik_node                   - inverse kinematics
    trajectory_node           - pick & place motion planning / execution
    uart_node                 - serial bridge to the ESP32

Does NOT launch robot_gui or visualizer_node - add them separately.

Usage:
    ros2 launch robot_control robot_bringup.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    control_mode_manager_node = Node(
        package='robot_control',
        executable='control_mode_manager_node',
        name='control_mode_manager_node',
        output='screen',
    )

    command_mux_node = Node(
        package='robot_control',
        executable='command_mux_node',
        name='command_mux_node',
        output='screen',
    )

    gripper_mux_node = Node(
        package='robot_control',
        executable='gripper_mux_node',
        name='gripper_mux_node',
        output='screen',
    )

    fk_node = Node(
        package='robot_control',
        executable='fk_node',
        name='fk_node',
        output='screen',
    )

    ik_node = Node(
        package='robot_control',
        executable='ik_node',
        name='ik_node',
        output='screen',
    )

    trajectory_node = Node(
        package='robot_control',
        executable='trajectory_node',
        name='trajectory_node',
        output='screen',
    )

    uart_node = Node(
        package='robot_control',
        executable='uart_node',
        name='uart_node',
        output='screen',
    )

    return LaunchDescription([
        control_mode_manager_node,
        command_mux_node,
        gripper_mux_node,
        fk_node,
        ik_node,
        trajectory_node,
        uart_node,
    ])