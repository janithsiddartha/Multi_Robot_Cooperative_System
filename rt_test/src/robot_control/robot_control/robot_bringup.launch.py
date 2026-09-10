"""
robot_bringup.launch.py

Brings up the core 5-DOF robot control stack:

    fk_node          - forward kinematics (/joint_commands -> /end_effector_position)
    ik_node          - inverse kinematics  (/ik_target -> /ik_solution, /ik_status)
    trajectory_node  - pick & place motion planning / execution
    uart_node        - serial bridge to the ESP32 (servo commands)

Does NOT launch robot_gui, visualizer_node, or command_mux_node -
add them separately if/when you want them.

Usage:
    ros2 launch robot_control robot_bringup.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

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
        fk_node,
        ik_node,
        trajectory_node,
        uart_node,
    ])