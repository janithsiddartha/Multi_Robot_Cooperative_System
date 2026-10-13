from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robot_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='iit',
    maintainer_email='iit@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hardware_translator_node = robot_control.nodes.hardware_translator_node:main',
            'uart_node = robot_control.nodes.uart_node:main',
            'fk_node = robot_control.nodes.fk_node:main',
            'visualizer_node = robot_control.nodes.visualizer_node:main',
            'ik_node = robot_control.nodes.ik_node:main',
            'trajectory_node = robot_control.nodes.trajectory_node:main',
            'command_mux_node = robot_control.nodes.command_mux_node:main',
            'gripper_mux_node = robot_control.nodes.gripper_mux_node:main',
            'control_mode_manager_node = robot_control.nodes.control_mode_manager_node:main',
            'robot_gui_test_1 = robot_control.gui.robot_gui_test_1:main',
            'robot_gui_test_2 = robot_control.gui.robot_gui_test_2:main',
            'robot_gui_test_3 = robot_control.gui.robot_gui_test_3:main',
            'multi_robot_gui = robot_control.gui.multi_robot_gui:main',
            'handover_planner_node = robot_control.nodes.handover_planner_node:main',
            'handover_visualizer_node = robot_control.nodes.handover_visualizer_node:main',
            'common_workspace_node = robot_control.nodes.common_workspace_node:main',
            'multi_robot_handover_gui = robot_control.gui.multi_robot_handover_gui:main',
            'handover_pose_node = robot_control.nodes.handover_pose_node:main',
            'stage1_gui = robot_control.gui.stage1_gui:main',
        ],
    },
)
