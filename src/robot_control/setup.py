from setuptools import find_packages, setup

package_name = 'robot_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/robot_bringup.launch.py']),
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
            'uart_node = robot_control.uart_node:main',
             'robot_gui = robot_control.robot_gui:main',
             'fk_node = robot_control.fk_node:main',
             'visualizer_node = robot_control.visualizer_node:main',
             'ik_node = robot_control.ik_node:main',
             'robot_gui_OLD = robot_control.robot_gui_OLD:main',
             'trajectory_node = robot_control.trajectory_node:main',
             'command_mux_node = robot_control.command_mux_node:main',
             'robot_gui_test = robot_control.robot_gui_test:main',
             'gripper_mux_node = robot_control.gripper_mux_node:main',
             'robot_gui_test_ctk = robot_control.robot_gui_test_ctk:main',
             'control_mode_manager_node = robot_control.control_mode_manager_node:main',
        ],

    },
)
