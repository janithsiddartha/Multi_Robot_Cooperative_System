import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('robot_control')

    launch_actions = []

    # Launch uart_node for Robot 1, 2, and 3 in their respective namespaces
    for i in range(1, 4):
        robot_name = f'robot{i}'
        config_file = f'robot{i}.yaml'
        config_path = os.path.join(pkg_share, 'config', config_file)

        launch_actions.append(
            Node(
                package='robot_control',
                executable='uart_node',
                name='uart_node',
                namespace=robot_name,
                output='screen',
                parameters=[config_path]
            )
        )

    return LaunchDescription(launch_actions)
