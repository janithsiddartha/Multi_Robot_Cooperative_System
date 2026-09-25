import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch.actions import GroupAction

def generate_launch_description():
    pkg_share = get_package_share_directory('robot_control')
    bringup_launch_path = os.path.join(pkg_share, 'launch', 'robot_bringup.launch.py')

    launch_actions = []
    
    # Bring up robot1, robot2, and robot3
    for i in range(1, 4):
        robot_name = f'robot{i}'
        config_file = f'robot{i}.yaml'
        
        launch_actions.append(
            GroupAction([
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(bringup_launch_path),
                    launch_arguments={
                        'robot_name': robot_name,
                        'config': config_file
                    }.items()
                )
            ])
        )
        
    return LaunchDescription(launch_actions)
