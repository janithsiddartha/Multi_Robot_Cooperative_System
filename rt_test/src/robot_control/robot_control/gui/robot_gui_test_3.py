#!/usr/bin/env python3

import sys
from pathlib import Path

import os
import rclpy
from ament_index_python.packages import get_package_share_directory


def _find_gui_dir():
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        candidate = parent / 'gui'
        if candidate.is_dir() and (candidate / 'robot_gui_test.py').is_file():
            return candidate
    raise FileNotFoundError('Could not find the workspace gui directory for robot_gui_test_3')


GUI_DIR = _find_gui_dir()
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from robot_gui_test import RobotGUI


def main():
    config_path = os.path.join(get_package_share_directory('robot_control'), 'config', 'robot3.yaml')
    args = sys.argv.copy()
    if '--ros-args' not in args:
        args.extend(['--ros-args', '--params-file', config_path])
        
    rclpy.init(args=args)
    node = RobotGUI(namespace='robot3')
    node.root.title(node.config.gui_title)

    try:
        node.root.after(10, node.process_ros)
        node.root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
