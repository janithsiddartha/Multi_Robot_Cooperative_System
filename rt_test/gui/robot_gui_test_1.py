import sys
import rclpy
from robot_gui_test import RobotGUI


def main():
    rclpy.init(args=sys.argv)
    node = RobotGUI(namespace='robot1')
    node.root.title('Robot 1 - Control Station')

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
