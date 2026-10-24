import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
import numpy as np

# ============================================================
# IMPORT IK SOLVER
# ============================================================

from robot_control.kinematics.ik_solver import ik
from robot_control.kinematics.fk_math import fk
from robot_control.kinematics.robot_kinematics import KinematicsConfig


# ============================================================
# IK ROS 2 NODE
# ============================================================

class IKNode(Node):

    def __init__(self):
        super().__init__('ik_node')

        self.config = KinematicsConfig(self)

        self.current_q_deg = np.zeros(self.config.num_joints)

        self.joint_subscription = self.create_subscription(
            Float64MultiArray,
            'joint_commands',
            self.joint_callback,
            10
        )

        self.gui_target_subscription = self.create_subscription(
            Float64MultiArray,
            'ik_gui_target',
            self.gui_target_callback,
            10
        )

        self.trajectory_target_subscription = self.create_subscription(
            Float64MultiArray,
            'ik_trajectory_target',
            self.trajectory_target_callback,
            10
        )

        self.legacy_target_subscription = self.create_subscription(
            Float64MultiArray,
            'ik_target',
            self.gui_target_callback,
            10
        )

        self.gui_solution_publisher = self.create_publisher(
            Float64MultiArray,
            'ik_gui_solution',
            10
        )

        self.gui_status_publisher = self.create_publisher(
            String,
            'ik_gui_status',
            10
        )

        self.legacy_solution_publisher = self.create_publisher(
            Float64MultiArray,
            'ik_solution',
            10
        )

        self.legacy_status_publisher = self.create_publisher(
            String,
            'ik_status',
            10
        )

        self.trajectory_solution_publisher = self.create_publisher(
            Float64MultiArray,
            'ik_trajectory_solution',
            10
        )

        self.trajectory_status_publisher = self.create_publisher(
            String,
            'ik_trajectory_status',
            10
        )


        self.get_logger().info('========================================')
        self.get_logger().info('5-DOF IK NODE STARTED')
        self.get_logger().info('========================================')

    def joint_callback(self, msg):
        if len(msg.data) != self.config.num_joints:
            self.get_logger().warn('Ignoring invalid joint command')
            return
        self.current_q_deg = np.array(msg.data, dtype=float)

    def gui_target_callback(self, msg):
        self.solve_ik_request(msg, source='GUI')

    def trajectory_target_callback(self, msg):
        self.solve_ik_request(msg, source='TRAJECTORY')

    def solve_ik_request(self, msg, source):
        if len(msg.data) != 3:
            error_message = 'ERROR: Invalid IK target. Expected [X, Y, Z]'
            self.get_logger().error(f'{source} | {error_message}')
            self.publish_status(source, error_message)
            return

        target_xyz = np.array(msg.data, dtype=float)

        try:
            current_q_rad = np.deg2rad(self.current_q_deg)
            result = ik(
                target=target_xyz,
                config=self.config,
                current_q=current_q_rad
            )

        except Exception as e:
            error_message = f'ERROR: IK solver exception: {e}'
            self.get_logger().error(f'{source} | {error_message}')
            self.publish_status(source, error_message)
            return

        if not result['success']:
            error_mm = result.get('error', -1.0)
            status_message = f'IK FAILED | Best error = {error_mm:.3f} mm'
            self.get_logger().warn(f'{source} | {status_message}')
            self.publish_status(source, status_message)
            return

        q_solution_rad = result['q']
        q_solution_deg = np.rad2deg(q_solution_rad)
        error_mm = result['error']

        achieved_xyz, _ = fk(q_solution_rad, self.config)

        solution_msg = Float64MultiArray()
        solution_msg.data = [float(q_solution_deg[i]) for i in range(self.config.num_joints)]

        if source == 'GUI':
            self.gui_solution_publisher.publish(solution_msg)
            self.legacy_solution_publisher.publish(solution_msg)
        elif source == 'TRAJECTORY':
            self.trajectory_solution_publisher.publish(solution_msg)

        status_message = f'IK SUCCESS | Error={error_mm:.4f} mm | Solutions={len(result["solutions"])}'
        self.publish_status(source, status_message)


    def publish_status(self, source, text):
        msg = String()
        msg.data = text
        if source == 'GUI':
            self.gui_status_publisher.publish(msg)
            self.legacy_status_publisher.publish(msg)
        elif source == 'TRAJECTORY':
            self.trajectory_status_publisher.publish(msg)

    def destroy_node(self):
        self.get_logger().info('IK node shutting down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = IKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()