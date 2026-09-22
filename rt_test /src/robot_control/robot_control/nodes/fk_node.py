#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import Float64MultiArray

from robot_control.kinematics.fk_math import fk
from robot_control.kinematics.robot_kinematics import KinematicsConfig


class FKNode(Node):

    def __init__(self):
        super().__init__('fk_node')

        self.config = KinematicsConfig(self)

        # ====================================================
        # SUBSCRIBE
        # ====================================================
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'joint_commands',
            self.joint_callback,
            10
        )

        # ====================================================
        # PUBLISH
        # ====================================================
        self.publisher = self.create_publisher(
            Float64MultiArray,
            'end_effector_position',
            10
        )

        # ====================================================
        # STARTUP LOG
        # ====================================================
        self.get_logger().info('========================================')
        self.get_logger().info('FK NODE STARTED (MODIFIED DH KINEMATICS)')
        self.get_logger().info('========================================')
        self.get_logger().info('Subscribed: joint_commands')
        self.get_logger().info('Publishing: end_effector_position')
        self.get_logger().info('Joint units: degrees')
        self.get_logger().info('Position units: mm')

    # ========================================================
    # JOINT COMMAND CALLBACK
    # ========================================================
    def joint_callback(self, msg):

        # ----------------------------------------------------
        # VALIDATE INPUT
        # ----------------------------------------------------
        if len(msg.data) != 5:
            self.get_logger().error('FK ERROR: Expected 5 joint angles')
            return

        # ----------------------------------------------------
        # READ JOINT ANGLES
        # ----------------------------------------------------
        q_deg = np.array(msg.data, dtype=float)

        # ----------------------------------------------------
        # RUN FORWARD KINEMATICS USING MODIFIED DH
        # ----------------------------------------------------
        q_rad = np.deg2rad(q_deg)
        
        # fk returns (position, transformation_matrix)
        position, _ = fk(q_rad, self.config)

        # ----------------------------------------------------
        # PUBLISH END-EFFECTOR POSITION
        # ----------------------------------------------------
        output_msg = Float64MultiArray()
        output_msg.data = [
            float(position[0]),
            float(position[1]),
            float(position[2])
        ]

        self.publisher.publish(output_msg)



# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = FKNode()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


# ============================================================

if __name__ == '__main__':

    main()