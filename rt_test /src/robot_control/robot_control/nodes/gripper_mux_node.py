#!/usr/bin/env python3

import rclpy

from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy
)

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String


# ============================================================
# GRIPPER MUX NODE
#
# PURPOSE:
#
# This node is the ONLY authority allowed to publish:
#
#     /gripper_command
#
#
# INPUTS:
#
#     /manual_gripper_command
#     /trajectory_gripper_command
#     /control_mode
#
#
# CONTROL RULES:
#
#     MANUAL
#         -> manual gripper commands allowed
#         -> trajectory gripper commands blocked
#
#     TRAJECTORY
#         -> trajectory gripper commands allowed
#         -> manual gripper commands blocked
#
#     HALTED
#         -> all gripper commands blocked
#
#
# OUTPUT:
#
#     /gripper_command
#
# ============================================================


class GripperMuxNode(Node):

    def __init__(self):

        super().__init__('gripper_mux_node')

        # ====================================================
        # CURRENT CONTROL MODE
        #
        # Safe boot default
        # ====================================================

        self.control_mode = 'MANUAL'


        # ====================================================
        # CONTROL MODE QoS
        #
        # Receives the last published mode even if this node
        # starts after the mode publisher.
        # ====================================================

        mode_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )


        # ====================================================
        # SUBSCRIBERS
        # ====================================================


        # ----------------------------------------------------
        # MANUAL GRIPPER COMMAND
        #
        # Published by GUI
        # ----------------------------------------------------

        self.manual_subscription = self.create_subscription(
            Float64MultiArray,
            'manual_gripper_command',
            self.manual_callback,
            10
        )


        # ----------------------------------------------------
        # TRAJECTORY GRIPPER COMMAND
        #
        # Published by trajectory_node
        # ----------------------------------------------------

        self.trajectory_subscription = self.create_subscription(
            Float64MultiArray,
            'trajectory_gripper_command',
            self.trajectory_callback,
            10
        )


        # ----------------------------------------------------
        # CONTROL MODE
        # ----------------------------------------------------

        self.mode_subscription = self.create_subscription(
            String,
            'control_mode',
            self.mode_callback,
            mode_qos
        )


        # ====================================================
        # FINAL GRIPPER COMMAND PUBLISHER
        #
        # ONLY THIS NODE publishes /gripper_command
        # ====================================================

        self.gripper_publisher = self.create_publisher(
            Float64MultiArray,
            'gripper_command',
            10
        )


        # ====================================================
        # STARTUP LOG
        # ====================================================

        self.get_logger().info(
            '========================================'
        )

        self.get_logger().info(
            'GRIPPER MUX NODE STARTED'
        )

        self.get_logger().info(
            '========================================'
        )

        self.get_logger().info(
            f'Initial mode: {self.control_mode}'
        )

        self.get_logger().info(
            'Input: /manual_gripper_command'
        )

        self.get_logger().info(
            'Input: /trajectory_gripper_command'
        )

        self.get_logger().info(
            'Input: /control_mode'
        )

        self.get_logger().info(
            'Output: /gripper_command'
        )


    # ========================================================
    # CONTROL MODE CALLBACK
    # ========================================================

    def mode_callback(self, msg):

        mode = msg.data.strip().upper()

        valid_modes = [
            'MANUAL',
            'TRAJECTORY',
            'HALTED'
        ]

        if mode not in valid_modes:

            self.get_logger().warn(
                f'Invalid control mode: {mode}'
            )

            return


        if mode != self.control_mode:

            old_mode = self.control_mode

            self.control_mode = mode

            self.get_logger().info(
                f'CONTROL MODE: '
                f'{old_mode} -> {mode}'
            )


    # ========================================================
    # VALIDATE GRIPPER COMMAND
    # ========================================================

    def validate_gripper_command(self, msg):

        if len(msg.data) != 1:

            self.get_logger().warn(
                'Invalid gripper command: expected one angle'
            )

            return False

        try:

            float(msg.data[0])

        except (TypeError, ValueError):

            self.get_logger().warn(
                'Invalid gripper command'
            )

            return False

        return True


    # ========================================================
    # MANUAL GRIPPER COMMAND
    #
    # Allowed only in MANUAL mode
    # ========================================================

    def manual_callback(self, msg):

        if self.control_mode != 'MANUAL':

            self.get_logger().debug(
                f'Manual gripper command blocked '
                f'(mode={self.control_mode})'
            )

            return


        if not self.validate_gripper_command(msg):

            return


        self.gripper_publisher.publish(msg)

        self.get_logger().info(
            f'Manual gripper command forwarded: '
            f'{msg.data[0]:.2f}'
        )


    # ========================================================
    # TRAJECTORY GRIPPER COMMAND
    #
    # Allowed only in TRAJECTORY mode
    # ========================================================

    def trajectory_callback(self, msg):

        if self.control_mode != 'TRAJECTORY':

            self.get_logger().debug(
                f'Trajectory gripper command blocked '
                f'(mode={self.control_mode})'
            )

            return


        if not self.validate_gripper_command(msg):

            return


        self.gripper_publisher.publish(msg)

        self.get_logger().info(
            f'Trajectory gripper command forwarded: '
            f'{msg.data[0]:.2f}'
        )


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = GripperMuxNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


# ============================================================

if __name__ == '__main__':

    main()
