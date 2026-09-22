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
# COMMAND MUX NODE
#
# PURPOSE:
#
# This node is the ONLY authority allowed to publish:
#
#     /joint_commands
#
#
# INPUT SOURCES:
#
#     /manual_joint_command
#     /ik_joint_command
#     /trajectory_joint_command
#
#
# CONTROL MODES:
#
#     MANUAL
#         Manual commands allowed
#         Direct IK commands allowed
#
#     TRAJECTORY
#         Only trajectory commands allowed
#
#     HALTED
#         All joint motion commands blocked
#
#
# OUTPUT:
#
#     /joint_commands
#
# ============================================================


class CommandMuxNode(Node):


    def __init__(self):

        super().__init__('command_mux_node')


        # ====================================================
        # CURRENT CONTROL MODE
        #
        # Safe boot default:
        #
        # MANUAL
        #
        # /control_mode will update this if a retained message
        # already exists.
        # ====================================================

        self.control_mode = 'MANUAL'


        # ====================================================
        # CONTROL MODE QoS
        #
        # TRANSIENT_LOCAL ensures a late-joining mux receives
        # the last published control mode.
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
        # MANUAL JOINT COMMAND
        #
        # GUI sliders / direct manual control
        # ----------------------------------------------------

        self.manual_subscription = self.create_subscription(

            Float64MultiArray,

            'manual_joint_command',

            self.manual_callback,

            10

        )


        # ----------------------------------------------------
        # DIRECT IK JOINT COMMAND
        #
        # GUI requests IK
        #
        # ik_node solves
        #
        # GUI receives solution and forwards it here
        # ----------------------------------------------------

        self.ik_subscription = self.create_subscription(

            Float64MultiArray,

            'ik_joint_command',

            self.ik_callback,

            10

        )


        # ----------------------------------------------------
        # TRAJECTORY JOINT COMMAND
        #
        # Published only by trajectory_node
        # ----------------------------------------------------

        self.trajectory_subscription = self.create_subscription(

            Float64MultiArray,

            'trajectory_joint_command',

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
        # FINAL JOINT COMMAND PUBLISHER
        #
        # ONLY THIS NODE publishes /joint_commands
        # ====================================================

        self.command_publisher = self.create_publisher(

            Float64MultiArray,

            'joint_commands',

            10

        )


        # ====================================================
        # STARTUP LOG
        # ====================================================

        self.get_logger().info(
            '========================================'
        )

        self.get_logger().info(
            'COMMAND MUX NODE STARTED'
        )

        self.get_logger().info(
            '========================================'
        )

        self.get_logger().info(
            f'Initial mode: {self.control_mode}'
        )

        self.get_logger().info(
            'Input: /manual_joint_command'
        )

        self.get_logger().info(
            'Input: /ik_joint_command'
        )

        self.get_logger().info(
            'Input: /trajectory_joint_command'
        )

        self.get_logger().info(
            'Input: /control_mode'
        )

        self.get_logger().info(
            'Output: /joint_commands'
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
    # VALIDATE JOINT COMMAND
    # ========================================================

    def validate_joint_command(self, msg):


        if len(msg.data) != 5:

            self.get_logger().warn(
                'Invalid joint command: expected 5 values'
            )

            return False


        return True


    # ========================================================
    # MANUAL COMMAND CALLBACK
    #
    # Allowed only in:
    #
    # MANUAL
    # ========================================================

    def manual_callback(self, msg):


        if self.control_mode != 'MANUAL':

            return


        if not self.validate_joint_command(msg):

            return


        self.command_publisher.publish(msg)


    # ========================================================
    # DIRECT IK COMMAND CALLBACK
    #
    # Allowed only in:
    #
    # MANUAL
    #
    # Direct IK is treated as operator-requested motion.
    # ========================================================

    def ik_callback(self, msg):


        if self.control_mode != 'MANUAL':

            return


        if not self.validate_joint_command(msg):

            return


        self.command_publisher.publish(msg)


    # ========================================================
    # TRAJECTORY COMMAND CALLBACK
    #
    # Allowed only in:
    #
    # TRAJECTORY
    # ========================================================

    def trajectory_callback(self, msg):


        if self.control_mode != 'TRAJECTORY':

            return


        if not self.validate_joint_command(msg):

            return


        self.command_publisher.publish(msg)


# ============================================================
# MAIN
# ============================================================

def main(args=None):


    rclpy.init(
        args=args
    )


    node = CommandMuxNode()


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
