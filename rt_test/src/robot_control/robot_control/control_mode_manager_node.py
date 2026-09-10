#!/usr/bin/env python3

import rclpy

from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy
)

from std_msgs.msg import String


# ============================================================
# CONTROL MODE MANAGER NODE
#
# PURPOSE:
#
# This node is the SINGLE authoritative publisher of:
#
#     /control_mode
#
#
# OTHER NODES NEVER PUBLISH /control_mode DIRECTLY.
#
# They request a mode transition using:
#
#     /control_mode_request
#
#
# VALID MODES:
#
#     MANUAL
#     TRAJECTORY
#     HALTED
#
#
# STATE TRANSITIONS:
#
#     MANUAL
#        |
#        | TRAJECTORY request
#        v
#     TRAJECTORY
#        |
#        | HALTED request (stop / failure)
#        | MANUAL request (normal completion)
#        v
#     HALTED or MANUAL
#        |
#        | MANUAL request (explicit RESUME)
#        v
#     MANUAL
#
#
# SAFETY RULE:
#
# HALTED -> TRAJECTORY is NOT allowed directly.
#
# Operator must explicitly:
#
#     HALTED -> MANUAL -> TRAJECTORY
#
# ============================================================


class ControlModeManagerNode(Node):

    def __init__(self):

        super().__init__('control_mode_manager_node')

        # ====================================================
        # CURRENT AUTHORITATIVE MODE
        #
        # Safe startup state:
        #
        # HALTED
        #
        # This is intentionally safer than booting directly
        # into MANUAL.
        #
        # ====================================================

        self.control_mode = 'HALTED'


        # ====================================================
        # CONTROL MODE QoS
        #
        # TRANSIENT_LOCAL:
        #
        # Late subscribers immediately receive the latest
        # authoritative control mode.
        #
        # ====================================================

        self.mode_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )


        # ====================================================
        # SUBSCRIBE:
        #
        # MODE CHANGE REQUESTS
        #
        # Multiple nodes may publish requests here.
        #
        # This is NOT the authoritative state topic.
        #
        # ====================================================

        self.request_subscription = self.create_subscription(
            String,
            '/control_mode_request',
            self.control_mode_request_callback,
            10
        )


        # ====================================================
        # PUBLISH:
        #
        # AUTHORITATIVE CONTROL MODE
        #
        # ONLY THIS NODE PUBLISHES THIS TOPIC.
        #
        # ====================================================

        self.mode_publisher = self.create_publisher(
            String,
            '/control_mode',
            self.mode_qos
        )


        # ====================================================
        # STARTUP LOG
        # ====================================================

        self.get_logger().info(
            '========================================'
        )

        self.get_logger().info(
            'CONTROL MODE MANAGER STARTED'
        )

        self.get_logger().info(
            '========================================'
        )

        self.get_logger().info(
            'Subscribed: /control_mode_request'
        )

        self.get_logger().info(
            'Publishing: /control_mode'
        )

        self.get_logger().info(
            f'Initial mode: {self.control_mode}'
        )


        # ====================================================
        # PUBLISH INITIAL STATE
        #
        # This retained message ensures every node knows
        # the system state immediately.
        #
        # ====================================================

        self.publish_control_mode()


    # ========================================================
    # MODE REQUEST CALLBACK
    # ========================================================

    def control_mode_request_callback(self, msg):

        requested_mode = msg.data.strip().upper()


        # ====================================================
        # VALIDATE REQUEST
        # ====================================================

        valid_modes = [

            'MANUAL',

            'TRAJECTORY',

            'HALTED'

        ]


        if requested_mode not in valid_modes:

            self.get_logger().warn(
                f'Ignoring invalid mode request: '
                f'{requested_mode}'
            )

            return


        # ====================================================
        # PROCESS STATE TRANSITION
        # ====================================================

        self.handle_mode_request(
            requested_mode
        )


    # ========================================================
    # HANDLE MODE REQUEST
    # ========================================================

    def handle_mode_request(self, requested_mode):


        # ----------------------------------------------------
        # REQUEST IS SAME AS CURRENT STATE
        # ----------------------------------------------------

        if requested_mode == self.control_mode:

            self.get_logger().info(
                f'Mode already {self.control_mode}'
            )

            # Re-publish authoritative state.
            #
            # This is harmless and helps ensure subscribers
            # have the latest retained value.

            self.publish_control_mode()

            return


        # ====================================================
        # CURRENT MODE: MANUAL
        # ====================================================

        if self.control_mode == 'MANUAL':


            # ------------------------------------------------
            # MANUAL -> TRAJECTORY
            # ------------------------------------------------

            if requested_mode == 'TRAJECTORY':

                self.set_control_mode(
                    'TRAJECTORY'
                )

                return


            # ------------------------------------------------
            # MANUAL -> HALTED
            # ------------------------------------------------

            if requested_mode == 'HALTED':

                self.set_control_mode(
                    'HALTED'
                )

                return


        # ====================================================
        # CURRENT MODE: TRAJECTORY
        # ====================================================

        elif self.control_mode == 'TRAJECTORY':


            # ------------------------------------------------
            # TRAJECTORY -> HALTED
            #
            # STOP / FAILURE / EMERGENCY
            # ------------------------------------------------

            if requested_mode == 'HALTED':

                self.set_control_mode(
                    'HALTED'
                )

                return


            # ------------------------------------------------
            # TRAJECTORY -> MANUAL
            #
            # Allowed on normal trajectory completion.
            # Operator STOP still uses TRAJECTORY -> HALTED.
            # ------------------------------------------------

            if requested_mode == 'MANUAL':

                self.set_control_mode(
                    'MANUAL'
                )

                return


        # ====================================================
        # CURRENT MODE: HALTED
        # ====================================================

        elif self.control_mode == 'HALTED':


            # ------------------------------------------------
            # HALTED -> MANUAL
            #
            # Explicit operator RESUME.
            # ------------------------------------------------

            if requested_mode == 'MANUAL':

                self.set_control_mode(
                    'MANUAL'
                )

                return


            # ------------------------------------------------
            # HALTED -> TRAJECTORY
            #
            # NOT allowed directly.
            # ------------------------------------------------

            if requested_mode == 'TRAJECTORY':

                self.get_logger().warn(
                    'Ignoring HALTED -> TRAJECTORY request. '
                    'Explicit MANUAL resume is required first.'
                )

                return


            # ------------------------------------------------
            # HALTED -> HALTED
            #
            # Already handled above, but kept conceptually
            # safe.
            # ------------------------------------------------

            if requested_mode == 'HALTED':

                return


    # ========================================================
    # SET CONTROL MODE
    # ========================================================

    def set_control_mode(self, new_mode):

        old_mode = self.control_mode

        self.control_mode = new_mode


        self.get_logger().info(
            f'CONTROL MODE: '
            f'{old_mode} -> {new_mode}'
        )


        self.publish_control_mode()


    # ========================================================
    # PUBLISH AUTHORITATIVE CONTROL MODE
    # ========================================================

    def publish_control_mode(self):

        mode_msg = String()

        mode_msg.data = self.control_mode


        self.mode_publisher.publish(
            mode_msg
        )


    # ========================================================
    # CLEANUP
    # ========================================================

    def destroy_node(self):

        self.get_logger().info(
            'Control mode manager shutting down'
        )

        super().destroy_node()


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(
        args=args
    )


    node = ControlModeManagerNode()


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