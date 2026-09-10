import rclpy

from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String

import numpy as np


# ============================================================
# IMPORT IK SOLVER
# ============================================================

from robot_control.ik_solver import ik
from robot_control.fk_math import fk


# ============================================================
# IK ROS 2 NODE
# ============================================================

class IKNode(Node):

    def __init__(self):

        super().__init__('ik_node')

        # ====================================================
        # CURRENT JOINT CONFIGURATION
        #
        # Stored in degrees
        #
        # [theta1, theta2, theta3, theta4, theta5]
        # ====================================================

        self.current_q_deg = np.zeros(5)


        # ====================================================
        # SUBSCRIBE: CURRENT ROBOT JOINT COMMAND
        #
        # This is the single source of truth for the
        # commanded robot configuration.
        #
        # Format:
        #
        # [theta1, theta2, theta3, theta4, theta5]
        #
        # Units: degrees
        # ====================================================

        self.joint_subscription = self.create_subscription(

            Float64MultiArray,

            '/joint_commands',

            self.joint_callback,

            10

        )


        # ====================================================
        # GUI IK TARGET INPUT
        #
        # Format:
        #
        # [X, Y, Z]
        #
        # Units: mm
        # ====================================================

        self.gui_target_subscription = self.create_subscription(

            Float64MultiArray,

            '/ik_gui_target',

            self.gui_target_callback,

            10

        )


        # ====================================================
        # TRAJECTORY IK TARGET INPUT
        #
        # Format:
        #
        # [X, Y, Z]
        #
        # Units: mm
        # ====================================================

        self.trajectory_target_subscription = self.create_subscription(

            Float64MultiArray,

            '/ik_trajectory_target',

            self.trajectory_target_callback,

            10

        )


        # Legacy GUI IK target used by robot_gui_OLD.

        self.legacy_target_subscription = self.create_subscription(

            Float64MultiArray,

            '/ik_target',

            self.gui_target_callback,

            10

        )


        # ====================================================
        # GUI IK SOLUTION OUTPUT
        # ====================================================

        self.gui_solution_publisher = self.create_publisher(

            Float64MultiArray,

            '/ik_gui_solution',

            10

        )


        # ====================================================
        # GUI IK STATUS OUTPUT
        # ====================================================

        self.gui_status_publisher = self.create_publisher(

            String,

            '/ik_gui_status',

            10

        )


        # Legacy GUI outputs used by robot_gui_OLD.

        self.legacy_solution_publisher = self.create_publisher(

            Float64MultiArray,

            '/ik_solution',

            10

        )


        self.legacy_status_publisher = self.create_publisher(

            String,

            '/ik_status',

            10

        )


        # ====================================================
        # TRAJECTORY IK SOLUTION OUTPUT
        # ====================================================

        self.trajectory_solution_publisher = self.create_publisher(

            Float64MultiArray,

            '/ik_trajectory_solution',

            10

        )


        # ====================================================
        # TRAJECTORY IK STATUS OUTPUT
        # ====================================================

        self.trajectory_status_publisher = self.create_publisher(

            String,

            '/ik_trajectory_status',

            10

        )


        # ====================================================
        # STARTUP LOG
        # ====================================================

        self.get_logger().info(
            '========================================'
        )

        self.get_logger().info(
            '5-DOF IK NODE STARTED'
        )

        self.get_logger().info(
            '========================================'
        )

        self.get_logger().info(
            'Subscribed: /joint_commands'
        )

        self.get_logger().info(
            'Subscribed: /ik_gui_target'
        )

        self.get_logger().info(
            'Subscribed: /ik_trajectory_target'
        )

        self.get_logger().info(
            'Publishing: /ik_gui_solution'
        )

        self.get_logger().info(
            'Publishing: /ik_gui_status'
        )

        self.get_logger().info(
            'Publishing: /ik_trajectory_solution'
        )

        self.get_logger().info(
            'Publishing: /ik_trajectory_status'
        )


    # ========================================================
    # CURRENT JOINT CALLBACK
    # ========================================================

    def joint_callback(self, msg):

        if len(msg.data) != 5:

            self.get_logger().warn(
                'Ignoring invalid joint command'
            )

            return


        self.current_q_deg = np.array(

            msg.data,

            dtype=float

        )


    # ========================================================
    # GUI IK TARGET CALLBACK
    # ========================================================

    def gui_target_callback(self, msg):

        self.solve_ik_request(

            msg,

            source='GUI'

        )


    # ========================================================
    # TRAJECTORY IK TARGET CALLBACK
    # ========================================================

    def trajectory_target_callback(self, msg):

        self.solve_ik_request(

            msg,

            source='TRAJECTORY'

        )


    # ========================================================
    # COMMON IK SOLVER
    #
    # Both GUI and trajectory requests use the same IK
    # mathematics.
    #
    # The source determines where the result is published.
    # ========================================================

    def solve_ik_request(self, msg, source):


        # ----------------------------------------------------
        # VALIDATE TARGET
        # ----------------------------------------------------

        if len(msg.data) != 3:

            error_message = (
                'ERROR: Invalid IK target. '
                'Expected [X, Y, Z]'
            )

            self.get_logger().error(

                f'{source} | {error_message}'

            )

            self.publish_status(

                source,

                error_message

            )

            return


        # ----------------------------------------------------
        # READ TARGET
        # ----------------------------------------------------

        target_xyz = np.array(

            msg.data,

            dtype=float

        )


        x = target_xyz[0]

        y = target_xyz[1]

        z = target_xyz[2]


        self.get_logger().info(

            f'{source} IK TARGET RECEIVED | '

            f'X={x:.2f}, '

            f'Y={y:.2f}, '

            f'Z={z:.2f} mm'

        )


        # ----------------------------------------------------
        # RUN IK
        # ----------------------------------------------------

        try:

            current_q_rad = np.deg2rad(self.current_q_deg)
            result = ik(
                target=target_xyz,
                current_q=current_q_rad
            )


        except Exception as e:


            error_message = (

                f'ERROR: IK solver exception: {e}'

            )


            self.get_logger().error(

                f'{source} | {error_message}'

            )


            self.publish_status(

                source,

                error_message

            )

            return


        # ----------------------------------------------------
        # CHECK IK RESULT
        # ----------------------------------------------------

        if not result['success']:


            error_mm = result.get(
                'error',
                -1.0
            )


            status_message = (

                f'IK FAILED | '

                f'Best error = '

                f'{error_mm:.3f} mm'

            )


            self.get_logger().warn(

                f'{source} | {status_message}'

            )


            self.publish_status(

                source,

                status_message

            )

            return


        # ----------------------------------------------------
        # EXTRACT SOLUTION
        # ----------------------------------------------------

        q_solution_rad = result['q']
        q_solution_deg = np.rad2deg(q_solution_rad)

        error_mm = result['error']

        achieved_xyz, _ = fk(q_solution_rad)


        # ----------------------------------------------------
        # LOG RESULT
        # ----------------------------------------------------

        self.get_logger().info(

            f'{source} IK SUCCESS'

        )


        self.get_logger().info(

            f'Joint solution [deg]: '

            f'[{q_solution_deg[0]:.2f}, '

            f'{q_solution_deg[1]:.2f}, '

            f'{q_solution_deg[2]:.2f}, '

            f'{q_solution_deg[3]:.2f}, '

            f'{q_solution_deg[4]:.2f}]'

        )


        self.get_logger().info(

            f'Achieved XYZ [mm]: '

            f'[{achieved_xyz[0]:.2f}, '

            f'{achieved_xyz[1]:.2f}, '

            f'{achieved_xyz[2]:.2f}]'

        )


        self.get_logger().info(

            f'Position error: '

            f'{error_mm:.4f} mm'

        )


        # ----------------------------------------------------
        # CREATE SOLUTION MESSAGE
        # ----------------------------------------------------

        solution_msg = Float64MultiArray()


        solution_msg.data = [

            float(q_solution_deg[0]),

            float(q_solution_deg[1]),

            float(q_solution_deg[2]),

            float(q_solution_deg[3]),

            float(q_solution_deg[4])

        ]


        # ----------------------------------------------------
        # PUBLISH TO CORRECT CHANNEL
        # ----------------------------------------------------

        if source == 'GUI':

            self.gui_solution_publisher.publish(

                solution_msg

            )

            self.legacy_solution_publisher.publish(

                solution_msg

            )


        elif source == 'TRAJECTORY':

            self.trajectory_solution_publisher.publish(

                solution_msg

            )


        # ----------------------------------------------------
        # PUBLISH SUCCESS STATUS
        # ----------------------------------------------------

        status_message = (
            f'IK SUCCESS | '
            f'Error={error_mm:.4f} mm | '
            f'Solutions={len(result["solutions"])}'
        )


        self.publish_status(

            source,

            status_message

        )


    # ========================================================
    # PUBLISH STATUS TO CORRECT CHANNEL
    # ========================================================

    def publish_status(self, source, text):


        msg = String()

        msg.data = text


        if source == 'GUI':

            self.gui_status_publisher.publish(

                msg

            )

            self.legacy_status_publisher.publish(

                msg

            )


        elif source == 'TRAJECTORY':

            self.trajectory_status_publisher.publish(

                msg

            )


    # ========================================================
    # CLEANUP
    # ========================================================

    def destroy_node(self):

        self.get_logger().info(

            'IK node shutting down'

        )

        super().destroy_node()


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(

        args=args

    )


    node = IKNode()


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