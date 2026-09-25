import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray

import numpy as np
import matplotlib.pyplot as plt

from robot_control.fk_math import all_frame_positions


# ============================================================
# ROS 2 VISUALIZER NODE
# ============================================================

class RobotVisualizer(Node):

    def __init__(self):

        super().__init__('visualizer_node')


        # ====================================================
        # CURRENT JOINT ANGLES
        #
        # Stored in degrees
        #
        # [BASE, J1, J2, J3, J4]
        # ====================================================

        self.q_deg = np.zeros(5)


        # ====================================================
        # SUBSCRIBE TO JOINT COMMANDS
        # ====================================================

        self.subscription = self.create_subscription(

            Float64MultiArray,

            'joint_commands',

            self.joint_callback,

            10
        )


        # ====================================================
        # MATPLOTLIB INTERACTIVE MODE
        # ====================================================

        plt.ion()


        # ====================================================
        # CREATE FIGURE
        # ====================================================

        self.fig = plt.figure(
            figsize=(8, 8)
        )


        self.ax = self.fig.add_subplot(
            111,
            projection='3d'
        )


        self.fig.canvas.manager.set_window_title(
            '5-DOF Robot FK Visualizer'
        )


        plt.show(
            block=False
        )


        # ====================================================
        # UPDATE TIMER
        #
        # 30 FPS approximately
        # ====================================================

        self.timer = self.create_timer(

            0.033,

            self.update_visualization
        )


        self.get_logger().info(
            'Robot Visualizer Started'
        )


        self.get_logger().info(
            'Subscribed to /joint_commands'
        )


    # ========================================================
    # JOINT CALLBACK
    # ========================================================

    def joint_callback(
        self,
        msg
    ):

        if len(msg.data) != 5:

            self.get_logger().error(
                'Expected 5 joint angles'
            )

            return


        self.q_deg = np.array(
            msg.data,
            dtype=float
        )


    # ========================================================
    # CALCULATE ALL JOINT POSITIONS
    # ========================================================

    def calculate_joint_positions(self):

        q_rad = np.deg2rad(self.q_deg)
        return all_frame_positions(q_rad)


    # ========================================================
    # UPDATE VISUALIZATION
    # ========================================================

    def update_visualization(self):


        # ====================================================
        # CALCULATE FK
        # ====================================================

        points = (
            self.calculate_joint_positions()
        )

        PTCP_X = points[-1, 0]
        PTCP_Y = points[-1, 1]
        PTCP_Z = points[-1, 2]

        X = points[:, 0]

        Y = points[:, 1]

        Z = points[:, 2]


        # ====================================================
        # CLEAR OLD PLOT
        # ====================================================

        self.ax.clear()


        # ====================================================
        # DRAW ROBOT LINKS
        # ====================================================

        self.ax.plot(

            X,

            Y,

            Z,

            '-o',

            linewidth=3,

            markersize=8

        )


        # ====================================================
        # LABEL JOINTS
        # ====================================================

        labels = [
            'Base',
            'F1',
            'F2',
            'F3',
            'F4',
            'F5',
            'F6',
            'TCP'
        ]


        for i in range(
            len(points)
        ):

            self.ax.text(

                X[i],

                Y[i],

                Z[i],

                labels[i],

                fontsize=9

            )


        # ====================================================
        # DISPLAY TCP POSITION
        # ====================================================

        tcp_text = (

            f'TCP\n'

            f'X = {PTCP_X:.1f} mm\n'

            f'Y = {PTCP_Y:.1f} mm\n'

            f'Z = {PTCP_Z:.1f} mm'

        )


        # ====================================================
        # AXIS LABELS
        # ====================================================

        self.ax.set_xlabel(
            'X (mm)'
        )

        self.ax.set_ylabel(
            'Y (mm)'
        )

        self.ax.set_zlabel(
            'Z (mm)'
        )


        self.ax.set_title(

            '5-DOF Robot\n'

            f'θ = '
            f'[{self.q_deg[0]:.1f}, '
            f'{self.q_deg[1]:.1f}, '
            f'{self.q_deg[2]:.1f}, '
            f'{self.q_deg[3]:.1f}, '
            f'{self.q_deg[4]:.1f}]°'

        )


        # ====================================================
        # FIXED AXIS LIMITS
        # ====================================================

        max_reach = 450


        self.ax.set_xlim(

            -max_reach,

            max_reach

        )


        self.ax.set_ylim(

            -max_reach,

            max_reach

        )


        self.ax.set_zlim(

            -50,

            100 + max_reach

        )


        # ====================================================
        # EQUAL ASPECT RATIO
        # ====================================================

        self.ax.set_box_aspect(

            [

                1,

                1,

                1

            ]

        )


        # ====================================================
        # GRID
        # ====================================================

        self.ax.grid(
            True
        )


        # ====================================================
        # REFRESH WINDOW
        # ====================================================

        self.fig.canvas.draw_idle()

        self.fig.canvas.flush_events()

        plt.pause(
            0.001
        )


# ============================================================
# MAIN
# ============================================================

def main(args=None):


    rclpy.init(
        args=args
    )


    node = (
        RobotVisualizer()
    )


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
