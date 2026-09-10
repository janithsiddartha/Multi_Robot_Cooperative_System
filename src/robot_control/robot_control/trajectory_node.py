#!/usr/bin/env python3

import rclpy

from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy
)

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String

import numpy as np

from robot_control.robot_kinematics import (
    TRAJECTORY_DT as DT, 
    APPROACH_HEIGHT
)
from robot_control.trajectory_math import joint_trajectory





# ============================================================
# TRAJECTORY NODE
# ============================================================

class TrajectoryNode(Node):


    def __init__(self):

        super().__init__(
            'trajectory_node'
        )


        # ====================================================
        # ROBOT DATA
        # ====================================================

        self.current_q_deg = np.zeros(
            5,
            dtype=float
        )

        self.pick_xyz = None

        self.place_xyz = None


        # ====================================================
        # EXECUTION STATE
        # ====================================================

        self.state = 'IDLE'

        self.busy = False

        self.waiting_for_ik = False

        self.pending_waypoint = None

        self.current_target_xyz = None

        self.q_target_deg = None


        # ====================================================
        # MODE TRANSITION STATE
        #
        # Prevent first trajectory activity from being sent
        # in the same instant as the TRAJECTORY mode switch.
        # ====================================================

        self.pending_start = False

        self.control_mode = 'HALTED'


        self.mode_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )


        # ====================================================
        # PICK AND PLACE PLAN
        # ====================================================

        self.waypoints = []

        self.waypoint_index = 0


        # ====================================================
        # TRAJECTORY DATA
        # ====================================================

        self.trajectory = None

        self.trajectory_index = 0

        self.motion_active = False


        # ====================================================
        # SUBSCRIBERS
        # ====================================================


        # ----------------------------------------------------
        # PICK POSITION
        # ----------------------------------------------------

        self.pick_subscription = (
            self.create_subscription(
                Float64MultiArray,
                '/pick_position',
                self.pick_callback,
                10
            )
        )


        # ----------------------------------------------------
        # PLACE POSITION
        # ----------------------------------------------------

        self.place_subscription = (
            self.create_subscription(
                Float64MultiArray,
                '/place_position',
                self.place_callback,
                10
            )
        )


        # ----------------------------------------------------
        # TRAJECTORY COMMAND
        #
        # START
        # STOP
        # ----------------------------------------------------

        self.command_subscription = (
            self.create_subscription(
                String,
                '/trajectory_command',
                self.command_callback,
                10
            )
        )


        # ----------------------------------------------------
        # TRAJECTORY IK SOLUTION
        # ----------------------------------------------------

        self.ik_solution_subscription = (
            self.create_subscription(
                Float64MultiArray,
                '/ik_trajectory_solution',
                self.ik_solution_callback,
                10
            )
        )


        # ----------------------------------------------------
        # TRAJECTORY IK STATUS
        # ----------------------------------------------------

        self.ik_status_subscription = (
            self.create_subscription(
                String,
                '/ik_trajectory_status',
                self.ik_status_callback,
                10
            )
        )


        # ----------------------------------------------------
        # FINAL JOINT COMMAND
        #
        # Read-only system source of truth.
        # ----------------------------------------------------

        self.joint_subscription = (
            self.create_subscription(
                Float64MultiArray,
                '/joint_commands',
                self.joint_callback,
                10
            )
        )


        # ----------------------------------------------------
        # AUTHORITATIVE CONTROL MODE
        # ----------------------------------------------------

        self.mode_subscription = (
            self.create_subscription(
                String,
                '/control_mode',
                self.mode_callback,
                self.mode_qos
            )
        )


        # ====================================================
        # PUBLISHERS
        # ====================================================


        # ----------------------------------------------------
        # TRAJECTORY IK TARGET
        # ----------------------------------------------------

        self.ik_target_publisher = (
            self.create_publisher(
                Float64MultiArray,
                '/ik_trajectory_target',
                10
            )
        )


        # ----------------------------------------------------
        # TRAJECTORY JOINT COMMAND
        #
        # Goes ONLY to command_mux_node.
        # ----------------------------------------------------

        self.joint_publisher = (
            self.create_publisher(
                Float64MultiArray,
                '/trajectory_joint_command',
                10
            )
        )


        # ----------------------------------------------------
        # TRAJECTORY GRIPPER COMMAND
        #
        # Goes to gripper_mux_node.
        #
        # Does NOT publish directly to /gripper_command.
        # ----------------------------------------------------

        self.gripper_publisher = (
            self.create_publisher(
                Float64MultiArray,
                '/trajectory_gripper_command',
                10
            )
        )


        # ----------------------------------------------------
        # TRAJECTORY STATUS
        # ----------------------------------------------------

        self.status_publisher = (
            self.create_publisher(
                String,
                '/trajectory_status',
                10
            )
        )


        # ----------------------------------------------------
        # CONTROL MODE REQUEST
        #
        # trajectory_node requests mode transitions only.
        # control_mode_manager_node is the sole /control_mode
        # publisher.
        #
        # TRAJECTORY
        # MANUAL
        # HALTED
        # ----------------------------------------------------

        self.control_mode_request_publisher = (
            self.create_publisher(
                String,
                '/control_mode_request',
                10
            )
        )


        # ====================================================
        # 50 Hz EXECUTION TIMER
        # ====================================================

        self.timer = self.create_timer(
            DT,
            self.timer_callback
        )


        # ====================================================
        # STARTUP
        # ====================================================

        self.get_logger().info(
            '========================================'
        )

        self.get_logger().info(
            '5-DOF TRAJECTORY NODE STARTED'
        )

        self.get_logger().info(
            '========================================'
        )

        self.get_logger().info(
            'Update rate: 50 Hz'
        )

        self.get_logger().info(
            f'Approach height: {APPROACH_HEIGHT} mm'
        )

        self.get_logger().info(
            'IK target: /ik_trajectory_target'
        )

        self.get_logger().info(
            'IK solution: /ik_trajectory_solution'
        )

        self.get_logger().info(
            'IK status: /ik_trajectory_status'
        )

        self.get_logger().info(
            'Trajectory output: /trajectory_joint_command'
        )

        self.get_logger().info(
            'Trajectory gripper: /trajectory_gripper_command'
        )


        # ====================================================
        # INITIAL STATUS
        # ====================================================

        self.publish_status(
            'IDLE'
        )


    # ========================================================
    # PICK CALLBACK
    # ========================================================

    def pick_callback(self, msg):

        if len(msg.data) != 3:

            self.get_logger().error(
                'Pick position must contain [X,Y,Z]'
            )

            return

        self.pick_xyz = np.array(
            msg.data,
            dtype=float
        )

        self.get_logger().info(
            f'Pick position set: {self.pick_xyz}'
        )

        self.publish_status(
            'PICK_POSITION_RECEIVED'
        )


    # ========================================================
    # PLACE CALLBACK
    # ========================================================

    def place_callback(self, msg):

        if len(msg.data) != 3:

            self.get_logger().error(
                'Place position must contain [X,Y,Z]'
            )

            return

        self.place_xyz = np.array(
            msg.data,
            dtype=float
        )

        self.get_logger().info(
            f'Place position set: {self.place_xyz}'
        )

        self.publish_status(
            'PLACE_POSITION_RECEIVED'
        )


    # ========================================================
    # CURRENT JOINT CALLBACK
    #
    # Final commanded joint state.
    # ========================================================

    def joint_callback(self, msg):

        if len(msg.data) != 5:
            return

        self.current_q_deg = np.array(
            msg.data,
            dtype=float
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
            return

        self.control_mode = mode


    # ========================================================
    # COMMAND CALLBACK
    # ========================================================

    def command_callback(self, msg):

        command = (
            msg.data
            .strip()
            .upper()
        )


        # ----------------------------------------------------
        # START
        # ----------------------------------------------------

        if command == 'START':

            if self.busy:

                self.get_logger().warn(
                    'Trajectory already running'
                )

                return


            if self.pick_xyz is None:

                self.publish_status(
                    'ERROR: PICK_POSITION_NOT_SET'
                )

                return


            if self.place_xyz is None:

                self.publish_status(
                    'ERROR: PLACE_POSITION_NOT_SET'
                )

                return


            self.start_pick_and_place()


        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        elif command == 'STOP':

            self.stop_trajectory(
                halted=True
            )


        else:

            self.get_logger().warn(
                f'Unknown trajectory command: {command}'
            )


    # ========================================================
    # START PICK AND PLACE
    # ========================================================

    def start_pick_and_place(self):

        self.busy = True

        self.motion_active = False

        self.waiting_for_ik = False

        self.pending_start = False

        self.waypoint_index = 0


        # ----------------------------------------------------
        # CREATE WAYPOINTS
        # ----------------------------------------------------

        pre_pick = np.array([
            self.pick_xyz[0],
            self.pick_xyz[1],
            self.pick_xyz[2] + APPROACH_HEIGHT
        ])


        pick = self.pick_xyz.copy()


        lift = pre_pick.copy()


        pre_place = np.array([
            self.place_xyz[0],
            self.place_xyz[1],
            self.place_xyz[2] + APPROACH_HEIGHT
        ])


        place = self.place_xyz.copy()


        retreat = pre_place.copy()


        self.waypoints = [

            ('MOVING_TO_PRE_PICK', pre_pick),

            ('MOVING_TO_PICK', pick),

            ('LIFTING', lift),

            ('MOVING_TO_PRE_PLACE', pre_place),

            ('MOVING_TO_PLACE', place),

            ('RETREATING', retreat)

        ]


        # ----------------------------------------------------
        # ENTER TRAJECTORY MODE
        # ----------------------------------------------------

        self.publish_control_mode(
            'TRAJECTORY'
        )


        # ----------------------------------------------------
        # WAIT ONE TIMER TICK
        #
        # Avoid cross-topic ordering race between:
        #
        # /control_mode
        # /trajectory_joint_command
        #
        # ----------------------------------------------------

        self.pending_start = True


        self.publish_status(
            'PLANNING'
        )


        self.get_logger().info(
            'Pick and place sequence armed'
        )


    # ========================================================
    # START FIRST WAYPOINT
    #
    # Called one timer tick after TRAJECTORY mode is published.
    # ========================================================

    def begin_trajectory_sequence(self):

        if not self.busy:
            return

        self.pending_start = False

        self.get_logger().info(
            'Pick and place sequence started'
        )

        self.request_next_waypoint()


    # ========================================================
    # REQUEST NEXT WAYPOINT
    # ========================================================

    def request_next_waypoint(self):

        if not self.busy:
            return


        if self.waypoint_index >= len(
            self.waypoints
        ):

            self.finish_trajectory()

            return


        name, xyz = self.waypoints[
            self.waypoint_index
        ]


        self.state = name

        self.pending_waypoint = name

        self.current_target_xyz = xyz.copy()

        self.waiting_for_ik = True


        self.publish_status(
            f'{name} | SOLVING_IK'
        )


        msg = Float64MultiArray()

        msg.data = [
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2])
        ]


        self.get_logger().info(
            f'Requesting IK for {name}: '
            f'X={xyz[0]:.2f}, '
            f'Y={xyz[1]:.2f}, '
            f'Z={xyz[2]:.2f}'
        )


        self.ik_target_publisher.publish(
            msg
        )


    # ========================================================
    # IK SOLUTION CALLBACK
    # ========================================================

    def ik_solution_callback(self, msg):

        if not self.busy:
            return


        if not self.waiting_for_ik:
            return


        if len(msg.data) != 5:

            self.publish_status(
                'ERROR: INVALID_IK_SOLUTION'
            )

            self.stop_trajectory(
                halted=True
            )

            return


        self.q_target_deg = np.array(
            msg.data,
            dtype=float
        )


        self.get_logger().info(
            f'IK solution received for '
            f'{self.pending_waypoint}: '
            f'{self.q_target_deg}'
        )


        self.create_motion(
            self.current_q_deg.copy(),
            self.q_target_deg
        )


    # ========================================================
    # IK STATUS CALLBACK
    # ========================================================

    def ik_status_callback(self, msg):

        if not self.busy:
            return


        if not self.waiting_for_ik:
            return


        status = (
            msg.data
            .strip()
            .upper()
        )


        # ----------------------------------------------------
        # Failure only.
        #
        # IK success is handled by the solution callback.
        # ----------------------------------------------------

        if (
            'FAILED' in status
            or 'ERROR' in status
        ):

            self.get_logger().error(
                f'IK failed: {msg.data}'
            )


            self.publish_status(
                f'ERROR: {msg.data}'
            )


            self.stop_trajectory(
                halted=True
            )


    # ========================================================
    # CREATE JOINT MOTION
    # ========================================================

    def create_motion(
            self,
            q_start_deg,
            q_target_deg
    ):

        if not self.busy:
            return


        q_start_rad = np.deg2rad(
            q_start_deg
        )

        q_target_rad = np.deg2rad(
            q_target_deg
        )


        _, trajectory_rad, *_ = joint_trajectory(
            q_start_rad,
            q_target_rad,
            duration=3.0,
            dt=DT
        )


        self.trajectory = [

            np.rad2deg(q)

            for q in trajectory_rad

        ]


        self.trajectory_index = 0

        self.motion_active = True

        self.waiting_for_ik = False


        self.publish_status(
            self.state
        )


        self.get_logger().info(
            f'Trajectory generated: '
            f'{len(self.trajectory)} points'
        )


    # ========================================================
    # 50 Hz TIMER
    # ========================================================

    def timer_callback(self):


        # ----------------------------------------------------
        # START DELAY AFTER MODE SWITCH
        # ----------------------------------------------------

        if self.pending_start:

            if self.control_mode != 'TRAJECTORY':
                return

            self.begin_trajectory_sequence()

            return


        # ----------------------------------------------------
        # NO ACTIVE MOTION
        # ----------------------------------------------------

        if not self.motion_active:
            return


        if self.trajectory is None:
            return


        # ----------------------------------------------------
        # FINISHED CURRENT MOTION
        # ----------------------------------------------------

        if self.trajectory_index >= len(
            self.trajectory
        ):

            self.finish_current_waypoint()

            return


        # ----------------------------------------------------
        # GET NEXT TRAJECTORY POINT
        # ----------------------------------------------------

        q = self.trajectory[
            self.trajectory_index
        ]


        # ----------------------------------------------------
        # PUBLISH TO COMMAND MUX
        # ----------------------------------------------------

        msg = Float64MultiArray()

        msg.data = [
            float(q[0]),
            float(q[1]),
            float(q[2]),
            float(q[3]),
            float(q[4])
        ]


        self.joint_publisher.publish(
            msg
        )


        self.trajectory_index += 1


    # ========================================================
    # FINISH CURRENT WAYPOINT
    # ========================================================

    def finish_current_waypoint(self):

        if not self.busy:
            return


        self.motion_active = False

        self.trajectory = None


        name = self.pending_waypoint


        self.get_logger().info(
            f'Waypoint completed: {name}'
        )


        # ----------------------------------------------------
        # PICK ACTION
        # ----------------------------------------------------

        if name == 'MOVING_TO_PICK':

            self.publish_status(
                'CLOSING_GRIPPER'
            )

            self.send_gripper(
                -55.0
            )


        # ----------------------------------------------------
        # PLACE ACTION
        # ----------------------------------------------------

        elif name == 'MOVING_TO_PLACE':

            self.publish_status(
                'OPENING_GRIPPER'
            )

            self.send_gripper(
                0.0
            )


        # ----------------------------------------------------
        # NEXT WAYPOINT
        # ----------------------------------------------------

        self.waypoint_index += 1

        self.request_next_waypoint()


    # ========================================================
    # SEND TRAJECTORY GRIPPER COMMAND
    # ========================================================

    def send_gripper(self, angle):

        msg = Float64MultiArray()

        msg.data = [
            float(angle)
        ]


        self.gripper_publisher.publish(
            msg
        )


        self.get_logger().info(
            f'Trajectory gripper command: {angle}'
        )


    # ========================================================
    # STOP TRAJECTORY
    #
    # STOP / ERROR -> HALTED
    # ========================================================

    def stop_trajectory(self, halted=True):

        self.busy = False

        self.motion_active = False

        self.waiting_for_ik = False

        self.pending_start = False


        self.trajectory = None

        self.trajectory_index = 0


        self.waypoints = []

        self.waypoint_index = 0


        self.pending_waypoint = None

        self.current_target_xyz = None

        self.q_target_deg = None


        self.state = 'STOPPED'


        self.publish_status(
            'STOPPED'
        )


        if halted:

            self.publish_control_mode(
                'HALTED'
            )


        self.get_logger().warn(
            'Trajectory stopped'
        )


    # ========================================================
    # FINISH TRAJECTORY
    #
    # Successful completion -> MANUAL
    # ========================================================

    def finish_trajectory(self):

        self.busy = False

        self.motion_active = False

        self.waiting_for_ik = False

        self.pending_start = False


        self.trajectory = None

        self.pending_waypoint = None


        self.state = 'COMPLETE'


        self.publish_status(
            'COMPLETE'
        )


        self.publish_control_mode(
            'MANUAL'
        )


        self.get_logger().info(
            'Pick and place COMPLETE'
        )


    # ========================================================
    # CONTROL MODE PUBLISHER
    # ========================================================

    def publish_control_mode(self, mode):

        mode = (
            mode
            .strip()
            .upper()
        )


        valid_modes = [

            'MANUAL',

            'TRAJECTORY',

            'HALTED'

        ]


        if mode not in valid_modes:

            self.get_logger().error(
                f'Invalid control mode: {mode}'
            )

            return


        msg = String()

        msg.data = mode


        self.control_mode_request_publisher.publish(
            msg
        )


        self.get_logger().info(
            f'CONTROL MODE REQUESTED: {mode}'
        )


    # ========================================================
    # STATUS PUBLISHER
    # ========================================================

    def publish_status(self, text):

        msg = String()

        msg.data = text


        self.status_publisher.publish(
            msg
        )


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = TrajectoryNode()


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