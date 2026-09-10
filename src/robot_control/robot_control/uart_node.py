#!/usr/bin/env python3

import rclpy

from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy
)

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int32
from std_msgs.msg import String

import serial


# ============================================================
# UART NODE
#
# PURPOSE:
#
# Converts ROS robot commands into UART commands for ESP32.
#
#
# SUBSCRIBES:
#
#     /joint_commands
#     /gripper_command
#     /robot_speed
#     /robot_command
#     /control_mode
#
#
# SERIAL PROTOCOL:
#
#     J,<channel>,<angle>
#     S,<speed>
#     HOME
#     STOP
#
#
# SAFETY:
#
#     STOP is ALWAYS allowed.
#
#     HOME is allowed only in MANUAL mode.
#
#     HALTED blocks HOME and gripper motion.
#
# Joint motion authority is handled upstream by:
#
#     command_mux_node
#
# ============================================================


class UARTNode(Node):


    def __init__(self):

        super().__init__('uart_node')


        # ====================================================
        # CURRENT CONTROL MODE
        #
        # Safe startup default.
        #
        # Updated by retained /control_mode.
        # ====================================================

        self.control_mode = 'MANUAL'


        # ====================================================
        # CONTROL MODE QoS
        # ====================================================

        mode_qos = QoSProfile(

            depth=1,

            reliability=ReliabilityPolicy.RELIABLE,

            durability=DurabilityPolicy.TRANSIENT_LOCAL

        )


        # ====================================================
        # SERIAL CONFIGURATION
        # ====================================================

        self.port = '/dev/ttyACM0'

        self.baudrate = 115200


        try:

            self.ser = serial.Serial(

                self.port,

                self.baudrate,

                timeout=0.1

            )


            self.get_logger().info(

                f'Connected to ESP32: '

                f'{self.port} @ {self.baudrate}'

            )


        except serial.SerialException as e:

            self.ser = None


            self.get_logger().error(

                f'Failed to open serial port: {e}'

            )


        # ====================================================
        # JOINT CALIBRATION
        #
        # servo_angle =
        #
        # sign * (robot_angle / scale)
        # + offset
        #
        # ====================================================


        # ====================================================
        # JOINT SCALE
        #
        # physical_angle = scale * servo_angle
        # ====================================================

        self.joint_scale = [

            0.75,   # CH0 BASE

            0.75,   # CH1 J1

            0.75,   # CH2 J2

            0.50,   # CH3 J3

            0.50    # CH4 J4

        ]


        # ====================================================
        # JOINT DIRECTION
        #
        # +1 = same direction
        # -1 = opposite direction
        # ====================================================

        self.joint_sign = [

            1,     # CH0 BASE

            -1,     # CH1 J1

            +1,     # CH2 J2

            1,     # CH3 J3

            -1      # CH4 J4

        ]


        # ====================================================
        # JOINT ZERO OFFSET
        #
        # Change these later after physical calibration.
        # ====================================================

        self.joint_offset = [

            0.0,    # CH0 BASE

            0.0,    # CH1 J1

            0.0,    # CH2 J2

            0.0,    # CH3 J3

            0.0     # CH4 J4

        ]


        # ====================================================
        # CONTROL MODE SUBSCRIBER
        # ====================================================

        self.mode_subscription = self.create_subscription(

            String,

            '/control_mode',

            self.mode_callback,

            mode_qos

        )


        # ====================================================
        # JOINT COMMAND SUBSCRIBER
        #
        # ONLY command_mux_node should publish this topic.
        # ====================================================

        self.joint_subscription = self.create_subscription(

            Float64MultiArray,

            '/joint_commands',

            self.joint_callback,

            10

        )


        # ====================================================
        # GRIPPER SUBSCRIBER
        #
        # Final gripper command after arbitration.
        # ====================================================

        self.gripper_subscription = self.create_subscription(

            Float64MultiArray,

            '/gripper_command',

            self.gripper_callback,

            10

        )


        # ====================================================
        # SPEED SUBSCRIBER
        # ====================================================

        self.speed_subscription = self.create_subscription(

            Int32,

            '/robot_speed',

            self.speed_callback,

            10

        )


        # ====================================================
        # HOME / STOP SUBSCRIBER
        # ====================================================

        self.system_subscription = self.create_subscription(

            String,

            '/robot_command',

            self.system_callback,

            10

        )


        # ====================================================
        # STARTUP MESSAGE
        # ====================================================

        self.get_logger().info(

            '========================================'

        )

        self.get_logger().info(

            'UART NODE READY'

        )

        self.get_logger().info(

            '========================================'

        )

        self.get_logger().info(

            f'Initial control mode: {self.control_mode}'

        )

        self.get_logger().info(

            f'Scale:  {self.joint_scale}'

        )

        self.get_logger().info(

            f'Sign:   {self.joint_sign}'

        )

        self.get_logger().info(

            f'Offset: {self.joint_offset}'

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
    # SEND SERIAL COMMAND
    # ========================================================

    def send_command(self, command):


        if (

            self.ser is None

            or

            not self.ser.is_open

        ):


            self.get_logger().error(

                'Serial port unavailable'

            )

            return False


        try:


            self.ser.write(

                (command + '\n').encode()

            )


            self.ser.flush()


            return True


        except serial.SerialException as e:


            self.get_logger().error(

                f'Serial write failed: {e}'

            )


            return False


    # ========================================================
    # JOINT CALLBACK
    #
    # Robot angle -> servo angle
    #
    # Final joint commands should already be authorized by
    # command_mux_node.
    # ========================================================

    def joint_callback(self, msg):


        if len(msg.data) != 5:


            self.get_logger().error(

                'Expected 5 joint angles'

            )

            return


        for channel in range(5):


            robot_angle = float(

                msg.data[channel]

            )


            # ------------------------------------------------
            # CALIBRATION VALUES
            # ------------------------------------------------

            scale = self.joint_scale[channel]

            sign = self.joint_sign[channel]

            offset = self.joint_offset[channel]


            # ------------------------------------------------
            # ROBOT ANGLE -> SERVO COMMAND
            #
            # servo =
            #
            # sign * (robot / scale)
            # + offset
            # ------------------------------------------------

            servo_angle = (

                sign *

                (robot_angle / scale)

                + offset

            )


            # ------------------------------------------------
            # ESP32 SERVO RANGE
            # ------------------------------------------------

            servo_angle = max(

                -180.0,

                min(

                    180.0,

                    servo_angle

                )

            )


            servo_angle = int(

                round(

                    servo_angle

                )

            )


            # ------------------------------------------------
            # ESP32 PROTOCOL
            # ------------------------------------------------

            command = (

                f'J,{channel},{servo_angle}'

            )


            self.send_command(

                command

            )


            self.get_logger().info(

                f'CH{channel}: '

                f'robot={robot_angle:.1f} '

                f'-> servo={servo_angle} '

                f'[scale={scale}, '

                f'sign={sign}, '

                f'offset={offset}]'

            )


    # ========================================================
    # GRIPPER CALLBACK
    # ========================================================

    def gripper_callback(self, msg):


        # ----------------------------------------------------
        # HALTED = no gripper motion
        # ----------------------------------------------------

        if self.control_mode == 'HALTED':


            self.get_logger().warn(

                'Gripper command blocked: HALTED'

            )

            return


        if len(msg.data) != 1:


            self.get_logger().error(

                'Expected one gripper angle'

            )

            return


        angle = float(

            msg.data[0]

        )


        # ----------------------------------------------------
        # GRIPPER RANGE
        #
        # OPEN  = 0
        # CLOSE = -55
        # ----------------------------------------------------

        angle = max(

            -55.0,

            min(

                0.0,

                angle

            )

        )


        angle = int(

            round(

                angle

            )

        )


        command = (

            f'J,5,{angle}'

        )


        self.send_command(

            command

        )


        self.get_logger().info(

            f'Gripper: {angle}'

        )


    # ========================================================
    # SPEED CALLBACK
    # ========================================================

    def speed_callback(self, msg):


        speed = int(

            msg.data

        )


        speed = max(

            1,

            min(

                10,

                speed

            )

        )


        command = (

            f'S,{speed}'

        )


        self.send_command(

            command

        )


        self.get_logger().info(

            f'Speed = {speed}'

        )


    # ========================================================
    # HOME / STOP CALLBACK
    # ========================================================

    def system_callback(self, msg):


        command = (

            msg.data

            .strip()

            .upper()

        )


        # ----------------------------------------------------
        # STOP
        #
        # ALWAYS ALLOWED.
        #
        # STOP must work independently of command_mux,
        # trajectory_node, and GUI state.
        # ----------------------------------------------------

        if command == 'STOP':


            self.send_command(

                'STOP'

            )


            self.get_logger().warn(

                'SYSTEM COMMAND: STOP'

            )


            return


        # ----------------------------------------------------
        # HOME
        #
        # Only allowed in MANUAL mode.
        #
        # This prevents HOME from bypassing HALTED or
        # interrupting an active trajectory.
        # ----------------------------------------------------

        if command == 'HOME':


            if self.control_mode != 'MANUAL':


                self.get_logger().warn(

                    f'HOME blocked in mode: '

                    f'{self.control_mode}'

                )

                return


            self.send_command(

                'HOME'

            )


            self.get_logger().info(

                'SYSTEM COMMAND: HOME'

            )


            return


        # ----------------------------------------------------
        # UNKNOWN
        # ----------------------------------------------------

        self.get_logger().warn(

            f'Unknown system command: {command}'

        )


    # ========================================================
    # CLEANUP
    # ========================================================

    def destroy_node(self):


        if (

            self.ser is not None

            and

            self.ser.is_open

        ):


            self.ser.close()


            self.get_logger().info(

                'Serial port closed'

            )


        super().destroy_node()


# ============================================================
# MAIN
# ============================================================

def main(args=None):


    rclpy.init(

        args=args

    )


    node = UARTNode()


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