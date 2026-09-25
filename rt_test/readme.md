                    ┌───────────────────────┐
                    │       ROS 2 GUI       │
                    │                       │
                    │ Joint sliders         │
                    │ Gripper control       │
                    │ Speed control         │
                    │ HOME button           │
                    │ STOP button           │
                    └───────────┬───────────┘
                                │
                                │ ROS 2 Topics
                                ▼
                    ┌───────────────────────┐
                    │       UART Node       │
                    │                       │
                    │ Angle scaling         │
                    │ Direction mapping     │
                    │ Zero offsets          │
                    │ ROS → Serial          │
                    └───────────┬───────────┘
                                │
                                │ USB Serial
                                │ 115200 baud
                                ▼
                    ┌───────────────────────┐
                    │         ESP32         │
                    │                       │
                    │ UART parser           │
                    │ Servo control         │
                    │ Smooth motion         │
                    │ Speed control         │
                    └───────────┬───────────┘
                                │
                                │ I2C
                                ▼
                    ┌───────────────────────┐
                    │       PCA9685         │
                    │    Servo Driver       │
                    └───────────┬───────────┘
                                │
                                ▼
                  ┌─────────────────────────┐
                  │       ROBOT ARM         │
                  │                         │
                  │ CH0 → Base              │
                  │ CH1 → J1                │
                  │ CH2 → J2                │
                  │ CH3 → J3                │
                  │ CH4 → J4                │
                  │ CH5 → Gripper           │
                  └─────────────────────────┘

Port:/dev/ttyACM0
Baud rate:115200


UART NODE

SUBSCRIBES:
✓ /joint_commands
✓ /gripper_command
✓ /robot_speed
✓ /robot_command

PUBLISHES:
Currently nothing


ROS 2 Topics
/joint_commands
Message Type
std_msgs/msg/Float64MultiArray
Format
[
    BASE,
    J1,
    J2,
    J3,
    J4
]

/gripper_command
Message Type
std_msgs/msg/Float64MultiArray
Format
[gripper_angle]

Current convention:
0°    = OPEN
-55°  = CLOSED

/robot_speed
Message Type
std_msgs/msg/Int32
Range
1 to 10

/robot_command
Message Type
std_msgs/msg/String
Supported commands:
HOME
STOP
Example:
HOME
moves the robot to its configured home position.
STOP
stops the current motion according to the ESP32 firmware implementation.

========================================================
5-DOF ROBOT GEOMETRIC FORWARD KINEMATICS
========================================================

Link lengths:

L1 = 116.62 mm
L2 = 120.00 mm
L3 = 90.25 mm
L4 = 30.00 mm
L5 = 118.63 mm

TCP / Gripper offsets:

dx = -21.00 mm
dy = -8.16 mm
dz = 0.00 mm

Joint angles are RELATIVE joint angles:

theta1 = base yaw
theta2 = joint 2 angle
theta3 = joint 3 angle
theta4 = roll about L4 axis
theta5 = joint 5 angle


========================================================
FORWARD KINEMATICS
========================================================

XE =
(
    L2*sin(theta2)
    + (L3 + L4)*sin(theta2 + theta3)
    + L5*(
          cos(theta5)*sin(theta2 + theta3)
          + sin(theta5)*cos(theta4)*cos(theta2 + theta3)
        )
)*cos(theta1)
- L5*sin(theta5)*sin(theta4)*sin(theta1)
+ dx


YE =
(
    L2*sin(theta2)
    + (L3 + L4)*sin(theta2 + theta3)
    + L5*(
          cos(theta5)*sin(theta2 + theta3)
          + sin(theta5)*cos(theta4)*cos(theta2 + theta3)
        )
)*sin(theta1)
+ L5*sin(theta5)*sin(theta4)*cos(theta1)
+ dy


ZE =
L1
+ L2*cos(theta2)
+ (L3 + L4)*cos(theta2 + theta3)
+ L5*(
      cos(theta5)*cos(theta2 + theta3)
      - sin(theta5)*cos(theta4)*sin(theta2 + theta3)
    )
+ dz

                 ┌──────────────────────┐
                 │   CONTROL PLANE      │
                 │                      │
                 │ control_mode_manager │
                 └──────────┬───────────┘
                            │
                     /control_mode
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
        command_mux     gripper_mux       GUI


                 ┌──────────────────────┐
                 │   MOTION PLANE       │
                 └──────────────────────┘

 Manual ────────┐
 IK ────────────┼──► command_mux ──► /joint_commands
 Trajectory ────┘                         │
                                          ├── UART → ESP32
                                          ├── FK
                                          ├── Visualizer
                                          ├── GUI display
                                          ├── IK current state
                                          └── Trajectory state


                 ┌──────────────────────┐
                 │   GRIPPER PLANE      │
                 └──────────────────────┘

 Manual gripper ──────┐
                      ├──► gripper_mux ──► UART
 Trajectory gripper ──┘


                 ┌──────────────────────┐
                 │     IK PLANE         │
                 └──────────────────────┘

 GUI IK ───────────────► IK node ──► GUI IK result/status

 Trajectory IK ────────► IK node ──► Trajectory result/status