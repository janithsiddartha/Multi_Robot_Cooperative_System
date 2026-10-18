# Multi-Robot Cooperative Pick-and-Place & Synchronized Handover System

An advanced multi-robot coordination and manipulation framework developed in ROS 2 (Jazzy) for heterogeneous robotic manipulators. The platform orchestrates 3 physical robotic arms cooperating in shared workspaces to perform pick-and-place, adaptive object-pose grasping, synchronized multi-robot handovers, and real-time whole-body collision avoidance.

---

## Table of Contents

1. [System Overview & Manipulators](#1-system-overview--manipulators)
2. [Software Architecture & Control Planes](#2-software-architecture--control-planes)
3. [Evolution of Stages (1, 1.5, 2, 2.5)](#3-evolution-of-stages)
4. [Configuration System](#4-configuration-system)
5. [Installation & Build Instructions](#5-installation--build-instructions)
6. [How to Run & GUI Operation Guide](#6-how-to-run--gui-operation-guide)
7. [ROS 2 Interface & Topics](#7-ros-2-interface--topics)
8. [Automated Test Suite](#8-automated-test-suite)

---

## 1. System Overview & Manipulators

The workcell comprises three heterogeneous articulated robotic arms mounted around a central shared workspace table:

```
                          World +Y (Ceiling / North)
                                      ▲
                                      │
                                 ┌─────────┐
                                 │ Robot 3 │ (6-DOF)
                                 │ [0,250] │
                                 └────┬────┘
                                      │
                                  ┌───▼───┐
                                  │Shared │
                                  │ Hand- │
                                  │ over  │
                                  │ Area  │
                                  └───┬───┘
                                      │
               ┌─────────┐            │            ┌─────────┐
(West) ◄───────┤ Robot 1 ├────────────┴────────────┤ Robot 2 ├───────► (East)
               │[-250, 0]│                         │ [250, 0]│
               └─────────┘                         └─────────┘
                                 Robot Base Placements
```

| Robot Identifier | Kinematics Structure | Degrees of Freedom | Base Position $(X, Y, Z)$ [mm] | Base Yaw Orientation | End-Effector Gripper |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`robot1`** | Articulated Anthromorphic | **5-DOF** | `[-250.0, 0.0, 0.0]` | $0.0^\circ$ (Facing $+X$) | Servo Parallel Gripper |
| **`robot2`** | Articulated Anthromorphic | **5-DOF** | `[250.0, 0.0, 0.0]` | $180.0^\circ$ (Facing $-X$) | Servo Parallel Gripper |
| **`robot3`** | Full Articulated Arm | **6-DOF** | `[0.0, 250.0, 0.0]` | $90.0^\circ$ (Facing $-Y$) | Servo Parallel Gripper |

- **Hardware Control**: Each robot arm is driven by an ESP32 microcontroller and PCA9685 16-channel PWM servo driver connected over USB-UART serial at 115200 baud.
- **Actuation**: Precision bus servos with real-time feedback and trajectory execution.

---

## 2. Software Architecture & Control Planes

The software stack separates planning, coordination, simulation, and hardware transmission into clean decoupled planes:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        GUI & VISUALIZATION PLANE                       │
│    Stage 1 GUI  │  Stage 1.5 GUI  │  Stage 2 GUI  │  Stage 2.5 GUI     │
│   • 3D Matplotlib Visualizer (Robot Links, TCP Frames, Point Clouds)   │
│   • Workspace Clouds • Real-time Clearance Monitor • Safe Preview      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Plan / Simulate / Execute
┌───────────────────────────────────▼────────────────────────────────────┐
│                        MOTION PLANNING ENGINE                          │
│   • Target Parsing & Normalization (XYZ mm, Yaw [0, 360)°)             │
│   • Stage 2.5 Object-Pose IK & Adaptive Approach Tilt Solver           │
│   • Stage 2 6D Orientation Error Formulations                          │
│   • Stage 1.5 Whole-Body Link-to-Link Collision Distance Engine        │
│   • Stage 1 Damped Least-Squares Position IK & Multi-Start Multi-Robot │
│   • Zero-Recalculation Plan Caching Engine                             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Trajectory Stream
┌───────────────────────────────────▼────────────────────────────────────┐
│                        ROS 2 HARDWARE INTERFACE                        │
│   • Topic Multiplexers (command_mux, gripper_mux)                      │
│   • Joint Command Publishers (`/robotX/joint_commands`)                │
│   • Gripper Command Publishers (`/robotX/gripper_command`)             │
│   • UART Serial Bridge Nodes (`uart_node`)                             │
│   • ESP32 Firmware Execution via PCA9685                               │
└────────────────────────────────────────────────────────────────────────┘
```

### Dual-Execution Architecture

1. **SIMULATE (3D Preview Only)**:
   - Visualizes the full 3D robotic motion path in the interactive Matplotlib canvas.
   - Evaluates link-to-link distances throughout the trajectory.
   - **Hardware output is strictly disabled** (no serial commands sent to physical robots).
   - Generates and verifies collision clearance; caches the verified plan.
2. **EXECUTE (Real Robot)**:
   - Reuses the cached simulation plan if coordinates match (**saving ~20 seconds** of re-computation).
   - Prompts a safety confirmation dialog.
   - Streams synchronized trajectory waypoints to `/joint_commands` and `/gripper_command` topics at configured step intervals.

---

## 3. Evolution of Stages

The framework is structured into four progressive stages, each addressing higher levels of manipulation autonomy and geometric fidelity.

### Stage 1: Basic Pick-and-Place with Manual Handover
- **Inverse Kinematics**: Numerical Damped Least Squares (DLS) Position IK reaching 3D Cartesian coordinates $(X, Y, Z)$.
- **Multi-Robot Selection**: If target falls in the workspace intersection of multiple robots, prompts an interactive selection dialog (`select_robot`).
- **Handover**: Executes fixed, pre-calibrated manual handover joint configurations from configuration files (`handover.yaml`).

### Stage 1.5: Whole-Body Link Clearance & Collision Avoidance
- **Continuous Distance Engine**: Evaluates minimum Euclidean 3D line-segment distances between all pairs of robotic links across time.
- **Collision Checking**: Rejects trajectories where minimum clearance drops below `safe_distance` (default: 40 mm).
- **Staggered Motion**: Implements velocity scaling and temporal delays between robots to prevent mid-air link collisions.

### Stage 2: 6D Pose Inverse Kinematics
- **Full Pose Control**: Simultaneously solves for position $(X, Y, Z)$ and end-effector 3D orientation matrix $R \in \text{SO}(3)$ using rotation error vectors.
- **Handover Alignment**: Computes opposing gripper orientations where robot grippers align face-to-face along complementary axes during transfer.

### Stage 2.5: Object-Pose IK & Synchronized Handover (Latest)
- **Object-Target Specification**: User provides object target position $(X, Y, Z)$ and optional table yaw angle $\psi$ (default: $0.0^\circ$).
- **Adaptive Approach Tilt ($\tau$)**: Evaluates candidate tilt angles from $[0.0^\circ, \pm 10^\circ, \pm 20^\circ, \dots, \pm 90^\circ]$ to find reachable solutions while preferring top-down picking.
- **Configurable Gripper Finger Axis**: Supports grippers mounted along either local $X$ or local $Y$ (`finger_axis: "x" | "y"`).
- **5-DOF Task-Priority Nullspace IK**: Cartesian position is solved as the hard primary task; finger-axis yaw alignment is projected into the position nullspace as a secondary task.
- **6-DOF Balanced Units DLS IK**: Dimensionally scaled Jacobian with adaptive Levenberg-Marquardt singularity damping.
- **Independent Pick/Place Solver Modes with Fallback**: If an object orientation cannot be satisfied by any robot, automatically falls back to Stage 1.5 position IK independently for Pick or Place.
- **Synchronized 9-Step Trajectory State Machine**: Synchronized approach, parallel pre-handover stationing, simultaneous advance, gripper transfer, simultaneous retraction, and return to home.

---

## 4. Configuration System

All kinematics models, hardware parameters, tolerances, and timings are decoupled into YAML configuration files in [`src/robot_control/config/`](file:///home/iit/rt_test/src/robot_control/config/):

### Robot Kinematics Configs: `robot1.yaml`, `robot2.yaml`, `robot3.yaml`

Each robot YAML file defines:
- **DH Parameters**: `alpha_deg`, `a`, `d`, `theta_offset_deg`, `theta_joint_index`, `theta_joint_multiplier`.
- **TCP Offset**: Vector `[dx, dy, dz]` from the wrist link to the tool center point.
- **Joint Limits**: `limits_deg_min` and `limits_deg_max` (e.g. `[-90.0, 90.0]`).
- **Home Position**: `home_deg`.
- **Gripper Configuration**:
  ```yaml
  gripper:
    open_angle: -55.0       # Open gripper servo angle (degrees)
    close_angle: 55.0       # Closed gripper servo angle (degrees)
    finger_axis: "x"        # Gripper finger pads axis: "x" or "y"
    approach_axis: "z"      # Approach vector: "z"
  ```

### Workcell Coordination: `handover.yaml`

Defines inter-robot coordination, workcell placement, and Stage 2.5 solver hyperparameters:

```yaml
robots:
  robot1:
    world_base_xyz: [-250.0, 0.0, 0.0]
    world_base_rpy_deg: [0.0, 0.0, 0.0]
  robot2:
    world_base_xyz: [250.0, 0.0, 0.0]
    world_base_rpy_deg: [0.0, 0.0, 180.0]
  robot3:
    world_base_xyz: [0.0, 250.0, 0.0]
    world_base_rpy_deg: [0.0, 0.0, 90.0]

object_pick_ik:
  finger_axis: "x"          # Global default finger axis ("x" or "y")
  approach_axis: "z"        # Global approach axis ("z")
  tolerances:
    position_mm: 1.0        # Cartesian position tolerance (mm)
    orientation_deg: 15.0   # 6-DOF orientation tolerance (deg)
    finger_alignment_deg: 5.0 # 5-DOF finger alignment tolerance (deg)
    joint_margin_deg: 0.1   # Minimum required margin from joint limits (deg)
    min_manipulability: 1e-4# Minimum acceptable singular value
  solver:
    characteristic_length_mm: 100.0
    max_iterations: 100
    damping_base: 0.05
    damping_max: 0.5
  candidate_tilts_deg: [0.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0, 90.0, -90.0]
  cost_weights:
    tilt: 0.40
    current_q: 0.25
    joint_margin: 0.15
    manipulability: 0.10
    finger_alignment: 0.10
```

---

## 5. Installation & Build Instructions

### Prerequisites

- **OS**: Ubuntu 24.04 LTS (or compatible Linux)
- **ROS 2**: Jazzy Jalisco
- **Python**: $\ge 3.10$ with `numpy`, `scipy`, `matplotlib`, `pyyaml`, `pytest`

### Build Workspace

From the workspace root directory (`/home/iit/rt_test`):

```bash
# Source ROS 2 environment
source /opt/ros/jazzy/setup.bash

# Build the robot_control package
colcon build --symlink-install --packages-select robot_control

# Source the local workspace
source install/setup.bash
```

---

## 6. How to Run & GUI Operation Guide

### Launching GUIs

Each stage has an independent graphical user interface:

```bash
# Stage 1: Basic Pick-and-Place GUI
ros2 run robot_control stage1_gui

# Stage 1.5: Whole-Body Clearance Monitoring GUI
ros2 run robot_control stage1_5_gui

# Stage 2: 6D Pose IK GUI
ros2 run robot_control stage2_gui

# Stage 2.5: Object-Pose IK & Synchronized Handover GUI (Recommended)
ros2 run robot_control stage2_5_gui
```

### Stage 2.5 User Walkthrough

1. **Enter Pick Target**:
   - `XYZ (mm)`: Enter world coordinates comma-separated (e.g. `-97.10, 0.0, 13.82`).
   - `Yaw (°)`: Enter object orientation on the table (e.g. `0.0`).
2. **Enter Place Target**:
   - `XYZ (mm)`: Enter target placement coordinates (e.g. `0.0, 150.0, 150.0`).
   - `Yaw (°)`: Enter desired placement yaw (e.g. `0.0`).
3. **Click `SIMULATE (3D PREVIEW ONLY)`**:
   - The visualizer displays candidate robot postures, workspace clouds, and collision clearance lines.
   - If multiple robots can reach a target, an interactive selection popup asks you to select your preferred robot.
   - Solvers automatically display badges: `[Object-Pose IK]` or `[Normal IK Fallback]`.
   - The verified trajectory is saved to the plan cache.
4. **Click `EXECUTE (REAL ROBOT)`**:
   - The system detects the cached simulation and reuses it immediately, bypassing re-computation.
   - A confirmation dialog ensures physical workspace clearance before sending trajectory commands to the robots.

---

## 7. ROS 2 Interface & Topics

### Published Topics

| Topic Name | Message Type | Description |
| :--- | :--- | :--- |
| `/robot1/joint_commands` | `std_msgs/msg/Float64MultiArray` | 5-DOF joint angles for Robot 1 (degrees) |
| `/robot2/joint_commands` | `std_msgs/msg/Float64MultiArray` | 5-DOF joint angles for Robot 2 (degrees) |
| `/robot3/joint_commands` | `std_msgs/msg/Float64MultiArray` | 6-DOF joint angles for Robot 3 (degrees) |
| `/robotX/gripper_command` | `std_msgs/msg/Float64MultiArray` | Gripper servo angle in degrees (`[angle]`) |
| `/robotX/robot_speed` | `std_msgs/msg/Int32` | Motion speed multiplier (1 to 10) |
| `/robotX/robot_command` | `std_msgs/msg/String` | Hardware macro commands (`HOME`, `STOP`) |

---

## 8. Automated Test Suite

A comprehensive test suite verifies forward kinematics, all IK solvers, collision avoidance, and trajectory synchronization:

```bash
# Run all unit tests
PYTHONPATH=src/robot_control python3 -m pytest \
    src/robot_control/test/test_object_pick_ik.py \
    src/robot_control/test/test_stage2_5.py \
    src/robot_control/test/test_stage1_ik.py -v
```

All 24 test suites execute deterministically and assert convergence, joint limit compliance, collision clearance, and fallback mechanisms.
