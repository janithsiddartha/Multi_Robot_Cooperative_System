# Multi-Robot Cooperative Pick-and-Place & Synchronized Handover System

An advanced multi-robot coordination and manipulation framework developed in ROS 2 (Jazzy) for heterogeneous robotic manipulators. The platform orchestrates 3 physical robotic arms cooperating in shared workspaces to perform pick-and-place, adaptive object-pose grasping, synchronized multi-robot handovers, and real-time whole-body collision avoidance.

---

## Table of Contents

1. [System Overview & Manipulators](#1-system-overview--manipulators)
2. [Software Architecture & Control Planes](#2-software-architecture--control-planes)
3. [Evolution of Stages (1, 1.5, 2, 2.5, 3, 3-Test)](#3-evolution-of-stages)
4. [Stage 3 Test: Final Operational Version](#4-stage-3-test-final-operational-version)
5. [Configuration System](#5-configuration-system)
6. [Installation & Build Instructions](#6-installation--build-instructions)
7. [How to Run & GUI Operation Guide](#7-how-to-run--gui-operation-guide)
8. [ROS 2 Interface & Topics](#8-ros-2-interface--topics)
9. [Automated Test Suite](#9-automated-test-suite)

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
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               GUI & VISUALIZATION PLANE                                │
│    Stage 1 GUI  │  Stage 1.5 GUI  │  Stage 2 GUI  │  Stage 2.5 GUI  │ Stage 3 Test GUI │
│   • 3D Matplotlib Visualizer (Robot Links, TCP Frames, Point Clouds)                   │
│   • Multi-Robot Handover Inspector • Live Clearance Gauge • Interactive Robot Chooser  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Plan / Simulate / Execute
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                                MOTION PLANNING ENGINE                                  │
│   • Target Parsing & Normalization (XYZ mm, Yaw [0, 360)°)                             │
│   • Stage 2.5 Object-Pose IK & Adaptive Approach Tilt Solver                           │
│   • Stage 3 Kinematic Handover Solver (Voxel Workspace Overlap & SO(3) Mating)         │
│   • Stage 3 Test Strategy Dispatch: TOP_OF_WORKSPACE vs CENTER_OF_WORKSPACE            │
│   • Approach Strategy Coordination: DUAL_SYNCHRONIZED vs ASYMMETRIC                    │
│   • Stage 2.5 Trajectory Engine: Cartesian TCP Z Corridors & Exclude-TCP Clearance     │
│   • High-Performance Two-Tier Disk Caching (< 1 ms Recall with Robot Chooser)          │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Trajectory Stream
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                                ROS 2 HARDWARE INTERFACE                                │
│   • Topic Multiplexers (command_mux, gripper_mux)                                      │
│   • Joint Command Publishers (`/robotX/joint_commands`)                                │
│   • Gripper Command Publishers (`/robotX/gripper_command`)                             │
│   • UART Serial Bridge Nodes (`uart_node`)                                             │
│   • ESP32 Firmware Execution via PCA9685                                               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Dual-Execution Architecture

1. **SIMULATE (3D Preview Only)**:
   - Visualizes the full 3D robotic motion path in the interactive Matplotlib canvas.
   - Evaluates link-to-link distances throughout the trajectory.
   - **Hardware output is strictly disabled** (no serial commands sent to physical robots).
   - Generates and verifies collision clearance; caches the verified plan.
2. **EXECUTE (Real Robot)**:
   - Evaluates hardware readiness and verified joint states.
   - Prompts a safety confirmation dialog before moving physical arms.
   - Streams synchronized trajectory waypoints to `/joint_commands` and `/gripper_command` topics at configured step intervals.

---

## 3. Evolution of Stages

The framework is structured into progressive stages, each addressing higher levels of manipulation autonomy and geometric fidelity:

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

### Stage 2.5: Object-Pose IK & Synchronized Handover
- **Object-Target Specification**: User provides object target position $(X, Y, Z)$ and optional table yaw angle $\psi$ (default: $0.0^\circ$).
- **Adaptive Approach Tilt ($\tau$)**: Evaluates candidate tilt angles from $[0.0^\circ, \pm 10^\circ, \dots, \pm 90^\circ]$ preferring top-down picking.
- **Configurable Gripper Finger Axis**: Supports grippers mounted along either local $X$ or local $Y$ (`finger_axis: "x" | "y"`).
- **5-DOF Task-Priority Nullspace IK**: Cartesian position is solved as the primary task; finger-axis yaw alignment is projected into the nullspace.
- **Synchronized 9-Step Trajectory State Machine**: Synchronized approach, parallel pre-handover stationing, simultaneous advance, gripper transfer, simultaneous retraction, and return to home.

### Stage 3: Automated Kinematic Handover Discovery
- **Workspace Voxel Overlap**: Samples reach volumes into 25 mm voxel grids to discover candidate handover points in 3D space.
- **Coupled $\text{SO}(3)$ Orientation Optimization**: Solves for opposing gripper orientations with anti-parallel approach vectors ($\mathbf{z}_A \cdot \mathbf{z}_B = -1$) and orthogonal finger pads ($\mathbf{x}_A \cdot \mathbf{x}_B = 0$).
- **Multi-Pair Support**: Evaluates and caches all available robot pairs (`robot1_robot2`, `robot2_robot3`, `robot3_robot1`).

---

## 4. Stage 3 Test: Final Operational Version

**Stage 3 Test (`stage3_test_gui`)** represents the final, production-ready operational standard of the system, synthesizing Stage 2.5's robust trajectory execution with Stage 3's automated kinematic handover intelligence.

### Key Capabilities & Enhancements:

1. **Candidate Selection Strategy Selector**:
   - **`TOP_OF_WORKSPACE` (High Z)**: Prioritizes handover candidates located in the upper volume of the workspace intersection ($Z \approx 362\text{ mm}$), maximizing ground clearance and avoiding arm collisions.
   - **`CENTER_OF_WORKSPACE` (Centroid)**: Prioritizes candidates near the geometric centroid of the shared reach volume with balanced distance from each base.

2. **Approach Strategy Telemetry**:
   - **`DUAL_SYNCHRONIZED`**: Both robots advance simultaneously along their continuous Cartesian TCP $Z$ corridors into the handover pose.
   - **`ASYMMETRIC`**: Handles asymmetric kinematics (such as 5-DOF manipulators) by stationing one robot safely at handover/pre-handover first while the other follows along the corridor.

3. **Stage 2.5-Derived Trajectory Engine**:
   - Uses `plan_stage1_5_synchronized_paths` for smooth, continuous joint interpolation.
   - **Proper TCP Tooltip Exclusion (`exclude_tcp=True`)**: Handover maneuvers exclude only the fingertip contact geometry, allowing opposing grippers to meet at 0–10 mm separation without triggering false-positive whole-body collision blocks.
   - Non-blocking execution safety monitoring: clearance gauge updates in real time (`safe` $\ge 50\text{ mm}$, `warn` $< 50\text{ mm}$) without halting simulation.

4. **Multi-Robot Handover Inspector**:
   - Live modal tool to inspect any robot pair (`robot1_robot2`, `robot2_robot3`, `robot3_robot1`).
   - Displays position error, angle alignment dev, whole-body clearances, and approach mode.
   - Buttons for immediate 3D visualization and simulation: `▶ Move to Pre-H (Sim)` and `▶ Move to HO (Sim)`.

5. **Two-Tier Instant Disk Caching with Active Chooser**:
   - **Tier 1 (Target IK Cache)**: Stores solved target kinematics in `trajectory_cache.json`. Subsequent planning requests load in **< 1 ms** (`⚡ [DISK CACHE HIT]`).
   - **Tier 2 (Test Handover Cache)**: Stores solved test handover poses in `auto_handover_poses_test.json`, fingerprinted with active robot configurations.
   - **Interactive Robot Selection**: Unlike rigid session caching, the robot selection modal (`_chooser`) is queried on every **Simulate** and **Execute** click when multiple robots can reach a target, giving full operator control to switch robots on demand.

---

## 5. Configuration System

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

Defines inter-robot coordination, workcell placement, and Stage 2.5/Stage 3 solver hyperparameters:

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

### Auto-Handover Poses Cache: `auto_handover_poses_test.json`

Stores pre-solved, collision-checked handover configurations for all robot pairs (`robot1_robot2`, `robot2_robot3`, `robot3_robot1`), including world coordinates, joint angles, standoff distances, and approach modes.

---

## 6. Installation & Build Instructions

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

## 7. How to Run & GUI Operation Guide

### Launching GUIs

```bash
# =========================================================================
# STAGE 3 TEST: Final Working Operational Station (Recommended)
# =========================================================================
ros2 run robot_control stage3_test_gui

# Previous Reference Implementations
ros2 run robot_control stage2_5_gui     # Stage 2.5: Object-Pose IK & Synchronized Handover
ros2 run robot_control stage3_gui       # Stage 3: Base Kinematic Discovery
ros2 run robot_control stage2_gui       # Stage 2: 6D Pose IK
ros2 run robot_control stage1_5_gui     # Stage 1.5: Whole-Body Clearance
ros2 run robot_control stage1_gui       # Stage 1: Basic Pick-and-Place
```

### Stage 3 Test Operational Walkthrough

1. **Select Handover Candidate Strategy**:
   - In the Handover Status Card, choose:
     - `Top (High Z)` (Recommended): Handover in upper shared reach space ($Z \approx 362\text{ mm}$), maximizing clearance from the table and arm bases.
     - `Center (Centroid)`: Handover centered between the robot bases.

2. **Inspect Handover Pairs**:
   - Click `🔍 Inspect Handover Pairs...` to open the Multi-Robot Handover Inspector.
   - Select any pair (`robot1_robot2`, `robot2_robot3`, `robot3_robot1`).
   - Click `▶ Move to Pre-H (Sim)` to preview the pre-handover corridor stance.
   - Click `▶ Move to HO (Sim)` to preview the final handover grip.
   - Motions interpolate smoothly with whole-body clearance telemetry (`exclude_tcp=True`).

3. **Define Pick & Place Tasks**:
   - `Pick Target`: Enter coordinates `XYZ (mm)` (e.g. `0.0, 150.0, 100.0`) and `Yaw (°)` (e.g. `0.0`).
   - `Place Target`: Enter coordinates `XYZ (mm)` (e.g. `150.0, 0.0, 100.0`) and `Yaw (°)` (e.g. `0.0`).

4. **Click `SIMULATE (3D PREVIEW ONLY)`**:
   - Target IK loads instantly from disk cache (< 1 ms).
   - If multiple robots can reach the pick/place locations, the **Robot Selection Dialog** prompts you to choose which robot performs Pick or Place.
   - You can click **Simulate** again at any time to test different robot assignments.
   - 3D visualizer animates the complete synchronized 9-phase trajectory.
   - Clearance gauge tracks whole-body physical distance in real time.

5. **Click `EXECUTE (REAL ROBOT)`**:
   - Verifies live hardware feedback.
   - Allows confirmation of chosen robot assignments.
   - Displays final Solution Dialog with verified clearance.
   - Streams trajectory waypoints to hardware controllers on physical robots.

---

## 8. ROS 2 Interface & Topics

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

## 9. Automated Test Suite

A comprehensive test suite verifies forward kinematics, all IK solvers, collision avoidance, and trajectory synchronization:

```bash
# Source ROS environment
source install/setup.bash

# Run Stage 3 Test Suite
colcon test --packages-select robot_control --pytest-args "-k test_stage3_test_env"

# Run Stage 3 GUI & Execution Suite
colcon test --packages-select robot_control --pytest-args "-k test_stage3_gui"

# Run Core Kinematics & Stage 2.5 Tests
colcon test --packages-select robot_control --pytest-args "-k test_stage2_5"

# View summary of all test results
colcon test-result --all
```

All core test suites execute deterministically to assert convergence, joint limit compliance, collision clearance, and fallback mechanisms.
