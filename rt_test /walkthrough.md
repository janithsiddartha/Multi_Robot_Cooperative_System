# Multi-Robot Refactoring Complete

I have completed the refactoring of the `robot_control` package and GUI to support a multi-robot architecture.

## Summary of Changes

### 1. Kinematics Configuration Class
- Refactored `robot_kinematics.py` into a `KinematicsConfig` class.
- The class takes a ROS 2 node instance and dynamically declares and loads DH parameters, joint limits, approaches heights, and segment times from the node's parameter server instead of relying on hardcoded global variables.

### 2. Mathematics Decoupling
- **`fk_math.py`**: Now requires `config` to be passed into the `fk(q, config)` function.
- **`ik_solver.py`**: Now requires `config` to be passed into `ik(target, config, ...)` to retrieve the DH parameters and joint limits dynamically.
- **`trajectory_math.py`**: `pick_and_place` now accepts `config` to read `approach_height`, `segment_time`, and `trajectory_dt`.

### 3. Node Refactoring
- Refactored all backend nodes to instantiate `KinematicsConfig(self)`.
- Replaced all absolute topic names (e.g., `'/joint_commands'`) with relative topic names (e.g., `'joint_commands'`) to allow `PushRosNamespace` to correctly scope them per-robot.
- `uart_node.py` now dynamically loads the UART port, baudrate, and joint calibration arrays (scale, offset, sign) directly from the ROS 2 parameter server.

### 4. GUI Namespace Separation
- Replaced all absolute topic paths in the GUI with relative ones.
- Added `namespace` support to the `RobotGUI` node initialization in `robot_gui_test.py`.
- Created three wrapper scripts in `BTP/gui/`:
  - [robot_gui_test_1.py](file:///c:/Users/IIT/Desktop/rt_test/gui/robot_gui_test_1.py)
  - [robot_gui_test_2.py](file:///c:/Users/IIT/Desktop/rt_test/gui/robot_gui_test_2.py)
  - [robot_gui_test_3.py](file:///c:/Users/IIT/Desktop/rt_test/gui/robot_gui_test_3.py)
- These scripts initialize the GUI with the correct `namespace` (`robot1`, `robot2`, `robot3`) and custom window titles.

### 5. `setup.py` Update
- Updated `setup.py` to point to the new subdirectories (`robot_control.nodes...`).
- Removed the GUI console scripts, as they now live independently in `BTP/gui/`.
- Configured `setup.py` to install the YAML templates from the `config/` directory to the ROS 2 share directory so they can be loaded by `robot_bringup.launch.py`.

## Next Steps
To use the new multi-robot system, open a ROS 2 powershell terminal, source your environment, and rebuild the workspace:

```bash
cd c:\Users\IIT\Desktop\rt_test
colcon build --packages-select robot_control
```

Once built, you can launch the backend for a specific robot:

```bash
# Launch robot 1 backend
ros2 launch robot_control robot_bringup.launch.py robot_name:=robot1 config:=robot1.yaml
```

In a separate terminal, launch the specific GUI:

```bash
# Launch robot 1 GUI
python gui\robot_gui_test_1.py
```
