# Multi-Robot Refactoring Plan

This refactoring plan will convert the existing single-robot ROS 2 project into a clean multi-robot architecture while preserving all functionality for Robot 1.

## User Review Required

> [!IMPORTANT]
> The target architecture requested placing `gui/` at the root of the BTP folder (outside `src/`). Scripts located here cannot be run using `ros2 run robot_control <script>` and won't be managed by colcon. Is it acceptable to run these GUI scripts via `python3 gui/robot_gui_test_1.py`?
>
> Also, to keep topics clean across all nodes and GUIs, I will remove the leading slash from all topic strings (e.g., `'/joint_commands'` becomes `'joint_commands'`). When launched under a namespace or when the GUI node explicitly defines a namespace, ROS 2 will automatically resolve the topics correctly to `/robot1/joint_commands`, etc. Please confirm this matches your expectations.

## Open Questions

- If the GUI scripts are moved to `BTP/gui/`, should they be completely removed from `setup.py`'s `console_scripts` entry points? I will remove them to keep `setup.py` strictly for backend ROS 2 nodes, and the GUIs will just be standalone Python scripts.

## Proposed Changes

### Configuration and Launch Layer
Extract hard-coded parameters from Python code into YAML configurations and update the launch system.

#### [NEW] [robot1.yaml](file:///c:/Users/IIT/Desktop/rt_test/src/robot_control/config/robot1.yaml)
Contains all existing parameters from Robot 1 (DH table, joint limits, calibration constants, UART config). The file will be structured using the `/**` wildcard so parameters map cleanly to any node in the namespace.

#### [NEW] [robot2.yaml](file:///c:/Users/IIT/Desktop/rt_test/src/robot_control/config/robot2.yaml)
Placeholder configuration for Robot 2, structurally identical to `robot1.yaml`.

#### [NEW] [robot3.yaml](file:///c:/Users/IIT/Desktop/rt_test/src/robot_control/config/robot3.yaml)
Placeholder configuration for Robot 3, structurally identical to `robot1.yaml`.

#### [MODIFY] [robot_bringup.launch.py](file:///c:/Users/IIT/Desktop/rt_test/src/robot_control/launch/robot_bringup.launch.py)
Move from `src/robot_control/robot_control/launch/` (or current location) to `src/robot_control/launch/`.
Update to accept `robot_name` and `config` launch arguments. It will use the `PushRosNamespace` action to enforce the namespace and will pass the selected YAML config to all instantiated backend nodes.

---

### Backend Logic Layer
Refactor the Python nodes into sub-modules and update them to read parameters from ROS 2 instead of importing hard-coded global variables.

#### [MODIFY] Node scripts (e.g. `uart_node.py`, `fk_node.py`, etc.)
- Move all node scripts to `src/robot_control/robot_control/nodes/`.
- Update topic names to be relative (e.g., remove leading slashes).
- Update imports to fetch from the new `kinematics` and `trajectory` modules.
- Refactor nodes that depend on `robot_kinematics.py` to `declare_parameter()` and read the values natively via ROS 2 parameters.

#### [MODIFY] Logic scripts (e.g. `robot_kinematics.py`, `ik_solver.py`, `fk_math.py`, `trajectory_math.py`)
- Move kinematics scripts to `src/robot_control/robot_control/kinematics/`.
- Move trajectory scripts to `src/robot_control/robot_control/trajectory/`.
- Update parameter usage. Functions relying on hard-coded globals (like `L1`, `L2`, `alpha_deg`) will be updated to accept these as arguments, passed in from the ROS 2 nodes that load the YAML config.

---

### GUI Layer
Move the GUI files out of the package to the workspace root and create isolated namespace wrappers.

#### [MODIFY] `robot_gui*.py`
- Move all existing GUI files to `BTP/gui/`.
- Modify all absolute topic subscriptions/publishers to relative strings.
- Refactor `robot_gui_test.py` to accept a `ROBOT_NAMESPACE` variable dynamically.

#### [NEW] `robot_gui_test_1.py`, `robot_gui_test_2.py`, `robot_gui_test_3.py`
Create wrapper scripts in `BTP/gui/` that set `ROBOT_NAMESPACE` and instantiate the main GUI class from `robot_gui_test.py`.

---

### Package Metadata
Update ROS 2 package files to match the new structure.

#### [MODIFY] [setup.py](file:///c:/Users/IIT/Desktop/rt_test/src/robot_control/setup.py)
- Remove GUI scripts from `console_scripts` if they are moved outside the package.
- Update node paths in `console_scripts` to use `robot_control.nodes.<node_name>:main`.
- Add the `config/` directory to `data_files` to ensure YAMLs are installed.

## Verification Plan

### Automated Tests
- `colcon build --symlink-install` in the workspace to verify package structure.

### Manual Verification
1. Run `ros2 launch robot_control robot_bringup.launch.py robot_name:=robot1 config:=robot1.yaml`.
2. Inspect namespaces with `ros2 topic list`. Everything should be under `/robot1/`.
3. Run `python3 gui/robot_gui_test_1.py` and verify it interacts with `/robot1/` topics.
4. Launch Robot 2 with `robot_name:=robot2 config:=robot2.yaml` and verify namespace isolation.
