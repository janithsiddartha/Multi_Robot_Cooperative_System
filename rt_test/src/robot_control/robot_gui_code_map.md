# robot_gui_test.py — Code Map & Quick Reference

A guide to finding your way around without needing AI help for small/medium edits.
Line numbers refer to the version uploaded on 2026-09-07.

---

## 1. Big picture

```
Lines 1-27      Imports
Lines 38-242    Reusable custom Tkinter widgets (no ROS logic here)
Lines 244-860   RobotGUI.__init__  -> ROS setup + state + window layout
Lines 860-1244  UI-building methods (build panels/tabs, wire buttons)
Lines 1244-1390 3D visualizer
Lines 1390-2062 Control-mode logic + manual joint control
Lines 2062-2394 IK + trajectory callbacks
Lines 2394-2547 Pick & place
Lines 2547-2700 Gripper + speed
Lines 2700-3036 Home / Stop / Resume / shutdown / main()
```

The file has exactly ONE class that does ROS + GUI work: `RobotGUI(Node)` (line 244).
Everything above it (`RoundedFrame`, `RoundedButton`, `RoundedSlider`) is just visual
styling — safe to tweak without touching any robot logic.

---

## 2. "I want to change the LOOK of something"

| What | Where | Notes |
|---|---|---|
| Colors (background, accent blue, text, etc.) | **Lines 461-474** (`self.bg_color`, `self.primary_blue`, ...) | Change the hex value once here — used everywhere. Don't hunt for individual widgets. |
| Fonts | **Lines 476-480** (`self.font_main`, `self.font_title`, ...) | Same idea — centralized. |
| Rounded corners / shadow style of cards | `RoundedFrame` class, **lines 38-65** | `radius` param controls corner roundness. |
| Button appearance/hover color | `RoundedButton` class, **lines 68-142** | `bg`, `hover_bg`, `radius` are passed in when a button is created — you don't need to edit the class itself for a single button. |
| Slider look | `RoundedSlider` class, **lines 145-243** | |
| Window title / initial size | **Lines 485-524ish**, look for `self.root = tk.Tk()` (line 485) | |

**Rule of thumb:** if it's a color/font, change it ONCE in the constants block
(lines 461-480), not in the individual widget-creation calls.

---

## 3. "I want to change a UI PANEL / TAB (add a label, rearrange controls)"

All panel/tab builders live in **lines 860-1160**. Each is a small self-contained
method — find the one whose name matches what you're editing:

| Method | Line | Builds |
|---|---|---|
| `create_header` | 882 | Top header bar |
| `create_overview_tab` | 954 | "Overview" tab |
| `create_fk_tab` | 959 | Forward kinematics tab |
| `create_ik_tab` | 967 | Inverse kinematics tab |
| `create_pick_place_tab` | 970 | Pick & place tab |
| `create_robot_info_tab` | 973 | Robot info / DH table tab |
| `create_manual_control` | 1003 | Manual joint sliders panel |
| `create_current_position_panel` | 1025 | Live XYZ / joint readout |
| `create_direct_ik_panel` | 1044 | XYZ target entry for IK |
| `create_gripper_panel` | 1070 | Open/close gripper buttons |
| `create_speed_panel` | 1081 | Speed slider |
| `create_pick_place_panel` | 1089 | Pick/place set/clear/swap buttons |
| `create_system_panel` | 1129 | Home / Stop / Resume buttons |
| `create_status_panel` | 1138 | Status text panel |
| `create_status_bar` | 1154 | Bottom status bar |

If you want to add a new label/button to an existing panel, edit the matching
method above — don't need to touch anything else.

---

## 4. "I want to change ROS topics / messages"

**All publishers:** lines 285-450ish (search `self.create_publisher`)
**All subscribers:** lines 470-620ish (search `self.create_subscription`)

Quick list (approx. line of the `create_publisher`/`create_subscription` call):

| Name | Line | Topic |
|---|---|---|
| `manual_joint_pub` | 285 | `/manual_joint_command` |
| `ik_gui_target_pub` | 302 | `/ik_gui_target` |
| `manual_gripper_pub` | ~320 | gripper command |
| `ik_joint_pub` | ~337 | |
| `speed_pub` | ~352 | |
| `command_pub` | ~370 | system command (HOME/STOP) |
| `control_mode_request_pub` | ~392 | |
| `pick_pub` / `place_pub` | ~407 / ~422 | |
| `trajectory_command_pub` | ~440 | |
| `joint_state_subscription` | ~472 | incoming joint states |
| `ee_subscription` | ~489 | end-effector pose |
| `ik_solution_subscription` | ~506 | |
| `ik_status_subscription` | ~523 | |
| `trajectory_status_subscription` | ~543 | |
| `control_mode_subscription` | ~561 | |

If you need to **change a topic name or message type**, it's a one-line edit at
the relevant `create_publisher(...)` / `create_subscription(...)` call — you do
NOT need to touch the callback logic elsewhere unless the message *type* changes.

**Callbacks that fire when a subscription receives data** (search by name):
- `control_mode_callback` — line 1396
- `joint_command_callback` — line 1967
- `end_effector_callback` — line 2066
- `ik_solution_callback` — line 2215
- `ik_status_callback` — line 2284
- `trajectory_status_callback` — line 2339

---

## 5. "I want to change robot BEHAVIOR / LOGIC"

| Behavior | Method | Line |
|---|---|---|
| What happens when a joint slider moves | `slider_changed` | 1737 |
| What happens when you type a joint angle | `angle_entry_changed` | 1793 |
| Publishing manual joint commands | `publish_manual_joints` | 1920 |
| Solve IK and move | `solve_ik_and_move` | 2123 |
| Start pick & place sequence | `start_pick_and_place` | 2398 |
| Open gripper | `gripper_open` | 2551 |
| Close gripper | `gripper_close` | 2590 |
| Manual gripper send | `send_manual_gripper` | 2629 |
| Speed slider changed | `speed_changed` | 2662 |
| HOME button | `home` | 2700 |
| STOP button | `stop` | 2791 |
| RESUME MANUAL button | `resume_manual` | 2866 |
| Enable/disable controls based on mode | `update_control_permissions` | 1514 |

---

## 6. "I want to change joint limits / names / gripper angles"

**Lines 332-420ish** — all in `__init__`:
- `self.joint_names` — line ~332
- `self.joint_min` / `self.joint_max` — lines ~347 / ~362
- `self.joint_angles` (initial values) — line ~383
- `self.gripper_open_angle` / `self.gripper_close_angle` — lines ~417-419
- `self.speed` (initial) — line ~426

These are also partly sourced from `robot_kinematics.py` (imported at the top,
line 23) — if the limits should be truly global across the project, that file
is the "source of truth", not this GUI file.

---

## 7. "I want to change the 3D visualizer"

- `setup_3d_visualizer` — line 1247 (creates the matplotlib figure/canvas)
- `get_workspace_points` — line 1256 (workspace boundary points)
- `update_visualization` — line 1282 (redraws frames/trace/grid on every update — this is the biggest single method, ~100 lines)
- Toggle checkboxes (trace, grid, joint frames, etc.) are `tk.BooleanVar`s set around **lines 524-533**

---

## 8. General tips for editing safely on your own

1. **Search by method name first.** Every meaningful chunk of logic is inside
   a `def some_name(self, ...):` — use Ctrl+F for the name in the table above.
2. **Constants over magic values.** Colors/fonts/joint limits are centralized
   near the top of `__init__` — always edit there, not inline elsewhere.
3. **UI vs. logic split:** methods named `create_*` only build widgets — they
   don't contain robot logic. Methods named as verbs (`home`, `stop`,
   `solve_ik_and_move`, `gripper_open`) contain the actual behavior.
4. **Don't touch `RoundedFrame`/`RoundedButton`/`RoundedSlider` (lines 38-243)**
   unless you're changing the widget style everywhere — they're generic and
   reused throughout.
5. When in doubt about a variable's origin, search for `self.variable_name =`
   — nearly everything is set once in `__init__` (lines 244-860).

