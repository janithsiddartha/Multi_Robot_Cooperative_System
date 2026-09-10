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
from std_msgs.msg import Int32
from std_msgs.msg import String

import tkinter as tk
from tkinter import ttk


# ============================================================
# ROBOT GUI
# ============================================================

class RobotGUI(Node):

    def __init__(self):

        super().__init__('robot_gui')

        # ====================================================
        # CONTROL MODE QoS
        #
        # Must match control_mode_manager_node publisher QoS.
        #
        # TRANSIENT_LOCAL allows the GUI to immediately receive
        # the latest mode when it starts or restarts.
        # ====================================================

        self.control_mode_qos = QoSProfile(

            reliability=ReliabilityPolicy.RELIABLE,

            durability=DurabilityPolicy.TRANSIENT_LOCAL,

            history=HistoryPolicy.KEEP_LAST,

            depth=1

        )

        # ====================================================
        # ROS PUBLISHERS
        # ====================================================


        # ----------------------------------------------------
        # MANUAL JOINT COMMAND
        #
        # GUI sliders publish here.
        #
        # command_mux_node decides whether it is allowed.
        # ----------------------------------------------------

        self.manual_joint_pub = self.create_publisher(

            Float64MultiArray,

            '/manual_joint_command',

            10

        )


        # ----------------------------------------------------
        # DIRECT IK TARGET
        #
        # GUI -> ik_node
        # ----------------------------------------------------

        self.ik_gui_target_pub = self.create_publisher(

            Float64MultiArray,

            '/ik_gui_target',

            10

        )


        # ----------------------------------------------------
        # MANUAL GRIPPER COMMAND
        #
        # GUI -> gripper_mux_node
        # ----------------------------------------------------

        self.manual_gripper_pub = self.create_publisher(

            Float64MultiArray,

            '/manual_gripper_command',

            10

        )


        # ----------------------------------------------------
        # DIRECT IK JOINT COMMAND
        #
        # GUI -> command_mux_node
        # ----------------------------------------------------

        self.ik_joint_pub = self.create_publisher(

            Float64MultiArray,

            '/ik_joint_command',

            10

        )


        # ----------------------------------------------------
        # ROBOT SPEED
        # ----------------------------------------------------

        self.speed_pub = self.create_publisher(

            Int32,

            '/robot_speed',

            10

        )


        # ----------------------------------------------------
        # ROBOT SYSTEM COMMAND
        #
        # HOME
        # STOP
        # ----------------------------------------------------

        self.command_pub = self.create_publisher(

            String,

            '/robot_command',

            10

        )


        # ----------------------------------------------------
        # CONTROL MODE REQUEST
        #
        # GUI requests mode transitions only.
        # control_mode_manager_node is the sole /control_mode
        # publisher.
        #
        # HALTED  -> STOP safety override
        # MANUAL  -> explicit RESUME MANUAL
        # ----------------------------------------------------

        self.control_mode_request_pub = self.create_publisher(

            String,

            '/control_mode_request',

            10

        )


        # ----------------------------------------------------
        # PICK POSITION
        # ----------------------------------------------------

        self.pick_pub = self.create_publisher(

            Float64MultiArray,

            '/pick_position',

            10

        )


        # ----------------------------------------------------
        # PLACE POSITION
        # ----------------------------------------------------

        self.place_pub = self.create_publisher(

            Float64MultiArray,

            '/place_position',

            10

        )


        # ----------------------------------------------------
        # TRAJECTORY COMMAND
        #
        # START
        # STOP
        # ----------------------------------------------------

        self.trajectory_command_pub = self.create_publisher(

            String,

            '/trajectory_command',

            10

        )


        # ====================================================
        # ROS SUBSCRIBERS
        # ====================================================


        # ----------------------------------------------------
        # FINAL JOINT COMMAND
        #
        # READ ONLY
        #
        # This is the GUI's single source of truth for
        # displayed joint positions.
        #
        # Every source:
        #
        # Manual
        # Direct IK
        # Trajectory
        #
        # eventually appears here after command_mux.
        # ----------------------------------------------------

        self.joint_state_subscription = self.create_subscription(

            Float64MultiArray,

            '/joint_commands',

            self.joint_command_callback,

            10

        )


        # ----------------------------------------------------
        # FK END EFFECTOR POSITION
        # ----------------------------------------------------

        self.ee_subscription = self.create_subscription(

            Float64MultiArray,

            '/end_effector_position',

            self.end_effector_callback,

            10

        )


        # ----------------------------------------------------
        # GUI IK SOLUTION
        # ----------------------------------------------------

        self.ik_solution_subscription = self.create_subscription(

            Float64MultiArray,

            '/ik_gui_solution',

            self.ik_solution_callback,

            10

        )


        # ----------------------------------------------------
        # GUI IK STATUS
        # ----------------------------------------------------

        self.ik_status_subscription = self.create_subscription(

            String,

            '/ik_gui_status',

            self.ik_status_callback,

            10

        )


        # ----------------------------------------------------
        # TRAJECTORY STATUS
        #
        # Informational only.
        #
        # It does NOT control GUI authority.
        # /control_mode is the authority.
        # ----------------------------------------------------

        self.trajectory_status_subscription = self.create_subscription(

            String,

            '/trajectory_status',

            self.trajectory_status_callback,

            10

        )


        # ----------------------------------------------------
        # CONTROL MODE
        #
        # Actual system authority state.
        # ----------------------------------------------------

        self.control_mode_subscription = self.create_subscription(

            String,

            '/control_mode',

            self.control_mode_callback,

            self.control_mode_qos

        )


        # ====================================================
        # ROBOT CONFIGURATION
        # ====================================================

        self.joint_names = [

            'BASE',

            'J1',

            'J2',

            'J3',

            'J4'

        ]


        from robot_control.robot_kinematics import JOINT_LIMITS_DEG, GRIPPER_OPEN_ANGLE, GRIPPER_CLOSE_ANGLE

        self.joint_min = JOINT_LIMITS_DEG[:, 0].tolist()
        self.joint_max = JOINT_LIMITS_DEG[:, 1].tolist()


        # ====================================================
        # CURRENT JOINT VALUES
        #
        # Updated from /joint_commands.
        # ====================================================

        self.joint_angles = [

            0.0,

            0.0,

            0.0,

            0.0,

            0.0

        ]


        # ====================================================
        # END EFFECTOR POSITION
        # ====================================================

        self.ee_position = [

            0.0,

            0.0,

            0.0

        ]


        # ====================================================
        # GRIPPER
        # ====================================================

        self.gripper_open_angle = GRIPPER_OPEN_ANGLE
        self.gripper_close_angle = GRIPPER_CLOSE_ANGLE


        # ====================================================
        # SPEED
        # ====================================================

        self.speed = 3


        # ====================================================
        # GUI STATE
        # ====================================================


        # Prevent GUI callback loops when sliders are updated
        # from /joint_commands.

        self.updating_gui = False

        self._pending_gui_unlock = False


        # GUI starts locked until it receives an actual
        # /control_mode message.

        self.control_mode = None

        self.control_mode_received = False


        # Only used for direct IK status display.
        #
        # NOT used for slider synchronization.

        self.direct_ik_pending = False


        # ====================================================
        # WINDOW
        # ====================================================

        self.root = tk.Tk()

        self.root.title(

            '5-DOF Robot Control Station'

        )


        self.root.geometry(

            '1500x900'

        )


        self.root.minsize(

            1250,

            780

        )


        self.root.configure(

            bg='#E2E8F0'

        )


        # ====================================================
        # STYLE
        # ====================================================

        self.setup_style()


        # ====================================================
        # HEADER
        # ====================================================

        self.create_header()


        # ====================================================
        # MAIN AREA
        # ====================================================

        main_frame = tk.Frame(

            self.root,

            bg='#E2E8F0'

        )

        main_frame.pack(

            fill='both',

            expand=True,

            padx=18,

            pady=15

        )


        main_frame.columnconfigure(

            0,

            weight=2

        )

        main_frame.columnconfigure(

            1,

            weight=1

        )

        main_frame.columnconfigure(

            2,

            weight=1

        )

        main_frame.rowconfigure(

            0,
            

            weight=1

        )


        # ====================================================
        # LEFT COLUMN
        # ====================================================

        left_frame = tk.Frame(

            main_frame,

            bg='#E2E8F0'

        )

        left_frame.grid(

            row=0,

            column=0,

            sticky='nsew',

            padx=(0, 12)

        )


        self.create_manual_control(

            left_frame

        )

        self.create_direct_ik_panel(

            left_frame

        )

        self.create_status_panel(

            left_frame

        )


        # ====================================================
        # MIDDLE COLUMN
        # ====================================================

        middle_frame = tk.Frame(

            main_frame,

            bg='#E2E8F0'

        )

        middle_frame.grid(

            row=0,

            column=1,

            sticky='nsew',

            padx=6

        )


        self.create_current_position_panel(

            middle_frame

        )

        self.create_gripper_panel(

            middle_frame

        )

        self.create_speed_panel(

            middle_frame

        )


        # ====================================================
        # RIGHT COLUMN
        # ====================================================

        right_frame = tk.Frame(

            main_frame,

            bg='#E2E8F0'

        )

        right_frame.grid(

            row=0,

            column=2,

            sticky='nsew',

            padx=(12, 0)

        )


        self.create_pick_place_panel(

            right_frame

        )

        self.create_system_panel(

            right_frame

        )


        # ====================================================
        # STATUS BAR
        # ====================================================

        self.create_status_bar()


        # ====================================================
        # WINDOW CLOSE
        # ====================================================

        self.root.protocol(

            'WM_DELETE_WINDOW',

            self.shutdown

        )


        # ====================================================
        # ROS SPIN
        #
        # Tkinter owns the main thread, so we poll rclpy from
        # the Tk event loop. A create_timer() callback would
        # never run because spin_once() is what drives timers.
        # ====================================================

        self.root.after(

            10,

            self.process_ros

        )


        # Start with controls disabled.

        self.update_control_permissions()


        self.get_logger().info(

            '========================================'

        )

        self.get_logger().info(

            '5-DOF ROBOT GUI STARTED'

        )

        self.get_logger().info(

            '========================================'

        )

        self.get_logger().info(

            'Waiting for /control_mode...'

        )


    # ========================================================
    # STYLE
    # ========================================================

    def setup_style(self):

        style = ttk.Style()

        try:

            style.theme_use(

                'clam'

            )

        except Exception:

            pass


    # ========================================================
    # HEADER
    # ========================================================

    def create_header(self):

        header = tk.Frame(

            self.root,

            bg='#0F172A',

            height=90

        )

        header.pack(

            fill='x'

        )

        header.pack_propagate(

            False

        )


        title_frame = tk.Frame(

            header,

            bg='#0F172A'

        )

        title_frame.pack(

            side='left',

            padx=30,

            pady=15

        )


        tk.Label(

            title_frame,

            text='5-DOF ROBOT CONTROL STATION',

            font=('Arial', 22, 'bold'),

            fg='white',

            bg='#0F172A'

        ).pack(

            anchor='w'

        )


        tk.Label(

            title_frame,

            text='ROS 2 Jazzy  •  ESP32  •  PCA9685  •  Robot 1',

            font=('Arial', 10),

            fg='#94A3B8',

            bg='#0F172A'

        ).pack(

            anchor='w',

            pady=(4, 0)

        )


        self.mode_label = tk.Label(

            header,

            text='WAITING',

            font=('Arial', 11, 'bold'),

            fg='white',

            bg='#64748B',

            padx=20,

            pady=9

        )

        self.mode_label.pack(

            side='right',

            padx=30

        )


    # ========================================================
    # SECTION FRAME
    # ========================================================

    def create_section_frame(

        self,

        parent,

        title

    ):

        outer = tk.Frame(

            parent,

            bg='#CBD5E1'

        )

        outer.pack(

            fill='x',

            pady=(0, 12)

        )


        tk.Label(

            outer,

            text=title,

            font=('Arial', 11, 'bold'),

            bg='#1E293B',

            fg='white',

            anchor='w',

            padx=12,

            pady=7

        ).pack(

            fill='x'

        )


        content = tk.Frame(

            outer,

            bg='#F8FAFC',

            padx=12,

            pady=12

        )

        content.pack(

            fill='both',

            expand=True

        )


        return content


    # ========================================================
    # MANUAL JOINT CONTROL
    # ========================================================

    def create_manual_control(

        self,

        parent

    ):

        frame = self.create_section_frame(

            parent,

            'MANUAL JOINT CONTROL'

        )


        self.sliders = []

        self.angle_entries = []


        for i in range(5):

            row = tk.Frame(

                frame,

                bg='#F8FAFC'

            )

            row.pack(

                fill='x',

                pady=8

            )


            tk.Label(

                row,

                text=self.joint_names[i],

                width=8,

                anchor='w',

                font=('Arial', 10, 'bold'),

                bg='#F8FAFC',

                fg='#334155'

            ).pack(

                side='left'

            )


            slider = tk.Scale(

                row,

                from_=self.joint_min[i],

                to=self.joint_max[i],

                orient='horizontal',

                resolution=1,

                showvalue=False,

                length=330,

                bg='#F8FAFC',

                highlightthickness=0,

                command=lambda value, idx=i:
                    self.slider_changed(
                        idx,
                        value
                    )

            )


            slider.set(

                self.joint_angles[i]

            )


            slider.pack(

                side='left',

                fill='x',

                expand=True,

                padx=8

            )


            entry = tk.Entry(

                row,

                width=9,

                justify='center',

                font=('Arial', 10),

                relief='solid'

            )

            entry.insert(

                0,

                '0.0'

            )


            entry.bind(

                '<Return>',

                lambda event, idx=i:
                    self.angle_entry_changed(
                        idx
                    )

            )


            entry.bind(

                '<FocusOut>',

                lambda event, idx=i:
                    self.angle_entry_changed(
                        idx
                    )

            )


            entry.pack(

                side='left',

                padx=5

            )


            tk.Label(

                row,

                text='deg',

                bg='#F8FAFC',

                fg='#64748B'

            ).pack(

                side='left'

            )


            self.sliders.append(

                slider

            )

            self.angle_entries.append(

                entry

            )


    # ========================================================
    # CURRENT END EFFECTOR POSITION
    # ========================================================

    def create_current_position_panel(

        self,

        parent

    ):

        frame = self.create_section_frame(

            parent,

            'CURRENT END EFFECTOR POSITION'

        )


        self.x_label = self.create_value_row(

            frame,

            'X',

            '0.00'

        )

        self.y_label = self.create_value_row(

            frame,

            'Y',

            '0.00'

        )

        self.z_label = self.create_value_row(

            frame,

            'Z',

            '0.00'

        )


        self.fk_status = tk.Label(

            frame,

            text='WAITING',

            font=('Arial', 10, 'bold'),

            fg='#64748B',

            bg='#F8FAFC'

        )

        self.fk_status.pack(

            pady=(8, 0)

        )


    # ========================================================
    # VALUE ROW
    # ========================================================

    def create_value_row(

        self,

        parent,

        name,

        value

    ):

        row = tk.Frame(

            parent,

            bg='#F8FAFC'

        )

        row.pack(

            fill='x',

            pady=5

        )


        tk.Label(

            row,

            text=name,

            width=4,

            anchor='w',

            font=('Arial', 11, 'bold'),

            bg='#F8FAFC'

        ).pack(

            side='left'

        )


        label = tk.Label(

            row,

            text=value,

            width=12,

            font=('Arial', 11, 'bold'),

            fg='#0F172A',

            bg='#E2E8F0',

            relief='solid',

            pady=5

        )

        label.pack(

            side='right'

        )


        return label


    # ========================================================
    # DIRECT IK
    # ========================================================

    def create_direct_ik_panel(

        self,

        parent

    ):

        frame = self.create_section_frame(

            parent,

            'DIRECT CARTESIAN IK'

        )


        self.target_x_entry = self.create_xyz_entry(

            frame,

            'X'

        )

        self.target_y_entry = self.create_xyz_entry(

            frame,

            'Y'

        )

        self.target_z_entry = self.create_xyz_entry(

            frame,

            'Z'

        )


        self.solve_ik_button = tk.Button(

            frame,

            text='SOLVE IK & MOVE',

            font=('Arial', 11, 'bold'),

            bg='#7C3AED',

            fg='white',

            command=self.solve_ik_and_move

        )

        self.solve_ik_button.pack(

            fill='x',

            pady=(12, 6)

        )


        self.ik_status_label = tk.Label(

            frame,

            text='IK Status: IDLE',

            bg='#F8FAFC',

            fg='#475569',

            anchor='w'

        )

        self.ik_status_label.pack(

            fill='x'

        )


    # ========================================================
    # XYZ ENTRY
    # ========================================================

    def create_xyz_entry(

        self,

        parent,

        axis

    ):

        row = tk.Frame(

            parent,

            bg='#F8FAFC'

        )

        row.pack(

            fill='x',

            pady=4

        )


        tk.Label(

            row,

            text=f'{axis} (mm)',

            width=10,

            anchor='w',

            bg='#F8FAFC'

        ).pack(

            side='left'

        )


        entry = tk.Entry(

            row,

            width=14,

            justify='center'

        )

        entry.insert(

            0,

            '0.0'

        )

        entry.pack(

            side='right'

        )


        return entry


    # ========================================================
    # GRIPPER
    # ========================================================

    def create_gripper_panel(

        self,

        parent

    ):

        frame = self.create_section_frame(

            parent,

            'GRIPPER CONTROL'

        )


        self.gripper_open_button = tk.Button(

            frame,

            text='OPEN',

            command=self.gripper_open

        )

        self.gripper_open_button.pack(

            fill='x',

            pady=4

        )


        self.gripper_close_button = tk.Button(

            frame,

            text='CLOSE',

            command=self.gripper_close

        )

        self.gripper_close_button.pack(

            fill='x',

            pady=4

        )


        self.gripper_status = tk.Label(

            frame,

            text='UNKNOWN',

            bg='#F8FAFC'

        )

        self.gripper_status.pack(

            pady=8

        )


    # ========================================================
    # SPEED
    # ========================================================

    def create_speed_panel(

        self,

        parent

    ):

        frame = self.create_section_frame(

            parent,

            'ROBOT SPEED'

        )


        self.speed_label = tk.Label(

            frame,

            text=f'SPEED LEVEL  {self.speed}/10',

            bg='#F8FAFC',

            font=('Arial', 10, 'bold')

        )

        self.speed_label.pack()


        self.speed_slider = tk.Scale(

            frame,

            from_=1,

            to=10,

            orient='horizontal',

            resolution=1,

            command=self.speed_changed

        )

        self.speed_slider.set(

            self.speed

        )

        self.speed_slider.pack(

            fill='x',

            pady=5

        )


    # ========================================================
    # PICK AND PLACE
    # ========================================================

    def create_pick_place_panel(

        self,

        parent

    ):

        frame = self.create_section_frame(

            parent,

            'PICK & PLACE TRAJECTORY'

        )


        tk.Label(

            frame,

            text='PICK POSITION',

            bg='#F8FAFC',

            font=('Arial', 10, 'bold')

        ).pack(

            anchor='w'

        )


        self.pick_x_entry = self.create_xyz_entry(

            frame,

            'X'

        )

        self.pick_y_entry = self.create_xyz_entry(

            frame,

            'Y'

        )

        self.pick_z_entry = self.create_xyz_entry(

            frame,

            'Z'

        )


        tk.Label(

            frame,

            text='PLACE POSITION',

            bg='#F8FAFC',

            font=('Arial', 10, 'bold')

        ).pack(

            anchor='w',

            pady=(10, 0)

        )


        self.place_x_entry = self.create_xyz_entry(

            frame,

            'X'

        )

        self.place_y_entry = self.create_xyz_entry(

            frame,

            'Y'

        )

        self.place_z_entry = self.create_xyz_entry(

            frame,

            'Z'

        )


        self.plan_move_button = tk.Button(

            frame,

            text='PLAN & MOVE',

            font=('Arial', 11, 'bold'),

            bg='#0F766E',

            fg='white',

            command=self.start_pick_and_place

        )

        self.plan_move_button.pack(

            fill='x',

            pady=(12, 6)

        )


        self.task_status_label = tk.Label(

            frame,

            text='Task Status: IDLE',

            bg='#F8FAFC',

            anchor='w'

        )

        self.task_status_label.pack(

            fill='x'

        )


        self.trajectory_status_display = tk.Label(

            frame,

            text='IDLE',

            bg='#E2E8F0',

            font=('Arial', 10, 'bold'),

            pady=6

        )

        self.trajectory_status_display.pack(

            fill='x',

            pady=(5, 0)

        )


    # ========================================================
    # SYSTEM PANEL
    # ========================================================

    def create_system_panel(

        self,

        parent

    ):

        frame = self.create_section_frame(

            parent,

            'SYSTEM CONTROL'

        )


        self.home_button = tk.Button(

            frame,

            text='HOME',

            font=('Arial', 10, 'bold'),

            command=self.home

        )

        self.home_button.pack(

            fill='x',

            pady=4

        )


        self.stop_button = tk.Button(

            frame,

            text='STOP',

            font=('Arial', 11, 'bold'),

            bg='#DC2626',

            fg='white',

            command=self.stop

        )

        self.stop_button.pack(

            fill='x',

            pady=4

        )


        self.resume_button = tk.Button(

            frame,

            text='RESUME MANUAL',

            font=('Arial', 10, 'bold'),

            bg='#2563EB',

            fg='white',

            command=self.resume_manual

        )

        self.resume_button.pack(

            fill='x',

            pady=4

        )


    # ========================================================
    # STATUS PANEL
    # ========================================================

    def create_status_panel(

        self,

        parent

    ):

        frame = self.create_section_frame(

            parent,

            'ROBOT STATUS'

        )


        self.robot_state = tk.Label(

            frame,

            text='WAITING',

            font=('Arial', 12, 'bold'),

            bg='#F8FAFC'

        )

        self.robot_state.pack(

            pady=5

        )


        self.status_label = tk.Label(

            frame,

            text='Waiting for /control_mode',

            bg='#F8FAFC',

            wraplength=260,

            justify='left'

        )

        self.status_label.pack(

            fill='x',

            pady=5

        )


    # ========================================================
    # STATUS BAR
    # ========================================================

    def create_status_bar(self):

        self.status_bar = tk.Label(

            self.root,

            text='ROS GUI READY',

            anchor='w',

            bg='#0F172A',

            fg='white',

            padx=15,

            pady=7

        )

        self.status_bar.pack(

            fill='x',

            side='bottom'

        )


    # ========================================================
    # CONTROL MODE CALLBACK
    # ========================================================

    def control_mode_callback(

        self,

        msg

    ):

        mode = msg.data.strip().upper()


        if mode not in [

            'MANUAL',

            'TRAJECTORY',

            'HALTED'

        ]:

            self.get_logger().warn(

                f'Invalid control mode received: {mode}'

            )

            return


        self.control_mode = mode

        self.control_mode_received = True


        if mode == 'MANUAL':

            self.mode_label.config(

                text='MANUAL',

                bg='#2563EB'

            )

            self.robot_state.config(

                text='READY'

            )


        elif mode == 'TRAJECTORY':

            self.mode_label.config(

                text='TRAJECTORY',

                bg='#0F766E'

            )

            self.robot_state.config(

                text='BUSY'

            )


        elif mode == 'HALTED':

            self.mode_label.config(

                text='HALTED',

                bg='#DC2626'

            )

            self.robot_state.config(

                text='HALTED'

            )


        if mode == 'HALTED':

            self.status_label.config(

                text='System HALTED - press RESUME MANUAL'

            )

        elif mode == 'MANUAL':

            self.status_label.config(

                text='Manual control enabled'

            )

        elif mode == 'TRAJECTORY':

            self.status_label.config(

                text='Trajectory running'

            )


        self.update_control_permissions()


    # ========================================================
    # UPDATE CONTROL PERMISSIONS
    # ========================================================

    def update_control_permissions(self):


        # ----------------------------------------------------
        # GUI starts disabled until mode is known.
        # ----------------------------------------------------

        if not self.control_mode_received:

            motion_state = 'disabled'

            gripper_state = 'disabled'

            home_state = 'disabled'

            plan_state = 'disabled'

            resume_state = 'disabled'


        # ----------------------------------------------------
        # MANUAL MODE
        # ----------------------------------------------------

        elif self.control_mode == 'MANUAL':

            motion_state = 'normal'

            gripper_state = 'normal'

            home_state = 'normal'

            plan_state = 'normal'

            resume_state = 'disabled'


        # ----------------------------------------------------
        # TRAJECTORY MODE
        # ----------------------------------------------------

        elif self.control_mode == 'TRAJECTORY':

            motion_state = 'disabled'

            gripper_state = 'disabled'

            home_state = 'disabled'

            plan_state = 'disabled'

            resume_state = 'disabled'


        # ----------------------------------------------------
        # HALTED MODE
        # ----------------------------------------------------

        else:

            motion_state = 'disabled'

            gripper_state = 'disabled'

            home_state = 'disabled'

            plan_state = 'disabled'

            resume_state = 'normal'


        # ----------------------------------------------------
        # JOINT CONTROLS
        # ----------------------------------------------------

        for slider in getattr(

            self,

            'sliders',

            []

        ):

            slider.config(

                state=motion_state

            )


        for entry in getattr(

            self,

            'angle_entries',

            []

        ):

            entry.config(

                state=motion_state

            )


        # ----------------------------------------------------
        # DIRECT IK
        # ----------------------------------------------------

        if hasattr(

            self,

            'solve_ik_button'

        ):

            self.solve_ik_button.config(

                state=motion_state

            )


        # ----------------------------------------------------
        # GRIPPER
        # ----------------------------------------------------

        if hasattr(

            self,

            'gripper_open_button'

        ):

            self.gripper_open_button.config(

                state=gripper_state

            )


        if hasattr(

            self,

            'gripper_close_button'

        ):

            self.gripper_close_button.config(

                state=gripper_state

            )


        # ----------------------------------------------------
        # HOME
        # ----------------------------------------------------

        if hasattr(

            self,

            'home_button'

        ):

            self.home_button.config(

                state=home_state

            )


        # ----------------------------------------------------
        # PICK & PLACE
        # ----------------------------------------------------

        if hasattr(

            self,

            'plan_move_button'

        ):

            self.plan_move_button.config(

                state=plan_state

            )


        # ----------------------------------------------------
        # RESUME
        # ----------------------------------------------------

        if hasattr(

            self,

            'resume_button'

        ):

            self.resume_button.config(

                state=resume_state

            )


    # ========================================================
    # JOINT SLIDER CHANGED
    # ========================================================

    def slider_changed(

        self,

        index,

        value

    ):

        if self.updating_gui:

            return


        if self.control_mode != 'MANUAL':

            return


        angle = float(

            value

        )


        self.joint_angles[index] = angle


        self.set_angle_entry(

            index,

            angle

        )


        self.publish_manual_joints()


        self.status_label.config(

            text=(
                f'{self.joint_names[index]} = '
                f'{angle:+.1f} deg'
            )

        )


    # ========================================================
    # ANGLE ENTRY CHANGED
    # ========================================================

    def angle_entry_changed(

        self,

        index

    ):

        if self.updating_gui:

            return


        if self.control_mode != 'MANUAL':

            return


        try:

            angle = float(

                self.angle_entries[index].get()

            )

        except ValueError:

            self.set_angle_entry(

                index,

                self.joint_angles[index]

            )

            return


        angle = max(

            self.joint_min[index],

            min(

                self.joint_max[index],

                angle

            )

        )


        self.joint_angles[index] = angle


        self.updating_gui = True

        self.sliders[index].set(

            angle

        )

        if not self._pending_gui_unlock:

            self._pending_gui_unlock = True

            self.root.after_idle(

                self._finish_gui_update

            )


        self.set_angle_entry(

            index,

            angle

        )


        self.publish_manual_joints()


    # ========================================================
    # SET ANGLE ENTRY
    # ========================================================

    def set_angle_entry(

        self,

        index,

        angle

    ):

        entry = self.angle_entries[index]


        entry.delete(

            0,

            tk.END

        )


        entry.insert(

            0,

            f'{float(angle):.1f}'

        )


    # ========================================================
    # PUBLISH MANUAL JOINTS
    # ========================================================

    def publish_manual_joints(self):


        # Server-side mux is the real authority.
        #
        # This check avoids unnecessary publishing.

        if self.control_mode != 'MANUAL':

            return


        msg = Float64MultiArray()


        msg.data = [

            float(v)

            for v in self.joint_angles

        ]


        self.manual_joint_pub.publish(

            msg

        )


    # ========================================================
    # JOINT COMMAND CALLBACK
    #
    # SINGLE DISPLAY SOURCE OF TRUTH
    #
    # IMPORTANT:
    #
    # No conditions based on:
    #
    # direct_ik_pending
    # trajectory state
    # status strings
    #
    # GUI always displays the actual command selected by mux.
    # ========================================================

    def joint_command_callback(

        self,

        msg

    ):

        if len(msg.data) != 5:

            return


        q = [

            float(v)

            for v in msg.data

        ]


        self.joint_angles = q


        self.set_gui_joint_values(

            q

        )


    # ========================================================
    # SET GUI JOINT VALUES
    # ========================================================

    def set_gui_joint_values(

        self,

        q

    ):

        if len(q) != 5:

            return


        self.updating_gui = True


        for i in range(5):

            angle = float(q[i])


            self.sliders[i].set(

                angle

            )


            self.set_angle_entry(

                i,

                angle

            )


        if not self._pending_gui_unlock:

            self._pending_gui_unlock = True

            self.root.after_idle(

                self._finish_gui_update

            )


    def _finish_gui_update(self):

        self._pending_gui_unlock = False

        self.updating_gui = False


    # ========================================================
    # FK CALLBACK
    # ========================================================

    def end_effector_callback(

        self,

        msg

    ):

        if len(msg.data) != 3:

            return


        self.ee_position = [

            float(msg.data[0]),

            float(msg.data[1]),

            float(msg.data[2])

        ]


        self.x_label.config(

            text=f'{self.ee_position[0]:.2f}'

        )

        self.y_label.config(

            text=f'{self.ee_position[1]:.2f}'

        )

        self.z_label.config(

            text=f'{self.ee_position[2]:.2f}'

        )


        self.fk_status.config(

            text='LIVE'

        )


    # ========================================================
    # DIRECT IK
    # ========================================================

    def solve_ik_and_move(self):


        if self.control_mode != 'MANUAL':

            self.status_label.config(

                text=(
                    'Direct IK unavailable unless '
                    'control mode is MANUAL'
                )

            )

            return


        xyz = self.read_xyz(

            self.target_x_entry,

            self.target_y_entry,

            self.target_z_entry

        )


        if xyz is None:

            self.status_label.config(

                text='ERROR - Invalid Cartesian target'

            )

            return


        msg = Float64MultiArray()

        msg.data = list(

            xyz

        )


        self.direct_ik_pending = True


        self.ik_gui_target_pub.publish(

            msg

        )


        self.robot_state.config(

            text='SOLVING IK'

        )


        self.ik_status_label.config(

            text='IK Status: Solving...'

        )


        self.status_label.config(

            text=(
                f'IK target sent - '
                f'X={xyz[0]:.1f}, '
                f'Y={xyz[1]:.1f}, '
                f'Z={xyz[2]:.1f}'
            )

        )


    # ========================================================
    # IK SOLUTION CALLBACK
    #
    # GUI receives only GUI-specific IK results.
    #
    # No trajectory cross-talk is possible.
    # ========================================================

    def ik_solution_callback(

        self,

        msg

    ):

        if len(msg.data) != 5:

            return


        if self.control_mode != 'MANUAL':

            return


        q = [

            float(v)

            for v in msg.data

        ]


        # Do not move sliders optimistically.
        #
        # Publish the solution to the mux input.
        #
        # Sliders will move only when the resulting command
        # appears on /joint_commands.

        command = Float64MultiArray()

        command.data = q


        self.ik_joint_pub.publish(

            command

        )


        self.direct_ik_pending = False


        self.ik_status_label.config(

            text='IK Status: SUCCESS - command sent'

        )


        self.status_label.config(

            text='IK solution sent through command mux'

        )


    # ========================================================
    # IK STATUS CALLBACK
    #
    # GUI receives only GUI IK status.
    # ========================================================

    def ik_status_callback(

        self,

        msg

    ):

        status = msg.data


        self.ik_status_label.config(

            text=f'IK Status: {status}'

        )


        upper = status.upper()


        if 'SUCCESS' in upper:

            return


        if 'FAILED' in upper or 'ERROR' in upper:

            self.direct_ik_pending = False


            if self.control_mode == 'MANUAL':

                self.robot_state.config(

                    text='READY'

                )


            self.status_label.config(

                text=status

            )


    # ========================================================
    # TRAJECTORY STATUS CALLBACK
    #
    # DISPLAY ONLY
    #
    # Does NOT change authority.
    # ========================================================

    def trajectory_status_callback(

        self,

        msg

    ):

        status = msg.data


        self.task_status_label.config(

            text=f'Task Status: {status}'

        )


        self.trajectory_status_display.config(

            text=status

        )


        upper = status.upper()


        if upper == 'COMPLETE':

            self.status_label.config(

                text='Pick & place COMPLETE'

            )


        elif upper == 'STOPPED':

            self.status_label.config(

                text='Trajectory STOPPED'

            )


        elif 'ERROR' in upper:

            self.status_label.config(

                text=status

            )


    # ========================================================
    # PICK AND PLACE
    # ========================================================

    def start_pick_and_place(self):


        if self.control_mode != 'MANUAL':

            self.status_label.config(

                text=(
                    'Pick & place can only start '
                    'from MANUAL mode'
                )

            )

            return


        pick = self.read_xyz(

            self.pick_x_entry,

            self.pick_y_entry,

            self.pick_z_entry

        )


        place = self.read_xyz(

            self.place_x_entry,

            self.place_y_entry,

            self.place_z_entry

        )


        if pick is None:

            self.status_label.config(

                text='ERROR - Invalid PICK XYZ'

            )

            return


        if place is None:

            self.status_label.config(

                text='ERROR - Invalid PLACE XYZ'

            )

            return


        # ----------------------------------------------------
        # Publish PICK
        # ----------------------------------------------------

        pick_msg = Float64MultiArray()

        pick_msg.data = list(

            pick

        )

        self.pick_pub.publish(

            pick_msg

        )


        # ----------------------------------------------------
        # Publish PLACE
        # ----------------------------------------------------

        place_msg = Float64MultiArray()

        place_msg.data = list(

            place

        )

        self.place_pub.publish(

            place_msg

        )


        # ----------------------------------------------------
        # START TRAJECTORY
        #
        # trajectory_node owns the transition to TRAJECTORY.
        # GUI does not directly set TRAJECTORY mode.
        # ----------------------------------------------------

        command = String()

        command.data = 'START'


        self.trajectory_command_pub.publish(

            command

        )


        self.task_status_label.config(

            text='Task Status: STARTING...'

        )


        self.trajectory_status_display.config(

            text='STARTING'

        )


        self.robot_state.config(

            text='STARTING'

        )


        self.status_label.config(

            text=(
                f'Pick & place requested - '
                f'PICK={pick}  PLACE={place}'
            )

        )


    # ========================================================
    # GRIPPER OPEN
    # ========================================================

    def gripper_open(self):


        if self.control_mode != 'MANUAL':

            self.status_label.config(

                text='Manual gripper control unavailable'

            )

            return


        self.send_manual_gripper(

            self.gripper_open_angle

        )


        self.gripper_status.config(

            text='OPEN  -  0 deg'

        )


        self.status_label.config(

            text='Gripper OPEN command sent'

        )


    # ========================================================
    # GRIPPER CLOSE
    # ========================================================

    def gripper_close(self):


        if self.control_mode != 'MANUAL':

            self.status_label.config(

                text='Manual gripper control unavailable'

            )

            return


        self.send_manual_gripper(

            self.gripper_close_angle

        )


        self.gripper_status.config(

            text='CLOSED  -  -55 deg'

        )


        self.status_label.config(

            text='Gripper CLOSE command sent'

        )


    # ========================================================
    # SEND MANUAL GRIPPER
    # ========================================================

    def send_manual_gripper(

        self,

        angle

    ):

        if self.control_mode != 'MANUAL':

            return


        msg = Float64MultiArray()

        msg.data = [

            float(angle)

        ]


        self.manual_gripper_pub.publish(

            msg

        )


    # ========================================================
    # SPEED
    # ========================================================

    def speed_changed(

        self,

        value

    ):

        self.speed = int(

            float(value)

        )


        self.speed_label.config(

            text=f'SPEED LEVEL  {self.speed}/10'

        )


        msg = Int32()

        msg.data = self.speed


        self.speed_pub.publish(

            msg

        )


    # ========================================================
    # HOME
    # ========================================================

    def home(self):


        if self.control_mode != 'MANUAL':

            self.status_label.config(

                text='HOME is allowed only in MANUAL mode'

            )

            return


        # ----------------------------------------------------
        # HOME THROUGH NORMAL JOINT PATH
        #
        # Keeps GUI/FK/joint_commands synchronized.
        # ----------------------------------------------------

        home_joints = [

            0.0,

            0.0,

            0.0,

            0.0,

            0.0

        ]


        self.joint_angles = home_joints.copy()


        joint_msg = Float64MultiArray()

        joint_msg.data = home_joints


        self.manual_joint_pub.publish(

            joint_msg

        )


        # ----------------------------------------------------
        # ALSO SEND FIRMWARE HOME COMMAND
        # ----------------------------------------------------

        system_msg = String()

        system_msg.data = 'HOME'


        self.command_pub.publish(

            system_msg

        )


        self.robot_state.config(

            text='HOMING'

        )


        self.status_label.config(

            text='HOME command sent'

        )


        # IMPORTANT:
        #
        # Do NOT directly move GUI sliders here.
        #
        # Sliders update only when /joint_commands arrives.


    # ========================================================
    # STOP
    # ========================================================

    def stop(self):


        # ----------------------------------------------------
        # FORCE HALTED
        #
        # Independent of trajectory_node.
        # ----------------------------------------------------

        mode_msg = String()

        mode_msg.data = 'HALTED'


        self.control_mode_request_pub.publish(

            mode_msg

        )


        # ----------------------------------------------------
        # STOP TRAJECTORY LOGIC
        #
        # If trajectory_node is alive, it will clean up.
        # ----------------------------------------------------

        trajectory_msg = String()

        trajectory_msg.data = 'STOP'


        self.trajectory_command_pub.publish(

            trajectory_msg

        )


        # ----------------------------------------------------
        # PHYSICAL ROBOT STOP
        #
        # Always reaches UART / ESP32.
        # ----------------------------------------------------

        robot_msg = String()

        robot_msg.data = 'STOP'


        self.command_pub.publish(

            robot_msg

        )


        self.robot_state.config(

            text='HALTED'

        )


        self.status_label.config(

            text='STOP sent - system HALTED'

        )


    # ========================================================
    # RESUME MANUAL
    # ========================================================

    def resume_manual(self):


        if self.control_mode != 'HALTED':

            self.status_label.config(

                text='RESUME MANUAL is only available from HALTED'

            )

            return


        msg = String()

        msg.data = 'MANUAL'


        self.control_mode_request_pub.publish(

            msg

        )


        self.status_label.config(

            text='Manual mode requested'

        )


    # ========================================================
    # READ XYZ
    # ========================================================

    def read_xyz(

        self,

        x_entry,

        y_entry,

        z_entry

    ):

        try:

            x = float(

                x_entry.get()

            )

            y = float(

                y_entry.get()

            )

            z = float(

                z_entry.get()

            )


            return (

                x,

                y,

                z

            )


        except ValueError:

            return None


    # ========================================================
    # PROCESS ROS
    # ========================================================

    def process_ros(self):

        if not rclpy.ok():

            return

        rclpy.spin_once(

            self,

            timeout_sec=0.0

        )

        self.root.after(

            10,

            self.process_ros

        )


    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self):

        self.get_logger().info(

            'Closing robot GUI'

        )


        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

def main(args=None):


    rclpy.init(

        args=args

    )


    node = RobotGUI()


    try:

        node.root.mainloop()


    except KeyboardInterrupt:

        pass


    finally:

        node.destroy_node()


        if rclpy.ok():

            rclpy.shutdown()


# ============================================================

if __name__ == '__main__':

    main()