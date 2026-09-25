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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

import tkinter as tk
from tkinter import ttk


# ============================================================
# ROBOT GUI
# ============================================================


# ============================================================
# CUSTOM ROUNDED WIDGETS (GLASSMORPHISM / IOS STYLE)
# ============================================================
class RoundedFrame(tk.Canvas):
    def __init__(self, parent, bg_color="#F3F8FC", card_color="#FFFFFF", border_color="#D9E7F2", radius=16, shadow=True, **kwargs):
        super().__init__(parent, bg=bg_color, highlightthickness=0, **kwargs)
        self.radius = radius
        self.card_color = card_color
        self.border_color = border_color
        self.shadow = shadow
        self.bind("<Configure>", self.draw)

    def draw(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2 * self.radius or h < 2 * self.radius: return
        
        if self.shadow:
            self.create_rounded_rect(3, 5, w-3, h-1, self.radius, fill="#E6F0F9", outline="")
            self.create_rounded_rect(2, 4, w-2, h-2, self.radius, fill="#EBF3FA", outline="")
            
        self.create_rounded_rect(2, 2, w-4, h-4, self.radius, fill=self.card_color, outline=self.border_color, width=1)

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = (x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1,
                  x2, y1, x2, y1+r, x2, y1+r, x2, y2-r,
                  x2, y2-r, x2, y2, x2-r, y2, x2-r, y2,
                  x1+r, y2, x1+r, y2, x1, y2, x1, y2-r,
                  x1, y2-r, x1, y1+r, x1, y1+r, x1, y1)
        return self.create_polygon(points, smooth=True, **kwargs)


class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command=None, bg_parent="#FFFFFF", bg="#4F9DE8", fg="#FFFFFF", hover_bg="#6FB7E9", border_color="", radius=10, height=38, font=("Inter", 10, "bold"), **kwargs):
        super().__init__(parent, height=height, bg=bg_parent, highlightthickness=0, **kwargs)
        self.command = command
        self.text = text
        self.bg_color = bg
        self.fg_color = fg
        self.hover_bg = hover_bg
        self.border_color = border_color if border_color else bg
        self.radius = radius
        self.font = font
        self.rect_id = None
        self.text_id = None
        self.bind("<Configure>", self.draw)
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def config(self, **kwargs):
        super().config(**kwargs)
        if "state" in kwargs:
            self.draw()
            
    def configure(self, **kwargs):
        self.config(**kwargs)

    def draw(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10: return
        is_disabled = str(self.cget("state")) == "disabled"
        fill_color = "#D9E7F2" if is_disabled else self.bg_color
        outline_color = "#D9E7F2" if is_disabled else self.border_color
        text_color = "#71849A" if is_disabled else self.fg_color
        
        if not is_disabled and self.bg_color != "#FFFFFF" and self.bg_color != self.cget("bg"):
            self.create_rounded_rect(2, 4, w-2, h-1, self.radius, fill="#DDEAF4", outline="")
            
        self.rect_id = self.create_rounded_rect(1, 1, w-3, h-3, self.radius, fill=fill_color, outline=outline_color, width=1)
        self.text_id = self.create_text(w/2, (h-2)/2, text=self.text, fill=text_color, font=self.font, justify="center")

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = (x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1,
                  x2, y1, x2, y1+r, x2, y1+r, x2, y2-r,
                  x2, y2-r, x2, y2, x2-r, y2, x2-r, y2,
                  x1+r, y2, x1+r, y2, x1, y2, x1, y2-r,
                  x1, y2-r, x1, y1+r, x1, y1+r, x1, y1)
        return self.create_polygon(points, smooth=True, **kwargs)

    def on_press(self, event):
        if str(self.cget("state")) == "disabled": return
        if self.rect_id:
            self.itemconfig(self.rect_id, fill=self.border_color)
            self.move(self.text_id, 0, 1)

    def on_release(self, event):
        if str(self.cget("state")) == "disabled": return
        if self.rect_id:
            self.itemconfig(self.rect_id, fill=self.hover_bg)
            self.move(self.text_id, 0, -1)
            if self.command: self.command()
        
    def on_enter(self, event):
        if str(self.cget("state")) == "disabled": return
        if self.rect_id:
            self.itemconfig(self.rect_id, fill=self.hover_bg)
            self.config(cursor="hand2")

    def on_leave(self, event):
        if str(self.cget("state")) == "disabled": return
        if self.rect_id:
            self.itemconfig(self.rect_id, fill=self.bg_color)
            self.config(cursor="")


class RoundedSlider(tk.Canvas):
    def __init__(self, parent, command=None, from_=0, to=100, bg_parent="#FFFFFF", **kwargs):
        super().__init__(parent, height=28, bg=bg_parent, highlightthickness=0, **kwargs)
        self.command = command
        self.val_min = from_
        self.val_max = to
        self.value = from_
        self.radius = 4
        self.handle_r = 10
        self.dragging = False
        
        self.bind("<Configure>", self.draw)
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Enter>", lambda e: self.config(cursor="hand2") if str(self.cget("state")) != "disabled" else None)
        self.bind("<Leave>", lambda e: self.config(cursor=""))

    def config(self, **kwargs):
        super().config(**kwargs)
        if "state" in kwargs:
            self.draw()
            
    def configure(self, **kwargs):
        self.config(**kwargs)

    def set(self, val):
        self.value = max(self.val_min, min(self.val_max, val))
        self.draw()

    def get(self):
        return self.value

    def val_to_x(self, w):
        pad = self.handle_r + 4
        span = w - 2 * pad
        if self.val_max == self.val_min: return pad
        pct = (self.value - self.val_min) / (self.val_max - self.val_min)
        return pad + pct * span

    def x_to_val(self, x, w):
        pad = self.handle_r + 4
        span = w - 2 * pad
        pct = (x - pad) / span
        pct = max(0.0, min(1.0, pct))
        return self.val_min + pct * (self.val_max - self.val_min)

    def draw(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 30: return
        
        cy = h / 2
        pad = self.handle_r + 4
        
        is_disabled = str(self.cget("state")) == "disabled"
        track_color = "#E2E8F0" if is_disabled else "#DDEAF4"
        active_color = "#94A3B8" if is_disabled else "#6FB7E9"
        border_color = "#94A3B8" if is_disabled else "#5AA5DD"
        
        self.create_rounded_rect(pad, cy-self.radius, w-pad, cy+self.radius, self.radius, fill=track_color, outline="")
        
        hx = self.val_to_x(w)
        if hx > pad:
            self.create_rounded_rect(pad, cy-self.radius, hx, cy+self.radius, self.radius, fill=active_color, outline="")
            
        if not is_disabled:
            self.create_oval(hx-self.handle_r, cy-self.handle_r+1, hx+self.handle_r, cy+self.handle_r+1, fill="#D9E7F2", outline="")
            
        self.create_oval(hx-self.handle_r, cy-self.handle_r, hx+self.handle_r, cy+self.handle_r, fill="#FFFFFF", outline=border_color, width=1.5)

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = (x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1,
                  x2, y1, x2, y1+r, x2, y1+r, x2, y2-r,
                  x2, y2-r, x2, y2, x2-r, y2, x2-r, y2,
                  x1+r, y2, x1+r, y2, x1, y2, x1, y2-r,
                  x1, y2-r, x1, y1+r, x1, y1+r, x1, y1)
        return self.create_polygon(points, smooth=True, **kwargs)

    def on_press(self, event):
        if str(self.cget("state")) == "disabled": return
        self.dragging = True
        self.on_drag(event)

    def on_drag(self, event):
        if not self.dragging or str(self.cget("state")) == "disabled": return
        w = self.winfo_width()
        self.value = self.x_to_val(event.x, w)
        self.draw()
        if self.command:
            self.command(str(round(self.value, 2)))

    def on_release(self, event):
        if str(self.cget("state")) == "disabled": return
        self.dragging = False
        if self.command:
            self.command(str(round(self.value, 2)))

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

            'manual_joint_command',

            10

        )


        # ----------------------------------------------------
        # DIRECT IK TARGET
        #
        # GUI -> ik_node
        # ----------------------------------------------------

        self.ik_gui_target_pub = self.create_publisher(

            Float64MultiArray,

            'ik_gui_target',

            10

        )


        # ----------------------------------------------------
        # MANUAL GRIPPER COMMAND
        #
        # GUI -> gripper_mux_node
        # ----------------------------------------------------

        self.manual_gripper_pub = self.create_publisher(

            Float64MultiArray,

            'manual_gripper_command',

            10

        )


        # ----------------------------------------------------
        # DIRECT IK JOINT COMMAND
        #
        # GUI -> command_mux_node
        # ----------------------------------------------------

        self.ik_joint_pub = self.create_publisher(

            Float64MultiArray,

            'ik_joint_command',

            10

        )


        # ----------------------------------------------------
        # ROBOT SPEED
        # ----------------------------------------------------

        self.speed_pub = self.create_publisher(

            Int32,

            'robot_speed',

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

            'robot_command',

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

            'control_mode_request',

            10

        )


        # ----------------------------------------------------
        # PICK POSITION
        # ----------------------------------------------------

        self.pick_pub = self.create_publisher(

            Float64MultiArray,

            'pick_position',

            10

        )


        # ----------------------------------------------------
        # PLACE POSITION
        # ----------------------------------------------------

        self.place_pub = self.create_publisher(

            Float64MultiArray,

            'place_position',

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

            'trajectory_command',

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

            'joint_commands',

            self.joint_command_callback,

            10

        )


        # ----------------------------------------------------
        # FK END EFFECTOR POSITION
        # ----------------------------------------------------

        self.ee_subscription = self.create_subscription(

            Float64MultiArray,

            'end_effector_position',

            self.end_effector_callback,

            10

        )


        # ----------------------------------------------------
        # GUI IK SOLUTION
        # ----------------------------------------------------

        self.ik_solution_subscription = self.create_subscription(

            Float64MultiArray,

            'ik_gui_solution',

            self.ik_solution_callback,

            10

        )


        # ----------------------------------------------------
        # GUI IK STATUS
        # ----------------------------------------------------

        self.ik_status_subscription = self.create_subscription(

            String,

            'ik_gui_status',

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

            'trajectory_status',

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

            'control_mode',

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


        self.joint_min = [

            -135.0,

            -135.0,

            -135.0,

            -90.0,

            -90.0

        ]


        self.joint_max = [

            135.0,

            135.0,

            135.0,

            90.0,

            90.0

        ]


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

        self.gripper_open_angle = 0.0

        self.gripper_close_angle = -55.0


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
        # DESIGN TOKENS (PHASE 1 POLISH - GLASSMORPHISM)
        # ====================================================
        self.bg_color = "#F3F8FC"
        self.card_color = "#FFFFFF"
        self.glass_color = "#F8FCFF"
        self.panel_border = "#D9E7F2"
        self.primary_blue = "#4F9DE8"
        self.sky_blue = "#6FB7E9"
        self.soft_blue = "#E8F4FD"
        self.deep_blue = "#315F8A"
        self.primary_text = "#24364B"
        self.secondary_text = "#71849A"
        self.success_color = "#46B982"
        self.warning_color = "#E6A34A"
        self.error_color = "#E66A6A"
        self.disabled_color = "#71849A"
        
        self.font_main = ("Inter", 10)
        self.font_bold = ("Inter", 10, "bold")
        self.font_title = ("Inter", 16, "bold")
        self.font_section = ("Inter", 11, "bold")
        self.font_small = ("Inter", 9)

        # ====================================================
        # WINDOW
        # ====================================================
        self.root = tk.Tk()
        self.root.title('Multi-Robot Control Station')
        self.root.geometry('1600x1000')
        self.root.minsize(1400, 900)
        self.root.configure(bg=self.bg_color)

        self.setup_style()
        self.create_header()

        # ====================================================
        # MAIN LAYOUT
        # ====================================================
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill='both', expand=True, padx=20, pady=12)
        main_frame.columnconfigure(0, weight=5) # 3D view
        main_frame.columnconfigure(1, weight=2) # Status, System
        main_frame.columnconfigure(2, weight=2) # TCP, Gripper, Speed
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=0)

        # --- COL 0 (3D VIEW) ---
        left_parent = tk.Frame(main_frame, bg=self.bg_color)
        left_parent.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 10))
        left_parent.grid_propagate(False)
        
        left_frame = RoundedFrame(left_parent, bg_color=self.bg_color, card_color=self.glass_color, border_color=self.panel_border, radius=18, shadow=True)
        left_frame.pack(fill='both', expand=True)
        
        content = tk.Frame(left_frame, bg=self.glass_color)
        content.pack(fill='both', expand=True, padx=16, pady=14)
        tk.Label(content, text="LIVE ROBOT VIEW", font=self.font_title, fg=self.deep_blue, bg=self.glass_color).pack(pady=(0, 2))
        
        self.trace_enabled = tk.BooleanVar(value=True)
        cb = tk.Checkbutton(content, text="TCP TRACE", variable=self.trace_enabled, bg=self.glass_color, fg=self.primary_text, selectcolor=self.glass_color, activebackground=self.glass_color, font=("Inter", 9, "bold"), bd=0, highlightthickness=0)
        cb.pack(pady=(0, 2))
        
        # Plot container
        self.plot_frame = tk.Frame(content, bg=self.glass_color)
        self.plot_frame.pack(fill='both', expand=True)
        self.setup_3d_visualizer(self.plot_frame)

        # --- COL 1 (STATUS & SYSTEM) ---
        mid_parent = tk.Frame(main_frame, bg=self.bg_color)
        mid_parent.grid(row=0, column=1, sticky="nsew", padx=(6, 6), pady=(0, 10))
        
        self.create_status_panel(mid_parent)
        self.create_system_panel(mid_parent)

        # --- COL 2 (TCP, GRIPPER) ---
        right_parent = tk.Frame(main_frame, bg=self.bg_color)
        right_parent.grid(row=0, column=2, sticky="nsew", padx=(6, 0), pady=(0, 10))
        
        self.create_current_position_panel(right_parent)
        self.create_gripper_panel(right_parent)
        self.create_speed_panel(right_parent)

        # --- BOTTOM AREA (MODE-SPECIFIC CONTROLS) ---
        bottom_frame = tk.Frame(main_frame, bg=self.bg_color)
        bottom_frame.grid(row=1, column=0, columnspan=3, sticky="nsew")
        bottom_frame.columnconfigure(0, weight=1, uniform="bot")
        bottom_frame.columnconfigure(1, weight=1, uniform="bot")
        bottom_frame.columnconfigure(2, weight=1, uniform="bot")

        bot_left = tk.Frame(bottom_frame, bg=self.bg_color)
        bot_left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        bot_mid = tk.Frame(bottom_frame, bg=self.bg_color)
        bot_mid.grid(row=0, column=1, sticky="nsew", padx=(8, 8))
        bot_right = tk.Frame(bottom_frame, bg=self.bg_color)
        bot_right.grid(row=0, column=2, sticky="nsew", padx=(8, 0))

        self.create_manual_control(bot_left)
        self.create_direct_ik_panel(bot_mid)
        self.create_pick_place_panel(bot_right)

        self.create_status_bar()

        self.root.protocol('WM_DELETE_WINDOW', self.shutdown)
        self.root.after(10, self.process_ros)
        self.update_control_permissions()

        self.get_logger().info('========================================')
        self.get_logger().info('5-DOF ROBOT GUI STARTED')
        self.get_logger().info('========================================')
        self.get_logger().info('Waiting for /control_mode...')


    # ========================================================
    # STYLE
    # ========================================================
    def setup_style(self):
        pass

    # ========================================================
    # HEADER
    # ========================================================
    def create_header(self):
        header = tk.Frame(self.root, bg=self.card_color, highlightbackground=self.panel_border, highlightthickness=1)
        header.pack(fill='x')

        title_frame = tk.Frame(header, bg=self.card_color)
        title_frame.pack(side='left', padx=24, pady=10)
        
        tk.Label(title_frame, text='MULTI-ROBOT CONTROL STATION', font=("Inter", 18, "bold"), fg=self.deep_blue, bg=self.card_color).pack(anchor='w')
        tk.Label(title_frame, text='ROS 2 • ESP32 • PCA9685 • ROBOT 01', font=("Inter", 11), fg=self.secondary_text, bg=self.card_color).pack(anchor='w', pady=(4, 0))

        right_header = tk.Frame(header, bg=self.card_color)
        right_header.pack(side='right', padx=24, pady=10)
        
        self.stop_button = RoundedButton(right_header, text='■  STOP', command=self.stop, bg_parent=self.card_color, bg=self.error_color, fg='white', hover_bg="#C53030", width=120)
        self.stop_button.pack(side='right', padx=(20, 0))

        self.mode_label = tk.Label(right_header, text='● ROBOT READY', font=self.font_bold, fg=self.success_color, bg=self.card_color, padx=20, pady=5)
        self.mode_label.pack(side='right')

    def create_section_frame(self, parent, title):
        outer = RoundedFrame(parent, bg_color=self.bg_color, card_color=self.card_color, border_color=self.panel_border, radius=12, shadow=True)
        outer.pack(fill='x', pady=(0, 10))
        content = tk.Frame(outer, bg=self.card_color)
        content.pack(fill='both', expand=True, padx=16, pady=10)
        if title:
            tk.Label(content, text=title, font=self.font_section, bg=self.card_color, fg=self.deep_blue, anchor='w').pack(fill='x', pady=(0, 6))
        return content

    def create_rounded_entry(self, parent, text_val='0.0'):
        container = tk.Frame(parent, bg=self.card_color, highlightbackground=self.panel_border, highlightthickness=1, bd=0)
        entry = tk.Entry(container, width=8, justify='center', font=self.font_main, bg="#FFFFFF", fg=self.primary_text, relief='flat', insertbackground=self.primary_text)
        entry.insert(0, text_val)
        entry.pack(padx=4, pady=2)
        def on_focus_in(e): container.config(highlightbackground=self.sky_blue, highlightthickness=2)
        def on_focus_out(e): container.config(highlightbackground=self.panel_border, highlightthickness=1)
        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out, add="+")
        return entry, container

    def create_manual_control(self, parent):
        frame = self.create_section_frame(parent, 'MANUAL JOINT CONTROL')
        self.sliders = []
        self.angle_entries = []

        for i in range(5):
            row = tk.Frame(frame, bg=self.card_color)
            row.pack(fill='x', pady=3)
            tk.Label(row, text=self.joint_names[i], width=5, anchor='w', font=self.font_bold, bg=self.card_color, fg=self.secondary_text).pack(side='left')
            slider = RoundedSlider(row, from_=self.joint_min[i], to=self.joint_max[i], command=lambda value, idx=i: self.slider_changed(idx, value))
            slider.set(self.joint_angles[i])
            slider.pack(side='left', fill='x', expand=True, padx=12)
            entry, container = self.create_rounded_entry(row, '0.0')
            entry.bind('<Return>', lambda event, idx=i: self.angle_entry_changed(idx))
            entry.bind('<FocusOut>', lambda event, idx=i: self.angle_entry_changed(idx))
            container.pack(side='right')
            self.sliders.append(slider)
            self.angle_entries.append(entry)
        
        btn = RoundedButton(frame, text='📋 COPY JOINTS', command=self.copy_joints, bg_parent=self.card_color, bg=self.glass_color, fg=self.primary_blue, hover_bg=self.soft_blue, border_color=self.sky_blue, height=30)
        btn.pack(fill='x', pady=(6, 0))

    def create_current_position_panel(self, parent):
        frame = self.create_section_frame(parent, 'CURRENT TCP')
        self.x_label = self.create_value_row(frame, 'X', '0.00 mm')
        self.y_label = self.create_value_row(frame, 'Y', '0.00 mm')
        self.z_label = self.create_value_row(frame, 'Z', '0.00 mm')
        self.fk_status = tk.Label(frame, text='● WAITING', font=self.font_bold, fg=self.disabled_color, bg=self.card_color)
        self.fk_status.pack(pady=(3, 3))
        
        btn = RoundedButton(frame, text='📋 COPY XYZ', command=self.copy_xyz, bg_parent=self.card_color, bg=self.glass_color, fg=self.primary_blue, hover_bg=self.soft_blue, border_color=self.sky_blue, height=30)
        btn.pack(fill='x', pady=(0, 0))

    def create_value_row(self, parent, name, value):
        row = tk.Frame(parent, bg=self.card_color)
        row.pack(fill='x', pady=2)
        tk.Label(row, text=name, width=4, anchor='w', font=self.font_bold, bg=self.card_color, fg=self.secondary_text).pack(side='left')
        label = tk.Label(row, text=value, font=self.font_bold, fg=self.deep_blue, bg=self.card_color, anchor='e')
        label.pack(side='right')
        return label

    def create_direct_ik_panel(self, parent):
        frame = self.create_section_frame(parent, 'CARTESIAN TARGET')
        self.target_x_entry = self.create_xyz_entry(frame, 'X')
        self.target_y_entry = self.create_xyz_entry(frame, 'Y')
        self.target_z_entry = self.create_xyz_entry(frame, 'Z')
        
        btn_frame = tk.Frame(frame, bg=self.card_color)
        btn_frame.pack(fill='x', pady=(4, 0))
        btn_set = RoundedButton(btn_frame, text='SET CURRENT', command=self.set_current_cartesian, bg_parent=self.card_color, bg=self.glass_color, fg=self.primary_blue, hover_bg=self.soft_blue, border_color=self.sky_blue, height=28)
        btn_set.pack(side='left', fill='x', expand=True, padx=(0, 4))
        btn_clear = RoundedButton(btn_frame, text='CLEAR', command=self.clear_cartesian, bg_parent=self.card_color, bg=self.glass_color, fg=self.primary_blue, hover_bg=self.soft_blue, border_color=self.sky_blue, height=28)
        btn_clear.pack(side='right', fill='x', expand=True, padx=(4, 0))
        
        self.solve_ik_button = RoundedButton(frame, text='⌖  SOLVE IK & MOVE', command=self.solve_ik_and_move_wrapped, bg_parent=self.card_color, bg=self.primary_blue, hover_bg=self.sky_blue, fg='white', height=36)
        self.solve_ik_button.pack(fill='x', pady=(8, 4))
        self.ik_status_label = tk.Label(frame, text='IK STATUS: IDLE', bg=self.card_color, fg=self.secondary_text, font=self.font_small, anchor='w')
        self.ik_status_label.pack(fill='x')

    def create_xyz_entry(self, parent, label_text):
        row = tk.Frame(parent, bg=self.card_color)
        row.pack(fill='x', pady=2)
        tk.Label(row, text=f'{label_text} (mm)', width=8, anchor='w', font=self.font_main, bg=self.card_color, fg=self.secondary_text).pack(side='left')
        entry, container = self.create_rounded_entry(row, '0.0')
        container.pack(side='right')
        return entry

    def create_gripper_panel(self, parent):
        frame = self.create_section_frame(parent, 'GRIPPER')
        btn_frame = tk.Frame(frame, bg=self.card_color)
        btn_frame.pack(fill='x', pady=(0, 6))
        self.gripper_open_button = RoundedButton(btn_frame, text='◱  OPEN', command=self.gripper_open, bg_parent=self.card_color, bg=self.card_color, border_color=self.sky_blue, fg=self.primary_blue, hover_bg=self.soft_blue, width=100, height=32)
        self.gripper_open_button.pack(side='left', fill='x', expand=True, padx=(0, 4))
        self.gripper_close_button = RoundedButton(btn_frame, text='◲  CLOSE', command=self.gripper_close, bg_parent=self.card_color, bg=self.card_color, border_color=self.sky_blue, fg=self.primary_blue, hover_bg=self.soft_blue, width=100, height=32)
        self.gripper_close_button.pack(side='right', fill='x', expand=True, padx=(4, 0))
        self.gripper_status = tk.Label(frame, text='● UNKNOWN', font=self.font_bold, fg=self.disabled_color, bg=self.card_color)
        self.gripper_status.pack(pady=(2, 0))

    def create_speed_panel(self, parent):
        frame = self.create_section_frame(parent, 'MOTION SPEED')
        self.speed_label = tk.Label(frame, text=f'SPEED LEVEL {self.speed} / 10', bg=self.card_color, fg=self.deep_blue, font=self.font_bold)
        self.speed_label.pack(pady=(0, 4))
        self.speed_slider = RoundedSlider(frame, from_=1, to=10, command=self.speed_changed)
        self.speed_slider.set(self.speed)
        self.speed_slider.pack(fill='x')

    def create_pick_place_panel(self, parent):
        frame = self.create_section_frame(parent, 'PICK & PLACE')
        cols = tk.Frame(frame, bg=self.card_color)
        cols.pack(fill='x')
        left_col = tk.Frame(cols, bg=self.card_color)
        left_col.pack(side='left', fill='x', expand=True, padx=(0, 6))
        right_col = tk.Frame(cols, bg=self.card_color)
        right_col.pack(side='right', fill='x', expand=True, padx=(6, 0))
        tk.Label(left_col, text='PICK POSITION', bg=self.card_color, fg=self.secondary_text, font=self.font_bold, anchor='w').pack(fill='x', pady=(0, 4))
        self.pick_x_entry = self.create_xyz_entry(left_col, 'X')
        self.pick_y_entry = self.create_xyz_entry(left_col, 'Y')
        self.pick_z_entry = self.create_xyz_entry(left_col, 'Z')
        
        btn_set_pick = RoundedButton(left_col, text='SET CURRENT', command=self.set_current_pick, bg_parent=self.card_color, bg=self.glass_color, fg=self.primary_blue, hover_bg=self.soft_blue, border_color=self.sky_blue, height=26, font=("Inter", 9, "bold"))
        btn_set_pick.pack(fill='x', pady=(4, 0))
        
        tk.Label(right_col, text='PLACE POSITION', bg=self.card_color, fg=self.secondary_text, font=self.font_bold, anchor='w').pack(fill='x', pady=(0, 4))
        self.place_x_entry = self.create_xyz_entry(right_col, 'X')
        self.place_y_entry = self.create_xyz_entry(right_col, 'Y')
        self.place_z_entry = self.create_xyz_entry(right_col, 'Z')
        
        btn_set_place = RoundedButton(right_col, text='SET CURRENT', command=self.set_current_place, bg_parent=self.card_color, bg=self.glass_color, fg=self.primary_blue, hover_bg=self.soft_blue, border_color=self.sky_blue, height=26, font=("Inter", 9, "bold"))
        btn_set_place.pack(fill='x', pady=(4, 0))

        action_frame = tk.Frame(frame, bg=self.card_color)
        action_frame.pack(fill='x', pady=(6, 0))
        
        swap_btn = RoundedButton(action_frame, text='⇅ SWAP', command=self.swap_pick_place, bg_parent=self.card_color, bg=self.glass_color, fg=self.primary_blue, hover_bg=self.soft_blue, border_color=self.sky_blue, height=28)
        swap_btn.pack(side='left', fill='x', expand=True, padx=(0, 4))
        
        clear_pp_btn = RoundedButton(action_frame, text='CLEAR', command=self.clear_pick_and_place, bg_parent=self.card_color, bg=self.glass_color, fg=self.primary_blue, hover_bg=self.soft_blue, border_color=self.sky_blue, height=28)
        clear_pp_btn.pack(side='right', fill='x', expand=True, padx=(4, 0))

        self.plan_move_button = RoundedButton(frame, text='⬈  PLAN & MOVE', command=self.start_pick_and_place_wrapped, bg_parent=self.card_color, bg=self.primary_blue, hover_bg=self.sky_blue, fg='white', height=36)
        self.plan_move_button.pack(fill='x', pady=(6, 4))
        self.task_status_label = tk.Label(frame, text='Task Status: IDLE', bg=self.card_color, fg=self.secondary_text, font=self.font_small, anchor='w')
        self.task_status_label.pack(fill='x')
        self.trajectory_status_display = tk.Label(frame, text='IDLE', bg=self.soft_blue, fg=self.primary_blue, font=self.font_bold, pady=4)
        self.trajectory_status_display.pack(fill='x', pady=(3, 0))

    def create_system_panel(self, parent):
        frame = self.create_section_frame(parent, 'SYSTEM CONTROL')
        self.home_button = RoundedButton(frame, text='⌂  HOME', command=self.home_wrapped, bg_parent=self.card_color, bg=self.card_color, border_color=self.sky_blue, fg=self.primary_blue, hover_bg=self.soft_blue, height=34)
        self.home_button.pack(fill='x', pady=3)
        self.resume_button = RoundedButton(frame, text='↺  RESUME MANUAL', command=self.resume_manual, bg_parent=self.card_color, bg=self.card_color, border_color=self.sky_blue, fg=self.primary_blue, hover_bg=self.soft_blue, height=34)
        self.resume_button.pack(fill='x', pady=3)
        self.stop_sys_button = RoundedButton(frame, text='■  STOP', command=self.stop, bg_parent=self.card_color, bg=self.error_color, fg='white', hover_bg="#C53030", height=34)
        self.stop_sys_button.pack(fill='x', pady=(8, 2))

    def create_status_panel(self, parent):
        frame = self.create_section_frame(parent, 'ROBOT 01')
        self.robot_state = tk.Label(frame, text='● READY', font=("Inter", 14, "bold"), fg=self.success_color, bg=self.card_color)
        self.robot_state.pack(pady=(0, 8))
        info_frame = tk.Frame(frame, bg=self.card_color)
        info_frame.pack(fill='x')
        tk.Label(info_frame, text='Control Mode', font=self.font_main, fg=self.secondary_text, bg=self.card_color, anchor='w').grid(row=0, column=0, sticky='w', pady=2)
        self.control_mode_lbl = tk.Label(info_frame, text='MANUAL', font=self.font_bold, fg=self.deep_blue, bg=self.card_color, anchor='e')
        self.control_mode_lbl.grid(row=0, column=1, sticky='e', pady=2)
        info_frame.columnconfigure(1, weight=1)
        tk.Label(info_frame, text='Connection', font=self.font_main, fg=self.secondary_text, bg=self.card_color, anchor='w').grid(row=1, column=0, sticky='w', pady=2)
        self.connection_lbl = tk.Label(info_frame, text='CONNECTED', font=self.font_bold, fg=self.success_color, bg=self.card_color, anchor='e')
        self.connection_lbl.grid(row=1, column=1, sticky='e', pady=2)
        self.status_label = tk.Label(frame, text='Waiting for /control_mode', bg=self.card_color, fg=self.secondary_text, font=self.font_small, wraplength=260, justify='center')
        self.status_label.pack(fill='x', pady=(10, 0))

    def create_status_bar(self):
        self.status_bar = tk.Label(self.root, text='ROS GUI READY', anchor='w', bg=self.bg_color, fg=self.secondary_text, font=self.font_small, padx=20, pady=6)
        self.status_bar.pack(fill='x', side='bottom')



    # ========================================================
    # UTILITY METHODS (PHASE 2.1)
    # ========================================================

    def home_wrapped(self):
        self.tcp_trace_x = []
        self.tcp_trace_y = []
        self.tcp_trace_z = []
        self.home()
    def copy_joints(self):
        txt = f"[{self.joint_angles[0]:.1f}, {self.joint_angles[1]:.1f}, {self.joint_angles[2]:.1f}, {self.joint_angles[3]:.1f}, {self.joint_angles[4]:.1f}]"
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)

    def copy_xyz(self):
        txt = f"{self.ee_position[0]:.2f}, {self.ee_position[1]:.2f}, {self.ee_position[2]:.2f}"
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)

    def set_current_cartesian(self):
        self.target_x_entry.delete(0, tk.END)
        self.target_x_entry.insert(0, f"{self.ee_position[0]:.2f}")
        self.target_y_entry.delete(0, tk.END)
        self.target_y_entry.insert(0, f"{self.ee_position[1]:.2f}")
        self.target_z_entry.delete(0, tk.END)
        self.target_z_entry.insert(0, f"{self.ee_position[2]:.2f}")

    def clear_cartesian(self):
        self.target_x_entry.delete(0, tk.END)
        self.target_y_entry.delete(0, tk.END)
        self.target_z_entry.delete(0, tk.END)

    def set_current_pick(self):
        self.pick_x_entry.delete(0, tk.END)
        self.pick_x_entry.insert(0, f"{self.ee_position[0]:.2f}")
        self.pick_y_entry.delete(0, tk.END)
        self.pick_y_entry.insert(0, f"{self.ee_position[1]:.2f}")
        self.pick_z_entry.delete(0, tk.END)
        self.pick_z_entry.insert(0, f"{self.ee_position[2]:.2f}")

    def clear_pick(self):
        self.pick_x_entry.delete(0, tk.END)
        self.pick_y_entry.delete(0, tk.END)
        self.pick_z_entry.delete(0, tk.END)

    def set_current_place(self):
        self.place_x_entry.delete(0, tk.END)
        self.place_x_entry.insert(0, f"{self.ee_position[0]:.2f}")
        self.place_y_entry.delete(0, tk.END)
        self.place_y_entry.insert(0, f"{self.ee_position[1]:.2f}")
        self.place_z_entry.delete(0, tk.END)
        self.place_z_entry.insert(0, f"{self.ee_position[2]:.2f}")

    def clear_place(self):
        self.place_x_entry.delete(0, tk.END)
        self.place_y_entry.delete(0, tk.END)
        self.place_z_entry.delete(0, tk.END)

    def clear_pick_and_place(self):
        self.clear_pick()
        self.clear_place()

    def swap_pick_place(self):
        px, py, pz = self.pick_x_entry.get(), self.pick_y_entry.get(), self.pick_z_entry.get()
        lx, ly, lz = self.place_x_entry.get(), self.place_y_entry.get(), self.place_z_entry.get()
        self.pick_x_entry.delete(0, tk.END); self.pick_x_entry.insert(0, lx)
        self.pick_y_entry.delete(0, tk.END); self.pick_y_entry.insert(0, ly)
        self.pick_z_entry.delete(0, tk.END); self.pick_z_entry.insert(0, lz)
        self.place_x_entry.delete(0, tk.END); self.place_x_entry.insert(0, px)
        self.place_y_entry.delete(0, tk.END); self.place_y_entry.insert(0, py)
        self.place_z_entry.delete(0, tk.END); self.place_z_entry.insert(0, pz)

    def solve_ik_and_move_wrapped(self):
        self.tcp_trace_x = []
        self.tcp_trace_y = []
        self.tcp_trace_z = []
        self.solve_ik_and_move()

    def start_pick_and_place_wrapped(self):
        self.tcp_trace_x = []
        self.tcp_trace_y = []
        self.tcp_trace_z = []
        self.start_pick_and_place()

    # ========================================================
    # 3D VISUALIZER SETUP & LOGIC
    # ========================================================
    def setup_3d_visualizer(self, parent):
        self.fig = plt.figure(facecolor=self.glass_color)
        self.fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor(self.glass_color)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.update_visualization()

    def calculate_joint_positions(self):
        L1 = 153.0
        L2 = 105.0
        L3 = 98.0
        L4 = 17.5
        L5 = 48.0
        DX = 0.0
        DY = -12.0
        DZ = 0.0

        q = self.joint_angles
        theta1 = np.radians(q[0])
        theta2 = np.radians(q[1])
        theta3 = np.radians(q[2])
        theta4 = np.radians(q[3])
        theta5 = np.radians(q[4])
        theta23 = theta2 + theta3

        P0 = np.array([0.0, 0.0, 0.0])
        P1 = np.array([0.0, 0.0, L1])
        P2 = np.array([L2 * np.sin(theta2) * np.cos(theta1),
                       L2 * np.sin(theta2) * np.sin(theta1),
                       L1 + L2 * np.cos(theta2)])
        P3 = np.array([(L2 * np.sin(theta2) + L3 * np.sin(theta23)) * np.cos(theta1),
                       (L2 * np.sin(theta2) + L3 * np.sin(theta23)) * np.sin(theta1),
                       L1 + L2 * np.cos(theta2) + L3 * np.cos(theta23)])
        P4 = np.array([(L2 * np.sin(theta2) + (L3 + L4) * np.sin(theta23)) * np.cos(theta1),
                       (L2 * np.sin(theta2) + (L3 + L4) * np.sin(theta23)) * np.sin(theta1),
                       L1 + L2 * np.cos(theta2) + (L3 + L4) * np.cos(theta23)])

        radial = L2 * np.sin(theta2) + (L3 + L4) * np.sin(theta23) + L5 * (np.cos(theta5) * np.sin(theta23) + np.sin(theta5) * np.cos(theta4) * np.cos(theta23))
        tcp_x = radial * np.cos(theta1) - L5 * np.sin(theta5) * np.sin(theta4) * np.sin(theta1)
        tcp_y = radial * np.sin(theta1) + L5 * np.sin(theta5) * np.sin(theta4) * np.cos(theta1)
        tcp_z = L1 + L2 * np.cos(theta2) + (L3 + L4) * np.cos(theta23) + L5 * (np.cos(theta5) * np.cos(theta23) - np.sin(theta5) * np.cos(theta4) * np.sin(theta23))

        PTCP = np.array([tcp_x + DX, tcp_y + DY, tcp_z + DZ])
        return np.array([P0, P1, P2, P3, P4, PTCP])

    def update_visualization(self):
        if not hasattr(self, 'ax'): return
        points = self.calculate_joint_positions()
        PTCP_X = points[5, 0]
        PTCP_Y = points[5, 1]
        PTCP_Z = points[5, 2]
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]

        self.ax.clear()
        self.ax.plot(X, Y, Z, '-o', linewidth=3, markersize=8, color=self.primary_blue)
        
        if hasattr(self, 'trace_enabled') and self.trace_enabled.get() and hasattr(self, 'tcp_trace_x') and len(self.tcp_trace_x) > 1:
            self.ax.plot(self.tcp_trace_x, self.tcp_trace_y, self.tcp_trace_z, '-', color=self.sky_blue, linewidth=2, alpha=0.7)
        
        # Customize background to match glass theme
        self.ax.set_facecolor(self.glass_color)
        self.fig.patch.set_facecolor(self.glass_color)
        
        # Grid lines color
        self.ax.xaxis._axinfo["grid"].update({"color": self.panel_border})
        self.ax.yaxis._axinfo["grid"].update({"color": self.panel_border})
        self.ax.zaxis._axinfo["grid"].update({"color": self.panel_border})
        
        # Pane colors
        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        labels = ['Base', 'J1', 'J2', 'J3', 'J4', 'TCP']
        for i in range(len(points)):
            self.ax.text(X[i], Y[i], Z[i], labels[i], fontsize=9, color=self.primary_text)

        self.ax.set_xlabel('X (mm)', color=self.primary_text)
        self.ax.set_ylabel('Y (mm)', color=self.primary_text)
        self.ax.set_zlabel('Z (mm)', color=self.primary_text)
        self.ax.tick_params(colors=self.secondary_text)

        max_reach = 105.0 + 98.0 + 17.5 + 48.0 + 50
        self.ax.set_xlim(-max_reach, max_reach)
        self.ax.set_ylim(-max_reach, max_reach)
        self.ax.set_zlim(-50, 153.0 + max_reach)
        self.ax.set_box_aspect([1, 1, 1])
        
        # Trigger redraw
        self.canvas.draw_idle()


    # ========================================================
    # CONTROL MODE CALLBACK

    # ========================================================
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
        
        # Also update the 3D visualizer natively
        if hasattr(self, 'update_visualization'):
            self.update_visualization()


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

        if not hasattr(self, 'tcp_trace_x'):
            self.tcp_trace_x = []
            self.tcp_trace_y = []
            self.tcp_trace_z = []

        # Only trace if a motion is likely happening, but safely append
        if len(self.tcp_trace_x) == 0 or (abs(self.tcp_trace_x[-1] - self.ee_position[0]) > 0.1 or abs(self.tcp_trace_y[-1] - self.ee_position[1]) > 0.1 or abs(self.tcp_trace_z[-1] - self.ee_position[2]) > 0.1):
            self.tcp_trace_x.append(self.ee_position[0])
            self.tcp_trace_y.append(self.ee_position[1])
            self.tcp_trace_z.append(self.ee_position[2])


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
