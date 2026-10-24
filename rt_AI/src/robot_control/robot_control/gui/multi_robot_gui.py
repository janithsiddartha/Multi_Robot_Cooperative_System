#!/usr/bin/env python3

import subprocess
import sys
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
import tkinter as tk
from tkinter import ttk


ROBOTS = [
    'robot1',
    'robot2',
    'robot3',
]


class MultiRobotTabGUI(Node):
    def __init__(self, root):
        super().__init__('multi_robot_gui')
        self.root = root
        self.status_vars = {}

        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        for robot in ROBOTS:
            self.status_vars[robot] = tk.StringVar(value='Waiting for /control_mode...')
            self.create_subscription(
                String,
                f'/{robot}/control_mode',
                lambda msg, robot_name=robot: self.control_mode_callback(msg, robot_name),
                self.qos,
            )

        self.build_ui()

    def control_mode_callback(self, msg, robot_name):
        self.status_vars[robot_name].set(msg.data)

    def open_robot_gui(self, robot_name):
        repo_root = Path(__file__).resolve().parents[4]
        gui_script = repo_root / 'gui' / f'robot_gui_test_{robot_name[-1]}.py'
        if not gui_script.exists():
            gui_script = repo_root / 'gui' / 'robot_gui_test.py'
        subprocess.Popen([sys.executable, str(gui_script)])

    def build_ui(self):
        self.root.title('Multi-Robot Control')
        self.root.geometry('700x360')
        self.root.minsize(650, 300)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=16, pady=16)

        for robot in ROBOTS:
            frame = ttk.Frame(notebook, padding=18)
            notebook.add(frame, text=robot.replace('robot', 'Robot '))

            ttk.Label(frame, text=robot, font=('Arial', 18, 'bold')).pack(anchor='w', pady=(0, 10))
            ttk.Label(frame, text='Control mode:').pack(anchor='w')
            ttk.Label(frame, textvariable=self.status_vars[robot], font=('Arial', 12, 'bold')).pack(anchor='w', pady=(0, 16))

            ttk.Button(
                frame,
                text=f'Open {robot} GUI',
                command=lambda robot_name=robot: self.open_robot_gui(robot_name),
            ).pack(anchor='w', pady=4)

            ttk.Label(
                frame,
                text=f'Namespace: /{robot}',
                foreground='#555555',
            ).pack(anchor='w', pady=(12, 0))


def main(args=None):
    rclpy.init(args=args)
    root = tk.Tk()
    gui = MultiRobotTabGUI(root)

    try:
        thread = threading.Thread(target=lambda: rclpy.spin(gui), daemon=True)
        thread.start()
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        gui.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
