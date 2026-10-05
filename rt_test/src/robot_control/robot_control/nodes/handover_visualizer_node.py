import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import matplotlib.pyplot as plt
import yaml

from robot_control.kinematics.fk_math import all_frame_positions
from robot_control.handover.planner import euler_to_matrix
from robot_control.nodes.handover_planner_node import YamlKinematicsConfig

class MultiRobotVisualizer(Node):
    def __init__(self):
        super().__init__('multi_robot_visualizer_node')
        
        # Load configs
        handover_yaml = '/home/iit/rt_test/src/robot_control/config/handover.yaml'
        with open(handover_yaml, 'r') as f:
            self.handover_config = yaml.safe_load(f)
            
        self.robots = {}
        for i in [1, 2, 3]:
            path = f'/home/iit/rt_test/src/robot_control/config/robot{i}.yaml'
            self.robots[i] = YamlKinematicsConfig(path)
            
        self.q_deg = {
            1: np.zeros(self.robots[1].num_joints),
            2: np.zeros(self.robots[2].num_joints),
            3: np.zeros(self.robots[3].num_joints)
        }
        
        self.subs = []
        for i in [1, 2, 3]:
            sub = self.create_subscription(
                Float64MultiArray,
                f'/robot{i}/joint_commands',
                lambda msg, rid=i: self.joint_callback(msg, rid),
                10
            )
            self.subs.append(sub)

        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.manager.set_window_title('Multi-Robot Visualizer')
        plt.show(block=False)
        self.timer = self.create_timer(0.05, self.update_visualization)
        self.get_logger().info('Multi-Robot Visualizer Started')

    def joint_callback(self, msg, robot_id):
        if len(msg.data) == self.robots[robot_id].num_joints:
            self.q_deg[robot_id] = np.array(msg.data, dtype=float)

    def get_world_to_base(self, robot_id):
        base_info = self.handover_config.get('robots', {}).get(f'robot{robot_id}', {})
        xyz = base_info.get('world_base_xyz', [0,0,0])
        rpy = np.deg2rad(base_info.get('world_base_rpy_deg', [0,0,0]))
        R = euler_to_matrix(rpy)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = xyz
        return T

    def update_visualization(self):
        self.ax.clear()
        
        colors = {1: 'b', 2: 'r', 3: 'g'}
        
        for i in [1, 2, 3]:
            q_rad = np.deg2rad(self.q_deg[i])
            points_base = all_frame_positions(q_rad, self.robots[i])
            
            T_W_B = self.get_world_to_base(i)
            R_W_B = T_W_B[:3, :3]
            p_W_B = T_W_B[:3, 3]
            
            points_world = (R_W_B @ points_base.T).T + p_W_B
            
            X = points_world[:, 0]
            Y = points_world[:, 1]
            Z = points_world[:, 2]
            
            self.ax.plot(X, Y, Z, '-o', linewidth=3, markersize=6, color=colors[i], label=f'Robot {i}')
            self.ax.text(X[-1], Y[-1], Z[-1], f'TCP {i}', fontsize=8)

        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        
        max_reach = 800
        self.ax.set_xlim(-max_reach, max_reach)
        self.ax.set_ylim(-max_reach, max_reach)
        self.ax.set_zlim(0, max_reach)
        self.ax.set_box_aspect([1, 1, 0.5])
        
        self.ax.legend()
        self.ax.grid(True)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
