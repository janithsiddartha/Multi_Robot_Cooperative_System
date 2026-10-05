"""ROS 2 node for computing and reporting static handover poses."""

import os
import rclpy
from rclpy.node import Node
import yaml

from robot_control.handover.static_map import StaticHandoverMap
from robot_control.nodes.handover_planner_node import YamlKinematicsConfig

class HandoverPoseNode(Node):
    """Compute and report cached handover poses for robot pairs."""

    def __init__(self):
        super().__init__("handover_pose_node")

        config_dir = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "config",
            )
        )

        if not os.path.isdir(config_dir):
            config_dir = "/home/iit/rt_test/src/robot_control/config"

        handover_yaml = os.path.join(config_dir, "handover.yaml")
        robot_yaml_paths = {
            rid: os.path.join(config_dir, f"robot{rid}.yaml")
            for rid in (1, 2, 3)
        }

        with open(handover_yaml, "r", encoding="utf-8") as handle:
            handover_config = yaml.safe_load(handle) or {}

        robots = {
            rid: YamlKinematicsConfig(robot_yaml_paths[rid])
            for rid in (1, 2, 3)
        }

        self.static_map = StaticHandoverMap(handover_config, robots)
        
        self.get_logger().info("Generating static handover map...")
        self.static_map.load_or_generate(force_rebuild=True)
        
        json_path = self.static_map.save_json_report()

        print("\n" + "="*50)
        print("STATIC HANDOVER POSE MAP")
        print("="*50 + "\n")
        
        for a, b in ((1, 2), (1, 3), (2, 3)):
            result = self.static_map.pair_results[(a, b)]
            if result is None:
                print(f"Robot {a} <-> Robot {b} : NULL")
            else:
                print(f"Robot {a} <-> Robot {b} : VALID")
                
        print("\nJSON:")
        print(json_path + "\n")
        
        for a, b in ((1, 2), (1, 3), (2, 3)):
            result = self.static_map.pair_results[(a, b)]
            if result is not None:
                p = result["position"]
                q_a = result["robot_a_joints"]
                q_b = result["robot_b_joints"]
                print(f"R{a}-R{b}: H = [{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}], branch = {result['branch']}")
                print(f"R{a} q = [{', '.join(f'{x:.4f}' for x in q_a)}]")
                print(f"R{b} q = [{', '.join(f'{x:.4f}' for x in q_b)}]\n")


def main(args=None):
    rclpy.init(args=args)
    node = HandoverPoseNode()
    
    # We do not need to spin since it's an offline/precomputation node.
    # It just generates the cache and reports.
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
