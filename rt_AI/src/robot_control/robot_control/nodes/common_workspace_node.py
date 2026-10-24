"""ROS 2 node for computing and reporting common position workspaces."""

import os

import rclpy
from rclpy.node import Node
import yaml

from robot_control.handover.workspace import PositionWorkspaceComputer
from robot_control.nodes.handover_planner_node import YamlKinematicsConfig


class CommonWorkspaceNode(Node):
    """Compute and report cached robot position workspaces."""

    def __init__(self):
        super().__init__("common_workspace_node")

        config_dir = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                "..",
                "..",
                "config",
            )
        )

        # Source-tree fallback for the current BTP workspace.
        if not os.path.isdir(config_dir):
            config_dir = (
                "/home/iit/rt_test/src/robot_control/config"
            )

        handover_yaml = os.path.join(
            config_dir,
            "handover.yaml",
        )

        robot_yaml_paths = {
            robot_id: os.path.join(
                config_dir,
                f"robot{robot_id}.yaml",
            )
            for robot_id in (1, 2, 3)
        }

        # --------------------------------------------------------
        # Load common handover configuration
        # --------------------------------------------------------

        with open(
            handover_yaml,
            "r",
            encoding="utf-8",
        ) as handle:
            handover_config = (
                yaml.safe_load(handle)
                or {}
            )

        # --------------------------------------------------------
        # Load actual robot kinematic configurations
        # --------------------------------------------------------

        robots = {
            robot_id: YamlKinematicsConfig(
                robot_yaml_paths[robot_id]
            )
            for robot_id in (1, 2, 3)
        }

        # --------------------------------------------------------
        # Compute/load actual position workspaces
        # --------------------------------------------------------

        self.workspace = PositionWorkspaceComputer(
            handover_config=handover_config,
            robots=robots,
            handover_yaml_path=handover_yaml,
            robot_yaml_paths=robot_yaml_paths,
        )

        self.workspace.load_or_compute()

        self.get_logger().info(
            "\n" + self.workspace.report()
        )


def main(args=None):
    """Run the common position-workspace node."""
    rclpy.init(args=args)

    node = CommonWorkspaceNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        # Avoid calling shutdown twice if ROS has already shut down.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()