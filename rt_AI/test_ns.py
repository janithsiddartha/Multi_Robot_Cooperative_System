import rclpy
from rclpy.node import Node
import sys
import os
import time

def main():
    rclpy.init(args=sys.argv)
    node = Node('test_node')
    print("Namespace is:", node.get_namespace())
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()
