#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import serial.tools.list_ports

class UARTNode(Node):
    def __init__(self):
        super().__init__('uart_node')
        
        # We only need baudrate from yaml. Port defaults to None.
        self.declare_parameter('uart.baudrate', 115200)
        self.baudrate = self.get_parameter('uart.baudrate').value
        self.ser = None
        self.port = None

        # Subscribe to raw string commands and port selection commands
        self.create_subscription(String, 'serial_tx', self.serial_tx_callback, 10)
        self.create_subscription(String, 'robot_command', self.system_callback, 10)

        # Publisher for GUI
        self.ports_pub = self.create_publisher(String, 'available_ports', 10)
        self.ports_timer = self.create_timer(0.5, self.publish_ports)

        self.get_logger().info('UART DUMB RELAY NODE READY')
        self.get_logger().info('Standing by. Waiting for port selection from GUI...')

    def serial_tx_callback(self, msg):
        command = msg.data.strip()
        self.send_command(command)

    def system_callback(self, msg):
        raw_command = msg.data.strip()
        if raw_command.upper().startswith('PORT,'):
            new_port = raw_command.split(',', 1)[1].strip()
            self.get_logger().info(f'Switching serial port to: {new_port}')
            
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
                self.get_logger().info('Closed old serial port.')

            self.port = new_port
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1, write_timeout=0.1)
                self.get_logger().info(f'Connected to ESP32: {self.port} @ {self.baudrate}')
            except serial.SerialException as e:
                self.ser = None
                self.get_logger().error(f'Failed to open serial port: {e}')

    def send_command(self, command):
        if self.ser is None or not self.ser.is_open:
            return False

        try:
            self.ser.write((command + '\n').encode())
            self.ser.flush()
            return True
        except serial.SerialException as e:
            self.get_logger().error(f'Serial write failed: {e}')
            return False

    def publish_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports() if 'USB' in p.device or 'ACM' in p.device or 'COM' in p.device]
        msg = String()
        msg.data = ','.join(ports) if ports else 'NONE'
        self.ports_pub.publish(msg)

    def destroy_node(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main(args=None):
    # This node no longer requires --params-file because it uses a default baudrate
    # and no longer needs joint/kinematics parameters!
    rclpy.init(args=args)
    node = UARTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
