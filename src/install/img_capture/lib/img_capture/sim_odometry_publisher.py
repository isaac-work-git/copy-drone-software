#!/usr/bin/env python3

from custom_msgs.msg import CustomOdometryMsg
import rclpy
from rclpy.node import Node
import numpy as np
import os

class SimulatedOdometryPublisher(Node):
    def __init__(self):
        super().__init__('simulated_odometry_publisher')
        self.publisher_ = self.create_publisher(CustomOdometryMsg, '/simulated_odometry', 10)
        self.timer = self.create_timer(1.0, self.publish_odometry)
        self.data = self.load_npy_data()
        self.current_index = 0

    def load_npy_data(self):
        file_path = self.declare_parameter('npy_file_path', '/home/username/ros2_ws/sim_data/clean_odometry.npy').value
        if not file_path:
            self.get_logger().error('NPY file path not specified')
            return None
        if not os.path.exists(file_path):
            self.get_logger().error(f'NPY file not found: {file_path}')
            return None
        return np.load(file_path)

    def publish_odometry(self):
        if self.data is None or self.current_index >= len(self.data):
            return

        pose_data = self.data[self.current_index]
        msg = CustomOdometryMsg(
            timestamp=float(pose_data[0]),
            latitude=float(pose_data[1]),
            longitude=float(pose_data[2]),
            altitude=float(pose_data[3]),
            pitch=float(pose_data[4]),
            roll=float(pose_data[5]),
            yaw=float(pose_data[6])
        )

        self.publisher_.publish(msg)
        
        self.get_logger().info(f'Published custom pose data: '
                               f'Timestamp: {msg.timestamp}, '
                               f'Position: ({msg.longitude}, {msg.latitude}, {msg.altitude}), '
                               f'Orientation: Pitch={msg.pitch}, Roll={msg.roll}, Yaw={msg.yaw}')

        self.current_index += 1

def main(args=None):
    rclpy.init(args=args)
    node = SimulatedOdometryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
