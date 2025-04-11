#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from px4_msgs.msg import VehicleAttitude, SensorGps
from std_msgs.msg import String
import math
import json
import pandas as pd
from datetime import datetime

class CustomPoseMsg:
    def __init__(self, timestamp, latitude, longitude, altitude, pitch, roll, yaw):
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw

class CombinedOdometryPublisher(Node):
    def __init__(self):
        super().__init__('combined_odometry_publisher')

        # Initialize variables
        self.roll = self.pitch = self.yaw = 0.0
        self.latitude = self.longitude = self.altitude = 0.0
        self.timestamp = 0

        # Set up QoS profile for subscriptions (BEST_EFFORT for PX4 topics)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # Create subscriptions
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.attitude_callback, qos_profile)
        self.create_subscription(SensorGps, '/fmu/out/vehicle_gps_position', self.gps_position_callback, qos_profile)

        # Create publisher for combined odometry data
        self.publisher = self.create_publisher(String, '/combined_odometry', 10)

        # Timer for periodic publishing at 20 Hz (50 ms interval)
        self.timer = self.create_timer(1.0 / 20.0, self.timer_callback)

    def timer_callback(self):
        """Timer callback to publish combined odometry data."""
        self.publish_combined_odometry()

    def attitude_callback(self, msg):
        """Callback for VehicleAttitude messages to extract roll, pitch, yaw."""
        w, x, y, z = msg.q  # Quaternion components from VehicleAttitude message
        self.roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        self.pitch = math.asin(2 * (w * y - z * x))
        self.yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def gps_position_callback(self, msg):
        """Callback for SensorGps messages to extract GPS data."""
        self.latitude = msg.latitude_deg
        self.longitude = msg.longitude_deg
        self.altitude = msg.altitude_msl_m
        self.timestamp = msg.timestamp

    def publish_combined_odometry(self):
        """Publish combined odometry data as a JSON string."""
        custom_pose = CustomPoseMsg(
            timestamp=self.timestamp,
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.altitude,
            pitch=self.pitch,
            roll=self.roll,
            yaw=self.yaw
        )

        # Convert CustomPoseMsg to JSON string and publish it
        json_str = json.dumps(custom_pose.__dict__)
        msg = String()
        msg.data = json_str

        # Publish the message on the /combined_odometry topic
        self.publisher.publish(msg)

        # Log
        self.get_logger().info(f"timestamp: {self.timestamp}, latitude: {self.latitude}, longitude: {self.longitude}, altitude: {self.altitude}, pitch: {self.pitch}, roll: {self.roll}, yaw: {self.yaw}")

def main(args=None):
    rclpy.init(args=args)
    node = CombinedOdometryPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
