#!/usr/bin/env python3

import os
import math
import json
import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String

from collections import deque
from datetime import datetime

"""
'High Alt Filter' node that:
- Subscribes to camera/thermal_image + /combined_odometry
- Queues last N=25 odometry messages
- Has a timer at 3 Hz that checks for the 'latest image', 
  finds best-match odometry, and stores them in lists.
- On shutdown (KeyboardInterrupt), saves images (numpy) + odometry (csv).
"""

class HighAltFilterNode(Node):
    def __init__(self, odom_queue_size=25, freq_hz=3.0):
        super().__init__("high_alt_filter_node")
        self.get_logger().info("HighAltFilterNode started.")

        # Odom queue
        self.odom_history = deque(maxlen=odom_queue_size)

        # For saving matched data
        self.saved_images = []   # list of 2D NumPy arrays
        self.saved_odom   = []   # list of dictionaries

        # Latest camera msg (None if not arrived yet)
        self.latest_img_msg = None

        # Subscribe to camera
        self.img_sub = self.create_subscription(
            Image,
            "camera/thermal_image",
            self.img_callback,
            10
        )

        # Subscribe to odom
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        self.odom_sub = self.create_subscription(
            String,
            "/combined_odometry",
            self.odom_callback,
            qos_profile
        )

        # Timer at freq_hz => gather data
        self.timer_period = 1.0 / freq_hz
        self.timer_ = self.create_timer(self.timer_period, self.timer_callback)

    def img_callback(self, msg):
        """Just store the latest image msg."""
        self.latest_img_msg = msg

    def odom_callback(self, msg):
        """Parse JSON, store in queue."""
        try:
            data = json.loads(msg.data)
            self.odom_history.append(data)  # store raw dict
        except Exception as e:
            self.get_logger().error(f"Failed to parse odom: {e}")

    def timer_callback(self):
        """At 3 Hz, if we have a new image, pair it with the best odometry, store it."""
        if self.latest_img_msg is None:
            return  # no camera yet

        # Convert the image to scaled [0..100]
        msg = self.latest_img_msg
        try:
            raw_16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        except ValueError as e:
            self.get_logger().error(f"Cannot reshape image: {e}")
            return

        scaled_img = (raw_16.astype(float) / 65535.0) * 100.0

        # Find best odom
        img_time_us = (msg.header.stamp.sec * 1_000_000) + (msg.header.stamp.nanosec // 1000)
        best_odom = self.find_closest_odom(img_time_us)
        if best_odom is None:
            # no odom => skip
            self.get_logger().info("No odom match found for this image. Skipping.")
            return

        # store them
        self.saved_images.append(scaled_img)
        self.saved_odom.append(best_odom)

        # Optionally log something
        self.get_logger().info(f"Stored 1 image & matching odom. Now have {len(self.saved_images)} samples.")

        # Mark the image as consumed if you want to skip re-using it 
        # (otherwise we'll store it again next cycle).
        self.latest_img_msg = None

    def find_closest_odom(self, img_time_us):
        """Find odom in self.odom_history with closest 'timestamp' field to img_time_us."""
        if not self.odom_history:
            return None

        best = None
        best_diff = float('inf')
        for odom in self.odom_history:
            if "timestamp" not in odom:
                continue
            diff = abs(odom["timestamp"] - img_time_us)
            if diff < best_diff:
                best_diff = diff
                best = odom
        return best

    def save_and_exit(self):
        """On shutdown, save images & odom to disk."""
        # Save images
        arr = np.array(self.saved_images, dtype=np.float32)  # shape = (N, H, W)
        img_filename = f"flight_logs/high_alt_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
        np.save(img_filename, arr)
        self.get_logger().info(f"Saved {len(self.saved_images)} images to {img_filename}")

        # Save odom
        # convert to DataFrame
        df = pd.DataFrame(self.saved_odom)
        odom_filename = f"flight_logs/flight_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
        np.save(odom_filename, df.values)
        self.get_logger().info(f"Saved {len(self.saved_odom)} odometry entries to {odom_filename}")

def main(args=None):
    rclpy.init(args=args)
    node = HighAltFilterNode(odom_queue_size=25, freq_hz=3.0)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_and_exit()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
