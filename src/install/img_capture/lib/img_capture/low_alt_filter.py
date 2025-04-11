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
from scipy.ndimage import gaussian_filter

# -------------------------
# Filtering function
# -------------------------
def filter_image(img, first_threshold, filter_boost, filter_sigma, second_threshold, camera_dims):
    # Make a float copy
    filtered_img = img.astype(float)
    # 1) Apply first threshold
    filtered_img[filtered_img < first_threshold] = 0
    # 2) Add boost to non-zero
    mask_nonzero = (filtered_img > 0)
    filtered_img[mask_nonzero] += filter_boost
    # 3) Gaussian blur
    filtered_img = gaussian_filter(filtered_img, sigma=filter_sigma)
    # 4) Second threshold
    filtered_img[filtered_img < second_threshold] = 0
    # 5) Make a visual copy
    visual_img = filtered_img.copy()
    visual_img[visual_img > 0] = 255
    # 6) Locate hot spots
    hot_spots = []
    weights = []
    black_out_edge = int(np.ceil(2 * filter_sigma))

    # Make a separate array for “peak finding”
    search_img = filtered_img.copy()

    while search_img.max() > 0:
        # find the maximum
        hot_spot = np.unravel_index(np.argmax(search_img), search_img.shape)
        # sum of the region near that hotspot
        lower_x = max(0, hot_spot[0] - black_out_edge)
        upper_x = min(camera_dims[0], hot_spot[0] + black_out_edge + 1)
        lower_y = max(0, hot_spot[1] - black_out_edge)
        upper_y = min(camera_dims[1], hot_spot[1] + black_out_edge + 1)

        weights.append(np.sum(search_img[lower_x:upper_x, lower_y:upper_y]))
        # zero out that area
        search_img[lower_x:upper_x, lower_y:upper_y] = 0
        hot_spots.append(hot_spot)

    return visual_img.astype(np.uint8), np.array(hot_spots).astype(int), np.array(weights)



class LowAltFilterNode(Node):
    def __init__(self, odom_queue_size=25, freq_hz=3.0):
        super().__init__("low_alt_filter_node")
        self.get_logger().info("LowAltFilterNode started.")

        ## data taken from topics ##
        self.odom_history = deque(maxlen=odom_queue_size)
        self.saved_images = []
        self.saved_odom = []
        self.latest_img_msg = None
        self.filtered_img = None
        self.hotspots = None
        self.weights = None

        ## Camera Intrinsic Characteristics ##
        self.fov_x = 56
        self.fov_y = 45
        self.alt = 100
        self.camera_dims = None
        self.first_threshold = 100
        self.filter_boost = 100
        self.filter_sigma = 2
        self.second_threshold = 35

        ## Subscribers ##
        self.img_sub = self.create_subscription(
            Image, "camera/thermal_image", self.img_callback, 10
        )

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, depth=10
        )
        self.odom_sub = self.create_subscription(
            String, "/combined_odometry", self.odom_callback, qos_profile
        )

        self.timer_period = 1.0 / freq_hz
        self.timer_ = self.create_timer(self.timer_period, self.timer_callback)

        ## Publishers ##
        global hotspot_pub
        hotspot_pub = self.create_publisher(String, 'hotspot_info', 10)


    def img_callback(self, msg):
        self.latest_img_msg = msg
        # Get filtered img and hot spots
        self.filtered_image, self.hot_spots, self.weights = filter_image(msg, self.first_threshold, self.filter_boost, self.filter_sigma, self.second_threshold)

    def odom_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.odom_history.append(data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse odom: {e}")

    def timer_callback(self):
        if self.latest_img_msg is None:
            return

        msg = self.latest_img_msg
        try:
            raw_16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        except ValueError as e:
            self.get_logger().error(f"Cannot reshape image: {e}")
            return

        scaled_img = (raw_16.astype(float) / 65535.0) * 100.0

        hotspot_y, hotspot_x = np.unravel_index(np.argmax(scaled_img), scaled_img.shape)
        center_x, center_y = msg.width // 2, msg.height // 2
        pixel_offset_x, pixel_offset_y = hotspot_x - center_x, hotspot_y - center_y

        img_time_us = (msg.header.stamp.sec * 1_000_000) + (msg.header.stamp.nanosec // 1000)
        best_odom = self.find_closest_odom(img_time_us)
        if best_odom is None:
            self.get_logger().info("No odom match found for this image. Skipping.")
            return

        self.altitude = 100 #best_odom.get("altitude", 100)
        self.camera_dims = scaled_img.shape
        self.calculate_hotspot_location(self.hotspots, scaled_img.shape)
        # ## take odom data and locate hotspots
        altitude = 100 #best_odom.get("altitude", 100)
        # pitch = math.radians(best_odom.get("pitch", 0))
        # roll = math.radians(best_odom.get("roll", 0))
        yaw = math.radians(best_odom.get("yaw", 0))

        # # Convert FOV to radians
        # fov_x_rad = np.radians(self.fov_x)
        # fov_y_rad = np.radians(self.fov_y)
        # img_width = scaled_img.shape[0]
        # img_height = scaled_img.shape[1]

        # # Calculate angular offsets
        # theta_x = ((hotspot_x - (img_width / 2)) / img_width) * fov_x_rad
        # theta_y = ((hotspot_y - (img_height / 2)) / img_height) * fov_y_rad
        
        meters_per_pixel = self.altitude / 100.0  # Example: Adjust based on actual camera specs
        offset_x_meters = pixel_offset_x * meters_per_pixel
        offset_y_meters = pixel_offset_y * meters_per_pixel

        adjusted_x = offset_x_meters * math.cos(yaw) - offset_y_meters * math.sin(yaw)
        adjusted_y = offset_x_meters * math.sin(yaw) + offset_y_meters * math.cos(yaw)

        self.get_logger().info(
            f"Hotspot offset (m): X={adjusted_x:.2f}, Y={adjusted_y:.2f}, Alt={altitude:.2f}"
        )
        ##

        self.saved_images.append(scaled_img)
        self.saved_odom.append(best_odom)
        self.latest_img_msg = None

    def find_closest_odom(self, img_time_us):
        if not self.odom_history:
            return None

        best, best_diff = None, float('inf')
        for odom in self.odom_history:
            if "timestamp" not in odom:
                continue
            diff = abs(odom["timestamp"] - img_time_us)
            if diff < best_diff:
                best_diff = diff
                best = odom
        return best
    
    def calculate_hotspot_distance(x, y, image_width, image_height, fov_x, fov_y, altitude):
        # Convert FOV to radians
        fov_x_rad = np.radians(fov_x)
        fov_y_rad = np.radians(fov_y)
    
        # Calculate angular offsets
        theta_x = ((x - (image_width / 2)) / image_width) * fov_x_rad
        theta_y = ((y - (image_height / 2)) / image_height) * fov_y_rad
    
        # Calculate ground distances
        dx = altitude * np.tan(theta_x)
        dy = altitude * np.tan(theta_y)
    
        # Euclidean distance from image center
        distance = np.sqrt(dx**2 + dy**2)
        return distance
    
    def calculate_hotspot_location(hot_spots, camera_dims, self):
        if len(hot_spots) > 0:
            # Find the hottest hotspot
            hottest_spot = hot_spots[0]  # The first hotspot is the hottest due to the filtering logic
            x, y = hottest_spot

            # Center of the image
            center_x, center_y = camera_dims[0] // 2, camera_dims[1] // 2

            # Calculate distances from the center of the image in meters
            horizontal_distance = self.calculate_hotspot_distance(
                x, center_y, camera_dims[0], camera_dims[1], self.fov_x, self.fov_y, self.altitude
            )
            vertical_distance = self.calculate_hotspot_distance(
                center_x, y, camera_dims[0], camera_dims[1], self.fov_x, self.fov_y, self.altitude
            )

            # Determine vertical direction
            if y > center_y:
                vertical_dir = f"{vertical_distance:.2f} meters down"
            elif y < center_y:
                vertical_dir = f"{vertical_distance:.2f} meters up"
            else:
                vertical_dir = "centered vertically"

            # Determine horizontal direction
            if x > center_x:
                horizontal_dir = f"{horizontal_distance:.2f} meters right"
            elif x < center_x:
                horizontal_dir = f"{horizontal_distance:.2f} meters left"
            else:
                horizontal_dir = "centered horizontally"

            # Create the message
            message = (
                f"Hottest Hotspot: Coordinates=({x}, {y}), "
                f"Vertical Offset={vertical_dir}, "
                f"Horizontal Offset={horizontal_dir}"
            )
            print(message)  # Print to console

            # Publish the message to the ROS topic
            hotspot_pub.publish(String(data=message))
        else:
            print("No hotspot detected.")


    def save_and_exit(self):
        arr = np.array(self.saved_images, dtype=np.float32)
        img_filename = f"low_alt_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
        np.save(img_filename, arr)
        self.get_logger().info(f"Saved {len(self.saved_images)} images to {img_filename}")

        df = pd.DataFrame(self.saved_odom)
        odom_filename = f"low_alt_odom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(odom_filename, index=False)
        self.get_logger().info(f"Saved {len(self.saved_odom)} odometry entries to {odom_filename}")


def main(args=None):
    rclpy.init(args=args)
    node = LowAltFilterNode(odom_queue_size=25, freq_hz=3.0)

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