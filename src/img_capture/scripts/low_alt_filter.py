#!/usr/bin/env python3

## module filtering for hotspots and 
# determines how far left/right, forward/back hotspot is from plane

## comparing home location to world coordinates. 
## sending hotspot location data to topic
## drop_mech is recieving the data
## TODO: Working on saving the hotspot_location sent from the user


import os
import math
import json
import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import String

from collections import deque
from datetime import datetime
from scipy.ndimage import gaussian_filter

from tf_transformations import euler_from_quaternion

## need to install tf-transformations


# Setting up json string message for location
class CustomPoseMsg:
    def __init__(self, x, y, alt):
        self.x = x
        self.y = y
        self.alt = alt


class LowAltFilterNode(Node):
    def __init__(self, odom_queue_size=25, freq_hz=3.0):
        super().__init__("low_alt_filter_node")
        self.get_logger().info("LowAltFilterNode started.")

        ## data taken from topics ##
        self.odom_history = deque(maxlen=odom_queue_size)
        self.saved_images = []
        self.saved_odom = []
        self.latest_img_msg = None

        ## data extracted from topics
        self.home_location = None
        self.home_altitude = 1300
        self.sim = False
        self.pitch = 0
        self.roll  = 0
        self.yaw   = 0
        self.altitude = 0

        ## data sent from User
        # self.hot_location = None
        self.user_hot_lat = None
        self.user_hot_long = None
        # self.user_hot_alt = None

        ## Camera Intrinsic Characteristics ##
        self.fov_x = 56
        self.fov_y = 42
        # self.camera_dims = np.array([200, 320])
        # self.first_threshold = 100
        # self.filter_boost = 100
        self.filter_sigma = 2
        # self.second_threshold = 35

        ## Subscribers ##
        self.img_sub = self.create_subscription(
            Image, "camera/thermal_image", self.img_callback, 10
        )

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, depth=10
        )
        self.odom_sub = self.create_subscription(
            String, "combined_odometry", self.odom_callback, qos_profile
        )

        # self.hot_sub = self.create_subscription(
        #     NavSatFix, "/user_hotspot", self.hot_coord_callback, 10
        # )

        self.timer_period = 1.0 / freq_hz
        self.timer_ = self.create_timer(self.timer_period, self.timer_callback)

        ## Publishers ##
        self.hotspot_pub = self.create_publisher(String, 'hotspot_info', 10)


    def img_callback(self, msg):
        self.latest_img_msg = msg
	
    def odom_callback(self, msg):
        try:
            data = json.loads(msg.data)

	        # If this is the first odometry message, save it as the home location
            if self.home_location is None:
                self.home_location = data
                self.sim = all(value == 0 for value in data.values())
                self.get_logger().info(f"Home location set: {self.home_location}, Sim mode: {self.sim}")

            self.odom_history.append(data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse odom: {e}")
    
    # def hot_coord_callback(self, msg):
    #     # self.hot_location = msg.data
    #     self.user_hot_lat = msg.latitude
    #     self.user_hot_long = msg.longitude
    #     # self.user_hot_alt = msg.altitude

    #     self.get_logger().debug(f"Recieved hotspot location from user - Lat: {self.user_hot_lat}, Long: {self.user_hot_long}")


    def timer_callback(self):
        if self.latest_img_msg is None:
            return

        ## filter image
        msg = self.latest_img_msg
        try:
            raw_16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        except ValueError as e:
            self.get_logger().error(f"Cannot reshape image: {e}")
            return

        scaled_img = (raw_16.astype(float) / 65535.0) * 100.0
        filtered_img = gaussian_filter(scaled_img, sigma=self.filter_sigma)

	    ## Find hotspot
        hotspot_y, hotspot_x = np.unravel_index(np.argmax(filtered_img), filtered_img.shape)

        img_time_us = (msg.header.stamp.sec * 1_000_000) + (msg.header.stamp.nanosec // 1000)
        best_odom = self.find_closest_odom(img_time_us)

        if best_odom is None:
            self.get_logger().debug("No odom match found for this image. Skipping.")
            return
        
        # center_x, center_y = msg.width // 2, msg.height // 2
        # pixel_offset_x, pixel_offset_y = hotspot_x - center_x, hotspot_y - center_y

        # ## take odom data and locate hotspots
        # self.altitude = best_odom.get("altitude", 1400)
        # self.pitch = math.radians(best_odom.get("pitch", 0))
        # self.roll = math.radians(best_odom.get("roll", 0))
        # self.yaw = math.radians(best_odom.get("yaw", 0))

        # ## take odom data and locate hotspots
        # self.altitude = best_odom.get("altitude_ellipsoid_m", 1400)
        # curr_qw = best_odom.get("qw", 1)
        # curr_qx = best_odom.get("qw", 0)
        # curr_qy = best_odom.get("qw", 0)
        # curr_qz = best_odom.get("qw", 0)
        # self.roll, self.pitch, self.yaw = self.quaternion_to_euler(curr_qw, curr_qx, curr_qy, curr_qz)

        

        ## used for debugging purposes
        # world_long = best_odom.get("longitude", 0)
        # world_lat = best_odom.get("lattitude", 0)
        # self.get_logger().info(f"World Location; Lat: {world_lat}, Long: {world_long}")
        
        # # Alt 70
        # # Compare altitude with home location
        # if(self.home_location is None):
        #     self.home_altitude = 1300
        # else:
        #     self.home_altitude = self.home_location.get("altitude_ellipsoid_m", 1300)  # Default home altitude
        
        # altitude_difference = self.altitude - self.home_altitude

        try:
            self.pitch = best_odom.get("pitch")
            self.roll  = best_odom.get("roll")
            self.yaw   = best_odom.get("yaw")
            altitude_difference = best_odom.get("z")
        except ValueError as e:
            self.get_logger().debug(f"Couldn't parse odom in filter: {e}")
            return

        # Log altitude to check
        # self.get_logger().info(
        #     f"First Alt: {self.home_altitude:.2f}, Curr Alt: {self.altitude:.2f} Altitude Difference: {altitude_difference:.2f}m"
        # )
        if(altitude_difference < 15):
            return
        

        # Camera properties
        FOV_x = 56
        FOV_y = 42
        img_width, img_height = msg.width, msg.height

        #camera to world 
        x, y = self.pixels_to_world(
            np.array([[hotspot_x, hotspot_y]]) , 
            np.array([img_width, img_height]), 
            np.array([FOV_x, FOV_y]), 
            self.pitch, self.roll, 0, 
            np.array([0, 0, altitude_difference]) )[0]
        
        
        self.get_logger().debug(
            f"Relative coords from UAV to fire: X:{x:.2f}m, Y: {y:.2f}m, Altitude: {altitude_difference:.2f}m"
        )

        # if(altitude_difference > 25):
        self.publish_location(x, y, altitude_difference)

        self.saved_images.append(scaled_img)
        self.saved_odom.append(best_odom)
        self.latest_img_msg = None

    def quaternion_to_euler(self, qw, qx, qy, qz):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        euler = euler_from_quaternion([qx, qy, qz, qw])
        return euler  # Returns a tuple: (roll, pitch, yaw)

    ## converts camera pixel locations to real world locations ##
    def pixels_to_world(self, pixel_coords, camera_dims, fov, pitch, roll, yaw, camera_coords):
        # Convert degrees to radians
        fov_rad = fov * np.pi / 180  # fov is now [fov_x, fov_y]
        fovx_rad, fovy_rad = fov_rad

        pitch_rad = pitch
        roll_rad = roll
        yaw_rad = yaw

        # Calculate rotation matrix
        R_roll = np.array([
            [np.cos(roll_rad), 0, np.sin(roll_rad)],
            [0,                 1, 0                ],
            [-np.sin(roll_rad), 0, np.cos(roll_rad)]
        ])
        R_pitch = np.array([
            [1, 0,                 0                ],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad),  np.cos(pitch_rad)]
        ])
        R_yaw = np.array([
            [np.cos(yaw_rad), np.sin(yaw_rad), 0],
            [-np.sin(yaw_rad),  np.cos(yaw_rad), 0],
            [0,               0,               1]
        ])
        R = R_yaw @ R_pitch @ R_roll

        # Calculate pixel ratios
        pixel_ratios = pixel_coords / (camera_dims - 1)

        # Calculate angle ratios
        angle_x = (pixel_ratios[:, 0] - 0.5) * fovx_rad
        angle_y = (pixel_ratios[:, 1] - 0.5) * fovy_rad

        # Calculate direction in the camera space
        sin_angle_x = np.sin(angle_x)
        sin_angle_y = np.sin(angle_y)
        cos_angle_x = np.cos(angle_x)
        cos_angle_y = np.cos(angle_y)
        dir_x = sin_angle_x * cos_angle_y
        dir_y = sin_angle_y
        dir_z = -cos_angle_x * cos_angle_y
        direction_camera_space = np.stack((dir_x, dir_y, dir_z), axis=-1)
        direction_camera_space /= np.linalg.norm(direction_camera_space, axis=1, keepdims=True)

        # Calculate the direction in the world space
        direction_world_space = (R @ direction_camera_space.T).T
        direction_world_space /= np.linalg.norm(direction_world_space, axis=1, keepdims=True)

        # Calculate the ground coordinates
        t = -camera_coords[2] / direction_world_space[:, 2]
        ground_coords = camera_coords[:2] + direction_world_space[:, :2] * t[:, np.newaxis]

        # Return the ground coordinates
        return ground_coords


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
    

    def publish_location(self, hot_x, hot_y, hot_alt):
        """Publish combined odometry data as a JSON string."""
        custom_pose = CustomPoseMsg(
            x=hot_x,
            y=hot_y,
            alt=hot_alt
        )

        # Convert CustomPoseMsg to JSON string and publish it
        json_str = json.dumps(custom_pose.__dict__)
        msg = String()
        msg.data = json_str

        # Publish the message on the /combined_odometry topic
        self.hotspot_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LowAltFilterNode(odom_queue_size=25, freq_hz=3.0)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()