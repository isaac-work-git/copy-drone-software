#!/usr/bin/env python3

import os
import math
import json
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String

# ---------------------------------------------------------------------
# 1) HELPER FUNCTIONS (copy your versions in full).
#    For brevity, they are shown condensed here:
# ---------------------------------------------------------------------

def latlon_to_ecef(lat_deg, lon_deg, alt_m):
    # ... same as your provided function ...
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2*f - f*f
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = a / math.sqrt(1 - e2 * sin_lat*sin_lat)
    X = (N + alt_m) * cos_lat * math.cos(lon)
    Y = (N + alt_m) * cos_lat * math.sin(lon)
    Z = (N*(1 - e2) + alt_m) * sin_lat
    return X, Y, Z

def ecef_to_enu(dX, dY, dZ, ref_lat_deg, ref_lon_deg):
    # ... same as your provided function ...
    lat0 = math.radians(ref_lat_deg)
    lon0 = math.radians(ref_lon_deg)
    sin_lat0 = math.sin(lat0)
    cos_lat0 = math.cos(lat0)
    sin_lon0 = math.sin(lon0)
    cos_lon0 = math.cos(lon0)

    r11 = -sin_lon0
    r12 =  cos_lon0
    r13 =  0.0

    r21 = -sin_lat0*cos_lon0
    r22 = -sin_lat0*sin_lon0
    r23 =  cos_lat0

    r31 =  cos_lat0*cos_lon0
    r32 =  cos_lat0*sin_lon0
    r33 =  sin_lat0

    e = r11*dX + r12*dY + r13*dZ
    n = r21*dX + r22*dY + r23*dZ
    u = r31*dX + r32*dY + r33*dZ
    return np.array([e, n, u])

def ecef_to_latlonalt(X, Y, Z):
    # ... same as your provided function ...
    a = 6378137.0
    f = 1.0/298.257223563
    e2 = 2*f - f*f
    b = a*(1 - f)
    lon = math.atan2(Y, X)
    p = math.sqrt(X*X + Y*Y)
    eps = 1e-11

    lat = math.atan2(Z, p*(1 - e2))
    while True:
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1 - e2*sin_lat*sin_lat)
        alt = p / math.cos(lat) - N
        new_lat = math.atan2(Z + e2*N*sin_lat, p)
        if abs(new_lat - lat) < eps:
            lat = new_lat
            break
        lat = new_lat

    lat_deg = math.degrees(lat)
    lon_deg = math.degrees(lon)
    return lat_deg, lon_deg, alt

def enu_to_ecef(e, n, u, ref_lat_deg, ref_lon_deg):
    # ... same as your provided function ...
    lat0 = math.radians(ref_lat_deg)
    lon0 = math.radians(ref_lon_deg)
    sin_lat0 = math.sin(lat0)
    cos_lat0 = math.cos(lat0)
    sin_lon0 = math.sin(lon0)
    cos_lon0 = math.cos(lon0)

    dX = -sin_lon0*e + -sin_lat0*cos_lon0*n + cos_lat0*cos_lon0*u
    dY =  cos_lon0*e + -sin_lat0*sin_lon0*n + cos_lat0*sin_lon0*u
    dZ =                        cos_lat0*n +          sin_lat0*u
    return dX, dY, dZ

def enu_to_latlonalt(e, n, u, ref_lat_deg, ref_lon_deg, ref_alt_m):
    X0, Y0, Z0 = latlon_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    dX, dY, dZ = enu_to_ecef(e, n, u, ref_lat_deg, ref_lon_deg)
    X = X0 + dX
    Y = Y0 + dY
    Z = Z0 + dZ
    return ecef_to_latlonalt(X, Y, Z)

def pixels_to_world(pixel_coords, camera_dims, fov_deg, pitch_deg, roll_deg, yaw_deg, camera_coords_enu):
    """
    Convert pixel coordinates in the 2D thermal image to 2D ground-plane
    coordinates in the local ENU frame (x=East, y=North), ignoring altitude variations.
    
    Inputs:
    -------
    - pixel_coords: Nx2 array of (x,y) pixel indices
    - camera_dims: 2D array [width, height], e.g. (320, 240)
    - fov_deg: [fov_x_deg, fov_y_deg]
    - pitch_deg, roll_deg, yaw_deg: camera orientation in degrees
    - camera_coords_enu: [E, N, U] of the camera in the local ENU frame
    """
    # Convert to radians
    fov_rad = np.radians(fov_deg)   # e.g. [fov_x_rad, fov_y_rad]
    pitch_rad = np.radians(pitch_deg)
    roll_rad  = np.radians(roll_deg)
    yaw_rad   = np.radians(yaw_deg)

    # Rotation matrices
    R_roll = np.array([
        [ np.cos(roll_rad), 0, np.sin(roll_rad)],
        [0,                 1,               0],
        [-np.sin(roll_rad), 0, np.cos(roll_rad)],
    ])
    R_pitch = np.array([
        [1,               0,                0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad),  np.cos(pitch_rad)],
    ])
    R_yaw = np.array([
        [ np.cos(yaw_rad),  np.sin(yaw_rad), 0],
        [-np.sin(yaw_rad),  np.cos(yaw_rad), 0],
        [0,                0,               1],
    ])
    R = R_yaw @ R_pitch @ R_roll

    # Pixel coords in [0..(w-1)], [0..(h-1)]
    # shape: pixel_coords is Nx2
    w, h = camera_dims
    # Pixel as fraction of the image dimension (0.0 to 1.0)
    pixel_ratios = pixel_coords / (camera_dims - 1)  # Nx2

    # Convert fraction to angle offset from center
    # (pixel_ratios - 0.5) => range [-0.5..+0.5]
    angle_x = (pixel_ratios[:,0] - 0.5) * fov_rad[0]
    angle_y = (pixel_ratios[:,1] - 0.5) * fov_rad[1]

    # Direction in camera space
    sin_ax = np.sin(angle_x)
    sin_ay = np.sin(angle_y)
    cos_ax = np.cos(angle_x)
    cos_ay = np.cos(angle_y)
    dir_cam_x = sin_ax * cos_ay
    dir_cam_y = sin_ay
    dir_cam_z = -cos_ax * cos_ay  # negative to look "down" in front

    # shape => Nx3
    direction_camera_space = np.stack([dir_cam_x, dir_cam_y, dir_cam_z], axis=-1)
    direction_camera_space /= np.linalg.norm(direction_camera_space, axis=1, keepdims=True)

    # Rotate to world space
    direction_world = (R @ direction_camera_space.T).T
    direction_world /= np.linalg.norm(direction_world, axis=1, keepdims=True)

    # Intersect with ground plane: solve for t s.t. world_z + t*dir_z = 0
    # But here, camera_coords_enu[2] = camera altitude above ground
    cam_alt = camera_coords_enu[2]
    eps = 1e-12
    t = -cam_alt / (direction_world[:,2] + eps)  # Nx

    # Project to ground plane in ENU
    ground_x = camera_coords_enu[0] + direction_world[:,0]*t
    ground_y = camera_coords_enu[1] + direction_world[:,1]*t

    return np.stack([ground_x, ground_y], axis=-1)

# Example thermal-image filter function (adapted).
from scipy.ndimage import gaussian_filter
def filter_image(img, first_threshold, filter_boost, filter_sigma, second_threshold):
    """
    Given a (H,W) thermal image, do a simple threshold->boost->blur->threshold pipeline
    and then extract "hot spots." Returns (filtered_img, hot_spots, weights).
    """
    filtered_img = img.astype(float)

    # 1) First threshold => zero out below
    filtered_img[filtered_img < first_threshold] = 0

    # 2) Boost anything not zero
    mask = (filtered_img > 0)
    filtered_img[mask] += filter_boost

    # 3) Gaussian blur
    filtered_img = gaussian_filter(filtered_img, sigma=filter_sigma)

    # 4) Second threshold
    filtered_img[filtered_img < second_threshold] = 0

    # Identify hot spots
    hot_spots = []
    weights = []
    black_out_edge = int(np.ceil(2 * filter_sigma))

    tmp = filtered_img.copy()
    while True:
        val = tmp.max()
        if val <= 0:
            break
        max_idx = np.unravel_index(np.argmax(tmp), tmp.shape)
        hot_spots.append(max_idx)
        # sum of that region
        x0, y0 = max_idx
        lower_x = max(0, x0 - black_out_edge)
        upper_x = min(tmp.shape[0], x0 + black_out_edge+1)
        lower_y = max(0, y0 - black_out_edge)
        upper_y = min(tmp.shape[1], y0 + black_out_edge+1)
        w = np.sum(tmp[lower_x:upper_x, lower_y:upper_y])
        weights.append(w)
        # zero out that region
        tmp[lower_x:upper_x, lower_y:upper_y] = 0

    hot_spots = np.array(hot_spots, dtype=int)  # shape: (N,2) in (row,col)
    weights = np.array(weights, dtype=float)
    return filtered_img, hot_spots, weights

# ---------------------------------------------------------------------
# 2) THE NEW NODE
# ---------------------------------------------------------------------
class HotspotDetectorNode(Node):
    def __init__(self, odom_queue_size=25, freq_hz=3.0):
        super().__init__("hotspot_detector_node")
        self.get_logger().info("HotspotDetectorNode started.")

        # --------------------------------------------------------------
        # Load detection parameters
        # (Adjust the filename/path to match your setup)
        # We'll assume they were stored in a dict with keys matching the usage below.
        # e.g. detection_params = {
        #   "first_threshold": 30,
        #   "filter_boost": 15,
        #   "filter_sigma": 2.0,
        #   "second_threshold": 40
        # }
        # Save them to member variables.
        # --------------------------------------------------------------
        detection_params = np.load("detection_params.npy", allow_pickle=True).item()
        self.first_threshold   = detection_params["first_threshold"]
        self.filter_boost      = detection_params["filter_boost"]
        self.filter_sigma      = detection_params["filter_sigma"]
        self.second_threshold  = detection_params["second_threshold"]

        # Camera parameters
        self.camera_dims = np.array([320, 240])       # width=320, height=240
        self.fov = np.array([56.0, 42.0])             # degrees: [FOV_x, FOV_y]
        self.fixed_camera_tilt = 30.0                 # degrees to add to pitch

        # Odom queue
        self.odom_history = deque(maxlen=odom_queue_size)

        # Reference ECEF / lat-lon-alt (None until first fix)
        self.X0 = None
        self.Y0 = None
        self.Z0 = None
        self.ref_lat = None
        self.ref_lon = None
        self.ref_alt = None

        # For storing latest image
        self.latest_img_msg = None

        # Subscribe to the thermal image
        self.img_sub = self.create_subscription(
            Image,
            "camera/thermal_image",
            self.img_callback,
            10
        )

        # Subscribe to odometry
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

        # Publisher for hot spots
        self.hotspot_pub = self.create_publisher(String, "/hotspots", 10)

        # Create a timer to process images
        self.timer_period = 1.0 / freq_hz
        self.timer_ = self.create_timer(self.timer_period, self.timer_callback)

    def img_callback(self, msg):
        self.latest_img_msg = msg

    def odom_callback(self, msg):
        """Parse JSON odometry; store in a queue."""
        try:
            data = json.loads(msg.data)
            self.odom_history.append(data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse odom: {e}")

    def timer_callback(self):
        """Process the latest image at the set frequency."""
        if self.latest_img_msg is None:
            return
        # Convert the image to a 2D np.float32 array scaled [0..100]
        msg = self.latest_img_msg
        try:
            raw_16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        except ValueError as e:
            self.get_logger().error(f"Cannot reshape image: {e}")
            return

        scaled_img = (raw_16.astype(np.float32) / 65535.0) * 100.0
        # Try to find matching odom
        img_time_us = (msg.header.stamp.sec * 1_000_000) + (msg.header.stamp.nanosec // 1000)
        best_odom = self.find_closest_odom(img_time_us)
        if best_odom is None:
            self.get_logger().info("No odom match found for this image. Skipping.")
            return

        # If reference hasn't been set, set it now
        # We assume odom has "lat", "lon", "alt"
        if self.X0 is None:
            self.ref_lat = best_odom["lat"]
            self.ref_lon = best_odom["lon"]
            self.ref_alt = best_odom["alt"]
            self.X0, self.Y0, self.Z0 = latlon_to_ecef(self.ref_lat, self.ref_lon, self.ref_alt)
            self.get_logger().info(f"Reference GPS set to ({self.ref_lat:.6f}, {self.ref_lon:.6f}, {self.ref_alt:.1f} m)")

        # Extract camera position/orientation from odom
        # We assume best_odom has: lat, lon, alt, pitch, roll, yaw (all in degrees).
        lat = best_odom["lat"]
        lon = best_odom["lon"]
        alt = best_odom["alt"]
        pitch = best_odom.get("pitch", 0.0)
        roll  = best_odom.get("roll",  0.0)
        yaw   = best_odom.get("yaw",   0.0)

        # Add the fixed camera tilt to pitch
        pitch += self.fixed_camera_tilt

        # Convert (lat,lon,alt) to ENU
        X, Y, Z = latlon_to_ecef(lat, lon, alt)
        dX = X - self.X0
        dY = Y - self.Y0
        dZ = Z - self.Z0
        camera_coords_enu = ecef_to_enu(dX, dY, dZ, self.ref_lat, self.ref_lon)
        # camera_coords_enu => [E, N, U]

        # Filter the image
        _, hot_spots, weights = filter_image(
            scaled_img,
            self.first_threshold,
            self.filter_boost,
            self.filter_sigma,
            self.second_threshold
        )

        # For each hot spot, convert (row,col) -> (x=col, y=row) so we do x as width
        if len(hot_spots) == 0:
            # No hot spots => publish empty list
            self.publish_hotspots([])
            self.get_logger().info("No hot spots found in this image.")
        else:
            # Convert from (row, col) to (x, y)
            # row => y, col => x, but your pixels_to_world function expects
            # pixel_coords in the form [x, y] = [col, row].
            pixel_coords = []
            for (r, c) in hot_spots:
                pixel_coords.append([c, r])
            pixel_coords = np.array(pixel_coords, dtype=float)

            # Project to ground in local ENU
            ground_points_enu = pixels_to_world(
                pixel_coords, 
                self.camera_dims,
                self.fov,
                pitch,
                roll,
                yaw,
                camera_coords_enu
            )

            # Convert each ground ENU point back to absolute lat/lon
            # We'll store (lat, lon, e, n) in a list
            hotspots_out = []
            for i, (gx, gy) in enumerate(ground_points_enu):
                # e = gx, n = gy
                e = gx
                n = gy
                u = 0.0  # ground level (assuming same reference altitude)
                lat_out, lon_out, alt_out = enu_to_latlonalt(
                    e, n, u,
                    self.ref_lat, self.ref_lon, self.ref_alt
                )
                hotspots_out.append( (lat_out, lon_out, gx, gy) )

            # Publish them
            self.publish_hotspots(hotspots_out)
            self.get_logger().info(f"Detected {len(hot_spots)} hot spots and published them.")

        # Mark image as consumed
        self.latest_img_msg = None

    def find_closest_odom(self, img_time_us):
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

    def publish_hotspots(self, hotspot_list):
        # hotspot_list: list of (lat, lon, x, y)
        # Publish as JSON
        # Example: [ [lat1, lon1, e1, n1], [lat2, lon2, e2, n2], ...]
        msg = String()
        msg.data = json.dumps(hotspot_list)
        self.hotspot_pub.publish(msg)

    def save_and_exit(self):
        # Not strictly required unless you want to save some logs to disk.
        # For now, do nothing or add code if needed.
        pass


def main(args=None):
    rclpy.init(args=args)
    node = HotspotDetectorNode(odom_queue_size=25, freq_hz=3.0)

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
