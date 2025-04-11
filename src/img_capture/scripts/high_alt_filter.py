#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

import numpy as np
import math
import json
import time
import threading
from collections import deque
from scipy.ndimage import gaussian_filter

###############################################################################
# Utilities for GPS <-> ENU conversions
###############################################################################

def wgs84_to_ecef(lat_deg, lon_deg, alt_m):
  a = 6378137.0             # semi-major axis
  f = 1 / 298.257223563     # flattening
  b = a * (1 - f)           # semi-minor axis
  e_sq = (a**2 - b**2) / a**2
  lat = np.radians(lat_deg)
  lon = np.radians(lon_deg)
  N = a / np.sqrt(1 - e_sq * np.sin(lat)**2)
  X = (N + alt_m) * np.cos(lat) * np.cos(lon)
  Y = (N + alt_m) * np.cos(lat) * np.sin(lon)
  Z = (N * (1 - e_sq) + alt_m) * np.sin(lat)
  return np.array([X, Y, Z])

def compute_enu_vectors(ref_lat, ref_lon):
  lat0 = np.radians(ref_lat)
  lon0 = np.radians(ref_lon)
  e_hat = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
  n_hat = np.array([
      -np.sin(lat0) * np.cos(lon0),
      -np.sin(lat0) * np.sin(lon0),
        np.cos(lat0)
  ])
  u_hat = np.array([
      np.cos(lat0) * np.cos(lon0),
      np.cos(lat0) * np.sin(lon0),
      np.sin(lat0)
  ])
  return e_hat, n_hat, u_hat

def ecef_to_geodetic(X, Y, Z):
  a = 6378137.0             # semi-major axis
  f = 1 / 298.257223563     # flattening
  b = a * (1 - f)           # semi-minor axis
  e_sq = (a**2 - b**2) / a**2
  ep_sq = (a**2 - b**2) / b**2

  # Compute longitude
  lon = np.arctan2(Y, X)

  # Distance from the z-axis
  p = np.sqrt(X**2 + Y**2)

  # Compute preliminary latitude
  theta = np.arctan2(Z * a, p * b)
  sin_theta = np.sin(theta)
  cos_theta = np.cos(theta)
  lat = np.arctan2(Z + ep_sq * b * sin_theta**3, p - e_sq * a * cos_theta**3)

  # Compute altitude
  N = a / np.sqrt(1 - e_sq * np.sin(lat)**2)
  alt = p / np.cos(lat) - N

  # Convert to degrees
  lat_deg = np.degrees(lat)
  lon_deg = np.degrees(lon)
  return lat_deg, lon_deg, alt

def enu_to_latlon(ref_lat, ref_lon, ref_alt, x, y, z):
  # Convert reference to ECEF
  ref_ecef = wgs84_to_ecef(ref_lat, ref_lon, ref_alt)

  # Compute ENU unit vectors at the reference
  e_hat, n_hat, u_hat = compute_enu_vectors(ref_lat, ref_lon)

  # Compute current ECEF coordinates using the ENU offset
  current_ecef = ref_ecef + x * e_hat + y * n_hat + z * u_hat

  # Convert current ECEF coordinates back to geodetic coordinates
  lat, lon, _ = ecef_to_geodetic(current_ecef[0], current_ecef[1], current_ecef[2])
  return lat, lon

###############################################################################
# Hotspot filtering
###############################################################################
def filter_image(img, first_threshold, filter_boost, filter_sigma, second_threshold):
    """
    Return (hot_spots, weights).
    Steps:
      1) Zero out anything < first_threshold
      2) Add filter_boost to surviving pixels
      3) Gaussian blur
      4) Zero out anything < second_threshold
      5) Repeatedly pick the max, record coordinate & sum in a small neighborhood,
         zero that out, repeat.
    """
    filtered_img = img.copy()

    # 1) First threshold
    filtered_img[filtered_img < first_threshold] = 0.0

    # 2) Boost
    mask = (filtered_img > 0)
    filtered_img[mask] += filter_boost

    # 3) Gaussian blur
    filtered_img = gaussian_filter(filtered_img, sigma=filter_sigma)

    # 4) Second threshold
    filtered_img[filtered_img < second_threshold] = 0.0

    # 5) Pick hotspots
    hot_spots = []
    weights = []
    black_out_edge = int(np.ceil(2 * filter_sigma))

    while True:
        max_val = filtered_img.max()
        if max_val <= 0:
            break
        # Argmax
        hotspot_idx = np.unravel_index(np.argmax(filtered_img), filtered_img.shape)
        x0, y0 = hotspot_idx
        # Local sum
        lx = max(0, x0 - black_out_edge)
        ux = min(filtered_img.shape[0], x0 + black_out_edge + 1)
        ly = max(0, y0 - black_out_edge)
        uy = min(filtered_img.shape[1], y0 + black_out_edge + 1)
        local_sum = np.sum(filtered_img[lx:ux, ly:uy])
        weights.append(local_sum)
        hot_spots.append(hotspot_idx)
        filtered_img[lx:ux, ly:uy] = 0.0

    hot_spots = np.array(hot_spots).astype(int)   # shape (N,2), row-major
    weights = np.array(weights)
    return hot_spots, weights

###############################################################################
# Pixel -> ground-plane projection
###############################################################################
def pixels_to_world(pixel_coords, camera_dims, fov_radians, pitch_rad, roll_rad, yaw_rad, camera_coords_enu):
    """
    Project pixel coordinates (x,y) onto the ground plane z=0 in local ENU coords.

    - pitch_rad, roll_rad, yaw_rad are already in radians!
    - camera_dims = [width, height]
    - fov_radians = [fov_x (rad), fov_y (rad)]  (Because user said no conversion needed.)
    - camera_coords_enu = [E, N, U]
    """
    fovx_rad = fov_radians[0]
    fovy_rad = fov_radians[1]

    # Build rotation matrices (already in rad).
    R_roll = np.array([
        [ np.cos(roll_rad), 0, np.sin(roll_rad)],
        [ 0,                1, 0               ],
        [-np.sin(roll_rad), 0, np.cos(roll_rad)]
    ])
    R_pitch = np.array([
        [1, 0,                0              ],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad),  np.cos(pitch_rad)]
    ])
    R_yaw = np.array([
        [ np.cos(yaw_rad),  np.sin(yaw_rad), 0],
        [-np.sin(yaw_rad),  np.cos(yaw_rad), 0],
        [ 0,                0,               1]
    ])

    R = R_yaw @ R_pitch @ R_roll  # 3x3

    w = camera_dims[0]
    h = camera_dims[1]
    px = pixel_coords[:, 0]  # x
    py = pixel_coords[:, 1]  # y

    # Scale to angles
    angle_x = (px / (w - 1) - 0.5) * fovx_rad
    angle_y = (py / (h - 1) - 0.5) * fovy_rad

    sin_ax = np.sin(angle_x)
    cos_ax = np.cos(angle_x)
    sin_ay = np.sin(angle_y)
    cos_ay = np.cos(angle_y)

    # direction in camera frame:
    dir_x = sin_ax * cos_ay
    dir_y = sin_ay
    dir_z = -cos_ax * cos_ay

    dir_cam = np.stack([dir_x, dir_y, dir_z], axis=-1)
    dir_cam /= np.linalg.norm(dir_cam, axis=1, keepdims=True)

    # rotate to world
    dir_world = (R @ dir_cam.T).T
    dir_world /= np.linalg.norm(dir_world, axis=1, keepdims=True)

    cam_e = camera_coords_enu[0]
    cam_n = camera_coords_enu[1]
    cam_u = camera_coords_enu[2]

    dz = dir_world[:, 2]
    # t for z=0 => -cam_u / dz
    t = -cam_u / dz

    ground_e = cam_e + t * dir_world[:, 0]
    ground_n = cam_n + t * dir_world[:, 1]
    return np.stack([ground_e, ground_n], axis=-1)

###############################################################################
# Main node
###############################################################################
class HotSpotNode(Node):
    def __init__(self):
        super().__init__("hot_spot_node")

        # self.get_logger().info("HotSpotNode is starting...")

        # Load filter parameters from JSON
        try:
            detection_params = np.load("detection_params.npy")
            self.first_threshold   = detection_params[0]
            self.filter_boost      = detection_params[1]
            self.filter_sigma      = detection_params[2]
            self.second_threshold  = detection_params[3]
        except Exception as e:
            self.get_logger().error(f"Failed to read detection_params.npy: {e}")
            # Fallback defaults:
            self.first_threshold   = 30
            self.filter_boost      = 5
            self.filter_sigma      = 2
            self.second_threshold  = 35


        # Don’t process images until altitude is at least 5 m above ref
        self.required_alt_buffer = 5.0

        # Rolling buffer of odometry
        self.odom_history = deque(maxlen=50)

        # Store only the most recent image
        self.newest_img_msg = None

        # Publishers
        self.hotspot_pub = self.create_publisher(
            String, "hot_spots", 10
        )

        # Subscribers
        self.create_subscription(
            Image,
            "camera/thermal_image",
            self.img_callback,
            10
        )
        self.create_subscription(
            String,
            "combined_odometry",
            self.odom_callback,
            10
        )

        # Start a background thread to process images
        self.processing_thread = threading.Thread(target=self.process_loop, daemon=True)
        self.processing_thread.start()

        self.get_logger().info("HotSpotNode started successfully.")

    def odom_callback(self, msg):
        """
        Odom JSON:
          {
            "timestamp": <int microseconds>,
            "x": <float m>,
            "y": <float m>,
            "z": <float m>,
            "pitch": <float rad>,
            "roll": <float rad>,
            "yaw": <float rad>,
            "ref_lat": <float deg>,
            "ref_lon": <float deg>,
            "ref_alt": <float m>
          }
        """
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Cannot parse odom JSON: {e}")
            return

        self.odom_history.append(data)

    def img_callback(self, msg):
        """
        Each image's header.stamp is also in microseconds (sec/nsec chosen by publisher).
        We just store the newest image.
        """
        self.newest_img_msg = msg

    def process_loop(self):
        CAMERA_DIMS = np.array([320, 240])  # Example: 320x240
        CAMERA_FOV = np.array([0.97738438, 0.73303829])  # ~ 56°, 42° in radians

        while rclpy.ok():
            if self.newest_img_msg is None:
                time.sleep(0.01)
                continue

            msg = self.newest_img_msg
            self.newest_img_msg = None

            if len(self.odom_history) == 0:
                self.get_logger().debug("No odometry yet - skipping this image.")
                continue

            # Convert image to float [0..100]
            try:
                img_16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            except ValueError as e:
                self.get_logger().error(f"Cannot reshape image: {e}")
                continue

            img_float = (img_16.astype(float) / 65535.0) * 100.0

            # Extract the image timestamp in microseconds
            # (Assuming the publisher set sec/nanosec so that this is consistent.)
            img_time_us = msg.header.stamp.sec * 1_000_000 + (msg.header.stamp.nanosec // 1000)

            # Find best odom
            odom = self.find_closest_odom(img_time_us)
            if odom is None:
                self.get_logger().debug("Could not find matching odom. Skipping image.")
                continue


            # Check altitude threshold
            curr_alt = odom["z"]
            if curr_alt < (self.required_alt_buffer):
                self.get_logger().debug(
                    f"Altitude {curr_alt:.2f} < {self.required_alt_buffer:.2f} "
                    "=> ignoring this image (drone not high enough)."
                )
                continue

            # Filter & get hotspots
            hot_spots, weights = filter_image(
                img_float,
                self.first_threshold,
                self.filter_boost,
                self.filter_sigma,
                self.second_threshold
            )

            if len(hot_spots) == 0:
                # Publish empty list
                out_msg = {
                    "timestamp_us": img_time_us,
                    "hotspots": []
                }
                self.hotspot_pub.publish(String(data=json.dumps(out_msg)))
                self.get_logger().debug(f"No hotspots found (timestamp={img_time_us}).")
                continue

            # Convert camera location to ENU
            x_   = odom["latitude"]
            y_   = odom["longitude"]
            z_   = odom["altitude"]
            pitch_ = odom["pitch"] + 15 * np.pi / 180
            roll_  = odom["roll"]
            yaw_   = odom["yaw"]
            ref_lat = odom["ref_lat"]
            ref_lon = odom["ref_lon"]
            ref_alt = odom["ref_alt"]

            camera_enu = np.array([x_, y_, z_])

            # Re-map hotspots from (row,col) => (x=col, y=row)
            pixel_coords = np.stack([hot_spots[:,1], hot_spots[:,0]], axis=-1)

            # Project to ground
            ground_enu_xy = pixels_to_world(
                pixel_coords,
                CAMERA_DIMS,
                CAMERA_FOV,
                pitch_, roll_, yaw_,
                camera_enu
            )

            # enu_to_latlon(ref_lat, ref_lon, ref_alt, x, y, z)

            # Convert each ENU hotspot to lat/lon
            final_hotspots = []
            for i in range(len(ground_enu_xy)):
                e_val = ground_enu_xy[i,0]
                n_val = ground_enu_xy[i,1]
                w_val = weights[i]
                lat2, lon2 = enu_to_latlon(ref_lat, ref_lon, ref_alt, e_val, n_val, 0)
                final_hotspots.append([lat2, lon2, float(e_val), float(n_val), float(w_val)])

            # Publish
            out_msg = {
                "timestamp_us": img_time_us,
                "hotspots": final_hotspots
            }
            self.hotspot_pub.publish(String(data=json.dumps(out_msg)))
            self.get_logger().debug(
                f"Published {len(final_hotspots)} hotspot(s) (ts={img_time_us})."
            )

    def find_closest_odom(self, stamp_us):
        best = None
        best_diff = float("inf")
        for od in self.odom_history:
            diff = abs(od["timestamp"] - stamp_us)
            if diff < best_diff:
                best_diff = diff
                best = od
        return best


def main(args=None):
    rclpy.init(args=args)
    node = HotSpotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt. Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()