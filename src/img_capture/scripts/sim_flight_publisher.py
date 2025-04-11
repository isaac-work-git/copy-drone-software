#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
import threading
import json

from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
from std_msgs.msg import String

###############################################################################
# Simple container for the odometry fields
###############################################################################
class CustomPoseMsg:
    def __init__(self, timestamp, latitude, longitude, altitude, pitch, roll, yaw):
        self.timestamp = timestamp
        self.latitude  = latitude
        self.longitude = longitude
        self.altitude  = altitude
        self.pitch     = pitch
        self.roll      = roll
        self.yaw       = yaw

###############################################################################
# Main Combined Publisher Node
###############################################################################
class SimCombinedPublisher(Node):
    def __init__(self):
        super().__init__('sim_combined_publisher')
        self.get_logger().info("Sim Flight Publisher started.\n")

        # Publishers
        self.camera_pub = self.create_publisher(Image, 'camera/thermal_image', 10)
        self.odom_pub   = self.create_publisher(String, 'combined_odometry', 10)

        # Load data from files
        try:
            self.video_matrix = np.load('sim_data/video_matrix16.npy')
        except Exception as e:
            self.get_logger().error(f"Failed to load video_matrix16.npy: {e}")
            rclpy.shutdown()
            return

        try:
            # Odometry: shape (N, 7) => [timestamp_us, lat, lon, alt, pitch, roll, yaw]
            self.odom_data = np.load('sim_data/clean_odometry.npy')
        except Exception as e:
            self.get_logger().error(f"Failed to load clean_odometry.npy: {e}")
            rclpy.shutdown()
            return

        # Validate shapes
        self.num_frames = self.video_matrix.shape[0]
        if self.odom_data.shape[0] != self.num_frames:
            self.get_logger().error(
                "Number of odometry rows does not match number of video frames!"
            )
            rclpy.shutdown()
            return
        if self.odom_data.shape[1] != 7:
            self.get_logger().error(
                "Odometry data must have 7 columns: [timestamp_us, lat, lon, alt, pitch, roll, yaw]."
            )
            rclpy.shutdown()
            return

        # Image dimensions
        self.height = int(self.video_matrix.shape[1])
        self.width  = int(self.video_matrix.shape[2])
        self.get_logger().debug(
            f"Loaded {self.num_frames} frames of size {self.width}x{self.height}."
        )

        # Extract the odometry timestamps (microseconds) for pacing only
        self.timestamps_us = self.odom_data[:, 0].astype(np.int64)

        # Estimate frequency from the odometry timestamps
        if self.num_frames > 1:
            dt_us = np.diff(self.timestamps_us)
            avg_dt_us = np.mean(dt_us)
            freq = 1e6 / avg_dt_us
            self.get_logger().debug(f"Estimated publishing frequency: {freq:.2f} Hz")
        else:
            self.get_logger().warn("Only one sample found; frequency estimation not possible.")

        # Start a thread to publish everything
        self.publish_thread = threading.Thread(target=self.publish_data_loop)
        self.publish_thread.daemon = True
        self.publish_thread.start()

    def publish_data_loop(self):
        """
        Loop over all frames/odometry rows. The sleep interval is determined by
        the difference between consecutive odometry timestamps. However, the *published*
        timestamp in both messages is the current system time in microseconds since 1971.
        """
        for i in range(self.num_frames):
            # ------------------- Pacing Logic: Sleep based on odometry timestamps -------------------
            # Sleep at the end of each iteration (except for i=0)
            # to emulate real-time intervals between frames.
            if i > 0:
                # dt in seconds based on difference in consecutive odometry timestamps
                prev_ts = self.timestamps_us[i - 1]
                curr_ts = self.timestamps_us[i]
                dt_sec  = (curr_ts - prev_ts) / 1e6
                time.sleep(dt_sec)

            # ------------------- Get "Now" in microseconds since 1971 -------------------
            # 1 year after 1970 epoch => 1971 offset = 31,536,000 seconds
            # This is approximate, ignoring leap years.
            offset_1971_s = 31536000
            now_sec = time.time() - offset_1971_s
            now_us  = int(now_sec * 1e6)

            # Convert microseconds to sec/nanosec
            sec  = now_us // 1_000_000
            nsec = (now_us % 1_000_000) * 1000

            # ------------------- Publish Camera Image -------------------
            img_msg = Image()
            stamp = Time()
            stamp.sec = int(sec)
            stamp.nanosec = int(nsec)

            img_msg.header.stamp = stamp
            img_msg.header.frame_id = "seek_thermal_frame"
            img_msg.height = self.height
            img_msg.width  = self.width
            img_msg.encoding = "mono16"
            img_msg.is_bigendian = 0
            img_msg.step = self.width * 2

            # Ensure correct endianness
            frame = self.video_matrix[i]
            if frame.dtype.byteorder == '>':
                frame = frame.byteswap().newbyteorder()

            img_msg.data = frame.tobytes()
            self.camera_pub.publish(img_msg)
            self.get_logger().debug(f"Published frame {i+1}/{self.num_frames}")

            # ------------------- Publish Odometry -------------------
            row = self.odom_data[i]
            lat   = float(row[1])
            lon   = float(row[2])
            alt   = float(row[3])
            pitch = float(row[4])
            roll  = float(row[5])
            yaw   = float(row[6])

            # Instead of the original odometry timestamp, we use now_us
            custom_pose = CustomPoseMsg(
                timestamp=now_us,
                latitude=lat,
                longitude=lon,
                altitude=alt,
                pitch=pitch,
                roll=roll,
                yaw=yaw
            )
            odom_msg = String()
            odom_msg.data = json.dumps(custom_pose.__dict__)

            self.odom_pub.publish(odom_msg)
            self.get_logger().debug(f"Published odometry: {odom_msg.data}")

        self.get_logger().info("All frames and odometry have been published. Shutting down node.")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = SimCombinedPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().debug("Keyboard interrupt, shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
