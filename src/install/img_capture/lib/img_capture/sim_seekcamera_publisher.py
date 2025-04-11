#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
import numpy as np
import time
import threading

class SimulatedSeekCameraNode(Node):
    def __init__(self):
        super().__init__('seekcamera_simulator')
        # Create publisher on the same topic and with the same queue size as in the C++ node
        self.publisher_ = self.create_publisher(Image, 'camera/thermal_image', 10)

        # Load the simulation data files
        try:
            # video_matrix16.npy is assumed to be an array of shape (num_frames, height, width)
            self.video_matrix = np.load('sim_data/video_matrix16.npy')
            # camera_timestamps.npy is assumed to be a vector (one timestamp per frame) in microseconds
            self.camera_timestamps = np.load('sim_data/camera_timestamps.npy').astype(int)
        except Exception as e:
            self.get_logger().error(f"Failed to load simulation data: {e}")
            rclpy.shutdown()
            return

        # Check that the number of frames matches the number of timestamps
        if self.video_matrix.shape[0] != self.camera_timestamps.shape[0]:
            self.get_logger().error("Number of frames and timestamps do not match!")
            rclpy.shutdown()
            return

        self.num_frames = self.video_matrix.shape[0]
        self.height = int(self.video_matrix.shape[1])
        self.width  = int(self.video_matrix.shape[2])
        self.get_logger().info(
            f"Loaded {self.num_frames} frames (size: {self.width}x{self.height}).")

        # Determine the frames per second (FPS) from the timestamp differences.
        # Timestamps are in microseconds so we convert the average delta to seconds.
        if self.num_frames > 1:
            dt_us = np.diff(self.camera_timestamps)
            avg_dt_us = np.mean(dt_us)
            fps = 1e6 / avg_dt_us
            self.get_logger().info(f"Estimated FPS from timestamps: {fps:.2f}")
        else:
            self.get_logger().warn("Only one frame found; FPS estimation not possible.")
        
        # Start a separate thread to publish frames at the proper rate
        self.publisher_thread = threading.Thread(target=self.publish_frames)
        self.publisher_thread.daemon = True
        self.publisher_thread.start()

    def publish_frames(self):
        """
        Loop over the simulation frames and publish them on the 'camera/thermal_image' topic.
        The header timestamp is set from the simulation data (converted from microseconds to sec/nsec)
        and the delay between messages is determined from the differences in these timestamps.
        """
        for i in range(self.num_frames):
            # Create a new Image message
            msg = Image()

            # Convert the simulation timestamp (in microseconds) to seconds and nanoseconds.
            # (This simulates the header stamp that in the C++ code is set via this->now().)
            timestamp_us = int(self.camera_timestamps[i])
            sec  = timestamp_us // 1_000_000
            nsec = (timestamp_us % 1_000_000) * 1000  # convert microseconds remainder to nanoseconds

            stamp = Time()
            stamp.sec = sec
            stamp.nanosec = nsec
            msg.header.stamp = stamp
            msg.header.frame_id = "seek_thermal_frame"

            # Fill in the image fields. The C++ node publishes the PRE_AGC frame as mono16:
            msg.height   = self.height
            msg.width    = self.width
            msg.encoding = "mono16"
            msg.is_bigendian = 0  # False
            msg.step     = self.width * 2  # each pixel is 2 bytes

            # Get the current frame (assumed to be a 2D array of dtype uint16)
            frame = self.video_matrix[i]

            # Ensure the byte order is little-endian (if not, swap the bytes)
            if frame.dtype.byteorder == '>':
                frame = frame.byteswap().newbyteorder()
            # Convert the numpy array to a bytes object and assign it to msg.data
            msg.data = frame.tobytes()

            # Publish the message
            self.publisher_.publish(msg)
            self.get_logger().info(f"Published frame {i+1}/{self.num_frames}")

            # If this is not the last frame, compute the delay until the next frame.
            if i < self.num_frames - 1:
                # dt (in seconds) from the difference between the next and current timestamp
                dt = (int(self.camera_timestamps[i+1]) - timestamp_us) / 1e6
                time.sleep(dt)

def main(args=None):
    rclpy.init(args=args)
    node = SimulatedSeekCameraNode()
    try:
        # Spin will keep the node alive until externally interrupted
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        node.get_logger().info("Shutting down simulated seek camera node.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
