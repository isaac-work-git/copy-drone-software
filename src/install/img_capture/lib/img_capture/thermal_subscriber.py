#!/usr/bin/env python3

import os
import json
import time
import math
import numpy as np
import threading
import tkinter as tk
from collections import deque
from datetime import datetime, timezone

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy

import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.ndimage import gaussian_filter

# -------------------------
# Filtering function
# -------------------------
def filter_image(img, first_threshold, filter_boost, filter_sigma, second_threshold):
    """
    img is in [0..100] float range.
    1) Zero out values below first_threshold
    2) Add filter_boost to remaining (non-zero) pixels
    3) Apply gaussian blur with sigma=filter_sigma
    4) Zero out values below second_threshold
    5) Binarize anything >0 to 100
    6) Return the filtered array, plus list of hot_spots & weights
    """

    filtered_img = img.copy()  # float in [0..100]

    # 1) Apply first threshold
    filtered_img[filtered_img < first_threshold] = 0

    # 2) Add boost
    mask_nonzero = (filtered_img > 0)
    filtered_img[mask_nonzero] += filter_boost

    # 3) Gaussian blur
    filtered_img = gaussian_filter(filtered_img, sigma=filter_sigma)

    # 4) Second threshold
    filtered_img[filtered_img < second_threshold] = 0

    # 5) Binarize for display
    visual_img = filtered_img.copy()
    visual_img[visual_img > 0] = 100.0

    # 6) Locate hot spots
    hot_spots = []
    weights = []
    black_out_edge = int(np.ceil(2 * filter_sigma))

    search_img = filtered_img.copy()
    while search_img.max() > 0:
        hot_spot = np.unravel_index(np.argmax(search_img), search_img.shape)
        lower_x = max(0, hot_spot[0] - black_out_edge)
        upper_x = min(search_img.shape[0], hot_spot[0] + black_out_edge + 1)
        lower_y = max(0, hot_spot[1] - black_out_edge)
        upper_y = min(search_img.shape[1], hot_spot[1] + black_out_edge + 1)

        weights.append(float(np.sum(search_img[lower_x:upper_x, lower_y:upper_y])))
        search_img[lower_x:upper_x, lower_y:upper_y] = 0
        hot_spots.append([int(hot_spot[0]), int(hot_spot[1])])

    return visual_img, hot_spots, weights

# -------------------------
# Minimal data class for odometry
# -------------------------
class OdomData:
    """Hold one odometry entry (parsed from JSON)."""
    __slots__ = ('timestamp_us','latitude','longitude','altitude','pitch','roll','yaw')
    def __init__(self, timestamp_us, lat, lon, alt, pitch, roll, yaw):
        self.timestamp_us = timestamp_us
        self.latitude = lat
        self.longitude = lon
        self.altitude = alt
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw

# -------------------------
# The node
# -------------------------
class ThermalImageSubscriber(Node):
    """
    1) Subscribes to 'camera/thermal_image' (mono16 -> scaled [0..100]).
    2) Subscribes to '/combined_odometry' (String JSON).
    3) Buffers the last n=100 odometry messages in a deque.
    4) For each new image, if not busy, tries to match odometry within 0.5s and publish a combined message.
    5) We store the "last successful" image in self.current_img/self.current_filtered_img.
    6) If the user has never gotten a valid matched image, the user can forcibly capture one image for preview.
    """

    def __init__(self, odom_queue_size=100, max_time_diff_s=0.5):
        super().__init__('thermal_subscriber_np')

        self.odom_history = deque(maxlen=odom_queue_size)
        self.max_time_diff_us = int(max_time_diff_s * 1e6)  # 0.5 => 500000 us

        self.processing_in_progress = False

        # The last successfully matched & published image
        self.current_img = None
        self.current_filtered_img = None

        # For forced capture
        self.manual_capture_requested = False
        self.manual_capture_event = threading.Event()
        self.manual_capture_img = None

        # Subscriber for images
        self.image_sub = self.create_subscription(
            Image,
            'camera/thermal_image',
            self.image_callback,
            10
        )

        # Subscriber for combined odometry
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        self.odom_sub = self.create_subscription(
            String,
            '/combined_odometry',
            self.odom_callback,
            qos_profile
        )

        # Publisher for final "thermal + odom" message
        self.combo_pub = self.create_publisher(String, '/thermal_with_odom', 10)

        self.get_logger().info("ThermalImageSubscriber node started.")

    def odom_callback(self, msg):
        """Parse JSON and store in odom_history."""
        try:
            data = json.loads(msg.data)
            new_odom = OdomData(
                timestamp_us=data["timestamp"],
                lat=data["latitude"],
                lon=data["longitude"],
                alt=data["altitude"],
                pitch=data["pitch"],
                roll=data["roll"],
                yaw=data["yaw"]
            )
            self.odom_history.append(new_odom)
        except Exception as e:
            self.get_logger().error(f"Failed to parse odometry JSON: {e}")

    def image_callback(self, msg):
        """Called for each new image."""
        # Check if we are forcibly capturing an image
        if self.manual_capture_requested:
            # We'll do a quick read & convert, skip odometry matching,
            # store it in self.manual_capture_img, set the event, and return.
            try:
                raw_16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                self.manual_capture_img = (raw_16.astype(float) / 65535.0) * 100.0
            except ValueError as e:
                self.get_logger().error(f"Failed to reshape manual capture: {e}")
                self.manual_capture_img = None
            # End the manual capture
            self.manual_capture_requested = False
            self.manual_capture_event.set()
            return

        # Normal logic: skip if we are already busy
        if self.processing_in_progress:
            return

        self.processing_in_progress = True

        # Convert ROS time to microseconds
        img_time_us = (msg.header.stamp.sec * 1_000_000) + (msg.header.stamp.nanosec // 1000)

        # Interpret data as 16-bit
        try:
            raw_16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        except ValueError as e:
            self.get_logger().error(f"Failed to reshape image data: {e}")
            self.processing_in_progress = False
            return

        scaled_img = (raw_16.astype(float) / 65535.0) * 100.0

        # Match odometry
        best_odom = self.find_closest_odom(img_time_us)
        if best_odom is None:
            self.processing_in_progress = False
            return

        # Hard-coded filter params (we'll do advanced param reading from the GUI below)
        first_threshold = 10.0
        filter_boost    = 10.0
        filter_sigma    = 2.0
        second_thresh   = 20.0

        filtered_img, hot_spots, weights = filter_image(
            scaled_img, first_threshold, filter_boost, filter_sigma, second_thresh
        )

        # Update current images
        self.current_img = scaled_img
        self.current_filtered_img = filtered_img

        # Publish combined data
        combo_data = {
            "image_time_us": int(img_time_us),
            "odometry_time_us": int(best_odom.timestamp_us),
            "latitude": best_odom.latitude,
            "longitude": best_odom.longitude,
            "altitude": best_odom.altitude,
            "pitch": best_odom.pitch,
            "roll": best_odom.roll,
            "yaw": best_odom.yaw,
            "hot_spots": hot_spots,
            "weights": weights
        }
        combo_json = json.dumps(combo_data)
        out_msg = String()
        out_msg.data = combo_json
        self.combo_pub.publish(out_msg)
        self.get_logger().info(
            f"Published combined data with {len(hot_spots)} hotspots. "
            f"Image TS={img_time_us}, Odom TS={best_odom.timestamp_us}"
        )

        self.processing_in_progress = False

    def find_closest_odom(self, img_time_us):
        """Return OdomData with timestamp within self.max_time_diff_us, else None."""
        if not self.odom_history:
            return None

        best_odom = None
        best_diff = float('inf')
        for odom in self.odom_history:
            diff = abs(odom.timestamp_us - img_time_us)
            if diff < best_diff:
                best_diff = diff
                best_odom = odom
        if best_diff > self.max_time_diff_us:
            return None
        return best_odom

    # ------------- EXTRA: For forced single capture -------------
    def capture_one_image_blocking(self, timeout=2.0):
        """
        Attempt to fetch exactly one new camera frame (ignoring odometry).
        This will not set self.current_img; returns the scaled float image or None.
        """
        self.manual_capture_img = None
        self.manual_capture_requested = True
        self.manual_capture_event.clear()

        # Wait up to 'timeout' seconds for the image
        got_it = self.manual_capture_event.wait(timeout=timeout)
        if not got_it:
            return None
        return self.manual_capture_img

# -------------------------
# Main GUI Application
# -------------------------
def main():
    rclpy.init()
    node = ThermalImageSubscriber()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root = tk.Tk()
    root.title("Thermal + Odom Viewer (Enhanced)")

    fig = Figure(figsize=(8,4), dpi=100)
    (ax1, ax2) = fig.subplots(1,2)

    dummy_img = np.zeros((10,10), dtype=float)
    im_unfiltered = ax1.imshow(dummy_img, cmap='gray', vmin=0, vmax=100)
    ax1.set_title("Unfiltered")

    im_filtered = ax2.imshow(dummy_img, cmap='gray', vmin=0, vmax=100)
    ax2.set_title("Filtered")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    # 1) Load or create a param file
    param_file = "detection_params.npy"
    if os.path.exists(param_file):
        params = np.load(param_file, allow_pickle=True)
        if len(params) != 4:
            print(f"Param file {param_file} has unexpected shape. Re-init defaults.")
            params = np.array([10, 10, 2, 20], dtype=float)
            np.save(param_file, params)
    else:
        params = np.array([10, 10, 2, 20], dtype=float)
        np.save(param_file, params)

    detection_first_threshold  = params[0]
    detection_filter_boost     = params[1]
    detection_filter_sigma     = params[2]
    detection_second_threshold = params[3]

    # 2) Param entries
    param_frame = tk.Frame(root)
    param_frame.pack(side=tk.TOP, pady=5)

    tk.Label(param_frame, text="First Threshold").grid(row=0, column=0, padx=5)
    ent_first_threshold = tk.Entry(param_frame, width=6)
    ent_first_threshold.insert(0, str(detection_first_threshold))
    ent_first_threshold.grid(row=0, column=1)

    tk.Label(param_frame, text="Filter Boost").grid(row=0, column=2, padx=5)
    ent_filter_boost = tk.Entry(param_frame, width=6)
    ent_filter_boost.insert(0, str(detection_filter_boost))
    ent_filter_boost.grid(row=0, column=3)

    tk.Label(param_frame, text="Filter Sigma").grid(row=1, column=0, padx=5)
    ent_filter_sigma = tk.Entry(param_frame, width=6)
    ent_filter_sigma.insert(0, str(detection_filter_sigma))
    ent_filter_sigma.grid(row=1, column=1)

    tk.Label(param_frame, text="Second Threshold").grid(row=1, column=2, padx=5)
    ent_second_threshold = tk.Entry(param_frame, width=6)
    ent_second_threshold.insert(0, str(detection_second_threshold))
    ent_second_threshold.grid(row=1, column=3)

    def save_params():
        try:
            p1 = float(ent_first_threshold.get())
            p2 = float(ent_filter_boost.get())
            p3 = float(ent_filter_sigma.get())
            p4 = float(ent_second_threshold.get())
            np.save(param_file, np.array([p1, p2, p3, p4]))
            print(f"Saved params to {param_file}")
        except ValueError:
            print("Invalid input in param fields. Could not save.")

    tk.Button(param_frame, text="Save Params", command=save_params).grid(
        row=2, column=0, columnspan=4, pady=5
    )

    # 3) Update logic
    def update_plot():
        # Parse param fields
        try:
            first_thresh  = float(ent_first_threshold.get())
            boost         = float(ent_filter_boost.get())
            sigma         = float(ent_filter_sigma.get())
            second_thresh = float(ent_second_threshold.get())
        except ValueError:
            print("Cannot parse param fields. Aborting update.")
            return

        if node.current_img is None:
            # No "current" image => forcibly grab a new one ignoring odometry
            print("No current image. Attempting a single capture for preview...")
            temp_img = node.capture_one_image_blocking(timeout=2.0)
            if temp_img is None:
                print("Failed to capture a new image. Try again.")
                return
            # Filter the forced image with the updated params, but do NOT store as "current".
            filtered, hot_spots, weights = filter_image(
                temp_img, first_thresh, boost, sigma, second_thresh
            )
            # Display them
            im_unfiltered.set_data(temp_img)
            ax1.set_title(f"Unfiltered (forced): {temp_img.shape}")
            im_unfiltered.set_clim(0, 100)

            im_filtered.set_data(filtered)
            ax2.set_title(f"Filtered (forced): {filtered.shape}, Spots={len(hot_spots)}")
            im_filtered.set_clim(0, 100)
            canvas.draw()
        else:
            # We have a current image => re-filter it with the new params
            # so the user can see the difference. We do NOT re-run odometry.
            # Just re-apply the filter in real-time.
            # (If you want to re-run the entire pipeline with new params, you'd code differently.)
            ref_img = node.current_img
            filtered, hot_spots, weights = filter_image(
                ref_img, first_thresh, boost, sigma, second_thresh
            )
            node.current_filtered_img = filtered  # if you want to store it

            im_unfiltered.set_data(ref_img)
            ax1.set_title(f"Unfiltered: {ref_img.shape}")
            im_unfiltered.set_clim(0, 100)

            im_filtered.set_data(filtered)
            ax2.set_title(f"Filtered: {filtered.shape}, Spots={len(hot_spots)}")
            im_filtered.set_clim(0, 100)
            canvas.draw()

    btn_update = tk.Button(root, text="Update", command=update_plot)
    btn_update.pack(pady=5)

    def on_closing():
        node.get_logger().info("Closing GUI, shutting down ROS...")
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()
