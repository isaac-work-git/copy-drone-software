#!/usr/bin/env python3

import os
import math
import json
import numpy as np
import threading
import tkinter as tk
from collections import deque
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.ndimage import gaussian_filter

# -------------------------
# Filter function
# -------------------------
def filter_image(img, first_threshold, filter_boost, filter_sigma, second_threshold):
    """
    img is in [0..100] float range.
    1) Zero out values below first_threshold
    2) Add filter_boost to remaining (non-zero) pixels
    3) Apply gaussian blur with sigma=filter_sigma
    4) Zero out values below second_threshold
    5) Binarize anything >0 to 100
    Return the filtered array.
    """
    filtered_img = img.copy()
    filtered_img[filtered_img < first_threshold] = 0
    mask = (filtered_img > 0)
    filtered_img[mask] += filter_boost
    filtered_img = gaussian_filter(filtered_img, sigma=filter_sigma)
    filtered_img[filtered_img < second_threshold] = 0
    # Binarize for display
    visual_img = filtered_img.copy()
    visual_img[visual_img > 0] = 100.0
    return visual_img

# -------------------------
# Node that publishes "live" feedback
# -------------------------
class LiveFeedbackNode(Node):
    """
    Subscribes to camera images (mono16 -> scaled to [0..100]) and odometry.
    Maintains latest camera and odometry.
    Has a timer at 5 Hz that updates a GUI with:
      1) Normalized image (image / image.max)
      2) Raw scaled image [0..100]
      3) Filtered image [0..100]
    Also shows the latest odometry in a label.
    Has param fields for filter thresholds, etc., with "Save Params" button.
    No "Update" button. It's all live.
    """

    def __init__(self, freq_hz=5.0):
        super().__init__("live_feedback_node")
        self.get_logger().info("Live Feedback Node started.")

        # Latest camera frame (None if not received yet)
        self.latest_image = None  # float in [0..100]
        # Latest odometry data (parsed from JSON)
        self.latest_odom = None

        # Default filter params
        self.param_file = "detection_params.npy"
        self.first_threshold = 10.0
        self.filter_boost    = 10.0
        self.filter_sigma    = 2.0
        self.second_thresh   = 20.0
        self.load_params()

        # Subscribers
        self.create_subscription(
            Image,
            "camera/thermal_image",
            self.camera_callback,
            10
        )
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        self.create_subscription(
            String,
            "/combined_odometry",
            self.odom_callback,
            qos_profile
        )

        # Timer at freq_hz to invoke the GUI update callback
        period_s = 1.0 / freq_hz
        self.gui_timer = self.create_timer(period_s, self.gui_update_callback)

        # We'll set up the GUI in a separate function after spin() starts
        # but we can start it now in constructor if we want.

    def load_params(self):
        """Load filter parameters from a .npy file if present."""
        if os.path.exists(self.param_file):
            arr = np.load(self.param_file, allow_pickle=True)
            if len(arr) == 4:
                self.first_threshold  = arr[0]
                self.filter_boost     = arr[1]
                self.filter_sigma     = arr[2]
                self.second_thresh    = arr[3]
            else:
                self.get_logger().warn("Param file shape mismatch. Using defaults.")
        else:
            # Save defaults
            arr = np.array([self.first_threshold, self.filter_boost,
                            self.filter_sigma, self.second_thresh], dtype=float)
            np.save(self.param_file, arr)

    def save_params(self):
        """Save current filter parameters to .npy file."""
        arr = np.array([self.first_threshold, self.filter_boost,
                        self.filter_sigma, self.second_thresh], dtype=float)
        np.save(self.param_file, arr)
        self.get_logger().info(f"Params saved to {self.param_file}")

    def camera_callback(self, msg):
        """Receive the camera image, convert to float [0..100]."""
        try:
            raw_16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        except ValueError as e:
            self.get_logger().error(f"Failed to decode camera image: {e}")
            return
        scaled = (raw_16.astype(float) / 65535.0) * 100.0
        self.latest_image = scaled

    def odom_callback(self, msg):
        """Store the latest odometry JSON as a dictionary."""
        try:
            data = json.loads(msg.data)
            self.latest_odom = data  # store as dictionary
        except Exception as e:
            self.get_logger().error(f"Failed to parse odometry JSON: {e}")

    def gui_update_callback(self):
        """
        Timer callback at 5 Hz to refresh the GUI. We'll store references
        to our figure/axes in the node or pass them in via a global.
        For minimal code, we can set a global GUI reference or static var.
        """
        # This is a placeholder. The actual GUI updates will happen in the
        # main "live_feedback.py" code, once we integrate node + GUI. We need
        # a shared approach or a callback from the outside. We'll handle that
        # in "main()" below.
        pass


# -------------------------
# The main function w/ GUI
# -------------------------
def main():
    rclpy.init()
    node = LiveFeedbackNode(freq_hz=5.0)

    # We'll create a background spin thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Now create the Tkinter UI
    root = tk.Tk()
    root.title("Live Feedback (5 Hz)")

    # 3 subplots in one row
    fig = Figure(figsize=(12,4), dpi=100)
    (ax1, ax2, ax3) = fig.subplots(1,3)

    # Create a canvas
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Create dummy images
    dummy_img = np.zeros((10,10), dtype=float)
    im_norm    = ax1.imshow(dummy_img, cmap='gray', vmin=0, vmax=1)
    ax1.set_title("Normalized")

    im_raw     = ax2.imshow(dummy_img, cmap='gray', vmin=0, vmax=100)
    ax2.set_title("Raw [0..100]")

    im_filt    = ax3.imshow(dummy_img, cmap='gray', vmin=0, vmax=100)
    ax3.set_title("Filtered")

    # A small label to show odometry
    odom_label = tk.Label(root, text="No odom yet", font=("Arial", 10))
    odom_label.pack(side=tk.TOP, pady=5)

    # Param area
    param_frame = tk.Frame(root)
    param_frame.pack(side=tk.TOP, pady=5)

    tk.Label(param_frame, text="First Threshold").grid(row=0, column=0, padx=5)
    ent_first_threshold = tk.Entry(param_frame, width=6)
    ent_first_threshold.insert(0, str(node.first_threshold))
    ent_first_threshold.grid(row=0, column=1)

    tk.Label(param_frame, text="Filter Boost").grid(row=0, column=2, padx=5)
    ent_filter_boost = tk.Entry(param_frame, width=6)
    ent_filter_boost.insert(0, str(node.filter_boost))
    ent_filter_boost.grid(row=0, column=3)

    tk.Label(param_frame, text="Filter Sigma").grid(row=1, column=0, padx=5)
    ent_filter_sigma = tk.Entry(param_frame, width=6)
    ent_filter_sigma.insert(0, str(node.filter_sigma))
    ent_filter_sigma.grid(row=1, column=1)

    tk.Label(param_frame, text="Second Threshold").grid(row=1, column=2, padx=5)
    ent_second_threshold = tk.Entry(param_frame, width=6)
    ent_second_threshold.insert(0, str(node.second_thresh))
    ent_second_threshold.grid(row=1, column=3)

    def save_params():
        try:
            node.first_threshold = float(ent_first_threshold.get())
            node.filter_boost    = float(ent_filter_boost.get())
            node.filter_sigma    = float(ent_filter_sigma.get())
            node.second_thresh   = float(ent_second_threshold.get())
            node.save_params()
        except ValueError:
            print("Invalid param input, cannot save.")

    tk.Button(param_frame, text="Save Params", command=save_params).grid(
        row=2, column=0, columnspan=4, pady=5
    )

    # We'll define a function that runs at 5 Hz
    def refresh_gui():
        if node.latest_image is not None:
            # Get the latest image (which is originally in [0,100] but may not use the full range)
            img = node.latest_image

            # Compute the raw image’s min and max values
            min_val = img.min()
            max_val = img.max()
            if max_val - min_val > 1e-9:
                # Rescale the raw image so that min becomes 0 and max becomes 100
                raw_img = (img - min_val) / (max_val - min_val) * 100.0
                # Also compute a normalized version (0–1) for display purposes
                norm_img = (img - min_val) / (max_val - min_val)
            else:
                raw_img = np.zeros_like(img)
                norm_img = np.zeros_like(img)

            # Apply the filter to the raw image.
            # (filter_image expects its input to be in [0,100].)
            filt_img = filter_image(raw_img,
                                    node.first_threshold,
                                    node.filter_boost,
                                    node.filter_sigma,
                                    node.second_thresh)

            # (Optionally, you can also rescale the filtered image if its dynamic range is limited.
            #  In this example, we do so so that its minimum maps to 0 and maximum to 100.)
            fmin = filt_img.min()
            fmax = filt_img.max()
            if fmax - fmin > 1e-9:
                filt_img_rescaled = (filt_img - fmin) / (fmax - fmin) * 100.0
            else:
                filt_img_rescaled = np.zeros_like(filt_img)

            # Update the plots.
            im_norm.set_data(norm_img)
            ax1.set_title(f"Normalized (min={min_val:.1f}, max={max_val:.1f})")
            im_norm.set_clim(0, 1)

            im_raw.set_data(raw_img)
            im_raw.set_clim(0, 100)

            im_filt.set_data(filt_img_rescaled)
            im_filt.set_clim(0, 100)

        # Show the latest odometry, if available.
        if node.latest_odom is not None:
            txt = f"Latest Odom:\n{json.dumps(node.latest_odom, indent=2)}"
            odom_label.config(text=txt)

        canvas.draw()
        # Schedule next refresh (every 200 ms, i.e. 5 Hz)
        root.after(200, refresh_gui)

    # Kick off the first refresh
    root.after(200, refresh_gui)

    def on_closing():
        node.get_logger().info("Shutting down live_feedback.py ...")
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
