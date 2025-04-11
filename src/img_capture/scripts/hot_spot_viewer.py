#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import numpy as np
import json
import threading
import tkinter as tk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class HotSpotViewerNode(Node):
    """
    Subscribes to /hot_spots, which publishes messages of the form:
    {
      "timestamp_us": 1234567890,
      "hotspots": [
         [lat, lon, enu_x, enu_y, weight],
         ...
      ]
    }

    We store these hotspots, then show two scatter plots:
      1) latitude vs. longitude (autoscaled)
      2) ENU X vs. Y in a fixed 30×30 meter window with arrow buttons to pan.
    """
    def __init__(self, freq_hz=5.0):
        super().__init__("hot_spot_viewer_node")
        self.get_logger().info("HotSpotViewerNode started.")

        # Rolling lists for all hotspot coordinates
        self.lats = []
        self.lons = []
        self.enu_xs = []
        self.enu_ys = []
        self.weights = []

        # ENU window: 30×30 meters, centered at (0,0) initially
        self.window_size = 30.0
        self.window_center_x = 0.0
        self.window_center_y = 0.0

        # Subscribe
        self.create_subscription(String, "hot_spots", self.hotspot_callback, 10)

        # Refresh rate for the GUI
        self.period_ms = int(1000 // freq_hz)

    def hotspot_callback(self, msg):
        """
        Parse the incoming JSON and accumulate hotspots.
        """
        try:
            data = json.loads(msg.data)
            hotspots = data.get("hotspots", [])
            for h in hotspots:
                lat, lon, ex, ey, w = h
                self.lats.append(lat)
                self.lons.append(lon)
                self.enu_xs.append(ex)
                self.enu_ys.append(ey)
                self.weights.append(w)

            self.get_logger().debug(f"Received {len(hotspots)} hotspots (total={len(self.lats)}).")
        except Exception as e:
            self.get_logger().error(f"Failed to parse hotspot JSON: {e}")


def main():
    rclpy.init()
    node = HotSpotViewerNode(freq_hz=5.0)

    # Spin in a background thread so that callbacks are handled
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Create Tkinter GUI
    root = tk.Tk()
    root.title("Hot Spot Viewer")

    fig = Figure(figsize=(10,5), dpi=100)
    ax_left = fig.add_subplot(1,2,1)
    ax_right = fig.add_subplot(1,2,2)

    ax_left.set_title("Hotspots (Lat vs. Lon)")
    ax_left.set_xlabel("Longitude")
    ax_left.set_ylabel("Latitude")

    ax_right.set_title("Hotspots (ENU X vs. Y)")
    ax_right.set_xlabel("X [m]")
    ax_right.set_ylabel("Y [m]")

    # Create initial scatter plots with some default size/color
    # so points are clearly visible.
    scat_left = ax_left.scatter([], [], s=20, c="red", marker='o')
    scat_right = ax_right.scatter([], [], s=20, c="blue", marker='o')

    # We'll define a function to update the right subplot's X/Y limits
    def update_enu_limits():
        half = node.window_size / 2.0
        ax_right.set_xlim(node.window_center_x - half, node.window_center_x + half)
        ax_right.set_ylim(node.window_center_y - half, node.window_center_y + half)

    # Initialize
    update_enu_limits()

    # Embed figure in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Navigation buttons for ENU subplot
    def move_left():
        node.window_center_x -= 5
        update_enu_limits()
        canvas.draw()

    def move_right():
        node.window_center_x += 5
        update_enu_limits()
        canvas.draw()

    def move_up():
        node.window_center_y += 5
        update_enu_limits()
        canvas.draw()

    def move_down():
        node.window_center_y -= 5
        update_enu_limits()
        canvas.draw()

    btn_frame = tk.Frame(root)
    btn_frame.pack(side=tk.BOTTOM, pady=5)

    tk.Button(btn_frame, text="Left",  command=move_left ).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Right", command=move_right).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Up",    command=move_up   ).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Down",  command=move_down ).pack(side=tk.LEFT, padx=5)

    def on_closing():
        node.get_logger().info("Shutting down HotSpotViewerNode...")
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    def refresh_gui():
        """
        Periodic update of scatter data. We do manual axis updates for lat/lon.
        """
        if len(node.lats) > 0:
            # Update the scatter data
            scat_left.set_offsets(np.column_stack([node.lons, node.lats]))
            scat_right.set_offsets(np.column_stack([node.enu_xs, node.enu_ys]))

            # 1) Autoscale lat/lon manually
            min_lon = np.min(node.lons)
            max_lon = np.max(node.lons)
            min_lat = np.min(node.lats)
            max_lat = np.max(node.lats)

            # If they are identical (only 1 point), expand them slightly so it's visible
            if min_lon == max_lon:
                min_lon -= 0.0001
                max_lon += 0.0001
            if min_lat == max_lat:
                min_lat -= 0.0001
                max_lat += 0.0001

            ax_left.set_xlim(min_lon, max_lon)
            ax_left.set_ylim(min_lat, max_lat)

            # 2) ENU is a fixed window around (window_center_x, window_center_y)
            #    so we do NOT autoscale.  We only do update_enu_limits() upon arrow clicks.

        canvas.draw()
        root.after(node.period_ms, refresh_gui)

    # Kick off first GUI update
    root.after(node.period_ms, refresh_gui)
    root.mainloop()


if __name__ == "__main__":
    main()
