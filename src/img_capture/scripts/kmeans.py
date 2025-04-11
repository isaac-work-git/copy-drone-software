#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import json
from std_msgs.msg import String

class DynamicKMeans(Node):
    def __init__(self):
        """
        Simplified node that keeps track of all hotspots received, and
        computes the average lat/lon and x/y across ALL received hotspots.
        Then it publishes these averages each time new hotspots arrive.
        """
        super().__init__('dynamic_kmeans')
        self.get_logger().info("Simplified K-Means (Averaging) Node started.")

        # We'll store all hotspots here as an Nx4 NumPy array: [lat, lon, x, y].
        self.all_hotspots = np.empty((0, 4))

        # Subscribe to the "hot_spots" topic
        self.create_subscription(String, "hot_spots", self.hotspot_callback, 10)

        # Publisher for the average hotspot data
        self.avg_pub = self.create_publisher(String, "average_hotspots", 10)

    def hotspot_callback(self, msg):
        """
        Each message contains:
          {
            "timestamp_us": <int>,
            "hotspots": [
                [lat, lon, x, y, weight],
                [lat, lon, x, y, weight],
                ...
            ]
          }
        We'll ignore the weights, just collecting (lat, lon, x, y).
        Then we compute the average lat/lon/x/y over ALL received hotspots,
        log it, and publish it.
        """
        try:
            data = json.loads(msg.data)
            hotspots = data.get("hotspots", [])

            # Extract lat, lon, x, y (ignore weight) and append them to self.all_hotspots
            new_data = []
            for h in hotspots:
                # h is [lat, lon, x, y, weight]
                if len(h) >= 4:
                    lat, lon, xx, yy = h[0], h[1], h[2], h[3]
                    new_data.append([lat, lon, xx, yy])

            if len(new_data) > 0:
                new_data = np.array(new_data)
                self.all_hotspots = np.vstack((self.all_hotspots, new_data))

                # Compute averages
                avg_lat = np.mean(self.all_hotspots[:, 0])
                avg_lon = np.mean(self.all_hotspots[:, 1])
                avg_x   = np.mean(self.all_hotspots[:, 2])
                avg_y   = np.mean(self.all_hotspots[:, 3])

                # Log to console
                self.get_logger().info(
                    f"Received {len(new_data)} new hotspots. "
                    f"Totals so far: {len(self.all_hotspots)}. "
                    f"Average lat/lon=({avg_lat:.6f}, {avg_lon:.6f}), "
                    f"Average x/y=({avg_x:.2f}, {avg_y:.2f})."
                )

                # Publish averages
                out_data = {
                    "average_lat": float(avg_lat),
                    "average_lon": float(avg_lon),
                    "average_x": float(avg_x),
                    "average_y": float(avg_y),
                    "total_count": len(self.all_hotspots)
                }
                self.avg_pub.publish(String(data=json.dumps(out_data)))

        except Exception as e:
            self.get_logger().error(f"Failed to parse hotspot JSON: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DynamicKMeans()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
