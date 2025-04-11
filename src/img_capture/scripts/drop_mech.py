#!/usr/bin/env python3

## opens drop mechanism when receives command
# changes servo value/angle when it recieves true from dropcommand topic


# import os
#import math
import json
#import numpy as np
#import pandas as pd
import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from gpiozero import Servo
from time import sleep


class DropMechNode(Node):
    def __init__(self):
        super().__init__("drop_mech_node")
        self.get_logger().info("DropMechNode started.")

        ## data taken from topics ##
        self.hot_location = None
        self.x = 0
        self.y = 0
        self.alt = 0

        self.servo = Servo(25)
        self.openVal = 0.7
        self.closeVal = -1
        self.servo.value = self.closeVal
        self.automatic = False

        ## Subscribers ##
        self.img_sub = self.create_subscription(
            String, "hotspot_info", self.drop_callback, 10
        )

        self.img_sub = self.create_subscription(
            String, "servo_command", self.servo_command_callback, 10
        )

    def drop_callback(self, msg):
        try: 
            data = json.loads(msg.data)
            self.x = data.get("x", 26)
            self.y = data.get("y", 26)
            self.alt = data.get("alt", 70)
            self.hot_location = data
            if abs(self.x) <  25 and abs(self.y) < 25:
                self.get_logger().info(f"Drop mech activated")
                self.servo.value = self.openVal
        except Exception as e:
            self.get_logger().warn(f"Failed to parse hotspot data in mech drop code: {e}")

    def servo_command_callback(self, msg):
        try:
            # using command words
            command = msg.data.strip().lower()  # Get the string and normalize
            
            if command == "auto" :
                self.automatic = True
            elif command == "manual" :
                self.automatic = False
            elif self.automatic == False:
                if command == "open":
                    self.servo.value = self.openVal
                elif command == "close" :
                    self.servo.value = self.closeVal
            
        except Exception as e:
            self.get_logger().error(f"Fail to parse user command: {e}")
	

def main(args=None):
    rclpy.init(args=args)
    node = DropMechNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()