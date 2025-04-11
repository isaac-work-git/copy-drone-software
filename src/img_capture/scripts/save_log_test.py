#!/usr/bin/env python3

## Code description:
# prints flight log to see if it is saved correctly

import numpy as np

try: 
    # data = np.load('/home/username/ros2_ws/flight_logs/high_flight_odom_20250310_095442.npy')
    data = np.load('/home/username/ros2_ws/flight_logs/high_therm_images_20250310_095442.npy')
    
    # print(type(data))
    print(data.shape)
    print(data)

except Exception as e:
    print(f"Error loading file: {e}")