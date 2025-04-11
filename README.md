# Drone Repository
This repository contains the code for the autonomous drones for the XPRIZE Snowflake team. Code scope covers subsystems of Communication, Detection, and Navigation. Steps have been taken to make sure all the necessary code is contained in this repository. If an onboard computer is corrupted, or more drones are added to the fleet, setting up the workspace is very simple.

## Setup:
1. Clone the repository into the home directory (`~/`) of the Raspberry Pi or similar onboard computer using `git clone git@github.com:XPRIZE-Snowflake/high_alt_drone.git`.

    - If you are cloning on a new onboard computer and receive an error similar to `Permission denied (publickey).`, follow [these steps](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) to generate a new ssh key-value pair and give you permission to perform Git actions.

2. Once the repository is cloned, it will be located in a directory named `~/high_alt_drone`. We now want to rename that directory to be called **ros2_ws** to be consistent with the rest of our workspaces. We can use `mv ~/high_alt_drone ~/ros2_ws` to rename it. Now go into that directory.

3. Once in that directory, we will install the Seek Thermal SDK files so we can communicate with the thermal camera. Use `sudo apt-get install ./seekthermal-sdk-dev-{version}_arm64.deb` to install the files. Then run `sudo udevadm control --reload` to reload the udev rules as per the Seek Thermal Quick Start Guide.
    - As a precaution, download this extra dependency for the Seek Thermal SDK. It is sometimes pre-installed, but not always. `sudo apt-get install libsdl2-dev`
  
4. Now try and build the workspace. Make sure you are in the main directory of the workspace (`~/ros2_ws`) and run `colcon build`. Due to the size of the **px4_msgs** package, this will take about 15-20 minutes. After you successfully build the first time you can build individual packages to save time.

## Running ROS Packages
In order to run each node, the following commands need to be sent from the terminal: 

1. **Enter the Pi wirelessly** 
**Option 1:** In the lab, with Wi-Fi, enter: `ssh username@10.2.118.167`  

**Option 2:** In the field, without Wi-Fi 

Create a hotspot from your computer or phone 

On a windows computer go to: settings->network and internet->mobile hotspot 

Find the new ip address of the Pi and enter using ssh: `ssh username@{ip_address}` 


2. **Go to the ros2 workspace and build the package**
```
cd ros2_ws 
colcon build --symlink-install --packages-select img_capture 
source install/setup.bash
```
Note: If the px4_msgs are creating an error, run: `colcon build --symlink-install --packages-select px4_msgs` 


3. **Run the desired nodes** 

Generic node activation: `ros2 run {package} {node}` 

**Option 1:** For testing in the lab: 
```
ros2 run img_capture sim_flight_publisher.py 
ros2 run img_capture high_alt_filter.py 
ros2 run img_capture k-means.py 
ros2 run img_capture live_feedback.py 
```
Run the last command in Putty, see FullSetupDoc

**Option 2:** For testing in the field: 
```
ros2 run img_capture seekcamera_publisher 
ros2 run img_capture odometry_publisher.py 
ros2 run img_capture flight_logger.py 
ros2 run img_capture live_feedback.py
```
Run the last command in Putty, see FullSetupDoc.

For more Field Testing, [go here](#running-in-the-field).

**Option 3:** Launch files 
```
ros2 launch img_capture new_img_capture.launch.py [optional arguments] 
ros2 launch img_capture new_img_capture.launch.py sim:=false log:=false high:=true 
```
 

Here are the possible arguments: 

sim:=false log:=false high:=true 

sim – false: launches odometry_publisher.py and seekcamera_publisher 

sim – true: launches sim_flight_publisher.py 

log – true: launches flight_logger.py 

log – false: doesn’t launch flight_logger.py 

high – true: launches high_alt_filter.py and kmeans.py 

high – false: launches low_alt_filter.py and drop_mech.py 

Print – log: only sends messages to the logger 

Print – screen: prints messages directly on the terminal 

Print – both: prints to screen and saves to the logger 

 

4. Shutdown the scripts 
This command will stop running code and save the log from flight_logger.py: `pgrep -f img_capture | xargs kill –9 ` 

## Running in the Field

**1.** From a different putty terminal run this (also from ros2_ws):
`ros2 run img_capture live_feedback`

**2.** Here are all the commands to run from ros2_ws in your main terminal:
```
nohup ros2 run img_capture seekcamera_publisher
nohup ros2 run img_capture odometry_publisher
nohup ros2 run img_capture flight_logger
```
Even if you lose wifi or close your first terminal, these nodes will continue running.
If you lose wifi, you will need to rerun live feedback to get visuals again.

**3.** When you are done with the flight and want to terminate the nohup commands to save the flight, run:
`pgrep -f img_capture | xargs kill -9`


# Code Organization: 

## Src:
All current and future packages will be found in the **/src** folder.

### img_capture
Contains the ROS2 nodes and python scripts for the subteams. The **/scripts** folder contains the python scripts to receive and send camera data. The **/src** folder contains the C++ seekcamera_publisher.cpp. This publisher sends the camera images to the ROS network so that our python scripts can filter and gather data.

### px4_msgs
This is a nested repository from PX4. From their README: ROS 2 message definitions for the PX4 Autopilot project.
Building this package generates all the required interfaces to interface ROS 2 nodes with the PX4 internals.

### px4_ros_com
This is a nested repository from PX4. From their README: This package provides example nodes for exchanging data and commands between ROS2 and PX4. It also provides a library to ease the conversion between ROS2 and PX4 frame conventions. It has a straight dependency on the px4_msgs package.
