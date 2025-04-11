from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='img_capture',
            executable='seekcamera_publisher',
            name='seekcamera_publisher',
            output='screen'
        )
	Node(
	    package='img_capture',
	    executable='thermal_subscriber',
	    name='thermal_subscriber',
	    output='screen'
	)
    ])
