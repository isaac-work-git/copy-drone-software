import os
import launch
import launch_ros.actions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
import launch.conditions


# Launch file for default field use

def generate_launch_description():
    namespace = os.getenv("ROS_NAMESPACE", "")

    sim_arg = DeclareLaunchArgument(
        'sim', default_value='false', 
        description='Set to true to launch sim_flight_publisher.py, otherwise start seekcamera_publisher.cpp and odometry_publisher.py'
    )
    log_arg = DeclareLaunchArgument(
        'log', default_value='true', 
        description='Set to true to enable flight_logger.py'
    )
    high_arg = DeclareLaunchArgument(
        'high', default_value='false', 
        description='Set to false to run low_alt_filter.py, otherwise runs high_alt_filter.py and kmeans.py'
    )
    live_arg = DeclareLaunchArgument(
        'live', default_value='true',
        description='Set to false to run low_alt_filter.py, otherwise runs high_alt_filter.py and kmeans.py'
    )

    sim_flight_publisher_node = launch_ros.actions.Node(
        package='img_capture', executable='sim_flight_publisher.py', 
        name='sim_flight_publisher', output={'both': 'log'}, 
        namespace=namespace
    )
    
    seekcamera_publisher_node = launch_ros.actions.Node(
        package='img_capture', executable='seekcamera_publisher', 
        name='seekcamera_publisher', output={'both': 'log'}, 
        namespace=namespace
    )
    
    odometry_publisher_python_node = launch_ros.actions.Node(
        package='img_capture', executable='odometry_publisher.py', 
        name='odometry_publisher', output={'both': 'log'},
        namespace=namespace
    )
    
    flight_logger_node = launch_ros.actions.Node(
        package='img_capture', executable='flight_logger.py', 
        name='flight_logger', output={'both': 'log'},
        namespace=namespace
    )
    
    low_alt_filter_node = launch_ros.actions.Node(
        package='img_capture', executable='low_alt_filter.py', 
        name='low_alt_filter', output={'both': 'log'}, 
        namespace=namespace
    )
    
    high_alt_filter_node = launch_ros.actions.Node(
        package='img_capture', executable='high_alt_filter.py', 
        name='hot_spot_node', output={'both': 'log'}, 
        namespace=namespace
    )
    
    kmeans_node = launch_ros.actions.Node(
        package='img_capture', executable='kmeans.py', 
        name='kmeans', output={'both': 'log'}, 
        parameters=[
            {"max_radius": 15.0},
            {"min_size": 10}
        ],
        namespace=namespace
    )

    live_feedback_node = launch_ros.actions.Node(
        package='img_capture', executable='live_feedback.py', 
        name='live_feedback_node', output='screen', 
        namespace=namespace
    )

    sim_group = GroupAction(
        condition=launch.conditions.IfCondition(LaunchConfiguration('sim')),
        actions=[sim_flight_publisher_node],
    )
    
    default_group = GroupAction(
        condition=launch.conditions.UnlessCondition(LaunchConfiguration('sim')),
        actions=[seekcamera_publisher_node, odometry_publisher_python_node],
    )
    
    logging_group = GroupAction(
        condition=launch.conditions.IfCondition(LaunchConfiguration('log')),
        actions=[flight_logger_node],
    )
    
    high_filter_group = GroupAction(
        condition=launch.conditions.IfCondition(LaunchConfiguration('high')),
        actions=[high_alt_filter_node, kmeans_node],
    )
    
    low_filter_group = GroupAction(
        condition=launch.conditions.UnlessCondition(LaunchConfiguration('high')),
        actions=[low_alt_filter_node],
    )

    live_feedback_group = GroupAction(
        condition=launch.conditions.IfCondition(LaunchConfiguration('live')),
        actions=[live_feedback_node],
    )

    return launch.LaunchDescription([
        sim_arg,
        log_arg,
        high_arg,
        live_arg,
        sim_group,
        default_group,
        logging_group,
        high_filter_group,
        low_filter_group,
        live_feedback_group,
    ])
