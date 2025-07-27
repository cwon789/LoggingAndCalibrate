import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('odom_logger')
    
    # Path to the parameter file
    default_params_file = os.path.join(pkg_share, 'config', 'logger_params.yaml')
    
    # Declare launch arguments
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to the ROS 2 parameters file'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    # Create the node
    odom_logger_node = Node(
        package='odom_logger',
        executable='odom_logger_node',
        name='odom_logger_node',
        parameters=[
            LaunchConfiguration('params_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen',
        emulate_tty=True
    )
    
    # Return launch description
    return LaunchDescription([
        params_file_arg,
        use_sim_time_arg,
        odom_logger_node
    ])