"""Launch the XR VR driver with its control service behind a grip-aware proxy."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    gui_argument = DeclareLaunchArgument(
        "gui",
        default_value="false",
        description="Start RViz2 automatically with this launch file.",
    )

    vr_driver_node = Node(
        package="xr_vr_driver_ros",
        executable="xr_vr_driver_ros_node",
        name="xr_vr_driver_ros_node",
        namespace="",
        output="screen",
        parameters=[{}],
        remappings=[
            (
                "/vr/meta_quest_control/trigger",
                "/vr/meta_quest_control/trigger_raw",
            )
        ],
    )

    gui = LaunchConfiguration("gui")
    rviz_config_path = PathJoinSubstitution(
        [FindPackageShare("xr_vr_driver_ros"), "rviz", "vr.rviz"]
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_path],
        condition=IfCondition(gui),
    )
    adb_monitor = Node(
        name="adb_monitor",
        package="xr_vr_driver_ros",
        executable="adb_monitor.py",
        output="both",
    )

    return LaunchDescription([gui_argument, vr_driver_node, rviz_node, adb_monitor])
