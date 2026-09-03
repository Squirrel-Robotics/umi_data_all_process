"""
Sensor Data Collection Configuration

Define all collectible sensor data streams and configuration options
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CollectionConfig:
    """Sensor collection configuration"""

    # Joint states (slave arm joint states)
    slave_joint_names: Optional[List[str]] = None
    """Slave arm joint state name list, e.g.: ['left_arm_joint_states', 'right_arm_joint_states']
    If None or empty list, slave arm joint states are not collected
    Supported joint state names:
    - left_arm_joint_states: Left arm joint states
    - right_arm_joint_states: Right arm joint states
    - lift_joint_states: Lift joint states (quanta_x1)
    - waist_joint_states: Waist joint states (quanta_x2)
    - left_gripper_joint_states: Left gripper joint states
    - right_gripper_joint_states: Right gripper joint states
    - head_joint_states: Head joint states"""

    slave_action_names: Optional[List[str]] = None
    """Slave arm action name list, e.g.: ['left_arm_actions', 'right_arm_actions']
    If None, will be automatically generated from slave_joint_names (replace '_joint_states' with '_actions')
    If [], no action data will be collected"""

    # Image sensors (4 fixed types)
    enable_head_rgb_stream: bool = False
    """Enable head RGB video stream"""

    enable_head_depth_stream: bool = False
    """Enable head depth video stream"""

    enable_left_arm_rgb_stream: bool = False
    """Enable left arm RGB video stream"""

    enable_right_arm_rgb_stream: bool = False
    """Enable right arm RGB video stream"""

    # End pose
    enable_left_arm_end_pose: bool = False
    """Enable left arm end pose"""

    enable_right_arm_end_pose: bool = False
    """Enable right arm end pose"""

    enable_wrench_ext_world: bool = False
    """Enable wrist external force"""

    enable_wrench_ext_local: bool = False
    """Enable wrist local force"""

    enable_waist_end_pose: bool = False
    """Enable waist end pose"""

    enable_odometry: bool = False
    """Enable chassis odometry (odom)"""

    enable_pose: bool = False
    """Enable robot pose data (tracked_pose)"""

    enable_chassis_imu: bool = False
    """Enable chassis IMU data"""

    # Depth sensors
    enable_depth_points: bool = False
    """Enable chassis depth points"""

    enable_head_depth_video: bool = False
    """Enable head depth video stream"""

    # Laser scanner
    enable_laser_scan: bool = False
    """Enable laser scanner"""

    enable_left_gripper_position: bool = False
    """Enable left gripper position"""

    enable_right_gripper_position: bool = False
    """Enable right gripper position"""

    enable_vr_action_commands: bool = False
    """Use the configured real command streams as episode actions.

    Legacy gripper robots record VR pose/gripper commands. Revo2 robots record
    only the left/right six-axis Revo2 commands; arm pose commands are not part
    of that contract. Commands are causally aligned to each observation
    (latest command at or before the observation timestamp); next-frame
    measured states are not used as actions.
    """

    enable_revo2_hands: bool = False
    """Collect BrainCo Revo2 motor state streams.

    Revo2 serial devices are owned by the dedicated control bridge.  The data
    collector subscribes only to the bridge's joint-state ROS JSON topics;
    tactile streams are intentionally excluded. Collection therefore never
    opens or competes for the RS485 ports.
    """

    camera_stream_profiles: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )
    """Required source-camera contracts keyed by camera stream name.

    RGB profiles may define ``capture_codec``, ``width``, ``height``, ``fps``
    and ``fps_tolerance``. H265 profiles also define the encoded source size
    and optional logical-view crop. The collector validates source encoding,
    resolution and measured rate before accepting an episode.
    """

    # Tactile sensors
    enable_left_gripper_tactile: bool = False
    """Enable left gripper tactile sensor"""

    enable_right_gripper_tactile: bool = False
    """Enable right gripper tactile sensor"""

    enable_left_hand_tactile: bool = False
    """Enable left hand tactile sensor"""

    enable_right_hand_tactile: bool = False
    """Enable right hand tactile sensor"""

    # 距离传感器
    enable_tof_sensors: bool = False
    """Enable ToF sensors (2)"""

    enable_ultrasonic_sensors: bool = False
    """Enable ultrasonic sensors (4)"""

    enable_master_arm_data: bool = False
    """Enable master arm data, joint and gripper joint states"""


    def get_enabled_sensors(self) -> List[str]:
        """Get all enabled sensors list"""
        enabled = []
        for field_name, field_value in self.__dict__.items():
            if field_name.startswith('enable_') and field_value:
                sensor_name = field_name.replace('enable_', '')
                enabled.append(sensor_name)
        return enabled

    def get_camera_names(self) -> List[str]:
        """Get enabled camera names list"""
        cameras = []
        if self.enable_head_rgb_stream:
            cameras.append('head_rgb_stream')
        if self.enable_head_depth_stream:
            cameras.append('head_depth_stream')
        if self.enable_left_arm_rgb_stream:
            cameras.append('left_arm_rgb_stream')
        if self.enable_right_arm_rgb_stream:
            cameras.append('right_arm_rgb_stream')
        return cameras
