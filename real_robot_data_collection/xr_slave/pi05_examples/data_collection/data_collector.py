"""
gRPC Real-time Data Collector - Supports all sensors, saves as universal JSON format

Usage:
    from x2robot import connect
    from x2robot.action_data_collection import DataCollector
    from x2robot.collection_config import CollectionConfigPresets

    robot = connect("x2://192.168.1.100:50051")

    # Use preset configuration
    collector = DataCollector(
        robot,
        output_dir="./data",
        target_hz=30,
        collection_config=CollectionConfigPresets.full_manipulation()
    )

    collector.start_recording(task="pick and place")
    # ... execute task ...
    collector.stop_recording()
"""

import json
from pathlib import Path
import time
import threading
import queue
from collections import defaultdict
import tempfile
import pickle
import bisect
import shutil
import os
import sys
import signal
import struct
import subprocess
from datetime import datetime
from typing import Optional, List, Dict, Any
import io
import atexit
import errno
import math
import re
import traceback
from copy import deepcopy

import numpy as np
from PIL import Image
import cv2

from x2robot import Robot
from .collection_config import CollectionConfig


VR_POSE_ACTION_STREAMS = {
    'vr_left_arm_pose_commands': 'left_arm_end_pose_action',
    'vr_right_arm_pose_commands': 'right_arm_end_pose_action',
}
VR_GRIPPER_ACTION_STREAMS = {
    'vr_left_gripper_joint_commands': 'left_gripper_position_action',
    'vr_right_gripper_joint_commands': 'right_gripper_position_action',
}
VR_REVO2_ACTION_STREAMS = {
    'vr_left_revo2_joint_commands': 'left_revo2_joint_action',
    'vr_right_revo2_joint_commands': 'right_revo2_joint_action',
}
LEGACY_VR_ACTION_SENSOR_NAMES = tuple(
    (*VR_POSE_ACTION_STREAMS, *VR_GRIPPER_ACTION_STREAMS)
)
REVO2_VR_ACTION_SENSOR_NAMES = tuple(
    VR_REVO2_ACTION_STREAMS
)
ALL_VR_ACTION_SENSOR_NAMES = tuple(
    dict.fromkeys(
        (*LEGACY_VR_ACTION_SENSOR_NAMES, *REVO2_VR_ACTION_SENSOR_NAMES)
    )
)
REVO2_STATE_SENSOR_NAMES = (
    'left_revo2_joint_states',
    'right_revo2_joint_states',
)
REVO2_ACTUATOR_NAMES = (
    'thumb',
    'thumb_aux',
    'index',
    'middle',
    'ring',
    'pinky',
)
VR_ROS_COMMAND_TOPICS = {
    'vr_left_arm_pose_commands': '/whole_body_controller/left_wrist_pose_cmd',
    'vr_right_arm_pose_commands': '/whole_body_controller/right_wrist_pose_cmd',
    'vr_left_gripper_joint_commands': '/gripper/cmd/left',
    'vr_right_gripper_joint_commands': '/gripper/cmd/right',
    'vr_left_revo2_joint_commands': '/revo2/command/left',
    'vr_right_revo2_joint_commands': '/revo2/command/right',
}
REVO2_ROS_TOPICS = {
    'left_revo2_joint_states': '/revo2/state/left',
    'right_revo2_joint_states': '/revo2/state/right',
}


class DataCollector:
    """Real-time data collector - saves as universal JSON format

    Collected data format:
    {
        "metadata": {
            "fps": 30,
            "joint_names": ["left_arm_joint1", ...],
            "camera_names": ["head_camera", ...],
            "robot_type": "x2_robot",
            "created_at": "2026-01-12T10:30:00"
        },
        "episodes": [
            {
                "episode_id": 0,
                "task": "pick and place",
                "timestamp": "2026-01-12T10:30:00",
                "duration": 10.5,
                "num_frames": 315,
                "joint_names": ["joint1", "joint2", ...],  // Joint name list
                "frames": [
                    {
                        "frame_id": 0,
                        "timestamp": 1234567890.123,
                        "observation": {
                            "joint_positions": [0.1, 0.2, ...],
                            "joint_velocities": [0.0, 0.0, ...],
                            "joint_efforts": [0.0, 0.0, ...]
                        },
                        "action": {
                            "joint_positions": [0.1, 0.2, ...]
                        },
                        "images": {
                            "head_camera": "episode_0/frame_0000_head_camera.jpg",
                            ...
                        }
                    },
                    ...
                ]
            }
        ]
    }
    """

    def __init__(
        self,
        robot: Robot,
        output_dir: str = "./collected_data",
        target_hz: float = 30.0,
        collection_config: Optional[CollectionConfig] = None,
        image_quality: int = 95,
        downsample_joint_states: bool = True,
        use_video_storage: bool = False,
        video_codec: str = 'XVID',  # Default to XVID, good compatibility
        keep_raw_data: bool = False,
        raw_topic_storage: bool = False,
        startup_readiness_timeout_seconds: Optional[float] = 8.0,
    ):
        """Initialize data collector

        Args:
            robot: Robot instance
            output_dir: Data save directory
            target_hz: Target collection frequency (for downsampling)
            collection_config: Sensor configuration object, use presets from SensorConfigPresets
                          If None, use basic_manipulation preset
            image_quality: JPEG image quality (1-100), only used when use_video_storage=False
            downsample_joint_states: Whether to downsample joint states
                - True: Downsample to target_hz (saves memory, suitable for training)
                - False: Keep original 500Hz (high precision, large memory usage)
            use_video_storage: Whether to use video format for image storage
                - True: Save as MP4 video (saves space, faster loading)
                - False: Save as JPG images (default, good compatibility)
            video_codec: Video codec (default: 'XVID')
                Recommended: 'XVID' (good compatibility), 'MJPG' (Motion JPEG, best compatibility)
                Optional: 'mp4v' (MPEG-4), 'avc1' (H.264, requires hardware support)
            keep_raw_data: Whether to also save the pre-alignment raw streams
                (sensor / pose / joint / action) into episode_xxxx/raw_data/.
                Camera frames are NOT included (they are large and duplicate the
                saved video/images). Default False (no behaviour change).
            raw_topic_storage: Save every enabled topic as its own timestamped
                pickle-record stream. This bypasses head-camera alignment,
                image decoding and video encoding. Camera payloads remain in
                their source JPEG/H265 encoding for later conversion.
            startup_readiness_timeout_seconds: Maximum time to wait for all
                enabled producers to become ready. Set to None to keep
                monitoring until they are ready or the episode is canceled.
        """
        self.robot = robot
        self.output_dir = Path(output_dir)
        self.target_hz = target_hz
        self.target_period = 1.0 / target_hz
        self.image_quality = image_quality
        self.downsample_joint_states = downsample_joint_states
        self.use_video_storage = use_video_storage
        self.video_codec = video_codec
        self.keep_raw_data = keep_raw_data
        self.raw_topic_storage = raw_topic_storage
        if (
            startup_readiness_timeout_seconds is not None
            and startup_readiness_timeout_seconds <= 0
        ):
            raise ValueError(
                "startup_readiness_timeout_seconds 必须大于 0 或为 None"
            )
        self.startup_readiness_timeout_seconds = (
            float(startup_readiness_timeout_seconds)
            if startup_readiness_timeout_seconds is not None
            else None
        )
        if self.raw_topic_storage and self.use_video_storage:
            print(
                "⚠️  raw topic 模式会忽略 use_video_storage；"
                "相机消息将按源 bytes 保存"
            )
        if self.raw_topic_storage and self.keep_raw_data:
            print(
                "⚠️  raw topic 模式本身已经保存全部原始流，"
                "keep_raw_data 不会额外生成 raw_data/"
            )

        # Use provided configuration or default configuration
        self.collection_config = collection_config or CollectionConfig()
        collection_config = self.collection_config
        self.camera_names = self.collection_config.get_camera_names()
        self.camera_stream_profiles = deepcopy(
            self.collection_config.camera_stream_profiles
        )
        unknown_profile_cameras = sorted(
            set(self.camera_stream_profiles) - set(self.camera_names)
        )
        if unknown_profile_cameras:
            raise ValueError(
                "相机采集契约包含未启用的流："
                f"{unknown_profile_cameras}"
            )
        self.camera_stream_profiles = {
            camera_name: self._normalize_camera_profile(
                camera_name,
                profile,
            )
            for camera_name, profile in self.camera_stream_profiles.items()
        }
        self.vr_action_sensor_names = ()
        if self.collection_config.enable_vr_action_commands:
            self.vr_action_sensor_names = (
                REVO2_VR_ACTION_SENSOR_NAMES
                if self.collection_config.enable_revo2_hands
                else LEGACY_VR_ACTION_SENSOR_NAMES
            )
        self.revo2_state_sensor_names = (
            REVO2_STATE_SENSOR_NAMES
            if self.collection_config.enable_revo2_hands
            else ()
        )
        self.ros_bridge_sensor_names = tuple(
            dict.fromkeys(
                (*self.revo2_state_sensor_names, *self.vr_action_sensor_names)
            )
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Episode counter
        self.episode_count = 0
        self._active_episode_staging_dir = None
        self._startup_cancel_event = threading.Event()

        # Data buffer queues - only created when needed
        # Queue size calculation: expected recording duration (seconds) × collection frequency (Hz)
        # Joint states: may be high frequency (500Hz)
        # Other sensors: usually lower frequency (10-100Hz)
        self.queues = {}  # Unified management of all data queues

        # Calculate joint state queue size (for display, even if joint states are not enabled)
        if downsample_joint_states:
            # Support 10 minutes of collection time with sufficient buffer
            self.queue_size = int(target_hz * 900)  # 900 seconds = 15 minutes
        else:
            # High frequency mode: support longer duration or larger buffer
            self.queue_size = int(500 * 900)  # 450,000, support 15 minutes @ 500Hz

        # Sensor data file locks (no longer using queues)
        sensor_names = []

        if collection_config.enable_left_arm_end_pose:
            sensor_names.append('left_arm_end_pose')
        if collection_config.enable_right_arm_end_pose:
            sensor_names.append('right_arm_end_pose')
        if collection_config.enable_waist_end_pose:
            sensor_names.append('waist_end_pose')

        if collection_config.enable_odometry:
            sensor_names.append('odometry')
        if collection_config.enable_pose:
            sensor_names.append('pose')
        if collection_config.enable_chassis_imu:
            sensor_names.append('chassis_imu')

        if collection_config.enable_depth_points:
            sensor_names.append('depth_points')
        if collection_config.enable_laser_scan:
            sensor_names.append('laser_scan')

        if collection_config.enable_left_gripper_tactile:
            sensor_names.append('left_gripper_tactile')
        if collection_config.enable_right_gripper_tactile:
            sensor_names.append('right_gripper_tactile')
        if collection_config.enable_left_hand_tactile:
            sensor_names.append('left_hand_tactile')
        if collection_config.enable_right_hand_tactile:
            sensor_names.append('right_hand_tactile')

        if collection_config.enable_tof_sensors:
            sensor_names.extend(['tof_1', 'tof_2'])
        if collection_config.enable_ultrasonic_sensors:
            sensor_names.extend([f'ultrasonic_{i}' for i in range(1, 5)])

        # Joint states use temporary file storage (high frequency collection, avoid memory issues)
        # Set according to configured slave_joint_names and slave_action_names
        self.slave_joint_names = []
        self.slave_action_names = []

        if collection_config.slave_joint_names:
            self.slave_joint_names = collection_config.slave_joint_names
            if downsample_joint_states:
                print(f"Joint state downsampling mode: 500Hz → {target_hz}Hz")
            else:
                print(f"Joint state original frequency mode: keeping 500Hz")
            print(f"Configured joint states: {self.slave_joint_names}")

            # Configure action names according to slave_action_names
            if collection_config.slave_action_names is not None:
                # If action_names are explicitly specified, use them
                self.slave_action_names = collection_config.slave_action_names
            else:
                # If None, automatically generate action_names from joint_names
                self.slave_action_names = [name.replace('_joint_states', '_actions') for name in self.slave_joint_names]

            if self.slave_action_names:
                print(f"Configured actions: {self.slave_action_names}")

            # Add joint states and actions for each part to sensor name list
            sensor_names.extend(self.slave_joint_names)
            sensor_names.extend(self.slave_action_names)

        if collection_config.enable_master_arm_data:
            sensor_names.append('master_left_arm_joint_state')
            sensor_names.append('master_right_arm_joint_state')
            sensor_names.append('master_left_arm_end_pose')
            sensor_names.append('master_right_arm_end_pose')
            sensor_names.append('master_left_gripper_joint_state')
            sensor_names.append('master_right_gripper_joint_state')

        if collection_config.enable_wrench_ext_world:
            sensor_names.append('left_arm_wrench_ext_world')
            sensor_names.append('right_arm_wrench_ext_world')
        if collection_config.enable_wrench_ext_local:
            sensor_names.append('left_arm_wrench_ext_local')
            sensor_names.append('right_arm_wrench_ext_local')

        if collection_config.enable_left_gripper_position:
            sensor_names.append('left_gripper_position')
        if collection_config.enable_right_gripper_position:
            sensor_names.append('right_gripper_position')
        if self.collection_config.enable_revo2_hands:
            sensor_names.extend(self.revo2_state_sensor_names)
        if collection_config.enable_vr_action_commands:
            sensor_names.extend(self.vr_action_sensor_names)

        if self.raw_topic_storage:
            self._validate_raw_topic_names(
                list(self.camera_names) + sensor_names
            )

        # All sensor data uses temporary file storage to avoid queue memory overflow
        self.sensor_temp_files = {}  # {sensor_name: file_handle}
        self.sensor_temp_paths = {}  # {sensor_name: file_path}
        self.sensor_file_locks = {}  # {sensor_name: lock}

        # Create file lock for each sensor
        for sensor_name in sensor_names:
            self.sensor_file_locks[sensor_name] = threading.Lock()

        # Image data uses temporary file storage to avoid memory overflow
        self.image_temp_files = {}  # {camera_name: file_handle}
        self.image_temp_paths = {}  # {camera_name: file_path}
        self.image_file_locks = {cam: threading.Lock() for cam in self.camera_names}

        # Control flags
        self.is_recording = False
        # Each episode owns an independent stop event.  A collector that is
        # still unwinding from an old gRPC stream must never become active
        # again merely because the next episode sets is_recording=True.
        self._episode_stop_event = None
        self._collector_context = threading.local()
        self._collector_errors_lock = threading.Lock()
        self._collector_errors = []
        self._vr_bridge_ready_event = threading.Event()
        self._revo2_feedback_ready_event = threading.Event()
        self._ros_bridge_seen_sensors = set()
        self._ros_bridge_seen_lock = threading.Lock()
        self._camera_profile_ready_event = threading.Event()
        self._camera_profile_lock = threading.Lock()
        self._camera_profile_status = {}
        self.thread_shutdown_timeout = 5.0
        self.threads = []
        self.thread_info = []

        # Temporary files are buffered and flushed at a bounded interval.
        # Closing the files at episode end still guarantees that all records
        # are available before alignment and saving.
        self.temp_file_flush_interval = 0.5
        self._temp_file_last_flush = {}

        # Current episode information
        self.current_episode_task = None
        self.current_episode_start_time = None
        self.current_episode_stop_time = None

        # Data collection statistics - dynamically create counters for each enabled sensor
        self.stats = {
            'start_time': None,
            'last_update': None
        }
        # Create counters for each enabled queue
        for queue_name in self.queues.keys():
            self.stats[f'{queue_name}_count'] = 0
        # Create counters for each camera
        for cam in self.camera_names:
            self.stats[f'{cam}_count'] = 0
        self.stats_lock = threading.Lock()
        self._raw_topic_stats_lock = threading.Lock()
        self._raw_topic_stats = {}

        # Joint name mapping
        self.joint_names = None
        self.joint_name_mapping = None

        # Error time record (for network error handling)
        self.last_error_time = 0

        # Camera data detection - record the last time each camera received data
        self.camera_last_data_time = {cam: None for cam in self.camera_names}
        self.camera_data_check_interval = 1.0  # Check interval (seconds)
        self.camera_data_timeout = 5.0  # Timeout (seconds) - error if no data within 5 seconds
        self.camera_monitor_thread = None
        self._reset_camera_profile_status()

        # Dataset metadata file
        self.metadata_file = self.output_dir / "dataset_metadata.json"
        self._load_or_create_metadata()

        # Register cleanup function on exit to ensure temporary files are cleaned up
        # Use lambda wrapper to ensure access to self
        atexit.register(lambda: self._cleanup_before_exit("Program exiting, cleaning up resources..."))

        # Register signal handler to ensure temporary files are cleaned up even on abnormal exit
        def signal_handler(sig, frame):
            """Handle interrupt signal, ensure temporary files are cleaned up"""
            err_msg = "\nReceived interrupt signal, cleaning up resources..."
            try:
                self._cleanup_before_exit(err_msg)
            except Exception as e:
                print(f"Error cleaning up resources: {e}")
            sys.exit(0)

        # Register SIGINT (Ctrl+C) and SIGTERM signal handlers
        # Note: Signal handlers may be overridden by handlers in example files, but at least try to clean up here
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except (ValueError, OSError):
            # Some signals may not be supported in certain environments (e.g., Windows or some test environments)
            pass

    def _load_or_create_metadata(self):
        """Load or create dataset metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.dataset_metadata = json.load(f)
            self.episode_count = len(self.dataset_metadata.get('episodes', []))
            print(f"已加载现有数据集，当前共有 {self.episode_count} 条数据")
        else:
            self.dataset_metadata = {
                "fps": self.target_hz,
                "camera_names": self.camera_names,
                "robot_type": "x2_robot",
                "created_at": datetime.now().isoformat(),
                "episodes": []
            }
            self._save_metadata()
        self._quarantine_orphan_staging_dirs()
        self._recover_complete_unindexed_episodes()

    def _save_metadata(self):
        """Atomically save dataset metadata."""
        temp_metadata_file = self.metadata_file.with_name(
            f".{self.metadata_file.name}.{os.getpid()}."
            f"{time.time_ns()}.tmp"
        )
        try:
            with open(temp_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(
                    self.dataset_metadata,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_metadata_file, self.metadata_file)
            self._fsync_directory(self.output_dir)
        finally:
            if temp_metadata_file.exists():
                try:
                    temp_metadata_file.unlink()
                except OSError:
                    pass

    @staticmethod
    def _fsync_directory(directory: Path):
        """Best-effort directory fsync for durable atomic renames."""
        directory_fd = None
        try:
            directory_fd = os.open(
                str(directory),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            os.fsync(directory_fd)
        except OSError:
            # Some filesystems/platforms do not support fsync on directories.
            pass
        finally:
            if directory_fd is not None:
                os.close(directory_fd)

    @staticmethod
    def _fsync_file(file_path: Path):
        """Flush a completed file before publishing its parent directory."""
        file_fd = None
        try:
            file_fd = os.open(str(file_path), os.O_RDONLY)
            os.fsync(file_fd)
        finally:
            if file_fd is not None:
                os.close(file_fd)

    @staticmethod
    def _validate_raw_topic_names(topic_names):
        """Reject duplicate or path-like names before creating raw files."""
        if len(topic_names) != len(set(topic_names)):
            raise ValueError(
                "raw topic 名称必须唯一，当前配置存在重名："
                f"{topic_names}"
            )
        invalid_names = [
            name
            for name in topic_names
            if (
                not isinstance(name, str)
                or re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None
                or name in {".", ".."}
            )
        ]
        if invalid_names:
            raise ValueError(
                "raw topic 名称只能包含字母、数字、下划线、点和连字符，"
                f"禁止路径字符：{invalid_names}"
            )

    @staticmethod
    def _normalize_camera_profile(camera_name, profile):
        """Validate and normalize one source-camera capture contract."""
        if not isinstance(profile, dict):
            raise ValueError(f"{camera_name} 相机采集契约必须是字典")
        codec = str(profile.get('capture_codec', '')).upper()
        if codec not in {'MJPEG', 'H265'}:
            raise ValueError(
                f"{camera_name} capture_codec 必须为 MJPEG 或 H265，"
                f"实际为 {codec!r}"
            )
        try:
            width = int(profile['width'])
            height = int(profile['height'])
            fps = float(profile['fps'])
            fps_tolerance = float(profile.get('fps_tolerance', 3.0))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{camera_name} 相机采集契约字段无效") from error
        if (
            width <= 0
            or height <= 0
            or not math.isfinite(fps)
            or fps <= 0
            or not math.isfinite(fps_tolerance)
            or fps_tolerance <= 0
            or fps_tolerance >= fps
        ):
            raise ValueError(f"{camera_name} 相机采集契约数值无效")
        normalized = {
            'capture_codec': codec,
            'storage_encoding': (
                'jpeg_frames'
                if codec == 'MJPEG'
                else 'h265_annexb_access_units'
            ),
            'width': width,
            'height': height,
            'fps': fps,
            'fps_tolerance': fps_tolerance,
        }
        if codec == 'H265':
            try:
                source_width = int(profile['source_width'])
                source_height = int(profile['source_height'])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{camera_name} H265 契约缺少有效的源画面尺寸"
                ) from error
            if source_width <= 0 or source_height <= 0:
                raise ValueError(f"{camera_name} H265 源画面尺寸无效")
            normalized['source_width'] = source_width
            normalized['source_height'] = source_height
            view_crop = profile.get('view_crop')
            if view_crop is not None:
                try:
                    normalized_crop = [int(value) for value in view_crop]
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{camera_name} view_crop 必须是四个整数"
                    ) from error
                if (
                    len(normalized_crop) != 4
                    or any(value < 0 for value in normalized_crop[:2])
                    or any(value <= 0 for value in normalized_crop[2:])
                    or normalized_crop[0] + normalized_crop[2] > source_width
                    or normalized_crop[1] + normalized_crop[3] > source_height
                    or normalized_crop[2] != width
                    or normalized_crop[3] != height
                ):
                    raise ValueError(f"{camera_name} view_crop 超出源画面或尺寸不符")
                normalized['view_crop'] = normalized_crop
        return normalized

    def _reset_camera_profile_status(self):
        with self._camera_profile_lock:
            self._camera_profile_status = {
                camera_name: {
                    'expected': deepcopy(profile),
                    'observed': None,
                    'valid': False,
                    'error': None,
                    '_timestamps': [],
                }
                for camera_name, profile in self.camera_stream_profiles.items()
            }
            self._camera_profile_ready_event.clear()
            if not self._camera_profile_status:
                self._camera_profile_ready_event.set()

    def camera_profile_status(self):
        """Return a JSON-safe snapshot used by manifests and the web UI."""
        with self._camera_profile_lock:
            return {
                camera_name: {
                    key: deepcopy(value)
                    for key, value in status.items()
                    if not key.startswith('_')
                }
                for camera_name, status in self._camera_profile_status.items()
            }

    def _observe_camera_profile_frame(
        self,
        camera_name,
        frame_message,
        payload_bytes,
        timestamp,
    ):
        """Validate source encoding/resolution and estimate its frame rate."""
        if camera_name not in self.camera_stream_profiles:
            return
        try:
            timestamp_value = float(timestamp)
            if not math.isfinite(timestamp_value):
                raise ValueError("相机时间戳不是有限数值")
            with self._camera_profile_lock:
                status = self._camera_profile_status[camera_name]
                expected = status['expected']
                if status['error']:
                    raise RuntimeError(status['error'])

                if status['observed'] is None:
                    source_format = str(
                        getattr(frame_message, 'format', '') or ''
                    )
                    if expected['capture_codec'] == 'MJPEG':
                        if (
                            len(payload_bytes) < 4
                            or payload_bytes[:2] != b'\xff\xd8'
                            or payload_bytes[-2:] != b'\xff\xd9'
                        ):
                            raise ValueError("消息不是完整 JPEG 帧")
                        if source_format and not any(
                            token in source_format.lower()
                            for token in ('jpeg', 'jpg', 'mjpeg')
                        ):
                            raise ValueError(
                                "消息 format 不是 JPEG/MJPEG："
                                f"{source_format!r}"
                            )
                        with Image.open(io.BytesIO(payload_bytes)) as image:
                            width, height = image.size
                            decoded_format = str(image.format or '').upper()
                        if decoded_format != 'JPEG':
                            raise ValueError(
                                f"解码格式不是 JPEG：{decoded_format!r}"
                            )
                        if (
                            width != expected['width']
                            or height != expected['height']
                        ):
                            raise ValueError(
                                f"分辨率为 {width}x{height}，要求 "
                                f"{expected['width']}x{expected['height']}"
                            )
                        status['observed'] = {
                            'source_format': source_format or 'jpeg',
                            'storage_encoding': 'jpeg_frames',
                            'width': width,
                            'height': height,
                            'fps': None,
                            'sample_count': 0,
                        }
                    else:
                        if (
                            len(payload_bytes) < 5
                            or b'\x00\x00\x01' not in payload_bytes[:128]
                        ):
                            raise ValueError("消息不是 Annex-B H265 access unit")
                        if source_format and not any(
                            token in source_format.lower()
                            for token in ('h265', 'hevc')
                        ):
                            raise ValueError(
                                f"消息 format 不是 H265/HEVC：{source_format!r}"
                            )
                        width = int(getattr(frame_message, 'width', 0))
                        height = int(getattr(frame_message, 'height', 0))
                        source_width = int(
                            getattr(frame_message, 'source_width', 0)
                        )
                        source_height = int(
                            getattr(frame_message, 'source_height', 0)
                        )
                        view_crop = [
                            int(value)
                            for value in getattr(
                                frame_message,
                                'view_crop',
                                (),
                            )
                        ]
                        if (
                            width != expected['width']
                            or height != expected['height']
                            or source_width != expected['source_width']
                            or source_height != expected['source_height']
                            or (
                                expected.get('view_crop') is not None
                                and view_crop != expected['view_crop']
                            )
                        ):
                            raise ValueError(
                                "逻辑/源分辨率为 "
                                f"{width}x{height}/{source_width}x{source_height}，"
                                "要求 "
                                f"{expected['width']}x{expected['height']}/"
                                f"{expected['source_width']}x"
                                f"{expected['source_height']}"
                            )
                        status['observed'] = {
                            'source_format': source_format or 'h265_annexb',
                            'storage_encoding': 'h265_annexb_access_units',
                            'width': width,
                            'height': height,
                            'source_width': source_width,
                            'source_height': source_height,
                            'view_crop': view_crop,
                            'fps': None,
                            'sample_count': 0,
                        }

                timestamps = status['_timestamps']
                if not timestamps or timestamp_value > timestamps[-1]:
                    timestamps.append(timestamp_value)
                    if len(timestamps) > 60:
                        del timestamps[:-60]
                status['observed']['sample_count'] += 1

                if len(timestamps) >= 30:
                    elapsed = timestamps[-1] - timestamps[0]
                    if elapsed > 0:
                        observed_fps = (len(timestamps) - 1) / elapsed
                        status['observed']['fps'] = observed_fps
                        if abs(observed_fps - expected['fps']) > expected[
                            'fps_tolerance'
                        ]:
                            raise ValueError(
                                f"实测 {observed_fps:.2f} Hz，要求 "
                                f"{expected['fps']:.2f}±"
                                f"{expected['fps_tolerance']:.2f} Hz"
                            )
                        status['valid'] = True
                        if all(
                            item['valid']
                            for item in self._camera_profile_status.values()
                        ):
                            self._camera_profile_ready_event.set()
        except Exception as error:
            message = f"{camera_name} 相机采集契约校验失败：{error}"
            with self._camera_profile_lock:
                if camera_name in self._camera_profile_status:
                    self._camera_profile_status[camera_name]['error'] = message
                    self._camera_profile_status[camera_name]['valid'] = False
            raise RuntimeError(message) from error

    def _assert_camera_profiles_ready(self):
        status = self.camera_profile_status()
        invalid = {
            camera_name: item
            for camera_name, item in status.items()
            if not item.get('valid')
        }
        if invalid:
            details = '; '.join(
                f"{name}: {item.get('error') or '尚未得到足够帧率样本'}"
                for name, item in invalid.items()
            )
            raise RuntimeError(f"RGB 相机未满足采集契约：{details}")

    def _quarantine_orphan_staging_dirs(self):
        """Preserve stale crash leftovers without touching active writers."""
        for staging_dir in sorted(
            self.output_dir.glob(".episode_*.staging")
        ):
            parts = staging_dir.name.split(".")
            owner_token = parts[2] if len(parts) >= 4 else ""
            owner_pid_text = owner_token.split("_", 1)[0]
            owner_alive = False
            try:
                owner_pid = int(owner_pid_text)
                os.kill(owner_pid, 0)
                owner_alive = True
            except (ValueError, ProcessLookupError):
                owner_alive = False
            except PermissionError:
                owner_alive = True

            if owner_alive:
                print(
                    "⚠️  检测到另一个仍在运行的采集进程临时目录，"
                    f"不会触碰：{staging_dir.name}"
                )
                continue

            quarantine_dir = self.output_dir / (
                f"incomplete_{staging_dir.name.lstrip('.')}_"
                f"{time.time_ns()}"
            )
            os.replace(staging_dir, quarantine_dir)
            self._fsync_directory(self.output_dir)
            print(
                "⚠️  检测到异常退出遗留的 Episode 临时目录，"
                f"已保留为：{quarantine_dir.name}"
            )

    def _recover_complete_unindexed_episodes(self):
        """Recover a complete Episode committed just before an interruption."""
        while True:
            episode_id = len(self.dataset_metadata.get('episodes', []))
            episode_dir = self.output_dir / f"episode_{episode_id:04d}"
            if not episode_dir.exists():
                self.episode_count = episode_id
                return

            episode_json_path = episode_dir / "episode.json"
            recovery_error = None
            try:
                with open(
                    episode_json_path,
                    'r',
                    encoding='utf-8',
                ) as f:
                    episode_json = json.load(f)
                if episode_json.get("episode_id") != episode_id:
                    raise ValueError("episode_id 不匹配")
                storage_format = episode_json.get("storage_format")
                recovered_effective_metadata: Dict[str, Any] = {}
                if storage_format == "raw_topics_pickle_v1":
                    raw_topics = episode_json.get("raw_topics", {})
                    if not raw_topics:
                        raise ValueError("raw_topics 为空")
                    if (
                        int(episode_json.get("num_topics", -1))
                        != len(raw_topics)
                    ):
                        raise ValueError("num_topics 与 raw_topics 不一致")

                    has_effective_start = (
                        "effective_start_timestamp" in episode_json
                    )
                    has_effective_end = (
                        "effective_end_timestamp" in episode_json
                    )
                    has_effective_context = (
                        "effective_context_topics" in episode_json
                    )
                    effective_companion_fields = (
                        "effective_duration",
                        "effective_interval_policy",
                        "capture_warmup_seconds",
                    )
                    if has_effective_start != has_effective_end:
                        raise ValueError(
                            "effective start/end 字段不成对"
                        )
                    recovered_effective_start = None
                    recovered_effective_end = None
                    recovered_context_topics = set()
                    if has_effective_start:
                        missing_effective_fields = [
                            field_name
                            for field_name in (
                                "effective_context_topics",
                                *effective_companion_fields,
                            )
                            if field_name not in episode_json
                        ]
                        if missing_effective_fields:
                            raise ValueError(
                                "effective 区间缺少字段："
                                f"{missing_effective_fields}"
                            )
                        if isinstance(
                            episode_json["effective_start_timestamp"],
                            bool,
                        ) or isinstance(
                            episode_json["effective_end_timestamp"],
                            bool,
                        ):
                            raise ValueError(
                                "effective 时间戳不是有效数值"
                            )
                        recovered_effective_start = float(
                            episode_json["effective_start_timestamp"]
                        )
                        recovered_effective_end = float(
                            episode_json["effective_end_timestamp"]
                        )
                        if (
                            not math.isfinite(
                                recovered_effective_start
                            )
                            or not math.isfinite(
                                recovered_effective_end
                            )
                            or recovered_effective_end
                            <= recovered_effective_start
                        ):
                            raise ValueError(
                                "effective 时间区间无效"
                            )
                        if isinstance(
                            episode_json["effective_duration"],
                            bool,
                        ) or isinstance(
                            episode_json["capture_warmup_seconds"],
                            bool,
                        ):
                            raise ValueError(
                                "effective 时长或预采集时长无效"
                            )
                        recovered_effective_duration = float(
                            episode_json["effective_duration"]
                        )
                        recovered_capture_warmup = float(
                            episode_json["capture_warmup_seconds"]
                        )
                        expected_effective_duration = (
                            recovered_effective_end
                            - recovered_effective_start
                        )
                        if (
                            not math.isfinite(
                                recovered_effective_duration
                            )
                            or recovered_effective_duration <= 0
                            or not math.isclose(
                                recovered_effective_duration,
                                expected_effective_duration,
                                rel_tol=1e-12,
                                abs_tol=1e-9,
                            )
                            or not math.isfinite(
                                recovered_capture_warmup
                            )
                            or recovered_capture_warmup < 0
                        ):
                            raise ValueError(
                                "effective 时长或预采集时长无效"
                            )
                        if (
                            episode_json["effective_interval_policy"]
                            != "timestamp_window_with_context_carry_forward"
                        ):
                            raise ValueError(
                                "effective interval policy 无效"
                            )
                        recovered_context_topic_list = episode_json[
                            "effective_context_topics"
                        ]
                        if (
                            not isinstance(
                                recovered_context_topic_list,
                                list,
                            )
                            or any(
                                not isinstance(topic_name, str)
                                or not topic_name
                                for topic_name
                                in recovered_context_topic_list
                            )
                            or len(set(recovered_context_topic_list))
                            != len(recovered_context_topic_list)
                        ):
                            raise ValueError(
                                "effective context topic 字段无效"
                            )
                        recovered_context_topics = set(
                            recovered_context_topic_list
                        )
                        unknown_context_topics = sorted(
                            recovered_context_topics
                            - set(raw_topics)
                        )
                        if unknown_context_topics:
                            raise ValueError(
                                "effective context topic 不存在："
                                f"{unknown_context_topics}"
                            )
                        recovered_effective_metadata = {
                            "effective_start_timestamp": (
                                recovered_effective_start
                            ),
                            "effective_end_timestamp": (
                                recovered_effective_end
                            ),
                            "effective_duration": (
                                recovered_effective_duration
                            ),
                            "effective_interval_policy": (
                                "timestamp_window_with_context_carry_forward"
                            ),
                            "capture_warmup_seconds": (
                                recovered_capture_warmup
                            ),
                            "effective_context_topics": (
                                recovered_context_topic_list
                            ),
                        }
                    elif has_effective_context or any(
                        field_name in episode_json
                        for field_name in effective_companion_fields
                    ):
                        raise ValueError(
                            "没有 effective 区间却存在关联字段"
                        )

                    episode_root = episode_dir.resolve()
                    recovered_total_records = 0
                    for topic_name, topic_info in raw_topics.items():
                        relative_path = Path(topic_info["path"])
                        topic_path = (
                            episode_dir / relative_path
                        ).resolve()
                        if episode_root not in topic_path.parents:
                            raise ValueError(
                                f"topic 路径越界：{topic_name}"
                            )
                        if (
                            int(topic_info.get("record_count", 0)) <= 0
                            or not topic_path.is_file()
                            or topic_path.stat().st_size
                            != int(topic_info.get("size_bytes", -1))
                        ):
                            raise ValueError(
                                f"raw topic 不完整：{topic_name}"
                            )
                        source_count_total = sum(
                            int(count)
                            for count in topic_info.get(
                                "timestamp_source_counts",
                                {},
                            ).values()
                        )
                        if source_count_total != int(
                            topic_info["record_count"]
                        ):
                            raise ValueError(
                                "时间戳来源计数不一致："
                                f"{topic_name}"
                            )
                        if recovered_effective_start is not None:
                            first_timestamp = float(
                                topic_info.get("first_timestamp")
                            )
                            last_timestamp = float(
                                topic_info.get("last_timestamp")
                            )
                            if (
                                not math.isfinite(first_timestamp)
                                or not math.isfinite(last_timestamp)
                                or last_timestamp < first_timestamp
                            ):
                                raise ValueError(
                                    "raw topic first/last 时间戳无效："
                                    f"{topic_name}"
                                )
                            if (
                                first_timestamp
                                > recovered_effective_start
                            ):
                                raise ValueError(
                                    "raw topic 未覆盖 effective start："
                                    f"{topic_name}"
                                )
                            if (
                                topic_name
                                not in recovered_context_topics
                                and last_timestamp
                                < recovered_effective_end
                            ):
                                raise ValueError(
                                    "raw topic 未覆盖 effective end："
                                    f"{topic_name}"
                                )
                        recovered_total_records += int(
                            topic_info["record_count"]
                        )
                    if (
                        int(episode_json.get("total_records", -1))
                        != recovered_total_records
                    ):
                        raise ValueError(
                            "total_records 与逐 topic 计数不一致"
                        )
                    reference_topic = episode_json.get("reference_topic")
                    if reference_topic not in raw_topics:
                        raise ValueError("reference_topic 不存在")
                    num_frames = int(
                        episode_json.get(
                            "reference_sample_count",
                            0,
                        )
                    )
                    if (
                        num_frames
                        != int(
                            raw_topics[reference_topic]["record_count"]
                        )
                    ):
                        raise ValueError(
                            "reference_sample_count 与参考流不一致"
                        )
                    if recovered_effective_start is not None:
                        reference_first_timestamp = float(
                            raw_topics[reference_topic].get(
                                "first_timestamp"
                            )
                        )
                        expected_capture_warmup = max(
                            0.0,
                            recovered_effective_start
                            - reference_first_timestamp,
                        )
                        if not math.isclose(
                            recovered_capture_warmup,
                            expected_capture_warmup,
                            rel_tol=1e-12,
                            abs_tol=1e-9,
                        ):
                            raise ValueError(
                                "capture_warmup_seconds 与参考流"
                                "首时间戳不一致"
                            )
                else:
                    num_frames = int(episode_json["num_frames"])
                    if (
                        num_frames <= 0
                        or len(episode_json.get("frames", []))
                        != num_frames
                    ):
                        raise ValueError("JSON 帧数不完整")
                    if storage_format == "video":
                        video_files = episode_json.get(
                            "video_files",
                            {},
                        )
                        if not video_files:
                            raise ValueError("video_files 为空")
                        for video_file in video_files.values():
                            video_path = episode_dir / video_file
                            if (
                                not video_path.is_file()
                                or video_path.stat().st_size < 1024
                            ):
                                raise ValueError(
                                    f"视频不完整：{video_file}"
                                )
            except Exception as error:
                recovery_error = error

            if recovery_error is not None:
                quarantine_dir = self.output_dir / (
                    f".episode_{episode_id:04d}.incomplete."
                    f"{time.time_ns()}"
                )
                os.replace(episode_dir, quarantine_dir)
                self._fsync_directory(self.output_dir)
                print(
                    "⚠️  发现未完成的 Episode，已保留到隔离目录："
                    f"{quarantine_dir}（{recovery_error}）"
                )
                self.episode_count = episode_id
                return

            recovered_entry = {
                "episode_id": episode_id,
                "model": episode_json.get("model", "unknown"),
                "task": episode_json.get("task", ""),
                "timestamp": episode_json.get(
                    "timestamp",
                    datetime.now().isoformat(),
                ),
                "duration": episode_json.get("duration", 0),
                "path": episode_dir.name,
                "storage_format": episode_json.get(
                    "storage_format",
                ),
            }
            if "capture_duration" in episode_json:
                recovered_entry["capture_duration"] = episode_json[
                    "capture_duration"
                ]
            if (
                episode_json.get("storage_format")
                == "raw_topics_pickle_v1"
            ):
                recovered_entry.update({
                    "reference_topic": episode_json.get(
                        "reference_topic",
                    ),
                    "reference_sample_count": num_frames,
                    "num_topics": episode_json.get("num_topics", 0),
                    "total_records": episode_json.get(
                        "total_records",
                        0,
                    ),
                })
                recovered_entry.update(recovered_effective_metadata)
            else:
                recovered_entry["num_frames"] = num_frames
            self.dataset_metadata['episodes'].append(recovered_entry)
            self._save_metadata()
            print(
                "✓ 已恢复中断前完整保存但尚未登记的 Episode："
                f"{episode_dir.name}"
            )

    def _prepare_episode_staging_dir(self) -> Path:
        """Create or reuse this Episode's private transaction directory."""
        if self._active_episode_staging_dir is not None:
            staging_dir = Path(self._active_episode_staging_dir)
            if not staging_dir.is_dir():
                raise RuntimeError(
                    "Episode 临时目录已丢失："
                    f"{staging_dir}"
                )
            return staging_dir

        episode_id = self.episode_count
        final_episode_dir = (
            self.output_dir / f"episode_{episode_id:04d}"
        )
        if final_episode_dir.exists():
            raise FileExistsError(
                "Episode 目标目录已经存在，拒绝覆盖："
                f"{final_episode_dir}"
            )
        staging_token = f"{os.getpid()}_{time.time_ns()}"
        staging_dir = self.output_dir / (
            f".episode_{episode_id:04d}.{staging_token}.staging"
        )
        staging_dir.mkdir(parents=True, exist_ok=False)
        self._active_episode_staging_dir = staging_dir
        return staging_dir

    def _cleanup_active_episode_staging(self):
        """Remove only the private staging directory owned by this collector."""
        staging_dir = self._active_episode_staging_dir
        if staging_dir is None:
            return
        staging_dir = Path(staging_dir)
        try:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
                print(
                    "  ✓ 已清理保存失败的 Episode 临时目录："
                    f"{staging_dir.name}"
                )
        finally:
            self._active_episode_staging_dir = None

    def _collector_entry(self, target, stop_event):
        """Run one collector with the stop event that belongs to its episode."""
        self._collector_context.stop_event = stop_event
        try:
            target()
        except BaseException as error:
            # Do not turn normal shutdown/cancellation noise into a failed
            # Episode. Any exception raised while this Episode is still active
            # is retained and checked synchronously before save.
            if self.is_recording and not stop_event.is_set():
                failure = {
                    "thread": threading.current_thread().name,
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
                with self._collector_errors_lock:
                    self._collector_errors.append(failure)
                stop_event.set()
                print(
                    "❌ 数据采集线程异常，当前 Episode 将拒绝保存："
                    f"{failure['thread']}：{failure['type']}："
                    f"{failure['message']}"
                )
        finally:
            try:
                del self._collector_context.stop_event
            except AttributeError:
                pass
            try:
                del self._collector_context.timestamp_source
            except AttributeError:
                pass

    def _raise_collector_errors(self):
        """Reject an Episode if any source collector exited abnormally."""
        with self._collector_errors_lock:
            failures = list(self._collector_errors)
        if not failures:
            return
        summary = "; ".join(
            f"{failure['thread']} ({failure['type']}): "
            f"{failure['message']}"
            for failure in failures
        )
        raise RuntimeError(
            "一个或多个采集线程在录制期间异常退出，"
            f"本条数据不会提交：{summary}"
        )

    def has_collection_errors(self):
        """Return whether an active Episode has a recorded source failure."""
        with self._collector_errors_lock:
            return bool(self._collector_errors)

    def _recording_active(self):
        """Return whether the current thread may still write episode data."""
        stop_event = getattr(
            self._collector_context,
            'stop_event',
            self._episode_stop_event,
        )
        return (
            self.is_recording
            and stop_event is not None
            and not stop_event.is_set()
        )

    def _record_raw_topic_sample(
        self,
        topic_name,
        timestamp,
        timestamp_source=None,
    ):
        """Track exact per-topic timestamp diagnostics without aligning data."""
        if not self.raw_topic_storage:
            return
        try:
            timestamp_value = float(timestamp)
        except (TypeError, ValueError):
            timestamp_value = float('nan')
        if timestamp_source is None:
            timestamp_source = getattr(
                self._collector_context,
                "timestamp_source",
                "unspecified",
            )

        with self._raw_topic_stats_lock:
            topic_stats = self._raw_topic_stats.setdefault(
                topic_name,
                {
                    "record_count": 0,
                    "first_timestamp": None,
                    "last_timestamp": None,
                    "timestamp_regressions": 0,
                    "duplicate_timestamps": 0,
                    "nonfinite_timestamps": 0,
                    "timestamp_source_counts": {},
                },
            )
            source_counts = topic_stats["timestamp_source_counts"]
            source_counts[timestamp_source] = (
                int(source_counts.get(timestamp_source, 0)) + 1
            )
            previous_timestamp = topic_stats["last_timestamp"]
            topic_stats["record_count"] += 1
            if not np.isfinite(timestamp_value):
                topic_stats["nonfinite_timestamps"] += 1
                return
            if topic_stats["first_timestamp"] is None:
                topic_stats["first_timestamp"] = timestamp_value
            if previous_timestamp is not None:
                if timestamp_value < previous_timestamp:
                    topic_stats["timestamp_regressions"] += 1
                elif timestamp_value == previous_timestamp:
                    topic_stats["duplicate_timestamps"] += 1
            topic_stats["last_timestamp"] = timestamp_value

    def _maybe_flush_temp_file(self, file_handle, file_key, force=False):
        """Flush a temporary stream periodically instead of on every sample."""
        if file_handle is None or file_handle.closed:
            return
        now = time.monotonic()
        last_flush = self._temp_file_last_flush.get(file_key, 0.0)
        if force or now - last_flush >= self.temp_file_flush_interval:
            file_handle.flush()
            self._temp_file_last_flush[file_key] = now

    def _flush_all_temp_files(self):
        """Flush all episode files after producers have stopped."""
        for camera_name, file_handle in list(self.image_temp_files.items()):
            with self.image_file_locks[camera_name]:
                self._maybe_flush_temp_file(
                    file_handle,
                    f'image:{camera_name}',
                    force=True,
                )
        for sensor_name, file_handle in list(self.sensor_temp_files.items()):
            lock = self.sensor_file_locks.get(sensor_name)
            if lock is None:
                self._maybe_flush_temp_file(
                    file_handle,
                    f'sensor:{sensor_name}',
                    force=True,
                )
            else:
                with lock:
                    self._maybe_flush_temp_file(
                        file_handle,
                        f'sensor:{sensor_name}',
                        force=True,
                    )

    def _close_and_fsync_temp_files(self):
        """Durably close every topic stream after all producers have exited."""
        for camera_name, file_handle in list(
            self.image_temp_files.items()
        ):
            with self.image_file_locks[camera_name]:
                if not file_handle.closed:
                    file_handle.flush()
                    os.fsync(file_handle.fileno())
                    file_handle.close()

        for sensor_name, file_handle in list(
            self.sensor_temp_files.items()
        ):
            lock = self.sensor_file_locks.get(sensor_name)
            if lock is None:
                if not file_handle.closed:
                    file_handle.flush()
                    os.fsync(file_handle.fileno())
                    file_handle.close()
                continue
            with lock:
                if not file_handle.closed:
                    file_handle.flush()
                    os.fsync(file_handle.fileno())
                    file_handle.close()

    def _release_unused_memory(self):
        """Release Python cycles and return free glibc arenas when supported."""
        import gc

        gc.collect()
        try:
            import ctypes

            libc = ctypes.CDLL(None)
            malloc_trim = getattr(libc, 'malloc_trim', None)
            if malloc_trim is not None:
                malloc_trim.argtypes = [ctypes.c_size_t]
                malloc_trim.restype = ctypes.c_int
                malloc_trim(0)
        except Exception:
            # malloc_trim is a Linux/glibc optimization, not a correctness
            # requirement.
            pass

    def _print_resource_usage(self, stage):
        """Print compact per-stage process memory and thread telemetry."""
        status_values = {}
        try:
            with open('/proc/self/status', 'r', encoding='utf-8') as status:
                for line in status:
                    key, _, value = line.partition(':')
                    if key in {'VmRSS', 'VmSwap', 'Threads'}:
                        status_values[key] = value.strip()
        except OSError:
            return

        print(
            f"  [资源] {stage}："
            f"RSS={status_values.get('VmRSS', '未知')}，"
            f"Swap={status_values.get('VmSwap', '未知')}，"
            f"系统线程={status_values.get('Threads', '未知')}，"
            f"Python线程={threading.active_count()}"
        )

    def _wait_for_collection_threads(self):
        """Wait for all collectors before temporary files are read or closed."""
        if self._episode_stop_event is not None:
            self._episode_stop_event.set()

        deadline = time.monotonic() + self.thread_shutdown_timeout
        for thread in list(self.threads):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)

        alive_threads = [thread for thread in self.threads if thread.is_alive()]
        self.threads = alive_threads
        if alive_threads:
            names = ', '.join(thread.name for thread in alive_threads)
            raise RuntimeError(
                f"以下采集线程在 {self.thread_shutdown_timeout:.1f} 秒内未停止："
                f"{names}。为避免跨 Episode 串写，请重启采集程序。"
            )

        self.thread_info = []

    def start_recording(self, task: str = "default_task"):
        """Start recording episode (start all data collection threads)

        Args:
            task: Task description
        """
        if self.is_recording:
            print("⚠️  警告：当前已在采集")
            return

        self._startup_cancel_event.clear()

        self.threads = [thread for thread in self.threads if thread.is_alive()]
        if self.threads:
            names = ', '.join(thread.name for thread in self.threads)
            raise RuntimeError(
                f"上一条数据仍有采集线程未退出：{names}。"
                "为避免旧线程写入新 Episode，请重启采集程序。"
            )
        if (
            self.camera_monitor_thread is not None
            and self.camera_monitor_thread.is_alive()
        ):
            raise RuntimeError("上一条数据的相机监控线程仍未退出")

        episode_stop_event = threading.Event()
        self._episode_stop_event = episode_stop_event

        # Start various data collection threads
        threads = []

        # Joint state collection - start corresponding collection threads according to configured slave_joint_names
        if self.slave_joint_names:
            # Create mapping from joint_name to collection method
            joint_collector_map = {
                'left_arm_joint_states': (self._collect_left_arm_joint_states, "LeftArmJointStateCollector"),
                'right_arm_joint_states': (self._collect_right_arm_joint_states, "RightArmJointStateCollector"),
                'lift_joint_states': (self._collect_lift_joint_states, "LiftJointStateCollector"),
                'waist_joint_states': (self._collect_waist_joint_states, "WaistJointStateCollector"),
                'left_gripper_joint_states': (self._collect_left_gripper_joint_states, "LeftGripperJointStateCollector"),
                'right_gripper_joint_states': (self._collect_right_gripper_joint_states, "RightGripperJointStateCollector"),
                'head_joint_states': (self._collect_head_joint_states, "HeadJointStateCollector"),
            }

            # Start corresponding threads according to configured joint_names
            for joint_name in self.slave_joint_names:
                if joint_name in joint_collector_map:
                    collector_func, thread_name = joint_collector_map[joint_name]
                    threads.append(threading.Thread(target=collector_func, daemon=True, name=thread_name))
                else:
                    raise ValueError(f"Unknown joint state name: {joint_name}. Supported names: {list(joint_collector_map.keys())}")

        # Image stream collection
        if self.collection_config.enable_head_rgb_stream:
            threads.append(threading.Thread(target=self._collect_head_rgb_stream, daemon=True, name="HeadRgbStreamCollector"))

        if self.collection_config.enable_head_depth_stream:
            threads.append(threading.Thread(target=self._collect_head_depth_stream, daemon=True, name="HeadDepthStreamCollector"))

        if self.collection_config.enable_left_arm_rgb_stream:
            threads.append(threading.Thread(target=self._collect_left_arm_rgb_stream, daemon=True, name="LeftArmRgbStreamCollector"))

        if self.collection_config.enable_right_arm_rgb_stream:
            threads.append(threading.Thread(target=self._collect_right_arm_rgb_stream, daemon=True, name="RightArmRgbStreamCollector"))

        # Sensor collection
        if self.collection_config.enable_chassis_imu:
            threads.append(threading.Thread(target=self._collect_imu, daemon=True, name="ImuCollector"))

        if self.collection_config.enable_depth_points:
            threads.append(threading.Thread(target=self._collect_depth, daemon=True, name="DepthCollector"))

        # End effector pose collection
        if self.collection_config.enable_left_arm_end_pose:
            threads.append(threading.Thread(target=self._collect_left_arm_end_pose, daemon=True, name="LeftArmEndPoseCollector"))

        if self.collection_config.enable_right_arm_end_pose:
            threads.append(threading.Thread(target=self._collect_right_arm_end_pose, daemon=True, name="RightArmEndPoseCollector"))

        if self.collection_config.enable_waist_end_pose:
            threads.append(threading.Thread(target=self._collect_waist_end_pose, daemon=True, name="WaistEndPoseCollector"))

        # Chassis sensor collection
        if self.collection_config.enable_odometry:
            threads.append(threading.Thread(target=self._collect_odometry, daemon=True, name="OdometryCollector"))

        if self.collection_config.enable_pose:
            threads.append(threading.Thread(target=self._collect_pose, daemon=True, name="PoseCollector"))

        # Depth and laser sensor collection
        if self.collection_config.enable_head_depth_video:
            threads.append(threading.Thread(target=self._collect_head_depth_video, daemon=True, name="HeadDepthVideoCollector"))

        if self.collection_config.enable_laser_scan:
            threads.append(threading.Thread(target=self._collect_laser_scan, daemon=True, name="LaserScanCollector"))

        # Tactile sensor collection
        if self.collection_config.enable_left_gripper_tactile:
            threads.append(threading.Thread(target=self._collect_left_gripper_tactile, daemon=True, name="LeftGripperTactileCollector"))

        if self.collection_config.enable_right_gripper_tactile:
            threads.append(threading.Thread(target=self._collect_right_gripper_tactile, daemon=True, name="RightGripperTactileCollector"))

        if self.collection_config.enable_left_hand_tactile:
            threads.append(threading.Thread(target=self._collect_left_hand_tactile, daemon=True, name="LeftHandTactileCollector"))

        if self.collection_config.enable_right_hand_tactile:
            threads.append(threading.Thread(target=self._collect_right_hand_tactile, daemon=True, name="RightHandTactileCollector"))

        # Distance sensor collection
        if self.collection_config.enable_tof_sensors:
            threads.append(threading.Thread(target=self._collect_tof_sensors, daemon=True, name="ToFSensorsCollector"))

        if self.collection_config.enable_ultrasonic_sensors:
            threads.append(threading.Thread(target=self._collect_ultrasonic_sensors, daemon=True, name="UltrasonicSensorsCollector"))

        if self.collection_config.enable_master_arm_data:
            threads.append(threading.Thread(target=self._collect_master_left_arm_end_pose, daemon=True, name="MasterLeftArmEndPoseCollector"))
            threads.append(threading.Thread(target=self._collect_master_left_arm_joint_state, daemon=True, name="MasterLeftArmJointStateCollector"))
            threads.append(threading.Thread(target=self._collect_master_right_arm_end_pose, daemon=True, name="MasterRightArmEndPoseCollector"))
            threads.append(threading.Thread(target=self._collect_master_right_arm_joint_state, daemon=True, name="MasterRightArmJointStateCollector"))
            threads.append(threading.Thread(target=self._collect_master_left_gripper_joint_state, daemon=True, name="MasterLeftGripperJointStateCollector"))
            threads.append(threading.Thread(target=self._collect_master_right_gripper_joint_state, daemon=True, name="MasterRightGripperJointStateCollector"))

        if self.collection_config.enable_wrench_ext_world:
            threads.append(threading.Thread(target=self._collect_left_arm_wrench_ext_world, daemon=True, name="LeftArmWrenchExtWorldCollector"))
            threads.append(threading.Thread(target=self._collect_right_arm_wrench_ext_world, daemon=True, name="RightArmWrenchExtWorldCollector"))
        if self.collection_config.enable_wrench_ext_local:
            threads.append(threading.Thread(target=self._collect_left_arm_wrench_ext_local, daemon=True, name="LeftArmWrenchExtLocalCollector"))
            threads.append(threading.Thread(target=self._collect_right_arm_wrench_ext_local, daemon=True, name="RightArmWrenchExtLocalCollector"))

        if self.collection_config.enable_left_gripper_position:
            threads.append(threading.Thread(target=self._collect_left_gripper_position, daemon=True, name="LeftGripperPositionCollector"))
        if self.collection_config.enable_right_gripper_position:
            threads.append(threading.Thread(target=self._collect_right_gripper_position, daemon=True, name="RightGripperPositionCollector"))
        if self.ros_bridge_sensor_names:
            threads.append(threading.Thread(
                target=self._collect_vr_ros_commands,
                daemon=True,
                name="ROSDataBridgeCollector",
            ))

        # Check if at least one data collection is enabled
        if not threads:
            print("错误：未启用任何数据类型，请至少启用一项采集内容")
            return

        # Initialize statistics
        with self.stats_lock:
            self.stats['start_time'] = time.time()
            self.stats['last_update'] = time.time()

        # Bind every collector to this episode's independent stop event.
        raw_threads = threads
        threads = [
            threading.Thread(
                target=self._collector_entry,
                args=(thread._target, episode_stop_event),
                daemon=True,
                name=thread.name,
            )
            for thread in raw_threads
        ]

        # Save thread information
        self.thread_info = []
        for t in threads:
            self.thread_info.append((t.name, t._target))
            self.threads.append(t)

        # Clear queues
        self._clear_queues()

        # Raw-topic diagnostics must be reset before any producer starts.
        with self._raw_topic_stats_lock:
            self._raw_topic_stats = {}
        with self._collector_errors_lock:
            self._collector_errors = []
        self._vr_bridge_ready_event.clear()
        self._revo2_feedback_ready_event.clear()
        with self._ros_bridge_seen_lock:
            self._ros_bridge_seen_sensors = set()
        self._reset_camera_profile_status()

        # Initialize every counter and camera timestamp before producers can
        # write. In particular, waiting for the VR bridge must not leave a
        # window where camera threads see a missing image_count dictionary.
        with self.stats_lock:
            for key in list(self.stats.keys()):
                if key.endswith('_count'):
                    if isinstance(self.stats[key], dict):
                        self.stats[key] = defaultdict(int)
                    else:
                        self.stats[key] = 0
            self.stats.setdefault('state_count', 0)
            self.stats.setdefault('action_count', 0)
            self.stats.setdefault(
                'image_count',
                defaultdict(int),
            )
            self.stats.setdefault('imu_count', 0)
            self.stats.setdefault('depth_count', 0)
            for camera_name in self.camera_names:
                self.camera_last_data_time[camera_name] = None

        # Create temporary files
        self._create_temp_files()

        self.current_episode_task = task
        self.current_episode_start_time = time.time()
        self.current_episode_stop_time = None
        self.is_recording = True

        # Start all threads
        for t in threads:
            t.start()

        readiness_deadline = (
            time.monotonic() + self.startup_readiness_timeout_seconds
            if self.startup_readiness_timeout_seconds is not None
            else None
        )
        readiness_errors = []

        def wait_until(event):
            while True:
                if event.wait(timeout=0.2):
                    return True
                if episode_stop_event.is_set() or not self.is_recording:
                    return False
                if self._startup_cancel_event.is_set():
                    return False
                if (
                    readiness_deadline is not None
                    and time.monotonic() >= readiness_deadline
                ):
                    return False

        if self.ros_bridge_sensor_names and not wait_until(
            self._vr_bridge_ready_event
        ):
            readiness_errors.append("ROS 数据桥未就绪")
        if (
            self.collection_config.enable_revo2_hands
            and not wait_until(self._revo2_feedback_ready_event)
        ):
            readiness_errors.append("双手 Revo2 关节反馈未全部就绪")
        if self.camera_stream_profiles and not wait_until(
            self._camera_profile_ready_event
        ):
            readiness_errors.append("RGB 相机采集契约未通过")

        if readiness_errors:
            self.is_recording = False
            episode_stop_event.set()
            try:
                self._wait_for_collection_threads()
            finally:
                self._cleanup_temp_files()
                self._cleanup_active_episode_staging()
                self.current_episode_task = None
                self.current_episode_start_time = None
                self.current_episode_stop_time = None
            self._raise_collector_errors()
            self._assert_camera_profiles_ready()
            timeout_text = (
                f"在 {self.startup_readiness_timeout_seconds:g} 秒内"
                if self.startup_readiness_timeout_seconds is not None
                else "取消前"
            )
            raise RuntimeError(
                f"采集预检{timeout_text}未通过："
                + "；".join(readiness_errors)
                + "。本条采集未开始，请检查 ROS 2 话题与硬件"
            )

        print(f"✓ 已开始采集第 {self.episode_count} 条数据（任务：{task}）")
        print(f"  - 输出目录：{self.output_dir}")
        if self.raw_topic_storage:
            print("  - 存储模式：各 topic 原始时间戳流")
            print(
                "  - RGB 相机存储：源压缩 bytes（不解码、不转码）"
            )
            print("  - 时间对齐：关闭")
        else:
            print(
                "  - 图像存储："
                f"{'MP4 视频' if self.use_video_storage else 'JPG 图片'}"
            )
            print(f"  - 目标频率：{self.target_hz} Hz")

        # Display enabled data streams
        sensor_count = len(self.sensor_file_locks)
        image_count = len(self.camera_names)

        if sensor_count > 0:
            print(f"  - 传感器数据流：✓（{sensor_count} 路）")
        else:
            print("  - 传感器数据流：✗（未启用）")

        if image_count > 0:
            print(f"  - 图像数据流：✓（{image_count} 路相机）")
        else:
            print("  - 图像数据流：✗（未启用）")

        # Start camera data monitoring thread
        self._start_camera_monitor()
        self._print_resource_usage("录制开始")

    def cancel_startup_wait(self):
        """Wake an indefinite readiness wait without stopping a live episode."""
        self._startup_cancel_event.set()

    def stop_recording(self) -> Optional[Dict[str, Any]]:
        """Stop recording and save episode (stop all data collection threads)

        Returns:
            episode_info: Episode information dictionary, including save path, etc.
        """
        if not self.is_recording:
            print("⚠️  警告：当前未在采集")
            return None

        self.current_episode_stop_time = time.time()
        self.is_recording = False
        if self._episode_stop_event is not None:
            self._episode_stop_event.set()
        print(f"正在停止第 {self.episode_count} 条数据采集...")

        try:
            # Stop every producer before reading or closing temporary files.
            self._stop_camera_monitor()
            print("  正在等待数据采集线程停止...")
            self._wait_for_collection_threads()
            self._raise_collector_errors()
            self._assert_camera_profiles_ready()
            self._flush_all_temp_files()
            print("  ✓ 所有采集线程均已停止，缓存已写入临时文件")
            self._print_resource_usage("采集线程停止")

            if self.raw_topic_storage:
                # Preserve every source stream exactly as recorded. This path
                # intentionally performs no head alignment, interpolation,
                # image decoding, ffmpeg encoding or ffprobe scan.
                self._close_and_fsync_temp_files()
                episode_info = self._save_raw_topic_episode(
                    self.current_episode_task
                )
            else:
                # Collect data (validate data during collection)
                episode_data = self._collect_episode_data()

                if episode_data is None:
                    print("错误：本条数据采集失败，无法读取数据")
                    sys.exit(1)

                # Validate the aligned legacy representation.
                if not self._validate_episode_data(episode_data):
                    print("错误：本条数据校验失败")
                    sys.exit(1)

                episode_info = self._save_episode(
                    episode_data,
                    self.current_episode_task,
                )
            self._release_unused_memory()
            self._print_resource_usage("Episode 保存完成")

            self.episode_count += 1
            self.current_episode_task = None
            self.current_episode_start_time = None
            self.current_episode_stop_time = None

            return episode_info
        finally:
            # Ensure temporary files are cleaned up (even if error occurs)
            try:
                if self._episode_stop_event is not None:
                    self._episode_stop_event.set()
                self._stop_camera_monitor()
                if self.threads:
                    try:
                        self._wait_for_collection_threads()
                    except Exception as error:
                        print(f"  ⚠️  采集线程回收失败：{error}")
                # Clean up temporary files
                self._cleanup_temp_files()
                try:
                    self._cleanup_active_episode_staging()
                except Exception as error:
                    print(
                        "  ⚠️  Episode 临时目录清理失败："
                        f"{error}"
                    )
            except Exception as e:
                print(f"  ⚠️  Error cleaning up resources: {e}")

            print("✓ 数据采集已停止")
            print(f"✓ 当前共保存 {self.episode_count} 条数据")
            print(f"✓ 数据保存目录：{self.output_dir}")

    def discard_recording(self) -> bool:
        """Stop all collectors and discard the current Episode safely."""
        if not self.is_recording:
            print("⚠️  警告：当前未在采集，没有可丢弃的数据")
            return False

        episode_id = self.episode_count
        self.current_episode_stop_time = time.time()
        self.is_recording = False
        if self._episode_stop_event is not None:
            self._episode_stop_event.set()
        print(f"正在停止并丢弃第 {episode_id} 条数据...")

        try:
            self._stop_camera_monitor()
            print("  正在等待数据采集线程停止...")
            self._wait_for_collection_threads()
        except Exception as error:
            raise RuntimeError(
                "丢弃失败：采集线程未能全部停止。"
                "为避免关闭仍在写入的临时文件，程序不会进入下一条；"
                "请重启采集程序。"
            ) from error

        alive_threads = [
            thread.name
            for thread in self.threads
            if thread.is_alive()
        ]
        if alive_threads:
            raise RuntimeError(
                "丢弃失败：仍有采集线程存活："
                f"{alive_threads}。请重启采集程序。"
            )

        print("  ✓ 所有数据采集线程均已停止")
        self._print_resource_usage("本条数据丢弃前")

        # Only remove files after every producer has definitely exited.
        self._cleanup_temp_files()
        self._cleanup_active_episode_staging()
        self._release_unused_memory()
        self.current_episode_task = None
        self.current_episode_start_time = None
        self.current_episode_stop_time = None

        print(f"✓ 第 {episode_id} 条数据已丢弃，未写入数据集")
        print(f"✓ 下一条仍使用编号 {self.episode_count}")
        return True

    def _collect_joint_states(self):
        """Collect full robot joint state data"""
        print("Starting full joint state stream...")

        try:
            stream = self.robot.state.get_all_joint_states_stream(timeout=None)

            for state_msg in stream:
                # If not recording, exit loop
                if not self._recording_active():
                    break

                # If not recording, exit loop (thread will stop)
                if not self._recording_active():
                    break

                try:
                    # Establish joint name mapping when receiving first message
                    if self.joint_name_mapping is None:
                        print(f"  🔍 Analyzing joint state message structure...")
                        print(f"    Message type: {type(state_msg)}")

                        # Check name field
                        has_name = hasattr(state_msg, 'name')
                        print(f"    name field: {'✓' if has_name else '✗'}")
                        if has_name:
                            name_val = getattr(state_msg, 'name', None)
                            if name_val and len(name_val) > 0:
                                print(f"    ✓ Joint names: {len(name_val)} ({name_val[0]}...{name_val[-1]})")
                            else:
                                print(f"    ⚠️  name field is empty or None")

                        # Check position field
                        has_position = hasattr(state_msg, 'position')
                        print(f"    position field: {'✓' if has_position else '✗'}")
                        if has_position:
                            pos_val = getattr(state_msg, 'position', None)
                            if pos_val and len(pos_val) > 0:
                                print(f"    ✓ Joint positions: {len(pos_val)}")
                            else:
                                print(f"    ⚠️  position field is empty or None (message may not be initialized)")

                        joint_names_obtained = False

                        # Prefer joint names from message
                        if hasattr(state_msg, 'name') and state_msg.name and len(state_msg.name) > 0:
                            self.joint_names = list(state_msg.name)
                            self.joint_name_mapping = {}
                            for idx, name in enumerate(state_msg.name):
                                self.joint_name_mapping[name] = idx

                            print(f"  ✅ Joint names set successfully: {len(self.joint_name_mapping)} joints")
                            print(f"     Joint list: {self.joint_names}")
                            joint_names_obtained = True

                        # If no name field, try to infer from position length
                        elif hasattr(state_msg, 'position') and state_msg.position and len(state_msg.position) > 0:
                            num_joints = len(state_msg.position)
                            self.joint_names = [f"joint_{i+1}" for i in range(num_joints)]
                            self.joint_name_mapping = {name: idx for idx, name in enumerate(self.joint_names)}

                            print(f"  ⚠️  Using default joint names: {len(self.joint_name_mapping)} joints")
                            print(f"     Default list: {self.joint_names}")
                            joint_names_obtained = True

                        else:
                            print(f"  ⏳ Waiting for valid joint data... (first message may not be initialized)")

                        # Update metadata
                        if joint_names_obtained and self.joint_names:
                            self.dataset_metadata['joint_names'] = self.joint_names
                            self._save_metadata()
                            print(f"  ✅ Joint mapping configuration completed")
                        elif not joint_names_obtained:
                            print(f"  ⏳ Joint name setting delayed...")

                    # Extract all joint data
                    joint_positions = np.array(state_msg.position, dtype=np.float32).flatten()
                    joint_velocities = np.array(state_msg.velocity, dtype=np.float32).flatten() if hasattr(state_msg, 'velocity') and state_msg.velocity else None
                    joint_efforts = np.array(state_msg.effort, dtype=np.float32).flatten() if hasattr(state_msg, 'effort') and state_msg.effort else None

                    timestamp = self._extract_timestamp_from_header(state_msg)

                    if self._recording_active():
                        # Ensure temporary file is created (only available after start_recording)
                        if hasattr(self, 'sensor_temp_files') and 'joint_states' in self.sensor_temp_files:
                            # Write joint state data to temporary file (avoid queue memory issues)
                            joint_data = (timestamp, joint_positions, joint_velocities, joint_efforts)
                            with self.sensor_file_locks['joint_states']:
                                if 'joint_states' in self.sensor_temp_files:
                                    pickle.dump(joint_data, self.sensor_temp_files['joint_states'])
                                    self._maybe_flush_temp_file(
                                        self.sensor_temp_files['joint_states'],
                                        'sensor:joint_states',
                                    )
                                    self._record_raw_topic_sample(
                                        'joint_states',
                                        timestamp,
                                    )

                            # Also write action data to file (same as state, for imitation learning)
                            action_data = (timestamp, joint_positions)
                            with self.sensor_file_locks['actions']:
                                if 'actions' in self.sensor_temp_files:
                                    pickle.dump(action_data, self.sensor_temp_files['actions'])
                                    self._maybe_flush_temp_file(
                                        self.sensor_temp_files['actions'],
                                        'sensor:actions',
                                    )
                                    self._record_raw_topic_sample(
                                        'actions',
                                        timestamp,
                                    )

                        with self.stats_lock:
                            self.stats['state_count'] += 1
                            self.stats['action_count'] += 1

                except Exception as e:
                    print(f"Joint state data processing error: {e}")
                    continue

        except Exception as e:
            print(f"Joint state stream error: {e}")
            raise

    def _collect_left_arm_joint_states(self):
        """Collect left arm joint state data"""
        print("Starting left arm joint state stream...")
        self._collect_joint_state_stream('left_arm_joint_states', self.robot.left_arm.get_joint_states_stream)

    def _collect_right_arm_joint_states(self):
        """Collect right arm joint state data"""
        print("Starting right arm joint state stream...")
        self._collect_joint_state_stream('right_arm_joint_states', self.robot.right_arm.get_joint_states_stream)

    def _collect_lift_joint_states(self):
        """Collect waist joint state data"""
        print("Starting waist joint state stream...")
        self._collect_joint_state_stream('lift_joint_states', self.robot.lift.get_joint_states_stream)

    def _collect_left_gripper_joint_states(self):
        """Collect left gripper joint state data"""
        print("Starting left gripper joint state stream...")
        self._collect_joint_state_stream('left_gripper_joint_states', self.robot.left_gripper.get_joint_states_stream)

    def _collect_right_gripper_joint_states(self):
        """Collect right gripper joint state data"""
        print("Starting right gripper joint state stream...")
        self._collect_joint_state_stream('right_gripper_joint_states', self.robot.right_gripper.get_joint_states_stream)

    def _collect_head_joint_states(self):
        """Collect head joint state data"""
        print("Starting head joint state stream...")
        self._collect_joint_state_stream('head_joint_states', self.robot.head.get_joint_states_stream)


    def _collect_waist_joint_states(self):
        """Collect waist joint state data"""
        print("Starting waist joint state stream...")
        self._collect_joint_state_stream('waist_joint_states', self.robot.waist.get_joint_states_stream)


    # ============ Image Stream Collection Methods ============

    def _collect_head_rgb_stream(self):
        """Collect head RGB video stream"""
        print("正在启动头部 RGB 视频流...")
        self._collect_camera_stream('head_rgb_stream', self.robot.head_camera.get_rgb_video_stream)

    def _collect_head_depth_stream(self):
        """Collect head depth video stream"""
        print("Starting head depth video stream...")
        self._collect_depth_stream('head_depth_stream', self.robot.head_camera.get_depth_video_stream)

    def _collect_left_arm_rgb_stream(self):
        """Collect left arm RGB video stream"""
        print("正在启动左臂 RGB 视频流...")
        self._collect_camera_stream('left_arm_rgb_stream', self.robot.left_arm_camera.get_video_stream)

    def _collect_right_arm_rgb_stream(self):
        """Collect right arm RGB video stream"""
        print("正在启动右臂 RGB 视频流...")
        self._collect_camera_stream('right_arm_rgb_stream', self.robot.right_arm_camera.get_video_stream)

    def _image_to_jpeg_bytes(self, image_data):
        """Return JPEG bytes, passing existing JPEG payloads through unchanged."""
        if isinstance(image_data, (bytes, bytearray, memoryview)):
            raw_bytes = bytes(image_data)
            if len(raw_bytes) >= 4 and raw_bytes[:2] == b'\xff\xd8':
                return raw_bytes
            with Image.open(io.BytesIO(raw_bytes)) as source_image:
                rgb_image = source_image.convert('RGB')
                try:
                    output = io.BytesIO()
                    rgb_image.save(
                        output,
                        'JPEG',
                        quality=self.image_quality,
                    )
                    return output.getvalue()
                finally:
                    rgb_image.close()

        if isinstance(image_data, np.ndarray):
            source_image = Image.fromarray(image_data)
            close_source = True
        elif isinstance(image_data, Image.Image):
            source_image = image_data
            close_source = False
        else:
            raise TypeError(f"不支持的图像类型：{type(image_data)}")

        converted_image = None
        try:
            image_to_save = source_image
            if source_image.mode != 'RGB':
                converted_image = source_image.convert('RGB')
                image_to_save = converted_image
            output = io.BytesIO()
            image_to_save.save(
                output,
                'JPEG',
                quality=self.image_quality,
            )
            return output.getvalue()
        finally:
            if converted_image is not None:
                converted_image.close()
            if close_source:
                source_image.close()

    def _collect_camera_stream(self, camera_name, stream_func):
        """Collect video stream from a single camera"""
        print(f"正在启动 {camera_name} 数据流...")

        try:
            stream = stream_func(timeout=None)

            for frame_msg in stream:
                # If not recording, exit loop
                if not self._recording_active():
                    break

                # If not recording, exit loop (thread will stop)
                if not self._recording_active():
                    break

                try:
                    # Check if data is empty
                    if not frame_msg or not frame_msg.data:
                        continue

                    # Raw-topic mode preserves the camera's compressed payload:
                    # no PIL/OpenCV decode and no re-encoding. Wrist SDK streams
                    # provide JPEG; E6 provides H265 Annex-B access units.
                    if self.raw_topic_storage:
                        if not isinstance(
                            frame_msg.data,
                            (bytes, bytearray, memoryview),
                        ):
                            raise TypeError(
                                "raw topic 模式要求相机消息提供原始 bytes，"
                                f"实际为 {type(frame_msg.data)}"
                            )
                        img_bytes_compressed = bytes(frame_msg.data)
                    else:
                        # Legacy mode retains its compatibility conversion for
                        # non-JPEG image representations.
                        img_bytes_compressed = self._image_to_jpeg_bytes(
                            frame_msg.data,
                        )

                    timestamp = self._extract_timestamp_from_header(frame_msg)
                    self._observe_camera_profile_frame(
                        camera_name,
                        frame_msg,
                        img_bytes_compressed,
                        timestamp,
                    )

                    if self._recording_active():
                        try:
                            with self.image_file_locks[camera_name]:
                                if camera_name in self.image_temp_files:
                                    pickle.dump((timestamp, img_bytes_compressed), self.image_temp_files[camera_name])
                                    self._maybe_flush_temp_file(
                                        self.image_temp_files[camera_name],
                                        f'image:{camera_name}',
                                    )
                                    self._record_raw_topic_sample(
                                        camera_name,
                                        timestamp,
                                    )

                                    with self.stats_lock:
                                        self.stats['image_count'][camera_name] += 1
                                        # Update last time camera received data
                                        self.camera_last_data_time[camera_name] = time.time()
                        except Exception as e:
                            print(f"❌ {camera_name} data processing error: {e}")
                            if hasattr(frame_msg, 'data'):
                                print(f"   数据类型：{type(frame_msg.data)}")
                                try:
                                    print(
                                        "   数据大小："
                                        f"{len(frame_msg.data)} bytes/items"
                                    )
                                except TypeError:
                                    pass
                            else:
                                print("   消息没有 data 字段")
                            raise RuntimeError(f"{camera_name} data processing failed: {e}")

                except Exception as e:
                    if self._is_grpc_connection_error(e):
                        self._handle_grpc_error_and_exit(camera_name, e)
                    print(f"❌ {camera_name} stream processing error: {e}, time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
                    raise RuntimeError(f"{camera_name} stream processing failed: {e}")

        except Exception as e:
            if self._is_grpc_connection_error(e):
                self._handle_grpc_error_and_exit(camera_name, e)
            print(f"❌ {camera_name} stream processing error: {e}, time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
            raise RuntimeError(f"{camera_name} stream processing failed: {e}")

    def _collect_depth_stream(self, camera_name, stream_func):
        """Collect depth video stream"""
        print(f"正在启动 {camera_name} 数据流...")

        try:
            stream = stream_func(timeout=None)

            for frame_msg in stream:
                # If not recording, exit loop
                if not self._recording_active():
                    break

                # If not recording, exit loop (thread will stop)
                if not self._recording_active():
                    break

                try:
                    # Check if data is empty
                    if not frame_msg or not frame_msg.data:
                        continue

                    depth_bytes = bytes(frame_msg.data)
                    timestamp = self._extract_timestamp_from_header(frame_msg)

                    if self._recording_active():
                        # Try multiple methods to process depth data

                        depth_data = None

                        # Method 1: Try to process as compressed image
                        try:
                            depth_img = Image.open(io.BytesIO(depth_bytes))
                            # Convert to numpy array
                            depth_data = np.array(depth_img, dtype=np.float32)
                            print(f"{camera_name} depth data parsed as image: {depth_data.shape}")
                        except Exception:
                            # Method 2: Try to process as raw float32 array
                            try:
                                # Check if data length is a multiple of 4
                                if len(depth_bytes) % 4 == 0:
                                    num_pixels = len(depth_bytes) // 4
                                    depth_values = struct.unpack(f'{num_pixels}f', depth_bytes)
                                    depth_data = np.array(depth_values, dtype=np.float32)

                                    # Try common depth map resolutions
                                    if len(depth_data) == 640 * 480:
                                        depth_data = depth_data.reshape(480, 640)
                                    elif len(depth_data) == 320 * 240:
                                        depth_data = depth_data.reshape(240, 320)
                                    elif len(depth_data) == 1280 * 720:
                                        depth_data = depth_data.reshape(720, 1280)
                                    elif len(depth_data) == 640 * 360:
                                        depth_data = depth_data.reshape(360, 640)
                                    # If not standard resolution, keep as 1D array
                                    print(f"{camera_name} depth data parsed as float32 array: {depth_data.shape}")
                                else:
                                    raise ValueError("Data length is not a multiple of 4")
                            except Exception:
                                # Method 3: Save raw byte data
                                depth_data = depth_bytes
                                print(f"{camera_name} saving raw depth data: {len(depth_bytes)} bytes")

                        # Save processed depth data
                        with self.image_file_locks[camera_name]:
                            if camera_name in self.image_temp_files:
                                pickle.dump((timestamp, depth_data), self.image_temp_files[camera_name])
                                self._maybe_flush_temp_file(
                                    self.image_temp_files[camera_name],
                                    f'image:{camera_name}',
                                )
                                self._record_raw_topic_sample(
                                    camera_name,
                                    timestamp,
                                )

                        with self.stats_lock:
                            self.stats['image_count'][camera_name] += 1
                            # Update last time camera received data
                            self.camera_last_data_time[camera_name] = time.time()

                except Exception as e:
                    if self._is_grpc_connection_error(e):
                        self._handle_grpc_error_and_exit(camera_name, e)
                    print(f"❌ {camera_name} data processing error: {e}, time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
                    print(f"   Data type: {type(frame_msg.data) if hasattr(frame_msg, 'data') else 'No data attribute'}")
                    print(f"   Data content: {repr(frame_msg.data) if hasattr(frame_msg, 'data') else 'No data attribute'}")
                    raise RuntimeError(f"{camera_name} data processing failed: {e}")

        except Exception as e:
            if self._is_grpc_connection_error(e):
                self._handle_grpc_error_and_exit(camera_name, e)
            # Other types of errors, re-raise
            print(f"❌ {camera_name} stream error: {e}, time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
            raise

    def _collect_imu(self):
        """Collect IMU data"""
        print("Starting IMU data stream...")

        try:
            stream = self.robot.imu.get_chassis_imu_stream(timeout=None)

            for imu_msg in stream:
                # If not recording, exit loop
                if not self._recording_active():
                    break

                # If not recording, exit loop (thread will stop)
                if not self._recording_active():
                    break

                try:
                    imu_data = {
                        'orientation': [
                            float(imu_msg.orientation.x),
                            float(imu_msg.orientation.y),
                            float(imu_msg.orientation.z),
                            float(imu_msg.orientation.w)
                        ],
                        'angular_velocity': [
                            float(imu_msg.angular_velocity.x),
                            float(imu_msg.angular_velocity.y),
                            float(imu_msg.angular_velocity.z)
                        ],
                        'linear_acceleration': [
                            float(imu_msg.linear_acceleration.x),
                            float(imu_msg.linear_acceleration.y),
                            float(imu_msg.linear_acceleration.z)
                        ]
                    }

                    timestamp = self._extract_timestamp_from_header(imu_msg)

                    if self._recording_active():
                        # Ensure temporary file is created (available after start_recording)
                        if hasattr(self, 'sensor_temp_files') and 'chassis_imu' in self.sensor_temp_files:
                            imu_data_tuple = (timestamp, imu_data)
                            with self.sensor_file_locks['chassis_imu']:
                                if 'chassis_imu' in self.sensor_temp_files:
                                    pickle.dump(imu_data_tuple, self.sensor_temp_files['chassis_imu'])
                                    self._maybe_flush_temp_file(
                                        self.sensor_temp_files['chassis_imu'],
                                        'sensor:chassis_imu',
                                    )
                                    self._record_raw_topic_sample(
                                        'chassis_imu',
                                        timestamp,
                                    )
                        with self.stats_lock:
                            self.stats['imu_count'] += 1

                except Exception as e:
                    if self._is_grpc_connection_error(e):
                        self._handle_grpc_error_and_exit('chassis_imu', e)
                    print(f"IMU data processing error: {e}, time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    continue

        except Exception as e:
            if self._is_grpc_connection_error(e):
                self._handle_grpc_error_and_exit('chassis_imu', e)
            print(f"IMU stream error: {e}, time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            raise

    def _collect_depth(self):
        """Collect depth data"""
        print("Starting depth data stream...")

        try:
            stream = self.robot.depth_points.get_chassis_depth_points_stream(timeout=None)

            for depth_msg in stream:
                # If not recording, exit loop
                if not self._recording_active():
                    break

                # If not recording, exit loop (thread will stop)
                if not self._recording_active():
                    break

                try:
                    timestamp = self._extract_timestamp_from_header(depth_msg)

                    if self._recording_active():
                        depth_info = {
                            'timestamp': timestamp,
                            'width': depth_msg.width,
                            'height': depth_msg.height,
                            'point_count': len(depth_msg.data) if hasattr(depth_msg, 'data') else 0
                        }
                        # Ensure temporary file is created (available after start_recording)
                        if hasattr(self, 'sensor_temp_files') and 'depth_points' in self.sensor_temp_files:
                            depth_data_tuple = (timestamp, depth_info)
                            with self.sensor_file_locks['depth_points']:
                                if 'depth_points' in self.sensor_temp_files:
                                    pickle.dump(depth_data_tuple, self.sensor_temp_files['depth_points'])
                                    self._maybe_flush_temp_file(
                                        self.sensor_temp_files['depth_points'],
                                        'sensor:depth_points',
                                    )
                                    self._record_raw_topic_sample(
                                        'depth_points',
                                        timestamp,
                                    )
                        with self.stats_lock:
                            self.stats['depth_count'] += 1

                except Exception as e:
                    print(f"Depth data processing error: {e}, time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    continue

        except Exception as e:
            print(f"Depth stream error: {e}, time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            raise

    # ============ End effector pose collection methods ============

    def _collect_left_arm_end_pose(self):
        """Collect left arm end effector pose"""
        print("正在启动左臂末端位姿数据流...")
        self._collect_pose_stream('left_arm_end_pose', self.robot.left_arm.get_end_pose_stream)

    def _collect_right_arm_end_pose(self):
        """Collect right arm end effector pose"""
        print("正在启动右臂末端位姿数据流...")
        self._collect_pose_stream('right_arm_end_pose', self.robot.right_arm.get_end_pose_stream)

    def _collect_waist_end_pose(self):
        """Collect waist end effector pose"""
        print("Starting waist end effector pose stream...")
        self._collect_pose_stream('waist_end_pose', self.robot.waist.get_end_pose_stream)

    # ============ Chassis sensor collection methods ============

    def _collect_odometry(self):
        """Collect chassis odometry data"""
        print("Starting odometry stream...")
        self._collect_generic_stream('odometry', self.robot.chassis.get_odometry_stream)

    def _collect_pose(self):
        """Collect robot pose data"""
        print("Starting pose data stream...")
        self._collect_generic_stream('pose', self.robot.chassis.get_pose_stream)

    # ============ Depth and LiDAR sensor collection methods ============

    def _collect_head_depth_video(self):
        """Collect head depth video stream"""
        print("Starting head depth video stream...")
        self._collect_camera_stream('head_depth_video', self.robot.head_camera.get_depth_video_stream)

    def _collect_laser_scan(self):
        """Collect LiDAR scan data"""
        print("Starting LiDAR stream...")
        self._collect_generic_stream('laser_scan', self.robot.radar.get_laser_scan_stream)

    # ============ Tactile sensor collection methods ============

    def _collect_left_gripper_tactile(self):
        """Collect left gripper tactile data"""
        print("Starting left gripper tactile sensor...")
        if hasattr(self.robot, 'left_gripper_tactile'):
            self._collect_generic_stream('left_gripper_tactile', self.robot.left_gripper_tactile.get_tactile_sensor_data_stream)
        else:
            raise RuntimeError("Left gripper tactile sensor not available (only quanta_x2 model supported)")

    def _collect_right_gripper_tactile(self):
        """Collect right gripper tactile data"""
        print("Starting right gripper tactile sensor...")
        if hasattr(self.robot, 'right_gripper_tactile'):
            self._collect_generic_stream('right_gripper_tactile', self.robot.right_gripper_tactile.get_tactile_sensor_data_stream)
        else:
            raise RuntimeError("Right gripper tactile sensor not available (only quanta_x2 model supported)")

    def _collect_left_hand_tactile(self):
        """Collect left dexterous hand tactile data"""
        print("Starting left dexterous hand tactile sensor...")
        if hasattr(self.robot, 'left_hand_tactile'):
            self._collect_generic_stream('left_hand_tactile', self.robot.left_hand_tactile.get_tactile_sensor_data_stream)
        else:
            raise RuntimeError("Left dexterous hand tactile sensor not available (only quanta_x2 model supported)")

    def _collect_right_hand_tactile(self):
        """Collect right dexterous hand tactile data"""
        print("Starting right dexterous hand tactile sensor...")
        if hasattr(self.robot, 'right_hand_tactile'):
            self._collect_generic_stream('right_hand_tactile', self.robot.right_hand_tactile.get_tactile_sensor_data_stream)
        else:
            raise RuntimeError("Right dexterous hand tactile sensor not available (only quanta_x2 model supported)")

    def _collect_master_left_arm_end_pose(self):
        """Collect master left arm end pose control command"""
        print("Starting master left arm end pose control command collection...")
        try:
            self._collect_pose_stream('master_left_arm_end_pose', self.robot.master_left_arm.get_end_pose_stream)  # Placeholder
        except Exception as e:
            print(f"Master left arm end pose control command collection failed: {e}")
            raise

    def _collect_master_right_arm_end_pose(self):
        """Collect master right arm end pose control command"""
        print("Starting master right arm end pose control command collection...")
        try:
            self._collect_pose_stream('master_right_arm_end_pose', self.robot.master_right_arm.get_end_pose_stream)  # Placeholder
        except Exception as e:
            print(f"Master right arm end pose control command collection failed: {e}")
            raise

    def _collect_master_left_arm_joint_state(self):
        """Collect master left arm joint control command"""
        print("Starting master left arm joint control command collection...")
        try:
            self._collect_joint_state_stream('master_left_arm_joint_state', self.robot.master_left_arm.get_joint_states_stream)  # Placeholder
        except Exception as e:
            print(f"Master left arm joint control command collection failed: {e}")
            raise

    def _collect_master_right_arm_joint_state(self):
        """Collect master right arm joint control command"""
        print("Starting master right arm joint control command collection...")
        try:
            self._collect_joint_state_stream('master_right_arm_joint_state', self.robot.master_right_arm.get_joint_states_stream)  # Placeholder
        except Exception as e:
            print(f"Master right arm joint control command collection failed: {e}")
            raise

    def _collect_master_left_gripper_joint_state(self):
        """Collect master left gripper joint control command"""
        print("Starting master left gripper joint control command collection...")
        try:
            self._collect_joint_state_stream('master_left_gripper_joint_state', self.robot.master_left_arm.get_gripper_joint_states_stream)  # Placeholder
        except Exception as e:
            print(f"Master left gripper joint control command collection failed: {e}")
            raise

    def _collect_master_right_gripper_joint_state(self):
        """Collect master right gripper joint control command"""
        print("Starting master right gripper joint control command collection...")
        try:
            self._collect_joint_state_stream('master_right_gripper_joint_state', self.robot.master_right_arm.get_gripper_joint_states_stream)  # Placeholder
        except Exception as e:
            print(f"Master right gripper joint control command collection failed: {e}")
            raise

    def _collect_left_arm_wrench_ext_world(self):
        """Collect master left arm wrench ext world control command"""
        print("Starting master left arm wrench ext world control command collection...")
        try:
            self._collect_generic_stream('left_arm_wrench_ext_world', self.robot.left_arm.get_wrench_ext_world_stream)  # Placeholder
        except Exception as e:
            print(f"Master left arm wrench ext world control command collection failed: {e}")
            raise

    def _collect_left_arm_wrench_ext_local(self):
        """Collect master left arm wrench ext local control command"""
        print("Starting master left arm wrench ext local control command collection...")
        try:
            self._collect_generic_stream('left_arm_wrench_ext_local', self.robot.left_arm.get_wrench_ext_local_stream)  # Placeholder
        except Exception as e:
            print(f"Master left arm wrench ext local control command collection failed: {e}")
            raise

    def _collect_right_arm_wrench_ext_world(self):
        """Collect master right arm wrench ext world control command"""
        print("Starting master right arm wrench ext world control command collection...")
        try:
            self._collect_generic_stream('right_arm_wrench_ext_world', self.robot.right_arm.get_wrench_ext_world_stream)  # Placeholder
        except Exception as e:
            print(f"Master right arm wrench ext world control command collection failed: {e}")
            raise

    def _collect_right_arm_wrench_ext_local(self):
        """Collect master right arm wrench ext local control command"""
        print("Starting master right arm wrench ext local control command collection...")
        try:
            self._collect_generic_stream('right_arm_wrench_ext_local', self.robot.right_arm.get_wrench_ext_local_stream)  # Placeholder
        except Exception as e:
            print(f"Master right arm wrench ext local control command collection failed: {e}")
            raise

    def _collect_left_gripper_position(self):
        """Collect left gripper position"""
        print("正在采集左夹爪实测位置...")
        try:
            self._collect_gripper_position_stream(
                'left_gripper_position',
                self.robot.left_gripper.get_position_stream,
            )
        except Exception as e:
            print(f"左夹爪实测位置采集失败：{e}")
            raise

    def _collect_right_gripper_position(self):
        """Collect right gripper position"""
        print("正在采集右夹爪实测位置...")
        try:
            self._collect_gripper_position_stream(
                'right_gripper_position',
                self.robot.right_gripper.get_position_stream,
            )
        except Exception as e:
            print(f"右夹爪实测位置采集失败：{e}")
            raise

    @staticmethod
    def _gripper_position_as_float(message):
        """Convert an SDK gripper position message to a portable scalar."""
        if isinstance(message, dict):
            value = message.get("position")
        else:
            value = getattr(message, "position", None)
            if value is None:
                value = getattr(message, "data", None)
        if value is None:
            raise ValueError("夹爪位置消息没有 position/data 字段")
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                raise ValueError("夹爪位置数组为空")
            value = value[0]
        position = float(value)
        if not np.isfinite(position):
            raise ValueError(f"夹爪位置不是有限数值：{position}")
        return position

    def _collect_gripper_position_stream(self, queue_name, stream_func):
        """Collect measured gripper state as a portable primitive dict."""
        try:
            stream = stream_func(timeout=None)
            for message in stream:
                if not self._recording_active():
                    break
                try:
                    timestamp = self._extract_timestamp_from_header(message)
                    position_data = {
                        "position": self._gripper_position_as_float(message)
                    }
                    if not self._recording_active():
                        break
                    with self.sensor_file_locks[queue_name]:
                        output = self.sensor_temp_files.get(queue_name)
                        if output is None or output.closed:
                            continue
                        pickle.dump((timestamp, position_data), output)
                        self._maybe_flush_temp_file(
                            output,
                            f"sensor:{queue_name}",
                        )
                        self._record_raw_topic_sample(
                            queue_name,
                            timestamp,
                        )
                    with self.stats_lock:
                        count_key = f"{queue_name}_count"
                        self.stats[count_key] = (
                            self.stats.get(count_key, 0) + 1
                        )
                except Exception as error:
                    raise RuntimeError(
                        f"{queue_name} 单条数据处理失败：{error}"
                    ) from error
        except Exception as error:
            if self._is_grpc_connection_error(error):
                self._handle_grpc_error_and_exit(queue_name, error)
            print(f"{queue_name} 数据流失败：{error}")
            raise

    def _observe_ros_bridge_sample(self, sensor_name, timestamp, payload):
        """Optional hook used by the collection website for live values."""

    def _collect_vr_ros_commands(self):
        """Collect VR commands plus Revo2 joint feedback from ROS JSON topics.

        The SDK DataCollection GetVR* streams are present in x2robot 1.0.5 but
        do not publish data on this robot.  The Revo2 control bridge is also
        the only process allowed to own the two RS485 ports; it publishes
        measured motor data and final six-axis commands on ROS.  Run
        the subscribers in a child process so the SDK Python environment does
        not need to source ROS 2 itself.
        """
        import select

        def write_sample(sensor_name, value):
            if not self._recording_active():
                return
            lock = self.sensor_file_locks.get(sensor_name)
            output = self.sensor_temp_files.get(sensor_name)
            if lock is None or output is None or output.closed:
                return
            payload = value['data']
            if isinstance(payload, dict):
                payload = dict(payload)
                payload['_timestamp_meta'] = {
                    'source': value.get(
                        'timestamp_source',
                        'receipt_wall_clock',
                    ),
                    'receive_timestamp': value.get(
                        'receive_timestamp',
                        value['timestamp'],
                    ),
                }
            with lock:
                if output.closed:
                    return
                pickle.dump((value['timestamp'], payload), output)
                self._maybe_flush_temp_file(
                    output,
                    f'sensor:{sensor_name}',
                )
                self._record_raw_topic_sample(
                    sensor_name,
                    value['timestamp'],
                    value.get(
                        'timestamp_source',
                        'receipt_wall_clock',
                    ),
                )
            with self.stats_lock:
                count_key = f'{sensor_name}_count'
                self.stats[count_key] = self.stats.get(count_key, 0) + 1

            try:
                self._observe_ros_bridge_sample(
                    sensor_name,
                    value['timestamp'],
                    payload,
                )
            except Exception as error:
                print(
                    f"⚠️  ROS 数据桥网页观察钩子失败：{error}"
                )

            if sensor_name in self.revo2_state_sensor_names:
                with self._ros_bridge_seen_lock:
                    self._ros_bridge_seen_sensors.add(sensor_name)
                    if set(self.revo2_state_sensor_names).issubset(
                        self._ros_bridge_seen_sensors
                    ):
                        self._revo2_feedback_ready_event.set()

        print("正在采集 ROS 最终控制指令与 Revo2 关节反馈（不含触觉）...")
        ros_topics = {**VR_ROS_COMMAND_TOPICS, **REVO2_ROS_TOPICS}
        for sensor_name in self.ros_bridge_sensor_names:
            topic = ros_topics[sensor_name]
            print(f"  {sensor_name} <- {topic}")

        ros_setup = Path('/opt/ros/jazzy/setup.bash')
        helper = Path(__file__).with_name('vr_ros_command_bridge.py')
        if not ros_setup.exists():
            raise RuntimeError(f"找不到 ROS 2 环境文件：{ros_setup}")
        if not helper.exists():
            raise RuntimeError(f"找不到 VR ROS 指令采集辅助程序：{helper}")

        process = None
        stderr_file = None

        def read_bridge_stderr():
            if stderr_file is None:
                return ""
            stderr_file.flush()
            stderr_file.seek(0, os.SEEK_END)
            size = stderr_file.tell()
            stderr_file.seek(max(0, size - 4096), os.SEEK_SET)
            return stderr_file.read().strip()

        try:
            # Resolve the environment in a separate shell. LD_LIBRARY_PATH must
            # be present before the child Python process starts; changing it in
            # this already-running SDK process is too late for rclpy's shared
            # libraries.
            env_blob = subprocess.check_output(
                [
                    'bash',
                    '-c',
                    f'source {ros_setup} >/dev/null 2>&1 && env -0',
                ],
            )
            ros_environment = os.environ.copy()
            for entry in env_blob.split(b'\0'):
                if not entry or b'=' not in entry:
                    continue
                key, value = entry.split(b'=', 1)
                ros_environment[os.fsdecode(key)] = os.fsdecode(value)

            # A regular temporary file cannot fill a pipe and deadlock a
            # long-running rclpy helper. Only the final 4 KiB are read if the
            # helper exits unexpectedly.
            stderr_file = tempfile.TemporaryFile(
                mode='w+',
                encoding='utf-8',
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(helper),
                    '--sensors',
                    *self.ros_bridge_sensor_names,
                ],
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                bufsize=1,
                env=ros_environment,
            )
            while self._recording_active():
                if process.poll() is not None:
                    stderr = read_bridge_stderr()
                    raise RuntimeError(
                        "VR ROS 指令采集辅助程序意外退出"
                        + (f"：{stderr}" if stderr else "")
                    )
                readable, _, _ = select.select(
                    [process.stdout],
                    [],
                    [],
                    0.1,
                )
                if not readable:
                    continue
                line = process.stdout.readline()
                if not line:
                    continue
                sample = json.loads(line)
                if sample.get('event') == 'ready':
                    ready_topics = sample.get('topics')
                    if (
                        not isinstance(ready_topics, dict)
                        or set(ready_topics) != set(self.ros_bridge_sensor_names)
                    ):
                        raise RuntimeError(
                            'ROS 数据桥订阅集合与采集配置不一致：'
                            f'期望 {sorted(self.ros_bridge_sensor_names)}，'
                            f'实际 {sorted(ready_topics or {})}'
                        )
                    self._vr_bridge_ready_event.set()
                    print("  ✓ ROS 数据桥已就绪")
                    continue
                sensor_name = sample.get('sensor_name')
                if sensor_name not in self.ros_bridge_sensor_names:
                    continue
                write_sample(sensor_name, sample)
        except Exception as error:
            print(f"ROS 指令/Revo2 反馈采集失败：{error}")
            raise
        finally:
            try:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1.0)
            finally:
                if stderr_file is not None:
                    stderr_file.close()

    def _collect_vr_left_arm_pose_commands(self):
        """Legacy SDK VR stream collector (not used on this robot)."""
        print("Starting VR left arm pose control command collection...")
        try:
            self._collect_pose_stream(
                'vr_left_arm_pose_commands',
                self.robot.action_data_collection.get_vr_left_arm_pose_stream,
            )
        except Exception as e:
            print(f"VR left arm pose control command collection failed: {e}")
            raise

    def _collect_vr_right_arm_pose_commands(self):
        """Collect VR right arm pose control command"""
        print("Starting VR right arm pose control command collection...")
        try:
            self._collect_pose_stream(
                'vr_right_arm_pose_commands',
                self.robot.action_data_collection.get_vr_right_arm_pose_stream,
            )
        except Exception as e:
            print(f"VR right arm pose control command collection failed: {e}")
            raise

    def _collect_vr_left_gripper_joint_commands(self):
        """Collect VR left gripper joint control command"""
        print("Starting VR left gripper joint control command collection...")
        try:
            self._collect_gripper_command_stream(
                'vr_left_gripper_joint_commands',
                self.robot.action_data_collection.get_vr_left_gripper_joint_state_stream,
            )
        except Exception as e:
            print(f"VR left gripper joint control command collection failed: {e}")
            raise

    def _collect_vr_right_gripper_joint_commands(self):
        """Collect VR right gripper joint control command"""
        print("Starting VR right gripper joint control command collection...")
        try:
            self._collect_gripper_command_stream(
                'vr_right_gripper_joint_commands',
                self.robot.action_data_collection.get_vr_right_gripper_joint_state_stream,
            )
        except Exception as e:
            print(f"VR right gripper joint control command collection failed: {e}")
            raise

    def _collect_gripper_command_stream(self, queue_name, stream_func):
        """Collect a VR gripper JointState command as one scalar position."""
        try:
            stream = stream_func(timeout=None)
            for command_msg in stream:
                if not self._recording_active():
                    break
                try:
                    timestamp = self._extract_timestamp_from_header(command_msg)
                    positions = list(getattr(command_msg, 'position', []) or [])
                    if not positions:
                        raise ValueError("VR gripper command has no position")
                    command_data = {'position': float(positions[0])}
                    if (
                        queue_name in self.sensor_file_locks
                        and queue_name in self.sensor_temp_files
                    ):
                        with self.sensor_file_locks[queue_name]:
                            pickle.dump(
                                (timestamp, command_data),
                                self.sensor_temp_files[queue_name],
                            )
                            self._maybe_flush_temp_file(
                                self.sensor_temp_files[queue_name],
                                f'sensor:{queue_name}',
                            )
                            self._record_raw_topic_sample(
                                queue_name,
                                timestamp,
                            )
                        with self.stats_lock:
                            count_key = f'{queue_name}_count'
                            self.stats[count_key] = self.stats.get(count_key, 0) + 1
                except Exception as e:
                    print(f"{queue_name} Error processing: {e}")
                    continue
        except Exception as e:
            if self._is_grpc_connection_error(e):
                self._handle_grpc_error_and_exit(queue_name, e)
            print(f"{queue_name} Stream error: {e}")
            raise


    # ============ Distance sensor collection methods ============

    def _collect_tof_sensors(self):
        """Collect ToF sensor data"""
        print("Starting ToF sensor collection...")
        # Collect two ToF sensors
        tof_threads = []
        for i in [1, 2]:
            queue_name = f'tof_{i}'
            stream_func = getattr(self.robot.tof, f'get_chassis_tof_{i}_stream')
            thread = threading.Thread(
                target=self._collect_generic_stream,
                args=(queue_name, stream_func),
                daemon=True
            )
            thread.start()
            tof_threads.append(thread)
            # Save thread information for restart (using lambda to wrap parameters)
            if hasattr(self, 'thread_info'):
                def _tof_collector_wrapper():
                    self._collect_generic_stream(queue_name, stream_func)
                self.thread_info.append((f'ToF_{i}_Collector', _tof_collector_wrapper))
                self.threads.append(thread)

        # Do not call join(), let the thread run in the background
        # When is_recording = False, the thread will exit the loop, and the method will end
        # When restarting, these threads will be recreated

    def _collect_ultrasonic_sensors(self):
        """Collect ultrasonic sensor data"""
        print("Starting ultrasonic sensor collection...")
        # Collect four ultrasonic sensors
        ultrasonic_threads = []
        for i in range(1, 5):
            queue_name = f'ultrasonic_{i}'
            stream_func = getattr(self.robot.ultrasonic, f'get_chassis_ultrasonic_{i}_stream')
            thread = threading.Thread(
                target=self._collect_generic_stream,
                args=(queue_name, stream_func),
                daemon=True
            )
            thread.start()
            ultrasonic_threads.append(thread)
            # Save thread information for restart (using lambda to wrap parameters)
            if hasattr(self, 'thread_info'):
                def _ultrasonic_collector_wrapper():
                    self._collect_generic_stream(queue_name, stream_func)
                self.thread_info.append((f'Ultrasonic_{i}_Collector', _ultrasonic_collector_wrapper))
                self.threads.append(thread)

        # Do not call join(), let the thread run in the background
        # When is_recording = False, the thread will exit the loop, and the method will end
        # When restarting, these threads will be recreated

    # ============ General auxiliary methods ============

    def _start_camera_monitor(self):
        """Start camera data monitoring thread"""
        if self.camera_monitor_thread is not None and self.camera_monitor_thread.is_alive():
            raise RuntimeError("上一条数据的相机监控线程仍未退出")

        self.camera_monitor_thread = threading.Thread(
            target=self._collector_entry,
            args=(self._monitor_camera_data, self._episode_stop_event),
            daemon=True,
            name="CameraDataMonitor"
        )
        self.camera_monitor_thread.start()

    def _stop_camera_monitor(self):
        """Stop camera data monitoring thread"""
        if self._episode_stop_event is not None:
            self._episode_stop_event.set()
        thread = self.camera_monitor_thread
        if thread is None:
            return
        if thread is threading.current_thread():
            self.camera_monitor_thread = None
            return
        if thread.is_alive():
            thread.join(timeout=self.camera_data_check_interval + 1.0)
        if thread.is_alive():
            raise RuntimeError("相机监控线程未能及时停止")
        self.camera_monitor_thread = None

    def _stop_all_threads(self):
        """Stop all data collection threads"""
        self.is_recording = False
        if self._episode_stop_event is not None:
            self._episode_stop_event.set()

        self._stop_camera_monitor()
        if hasattr(self, 'threads') and self.threads:
            print("正在等待数据采集线程停止...")
            self._wait_for_collection_threads()

    def _cleanup_before_exit(self, message="Cleaning up resources..."):
        """Uniform cleanup function: stop threads first, then clean up temporary files (for cleanup before exit)

        Note: This function only handles cleanup, not exit, exit is decided by caller

        Args:
            message: Prompt message before cleanup
        """

        # First step: stop all threads (to avoid writing to closed files)
        try:
            self._stop_all_threads()
        except Exception as e:
            print(f"Error stopping threads: {e}")

        # Second step: clean up temporary files
        try:
            if message:
                print(message)

            self._cleanup_temp_files()
            self._cleanup_active_episode_staging()
            print("临时文件已清理")
        except Exception as e:
            print(f"  ⚠️  Clean up temporary files failed: {e}")

    def _get_enabled_cameras(self):
        """Get the list of actually enabled cameras (according to configuration)"""
        enabled_cameras = []
        if self.collection_config.enable_head_rgb_stream:
            enabled_cameras.append('head_rgb_stream')
        if self.collection_config.enable_head_depth_stream:
            enabled_cameras.append('head_depth_stream')
        if self.collection_config.enable_left_arm_rgb_stream:
            enabled_cameras.append('left_arm_rgb_stream')
        if self.collection_config.enable_right_arm_rgb_stream:
            enabled_cameras.append('right_arm_rgb_stream')
        return enabled_cameras

    def _monitor_camera_data(self):
        """Monitor camera data, if a camera has no data, report an error and exit"""
        recording_start_time = time.time()

        # Get the list of actually enabled cameras
        enabled_cameras = self._get_enabled_cameras()

        # If there are no enabled cameras, no need to monitor
        if not enabled_cameras:
            return

        while self._recording_active():
            current_time = time.time()
            elapsed_time = current_time - recording_start_time

            # Only check after recording starts (give cameras some time to start collecting)
            # At least wait for the timeout time to determine if the camera has no data
            if elapsed_time < self.camera_data_timeout:
                time.sleep(self.camera_data_check_interval)
                continue

            # Only check actually enabled cameras
            cameras_without_data = []
            with self.stats_lock:
                for cam in enabled_cameras:
                    # Ensure camera is in tracking list
                    if cam not in self.camera_last_data_time:
                        continue

                    last_data_time = self.camera_last_data_time.get(cam)
                    if last_data_time is None:
                        # Never received data, but only report error after recording time exceeds timeout
                        if elapsed_time >= self.camera_data_timeout:
                            cameras_without_data.append(cam)
                    else:
                        # Check if timeout
                        time_since_last_data = current_time - last_data_time
                        if time_since_last_data > self.camera_data_timeout:
                            cameras_without_data.append(cam)

            if cameras_without_data:
                error_msg = f"❌ Detected the following cameras have no data (timeout {self.camera_data_timeout} seconds):\n"
                for cam in cameras_without_data:
                    last_time = self.camera_last_data_time.get(cam)
                    if last_time is None:
                        error_msg += f"  - {cam}: Never received data\n"
                    else:
                        time_since = current_time - last_time
                        error_msg += f"  - {cam}: Last data time {time_since:.2f} seconds ago\n"
                error_msg += "Process will exit..."
                # Clean up temporary files before exiting
                self._cleanup_before_exit(error_msg)
                os._exit(1)

            time.sleep(self.camera_data_check_interval)

    def _restart_stopped_threads(self):
        """Restart stopped threads"""
        if not hasattr(self, 'thread_info') or not self.thread_info:
            return

        # Check which threads have stopped
        stopped_threads = []
        for i, thread in enumerate(self.threads):
            if not thread.is_alive():
                stopped_threads.append(i)

        if stopped_threads:
            print(f"Detected {len(stopped_threads)} threads have stopped, restarting...")
            for i in stopped_threads:
                thread_name, target_func = self.thread_info[i]
                # Create new thread
                new_thread = threading.Thread(target=target_func, daemon=True, name=thread_name)
                new_thread.start()
                self.threads[i] = new_thread
                print(f"    ✓ Restart thread: {thread_name}")

    def _is_grpc_connection_error(self, error):
        """Check if it is a gRPC connection error"""
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Check error type
        if 'MultiThreadedRendezvous' in error_type or 'grpc' in error_type.lower():
            return True

        # Check error message
        if any(keyword in error_str for keyword in [
            'connection reset',
            'unavailable',
            'peer',
            'network',
            'grpc_status',
            'statuscode.unavailable',
            'recvmsg:connection reset'
        ]):
            return True

        return False

    def _handle_grpc_error_and_exit(self, queue_name, error):
        """Handle gRPC connection error and exit process"""
        err_msg = f"❌ {queue_name} Network connection interrupted: {error}\n"
        err_msg += f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}\n"
        err_msg += f"   Please check the network connection and restart data collection"
        # Clean up resources before exiting
        self._cleanup_before_exit(err_msg)
        print(f"Process will exit...")
        # Exit process directly (using os._exit to avoid triggering other cleanup logic, but we have already manually cleaned up)
        os._exit(1)

    def _collect_joint_state_stream(self, queue_name, stream_func):
        """General joint state stream collection method"""
        # Store the joint names of each part (save the first time the message is received)
        joint_names_cache = {}

        try:
            stream = stream_func(timeout=None)
            message_count = 0

            for joint_state_msg in stream:
                # If not in recording, exit the loop
                if not self._recording_active():
                    break

                try:
                    timestamp = self._extract_timestamp_from_header(joint_state_msg)
                    message_count += 1

                    # The first time the message is received, save the joint names
                    if queue_name not in joint_names_cache:
                        if hasattr(joint_state_msg, 'name') and joint_state_msg.name and len(joint_state_msg.name) > 0:
                            joint_names_cache[queue_name] = list(joint_state_msg.name)
                            print(f"  ✓ {queue_name} joint names: {joint_names_cache[queue_name]}")
                            # Save to instance variable for subsequent use
                            if not hasattr(self, '_joint_names_by_part'):
                                self._joint_names_by_part = {}
                            self._joint_names_by_part[queue_name] = joint_names_cache[queue_name]

                    if self._recording_active():
                        # Extract joint data
                        positions = np.array(joint_state_msg.position, dtype=np.float32).flatten() if joint_state_msg.position else np.array([], dtype=np.float32)
                        velocities = np.array(joint_state_msg.velocity, dtype=np.float32).flatten() if hasattr(joint_state_msg, 'velocity') and joint_state_msg.velocity else None
                        efforts = np.array(joint_state_msg.effort, dtype=np.float32).flatten() if hasattr(joint_state_msg, 'effort') and joint_state_msg.effort else None

                        # Store joint state data to temporary file
                        if queue_name in self.sensor_file_locks and hasattr(self, 'sensor_temp_files') and queue_name in self.sensor_temp_files:
                            joint_state_data_tuple = (timestamp, positions, velocities, efforts)
                            with self.sensor_file_locks[queue_name]:
                                if queue_name in self.sensor_temp_files:
                                    pickle.dump(joint_state_data_tuple, self.sensor_temp_files[queue_name])
                                    self._maybe_flush_temp_file(
                                        self.sensor_temp_files[queue_name],
                                        f'sensor:{queue_name}',
                                    )
                                    self._record_raw_topic_sample(
                                        queue_name,
                                        timestamp,
                                    )
                            with self.stats_lock:
                                self.stats[f'{queue_name}_count'] = self.stats.get(f'{queue_name}_count', 0) + 1
                        else:
                            if message_count == 1:  # Only print once, to avoid spamming
                                print(f"  ⚠️  Warning: {queue_name} temporary file does not exist, cannot save data")

                        # Note: action data does not need to be saved separately, the next frame's state will be used as action when saving episode
                except Exception as e:
                    print(f"{queue_name} Error processing: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # If the thread exits but no messages are received, print a warning
            if message_count == 0:
                print(f"  ⚠️  Warning: {queue_name} thread exited, but no data messages were received")
        except Exception as e:
            if self._is_grpc_connection_error(e):
                self._handle_grpc_error_and_exit(queue_name, e)
            print(f"{queue_name} Stream error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _collect_pose_stream(self, queue_name, stream_func):
        """General end-effector pose stream collection method"""
        try:
            stream = stream_func(timeout=None)
            for pose_msg in stream:
                # If not in recording, exit the loop
                if not self._recording_active():
                    break

                try:
                    timestamp = self._extract_timestamp_from_header(pose_msg)
                    if self._recording_active():
                        pose_data = {
                            'position': {
                                'x': pose_msg.pose.position.x,
                                'y': pose_msg.pose.position.y,
                                'z': pose_msg.pose.position.z,
                            },
                            'orientation': {
                                'x': pose_msg.pose.orientation.x,
                                'y': pose_msg.pose.orientation.y,
                                'z': pose_msg.pose.orientation.z,
                                'w': pose_msg.pose.orientation.w,
                            }
                        }
                        # Store to corresponding temporary file
                        if queue_name in self.sensor_file_locks and hasattr(self, 'sensor_temp_files') and queue_name in self.sensor_temp_files:
                            pose_data_tuple = (timestamp, pose_data)
                            with self.sensor_file_locks[queue_name]:
                                if queue_name in self.sensor_temp_files:
                                    pickle.dump(pose_data_tuple, self.sensor_temp_files[queue_name])
                                    self._maybe_flush_temp_file(
                                        self.sensor_temp_files[queue_name],
                                        f'sensor:{queue_name}',
                                    )
                                    self._record_raw_topic_sample(
                                        queue_name,
                                        timestamp,
                                    )
                            with self.stats_lock:
                                self.stats[f'{queue_name}_count'] = self.stats.get(f'{queue_name}_count', 0) + 1

                except Exception as e:
                    print(f"{queue_name} Error processing: {e}")
                    continue

        except Exception as e:
            if self._is_grpc_connection_error(e):
                self._handle_grpc_error_and_exit(queue_name, e)
            print(f"{queue_name} Stream error: {e}")
            raise

    def _convert_ros_msg_to_dict(self, msg, exclude_fields=None):
        """Recursively convert protobuf message object to a JSON serializable dictionary

        Args:
            msg: protobuf message object
            exclude_fields: Collection of field names to exclude (optional)

        Returns:
            dict or basic type
        """
        # Default excluded fields: timestamp (aligned) and private fields
        if exclude_fields is None:
            exclude_fields = {'stamp', 'header'}  # stamp for sensor timestamp, header for ROS message header

        # If it is a basic type, return directly
        if isinstance(msg, (int, float, str, bool, type(None))):
            return msg

        # If it is a NumPy array, convert to list
        if isinstance(msg, np.ndarray):
            return msg.tolist()

        # If it is a list or tuple, recursively process each element
        if isinstance(msg, (list, tuple)):
            return [self._convert_ros_msg_to_dict(item, exclude_fields) for item in msg]

        # If it is a dictionary, recursively process each value
        if isinstance(msg, dict):
            return {k: self._convert_ros_msg_to_dict(v, exclude_fields) for k, v in msg.items() if k not in exclude_fields}

        # If it is a protobuf message object (has DESCRIPTOR attribute)
        if hasattr(msg, 'DESCRIPTOR'):
            result = {}
            # Use ListFields() to get all set fields
            if hasattr(msg, 'ListFields'):
                for field_descriptor, value in msg.ListFields():
                    field_name = field_descriptor.name
                    if field_name not in exclude_fields:
                        result[field_name] = self._convert_ros_msg_to_dict(value, exclude_fields)
            # If ListFields() is empty, try to get all fields from DESCRIPTOR
            elif hasattr(msg, 'DESCRIPTOR'):
                for field in msg.DESCRIPTOR.fields:
                    field_name = field.name
                    if field_name not in exclude_fields and hasattr(msg, field_name):
                        value = getattr(msg, field_name)
                        # Only add fields with non-default values
                        if value is not None and value != field.default_value:
                            result[field_name] = self._convert_ros_msg_to_dict(value, exclude_fields)
            return result

        # If it has __dict__ attribute (preferred over __slots__, because dataclass may have empty __slots__)
        if hasattr(msg, '__dict__'):
            msg_dict = msg.__dict__
            # If __dict__ is not empty, use it (filter out private fields starting with underscore and excluded fields)
            if msg_dict:
                return {k: self._convert_ros_msg_to_dict(v, exclude_fields)
                       for k, v in msg_dict.items()
                       if not k.startswith('_') and k not in exclude_fields}

        # If it is a ROS message object (has __slots__ attribute and __slots__ is not empty)
        if hasattr(msg, '__slots__') and msg.__slots__:
            result = {}
            for slot in msg.__slots__:
                if slot not in exclude_fields and hasattr(msg, slot):
                    value = getattr(msg, slot)
                    result[slot] = self._convert_ros_msg_to_dict(value, exclude_fields)
            return result

        # Other cases, try to convert to string
        try:
            return str(msg)
        except:
            return None

    def _collect_generic_stream(self, queue_name, stream_func):
        """General sensor stream collection method"""
        try:
            stream = stream_func(timeout=None)
            for msg in stream:
                # If not in recording, exit the loop
                if not self._recording_active():
                    break

                # If not in recording, exit the loop (thread will stop)
                if not self._recording_active():
                    break

                try:
                    timestamp = self._extract_timestamp_from_header(msg)
                    if self._recording_active():
                        # Store original message data
                        data = {
                            'timestamp': timestamp,
                            'data': msg
                        }
                        # Store to corresponding temporary file
                        if queue_name in self.sensor_file_locks and hasattr(self, 'sensor_temp_files') and queue_name in self.sensor_temp_files:
                            sensor_data_tuple = (timestamp, data)
                            with self.sensor_file_locks[queue_name]:
                                if queue_name in self.sensor_temp_files:
                                    pickle.dump(sensor_data_tuple, self.sensor_temp_files[queue_name])
                                    self._maybe_flush_temp_file(
                                        self.sensor_temp_files[queue_name],
                                        f'sensor:{queue_name}',
                                    )
                                    self._record_raw_topic_sample(
                                        queue_name,
                                        timestamp,
                                    )
                            with self.stats_lock:
                                self.stats[f'{queue_name}_count'] = self.stats.get(f'{queue_name}_count', 0) + 1

                except Exception as e:
                    print(f"{queue_name} Error processing: {e}")
                    continue

        except Exception as e:
            if self._is_grpc_connection_error(e):
                self._handle_grpc_error_and_exit(queue_name, e)
            # Other types of errors, rethrow
            print(f"❌ {queue_name} Stream error: {e}, time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
            raise


    def get_stats(self):
        """Get current statistics"""
        with self.stats_lock:
            stats = self.stats.copy()
            if self.current_episode_start_time:
                stats['recording_duration'] = time.time() - self.current_episode_start_time
            return stats

    def print_stats(self):
        """Print current statistics"""
        stats = self.get_stats()
        print("\n=== 数据采集统计 ===")
        if 'recording_duration' in stats:
            print(f"采集时长：{stats['recording_duration']:.2f} 秒")
        print(f"State 帧数：{stats.get('state_count', 0)}")
        print(f"Action 帧数：{stats.get('action_count', 0)}")
        for cam, count in stats.get('image_count', {}).items():
            print(f"{cam}：{count} 帧")
        # Show all sensor statistics
        for stat_key, count in sorted(stats.items()):
            if stat_key.endswith('_count') and stat_key not in ['state_count', 'action_count', 'image_count']:
                # Only check for integer type count > 0
                if isinstance(count, int) and count > 0:
                    sensor_name = stat_key.replace('_count', '')
                    if sensor_name in ['head_camera', 'left_arm_camera', 'right_arm_camera',
                                       'head_depth_video', 'head_rgb_video']:
                        print(f"{sensor_name}：{count} 帧")
                    else:
                        print(f"{sensor_name}: {count}")

        # Compatibility with old display method
        if self.collection_config.enable_chassis_imu:
            print(f"IMU: {stats.get('chassis_imu_count', 0)} frames")
        if self.collection_config.enable_depth_points:
            print(f"Depth: {stats.get('depth_points_count', 0)} frames")

        # Show file storage status
        sensor_files = len([s for s in self.sensor_file_locks.keys() if hasattr(self, 'sensor_temp_files') and s in self.sensor_temp_files])
        hf_files = len([h for h in ['joint_states', 'actions'] if hasattr(self, 'sensor_temp_files') and h in self.sensor_temp_files])

        if sensor_files > 0 or hf_files > 0:
            print(f"数据存储状态：✓ {sensor_files} 个传感器文件，✓ {hf_files} 个高频文件")
            print("存储策略：全部写入临时文件")

            # Check if there is any overwrite
            if hasattr(self, '_queue_overwrites'):
                overwrites = self._queue_overwrites
                if overwrites > 0:
                    print(f"⚠️  Warning: Some data may be overwritten due to processing delay")
        else:
            print("数据存储：✗ 没有可用文件")

        print("================\n")

    def _extract_timestamp_from_header(self, msg):
        """Extract timestamp from the header of the message"""
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            sec = int(msg.header.stamp.sec)
            nanosec = int(msg.header.stamp.nanosec)
            if sec != 0 or nanosec != 0:
                self._collector_context.timestamp_source = "message_header"
                return float(sec) + float(nanosec) / 1e9
        # Messages without a usable source stamp retain their independent
        # collector receipt time; they are never assigned another topic's
        # timestamp.
        self._collector_context.timestamp_source = (
            "collector_receipt_wall_clock"
        )
        return time.time()

    def _get_enabled_data_mapping(self):
        """Get the mapping of enabled configuration items to data key names

        Returns:
            dict: {configuration item name: [data key name list]}
        """
        mapping = {}

        # Joint state collection
        if self.slave_joint_names:
            mapping['slave_joint_names'] = self.slave_joint_names

        # Image stream collection
        if self.collection_config.enable_head_rgb_stream:
            mapping['enable_head_rgb_stream'] = ['head_rgb_stream']
        if self.collection_config.enable_head_depth_stream:
            mapping['enable_head_depth_stream'] = ['head_depth_stream']
        if self.collection_config.enable_left_arm_rgb_stream:
            mapping['enable_left_arm_rgb_stream'] = ['left_arm_rgb_stream']
        if self.collection_config.enable_right_arm_rgb_stream:
            mapping['enable_right_arm_rgb_stream'] = ['right_arm_rgb_stream']

        # Sensor collection
        if self.collection_config.enable_left_arm_end_pose:
            mapping['enable_left_arm_end_pose'] = ['left_arm_end_pose']
        if self.collection_config.enable_right_arm_end_pose:
            mapping['enable_right_arm_end_pose'] = ['right_arm_end_pose']
        if self.collection_config.enable_waist_end_pose:
            mapping['enable_waist_end_pose'] = ['waist_end_pose']
        if self.collection_config.enable_odometry:
            mapping['enable_odometry'] = ['odometry']
        if self.collection_config.enable_pose:
            mapping['enable_pose'] = ['pose']
        if self.collection_config.enable_chassis_imu:
            mapping['enable_chassis_imu'] = ['chassis_imu']
        if self.collection_config.enable_depth_points:
            mapping['enable_depth_points'] = ['depth_points']
        if self.collection_config.enable_head_depth_video:
            mapping['enable_head_depth_video'] = ['head_depth_video']
        if self.collection_config.enable_laser_scan:
            mapping['enable_laser_scan'] = ['laser_scan']

        # Tactile sensors
        if self.collection_config.enable_left_gripper_tactile:
            mapping['enable_left_gripper_tactile'] = ['left_gripper_tactile']
        if self.collection_config.enable_right_gripper_tactile:
            mapping['enable_right_gripper_tactile'] = ['right_gripper_tactile']
        if self.collection_config.enable_left_hand_tactile:
            mapping['enable_left_hand_tactile'] = ['left_hand_tactile']
        if self.collection_config.enable_right_hand_tactile:
            mapping['enable_right_hand_tactile'] = ['right_hand_tactile']

        # Distance sensors
        if self.collection_config.enable_tof_sensors:
            mapping['enable_tof_sensors'] = ['tof_1', 'tof_2']
        if self.collection_config.enable_ultrasonic_sensors:
            mapping['enable_ultrasonic_sensors'] = [f'ultrasonic_{i}' for i in range(1, 5)]

        # Master arm data
        if self.collection_config.enable_master_arm_data:
            mapping['enable_master_arm_data'] = [
                'master_left_arm_joint_state',
                'master_right_arm_joint_state',
                'master_left_arm_end_pose',
                'master_right_arm_end_pose',
                'master_left_gripper_joint_state',
                'master_right_gripper_joint_state'
            ]

        # Force sensors
        if self.collection_config.enable_wrench_ext_world:
            mapping['enable_wrench_ext_world'] = [
                'left_arm_wrench_ext_world',
                'right_arm_wrench_ext_world'
            ]
        if self.collection_config.enable_wrench_ext_local:
            mapping['enable_wrench_ext_local'] = [
                'left_arm_wrench_ext_local',
                'right_arm_wrench_ext_local'
            ]

        if self.collection_config.enable_left_gripper_position:
            mapping['enable_left_gripper_position'] = ['left_gripper_position']
        if self.collection_config.enable_right_gripper_position:
            mapping['enable_right_gripper_position'] = ['right_gripper_position']
        if self.collection_config.enable_vr_action_commands:
            mapping['enable_vr_action_commands'] = list(
                self.vr_action_sensor_names
            )
        if self.collection_config.enable_revo2_hands:
            mapping['enable_revo2_hands'] = list(
                self.revo2_state_sensor_names
            )

        return mapping

    def _validate_collected_data(self, sensor_data: Dict[str, Any], images: Dict[str, Any]) -> bool:
        """Validate collected raw data (before alignment)

        Check if all enabled data items have actual data (read from temporary files)

        Args:
            sensor_data: Raw sensor data collected from temporary files
            images: Raw image data collected from temporary files

        Returns:
            True if all enabled data items have data, False otherwise
        """
        # Get the mapping of enabled configuration items
        enabled_mapping = self._get_enabled_data_mapping()

        # Check if all enabled data items have data
        missing_data_items = []

        # For recording the missing cases of master arm data (only warning, not error, exclude action data)
        master_arm_missing_items = []

        # Camera name mapping: internal stream name -> user-friendly name
        camera_name_mapping = {
            'head_rgb_stream': 'head_camera',
            'head_depth_stream': 'head_depth_camera',
            'left_arm_rgb_stream': 'left_arm_camera',
            'right_arm_rgb_stream': 'right_arm_camera'
        }

        for config_name, data_keys in enabled_mapping.items():
            # Check the master arm data (only warning, not required, exclude action data)
            if config_name == 'enable_master_arm_data':
                # Only check the joint_state and end_pose of the master arm, not check action
                for key in data_keys:
                    # Skip data related to action
                    if 'action' in key.lower():
                        continue

                    # Check sensor data
                    if key in sensor_data:
                        if len(sensor_data[key]) == 0:
                            master_arm_missing_items.append((key, f"数据为空（{len(sensor_data[key])} 条）"))
                    else:
                        available_keys = list(sensor_data.keys())[:10]  # Only show the first 10 keys
                        master_arm_missing_items.append((key, f"数据不存在（sensor_data 当前包含：{available_keys}）"))
                continue

            # Check image data
            if any(key.endswith('_rgb_stream') or key.endswith('_depth_stream') or key.endswith('_depth_video')
                   for key in data_keys):
                # Image data in images dictionary (using original internal key names)
                for internal_key in data_keys:
                    if internal_key in images:
                        if len(images[internal_key]) == 0:
                            friendly_name = camera_name_mapping.get(internal_key, internal_key)
                            missing_data_items.append((config_name, friendly_name, "图像数据为空"))
                    else:
                        friendly_name = camera_name_mapping.get(internal_key, internal_key)
                        missing_data_items.append((config_name, friendly_name, f"图像数据不存在（内部名称：{internal_key}）"))
            else:
                # Sensor data in sensor_data dictionary
                for key in data_keys:
                    if key in sensor_data:
                        if len(sensor_data[key]) == 0:
                            missing_data_items.append((config_name, key, f"数据为空（{len(sensor_data[key])} 条）"))
                    else:
                        # Provide more detailed error information
                        available_keys = list(sensor_data.keys())[:10]  # Only show the first 10 keys
                        missing_data_items.append((config_name, key, f"数据不存在（sensor_data 当前包含：{available_keys}）"))

        # If there are missing data items, exit with error
        if missing_data_items:
            error_parts = ["错误：以下已启用的数据项没有采集到数据：\n"]

            # Group by configuration item
            by_config = {}
            for config_name, data_key, reason in missing_data_items:
                if config_name not in by_config:
                    by_config[config_name] = []
                by_config[config_name].append((data_key, reason))

            for config_name, items in by_config.items():
                error_parts.append(f"  - {config_name}:")
                for data_key, reason in items:
                    error_parts.append(f"    • {data_key}: {reason}")

            error_parts.append("\n请检查机器人连接和采集配置，确保所有启用的数据项均能正常输出。")

            error_msg = "\n".join(error_parts)
            print(f"  {error_msg}")
            return False

        # Master arm data missing warning (not error, exit)
        if master_arm_missing_items:
            print("\n  ⚠️  警告：已启用 enable_master_arm_data，但以下主臂数据缺失（不影响其他数据采集）：")
            for data_key, reason in master_arm_missing_items:
                print(f"    • {data_key}: {reason}")

        return True

    def _validate_episode_data(self, episode_data):
        """Validate episode data quality"""
        if episode_data is None:
            return False

        # Get data from the aligned data structure
        sensor_data = episode_data.get('sensor_data', {})
        action_data = episode_data.get('action_data', {})
        images = episode_data.get('images', {})

        # Get the mapping of enabled configuration items
        enabled_mapping = self._get_enabled_data_mapping()

        # Check if all enabled data items have data
        missing_data_items = []

        # Camera name mapping: internal stream name -> user-friendly name
        camera_name_mapping = episode_data.get('camera_name_mapping', {
            'head_rgb_stream': 'head_camera',
            'head_depth_stream': 'head_depth_camera',
            'left_arm_rgb_stream': 'left_arm_camera',
            'right_arm_rgb_stream': 'right_arm_camera'
        })

        # For recording the missing cases of master arm data (only warning, not error)
        master_arm_missing_items = []

        for config_name, data_keys in enabled_mapping.items():
            # Check the master arm data (only warning, not required)
            if config_name == 'enable_master_arm_data':
                # Check image data
                if any(key.endswith('_rgb_stream') or key.endswith('_depth_stream') or key.endswith('_depth_video')
                       for key in data_keys):
                    # Image data in images dictionary (using mapped friendly name)
                    if self.use_video_storage and 'image_index_mapping' in episode_data:
                        # Video storage mode: check image_index_mapping
                        image_index_mapping = episode_data.get('image_index_mapping', {})
                        for internal_key in data_keys:
                            friendly_name = camera_name_mapping.get(internal_key, internal_key)
                            if friendly_name in image_index_mapping:
                                if len(image_index_mapping[friendly_name]) == 0:
                                    master_arm_missing_items.append((friendly_name, "Image data is empty"))
                            else:
                                master_arm_missing_items.append((friendly_name, f"Image data does not exist (internal key: {internal_key})"))
                    else:
                        # Image storage mode: check aligned_images (in images dictionary)
                        for internal_key in data_keys:
                            friendly_name = camera_name_mapping.get(internal_key, internal_key)
                            if friendly_name in images:
                                if len(images[friendly_name]) == 0:
                                    master_arm_missing_items.append((friendly_name, "Image data is empty"))
                            else:
                                # Also check the original internal key names (before data alignment)
                                if internal_key in images:
                                    if len(images[internal_key]) == 0:
                                        master_arm_missing_items.append((friendly_name, "Image data is empty"))
                                else:
                                    master_arm_missing_items.append((friendly_name, f"Image data does not exist (internal key: {internal_key})"))
                else:
                    # Sensor data in sensor_data or action_data dictionary
                    # Only check the joint_state and end_pose of the master arm, not check action
                    for key in data_keys:
                        # Skip data related to action
                        if 'action' in key.lower():
                            continue

                        found = False

                        # Check sensor_data first
                        if key in sensor_data:
                            if len(sensor_data[key]) > 0:
                                found = True
                            else:
                                master_arm_missing_items.append((key, f"Data is empty ({len(sensor_data[key])} items)"))
                                continue

                        # If not found, record warning
                        if not found:
                            available_sensor_keys = list(sensor_data.keys())[:10]  # Only show the first 10 keys
                            error_info = f"Data does not exist (sensor_data has: {available_sensor_keys})"
                            master_arm_missing_items.append((key, error_info))
                continue

            # Check image data
            if any(key.endswith('_rgb_stream') or key.endswith('_depth_stream') or key.endswith('_depth_video')
                   for key in data_keys):
                # Image data in images dictionary (using mapped friendly name)
                if self.use_video_storage and 'image_index_mapping' in episode_data:
                    # Video storage mode: check image_index_mapping
                    image_index_mapping = episode_data.get('image_index_mapping', {})
                    for internal_key in data_keys:
                        friendly_name = camera_name_mapping.get(internal_key, internal_key)
                        if friendly_name in image_index_mapping:
                            if len(image_index_mapping[friendly_name]) == 0:
                                missing_data_items.append((config_name, friendly_name, "图像数据为空"))
                        else:
                            missing_data_items.append((config_name, friendly_name, f"图像数据不存在（内部名称：{internal_key}）"))
                else:
                    # Image storage mode: check aligned_images (in images dictionary)
                    for internal_key in data_keys:
                        friendly_name = camera_name_mapping.get(internal_key, internal_key)
                        if friendly_name in images:
                            if len(images[friendly_name]) == 0:
                                missing_data_items.append((config_name, friendly_name, "图像数据为空"))
                        else:
                            # Also check the original internal key names (before data alignment)
                            if internal_key in images:
                                if len(images[internal_key]) == 0:
                                    missing_data_items.append((config_name, friendly_name, "图像数据为空"))
                            else:
                                missing_data_items.append((config_name, friendly_name, f"图像数据不存在（内部名称：{internal_key}）"))
            else:
                # Sensor data in sensor_data or action_data dictionary
                for key in data_keys:
                    found = False
                    data_source = None

                    # Check sensor_data first
                    if key in sensor_data:
                        if len(sensor_data[key]) > 0:
                            found = True
                            data_source = "sensor_data"
                        else:
                            missing_data_items.append((config_name, key, f"数据为空（{len(sensor_data[key])} 条）"))
                            continue

                    # If not in sensor_data, check action_data (some data may be there)
                    if not found and key in action_data:
                        if len(action_data[key]) > 0:
                            found = True
                            data_source = "action_data"
                        else:
                            missing_data_items.append((config_name, key, f"数据为空（{len(action_data[key])} 条）"))
                            continue

                    # If not found, exit with error
                    if not found:
                        # Provide more detailed error information
                        available_sensor_keys = list(sensor_data.keys())[:10]  # Only show the first 10 keys
                        available_action_keys = list(action_data.keys())[:10]
                        error_info = (
                            "数据不存在（sensor_data 当前包含："
                            f"{available_sensor_keys}；action_data 当前包含："
                            f"{available_action_keys}）"
                        )
                        missing_data_items.append((config_name, key, error_info))

        # If there are missing data items, exit with error
        if missing_data_items:
            error_parts = ["错误：以下已启用的数据项没有采集到数据：\n"]

            # Group by configuration item
            by_config = {}
            for config_name, data_key, reason in missing_data_items:
                if config_name not in by_config:
                    by_config[config_name] = []
                by_config[config_name].append((data_key, reason))

            for config_name, items in by_config.items():
                error_parts.append(f"  - {config_name}:")
                for data_key, reason in items:
                    error_parts.append(f"    • {data_key}: {reason}")

            error_parts.append("\n请检查机器人连接和采集配置，确保所有启用的数据项均能正常输出。")

            error_msg = "\n".join(error_parts)
            # Clean up temporary files before exiting
            self._cleanup_before_exit(error_msg)
            sys.exit(1)

        # Master arm data missing warning (not error, exit)
        # Note: Warning is only printed in _validate_collected_data, here is not repeated to avoid repetition

        # Use the joint state names in the member variable (for subsequent statistics)
        joint_state_names = self.slave_joint_names if self.slave_joint_names else []

        # Check if there is any joint state data (compatible with old format)
        states = sensor_data.get('joint_states', [])  # Compatible with old format
        actions = sensor_data.get('actions', [])  # Compatible with old format

        # Check if there is any joint state data (compatible with old format)
        joint_states_by_part = {}
        for joint_state_name in joint_state_names:
            if joint_state_name in sensor_data and len(sensor_data[joint_state_name]) > 0:
                joint_states_by_part[joint_state_name] = sensor_data[joint_state_name]

        # Check if there is any data
        has_joint_data = len(states) > 0 or len(actions) > 0 or len(joint_states_by_part) > 0
        state_sensor_data = {
            name: values
            for name, values in sensor_data.items()
            if name not in ALL_VR_ACTION_SENSOR_NAMES and values
        }
        has_state_sensor_data = bool(state_sensor_data)
        if self.use_video_storage and 'image_index_mapping' in episode_data:
            has_image_data = any(
                len(indices) > 0
                for indices in episode_data['image_index_mapping'].values()
            )
        else:
            has_image_data = any(len(imgs) > 0 for imgs in images.values())

        if not has_joint_data and not has_state_sensor_data and not has_image_data:
            print("  Validation failed: no data")
            return False

        # Get the number of state frames. Pi0.5 VR collection intentionally
        # uses measured EEF/gripper state without joint-state observations.
        if joint_states_by_part:
            # Use the data of each part
            state_frame_count = max(len(data) for data in joint_states_by_part.values())
        elif state_sensor_data:
            state_frame_count = max(
                len(values)
                for values in state_sensor_data.values()
            )
        else:
            # Compatible with old format
            state_frame_count = len(states) if states else 0

        # If image collection is enabled, check the image data
        if len(self.camera_names) > 0:
            # Video storage mode: use image_index_mapping to validate
            if self.use_video_storage and 'image_index_mapping' in episode_data:
                image_index_mapping = episode_data['image_index_mapping']
                for cam_name, indices in image_index_mapping.items():
                    if len(indices) == 0:
                        print(f"  ⚠️  Warning: {cam_name} has no image data")
                    elif has_joint_data and state_frame_count > 0:
                        if abs(len(indices) - state_frame_count) > max(state_frame_count * 0.1, 1):
                            print(f"  ⚠️  Warning: the number of images ({len(indices)}) of {cam_name} is significantly different from the number of states ({state_frame_count})")
            # Image storage mode: check aligned_images
            else:
                for cam_name, cam_images in images.items():
                    if len(cam_images) == 0:
                        print(f"  ⚠️  Warning: {cam_name} has no image data")
                    elif has_joint_data and state_frame_count > 0:
                        if abs(len(cam_images) - state_frame_count) > max(state_frame_count * 0.1, 1):
                            print(f"  ⚠️  Warning: the number of images ({len(cam_images)}) of {cam_name} is significantly different from the number of states ({state_frame_count})")

        # Check the data amount
        max_frames = max(state_frame_count, max((len(imgs) for imgs in images.values()), default=0))
        min_frames = int(self.target_hz * 0.5)  # At least 0.5 seconds of data
        if max_frames < min_frames:
            print(f"  ⚠️  警告：数据量过少（{max_frames} 帧 < {min_frames} 帧）")

        # Print the data statistics of each part
        if joint_states_by_part:
            print(f"  ✓ Data validation completed: {len(joint_states_by_part)} parts of joint states, {state_frame_count} state frames, {len(images)} cameras")
            for part_name, data in joint_states_by_part.items():
                print(f"    - {part_name}: {len(data)} frames")
        else:
            print(f"  ✓ 数据校验完成：{state_frame_count} 帧 state，{len(images)} 路相机")
        return True

    def _clear_queues(self):
        """Clear all queues"""
        # Only clear the queues that actually exist
        for queue_name, queue in self.queues.items():
            if queue is not None:
                try:
                    while not queue.empty():
                        queue.get_nowait()  # Non-blocking get, avoid blocking
                except:
                    pass  # The queue may have been modified elsewhere, ignore the error

    def _create_temp_files(self):
        """Create temporary files for cameras and high-frequency data"""
        # Ensure that all temporary file dictionaries are initialized
        if not hasattr(self, 'image_temp_files'):
            self.image_temp_files = {}
        if not hasattr(self, 'image_temp_paths'):
            self.image_temp_paths = {}
        if not hasattr(self, 'sensor_temp_files'):
            self.sensor_temp_files = {}
        if not hasattr(self, 'sensor_temp_paths'):
            self.sensor_temp_paths = {}

        self._cleanup_temp_files()
        self._temp_file_last_flush.clear()

        temp_directory = None
        if self.raw_topic_storage:
            staging_dir = self._prepare_episode_staging_dir()
            raw_topics_dir = staging_dir / "raw_topics"
            raw_topics_dir.mkdir(parents=True, exist_ok=True)
            temp_directory = str(raw_topics_dir)

        def create_topic_temp_file(topic_name):
            if self.raw_topic_storage:
                return tempfile.NamedTemporaryFile(
                    mode='wb',
                    delete=False,
                    dir=temp_directory,
                    prefix=f".{topic_name}.",
                    suffix=".pending.pklseq",
                )
            return tempfile.NamedTemporaryFile(
                mode='wb',
                delete=False,
                suffix=f"_{topic_name}.pkl",
            )

        # Camera temporary files
        for camera_name in self.camera_names:
            temp_file = create_topic_temp_file(camera_name)
            self.image_temp_files[camera_name] = temp_file
            self.image_temp_paths[camera_name] = temp_file.name

        # Sensor data temporary files
        for sensor_name in self.sensor_file_locks.keys():
            temp_file = create_topic_temp_file(sensor_name)
            self.sensor_temp_files[sensor_name] = temp_file
            self.sensor_temp_paths[sensor_name] = temp_file.name

    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        cleaned_count = 0

        # Clean up camera temporary files
        if hasattr(self, 'image_temp_files'):
            for camera_name in list(self.image_temp_files.keys()):
                if camera_name in self.image_temp_files:
                    try:
                        if not self.image_temp_files[camera_name].closed:
                            self.image_temp_files[camera_name].close()
                    except Exception:
                        pass  # Ignore the error of closing
                    try:
                        if hasattr(self, 'image_temp_paths') and camera_name in self.image_temp_paths:
                            temp_path = self.image_temp_paths[camera_name]
                            if temp_path and os.path.exists(temp_path):
                                os.unlink(temp_path)
                                cleaned_count += 1
                    except Exception as e:
                        print(f"Error cleaning up camera temporary file ({camera_name}): {e}")

        # Clean up sensor data temporary files (now including joint state and action)
        if hasattr(self, 'sensor_temp_files'):
            for sensor_name in list(self.sensor_temp_files.keys()):
                if sensor_name in self.sensor_temp_files:
                    try:
                        if not self.sensor_temp_files[sensor_name].closed:
                            self.sensor_temp_files[sensor_name].close()
                    except Exception:
                        pass  # Ignore the error of closing
                    try:
                        if hasattr(self, 'sensor_temp_paths') and sensor_name in self.sensor_temp_paths:
                            temp_path = self.sensor_temp_paths[sensor_name]
                            if temp_path and os.path.exists(temp_path):
                                os.unlink(temp_path)
                                cleaned_count += 1
                    except Exception as e:
                        print(f"Error cleaning up sensor temporary file ({sensor_name}): {e}")

        # Clear the dictionaries (if they exist)
        if hasattr(self, 'image_temp_files'):
            self.image_temp_files.clear()
        if hasattr(self, 'image_temp_paths'):
            self.image_temp_paths.clear()
        if hasattr(self, 'sensor_temp_files'):
            self.sensor_temp_files.clear()
        if hasattr(self, 'sensor_temp_paths'):
            self.sensor_temp_paths.clear()
        if hasattr(self, '_temp_file_last_flush'):
            self._temp_file_last_flush.clear()

        # Additional cleanup: scan the temporary directory for possible leftover temporary files (with our suffix)
        # This can clean up files left over when the program exits abnormally
        try:
            temp_dir = tempfile.gettempdir()
            if os.path.isdir(temp_dir):
                # Find all pkl files with our suffix
                suffixes_to_clean = []
                if hasattr(self, 'camera_names'):
                    suffixes_to_clean.extend([f'_{cam}.pkl' for cam in self.camera_names])
                if hasattr(self, 'sensor_file_locks'):
                    suffixes_to_clean.extend([f'_{sensor}.pkl' for sensor in self.sensor_file_locks.keys()])
                if hasattr(self, 'slave_joint_names') and self.slave_joint_names:
                    suffixes_to_clean.extend([f'_{name}.pkl' for name in self.slave_joint_names])
                    if hasattr(self, 'slave_action_names') and self.slave_action_names:
                        suffixes_to_clean.extend([f'_{name}.pkl' for name in self.slave_action_names])

                for filename in os.listdir(temp_dir):
                    if filename.startswith('tmp') and filename.endswith('.pkl'):
                        # Check if it matches our suffix
                        for suffix in suffixes_to_clean:
                            if filename.endswith(suffix):
                                temp_file_path = os.path.join(temp_dir, filename)
                                try:
                                    # Check if the file is very old (more than 1 hour) or is 0 bytes
                                    file_stat = os.stat(temp_file_path)
                                    file_age = time.time() - file_stat.st_mtime
                                    if file_age > 3600 or file_stat.st_size == 0:
                                        os.unlink(temp_file_path)
                                        cleaned_count += 1
                                except Exception:
                                    pass  # Ignore the error of deleting
                                break
        except Exception as e:
            # Ignore the error of scanning, does not affect main flow
            pass

        if cleaned_count > 0:
            print(f"✓ 已清理 {cleaned_count} 个临时文件")

    def _collect_episode_data(self):
        """Collect episode data from queues and temporary files"""
        # Collect data from sensor temporary files
        sensor_data = {}
        episode_data = {}

        for sensor_name in self.sensor_file_locks.keys():
            if sensor_name not in self.sensor_temp_files:
                continue

            # Close the write handle
            with self.sensor_file_locks[sensor_name]:
                self.sensor_temp_files[sensor_name].close()

            # Reopen the file for reading
            temp_path = self.sensor_temp_paths[sensor_name]
            if not os.path.exists(temp_path):
                print(f"⚠️  Warning: the sensor temporary file does not exist {sensor_name}: {temp_path}")
                continue

            data_list = []
            try:
                with open(temp_path, 'rb') as f:
                    while True:
                        try:
                            data = pickle.load(f)
                            data_list.append(data)
                        except EOFError:
                            break  # File end
                        except Exception as e:
                            print(f"Error reading {sensor_name} data: {e}")
                            break

                if data_list:
                    sensor_data[sensor_name] = data_list
                    print(f"  {sensor_name}：采集到 {len(data_list)} 条数据")

            except Exception as e:
                print(f"Error reading {sensor_name} temporary file: {e}")

        # Read the joint state and action data from the temporary files
        print("正在从临时文件读取状态和动作数据...")

        # Use the joint state and action names in the member variable
        joint_state_names = self.slave_joint_names if self.slave_joint_names else []
        action_names = self.slave_action_names if self.slave_action_names else []

        # Read the joint state data of each part
        for data_type in joint_state_names:
            if not hasattr(self, 'sensor_temp_files') or data_type not in self.sensor_temp_files:
                continue

            # Close the write handle
            if data_type in self.sensor_file_locks:
                with self.sensor_file_locks[data_type]:
                    if data_type in self.sensor_temp_files:
                        self.sensor_temp_files[data_type].close()
            else:
                if data_type in self.sensor_temp_files:
                    self.sensor_temp_files[data_type].close()

            # Reopen the file for reading
            temp_path = self.sensor_temp_paths.get(data_type)
            if not temp_path:
                print(f"⚠️  Warning: the temporary file path does not exist for {data_type}")
                continue
            if not os.path.exists(temp_path):
                print(f"⚠️  Warning: the joint state temporary file does not exist {data_type}: {temp_path}")
                continue

            data_list = []
            try:
                with open(temp_path, 'rb') as f:
                    while True:
                        try:
                            data = pickle.load(f)
                            data_list.append(data)
                        except EOFError:
                            break  # File end
                        except Exception as e:
                            print(f"Error reading {data_type} data: {e}")
                            break

                if data_list:
                    sensor_data[data_type] = data_list
                    print(f"  {data_type}：采集到 {len(data_list)} 条数据")
                else:
                    print(f"  ⚠️  Warning: the temporary file exists for {data_type} but the data is empty")

            except Exception as e:
                print(f"Error reading {data_type} temporary file: {e}")
                import traceback
                traceback.print_exc()

        # Read the action data of each part
        for data_type in action_names:
            if not hasattr(self, 'sensor_temp_files') or data_type not in self.sensor_temp_files:
                continue

            # Close the write handle
            if data_type in self.sensor_file_locks:
                with self.sensor_file_locks[data_type]:
                    if data_type in self.sensor_temp_files:
                        self.sensor_temp_files[data_type].close()
            else:
                if data_type in self.sensor_temp_files:
                    self.sensor_temp_files[data_type].close()

            # Reopen the file for reading
            temp_path = self.sensor_temp_paths[data_type]
            if not os.path.exists(temp_path):
                print(f"⚠️  Warning: the action data temporary file does not exist {data_type}: {temp_path}")
                continue

            data_list = []
            try:
                with open(temp_path, 'rb') as f:
                    while True:
                        try:
                            data = pickle.load(f)
                            data_list.append(data)
                        except EOFError:
                            break  # File end
                        except Exception as e:
                            print(f"Error reading {data_type} data: {e}")
                            break

                if data_list:
                    sensor_data[data_type] = data_list
                    print(f"  {data_type}：采集到 {len(data_list)} 条数据")
                else:
                    # Skip the warning for action data (action data is not required)
                    if 'action' not in data_type.lower():
                        print(f"  ⚠️  {data_type}: the temporary file exists but the data is empty")

            except Exception as e:
                print(f"Error reading {data_type} temporary file: {e}")
                import traceback
                traceback.print_exc()

        # use video storage to read image data
        if self.use_video_storage:
            # Video mode: only read the timestamp information, not load the image data
            images = defaultdict(list)
            print("正在从临时文件读取图像时间戳...")

            for camera_name in self.camera_names:
                if camera_name not in self.image_temp_files:
                    continue

                # Close the write handle
                with self.image_file_locks[camera_name]:
                    self.image_temp_files[camera_name].close()

                # Reopen the file for reading, only read the timestamp
                temp_path = self.image_temp_paths[camera_name]
                if not os.path.exists(temp_path):
                    print(f"⚠️  Warning: the temporary file does not exist {camera_name}: {temp_path}")
                    continue

                try:
                    with open(temp_path, 'rb') as f:
                        while True:
                            try:
                                timestamp, img_data = pickle.load(f)
                                # Only save the timestamp, not save the image data to save memory
                                # img_data is now compressed JPEG bytes, no need to load
                                images[camera_name].append((timestamp, None))
                            except EOFError:
                                break

                    print(f"  {camera_name}：读取到 {len(images[camera_name])} 帧时间戳")

                except Exception as e:
                    print(f"Error reading temporary file ({camera_name}): {e}")
        else:
            # Image mode: load all images normally
            images = defaultdict(list)
            print("正在从临时文件读取图像数据...")

            for camera_name in self.camera_names:
                if camera_name not in self.image_temp_files:
                    continue

                # Close the write handle
                with self.image_file_locks[camera_name]:
                    self.image_temp_files[camera_name].close()

                # Reopen the file for reading
                temp_path = self.image_temp_paths[camera_name]
                if not os.path.exists(temp_path):
                    print(f"⚠️  Warning: the temporary file does not exist {camera_name}: {temp_path}")
                    continue

                try:
                    with open(temp_path, 'rb') as f:
                        while True:
                            try:
                                timestamp, img_data = pickle.load(f)
                                # img_data is compressed JPEG bytes, need to convert back to PIL Image
                                if isinstance(img_data, bytes):
                                    # Compressed JPEG data, need to decode
                                    img = Image.open(io.BytesIO(img_data))
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                else:
                                    # Compatible with old format: if it is already a PIL Image object (backward compatible)
                                    img = img_data
                                images[camera_name].append((timestamp, img))
                            except EOFError:
                                break

                    print(f"  {camera_name}：读取到 {len(images[camera_name])} 帧")

                except Exception as e:
                    print(f"Error reading temporary file ({camera_name}): {e}")

        # Validate before data alignment: check if all enabled data items have actual data
        # This can avoid the error of missing data during alignment
        validation_result = self._validate_collected_data(sensor_data, images)
        if not validation_result:
            print("错误：数据校验失败，部分已启用的数据项没有采集到实际数据")
            return None

        # Optionally persist the pre-alignment raw streams before temp files are
        # cleaned up. Done after validation so we don't leave raw_data for an
        # episode that ultimately fails to save.
        if self.keep_raw_data:
            self._save_raw_data(sensor_data)

        # Align data by timestamp interpolation
        print("正在按时间戳对齐数据...")
        episode_data = self._align_data_by_timestamp(sensor_data, images)

        if episode_data is None:
            print("⚠️  警告：数据对齐失败")
            return None

        # Note: when using video storage, do not clean up temporary files here, clean up after saving the video
        if not self.use_video_storage:
            self._cleanup_temp_files()

        return episode_data

    def _save_raw_data(self, sensor_data):
        """Persist pre-alignment raw streams into episode_xxxx/raw_data/.

        Copies the raw sensor/pose/joint/action temp files (exact, lossless,
        same `(timestamp, value)` pickle stream the collectors wrote) and writes
        a manifest.json describing each file. Camera frames are intentionally
        excluded (large, and already stored as video/images).

        Must run while the temp files still exist, i.e. after read-back and
        before _cleanup_temp_files(). episode_count is the index of the episode
        currently being saved (matches _save_episode).
        """
        try:
            episode_dir = self._prepare_episode_staging_dir()
            raw_dir = episode_dir / "raw_data"
            raw_dir.mkdir(parents=True, exist_ok=True)

            def describe(records):
                if not records:
                    return "empty"
                rec = records[0]
                if isinstance(rec, tuple) and len(rec) == 4:
                    return "(timestamp, positions, velocities, efforts)"
                if isinstance(rec, tuple) and len(rec) == 2:
                    val = rec[1]
                    if isinstance(val, dict):
                        return f"(timestamp, dict[{', '.join(val.keys())}])"
                    return f"(timestamp, {type(val).__name__})"
                return f"record_type={type(rec).__name__}"

            manifest = {
                "episode_id": self.episode_count,
                "created": datetime.now().isoformat(),
                "note": "Pre-alignment raw streams. Each .pkl is a sequence of "
                        "pickled records; read with repeated pickle.load until EOF.",
                "files": {},
            }

            copied = 0
            for name, temp_path in self.sensor_temp_paths.items():
                if not temp_path or not os.path.exists(temp_path):
                    continue
                try:
                    shutil.copy2(temp_path, raw_dir / f"{name}.pkl")
                    manifest["files"][f"{name}.pkl"] = {
                        "records": len(sensor_data.get(name, [])),
                        "record_format": describe(sensor_data.get(name, [])),
                    }
                    copied += 1
                except Exception as e:
                    print(f"  ⚠️  Failed to copy raw data for {name}: {e}")

            with open(raw_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            print(f"  ✓ Saved raw data: {copied} streams -> {raw_dir}")
        except Exception as e:
            # Never let raw-data saving break the main save flow
            print(f"  ⚠️  Failed to save raw data: {e}")

    def _align_data_by_timestamp(self, sensor_data, images):
        """Align all data to a uniform time grid based on timestamp"""

        # Camera name mapping: internal stream name -> user friendly name
        camera_name_mapping = {
            'head_rgb_stream': 'head_camera',
            'head_depth_stream': 'head_depth_camera',
            'left_arm_rgb_stream': 'left_arm_camera',
            'right_arm_rgb_stream': 'right_arm_camera'
        }

        # 1. Align image data frames based on the timestamp of the main camera (head camera)
        # First find the internal key name of the main camera
        head_camera_internal_name = None
        for internal_name, friendly_name in camera_name_mapping.items():
            if friendly_name == 'head_camera' and internal_name in images and images[internal_name]:
                head_camera_internal_name = internal_name
                break

        if not head_camera_internal_name:
            print("⚠️  警告：主相机 head_camera 没有数据，无法进行对齐")
            print(f"   当前可用图像流：{list(images.keys())}")
            return None

        # Use the timestamp of the main camera as the baseline time line
        target_timestamps = [t for t, _ in images[head_camera_internal_name]]
        num_frames = len(target_timestamps)

        print(f"   使用主相机 {head_camera_internal_name} 的时间戳作为基准：{num_frames} 帧")

        # Count the data of each sensor
        sensor_stats = [(name, len(data_list)) for name, data_list in sensor_data.items() if data_list]
        hf_stats = [(name, len(data_list)) for name, data_list in sensor_data.items() if data_list and name in ['joint_states', 'actions']]
        image_stats = [(cam, len(imgs)) for cam, imgs in images.items() if imgs]
        print(f"   传感器数据：{sensor_stats}")
        print(f"   高频数据：{hf_stats}")
        print(f"   图像数据：{image_stats}")

        # 2. Align the data of each sensor by timestamp
        aligned_sensor_data = {}
        aligned_action_data = {}

        # Use the joint state and action names in the member variable
        joint_state_names = self.slave_joint_names if self.slave_joint_names else []
        action_names = self.slave_action_names if self.slave_action_names else []

        for sensor_name, data_list in sensor_data.items():
            if data_list:
                # End-effector pose: position is linearly interpolated and the
                # orientation quaternion is SLERP'd (component-wise lerp of a
                # quaternion is incorrect). This avoids the stair-step / jitter
                # that nearest-neighbor produces during playback.
                if sensor_name.endswith('_end_pose') and self.downsample_joint_states:
                    print(f"   对 {sensor_name} 使用位置线性插值和四元数 SLERP")
                    aligned_sensor_data[sensor_name] = self._interpolate_pose(data_list, target_timestamps)
                elif sensor_name in self.vr_action_sensor_names:
                    print(f"   对 {sensor_name} 使用因果前值对齐")
                    aligned_sensor_data[sensor_name] = self._interpolate_previous(
                        data_list,
                        target_timestamps,
                    )
                # When downsampling, the joint state and action need linear interpolation
                elif sensor_name in joint_state_names + action_names and self.downsample_joint_states:
                    print(f"   Use linear interpolation to align the data of {sensor_name}")
                    if sensor_name in action_names:
                        aligned_action_data[sensor_name] = self._interpolate_linear(data_list, target_timestamps)
                    else:
                        aligned_sensor_data[sensor_name] = self._interpolate_linear(data_list, target_timestamps)
                else:
                    # Other sensor data use nearest neighbor interpolation
                    if sensor_name in action_names:
                        aligned_action_data[sensor_name] = self._interpolate_nearest(data_list, target_timestamps)
                    else:
                        aligned_sensor_data[sensor_name] = self._interpolate_nearest(data_list, target_timestamps)

        # Align the image data by timestamp (use nearest neighbor interpolation, because the image data is already the baseline)
        aligned_images = {}
        image_index_mapping = {}  # Used for frame index mapping in video mode

        if self.use_video_storage:
            # Video mode: only create the mapping from timestamp to frame index, not load the image
            for cam_name, cam_imgs in images.items():
                friendly_name = camera_name_mapping.get(cam_name, cam_name)
                if cam_name == head_camera_internal_name:
                    # Main camera: create 1:1 mapping
                    image_index_mapping[friendly_name] = list(range(len(cam_imgs)))
                    aligned_images[friendly_name] = []  # Empty list
                else:
                    # Other cameras: create the mapping from timestamp to frame index
                    cam_timestamps = [t for t, _ in cam_imgs]
                    indices = self._interpolate_nearest_indices(cam_timestamps, target_timestamps)
                    image_index_mapping[friendly_name] = indices
                    aligned_images[friendly_name] = []  # Empty list
        else:
            # Image mode: align the image data normally
            for cam_name, cam_imgs in images.items():
                friendly_name = camera_name_mapping.get(cam_name, cam_name)
                if cam_name == head_camera_internal_name:
                    # Main camera: use directly, no interpolation
                    aligned_images[friendly_name] = [img for _, img in cam_imgs]
                else:
                    # Other cameras use nearest neighbor interpolation to align to the timestamp of the main camera
                    aligned_images[friendly_name] = self._interpolate_nearest(cam_imgs, target_timestamps)

        # Stream subscriptions start concurrently, so one or more initial camera
        # frames can precede the first VR command. Drop only that leading prefix;
        # never fill it with a future command.
        if self.collection_config.enable_vr_action_commands:
            first_complete_command_frame = next(
                (
                    index
                    for index in range(len(target_timestamps))
                    if all(
                        index < len(aligned_sensor_data.get(sensor_name, []))
                        and aligned_sensor_data[sensor_name][index] is not None
                        for sensor_name in self.vr_action_sensor_names
                    )
                ),
                None,
            )
            if first_complete_command_frame is None:
                raise RuntimeError(
                    "没有任何相机帧同时具备"
                    f"{len(self.vr_action_sensor_names)} 路有效指令；"
                    "请先启动 VR 遥操，再开始采集"
                )
            if first_complete_command_frame:
                print(
                    f"   丢弃开头 {first_complete_command_frame} 帧；"
                    "这些帧尚未同时收到全部有效指令"
                )
                target_timestamps = target_timestamps[first_complete_command_frame:]
                aligned_sensor_data = {
                    name: values[first_complete_command_frame:]
                    for name, values in aligned_sensor_data.items()
                }
                aligned_action_data = {
                    name: values[first_complete_command_frame:]
                    for name, values in aligned_action_data.items()
                }
                if self.use_video_storage:
                    image_index_mapping = {
                        name: indices[first_complete_command_frame:]
                        for name, indices in image_index_mapping.items()
                    }
                else:
                    aligned_images = {
                        name: values[first_complete_command_frame:]
                        for name, values in aligned_images.items()
                    }

        print(f"   对齐后的帧数：{len(target_timestamps)}")

        # Ensure joint_names is not None - infer the joint names from the joint state data of each part
        joint_names = self.joint_names
        if not joint_names or not isinstance(joint_names, dict):
            # Use the saved real joint names first
            if hasattr(self, '_joint_names_by_part') and self._joint_names_by_part:
                joint_names = {}
                for joint_state_name, names in self._joint_names_by_part.items():
                    part_name = joint_state_name.replace('_joint_states', '')
                    joint_names[part_name] = names
                print(f"✓ Use the saved joint names: {len(joint_names)} parts")
                for part_name, names in joint_names.items():
                    print(f"    - {part_name}: {names}")
            else:
                # Infer the joint names from the joint state data of each part (separated by parts)
                joint_names = {}
                for joint_state_name in joint_state_names:
                    if joint_state_name in aligned_sensor_data and len(aligned_sensor_data[joint_state_name]) > 0:
                        # Infer the number of joints from the first frame data
                        first_state = aligned_sensor_data[joint_state_name][0]
                        if isinstance(first_state, tuple) and len(first_state) > 0:
                            num_joints = len(first_state[0])
                        else:
                            num_joints = len(first_state) if hasattr(first_state, '__len__') else 0

                        # Generate default joint names for each part
                        part_name = joint_state_name.replace('_joint_states', '')
                        part_joint_names = []
                        for i in range(num_joints):
                            part_joint_names.append(f"{part_name}_joint{i+1}")
                        joint_names[part_name] = part_joint_names

                if joint_names:
                    total_joints = sum(len(names) for names in joint_names.values())
                    print(f"⚠️  joint_names is empty, infer the default names from the data: {len(joint_names)} parts, total {total_joints} joints")
                    for part_name, names in joint_names.items():
                        print(f"    - {part_name}: {len(names)} joints")

        episode_data = {
            'timestamps': target_timestamps,
            'sensor_data': aligned_sensor_data,  # Normal sensor data
            'action_data': aligned_action_data,  # Action data
            'images': aligned_images,  # Image data
            'joint_names': joint_names,
        }

        # Video mode: add the image index mapping and the original camera name mapping
        if self.use_video_storage:
            episode_data['image_index_mapping'] = image_index_mapping
            episode_data['camera_name_mapping'] = camera_name_mapping

        print("✓ 时间戳对齐完成")
        return episode_data

    def _interpolate_linear(self, data_with_timestamps, target_timestamps):
        """Use linear interpolation to align the data to the target timestamps"""
        import numpy as np

        if not data_with_timestamps:
            return []

        sorted_data = sorted(data_with_timestamps, key=lambda x: x[0])
        aligned_data = []

        # Check the data format: the joint state is a 4 element tuple, other is a 2 element tuple
        first_item = sorted_data[0]
        is_joint_state = len(first_item) == 4  # The joint state format: (timestamp, positions, velocities, efforts)

        # Extract the timestamp and data
        timestamps = np.array([item[0] for item in sorted_data])

        if is_joint_state:
            # The joint state data: positions, velocities, efforts
            positions = np.array([item[1] for item in sorted_data])
            velocities = np.array([item[2] for item in sorted_data])
            efforts = np.array([item[3] for item in sorted_data])
        else:
            # Other data format
            data_values = np.array([item[1] for item in sorted_data])

        for target_t in target_timestamps:
            if target_t <= timestamps[0]:
                # Use the first data point
                if is_joint_state:
                    aligned_data.append((sorted_data[0][1], sorted_data[0][2], sorted_data[0][3]))
                else:
                    aligned_data.append(sorted_data[0][1])
            elif target_t >= timestamps[-1]:
                # Use the last data point
                if is_joint_state:
                    aligned_data.append((sorted_data[-1][1], sorted_data[-1][2], sorted_data[-1][3]))
                else:
                    aligned_data.append(sorted_data[-1][1])
            else:
                # Linear interpolation
                # Find the interval where target_t is located
                idx = np.searchsorted(timestamps, target_t) - 1
                if idx < 0:
                    idx = 0

                t1, t2 = timestamps[idx], timestamps[idx + 1]
                ratio = (target_t - t1) / (t2 - t1) if t2 != t1 else 0

                if is_joint_state:
                    # Linear interpolation for positions, velocities, efforts separately
                    pos1, pos2 = positions[idx], positions[idx + 1]
                    vel1, vel2 = velocities[idx], velocities[idx + 1]
                    eff1, eff2 = efforts[idx], efforts[idx + 1]

                    interpolated_pos = pos1 + ratio * (pos2 - pos1)
                    interpolated_vel = vel1 + ratio * (vel2 - vel1)
                    interpolated_eff = eff1 + ratio * (eff2 - eff1)

                    aligned_data.append((interpolated_pos, interpolated_vel, interpolated_eff))
                else:
                    # Linear interpolation for single value or array
                    val1, val2 = data_values[idx], data_values[idx + 1]
                    interpolated_val = val1 + ratio * (val2 - val1)
                    aligned_data.append(interpolated_val)

        return aligned_data

    def _interpolate_pose(self, data_with_timestamps, target_timestamps):
        """Align end-effector pose records to target timestamps.

        Position (x, y, z) is linearly interpolated; the orientation quaternion
        is interpolated with SLERP (shortest path). Records are
        ``(timestamp, {'position': {...}, 'orientation': {...}})`` and the
        output preserves that dict structure. Robust to duplicate/boundary
        timestamps.
        """
        if not data_with_timestamps:
            return []

        sorted_data = sorted(data_with_timestamps, key=lambda x: x[0])
        ts = [r[0] for r in sorted_data]
        n = len(sorted_data)

        def pos_of(d):
            p = d['position']
            return np.array([p['x'], p['y'], p['z']], dtype=float)

        def quat_of(d):
            o = d['orientation']
            return np.array([o['x'], o['y'], o['z'], o['w']], dtype=float)

        def make(pos, quat):
            return {
                'position': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])},
                'orientation': {'x': float(quat[0]), 'y': float(quat[1]),
                                'z': float(quat[2]), 'w': float(quat[3])},
            }

        def slerp(q0, q1, t):
            dot = float(np.dot(q0, q1))
            # take the shortest path on the quaternion hypersphere
            if dot < 0.0:
                q1 = -q1
                dot = -dot
            if dot > 0.9995:
                # nearly aligned -> normalized linear interpolation is accurate
                q = q0 + t * (q1 - q0)
            else:
                theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
                sin0 = np.sin(theta0)
                s0 = np.sin((1.0 - t) * theta0) / sin0
                s1 = np.sin(t * theta0) / sin0
                q = s0 * q0 + s1 * q1
            norm = np.linalg.norm(q)
            return q / norm if norm > 0 else q0

        aligned_data = []
        for target_t in target_timestamps:
            pos = bisect.bisect_left(ts, target_t)
            if pos <= 0:
                d = sorted_data[0][1]
                aligned_data.append(make(pos_of(d), quat_of(d)))
                continue
            if pos >= n:
                d = sorted_data[-1][1]
                aligned_data.append(make(pos_of(d), quat_of(d)))
                continue

            i1, i2 = pos - 1, pos
            t1, t2 = ts[i1], ts[i2]
            d1, d2 = sorted_data[i1][1], sorted_data[i2][1]
            ratio = 0.0 if t2 <= t1 else min(1.0, max(0.0, (target_t - t1) / (t2 - t1)))

            p1, p2 = pos_of(d1), pos_of(d2)
            interp_pos = p1 + ratio * (p2 - p1)
            interp_quat = slerp(quat_of(d1), quat_of(d2), ratio)
            aligned_data.append(make(interp_pos, interp_quat))

        return aligned_data

    def _interpolate_nearest(self, data_with_timestamps, target_timestamps):
        """Use nearest neighbor interpolation to align the data to the target timestamps.

        Robust to repeated/duplicate timestamps in the source data. The previous
        forward-scan implementation advanced its cursor only when the diff strictly
        decreased and broke on the first non-decrease; a run of identical source
        timestamps (which the end-pose stream produces frequently) made the diff
        plateau, so the cursor stalled on the first element of that run and every
        later frame collapsed to the same value (the "frozen pose" bug).
        """
        if not data_with_timestamps:
            return []

        sorted_data = sorted(data_with_timestamps, key=lambda x: x[0])
        timestamps = [item[0] for item in sorted_data]
        n = len(sorted_data)

        def value_at(idx):
            item = sorted_data[idx]
            if len(item) == 4:  # joint state: (timestamp, positions, velocities, efforts)
                return (item[1], item[2], item[3])
            return item[1]  # other format: (timestamp, data)

        aligned_data = []
        for target_t in target_timestamps:
            pos = bisect.bisect_left(timestamps, target_t)
            if pos <= 0:
                best = 0
            elif pos >= n:
                best = n - 1
            else:
                # nearest of the two neighbours straddling target_t
                best = pos if (timestamps[pos] - target_t) < (target_t - timestamps[pos - 1]) else pos - 1
            aligned_data.append(value_at(best))

        return aligned_data

    def _interpolate_previous(self, data_with_timestamps, target_timestamps):
        """Causally align commands with the latest value at or before each target.

        A target earlier than the first command maps to ``None`` instead of
        borrowing a future command.  The episode writer rejects such frames so
        command/state datasets cannot silently contain future leakage.
        """
        if not data_with_timestamps:
            return []

        sorted_data = sorted(data_with_timestamps, key=lambda x: x[0])
        timestamps = [item[0] for item in sorted_data]
        aligned_data = []
        for target_t in target_timestamps:
            index = bisect.bisect_right(timestamps, target_t) - 1
            aligned_data.append(
                None if index < 0 else sorted_data[index][1]
            )
        return aligned_data

    def _interpolate_nearest_indices(self, timestamps, target_timestamps):
        """Use nearest neighbor interpolation to return the index mapping (for video mode).

        Assumes `timestamps` is sorted ascending (camera frame timestamps are
        monotonic). Robust to duplicate timestamps (see _interpolate_nearest).
        """
        if not timestamps:
            return []

        n = len(timestamps)
        indices = []

        for target_t in target_timestamps:
            pos = bisect.bisect_left(timestamps, target_t)
            if pos <= 0:
                closest_idx = 0
            elif pos >= n:
                closest_idx = n - 1
            else:
                closest_idx = pos if (timestamps[pos] - target_t) < (target_t - timestamps[pos - 1]) else pos - 1

            indices.append(closest_idx)

        return indices

    def _create_video_with_ffmpeg(self, temp_path: str, indices: List[int], output_path: str, fps: float, expected_frames: int) -> bool:
        """Create one aligned MP4 while keeping only one source frame in memory."""
        if (
            not indices
            or expected_frames <= 0
            or not os.path.exists(temp_path)
        ):
            print("  ⚠️  没有可编码的图像数据或临时文件不存在")
            return False
        if not np.isfinite(fps) or float(fps) <= 0:
            print(f"  ⚠️  视频帧率无效：{fps}")
            return False
        if len(indices) != expected_frames:
            print(
                "  ⚠️  视频索引数量与目标帧数不一致："
                f"{len(indices)} != {expected_frames}"
            )
            return False

        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            for index in indices
        ):
            print("  ⚠️  视频索引必须全部为整数")
            return False
        requested_indices = [int(index) for index in indices]
        if any(index < 0 for index in requested_indices):
            print("  ⚠️  视频索引中存在负数")
            return False
        if any(
            current < previous
            for previous, current in zip(
                requested_indices,
                requested_indices[1:],
            )
        ):
            print("  ⚠️  视频索引不是单调递增，无法顺序流式编码")
            return False

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        partial_output_path = output_path_obj.with_name(
            f".{output_path_obj.stem}.{os.getpid()}."
            f"{time.time_ns()}.part.mp4"
        )

        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-nostats",
            "-xerror",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-framerate", str(fps),
            "-i", "pipe:0",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-threads", "4",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-frames:v", str(expected_frames),
            "-y",
            str(partial_output_path),
        ]

        process = None
        frames_written = 0
        source_frames_read = 0
        current_source_index = -1
        current_jpeg = None
        completed = False

        try:
            with (
                open(temp_path, 'rb') as image_file,
                tempfile.TemporaryFile() as stderr_file,
            ):
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                    bufsize=0,
                )
                if process.stdin is None:
                    raise RuntimeError("无法打开 ffmpeg 输入管道")

                import select as select_module

                stdin_fd = process.stdin.fileno()
                os.set_blocking(stdin_fd, False)

                def write_with_timeout(payload, timeout=30.0):
                    remaining_payload = memoryview(payload)
                    deadline = time.monotonic() + timeout
                    while remaining_payload:
                        if process.poll() is not None:
                            raise BrokenPipeError(
                                f"ffmpeg 已提前退出：{process.returncode}"
                            )
                        remaining_time = deadline - time.monotonic()
                        if remaining_time <= 0:
                            raise TimeoutError(
                                f"向 ffmpeg 写入单帧超过 {timeout:.1f} 秒"
                            )
                        _, writable, _ = select_module.select(
                            [],
                            [stdin_fd],
                            [],
                            remaining_time,
                        )
                        if not writable:
                            continue
                        bytes_written = os.write(
                            stdin_fd,
                            remaining_payload,
                        )
                        if bytes_written <= 0:
                            raise BrokenPipeError("ffmpeg 输入管道已关闭")
                        remaining_payload = remaining_payload[bytes_written:]

                for requested_index in requested_indices:
                    source_record = None
                    while current_source_index < requested_index:
                        try:
                            source_record = pickle.load(image_file)
                        except EOFError:
                            raise RuntimeError(
                                "视频索引超出源图像范围："
                                f"{requested_index} > "
                                f"{current_source_index}"
                            )

                        current_source_index += 1
                        source_frames_read += 1

                    if source_record is not None:
                        if (
                            not isinstance(source_record, tuple)
                            or len(source_record) != 2
                        ):
                            raise ValueError(
                                f"源图像 {current_source_index} "
                                "的临时记录格式无效"
                            )
                        _, image_data = source_record
                        try:
                            current_jpeg = self._image_to_jpeg_bytes(
                                image_data
                            )
                        except Exception as error:
                            raise RuntimeError(
                                f"源图像 {current_source_index} "
                                f"无法转换：{error}"
                            ) from error

                    if current_jpeg is None:
                        raise RuntimeError("没有有效源图像可供编码")

                    write_with_timeout(current_jpeg)
                    frames_written += 1

                if process.stdin and not process.stdin.closed:
                    process.stdin.close()

                video_duration = expected_frames / max(float(fps), 1.0)
                wait_timeout = max(60.0, video_duration * 2.0 + 30.0)
                try:
                    returncode = process.wait(timeout=wait_timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
                    print(
                        "  ⚠️  ffmpeg 编码超时："
                        f"{wait_timeout:.1f} 秒"
                    )
                    return False

                stderr_file.seek(0)
                stderr = stderr_file.read()

            print(
                "    流式读取 "
                f"{source_frames_read} 个源帧，写入 "
                f"{frames_written}/{expected_frames} 个视频帧"
            )
            if returncode != 0:
                error_message = stderr.decode(
                    'utf-8',
                    errors='ignore',
                ).strip()
                print(
                    f"  ⚠️  ffmpeg 返回 {returncode}："
                    f"{error_message[:500] or '无错误输出'}"
                )
                return False
            if frames_written != expected_frames:
                print(
                    "  ⚠️  实际写入帧数与目标不一致："
                    f"{frames_written} != {expected_frames}"
                )
                return False

            probe_result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-count_frames",
                    "-show_entries", "stream=nb_read_frames",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(partial_output_path),
                ],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            if probe_result.returncode != 0:
                raise RuntimeError(
                    "ffprobe 校验失败："
                    f"{probe_result.stderr.strip()[:500]}"
                )
            try:
                probed_frames = int(probe_result.stdout.strip())
            except ValueError as error:
                raise RuntimeError(
                    "ffprobe 未返回有效帧数："
                    f"{probe_result.stdout!r}"
                ) from error
            if probed_frames != expected_frames:
                raise RuntimeError(
                    "ffprobe 视频帧数不一致："
                    f"{probed_frames} != {expected_frames}"
                )

            os.replace(partial_output_path, output_path_obj)
            completed = True
            return True

        except FileNotFoundError:
            print("  ⚠️  未安装 ffmpeg，请执行：sudo apt-get install ffmpeg")
            return False
        except Exception as error:
            print(f"  ⚠️  流式视频生成失败：{error}")
            return False
        finally:
            if process is not None and process.poll() is None:
                try:
                    if process.stdin and not process.stdin.closed:
                        process.stdin.close()
                except Exception:
                    pass
                process.kill()
                try:
                    process.wait(timeout=5.0)
                except Exception:
                    pass
            if not completed and partial_output_path.exists():
                try:
                    partial_output_path.unlink()
                except Exception:
                    pass

    def _create_video_with_ffmpeg_legacy(self, temp_path: str, indices: List[int], output_path: str, fps: float, expected_frames: int) -> bool:
        """Use ffmpeg to create a video (streaming processing, saving memory)

        Args:
            temp_path: temporary image file path
            indices: index mapping list, specify which original image should be used for each frame
            output_path: output video path
            fps: frame rate
            expected_frames: expected number of frames

        Returns:
            True if successful, False if failed
        """
        if not indices or not os.path.exists(temp_path):
            print(f"  ⚠️  Warning: no image data or temporary file does not exist, cannot create video")
            return False

        try:
            # Build ffmpeg command
            ffmpeg_cmd = [
                "ffmpeg",
                "-r", str(fps),
                "-f", "image2pipe",
                "-vcodec", "mjpeg",
                "-pix_fmt", "yuvj420p",
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "medium",
                "-crf", "23",
                "-y",
                output_path
            ]

            # Start ffmpeg process
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )

            # First load all original images into memory (to avoid repeated reading of files)
            # But this is separated by cameras, so the memory pressure is much smaller
            all_images = []
            with open(temp_path, 'rb') as f:
                while True:
                    try:
                        _, img_data = pickle.load(f)
                        # img_data may be compressed JPEG bytes or PIL Image object (backward compatibility)
                        if isinstance(img_data, bytes):
                            # Compressed JPEG data, need to decode to PIL Image
                            img = Image.open(io.BytesIO(img_data))
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                        elif isinstance(img_data, Image.Image):
                            # Already PIL Image object (backward compatibility for old format)
                            img = img_data
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                        else:
                            # Other format (e.g. numpy array)
                            img = img_data
                        all_images.append(img)
                    except EOFError:
                        break

            print(f"    Loaded {len(all_images)} original images, need to align to {expected_frames} frames")

            # Get the size of the first frame (for creating black frame)
            first_img = all_images[0] if all_images else None
            if first_img is None:
                print(f"  ⚠️  Warning: no valid first frame")
                process.stdin.close()
                process.kill()
                return False

            if isinstance(first_img, Image.Image):
                width, height = first_img.size
            elif isinstance(first_img, np.ndarray):
                height, width = first_img.shape[:2]
            else:
                print(f"  ⚠️  Warning: cannot determine the image size (type: {type(first_img)})")
                process.stdin.close()
                process.kill()
                return False

            # Create black frame (for error frame)
            black_frame = Image.new('RGB', (width, height), (0, 0, 0))

            # Streaming processing: read, convert and write frame by frame
            frames_written = 0
            for frame_idx, img_idx in enumerate(indices):
                try:
                    pil_img = None

                    # Get the corresponding original image
                    if img_idx >= len(all_images):
                        print(f"  ⚠️  Warning: index out of range {img_idx}/{len(all_images)}, using black frame")
                        pil_img = black_frame
                    else:
                        img = all_images[img_idx]

                        if img is None:
                            print(f"  ⚠️  Warning: frame {frame_idx} image is empty, using black frame")
                            pil_img = black_frame
                        else:
                            # Convert to PIL Image
                            # img should already be PIL Image (processed when loaded)
                            if isinstance(img, Image.Image):
                                pil_img = img
                            elif isinstance(img, np.ndarray):
                                pil_img = Image.fromarray(img)
                            elif isinstance(img, bytes):
                                # If it is bytes (should not happen, because it is processed when loaded), try to decode
                                pil_img = Image.open(io.BytesIO(img))
                                if pil_img.mode != 'RGB':
                                    pil_img = pil_img.convert('RGB')
                            else:
                                print(f"  ⚠️  Warning: frame {frame_idx} format not supported: {type(img)}, using black frame")
                                pil_img = black_frame

                    # Ensure it is in RGB mode
                    if pil_img and pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')

                    # Save as JPEG and write to pipe
                    if pil_img:
                        buffer = io.BytesIO()
                        pil_img.save(buffer, format='JPEG', quality=95)
                        process.stdin.write(buffer.getvalue())
                        frames_written += 1
                        del buffer

                    # Periodically release the processed images (if not needed anymore)
                    if frame_idx % 100 == 99:
                        # Find the indices of the images that will not be used in the subsequent frames
                        remaining_indices = set(indices[frame_idx+1:])
                        for i in range(len(all_images)):
                            if i not in remaining_indices and i <= img_idx:
                                all_images[i] = None

                        import gc
                        gc.collect()

                except Exception as e:
                    print(f"  ⚠️  Error: failed to process frame {frame_idx}: {e}")
                    # Even if it fails, write the black frame to ensure the number of frames is consistent
                    try:
                        buffer = io.BytesIO()
                        black_frame.save(buffer, format='JPEG', quality=95)
                        process.stdin.write(buffer.getvalue())
                        frames_written += 1
                        del buffer
                    except:
                        pass

            print(f"    Successfully written {frames_written}/{expected_frames} frames")

            # Clean up
            del all_images

            # Close input and wait for completion
            try:
                # flush and close stdin
                if process.stdin and not process.stdin.closed:
                    process.stdin.flush()
                    process.stdin.close()

                # Use wait() instead of communicate(), because we have written the data
                # communicate() will try to flush the closed stdin, causing an error
                returncode = process.wait(timeout=60)

                # Read stderr to get error information
                if process.stderr:
                    stderr = process.stderr.read()
                else:
                    stderr = b''

            except subprocess.TimeoutExpired:
                print(f"  ⚠️  ffmpeg processing timeout")
                try:
                    process.kill()
                    process.wait()
                except:
                    pass
                return False
            except Exception as e:
                print(f"  ⚠️  Processing error: {e}")
                try:
                    process.kill()
                    process.wait()
                except:
                    pass
                return False

            if returncode == 0:
                return True
            else:
                error_msg = stderr.decode('utf-8', errors='ignore') if stderr else 'Unknown error'
                print(f"  ⚠️  ffmpeg error (return code: {returncode}): {error_msg[:200]}")
                return False

        except FileNotFoundError:
            print("  ⚠️  未安装 ffmpeg，请执行：sudo apt-get install ffmpeg")
            return False
        except Exception as e:
            print(f"  ⚠️  视频生成失败：{e}")
            import traceback
            traceback.print_exc()
            return False

    def _add_vr_commands_to_frame(
        self,
        frame_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        frame_index: int,
    ) -> None:
        """Write causally aligned VR commands into canonical action fields."""
        for sensor_name, action_name in VR_POSE_ACTION_STREAMS.items():
            values = sensor_data.get(sensor_name)
            if not values or frame_index >= len(values) or values[frame_index] is None:
                raise RuntimeError(
                    f"第 {frame_index} 帧缺少因果对齐后的 {sensor_name}；"
                    "请先启动 VR 指令流，再开始采集"
                )
            pose = values[frame_index]
            try:
                position = pose['position']
                orientation = pose['orientation']
                pose_values = np.asarray(
                    [
                        position['x'], position['y'], position['z'],
                        orientation['x'], orientation['y'],
                        orientation['z'], orientation['w'],
                    ],
                    dtype=np.float64,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"第 {frame_index} 帧的 {sensor_name} 格式无效"
                ) from error
            if not np.all(np.isfinite(pose_values)):
                raise RuntimeError(
                    f"第 {frame_index} 帧的 {sensor_name} 包含非有限数值"
                )
            quaternion_norm = float(np.linalg.norm(pose_values[3:7]))
            if quaternion_norm < 1.0e-8:
                raise RuntimeError(
                    f"第 {frame_index} 帧的 {sensor_name} 四元数无效"
                )
            frame_data['action'][action_name] = {
                'position': {
                    'x': float(pose_values[0]),
                    'y': float(pose_values[1]),
                    'z': float(pose_values[2]),
                },
                'orientation': {
                    'x': float(pose_values[3] / quaternion_norm),
                    'y': float(pose_values[4] / quaternion_norm),
                    'z': float(pose_values[5] / quaternion_norm),
                    'w': float(pose_values[6] / quaternion_norm),
                },
            }

        gripper_action_streams = (
            VR_REVO2_ACTION_STREAMS
            if self.collection_config.enable_revo2_hands
            else VR_GRIPPER_ACTION_STREAMS
        )
        for sensor_name, action_name in gripper_action_streams.items():
            values = sensor_data.get(sensor_name)
            if not values or frame_index >= len(values) or values[frame_index] is None:
                raise RuntimeError(
                    f"第 {frame_index} 帧缺少因果对齐后的 {sensor_name}；"
                    "请先启动 VR 指令流，再开始采集"
                )
            command = values[frame_index]
            if self.collection_config.enable_revo2_hands:
                try:
                    actuator_names = tuple(command['actuator_names'])
                    positions = np.asarray(
                        command['positions'],
                        dtype=np.float64,
                    )
                    duration_ms = int(command['duration_ms'])
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"第 {frame_index} 帧的 {sensor_name} 格式无效"
                    ) from error
                if (
                    actuator_names != REVO2_ACTUATOR_NAMES
                    or positions.shape != (len(REVO2_ACTUATOR_NAMES),)
                    or not np.all(np.isfinite(positions))
                    or np.any(positions < 0)
                    or np.any(positions > 1000)
                    or duration_ms <= 0
                ):
                    raise RuntimeError(
                        f"第 {frame_index} 帧的 {sensor_name} 六轴命令无效"
                    )
                frame_data['action'][action_name] = {
                    'actuator_names': list(REVO2_ACTUATOR_NAMES),
                    'positions': [int(value) for value in positions],
                    'position_unit': 'normalized_0_1000',
                    'duration_ms': duration_ms,
                }
                continue
            try:
                position = float(command['position'])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"第 {frame_index} 帧的 {sensor_name} 格式无效"
                ) from error
            if not np.isfinite(position):
                raise RuntimeError(
                    f"第 {frame_index} 帧的 {sensor_name} 包含非有限数值"
                )
            frame_data['action'][action_name] = {'position': position}

    def _raw_topic_descriptor(self, topic_name, is_camera):
        """Describe one unaligned source stream in the raw-topic manifest."""
        if is_camera:
            if "depth" in topic_name:
                return {
                    "kind": "depth_camera",
                    "payload_encoding": (
                        "sdk_depth_array_or_source_bytes"
                    ),
                    "record_schema": (
                        "(timestamp_seconds, depth_payload)"
                    ),
                    "timestamp_source": (
                        "message_header_or_collector_receipt_wall_clock"
                    ),
                }
            descriptor = {
                "kind": "camera",
                "payload_encoding": "jpeg_bytes",
                "record_schema": "(timestamp_seconds, jpeg_bytes)",
                "timestamp_source": (
                    "message_header_or_collector_receipt_wall_clock"
                ),
            }
            camera_status = self.camera_profile_status().get(topic_name)
            if camera_status is not None:
                descriptor['capture_profile'] = camera_status['expected']
                descriptor['observed_profile'] = camera_status['observed']
                descriptor['capture_profile_valid'] = bool(
                    camera_status['valid']
                )
            return descriptor
        if topic_name in VR_POSE_ACTION_STREAMS:
            return {
                "kind": "action",
                "payload_encoding": "primitive_dict",
                "record_schema": "(timestamp_seconds, pose_command_dict)",
                "timestamp_source": (
                    "message_header_or_bridge_receipt_wall_clock"
                ),
                "ros_topic": VR_ROS_COMMAND_TOPICS.get(topic_name),
            }
        if topic_name in VR_GRIPPER_ACTION_STREAMS:
            return {
                "kind": "action",
                "payload_encoding": "primitive_dict",
                "record_schema": (
                    "(timestamp_seconds, gripper_command_dict)"
                ),
                "timestamp_source": (
                    "bridge_receipt_wall_clock_no_message_header"
                ),
                "ros_topic": VR_ROS_COMMAND_TOPICS.get(topic_name),
            }
        if topic_name in VR_REVO2_ACTION_STREAMS:
            return {
                "kind": "action",
                "payload_encoding": "primitive_dict",
                "payload_schema": "revo2_joint_command_v1",
                "record_schema": (
                    "(timestamp_seconds, revo2_joint_command_dict)"
                ),
                "actuator_names": list(REVO2_ACTUATOR_NAMES),
                "position_unit": "normalized_0_1000",
                "timestamp_source": "revo2_bridge_wall_clock",
                "ros_topic": VR_ROS_COMMAND_TOPICS.get(topic_name),
            }
        if topic_name in {
            'left_revo2_joint_states',
            'right_revo2_joint_states',
        }:
            return {
                "kind": "state",
                "payload_encoding": "primitive_dict",
                "payload_schema": "revo2_joint_state_v1",
                "record_schema": (
                    "(timestamp_seconds, revo2_joint_state_dict)"
                ),
                "actuator_names": list(REVO2_ACTUATOR_NAMES),
                "position_unit": "normalized_0_1000",
                "timestamp_source": "revo2_bridge_wall_clock",
                "ros_topic": REVO2_ROS_TOPICS.get(topic_name),
            }
        if topic_name in {
            "left_gripper_position",
            "right_gripper_position",
        }:
            return {
                "kind": "state",
                "payload_encoding": "primitive_dict",
                "record_schema": (
                    "(timestamp_seconds, {'position': float})"
                ),
                "timestamp_source": (
                    "message_header_or_collector_receipt_wall_clock"
                ),
            }
        if topic_name.endswith("_end_pose"):
            return {
                "kind": "state",
                "payload_encoding": "primitive_dict",
                "record_schema": (
                    "(timestamp_seconds, pose_state_dict)"
                ),
                "timestamp_source": (
                    "message_header_or_collector_receipt_wall_clock"
                ),
            }
        if topic_name in self.slave_joint_names:
            record_schema = (
                "(timestamp_seconds, positions, velocities, efforts)"
            )
        else:
            record_schema = "(timestamp_seconds, payload)"
        return {
            "kind": "state",
            "payload_encoding": "python_pickle_object",
            "record_schema": record_schema,
            "timestamp_source": (
                "message_header_or_collector_receipt_wall_clock"
            ),
        }

    def _publish_raw_topic_file(
        self,
        source_path: Path,
        destination_path: Path,
    ):
        """Publish a closed topic file, with a safe cross-device fallback."""
        try:
            os.replace(source_path, destination_path)
            return
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise

        copying_path = destination_path.with_name(
            f".{destination_path.name}.{os.getpid()}."
            f"{time.time_ns()}.copying"
        )
        try:
            shutil.copyfile(source_path, copying_path)
            self._fsync_file(copying_path)
            os.replace(copying_path, destination_path)
            source_path.unlink()
        finally:
            if copying_path.exists():
                try:
                    copying_path.unlink()
                except OSError:
                    pass

    def _save_raw_topic_episode(
        self,
        task: str,
        *,
        effective_start_timestamp: Optional[float] = None,
        effective_end_timestamp: Optional[float] = None,
        effective_context_topics: Optional[List[str]] = None,
        capture_warmup_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Atomically save independent topic streams without any alignment."""
        episode_id = self.episode_count
        final_episode_dir = (
            self.output_dir / f"episode_{episode_id:04d}"
        )
        episode_dir = self._prepare_episode_staging_dir()
        raw_topics_dir = episode_dir / "raw_topics"
        raw_topics_dir.mkdir(parents=True, exist_ok=True)

        with self._raw_topic_stats_lock:
            topic_stats = {
                name: dict(values)
                for name, values in self._raw_topic_stats.items()
            }

        derived_topic_names = set(self.slave_action_names)
        expected_camera_topics = list(self.camera_names)
        expected_sensor_topics = [
            name
            for name in self.sensor_file_locks
            if name not in derived_topic_names
        ]
        expected_topics = (
            expected_camera_topics + expected_sensor_topics
        )
        if not expected_topics:
            raise RuntimeError("raw topic 模式没有启用任何数据流")
        self._assert_camera_profiles_ready()

        source_paths = {}
        for topic_name in expected_camera_topics:
            source_paths[topic_name] = Path(
                self.image_temp_paths[topic_name]
            )
        for topic_name in expected_sensor_topics:
            source_paths[topic_name] = Path(
                self.sensor_temp_paths[topic_name]
            )

        missing_topics = []
        for topic_name in expected_topics:
            stats = topic_stats.get(topic_name, {})
            source_path = source_paths.get(topic_name)
            if (
                int(stats.get("record_count", 0)) <= 0
                or source_path is None
                or not source_path.is_file()
                or source_path.stat().st_size <= 0
            ):
                missing_topics.append(topic_name)
        if missing_topics:
            raise RuntimeError(
                "以下已启用 topic 没有采集到数据，raw Episode "
                f"不会提交：{missing_topics}"
            )

        invalid_timestamp_topics = [
            topic_name
            for topic_name in expected_topics
            if int(
                topic_stats[topic_name].get(
                    "nonfinite_timestamps",
                    0,
                )
            ) > 0
        ]
        if invalid_timestamp_topics:
            raise RuntimeError(
                "以下 topic 包含非有限时间戳，raw Episode 不会提交："
                f"{invalid_timestamp_topics}"
            )

        has_effective_start = effective_start_timestamp is not None
        has_effective_end = effective_end_timestamp is not None
        if has_effective_start != has_effective_end:
            raise RuntimeError(
                "effective_start_timestamp 与 "
                "effective_end_timestamp 必须同时提供"
            )

        effective_metadata: Dict[str, Any] = {}
        if not has_effective_start:
            if effective_context_topics not in (None, []):
                raise RuntimeError(
                    "未提供 effective 时间区间时不能设置 "
                    "effective_context_topics"
                )
            if capture_warmup_seconds is not None:
                raise RuntimeError(
                    "未提供 effective 时间区间时不能设置 "
                    "capture_warmup_seconds"
                )
        else:
            if isinstance(effective_start_timestamp, bool) or isinstance(
                effective_end_timestamp,
                bool,
            ):
                raise RuntimeError("effective 时间戳必须是有限数值")
            try:
                normalized_effective_start = float(
                    effective_start_timestamp
                )
                normalized_effective_end = float(
                    effective_end_timestamp
                )
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "effective 时间戳必须是有限数值"
                ) from error
            if (
                not math.isfinite(normalized_effective_start)
                or not math.isfinite(normalized_effective_end)
            ):
                raise RuntimeError("effective 时间戳必须是有限数值")
            if normalized_effective_end <= normalized_effective_start:
                raise RuntimeError(
                    "effective_end_timestamp 必须晚于 "
                    "effective_start_timestamp"
                )

            if capture_warmup_seconds is None:
                normalized_capture_warmup = None
            else:
                if isinstance(capture_warmup_seconds, bool):
                    raise RuntimeError(
                        "capture_warmup_seconds 必须是非负有限数值"
                    )
                try:
                    normalized_capture_warmup = float(
                        capture_warmup_seconds
                    )
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        "capture_warmup_seconds 必须是非负有限数值"
                    ) from error
                if (
                    not math.isfinite(normalized_capture_warmup)
                    or normalized_capture_warmup < 0
                ):
                    raise RuntimeError(
                        "capture_warmup_seconds 必须是非负有限数值"
                    )

            if effective_context_topics is None:
                normalized_context_topics: List[str] = []
            elif isinstance(effective_context_topics, list):
                normalized_context_topics = list(
                    effective_context_topics
                )
            else:
                raise RuntimeError(
                    "effective_context_topics 必须是字符串列表"
                )
            if any(
                not isinstance(topic_name, str) or not topic_name
                for topic_name in normalized_context_topics
            ):
                raise RuntimeError(
                    "effective_context_topics 必须是非空字符串列表"
                )
            if len(set(normalized_context_topics)) != len(
                normalized_context_topics
            ):
                raise RuntimeError(
                    "effective_context_topics 不能包含重复 topic"
                )
            unknown_context_topics = sorted(
                set(normalized_context_topics) - set(expected_topics)
            )
            if unknown_context_topics:
                raise RuntimeError(
                    "effective_context_topics 包含未启用 topic："
                    f"{unknown_context_topics}"
                )

            context_topic_set = set(normalized_context_topics)
            normalized_context_topics = [
                topic_name
                for topic_name in expected_topics
                if topic_name in context_topic_set
            ]
            coverage_errors = []
            for topic_name in expected_topics:
                stats = topic_stats[topic_name]
                try:
                    first_timestamp = float(
                        stats.get("first_timestamp")
                    )
                    last_timestamp = float(
                        stats.get("last_timestamp")
                    )
                except (TypeError, ValueError):
                    coverage_errors.append(
                        f"{topic_name}: first/last 时间戳无效"
                    )
                    continue
                if (
                    not math.isfinite(first_timestamp)
                    or not math.isfinite(last_timestamp)
                    or last_timestamp < first_timestamp
                ):
                    coverage_errors.append(
                        f"{topic_name}: first/last 时间戳无效"
                    )
                    continue
                if first_timestamp > normalized_effective_start:
                    coverage_errors.append(
                        f"{topic_name}: first_timestamp "
                        f"{first_timestamp} 晚于 effective start "
                        f"{normalized_effective_start}"
                    )
                if (
                    topic_name not in context_topic_set
                    and last_timestamp < normalized_effective_end
                ):
                    coverage_errors.append(
                        f"{topic_name}: last_timestamp "
                        f"{last_timestamp} 早于 effective end "
                        f"{normalized_effective_end}"
                    )
            if coverage_errors:
                raise RuntimeError(
                    "raw topic 未覆盖 effective 时间区间，"
                    "Episode 不会提交："
                    + "; ".join(coverage_errors)
                )

            warmup_reference_topic = (
                "head_rgb_stream"
                if "head_rgb_stream" in expected_topics
                else expected_topics[0]
            )
            reference_first_timestamp = float(
                topic_stats[warmup_reference_topic].get(
                    "first_timestamp"
                )
            )
            computed_capture_warmup = max(
                0.0,
                normalized_effective_start
                - reference_first_timestamp,
            )
            if normalized_capture_warmup is None:
                normalized_capture_warmup = computed_capture_warmup
            elif not math.isclose(
                normalized_capture_warmup,
                computed_capture_warmup,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "capture_warmup_seconds 与参考流首时间戳不一致"
                )

            effective_metadata = {
                "effective_start_timestamp": (
                    normalized_effective_start
                ),
                "effective_end_timestamp": normalized_effective_end,
                "effective_duration": (
                    normalized_effective_end
                    - normalized_effective_start
                ),
                "effective_interval_policy": (
                    "timestamp_window_with_context_carry_forward"
                ),
                "capture_warmup_seconds": normalized_capture_warmup,
                "effective_context_topics": normalized_context_topics,
            }

        # Empty files created only for legacy derived actions are not source
        # topics and must never appear in a raw-topic Episode.
        for topic_name in derived_topic_names:
            temp_path_value = self.sensor_temp_paths.get(topic_name)
            if temp_path_value:
                temp_path = Path(temp_path_value)
                if temp_path.exists():
                    temp_path.unlink()

        raw_topics = {}
        for topic_name in expected_topics:
            source_path = source_paths[topic_name]
            destination_path = (
                raw_topics_dir / f"{topic_name}.pklseq"
            )
            if destination_path.exists():
                raise FileExistsError(
                    f"raw topic 目标文件已存在：{destination_path}"
                )
            self._publish_raw_topic_file(
                source_path,
                destination_path,
            )
            descriptor = self._raw_topic_descriptor(
                topic_name,
                topic_name in expected_camera_topics,
            )
            stats = topic_stats[topic_name]
            descriptor.update({
                "path": str(
                    destination_path.relative_to(episode_dir)
                ),
                "record_count": int(stats["record_count"]),
                "first_timestamp": stats.get("first_timestamp"),
                "last_timestamp": stats.get("last_timestamp"),
                "timestamp_regressions": int(
                    stats.get("timestamp_regressions", 0)
                ),
                "duplicate_timestamps": int(
                    stats.get("duplicate_timestamps", 0)
                ),
                "nonfinite_timestamps": int(
                    stats.get("nonfinite_timestamps", 0)
                ),
                "timestamp_source_counts": {
                    str(source): int(count)
                    for source, count in stats.get(
                        "timestamp_source_counts",
                        {},
                    ).items()
                },
                "timestamp_unit": "seconds",
                "clock_domain": (
                    "source-defined; receipt fallbacks use Unix wall time"
                ),
                "size_bytes": destination_path.stat().st_size,
            })
            raw_topics[topic_name] = descriptor

        unexpected_pending = [
            path
            for path in raw_topics_dir.iterdir()
            if path.name.endswith(".pending.pklseq")
        ]
        if unexpected_pending:
            raise RuntimeError(
                "raw topic 临时文件未全部封存："
                f"{[path.name for path in unexpected_pending]}"
            )

        capture_stopped_at = (
            float(self.current_episode_stop_time)
            if self.current_episode_stop_time is not None
            else time.time()
        )
        capture_started_at = (
            float(self.current_episode_start_time)
            if self.current_episode_start_time is not None
            else capture_stopped_at
        )
        capture_duration = max(
            0.0,
            capture_stopped_at - capture_started_at,
        )
        recording_started_at = float(
            effective_metadata.get(
                "effective_start_timestamp",
                capture_started_at,
            )
        )
        recording_stopped_at = float(
            effective_metadata.get(
                "effective_end_timestamp",
                capture_stopped_at,
            )
        )
        episode_duration = float(
            effective_metadata.get(
                "effective_duration",
                capture_duration,
            )
        )
        reference_topic = (
            "head_rgb_stream"
            if "head_rgb_stream" in raw_topics
            else expected_topics[0]
        )
        reference_sample_count = int(
            raw_topics[reference_topic]["record_count"]
        )
        total_records = sum(
            int(info["record_count"])
            for info in raw_topics.values()
        )
        robot_model = self.robot.get_robot_model()
        episode_json = {
            "schema_version": 2,
            "episode_id": episode_id,
            "model": robot_model,
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "capture_started_at": datetime.fromtimestamp(
                capture_started_at
            ).isoformat(),
            "capture_stopped_at": datetime.fromtimestamp(
                capture_stopped_at
            ).isoformat(),
            "capture_duration": capture_duration,
            "recording_started_at": datetime.fromtimestamp(
                recording_started_at
            ).isoformat(),
            "recording_stopped_at": datetime.fromtimestamp(
                recording_stopped_at
            ).isoformat(),
            "duration": episode_duration,
            "storage_format": "raw_topics_pickle_v1",
            "timestamp_alignment": "none",
            "timestamp_policy": "independent_per_topic",
            "reference_topic": reference_topic,
            "reference_sample_count": reference_sample_count,
            "num_topics": len(raw_topics),
            "total_records": total_records,
            "raw_topics": raw_topics,
            "security_note": (
                "Only unpickle trusted local files. Some SDK state "
                "payloads require a compatible x2robot Python environment."
            ),
        }
        episode_json.update(effective_metadata)

        episode_json_path = episode_dir / "episode.json"
        with open(episode_json_path, 'w', encoding='utf-8') as f:
            json.dump(
                episode_json,
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.flush()
            os.fsync(f.fileno())

        # Re-read the manifest and validate the complete staged transaction.
        with open(episode_json_path, 'r', encoding='utf-8') as f:
            staged_json = json.load(f)
        if (
            staged_json.get("storage_format")
            != "raw_topics_pickle_v1"
            or set(staged_json.get("raw_topics", {}))
            != set(expected_topics)
            or any(
                staged_json.get(field_name) != field_value
                for field_name, field_value
                in effective_metadata.items()
            )
        ):
            raise RuntimeError("raw topic Episode manifest 校验失败")
        for topic_name, topic_info in raw_topics.items():
            topic_path = episode_dir / topic_info["path"]
            if (
                not topic_path.is_file()
                or topic_path.stat().st_size
                != int(topic_info["size_bytes"])
                or int(topic_info["record_count"]) <= 0
            ):
                raise RuntimeError(
                    f"raw topic 文件校验失败：{topic_name}"
                )

        self._fsync_directory(raw_topics_dir)
        self._fsync_directory(episode_dir)
        metadata_entry = {
            "episode_id": episode_id,
            "model": robot_model,
            "task": task,
            "timestamp": episode_json["timestamp"],
            "duration": episode_json["duration"],
            "capture_duration": episode_json["capture_duration"],
            "storage_format": "raw_topics_pickle_v1",
            "reference_topic": reference_topic,
            "reference_sample_count": reference_sample_count,
            "num_topics": len(raw_topics),
            "total_records": total_records,
            "path": final_episode_dir.name,
        }
        metadata_entry.update(effective_metadata)

        try:
            if final_episode_dir.exists():
                raise FileExistsError(
                    "Episode 提交前发现目标目录已经存在："
                    f"{final_episode_dir}"
                )
            os.replace(episode_dir, final_episode_dir)
            self._fsync_directory(self.output_dir)
            episode_json_path = (
                final_episode_dir / "episode.json"
            )

            self.dataset_metadata['episodes'].append(metadata_entry)
            self._save_metadata()
        except BaseException:
            if (
                self.dataset_metadata.get('episodes')
                and self.dataset_metadata['episodes'][-1]
                is metadata_entry
            ):
                self.dataset_metadata['episodes'].pop()

            metadata_rollback_ok = True
            try:
                self._save_metadata()
            except BaseException as metadata_rollback_error:
                metadata_rollback_ok = False
                print(
                    "  ⚠️  metadata 回滚失败；将保留完整 raw "
                    "Episode，程序下次启动时会自动恢复登记："
                    f"{metadata_rollback_error}"
                )

            try:
                if (
                    metadata_rollback_ok
                    and final_episode_dir.exists()
                    and not episode_dir.exists()
                ):
                    os.replace(final_episode_dir, episode_dir)
                    self._fsync_directory(self.output_dir)
                    self._active_episode_staging_dir = episode_dir
            except BaseException as rollback_error:
                print(
                    "  ⚠️  raw Episode 目录回滚失败，但数据完整；"
                    "程序下次启动时会自动恢复登记："
                    f"{rollback_error}"
                )
            raise

        self._active_episode_staging_dir = None
        print(f"✓ 第 {episode_id} 条 raw topic 数据已保存")
        print(f"  - topic 数量：{len(raw_topics)}")
        print(f"  - 总记录数：{total_records}")
        print("  - 视频转码：未执行")
        print(f"  - JSON 文件：{episode_json_path}")

        episode_info = {
            "episode_id": episode_id,
            "episode_dir": str(final_episode_dir),
            "episode_json": str(episode_json_path),
            "storage_format": "raw_topics_pickle_v1",
            "reference_topic": reference_topic,
            "reference_sample_count": reference_sample_count,
            "num_topics": len(raw_topics),
            "total_records": total_records,
            "topic_counts": {
                name: int(info["record_count"])
                for name, info in raw_topics.items()
            },
            "duration": episode_json["duration"],
            "capture_duration": episode_json["capture_duration"],
        }
        episode_info.update(effective_metadata)
        return episode_info

    def _save_episode(self, episode_data: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Save episode data to JSON and image or video file"""
        episode_id = self.episode_count
        final_episode_dir = (
            self.output_dir / f"episode_{episode_id:04d}"
        )
        episode_dir = self._prepare_episode_staging_dir()
        staging_token = f"{os.getpid()}_{time.time_ns()}"

        print(
            f"正在将第 {episode_id} 条数据保存到 "
            f"{final_episode_dir}..."
        )

        timestamps = episode_data['timestamps']
        sensor_data = episode_data['sensor_data']
        action_data = episode_data.get('action_data', {})
        images = episode_data['images']

        # Use the joint state and action names in the member variables
        joint_state_names = self.slave_joint_names if self.slave_joint_names else []
        action_names = self.slave_action_names if self.slave_action_names else []

        # Collect the joint state data for each part
        joint_states_by_part = {}
        for joint_state_name in joint_state_names:
            if joint_state_name in sensor_data:
                joint_states_by_part[joint_state_name] = sensor_data[joint_state_name]

        # Collect the action data for each part (for future state as action)
        actions_by_part = {}
        for action_name in action_names:
            if action_name in action_data:
                actions_by_part[action_name] = action_data[action_name]

        # Build the joint name dictionary (separated by parts, using the real joint names)
        joint_names = {}
        for joint_state_name in joint_state_names:
            if joint_state_name in joint_states_by_part and len(joint_states_by_part[joint_state_name]) > 0:
                part_name = joint_state_name.replace('_joint_states', '')

                # Use the saved real joint names first
                if hasattr(self, '_joint_names_by_part') and joint_state_name in self._joint_names_by_part:
                    joint_names[part_name] = self._joint_names_by_part[joint_state_name]
                else:
                    # If there are no saved joint names, infer the number of joints from the first frame data and generate default names
                    first_state = joint_states_by_part[joint_state_name][0]
                    if isinstance(first_state, tuple) and len(first_state) > 0:
                        num_joints = len(first_state[0])
                    else:
                        num_joints = len(first_state) if hasattr(first_state, '__len__') else 0

                    # Generate default joint names for each part
                    part_joint_names = []
                    for i in range(num_joints):
                        part_joint_names.append(f"{part_name}_joint{i+1}")
                    joint_names[part_name] = part_joint_names

        if not joint_names:
            joint_names = {}

        num_frames = len(timestamps)

        # Get the robot model and map it to the product name
        robot_model = self.robot.get_robot_model()
        # Build the episode JSON data
        episode_json = {
            "episode_id": episode_id,
            "model": robot_model,
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "duration": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            "num_frames": num_frames,
            "joint_names": joint_names if joint_names else {},  # Dictionary separated by parts
            "storage_format": "video" if self.use_video_storage else "images",
            "frames": []
        }

        # If using video storage, use ffmpeg to create video
        video_files = {}

        if self.use_video_storage:
            # Get the index mapping and camera name mapping from episode_data
            image_index_mapping = episode_data.get('image_index_mapping', {})
            camera_name_mapping = episode_data.get('camera_name_mapping', {})

            # Reverse mapping: friendly name -> internal name
            reverse_mapping = {v: k for k, v in camera_name_mapping.items()}

            expected_video_names = {
                camera_name_mapping.get(name, name)
                for name in self.camera_names
            }
            if set(image_index_mapping) != expected_video_names:
                raise RuntimeError(
                    "待编码的视频流不完整："
                    f"预期 {sorted(expected_video_names)}，"
                    f"实际 {sorted(image_index_mapping)}"
                )

            print("  正在生成视频文件...")
            pending_video_paths = {}
            try:
                for friendly_name, indices in image_index_mapping.items():
                    internal_name = reverse_mapping.get(
                        friendly_name,
                        friendly_name,
                    )
                    if internal_name not in self.image_temp_paths:
                        raise RuntimeError(
                            f"找不到 {friendly_name} 的临时文件"
                        )

                    temp_path = self.image_temp_paths[internal_name]
                    if not os.path.exists(temp_path):
                        raise RuntimeError(
                            f"临时文件不存在：{temp_path}"
                        )

                    final_video_path = (
                        episode_dir / f"{friendly_name}.mp4"
                    )
                    pending_video_path = episode_dir / (
                        f".{friendly_name}.{staging_token}.pending.mp4"
                    )
                    print(
                        "  正在使用 ffmpeg 生成视频："
                        f"{friendly_name}.mp4..."
                    )

                    success = self._create_video_with_ffmpeg(
                        temp_path,
                        indices,
                        str(pending_video_path),
                        self.target_hz,
                        num_frames,
                    )
                    if not success:
                        raise RuntimeError(
                            f"视频生成失败：{friendly_name}"
                        )
                    pending_video_paths[friendly_name] = (
                        pending_video_path,
                        final_video_path,
                    )
                    print(
                        f"  ✓ 视频编码完成：{friendly_name}.mp4"
                        f"（{num_frames} 帧）"
                    )
                    self._release_unused_memory()

                # Commit only after every camera has encoded and passed
                # ffprobe, so an incomplete set never replaces good videos.
                for friendly_name, (
                    pending_video_path,
                    final_video_path,
                ) in pending_video_paths.items():
                    os.replace(pending_video_path, final_video_path)
                    video_files[friendly_name] = (
                        f"{friendly_name}.mp4"
                    )
                    print(
                        f"  ✓ 视频提交成功：{friendly_name}.mp4"
                    )
            except Exception:
                for pending_video_path, _ in pending_video_paths.values():
                    if pending_video_path.exists():
                        try:
                            pending_video_path.unlink()
                        except Exception:
                            pass
                raise

            episode_json["video_files"] = video_files

        # Save each frame
        import gc
        for i in range(num_frames):
            frame_data = {
                "frame_id": i,
                "timestamp": timestamps[i],
                "images": {}
            }

            # Add the joint state data for each part
            if "observation" not in frame_data:
                frame_data["observation"] = {}

            for joint_state_name in joint_state_names:
                if joint_state_name in joint_states_by_part and i < len(joint_states_by_part[joint_state_name]):
                    # Extract the state data (positions, velocities, efforts)
                    state_data = joint_states_by_part[joint_state_name][i]
                    if isinstance(state_data, tuple):
                        positions, velocities, efforts = state_data
                    else:
                        positions = state_data
                        velocities = None
                        efforts = None

                    # Convert to list format
                    positions_list = positions.tolist() if isinstance(positions, np.ndarray) else list(positions)

                    # Save the joint state for each part
                    frame_data["observation"][joint_state_name] = {
                        "positions": positions_list
                    }

                    # Add optional velocity and effort
                    if velocities is not None:
                        frame_data["observation"][joint_state_name]["velocities"] = velocities.tolist() if isinstance(velocities, np.ndarray) else list(velocities)
                    if efforts is not None:
                        frame_data["observation"][joint_state_name]["efforts"] = efforts.tolist() if isinstance(efforts, np.ndarray) else list(efforts)

            # Write actions. VR mode uses recorded commands; legacy modes keep
            # the historical next-state proxy below for backward compatibility.
            if "action" not in frame_data:
                frame_data["action"] = {}
            if self.collection_config.enable_vr_action_commands:
                self._add_vr_commands_to_frame(
                    frame_data,
                    sensor_data,
                    i,
                )

            proxy_joint_state_names = (
                []
                if self.collection_config.enable_vr_action_commands
                else joint_state_names
            )
            for joint_state_name in proxy_joint_state_names:
                part_name = joint_state_name.replace('_joint_states', '')
                action_name = f"{part_name}_actions"

                # Use the state of the next frame as action
                if joint_state_name in joint_states_by_part:
                    if i + 1 < len(joint_states_by_part[joint_state_name]):
                        # There is a next frame, use the state of the next frame as action
                        next_state_data = joint_states_by_part[joint_state_name][i + 1]
                        if isinstance(next_state_data, tuple):
                            next_positions, _, _ = next_state_data
                        else:
                            next_positions = next_state_data

                        next_positions_list = next_positions.tolist() if isinstance(next_positions, np.ndarray) else list(next_positions)
                        frame_data["action"][action_name] = {
                            "positions": next_positions_list
                        }
                    elif i < len(joint_states_by_part[joint_state_name]):
                        # The last frame, use the state of the current frame as action
                        current_state_data = joint_states_by_part[joint_state_name][i]
                        if isinstance(current_state_data, tuple):
                            current_positions, _, _ = current_state_data
                        else:
                            current_positions = current_state_data

                        current_positions_list = current_positions.tolist() if isinstance(current_positions, np.ndarray) else list(current_positions)
                        frame_data["action"][action_name] = {
                            "positions": current_positions_list
                        }

            # Add other sensor data
            for sensor_name, sensor_data_list in sensor_data.items():
                # Skip the joint state data that has already been processed
                if sensor_name in joint_state_names:
                    continue
                # VR command streams are actions, never observations.
                if sensor_name in ALL_VR_ACTION_SENSOR_NAMES:
                    continue

                # Skip the empty data list
                if not sensor_data_list:
                    continue

                if i < len(sensor_data_list):
                    sensor_value = sensor_data_list[i]

                    # If there is no observation dictionary, create one
                    if "observation" not in frame_data:
                        frame_data["observation"] = {}

                    # Process the data based on the sensor type
                    if sensor_name.endswith('_end_pose'):
                        # End pose data
                        if isinstance(sensor_value, dict) and 'data' in sensor_value:
                            pose_data = sensor_value['data']
                            frame_data["observation"][sensor_name] = {
                                'position': {
                                    'x': pose_data.position.x if hasattr(pose_data, 'position') else 0,
                                    'y': pose_data.position.y if hasattr(pose_data, 'position') else 0,
                                    'z': pose_data.position.z if hasattr(pose_data, 'position') else 0,
                                },
                                'orientation': {
                                    'x': pose_data.orientation.x if hasattr(pose_data, 'orientation') else 0,
                                    'y': pose_data.orientation.y if hasattr(pose_data, 'orientation') else 0,
                                    'z': pose_data.orientation.z if hasattr(pose_data, 'orientation') else 0,
                                    'w': pose_data.orientation.w if hasattr(pose_data, 'orientation') else 1,
                                }
                            }
                        else:
                            # If the data format is different, save it directly
                            frame_data["observation"][sensor_name] = sensor_value

                        # Add the action of end_pose (the target pose = next frame's pose).
                        action_name = sensor_name.replace('_end_pose', '_end_pose_action')
                        if action_name in frame_data["action"]:
                            # A real VR command was already written for this arm.
                            pass
                        elif i + 1 < len(sensor_data_list):
                            next_sensor_value = sensor_data_list[i + 1]
                            if isinstance(next_sensor_value, dict) and 'data' in next_sensor_value:
                                # Legacy protobuf-wrapped format.
                                next_pose_data = next_sensor_value['data']
                                frame_data["action"][action_name] = {
                                    'position': {
                                        'x': next_pose_data.position.x if hasattr(next_pose_data, 'position') else 0,
                                        'y': next_pose_data.position.y if hasattr(next_pose_data, 'position') else 0,
                                        'z': next_pose_data.position.z if hasattr(next_pose_data, 'position') else 0,
                                    },
                                    'orientation': {
                                        'x': next_pose_data.orientation.x if hasattr(next_pose_data, 'orientation') else 0,
                                        'y': next_pose_data.orientation.y if hasattr(next_pose_data, 'orientation') else 0,
                                        'z': next_pose_data.orientation.z if hasattr(next_pose_data, 'orientation') else 0,
                                        'w': next_pose_data.orientation.w if hasattr(next_pose_data, 'orientation') else 1,
                                    }
                                }
                            elif isinstance(next_sensor_value, dict) and 'position' in next_sensor_value:
                                # Normal aligned format: copy the next frame's pose so the
                                # action is the target pose (independent dict, no aliasing).
                                np_pos = next_sensor_value['position']
                                np_ori = next_sensor_value['orientation']
                                frame_data["action"][action_name] = {
                                    'position': {'x': np_pos['x'], 'y': np_pos['y'], 'z': np_pos['z']},
                                    'orientation': {
                                        'x': np_ori['x'], 'y': np_ori['y'],
                                        'z': np_ori['z'], 'w': np_ori['w'],
                                    }
                                }
                            else:
                                # Unknown format: fall back to current frame.
                                frame_data["action"][action_name] = frame_data["observation"][sensor_name]
                        else:
                            # The last frame, use the current frame as action.
                            frame_data["action"][action_name] = frame_data["observation"][sensor_name]

                    elif sensor_name == 'odometry':
                        # Odometry data
                        if isinstance(sensor_value, dict) and 'data' in sensor_value:
                            odom_data = sensor_value['data']
                            frame_data["observation"][sensor_name] = {
                                'pose': {
                                    'position': {
                                        'x': odom_data.pose.pose.position.x if hasattr(odom_data, 'pose') and hasattr(odom_data.pose, 'pose') else 0,
                                        'y': odom_data.pose.pose.position.y if hasattr(odom_data, 'pose') and hasattr(odom_data.pose, 'pose') else 0,
                                        'z': odom_data.pose.pose.position.z if hasattr(odom_data, 'pose') and hasattr(odom_data.pose, 'pose') else 0,
                                    },
                                    'orientation': {
                                        'x': odom_data.pose.pose.orientation.x if hasattr(odom_data, 'pose') and hasattr(odom_data.pose, 'pose') else 0,
                                        'y': odom_data.pose.pose.orientation.y if hasattr(odom_data, 'pose') and hasattr(odom_data.pose, 'pose') else 0,
                                        'z': odom_data.pose.pose.orientation.z if hasattr(odom_data, 'pose') and hasattr(odom_data.pose, 'pose') else 0,
                                        'w': odom_data.pose.pose.orientation.w if hasattr(odom_data, 'pose') and hasattr(odom_data.pose, 'pose') else 1,
                                    }
                                },
                                'twist': {
                                    'linear': {
                                        'x': odom_data.twist.twist.linear.x if hasattr(odom_data, 'twist') and hasattr(odom_data.twist, 'twist') else 0,
                                        'y': odom_data.twist.twist.linear.y if hasattr(odom_data, 'twist') and hasattr(odom_data.twist, 'twist') else 0,
                                        'z': odom_data.twist.twist.linear.z if hasattr(odom_data, 'twist') and hasattr(odom_data.twist, 'twist') else 0,
                                    },
                                    'angular': {
                                        'x': odom_data.twist.twist.angular.x if hasattr(odom_data, 'twist') and hasattr(odom_data.twist, 'twist') else 0,
                                        'y': odom_data.twist.twist.angular.y if hasattr(odom_data, 'twist') and hasattr(odom_data.twist, 'twist') else 0,
                                        'z': odom_data.twist.twist.angular.z if hasattr(odom_data, 'twist') and hasattr(odom_data.twist, 'twist') else 0,
                                    }
                                }
                            }
                        else:
                            frame_data["observation"][sensor_name] = sensor_value

                        # Add the action of odometry (using the data of the next frame)
                        if i + 1 < len(sensor_data_list):
                            next_sensor_value = sensor_data_list[i + 1]
                            if isinstance(next_sensor_value, dict) and 'data' in next_sensor_value:
                                next_odom_data = next_sensor_value['data']
                                frame_data["action"]["odometry_action"] = {
                                    'pose': {
                                        'position': {
                                            'x': next_odom_data.pose.pose.position.x if hasattr(next_odom_data, 'pose') and hasattr(next_odom_data.pose, 'pose') else 0,
                                            'y': next_odom_data.pose.pose.position.y if hasattr(next_odom_data, 'pose') and hasattr(next_odom_data.pose, 'pose') else 0,
                                            'z': next_odom_data.pose.pose.position.z if hasattr(next_odom_data, 'pose') and hasattr(next_odom_data.pose, 'pose') else 0,
                                        },
                                        'orientation': {
                                            'x': next_odom_data.pose.pose.orientation.x if hasattr(next_odom_data, 'pose') and hasattr(next_odom_data.pose, 'pose') else 0,
                                            'y': next_odom_data.pose.pose.orientation.y if hasattr(next_odom_data, 'pose') and hasattr(next_odom_data.pose, 'pose') else 0,
                                            'z': next_odom_data.pose.pose.orientation.z if hasattr(next_odom_data, 'pose') and hasattr(next_odom_data.pose, 'pose') else 0,
                                            'w': next_odom_data.pose.pose.orientation.w if hasattr(next_odom_data, 'pose') and hasattr(next_odom_data.pose, 'pose') else 1,
                                        }
                                    },
                                    'twist': {
                                        'linear': {
                                            'x': next_odom_data.twist.twist.linear.x if hasattr(next_odom_data, 'twist') and hasattr(next_odom_data.twist, 'twist') else 0,
                                            'y': next_odom_data.twist.twist.linear.y if hasattr(next_odom_data, 'twist') and hasattr(next_odom_data.twist, 'twist') else 0,
                                            'z': next_odom_data.twist.twist.linear.z if hasattr(next_odom_data, 'twist') and hasattr(next_odom_data.twist, 'twist') else 0,
                                        },
                                        'angular': {
                                            'x': next_odom_data.twist.twist.angular.x if hasattr(next_odom_data, 'twist') and hasattr(next_odom_data.twist, 'twist') else 0,
                                            'y': next_odom_data.twist.twist.angular.y if hasattr(next_odom_data, 'twist') and hasattr(next_odom_data.twist, 'twist') else 0,
                                            'z': next_odom_data.twist.twist.angular.z if hasattr(next_odom_data, 'twist') and hasattr(next_odom_data.twist, 'twist') else 0,
                                        }
                                    }
                                }
                            else:
                                # If there is no next frame or the format is different, use the current frame as action
                                frame_data["action"]["odometry_action"] = frame_data["observation"][sensor_name]
                        else:
                            # The last frame, use the current frame as action
                            frame_data["action"]["odometry_action"] = frame_data["observation"][sensor_name]

                    elif sensor_name.endswith('_wrench_ext_world') or sensor_name.endswith('_wrench_ext_local'):
                        if isinstance(sensor_value, dict) and 'data' in sensor_value:
                            wrench_data = sensor_value['data']
                            frame_data["observation"][sensor_name] = {
                                    'force': {
                                        'x': wrench_data.wrench.force.x if hasattr(wrench_data.wrench, 'force') else 0,
                                        'y': wrench_data.wrench.force.y if hasattr(wrench_data.wrench, 'force') else 0,
                                        'z': wrench_data.wrench.force.z if hasattr(wrench_data.wrench, 'force') else 0,
                                    },
                                    'torque': {
                                        'x': wrench_data.wrench.torque.x if hasattr(wrench_data.wrench, 'torque') else 0,
                                        'y': wrench_data.wrench.torque.y if hasattr(wrench_data.wrench, 'torque') else 0,
                                        'z': wrench_data.wrench.torque.z if hasattr(wrench_data.wrench, 'torque') else 0,
                                    }
                                }
                        else:
                            frame_data["observation"][sensor_name] = sensor_value

                    elif sensor_name == 'chassis_imu':
                        # IMU data
                        if isinstance(sensor_value, dict) and 'data' in sensor_value:
                            imu_data = sensor_value['data']
                            frame_data["observation"][sensor_name] = {
                                'orientation': {
                                    'x': imu_data.orientation.x if hasattr(imu_data, 'orientation') else 0,
                                    'y': imu_data.orientation.y if hasattr(imu_data, 'orientation') else 0,
                                    'z': imu_data.orientation.z if hasattr(imu_data, 'orientation') else 0,
                                    'w': imu_data.orientation.w if hasattr(imu_data, 'orientation') else 1,
                                },
                                'angular_velocity': {
                                    'x': imu_data.angular_velocity.x if hasattr(imu_data, 'angular_velocity') else 0,
                                    'y': imu_data.angular_velocity.y if hasattr(imu_data, 'angular_velocity') else 0,
                                    'z': imu_data.angular_velocity.z if hasattr(imu_data, 'angular_velocity') else 0,
                                },
                                'linear_acceleration': {
                                    'x': imu_data.linear_acceleration.x if hasattr(imu_data, 'linear_acceleration') else 0,
                                    'y': imu_data.linear_acceleration.y if hasattr(imu_data, 'linear_acceleration') else 0,
                                    'z': imu_data.linear_acceleration.z if hasattr(imu_data, 'linear_acceleration') else 0,
                                }
                            }
                        else:
                            frame_data["observation"][sensor_name] = sensor_value

                    elif sensor_name == 'pose':
                        # Pose data
                        if isinstance(sensor_value, dict) and 'data' in sensor_value:
                            pose_data = sensor_value['data']
                            frame_data["observation"][sensor_name] = {
                                'position': {
                                    'x': pose_data.position.x if hasattr(pose_data, 'position') else 0,
                                    'y': pose_data.position.y if hasattr(pose_data, 'position') else 0,
                                    'z': pose_data.position.z if hasattr(pose_data, 'position') else 0,
                                },
                                'orientation': {
                                    'x': pose_data.orientation.x if hasattr(pose_data, 'orientation') else 0,
                                    'y': pose_data.orientation.y if hasattr(pose_data, 'orientation') else 0,
                                    'z': pose_data.orientation.z if hasattr(pose_data, 'orientation') else 0,
                                    'w': pose_data.orientation.w if hasattr(pose_data, 'orientation') else 1,
                                }
                            }
                        else:
                            frame_data["observation"][sensor_name] = sensor_value

                    elif sensor_name.endswith('_gripper_position'):
                        # Gripper position data
                        if isinstance(sensor_value, dict) and 'data' in sensor_value:
                            # Try to convert the message object to a dictionary
                            data_obj = sensor_value['data']
                            # Use the recursive method to convert
                            frame_data["observation"][sensor_name] = self._convert_ros_msg_to_dict(data_obj)

                            action_name = sensor_name.replace(
                                '_gripper_position',
                                '_gripper_position_action',
                            )
                            # Add the action of gripper_position only when a
                            # real VR command was not already written.
                            if action_name in frame_data["action"]:
                                pass
                            elif i + 1 < len(sensor_data_list):
                                next_sensor_value = sensor_data_list[i + 1]
                                if isinstance(next_sensor_value, dict) and 'data' in next_sensor_value:
                                    next_data_obj = next_sensor_value['data']
                                    frame_data["action"][action_name] = self._convert_ros_msg_to_dict(next_data_obj)
                                else:
                                    # If there is no next frame or the format is different, use the current frame as action
                                    frame_data["action"][action_name] = frame_data["observation"][sensor_name]
                            else:
                                # The last frame, use the current frame as action
                                frame_data["action"][action_name] = frame_data["observation"][sensor_name]
                        else:
                            # If the data format is different, save it directly
                            frame_data["observation"][sensor_name] = sensor_value
                            action_name = sensor_name.replace(
                                '_gripper_position',
                                '_gripper_position_action',
                            )
                            # Add the action only when a real VR command was
                            # not already written.
                            if action_name in frame_data["action"]:
                                pass
                            elif i + 1 < len(sensor_data_list):
                                frame_data["action"][action_name] = sensor_data_list[i + 1]
                            else:
                                # The last frame, use the current frame as action
                                frame_data["action"][action_name] = sensor_value

                    else:
                        # Other sensor data
                        if isinstance(sensor_value, dict) and 'data' in sensor_value:
                            # Try to convert the message object to a dictionary
                            data_obj = sensor_value['data']
                            # Use the recursive method to convert
                            frame_data["observation"][sensor_name] = self._convert_ros_msg_to_dict(data_obj)
                        elif isinstance(sensor_value, tuple):
                            # Process the joint state tuple format
                            # Interpolated format: (positions, velocities, efforts) - 3 elements
                            # Or the original format: (timestamp, positions, velocities, efforts) - 4 elements
                            # Or (timestamp, data) - 2 elements
                            if sensor_name.endswith('_joint_state') and len(sensor_value) == 3:
                                # Master joint state format (interpolated): (positions, velocities, efforts)
                                positions = sensor_value[0]
                                velocities = sensor_value[1] if len(sensor_value) > 1 else None
                                efforts = sensor_value[2] if len(sensor_value) > 2 else None

                                joint_state_dict = {}
                                if positions is not None:
                                    joint_state_dict['positions'] = self._convert_ros_msg_to_dict(positions)
                                if velocities is not None:
                                    joint_state_dict['velocities'] = self._convert_ros_msg_to_dict(velocities)
                                if efforts is not None:
                                    joint_state_dict['efforts'] = self._convert_ros_msg_to_dict(efforts)

                                frame_data["observation"][sensor_name] = joint_state_dict
                            else:
                                # Other tuple format, recursive conversion (will handle NumPy arrays)
                                frame_data["observation"][sensor_name] = self._convert_ros_msg_to_dict(sensor_value)
                        else:
                            # Other format, use recursive conversion to ensure NumPy arrays are converted
                            frame_data["observation"][sensor_name] = self._convert_ros_msg_to_dict(sensor_value)

            # Save images
            if self.use_video_storage:
                # Video storage mode: video has already been created, only record the frame number to JSON
                for cam_name in video_files.keys():
                    frame_data["images"][cam_name] = i
            else:
                # Image storage mode: read from the images dictionary and save as files
                for cam_name, cam_images in images.items():
                    if i < len(cam_images):
                        img = cam_images[i]
                        # Image storage mode: save as separate files
                        img_filename = f"frame_{i:04d}_{cam_name}.jpg"
                        img_path = episode_dir / img_filename

                        # Select the save format based on the image type
                        if 'depth' in cam_name:
                            # Depth image: use PNG format (supports floating point) or save as numpy array
                            if isinstance(img, Image.Image):
                                # Depth image is usually in floating point mode, needs special handling
                                if img.mode == 'F':
                                    # Convert the floating point depth image to a visual image and save
                                    # Normalize to 0-255 range for visualization
                                    depth_normalized = (img - img.min()) / (img.max() - img.min()) * 255
                                    depth_vis = Image.fromarray(depth_normalized.astype(np.uint8), mode='L')
                                    img_path = img_path.with_suffix('.png')
                                    depth_vis.save(img_path, 'PNG')
                                    img_filename = f"frame_{i:04d}_{cam_name}.png"
                                else:
                                    img_path = img_path.with_suffix('.png')
                                    img.save(img_path, 'PNG')
                                    img_filename = f"frame_{i:04d}_{cam_name}.png"
                            elif isinstance(img, np.ndarray):
                                if img.dtype == np.float32 or img.dtype == np.float64:
                                    # Depth data: save as numpy array file
                                    img_path = img_path.with_suffix('.npz')
                                    np.savez_compressed(img_path, depth=img)
                                    img_filename = f"frame_{i:04d}_{cam_name}.npz"
                                else:
                                    # Other numpy arrays: convert to image and save
                                    img_path = img_path.with_suffix('.png')
                                    img_pil = Image.fromarray(img.astype(np.uint8))
                                    img_pil.save(img_path, 'PNG')
                                    img_filename = f"frame_{i:04d}_{cam_name}.png"
                            elif isinstance(img, bytes):
                                # Original byte data: save as binary file
                                img_path = img_path.with_suffix('.bin')
                                with open(img_path, 'wb') as f:
                                    f.write(img)
                                img_filename = f"frame_{i:04d}_{cam_name}.bin"
                            else:
                                # Other format: try to save as pickle
                                img_path = img_path.with_suffix('.pkl')
                                with open(img_path, 'wb') as f:
                                    pickle.dump(img, f)
                                img_filename = f"frame_{i:04d}_{cam_name}.pkl"
                        else:
                            # Ordinary RGB image: use JPEG format
                            if isinstance(img, Image.Image):
                                img.save(img_path, 'JPEG', quality=self.image_quality)
                            else:
                                # If it is a numpy array, convert to PIL Image
                                if isinstance(img, np.ndarray):
                                    img = Image.fromarray(img)
                                    img.save(img_path, 'JPEG', quality=self.image_quality)

                        frame_data["images"][cam_name] = str(img_filename)

            episode_json["frames"].append(frame_data)

        # Video storage mode: video has already been created through ffmpeg
        if self.use_video_storage:
            print(f"  ✓ 已保存 {len(video_files)} 个视频文件")

            # Verify the video files
            for cam_name, video_file in video_files.items():
                video_path = episode_dir / video_file
                if video_path.exists():
                    file_size = video_path.stat().st_size
                    print(f"    {cam_name}: {file_size / 1024 / 1024:.2f} MB")
                    if file_size < 1024:  # Less than 1KB
                        print(f"    ⚠️  警告：{cam_name} 文件过小，可能已损坏")

            # Clean up the image data
            if 'aligned_video_images' in locals():
                aligned_video_images.clear()
                del aligned_video_images

            self._release_unused_memory()

        # Save episode JSON file (using temporary file to ensure atomic write)
        episode_json_path = episode_dir / "episode.json"
        temp_json_path = episode_dir / "episode.json.tmp"

        try:
            # Write to temporary file first
            with open(temp_json_path, 'w', encoding='utf-8') as f:
                json.dump(episode_json, f, indent=2, ensure_ascii=False)
                # Ensure the file is fully written to disk
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename (replace existing file)
            if episode_json_path.exists():
                episode_json_path.unlink()
            temp_json_path.rename(episode_json_path)

        except Exception as e:
            print(f"  ⚠️  Error saving JSON file: {e}")
            import traceback
            traceback.print_exc()
            # Try to save directly (if the temporary file method fails)
            try:
                with open(episode_json_path, 'w', encoding='utf-8') as f:
                    json.dump(episode_json, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e2:
                print(f"  ✗ Error saving JSON file: {e2}")
                import traceback
                traceback.print_exc()
                raise
            finally:
                # Clean up the temporary files (if they exist)
                if temp_json_path.exists():
                    try:
                        temp_json_path.unlink()
                    except:
                        pass

        # Validate the complete staged Episode before making it visible.
        with open(episode_json_path, 'r', encoding='utf-8') as f:
            staged_episode_json = json.load(f)
        if staged_episode_json.get("num_frames") != num_frames:
            raise RuntimeError(
                "Episode JSON 帧数校验失败："
                f"预期 {num_frames}，"
                f"实际 {staged_episode_json.get('num_frames')}"
            )
        if len(staged_episode_json.get("frames", [])) != num_frames:
            raise RuntimeError(
                "Episode JSON frames 数量与 num_frames 不一致"
            )
        if self.use_video_storage:
            for video_file in video_files.values():
                staged_video_path = episode_dir / video_file
                if (
                    not staged_video_path.is_file()
                    or staged_video_path.stat().st_size < 1024
                ):
                    raise RuntimeError(
                        f"Episode 视频校验失败：{video_file}"
                    )
                self._fsync_file(staged_video_path)

        self._fsync_directory(episode_dir)
        metadata_entry = {
            "episode_id": episode_id,
            "model": robot_model,
            "task": task,
            "timestamp": episode_json["timestamp"],
            "duration": episode_json["duration"],
            "num_frames": num_frames,
            "path": str(final_episode_dir.relative_to(self.output_dir))
        }

        # Commit the directory and metadata as one recoverable transaction.
        # BaseException is intentional: SIGINT/SystemExit must also roll back.
        try:
            if final_episode_dir.exists():
                raise FileExistsError(
                    "Episode 提交前发现目标目录已经存在："
                    f"{final_episode_dir}"
                )
            os.replace(episode_dir, final_episode_dir)
            self._fsync_directory(self.output_dir)
            episode_json_path = final_episode_dir / "episode.json"

            self.dataset_metadata['episodes'].append(metadata_entry)
            self._save_metadata()
        except BaseException:
            if (
                self.dataset_metadata.get('episodes')
                and self.dataset_metadata['episodes'][-1]
                is metadata_entry
            ):
                self.dataset_metadata['episodes'].pop()

            metadata_rollback_ok = True
            try:
                self._save_metadata()
            except BaseException as metadata_rollback_error:
                metadata_rollback_ok = False
                print(
                    "  ⚠️  metadata 回滚失败；将保留完整 Episode "
                    "目录，程序下次启动时会自动恢复登记："
                    f"{metadata_rollback_error}"
                )

            try:
                if (
                    metadata_rollback_ok
                    and final_episode_dir.exists()
                    and not episode_dir.exists()
                ):
                    os.replace(final_episode_dir, episode_dir)
                    self._fsync_directory(self.output_dir)
                    self._active_episode_staging_dir = episode_dir
            except BaseException as rollback_error:
                print(
                    "  ⚠️  Episode 目录回滚失败，但目录中的 "
                    "JSON 和视频均已完整；程序下次启动时会"
                    f"自动恢复登记：{rollback_error}"
                )
            raise

        self._active_episode_staging_dir = None
        try:
            self._cleanup_temp_files()
        except Exception as error:
            print(f"  ⚠️  清理临时文件时出错：{error}")
        self._release_unused_memory()

        print(f"✓ 第 {episode_id} 条数据已保存")
        print(f"  - 帧数：{num_frames}")
        print(f"  - 时长：{episode_json['duration']:.2f} 秒")
        print(f"  - JSON 文件：{episode_json_path}")
        if self.use_video_storage:
            total_video_frames = num_frames * len(video_files)
            print(f"  - 视频：{len(video_files)} 个 MP4 文件（每个 {num_frames} 帧）")
        else:
            total_image_count = sum(len(imgs) for imgs in images.values())
            print(f"  - 图片：{total_image_count} 个 JPG 文件")

        return {
            "episode_id": episode_id,
            "episode_dir": str(final_episode_dir),
            "episode_json": str(episode_json_path),
            "num_frames": num_frames,
            "duration": episode_json["duration"]
        }
