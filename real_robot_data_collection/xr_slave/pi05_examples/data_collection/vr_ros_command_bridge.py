"""ROS 2 subscriber bridge for selected commands and Revo2 telemetry.

This helper runs in a child process with /opt/ros/jazzy/setup.bash loaded.
It emits one compact JSON object per command on stdout for DataCollector.
"""

import argparse
import json
import math
import os
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32, String


TOPICS = {
    'vr_left_arm_pose_commands': '/whole_body_controller/left_wrist_pose_cmd',
    'vr_right_arm_pose_commands': '/whole_body_controller/right_wrist_pose_cmd',
    'vr_left_gripper_joint_commands': '/gripper/cmd/left',
    'vr_right_gripper_joint_commands': '/gripper/cmd/right',
    'vr_left_revo2_joint_commands': '/revo2/command/left',
    'vr_right_revo2_joint_commands': '/revo2/command/right',
    'left_revo2_joint_states': '/revo2/state/left',
    'right_revo2_joint_states': '/revo2/state/right',
    'left_revo2_tactile': '/revo2/tactile/left',
    'right_revo2_tactile': '/revo2/tactile/right',
}

REVO2_JSON_SCHEMAS = {
    'vr_left_revo2_joint_commands': 'revo2_joint_command_v1',
    'vr_right_revo2_joint_commands': 'revo2_joint_command_v1',
    'left_revo2_joint_states': 'revo2_joint_state_v1',
    'right_revo2_joint_states': 'revo2_joint_state_v1',
    'left_revo2_tactile': 'revo2_tactile_v1',
    'right_revo2_tactile': 'revo2_tactile_v1',
}
REVO2_ACTUATOR_NAMES = (
    'thumb', 'thumb_aux', 'index', 'middle', 'ring', 'pinky'
)
REVO2_FINGER_NAMES = ('thumb', 'index', 'middle', 'ring', 'pinky')


def _finite_vector(value, length, field_name):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f'{field_name} must contain {length} values')
    result = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f'{field_name} contains a boolean')
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f'{field_name} contains a non-finite value')
        result.append(number)
    return result


def validate_revo2_payload(sensor_name, payload):
    expected_side = 'left' if 'left' in sensor_name else 'right'
    if payload.get('side') != expected_side:
        raise ValueError(
            f"side={payload.get('side')!r}, expected={expected_side!r}"
        )

    if sensor_name.endswith('_joint_states'):
        if tuple(payload.get('actuator_names', ())) != REVO2_ACTUATOR_NAMES:
            raise ValueError('invalid actuator_names')
        positions = _finite_vector(payload.get('positions'), 6, 'positions')
        speeds = _finite_vector(payload.get('speeds'), 6, 'speeds')
        currents = _finite_vector(payload.get('currents'), 6, 'currents')
        states = payload.get('states')
        if (
            any(not 0 <= value <= 1000 for value in positions)
            or any(not -1000 <= value <= 1000 for value in speeds)
            or any(not -1000 <= value <= 1000 for value in currents)
            or not isinstance(states, list)
            or len(states) != 6
            or any(not isinstance(value, str) or not value for value in states)
        ):
            raise ValueError('invalid Revo2 joint state values')
        return

    if sensor_name.endswith('_tactile'):
        if tuple(payload.get('finger_names', ())) != REVO2_FINGER_NAMES:
            raise ValueError('invalid finger_names')
        fingers = payload.get('fingers')
        if not isinstance(fingers, list) or len(fingers) != 5:
            raise ValueError('fingers must contain five records')
        for expected_name, finger in zip(REVO2_FINGER_NAMES, fingers):
            if not isinstance(finger, dict) or finger.get('name') != expected_name:
                raise ValueError('tactile finger order is invalid')
            if not isinstance(finger.get('status'), str):
                raise ValueError(f'{expected_name} status is invalid')
            if not isinstance(finger.get('description'), str):
                raise ValueError(f'{expected_name} description is invalid')
            for field_name in (
                'normal_force',
                'tangential_force',
                'tangential_direction',
                'proximity',
            ):
                _finite_vector(
                    finger.get(field_name),
                    3,
                    f'{expected_name}.{field_name}',
                )
        return

    if sensor_name.startswith('vr_'):
        if tuple(payload.get('actuator_names', ())) != REVO2_ACTUATOR_NAMES:
            raise ValueError('invalid actuator_names')
        positions = _finite_vector(payload.get('positions'), 6, 'positions')
        if any(not 0 <= value <= 1000 for value in positions):
            raise ValueError('command positions are outside 0..1000')
        if payload.get('position_unit') != 'normalized_0_1000':
            raise ValueError('invalid position_unit')
        duration_ms = payload.get('duration_ms')
        if (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, int)
            or duration_ms <= 0
        ):
            raise ValueError('invalid duration_ms')
        _finite_vector(
            [payload.get('raw_trigger'), payload.get('normalized_trigger')],
            2,
            'trigger values',
        )
        return

    raise ValueError(f'unsupported Revo2 sensor name: {sensor_name}')


def emit(
    sensor_name,
    data,
    timestamp=None,
    timestamp_source='receipt_wall_clock',
):
    receive_timestamp = time.time()
    effective_timestamp = (
        receive_timestamp if timestamp is None else float(timestamp)
    )
    print(
        json.dumps(
            {
                'sensor_name': sensor_name,
                'timestamp': effective_timestamp,
                'receive_timestamp': receive_timestamp,
                'timestamp_source': timestamp_source,
                'data': data,
            },
            separators=(',', ':'),
        ),
        flush=True,
    )


def pose_callback(sensor_name):
    def callback(message):
        pose = message.pose
        stamp = message.header.stamp
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
        if sec != 0 or nanosec != 0:
            timestamp = float(sec) + float(nanosec) / 1e9
            timestamp_source = 'message_header'
        else:
            timestamp = None
            timestamp_source = 'receipt_wall_clock_zero_header'

        emit(
            sensor_name,
            {
                'position': {
                    'x': float(pose.position.x),
                    'y': float(pose.position.y),
                    'z': float(pose.position.z),
                },
                'orientation': {
                    'x': float(pose.orientation.x),
                    'y': float(pose.orientation.y),
                    'z': float(pose.orientation.z),
                    'w': float(pose.orientation.w),
                },
            },
            timestamp=timestamp,
            timestamp_source=timestamp_source,
        )

    return callback


def gripper_callback(sensor_name):
    def callback(message):
        # std_msgs/Float32 has no Header, so its own source timestamp does not
        # exist. Preserve the bridge receipt time and label it explicitly.
        emit(
            sensor_name,
            {'position': float(message.data)},
            timestamp_source='receipt_wall_clock_no_message_header',
        )

    return callback


def revo2_json_callback(sensor_name):
    expected_schema = REVO2_JSON_SCHEMAS[sensor_name]

    def callback(message):
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise TypeError("payload is not an object")
            if payload.get('schema') != expected_schema:
                raise ValueError(
                    f"schema={payload.get('schema')!r}, "
                    f"expected={expected_schema!r}"
                )
            validate_revo2_payload(sensor_name, payload)
            timestamp = float(payload['timestamp'])
            if not math.isfinite(timestamp):
                raise ValueError("timestamp is not finite")
            emit(
                sensor_name,
                payload,
                timestamp=timestamp,
                timestamp_source='revo2_bridge_wall_clock',
            )
        except Exception as error:
            print(
                f"invalid {sensor_name} Revo2 JSON: {error}",
                file=sys.stderr,
                flush=True,
            )

    return callback


def emit_ready(sensor_names):
    print(
        json.dumps(
            {
                'event': 'ready',
                'timestamp': time.time(),
                'topics': {
                    sensor_name: TOPICS[sensor_name]
                    for sensor_name in sensor_names
                },
            },
            separators=(',', ':'),
        ),
        flush=True,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sensors',
        nargs='+',
        choices=tuple(TOPICS),
        default=tuple(TOPICS),
        help='Only subscribe to the sensor names requested by DataCollector.',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sensor_names = tuple(dict.fromkeys(args.sensors))
    rclpy.init()
    node = rclpy.create_node(f'pi05_vr_command_bridge_{os.getpid()}')
    for sensor_name in (
        'vr_left_arm_pose_commands',
        'vr_right_arm_pose_commands',
    ):
        if sensor_name in sensor_names:
            node.create_subscription(
                PoseStamped,
                TOPICS[sensor_name],
                pose_callback(sensor_name),
                qos_profile_sensor_data,
            )
    for sensor_name in (
        'vr_left_gripper_joint_commands',
        'vr_right_gripper_joint_commands',
    ):
        if sensor_name in sensor_names:
            node.create_subscription(
                Float32,
                TOPICS[sensor_name],
                gripper_callback(sensor_name),
                qos_profile_sensor_data,
            )
    for sensor_name in REVO2_JSON_SCHEMAS:
        if sensor_name in sensor_names:
            node.create_subscription(
                String,
                TOPICS[sensor_name],
                revo2_json_callback(sensor_name),
                qos_profile_sensor_data,
            )
    emit_ready(sensor_names)

    try:
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
