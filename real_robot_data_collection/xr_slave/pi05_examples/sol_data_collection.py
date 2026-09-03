"""
Data Collection Example

This script demonstrates how to use DataCollector to collect robot data
"""

import fcntl
import functools
import json
import math
import os
import socket
import struct
import subprocess
import time
from types import SimpleNamespace
from typing import Annotated
from pathlib import Path
import typer
import signal
import sys

# Add current directory to Python path (to import data_collection module)
sys.path.insert(0, str(Path(__file__).parent))

from data_collection.data_collector import (
    DataCollector,
    REVO2_ACTUATOR_NAMES,
)
from data_collection.collection_config import CollectionConfig
from x2robot import connect


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'sol_data_collection' / 'task6'
COLLECTION_LOCK_FILENAME = '.collection_web.lock'

# The physical E6 is a USB KONA device. Its debug streaming app publishes a
# framed Stage-4 H265 stream on headset port 8554. One encoded 3200x1200 frame
# contains side-by-side stereo; the physical right eye is the right 1600x1200
# half. Robot video0/video1/video2 topics are deliberately never used here.
E6_SERIAL = os.environ.get('E6_SERIAL', '').strip()
E6_ADB_BIN = os.environ.get(
    'E6_ADB_BIN',
    '/home/xr/.local/adb-runtime/root/usr/bin/adb',
)
E6_ADB_LIBRARY_PATH = os.environ.get(
    'E6_ADB_LIBRARY_PATH',
    '/home/xr/.local/adb-runtime/root/usr/lib/x86_64-linux-gnu/android',
)
E6_ADB_VENDOR_KEYS = os.environ.get(
    'E6_ADB_VENDOR_KEYS',
    '/home/xr/.android',
)
E6_STREAM_PACKAGE = 'com.ssnwt.e6stream.debug'
E6_STREAM_ACTIVITY = (
    'com.ssnwt.e6stream.debug/com.ssnwt.e6stream.VrNativeActivity'
)
E6_REMOTE_STREAM_PORT = int(os.environ.get('E6_REMOTE_STREAM_PORT', '8554'))
E6_LOCAL_STREAM_PORT = int(os.environ.get('E6_LOCAL_STREAM_PORT', '18554'))
E6_SOURCE_WIDTH = 3200
E6_SOURCE_HEIGHT = 1200
E6_RIGHT_WIDTH = E6_SOURCE_WIDTH // 2
E6_STREAM_FPS = 60.0
E6_FRAME_METADATA_SIZE = 96
E6_PACKET_MAX_PAYLOAD = 32 * 1024 * 1024
E6_PACKET_IDLE_TIMEOUT_SECONDS = 5.0
E6_TIMESTAMP_MAX_SKEW_SECONDS = 30.0
E6_PACKET_HEADER = struct.Struct('>4sHHIHHIQQIq')

QUANTA_X2_RAW_TOPICS = {
    'head_rgb_stream',
    'left_arm_rgb_stream',
    'right_arm_rgb_stream',
    'left_arm_end_pose',
    'right_arm_end_pose',
    'left_revo2_joint_states',
    'right_revo2_joint_states',
    'vr_left_revo2_joint_commands',
    'vr_right_revo2_joint_commands',
}
WRIST_FISHEYE_TOPICS = {
    'left_arm_rgb_stream',
    'right_arm_rgb_stream',
}


class E6TransportError(RuntimeError):
    """Transient ADB/TCP failure; preflight may keep retrying."""


class E6ProtocolError(RuntimeError):
    """The connected endpoint does not satisfy the E6 stream contract."""


class E6H265Frame:
    """One E6 stereo HEVC access unit with a logical right-eye view."""

    def __init__(
        self,
        *,
        access_unit: bytes,
        codec_config: bytes,
        timestamp: float,
        timestamp_ns: int,
        frame_id: int,
        sequence_number: int,
        session_id: int,
        is_keyframe: bool,
        preview_jpeg: bytes | None,
    ):
        seconds = int(timestamp)
        nanoseconds = int(round((timestamp - seconds) * 1_000_000_000))
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds -= 1_000_000_000
        # Repeating VPS/SPS/PPS on every keyframe keeps a concatenated raw
        # topic independently decodable without decoding or transcoding here.
        self.data = (
            bytes(codec_config) + bytes(access_unit)
            if is_keyframe
            else bytes(access_unit)
        )
        self.format = 'h265_annexb'
        self.width = E6_RIGHT_WIDTH
        self.height = E6_SOURCE_HEIGHT
        self.source_width = E6_SOURCE_WIDTH
        self.source_height = E6_SOURCE_HEIGHT
        self.view_crop = (
            E6_RIGHT_WIDTH,
            0,
            E6_RIGHT_WIDTH,
            E6_SOURCE_HEIGHT,
        )
        self.frame_id = int(frame_id)
        self.sequence_number = int(sequence_number)
        self.session_id = int(session_id)
        self.device_timestamp_ns = int(timestamp_ns)
        self.is_keyframe = bool(is_keyframe)
        self.preview_jpeg = preview_jpeg
        self.header = SimpleNamespace(
            stamp=SimpleNamespace(sec=seconds, nanosec=nanoseconds),
        )


class E6RightHeadDataCollector(DataCollector):
    """Collect the E6 KONA headset's physical right-eye RGB camera."""

    def _run_e6_adb(self, *arguments, check=True, timeout=8.0):
        environment = os.environ.copy()
        existing_library_path = environment.get('LD_LIBRARY_PATH', '')
        environment['LD_LIBRARY_PATH'] = E6_ADB_LIBRARY_PATH + (
            f':{existing_library_path}' if existing_library_path else ''
        )
        environment['ADB_VENDOR_KEYS'] = E6_ADB_VENDOR_KEYS
        command = [E6_ADB_BIN, '-s', E6_SERIAL, *arguments]
        try:
            result = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise E6TransportError(f'ADB 命令无法执行：{error}') from error
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout or '').strip()
            raise E6TransportError(
                f"ADB 命令失败（{result.returncode}）：{detail or '无输出'}"
            )
        return result.stdout.strip(), result.returncode

    def _prepare_e6_transport(self):
        if not E6_SERIAL:
            raise E6TransportError(
                '未配置 E6_SERIAL；请在启动数采服务前设置头环 ADB 序列号'
            )
        state, _ = self._run_e6_adb('get-state')
        if state != 'device':
            raise E6TransportError(
                f'E6 ADB 状态为 {state!r}，请检查 USB 连接和授权'
            )

        process_id, _ = self._run_e6_adb(
            'shell',
            'pidof',
            E6_STREAM_PACKAGE,
            check=False,
        )
        if not process_id:
            self._run_e6_adb(
                'shell',
                'am',
                'start',
                '-n',
                E6_STREAM_ACTIVITY,
            )
            stop_event = getattr(self, '_episode_stop_event', None)
            if stop_event is not None and stop_event.wait(2.0):
                return False

        self._run_e6_adb(
            'forward',
            f'tcp:{E6_LOCAL_STREAM_PORT}',
            f'tcp:{E6_REMOTE_STREAM_PORT}',
        )
        return True

    def _receive_e6_exact(self, stream_socket, size):
        buffer = bytearray()
        deadline = time.monotonic() + E6_PACKET_IDLE_TIMEOUT_SECONDS
        while len(buffer) < size:
            if not self._recording_active():
                return None
            try:
                chunk = stream_socket.recv(size - len(buffer))
            except socket.timeout:
                if time.monotonic() >= deadline:
                    raise E6TransportError(
                        'E6 Stage-4 TCP 已连接，但 5 秒内没有收到数据'
                    )
                continue
            if not chunk:
                raise E6TransportError('E6 Stage-4 TCP 连接已关闭')
            buffer.extend(chunk)
            deadline = time.monotonic() + E6_PACKET_IDLE_TIMEOUT_SECONDS
        return bytes(buffer)

    def _decode_e6_right_preview(self, encoded_keyframe):
        """Decode only one IDR per second for the web's right-eye preview."""
        command = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel',
            'error',
            '-threads',
            '1',
            '-f',
            'hevc',
            '-i',
            'pipe:0',
            '-an',
            '-frames:v',
            '1',
            '-vf',
            (
                f'crop={E6_RIGHT_WIDTH}:{E6_SOURCE_HEIGHT}:'
                f'{E6_RIGHT_WIDTH}:0,scale=640:480'
            ),
            '-q:v',
            '6',
            '-f',
            'image2pipe',
            '-vcodec',
            'mjpeg',
            'pipe:1',
        ]
        try:
            result = subprocess.run(
                command,
                input=encoded_keyframe,
                capture_output=True,
                timeout=3.0,
                check=False,
            )
            jpeg = result.stdout
            if (
                result.returncode != 0
                or len(jpeg) < 4
                or jpeg[:2] != b'\xff\xd8'
                or jpeg[-2:] != b'\xff\xd9'
            ):
                detail = result.stderr.decode('utf-8', errors='replace').strip()
                raise RuntimeError(detail or f'ffmpeg 返回 {result.returncode}')
            self._e6_preview_warning_emitted = False
            return jpeg
        except Exception as error:
            if not getattr(self, '_e6_preview_warning_emitted', False):
                print(
                    '⚠️  E6 右目网页预览解码失败；压缩原始流仍会继续保存：'
                    f'{type(error).__name__}: {error}',
                    flush=True,
                )
                self._e6_preview_warning_emitted = True
            return None

    def _iter_e6_connection(self):
        address = ('127.0.0.1', E6_LOCAL_STREAM_PORT)
        try:
            stream_socket = socket.create_connection(address, timeout=5.0)
        except OSError as error:
            raise E6TransportError(
                f'无法连接 E6 转发端口 {address[0]}:{address[1]}：{error}'
            ) from error

        codec_config = None
        clock_offset_ns = None
        session_width = None
        session_height = None
        session_fps = None
        last_preview_monotonic = 0.0
        try:
            stream_socket.settimeout(0.5)
            while self._recording_active():
                raw_header = self._receive_e6_exact(
                    stream_socket,
                    E6_PACKET_HEADER.size,
                )
                if raw_header is None:
                    return
                (
                    magic,
                    version,
                    packet_type,
                    flags,
                    header_size,
                    _reserved,
                    payload_size,
                    session_id,
                    sequence_number,
                    frame_id,
                    timestamp_ns,
                ) = E6_PACKET_HEADER.unpack(raw_header)
                if magic != b'E6S3' or version != 3:
                    raise E6ProtocolError(
                        f'E6 协议头无效：magic={magic!r}, version={version}'
                    )
                if header_size != E6_PACKET_HEADER.size:
                    raise E6ProtocolError(
                        f'E6 协议头长度为 {header_size}，要求 '
                        f'{E6_PACKET_HEADER.size}'
                    )
                if payload_size > E6_PACKET_MAX_PAYLOAD:
                    raise E6ProtocolError(
                        f'E6 单包长度异常：{payload_size} bytes'
                    )
                payload = self._receive_e6_exact(stream_socket, payload_size)
                if payload is None:
                    return

                if packet_type == 1:
                    if len(payload) < 36:
                        raise E6ProtocolError('E6 Session Start 数据过短')
                    session_width, session_height, session_fps = (
                        struct.unpack_from('>III', payload, 0)
                    )
                    boot_time_ns, realtime_ns, uncertainty_ns = (
                        struct.unpack_from('>qqq', payload, len(payload) - 24)
                    )
                    if boot_time_ns <= 0 or realtime_ns <= 0:
                        raise E6ProtocolError('E6 Session Start 时钟锚点无效')
                    if uncertainty_ns < 0:
                        raise E6ProtocolError('E6 Session Start 时钟误差无效')
                    # The headset wall clock can differ from XR by seconds.
                    # Preserve the E6 boot-clock intervals, but anchor that
                    # clock to XR at Session Start so every robot topic shares
                    # one epoch without a hard-coded timezone/skew offset.
                    xr_session_receipt_ns = time.time_ns()
                    headset_to_xr_clock_ns = (
                        xr_session_receipt_ns - realtime_ns
                    )
                    clock_offset_ns = (
                        realtime_ns
                        - boot_time_ns
                        + headset_to_xr_clock_ns
                    )
                    if (
                        session_width != E6_SOURCE_WIDTH
                        or session_height != E6_SOURCE_HEIGHT
                        or not 55 <= session_fps <= 65
                    ):
                        raise E6ProtocolError(
                            'E6 RGB 参数不满足采集契约：'
                            f'{session_width}x{session_height}@{session_fps}Hz，'
                            f'要求 {E6_SOURCE_WIDTH}x{E6_SOURCE_HEIGHT}@60Hz'
                        )
                    continue

                if packet_type == 2:
                    if b'\x00\x00\x01' not in payload[:128]:
                        raise E6ProtocolError('E6 H265 codec config 不是 Annex-B')
                    codec_config = bytes(payload)
                    continue

                if packet_type == 4:
                    raise E6TransportError('E6 主动结束了当前视频会话')

                if packet_type != 3:
                    # Audio config/data are intentionally not part of this task.
                    continue
                if (
                    clock_offset_ns is None
                    or codec_config is None
                    or session_width is None
                    or session_height is None
                    or session_fps is None
                ):
                    raise E6ProtocolError('E6 RGB 帧早于 Session/codec config')
                if len(payload) <= E6_FRAME_METADATA_SIZE:
                    raise E6ProtocolError('E6 RGB 帧 payload 过短')

                access_unit = bytes(payload[E6_FRAME_METADATA_SIZE:])
                if b'\x00\x00\x01' not in access_unit[:128]:
                    raise E6ProtocolError('E6 RGB 帧不是 Annex-B H265')
                timestamp = (timestamp_ns + clock_offset_ns) / 1_000_000_000
                skew = abs(timestamp - time.time())
                if (
                    not math.isfinite(timestamp)
                    or skew > E6_TIMESTAMP_MAX_SKEW_SECONDS
                ):
                    raise E6ProtocolError(
                        'E6 RGB 时间戳与 XR 时钟不一致：'
                        f'偏差 {skew:.3f}s，限制 '
                        f'{E6_TIMESTAMP_MAX_SKEW_SECONDS:.1f}s'
                    )

                is_keyframe = bool(flags & 1)
                preview_jpeg = None
                now_monotonic = time.monotonic()
                if (
                    is_keyframe
                    and now_monotonic - last_preview_monotonic >= 0.8
                ):
                    preview_jpeg = self._decode_e6_right_preview(
                        codec_config + access_unit
                    )
                    last_preview_monotonic = now_monotonic

                yield E6H265Frame(
                    access_unit=access_unit,
                    codec_config=codec_config,
                    timestamp=timestamp,
                    timestamp_ns=timestamp_ns,
                    frame_id=frame_id,
                    sequence_number=sequence_number,
                    session_id=session_id,
                    is_keyframe=is_keyframe,
                    preview_jpeg=preview_jpeg,
                )
        finally:
            stream_socket.close()

    def _e6_right_stream(self, timeout=None):
        del timeout
        retry_delay = 0.5
        while self._recording_active():
            received_frame = False
            try:
                if not self._prepare_e6_transport():
                    return
                for frame in self._iter_e6_connection():
                    received_frame = True
                    retry_delay = 0.5
                    yield frame
                if self._recording_active():
                    raise E6TransportError('E6 视频会话意外结束')
                return
            except E6ProtocolError:
                raise
            except (E6TransportError, OSError) as error:
                if not self._recording_active():
                    return
                suffix = '' if received_frame else '（尚未收到 RGB 帧）'
                print(
                    '⚠️  E6 头环右目 RGB 暂时不可用，正在重连'
                    f'{suffix}：{error}',
                    flush=True,
                )
                stop_event = getattr(self, '_episode_stop_event', None)
                if stop_event is not None and stop_event.wait(retry_delay):
                    return
                if stop_event is None:
                    time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 2.0)

    def _collect_head_rgb_stream(self):
        print(
            '正在启动 E6 头环右目 RGB 作为 head_rgb_stream',
            flush=True,
        )
        self._collect_camera_stream(
            'head_rgb_stream',
            self._e6_right_stream,
        )

    def _raw_topic_descriptor(self, topic_name, is_camera):
        if topic_name == 'head_rgb_stream':
            camera_status = self.camera_profile_status().get(topic_name)
            descriptor = {
                'kind': 'camera',
                'payload_encoding': 'h265_annexb_stereo_access_unit_bytes',
                'record_schema': (
                    '(timestamp_seconds, h265_annexb_access_unit_bytes)'
                ),
                'timestamp_source': (
                    'e6_boottime_mapped_to_xr_at_session_start'
                ),
                'source_device': 'E6',
                'source_serial': E6_SERIAL,
                'source_package': E6_STREAM_PACKAGE,
                'source_view': 'physical_right_eye_rgb',
                'encoded_layout': 'side_by_side_stereo',
                'source_resolution': [E6_SOURCE_WIDTH, E6_SOURCE_HEIGHT],
                'logical_resolution': [E6_RIGHT_WIDTH, E6_SOURCE_HEIGHT],
                'logical_crop_xywh': [
                    E6_RIGHT_WIDTH,
                    0,
                    E6_RIGHT_WIDTH,
                    E6_SOURCE_HEIGHT,
                ],
                'transport': 'adb_forward_tcp_e6_stage4_v3',
                'headset_port': E6_REMOTE_STREAM_PORT,
                'codec': 'H265',
                'codec_config_policy': 'vps_sps_pps_repeated_on_keyframes',
                'live_decode_policy': 'right_eye_keyframes_only_for_preview',
            }
            if camera_status is not None:
                descriptor['capture_profile'] = camera_status['expected']
                descriptor['observed_profile'] = camera_status['observed']
                descriptor['capture_profile_valid'] = bool(
                    camera_status['valid']
                )
            return descriptor
        return super()._raw_topic_descriptor(topic_name, is_camera)


def exclusive_collection_lock(function):
    """Prevent the CLI and web collector from writing one dataset together."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        lock_path = DEFAULT_OUTPUT_DIR / COLLECTION_LOCK_FILENAME
        lock_handle = lock_path.open('a+', encoding='utf-8')
        try:
            try:
                fcntl.flock(
                    lock_handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError as error:
                typer.echo(
                    '错误：数据集正在被网页或另一个数采进程占用；'
                    '请勿同时运行网页与命令行采集',
                    err=True,
                )
                raise typer.Exit(code=1) from error

            lock_handle.seek(0)
            lock_handle.truncate()
            lock_handle.write(
                f'mode=cli pid={os.getpid()} started_at={time.time()}\n'
            )
            lock_handle.flush()
            return function(*args, **kwargs)
        finally:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock_handle.close()

    return wrapper


def signal_handler(sig, frame):
    """Handle Ctrl+C signal"""
    print("\n\n收到中断信号，正在停止...")
    # Note: DataCollector has already registered a signal handler to clean up temporary files
    # Just exit here, cleanup work is done by DataCollector's signal handler
    sys.exit(0)


def verify_saved_episode(episode_info: dict) -> None:
    """确认 Episode 已完整落盘后才允许进入下一条数据。"""
    episode_json_path = Path(episode_info['episode_json'])
    if not episode_json_path.is_file() or episode_json_path.stat().st_size == 0:
        raise RuntimeError(f"JSON 文件未正确保存：{episode_json_path}")

    with episode_json_path.open('r', encoding='utf-8') as file:
        episode_json = json.load(file)

    episode_dir = Path(episode_info['episode_dir'])
    storage_format = episode_json.get('storage_format')
    if storage_format == 'raw_topics_pickle_v1':
        if episode_info.get('storage_format') != storage_format:
            raise RuntimeError("raw topic 保存结果与 Episode JSON 格式不一致")
        if (
            episode_json.get('timestamp_alignment') != 'none'
            or episode_json.get('timestamp_policy')
            != 'independent_per_topic'
        ):
            raise RuntimeError("raw topic 时间戳策略校验失败")

        raw_topics = episode_json.get('raw_topics', {})
        if (
            not raw_topics
            or len(raw_topics) != int(episode_info['num_topics'])
            or len(raw_topics) != int(episode_json.get('num_topics', -1))
        ):
            raise RuntimeError("raw topic 数量校验失败")

        if episode_json.get('model') == 'quanta_x2':
            actual_topics = set(raw_topics)
            if actual_topics != QUANTA_X2_RAW_TOPICS:
                raise RuntimeError(
                    "Quanta X2 的 9 路采集契约不完整："
                    f"缺少 {sorted(QUANTA_X2_RAW_TOPICS - actual_topics)}；"
                    f"多出 {sorted(actual_topics - QUANTA_X2_RAW_TOPICS)}"
                )
            head_descriptor = raw_topics['head_rgb_stream']
            head_expected = head_descriptor.get('capture_profile', {})
            head_observed = head_descriptor.get('observed_profile', {})
            try:
                head_observed_fps = float(head_observed['fps'])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError('E6 右目 RGB 缺少实测帧率') from error
            if (
                head_descriptor.get('payload_encoding')
                != 'h265_annexb_stereo_access_unit_bytes'
                or head_descriptor.get('source_device') != 'E6'
                or head_descriptor.get('source_package') != E6_STREAM_PACKAGE
                or head_descriptor.get('source_view')
                != 'physical_right_eye_rgb'
                or head_descriptor.get('encoded_layout')
                != 'side_by_side_stereo'
                or head_descriptor.get('source_resolution')
                != [E6_SOURCE_WIDTH, E6_SOURCE_HEIGHT]
                or head_descriptor.get('logical_resolution')
                != [E6_RIGHT_WIDTH, E6_SOURCE_HEIGHT]
                or head_descriptor.get('logical_crop_xywh')
                != [
                    E6_RIGHT_WIDTH,
                    0,
                    E6_RIGHT_WIDTH,
                    E6_SOURCE_HEIGHT,
                ]
                or head_descriptor.get('transport')
                != 'adb_forward_tcp_e6_stage4_v3'
                or not head_descriptor.get('capture_profile_valid')
                or head_expected.get('capture_codec') != 'H265'
                or head_expected.get('storage_encoding')
                != 'h265_annexb_access_units'
                or int(head_expected.get('source_width', 0))
                != E6_SOURCE_WIDTH
                or int(head_expected.get('source_height', 0))
                != E6_SOURCE_HEIGHT
                or int(head_expected.get('width', 0)) != E6_RIGHT_WIDTH
                or int(head_expected.get('height', 0)) != E6_SOURCE_HEIGHT
                or not math.isclose(
                    float(head_expected.get('fps', 0)),
                    E6_STREAM_FPS,
                    abs_tol=1e-9,
                )
                or int(head_observed.get('source_width', 0))
                != E6_SOURCE_WIDTH
                or int(head_observed.get('source_height', 0))
                != E6_SOURCE_HEIGHT
                or int(head_observed.get('width', 0)) != E6_RIGHT_WIDTH
                or int(head_observed.get('height', 0)) != E6_SOURCE_HEIGHT
                or not 52.0 <= head_observed_fps <= 68.0
            ):
                raise RuntimeError(
                    'head_rgb_stream 不是经过预检的 E6 物理右目 H265 视频流'
                )
            for camera_topic in WRIST_FISHEYE_TOPICS:
                descriptor = raw_topics[camera_topic]
                expected_profile = descriptor.get('capture_profile', {})
                observed_profile = descriptor.get('observed_profile', {})
                try:
                    observed_fps = float(observed_profile['fps'])
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"{camera_topic} 缺少实测帧率"
                    ) from error
                if (
                    descriptor.get('payload_encoding') != 'jpeg_bytes'
                    or not descriptor.get('capture_profile_valid')
                    or expected_profile.get('capture_codec') != 'MJPEG'
                    or expected_profile.get('storage_encoding')
                    != 'jpeg_frames'
                    or int(expected_profile.get('width', 0)) != 640
                    or int(expected_profile.get('height', 0)) != 480
                    or not math.isclose(
                        float(expected_profile.get('fps', 0)),
                        30.0,
                        abs_tol=1e-9,
                    )
                    or int(observed_profile.get('width', 0)) != 640
                    or int(observed_profile.get('height', 0)) != 480
                    or not 27.0 <= observed_fps <= 33.0
                ):
                    raise RuntimeError(
                        f"{camera_topic} 未满足鱼眼 MJPEG "
                        "640x480@30Hz 采集契约"
                    )

            for side in ('left', 'right'):
                joint_descriptor = raw_topics[
                    f'{side}_revo2_joint_states'
                ]
                command_descriptor = raw_topics[
                    f'vr_{side}_revo2_joint_commands'
                ]
                if (
                    joint_descriptor.get('payload_schema')
                    != 'revo2_joint_state_v1'
                    or tuple(joint_descriptor.get('actuator_names', ()))
                    != REVO2_ACTUATOR_NAMES
                    or command_descriptor.get('payload_schema')
                    != 'revo2_joint_command_v1'
                    or tuple(command_descriptor.get('actuator_names', ()))
                    != REVO2_ACTUATOR_NAMES
                ):
                    raise RuntimeError(
                        f"{side} Revo2 关节/动作描述符不完整"
                    )

        has_effective_start = (
            'effective_start_timestamp' in episode_json
        )
        has_effective_end = (
            'effective_end_timestamp' in episode_json
        )
        has_effective_context = (
            'effective_context_topics' in episode_json
        )
        effective_companion_fields = (
            'effective_duration',
            'effective_interval_policy',
            'capture_warmup_seconds',
        )
        if has_effective_start != has_effective_end:
            raise RuntimeError(
                "effective start/end 字段必须成对保存"
            )

        effective_start = None
        effective_end = None
        effective_context_topics = set()
        effective_field_names = (
            'effective_start_timestamp',
            'effective_end_timestamp',
            *effective_companion_fields,
            'effective_context_topics',
        )
        if has_effective_start:
            missing_effective_fields = [
                field_name
                for field_name in (
                    'effective_context_topics',
                    *effective_companion_fields,
                )
                if field_name not in episode_json
            ]
            if missing_effective_fields:
                raise RuntimeError(
                    "effective 时间区间缺少字段："
                    f"{missing_effective_fields}"
                )
            if isinstance(
                episode_json['effective_start_timestamp'],
                bool,
            ) or isinstance(
                episode_json['effective_end_timestamp'],
                bool,
            ):
                raise RuntimeError(
                    "effective 时间戳不是有效数值"
                )
            try:
                effective_start = float(
                    episode_json['effective_start_timestamp']
                )
                effective_end = float(
                    episode_json['effective_end_timestamp']
                )
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "effective 时间戳不是有效数值"
                ) from error
            if (
                not math.isfinite(effective_start)
                or not math.isfinite(effective_end)
            ):
                raise RuntimeError("effective 时间戳必须是有限数值")
            if effective_end <= effective_start:
                raise RuntimeError(
                    "effective end 必须晚于 effective start"
                )

            if isinstance(
                episode_json['effective_duration'],
                bool,
            ) or isinstance(
                episode_json['capture_warmup_seconds'],
                bool,
            ):
                raise RuntimeError(
                    "effective 时长或预采集时长无效"
                )
            try:
                effective_duration = float(
                    episode_json['effective_duration']
                )
                capture_warmup_seconds = float(
                    episode_json['capture_warmup_seconds']
                )
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "effective 时长或预采集时长无效"
                ) from error
            if (
                not math.isfinite(effective_duration)
                or effective_duration <= 0
                or not math.isclose(
                    effective_duration,
                    effective_end - effective_start,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                or not math.isfinite(capture_warmup_seconds)
                or capture_warmup_seconds < 0
            ):
                raise RuntimeError(
                    "effective 时长或预采集时长无效"
                )
            try:
                stored_duration = float(episode_json['duration'])
                result_duration = float(episode_info['duration'])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "Episode 主时长字段无效"
                ) from error
            if (
                not math.isfinite(stored_duration)
                or not math.isclose(
                    stored_duration,
                    effective_duration,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    result_duration,
                    effective_duration,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
            ):
                raise RuntimeError(
                    "Episode duration 未按正式采集有效区间保存"
                )
            if 'capture_duration' in episode_json:
                try:
                    capture_duration = float(
                        episode_json['capture_duration']
                    )
                    result_capture_duration = float(
                        episode_info['capture_duration']
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        "Episode capture_duration 字段无效"
                    ) from error
                if (
                    not math.isfinite(capture_duration)
                    or capture_duration < effective_duration
                    or not math.isclose(
                        result_capture_duration,
                        capture_duration,
                        rel_tol=1e-12,
                        abs_tol=1e-9,
                    )
                ):
                    raise RuntimeError(
                        "Episode capture_duration 与保存结果不一致"
                    )
            if (
                episode_json['effective_interval_policy']
                != 'timestamp_window_with_context_carry_forward'
            ):
                raise RuntimeError(
                    "effective interval policy 无效"
                )

            context_topic_list = episode_json[
                'effective_context_topics'
            ]
            if (
                not isinstance(context_topic_list, list)
                or any(
                    not isinstance(topic_name, str)
                    or not topic_name
                    for topic_name in context_topic_list
                )
                or len(set(context_topic_list))
                != len(context_topic_list)
            ):
                raise RuntimeError(
                    "effective context topic 字段格式无效"
                )
            effective_context_topics = set(context_topic_list)
            unknown_context_topics = sorted(
                effective_context_topics - set(raw_topics)
            )
            if unknown_context_topics:
                raise RuntimeError(
                    "effective context topic 不在 raw_topics 中："
                    f"{unknown_context_topics}"
                )
            for field_name in effective_field_names:
                if episode_info.get(field_name) != episode_json[field_name]:
                    raise RuntimeError(
                        "保存结果与 Episode JSON 的 effective "
                        f"字段不一致：{field_name}"
                    )
        elif has_effective_context or any(
            field_name in episode_json
            for field_name in effective_companion_fields
        ):
            raise RuntimeError(
                "没有 effective 时间区间时不能保存关联字段"
            )

        episode_root = episode_dir.resolve()
        total_records = 0
        reference_topic = episode_json.get('reference_topic')
        if reference_topic not in raw_topics:
            raise RuntimeError("raw topic 参考流不存在")
        for topic_name, topic_info in raw_topics.items():
            topic_path = (
                episode_dir / Path(topic_info['path'])
            ).resolve()
            if episode_root not in topic_path.parents:
                raise RuntimeError(f"{topic_name} 文件路径越界")
            record_count = int(topic_info.get('record_count', 0))
            total_records += record_count
            source_count_total = sum(
                int(count)
                for count in topic_info.get(
                    'timestamp_source_counts',
                    {},
                ).values()
            )
            try:
                first_timestamp = float(
                    topic_info.get('first_timestamp')
                )
                last_timestamp = float(
                    topic_info.get('last_timestamp')
                )
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"{topic_name} first/last 时间戳无效"
                ) from error
            if (
                record_count <= 0
                or source_count_total != record_count
                or int(topic_info.get('nonfinite_timestamps', -1)) != 0
                or not math.isfinite(first_timestamp)
                or not math.isfinite(last_timestamp)
                or last_timestamp < first_timestamp
                or not topic_path.is_file()
                or topic_path.stat().st_size
                != int(topic_info.get('size_bytes', -1))
            ):
                raise RuntimeError(
                    f"{topic_name} 原始流未正确保存：{topic_path}"
                )
            if effective_start is not None:
                if first_timestamp > effective_start:
                    raise RuntimeError(
                        f"{topic_name} 未覆盖 effective start："
                        f"{first_timestamp} > {effective_start}"
                    )
                if (
                    topic_name not in effective_context_topics
                    and last_timestamp < effective_end
                ):
                    raise RuntimeError(
                        f"{topic_name} 未覆盖 effective end："
                        f"{last_timestamp} < {effective_end}"
                    )

        if effective_start is not None:
            reference_first_timestamp = float(
                raw_topics[reference_topic]['first_timestamp']
            )
            expected_capture_warmup = max(
                0.0,
                effective_start - reference_first_timestamp,
            )
            if not math.isclose(
                capture_warmup_seconds,
                expected_capture_warmup,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "capture_warmup_seconds 与参考流首时间戳不一致"
                )

        if (
            total_records != int(episode_info['total_records'])
            or total_records != int(episode_json.get('total_records', -1))
        ):
            raise RuntimeError("raw topic 总记录数校验失败")
        pending_files = [
            path
            for path in (episode_dir / 'raw_topics').iterdir()
            if path.name.endswith('.pending.pklseq')
        ]
        if pending_files:
            raise RuntimeError(f"仍有未封存的 raw topic 文件：{pending_files}")
        return

    expected_frames = int(episode_info['num_frames'])
    saved_frames = episode_json.get('frames', [])
    if (
        episode_json.get('num_frames') != expected_frames
        or len(saved_frames) != expected_frames
    ):
        raise RuntimeError(
            "JSON 帧数校验失败："
            f"预期 {expected_frames} 帧，实际 {len(saved_frames)} 帧"
        )

    video_files = episode_json.get('video_files', {})
    expected_cameras = {
        'head_camera',
        'left_arm_camera',
        'right_arm_camera',
    }
    if set(video_files) != expected_cameras:
        raise RuntimeError(
            "三路视频文件不完整："
            f"预期 {sorted(expected_cameras)}，实际 {sorted(video_files)}"
        )

    for camera_name, filename in video_files.items():
        video_path = episode_dir / filename
        if not video_path.is_file() or video_path.stat().st_size < 1024:
            raise RuntimeError(
                f"{camera_name} 视频未正确保存：{video_path}"
            )


def create_collection_config_for_quanta_x1() -> CollectionConfig:
    # Do not set SDK mode
    collection_config = CollectionConfig()

    collection_config.slave_joint_names = [
        'left_arm_joint_states',
        'right_arm_joint_states',
        'lift_joint_states',
        'left_gripper_joint_states',
        'right_gripper_joint_states',
        'head_joint_states'
    ]
    # action_names will be generated automatically from joint_names, or can be specified manually
    collection_config.enable_head_rgb_stream = True # Collect head RGB video stream
    collection_config.enable_left_arm_rgb_stream = True # Collect left arm RGB video stream
    collection_config.enable_right_arm_rgb_stream = True # Collect right arm RGB video stream
    collection_config.enable_left_arm_end_pose = True # Collect left arm end pose
    collection_config.enable_right_arm_end_pose = True # Collect right arm end pose
    collection_config.enable_odometry = True # Collect odometry data
    collection_config.enable_master_arm_data = True # Collect master arm joint states and end pose
    collection_config.enable_wrench_ext_world = True # Collect wrist external force
    collection_config.enable_wrench_ext_local = True # Collect wrist local force

    return collection_config

def create_collection_config_for_quanta_x2() -> CollectionConfig:
    collection_config = CollectionConfig()
    # Pi0.5 state: measured dual-arm EEF poses plus Revo2 joint data.
    # The Revo2 control bridge owns both RS485 ports and publishes versioned
    # primitive JSON; this process never opens a hand serial device directly.
    collection_config.slave_joint_names = []
    collection_config.enable_head_rgb_stream = True # Collect head RGB video stream
    collection_config.enable_left_arm_rgb_stream = True # Left wrist fisheye
    collection_config.enable_right_arm_rgb_stream = True # Right wrist fisheye
    collection_config.camera_stream_profiles = {
        'head_rgb_stream': {
            'capture_codec': 'H265',
            'source_width': E6_SOURCE_WIDTH,
            'source_height': E6_SOURCE_HEIGHT,
            'width': E6_RIGHT_WIDTH,
            'height': E6_SOURCE_HEIGHT,
            'view_crop': [
                E6_RIGHT_WIDTH,
                0,
                E6_RIGHT_WIDTH,
                E6_SOURCE_HEIGHT,
            ],
            'fps': E6_STREAM_FPS,
            'fps_tolerance': 8,
        },
        'left_arm_rgb_stream': {
            'capture_codec': 'MJPEG',
            'width': 640,
            'height': 480,
            'fps': 30,
            'fps_tolerance': 3,
        },
        'right_arm_rgb_stream': {
            'capture_codec': 'MJPEG',
            'width': 640,
            'height': 480,
            'fps': 30,
            'fps_tolerance': 3,
        },
    }
    collection_config.enable_left_arm_end_pose = True # Collect left arm end pose
    collection_config.enable_right_arm_end_pose = True # Collect right arm end pose
    collection_config.enable_revo2_hands = True
    # Pi0.5 action: only the six-axis Revo2 commands successfully sent by the
    # bridge. Arm pose commands are intentionally excluded from this task.
    collection_config.enable_vr_action_commands = True

    # No master arm joint states and end pose
    # No wrist external and local force
    return collection_config

def create_collection_config_for_desktop() -> CollectionConfig:
    collection_config = CollectionConfig()
    # Collect slave arm joint states (Desktop model, only left and right arms)
    collection_config.slave_joint_names = [
        'left_arm_joint_states',
        'right_arm_joint_states',
        'left_gripper_joint_states',
        'right_gripper_joint_states'
    ]
    # action_names will be generated automatically from joint_names, or can be specified manually
    collection_config.enable_head_rgb_stream = True # Collect head RGB video stream
    collection_config.enable_left_arm_rgb_stream = True # Collect left arm RGB video stream
    collection_config.enable_right_arm_rgb_stream = True # Collect right arm RGB video stream
    collection_config.enable_left_arm_end_pose = True # Collect left arm end pose
    collection_config.enable_right_arm_end_pose = True # Collect right arm end pose
    return collection_config


@exclusive_collection_lock
def main(
    server: Annotated[str, typer.Option(help="机器人服务地址，例如 localhost:50051")] = "localhost:50051",
    keep_raw_data: Annotated[bool, typer.Option(help="同时将对齐前的原始数据流保存到 episode_xxxx/raw_data/")] = False,
    raw_topic_storage: Annotated[bool, typer.Option(help="每个 topic 按自己的时间戳独立保存；相机保留源 JPEG/H265 压缩 bytes，不解码、不对齐、不转码")] = True,
    cooldown_seconds: Annotated[int, typer.Option(help="下一条数据开始采集前的等待秒数")] = 30,
):
    if cooldown_seconds < 0:
        raise typer.BadParameter("cooldown-seconds 不能为负数")

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Connect to robot
    print(f"正在连接机器人 {server}...")
    robot = connect(f"x2://{server}")
    print("✓ 机器人连接成功")

    robot_model = robot.get_robot_model()
    if robot_model == "quanta_x1":
        collection_config = create_collection_config_for_quanta_x1()
    elif robot_model == "quanta_x2":
        collection_config = create_collection_config_for_quanta_x2()
    elif robot_model == "desktop":
        collection_config = create_collection_config_for_desktop()
    else:
        raise ValueError(
            f"不支持的机器人型号：{robot_model}；"
            "支持的型号：quanta_x1、quanta_x2、desktop"
        )


    # Create data collector - optimize configuration to improve performance
    collector = E6RightHeadDataCollector(
        robot=robot,
        # Always use the dataset next to this script, regardless of cwd.
        output_dir=str(DEFAULT_OUTPUT_DIR),
        target_hz=30,                      # Target frequency (after downsampling)
        collection_config=collection_config,
        image_quality=95,                  # JPEG quality
        downsample_joint_states=True,       # Whether to downsample joint states, recommended to enable (eg: 500Hz -> target_hz=60Hz)
        use_video_storage=not raw_topic_storage,
        keep_raw_data=keep_raw_data,  # Save raw pre-alignment streams when --keep-raw-data is passed
        raw_topic_storage=raw_topic_storage,
    )

    print("\n" + "="*60)
    print("数据采集器已就绪")
    print("="*60)
    print(f"输出目录：{collector.output_dir}")
    if collector.raw_topic_storage:
        print("存储模式：各 topic 独立原始流")
        print("时间戳：每个 topic 使用自己的消息时间戳或接收时间")
        print(
            "RGB 相机存储：E6 双目 H265 access unit（逻辑视图为右目）；"
            "双腕 MJPEG 原始 JPEG bytes 直存"
        )
        print(
            "E6 右侧源："
            f"{E6_STREAM_PACKAGE}，ADB {E6_SERIAL}，"
            f"headset tcp/{E6_REMOTE_STREAM_PORT}"
        )
        if collector.camera_stream_profiles:
            print("腕部鱼眼相机预检：MJPEG 640x480 @ 30Hz（左右腕均需通过）")
        if collector.collection_config.enable_revo2_hands:
            print("Revo2：双手 6 轴关节状态 + 实际下发动作（不采集触觉）")
        print("时间对齐：关闭")
    else:
        print(f"目标频率：{collector.target_hz} Hz")
        print(f"图像存储：{'MP4 视频' if collector.use_video_storage else 'JPG 图片'}")
        print(f"保留原始数据：{'是（episode_xxxx/raw_data/）' if collector.keep_raw_data else '否'}")
    print("\n提示：")
    print("  - start_recording() 会自动启动全部数据采集线程")
    print("  - stop_recording() 会自动停止全部线程并保存数据")
    print("  - 随时可以按 Ctrl+C 中断程序")
    print("="*60 + "\n")

    try:
        # Record multiple episodes
        episode_index = 0
        while True:
            # Show current episode number to record (based on number of existing episodes)
            current_episode_num = collector.episode_count
            print(f"\n{'='*60}")
            print(f"准备采集第 {current_episode_num} 条数据")
            print(f"{'='*60}")

            # Ask for task name
            task_name = input(
                "请输入任务名称（直接按回车使用默认名称）："
            ).strip()
            if not task_name:
                task_name = (
                    "Put the object on the box, then take it down."
                )

            print(f"\n任务名称：{task_name}")
            print("请准备好机器人，按回车开始采集...")
            input()

            # Start recording (will start all data collection threads automatically)
            print("\n" + "="*60)
            print("开始采集...")
            print("="*60)
            collector.start_recording(task=task_name)

            print("\n✓ 所有数据采集线程均已启动")
            print("正在采集数据...")
            print("提示：请执行任务，完成后选择是否保存本条数据：")
            print("      - 直接按回车或输入 y：停止并保存")
            print("      - 输入 n：停止并丢弃")
            print("      - 也可以按 Ctrl+C 中断本次采集\n")

            # Print statistics every second
            recording_interrupted = False
            save_recording = True
            while True:
                try:
                    if collector.has_collection_errors():
                        print(
                            "\n检测到采集线程异常，正在停止并拒绝保存本条数据..."
                        )
                        collector.stop_recording()

                    # Non-blocking input detection
                    import select
                    if select.select([sys.stdin], [], [], 1)[0]:
                        save_answer = input().strip().lower()
                        if save_answer in ('n', 'no'):
                            save_recording = False
                        elif save_answer not in ('', 'y', 'yes'):
                            print(
                                f"未识别输入 {save_answer!r}，"
                                "为避免误删，本条将按默认设置保存。"
                            )
                        break

                    # Print statistics
                    collector.print_stats()
                except KeyboardInterrupt:
                    print("\n\n收到中断信号，正在停止本次采集...")
                    recording_interrupted = True
                    break

            # 同步停止、保存并校验；完成前绝不进入下一条数据。
            episode_info = None
            episode_discarded = False
            if collector.is_recording:
                if save_recording:
                    print("\n" + "="*60)
                    print("本条采集已结束，正在同步保存数据，请勿开始下一次动作...")
                    if collector.raw_topic_storage:
                        print(
                            "正在封存各 topic 原始流和 JSON；"
                            "不会解码、对齐或转码视频。"
                        )
                    else:
                        print(
                            "正在生成三路视频和 JSON，"
                            "保存完成前不会进入下一条数据。"
                        )
                    print("="*60)
                    save_started_at = time.monotonic()
                    episode_info = collector.stop_recording()
                    if episode_info:
                        verify_saved_episode(episode_info)
                        save_elapsed = time.monotonic() - save_started_at
                        print(f"✓ 数据保存及完整性校验完成，用时 {save_elapsed:.2f} 秒")
                else:
                    print("\n本条选择不保存，正在停止线程并清理临时数据...")
                    episode_discarded = collector.discard_recording()

            if episode_info:
                episode_id = episode_info['episode_id']
                print(f"\n{'='*60}")
                print(f"✓ 第 {episode_id} 条数据采集并保存完成！")
                print(f"{'='*60}")
                print(f"  - 数据编号：{episode_id}")
                print(f"  - 任务名称：{episode_info.get('task', task_name)}")
                if episode_info.get('storage_format') == 'raw_topics_pickle_v1':
                    print(f"  - topic 数量：{episode_info['num_topics']}")
                    print(f"  - 总记录数：{episode_info['total_records']}")
                    print(
                        "  - 参考流（仅统计，不用于对齐）："
                        f"{episode_info['reference_topic']} / "
                        f"{episode_info['reference_sample_count']} 条"
                    )
                else:
                    print(f"  - 帧数：{episode_info['num_frames']}")
                print(f"  - 时长：{episode_info['duration']:.2f} 秒")
                print(f"  - 保存路径：{episode_info['episode_dir']}")
            elif episode_discarded:
                print(f"\n{'='*60}")
                print("✓ 本条数据已丢弃，没有写入数据集")
                print(f"下一条数据编号仍为 {collector.episode_count}")
                print(f"{'='*60}")
            else:
                current_episode_num = collector.episode_count
                print(f"\n⚠️  第 {current_episode_num} 条数据保存失败，不会进入下一条")
                break

            if recording_interrupted:
                print("\n本次采集已中断")
                break

            # Ask whether to continue recording
            print("\n" + "-"*60)
            while True:
                continue_recording = input(
                    "是否继续采集下一条数据？（y/n，默认 y）："
                ).strip().lower()
                if continue_recording in ('', 'y', 'yes'):
                    break
                if continue_recording in ('n', 'no'):
                    print("停止采集")
                    return
                print("请输入 y 或 n（直接按回车表示 y）")

            if cooldown_seconds > 0:
                print(
                    f"\n等待 {cooldown_seconds} 秒后才能采集下一条数据..."
                )
                for remaining in range(cooldown_seconds, 0, -1):
                    print(
                        f"\r距离下一条数据可采集还有 {remaining:2d} 秒",
                        end="",
                        flush=True,
                    )
                    time.sleep(1)
                print("\r等待结束，可以采集下一条数据。              ")

            episode_index += 1

    except KeyboardInterrupt:
        print("\n\n收到中断信号，正在停止...")
    except Exception as e:
        print(f"\n错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure stopping recording (if still recording)
        if collector.is_recording:
            print("\n正在停止采集...")
            collector.stop_recording()

        print("\n" + "="*60)
        print("数据采集结束！")
        print("="*60)
        print(f"共采集 {collector.episode_count} 条数据")
        print(f"数据保存目录：{collector.output_dir}")
        if collector.raw_topic_storage:
            print("\n当前为 raw topic 格式：")
            print("  - episode_xxxx/raw_topics/*.pklseq")
            print("  - 每个文件是连续 pickle record，保留该 topic 自己的时间戳")
            print("  - E6 record 为 (timestamp_seconds, h265_access_unit_bytes)")
            print("  - 双腕 record 为 (timestamp_seconds, jpeg_bytes)")
            print("  - Revo2 关节和六轴动作均为版本化 primitive dict（无触觉流）")
            print("请使用你的转换程序按各 topic 独立时间轴读取。")
        else:
            print(f"\n转换为 LeRobot 格式可执行：")
            print(f"python3 tools/convert_to_lerobot.py \\")
            print(f"    --input-dir {collector.output_dir} \\")
            print(f"    --output-dir ./lerobot_data \\")
            print(f"    --repo-id my_robot/dataset \\")
            print(f"    --robot-type {robot.get_robot_model()} \\")
            print(f"    --use-videos")


if __name__ == "__main__":
    typer.run(main)
