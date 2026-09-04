#!/usr/bin/env python3
"""Lightweight LAN control panel for ``sol_data_collection.py``.

The production path deliberately uses only Python's standard-library HTTP
server.  The robot virtual environment does not need Flask/FastAPI/Uvicorn,
and the collector remains in the same process so there is exactly one owner
of the SDK connection and episode lifecycle.
"""

from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import hmac
import json
import math
import os
import pickle
import secrets
import shutil
import signal
import sys
import threading
import time
import traceback
from contextlib import ExitStack
from dataclasses import dataclass
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
REFERENCE_TOPIC = "head_rgb_stream"
DEFAULT_TASK = "Put the object on the box, then take it down."

TOPICS: dict[str, dict[str, str]] = {
    "head_rgb_stream": {
        "label": "E6 头环右目 RGB（H.265 · 60 Hz）",
        "group": "camera",
    },
    "left_arm_rgb_stream": {
        "label": "左腕鱼眼（MJPEG 640×480 · 30 Hz）",
        "group": "camera",
    },
    "right_arm_rgb_stream": {
        "label": "右腕鱼眼（MJPEG 640×480 · 30 Hz）",
        "group": "camera",
    },
    "left_arm_end_pose": {"label": "左臂实测末端", "group": "state"},
    "right_arm_end_pose": {"label": "右臂实测末端", "group": "state"},
    "left_revo2_joint_states": {
        "label": "左 Revo2 关节反馈",
        "group": "state",
    },
    "right_revo2_joint_states": {
        "label": "右 Revo2 关节反馈",
        "group": "state",
    },
    "vr_left_revo2_joint_commands": {
        "label": "左 Revo2 Command",
        "group": "action",
    },
    "vr_right_revo2_joint_commands": {
        "label": "右 Revo2 Command",
        "group": "action",
    },
}

TOPIC_COUNT = len(TOPICS)

CAMERA_TOPICS = tuple(
    name for name, metadata in TOPICS.items() if metadata["group"] == "camera"
)
CAMERA_PREVIEW_FPS = 15.0
CAMERA_PREVIEW_INTERVAL_SECONDS = 1.0 / CAMERA_PREVIEW_FPS
CAMERA_STREAM_IDLE_TIMEOUT_SECONDS = 5.0
CAMERA_STREAM_BOUNDARY = b"xr-camera-frame"
ACTION_TOPICS = tuple(
    name for name, metadata in TOPICS.items() if metadata["group"] == "action"
)
EVENT_DRIVEN_TOPICS = frozenset(ACTION_TOPICS)
CONTINUOUS_TOPICS = tuple(
    name for name in TOPICS if name not in EVENT_DRIVEN_TOPICS
)
CONTINUOUS_TOPIC_MAX_GAP_SECONDS = 2.0
EFFECTIVE_START_SYNC_TIMEOUT_SECONDS = 2.0
EFFECTIVE_START_SYNC_POLL_SECONDS = 0.01
TOPIC_PREFLIGHT_POLL_SECONDS = 0.2
TOPIC_PREFLIGHT_STABLE_POLLS = 3

STATE_LABELS = {
    "initializing": "正在连接",
    "ready": "数采已关闭",
    "starting": "正在预检",
    "armed": "监测正常",
    "canceling_start": "正在关闭数采",
    "starting_recording": "正在开始采集",
    "recording": "正在采集",
    "stopping": "正在停止",
    "review": "等待保存决定",
    "saving": "正在保存",
    "discarding": "正在删除",
    "cooldown": "冷却等待",
    "closing": "正在关闭数采",
    "error": "发生错误",
}


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def json_safe(value: Any) -> Any:
    """Convert common SDK/numpy-like values into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


class SessionStore:
    """Short-lived anonymous sessions used only for CSRF protection."""

    SESSION_TTL_SECONDS = 12 * 3600

    def __init__(self):
        self.lock = threading.Lock()
        self.sessions: dict[str, dict[str, Any]] = {}

    def create_session(self, remote_ip: str) -> tuple[str, str]:
        now = time.time()
        session_id = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(24)
        with self.lock:
            self.sessions = {
                key: value
                for key, value in self.sessions.items()
                if float(value["expires_at"]) > now
            }
            self.sessions[session_id] = {
                "csrf": csrf_token,
                "expires_at": now + self.SESSION_TTL_SECONDS,
                "remote_ip": remote_ip,
            }
        return session_id, csrf_token

    def get_session(self, session_id: str | None) -> dict[str, Any] | None:
        if not session_id:
            return None
        now = time.time()
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return None
            if float(session["expires_at"]) <= now:
                self.sessions.pop(session_id, None)
                return None
            session["expires_at"] = now + self.SESSION_TTL_SECONDS
            return dict(session)


class _PreflightGatedFile:
    """Discard pickle writes until the user confirms the effective start."""

    def __init__(self, file_handle, enabled):
        self._file_handle = file_handle
        self._enabled = enabled

    @property
    def closed(self):
        return self._file_handle.closed

    def write(self, data):
        if self._enabled():
            return self._file_handle.write(data)
        # BinaryIO.write() reports consumed bytes. pickle can therefore finish
        # normally while preflight keeps only counters and latest previews.
        return len(data)

    def flush(self):
        return self._file_handle.flush()

    def fileno(self):
        return self._file_handle.fileno()

    def close(self):
        return self._file_handle.close()

    def reset(self):
        self._file_handle.flush()
        self._file_handle.seek(0)
        self._file_handle.truncate(0)

    def __getattr__(self, name):
        return getattr(self._file_handle, name)


class ReviewableCollectorMixin:
    """Adds live telemetry and a stop-before-save review state.

    This mixin is combined with the existing DataCollector at runtime.  It
    intentionally does not modify any producer thread implementation.
    """

    def __init__(self, *args, verify_callback: Callable | None = None, **kwargs):
        self._web_live_lock = threading.Lock()
        self._web_camera_condition = threading.Condition(self._web_live_lock)
        self._web_receive_times: dict[str, float] = {}
        self._web_receive_monotonic: dict[str, float] = {}
        self._web_max_receive_gap: dict[str, float] = {}
        self._web_latest_values: dict[str, Any] = {}
        self._web_camera_frames: dict[str, tuple[int, float, bytes]] = {}
        self._web_camera_sequences: dict[str, int] = {}
        self._web_camera_preview_monotonic: dict[str, float] = {}
        self._web_episode_stop_monotonic: float | None = None
        self._web_effective_start_timestamp: float | None = None
        self._web_effective_start_monotonic: float | None = None
        self._web_records_after_effective: dict[str, int] = {}
        self._web_review_pending = False
        self._web_review_errors: list[dict[str, Any]] = []
        self._web_verify_callback = verify_callback
        self._web_persist_enabled = False
        super().__init__(*args, **kwargs)
        if not self.raw_topic_storage:
            raise ValueError("网页采集器只支持 raw_topic_storage=True")

    def start_recording(self, task: str = "default_task"):
        if self._web_review_pending:
            raise RuntimeError("上一条数据仍在等待保存或丢弃")
        self._web_persist_enabled = False
        with self._web_live_lock:
            self._web_receive_times = {}
            self._web_receive_monotonic = {}
            self._web_max_receive_gap = {}
            self._web_latest_values = {}
            self._web_camera_frames = {}
            self._web_camera_sequences = {}
            self._web_camera_preview_monotonic = {}
            self._web_camera_condition.notify_all()
            self._web_episode_stop_monotonic = None
            self._web_effective_start_timestamp = None
            self._web_effective_start_monotonic = None
            self._web_records_after_effective = {}
        self._web_review_errors = []
        return super().start_recording(task=task)

    def _create_temp_files(self):
        """Use zero-growth sinks during preflight, then reuse the same files."""
        super()._create_temp_files()
        if not self.raw_topic_storage:
            return
        enabled = lambda: bool(self._web_persist_enabled)
        for mapping in (self.image_temp_files, self.sensor_temp_files):
            for topic_name, file_handle in list(mapping.items()):
                mapping[topic_name] = _PreflightGatedFile(
                    file_handle,
                    enabled,
                )

    def _record_raw_topic_sample(
        self,
        topic_name,
        timestamp,
        timestamp_source=None,
    ):
        super()._record_raw_topic_sample(
            topic_name,
            timestamp,
            timestamp_source,
        )
        now_wall = time.time()
        now_monotonic = time.monotonic()
        with self._web_live_lock:
            previous = self._web_receive_monotonic.get(topic_name)
            if previous is not None:
                gap = max(0.0, now_monotonic - previous)
                self._web_max_receive_gap[topic_name] = max(
                    gap,
                    self._web_max_receive_gap.get(topic_name, 0.0),
                )
            self._web_receive_times[topic_name] = now_wall
            self._web_receive_monotonic[topic_name] = now_monotonic
            if (
                self._web_effective_start_monotonic is not None
                and now_monotonic >= self._web_effective_start_monotonic
            ):
                self._web_records_after_effective[topic_name] = (
                    self._web_records_after_effective.get(topic_name, 0) + 1
                )

    def mark_effective_start(self) -> float:
        """Mark the first timestamp that belongs to the usable task window."""
        if not self.is_recording:
            raise RuntimeError("正式采集尚未启动，不能记录有效起点")
        effective_start_timestamp = time.time()
        effective_start_monotonic = time.monotonic()

        with self._web_live_lock:
            if self._web_effective_start_timestamp is not None:
                raise RuntimeError("本条数据已经记录过有效起点")
        action_context: dict[str, Any] = {}

        topic_locks = [
            (f"camera:{name}", lock)
            for name, lock in self.image_file_locks.items()
        ] + [
            (f"sensor:{name}", lock)
            for name, lock in self.sensor_file_locks.items()
        ]
        with ExitStack() as lock_stack:
            for _, topic_lock in sorted(topic_locks, key=lambda item: item[0]):
                lock_stack.enter_context(topic_lock)

            # Producers are now paused at their per-topic file locks. Remove
            # any accidental bytes, reset preflight-only counters, and switch
            # every stream to persistent writes as one transition.
            for mapping in (self.image_temp_files, self.sensor_temp_files):
                for file_handle in mapping.values():
                    if isinstance(file_handle, _PreflightGatedFile):
                        file_handle.reset()
            with self._raw_topic_stats_lock:
                self._raw_topic_stats = {}
            self._web_persist_enabled = True

            with self._web_live_lock:
                action_context = {
                    topic_name: copy.deepcopy(
                        self._web_latest_values.get(topic_name)
                    )
                    for topic_name in ACTION_TOPICS
                }
                self._web_effective_start_timestamp = (
                    effective_start_timestamp
                )
                self._web_effective_start_monotonic = (
                    effective_start_monotonic
                )
                self._web_records_after_effective = {}
                self._web_receive_times = {}
                self._web_receive_monotonic = {}
                # Readiness warm-up gaps do not belong to the task interval.
                self._web_max_receive_gap = {}

            # Event-driven commands may not change immediately after the
            # button press. Persist their latest preflight values exactly at
            # the effective start so the saved Episode has explicit context.
            for topic_name in ACTION_TOPICS:
                payload = action_context.get(topic_name)
                output = self.sensor_temp_files.get(topic_name)
                if payload is None or output is None or output.closed:
                    continue
                pickle.dump((effective_start_timestamp, payload), output)
                self._record_raw_topic_sample(
                    topic_name,
                    effective_start_timestamp,
                    "effective_start_carry_forward",
                )

        # Producers can only persist their first sample after the transition
        # locks above are released. Use the first common sampled point as the
        # effective boundary so normal scheduling latency cannot invalidate
        # an otherwise complete Episode during save.
        expected_topics = self._expected_web_topics()
        deadline = time.monotonic() + EFFECTIVE_START_SYNC_TIMEOUT_SECONDS
        first_timestamps: dict[str, float] = {}
        while True:
            if not self.is_recording:
                raise RuntimeError("有效起点同步期间采集已停止")
            self._raise_collector_errors()
            with self._raw_topic_stats_lock:
                raw_stats = copy.deepcopy(self._raw_topic_stats)
            first_timestamps = {}
            for topic_name in expected_topics:
                statistics = raw_stats.get(topic_name, {})
                if int(statistics.get("record_count", 0)) <= 0:
                    continue
                try:
                    first_timestamp = float(
                        statistics.get("first_timestamp")
                    )
                except (TypeError, ValueError):
                    continue
                if math.isfinite(first_timestamp):
                    first_timestamps[topic_name] = first_timestamp
            if len(first_timestamps) == len(expected_topics):
                break
            if time.monotonic() >= deadline:
                missing = [
                    name
                    for name in expected_topics
                    if name not in first_timestamps
                ]
                raise RuntimeError(
                    "有效起点同步失败，以下数据流未及时写入首条样本："
                    + ", ".join(missing)
                )
            time.sleep(EFFECTIVE_START_SYNC_POLL_SECONDS)

        synchronized_start_timestamp = max(
            [effective_start_timestamp, *first_timestamps.values()]
        )
        with self._web_live_lock:
            self._web_effective_start_timestamp = (
                synchronized_start_timestamp
            )
        return synchronized_start_timestamp

    @staticmethod
    def _peek_message_timestamp(message: Any) -> float:
        try:
            stamp = message.header.stamp
            seconds = int(stamp.sec)
            nanoseconds = int(stamp.nanosec)
            if seconds or nanoseconds:
                return seconds + nanoseconds / 1e9
        except Exception:
            pass
        return time.time()

    def _collect_camera_stream(self, camera_name, stream_func):
        def observed_stream(timeout=None):
            stream = stream_func(timeout=timeout)
            for frame_message in stream:
                now_monotonic = time.monotonic()
                last_preview = self._web_camera_preview_monotonic.get(
                    camera_name,
                    0.0,
                )
                raw_data = getattr(frame_message, "data", None)
                preview_data = getattr(frame_message, "preview_jpeg", None)
                if (
                    preview_data is None
                    and isinstance(raw_data, (bytes, bytearray, memoryview))
                ):
                    candidate = bytes(raw_data)
                    if (
                        len(candidate) >= 4
                        and candidate[:2] == b"\xff\xd8"
                        and candidate[-2:] == b"\xff\xd9"
                    ):
                        preview_data = candidate
                if (
                    now_monotonic - last_preview
                    >= CAMERA_PREVIEW_INTERVAL_SECONDS
                    and frame_message
                    and preview_data
                    and isinstance(
                        preview_data,
                        (bytes, bytearray, memoryview),
                    )
                ):
                    jpeg_bytes = bytes(preview_data)
                    timestamp = self._peek_message_timestamp(frame_message)
                    with self._web_live_lock:
                        sequence = (
                            self._web_camera_sequences.get(camera_name, 0) + 1
                        )
                        self._web_camera_sequences[camera_name] = sequence
                        self._web_camera_frames[camera_name] = (
                            sequence,
                            timestamp,
                            jpeg_bytes,
                        )
                        self._web_camera_preview_monotonic[
                            camera_name
                        ] = now_monotonic
                        self._web_camera_condition.notify_all()
                yield frame_message

        return super()._collect_camera_stream(camera_name, observed_stream)

    def _collect_pose_stream(self, queue_name, stream_func):
        def observed_stream(timeout=None):
            stream = stream_func(timeout=timeout)
            for message in stream:
                try:
                    pose = message.pose
                    value = {
                        "position": {
                            "x": float(pose.position.x),
                            "y": float(pose.position.y),
                            "z": float(pose.position.z),
                        },
                        "orientation": {
                            "x": float(pose.orientation.x),
                            "y": float(pose.orientation.y),
                            "z": float(pose.orientation.z),
                            "w": float(pose.orientation.w),
                        },
                    }
                    with self._web_live_lock:
                        self._web_latest_values[queue_name] = value
                except Exception:
                    pass
                yield message

        return super()._collect_pose_stream(queue_name, observed_stream)

    def _observe_ros_bridge_sample(
        self,
        sensor_name: str,
        timestamp: float,
        payload: Any,
    ):
        """Cache the latest ROS feedback/action supplied by DataCollector.

        The collector owns persistence and timestamp statistics.  This hook is
        intentionally observation-only so a web-preview conversion failure
        can never interrupt data collection.
        """
        parent_observer = getattr(
            super(),
            "_observe_ros_bridge_sample",
            None,
        )
        if callable(parent_observer):
            parent_observer(sensor_name, timestamp, payload)
        if sensor_name not in TOPICS:
            return
        try:
            latest_value = json_safe(payload)
            with self._web_live_lock:
                self._web_latest_values[sensor_name] = latest_value
        except Exception:
            # The preview is best-effort; raw recording must keep running.
            pass

    def _expected_web_topics(self) -> list[str]:
        derived_topic_names = set(self.slave_action_names)
        return list(self.camera_names) + [
            name
            for name in self.sensor_file_locks
            if name not in derived_topic_names
        ]

    def stop_for_review(self) -> dict[str, Any]:
        if self._web_review_pending:
            raise RuntimeError("当前已经处于待审核状态")
        if not self.is_recording:
            raise RuntimeError("当前没有正在采集的数据")

        self.current_episode_stop_time = time.time()
        with self._web_live_lock:
            self._web_episode_stop_monotonic = time.monotonic()
        self.is_recording = False
        if self._episode_stop_event is not None:
            self._episode_stop_event.set()

        self._stop_camera_monitor()
        self._wait_for_collection_threads()
        self._flush_all_temp_files()
        self._close_and_fsync_temp_files()
        with self._collector_errors_lock:
            self._web_review_errors = copy.deepcopy(self._collector_errors)
        self._web_review_pending = True
        self._print_resource_usage("网页审核冻结")
        return self.pending_validation()

    def pending_validation(self) -> dict[str, Any]:
        expected_topics = self._expected_web_topics()
        with self._raw_topic_stats_lock:
            raw_stats = copy.deepcopy(self._raw_topic_stats)
        with self._web_live_lock:
            receive_monotonic = dict(self._web_receive_monotonic)
            max_receive_gap = dict(self._web_max_receive_gap)
            stop_monotonic = self._web_episode_stop_monotonic
            effective_start = self._web_effective_start_timestamp
            records_after_effective = dict(
                self._web_records_after_effective
            )

        missing_topics = []
        invalid_topics = []
        interrupted_topics = []
        effective_missing_topics = []
        start_boundary_topics = []
        tail_gap_seconds = {}
        for topic_name in expected_topics:
            statistics = raw_stats.get(topic_name, {})
            record_count = int(statistics.get("record_count", 0))
            if record_count <= 0:
                missing_topics.append(topic_name)
            if (
                int(statistics.get("nonfinite_timestamps", 0)) > 0
                or int(statistics.get("timestamp_regressions", 0)) > 0
            ):
                invalid_topics.append(topic_name)
            if effective_start is not None and record_count > 0:
                try:
                    first_timestamp = float(
                        statistics.get("first_timestamp")
                    )
                except (TypeError, ValueError):
                    first_timestamp = None
                if (
                    first_timestamp is not None
                    and math.isfinite(first_timestamp)
                    and first_timestamp > effective_start
                ):
                    start_boundary_topics.append(topic_name)
            if topic_name in CONTINUOUS_TOPICS and record_count > 0:
                if records_after_effective.get(topic_name, 0) <= 0:
                    effective_missing_topics.append(topic_name)
                last_receive = receive_monotonic.get(topic_name)
                tail_gap = (
                    max(0.0, stop_monotonic - last_receive)
                    if stop_monotonic is not None
                    and last_receive is not None
                    else float("inf")
                )
                tail_gap_seconds[topic_name] = tail_gap
                if (
                    tail_gap > CONTINUOUS_TOPIC_MAX_GAP_SECONDS
                    or max_receive_gap.get(topic_name, 0.0)
                    > CONTINUOUS_TOPIC_MAX_GAP_SECONDS
                ):
                    interrupted_topics.append(topic_name)

        effective_end_candidates = []
        for topic_name in CONTINUOUS_TOPICS:
            if topic_name not in expected_topics:
                continue
            last_timestamp = raw_stats.get(topic_name, {}).get(
                "last_timestamp"
            )
            try:
                candidate = float(last_timestamp)
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate):
                effective_end_candidates.append(candidate)
        effective_end = (
            min(effective_end_candidates)
            if len(effective_end_candidates)
            == len(set(CONTINUOUS_TOPICS).intersection(expected_topics))
            else None
        )
        effective_interval_valid = bool(
            effective_start is not None
            and effective_end is not None
            and effective_end > effective_start
        )
        valid = not (
            missing_topics
            or invalid_topics
            or interrupted_topics
            or effective_missing_topics
            or start_boundary_topics
            or self._web_review_errors
            or not effective_interval_valid
        )
        return {
            "valid": valid,
            "missing_topics": missing_topics,
            "invalid_topics": invalid_topics,
            "interrupted_topics": interrupted_topics,
            "effective_missing_topics": effective_missing_topics,
            "start_boundary_topics": start_boundary_topics,
            "tail_gap_seconds": tail_gap_seconds,
            "max_receive_gap_seconds": {
                name: max_receive_gap.get(name, 0.0)
                for name in CONTINUOUS_TOPICS
                if name in expected_topics
            },
            "collector_errors": copy.deepcopy(self._web_review_errors),
            "effective_start_timestamp": effective_start,
            "effective_end_timestamp": effective_end,
            "effective_duration": (
                effective_end - effective_start
                if effective_interval_valid
                else None
            ),
            "effective_interval_valid": effective_interval_valid,
            "effective_records": {
                name: records_after_effective.get(name, 0)
                for name in expected_topics
            },
        }

    def save_pending_recording(self) -> dict[str, Any]:
        if not self._web_review_pending:
            raise RuntimeError("没有等待保存的数据")
        validation = self.pending_validation()
        if not validation["valid"]:
            raise RuntimeError(
                "本条存在缺失或无效数据，已禁止保存；请丢弃后重新采集"
            )

        episode_info = self._save_raw_topic_episode(
            self.current_episode_task,
            effective_start_timestamp=validation[
                "effective_start_timestamp"
            ],
            effective_end_timestamp=validation[
                "effective_end_timestamp"
            ],
            effective_context_topics=sorted(EVENT_DRIVEN_TOPICS),
        )
        self.episode_count += 1
        try:
            if self._web_verify_callback is not None:
                self._web_verify_callback(episode_info)
        finally:
            # _save_raw_topic_episode() has already atomically committed the
            # episode.  Keep the in-memory id and lifecycle consistent even
            # if the post-commit verifier reports a problem.
            self._finish_pending_cleanup(remove_staging=False)
        return episode_info

    def discard_pending_recording(self) -> bool:
        if not self._web_review_pending:
            raise RuntimeError("没有等待丢弃的数据")
        alive_threads = [
            thread.name for thread in self.threads if thread.is_alive()
        ]
        if alive_threads:
            raise RuntimeError(f"仍有采集线程存活：{alive_threads}")
        self._finish_pending_cleanup(remove_staging=True)
        return True

    def abort_active_recording(self):
        """Discard any complete or partially initialized recording."""
        if self.is_recording:
            return self.discard_recording()
        if self._web_review_pending:
            return self.discard_pending_recording()

        # start_recording() can fail after creating files or starting only a
        # subset of producers.  This path is intentionally idempotent so a
        # failed readiness gate cannot leak threads or staging into the next
        # Episode.
        if self._episode_stop_event is not None:
            self._episode_stop_event.set()
        self._stop_camera_monitor()
        if self.threads:
            self._wait_for_collection_threads()
        self._cleanup_temp_files()
        self._cleanup_active_episode_staging()
        self._release_unused_memory()
        self.current_episode_task = None
        self.current_episode_start_time = None
        self.current_episode_stop_time = None
        self._web_review_pending = False
        self._web_review_errors = []
        with self._web_live_lock:
            self._web_episode_stop_monotonic = None
            self._web_effective_start_timestamp = None
            self._web_effective_start_monotonic = None
            self._web_records_after_effective = {}
        return True

    def _finish_pending_cleanup(self, remove_staging: bool):
        self._cleanup_temp_files()
        if remove_staging:
            self._cleanup_active_episode_staging()
        self._release_unused_memory()
        self.current_episode_task = None
        self.current_episode_start_time = None
        self.current_episode_stop_time = None
        self._web_review_pending = False
        self._web_review_errors = []
        with self._web_live_lock:
            self._web_episode_stop_monotonic = None
            self._web_effective_start_timestamp = None
            self._web_effective_start_monotonic = None
            self._web_records_after_effective = {}

    def shutdown_without_saving(self):
        if self.is_recording:
            self.discard_recording()
        elif self._web_review_pending:
            self.discard_pending_recording()

    def latest_camera_jpeg(self, camera_name: str) -> bytes | None:
        with self._web_live_lock:
            item = self._web_camera_frames.get(camera_name)
            return item[2] if item else None

    def wait_for_camera_jpeg(
        self,
        camera_name: str,
        after_sequence: int,
        timeout: float,
    ) -> tuple[int, float, bytes] | None:
        """Wait for a newer compressed preview frame without building a queue."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._web_camera_condition:
            while True:
                item = self._web_camera_frames.get(camera_name)
                if item is not None and item[0] > after_sequence:
                    return item
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._web_camera_condition.wait(remaining)

    def live_snapshot(self) -> dict[str, Any]:
        with self._raw_topic_stats_lock:
            raw_stats = copy.deepcopy(self._raw_topic_stats)
        with self._web_live_lock:
            receive_times = dict(self._web_receive_times)
            latest_values = copy.deepcopy(self._web_latest_values)

        return build_alignment_snapshot(
            raw_stats=raw_stats,
            receive_times=receive_times,
            latest_values=latest_values,
            expected_topics=self._expected_web_topics(),
            collector_errors=self.has_collection_errors(),
            recording=bool(self.is_recording),
        )


def make_reviewable_collector_class(base_class):
    class ReviewableDataCollector(ReviewableCollectorMixin, base_class):
        pass

    ReviewableDataCollector.__name__ = "ReviewableDataCollector"
    return ReviewableDataCollector


def build_alignment_snapshot(
    *,
    raw_stats: dict[str, dict[str, Any]],
    receive_times: dict[str, float],
    latest_values: dict[str, Any],
    expected_topics: list[str],
    collector_errors: bool,
    recording: bool,
) -> dict[str, Any]:
    now = time.time()
    reference = raw_stats.get(REFERENCE_TOPIC, {})
    reference_last = reference.get("last_timestamp")
    topic_rows = []
    missing_topics = []
    invalid_topics = []
    stale_topics = []

    first_timestamps = []
    last_timestamps = []
    for topic_name in expected_topics:
        metadata = TOPICS.get(
            topic_name,
            {"label": topic_name, "group": "other"},
        )
        statistics = raw_stats.get(topic_name, {})
        count = int(statistics.get("record_count", 0))
        first_timestamp = statistics.get("first_timestamp")
        last_timestamp = statistics.get("last_timestamp")
        last_receive = receive_times.get(topic_name)
        age_ms = (
            max(0.0, (now - last_receive) * 1000.0)
            if last_receive is not None
            else None
        )
        delta_ms = None
        if reference_last is not None and last_timestamp is not None:
            delta_ms = (float(last_timestamp) - float(reference_last)) * 1000.0

        rate_hz = None
        if (
            count > 1
            and first_timestamp is not None
            and last_timestamp is not None
        ):
            span = float(last_timestamp) - float(first_timestamp)
            if span > 0:
                rate_hz = (count - 1) / span

        regressions = int(statistics.get("timestamp_regressions", 0))
        nonfinite = int(statistics.get("nonfinite_timestamps", 0))
        duplicates = int(statistics.get("duplicate_timestamps", 0))
        status = "ok"
        note = "正常"
        if count <= 0:
            status = "bad"
            note = "尚未收到"
            missing_topics.append(topic_name)
        elif regressions or nonfinite:
            status = "bad"
            note = "时间戳异常"
            invalid_topics.append(topic_name)
        elif (
            recording
            and age_ms is not None
            and topic_name not in EVENT_DRIVEN_TOPICS
        ):
            group = metadata["group"]
            warn_age = 1500.0 if group == "action" else 500.0
            bad_age = 5000.0 if group == "action" else 1500.0
            if age_ms > bad_age:
                status = "bad"
                note = "数据已中断"
                stale_topics.append(topic_name)
            elif age_ms > warn_age:
                status = "warn"
                note = "更新较慢" if group != "action" else "保持上一指令"

        if status == "ok" and delta_ms is not None:
            absolute_delta = abs(delta_ms)
            if metadata["group"] != "action" and absolute_delta > 500:
                status = "bad"
                note = "与头相机偏差过大"
            elif metadata["group"] != "action" and absolute_delta > 150:
                status = "warn"
                note = "与头相机偏差偏大"
            elif metadata["group"] == "action" and delta_ms > 500:
                status = "warn"
                note = "指令时间领先参考流"

        if first_timestamp is not None and math.isfinite(float(first_timestamp)):
            first_timestamps.append(float(first_timestamp))
        if last_timestamp is not None and math.isfinite(float(last_timestamp)):
            last_timestamps.append(float(last_timestamp))

        topic_rows.append(
            {
                "name": topic_name,
                "label": metadata["label"],
                "group": metadata["group"],
                "count": count,
                "rate_hz": rate_hz,
                "first_timestamp": first_timestamp,
                "last_timestamp": last_timestamp,
                "age_ms": age_ms,
                "delta_to_head_ms": delta_ms,
                "timestamp_sources": statistics.get(
                    "timestamp_source_counts",
                    {},
                ),
                "timestamp_regressions": regressions,
                "duplicate_timestamps": duplicates,
                "nonfinite_timestamps": nonfinite,
                "status": status,
                "note": note,
                "latest_value": latest_values.get(topic_name),
            }
        )

    common_start = max(first_timestamps) if len(first_timestamps) == len(expected_topics) else None
    common_end = min(last_timestamps) if len(last_timestamps) == len(expected_topics) else None
    common_duration = (
        max(0.0, common_end - common_start)
        if common_start is not None and common_end is not None
        else 0.0
    )
    reference_first = reference.get("first_timestamp")
    uncovered_prefix = (
        max(0.0, common_start - float(reference_first))
        if common_start is not None and reference_first is not None
        else None
    )
    estimated_10hz_frames = (
        max(0, int(math.floor(common_duration * 10.0)) + 1)
        if common_start is not None and common_end is not None
        else 0
    )
    full_horizon_windows = max(0, estimated_10hz_frames - 49)
    required_topic_count = len(TOPICS)
    expected_topic_count = len(expected_topics)
    detected_topic_count = max(
        0,
        expected_topic_count - len(missing_topics),
    )
    all_topics_detected = (
        expected_topic_count == required_topic_count
        and detected_topic_count == required_topic_count
    )

    overall = "ok"
    summary = f"{TOPIC_COUNT} 路时间戳可用于后续对齐"
    bad_rows = [row["name"] for row in topic_rows if row["status"] == "bad"]
    if (
        collector_errors
        or missing_topics
        or invalid_topics
        or stale_topics
        or bad_rows
    ):
        overall = "bad"
        summary = "存在缺失、中断、无效时间戳或严重对齐偏差"
    elif uncovered_prefix is not None and uncovered_prefix > 0.5:
        overall = "warn"
        summary = f"共同覆盖晚于头相机 {uncovered_prefix:.2f} 秒"
    elif any(row["status"] == "warn" for row in topic_rows):
        overall = "warn"
        summary = "部分数据流需要关注"

    return {
        "overall": overall,
        "summary": summary,
        "reference_topic": REFERENCE_TOPIC,
        "topics": topic_rows,
        "missing_topics": missing_topics,
        "invalid_topics": invalid_topics,
        "stale_topics": stale_topics,
        "common_start_timestamp": common_start,
        "common_end_timestamp": common_end,
        "common_duration_seconds": common_duration,
        "uncovered_prefix_seconds": uncovered_prefix,
        "estimated_10hz_frames": estimated_10hz_frames,
        "full_50_action_windows": full_horizon_windows,
        "collector_errors": collector_errors,
        "required_topic_count": required_topic_count,
        "expected_topic_count": expected_topic_count,
        "detected_topic_count": detected_topic_count,
        "all_topics_detected": all_topics_detected,
    }


def project_alignment_to_effective_interval(
    alignment: dict[str, Any],
    effective_start_timestamp: float | None,
    effective_end_timestamp: float | None = None,
) -> dict[str, Any]:
    """Project alignment metrics onto the usable task interval.

    Event-driven VR/Revo2 commands provide carry-forward context, so they must
    exist before the effective start but do not truncate the interval end.
    """
    projected = copy.deepcopy(alignment)
    try:
        effective_start = float(effective_start_timestamp)
    except (TypeError, ValueError):
        return projected
    if not math.isfinite(effective_start):
        return projected

    rows = projected.get("topics", [])
    finite_first = []
    continuous_last = []
    for row in rows:
        try:
            first_timestamp = float(row.get("first_timestamp"))
        except (TypeError, ValueError):
            first_timestamp = None
        if first_timestamp is not None and math.isfinite(first_timestamp):
            finite_first.append(first_timestamp)

        if row.get("name") not in CONTINUOUS_TOPICS:
            continue
        try:
            last_timestamp = float(row.get("last_timestamp"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(last_timestamp):
            continuous_last.append(last_timestamp)

    expected_topic_count = int(
        projected.get("expected_topic_count", len(TOPICS))
    )
    effective_common_start = (
        max(effective_start, max(finite_first))
        if len(finite_first) == expected_topic_count
        else None
    )
    effective_common_end = (
        min(continuous_last)
        if len(continuous_last) == len(CONTINUOUS_TOPICS)
        else None
    )
    if effective_end_timestamp is not None:
        try:
            requested_end = float(effective_end_timestamp)
        except (TypeError, ValueError):
            requested_end = None
        if requested_end is not None and math.isfinite(requested_end):
            effective_common_end = (
                min(effective_common_end, requested_end)
                if effective_common_end is not None
                else requested_end
            )

    duration = (
        max(0.0, effective_common_end - effective_common_start)
        if effective_common_start is not None
        and effective_common_end is not None
        else 0.0
    )
    reference_first = next(
        (
            row.get("first_timestamp")
            for row in rows
            if row.get("name") == REFERENCE_TOPIC
        ),
        None,
    )
    try:
        warmup_prefix = max(
            0.0,
            effective_start - float(reference_first),
        )
    except (TypeError, ValueError):
        warmup_prefix = None

    projected["capture_common_start_timestamp"] = projected.get(
        "common_start_timestamp"
    )
    projected["capture_common_end_timestamp"] = projected.get(
        "common_end_timestamp"
    )
    projected["capture_uncovered_prefix_seconds"] = projected.get(
        "uncovered_prefix_seconds"
    )
    projected["effective_start_timestamp"] = effective_start
    projected["effective_end_timestamp"] = effective_common_end
    projected["warmup_prefix_seconds"] = warmup_prefix
    projected["common_start_timestamp"] = effective_common_start
    projected["common_end_timestamp"] = effective_common_end
    projected["common_duration_seconds"] = duration
    projected["uncovered_prefix_seconds"] = (
        max(0.0, effective_common_start - effective_start)
        if effective_common_start is not None
        else None
    )
    projected["estimated_10hz_frames"] = (
        max(0, int(math.floor(duration * 10.0)) + 1)
        if effective_common_start is not None
        and effective_common_end is not None
        else 0
    )
    projected["full_50_action_windows"] = max(
        0,
        projected["estimated_10hz_frames"] - 49,
    )

    bad_rows = [
        row["name"] for row in rows if row.get("status") == "bad"
    ]
    warn_rows = [
        row["name"] for row in rows if row.get("status") == "warn"
    ]
    if (
        not projected.get("collector_errors")
        and not projected.get("missing_topics")
        and not projected.get("invalid_topics")
        and not projected.get("stale_topics")
        and not bad_rows
        and not warn_rows
        and effective_common_start is not None
        and effective_common_end is not None
        and effective_common_end > effective_common_start
    ):
        projected["overall"] = "ok"
        projected["summary"] = f"有效采集区间 {TOPIC_COUNT} 路可对齐"
    return projected


class DemoCollector:
    """Safe in-memory collector used for UI/API acceptance tests."""

    def __init__(self):
        self.episode_count = 177
        self.current_episode_task = None
        self.current_episode_start_time = None
        self.current_episode_stop_time = None
        self.is_recording = False
        self._review_pending = False
        self._start_wall = None
        self._effective_start_timestamp = None
        self._effective_end_timestamp = None
        self._capture_duration = 0.0
        self._run_duration = 0.0
        self._demo_camera_lock = threading.Lock()
        self._demo_camera_sequences = {name: 0 for name in CAMERA_TOPICS}
        self.dataset_metadata = {
            "episodes": [
                {
                    "episode_id": 176,
                    "task": "pick_trash",
                    "duration": 8.66,
                    "path": "episode_0176",
                }
            ]
        }
        self._tiny_jpeg = base64.b64decode(
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsL"
            "DBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/"
            "2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAACAAIDASIAAhEBAxEB/8QA"
            "HwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUF"
            "BAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkK"
            "FhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1"
            "dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXG"
            "x8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEB"
            "AQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAEC"
            "AxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRom"
            "JygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOE"
            "hYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU"
            "1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDk6KKK+mPn"
            "j//Z"
        )

    def start_recording(self, task="default_task"):
        self.current_episode_task = task
        self.current_episode_start_time = time.time()
        self._start_wall = self.current_episode_start_time
        self.is_recording = True
        self._review_pending = False
        self._effective_start_timestamp = None
        self._effective_end_timestamp = None
        self._capture_duration = 0.0

    def mark_effective_start(self):
        if not self.is_recording:
            raise RuntimeError("正式采集尚未启动，不能记录有效起点")
        self._effective_start_timestamp = time.time()
        return self._effective_start_timestamp

    def stop_for_review(self):
        self.current_episode_stop_time = time.time()
        self._capture_duration = max(
            0.0,
            self.current_episode_stop_time
            - float(
                self.current_episode_start_time
                or self.current_episode_stop_time
            ),
        )
        self._effective_end_timestamp = self.current_episode_stop_time
        self._run_duration = max(
            0.0,
            self._effective_end_timestamp
            - float(
                self._effective_start_timestamp
                or self.current_episode_stop_time
            ),
        )
        self.is_recording = False
        self._review_pending = True
        return self.pending_validation()

    def pending_validation(self):
        return {
            "valid": True,
            "missing_topics": [],
            "invalid_topics": [],
            "collector_errors": [],
            "effective_missing_topics": [],
            "interrupted_topics": [],
            "effective_start_timestamp": self._effective_start_timestamp,
            "effective_end_timestamp": self._effective_end_timestamp,
            "effective_duration": self._run_duration,
            "effective_interval_valid": bool(
                self._effective_start_timestamp is not None
                and self._effective_end_timestamp is not None
                and self._effective_end_timestamp
                > self._effective_start_timestamp
            ),
        }

    def save_pending_recording(self):
        if not self._review_pending:
            raise RuntimeError("没有等待保存的数据")
        episode_id = self.episode_count
        info = {
            "episode_id": episode_id,
            "task": self.current_episode_task,
            "duration": self._run_duration,
            "episode_dir": f"/demo/episode_{episode_id:04d}",
            "storage_format": "raw_topics_pickle_v1",
            "num_topics": TOPIC_COUNT,
            "total_records": max(
                TOPIC_COUNT,
                int(self._run_duration * 900),
            ),
            "effective_start_timestamp": self._effective_start_timestamp,
            "effective_end_timestamp": self._effective_end_timestamp,
            "effective_duration": self._run_duration,
            "effective_context_topics": sorted(EVENT_DRIVEN_TOPICS),
        }
        self.dataset_metadata["episodes"].append(
            {
                "episode_id": episode_id,
                "task": self.current_episode_task,
                "duration": self._run_duration,
                "path": f"episode_{episode_id:04d}",
            }
        )
        self.episode_count += 1
        self._review_pending = False
        self.current_episode_task = None
        self.current_episode_start_time = None
        self._effective_start_timestamp = None
        self._effective_end_timestamp = None
        return info

    def discard_pending_recording(self):
        if not self._review_pending:
            raise RuntimeError("没有等待丢弃的数据")
        self._review_pending = False
        self.current_episode_task = None
        self.current_episode_start_time = None
        self._effective_start_timestamp = None
        self._effective_end_timestamp = None
        return True

    def abort_active_recording(self):
        self.is_recording = False
        self._review_pending = False
        self.current_episode_task = None
        self.current_episode_start_time = None
        self.current_episode_stop_time = None
        self._effective_start_timestamp = None
        self._effective_end_timestamp = None
        return True

    def latest_camera_jpeg(self, camera_name):
        if camera_name not in CAMERA_TOPICS or not (
            self.is_recording or self._review_pending
        ):
            return None
        return self._tiny_jpeg

    def wait_for_camera_jpeg(
        self,
        camera_name,
        after_sequence,
        timeout,
    ):
        if camera_name not in CAMERA_TOPICS or not self.is_recording:
            time.sleep(min(max(0.0, float(timeout)), 0.05))
            return None
        time.sleep(min(CAMERA_PREVIEW_INTERVAL_SECONDS, max(0.0, timeout)))
        with self._demo_camera_lock:
            sequence = max(
                int(after_sequence) + 1,
                self._demo_camera_sequences[camera_name] + 1,
            )
            self._demo_camera_sequences[camera_name] = sequence
        return sequence, time.time(), self._tiny_jpeg

    def live_snapshot(self):
        elapsed = (
            time.time() - self._start_wall
            if self.is_recording and self._start_wall
            else self._capture_duration
        )
        now = time.time()
        raw_stats = {}
        receive_times = {}
        if elapsed > 0:
            for index, name in enumerate(TOPICS):
                late = 1.2 if name in ACTION_TOPICS else 0.0
                effective = max(0.0, elapsed - late)
                group = TOPICS[name]["group"]
                rate = 30 if group == "camera" else (120 if group == "state" else 80)
                count = int(effective * rate)
                if count <= 0:
                    continue
                first = float(self._start_wall or now) + late
                raw_stats[name] = {
                    "record_count": count,
                    "first_timestamp": first,
                    "last_timestamp": first + (count - 1) / rate,
                    "timestamp_regressions": 0,
                    "duplicate_timestamps": 0,
                    "nonfinite_timestamps": 0,
                    "timestamp_source_counts": {
                        "message_header" if group != "action" else "receipt_wall_clock": count
                    },
                }
                receive_times[name] = now - index * 0.001
        return build_alignment_snapshot(
            raw_stats=raw_stats,
            receive_times=receive_times,
            latest_values={
                "left_revo2_joint_states": {
                    "positions": [12.0, 8.0, 35.0, 31.0, 28.0, 24.0],
                    "currents": [35, 21, 48, 44, 40, 37],
                },
                "right_revo2_joint_states": {
                    "positions": [11.0, 7.0, 34.0, 30.0, 27.0, 23.0],
                    "currents": [34, 20, 47, 43, 39, 36],
                },
            },
            expected_topics=list(TOPICS),
            collector_errors=False,
            recording=self.is_recording,
        )

    def shutdown_without_saving(self):
        self.is_recording = False
        self._review_pending = False


@dataclass
class ControllerSnapshot:
    state: str
    version: int
    run_id: str | None
    task: str | None
    cooldown_remaining: int
    last_error: str | None
    last_result: dict[str, Any] | None


class CollectionController:
    def __init__(
        self,
        collector,
        cooldown_seconds: int,
        output_dir: Path,
        start_guard: Callable[[], None] | None = None,
    ):
        self.collector = collector
        self.cooldown_seconds = cooldown_seconds
        self.output_dir = output_dir
        self.start_guard = start_guard
        self.lock = threading.RLock()
        self.state = "ready"
        self.version = 1
        self.run_id: str | None = None
        self.task: str | None = None
        self.cooldown_until = 0.0
        self.last_error: str | None = None
        self.last_traceback: str | None = None
        self.last_result: dict[str, Any] | None = None
        self.pending_validation: dict[str, Any] | None = None
        self._operation_thread: threading.Thread | None = None
        self._shutting_down = False
        self._restart_requested = False
        self._start_cancel_event = threading.Event()
        self.preflight_alignment: dict[str, Any] | None = None
        self.armed_until = 0.0
        self.monitoring_active = False
        self.effective_start_timestamp: float | None = None
        self.effective_end_timestamp: float | None = None
        self._e6_recovery_operation_lock = threading.Lock()
        self._e6_recovery_status: dict[str, Any] = {
            "running": False,
            "started_at": None,
            "finished_at": None,
            "last_result": None,
        }

    def _ensure_e6_recovery_idle_locked(self):
        if self._e6_recovery_status["running"]:
            raise ApiError(
                HTTPStatus.CONFLICT,
                "E6 正在恢复，请等待操作完成",
            )

    def _clear_effective_interval_locked(self):
        self.effective_start_timestamp = None
        self.effective_end_timestamp = None

    def _refresh_cooldown_locked(self):
        if self.state == "cooldown" and time.monotonic() >= self.cooldown_until:
            self.state = "ready"
            self.version += 1
            self.run_id = None
            self.task = None
            self.pending_validation = None
            self._clear_effective_interval_locked()

    def _snapshot_locked(self) -> ControllerSnapshot:
        self._refresh_cooldown_locked()
        remaining = (
            max(0, int(math.ceil(self.cooldown_until - time.monotonic())))
            if self.state == "cooldown"
            else 0
        )
        return ControllerSnapshot(
            state=self.state,
            version=self.version,
            run_id=self.run_id,
            task=self.task,
            cooldown_remaining=remaining,
            last_error=self.last_error,
            last_result=copy.deepcopy(self.last_result),
        )

    def _begin_operation(
        self,
        *,
        allowed_state: str,
        new_state: str,
        body: dict[str, Any],
        target: Callable[[], None],
        require_run_id: bool = True,
    ) -> dict[str, Any]:
        with self.lock:
            if self._shutting_down:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "服务正在安全退出，请稍后重试",
                )
            self._refresh_cooldown_locked()
            if self.state != allowed_state:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"当前状态为“{STATE_LABELS[self.state]}”，不能执行此操作",
                )
            if require_run_id and body.get("run_id") != self.run_id:
                raise ApiError(HTTPStatus.CONFLICT, "页面状态已过期，请刷新后重试")
            self.state = new_state
            self.version += 1
            self.last_error = None
            self.last_traceback = None
            thread = threading.Thread(
                target=target,
                daemon=False,
                name=f"WebCollection-{new_state}",
            )
            self._operation_thread = thread
            thread.start()
            return {
                "accepted": True,
                "state": self.state,
                "version": self.version,
                "run_id": self.run_id,
            }

    def start(self, body: dict[str, Any]) -> dict[str, Any]:
        task = str(body.get("task", "")).strip() or DEFAULT_TASK
        if len(task) > 100 or any(ord(character) < 32 for character in task):
            raise ApiError(HTTPStatus.BAD_REQUEST, "任务名称无效")

        configured_topic_rows = [
            str(topic.get("name", ""))
            for topic in self.collector.live_snapshot().get("topics", [])
        ]
        configured_topics = set(configured_topic_rows)
        required_topics = set(TOPICS)
        duplicate_topics = sorted(
            {
                name
                for name in configured_topic_rows
                if configured_topic_rows.count(name) > 1
            }
        )
        if (
            len(configured_topic_rows) != len(TOPICS)
            or configured_topics != required_topics
        ):
            missing = sorted(required_topics - configured_topics)
            extra = sorted(configured_topics - required_topics)
            details = []
            if missing:
                details.append(f"缺少：{', '.join(missing)}")
            if extra:
                details.append(f"多出：{', '.join(extra)}")
            if duplicate_topics:
                details.append(f"重复：{', '.join(duplicate_topics)}")
            if len(configured_topic_rows) != len(TOPICS):
                details.append(
                    f"实际 {len(configured_topic_rows)} 路，要求 {len(TOPICS)} 路"
                )
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"当前采集配置不是规定的 {TOPIC_COUNT} 路 Topic"
                + (f"（{'；'.join(details)}）" if details else ""),
            )

        with self.lock:
            self._refresh_cooldown_locked()
            if self.state != "ready":
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"当前状态为“{STATE_LABELS[self.state]}”，不能开始",
                )
            self._ensure_e6_recovery_idle_locked()
            if self.start_guard is not None:
                self.start_guard()
            self.run_id = secrets.token_hex(12)
            self.task = task
            self.monitoring_active = True
            self.preflight_alignment = None
            self.armed_until = 0.0
            self._clear_effective_interval_locked()
            self._start_cancel_event.clear()
            # Keep the readiness check, run id allocation, task assignment and
            # transition atomic across simultaneous browser requests.
            return self._begin_operation(
                allowed_state="ready",
                new_state="starting",
                body=body,
                require_run_id=False,
                target=self._do_start,
            )

    @staticmethod
    def _topic_gate_ready(alignment: dict[str, Any] | None) -> bool:
        if not alignment:
            return False
        topic_names = [
            str(topic.get("name", ""))
            for topic in alignment.get("topics", [])
        ]
        return bool(
            len(topic_names) == len(TOPICS)
            and len(set(topic_names)) == len(TOPICS)
            and set(topic_names) == set(TOPICS)
            and alignment.get("all_topics_detected")
            and not any(
                topic.get("status") == "bad"
                for topic in alignment.get("topics", [])
            )
            and not alignment.get("invalid_topics")
            and not alignment.get("stale_topics")
            and not alignment.get("collector_errors")
        )

    @staticmethod
    def _blocked_topics(
        alignment: dict[str, Any] | None,
    ) -> list[str]:
        snapshot = alignment or {}
        blocked = set()
        for key in ("missing_topics", "stale_topics", "invalid_topics"):
            blocked.update(str(name) for name in snapshot.get(key, []))
        blocked.update(
            str(topic.get("name", ""))
            for topic in snapshot.get("topics", [])
            if topic.get("status") == "bad"
        )
        return [name for name in TOPICS if name in blocked]

    def _wait_for_topic_gate(
        self,
        timeout_seconds: float | None,
    ) -> tuple[bool, dict[str, Any] | None]:
        deadline = (
            time.monotonic() + timeout_seconds
            if timeout_seconds is not None
            else None
        )
        stable_polls = 0
        latest_alignment: dict[str, Any] | None = None
        while deadline is None or time.monotonic() < deadline:
            if self._start_cancel_event.is_set():
                break
            with self.lock:
                if self._shutting_down:
                    self._start_cancel_event.set()
                    break

            latest_alignment = self.collector.live_snapshot()
            stable_polls = (
                stable_polls + 1
                if self._topic_gate_ready(latest_alignment)
                else 0
            )
            if stable_polls >= TOPIC_PREFLIGHT_STABLE_POLLS:
                return True, latest_alignment
            self._start_cancel_event.wait(TOPIC_PREFLIGHT_POLL_SECONDS)
        return False, latest_alignment

    def _abort_active_recording(self):
        abort = getattr(self.collector, "abort_active_recording", None)
        if not callable(abort):
            raise RuntimeError("采集器不支持安全取消启动")
        abort()

    def _finish_start_without_recording(
        self,
        *,
        kind: str,
        alignment: dict[str, Any] | None,
        error: str | None = None,
        preserve_last_result: bool = False,
    ):
        with self.lock:
            current_run_id = self.run_id
            self.state = "ready"
            result = {
                "kind": kind,
                "run_id": current_run_id,
                "missing_topics": list(
                    (alignment or {}).get("missing_topics", list(TOPICS))
                ),
                "stale_topics": list(
                    (alignment or {}).get("stale_topics", [])
                ),
                "invalid_topics": list(
                    (alignment or {}).get("invalid_topics", [])
                ),
                "blocked_topics": self._blocked_topics(alignment),
            }
            if error:
                result["error"] = error
                self.last_error = error
            if preserve_last_result and self.last_result:
                self.last_result = {
                    **self.last_result,
                    "monitoring_result": result,
                }
            else:
                self.last_result = result
            self.run_id = None
            self.task = None
            self.monitoring_active = False
            self.preflight_alignment = None
            self.armed_until = 0.0
            self._clear_effective_interval_locked()
            self.version += 1

    def _run_monitoring_cycle(
        self,
        *,
        timeout_kind: str,
        failure_kind: str,
        preserve_last_result: bool = False,
    ):
        latest_alignment: dict[str, Any] | None = None
        try:
            self.collector.start_recording(task=self.task)
            ready, latest_alignment = self._wait_for_topic_gate(
                None
            )
            with self.lock:
                canceled = (
                    self._start_cancel_event.is_set()
                    or self.state in {"canceling_start", "closing"}
                    or not self.monitoring_active
                )
                if ready and not canceled:
                    self.preflight_alignment = json_safe(latest_alignment)
                    self.armed_until = 0.0
                    self.state = "armed"
                    self.version += 1
                    return

            self._abort_active_recording()
            self._finish_start_without_recording(
                kind=(
                    "collection_closed"
                    if canceled
                    else timeout_kind
                ),
                alignment=latest_alignment,
                preserve_last_result=(
                    preserve_last_result and not canceled
                ),
            )
        except BaseException as error:
            failure_traceback = traceback.format_exc()
            try:
                self._abort_active_recording()
            except Exception:
                traceback.print_exc()
            with self.lock:
                canceled = (
                    self._start_cancel_event.is_set()
                    or self.state in {"canceling_start", "closing"}
                    or not self.monitoring_active
                )
            if canceled:
                self._finish_start_without_recording(
                    kind="collection_closed",
                    alignment=latest_alignment,
                )
                return
            print(failure_traceback, file=sys.stderr, flush=True)
            self._finish_start_without_recording(
                kind=failure_kind,
                alignment=latest_alignment,
                error=f"{type(error).__name__}: {error}",
                preserve_last_result=preserve_last_result,
            )

    def _do_start(self):
        self._run_monitoring_cycle(
            timeout_kind="preflight_timeout",
            failure_kind="preflight_failed",
        )

    def confirm_start(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._refresh_cooldown_locked()
            self._ensure_e6_recovery_idle_locked()
            if self.state != "armed":
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"当前状态为“{STATE_LABELS[self.state]}”，不能开始采集",
                )
            if body.get("run_id") != self.run_id:
                raise ApiError(HTTPStatus.CONFLICT, "页面状态已过期，请刷新后重试")
        live = self.collector.live_snapshot()
        if not self._topic_gate_ready(live):
            blocked = self._blocked_topics(live)
            detail = f"：{', '.join(blocked)}" if blocked else ""
            raise ApiError(
                HTTPStatus.CONFLICT,
                f"数据监测未通过，暂不能开始采集{detail}",
            )
        return self._begin_operation(
            allowed_state="armed",
            new_state="starting_recording",
            body=body,
            target=self._do_confirm_start,
        )

    def recover_e6(self, body: dict[str, Any]) -> dict[str, Any]:
        del body
        with self.lock:
            self._refresh_cooldown_locked()
            if self._shutting_down:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "服务正在安全退出，请稍后重试",
                )
            if self.state not in {"ready", "starting"}:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "只能在数采关闭或预检阶段恢复 E6；正式采集期间禁止重置",
                )
            if not self._e6_recovery_operation_lock.acquire(blocking=False):
                raise ApiError(HTTPStatus.CONFLICT, "E6 恢复操作正在进行")
            self._e6_recovery_status = {
                "running": True,
                "started_at": time.time(),
                "finished_at": None,
                "last_result": None,
            }
            self.version += 1

        result: dict[str, Any]
        try:
            recovery = getattr(self.collector, "recover_e6_transport", None)
            if not callable(recovery):
                result = {
                    "ok": False,
                    "code": "unsupported",
                    "message": "当前采集器不支持 E6 恢复",
                }
            else:
                raw_result = recovery()
                if not isinstance(raw_result, dict):
                    raise RuntimeError("E6 恢复函数返回了无效结果")
                result = json_safe(raw_result)
        except BaseException as error:
            traceback.print_exc()
            result = {
                "ok": False,
                "code": "recovery_failed",
                "message": f"{type(error).__name__}: {error}",
            }
        finally:
            with self.lock:
                self._e6_recovery_status = {
                    "running": False,
                    "started_at": self._e6_recovery_status["started_at"],
                    "finished_at": time.time(),
                    "last_result": result,
                }
                self.version += 1
            self._e6_recovery_operation_lock.release()

        return {
            "accepted": True,
            "state": self.state,
            "version": self.version,
            "e6_recovery": result,
        }

    def _do_confirm_start(self):
        try:
            latest_alignment = self.collector.live_snapshot()
            with self.lock:
                if not self._topic_gate_ready(latest_alignment):
                    self.preflight_alignment = json_safe(latest_alignment)
                    self.last_error = "数据状态刚刚发生变化，请恢复后再开始采集"
                    self.state = "armed"
                    self.version += 1
                    return
            effective_start = self.collector.mark_effective_start()
            with self.lock:
                self.preflight_alignment = None
                self.armed_until = 0.0
                self.effective_start_timestamp = effective_start
                self.effective_end_timestamp = None
                self.state = "recording"
                self.version += 1
        except BaseException as error:
            self._operation_failed(error)

    def cancel_start(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.close_collection(body)

    def close_collection(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._refresh_cooldown_locked()
            if self.state not in {"starting", "armed"}:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "请先停止当前 Episode 并完成保存或删除，再关闭数采",
                )
            if body.get("run_id") != self.run_id:
                raise ApiError(HTTPStatus.CONFLICT, "页面状态已过期，请刷新后重试")
            self.monitoring_active = False
            if self.state == "starting":
                self.state = "closing"
                self.version += 1
                self._start_cancel_event.set()
                cancel_startup = getattr(
                    self.collector,
                    "cancel_startup_wait",
                    None,
                )
                if callable(cancel_startup):
                    cancel_startup()
                return {
                    "accepted": True,
                    "state": self.state,
                    "version": self.version,
                    "run_id": self.run_id,
                }
            return self._begin_operation(
                allowed_state="armed",
                new_state="closing",
                body=body,
                target=self._do_close_collection,
            )

    def _do_close_collection(self):
        try:
            self._abort_active_recording()
            self._finish_start_without_recording(
                kind="collection_closed",
                alignment=None,
            )
        except BaseException as error:
            self._operation_failed(error)

    def _restart_monitoring_after_episode(self):
        with self.lock:
            if not self.monitoring_active:
                self.state = "ready"
                self.run_id = None
                self.task = None
                self.pending_validation = None
                self._clear_effective_interval_locked()
                self.version += 1
                return
            self.run_id = secrets.token_hex(12)
            self.pending_validation = None
            self.preflight_alignment = None
            self.armed_until = 0.0
            self._clear_effective_interval_locked()
            self._start_cancel_event.clear()
            self.state = "starting"
            self.version += 1
        self._run_monitoring_cycle(
            timeout_kind="monitoring_restart_timeout",
            failure_kind="monitoring_restart_failed",
            preserve_last_result=True,
        )

    def stop(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._begin_operation(
            allowed_state="recording",
            new_state="stopping",
            body=body,
            target=self._do_stop,
        )

    def _do_stop(self):
        try:
            validation = self.collector.stop_for_review()
            with self.lock:
                self.pending_validation = json_safe(validation)
                self.effective_end_timestamp = validation.get(
                    "effective_end_timestamp"
                )
                self.state = "review"
                self.version += 1
        except BaseException as error:
            self._operation_failed(error)

    def save(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._begin_operation(
            allowed_state="review",
            new_state="saving",
            body=body,
            target=self._do_save,
        )

    def _do_save(self):
        save_started_monotonic = time.monotonic()
        try:
            episode_info = self.collector.save_pending_recording()
            save_elapsed_seconds = max(
                0.0,
                time.monotonic() - save_started_monotonic,
            )
            with self.lock:
                self.last_result = {
                    "kind": "saved",
                    **json_safe(episode_info),
                    "save_elapsed_seconds": save_elapsed_seconds,
                }
                self.version += 1
            self._restart_monitoring_after_episode()
        except BaseException as error:
            self._operation_failed(error)

    def discard(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._begin_operation(
            allowed_state="review",
            new_state="discarding",
            body=body,
            target=self._do_discard,
        )

    def _do_discard(self):
        try:
            discarded_episode = self.collector.episode_count
            self.collector.discard_pending_recording()
            with self.lock:
                self.last_result = {
                    "kind": "discarded",
                    "episode_id": discarded_episode,
                }
                self.version += 1
            self._restart_monitoring_after_episode()
        except BaseException as error:
            self._operation_failed(error)

    def _operation_failed(self, error: BaseException):
        with self.lock:
            self.last_error = f"{type(error).__name__}: {error}"
            self.last_traceback = traceback.format_exc()
            self.state = "error"
            self._restart_requested = True
            self.version += 1
        print(self.last_traceback, file=sys.stderr, flush=True)
        # The collector can be left partially initialized after a producer or
        # transaction error.  Exit through the normal SIGTERM cleanup path and
        # return a non-zero status so systemd reconnects a fresh collector.
        os.kill(os.getpid(), signal.SIGTERM)

    def _recent_episodes(self, limit: int = 8) -> list[dict[str, Any]]:
        """Return saved Episodes that still exist in the output directory.

        ``dataset_metadata.json`` is intentionally kept as an append-only
        collection ledger while the service is running.  An operator may
        remove an Episode directory out of band, so using the in-memory ledger
        alone leaves stale rows in the web UI until the next service restart.
        The status endpoint polls this helper and treats the committed
        directory plus ``episode.json`` as the source of truth for visibility.
        """
        episodes = list(
            getattr(self.collector, "dataset_metadata", {}).get(
                "episodes",
                [],
            )
        )

        def episode_sort_key(item: Any) -> int:
            if not isinstance(item, dict):
                return -1
            try:
                episode_id = item.get("episode_id", -1)
                if isinstance(episode_id, bool):
                    return -1
                return int(episode_id)
            except (TypeError, ValueError):
                return -1

        # DemoCollector is deliberately in-memory and has no metadata file or
        # Episode directories.  Real DataCollector instances always expose
        # ``metadata_file`` and must be reconciled against durable storage.
        require_disk_presence = hasattr(self.collector, "metadata_file")
        recent_episodes: list[dict[str, Any]] = []
        for item in sorted(episodes, key=episode_sort_key, reverse=True):
            if not isinstance(item, dict):
                continue
            episode_id = episode_sort_key(item)
            if episode_id < 0:
                continue

            expected_name = f"episode_{episode_id:04d}"
            if require_disk_presence:
                episode_dir = self.output_dir / expected_name
                try:
                    committed_on_disk = bool(
                        not episode_dir.is_symlink()
                        and episode_dir.is_dir()
                        and (episode_dir / "episode.json").is_file()
                    )
                except OSError:
                    committed_on_disk = False
                if not committed_on_disk:
                    continue

            recent_episodes.append(
                {
                    "episode_id": episode_id,
                    "task": item.get("task"),
                    "duration": item.get(
                        "effective_duration",
                        item.get("duration"),
                    ),
                    "path": expected_name,
                }
            )
            if len(recent_episodes) >= limit:
                break
        return recent_episodes

    def status(self) -> dict[str, Any]:
        with self.lock:
            state_snapshot = self._snapshot_locked()
            pending_validation = copy.deepcopy(self.pending_validation)
            preflight_alignment = copy.deepcopy(self.preflight_alignment)
            effective_start_timestamp = self.effective_start_timestamp
            effective_end_timestamp = self.effective_end_timestamp
            monitoring_active = self.monitoring_active
            task = self.task
            e6_recovery_status = copy.deepcopy(self._e6_recovery_status)

        live = self.collector.live_snapshot()
        if state_snapshot.state == "ready" and not monitoring_active:
            live = build_alignment_snapshot(
                raw_stats={},
                receive_times={},
                latest_values={},
                expected_topics=list(TOPICS),
                collector_errors=False,
                recording=False,
            )
        if (
            state_snapshot.state
            in {"recording", "stopping", "review", "saving"}
            and effective_start_timestamp is not None
        ):
            live = project_alignment_to_effective_interval(
                live,
                effective_start_timestamp,
                effective_end_timestamp,
            )
        if (
            state_snapshot.state == "review"
            and pending_validation
            and not pending_validation.get("valid", False)
        ):
            live["overall"] = "bad"
            if pending_validation.get("interrupted_topics"):
                live["summary"] = (
                    "采集期间检测到连续数据流中断，本条禁止保存"
                )
            elif pending_validation.get("effective_missing_topics"):
                live["summary"] = (
                    "有效起点后有连续数据流未产生新样本，本条禁止保存"
                )
            elif pending_validation.get("start_boundary_topics"):
                live["summary"] = (
                    "部分数据流未覆盖有效起点，本条禁止保存"
                )
            elif not pending_validation.get(
                "effective_interval_valid",
                False,
            ):
                live["summary"] = (
                    "有效采集区间为空或未被连续数据流完整覆盖"
                )
            else:
                live["summary"] = "本条数据未通过保存前完整性校验"
        if state_snapshot.state in {
            "ready",
            "starting",
            "canceling_start",
            "starting_recording",
            "closing",
        } and not any(
            int(topic.get("count", 0)) > 0
            for topic in live.get("topics", [])
        ):
            live["overall"] = "neutral"
            live["summary"] = (
                f"开始预检后显示 {TOPIC_COUNT} 路数据的在线状态"
            )
        duration = 0.0
        if effective_start_timestamp is not None:
            duration_end = effective_end_timestamp or time.time()
            duration = max(
                0.0,
                float(duration_end) - float(effective_start_timestamp),
            )

        disk = shutil.disk_usage(self.output_dir)
        memory = read_memory_status()
        recent_episodes = self._recent_episodes()

        save_allowed = bool(
            state_snapshot.state == "review"
            and (pending_validation or {}).get("valid", False)
        )
        return {
            "ok": True,
            "server_time": time.time(),
            "state": state_snapshot.state,
            "state_label": STATE_LABELS[state_snapshot.state],
            "version": state_snapshot.version,
            "run_id": state_snapshot.run_id,
            "task": task,
            "duration_seconds": duration,
            "episode_id": int(self.collector.episode_count),
            "cooldown_remaining_seconds": state_snapshot.cooldown_remaining,
            "last_error": state_snapshot.last_error,
            "last_result": state_snapshot.last_result,
            "pending_validation": pending_validation,
            "save_allowed": save_allowed,
            "monitoring_active": monitoring_active,
            "e6_recovery": e6_recovery_status,
            "alignment": live,
            "effective_interval": {
                "start_timestamp": effective_start_timestamp,
                "end_timestamp": effective_end_timestamp,
                "duration_seconds": duration,
                "warmup_prefix_seconds": live.get(
                    "warmup_prefix_seconds"
                ),
                "context_topics": sorted(EVENT_DRIVEN_TOPICS),
            },
            "start_gate": {
                "phase": state_snapshot.state,
                "ready": bool(
                    state_snapshot.state == "armed"
                    and self._topic_gate_ready(live)
                ),
                "detected_topic_count": int(
                    live.get("detected_topic_count", 0)
                ),
                "required_topic_count": int(
                    live.get("required_topic_count", len(TOPICS))
                ),
                "missing_topics": list(live.get("missing_topics", [])),
                "stale_topics": list(live.get("stale_topics", [])),
                "invalid_topics": list(live.get("invalid_topics", [])),
                "blocked_topics": self._blocked_topics(live),
                "confirm_allowed": bool(
                    state_snapshot.state == "armed"
                    and self._topic_gate_ready(live)
                ),
                "effective_start_recorded": (
                    effective_start_timestamp is not None
                ),
                "expires_in_seconds": 0,
            },
            "system": {
                "disk_free_gib": disk.free / (1024**3),
                "disk_total_gib": disk.total / (1024**3),
                **memory,
            },
            "recent_episodes": recent_episodes,
            "camera_topics": list(CAMERA_TOPICS),
            "camera_preview_fps": CAMERA_PREVIEW_FPS,
            "head_camera_source": {
                "device": "E6",
                "view": "physical_right_eye_rgb",
                "label": "E6 头环右目 RGB（H.265 · 60 Hz）",
                "source_package": "com.ssnwt.e6stream.debug",
                "encoded_resolution": [3200, 1200],
                "resolution": [1600, 1200],
                "transport": "adb_forward_tcp_e6_stage4_v3",
                "preview_decode": "keyframes_only",
                "preview_fps": 1.0,
            },
            "raw_topic_storage": True,
            "target_training_hz": 10,
            "action_horizon": 50,
        }

    def camera(self, camera_name: str) -> bytes | None:
        if camera_name not in CAMERA_TOPICS:
            raise ApiError(HTTPStatus.NOT_FOUND, "相机不存在")
        return self.collector.latest_camera_jpeg(camera_name)

    def wait_for_camera(
        self,
        camera_name: str,
        after_sequence: int,
        timeout: float,
    ) -> tuple[int, float, bytes] | None:
        if camera_name not in CAMERA_TOPICS:
            raise ApiError(HTTPStatus.NOT_FOUND, "相机不存在")
        waiter = getattr(self.collector, "wait_for_camera_jpeg", None)
        if not callable(waiter):
            jpeg = self.collector.latest_camera_jpeg(camera_name)
            if jpeg is None:
                time.sleep(min(max(0.0, timeout), 0.05))
                return None
            return after_sequence + 1, time.time(), jpeg
        return waiter(camera_name, after_sequence, timeout)

    def begin_shutdown(self):
        with self.lock:
            self._shutting_down = True
            self._start_cancel_event.set()
            cancel_startup = getattr(
                self.collector,
                "cancel_startup_wait",
                None,
            )
            if callable(cancel_startup):
                cancel_startup()

    def restart_requested(self) -> bool:
        with self.lock:
            return self._restart_requested

    def shutdown(self):
        self.begin_shutdown()
        with self.lock:
            operation_thread = self._operation_thread
        if (
            operation_thread is not None
            and operation_thread is not threading.current_thread()
            and operation_thread.is_alive()
        ):
            print(
                f"等待当前操作安全结束：{operation_thread.name}",
                flush=True,
            )
            operation_thread.join(timeout=110)
            if operation_thread.is_alive():
                print(
                    "当前操作在 110 秒内未结束；为避免并发破坏 staging，"
                    "本次退出不执行清理",
                    file=sys.stderr,
                    flush=True,
                )
                return
        try:
            self.collector.shutdown_without_saving()
        except Exception:
            traceback.print_exc()


def read_memory_status() -> dict[str, float]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, raw_value = line.partition(":")
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                values[key] = int(raw_value.strip().split()[0])
    except Exception:
        return {}
    return {
        "memory_available_gib": values.get("MemAvailable", 0) / (1024**2),
        "memory_total_gib": values.get("MemTotal", 0) / (1024**2),
        "swap_used_gib": (
            values.get("SwapTotal", 0) - values.get("SwapFree", 0)
        )
        / (1024**2),
    }


class CollectionHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address,
        handler_class,
        *,
        controller: CollectionController,
        sessions: SessionStore,
    ):
        self.controller = controller
        self.sessions = sessions
        super().__init__(address, handler_class)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "XRCollectionWeb/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, message_format, *args):
        sys.stdout.write(
            "%s - - [%s] %s\n"
            % (
                self.address_string(),
                self.log_date_time_string(),
                message_format % args,
            )
        )
        sys.stdout.flush()

    def _security_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'self'; script-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'",
        )

    def _send_bytes(
        self,
        status: int,
        payload: bytes,
        content_type: str,
        *,
        headers: dict[str, str] | None = None,
        cache: str = "no-store",
    ):
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", cache)
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ):
        encoded = json.dumps(
            json_safe(payload),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self._send_bytes(
            status,
            encoded,
            "application/json; charset=utf-8",
            headers=headers,
        )

    def _body_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Content-Length 无效") from error
        if length > 65_536:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "请求过大")
        if length <= 0:
            return {}
        try:
            payload = json.loads(self.rfile.read(length))
        except Exception as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, "JSON 请求无效") from error
        if not isinstance(payload, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "JSON 必须是对象")
        return payload

    def _cookie_session_id(self) -> str | None:
        raw_cookie = self.headers.get("Cookie", "")
        cookie = SimpleCookie()
        try:
            cookie.load(raw_cookie)
        except Exception:
            return None
        morsel = cookie.get("xr_collection_session")
        return morsel.value if morsel else None

    def _require_session(self, *, csrf: bool = False) -> dict[str, Any]:
        session_id = self._cookie_session_id()
        session = self.server.sessions.get_session(session_id)
        if not session:
            raise ApiError(HTTPStatus.UNAUTHORIZED, "页面会话已过期，请刷新重试")
        if csrf:
            submitted = self.headers.get("X-CSRF-Token", "")
            if not hmac.compare_digest(submitted, str(session["csrf"])):
                raise ApiError(HTTPStatus.FORBIDDEN, "页面令牌无效，请刷新重试")
        session["id"] = session_id
        return session

    def _route(self) -> str:
        return unquote(urlparse(self.path).path)

    def _serve_camera_stream(self, camera_name: str):
        """Serve a bounded-rate MJPEG stream made only from the latest frame.

        The producer cache is one frame deep.  A slow browser therefore drops
        old previews instead of adding latency or back-pressure to recording.
        """
        if camera_name not in CAMERA_TOPICS:
            raise ApiError(HTTPStatus.NOT_FOUND, "相机不存在")
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; "
            f"boundary={CAMERA_STREAM_BOUNDARY.decode('ascii')}",
        )
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        if self.command == "HEAD":
            return

        sequence = 0
        idle_deadline = time.monotonic() + CAMERA_STREAM_IDLE_TIMEOUT_SECONDS
        try:
            while True:
                frame = self.server.controller.wait_for_camera(
                    camera_name,
                    sequence,
                    timeout=1.0,
                )
                if frame is None:
                    if time.monotonic() >= idle_deadline:
                        return
                    continue
                sequence, timestamp, jpeg = frame
                idle_deadline = (
                    time.monotonic() + CAMERA_STREAM_IDLE_TIMEOUT_SECONDS
                )
                header = (
                    b"--"
                    + CAMERA_STREAM_BOUNDARY
                    + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode("ascii")
                    + b"\r\nX-Frame-Timestamp: "
                    + f"{timestamp:.9f}".encode("ascii")
                    + b"\r\n\r\n"
                )
                self.wfile.write(header)
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            route = self._route()
            if route == "/healthz":
                self._send_json(HTTPStatus.OK, {"ok": True})
                return
            if route in {"/", "/index.html"}:
                self._serve_static("index.html")
                return
            if route in {"/app.css", "/app.js"}:
                self._serve_static(route[1:])
                return
            if route == "/api/session":
                session_id = self._cookie_session_id()
                session = self.server.sessions.get_session(session_id)
                headers = None
                if session is None:
                    session_id, csrf_token = (
                        self.server.sessions.create_session(
                            self.client_address[0]
                        )
                    )
                    session = {"csrf": csrf_token}
                    headers = {
                        "Set-Cookie": (
                            "xr_collection_session="
                            f"{session_id}; Path=/; HttpOnly; "
                            "SameSite=Strict; Max-Age=43200"
                        )
                    }
                self._send_json(
                    HTTPStatus.OK,
                    {"authenticated": True, "csrf": session["csrf"]},
                    headers=headers,
                )
                return
            if route == "/api/status":
                self._require_session()
                self._send_json(HTTPStatus.OK, self.server.controller.status())
                return
            if route.startswith("/api/camera/") and route.endswith(".mjpeg"):
                self._require_session()
                camera_name = route[
                    len("/api/camera/") : -len(".mjpeg")
                ]
                self._serve_camera_stream(camera_name)
                return
            if route.startswith("/api/camera/") and route.endswith(".jpg"):
                self._require_session()
                camera_name = route[len("/api/camera/") : -len(".jpg")]
                jpeg = self.server.controller.camera(camera_name)
                if jpeg is None:
                    raise ApiError(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "相机预览尚未就绪",
                    )
                self._send_bytes(
                    HTTPStatus.OK,
                    jpeg,
                    "image/jpeg",
                    cache="no-store, max-age=0",
                )
                return
            raise ApiError(HTTPStatus.NOT_FOUND, "页面不存在")
        except ApiError as error:
            self._send_json(error.status, {"ok": False, "error": str(error)})
        except BrokenPipeError:
            return
        except Exception as error:
            traceback.print_exc()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": f"服务器错误：{error}"},
            )

    def _serve_static(self, filename: str):
        safe_name = Path(filename).name
        path = STATIC_ROOT / safe_name
        if not path.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "静态文件不存在")
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }
        self._send_bytes(
            HTTPStatus.OK,
            path.read_bytes(),
            content_types.get(path.suffix, "application/octet-stream"),
            cache="no-cache",
        )

    def do_POST(self):
        try:
            route = self._route()
            self._require_session(csrf=True)
            body = self._body_json()
            operations = {
                "/api/start": self.server.controller.start,
                "/api/confirm-start": self.server.controller.confirm_start,
                "/api/cancel-start": self.server.controller.cancel_start,
                "/api/close": self.server.controller.close_collection,
                "/api/stop": self.server.controller.stop,
                "/api/save": self.server.controller.save,
                "/api/discard": self.server.controller.discard,
                "/api/recover-e6": self.server.controller.recover_e6,
            }
            operation = operations.get(route)
            if operation is None:
                raise ApiError(HTTPStatus.NOT_FOUND, "接口不存在")
            self._send_json(HTTPStatus.ACCEPTED, operation(body))
        except ApiError as error:
            self._send_json(error.status, {"ok": False, "error": str(error)})
        except BrokenPipeError:
            return
        except Exception as error:
            traceback.print_exc()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": f"服务器错误：{error}"},
            )


def acquire_process_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".collection_web.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError("另一个网页采集服务已经占用该输出目录") from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\nstarted_at={time.time()}\n")
    handle.flush()
    return handle


def reject_running_legacy_collector():
    """Prevent the common case of the old CLI and web UI collecting together."""
    conflicts = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == os.getpid():
            continue
        try:
            arguments = [
                item.decode("utf-8", errors="replace")
                for item in (process_dir / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(Path(argument).name == "sol_data_collection.py" for argument in arguments):
            conflicts.append(int(process_dir.name))
    if conflicts:
        pid_list = ", ".join(str(pid) for pid in sorted(conflicts))
        raise ApiError(
            HTTPStatus.CONFLICT,
            f"检测到旧版 sol_data_collection.py 正在运行（PID {pid_list}）；"
            "请先停止旧脚本，再从网页开始采集",
        )


def build_real_collector(args):
    examples_dir = Path(args.examples_dir).resolve()
    sys.path.insert(0, str(examples_dir))
    from sol_data_collection import (
        E6RightHeadDataCollector,
        create_collection_config_for_quanta_x2,
        verify_saved_episode,
    )
    from x2robot import connect

    print(f"正在连接机器人 {args.robot_server}...", flush=True)
    robot = connect(f"x2://{args.robot_server}")
    model = robot.get_robot_model()
    if model != "quanta_x2":
        raise RuntimeError(f"网页采集器仅配置 Quanta X2，实际型号：{model}")
    collector_class = make_reviewable_collector_class(
        E6RightHeadDataCollector
    )
    collector = collector_class(
        robot=robot,
        output_dir=str(Path(args.output_dir).resolve()),
        target_hz=30,
        collection_config=create_collection_config_for_quanta_x2(),
        image_quality=95,
        downsample_joint_states=True,
        use_video_storage=False,
        keep_raw_data=False,
        raw_topic_storage=True,
        startup_readiness_timeout_seconds=None,
        verify_callback=verify_saved_episode,
    )
    print("✓ 机器人和数据集已就绪", flush=True)
    return collector


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="XR Pi0.5 局域网数据采集控制台"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--examples-dir",
        default="/home/xr/pi05/sdk_robot/examples",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/xr/pi05/sdk_robot/examples/sol_data_collection/task6",
    )
    parser.add_argument("--robot-server", default="localhost:50051")
    parser.add_argument("--cooldown-seconds", type=int, default=10)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="使用不连接机器人的内存演示采集器",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not (0 <= args.cooldown_seconds <= 3600):
        raise SystemExit("--cooldown-seconds 必须位于 0..3600")
    if not (1 <= args.port <= 65535):
        raise SystemExit("--port 无效")

    output_dir = Path(args.output_dir).resolve()
    process_lock = acquire_process_lock(output_dir)
    collector = DemoCollector() if args.demo else build_real_collector(args)
    controller = CollectionController(
        collector=collector,
        cooldown_seconds=args.cooldown_seconds,
        output_dir=output_dir,
        start_guard=None if args.demo else reject_running_legacy_collector,
    )
    server = CollectionHTTPServer(
        (args.host, args.port),
        RequestHandler,
        controller=controller,
        sessions=SessionStore(),
    )

    stop_once = threading.Event()

    def stop_handler(signum, frame):
        del signum, frame
        if stop_once.is_set():
            return
        stop_once.set()
        controller.begin_shutdown()
        threading.Thread(
            target=server.shutdown,
            daemon=True,
            name="WebServerShutdown",
        ).start()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    print(
        f"XR 数据采集网页已启动：http://{args.host}:{args.port}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        controller.shutdown()
        server.server_close()
        process_lock.close()
        print("XR 数据采集网页已停止", flush=True)
    return 1 if controller.restart_requested() else 0


if __name__ == "__main__":
    raise SystemExit(main())
