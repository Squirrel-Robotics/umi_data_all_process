#!/usr/bin/env python3
"""Convert UMI hand pose plus absolute finger states to LeRobot v2.1.

This converter implements the following *explicit* data contract:

* output rate: 10 Hz;
* state[t] is 30-D: left/right ``inverse(T[t-1]) @ T[t]`` encoded as
  translation3 + rotation-6D6, followed by left/right absolute finger6;
* action[t, k] is 30-D: left/right ``inverse(T[t]) @ T[t+k]`` followed by
  the left/right absolute finger targets at t+k, k=1..50;
* every action in the H=50 chunk uses the same current T[t] anchor;
* every real observation/video row is preserved; action slots beyond the
  episode boundary repeat the terminal target and are explicitly masked by
  ``action_is_pad`` (videos are never padded);
* the three observation videos are E6 RGB, left hand RGB, and right hand RGB.

Camera alignment follows the timestamp/strict-uniqueness contract from the
E6/camera handoff.  It never uses ``camera/*_frames_aligned.csv`` and never
assumes that equal video frame numbers are synchronized.  For each episode it:

1. reads E6 global timestamps from ``sync/e6_rgb_timing.csv``;
2. reads camera timestamps from the original ``camera/*_frames.csv`` files;
3. independently matches the nearest left/right camera frame by timestamp;
4. retains the longest contiguous, strictly monotonic, one-to-one interval for
   which both camera deltas are within the configured limit (100 ms by default);
5. samples that interval to 10 Hz and records all source frame indices/deltas.

Dual-hand transforms are reconstructed only from ``e6/hand_pose.csv``.  The
converter deliberately does not read ``controller_poses.csv`` or
``raw/*/head_pose.csv``.  Absolute six-joint Revo2 finger positions are decoded
from the serial packet logs and selected causally on the global timestamp axis.

The source directory is read-only.  Conversion is staged beside the target and
atomically published only after all episodes, parquet files, videos, metadata,
and audits have been written.  ``inspect`` performs no writes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
from typing import Any, Sequence

import numpy as np


DEFAULT_SOURCE = Path("/mnt/data/dzq/umi/data/最新100条")
DEFAULT_TARGET = Path("/mnt/data/dzq/umi/datasets/umi_lerobot_h50")
DEFAULT_REPO_ID = "local/umi_lerobot_h50"
OUTPUT_FPS = 10
ACTION_HORIZON = 50
DEFAULT_MAX_ALIGNMENT_MS = 100.0
DEFAULT_MAX_HAND_AGE_MS = 100.0
POSE_DIM = 9
STATE_DIM = 30
EPISODE_PATTERN = re.compile(r"^\d{8}_\d{6}_\d+_\d+$")
SIDES = ("left", "right")
JOINTS = ("thumb_flex", "thumb_aux", "index", "middle", "ring", "little")
VIDEO_KEYS = {
    "observation.images.head_rgb": "head",
    "observation.images.left_wrist_rgb": "left",
    "observation.images.right_wrist_rgb": "right",
}

POSE_COMPONENTS = (
    "x_m", "y_m", "z_m",
    "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
)
STATE_NAMES = [
    f"{side}_eef_body_delta_{component}"
    for side in SIDES for component in POSE_COMPONENTS
] + [
    f"{side}_hand_{joint}_absolute_deg"
    for side in SIDES for joint in JOINTS
]
ACTION_NAMES = [
    f"{side}_eef_shared_anchor_relative_{component}"
    for side in SIDES for component in POSE_COMPONENTS
] + [
    f"{side}_hand_{joint}_target_absolute_deg"
    for side in SIDES for joint in JOINTS
]


@dataclass(frozen=True)
class EpisodeFiles:
    episode: Path
    raw_e6: Path
    relative_hand_pose_csv: Path
    rgb_metainfo_csv: Path
    head_video: Path
    e6_timing_csv: Path
    nearest_csv: Path
    clock_model_json: Path
    left_frames_csv: Path
    right_frames_csv: Path
    left_video: Path
    right_video: Path
    left_hand_packets_csv: Path
    right_hand_packets_csv: Path


@dataclass(frozen=True)
class CameraSeries:
    side: str
    frame_index: np.ndarray
    sequence: np.ndarray
    timestamp_ns: np.ndarray
    capture_ok: np.ndarray
    source_rows: list[dict[str, str]]
    fps: float


@dataclass
class EpisodePlan:
    files: EpisodeFiles
    source_episode_index: int
    e6_rows: list[dict[str, str]]
    source_nearest_rows: list[dict[str, Any]]
    recomputed_nearest_rows: list[dict[str, Any]]
    strict_rows: list[dict[str, Any]]
    e6_indices: np.ndarray
    left_frame_indices: np.ndarray
    right_frame_indices: np.ndarray
    hand_pose_timestamps_ns: np.ndarray
    state: np.ndarray
    action: np.ndarray
    action_is_pad: np.ndarray
    sample_valid_h50: np.ndarray
    source_fps: int
    source_stride: int
    camera_ratio: int
    left_offset: int
    right_offset: int
    strict_run_start: int
    strict_run_end_exclusive: int
    output_count: int
    quality: dict[str, Any]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(command: Sequence[str]) -> None:
    completed = subprocess.run(
        list(command), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def locate_episode_files(episode: Path) -> EpisodeFiles:
    raw_dirs = sorted(
        path.parent
        for path in (episode / "e6" / "raw").glob("*/rgb.mp4")
        if path.is_file()
    )
    if len(raw_dirs) != 1:
        raise ValueError(
            f"{episode.name}: expected exactly one e6/raw/*/rgb.mp4, found {len(raw_dirs)}"
        )
    raw = raw_dirs[0]
    files = EpisodeFiles(
        episode=episode,
        raw_e6=raw,
        relative_hand_pose_csv=episode / "e6" / "hand_pose.csv",
        rgb_metainfo_csv=raw / "rgb_metainfo.csv",
        head_video=raw / "rgb.mp4",
        e6_timing_csv=episode / "sync" / "e6_rgb_timing.csv",
        nearest_csv=episode / "sync" / "frame_associations.csv",
        clock_model_json=episode / "sync" / "clock_model.json",
        left_frames_csv=episode / "camera" / "cam0_left_hand_frames.csv",
        right_frames_csv=episode / "camera" / "cam1_right_hand_frames.csv",
        left_video=episode / "camera" / "cam0_left_hand.mkv",
        right_video=episode / "camera" / "cam1_right_hand.mkv",
        left_hand_packets_csv=episode / "serial" / "left_rx_packets.csv",
        right_hand_packets_csv=episode / "serial" / "right_rx_packets.csv",
    )
    missing = [path for path in files.__dict__.values() if isinstance(path, Path) and not path.exists()]
    if missing:
        raise FileNotFoundError(f"{episode.name}: missing required files: {missing}")
    return files


def discover_episodes(
    root: Path, only: Sequence[str], exclude: Sequence[str]
) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    episodes = sorted(
        path for path in root.iterdir()
        if path.is_dir() and EPISODE_PATTERN.fullmatch(path.name)
    )
    only_set = set(only)
    exclude_set = set(exclude)
    if only_set & exclude_set:
        raise ValueError(f"episode present in both --only and --exclude: {sorted(only_set & exclude_set)}")
    available = {path.name for path in episodes}
    unknown = (only_set | exclude_set) - available
    if unknown:
        raise ValueError(f"unknown episode IDs: {sorted(unknown)}")
    if only_set:
        episodes = [path for path in episodes if path.name in only_set]
    if exclude_set:
        episodes = [path for path in episodes if path.name not in exclude_set]
    if not episodes:
        raise ValueError("no source episodes selected")
    return episodes


def measured_fps(timestamp_ns: np.ndarray) -> float:
    if len(timestamp_ns) < 2:
        raise ValueError("need at least two timestamps")
    elapsed = int(timestamp_ns[-1]) - int(timestamp_ns[0])
    if elapsed <= 0 or np.any(np.diff(timestamp_ns) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return (len(timestamp_ns) - 1) * 1e9 / elapsed


def parse_camera(path: Path, side: str) -> CameraSeries:
    rows = read_csv(path)
    frame_index = np.asarray([int(row["frame_index"]) for row in rows], dtype=np.int64)
    sequence = np.asarray([int(row["v4l2_sequence"]) for row in rows], dtype=np.int64)
    timestamp_ns = np.asarray([int(row["host_frame_time_ns"]) for row in rows], dtype=np.int64)
    capture_ok = np.asarray([row["capture_status"] == "ok" for row in rows], dtype=bool)
    if not np.array_equal(frame_index, np.arange(len(rows), dtype=np.int64)):
        raise ValueError(f"{side}: camera frame_index is not contiguous from zero")
    if len(sequence) > 1 and not np.all(np.diff(sequence) == 1):
        raise ValueError(f"{side}: V4L2 sequence contains a gap")
    if not np.all(capture_ok):
        raise ValueError(f"{side}: camera capture_status contains non-ok rows")
    return CameraSeries(
        side=side,
        frame_index=frame_index,
        sequence=sequence,
        timestamp_ns=timestamp_ns,
        capture_ok=capture_ok,
        source_rows=rows,
        fps=measured_fps(timestamp_ns),
    )


def nearest_indices(sorted_timestamp_ns: np.ndarray, query_ns: np.ndarray) -> np.ndarray:
    """Return the nearest row index for each query in a sorted timestamp array."""
    right = np.searchsorted(sorted_timestamp_ns, query_ns, side="left")
    right = np.clip(right, 0, len(sorted_timestamp_ns) - 1)
    left = np.clip(right - 1, 0, len(sorted_timestamp_ns) - 1)
    choose_left = (
        np.abs(sorted_timestamp_ns[left] - query_ns)
        <= np.abs(sorted_timestamp_ns[right] - query_ns)
    )
    return np.where(choose_left, left, right).astype(np.int64)


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    return runs


def transforms_from_pose7(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64).copy()
    if position.shape[-1] != 3 or quaternion.shape[-1] != 4:
        raise ValueError("pose position/quaternion shape is invalid")
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("pose contains NaN or Inf")
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("pose contains a zero-norm quaternion")
    quaternion /= norm
    x, y, z, w = quaternion.T
    rotation = np.empty((len(quaternion), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - z * w)
    rotation[:, 0, 2] = 2 * (x * z + y * w)
    rotation[:, 1, 0] = 2 * (x * y + z * w)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - x * w)
    rotation[:, 2, 0] = 2 * (x * z - y * w)
    rotation[:, 2, 1] = 2 * (y * z + x * w)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    transforms = np.broadcast_to(np.eye(4), (len(position), 4, 4)).copy()
    transforms[:, :3, :3] = rotation
    transforms[:, :3, 3] = position
    return transforms


def reconstruct_hand_pose(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Reconstruct arbitrary-origin absolute trajectories from adjacent deltas.

    ``e6/hand_pose.csv`` already stores ``inverse(T[i-1]) @ T[i]`` for each
    hand.  Cumulative composition is sufficient for shared-anchor actions;
    the arbitrary identity chosen at the beginning of a continuous segment
    cancels from every relative transform.
    """
    timestamps = np.asarray([int(row["timestamp_ns"]) for row in rows], dtype=np.int64)
    frame_number = np.asarray([int(row["frame_number"]) for row in rows], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("e6/hand_pose.csv timestamps are not strictly increasing")
    trajectories: dict[str, np.ndarray] = {}
    transition_valid: dict[str, np.ndarray] = {}
    segment_ids: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {"source_rows": len(rows)}
    for side in SIDES:
        valid = np.asarray(
            [bool(int(row[f"{side}_relative_valid"])) for row in rows], dtype=bool
        )
        position = np.asarray([
            [float(row[f"{side}_local_dp{axis}"]) for axis in "xyz"]
            for row in rows
        ], dtype=np.float64)
        quaternion = np.asarray([
            [float(row[f"{side}_local_dq{axis}"]) for axis in "xyzw"]
            for row in rows
        ], dtype=np.float64)
        delta = transforms_from_pose7(position, quaternion)
        absolute = np.broadcast_to(np.eye(4), (len(rows), 4, 4)).copy()
        segment = np.zeros(len(rows), dtype=np.int64)
        effective_valid = np.zeros(len(rows), dtype=bool)
        for index in range(1, len(rows)):
            declared_previous = rows[index]["previous_frame_number"]
            links_previous = (
                declared_previous != ""
                and int(declared_previous) == int(frame_number[index - 1])
            )
            if valid[index] and links_previous:
                absolute[index] = absolute[index - 1] @ delta[index]
                segment[index] = segment[index - 1]
                effective_valid[index] = True
            else:
                # Start a new arbitrary-origin segment.  The invalid boundary
                # is excluded below, so no state/action crosses this reset.
                absolute[index] = np.eye(4)
                segment[index] = segment[index - 1] + 1
        trajectories[side] = absolute
        transition_valid[side] = effective_valid
        segment_ids[side] = segment
        metrics[f"{side}_declared_valid_rows"] = int(valid.sum())
        metrics[f"{side}_effective_valid_rows"] = int(effective_valid.sum())
        metrics[f"{side}_segments"] = int(segment[-1] + 1)
    pair_valid = transition_valid["left"] & transition_valid["right"]
    metrics["pair_effective_valid_rows"] = int(pair_valid.sum())
    return timestamps, trajectories, pair_valid, metrics


def decode_absolute_hand_packets(
    path: Path, side: str
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rows = read_csv(path)
    selected: list[dict[str, str]] = []
    decoded: list[list[int]] = []
    for row in rows:
        if row.get("packet_type_name") != "HAND_STATE_FRAME" or row.get("payload_status") != "full":
            continue
        values = [int(value) for value in row["decoded_values"].split(";")]
        if len(values) != 25:
            raise ValueError(f"{path}: HAND_STATE_FRAME does not contain 25 values")
        selected.append(row)
        decoded.append(values)
    if not selected:
        raise ValueError(f"{path}: no full HAND_STATE_FRAME packets")
    timestamp = np.asarray([int(row["host_rx_time_ns"]) for row in selected], dtype=np.int64)
    if np.any(np.diff(timestamp) <= 0):
        raise ValueError(f"{path}: hand packet timestamps are not strictly increasing")
    # First six decoded values are absolute joint position in 0.1 degree.
    position_deg = np.asarray(decoded, dtype=np.float64)[:, :6] / 10.0
    return timestamp, position_deg, {
        f"{side}_packet_csv_rows": len(rows),
        f"{side}_full_state_rows": len(selected),
    }


def sample_absolute_hand(
    source_timestamp_ns: np.ndarray,
    source_position_deg: np.ndarray,
    query_timestamp_ns: np.ndarray,
    max_age_ns: int = int(DEFAULT_MAX_HAND_AGE_MS * 1_000_000),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = np.searchsorted(source_timestamp_ns, query_timestamp_ns, side="right") - 1
    causal_valid = selected >= 0
    safe = np.maximum(selected, 0)
    chosen_timestamp = source_timestamp_ns[safe]
    age_ns = query_timestamp_ns - chosen_timestamp
    fresh = causal_valid & (age_ns >= 0) & (age_ns <= max_age_ns)
    position = source_position_deg[safe].copy()
    position[~causal_valid] = np.nan
    return position, fresh, age_ns, selected


def invert(transforms: np.ndarray) -> np.ndarray:
    result = np.zeros_like(transforms)
    transpose = transforms[..., :3, :3].swapaxes(-1, -2)
    result[..., :3, :3] = transpose
    result[..., :3, 3] = -np.einsum("...ij,...j->...i", transpose, transforms[..., :3, 3])
    result[..., 3, 3] = 1.0
    return result


def encode_relative(transforms: np.ndarray) -> np.ndarray:
    rotation = transforms[..., :3, :3]
    rot6d = np.concatenate([rotation[..., :, 0], rotation[..., :, 1]], axis=-1)
    return np.concatenate([transforms[..., :3, 3], rot6d], axis=-1)


def make_plan(
    files: EpisodeFiles,
    source_episode_index: int,
    max_alignment_ns: int,
    max_hand_age_ns: int,
) -> EpisodePlan:
    clock = load_json(files.clock_model_json)
    if clock.get("status") != "ok":
        raise ValueError(f"{files.episode.name}: clock_model.status != ok")

    timing = read_csv(files.e6_timing_csv)
    meta = read_csv(files.rgb_metainfo_csv)
    raw_hand_pose = read_csv(files.relative_hand_pose_csv)
    source_nearest_rows = read_csv(files.nearest_csv)
    if len(timing) != len(meta):
        raise ValueError(
            f"{files.episode.name}: E6 timing/meta row counts differ: "
            f"{len(timing)}/{len(meta)}"
        )
    e6_index = np.asarray([int(row["e6_frame_index"]) for row in timing], dtype=np.int64)
    if not np.array_equal(e6_index, np.arange(len(timing), dtype=np.int64)):
        raise ValueError(f"{files.episode.name}: E6 frame index is not contiguous")
    if any(row["mapping_status"] != "ok" for row in timing):
        raise ValueError(f"{files.episode.name}: E6 mapping_status contains non-ok rows")
    e6_global = np.asarray([int(row["global_ts_ns"]) for row in timing], dtype=np.int64)
    e6_mid = np.asarray([int(row["e6_mid_exposure_utc_ns"]) for row in timing], dtype=np.int64)
    meta_index = np.asarray([int(row["frame_index"]) for row in meta], dtype=np.int64)
    meta_mid = np.asarray([int(row["mid_exposure_utc_ns"]) for row in meta], dtype=np.int64)
    (
        hand_pose_timestamp,
        raw_hand_trajectories,
        raw_pair_pose_valid,
        hand_pose_metrics,
    ) = reconstruct_hand_pose(raw_hand_pose)
    hand_pose_index_by_timestamp: dict[int, int] = {}
    for index, timestamp_value in enumerate(hand_pose_timestamp):
        timestamp = int(timestamp_value)
        if timestamp in hand_pose_index_by_timestamp:
            raise ValueError(
                f"{files.episode.name}: duplicate hand_pose timestamp {timestamp}"
            )
        hand_pose_index_by_timestamp[timestamp] = index
    hand_pose_rows = np.asarray(
        [hand_pose_index_by_timestamp.get(int(timestamp), -1) for timestamp in e6_mid],
        dtype=np.int64,
    )
    hand_pose_matched = hand_pose_rows >= 0
    hand_trajectories: dict[str, np.ndarray] = {}
    for side in SIDES:
        trajectory = np.broadcast_to(np.eye(4), (len(e6_mid), 4, 4)).copy()
        trajectory[hand_pose_matched] = raw_hand_trajectories[side][
            hand_pose_rows[hand_pose_matched]
        ]
        hand_trajectories[side] = trajectory
    pair_pose_valid = np.zeros(len(e6_mid), dtype=bool)
    pair_pose_valid[hand_pose_matched] = raw_pair_pose_valid[
        hand_pose_rows[hand_pose_matched]
    ]
    if not np.array_equal(meta_index, e6_index):
        raise ValueError(f"{files.episode.name}: rgb_metainfo frame_index mismatch")
    if not np.array_equal(meta_mid, e6_mid):
        raise ValueError(f"{files.episode.name}: RGB/timing timestamps do not match exactly")

    left_hand_time, left_hand_position, left_hand_metrics = decode_absolute_hand_packets(
        files.left_hand_packets_csv, "left"
    )
    right_hand_time, right_hand_position, right_hand_metrics = decode_absolute_hand_packets(
        files.right_hand_packets_csv, "right"
    )
    left_hand, left_hand_fresh, left_hand_age, left_hand_source_row = sample_absolute_hand(
        left_hand_time, left_hand_position, e6_global, max_hand_age_ns
    )
    right_hand, right_hand_fresh, right_hand_age, right_hand_source_row = sample_absolute_hand(
        right_hand_time, right_hand_position, e6_global, max_hand_age_ns
    )
    absolute_hands12 = np.concatenate([left_hand, right_hand], axis=1)
    pair_hand_fresh = left_hand_fresh & right_hand_fresh

    e6_intervals_ns = np.diff(e6_mid)
    e6_fps_float = 1e9 / float(np.median(e6_intervals_ns))
    e6_average_fps = measured_fps(e6_mid)
    source_fps = int(round(e6_fps_float))
    if abs(e6_fps_float - source_fps) > 0.25 or source_fps % OUTPUT_FPS:
        raise ValueError(
            f"{files.episode.name}: unsupported median E6 rate {e6_fps_float:.6f} Hz; "
            f"must be an integer multiple of {OUTPUT_FPS} Hz"
        )
    source_stride = source_fps // OUTPUT_FPS
    nominal_interval_ns = int(round(1e9 / source_fps))
    interval_tolerance_ns = max(2_000_000, int(round(nominal_interval_ns * 0.10)))
    e6_interval_ok = np.ones(len(e6_mid), dtype=bool)
    e6_interval_ok[1:] = (
        np.abs(e6_intervals_ns - nominal_interval_ns) <= interval_tolerance_ns
    )

    left = parse_camera(files.left_frames_csv, "left")
    right = parse_camera(files.right_frames_csv, "right")
    camera_fps = statistics.mean((left.fps, right.fps))
    camera_ratio_float = camera_fps / e6_fps_float
    camera_ratio = int(round(camera_ratio_float))
    if camera_ratio <= 0 or abs(camera_ratio_float - camera_ratio) > 0.05:
        raise ValueError(
            f"{files.episode.name}: camera/E6 FPS ratio is not near an integer: "
            f"{camera_ratio_float:.6f}"
        )

    # Match each E6 exposure to the nearest frame on each wrist-camera clock.
    # Do not force a constant frame-number phase: the independent camera
    # clocks can drift enough that a fixed offset discards otherwise valid
    # portions of an episode. Strict monotonicity below prevents frame reuse.
    left_nearest_row = nearest_indices(left.timestamp_ns, e6_global)
    right_nearest_row = nearest_indices(right.timestamp_ns, e6_global)
    left_nearest = left.frame_index[left_nearest_row]
    right_nearest = right.frame_index[right_nearest_row]
    left_delta = left.timestamp_ns[left_nearest_row] - e6_global
    right_delta = right.timestamp_ns[right_nearest_row] - e6_global
    left_monotonic = np.ones(len(e6_index), dtype=bool)
    right_monotonic = np.ones(len(e6_index), dtype=bool)
    left_monotonic[1:] = np.diff(left_nearest) > 0
    right_monotonic[1:] = np.diff(right_nearest) > 0
    left_offset = int(round(float(np.median(left_nearest - camera_ratio * e6_index))))
    right_offset = int(round(float(np.median(right_nearest - camera_ratio * e6_index))))
    strict_mask = (
        left_monotonic & right_monotonic
        & (np.abs(left_delta) <= max_alignment_ns)
        & (np.abs(right_delta) <= max_alignment_ns)
        & pair_pose_valid
        & pair_hand_fresh
        & e6_interval_ok
    )
    runs = true_runs(strict_mask)
    if not runs:
        raise ValueError(f"{files.episode.name}: no strict three-camera alignment run")
    run_start, run_end = max(runs, key=lambda item: item[1] - item[0])
    strict_length = run_end - run_start
    sample_rows = np.arange(run_start, run_end, source_stride, dtype=np.int64)
    # One sample is needed before the first output state.  Every remaining
    # aligned observation is preserved.  Future action slots past the episode
    # boundary repeat the terminal target and carry an explicit pad mask; no
    # video frame is ever synthesized or repeated.
    output_count = len(sample_rows) - 1
    if output_count <= 0:
        raise ValueError(
            f"{files.episode.name}: strict run has only {len(sample_rows)} 10 Hz points; "
            "need at least 2"
        )
    anchor_positions = np.arange(1, 1 + output_count, dtype=np.int64)
    previous_rows = sample_rows[anchor_positions - 1]
    anchor_rows = sample_rows[anchor_positions]
    requested_future_positions = anchor_positions[:, None] + np.arange(
        1, ACTION_HORIZON + 1, dtype=np.int64
    )[None, :]
    action_is_pad = requested_future_positions >= len(sample_rows)
    future_positions = np.minimum(requested_future_positions, len(sample_rows) - 1)
    target_rows = sample_rows[future_positions]
    sample_valid_h50 = ~np.any(action_is_pad, axis=1)
    state_eef: list[np.ndarray] = []
    action_eef: list[np.ndarray] = []
    for side in SIDES:
        absolute = hand_trajectories[side]
        state_eef.append(
            encode_relative(invert(absolute[previous_rows]) @ absolute[anchor_rows])
        )
        action_eef.append(
            encode_relative(
                invert(absolute[anchor_rows])[:, None] @ absolute[target_rows]
            )
        )
    state = np.concatenate(
        [state_eef[0], state_eef[1], absolute_hands12[anchor_rows]], axis=1
    ).astype(np.float32)
    action = np.concatenate(
        [action_eef[0], action_eef[1], absolute_hands12[target_rows]], axis=2
    ).astype(np.float32)
    if state.shape != (output_count, STATE_DIM):
        raise AssertionError("state shape changed")
    if action.shape != (output_count, ACTION_HORIZON, STATE_DIM):
        raise AssertionError("action shape changed")
    if action_is_pad.shape != (output_count, ACTION_HORIZON):
        raise AssertionError("action pad-mask shape changed")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{files.episode.name}: state/action contains NaN or Inf")
    # k=1 is the next adjacent 10 Hz delta.  It should equal the next state,
    # except for the last emitted row whose next state is deliberately not an
    # emitted observation.
    if output_count > 1 and not np.allclose(action[:-1, 0], state[1:], atol=2e-6):
        raise AssertionError("shared-anchor k=1 does not match next adjacent state")

    recomputed_nearest_rows: list[dict[str, Any]] = []
    previous_left = previous_right = None
    for row in range(len(e6_index)):
        left_frame = int(left_nearest[row])
        right_frame = int(right_nearest[row])
        recomputed_nearest_rows.append({
            "e6_frame_index": int(e6_index[row]),
            "e6_global_ts_ns": int(e6_global[row]),
            "left_frame_index": left_frame,
            "left_signed_delta_ns": int(left_delta[row]),
            "left_reused": int(left_frame == previous_left),
            "left_within_limit": int(abs(int(left_delta[row])) <= max_alignment_ns),
            "right_frame_index": right_frame,
            "right_signed_delta_ns": int(right_delta[row]),
            "right_reused": int(right_frame == previous_right),
            "right_within_limit": int(abs(int(right_delta[row])) <= max_alignment_ns),
        })
        previous_left, previous_right = left_frame, right_frame

    output_anchor_set = set(anchor_rows.tolist())
    action_target_set = set(target_rows.reshape(-1).tolist())
    strict_rows: list[dict[str, Any]] = []
    for position, row in enumerate(sample_rows):
        strict_rows.append({
            "sample_grid_index": position,
            "is_output_anchor": int(int(row) in output_anchor_set),
            "is_action_target": int(int(row) in action_target_set),
            "e6_frame_index": int(e6_index[row]),
            "hand_pose_timestamp_ns": int(e6_mid[row]),
            "e6_global_ts_ns": int(e6_global[row]),
            "left_frame_index": int(left_nearest[row]),
            "left_signed_delta_ns": int(left_delta[row]),
            "right_frame_index": int(right_nearest[row]),
            "right_signed_delta_ns": int(right_delta[row]),
            "left_hand_packet_row": int(left_hand_source_row[row]),
            "left_hand_age_ns": int(left_hand_age[row]),
            "right_hand_packet_row": int(right_hand_source_row[row]),
            "right_hand_age_ns": int(right_hand_age[row]),
        })

    anchor_e6 = e6_index[anchor_rows]
    quality = {
        "episode_id": files.episode.name,
        "clock_model_status": clock.get("status"),
        "e6_rgb_rows": len(timing),
        "hand_pose_source_rows": len(raw_hand_pose),
        "hand_pose_rows_matched_to_rgb": int(hand_pose_matched.sum()),
        "rgb_rows_missing_hand_pose": int((~hand_pose_matched).sum()),
        "rgb_frame_indices_missing_hand_pose": e6_index[~hand_pose_matched].tolist(),
        "hand_pose_rows_outside_rgb_timeline": int(
            len(raw_hand_pose) - hand_pose_matched.sum()
        ),
        "hand_pose_quality": hand_pose_metrics,
        "absolute_hand_quality": {
            **left_hand_metrics,
            **right_hand_metrics,
            "max_hand_age_ns": max_hand_age_ns,
            "pair_fresh_within_limit_rows": int(pair_hand_fresh.sum()),
            "left_max_fresh_age_ns": int(left_hand_age[left_hand_fresh].max(initial=0)),
            "right_max_fresh_age_ns": int(right_hand_age[right_hand_fresh].max(initial=0)),
        },
        "strict_max_delta_ns": max_alignment_ns,
        "e6_fps_measured": e6_fps_float,
        "e6_fps_full_interval_average": e6_average_fps,
        "e6_nominal_interval_ns": nominal_interval_ns,
        "e6_interval_tolerance_ns": interval_tolerance_ns,
        "e6_interval_discontinuity_count": int((~e6_interval_ok).sum()),
        "e6_interval_discontinuity_after_frame_indices": e6_index[~e6_interval_ok].tolist(),
        "camera_left_fps_measured": left.fps,
        "camera_right_fps_measured": right.fps,
        "camera_to_e6_integer_ratio": camera_ratio,
        "left_median_frame_offset": left_offset,
        "right_median_frame_offset": right_offset,
        "strict_run_first_e6_frame": int(e6_index[run_start]),
        "strict_run_last_e6_frame": int(e6_index[run_end - 1]),
        "strict_run_rows": strict_length,
        "strict_left_max_abs_delta_ns": int(np.max(np.abs(left_delta[run_start:run_end]))),
        "strict_right_max_abs_delta_ns": int(np.max(np.abs(right_delta[run_start:run_end]))),
        "output_fps": OUTPUT_FPS,
        "output_source_stride": source_stride,
        "action_horizon": ACTION_HORIZON,
        "action_horizon_seconds": ACTION_HORIZON / OUTPUT_FPS,
        "tail_policy": "preserve observations/videos; repeat terminal action target only in explicitly masked slots",
        "output_rows": output_count,
        "sample_valid_h50_rows": int(sample_valid_h50.sum()),
        "terminal_padded_anchor_rows": int((~sample_valid_h50).sum()),
        "terminal_padded_action_slots": int(action_is_pad.sum()),
        "output_first_e6_frame": int(anchor_e6[0]),
        "output_last_e6_frame": int(anchor_e6[-1]),
    }
    return EpisodePlan(
        files=files,
        source_episode_index=source_episode_index,
        e6_rows=timing,
        source_nearest_rows=source_nearest_rows,
        recomputed_nearest_rows=recomputed_nearest_rows,
        strict_rows=strict_rows,
        e6_indices=anchor_e6,
        left_frame_indices=left_nearest[anchor_rows],
        right_frame_indices=right_nearest[anchor_rows],
        hand_pose_timestamps_ns=e6_mid[anchor_rows],
        state=state,
        action=action,
        action_is_pad=action_is_pad,
        sample_valid_h50=sample_valid_h50,
        source_fps=source_fps,
        source_stride=source_stride,
        camera_ratio=camera_ratio,
        left_offset=left_offset,
        right_offset=right_offset,
        strict_run_start=run_start,
        strict_run_end_exclusive=run_end,
        output_count=output_count,
        quality=quality,
    )


def resize_filter(width: int, height: int, mode: str) -> str:
    if mode == "stretch":
        return f"scale={width}:{height}"
    if mode == "letterbox":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        )
    if mode == "crop":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
    raise ValueError(mode)


def encode_frame_selection(
    source: Path,
    target: Path,
    frame_indices: np.ndarray,
    width: int,
    height: int,
    resize_mode: str,
    crf: int,
    source_crop: str | None = None,
) -> dict[str, Any]:
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if (
        frame_indices.ndim != 1
        or len(frame_indices) == 0
        or np.any(frame_indices < 0)
        or np.any(np.diff(frame_indices) <= 0)
    ):
        raise ValueError(f"video frame selection must be non-empty and strictly increasing: {source}")
    length = len(frame_indices)
    target.parent.mkdir(parents=True, exist_ok=True)
    select_expression = "+".join(f"eq(n\\,{int(index)})" for index in frame_indices)
    filters = [f"select='{select_expression}'"]
    if source_crop is not None:
        filters.append(source_crop)
    filters.extend((
        f"setpts=N/({OUTPUT_FPS}*TB)",
        resize_filter(width, height, resize_mode),
    ))
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-an", "-vf", ",".join(filters), "-c:v", "libx264", "-preset", "medium",
        "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(OUTPUT_FPS),
        "-vsync", "cfr", "-movflags", "+faststart", str(target),
    ])
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_packets,pix_fmt",
        "-of", "json", str(target),
    ], text=True))["streams"][0]
    if (
        int(probe["width"]) != width
        or int(probe["height"]) != height
        or probe["avg_frame_rate"] != f"{OUTPUT_FPS}/1"
        or int(probe["nb_read_packets"]) != length
    ):
        raise RuntimeError(f"encoded video validation failed: {target}: {probe}")
    return {
        "frames": length,
        "width": width,
        "height": height,
        "fps": OUTPUT_FPS,
        "codec": probe["codec_name"],
        "pix_fmt": probe["pix_fmt"],
        "size_bytes": target.stat().st_size,
        "sha256": sha256(target),
    }


def numeric_stats(value: np.ndarray) -> dict[str, np.ndarray]:
    array = np.asarray(value)
    return {
        # LeRobot's stats aggregator requires every entry to be an ndarray
        # with at least one dimension.  Reductions over scalar features such
        # as timestamp otherwise return NumPy scalar objects.
        "min": np.atleast_1d(np.min(array, axis=0)),
        "max": np.atleast_1d(np.max(array, axis=0)),
        "mean": np.atleast_1d(np.mean(array, axis=0)),
        "std": np.atleast_1d(np.std(array, axis=0)),
        "count": np.asarray([len(array)], dtype=np.int64),
    }


def video_stats(path: Path, expected_frames: int, sample_count: int = 64) -> dict[str, np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    wanted = set(
        np.linspace(0, expected_frames - 1, min(sample_count, expected_frames))
        .round().astype(int).tolist()
    )
    samples: list[np.ndarray] = []
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if count in wanted:
            samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count += 1
    capture.release()
    if count != expected_frames or len(samples) != len(wanted):
        raise RuntimeError(f"video decode count mismatch: {path}: {count}/{expected_frames}")
    values = np.stack(samples).astype(np.float64) / 255.0
    return {
        "min": values.min(axis=(0, 1, 2))[:, None, None],
        "max": values.max(axis=(0, 1, 2))[:, None, None],
        "mean": values.mean(axis=(0, 1, 2))[:, None, None],
        "std": values.std(axis=(0, 1, 2))[:, None, None],
        "count": np.asarray([len(values)], dtype=np.int64),
    }


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty audit CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_features(width: int, height: int) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32", "shape": (STATE_DIM,), "names": [STATE_NAMES],
        },
        "action": {
            "dtype": "float32", "shape": (ACTION_HORIZON, STATE_DIM),
            "names": [
                [f"future_step_{step}" for step in range(1, ACTION_HORIZON + 1)],
                ACTION_NAMES,
            ],
        },
        "action_is_pad": {
            "dtype": "bool", "shape": (ACTION_HORIZON,),
            "names": [f"future_step_{step}" for step in range(1, ACTION_HORIZON + 1)],
        },
        "sample_valid_h50": {
            "dtype": "int64", "shape": (1,), "names": ["all_50_targets_real"],
        },
        "source.e6_frame_index": {
            "dtype": "int64", "shape": (1,), "names": ["frame_index"],
        },
        "source.hand_pose_timestamp_ns": {
            "dtype": "int64", "shape": (1,), "names": ["timestamp_ns"],
        },
    }
    for key in VIDEO_KEYS:
        features[key] = {
            "dtype": "video", "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def write_parquet(
    metadata: Any,
    staging: Path,
    plan: EpisodePlan,
    episode_index: int,
    global_start: int,
) -> tuple[Path, dict[str, np.ndarray]]:
    import datasets
    from lerobot.common.datasets.utils import get_hf_features_from_features

    length = plan.output_count
    arrays = {
        "observation.state": plan.state,
        "action": plan.action,
        "action_is_pad": plan.action_is_pad,
        "sample_valid_h50": plan.sample_valid_h50.astype(np.int64)[:, None],
        "source.e6_frame_index": plan.e6_indices[:, None],
        "source.hand_pose_timestamp_ns": plan.hand_pose_timestamps_ns[:, None],
        "timestamp": np.arange(length, dtype=np.float32) / np.float32(OUTPUT_FPS),
        "frame_index": np.arange(length, dtype=np.int64),
        "episode_index": np.full(length, episode_index, dtype=np.int64),
        "index": np.arange(global_start, global_start + length, dtype=np.int64),
        "task_index": np.zeros(length, dtype=np.int64),
    }
    dataset = datasets.Dataset.from_dict(
        arrays,
        features=get_hf_features_from_features(metadata.features),
        split="train",
    )
    target = staging / metadata.get_data_file_path(episode_index)
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(target)
    return target, arrays


def inspect(args: argparse.Namespace) -> int:
    episodes = discover_episodes(args.source.resolve(), args.only, args.exclude)
    print(f"selected_episodes={len(episodes)}")
    if args.expected_episodes and len(episodes) != args.expected_episodes:
        print(
            f"WARNING: expected {args.expected_episodes} episodes but selected {len(episodes)}; "
            "convert will refuse until --expected-episodes or selection is corrected"
        )
    plans: list[EpisodePlan] = []
    for index, episode in enumerate(episodes):
        plan = make_plan(
            locate_episode_files(episode),
            index,
            max_alignment_ns=int(round(args.max_alignment_ms * 1_000_000)),
            max_hand_age_ns=int(round(args.max_hand_age_ms * 1_000_000)),
        )
        plans.append(plan)
        print(
            f"[{index + 1}/{len(episodes)}] {episode.name}: "
            f"E6={plan.source_fps}Hz stride={plan.source_stride} "
            f"ratio={plan.camera_ratio} offsets=L{plan.left_offset}/R{plan.right_offset} "
            f"strict={plan.quality['strict_run_rows']} output={plan.output_count}"
        )
    summary = {
        "mode": "inspect-only-no-writes",
        "source": str(args.source.resolve()),
        "target_not_created": str(args.target.resolve()),
        "selected_episodes": len(plans),
        "expected_episodes": args.expected_episodes,
        "output_fps": OUTPUT_FPS,
        "action_horizon": ACTION_HORIZON,
        "max_alignment_ms": args.max_alignment_ms,
        "max_hand_age_ms": args.max_hand_age_ms,
        "state_shape": [STATE_DIM],
        "action_shape": [ACTION_HORIZON, STATE_DIM],
        "total_output_rows_planned": sum(plan.output_count for plan in plans),
        "total_full_real_h50_rows_planned": int(
            sum(plan.sample_valid_h50.sum() for plan in plans)
        ),
        "episodes_with_zero_full_real_h50_rows": [
            plan.files.episode.name
            for plan in plans
            if not bool(plan.sample_valid_h50.any())
        ],
    }
    if not args.compact:
        summary["episodes"] = [plan.quality for plan in plans]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def convert(args: argparse.Namespace) -> int:
    if args.confirm != "CREATE_LEROBOT_DATASET":
        raise ValueError("conversion requires --confirm CREATE_LEROBOT_DATASET")
    source = args.source.resolve()
    target = args.target.resolve()
    episodes = discover_episodes(source, args.only, args.exclude)
    if args.expected_episodes and len(episodes) != args.expected_episodes:
        raise ValueError(
            f"expected {args.expected_episodes} selected episodes, found {len(episodes)}"
        )
    if target.exists():
        if (
            args.replace_empty_target
            and target.is_dir()
            and not any(target.iterdir())
        ):
            # This flag never permits replacing data.  It only removes an
            # already-created, completely empty target directory so atomic
            # publication can claim the requested path later.
            target.rmdir()
        else:
            raise FileExistsError(
                f"refusing to overwrite existing target: {target}; "
                "--replace-empty-target is accepted only for an empty directory"
            )
    if not target.parent.is_dir():
        raise FileNotFoundError(f"target parent does not exist: {target.parent}")

    plans = [
        make_plan(
            locate_episode_files(episode),
            index,
            max_alignment_ns=int(round(args.max_alignment_ms * 1_000_000)),
            max_hand_age_ns=int(round(args.max_hand_age_ms * 1_000_000)),
        )
        for index, episode in enumerate(episodes)
    ]
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import write_info, write_stats

    staging_parent = Path(tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=target.parent))
    staging = staging_parent / "dataset"
    try:
        metadata = LeRobotDatasetMetadata.create(
            repo_id=args.repo_id,
            fps=OUTPUT_FPS,
            root=staging,
            robot_type="umi_dual_hand_pose_body_delta_rot6d_hand30_h50",
            features=build_features(args.video_width, args.video_height),
            use_videos=True,
        )
        metadata.add_task(args.task)

        video_records: dict[tuple[int, str], dict[str, Any]] = {}
        jobs: dict[Any, tuple[int, str]] = {}
        with ThreadPoolExecutor(max_workers=max(1, args.video_workers)) as pool:
            for episode_index, plan in enumerate(plans):
                sources = {
                    "head": (
                        plan.files.head_video, plan.e6_indices,
                        "crop=1600:1200:1600:0",
                    ),
                    "left": (
                        plan.files.left_video, plan.left_frame_indices,
                        None,
                    ),
                    "right": (
                        plan.files.right_video, plan.right_frame_indices,
                        None,
                    ),
                }
                for key, role in VIDEO_KEYS.items():
                    video_target = staging / metadata.get_video_file_path(episode_index, key)
                    source_video, frame_indices, source_crop = sources[role]
                    future = pool.submit(
                        encode_frame_selection,
                        source_video, video_target, frame_indices,
                        args.video_width, args.video_height, args.resize_mode, args.crf,
                        source_crop,
                    )
                    jobs[future] = (episode_index, key)
            for completed, future in enumerate(as_completed(jobs), start=1):
                identity = jobs[future]
                video_records[identity] = future.result()
                if completed % 20 == 0 or completed == len(jobs):
                    print(f"encoded {completed}/{len(jobs)} videos", flush=True)

        manifests: list[dict[str, Any]] = []
        global_start = 0
        for episode_index, plan in enumerate(plans):
            parquet, arrays = write_parquet(
                metadata, staging, plan, episode_index, global_start
            )
            episode_stats = {key: numeric_stats(value) for key, value in arrays.items()}
            for key in VIDEO_KEYS:
                path = staging / metadata.get_video_file_path(episode_index, key)
                episode_stats[key] = video_stats(path, plan.output_count)
            metadata.save_episode(
                episode_index, plan.output_count, [args.task], episode_stats
            )

            audit_dir = staging / "meta" / "alignment" / f"episode_{episode_index:06d}"
            # Preserve the exporter-produced nearest table byte-for-row as the
            # primary audit requested by the handoff.  The independently
            # recomputed nearest table is separate and never masquerades as the
            # source artifact.
            write_csv_rows(
                audit_dir / "alignment_nearest_full.csv", plan.source_nearest_rows
            )
            write_csv_rows(
                audit_dir / "alignment_nearest_recomputed.csv",
                plan.recomputed_nearest_rows,
            )
            write_csv_rows(audit_dir / "alignment_10hz_strict_grid.csv", plan.strict_rows)
            atomic_json(audit_dir / "quality_summary.json", plan.quality)
            manifest = {
                **plan.quality,
                "episode_index": episode_index,
                "source_episode": str(plan.files.episode),
                "source_hashes": {
                    "e6/hand_pose.csv": sha256(plan.files.relative_hand_pose_csv),
                    "e6_rgb_timing.csv": sha256(plan.files.e6_timing_csv),
                    "left_frames.csv": sha256(plan.files.left_frames_csv),
                    "right_frames.csv": sha256(plan.files.right_frames_csv),
                    "left_rx_packets.csv": sha256(plan.files.left_hand_packets_csv),
                    "right_rx_packets.csv": sha256(plan.files.right_hand_packets_csv),
                },
                "parquet": {
                    "path": str(parquet.relative_to(staging)),
                    "size_bytes": parquet.stat().st_size,
                    "sha256": sha256(parquet),
                },
                "videos": {
                    key: {
                        **video_records[(episode_index, key)],
                        "path": str(metadata.get_video_file_path(episode_index, key)),
                    }
                    for key in VIDEO_KEYS
                },
                "global_index_start": global_start,
                "global_index_end_exclusive": global_start + plan.output_count,
            }
            manifests.append(manifest)
            global_start += plan.output_count
            print(
                f"[{episode_index + 1}/{len(plans)}] {plan.files.episode.name}: "
                f"{plan.output_count} rows",
                flush=True,
            )

        write_stats(metadata.stats, staging)
        metadata.info["splits"] = (
            {"train": "0:80", "validation": "80:90", "test": "90:100"}
            if len(plans) == 100
            else {"train": f"0:{len(plans)}"}
        )
        write_info(metadata.info, staging)
        contract = {
            "schema": "umi-dual-hand-pose-lerobot30-h50",
            "schema_version": 1,
            "status": "complete",
            "source": str(source),
            "target": str(target),
            "repo_id": args.repo_id,
            "task": args.task,
            "fps": OUTPUT_FPS,
            "state_dim": STATE_DIM,
            "state_shape": [STATE_DIM],
            "action_dim": STATE_DIM,
            "action_shape": [ACTION_HORIZON, STATE_DIM],
            "state_names": STATE_NAMES,
            "action_names": ACTION_NAMES,
            "state_semantics": "10Hz [left inv(T[t-1])T[t] pose9, right pose9, left/right absolute finger12]",
            "action_semantics": "action[t,k]=[left inv(T[t])T[t+k] pose9, right pose9, absolute finger targets12], k=1..50; one shared anchor",
            "pose_encoding": "body-frame translation xyz in metres + rotation6d [R[:,0],R[:,1]]; fingers in degrees",
            "tail_policy": "preserve every real observation/video row; repeat terminal action target only in slots marked action_is_pad=true",
            "camera_alignment": {
                "global_clock": "rock5b.clock_monotonic.ns",
                "e6_source": "sync/e6_rgb_timing.csv",
                "camera_sources": [
                    "camera/cam0_left_hand_frames.csv",
                    "camera/cam1_right_hand_frames.csv",
                ],
                "strict_max_delta_ns": int(round(args.max_alignment_ms * 1_000_000)),
                "max_hand_age_ns": int(round(args.max_hand_age_ms * 1_000_000)),
                "policy": "nearest timestamp, strict monotonic one-to-one longest run",
                "forbidden_source": "camera/*_frames_aligned.csv",
            },
            "video_keys": VIDEO_KEYS,
            "video_preprocessing": {
                "head_rgb_source": "E6 3200x1200 stereo rgb.mp4",
                "head_rgb_crop": "right eye crop=1600:1200:1600:0",
                "left_right_wrist_crop": None,
                "output_width": args.video_width,
                "output_height": args.video_height,
                "resize_mode": args.resize_mode,
            },
            "training_contract": {
                "action_is_prechunked": True,
                "action_horizon": ACTION_HORIZON,
                "ordinary_future_row_slicing_allowed": False,
                "apply_delta_transform_again": False,
                "action_padding_mask_key": "action_is_pad",
                "full_real_horizon_filter_key": "sample_valid_h50",
                "padding_contract": "training must mask action_is_pad, or filter sample_valid_h50==1",
            },
            "total_episodes": len(plans),
            "total_frames": global_start,
            "total_full_real_h50_frames": int(
                sum(plan.sample_valid_h50.sum() for plan in plans)
            ),
            "episodes_with_zero_full_real_h50_frames": [
                plan.files.episode.name
                for plan in plans
                if not bool(plan.sample_valid_h50.any())
            ],
            "episodes": manifests,
        }
        atomic_json(staging / "meta" / "umi_conversion.json", contract)
        shutil.copy2(Path(__file__).resolve(), staging / "meta" / Path(__file__).name)
        (staging / "README.md").write_text(
            "# UMI dual hand-pose + absolute fingers, 10 Hz / H50\n\n"
            "State is 30-D: adjacent body-frame left/right hand-pose deltas "
            "from `e6/hand_pose.csv`, followed by absolute finger12. Each row "
            "already contains one H=50 action chunk whose two EEF targets all "
            "share the current row anchor; future fingers remain absolute. Do "
            "not slice future action rows or apply another delta transform. "
            "Terminal action slots are explicit in `action_is_pad`; training "
            "must mask them or filter on `sample_valid_h50`. Videos are never padded. "
            "See `meta/umi_conversion.json`.\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        staging_parent.rmdir()
        print(json.dumps({
            "status": "complete",
            "target": str(target),
            "episodes": len(plans),
            "frames": global_start,
        }, ensure_ascii=False))
        return 0
    except Exception:
        print(f"conversion failed; staging retained at {staging_parent}")
        raise


def self_test() -> int:
    rng = np.random.default_rng(7)
    count = ACTION_HORIZON + 8
    position = np.cumsum(rng.normal(scale=0.01, size=(count, 3)), axis=0)
    angle = np.linspace(0.0, 0.8, count)
    transforms = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    transforms[:, :3, 3] = position
    transforms[:, 0, 0] = np.cos(angle)
    transforms[:, 0, 1] = -np.sin(angle)
    transforms[:, 1, 0] = np.sin(angle)
    transforms[:, 1, 1] = np.cos(angle)
    state = encode_relative(invert(transforms[:-1]) @ transforms[1:])
    anchor = 2
    targets = np.arange(anchor + 1, anchor + ACTION_HORIZON + 1)
    action = encode_relative(invert(transforms[anchor])[None] @ transforms[targets])
    if not np.allclose(action[0], state[anchor]):
        raise AssertionError("k=1 invariant failed")
    composed = transforms[anchor] @ (invert(transforms[anchor])[None] @ transforms[targets])
    if not np.allclose(composed, transforms[targets]):
        raise AssertionError("shared-anchor reconstruction failed")
    query = np.asarray([9, 21, 39], dtype=np.int64)
    source = np.asarray([0, 10, 20, 30, 40], dtype=np.int64)
    if not np.array_equal(nearest_indices(source, query), np.asarray([1, 2, 4])):
        raise AssertionError("nearest timestamp selection failed")
    relative_rows: list[dict[str, str]] = []
    for index in range(3):
        row = {
            "timestamp_ns": str(index * 100),
            "frame_number": str(index),
            "previous_frame_number": "" if index == 0 else str(index - 1),
        }
        for side in SIDES:
            row[f"{side}_relative_valid"] = str(int(index > 0))
            for axis, value in zip("xyz", (1.0 if index > 0 else 0.0, 0.0, 0.0)):
                row[f"{side}_local_dp{axis}"] = str(value)
            for axis, value in zip("xyzw", (0.0, 0.0, 0.0, 1.0)):
                row[f"{side}_local_dq{axis}"] = str(value)
        relative_rows.append(row)
    _, reconstructed, pair_valid, _ = reconstruct_hand_pose(relative_rows)
    if not np.array_equal(pair_valid, np.asarray([False, True, True])):
        raise AssertionError("hand_pose validity reconstruction failed")
    if not all(np.allclose(reconstructed[side][:, 0, 3], (0.0, 1.0, 2.0)) for side in SIDES):
        raise AssertionError("hand_pose cumulative reconstruction failed")
    sampled, fresh, age, selected = sample_absolute_hand(
        np.asarray([0, 90, 210]),
        np.arange(18, dtype=np.float64).reshape(3, 6),
        np.asarray([100, 220]),
    )
    if not np.array_equal(selected, (1, 2)) or not fresh.all() or not np.array_equal(age, (10, 10)):
        raise AssertionError("causal absolute-hand sampling failed")
    if sampled.shape != (2, 6):
        raise AssertionError("absolute-hand sample shape failed")
    print("self-test passed: hand_pose reconstruction, absolute hands, SE(3), H50, alignment")
    return 0


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--expected-episodes", type=int, default=0,
        help="require exactly N selected episodes; 0 disables the count check",
    )
    parser.add_argument(
        "--max-alignment-ms", type=float, default=DEFAULT_MAX_ALIGNMENT_MS,
        help="maximum absolute camera-to-E6 timestamp error (default: 100 ms)",
    )
    parser.add_argument(
        "--max-hand-age-ms", type=float, default=DEFAULT_MAX_HAND_AGE_MS,
        help="maximum age of a causal Revo2 hand-state sample (default: 100 ms)",
    )
    parser.add_argument("--only", action="append", default=[], metavar="EPISODE_ID")
    parser.add_argument("--exclude", action="append", default=[], metavar="EPISODE_ID")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser(
        "inspect", help="read and validate every source; never create the target"
    )
    add_common_arguments(inspect_parser)
    inspect_parser.add_argument(
        "--compact", action="store_true", help="omit per-episode quality objects from final JSON"
    )
    inspect_parser.set_defaults(handler=inspect)

    convert_parser = subparsers.add_parser(
        "convert", help="perform the staged, atomic LeRobot conversion"
    )
    add_common_arguments(convert_parser)
    convert_parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    convert_parser.add_argument("--task", required=True)
    convert_parser.add_argument("--video-width", type=int, default=640)
    convert_parser.add_argument("--video-height", type=int, default=480)
    convert_parser.add_argument(
        "--resize-mode", choices=("letterbox", "crop", "stretch"), default="letterbox"
    )
    convert_parser.add_argument("--crf", type=int, default=20)
    convert_parser.add_argument("--video-workers", type=int, default=4)
    convert_parser.add_argument(
        "--confirm", help="must be the literal CREATE_LEROBOT_DATASET"
    )
    convert_parser.add_argument(
        "--replace-empty-target",
        action="store_true",
        help="remove the target only when it is an existing completely empty directory",
    )
    convert_parser.set_defaults(handler=convert)

    test_parser = subparsers.add_parser("self-test", help="run math-only unit checks")
    test_parser.set_defaults(handler=lambda _args: self_test())
    args = parser.parse_args()
    if hasattr(args, "expected_episodes") and args.expected_episodes < 0:
        parser.error("--expected-episodes must be >= 0 (0 disables the check)")
    if hasattr(args, "max_alignment_ms") and (
        not math.isfinite(args.max_alignment_ms) or args.max_alignment_ms <= 0
    ):
        parser.error("--max-alignment-ms must be finite and positive")
    if hasattr(args, "max_hand_age_ms") and (
        not math.isfinite(args.max_hand_age_ms) or args.max_hand_age_ms <= 0
    ):
        parser.error("--max-hand-age-ms must be finite and positive")
    if hasattr(args, "video_width") and (args.video_width <= 0 or args.video_height <= 0):
        parser.error("video dimensions must be positive")
    if hasattr(args, "crf") and not 0 <= args.crf <= 51:
        parser.error("--crf must be in [0,51]")
    return args


if __name__ == "__main__":
    try:
        parsed = parse_args()
        raise SystemExit(parsed.handler(parsed))
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        raise SystemExit(2) from error
