#!/usr/bin/env python3
"""Convert the native ``task_v1`` UMI layout to LeRobot v2.1.

This converter implements the following *explicit* data contract:

* output rate: 10 Hz;
* state[t] is 30-D: left/right ``inverse(T[t-1]) @ T[t]`` encoded as
  translation3 + rotation-6D6, followed by left/right absolute finger6;
* action[t, k] is 30-D: left/right ``inverse(T[t]) @ T[t+k]`` followed by
  the left/right absolute finger targets at t+k, k=1..50;
* every action in the H=50 chunk uses the same current T[t] anchor;
* after the one-sample state baseline of each valid run, every real aligned
  observation/video row is preserved; action slots beyond that run boundary
  repeat its terminal target and are explicitly masked by ``action_is_pad``
  (videos are never padded); singleton runs cannot form a state and are
  retained in the audit rather than emitted;
* the three observation videos are the E6 right eye, cam0, and cam1; the
  project-level OpenPI mapping is cam0 -> left wrist and cam1 -> right wrist.

Camera alignment follows the timestamp/strict-uniqueness contract from the
E6/camera handoff.  It never uses ``camera/*_frames_aligned.csv`` and never
assumes that equal video frame numbers are synchronized.  For each episode it:

1. reads E6 global timestamps from ``sync/e6_rgb_timing.csv``;
2. reads camera timestamps from the original customer-camera ``frames.csv``;
3. independently matches the nearest cam0/cam1 frame by timestamp;
4. builds the E6 10 Hz grid, then retains every contiguous, strictly
   monotonic, one-to-one interval with at least two samples for which both
   camera deltas are within the configured limit (100 ms by default); each
   retained interval becomes an independent LeRobot episode, so neither state
   nor action can cross an invalid alignment boundary;
5. records every selected source frame index and signed timestamp delta.

Dual-hand transforms are reconstructed only from ``camera/hand_pose.csv``.  The
converter deliberately does not read ``controller_poses.csv`` or
recompute the controller-to-EEF calibration.  Absolute six-joint Revo2 finger
positions are decoded from the serial packet logs and aligned offline on the
global timestamp axis (nearest within 100 ms by default; this may select a
slightly later packet, while strictly causal ZOH is available as an option).

The converter directly decodes the native raw media without changing the
source episodes: ``camera/e6_rgb.h265`` for E6 and the two
``extensions/customer_camera/cam*/media.mjpeg`` streams for the wrist cameras.
The native files retain acquisition IDs cam0/cam1.  This task's established
OpenPI contract maps them to the canonical LeRobot keys ``left_wrist_rgb`` and
``right_wrist_rgb`` respectively; the source acquisition IDs remain recorded
in every row and in the conversion audit.

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


DEFAULT_SOURCE = Path("/mnt/data/dzq/umi/data/task_v1")
DEFAULT_TARGET = Path("/mnt/data/dzq/umi/datasets/task_v1_lerobot_h50")
DEFAULT_REPO_ID = "local/task_v1_lerobot_h50"
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
    "observation.images.left_wrist_rgb": "cam0",
    "observation.images.right_wrist_rgb": "cam1",
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
    relative_hand_pose_csv: Path
    rgb_metainfo_csv: Path
    head_video: Path
    e6_timing_csv: Path
    clock_model_json: Path
    cam0_frames_csv: Path
    cam1_frames_csv: Path
    cam0_video: Path
    cam1_video: Path
    cam0_associations_csv: Path
    cam1_associations_csv: Path
    alignment_report_json: Path
    customer_camera_summary_json: Path
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
    segment_index: int
    segment_count: int
    valid_run_index: int
    valid_run_count: int
    output_segment_id: str
    e6_rows: list[dict[str, str]]
    source_cam0_association_rows: list[dict[str, Any]]
    source_cam1_association_rows: list[dict[str, Any]]
    recomputed_nearest_rows: list[dict[str, Any]]
    strict_rows: list[dict[str, Any]]
    e6_source_row_indices: np.ndarray
    e6_frame_ids: np.ndarray
    cam0_frame_indices: np.ndarray
    cam1_frame_indices: np.ndarray
    cam0_signed_delta_ns: np.ndarray
    cam1_signed_delta_ns: np.ndarray
    hand_pose_timestamps_ns: np.ndarray
    state: np.ndarray
    action: np.ndarray
    action_is_pad: np.ndarray
    sample_valid_h50: np.ndarray
    source_fps: int
    source_stride: int
    camera_ratio: int
    cam0_offset: int
    cam1_offset: int
    strict_run_start: int
    strict_run_end_exclusive: int
    output_count: int
    quality: dict[str, Any]


@dataclass
class SourceEpisodePlan:
    files: EpisodeFiles
    source_episode_index: int
    segments: list[EpisodePlan]
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
    files = EpisodeFiles(
        episode=episode,
        relative_hand_pose_csv=episode / "camera" / "hand_pose.csv",
        rgb_metainfo_csv=episode / "camera" / "e6_rgb_stream_metainfo.csv",
        head_video=episode / "camera" / "e6_rgb.h265",
        e6_timing_csv=episode / "sync" / "e6_rgb_timing.csv",
        clock_model_json=episode / "sync" / "e6_clock_model.json",
        cam0_frames_csv=episode / "extensions" / "customer_camera" / "cam0" / "frames.csv",
        cam1_frames_csv=episode / "extensions" / "customer_camera" / "cam1" / "frames.csv",
        cam0_video=episode / "extensions" / "customer_camera" / "cam0" / "media.mjpeg",
        cam1_video=episode / "extensions" / "customer_camera" / "cam1" / "media.mjpeg",
        cam0_associations_csv=(
            episode / "extensions" / "customer_camera" / "derived" / "cam0_associations.csv"
        ),
        cam1_associations_csv=(
            episode / "extensions" / "customer_camera" / "derived" / "cam1_associations.csv"
        ),
        alignment_report_json=(
            episode / "extensions" / "customer_camera" / "derived" / "alignment_report.json"
        ),
        customer_camera_summary_json=(
            episode / "extensions" / "customer_camera" / "summary.json"
        ),
        left_hand_packets_csv=episode / "serial" / "left_rx_packets.csv",
        right_hand_packets_csv=episode / "serial" / "right_rx_packets.csv",
    )
    missing = [
        path
        for name, path in files.__dict__.items()
        if name != "episode" and isinstance(path, Path) and not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{episode.name}: missing required files: {missing}")
    empty = [
        path
        for name, path in files.__dict__.items()
        if name != "episode" and isinstance(path, Path) and path.stat().st_size == 0
    ]
    if empty:
        raise ValueError(f"{episode.name}: required files are empty: {empty}")
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
    timestamp_ns = np.asarray([int(row["alignment_time_ns"]) for row in rows], dtype=np.int64)
    capture_ok = np.asarray([row["capture_status"] == "ok" for row in rows], dtype=bool)
    if not np.array_equal(frame_index, np.arange(len(rows), dtype=np.int64)):
        raise ValueError(f"{side}: camera frame_index is not contiguous from zero")
    if len(sequence) > 1 and not np.all(np.diff(sequence) == 1):
        raise ValueError(f"{side}: V4L2 sequence contains a gap")
    if not np.all(capture_ok):
        raise ValueError(f"{side}: camera capture_status contains non-ok rows")
    alignment_sources = {row.get("alignment_time_source", "") for row in rows}
    if alignment_sources != {"v4l2_monotonic"}:
        raise ValueError(
            f"{side}: expected v4l2_monotonic alignment timestamps, "
            f"found {sorted(alignment_sources)}"
        )
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


def valid_sample_sequences(
    sample_valid: np.ndarray, transition_valid: np.ndarray
) -> list[tuple[int, int]]:
    """Return half-open runs of valid samples connected by valid transitions.

    ``transition_valid[i]`` describes the edge from sample ``i-1`` to ``i``.
    A failed incoming edge does not prevent sample ``i`` from becoming the
    baseline of a new sequence; this avoids dropping one extra 10 Hz point at
    every discontinuity.
    """
    sample_valid = np.asarray(sample_valid, dtype=bool)
    transition_valid = np.asarray(transition_valid, dtype=bool)
    if sample_valid.shape != transition_valid.shape:
        raise ValueError("sample/transition validity shape mismatch")
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, valid in enumerate(sample_valid):
        if not valid:
            if start is not None:
                runs.append((start, index))
                start = None
            continue
        if start is None:
            start = index
        elif not transition_valid[index]:
            runs.append((start, index))
            start = index
    if start is not None:
        runs.append((start, len(sample_valid)))
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

    ``camera/hand_pose.csv`` already stores ``inverse(T[i-1]) @ T[i]`` for each
    hand.  Cumulative composition is sufficient for shared-anchor actions;
    the arbitrary identity chosen at the beginning of a continuous segment
    cancels from every relative transform.
    """
    timestamps = np.asarray([int(row["timestamp_ns"]) for row in rows], dtype=np.int64)
    frame_number = np.asarray([int(row["frame_number"]) for row in rows], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("camera/hand_pose.csv timestamps are not strictly increasing")
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
    deduplicated_rows: list[dict[str, str]] = []
    deduplicated_values: list[list[int]] = []
    duplicate_timestamps = 0
    for row, values in zip(selected, decoded):
        timestamp = int(row["host_rx_time_ns"])
        if deduplicated_rows:
            previous_timestamp = int(deduplicated_rows[-1]["host_rx_time_ns"])
            if timestamp < previous_timestamp:
                raise ValueError(f"{path}: hand packet timestamps decrease")
            if timestamp == previous_timestamp:
                if values[:6] != deduplicated_values[-1][:6]:
                    raise ValueError(
                        f"{path}: duplicate timestamp has conflicting joint positions"
                    )
                # Stable source order makes the later row the best packet when
                # timestamps collide (it also has the later packet sequence in
                # the one observed task_v1 duplicate).
                deduplicated_rows[-1] = row
                deduplicated_values[-1] = values
                duplicate_timestamps += 1
                continue
        deduplicated_rows.append(row)
        deduplicated_values.append(values)
    timestamp = np.asarray(
        [int(row["host_rx_time_ns"]) for row in deduplicated_rows], dtype=np.int64
    )
    if np.any(np.diff(timestamp) <= 0):
        raise AssertionError("hand timestamp de-duplication failed")
    # First six decoded values are absolute joint position in 0.1 degree.
    position_deg = np.asarray(deduplicated_values, dtype=np.float64)[:, :6] / 10.0
    return timestamp, position_deg, {
        f"{side}_packet_csv_rows": len(rows),
        f"{side}_full_state_rows": len(selected),
        f"{side}_deduplicated_full_state_rows": len(deduplicated_rows),
        f"{side}_duplicate_timestamp_rows": duplicate_timestamps,
    }


def sample_absolute_hand(
    source_timestamp_ns: np.ndarray,
    source_position_deg: np.ndarray,
    query_timestamp_ns: np.ndarray,
    max_age_ns: int = int(DEFAULT_MAX_HAND_AGE_MS * 1_000_000),
    alignment: str = "nearest",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if alignment == "nearest":
        selected = nearest_indices(source_timestamp_ns, query_timestamp_ns)
        signed_delta_ns = source_timestamp_ns[selected] - query_timestamp_ns
        fresh = np.abs(signed_delta_ns) <= max_age_ns
        return source_position_deg[selected].copy(), fresh, signed_delta_ns, selected
    if alignment == "causal":
        selected = np.searchsorted(source_timestamp_ns, query_timestamp_ns, side="right") - 1
        causal_valid = selected >= 0
        safe = np.maximum(selected, 0)
        chosen_timestamp = source_timestamp_ns[safe]
        age_ns = query_timestamp_ns - chosen_timestamp
        fresh = causal_valid & (age_ns >= 0) & (age_ns <= max_age_ns)
        position = source_position_deg[safe].copy()
        position[~causal_valid] = np.nan
        return position, fresh, -age_ns, selected
    raise ValueError(f"unsupported hand alignment: {alignment}")


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
    hand_alignment: str,
) -> SourceEpisodePlan:
    clock = load_json(files.clock_model_json)
    if clock.get("status") != "ok":
        raise ValueError(f"{files.episode.name}: clock_model.status != ok")
    alignment_report = load_json(files.alignment_report_json)
    camera_summary = load_json(files.customer_camera_summary_json)
    for label, report in (
        ("alignment_report", alignment_report),
        ("customer_camera_summary", camera_summary),
    ):
        if report.get("status") != "ok":
            raise ValueError(f"{files.episode.name}: {label}.status != ok")
        if report.get("session_id") != files.episode.name:
            raise ValueError(f"{files.episode.name}: {label} session_id mismatch")

    timing = read_csv(files.e6_timing_csv)
    meta = read_csv(files.rgb_metainfo_csv)
    raw_hand_pose = read_csv(files.relative_hand_pose_csv)
    source_cam0_association_rows = read_csv(files.cam0_associations_csv)
    source_cam1_association_rows = read_csv(files.cam1_associations_csv)
    if len(timing) != len(meta):
        raise ValueError(
            f"{files.episode.name}: E6 timing/meta row counts differ: "
            f"{len(timing)}/{len(meta)}"
        )
    declared_e6_payload_bytes = sum(int(row["payload_bytes"]) for row in meta)
    e6_media_overhead_bytes = files.head_video.stat().st_size - declared_e6_payload_bytes
    if not 0 <= e6_media_overhead_bytes <= 4096:
        raise ValueError(
            f"{files.episode.name}: E6 H.265 size disagrees with payload_bytes: "
            f"file={files.head_video.stat().st_size}, payload={declared_e6_payload_bytes}, "
            f"overhead={e6_media_overhead_bytes}"
        )
    e6_index = np.asarray([int(row["source_row_index"]) for row in timing], dtype=np.int64)
    if not np.array_equal(e6_index, np.arange(len(timing), dtype=np.int64)):
        raise ValueError(f"{files.episode.name}: E6 source_row_index is not contiguous")
    if any(row["mapping_status"] != "ok" for row in timing):
        raise ValueError(f"{files.episode.name}: E6 mapping_status contains non-ok rows")
    e6_sequence = np.asarray([int(row["sequence_number"]) for row in timing], dtype=np.int64)
    e6_frame_id = np.asarray([int(row["frame_id"]) for row in timing], dtype=np.int64)
    e6_global = np.asarray(
        [int(row["mapped_5b_monotonic_ns"]) for row in timing], dtype=np.int64
    )
    e6_mid = np.asarray(
        [int(row["e6_mid_exposure_realtime_ns"]) for row in timing], dtype=np.int64
    )
    meta_sequence = np.asarray(
        [int(row["sequence_number"]) for row in meta], dtype=np.int64
    )
    meta_frame_id = np.asarray([int(row["frame_id"]) for row in meta], dtype=np.int64)
    meta_mid = np.asarray(
        [int(row["e6_mid_exposure_realtime_ns"]) for row in meta], dtype=np.int64
    )
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
    if not np.array_equal(meta_sequence, e6_sequence):
        raise ValueError(f"{files.episode.name}: RGB/timing sequence_number mismatch")
    if not np.array_equal(meta_frame_id, e6_frame_id):
        raise ValueError(f"{files.episode.name}: RGB/timing frame_id mismatch")
    if not np.array_equal(meta_mid, e6_mid):
        raise ValueError(f"{files.episode.name}: RGB/timing timestamps do not match exactly")
    hand_frame_by_timestamp = {
        int(row["timestamp_ns"]): int(row["frame_number"])
        for row in raw_hand_pose
    }
    matched_hand_frame_ids = np.asarray(
        [hand_frame_by_timestamp.get(int(timestamp), -1) for timestamp in e6_mid],
        dtype=np.int64,
    )
    if not np.array_equal(matched_hand_frame_ids, e6_frame_id):
        raise ValueError(f"{files.episode.name}: hand_pose frame_number/timestamp mismatch")

    left_hand_time, left_hand_position, left_hand_metrics = decode_absolute_hand_packets(
        files.left_hand_packets_csv, "left"
    )
    right_hand_time, right_hand_position, right_hand_metrics = decode_absolute_hand_packets(
        files.right_hand_packets_csv, "right"
    )
    left_hand, left_hand_fresh, left_hand_delta, left_hand_source_row = sample_absolute_hand(
        left_hand_time, left_hand_position, e6_global, max_hand_age_ns, hand_alignment
    )
    right_hand, right_hand_fresh, right_hand_delta, right_hand_source_row = sample_absolute_hand(
        right_hand_time, right_hand_position, e6_global, max_hand_age_ns, hand_alignment
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

    cam0 = parse_camera(files.cam0_frames_csv, "cam0")
    cam1 = parse_camera(files.cam1_frames_csv, "cam1")
    declared_cam0_bytes = sum(int(row["bytes"]) for row in cam0.source_rows)
    declared_cam1_bytes = sum(int(row["bytes"]) for row in cam1.source_rows)
    if files.cam0_video.stat().st_size != declared_cam0_bytes:
        raise ValueError(
            f"{files.episode.name}: cam0 MJPEG size disagrees with frames.csv bytes"
        )
    if files.cam1_video.stat().st_size != declared_cam1_bytes:
        raise ValueError(
            f"{files.episode.name}: cam1 MJPEG size disagrees with frames.csv bytes"
        )
    camera_fps = statistics.mean((cam0.fps, cam1.fps))
    camera_ratio_float = camera_fps / e6_fps_float
    camera_ratio = int(round(camera_ratio_float))
    if camera_ratio <= 0 or abs(camera_ratio_float - camera_ratio) > 0.05:
        raise ValueError(
            f"{files.episode.name}: camera/E6 FPS ratio is not near an integer: "
            f"{camera_ratio_float:.6f}"
        )

    # Recompute nearest matches for every E6 exposure for audit, but enforce
    # uniqueness on the requested 10 Hz output grid.  Enforcing one-to-one at
    # the native 60 Hz rate before downsampling would reject useful 10 Hz data
    # whenever camera jitter maps two adjacent raw E6 frames to one camera
    # frame, even though the selected 10 Hz frames remain unique.
    cam0_nearest_row = nearest_indices(cam0.timestamp_ns, e6_global)
    cam1_nearest_row = nearest_indices(cam1.timestamp_ns, e6_global)
    cam0_nearest = cam0.frame_index[cam0_nearest_row]
    cam1_nearest = cam1.frame_index[cam1_nearest_row]
    cam0_delta = cam0.timestamp_ns[cam0_nearest_row] - e6_global
    cam1_delta = cam1.timestamp_ns[cam1_nearest_row] - e6_global
    cam0_offset = int(round(float(np.median(cam0_nearest - camera_ratio * e6_index))))
    cam1_offset = int(round(float(np.median(cam1_nearest - camera_ratio * e6_index))))

    grid_rows = np.arange(0, len(e6_index), source_stride, dtype=np.int64)
    grid_sample_valid = (
        (np.abs(cam0_delta[grid_rows]) <= max_alignment_ns)
        & (np.abs(cam1_delta[grid_rows]) <= max_alignment_ns)
        & pair_hand_fresh[grid_rows]
    )
    grid_transition_valid = np.ones(len(grid_rows), dtype=bool)
    for position in range(1, len(grid_rows)):
        previous = int(grid_rows[position - 1])
        current = int(grid_rows[position])
        grid_transition_valid[position] = bool(
            cam0_nearest[current] > cam0_nearest[previous]
            and cam1_nearest[current] > cam1_nearest[previous]
            and np.all(pair_pose_valid[previous + 1:current + 1])
            and np.all(e6_interval_ok[previous + 1:current + 1])
        )
    runs = valid_sample_sequences(grid_sample_valid, grid_transition_valid)
    if not runs:
        raise ValueError(f"{files.episode.name}: no valid 10 Hz three-camera alignment run")
    emitted_runs = [
        (valid_run_index, run_start, run_end)
        for valid_run_index, (run_start, run_end) in enumerate(runs)
        if run_end - run_start >= 2
    ]
    if not emitted_runs:
        raise ValueError(
            f"{files.episode.name}: all {len(runs)} valid 10 Hz runs are singletons; "
            "at least two samples are required to form state"
        )

    recomputed_nearest_rows: list[dict[str, Any]] = []
    previous_cam0 = previous_cam1 = None
    for row in range(len(e6_index)):
        cam0_frame = int(cam0_nearest[row])
        cam1_frame = int(cam1_nearest[row])
        recomputed_nearest_rows.append({
            "e6_source_row_index": int(e6_index[row]),
            "e6_frame_id": int(e6_frame_id[row]),
            "e6_global_ts_ns": int(e6_global[row]),
            "cam0_frame_index": cam0_frame,
            "cam0_signed_delta_ns": int(cam0_delta[row]),
            "cam0_reused": int(cam0_frame == previous_cam0),
            "cam0_within_limit": int(abs(int(cam0_delta[row])) <= max_alignment_ns),
            "cam1_frame_index": cam1_frame,
            "cam1_signed_delta_ns": int(cam1_delta[row]),
            "cam1_reused": int(cam1_frame == previous_cam1),
            "cam1_within_limit": int(abs(int(cam1_delta[row])) <= max_alignment_ns),
        })
        previous_cam0, previous_cam1 = cam0_frame, cam1_frame

    emitted_index_by_valid_run = {
        valid_run_index: segment_index
        for segment_index, (valid_run_index, _run_start, _run_end) in enumerate(emitted_runs)
    }
    run_records: list[dict[str, Any]] = []
    for valid_run_index, (run_start, run_end) in enumerate(runs):
        run_rows = grid_rows[run_start:run_end]
        segment_index = emitted_index_by_valid_run.get(valid_run_index)
        record: dict[str, Any] = {
            "valid_run_index": valid_run_index,
            "grid_start_index": run_start,
            "grid_end_index_exclusive": run_end,
            "sample_count": len(run_rows),
            "first_e6_source_row": int(e6_index[run_rows[0]]),
            "last_e6_source_row": int(e6_index[run_rows[-1]]),
            "first_e6_frame_id": int(e6_frame_id[run_rows[0]]),
            "last_e6_frame_id": int(e6_frame_id[run_rows[-1]]),
            "emitted": segment_index is not None,
            "segment_index": segment_index,
            "output_segment_id": (
                f"{files.episode.name}__segment_{segment_index:03d}"
                if segment_index is not None else None
            ),
        }
        if segment_index is None:
            record["discard_reason"] = (
                "singleton valid run cannot form inv(T[t-1]) @ T[t] state"
            )
        run_records.append(record)

    emitted_sample_rows = np.concatenate([
        grid_rows[run_start:run_end]
        for _valid_run_index, run_start, run_end in emitted_runs
    ])
    source_quality_common = {
        "episode_id": files.episode.name,
        "source_episode_id": files.episode.name,
        "source_episode_index": source_episode_index,
        "clock_model_status": clock.get("status"),
        "e6_rgb_rows": len(timing),
        "source_media_integrity": {
            "e6_h265_file_bytes": files.head_video.stat().st_size,
            "e6_declared_payload_bytes": declared_e6_payload_bytes,
            "e6_container_overhead_bytes": e6_media_overhead_bytes,
            "cam0_mjpeg_file_bytes": files.cam0_video.stat().st_size,
            "cam0_declared_frame_bytes": declared_cam0_bytes,
            "cam1_mjpeg_file_bytes": files.cam1_video.stat().st_size,
            "cam1_declared_frame_bytes": declared_cam1_bytes,
        },
        "hand_pose_source_rows": len(raw_hand_pose),
        "hand_pose_rows_matched_to_rgb": int(hand_pose_matched.sum()),
        "rgb_rows_missing_hand_pose": int((~hand_pose_matched).sum()),
        "rgb_source_rows_missing_hand_pose": e6_index[~hand_pose_matched].tolist(),
        "rgb_frame_ids_missing_hand_pose": e6_frame_id[~hand_pose_matched].tolist(),
        "hand_pose_rows_outside_rgb_timeline": int(
            len(raw_hand_pose) - hand_pose_matched.sum()
        ),
        "hand_pose_quality": hand_pose_metrics,
        "absolute_hand_quality": {
            **left_hand_metrics,
            **right_hand_metrics,
            "alignment_policy": hand_alignment,
            "max_hand_age_ns": max_hand_age_ns,
            "pair_fresh_within_limit_rows": int(pair_hand_fresh.sum()),
            "pair_fresh_on_10hz_grid_rows": int(pair_hand_fresh[grid_rows].sum()),
            "left_max_fresh_abs_delta_ns": int(
                np.abs(left_hand_delta[left_hand_fresh]).max(initial=0)
            ),
            "right_max_fresh_abs_delta_ns": int(
                np.abs(right_hand_delta[right_hand_fresh]).max(initial=0)
            ),
        },
        "strict_max_delta_ns": max_alignment_ns,
        "e6_fps_measured": e6_fps_float,
        "e6_fps_full_interval_average": e6_average_fps,
        "e6_nominal_interval_ns": nominal_interval_ns,
        "e6_interval_tolerance_ns": interval_tolerance_ns,
        "e6_interval_discontinuity_count": int((~e6_interval_ok).sum()),
        "e6_interval_discontinuity_after_frame_indices": e6_index[~e6_interval_ok].tolist(),
        "camera_cam0_fps_measured": cam0.fps,
        "camera_cam1_fps_measured": cam1.fps,
        "camera_to_e6_integer_ratio": camera_ratio,
        "cam0_median_frame_offset": cam0_offset,
        "cam1_median_frame_offset": cam1_offset,
        "candidate_10hz_grid_points": len(grid_rows),
        "valid_sample_sequence_count": len(runs),
        "emitted_segment_count": len(emitted_runs),
        "discarded_singleton_sequence_count": len(runs) - len(emitted_runs),
        "valid_sample_sequences": run_records,
        "discarded_singleton_sequences": [
            record for record in run_records if not record["emitted"]
        ],
        "valid_10hz_grid_points": int(sum(end - start for start, end in runs)),
        "emitted_10hz_grid_points": len(emitted_sample_rows),
        "strict_10hz_grid_points": len(emitted_sample_rows),
        "strict_10hz_retention_ratio": len(emitted_sample_rows) / len(grid_rows),
        "strict_cam0_max_abs_delta_ns": int(
            np.max(np.abs(cam0_delta[emitted_sample_rows]))
        ),
        "strict_cam1_max_abs_delta_ns": int(
            np.max(np.abs(cam1_delta[emitted_sample_rows]))
        ),
        "output_fps": OUTPUT_FPS,
        "output_source_stride": source_stride,
        "action_horizon": ACTION_HORIZON,
        "action_horizon_seconds": ACTION_HORIZON / OUTPUT_FPS,
        "tail_policy": "preserve observations/videos; repeat terminal action target only in explicitly masked slots",
    }
    segments: list[EpisodePlan] = []
    segment_count = len(emitted_runs)
    for segment_index, (valid_run_index, run_start, run_end) in enumerate(emitted_runs):
        sample_rows = grid_rows[run_start:run_end]
        strict_length = len(sample_rows)
        # One sample is needed before the first output state.  Every remaining
        # aligned observation is preserved.  Future action slots past this
        # segment boundary repeat its terminal target and carry an explicit
        # pad mask; no video frame is synthesized or repeated, and an action
        # can never cross into a later valid run.
        output_count = strict_length - 1
        anchor_positions = np.arange(1, 1 + output_count, dtype=np.int64)
        previous_rows = sample_rows[anchor_positions - 1]
        anchor_rows = sample_rows[anchor_positions]
        requested_future_positions = anchor_positions[:, None] + np.arange(
            1, ACTION_HORIZON + 1, dtype=np.int64
        )[None, :]
        action_is_pad = requested_future_positions >= strict_length
        future_positions = np.minimum(requested_future_positions, strict_length - 1)
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
        # k=1 is the next adjacent 10 Hz delta.  It should equal the next
        # state, except for the last emitted row whose next state is not an
        # emitted observation.
        if output_count > 1 and not np.allclose(action[:-1, 0], state[1:], atol=2e-6):
            raise AssertionError("shared-anchor k=1 does not match next adjacent state")

        output_anchor_set = set(anchor_rows.tolist())
        action_target_set = set(target_rows.reshape(-1).tolist())
        strict_rows: list[dict[str, Any]] = []
        for position, row in enumerate(sample_rows):
            strict_rows.append({
                "source_grid_index": run_start + position,
                "segment_grid_index": position,
                "valid_run_index": valid_run_index,
                "segment_index": segment_index,
                "is_output_anchor": int(int(row) in output_anchor_set),
                "is_action_target": int(int(row) in action_target_set),
                "e6_source_row_index": int(e6_index[row]),
                "e6_frame_id": int(e6_frame_id[row]),
                "hand_pose_timestamp_ns": int(e6_mid[row]),
                "e6_global_ts_ns": int(e6_global[row]),
                "cam0_frame_index": int(cam0_nearest[row]),
                "cam0_signed_delta_ns": int(cam0_delta[row]),
                "cam1_frame_index": int(cam1_nearest[row]),
                "cam1_signed_delta_ns": int(cam1_delta[row]),
                "left_hand_packet_row": int(left_hand_source_row[row]),
                "left_hand_signed_delta_ns": int(left_hand_delta[row]),
                "right_hand_packet_row": int(right_hand_source_row[row]),
                "right_hand_signed_delta_ns": int(right_hand_delta[row]),
            })

        anchor_e6 = e6_index[anchor_rows]
        output_segment_id = f"{files.episode.name}__segment_{segment_index:03d}"
        quality = {
            **source_quality_common,
            "output_segment_id": output_segment_id,
            "segment_index": segment_index,
            "segment_count": segment_count,
            "valid_run_index": valid_run_index,
            "valid_run_count": len(runs),
            "strict_run_grid_start_index": run_start,
            "strict_run_grid_end_index_exclusive": run_end,
            "strict_run_first_e6_source_row": int(e6_index[sample_rows[0]]),
            "strict_run_last_e6_source_row": int(e6_index[sample_rows[-1]]),
            "strict_run_first_e6_frame_id": int(e6_frame_id[sample_rows[0]]),
            "strict_run_last_e6_frame_id": int(e6_frame_id[sample_rows[-1]]),
            "strict_run_rows": strict_length,
            "strict_10hz_grid_points": strict_length,
            "strict_10hz_retention_ratio": strict_length / len(grid_rows),
            "strict_cam0_max_abs_delta_ns": int(
                np.max(np.abs(cam0_delta[sample_rows]))
            ),
            "strict_cam1_max_abs_delta_ns": int(
                np.max(np.abs(cam1_delta[sample_rows]))
            ),
            "output_rows": output_count,
            "sample_valid_h50_rows": int(sample_valid_h50.sum()),
            "terminal_padded_anchor_rows": int((~sample_valid_h50).sum()),
            "terminal_padded_action_slots": int(action_is_pad.sum()),
            "output_first_e6_source_row": int(anchor_e6[0]),
            "output_last_e6_source_row": int(anchor_e6[-1]),
            "output_first_e6_frame_id": int(e6_frame_id[anchor_rows][0]),
            "output_last_e6_frame_id": int(e6_frame_id[anchor_rows][-1]),
        }
        segments.append(EpisodePlan(
            files=files,
            source_episode_index=source_episode_index,
            segment_index=segment_index,
            segment_count=segment_count,
            valid_run_index=valid_run_index,
            valid_run_count=len(runs),
            output_segment_id=output_segment_id,
            e6_rows=timing,
            source_cam0_association_rows=source_cam0_association_rows,
            source_cam1_association_rows=source_cam1_association_rows,
            recomputed_nearest_rows=recomputed_nearest_rows,
            strict_rows=strict_rows,
            e6_source_row_indices=anchor_e6,
            e6_frame_ids=e6_frame_id[anchor_rows],
            cam0_frame_indices=cam0_nearest[anchor_rows],
            cam1_frame_indices=cam1_nearest[anchor_rows],
            cam0_signed_delta_ns=cam0_delta[anchor_rows],
            cam1_signed_delta_ns=cam1_delta[anchor_rows],
            hand_pose_timestamps_ns=e6_mid[anchor_rows],
            state=state,
            action=action,
            action_is_pad=action_is_pad,
            sample_valid_h50=sample_valid_h50,
            source_fps=source_fps,
            source_stride=source_stride,
            camera_ratio=camera_ratio,
            cam0_offset=cam0_offset,
            cam1_offset=cam1_offset,
            strict_run_start=int(sample_rows[0]),
            strict_run_end_exclusive=int(sample_rows[-1]) + 1,
            output_count=output_count,
            quality=quality,
        ))

    source_quality = {
        **source_quality_common,
        "output_segment_ids": [plan.output_segment_id for plan in segments],
        "total_output_rows": int(sum(plan.output_count for plan in segments)),
        "total_sample_valid_h50_rows": int(
            sum(plan.sample_valid_h50.sum() for plan in segments)
        ),
        "total_terminal_padded_anchor_rows": int(
            sum((~plan.sample_valid_h50).sum() for plan in segments)
        ),
        "total_terminal_padded_action_slots": int(
            sum(plan.action_is_pad.sum() for plan in segments)
        ),
    }
    return SourceEpisodePlan(
        files=files,
        source_episode_index=source_episode_index,
        segments=segments,
        quality=source_quality,
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
    input_format: str | None = None,
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
    input_args = ["-f", input_format] if input_format is not None else []
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *input_args,
        "-i", str(source),
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
        "source.e6_source_row_index": {
            "dtype": "int64", "shape": (1,), "names": ["source_row_index"],
        },
        "source.e6_frame_id": {
            "dtype": "int64", "shape": (1,), "names": ["frame_id"],
        },
        "source.cam0_frame_index": {
            "dtype": "int64", "shape": (1,), "names": ["frame_index"],
        },
        "source.cam1_frame_index": {
            "dtype": "int64", "shape": (1,), "names": ["frame_index"],
        },
        "source.cam0_signed_delta_ns": {
            "dtype": "int64", "shape": (1,), "names": ["signed_delta_ns"],
        },
        "source.cam1_signed_delta_ns": {
            "dtype": "int64", "shape": (1,), "names": ["signed_delta_ns"],
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
        "source.e6_source_row_index": plan.e6_source_row_indices[:, None],
        "source.e6_frame_id": plan.e6_frame_ids[:, None],
        "source.cam0_frame_index": plan.cam0_frame_indices[:, None],
        "source.cam1_frame_index": plan.cam1_frame_indices[:, None],
        "source.cam0_signed_delta_ns": plan.cam0_signed_delta_ns[:, None],
        "source.cam1_signed_delta_ns": plan.cam1_signed_delta_ns[:, None],
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
    print(f"selected_source_episodes={len(episodes)}")
    if args.expected_episodes and len(episodes) != args.expected_episodes:
        print(
            f"WARNING: expected {args.expected_episodes} episodes but selected {len(episodes)}; "
            "convert will refuse until --expected-episodes or selection is corrected"
        )
    source_plans: list[SourceEpisodePlan] = []
    plans: list[EpisodePlan] = []
    for index, episode in enumerate(episodes):
        source_plan = make_plan(
            locate_episode_files(episode),
            index,
            max_alignment_ns=int(round(args.max_alignment_ms * 1_000_000)),
            max_hand_age_ns=int(round(args.max_hand_age_ms * 1_000_000)),
            hand_alignment=args.hand_alignment,
        )
        source_plans.append(source_plan)
        plans.extend(source_plan.segments)
        print(
            f"[{index + 1}/{len(episodes)}] {episode.name}: "
            f"E6={source_plan.segments[0].source_fps}Hz "
            f"stride={source_plan.segments[0].source_stride} "
            f"ratio={source_plan.segments[0].camera_ratio} "
            f"offsets=cam0:{source_plan.segments[0].cam0_offset}/"
            f"cam1:{source_plan.segments[0].cam1_offset} "
            f"valid_runs={source_plan.quality['valid_sample_sequence_count']} "
            f"segments={len(source_plan.segments)} "
            f"singletons={source_plan.quality['discarded_singleton_sequence_count']} "
            f"output={source_plan.quality['total_output_rows']}"
        )
    source_ids = [source_plan.files.episode.name for source_plan in source_plans]
    output_segment_ids = [plan.output_segment_id for plan in plans]
    source_ids_sha256 = hashlib.sha256("\n".join(source_ids).encode("utf-8")).hexdigest()
    output_segment_ids_sha256 = hashlib.sha256(
        "\n".join(output_segment_ids).encode("utf-8")
    ).hexdigest()
    summary = {
        "mode": "inspect-only-no-writes",
        "source": str(args.source.resolve()),
        "target_not_created": str(args.target.resolve()),
        "selected_episodes": len(source_plans),
        "selected_source_episodes": len(source_plans),
        "planned_output_episodes": len(plans),
        "selected_episode_ids_sha256": source_ids_sha256,
        "source_episode_ids_sha256": source_ids_sha256,
        "output_segment_ids_sha256": output_segment_ids_sha256,
        "expected_episodes": args.expected_episodes,
        "output_fps": OUTPUT_FPS,
        "action_horizon": ACTION_HORIZON,
        "max_alignment_ms": args.max_alignment_ms,
        "max_hand_age_ms": args.max_hand_age_ms,
        "hand_alignment": args.hand_alignment,
        "state_shape": [STATE_DIM],
        "action_shape": [ACTION_HORIZON, STATE_DIM],
        "total_output_rows_planned": sum(plan.output_count for plan in plans),
        "total_full_real_h50_rows_planned": int(
            sum(plan.sample_valid_h50.sum() for plan in plans)
        ),
        "total_padded_action_slots_planned": int(
            sum(plan.action_is_pad.sum() for plan in plans)
        ),
        "min_output_rows_per_output_episode": min(plan.output_count for plan in plans),
        "max_output_rows_per_output_episode": max(plan.output_count for plan in plans),
        "source_episodes_with_multiple_output_segments": [
            source_plan.files.episode.name
            for source_plan in source_plans
            if len(source_plan.segments) > 1
        ],
        "total_discarded_singleton_sequences": int(sum(
            source_plan.quality["discarded_singleton_sequence_count"]
            for source_plan in source_plans
        )),
        "discarded_singleton_sequences": [
            {
                "source_episode_id": source_plan.files.episode.name,
                **record,
            }
            for source_plan in source_plans
            for record in source_plan.quality["discarded_singleton_sequences"]
        ],
        "min_strict_10hz_retention_ratio": min(
            source_plan.quality["strict_10hz_retention_ratio"]
            for source_plan in source_plans
        ),
        "max_abs_cam0_alignment_ms": max(
            source_plan.quality["strict_cam0_max_abs_delta_ns"]
            for source_plan in source_plans
        ) / 1_000_000,
        "max_abs_cam1_alignment_ms": max(
            source_plan.quality["strict_cam1_max_abs_delta_ns"]
            for source_plan in source_plans
        ) / 1_000_000,
        "max_abs_left_hand_alignment_ms": max(
            source_plan.quality["absolute_hand_quality"]["left_max_fresh_abs_delta_ns"]
            for source_plan in source_plans
        ) / 1_000_000,
        "max_abs_right_hand_alignment_ms": max(
            source_plan.quality["absolute_hand_quality"]["right_max_fresh_abs_delta_ns"]
            for source_plan in source_plans
        ) / 1_000_000,
        "output_segments_with_zero_full_real_h50_rows": [
            plan.output_segment_id
            for plan in plans
            if not bool(plan.sample_valid_h50.any())
        ],
    }
    if not args.compact:
        summary["source_episodes"] = [
            source_plan.quality for source_plan in source_plans
        ]
        summary["output_segments"] = [plan.quality for plan in plans]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def convert(args: argparse.Namespace) -> int:
    if args.confirm != "CREATE_LEROBOT_DATASET":
        raise ValueError("conversion requires --confirm CREATE_LEROBOT_DATASET")
    if args.expected_episodes <= 0:
        raise ValueError("conversion requires --expected-episodes N to lock the source snapshot")
    source = args.source.resolve()
    target = args.target.resolve()
    if (
        target == source
        or target.is_relative_to(source)
        or source.is_relative_to(target)
    ):
        raise ValueError(f"source and target directories must not overlap: {source} / {target}")
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

    source_plans = [
        make_plan(
            locate_episode_files(episode),
            index,
            max_alignment_ns=int(round(args.max_alignment_ms * 1_000_000)),
            max_hand_age_ns=int(round(args.max_hand_age_ms * 1_000_000)),
            hand_alignment=args.hand_alignment,
        )
        for index, episode in enumerate(episodes)
    ]
    plans = [
        segment
        for source_plan in source_plans
        for segment in source_plan.segments
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
            robot_type="umi_task_v1_dual_hand_se3_rot6d_hand30_h50",
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
                        plan.files.head_video, plan.e6_source_row_indices,
                        "crop=1600:1200:1600:0", "hevc",
                    ),
                    "cam0": (
                        plan.files.cam0_video, plan.cam0_frame_indices,
                        None, "mjpeg",
                    ),
                    "cam1": (
                        plan.files.cam1_video, plan.cam1_frame_indices,
                        None, "mjpeg",
                    ),
                }
                for key, role in VIDEO_KEYS.items():
                    video_target = staging / metadata.get_video_file_path(episode_index, key)
                    source_video, frame_indices, source_crop, input_format = sources[role]
                    future = pool.submit(
                        encode_frame_selection,
                        source_video, video_target, frame_indices,
                        args.video_width, args.video_height, args.resize_mode, args.crf,
                        source_crop, input_format,
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
            episode_stats = {
                key: numeric_stats(value)
                for key, value in arrays.items()
                if key != "action"
            }
            real_action_slots = plan.action[~plan.action_is_pad]
            # A two-sample run legitimately emits one observation whose whole
            # H50 action is terminal padding.  Keep that episode and its mask,
            # but omit its zero-count action stats so it cannot introduce NaN
            # or fake values into LeRobot's aggregate action statistics.
            if len(real_action_slots):
                episode_stats["action"] = numeric_stats(real_action_slots)
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
                audit_dir / "source_cam0_associations.csv",
                plan.source_cam0_association_rows,
            )
            write_csv_rows(
                audit_dir / "source_cam1_associations.csv",
                plan.source_cam1_association_rows,
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
                "source_episode_id": plan.files.episode.name,
                "source_episode_index": plan.source_episode_index,
                "segment_index": plan.segment_index,
                "segment_count": plan.segment_count,
                "valid_run_index": plan.valid_run_index,
                "valid_run_count": plan.valid_run_count,
                "output_segment_id": plan.output_segment_id,
                "source_episode": str(plan.files.episode),
                "source_hashes": {
                    "camera/hand_pose.csv": sha256(plan.files.relative_hand_pose_csv),
                    "camera/e6_rgb_stream_metainfo.csv": sha256(plan.files.rgb_metainfo_csv),
                    "sync/e6_rgb_timing.csv": sha256(plan.files.e6_timing_csv),
                    "extensions/customer_camera/cam0/frames.csv": sha256(plan.files.cam0_frames_csv),
                    "extensions/customer_camera/cam1/frames.csv": sha256(plan.files.cam1_frames_csv),
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
                f"[{episode_index + 1}/{len(plans)}] {plan.output_segment_id}: "
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
            "schema": "umi-task-v1-dual-hand-pose-lerobot30-h50",
            "schema_version": 4,
            "status": "complete",
            "source": str(source),
            "source_episode_ids_sha256": hashlib.sha256(
                "\n".join(
                    source_plan.files.episode.name for source_plan in source_plans
                ).encode("utf-8")
            ).hexdigest(),
            "output_segment_ids_sha256": hashlib.sha256(
                "\n".join(plan.output_segment_id for plan in plans).encode("utf-8")
            ).hexdigest(),
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
                    "extensions/customer_camera/cam0/frames.csv",
                    "extensions/customer_camera/cam1/frames.csv",
                ],
                "strict_max_delta_ns": int(round(args.max_alignment_ms * 1_000_000)),
                "max_hand_age_ns": int(round(args.max_hand_age_ms * 1_000_000)),
                "hand_state_policy": args.hand_alignment,
                "hand_state_offline_noncausal": args.hand_alignment == "nearest",
                "camera_policy": (
                    "E6 10Hz grid, nearest timestamp, strict monotonic one-to-one; "
                    "every valid run with >=2 samples becomes a separate output episode"
                ),
                "source_associations_are_audit_only": [
                    "extensions/customer_camera/derived/cam0_associations.csv",
                    "extensions/customer_camera/derived/cam1_associations.csv",
                ],
            },
            "video_keys": VIDEO_KEYS,
            "video_preprocessing": {
                "head_rgb_source": "camera/e6_rgb.h265 (E6 3200x1200 stereo)",
                "head_rgb_crop": "right eye crop=1600:1200:1600:0",
                "cam0_source": "extensions/customer_camera/cam0/media.mjpeg",
                "cam1_source": "extensions/customer_camera/cam1/media.mjpeg",
                "cam0_output_key": "observation.images.left_wrist_rgb",
                "cam1_output_key": "observation.images.right_wrist_rgb",
                "cam0_cam1_physical_hand_mapping": (
                    "project contract cam0=left wrist, cam1=right wrist; "
                    "native source metadata itself exposes acquisition IDs only"
                ),
                "cam0_cam1_crop": None,
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
                "openpi_action_padding_mask_key": "action_is_pad",
                "full_real_horizon_filter_key": None,
                "full_real_horizon_audit_key": "sample_valid_h50",
                "current_openpi_sample_filter_key": None,
                "loss_reduction": "sum(loss[~action_is_pad]) / count(~action_is_pad)",
                "padding_contract": (
                    "the complete dataset retains terminal rows and action_is_pad; "
                    "the 5090 task_v1 OpenPI loader propagates action_is_pad as a "
                    "per-slot loss mask, so terminal anchors must not be filtered; "
                    "sample_valid_h50 is audit-only/optional compatibility metadata"
                ),
                "action_normalization_stats": "meta/stats.json action stats are 30-D and use only unpadded action slots",
            },
            "total_source_episodes": len(source_plans),
            "total_episodes": len(plans),
            "total_output_episodes": len(plans),
            "source_episodes_with_multiple_output_segments": [
                source_plan.files.episode.name
                for source_plan in source_plans
                if len(source_plan.segments) > 1
            ],
            "total_discarded_singleton_sequences": int(sum(
                source_plan.quality["discarded_singleton_sequence_count"]
                for source_plan in source_plans
            )),
            "discarded_singleton_sequences": [
                {
                    "source_episode_id": source_plan.files.episode.name,
                    **record,
                }
                for source_plan in source_plans
                for record in source_plan.quality["discarded_singleton_sequences"]
            ],
            "total_frames": global_start,
            "total_full_real_h50_frames": int(
                sum(plan.sample_valid_h50.sum() for plan in plans)
            ),
            "total_partial_h50_frames": int(
                sum((~plan.sample_valid_h50).sum() for plan in plans)
            ),
            "total_padded_action_slots": int(
                sum(plan.action_is_pad.sum() for plan in plans)
            ),
            "total_real_action_slots": int(
                global_start * ACTION_HORIZON
                - sum(plan.action_is_pad.sum() for plan in plans)
            ),
            "episodes_with_zero_full_real_h50_frames": [
                plan.output_segment_id
                for plan in plans
                if not bool(plan.sample_valid_h50.any())
            ],
            "source_episodes": [
                source_plan.quality for source_plan in source_plans
            ],
            "episodes": manifests,
        }
        atomic_json(staging / "meta" / "umi_conversion.json", contract)
        shutil.copy2(Path(__file__).resolve(), staging / "meta" / Path(__file__).name)
        (staging / "README.md").write_text(
            "# UMI dual hand-pose + absolute fingers, 10 Hz / H50\n\n"
            "State is 30-D: adjacent body-frame left/right hand-pose deltas "
            "from `camera/hand_pose.csv`, followed by absolute finger12. Each row "
            "already contains one H=50 action chunk whose two EEF targets all "
            "share the current row anchor; future fingers remain absolute. Do "
            "not slice future action rows or apply another delta transform. "
            "Every valid aligned run with at least two samples is an independent "
            "LeRobot episode, so actions never cross an invalid boundary. "
            "Terminal action slots are explicit in `action_is_pad`; training "
            "must mask them per slot. The 5090 task_v1 OpenPI stack propagates this "
            "mask into the loss, so do not filter terminal rows with "
            "`sample_valid_h50`; that field is audit-only/optional compatibility "
            "metadata. Videos are never padded. "
            "The 30-D action statistics in `meta/stats.json` exclude every padded slot. "
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
    sequences = valid_sample_sequences(
        np.asarray([True, True, False, True, True, True]),
        np.asarray([True, True, True, False, True, True]),
    )
    if sequences != [(0, 2), (3, 6)]:
        raise AssertionError("10 Hz validity segmentation failed")
    split_sequences = valid_sample_sequences(
        np.asarray([True, False, True, True, False, True, True, True, False, True]),
        np.ones(10, dtype=bool),
    )
    emitted_sequences = [
        (start, end) for start, end in split_sequences if end - start >= 2
    ]
    discarded_singletons = [
        (start, end) for start, end in split_sequences if end - start == 1
    ]
    if emitted_sequences != [(2, 4), (5, 8)] or discarded_singletons != [(0, 1), (9, 10)]:
        raise AssertionError("multi-run emission/singleton audit failed")
    synthetic_segment_length = 4
    synthetic_anchors = np.arange(1, synthetic_segment_length, dtype=np.int64)
    synthetic_requested = synthetic_anchors[:, None] + np.arange(
        1, ACTION_HORIZON + 1, dtype=np.int64
    )[None, :]
    synthetic_pad = synthetic_requested >= synthetic_segment_length
    synthetic_targets = np.minimum(synthetic_requested, synthetic_segment_length - 1)
    if (
        not synthetic_pad[-1].all()
        or synthetic_pad[0, 0]
        or not synthetic_pad[0, 2:].all()
        or int(synthetic_targets.max()) != synthetic_segment_length - 1
    ):
        raise AssertionError("per-segment terminal H50 padding failed")
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
    sampled, fresh, delta, selected = sample_absolute_hand(
        np.asarray([0, 90, 210]),
        np.arange(18, dtype=np.float64).reshape(3, 6),
        np.asarray([100, 220]),
    )
    if (
        not np.array_equal(selected, (1, 2))
        or not fresh.all()
        or not np.array_equal(delta, (-10, -10))
    ):
        raise AssertionError("nearest absolute-hand sampling failed")
    _, causal_fresh, causal_delta, causal_selected = sample_absolute_hand(
        np.asarray([0, 90, 210]),
        np.arange(18, dtype=np.float64).reshape(3, 6),
        np.asarray([100, 220]),
        alignment="causal",
    )
    if (
        not np.array_equal(causal_selected, (1, 2))
        or not causal_fresh.all()
        or not np.array_equal(causal_delta, (-10, -10))
    ):
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
        help="maximum absolute Revo2 hand-state timestamp error (default: 100 ms)",
    )
    parser.add_argument(
        "--hand-alignment", choices=("nearest", "causal"), default="nearest",
        help=(
            "align Revo2 hand states by nearest timestamp (default) or causal ZOH; "
            "nearest is recommended for task_v1 because periodic roughly "
            "116-118 ms packet gaps can fragment causal ZOH under a 100 ms bound"
        ),
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
    if hasattr(args, "max_alignment_ms") and args.max_alignment_ms > 100.0:
        parser.error("--max-alignment-ms must be <= 100 to satisfy the task_v1 contract")
    if hasattr(args, "max_hand_age_ms") and (
        not math.isfinite(args.max_hand_age_ms) or args.max_hand_age_ms <= 0
    ):
        parser.error("--max-hand-age-ms must be finite and positive")
    if hasattr(args, "max_hand_age_ms") and args.max_hand_age_ms > 100.0:
        parser.error("--max-hand-age-ms must be <= 100 to satisfy the task_v1 contract")
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
