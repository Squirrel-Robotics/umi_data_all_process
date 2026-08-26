#!/usr/bin/env python3
"""Convert UMI controller poses to validated frame-local hand-pose deltas.

The built-in calibration is the measured v3 controller-to-hand transform:
local +X points forward, local +Z points up, and +Y = Z cross X.  For each
hand, the output stores ``inverse(T_hand_previous) * T_hand_current``.

The script accepts either one CSV file or a dataset root.  Dataset mode finds
inputs with a configurable glob and writes one output next to every input (or
into a mirrored output tree).  It previews by default; pass --execute to write.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]

OUTPUT_COLUMNS = (
    "frame_number",
    "previous_frame_number",
    "timestamp_ns",
    "dt_ns",
    "left_relative_valid",
    "left_local_dpx",
    "left_local_dpy",
    "left_local_dpz",
    "left_local_dqx",
    "left_local_dqy",
    "left_local_dqz",
    "left_local_dqw",
    "right_relative_valid",
    "right_local_dpx",
    "right_local_dpy",
    "right_local_dpz",
    "right_local_dqx",
    "right_local_dqy",
    "right_local_dqz",
    "right_local_dqw",
)

FRAME_COLUMN_CANDIDATES = ("frame_number", "frame_id", "sequence_number")
TIMESTAMP_COLUMN_CANDIDATES = (
    "timestamp_ns",
    "e6_mid_exposure_realtime_ns",
    "e6_mid_exposure_utc_ns",
    "e6_mid_exposure_boot_ns",
    "controller_sample_realtime_ns",
    "controller_sample_boot_ns",
)

# Native UMI coordinates, millimetres.  The left points are the full X mirror
# of the right points.  These values reproduce the measured v3 calibration.
DEFAULT_CALIBRATION: dict[str, dict[str, list[float]]] = {
    "left": {
        "origin_mm": [-16.506, 57.416, 135.536],
        "forward_mm": [85.494, 57.416, 135.536],
        "up_mm": [-16.506, 52.244, 126.976],
    },
    "right": {
        "origin_mm": [16.506, 57.416, 135.536],
        "forward_mm": [-85.494, 57.416, 135.536],
        "up_mm": [16.506, 52.244, 126.976],
    },
}

IDENTITY_VALUES = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
Q_UMI_TO_ROS: Quaternion = (0.5, -0.5, -0.5, 0.5)


@dataclass(frozen=True)
class Pose:
    timestamp_ns: int
    position: Vector3
    quaternion: Quaternion


@dataclass(frozen=True)
class Frame:
    frame_number: int
    timestamp_ns: int
    left: Pose | None
    right: Pose | None


@dataclass(frozen=True)
class Job:
    source: Path
    target: Path


@dataclass(frozen=True)
class InputSchema:
    frame_column: str
    timestamp_column: str
    matched_column: str | None


@dataclass(frozen=True)
class JobResult:
    source: str
    target: str
    rows: int
    first_timestamp_ns: int
    last_timestamp_ns: int
    duration_s: float
    left_valid_deltas: int
    right_valid_deltas: int
    max_quaternion_norm_error: float
    max_position_reconstruction_error_m: float
    max_rotation_reconstruction_error_rad: float
    frame_column: str
    timestamp_column: str
    matched_column: str | None
    source_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="单个输入 CSV，或包含多个数据单元的根目录")
    parser.add_argument(
        "--input-glob",
        default="*/camera/e6_rgb_controller_poses.csv",
        help="目录模式下的输入匹配规则",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="单文件模式的输出路径；默认写到输入旁边的 hand_pose.csv",
    )
    parser.add_argument(
        "--output-name",
        default="hand_pose.csv",
        help="目录模式下每个输出的文件名",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="目录模式下写入镜像目录树；默认写到每个输入文件旁边",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        help="要求发现的输入文件数量严格等于此值",
    )
    parser.add_argument(
        "--frame-column",
        default="auto",
        help="帧号列名；默认 auto",
    )
    parser.add_argument(
        "--timestamp-column",
        default="auto",
        help="时间戳列名；默认 auto",
    )
    parser.add_argument(
        "--matched-column",
        default="auto",
        help="匹配有效标志列；默认自动使用 pose_matched，设为 none 可禁用",
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=("ros", "native"),
        default="ros",
        help="输出坐标基；默认 ros: (x,y,z)=(-z,-x,y)_UMI",
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        help="自定义左右手 origin_mm/forward_mm/up_mm 标定 JSON",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="可选 JSON 处理报告路径",
    )
    existing_group = parser.add_mutually_exclusive_group()
    existing_group.add_argument(
        "--overwrite",
        action="store_true",
        help="原子替换已经存在的输出；默认遇到现有输出即停止",
    )
    existing_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已经存在的输出，只处理尚未生成的文件",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际生成文件；不加时只预览",
    )
    parser.add_argument("--verbose", action="store_true", help="逐项显示输入和输出")
    return parser.parse_args()


def normalize_quaternion(values: Sequence[float]) -> Quaternion:
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("encountered an invalid or zero-length quaternion")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def quaternion_multiply(a: Sequence[float], b: Sequence[float]) -> Quaternion:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quaternion_conjugate(q: Sequence[float]) -> Quaternion:
    return (-q[0], -q[1], -q[2], q[3])


def rotate_vector(q: Sequence[float], vector: Sequence[float]) -> Vector3:
    rotated = quaternion_multiply(
        quaternion_multiply(q, (vector[0], vector[1], vector[2], 0.0)),
        quaternion_conjugate(q),
    )
    return (rotated[0], rotated[1], rotated[2])


def normalize_vector(values: Sequence[float]) -> Vector3:
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("calibration contains an invalid or zero-length direction")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def subtract(a: Sequence[float], b: Sequence[float]) -> Vector3:
    return tuple(a[index] - b[index] for index in range(3))  # type: ignore[return-value]


def cross(a: Sequence[float], b: Sequence[float]) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(a[index] * b[index] for index in range(3))


def quaternion_from_axes(
    x_axis: Sequence[float], y_axis: Sequence[float], z_axis: Sequence[float]
) -> Quaternion:
    r00, r10, r20 = x_axis
    r01, r11, r21 = y_axis
    r02, r12, r22 = z_axis
    trace = r00 + r11 + r22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = ((r21 - r12) / scale, (r02 - r20) / scale, (r10 - r01) / scale, 0.25 * scale)
    elif r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        q = (0.25 * scale, (r01 + r10) / scale, (r02 + r20) / scale, (r21 - r12) / scale)
    elif r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        q = ((r01 + r10) / scale, 0.25 * scale, (r12 + r21) / scale, (r02 - r20) / scale)
    else:
        scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        q = ((r02 + r20) / scale, (r12 + r21) / scale, 0.25 * scale, (r10 - r01) / scale)
    return normalize_quaternion(q)


def vector_umi_to_ros(vector: Sequence[float]) -> Vector3:
    return (-vector[2], -vector[0], vector[1])


def convert_umi_to_ros(sample: Pose) -> Pose:
    quaternion = quaternion_multiply(
        quaternion_multiply(Q_UMI_TO_ROS, sample.quaternion),
        quaternion_conjugate(Q_UMI_TO_ROS),
    )
    return Pose(
        sample.timestamp_ns,
        vector_umi_to_ros(sample.position),
        normalize_quaternion(quaternion),
    )


def compose_pose(parent: Pose, child: Pose) -> Pose:
    rotated_offset = rotate_vector(parent.quaternion, child.position)
    position = tuple(parent.position[index] + rotated_offset[index] for index in range(3))
    quaternion = normalize_quaternion(quaternion_multiply(parent.quaternion, child.quaternion))
    return Pose(parent.timestamp_ns, position, quaternion)  # type: ignore[arg-type]


def relative_pose(previous: Pose, current: Pose) -> Pose:
    previous_q = normalize_quaternion(previous.quaternion)
    current_q = normalize_quaternion(current.quaternion)
    inverse_previous_q = quaternion_conjugate(previous_q)
    world_delta = tuple(
        current.position[index] - previous.position[index] for index in range(3)
    )
    position = rotate_vector(inverse_previous_q, world_delta)
    quaternion = normalize_quaternion(quaternion_multiply(inverse_previous_q, current_q))
    if quaternion[3] < 0.0:
        quaternion = tuple(-value for value in quaternion)  # type: ignore[assignment]
    return Pose(current.timestamp_ns, position, quaternion)


def calibration_point(
    calibration: dict[str, dict[str, list[float]]], side: str, name: str
) -> Vector3:
    try:
        raw = calibration[side][name]
        values = tuple(float(value) for value in raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid calibration field {side}.{name}") from exc
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"calibration field {side}.{name} must contain 3 finite numbers")
    return values  # type: ignore[return-value]


def load_calibration(path: Path | None) -> dict[str, dict[str, list[float]]]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_CALIBRATION))
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("calibration JSON root must be an object")
    for side in ("left", "right"):
        for field in ("origin_mm", "forward_mm", "up_mm"):
            calibration_point(value, side, field)
    return value


def hand_offset(
    calibration: dict[str, dict[str, list[float]]], side: str, coordinate_mode: str
) -> Pose:
    origin = calibration_point(calibration, side, "origin_mm")
    forward = calibration_point(calibration, side, "forward_mm")
    up = calibration_point(calibration, side, "up_mm")
    x_axis = normalize_vector(subtract(forward, origin))
    z_hint = normalize_vector(subtract(up, origin))
    if abs(dot(x_axis, z_hint)) > 1.0 - 1e-9:
        raise ValueError(f"{side} calibration forward and up directions are collinear")
    y_axis = normalize_vector(cross(z_hint, x_axis))
    z_axis = normalize_vector(cross(x_axis, y_axis))
    position: Vector3 = tuple(value * 0.001 for value in origin)  # type: ignore[assignment]
    if coordinate_mode == "ros":
        position = vector_umi_to_ros(position)
        x_axis = vector_umi_to_ros(x_axis)
        y_axis = vector_umi_to_ros(y_axis)
        z_axis = vector_umi_to_ros(z_axis)
    return Pose(0, position, quaternion_from_axes(x_axis, y_axis, z_axis))


def is_true(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes"):
        return True
    if normalized in ("0", "false", "no"):
        return False
    raise ValueError(f"invalid boolean flag: {value!r}")


def detect_column(
    fieldnames: Sequence[str], requested: str, candidates: Sequence[str], label: str
) -> str:
    if requested != "auto":
        if requested not in fieldnames:
            raise ValueError(f"requested {label} column is missing: {requested}")
        return requested
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise ValueError(f"cannot auto-detect {label} column; tried: {', '.join(candidates)}")


def detect_matched_column(fieldnames: Sequence[str], requested: str) -> str | None:
    if requested.lower() == "none":
        return None
    if requested == "auto":
        return "pose_matched" if "pose_matched" in fieldnames else None
    if requested not in fieldnames:
        raise ValueError(f"requested matched column is missing: {requested}")
    return requested


def pose_from_row(row: dict[str, str], side: str, timestamp_ns: int) -> Pose | None:
    if not is_true(row[f"{side}_active"]):
        return None
    position = tuple(float(row[f"{side}_p{axis}"]) for axis in "xyz")
    quaternion = normalize_quaternion(tuple(float(row[f"{side}_q{axis}"]) for axis in "xyzw"))
    if not all(math.isfinite(value) for value in (*position, *quaternion)):
        raise ValueError(f"non-finite {side} controller pose")
    return Pose(timestamp_ns, position, quaternion)  # type: ignore[arg-type]


def load_frames(path: Path, args: argparse.Namespace) -> tuple[list[Frame], InputSchema]:
    frames: list[Frame] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        frame_column = detect_column(fieldnames, args.frame_column, FRAME_COLUMN_CANDIDATES, "frame")
        timestamp_column = detect_column(
            fieldnames, args.timestamp_column, TIMESTAMP_COLUMN_CANDIDATES, "timestamp"
        )
        matched_column = detect_matched_column(fieldnames, args.matched_column)
        required = [
            f"{side}_{kind}{axis}"
            for side in ("left", "right")
            for kind, axes in (("p", "xyz"), ("q", "xyzw"))
            for axis in axes
        ] + ["left_active", "right_active"]
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise ValueError(f"missing controller columns: {', '.join(missing)}")

        previous_timestamp = -1
        active_pose_count = 0
        for line_number, row in enumerate(reader, start=2):
            try:
                frame_number = int(row[frame_column])
                timestamp_ns = int(row[timestamp_column])
                matched = matched_column is None or is_true(row[matched_column])
                left = pose_from_row(row, "left", timestamp_ns) if matched else None
                right = pose_from_row(row, "right", timestamp_ns) if matched else None
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid data at CSV line {line_number}: {exc}") from exc
            if timestamp_ns <= previous_timestamp:
                raise ValueError(f"timestamps are not strictly increasing at CSV line {line_number}")
            active_pose_count += int(left is not None) + int(right is not None)
            frames.append(Frame(frame_number, timestamp_ns, left, right))
            previous_timestamp = timestamp_ns

    if not frames:
        raise ValueError("input CSV contains no data rows")
    if active_pose_count == 0:
        raise ValueError("input CSV contains no matched active controller poses")
    return frames, InputSchema(frame_column, timestamp_column, matched_column)


def mapped_hand_frames(
    frames: Iterable[Frame], offsets: dict[str, Pose], coordinate_mode: str
) -> list[Frame]:
    mapped = []
    for frame in frames:
        poses: dict[str, Pose | None] = {}
        for side in ("left", "right"):
            controller = getattr(frame, side)
            if controller is not None and coordinate_mode == "ros":
                controller = convert_umi_to_ros(controller)
            poses[side] = compose_pose(controller, offsets[side]) if controller is not None else None
        mapped.append(Frame(frame.frame_number, frame.timestamp_ns, poses["left"], poses["right"]))
    return mapped


def delta_fields(previous: Pose | None, current: Pose | None) -> tuple[list[str | int], int]:
    valid = previous is not None and current is not None
    values = IDENTITY_VALUES
    if valid:
        delta = relative_pose(previous, current)  # type: ignore[arg-type]
        values = (*delta.position, *delta.quaternion)
    return [int(valid), *(format(value, ".17g") for value in values)], int(valid)


def stage_output(target: Path, frames: list[Frame]) -> tuple[Path, dict[str, Any]]:
    temporary_name: str | None = None
    valid_counts = {"left": 0, "right": 0}
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{target.name}.",
            suffix=".staged",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            writer = csv.writer(stream)
            writer.writerow(OUTPUT_COLUMNS)
            for index, current in enumerate(frames):
                previous = frames[index - 1] if index else None
                previous_number: str | int = previous.frame_number if previous is not None else ""
                dt_ns: str | int = (
                    current.timestamp_ns - previous.timestamp_ns if previous is not None else ""
                )
                left_fields, left_valid = delta_fields(
                    previous.left if previous is not None else None, current.left
                )
                right_fields, right_valid = delta_fields(
                    previous.right if previous is not None else None, current.right
                )
                valid_counts["left"] += left_valid
                valid_counts["right"] += right_valid
                writer.writerow(
                    [
                        current.frame_number,
                        previous_number,
                        current.timestamp_ns,
                        dt_ns,
                        *left_fields,
                        *right_fields,
                    ]
                )
        stage = Path(temporary_name)
        os.chmod(stage, 0o644)
        metrics = validate_output(stage, frames)
        if metrics["left_valid_deltas"] != valid_counts["left"]:
            raise ValueError("left valid-delta count changed during validation")
        if metrics["right_valid_deltas"] != valid_counts["right"]:
            raise ValueError("right valid-delta count changed during validation")
        return stage, metrics
    except Exception:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        raise


def output_pose(row: dict[str, str], side: str, timestamp_ns: int) -> Pose:
    return Pose(
        timestamp_ns,
        tuple(float(row[f"{side}_local_dp{axis}"]) for axis in "xyz"),  # type: ignore[arg-type]
        tuple(float(row[f"{side}_local_dq{axis}"]) for axis in "xyzw"),  # type: ignore[arg-type]
    )


def quaternion_error(a: Sequence[float], b: Sequence[float]) -> float:
    value = abs(sum(left * right for left, right in zip(a, b)))
    return 2.0 * math.acos(max(-1.0, min(1.0, value)))


def validate_output(path: Path, mapped: list[Frame]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
            raise ValueError(f"unexpected output schema in {path}")
        rows = list(reader)
    if len(rows) != len(mapped):
        raise ValueError(f"output row count mismatch: {len(rows)} != {len(mapped)}")

    valid_counts = {"left": 0, "right": 0}
    max_norm_error = 0.0
    max_position_error = 0.0
    max_rotation_error = 0.0
    for index, (row, current) in enumerate(zip(rows, mapped)):
        previous = mapped[index - 1] if index else None
        if int(row["frame_number"]) != current.frame_number:
            raise ValueError(f"frame number mismatch at output row {index + 2}")
        if int(row["timestamp_ns"]) != current.timestamp_ns:
            raise ValueError(f"timestamp mismatch at output row {index + 2}")
        expected_previous = "" if previous is None else str(previous.frame_number)
        expected_dt = "" if previous is None else str(current.timestamp_ns - previous.timestamp_ns)
        if row["previous_frame_number"] != expected_previous or row["dt_ns"] != expected_dt:
            raise ValueError(f"previous-frame metadata mismatch at output row {index + 2}")

        for side in ("left", "right"):
            current_pose = getattr(current, side)
            previous_pose = getattr(previous, side) if previous is not None else None
            expected_valid = previous_pose is not None and current_pose is not None
            if int(row[f"{side}_relative_valid"]) != int(expected_valid):
                raise ValueError(f"{side} validity mismatch at output row {index + 2}")
            delta = output_pose(row, side, current.timestamp_ns)
            values = (*delta.position, *delta.quaternion)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"non-finite {side} output at row {index + 2}")
            norm_error = abs(math.sqrt(sum(value * value for value in delta.quaternion)) - 1.0)
            max_norm_error = max(max_norm_error, norm_error)
            if norm_error > 1e-12:
                raise ValueError(f"non-unit {side} quaternion at output row {index + 2}")
            if not expected_valid:
                if any(abs(actual - expected) > 1e-15 for actual, expected in zip(values, IDENTITY_VALUES)):
                    raise ValueError(f"non-identity invalid {side} delta at row {index + 2}")
                continue

            valid_counts[side] += 1
            reconstructed = compose_pose(previous_pose, delta)  # type: ignore[arg-type]
            position_error = math.sqrt(
                sum(
                    (actual - expected) ** 2
                    for actual, expected in zip(reconstructed.position, current_pose.position)  # type: ignore[union-attr]
                )
            )
            rotation_error = quaternion_error(reconstructed.quaternion, current_pose.quaternion)  # type: ignore[union-attr]
            max_position_error = max(max_position_error, position_error)
            max_rotation_error = max(max_rotation_error, rotation_error)
            if position_error > 1e-12 or rotation_error > 1e-7:
                raise ValueError(f"failed to reconstruct {side} pose at output row {index + 2}")

    return {
        "rows": len(rows),
        "left_valid_deltas": valid_counts["left"],
        "right_valid_deltas": valid_counts["right"],
        "max_quaternion_norm_error": max_norm_error,
        "max_position_reconstruction_error_m": max_position_error,
        "max_rotation_reconstruction_error_rad": max_rotation_error,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_jobs(args: argparse.Namespace) -> tuple[Path, list[Job], list[Job], int]:
    input_path = args.input.expanduser().resolve()
    if input_path.is_file():
        if args.output_root is not None:
            raise ValueError("--output-root is only valid when input is a directory")
        target = (
            args.output.expanduser().resolve()
            if args.output is not None
            else input_path.with_name(args.output_name)
        )
        jobs = [Job(input_path, target)]
        root = input_path.parent
    elif input_path.is_dir():
        if args.output is not None:
            raise ValueError("--output is only valid when input is one CSV file")
        output_root = args.output_root.expanduser().resolve() if args.output_root else None
        sources = sorted(path.resolve() for path in input_path.glob(args.input_glob) if path.is_file())
        jobs = []
        for source in sources:
            if output_root is None:
                target = source.with_name(args.output_name)
            else:
                target = output_root / source.relative_to(input_path).parent / args.output_name
            jobs.append(Job(source, target))
        root = input_path
    else:
        raise FileNotFoundError(f"input does not exist: {input_path}")

    if not jobs:
        raise ValueError(f"no input CSV files matched under {input_path}")
    if args.expected_count is not None and len(jobs) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} inputs, found {len(jobs)}")
    target_names = [os.fspath(job.target) for job in jobs]
    if len(set(target_names)) != len(target_names):
        raise ValueError("multiple inputs resolve to the same output path")
    pending = []
    skipped_existing = []
    for job in jobs:
        if job.source == job.target:
            raise ValueError(f"source and output are the same file: {job.source}")
        if os.path.lexists(job.target):
            if args.overwrite:
                pending.append(job)
            elif args.skip_existing:
                skipped_existing.append(job)
            else:
                raise FileExistsError(
                    f"output exists: {job.target} "
                    "(use --skip-existing or --overwrite)"
                )
        else:
            pending.append(job)
    return root, pending, skipped_existing, len(jobs)


def write_report(path: Path, report: dict[str, Any], overwrite: bool) -> None:
    if os.path.lexists(path) and not overwrite:
        raise FileExistsError(f"report exists: {path} (use --overwrite to replace)")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".staged",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.chmod(temporary_name, 0o644)
        if overwrite:
            os.replace(temporary_name, path)
        else:
            os.link(temporary_name, path)
            os.unlink(temporary_name)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def main() -> int:
    args = parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in ("", ".", ".."):
        raise ValueError("--output-name must be one plain file name")
    if args.expected_count is not None and args.expected_count < 1:
        raise ValueError("--expected-count must be positive")

    calibration = load_calibration(args.calibration_json)
    offsets = {
        side: hand_offset(calibration, side, args.coordinate_mode)
        for side in ("left", "right")
    }
    root, jobs, skipped_existing, discovered_count = discover_jobs(args)
    print(
        f"{'执行' if args.execute else '预览'}：发现 {discovered_count} 个输入，"
        f"计划处理 {len(jobs)} 个，跳过已有输出 {len(skipped_existing)} 个，"
        f"输出名 {args.output_name}"
    )
    if args.verbose:
        for job in jobs:
            print(f"  {job.source} -> {job.target}")
    if not args.execute:
        print("未修改数据；确认后添加 --execute。")
        return 0

    report_path = args.report.expanduser().resolve() if args.report is not None else None
    if report_path is not None and os.path.lexists(report_path) and not args.overwrite:
        raise FileExistsError(f"report exists: {report_path} (use --overwrite to replace)")

    stages: list[tuple[Path, Path]] = []
    results: list[JobResult] = []
    try:
        for index, job in enumerate(jobs, start=1):
            job.target.parent.mkdir(parents=True, exist_ok=True)
            frames, schema = load_frames(job.source, args)
            mapped = mapped_hand_frames(frames, offsets, args.coordinate_mode)
            stage, metrics = stage_output(job.target, mapped)
            stages.append((stage, job.target))
            results.append(
                JobResult(
                    source=os.fspath(job.source),
                    target=os.fspath(job.target),
                    first_timestamp_ns=mapped[0].timestamp_ns,
                    last_timestamp_ns=mapped[-1].timestamp_ns,
                    duration_s=(mapped[-1].timestamp_ns - mapped[0].timestamp_ns) * 1e-9,
                    frame_column=schema.frame_column,
                    timestamp_column=schema.timestamp_column,
                    matched_column=schema.matched_column,
                    source_sha256=sha256_file(job.source),
                    **metrics,
                )
            )
            print(
                f"staged {index}/{len(jobs)} {job.source}: rows={metrics['rows']}, "
                f"left_valid={metrics['left_valid_deltas']}, "
                f"right_valid={metrics['right_valid_deltas']}"
            )

        for stage, target in stages:
            if args.overwrite:
                os.replace(stage, target)
            else:
                os.link(stage, target)
                stage.unlink()
        stages.clear()

        report = {
            "schema_version": "umi.hand_pose.relative.v3.measured_flange_forward_up",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "root": os.fspath(root),
            "input_glob": args.input_glob,
            "output_name": args.output_name,
            "discovered_input_count": discovered_count,
            "skipped_existing": [os.fspath(job.target) for job in skipped_existing],
            "coordinate_mode": args.coordinate_mode,
            "controller_to_hand_calibration_native_umi_mm": calibration,
            "axis_definition": {
                "x": "normalize(forward_mm - origin_mm)",
                "y": "normalize((up_mm - origin_mm) cross x)",
                "z": "normalize(x cross y)",
            },
            "relative_transform": "inverse(T_hand_previous) * T_hand_current",
            "translation_frame": "previous hand local frame",
            "translation_unit": "metre",
            "quaternion_order": "xyzw",
            "invalid_delta": "identity transform with relative_valid=0",
            "file_count": len(results),
            "total_rows": sum(result.rows for result in results),
            "files": [asdict(result) for result in results],
        }
        if report_path is not None:
            write_report(report_path, report, args.overwrite)
            print(f"report={report_path}")
        print(
            f"完成：生成 {len(results)} 个 {args.output_name}，"
            f"总行数 {report['total_rows']}。"
        )
        return 0
    finally:
        for stage, _ in stages:
            try:
                stage.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileExistsError,
        FileNotFoundError,
        PermissionError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
