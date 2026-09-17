#!/usr/bin/env python3
"""批量把 controller 绝对位姿转换为上一帧末端自身坐标系中的相对位姿。

Python 3.8+，仅标准库，复制本文件即可使用。不依赖旧脚本、ROS 或配置目录。
内置 2026-09-09 确认的 test_true 标定；默认物理左手读取 left_*，物理右手读取 right_*。
仅当原始通道与实物相反时，显式使用 --side-map swapped。
T_W_H = T_W_C @ T_C_H; T_W_F = T_W_H @ Trans(target_origin_x_m, 0, 0)。
D[t] = inverse(T_W_F[t-1]) @ T_W_F[t]，F 与 Hand 同方向，仅原点可偏移。
默认沿最终 Hand 局部 X 轴偏移 -42 mm；--target-origin-x-m 0 可取消偏移。
标定已包含最终轴方向；不再叠加旋转，网页世界坐标显示变换不参与本转换。
默认仅预览；--execute 写入输入旁的 hand_pose.csv。
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


SIDES = ("left", "right")
COMPONENTS = ("px", "py", "pz", "qx", "qy", "qz", "qw")
OUTPUT_COLUMNS = (
    "frame_number", "previous_frame_number", "timestamp_ns", "dt_ns",
) + tuple(
    name for side in SIDES for name in
    (f"{side}_relative_valid", *(f"{side}_local_d{c}" for c in COMPONENTS))
)
FRAME_COLUMNS = ("frame_number", "frame_id", "sequence_number")
TIME_COLUMNS = (
    "timestamp_ns", "e6_mid_exposure_realtime_ns", "e6_mid_exposure_utc_ns",
    "e6_mid_exposure_boot_ns", "controller_sample_realtime_ns", "controller_sample_boot_ns",
)
Vector = Tuple[float, float, float]
Matrix = Tuple[Vector, Vector, Vector]
IDENTITY: Matrix = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
INVALID_VALUES = (0, 0, 0, 0, 0, 0, 1)
DEFAULT_TARGET_ORIGIN_X_M = -0.042

# Row-major R_C_H: each COLUMN is a new Hand axis expressed in native controller C.
# Confirmed test_true axes: P1 -> +X for both hands, P2 -> -Y (left) / +Y (right).
# The measured Hand origins below EXCLUDE the target shift, which is applied once
# by load_hand_frames, along the FINAL Hand local X axis.
DEFAULT_CALIBRATION = {
    "schema_version": "umi.controller_to_hand.direct.v2",
    "calibration_id": "test_true_xy_aligned_20260909",
    "coordinate_system": "umi_native",
    "translation_unit": "m",
    "transforms": {
        "left": {
            "translation": [-0.007435, -0.015192, -0.053839],
            "rotation_matrix": [
                [0., 0., -1.],
                [-0.5439932518055669, 0.8390895911581822, 0.],
                [0.839089591158182, 0.5439932518055669, 0.],
            ],
        },
        "right": {
            "translation": [0.007435, -0.015192, -0.053839],
            "rotation_matrix": [
                [0., 0., 1.],
                [-0.5439932518055669, -0.8390895911581822, 0.],
                [0.839089591158182, -0.5439932518055669, 0.],
            ],
        },
    },
}


@dataclass(frozen=True)
class Pose:
    position: Vector
    rotation: Matrix


@dataclass(frozen=True)
class Frame:
    number: int
    timestamp: int
    left: Optional[Pose]
    right: Optional[Pose]


@dataclass(frozen=True)
class Job:
    source: Path
    target: Path


def transpose(a):
    return tuple(zip(*a))


def matvec(a, b):
    return tuple(sum(x * y for x, y in zip(row, b)) for row in a)


def matmul(a, b):
    return tuple(tuple(sum(x * y for x, y in zip(row, col))
                       for col in zip(*b)) for row in a)


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def compose(a, b):
    offset = matvec(a.rotation, b.position)
    return Pose(tuple(x + y for x, y in zip(a.position, offset)),
                matmul(a.rotation, b.rotation))


def relative(previous, current):
    inverse_rotation = transpose(previous.rotation)
    return Pose(matvec(inverse_rotation, subtract(current.position, previous.position)),
                matmul(inverse_rotation, current.rotation))


def finite(values, count):
    result = tuple(float(v) for v in values)
    if len(result) != count or not all(math.isfinite(v) for v in result):
        raise ValueError(f"需要 {count} 个有限数值，不能含 NaN/Inf")
    return result


def normalized_quaternion(values):
    q = finite(values, 4)
    norm = math.sqrt(sum(v * v for v in q))
    if not math.isfinite(norm) or norm < 1e-12 or abs(norm - 1.) > 0.05:
        raise ValueError(f"四元数长度异常：{norm}，允许归一化误差不超过 0.05")
    return tuple(v / norm for v in q)


def quaternion_matrix(values):
    x, y, z, w = normalized_quaternion(values)
    return (
        (1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
        (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
        (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)),
    )


def matrix_quaternion(r):
    trace = sum(r[i][i] for i in range(3))
    if trace > 0:
        s = math.sqrt(trace + 1.) * 2
        q = ((r[2][1]-r[1][2])/s, (r[0][2]-r[2][0])/s,
             (r[1][0]-r[0][1])/s, s/4)
    else:
        i = max(range(3), key=lambda k: r[k][k])
        j, k = (i+1) % 3, (i+2) % 3
        s = math.sqrt(1. + r[i][i] - r[j][j] - r[k][k]) * 2
        q = [0.] * 4
        q[i], q[j], q[k], q[3] = s/4, (r[j][i]+r[i][j])/s, (r[k][i]+r[i][k])/s, (r[k][j]-r[j][k])/s
    q = normalized_quaternion(q)
    return tuple(-v for v in q) if q[3] < 0 else q


def validate_rotation(r):
    if len(r) != 3:
        raise ValueError("rotation_matrix 必须是 3x3 矩阵")
    r = tuple(finite(row, 3) for row in r)
    product = matmul(transpose(r), r)
    error = max(abs(product[i][j] - IDENTITY[i][j]) for i in range(3) for j in range(3))
    det = (r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
           - r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
           + r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    if error > 1e-9 or abs(det - 1.) > 1e-9:
        raise ValueError(f"矩阵不是右手正交旋转：det={det}, orthogonal_error={error}")
    return r


def load_calibration(path):
    if path:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    else:
        value = DEFAULT_CALIBRATION
    if (value.get("schema_version") != "umi.controller_to_hand.direct.v2"
            or value.get("coordinate_system") != "umi_native"
            or value.get("translation_unit") != "m"):
        raise ValueError("标定需要 direct.v2 schema、umi_native 坐标、m 单位；不接受旧三点配置")
    offsets = {}
    for side in SIDES:
        entry = value["transforms"][side]
        offsets[side] = Pose(finite(entry["translation"], 3), validate_rotation(entry["rotation_matrix"]))
    return value, offsets


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="单个 controller CSV 或数据集根目录")
    parser.add_argument("--input-glob", default="**/camera/e6_rgb_controller_poses.csv")
    parser.add_argument("--output-name", default="hand_pose.csv", help="每个输入旁的输出文件名")
    parser.add_argument("--output", type=Path, help="仅单文件模式：指定输出 CSV")
    parser.add_argument("--output-root", type=Path, help="仅目录模式：保持相对目录层级另存")
    parser.add_argument("--expected-count", type=int, help="要求匹配文件数量恰好等于此值")
    parser.add_argument("--side-map", choices=("identity", "swapped"), default="identity",
                        help="默认 identity：物理左右手读取同名通道；swapped 显式交换左右输入")
    parser.add_argument("--calibration-json", type=Path, help="可选自定义最终 T_C_H；格式见配套说明")
    parser.add_argument("--target-origin-x-m", type=float, default=DEFAULT_TARGET_ORIGIN_X_M,
                        help="目标原点沿最终 Hand 局部 X 轴的偏移（米），左右各自应用；默认-0.042（-42mm），0取消偏移")
    parser.add_argument("--position-unit", choices=("m", "mm"), default="m", help="输入位置单位，默认 m；输出始终 m")
    parser.add_argument("--frame-column", default="auto")
    parser.add_argument("--timestamp-column", default="auto", help="整数纳秒列；不经过浮点转换")
    parser.add_argument("--matched-column", default="auto", help="auto 使用 pose_matched；none 禁用")
    parser.add_argument("--require-frame-step", type=int, default=1,
                        help="默认帧号连续 +1 才产生有效增量；0 表示按相邻 CSV 行计算")
    parser.add_argument("--max-dt-ns", type=int, help="可选最大帧间时间；超出则标记增量无效")
    parser.add_argument("--report", type=Path, help="可选处理报告 JSON，仅 --execute 时写入")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--skip-existing", action="store_true", help="跳过已有输出，不检查其标定版本")
    group.add_argument("--overwrite", action="store_true", help="明确允许替换已有输出")
    parser.add_argument("--execute", action="store_true", help="实际转换；不加仅预览文件计划")
    parser.add_argument("--verbose", action="store_true", help="显示每个输入和输出及处理统计")
    args = parser.parse_args(argv)
    if not math.isfinite(args.target_origin_x_m):
        parser.error("target-origin-x-m 必须是有限数值，单位为米")
    if args.require_frame_step < 0 or (args.max_dt_ns is not None and args.max_dt_ns <= 0):
        parser.error("require-frame-step 必须非负，max-dt-ns 必须为正")
    if args.expected_count is not None and args.expected_count <= 0:
        parser.error("expected-count 必须为正")
    if (not args.output_name or args.output_name in (".", "..")
            or "/" in args.output_name or "\\" in args.output_name):
        parser.error("output-name 必须是文件名，不得包含目录")
    return args


def output_path(path):
    # Do not resolve the final component: reject output symlinks, never follow them.
    path = path.expanduser().absolute()
    return path.parent.resolve() / path.name


def same_file(a, b):
    return a == b or (a.exists() and b.exists() and os.path.samefile(a, b))


def check_target(target, protected):
    if target.is_symlink():
        raise ValueError(f"拒绝向符号链接输出：{target}")
    if target.exists() and not target.is_file():
        raise ValueError(f"输出不是普通文件：{target}")
    if any(same_file(target, path) for path in protected):
        raise ValueError(f"输出与输入、配置、脚本或其他输出冲突：{target}")


def discover_jobs(args):
    root = args.input.expanduser().resolve()
    if root.is_file():
        if args.output_root:
            raise ValueError("单文件模式不能使用 --output-root")
        jobs = [Job(root, output_path(args.output or root.with_name(args.output_name)))]
    elif root.is_dir():
        if args.output:
            raise ValueError("目录模式请使用 --output-name 或 --output-root")
        jobs = []
        seen_sources = set()
        for path in sorted(root.glob(args.input_glob)):
            if not path.is_file():
                continue
            source = path.resolve()
            if source in seen_sources:
                raise ValueError(f"同一输入被重复匹配（可能是符号链接）：{source}")
            seen_sources.add(source)
            target = path.with_name(args.output_name)
            if args.output_root:
                target = args.output_root / path.relative_to(root).parent / args.output_name
            jobs.append(Job(source, output_path(target)))
    else:
        raise FileNotFoundError(f"输入不存在：{root}")
    if not jobs:
        raise ValueError("未匹配到输入 CSV，请检查路径和 --input-glob")
    if args.expected_count is not None and len(jobs) != args.expected_count:
        raise ValueError(f"匹配 {len(jobs)} 个文件，与 expected-count={args.expected_count} 不符")
    protected = [job.source for job in jobs] + [Path(__file__).resolve()]
    if args.calibration_json:
        protected.append(args.calibration_json.expanduser().resolve())
    targets, pending, skipped = [], [], []
    for job in jobs:
        check_target(job.target, protected + targets)
        targets.append(job.target)
        if job.target.exists():
            if args.skip_existing:
                skipped.append(job)
                continue
            if not args.overwrite:
                raise FileExistsError(f"输出已存在：{job.target}；使用 --skip-existing 或 --overwrite")
        pending.append(job)
    if args.report:
        args.report = output_path(args.report)
        check_target(args.report, protected + targets)
        if args.report.exists() and not args.overwrite:
            raise FileExistsError(f"报告已存在：{args.report}；请换名字或显式 --overwrite")
    return jobs, pending, skipped, protected


def detect_column(fields, requested, candidates):
    if requested != "auto":
        if requested not in fields:
            raise ValueError(f"缺少列：{requested}")
        return requested
    for name in candidates:
        if name in fields:
            return name
    raise ValueError(f"无法自动识别列，候选：{', '.join(candidates)}")


def boolean(value):
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes"):
        return True
    if normalized in ("0", "false", "no"):
        return False
    raise ValueError(f"无效布尔标记：{value!r}")


def load_hand_frames(path, args, offsets):
    """Return absolute poses of target F (default local -X 42 mm), or Hand at 0.

    The optional offset is right-multiplied in each final Hand frame, AFTER
    channel selection and calibrated axes. It never changes controller data.
    """
    frames = []
    side_map = dict(zip(SIDES, reversed(SIDES) if args.side_map == "swapped" else SIDES))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields):
            raise ValueError("CSV 表头有重复列")
        frame_col = detect_column(fields, args.frame_column, FRAME_COLUMNS)
        time_col = detect_column(fields, args.timestamp_column, TIME_COLUMNS)
        matched_col = args.matched_column
        if matched_col == "auto":
            matched_col = "pose_matched" if "pose_matched" in fields else None
        elif matched_col.lower() == "none":
            matched_col = None
        elif matched_col not in fields:
            raise ValueError(f"缺少匹配标志列：{matched_col}")
        required = [f"{s}_{c}" for s in SIDES for c in (*COMPONENTS, "active")]
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError(f"缺少 controller 列：{', '.join(missing)}")
        valid_flags = {s: [f"{s}_{c}" for c in ("active", "position_valid", "orientation_valid")
                           if f"{s}_{c}" in fields] for s in SIDES}
        for line, row in enumerate(reader, start=2):
            try:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError("列数与表头不一致")
                number, timestamp = int(row[frame_col]), int(row[time_col])
                if frames and timestamp <= frames[-1].timestamp:
                    raise ValueError("时间戳必须严格递增，不能重复或倒序")
                matched = matched_col is None or boolean(row[matched_col])
                poses = {}
                for side, source in side_map.items():
                    if not matched or not all(boolean(row[f]) for f in valid_flags[source]):
                        poses[side] = None
                        continue
                    scale = 0.001 if args.position_unit == "mm" else 1.
                    p = tuple(v * scale for v in finite((row[f"{source}_p{a}"] for a in "xyz"), 3))
                    r = quaternion_matrix(row[f"{source}_q{a}"] for a in "xyzw")
                    # Choose calibration by PHYSICAL side, after resolving the source channel.
                    poses[side] = compose(Pose(p, r), offsets[side])
                    if args.target_origin_x_m != 0.:
                        poses[side] = compose(poses[side], Pose((args.target_origin_x_m, 0., 0.), IDENTITY))
                        finite(poses[side].position, 3)
                frames.append(Frame(number, timestamp, poses["left"], poses["right"]))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"CSV 第 {line} 行：{exc}") from exc
    if not frames or not any(f.left is not None or f.right is not None for f in frames):
        raise ValueError("CSV 为空或没有任何匹配且有效的手柄位姿")
    return frames, {"frame_column": frame_col, "timestamp_column": time_col,
                    "matched_column": matched_col, "validity_columns": valid_flags}


def relative_rows(frames, args):
    rows = []
    stats = {"rows": len(frames), "left_valid_deltas": 0, "right_valid_deltas": 0,
             "discontinuous_pairs": 0, "max_reconstruction_position_error_m": 0.,
             "max_reconstruction_rotation_matrix_error": 0.}
    for index, current in enumerate(frames):
        previous = frames[index-1] if index else None
        dt = current.timestamp - previous.timestamp if previous else None
        continuous = previous is not None
        if previous and ((args.require_frame_step and current.number-previous.number != args.require_frame_step)
                         or (args.max_dt_ns is not None and dt > args.max_dt_ns)):
            continuous = False
            stats["discontinuous_pairs"] += 1
        row = [current.number, previous.number if previous else "", current.timestamp, dt if previous else ""]
        for side in SIDES:
            old = getattr(previous, side) if previous else None
            new = getattr(current, side)
            valid = continuous and old is not None and new is not None
            values = INVALID_VALUES
            if valid:
                delta = relative(old, new)
                values = (*delta.position, *matrix_quaternion(delta.rotation))
                reconstructed = compose(old, Pose(delta.position, quaternion_matrix(values[3:])))
                pe = math.sqrt(sum(v*v for v in subtract(reconstructed.position, new.position)))
                re = max(abs(reconstructed.rotation[i][j]-new.rotation[i][j]) for i in range(3) for j in range(3))
                if not math.isfinite(pe) or pe > 1e-9 or re > 1e-9:
                    raise ValueError(f"帧 {current.number} {side} 右乘重建校验失败")
                stats["max_reconstruction_position_error_m"] = max(stats["max_reconstruction_position_error_m"], pe)
                stats["max_reconstruction_rotation_matrix_error"] = max(stats["max_reconstruction_rotation_matrix_error"], re)
                stats[f"{side}_valid_deltas"] += 1
            row.extend([int(valid), *(format(v, ".17g") for v in values)])
        rows.append(row)
    return rows, stats


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_file(target, rows=None, report=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False,
                                         dir=target.parent, prefix=f".{target.name}.", suffix=".staged") as stream:
            stage = Path(stream.name)
            if rows is not None:
                writer = csv.writer(stream)
                writer.writerow(OUTPUT_COLUMNS)
                writer.writerows(rows)
            else:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(stage, 0o644)
        if rows is not None:
            with stage.open(encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream)
                if next(reader) != list(OUTPUT_COLUMNS) or list(reader) != [[str(v) for v in row] for row in rows]:
                    raise ValueError("暂存 CSV 读回校验失败")
        return stage
    except BaseException:
        if stage is not None and stage.exists():
            stage.unlink()
        raise


def main(argv=None):
    staged, committed = [], []
    try:
        args = parse_args(argv)
        calibration, offsets = load_calibration(args.calibration_json.expanduser().resolve() if args.calibration_json else None)
        all_jobs, jobs, skipped, protected = discover_jobs(args)
        print(f"{'执行' if args.execute else '预览'}：发现 {len(all_jobs)} 个，待处理 {len(jobs)} 个，跳过 {len(skipped)} 个")
        label = "自定义最终矩阵" if args.calibration_json else "20260909 test_true 确认坐标定义（内置）"
        print(f"标定：{label}；side-map={args.side_map}；不叠加额外旋转")
        print(f"目标原点：左右分别沿 Hand 局部 X 偏移 {args.target_origin_x_m:g} m（{args.target_origin_x_m * 1000:g} mm），方向与 Hand 相同")
        print("输出：上一帧目标末端自身坐标系中的增量，m / quaternion XYZW")
        if skipped:
            print("提示：跳过已有文件不会更新其左右映射或原点偏移；要重算请另取 --output-name，或确认后使用 --overwrite。")
        if args.verbose:
            for job in jobs:
                print(f"  {job.source} -> {job.target}")
            for job in skipped:
                print(f"  SKIP {job.target}")
        if not args.execute:
            print("未读取并校验 CSV 内容，也未写入数据；确认计划后加 --execute。")
            return 0
        results = []
        for job in jobs:
            try:
                source_hash = sha256(job.source)
                frames, schema = load_hand_frames(job.source, args, offsets)
                rows, stats = relative_rows(frames, args)
                staged.append((stage_file(job.target, rows=rows), job.target))
                results.append({"source": str(job.source), "target": str(job.target),
                                "source_sha256": source_hash, "output_sha256": sha256(staged[-1][0]),
                                "first_timestamp_ns": frames[0].timestamp, "last_timestamp_ns": frames[-1].timestamp,
                                "duration_s": (frames[-1].timestamp-frames[0].timestamp)/1e9,
                                "schema": schema, **stats})
                if args.verbose:
                    print(f"  校验通过 {job.source}: {len(rows)} 行，有效增量 L={stats['left_valid_deltas']}, R={stats['right_valid_deltas']}")
            except (ValueError, OSError, TypeError, KeyError) as exc:
                raise ValueError(f"{job.source}: {exc}") from exc
        if args.report:
            report = {"schema_version": "umi.controller_to_hand.batch_report.v2",
                      "created_at": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256(Path(__file__)),
                      "calibration": calibration, "source_side_map": dict(zip(SIDES, reversed(SIDES) if args.side_map == "swapped" else SIDES)),
                      "delta_definition": ("inverse(T_world_hand_previous) @ T_world_hand_current" if args.target_origin_x_m == 0.
                                           else "inverse(T_world_target_previous) @ T_world_target_current"),
                      "target_pose_definition": "T_world_target = T_world_hand @ Trans(target_origin_x_m, 0, 0)",
                      "target_origin_x_m": args.target_origin_x_m,
                      "target_origin_in_hand_m": [args.target_origin_x_m, 0., 0.],
                      "target_rotation_in_hand_xyzw": [0., 0., 0., 1.],
                      "target_frame": "hand" if args.target_origin_x_m == 0. else "hand_offset",
                      "origin_shift_applied": args.target_origin_x_m != 0.,
                      "quaternion_order": "xyzw", "input_position_unit": args.position_unit, "output_position_unit": "m",
                      "robot_flange_transform_applied": args.target_origin_x_m != 0.,
                      "robot_flange_calibration_verified": False, "require_frame_step": args.require_frame_step,
                      "max_dt_ns": args.max_dt_ns, "processed": results,
                      "skipped": [{"source": str(j.source), "target": str(j.target)} for j in skipped]}
            staged.append((stage_file(args.report, report=report), args.report))
        # Validate all inputs and stage all outputs BEFORE replacing any final file.
        for result in results:
            if sha256(Path(result["source"])) != result["source_sha256"]:
                raise ValueError(f"输入在处理过程中发生变化，取消提交：{result['source']}")
        for _, target in staged:
            check_target(target, protected)
            if target.exists() and not args.overwrite:
                raise FileExistsError(f"处理过程中出现同名输出，取消提交：{target}")
        for temporary, target in staged:
            if args.overwrite:
                os.replace(temporary, target)
            else:
                # Atomic no-clobber publication; fail if another writer won the race.
                os.link(temporary, target)
                temporary.unlink()
            committed.append(str(target))
        print(f"完成：生成 {len(results)} 个 CSV，跳过 {len(skipped)} 个；未修改 controller 原始文件。")
        return 0
    except (ValueError, OSError, TypeError, KeyError, AttributeError, NotImplementedError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        if committed:
            print("提交中断；以下文件已保存（批量提交不是跨文件事务）：\n" + "\n".join(committed), file=sys.stderr)
        return 1
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()


if __name__ == "__main__":
    sys.exit(main())
