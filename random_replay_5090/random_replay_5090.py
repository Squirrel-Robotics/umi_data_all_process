#!/usr/bin/env python3
"""Randomly sample one episode and replay its hand poses on the 5090 MuJoCo host.

This orchestration script is intended to run on the data server.  It discovers
CSV files, chooses one reproducibly, automatically accepts an existing
hand_pose CSV or converts a controller-pose CSV, validates it, transfers a small
replay bundle to the 5090 host, renders an MP4, and optionally starts the viewer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import secrets
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
EXPECTED_COLUMNS = (
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
IDENTITY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="数据集根目录")
    parser.add_argument(
        "--source-glob",
        default="*/camera/e6_rgb_controller_poses.csv",
        help="相对于数据集根目录的 CSV 匹配规则",
    )
    parser.add_argument("--episode", help="指定数据单元名称；不指定则随机选择")
    parser.add_argument("--seed", type=int, help="随机种子；报告会记录实际使用的种子")
    parser.add_argument(
        "--converter",
        type=Path,
        default=PROJECT_DIR / "controller_to_hand_pose.py",
        help="控制器到 hand_pose.csv 的转换脚本",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_DIR / ".replay_cache",
        help="本机临时回放缓存目录",
    )
    parser.add_argument(
        "--remote-host",
        default="dzq@192.168.110.199",
        help="5090 SSH 主机",
    )
    parser.add_argument(
        "--remote-dir",
        default="/home/dzq/umi_x2_mujoco/random_replay",
        help="5090 上的回放目录",
    )
    parser.add_argument(
        "--remote-python",
        default="/mnt/mujoco/.venv/bin/python",
        help="5090 MuJoCo Python",
    )
    parser.add_argument(
        "--remote-assets-root",
        default="/home/dzq/umi_x2_mujoco/assets/generated_revo2",
        help="5090 上 Quanta X2 + Revo2 资产目录",
    )
    parser.add_argument(
        "--remote-quanta-src",
        default="/mnt/mujoco/quanta-x2-mujoco/src",
        help="5090 上 quanta_x2_mujoco Python 源码目录",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=PROJECT_DIR / "replay_videos",
        help="数据服务器上的默认 MP4 输出目录",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        help="自定义 MP4 输出路径（覆盖 --video-dir）",
    )
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-fps", type=float, default=60.0)
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--playback-rate", type=float, default=1.0)
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument(
        "--frame-profile",
        choices=("auto", "identity", "hand-pose-v3"),
        default="auto",
        help=(
            "EEF 局部坐标映射；auto 对 hand_pose_v3.csv 使用左 +90°/"
            "右 -90°，其他 CSV 使用单位映射"
        ),
    )
    parser.add_argument("--max-frames", type=int, help="只回放前 N 帧")
    parser.add_argument(
        "--camera-view",
        choices=("operator", "front"),
        default="operator",
    )
    parser.add_argument("--replace-running", action="store_true", help="关闭此前由本工具启动的回放")
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="只生成并下载 MP4，不启动 5090 桌面窗口",
    )
    parser.add_argument("--prepare-only", action="store_true", help="只抽样、转换和校验，不连接 5090")
    parser.add_argument("--dry-run", action="store_true", help="只显示随机选择，不生成文件")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--max-step-mm", type=float, default=100.0)
    parser.add_argument("--max-rotation-deg", type=float, default=90.0)
    parser.add_argument("--max-dt-ms", type=float, default=200.0)
    parser.add_argument(
        "--min-path-mm",
        type=float,
        default=1.0,
        help="单手累计轨迹低于此值时报告疑似静止/冻结；设为 0 可禁用",
    )
    args = parser.parse_args()

    for name in (
        "playback_rate",
        "translation_scale",
        "rotation_scale",
        "video_fps",
        "max_step_mm",
        "max_rotation_deg",
        "max_dt_ms",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.min_path_mm) or args.min_path_mm < 0.0:
        parser.error("--min-path-mm must be finite and non-negative")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.video_width <= 0 or args.video_width % 2:
        parser.error("--video-width must be a positive even integer")
    if args.video_height <= 0 or args.video_height % 2:
        parser.error("--video-height must be a positive even integer")
    if len(args.video_codec) != 4 or not args.video_codec.isascii():
        parser.error("--video-codec must contain exactly four ASCII characters")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_name(root: Path, source: Path) -> str:
    relative = source.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else source.parent.name


def discover_sources(root: Path, pattern: str) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root}")
    sources = sorted(path.resolve() for path in root.glob(pattern) if path.is_file())
    if not sources:
        raise ValueError(f"no CSV matched {pattern!r} under {root}")
    return sources


def csv_columns(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return tuple(csv.DictReader(stream).fieldnames or ())


def resolve_frame_profile(requested: str, source: Path) -> tuple[str, float, float]:
    profile = (
        "hand-pose-v3"
        if requested == "auto" and source.name == "hand_pose_v3.csv"
        else "identity"
        if requested == "auto"
        else requested
    )
    if profile == "hand-pose-v3":
        return profile, 90.0, -90.0
    return profile, 0.0, 0.0


def stage_existing_hand_pose(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with source.open("rb") as input_stream, tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as output_stream:
            temporary_name = output_stream.name
            for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                output_stream.write(block)
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def select_source(
    root: Path,
    sources: list[Path],
    requested_episode: str | None,
    requested_seed: int | None,
) -> tuple[Path, str, int, int]:
    if requested_episode:
        matches = [path for path in sources if episode_name(root, path) == requested_episode]
        if len(matches) != 1:
            raise ValueError(
                f"episode {requested_episode!r} matched {len(matches)} controller CSV files"
            )
        source = matches[0]
        seed = requested_seed if requested_seed is not None else 0
    else:
        seed = requested_seed if requested_seed is not None else secrets.randbits(64)
        source = random.Random(seed).choice(sources)
    return source, episode_name(root, source), seed, sources.index(source)


def quaternion_angle_deg(values: Sequence[float]) -> float:
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-12:
        raise ValueError("zero-length quaternion")
    w = abs(values[3] / norm)
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, w))))


def validate_hand_pose(
    path: Path,
    max_step_mm: float,
    max_rotation_deg: float,
    max_dt_ms: float,
    min_path_mm: float,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError(
                f"unexpected hand_pose.csv schema in {path}: {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"hand_pose.csv has no data rows: {path}")

    side_metrics: dict[str, dict[str, Any]] = {
        side: {
            "valid_deltas": 0,
            "invalid_deltas": 0,
            "path_length_mm": 0.0,
            "total_rotation_deg": 0.0,
            "max_step_mm": 0.0,
            "max_rotation_deg": 0.0,
        }
        for side in ("left", "right")
    }
    previous_timestamp: int | None = None
    previous_frame: int | None = None
    dt_values_ms: list[float] = []

    for index, row in enumerate(rows):
        line_number = index + 2
        try:
            frame = int(row["frame_number"])
            timestamp = int(row["timestamp_ns"])
            if previous_timestamp is None:
                if row["previous_frame_number"] or row["dt_ns"]:
                    raise ValueError("first row must not have previous-frame metadata")
            else:
                if int(row["previous_frame_number"]) != previous_frame:
                    raise ValueError("previous_frame_number mismatch")
                dt_ns = int(row["dt_ns"])
                if timestamp - previous_timestamp != dt_ns or dt_ns <= 0:
                    raise ValueError("dt_ns mismatch or non-positive interval")
                dt_values_ms.append(dt_ns * 1e-6)

            for side in ("left", "right"):
                valid = int(row[f"{side}_relative_valid"])
                if valid not in (0, 1):
                    raise ValueError(f"{side}_relative_valid must be 0 or 1")
                position = tuple(float(row[f"{side}_local_dp{axis}"]) for axis in "xyz")
                quaternion = tuple(float(row[f"{side}_local_dq{axis}"]) for axis in "xyzw")
                values = (*position, *quaternion)
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"{side} delta contains NaN or Inf")
                quaternion_norm_error = abs(
                    math.sqrt(sum(value * value for value in quaternion)) - 1.0
                )
                if quaternion_norm_error > 1e-10:
                    raise ValueError(f"{side} quaternion is not unit length")
                if not valid:
                    side_metrics[side]["invalid_deltas"] += 1
                    if any(
                        abs(actual - expected) > 1e-12
                        for actual, expected in zip(values, IDENTITY)
                    ):
                        raise ValueError(f"invalid {side} delta is not identity")
                    continue
                step_mm = 1000.0 * math.sqrt(sum(value * value for value in position))
                rotation_deg = quaternion_angle_deg(quaternion)
                metrics = side_metrics[side]
                metrics["valid_deltas"] += 1
                metrics["path_length_mm"] += step_mm
                metrics["total_rotation_deg"] += rotation_deg
                metrics["max_step_mm"] = max(metrics["max_step_mm"], step_mm)
                metrics["max_rotation_deg"] = max(
                    metrics["max_rotation_deg"], rotation_deg
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid hand_pose.csv line {line_number}: {exc}") from exc
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(f"timestamps are not increasing at line {line_number}")
        previous_timestamp = timestamp
        previous_frame = frame

    max_observed_dt_ms = max(dt_values_ms, default=0.0)
    warnings = []
    if max_observed_dt_ms > max_dt_ms:
        warnings.append(
            f"max dt {max_observed_dt_ms:.3f} ms exceeds {max_dt_ms:.3f} ms"
        )
    for side, metrics in side_metrics.items():
        if min_path_mm > 0.0 and metrics["path_length_mm"] < min_path_mm:
            warnings.append(
                f"{side} path length {metrics['path_length_mm']:.3f} mm is below "
                f"{min_path_mm:.3f} mm (possible static/frozen episode)"
            )
        if metrics["max_step_mm"] > max_step_mm:
            warnings.append(
                f"{side} max step {metrics['max_step_mm']:.3f} mm exceeds "
                f"{max_step_mm:.3f} mm"
            )
        if metrics["max_rotation_deg"] > max_rotation_deg:
            warnings.append(
                f"{side} max rotation {metrics['max_rotation_deg']:.3f} deg exceeds "
                f"{max_rotation_deg:.3f} deg"
            )

    duration_s = (
        int(rows[-1]["timestamp_ns"]) - int(rows[0]["timestamp_ns"])
    ) * 1e-9
    return {
        "rows": len(rows),
        "duration_s": duration_s,
        "mean_rate_hz": (len(rows) - 1) / duration_s if duration_s > 0 else 0.0,
        "max_dt_ms": max_observed_dt_ms,
        "sides": side_metrics,
        "warnings": warnings,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def run_checked(command: list[str]) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def remote_command(arguments: list[str]) -> str:
    return shlex.join(arguments)


def deploy_and_replay(
    args: argparse.Namespace,
    hand_pose: Path,
    metadata_path: Path,
    local_video: Path,
) -> dict[str, Any]:
    runner = SCRIPT_DIR / "replay_x2_revo2_relative_eef.py"
    launcher = SCRIPT_DIR / "launch_random_replay_5090.sh"
    for required in (runner, launcher):
        if not required.is_file():
            raise FileNotFoundError(f"missing replay component: {required}")

    socket_path = f"/tmp/umi-replay-ssh-{os.getuid()}-{os.getpid()}.sock"
    ssh_options = [
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=120",
        "-o",
        f"ControlPath={socket_path}",
    ]
    ssh_prefix = ["ssh", *ssh_options]
    scp_prefix = ["scp", *ssh_options]
    remote_dir = args.remote_dir.rstrip("/")
    remote_csv = f"{remote_dir}/hand_pose.csv"
    remote_runner = f"{remote_dir}/{runner.name}"
    remote_launcher = f"{remote_dir}/{launcher.name}"
    remote_summary = f"{remote_dir}/simulation_summary.json"
    remote_trajectory = f"{remote_dir}/trajectory.npz"
    remote_video = f"{remote_dir}/replay.mp4"
    local_video.parent.mkdir(parents=True, exist_ok=True)
    temporary_video = local_video.with_name(
        f".{local_video.stem}.{os.getpid()}.tmp.mp4"
    )
    temporary_video.unlink(missing_ok=True)

    try:
        run_checked(
            [
                *ssh_prefix,
                args.remote_host,
                remote_command(["mkdir", "-p", remote_dir]),
            ]
        )
        run_checked(
            [
                *scp_prefix,
                os.fspath(hand_pose),
                os.fspath(metadata_path),
                os.fspath(runner),
                os.fspath(launcher),
                f"{args.remote_host}:{remote_dir}/",
            ]
        )
        preflight = [
            "env",
            "MUJOCO_GL=egl",
            f"PYTHONPATH={args.remote_quanta_src}",
            args.remote_python,
            remote_runner,
            "--csv",
            remote_csv,
            "--assets-root",
            args.remote_assets_root,
            "--translation-scale",
            str(args.translation_scale),
            "--rotation-scale",
            str(args.rotation_scale),
            "--left-frame-x-deg",
            str(args.left_frame_x_deg),
            "--right-frame-x-deg",
            str(args.right_frame_x_deg),
            "--playback-rate",
            str(args.playback_rate),
            "--summary",
            remote_summary,
            "--trajectory-npz",
            remote_trajectory,
            "--video",
            remote_video,
            "--video-width",
            str(args.video_width),
            "--video-height",
            str(args.video_height),
            "--video-fps",
            str(args.video_fps),
            "--video-codec",
            args.video_codec,
            "--camera-view",
            args.camera_view,
        ]
        if args.max_frames is not None:
            preflight.extend(["--max-frames", str(args.max_frames)])
        run_checked(
            [*ssh_prefix, args.remote_host, remote_command(preflight)]
        )

        run_checked(
            [
                *scp_prefix,
                f"{args.remote_host}:{remote_video}",
                os.fspath(temporary_video),
            ]
        )
        if not temporary_video.is_file() or temporary_video.stat().st_size == 0:
            raise RuntimeError(f"downloaded MP4 is empty: {temporary_video}")
        with temporary_video.open("rb") as stream:
            if b"ftyp" not in stream.read(64):
                raise ValueError(f"downloaded file is not an MP4: {temporary_video}")
        os.replace(temporary_video, local_video)
        video_result = {
            "path": os.fspath(local_video),
            "remote_path": f"{args.remote_host}:{remote_video}",
            "size_bytes": local_video.stat().st_size,
            "sha256": sha256_file(local_video),
        }
        print(f"MP4 video: {local_video}")

        if not args.no_viewer:
            launch = [
                remote_launcher,
                "--csv",
                remote_csv,
                "--assets-root",
                args.remote_assets_root,
                "--python",
                args.remote_python,
                "--quanta-src",
                args.remote_quanta_src,
                "--playback-rate",
                str(args.playback_rate),
                "--translation-scale",
                str(args.translation_scale),
                "--rotation-scale",
                str(args.rotation_scale),
                "--left-frame-x-deg",
                str(args.left_frame_x_deg),
                "--right-frame-x-deg",
                str(args.right_frame_x_deg),
                "--camera-view",
                args.camera_view,
            ]
            if args.max_frames is not None:
                launch.extend(["--max-frames", str(args.max_frames)])
            if args.replace_running:
                launch.append("--replace")
            run_checked(
                [
                    *ssh_prefix,
                    args.remote_host,
                    remote_command(["chmod", "755", remote_launcher]),
                ]
            )
            run_checked(
                [*ssh_prefix, args.remote_host, remote_command(launch)]
            )
        print(f"5090 simulation summary: {args.remote_host}:{remote_summary}")
        if not args.no_viewer:
            print(f"5090 viewer log: {args.remote_host}:{remote_dir}/viewer.log")
        return video_result
    finally:
        temporary_video.unlink(missing_ok=True)
        subprocess.run(
            [*ssh_prefix, "-O", "exit", args.remote_host],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass


def main() -> int:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    sources = discover_sources(root, args.source_glob)
    source, episode, seed, selected_index = select_source(
        root, sources, args.episode, args.seed
    )
    resolved_profile, left_frame_x_deg, right_frame_x_deg = resolve_frame_profile(
        args.frame_profile,
        source,
    )
    args.left_frame_x_deg = left_frame_x_deg
    args.right_frame_x_deg = right_frame_x_deg
    print(
        f"selected episode={episode} index={selected_index}/{len(sources) - 1} "
        f"seed={seed}\nsource={source}\n"
        f"frame_profile={resolved_profile} left_x={left_frame_x_deg:+g}deg "
        f"right_x={right_frame_x_deg:+g}deg initial_pose=sdk_home"
    )
    if args.dry_run:
        return 0

    cache_dir = args.cache_dir.expanduser().resolve() / episode
    cache_dir.mkdir(parents=True, exist_ok=True)
    hand_pose = cache_dir / "hand_pose.csv"
    source_kind = "hand_pose" if csv_columns(source) == EXPECTED_COLUMNS else "controller"
    if source_kind == "hand_pose":
        stage_existing_hand_pose(source, hand_pose)
        print(f"using existing hand-pose CSV: {source}")
    else:
        converter = args.converter.expanduser().resolve()
        if not converter.is_file():
            raise FileNotFoundError(f"converter does not exist: {converter}")
        run_checked(
            [
                sys.executable,
                os.fspath(converter),
                os.fspath(source),
                "--output",
                os.fspath(hand_pose),
                "--overwrite",
                "--execute",
            ]
        )
    qa = validate_hand_pose(
        hand_pose,
        args.max_step_mm,
        args.max_rotation_deg,
        args.max_dt_ms,
        args.min_path_mm,
    )
    metadata = {
        "schema_version": "umi.random-replay-selection.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": os.fspath(root),
        "source_glob": args.source_glob,
        "candidate_count": len(sources),
        "episode": episode,
        "selected_index": selected_index,
        "random_seed": seed,
        "source_kind": source_kind,
        "source_csv": os.fspath(source),
        "source_csv_sha256": sha256_file(source),
        "frame_profile": resolved_profile,
        "frame_redefinition": {
            "left": f"Rx({left_frame_x_deg:+g} deg)",
            "right": f"Rx({right_frame_x_deg:+g} deg)",
        },
        "initial_pose": "Quanta X2 pro + Revo2 sdk_home",
        "hand_pose_csv": os.fspath(hand_pose),
        "hand_pose_csv_sha256": sha256_file(hand_pose),
        "qa": qa,
    }
    video_output = (
        args.video_output.expanduser().resolve()
        if args.video_output is not None
        else args.video_dir.expanduser().resolve() / f"{episode}_seed{seed}.mp4"
    )
    if video_output.suffix.lower() != ".mp4":
        raise ValueError(f"--video-output must use the .mp4 suffix: {video_output}")
    metadata["video_plan"] = {
        "path": os.fspath(video_output),
        "width": args.video_width,
        "height": args.video_height,
        "fps": args.video_fps,
        "codec": args.video_codec,
        "camera_view": args.camera_view,
        "playback_rate": args.playback_rate,
        "frame_profile": resolved_profile,
        "left_frame_x_deg": left_frame_x_deg,
        "right_frame_x_deg": right_frame_x_deg,
        "initial_pose": "sdk_home",
    }
    metadata_path = cache_dir / "selection.json"
    atomic_write_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if qa["warnings"] and args.fail_on_warning:
        raise ValueError("quality warnings present and --fail-on-warning was requested")
    if args.prepare_only:
        print(f"prepared replay bundle: {cache_dir}")
        return 0
    metadata["video"] = deploy_and_replay(
        args,
        hand_pose,
        metadata_path,
        video_output,
    )
    atomic_write_json(metadata_path, metadata)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileExistsError,
        FileNotFoundError,
        PermissionError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
