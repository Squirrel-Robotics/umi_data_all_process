#!/usr/bin/env python3
"""Replay hand_pose_v3 local-frame EEF deltas on Quanta X2 + Revo2.

The CSV deltas use ``D = inverse(T_hand_previous) @ T_hand_current``.  Because
they are expressed in each hand's own previous frame, targets are accumulated
by right multiplication.  A fixed local frame redefinition ``X`` is handled
exactly by conjugation: ``T_target = T_target @ inverse(X) @ D @ X``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


QUANTA_SRC = Path(
    os.environ.get(
        "QUANTA_X2_MUJOCO_SRC",
        "/mnt/mujoco/quanta-x2-mujoco/src",
    )
)
if str(QUANTA_SRC) not in sys.path:
    sys.path.insert(0, str(QUANTA_SRC))

from quanta_x2_mujoco import QuantaX2Sim, SimulationConfig  # noqa: E402
from quanta_x2_mujoco.umi import (  # noqa: E402
    matrix_to_pose7_xyzw,
    pose7_xyzw_to_matrix,
)


META_COLUMNS = (
    "frame_number",
    "previous_frame_number",
    "timestamp_ns",
    "dt_ns",
)
SIDE_COLUMNS = tuple(
    name
    for side in ("left", "right")
    for name in (
        f"{side}_relative_valid",
        *(f"{side}_local_dp{axis}" for axis in "xyz"),
        *(f"{side}_local_dq{axis}" for axis in "xyzw"),
    )
)
EXPECTED_COLUMNS = (*META_COLUMNS, *SIDE_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("/home/dzq/umi_x2_mujoco/assets/generated_revo2"),
    )
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument(
        "--left-frame-x-deg",
        type=float,
        default=0.0,
        help="left EEF local-X frame redefinition in degrees",
    )
    parser.add_argument(
        "--right-frame-x-deg",
        type=float,
        default=0.0,
        help="right EEF local-X frame redefinition in degrees",
    )
    parser.add_argument("--playback-rate", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--camera-view",
        choices=("operator", "front"),
        default="operator",
        help="operator uses the model camera; front faces the robot along +X",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="allow viewing/saving the prefix before an IK failure",
    )
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--trajectory-npz", type=Path)
    parser.add_argument("--video", type=Path, help="write an offscreen MP4 replay")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-fps", type=float, default=60.0)
    parser.add_argument(
        "--video-codec",
        default="mp4v",
        help="four-character OpenCV codec (default: mp4v)",
    )
    parser.add_argument("--initial-left-q", type=float, nargs=7)
    parser.add_argument("--initial-right-q", type=float, nargs=7)
    args = parser.parse_args()
    for name in (
        "translation_scale",
        "rotation_scale",
        "playback_rate",
        "video_fps",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("left_frame_x_deg", "right_frame_x_deg"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.video_width <= 0 or args.video_width % 2:
        parser.error("--video-width must be a positive even integer")
    if args.video_height <= 0 or args.video_height % 2:
        parser.error("--video-height must be a positive even integer")
    if len(args.video_codec) != 4 or not args.video_codec.isascii():
        parser.error("--video-codec must contain exactly four ASCII characters")
    if (args.initial_left_q is None) != (args.initial_right_q is None):
        parser.error("--initial-left-q and --initial-right-q must be used together")
    return args


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError(
                "CSV schema mismatch:\n"
                f"expected={list(EXPECTED_COLUMNS)}\nactual={reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no data rows")

    previous_timestamp = -1
    for index, row in enumerate(rows):
        try:
            timestamp = int(row["timestamp_ns"])
            int(row["frame_number"])
            if index > 0:
                int(row["previous_frame_number"])
                dt_ns = int(row["dt_ns"])
                if dt_ns <= 0:
                    raise ValueError("dt_ns must be positive")
            for side in ("left", "right"):
                valid = int(row[f"{side}_relative_valid"])
                if valid not in (0, 1):
                    raise ValueError(f"{side}_relative_valid must be 0 or 1")
                values = np.asarray(
                    [
                        *(float(row[f"{side}_local_dp{axis}"]) for axis in "xyz"),
                        *(float(row[f"{side}_local_dq{axis}"]) for axis in "xyzw"),
                    ],
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"{side} delta contains NaN or Inf")
                if np.linalg.norm(values[3:]) < 1e-12:
                    raise ValueError(f"{side} delta quaternion is degenerate")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid CSV row {index + 2}: {exc}") from exc
        if timestamp <= previous_timestamp:
            raise ValueError(f"timestamps are not increasing at CSV row {index + 2}")
        previous_timestamp = timestamp
    return rows


def _delta_transform(
    row: dict[str, str],
    side: str,
    translation_scale: float,
    rotation_scale: float,
    frame_redefinition: np.ndarray,
) -> np.ndarray:
    translation = translation_scale * np.asarray(
        [float(row[f"{side}_local_dp{axis}"]) for axis in "xyz"],
        dtype=np.float64,
    )
    quaternion = np.asarray(
        [float(row[f"{side}_local_dq{axis}"]) for axis in "xyzw"],
        dtype=np.float64,
    )
    quaternion /= np.linalg.norm(quaternion)
    rotation = Rotation.from_quat(quaternion)
    if rotation_scale != 1.0:
        rotation = Rotation.from_rotvec(rotation.as_rotvec() * rotation_scale)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.as_matrix()
    result[:3, 3] = translation
    # If T_new = T_old @ X, then the same per-frame motion expressed in the
    # new local frame is D_new = inverse(X) @ D_old @ X.
    return frame_redefinition.T @ result @ frame_redefinition


def _local_x_frame_redefinition(angle_degrees: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler(
        "x", angle_degrees, degrees=True
    ).as_matrix()
    return transform


def _snap_physical(sim: QuantaX2Sim, physical: np.ndarray) -> None:
    sim._target_qpos[:] = physical
    sim.data.qpos[sim._qpos_adr] = physical
    sim.data.qvel[:] = 0.0
    sim._project_mimics()
    mujoco.mj_forward(sim.model, sim.data)


def _quat_error(a: np.ndarray, b: np.ndarray) -> float:
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    qa /= np.linalg.norm(qa)
    qb /= np.linalg.norm(qb)
    return float(2.0 * math.acos(float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))))


def build_trajectory(
    csv_path: Path,
    assets_root: Path,
    translation_scale: float,
    rotation_scale: float,
    left_frame_x_deg: float,
    right_frame_x_deg: float,
    max_frames: int | None,
    viewer_backend: bool,
    initial_left_q: list[float] | None = None,
    initial_right_q: list[float] | None = None,
) -> tuple[QuantaX2Sim, list[np.ndarray], list[float], dict[str, Any]]:
    rows = _load_rows(csv_path)
    if max_frames is not None:
        rows = rows[:max_frames]
    sim = QuantaX2Sim(
        SimulationConfig(
            variant="pro",
            end_effector="revo2",
            assets_root=assets_root,
            render_engine="classic",
            render_backend="glfw" if viewer_backend else "egl",
            enable_rendering=False,
        )
    )
    home = sim.reset(seed=0, keyframe="sdk_home")
    if initial_left_q is not None and initial_right_q is not None:
        sim.data.qpos[8:15] = np.asarray(initial_left_q, dtype=np.float64)
        sim.data.qpos[26:33] = np.asarray(initial_right_q, dtype=np.float64)
        sim.data.qvel[:] = 0.0
        sim._target_qpos[:] = sim.data.qpos[sim._qpos_adr]
        sim._project_mimics()
        mujoco.mj_forward(sim.model, sim.data)
        home = sim.observe()
    home_q32 = np.asarray(home.joint_position, dtype=np.float64)
    home_physical = sim._canonical_to_physical(home_q32)
    sim._velocity_limits[:] = np.inf
    sim._cartesian_waist_step_limit_rad = 0.0
    sim._cartesian_waist_reference_qpos = home_physical[26:30].copy()

    targets = {
        "left": pose7_xyzw_to_matrix(home.left_end_pose),
        "right": pose7_xyzw_to_matrix(home.right_end_pose),
    }
    frame_redefinitions = {
        "left": _local_x_frame_redefinition(left_frame_x_deg),
        "right": _local_x_frame_redefinition(right_frame_x_deg),
    }
    frames: list[np.ndarray] = []
    frame_delays: list[float] = []
    frame_numbers: list[int] = []
    max_position_error = 0.0
    max_rotation_error = 0.0
    max_contacts = int(sim.data.ncon)
    first_failure: dict[str, Any] | None = None

    for index, row in enumerate(rows):
        candidates = {side: targets[side].copy() for side in ("left", "right")}
        for side in ("left", "right"):
            if int(row[f"{side}_relative_valid"]):
                candidates[side] = targets[side] @ _delta_transform(
                    row,
                    side,
                    translation_scale,
                    rotation_scale,
                    frame_redefinitions[side],
                )

        # Row zero intentionally holds sdk_home because it has no predecessor.
        if index > 0 or any(int(row[f"{side}_relative_valid"]) for side in ("left", "right")):
            left_pose = matrix_to_pose7_xyzw(candidates["left"])
            right_pose = matrix_to_pose7_xyzw(candidates["right"])
            cartesian23 = np.concatenate(
                [left_pose, right_pose, np.asarray(home.waist_pose), home_q32[30:32]]
            )
            physical, diagnostics, _ = sim._ik_joint_target(cartesian23)
            if physical is None or diagnostics.get("ik") != "converged":
                first_failure = {
                    "csv_row": index + 2,
                    "trajectory_index": index,
                    "frame_number": int(row["frame_number"]),
                    "diagnostics": diagnostics,
                }
                break

            canonical = sim._physical_to_canonical(physical)
            canonical[7:13] = home_q32[7:13]
            canonical[20:26] = home_q32[20:26]
            canonical[26:32] = home_q32[26:32]
            _snap_physical(sim, sim._canonical_to_physical(canonical))
            observation = sim.observe()
            max_position_error = max(
                max_position_error,
                float(np.linalg.norm(left_pose[:3] - observation.left_end_pose[:3])),
                float(np.linalg.norm(right_pose[:3] - observation.right_end_pose[:3])),
            )
            max_rotation_error = max(
                max_rotation_error,
                _quat_error(left_pose[3:], observation.left_end_pose[3:]),
                _quat_error(right_pose[3:], observation.right_end_pose[3:]),
            )
            targets = candidates

        frames.append(sim.data.qpos.copy())
        frame_numbers.append(int(row["frame_number"]))
        frame_delays.append(
            int(row["dt_ns"]) * 1e-9 if row["dt_ns"] else 1.0 / 30.0
        )
        max_contacts = max(max_contacts, int(sim.data.ncon))

    summary: dict[str, Any] = {
        "csv": str(csv_path.resolve()),
        "assets_root": str(assets_root.resolve()),
        "input_rows": len(rows),
        "solved_frames": len(frames),
        "complete": first_failure is None and len(frames) == len(rows),
        "first_failure": first_failure,
        "translation_scale": translation_scale,
        "rotation_scale": rotation_scale,
        "delta_frame": "redefined previous robot hand EEF (body/local)",
        "frame_redefinition": {
            "left": f"Rx({left_frame_x_deg:+.6g} deg)",
            "right": f"Rx({right_frame_x_deg:+.6g} deg)",
        },
        "delta_composition": (
            "T_target[i] = T_target[i-1] @ inverse(X_side) "
            "@ D_local[i] @ X_side"
        ),
        "initial_pose": (
            "Quanta X2 pro + Revo2 supplied arm qpos"
            if initial_left_q is not None
            else "Quanta X2 pro + Revo2 sdk_home"
        ),
        "revo2_hand_pose": "held at sdk_home",
        "waist_and_head": "held at sdk_home",
        "frame_numbers": frame_numbers,
        "max_fk_position_error_m": max_position_error,
        "max_fk_rotation_error_rad": max_rotation_error,
        "max_contact_count": max_contacts,
    }
    return sim, frames, frame_delays, summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _view(
    sim: QuantaX2Sim,
    frames: list[np.ndarray],
    frame_delays: list[float],
    playback_rate: float,
    loop: bool,
    camera_view: str,
) -> None:
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    viewer.opt.sitegroup[:] = True
    sim.model.vis.scale.framelength = 0.12
    sim.model.vis.scale.framewidth = 0.008
    if camera_view == "front":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (0.28, 0.0, 0.82)
        viewer.cam.distance = 2.25
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -8.0
    else:
        operator_camera = mujoco.mj_name2id(
            sim.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "operator",
        )
        if operator_camera >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = operator_camera
    try:
        while viewer.is_running():
            for qpos, delay in zip(frames, frame_delays, strict=True):
                if not viewer.is_running():
                    break
                started = time.monotonic()
                sim.data.qpos[:] = qpos
                sim.data.qvel[:] = 0.0
                mujoco.mj_forward(sim.model, sim.data)
                viewer.sync()
                remaining = delay / playback_rate - (time.monotonic() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
            if not loop:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.03)
                break
    except KeyboardInterrupt:
        pass
    # Avoid a known GLFW teardown crash on this host.
    viewer = None
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _offscreen_camera(model: mujoco.MjModel, camera_view: str) -> Any:
    if camera_view == "operator":
        operator_camera = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "operator",
        )
        if operator_camera >= 0:
            return "operator"

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.28, 0.0, 0.82)
    camera.distance = 2.25
    camera.azimuth = 180.0
    camera.elevation = -8.0
    return camera


def _video_source_indices(
    frame_delays: list[float],
    playback_rate: float,
    fps: float,
) -> tuple[np.ndarray, float]:
    """Resample variable-rate source frames onto a fixed-rate video timeline."""
    delays = np.asarray(frame_delays, dtype=np.float64)
    source_times = np.zeros(len(delays), dtype=np.float64)
    if len(delays) > 1:
        source_times[1:] = np.cumsum(np.maximum(delays[1:], 0.0))
        positive_delays = delays[1:][delays[1:] > 0.0]
    else:
        positive_delays = delays[delays > 0.0]
    tail_duration = (
        float(np.median(positive_delays)) if len(positive_delays) else 1.0 / fps
    )
    duration_s = (float(source_times[-1]) + tail_duration) / playback_rate
    output_frames = max(1, int(math.floor(duration_s * fps + 0.5)))
    sample_times = np.arange(output_frames, dtype=np.float64) / fps * playback_rate
    indices = np.searchsorted(source_times, sample_times, side="right") - 1
    return np.clip(indices, 0, len(delays) - 1), duration_s


def _render_video(
    sim: QuantaX2Sim,
    frames: list[np.ndarray],
    frame_delays: list[float],
    playback_rate: float,
    camera_view: str,
    output: Path,
    width: int,
    height: int,
    fps: float,
    codec: str,
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for MP4 encoding; install opencv-python"
        ) from exc
    if not frames:
        raise RuntimeError("no solved frames to render")
    if output.suffix.lower() != ".mp4":
        raise ValueError(f"video output must use the .mp4 suffix: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        os.fspath(temporary),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"OpenCV could not open MP4 encoder codec={codec!r}; "
            "try --video-codec mp4v"
        )

    sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, width)
    sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, height)
    renderer: mujoco.Renderer | None = None
    indices, duration_s = _video_source_indices(frame_delays, playback_rate, fps)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.frame = mujoco.mjtFrame.mjFRAME_SITE
    option.sitegroup[:] = True
    sim.model.vis.scale.framelength = 0.12
    sim.model.vis.scale.framewidth = 0.008
    camera = _offscreen_camera(sim.model, camera_view)
    try:
        renderer = mujoco.Renderer(sim.model, height=height, width=width)
        for output_index, source_index in enumerate(indices):
            sim.data.qpos[:] = frames[int(source_index)]
            sim.data.qvel[:] = 0.0
            mujoco.mj_forward(sim.model, sim.data)
            renderer.update_scene(sim.data, camera=camera, scene_option=option)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr,
                (
                    f"frame {int(source_index) + 1}/{len(frames)}  "
                    f"time {output_index / fps:.2f}s  rate {playback_rate:g}x"
                ),
                (28, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(bgr)
    except BaseException:
        writer.release()
        if renderer is not None:
            renderer.close()
        temporary.unlink(missing_ok=True)
        raise
    writer.release()
    if renderer is not None:
        renderer.close()
    if not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"MP4 encoder produced an empty file: {temporary}")
    os.replace(temporary, output)
    return {
        "path": os.fspath(output),
        "codec": codec,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": int(len(indices)),
        "duration_s": duration_s,
        "size_bytes": output.stat().st_size,
        "camera_view": camera_view,
    }


def main() -> int:
    args = parse_args()
    sim, frames, frame_delays, summary = build_trajectory(
        args.csv.expanduser().resolve(),
        args.assets_root.expanduser().resolve(),
        args.translation_scale,
        args.rotation_scale,
        args.left_frame_x_deg,
        args.right_frame_x_deg,
        args.max_frames,
        args.view,
        args.initial_left_q,
        args.initial_right_q,
    )
    if args.trajectory_npz:
        output = args.trajectory_npz.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            qpos=np.asarray(frames),
            frame_delay_s=np.asarray(frame_delays),
            frame_number=np.asarray(summary["frame_numbers"], dtype=np.int64),
        )
    if not summary["complete"] and not args.allow_partial:
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        if args.summary:
            _write_json(args.summary.expanduser().resolve(), summary)
        sim.close()
        return 3
    if args.video:
        summary["video"] = _render_video(
            sim,
            frames,
            frame_delays,
            args.playback_rate,
            args.camera_view,
            args.video.expanduser().resolve(),
            args.video_width,
            args.video_height,
            args.video_fps,
            args.video_codec,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.summary:
        _write_json(args.summary.expanduser().resolve(), summary)
    if args.view:
        if not frames:
            sim.close()
            raise RuntimeError("no solved frames to view")
        _view(
            sim,
            frames,
            frame_delays,
            args.playback_rate,
            args.loop,
            args.camera_view,
        )
    sim.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
