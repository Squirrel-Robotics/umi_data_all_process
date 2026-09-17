#!/usr/bin/env python3
"""Verify the task_v1 head crop against one real LeRobot frame on CPU.

The training config must crop the LeRobot CHW float32 head frame.  The separate
deployment config must keep the decoded HWC uint8 4:3 head frame intact.  Both
paths then exercise their complete configured OpenPI transforms and emit an
auditable JSON receipt plus a visual PNG preview.
"""

from __future__ import annotations

# These must be set before importing JAX, Torch, LeRobot, or OpenPI.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from openpi import transforms as openpi_transforms
from openpi.policies import cx002_policy
from openpi.training import config as training_config
from openpi.training import data_loader
from openpi_client import image_tools


DEFAULT_CONFIG = "pi05_umi_task_v1_hand_pose_10hz_h50_masked_with_head_roi_v1"
DEFAULT_DEPLOYMENT_CONFIG = (
    "pi05_umi_task_v1_hand_pose_10hz_h50_masked_with_head_roi_v1_deploy_uncropped"
)
EXPECTED_ROI = (0.234375, 0.35, 0.734375, 0.65)
EXPECTED_HEAD_SHAPE_CHW = (3, 480, 640)
EXPECTED_CROP = (150, 168, 470, 312)  # x0, y0, x1, y1; right/bottom exclusive.
EXPECTED_CROP_SHAPE_HWC = (144, 320, 3)
EXPECTED_MODEL_SHAPE_HWC = (224, 224, 3)


def _array(value: Any) -> np.ndarray:
    """Convert a CPU tensor/array to NumPy without changing its values."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _deployment_hwc_uint8(value: Any) -> np.ndarray:
    """Reconstruct the uint8 HWC frame represented by a LeRobot image tensor."""

    image = _array(value)
    if image.shape != EXPECTED_HEAD_SHAPE_CHW:
        raise ValueError(
            "task_v1 verification requires a 3x480x640 CHW frame, "
            f"got {image.shape}"
        )
    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError(f"expected a floating-point LeRobot image, got {image.dtype}")
    if not np.isfinite(image).all() or float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise ValueError("LeRobot image must contain finite values in [0, 1]")

    # LeRobot represents decoded uint8 pixels as float / 255. Rounding recovers
    # the camera/deployment uint8 image and lets the equality assertion detect
    # any lossy or inconsistent conversion inside the shared transform.
    image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(np.transpose(image, (1, 2, 0)))


def _sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(_array(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(value: Any) -> int | float | bool | str | None:
    if isinstance(value, str):
        return value
    array = _array(value)
    if array.size != 1:
        return None
    result = array.reshape(-1)[0].item()
    if isinstance(result, (int, float, bool, str)):
        return result
    return str(result)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "workspace/unversioned"


def _atomic_save_png(image: Image.Image, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def _atomic_save_json(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _make_preview(
    original: np.ndarray,
    cropped: np.ndarray,
    training_model_image: np.ndarray,
    deployment_model_image: np.ndarray,
) -> Image.Image:
    """Build a vertical proof sheet without altering the receipt arrays."""

    x0, y0, x1, y1 = EXPECTED_CROP
    marked = Image.fromarray(original, mode="RGB")
    marked_draw = ImageDraw.Draw(marked)
    marked_draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 64, 64), width=4)

    crop_preview = Image.fromarray(cropped, mode="RGB").resize(
        (640, 288), resample=Image.Resampling.NEAREST
    )
    training_model_preview = Image.fromarray(training_model_image, mode="RGB").resize(
        (448, 448), resample=Image.Resampling.NEAREST
    )
    deployment_model_preview = Image.fromarray(deployment_model_image, mode="RGB").resize(
        (448, 448), resample=Image.Resampling.NEAREST
    )

    margin = 20
    label_height = 24
    gap = 16
    canvas_width = 680
    canvas_height = (
        margin
        + label_height
        + marked.height
        + gap
        + label_height
        + crop_preview.height
        + gap
        + label_height
        + training_model_preview.height
        + gap
        + label_height
        + deployment_model_preview.height
        + margin
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    y = margin

    draw.text(
        (margin, y),
        "Original 640x480; ROI x=[150,470), y=[168,312)",
        fill=(245, 245, 245),
    )
    y += label_height
    canvas.paste(marked, (margin, y))
    y += marked.height + gap

    draw.text((margin, y), "Shared CX002Inputs crop 320x144 (shown 2x)", fill=(245, 245, 245))
    y += label_height
    canvas.paste(crop_preview, (margin, y))
    y += crop_preview.height + gap

    draw.text((margin, y), "Training model input: cropped 224x224 letterbox (shown 2x)", fill=(245, 245, 245))
    y += label_height
    model_x = (canvas_width - training_model_preview.width) // 2
    canvas.paste(training_model_preview, (model_x, y))
    draw.rectangle(
        (
            model_x - 1,
            y - 1,
            model_x + training_model_preview.width,
            y + training_model_preview.height,
        ),
        outline=(225, 225, 225),
        width=1,
    )
    y += training_model_preview.height + gap

    draw.text(
        (margin, y),
        "Deployment model input: uncropped 224x224 letterbox (shown 2x)",
        fill=(245, 245, 245),
    )
    y += label_height
    canvas.paste(deployment_model_preview, (model_x, y))
    draw.rectangle(
        (
            model_x - 1,
            y - 1,
            model_x + deployment_model_preview.width,
            y + deployment_model_preview.height,
        ),
        outline=(225, 225, 225),
        width=1,
    )
    return canvas


def _build_deployment_input(repacked: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source_images = repacked["images"]
    canonical = {
        key: _deployment_hwc_uint8(source_images[key])
        for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
    }
    deployment = {
        "images": canonical,
        "state": np.array(_array(repacked["state"]), copy=True),
        "actions": np.array(_array(repacked["actions"]), copy=True),
        "action_is_pad": np.array(_array(repacked["action_is_pad"]), dtype=bool, copy=True),
    }
    if "prompt" in repacked:
        deployment["prompt"] = repacked["prompt"]
    return deployment, canonical


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--deployment-config-name", default=DEFAULT_DEPLOYMENT_CONFIG)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports" / "head_crop_verification",
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.frame_index < 0:
        raise ValueError("--frame-index must be non-negative")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)

    training = training_config.get_config(args.config_name)
    training_data_config = training.data.create(training.assets_dirs, training.model)
    training_shared_candidates = [
        transform
        for transform in training_data_config.data_transforms.inputs
        if isinstance(transform, cx002_policy.CX002Inputs)
    ]
    if len(training_shared_candidates) != 1:
        raise RuntimeError(
            "training ROI config must expose exactly one shared CX002Inputs transform; "
            f"found {len(training_shared_candidates)}"
        )
    training_shared_inputs = training_shared_candidates[0]
    configured_roi = tuple(
        float(value) for value in training_shared_inputs.head_crop_normalized or ()
    )
    if configured_roi != EXPECTED_ROI:
        raise RuntimeError(
            "training config ROI "
            f"{configured_roi!r} does not match expected task_v1 ROI {EXPECTED_ROI!r}"
        )
    if tuple(cx002_policy.TASK_V1_HEAD_CROP_NORMALIZED) != EXPECTED_ROI:
        raise RuntimeError("CX002 task_v1 ROI constant differs from the verification contract")

    deployment = training_config.get_config(args.deployment_config_name)
    deployment_data_config = deployment.data.create(
        deployment.assets_dirs,
        deployment.model,
    )
    deployment_shared_candidates = [
        transform
        for transform in deployment_data_config.data_transforms.inputs
        if isinstance(transform, cx002_policy.CX002Inputs)
    ]
    if len(deployment_shared_candidates) != 1:
        raise RuntimeError(
            "deployment config must expose exactly one shared CX002Inputs transform; "
            f"found {len(deployment_shared_candidates)}"
        )
    deployment_shared_inputs = deployment_shared_candidates[0]
    if deployment_shared_inputs.head_crop_normalized is not None:
        raise RuntimeError(
            "deployment config must keep the head image uncropped, got "
            f"{deployment_shared_inputs.head_crop_normalized!r}"
        )
    if training_data_config.repo_id != deployment_data_config.repo_id:
        raise RuntimeError(
            "training and deployment configs refer to different datasets: "
            f"{training_data_config.repo_id!r} vs {deployment_data_config.repo_id!r}"
        )
    if training.model.action_horizon != deployment.model.action_horizon:
        raise RuntimeError("training and deployment action horizons differ")

    raw_dataset = data_loader.create_torch_dataset(
        training_data_config,
        training.model.action_horizon,
        training.model,
    )
    if args.frame_index >= len(raw_dataset):
        raise IndexError(f"frame index {args.frame_index} is outside dataset length {len(raw_dataset)}")
    raw_item = raw_dataset[args.frame_index]

    # Repack exactly as training does, then keep its CHW float images untouched
    # for the first invocation of the shared policy transform.
    repacked = copy.deepcopy(raw_item)
    for transform in training_data_config.repack_transforms.inputs:
        repacked = transform(repacked)
    training_output = training_shared_inputs(copy.deepcopy(repacked))

    # Reconstruct decoded HWC uint8 camera frames. The training transform must
    # crop this representation identically to the real LeRobot CHW float path.
    deployment_input, canonical_images = _build_deployment_input(repacked)
    training_hwc_output = training_shared_inputs(copy.deepcopy(deployment_input))

    head_training = _array(training_output["image"]["base_0_rgb"])
    head_training_hwc = _array(training_hwc_output["image"]["base_0_rgb"])
    x0, y0, x1, y1 = EXPECTED_CROP
    expected_crop = canonical_images["cam_high"][y0:y1, x0:x1]
    if head_training.shape != EXPECTED_CROP_SHAPE_HWC:
        raise AssertionError(f"training crop has shape {head_training.shape}")
    if head_training_hwc.shape != EXPECTED_CROP_SHAPE_HWC:
        raise AssertionError(f"training HWC crop has shape {head_training_hwc.shape}")
    np.testing.assert_array_equal(head_training, head_training_hwc)
    np.testing.assert_array_equal(head_training, expected_crop)

    # Exercise every deployment data transform on the real HWC image payload.
    # Its CX002Inputs is configured with crop=None, so the original 4:3 head
    # frame must survive this stage pixel-for-pixel.
    deployment_output = copy.deepcopy(deployment_input)
    for transform in deployment_data_config.data_transforms.inputs:
        deployment_output = transform(deployment_output)
    head_deployment = _array(deployment_output["image"]["base_0_rgb"])
    if head_deployment.shape != (480, 640, 3):
        raise AssertionError(f"uncropped deployment head has shape {head_deployment.shape}")
    np.testing.assert_array_equal(head_deployment, canonical_images["cam_high"])

    wrist_mapping = {
        "left_wrist_0_rgb": "cam_left_wrist",
        "right_wrist_0_rgb": "cam_right_wrist",
    }
    for output_key, source_key in wrist_mapping.items():
        expected_wrist = canonical_images[source_key]
        if expected_wrist.shape != (480, 640, 3):
            raise AssertionError(f"{source_key} has shape {expected_wrist.shape}")
        np.testing.assert_array_equal(training_output["image"][output_key], expected_wrist)
        np.testing.assert_array_equal(training_hwc_output["image"][output_key], expected_wrist)
        np.testing.assert_array_equal(deployment_output["image"][output_key], expected_wrist)

    original_mask = np.asarray(_array(repacked["action_is_pad"]), dtype=bool)
    if original_mask.shape != (training.model.action_horizon,):
        raise AssertionError(f"action_is_pad has shape {original_mask.shape}")
    np.testing.assert_array_equal(training_output["action_is_pad"], original_mask)
    np.testing.assert_array_equal(training_hwc_output["action_is_pad"], original_mask)
    np.testing.assert_array_equal(deployment_output["action_is_pad"], original_mask)
    np.testing.assert_array_equal(training_output["actions"], _array(repacked["actions"]))
    np.testing.assert_array_equal(training_hwc_output["actions"], _array(repacked["actions"]))
    np.testing.assert_array_equal(deployment_output["actions"], _array(repacked["actions"]))

    # Exercise the actual end-to-end training dataset pipeline. This repeats
    # repack + cropped CX002Inputs and then runs normalization and every model
    # transform, including ResizeImages(224, 224).
    complete_training_dataset = data_loader.transform_dataset(
        raw_dataset,
        training_data_config,
    )
    training_model_ready = complete_training_dataset[args.frame_index]
    training_model_images = {
        key: _array(value) for key, value in training_model_ready["image"].items()
    }
    expected_model_keys = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    if set(training_model_images) != expected_model_keys:
        raise AssertionError(
            f"unexpected training model image keys: {sorted(training_model_images)}"
        )
    for key, image in training_model_images.items():
        if image.shape != EXPECTED_MODEL_SHAPE_HWC:
            raise AssertionError(f"training model image {key} has shape {image.shape}")
    np.testing.assert_array_equal(training_model_ready["action_is_pad"], original_mask)

    expected_training_letterbox = image_tools.resize_with_pad(head_training, 224, 224)
    np.testing.assert_array_equal(
        training_model_images["base_0_rgb"],
        expected_training_letterbox,
    )
    if not (
        np.all(training_model_images["base_0_rgb"][:62] == 0)
        and np.all(training_model_images["base_0_rgb"][162:] == 0)
    ):
        raise AssertionError(
            "training 224x224 head does not contain the expected 62-row letterbox padding"
        )

    # Continue from the proven uncropped HWC deployment data output through
    # normalization and the deployment config's complete model transforms.
    deployment_model_ready = openpi_transforms.Normalize(
        deployment_data_config.norm_stats,
        use_quantiles=deployment_data_config.use_quantile_norm,
    )(copy.deepcopy(deployment_output))
    for transform in deployment_data_config.model_transforms.inputs:
        deployment_model_ready = transform(deployment_model_ready)
    deployment_model_images = {
        key: _array(value) for key, value in deployment_model_ready["image"].items()
    }
    if set(deployment_model_images) != expected_model_keys:
        raise AssertionError(
            f"unexpected deployment model image keys: {sorted(deployment_model_images)}"
        )
    for key, image in deployment_model_images.items():
        if image.shape != EXPECTED_MODEL_SHAPE_HWC:
            raise AssertionError(f"deployment model image {key} has shape {image.shape}")
    np.testing.assert_array_equal(deployment_model_ready["action_is_pad"], original_mask)

    expected_deployment_letterbox = image_tools.resize_with_pad(
        canonical_images["cam_high"],
        224,
        224,
    )
    np.testing.assert_array_equal(
        deployment_model_images["base_0_rgb"],
        expected_deployment_letterbox,
    )
    if not (
        np.all(deployment_model_images["base_0_rgb"][:28] == 0)
        and np.all(deployment_model_images["base_0_rgb"][196:] == 0)
    ):
        raise AssertionError(
            "deployment 224x224 head does not contain the expected 28-row letterbox padding"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"task_v1_head_crop_frame_{args.frame_index:06d}"
    preview_path = output_dir / f"{stem}_preview.png"
    receipt_path = output_dir / f"{stem}_receipt.json"
    preview = _make_preview(
        canonical_images["cam_high"],
        head_training,
        training_model_images["base_0_rgb"],
        deployment_model_images["base_0_rgb"],
    )
    _atomic_save_png(preview, preview_path)

    source_fields = {}
    for key in (
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "source.e6_source_row_index",
        "source.e6_frame_id",
        "task",
        "prompt",
    ):
        if key in raw_item:
            source_fields[key] = _scalar(raw_item[key])

    receipt: dict[str, Any] = {
        "schema": "umi-task-v1-head-crop-verification",
        "schema_version": 2,
        "status": "complete",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "device": "cpu",
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "jax_platforms": os.environ["JAX_PLATFORMS"],
            "cpu_threads": args.cpu_threads,
        },
        "config": {
            "training_name": args.config_name,
            "deployment_name": args.deployment_config_name,
            "dataset": str(training_data_config.repo_id),
            "dataset_rows": len(raw_dataset),
            "shared_input_transform": type(training_shared_inputs).__name__,
            "training_model_input_transforms": [
                type(transform).__name__
                for transform in training_data_config.model_transforms.inputs
            ],
            "deployment_model_input_transforms": [
                type(transform).__name__
                for transform in deployment_data_config.model_transforms.inputs
            ],
        },
        "sample": {
            "requested_global_index": args.frame_index,
            "source_fields": source_fields,
        },
        "training_cropped": {
            "head_crop_normalized_left_top_right_bottom": list(EXPECTED_ROI),
            "pixel_xyxy_exclusive": list(EXPECTED_CROP),
            "input_hwc_shape": list(canonical_images["cam_high"].shape),
            "crop_hwc_shape": list(head_training.shape),
            "training_chw_crop_sha256": _sha256_array(head_training),
            "training_hwc_crop_sha256": _sha256_array(head_training_hwc),
            "chw_hwc_crop_pixel_equal": True,
            "model_image_shapes": {
                key: list(value.shape) for key, value in training_model_images.items()
            },
            "head_letterbox_sha256": _sha256_array(
                training_model_images["base_0_rgb"]
            ),
            "letterbox_content_xyxy_exclusive": [0, 62, 224, 162],
            "letterbox_padding_rows_top_bottom": [62, 62],
            "matches_openpi_resize_with_pad": True,
        },
        "deployment_uncropped": {
            "head_crop_normalized": None,
            "input_hwc_shape": list(canonical_images["cam_high"].shape),
            "data_transform_output_hwc_shape": list(head_deployment.shape),
            "input_head_sha256": _sha256_array(canonical_images["cam_high"]),
            "data_transform_output_head_sha256": _sha256_array(head_deployment),
            "full_4_by_3_head_pixel_equal": True,
            "model_image_shapes": {
                key: list(value.shape) for key, value in deployment_model_images.items()
            },
            "head_letterbox_sha256": _sha256_array(
                deployment_model_images["base_0_rgb"]
            ),
            "letterbox_content_xyxy_exclusive": [0, 28, 224, 196],
            "letterbox_padding_rows_top_bottom": [28, 28],
            "matches_full_640x480_openpi_resize_with_pad": True,
        },
        "preservation": {
            "left_wrist_hwc_shape": list(canonical_images["cam_left_wrist"].shape),
            "right_wrist_hwc_shape": list(canonical_images["cam_right_wrist"].shape),
            "left_wrist_unchanged": True,
            "right_wrist_unchanged": True,
            "action_is_pad_shape": list(original_mask.shape),
            "action_is_pad_true_count": int(original_mask.sum()),
            "action_is_pad_unchanged": True,
            "actions_unchanged_by_cx002_inputs": True,
        },
        "artifacts": {
            "preview_png": str(preview_path),
            "preview_png_sha256": _sha256_file(preview_path),
            "receipt_json": str(receipt_path),
        },
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": _package_version("Pillow"),
            "torch": torch.__version__,
            "jax": _package_version("jax"),
            "lerobot": _package_version("lerobot"),
            "openpi": _package_version("openpi-client"),
        },
    }
    _atomic_save_json(receipt, receipt_path)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
