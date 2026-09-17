#!/usr/bin/env python3
"""Generate exact OpenPI norm stats for a finalized prechunked UMI dataset.

All observation states are included.  Actions are flattened over the horizon
only after dropping slots for which ``action_is_pad`` is true.  The result is
published atomically as one OpenPI asset directory and includes a hash audit.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pyarrow.parquet as pq


OPENPI_SRC = Path(os.environ.get("OPENPI_SRC", "/home/dzq/openpi/src"))
if OPENPI_SRC.is_dir():
    sys.path.insert(0, str(OPENPI_SRC))

import openpi.shared.normalize as normalize  # noqa: E402


CONTRACT_SCHEMA = "umi-folder-dual-hand-pose-lerobot"
VIDEO_KEYS = {
    "observation.images.head_rgb",
    "observation.images.left_wrist_rgb",
    "observation.images.right_wrist_rgb",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def statistics(value: np.ndarray) -> normalize.NormStats:
    if value.ndim != 2 or not len(value):
        raise ValueError(f"expected a non-empty 2-D array, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("normalization input contains NaN or Inf")
    return normalize.NormStats(
        mean=np.mean(value, axis=0, dtype=np.float64),
        std=np.std(value, axis=0, dtype=np.float64),
        q01=np.quantile(value, 0.01, axis=0, method="linear"),
        q99=np.quantile(value, 0.99, axis=0, method="linear"),
    )


def validate_contract(
    dataset: Path,
    contract: dict[str, Any],
    info: dict[str, Any],
    *,
    expected_fps: int | None,
    expected_horizon: int | None,
    expected_task: str | None,
) -> tuple[int, int, int, int, int]:
    if ".building-" in dataset.name:
        raise ValueError(f"refusing a conversion staging directory: {dataset}")
    staging = sorted(dataset.parent.glob(f".{dataset.name}.building-*"))
    if staging:
        raise ValueError(f"conversion staging still exists: {staging}")
    if not (
        contract.get("schema") == CONTRACT_SCHEMA
        and contract.get("schema_version") == 1
        and contract.get("status") == "complete"
        and contract.get("target") == str(dataset)
        and contract.get("one_source_one_output_episode") is True
        and contract.get("source_snapshot_verified_before_publish") is True
    ):
        raise ValueError("dataset is not a finalized atomic UMI conversion")
    if contract.get("source_episode_ids_sha256") != contract.get("output_episode_ids_sha256"):
        raise ValueError("source/output episode ID hashes disagree")

    fps = int(contract.get("fps", -1))
    state_shape = contract.get("state_shape")
    action_shape = contract.get("action_shape")
    if not (
        isinstance(state_shape, list)
        and len(state_shape) == 1
        and isinstance(action_shape, list)
        and len(action_shape) == 2
    ):
        raise ValueError("contract state/action shapes are malformed")
    state_dim = int(state_shape[0])
    horizon, action_dim = (int(value) for value in action_shape)
    if state_dim <= 0 or action_dim != state_dim or horizon <= 0:
        raise ValueError(
            f"expected matching positive state/action dims, got {state_shape}/{action_shape}"
        )
    if expected_fps is not None and fps != expected_fps:
        raise ValueError(f"dataset fps={fps}, expected {expected_fps}")
    if expected_horizon is not None and horizon != expected_horizon:
        raise ValueError(f"dataset horizon={horizon}, expected {expected_horizon}")
    if expected_task is not None and contract.get("task") != expected_task:
        raise ValueError(
            f"dataset task={contract.get('task')!r}, expected {expected_task!r}"
        )

    training = contract.get("training_contract")
    if not isinstance(training, dict) or not (
        training.get("action_is_prechunked") is True
        and training.get("action_horizon") == horizon
        and training.get("ordinary_future_row_slicing_allowed") is False
        and training.get("apply_delta_transform_again") is False
        and training.get("action_padding_mask_key") == "action_is_pad"
        and training.get("current_openpi_sample_filter_key") is None
        and training.get("required_dataset_fps") == fps
        and training.get("required_action_horizon") == horizon
    ):
        raise ValueError("dataset training contract is incompatible with masked prechunked actions")

    episodes = int(contract.get("total_output_episodes", -1))
    source_episodes = int(contract.get("total_source_episodes", -2))
    frames = int(contract.get("total_frames", -1))
    padded_slots = int(contract.get("total_padded_action_slots", -1))
    real_slots = int(contract.get("total_real_action_slots", -1))
    if episodes <= 0 or episodes != source_episodes:
        raise ValueError("conversion is not one source episode to one output episode")
    if frames <= 0 or padded_slots < 0 or real_slots <= 0:
        raise ValueError("contract frame/action totals are invalid")
    if real_slots + padded_slots != frames * horizon:
        raise ValueError("contract action slot totals do not equal frames * horizon")

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("meta/info.json has no features object")
    expected_features = {
        "observation.state": ("float32", [state_dim]),
        "action": ("float32", [horizon, action_dim]),
        "action_is_pad": ("bool", [horizon]),
        "sample_valid_action_horizon": ("int64", [1]),
    }
    for key, (dtype, shape) in expected_features.items():
        feature = features.get(key)
        if not isinstance(feature, dict) or not (
            feature.get("dtype") == dtype and feature.get("shape") == shape
        ):
            raise ValueError(f"bad feature contract for {key}: {feature}")
    if not (
        info.get("codebase_version") == "v2.1"
        and info.get("fps") == fps
        and info.get("total_episodes") == episodes
        and info.get("total_frames") == frames
        and info.get("total_tasks") == 1
        and info.get("total_videos") == episodes * len(VIDEO_KEYS)
    ):
        raise ValueError("meta/info.json totals disagree with the conversion contract")
    for key in VIDEO_KEYS:
        feature = features.get(key)
        video = feature.get("info") if isinstance(feature, dict) else None
        if not isinstance(feature, dict) or not isinstance(video, dict) or not (
            feature.get("dtype") == "video"
            and feature.get("shape") == [480, 640, 3]
            and video.get("video.fps") == fps
            and video.get("video.height") == 480
            and video.get("video.width") == 640
            and video.get("video.codec") == "h264"
            and video.get("video.pix_fmt") == "yuv420p"
            and video.get("video.channels") == 3
            and video.get("has_audio") is False
        ):
            raise ValueError(f"bad video feature contract for {key}: {feature}")

    parquet_files = sorted((dataset / "data").rglob("*.parquet"))
    if len(parquet_files) != episodes:
        raise ValueError(f"Parquet count={len(parquet_files)}, expected {episodes}")
    if jsonl_count(dataset / "meta/episodes.jsonl") != episodes:
        raise ValueError("meta/episodes.jsonl count does not match the contract")
    if jsonl_count(dataset / "meta/episodes_stats.jsonl") != episodes:
        raise ValueError("meta/episodes_stats.jsonl count does not match the contract")
    for key in VIDEO_KEYS:
        count = len(list((dataset / "videos").rglob(f"{key}/episode_*.mp4")))
        if count != episodes:
            raise ValueError(f"video count for {key}={count}, expected {episodes}")

    manifests = contract.get("episodes")
    if not isinstance(manifests, list) or len(manifests) != episodes:
        raise ValueError("contract episode manifest count is invalid")
    global_offset = 0
    for episode_index, manifest in enumerate(manifests):
        if not isinstance(manifest, dict) or manifest.get("episode_index") != episode_index:
            raise ValueError(f"bad contract episode manifest at index {episode_index}")
        parquet = manifest.get("parquet")
        if not isinstance(parquet, dict):
            raise ValueError(f"missing parquet manifest at episode {episode_index}")
        relative = Path(str(parquet.get("path", "")))
        path = (dataset / relative).resolve()
        if dataset not in path.parents or not path.is_file():
            raise ValueError(f"invalid parquet manifest path: {relative}")
        if path.stat().st_size != parquet.get("size_bytes") or sha256(path) != parquet.get("sha256"):
            raise ValueError(f"parquet manifest hash/size mismatch: {relative}")
        if manifest.get("global_index_start") != global_offset:
            raise ValueError(f"non-contiguous global index at episode {episode_index}")
        global_offset = int(manifest.get("global_index_end_exclusive", -1))
    if global_offset != frames:
        raise ValueError(f"manifest global frame count={global_offset}, expected {frames}")
    return state_dim, horizon, episodes, frames, real_slots


def compute(
    dataset: Path,
    contract: dict[str, Any],
    *,
    state_dim: int,
    horizon: int,
    episodes: int,
    frames: int,
    real_slots: int,
) -> tuple[dict[str, normalize.NormStats], dict[str, Any]]:
    parquet_files = sorted((dataset / "data").rglob("*.parquet"))
    all_states = np.empty((frames, state_dim), dtype=np.float64)
    all_actions = np.empty((real_slots, state_dim), dtype=np.float64)
    state_offset = 0
    action_offset = 0
    padded_slots = 0
    full_rows = 0
    episode_audit: list[dict[str, Any]] = []

    for episode_index, path in enumerate(parquet_files):
        table = pq.read_table(
            path,
            columns=[
                "observation.state",
                "action",
                "action_is_pad",
                "sample_valid_action_horizon",
            ],
        )
        length = len(table)
        state = np.asarray(
            table["observation.state"].combine_chunks().to_pylist(), dtype=np.float64
        )
        action = np.asarray(
            table["action"].combine_chunks().to_pylist(), dtype=np.float64
        )
        mask = np.asarray(
            table["action_is_pad"].combine_chunks().to_pylist(), dtype=bool
        )
        sample_valid = np.asarray(
            table["sample_valid_action_horizon"].combine_chunks().to_pylist(),
            dtype=np.int64,
        )
        if state.shape != (length, state_dim):
            raise ValueError(f"{path}: state shape {state.shape}")
        if action.shape != (length, horizon, state_dim):
            raise ValueError(f"{path}: action shape {action.shape}")
        if mask.shape != (length, horizon):
            raise ValueError(f"{path}: action_is_pad shape {mask.shape}")
        if sample_valid.size != length:
            raise ValueError(f"{path}: sample_valid_action_horizon shape {sample_valid.shape}")
        sample_valid = sample_valid.reshape(length).astype(bool)
        if not np.array_equal(sample_valid, ~np.any(mask, axis=1)):
            raise ValueError(f"{path}: sample_valid_action_horizon disagrees with mask")
        expected_mask = (
            np.arange(length, dtype=np.int64)[:, None]
            + np.arange(1, horizon + 1, dtype=np.int64)[None, :]
            >= length
        )
        if not np.array_equal(mask, expected_mask):
            raise ValueError(f"{path}: padding mask does not match episode boundary")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"{path}: state/action contains NaN or Inf")

        selected = action[~mask]
        all_states[state_offset : state_offset + length] = state
        all_actions[action_offset : action_offset + len(selected)] = selected
        state_offset += length
        action_offset += len(selected)
        padding = int(mask.sum())
        padded_slots += padding
        full_rows += int(sample_valid.sum())
        episode_audit.append({
            "episode_index": episode_index,
            "path": str(path.relative_to(dataset)),
            "rows": length,
            "real_action_slots": int(len(selected)),
            "padded_action_slots": padding,
        })
        if (episode_index + 1) % 10 == 0 or episode_index + 1 == episodes:
            print(
                f"[masked-norm] validated {episode_index + 1}/{episodes} episodes; "
                f"states={state_offset}, real_actions={action_offset}",
                flush=True,
            )

    expected_padding = int(contract["total_padded_action_slots"])
    expected_full = int(contract["total_full_real_action_horizon_frames"])
    expected_partial = int(contract["total_partial_action_horizon_frames"])
    if state_offset != frames or action_offset != real_slots:
        raise ValueError(
            f"observed counts state={state_offset}, action={action_offset}; "
            f"expected {frames}/{real_slots}"
        )
    if padded_slots != expected_padding:
        raise ValueError(f"observed padded slots={padded_slots}, expected {expected_padding}")
    if full_rows != expected_full or frames - full_rows != expected_partial:
        raise ValueError("full/partial horizon row totals disagree with the contract")

    norm_stats = {
        "state": statistics(all_states),
        "actions": statistics(all_actions),
    }
    audit = {
        "episodes": episode_audit,
        "output_episodes": episodes,
        "state_vectors": frames,
        "action_vectors": real_slots,
        "padded_action_slots": padded_slots,
        "full_action_horizon_rows": full_rows,
        "partial_action_horizon_rows": frames - full_rows,
        "state_dim": state_dim,
        "action_dim": state_dim,
        "action_horizon": horizon,
    }
    return norm_stats, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--assets-base-dir",
        type=Path,
        default=Path("/mnt/data/dzq/openpi/data/assets"),
    )
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--expected-fps", type=int)
    parser.add_argument("--expected-horizon", type=int)
    parser.add_argument("--expected-task")
    args = parser.parse_args()
    if not args.asset_id or "/" in args.asset_id or args.asset_id in {".", ".."}:
        raise ValueError("--asset-id must be one path-free directory name")

    dataset = args.dataset.resolve(strict=True)
    contract_path = dataset / "meta/umi_conversion.json"
    info_path = dataset / "meta/info.json"
    contract = read_json(contract_path)
    info = read_json(info_path)
    initial_contract_sha256 = sha256(contract_path)
    initial_info_sha256 = sha256(info_path)
    state_dim, horizon, episodes, frames, real_slots = validate_contract(
        dataset,
        contract,
        info,
        expected_fps=args.expected_fps,
        expected_horizon=args.expected_horizon,
        expected_task=args.expected_task,
    )

    assets_base = args.assets_base_dir.resolve()
    assets_base.mkdir(parents=True, exist_ok=True)
    output = assets_base / args.asset_id
    lock_path = assets_base / ".masked_norm_stats.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing asset: {output}")
        stale = sorted(assets_base.glob(f".{args.asset_id}.building-*"))
        if stale:
            raise FileExistsError(f"unfinished normalization staging exists: {stale}")

        norm_stats, counts = compute(
            dataset,
            contract,
            state_dim=state_dim,
            horizon=horizon,
            episodes=episodes,
            frames=frames,
            real_slots=real_slots,
        )
        staging = Path(tempfile.mkdtemp(prefix=f".{args.asset_id}.building-", dir=assets_base))
        try:
            norm_path = staging / "norm_stats.json"
            norm_path.write_text(normalize.serialize_json(norm_stats) + "\n", encoding="utf-8")
            roundtrip = normalize.deserialize_json(norm_path.read_text(encoding="utf-8"))
            if set(roundtrip) != {"state", "actions"}:
                raise ValueError(f"bad norm stats keys after roundtrip: {set(roundtrip)}")
            for key in ("state", "actions"):
                value = roundtrip[key]
                arrays = (value.mean, value.std, value.q01, value.q99)
                if any(array is None or np.asarray(array).shape != (state_dim,) for array in arrays):
                    raise ValueError(f"bad {key} norm stats shapes after roundtrip")
                if any(not np.isfinite(np.asarray(array)).all() for array in arrays):
                    raise ValueError(f"non-finite {key} norm stats after roundtrip")
            if sha256(contract_path) != initial_contract_sha256 or sha256(info_path) != initial_info_sha256:
                raise RuntimeError("dataset metadata changed while normalization was running")
            final_norm_path = output / "norm_stats.json"
            audit = {
                "schema": "umi-openpi-masked-norm-stats-audit",
                "schema_version": 1,
                "status": "complete",
                "asset_id": args.asset_id,
                "dataset": str(dataset),
                "dataset_contract_sha256": initial_contract_sha256,
                "dataset_info_sha256": initial_info_sha256,
                "source_episode_ids_sha256": contract.get("source_episode_ids_sha256"),
                "norm_stats_path": str(final_norm_path),
                "norm_stats_sha256": sha256(norm_path),
                "padding_policy": "all state rows; action[~action_is_pad] only",
                "accumulator_dtype": "float64",
                "quantiles": "exact numpy linear 0.01/0.99",
                "generator": str(Path(__file__).resolve()),
                "generator_sha256": sha256(Path(__file__).resolve()),
                **counts,
            }
            (staging / "norm_stats_audit.json").write_text(
                json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(staging, output)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    print(json.dumps({
        key: value for key, value in audit.items() if key != "episodes"
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
