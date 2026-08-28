#!/usr/bin/env python3
"""Compute exact OpenPI norm stats for the task_v1 masked H50 dataset.

Every real observation state is included.  Actions are flattened across the
H=50 axis only after removing slots where ``action_is_pad`` is true.  This
implements terminal padding without dropping terminal observation anchors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import openpi.shared.normalize as normalize


STATE_DIM = 30
ACTION_HORIZON = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def statistics(value: np.ndarray) -> normalize.NormStats:
    if value.ndim != 2 or value.shape[1] != STATE_DIM or not np.isfinite(value).all():
        raise ValueError(f"expected finite (N,{STATE_DIM}), got {value.shape}")
    if len(value) == 0:
        raise ValueError("cannot compute statistics from zero real vectors")
    return normalize.NormStats(
        mean=np.mean(value, axis=0, dtype=np.float64),
        std=np.std(value, axis=0, dtype=np.float64),
        q01=np.quantile(value, 0.01, axis=0, method="linear"),
        q99=np.quantile(value, 0.99, axis=0, method="linear"),
    )


def validate_contract(contract: dict[str, object]) -> None:
    if not (
        contract.get("schema") == "umi-task-v1-dual-hand-pose-lerobot30-h50"
        and int(contract.get("schema_version", 0)) >= 4
        and contract.get("status") == "complete"
        and contract.get("fps") == 10
        and contract.get("state_shape") == [STATE_DIM]
        and contract.get("action_shape") == [ACTION_HORIZON, STATE_DIM]
    ):
        raise ValueError("dataset is not the expected task_v1 10 Hz H50 schema")
    training = contract.get("training_contract")
    if not isinstance(training, dict) or not (
        training.get("action_is_prechunked") is True
        and training.get("action_horizon") == ACTION_HORIZON
        and training.get("ordinary_future_row_slicing_allowed") is False
        and training.get("apply_delta_transform_again") is False
        and training.get("action_padding_mask_key") == "action_is_pad"
        and training.get("full_real_horizon_filter_key") is None
        and training.get("current_openpi_sample_filter_key") is None
    ):
        raise ValueError("training contract does not require per-slot action padding masks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--assets-base-dir",
        type=Path,
        default=Path("/mnt/data/dzq/openpi/data/assets"),
    )
    parser.add_argument(
        "--asset-id",
        default="umi_task_v1_hand_pose_10hz_h50_masked_v1",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    contract_path = dataset / "meta" / "umi_conversion.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract(contract)

    parquet_files = sorted((dataset / "data").rglob("*.parquet"))
    expected_episodes = int(contract["total_output_episodes"])
    if len(parquet_files) != expected_episodes:
        raise ValueError(
            f"Parquet count {len(parquet_files)} differs from contract {expected_episodes}"
        )

    states: list[np.ndarray] = []
    real_actions: list[np.ndarray] = []
    episode_audit: list[dict[str, int | str]] = []
    total_padding = 0
    for episode_index, path in enumerate(parquet_files):
        table = pq.read_table(
            path,
            columns=[
                "observation.state",
                "action",
                "action_is_pad",
                "sample_valid_h50",
            ],
        )
        state = np.asarray(
            table["observation.state"].combine_chunks().to_pylist(), dtype=np.float64
        )
        action = np.asarray(
            table["action"].combine_chunks().to_pylist(), dtype=np.float64
        )
        action_is_pad = np.asarray(
            table["action_is_pad"].combine_chunks().to_pylist(), dtype=bool
        )
        sample_valid = np.asarray(
            table["sample_valid_h50"].combine_chunks().to_pylist(), dtype=np.int64
        ).reshape(-1).astype(bool)
        length = len(table)
        if state.shape != (length, STATE_DIM):
            raise ValueError(f"{path}: state shape {state.shape}")
        if action.shape != (length, ACTION_HORIZON, STATE_DIM):
            raise ValueError(f"{path}: action shape {action.shape}")
        if action_is_pad.shape != (length, ACTION_HORIZON):
            raise ValueError(f"{path}: action_is_pad shape {action_is_pad.shape}")
        if not np.array_equal(sample_valid, ~np.any(action_is_pad, axis=1)):
            raise ValueError(f"{path}: sample_valid_h50 disagrees with action_is_pad")

        expected_pad = (
            np.arange(length, dtype=np.int64)[:, None]
            + np.arange(1, ACTION_HORIZON + 1, dtype=np.int64)[None, :]
            >= length
        )
        if not np.array_equal(action_is_pad, expected_pad):
            raise ValueError(f"{path}: action_is_pad does not match the episode boundary")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"{path}: NaN or Inf in state/action")

        selected_actions = action[~action_is_pad]
        states.append(state)
        if len(selected_actions):
            real_actions.append(selected_actions)
        padding = int(action_is_pad.sum())
        total_padding += padding
        episode_audit.append({
            "episode_index": episode_index,
            "path": str(path.relative_to(dataset)),
            "rows": length,
            "padded_action_slots": padding,
            "real_action_slots": int(len(selected_actions)),
        })

    state = np.concatenate(states, axis=0)
    action = np.concatenate(real_actions, axis=0)
    expected_rows = int(contract["total_frames"])
    expected_padding = int(contract["total_padded_action_slots"])
    expected_real_actions = int(contract["total_real_action_slots"])
    if len(state) != expected_rows:
        raise ValueError(f"state count {len(state)} differs from contract {expected_rows}")
    if total_padding != expected_padding:
        raise ValueError(
            f"padding count {total_padding} differs from contract {expected_padding}"
        )
    if len(action) != expected_real_actions:
        raise ValueError(
            f"real action count {len(action)} differs from contract {expected_real_actions}"
        )
    if len(action) != len(state) * ACTION_HORIZON - total_padding:
        raise ValueError("real action count does not equal rows*H minus padding")

    norm_stats = {"state": statistics(state), "actions": statistics(action)}
    output = args.assets_base_dir.resolve() / args.asset_id
    norm_path = output / "norm_stats.json"
    atomic_text(norm_path, normalize.serialize_json(norm_stats))
    audit = {
        "schema": "umi-task-v1-openpi-masked-norm-stats-audit",
        "schema_version": 1,
        "status": "complete",
        "asset_id": args.asset_id,
        "dataset": str(dataset),
        "dataset_contract_sha256": sha256(contract_path),
        "output_episodes": len(parquet_files),
        "state_vectors": len(state),
        "action_vectors": len(action),
        "padded_action_slots": total_padding,
        "state_dim": STATE_DIM,
        "action_dim": STATE_DIM,
        "action_horizon": ACTION_HORIZON,
        "padding_policy": "all state rows; action[~action_is_pad] only",
        "accumulator_dtype": "float64",
        "quantiles": "exact numpy linear 0.01/0.99",
        "norm_stats_path": str(norm_path),
        "norm_stats_sha256": sha256(norm_path),
        "episodes": episode_audit,
    }
    audit_path = output / "norm_stats_audit.json"
    atomic_text(
        audit_path,
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps({
        key: value for key, value in audit.items() if key != "episodes"
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
