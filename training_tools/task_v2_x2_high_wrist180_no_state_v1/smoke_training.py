#!/usr/bin/env python3
"""CPU-only no-state release gate: 95 reviewed - 10 excluded = 85, no splitting.

Run next to the three episode lists, selection_audit.json and converter_strict.py.
No GPU model is constructed and no training/checkpoint/output is written.
The final stdout record is JSON; exceptions exit nonzero and prevent launch.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import ast
import copy
import csv
import dataclasses
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

sys.dont_write_bytecode = True

OPENPI = Path("/home/dzq/openpi")
PACKAGE = Path(__file__).resolve().parent
SOURCE = Path("/mnt/data/dzq/umi_v2/data/task_v2_x2")
OLD_DATASET = Path("/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50")
BASE_DATASET = Path("/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50_right_eye_v2")
DATASET = BASE_DATASET.with_name(BASE_DATASET.name + "_wrist180")
OLD_ASSET = Path("/mnt/data/dzq/openpi/data/assets/umi_task_v2_x2_high_hand_pose_10hz_h50_masked_v1")
ASSET_ID = "umi_task_v2_x2_high_hand_pose_10hz_h50_right_eye_wrist180_masked_v1"
PROMPT = "Put the two objects into the box."
ORIGINAL_SHA256 = "b6a05245e6b0eaf68746fa43851db2f2b097a20f87e0131444f692082a4ebffd"
STRICT_CONVERTER_SHA256 = "45d90f403f7ea0214cbdf12412b70f59b5c4a6ab4cf4e3a944925ffdc6e92e9b"
EXCLUDED = {
    "20260908_171356_196236498_86862", "20260908_171920_199894486_86862",
    "20260908_172448_969727935_86862", "20260908_173008_477321212_86862",
    "20260908_173514_363156274_86862", "20260908_173536_630832952_86862",
    "20260908_181618_164061991_96413", "20260908_182238_803632672_96413",
    "20260908_182257_037984890_96413", "20260908_190437_866038415_104264",
}
IMAGE_KEYS = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
HEAD_KEY = "observation.images.head_rgb"
WRIST_KEYS = ("observation.images.left_wrist_rgb", "observation.images.right_wrist_rgb")
EXPECTED_CONFIG = "pi05_umi_task_v2_x2_high_10hz_h50_wrist180_no_state_10k_b64_w32_8gpu_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def episode_list(path: Path) -> list[str]:
    values = path.read_text(encoding="utf-8").splitlines()
    require(bool(values), f"Empty episode list: {path}")
    require(all(re.fullmatch(r"\d{8}_\d{6}_\d+_\d+", value) for value in values),
            f"Invalid episode IDs in {path}")
    require(len(values) == len(set(values)), f"Duplicate episode IDs in {path}")
    return values


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def no_state_model_paths(pi0_path: Path, model_path: Path) -> dict[str, Any]:
    """Check the actual source branches used by Pi05 without building a model."""
    pi0_tree = ast.parse(pi0_path.read_text(encoding="utf-8"))
    model_tree = ast.parse(model_path.read_text(encoding="utf-8"))
    classes = [node for node in pi0_tree.body if isinstance(node, ast.ClassDef) and node.name == "Pi0"]
    require(len(classes) == 1, "Expected one Pi0 model class")
    methods = {node.name: node for node in classes[0].body if isinstance(node, ast.FunctionDef)}

    def is_attribute(node: ast.AST, owner: str, field: str) -> bool:
        return (isinstance(node, ast.Attribute) and node.attr == field
                and isinstance(node.value, ast.Name) and node.value.id == owner)

    def state_reads(node: ast.AST) -> list[ast.Attribute]:
        return [item for item in ast.walk(node) if isinstance(item, ast.Attribute)
                and item.attr == "state" and isinstance(item.ctx, ast.Load)]

    def branch_nodes(statements: list[ast.stmt]) -> set[ast.AST]:
        return set(ast.walk(ast.Module(body=statements, type_ignores=[])))

    prefix, suffix = methods["embed_prefix"], methods["embed_suffix"]
    require(not state_reads(prefix), "Pi05 prefix unexpectedly reads observation state")
    guards = [node for node in ast.walk(suffix) if isinstance(node, ast.If)
              and isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
              and is_attribute(node.test.operand, "self", "pi05")]
    require(len(guards) == 1, "Expected one explicit not-self.pi05 state-token branch")
    guarded = branch_nodes(guards[0].body)
    suffix_reads = state_reads(suffix)
    require(bool(suffix_reads) and all(node in guarded for node in suffix_reads),
            "Suffix has state access outside the disabled Pi0-only branch")
    state_proj_uses = [node for node in ast.walk(suffix) if is_attribute(node, "self", "state_proj")]
    require(bool(state_proj_uses) and all(node in guarded for node in state_proj_uses),
            "State projection is reachable from the Pi05 suffix")

    init = methods["__init__"]
    pi05_assignments = [node for node in ast.walk(init) if isinstance(node, ast.Assign)
                       and any(is_attribute(target, "self", "pi05") for target in node.targets)]
    require(len(pi05_assignments) == 1
            and is_attribute(pi05_assignments[0].value, "config", "pi05"),
            "Model pi05 mode no longer comes directly from the resolved config")
    init_pi05_guards = [node for node in ast.walk(init) if isinstance(node, ast.If)
                       and is_attribute(node.test, "config", "pi05")]
    else_nodes = set().union(*(branch_nodes(node.orelse) for node in init_pi05_guards))
    projection_allocations = [node for node in ast.walk(init) if isinstance(node, ast.Assign)
                              and any(is_attribute(target, "self", "state_proj")
                                      for target in node.targets)]
    require(len(projection_allocations) == 1 and projection_allocations[0] in else_nodes,
            "Pi05 unexpectedly allocates a continuous state projection")
    require(not state_reads(methods["compute_loss"]), "Training loss unexpectedly conditions on state")
    sampling = methods["sample_actions"]
    sampling_parents = {child: node for node in ast.walk(sampling) for child in ast.iter_child_nodes(node)}
    for node in state_reads(sampling):
        parent = sampling_parents[node]
        require(isinstance(parent, ast.Attribute) and parent.attr == "shape",
                "Inference uses state values rather than only its batch shape")

    preprocess = [node for node in model_tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "preprocess_observation"]
    require(len(preprocess) == 1, "Expected one observation preprocessing function")
    parents = {child: node for node in ast.walk(preprocess[0]) for child in ast.iter_child_nodes(node)}
    for node in state_reads(preprocess[0]):
        parent = parents[node]
        shape_only = isinstance(parent, ast.Attribute) and parent.attr == "shape"
        interface_passthrough = (isinstance(parent, ast.keyword) and parent.arg == "state"
                                 and parent.value is node)
        require(shape_only or interface_passthrough,
                "Observation preprocessing uses state values for conditioning")
    return {"method": "AST inspection of actual installed Pi05 model paths",
            "pi0_source_sha256": digest(pi0_path), "observation_source_sha256": digest(model_path),
            "prefix_reads_state": False, "suffix_state_branch_condition": "not self.pi05",
            "pi05_allocates_continuous_state_projection": False,
            "compute_loss_reads_state": False, "inference_state_use": "batch shape only",
            "preprocess_state_use": "batch shape and unchanged interface field only"}


def selection_check(contract: dict[str, Any]) -> dict[str, Any]:
    original_file = PACKAGE / "track_pass_episodes.txt"
    selected_file = PACKAGE / "selected_85_episodes.txt"
    excluded_file = PACKAGE / "excluded_10_episodes.txt"
    original, selected, excluded = map(episode_list, (original_file, selected_file, excluded_file))
    require(digest(original_file) == ORIGINAL_SHA256, "Original 95-entry allowlist SHA changed")
    require(len(original) == 95 and len(excluded) == 10 and set(excluded) == EXCLUDED,
            "Approved original/exclusion inventory changed")
    require(EXCLUDED <= set(original), "Excluded IDs are not all in the original allowlist")
    require(selected == [value for value in original if value not in EXCLUDED]
            and len(selected) == 85, "Final allowlist must be original 95 minus exact 10, in order")
    require(isinstance(read_json(PACKAGE / "selection_audit.json"), dict),
            "selection_audit.json must be an object")
    require(digest(DATASET / "meta/episode_selection.txt") == digest(selected_file),
            "Dataset's copied allowlist differs from the approved 85")
    selected_ids_sha = hashlib.sha256("\n".join(selected).encode()).hexdigest()
    require(contract["source_episode_ids_sha256"] == selected_ids_sha
            and contract["output_episode_ids_sha256"] == selected_ids_sha,
            "Source/output episode IDs changed or split")
    selection = contract["episode_selection"]
    require(selection["selected_episode_ids"] == selected
            and selection["episode_list_sha256"] == digest(selected_file)
            and selection["selected_episode_count"] == 85,
            "Conversion selection metadata disagrees with approved 85")
    require(contract["source_snapshot"]["episode_ids"] == selected,
            "Source snapshot does not contain exactly approved 85")
    for count_key in ("total_source_episodes", "total_episodes", "total_output_episodes"):
        require(contract[count_key] == 85, f"Wrong conversion count: {count_key}")
    manifests = contract["episodes"]
    require(len(manifests) == 85, "Expected 85 output manifests")
    for index, (item, episode_id) in enumerate(zip(manifests, selected, strict=True)):
        require(item["episode_index"] == index and item["source_episode_id"] == episode_id
                and item["output_episode_id"] == episode_id
                and item["one_source_one_output_episode"] is True,
                f"Output episode {index} was split, omitted or renamed")
    for path in (PACKAGE / "converter_strict.py", DATASET / "meta/converter_strict.py"):
        require(digest(path) == STRICT_CONVERTER_SHA256, f"Strict converter SHA changed: {path}")
    return {"original_episodes": 95, "excluded_episodes": 10, "output_episodes": 85,
            "original_sha256": ORIGINAL_SHA256, "selected_sha256": digest(selected_file),
            "excluded_sha256": digest(excluded_file), "episode_ids_sha256": selected_ids_sha,
            "selection_audit_sha256": digest(PACKAGE / "selection_audit.json"),
            "one_source_one_output_episode": True}


def alignment_check(contract: dict[str, Any]) -> dict[str, Any]:
    camera = contract["camera_alignment"]
    require(camera["strict_max_delta_ns"] == 100_000_000
            and camera["max_hand_age_ns"] == 100_000_000
            and camera["hand_state_policy"] == "nearest",
            "100ms nearest-alignment settings changed")
    maximums = {key: 0 for key in ("cam0", "cam1", "left_hand", "right_hand")}
    checked_rows = 0
    for index, manifest in enumerate(contract["episodes"]):
        path = DATASET / "meta/alignment" / f"episode_{index:06d}" / "alignment_output_grid.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == manifest["output_rows"] + 1,
                f"Alignment audit row count mismatch in episode {index}")
        for row in rows:
            for key in maximums:
                value = abs(int(row[f"{key}_signed_delta_ns"]))
                require(value <= 100_000_000, f">100ms {key} alignment in episode {index}")
                maximums[key] = max(maximums[key], value)
        for field in ("source_grid_index", "episode_grid_index"):
            require(all(int(b[field]) - int(a[field]) == 1 for a, b in zip(rows[:-1], rows[1:])),
                    f"Non-contiguous output grid in episode {index}")
        for field in ("cam0_frame_index", "cam1_frame_index", "e6_source_row_index"):
            require(all(int(b[field]) > int(a[field]) for a, b in zip(rows[:-1], rows[1:])),
                    f"Reused/non-monotonic {field} in episode {index}")
        require(len({row["valid_run_index"] for row in rows}) == 1,
                f"Output crosses continuity runs in episode {index}")
        checked_rows += len(rows)
    return {"checked_grid_rows_including_initial_anchors": checked_rows,
            "maximum_abs_alignment_ms": {key: value / 1e6 for key, value in maximums.items()}}


def contained_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    require(root.resolve() in path.parents and path.is_file(),
            f"Artifact path escapes dataset or is not a file: {relative}")
    return path


def probe_dimensions(path: Path, *, raw_hevc: bool = False) -> tuple[int, int]:
    command = ["ffprobe", "-v", "error"]
    if raw_hevc:
        command += ["-f", "hevc"]
    command += ["-select_streams", "v:0", "-show_entries", "stream=width,height",
                "-of", "json", str(path)]
    result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=120)
    streams = json.loads(result.stdout).get("streams", [])
    require(len(streams) == 1, f"Expected one probed video stream: {path}")
    return int(streams[0]["width"]), int(streams[0]["height"])


def decode_reference_or_head(path: Path, *, source_frame_index: int | None = None) -> Any:
    """Decode RGB pixels; derive the reference crop from decoded iw/ih, not audit metadata."""
    import numpy as np
    command = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-filter_threads", "1"]
    if source_frame_index is not None:
        command += ["-f", "hevc"]
    command += ["-i", str(path)]
    if source_frame_index is not None:
        require(source_frame_index >= 0, "Negative raw frame index")
        command += ["-vf", f"select=eq(n\\,{source_frame_index}),"
                    "crop=iw/2:ih:iw/2:0,"
                    "scale=640:480:force_original_aspect_ratio=decrease,"
                    "pad=640:480:(ow-iw)/2:(oh-ih)/2"]
    command += ["-frames:v", "1", "-vsync", "0", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    result = subprocess.run(command, capture_output=True, check=True, timeout=180)
    require(len(result.stdout) == 640 * 480 * 3, f"Unexpected decoded RGB size: {path}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(480, 640, 3)


def pixel_metrics(reference: Any, image: Any) -> dict[str, float]:
    import numpy as np
    difference = np.asarray(reference, dtype=np.float64) - np.asarray(image, dtype=np.float64)
    mse = float(np.mean(np.square(difference)))
    return {"mse": mse, "mae": float(np.mean(np.abs(difference))),
            "psnr_db": math.inf if mse == 0 else 10 * math.log10(255**2 / mse)}


def repair_check(contract: dict[str, Any], *, dataset: Path) -> dict[str, Any]:
    """Verify recorded crop geometry and the unchanged numerical/wrist payloads."""
    audit_path = dataset / "meta/head_video_repair.json"
    repair = read_json(audit_path)
    expected = {
        "schema": "umi-head-video-right-eye-repair", "schema_version": 1, "status": "complete",
        "source_dataset": str(OLD_DATASET), "target_dataset": str(dataset),
        "source_dataset_contract_sha256": digest(OLD_DATASET / "meta/umi_conversion.json"),
        "source_dataset_info_sha256": digest(OLD_DATASET / "meta/info.json"),
    }
    for key, value in expected.items():
        require(repair.get(key) == value, f"Right-eye repair audit mismatch: {key}")
    original_contract = read_json(OLD_DATASET / "meta/umi_conversion.json")
    for key in ("source", "source_snapshot", "source_episode_ids_sha256",
                "output_episode_ids_sha256", "episode_selection", "camera_alignment",
                "training_contract", "fps", "state_shape", "action_shape",
                "state_names", "action_names", "state_semantics", "action_semantics",
                "total_source_episodes", "total_episodes", "total_output_episodes", "total_frames",
                "total_real_action_slots", "total_padded_action_slots"):
        require(contract[key] == original_contract[key], f"Head repair changed numerical contract: {key}")
    items = repair.get("episodes", [])
    require(len(items) == 85, "Right-eye repair must cover all 85 episodes")
    preserved_count = 0
    frame_count = 0
    pixel_checks = []
    for index, (item, manifest, original) in enumerate(
        zip(items, contract["episodes"], original_contract["episodes"], strict=True)
    ):
        episode_id = manifest["source_episode_id"]
        require(item["episode_index"] == index and item["source_episode_id"] == episode_id,
                f"Repair episode identity mismatch at index {index}")
        require(original["source_episode_id"] == episode_id, "Repair reordered episodes")
        raw_video = SOURCE / episode_id / "camera/e6_rgb.h265"
        require(probe_dimensions(raw_video, raw_hevc=True) == (3840, 1200),
                f"Unexpected actual decoded E6 dimensions: {episode_id}")
        require(item["input_probe"]["width"] == 3840 and item["input_probe"]["height"] == 1200
                and item["crop_xywh"] == [1920, 0, 1920, 1200],
                f"Incorrect right-half crop audit: {episode_id}")
        head = item["head_video"]
        require(head["path"] == manifest["videos"][HEAD_KEY]["path"],
                f"Head path mismatch: {episode_id}")
        head_file = contained_file(dataset, head["path"])
        head_sha = digest(head_file)
        require(head_sha == head["sha256"] == manifest["videos"][HEAD_KEY]["sha256"],
                f"Head video hash mismatch: {episode_id}")
        require(digest(raw_video) == head["source_raw_sha256"],
                f"Raw E6 video changed after repair: {episode_id}")
        require(probe_dimensions(head_file) == (640, 480),
                f"Repaired head output dimensions mismatch: {episode_id}")

        alignment_path = dataset / "meta/alignment" / f"episode_{index:06d}" / "alignment_output_grid.csv"
        old_alignment_path = OLD_DATASET / "meta/alignment" / f"episode_{index:06d}" / "alignment_output_grid.csv"
        alignment_sha = digest(alignment_path)
        require(alignment_sha == item["alignment_output_grid_sha256"] == digest(old_alignment_path),
                f"Repair changed alignment/frame selection: {episode_id}")
        with alignment_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        anchors = [int(row["e6_source_row_index"]) for row in rows if int(row["is_output_anchor"]) == 1]
        anchor_sha = hashlib.sha256(",".join(map(str, anchors)).encode("utf-8")).hexdigest()
        require(item["source_row_indices"] == anchors
                and item["source_row_indices_sha256"] == anchor_sha
                and item["frame_selection_origin"] == "existing_alignment_output_grid.is_output_anchor"
                and len(anchors) == manifest["output_rows"],
                f"Repaired video selected different frames: {episode_id}")
        reference = decode_reference_or_head(raw_video, source_frame_index=anchors[0])
        corrected_metrics = pixel_metrics(reference, decode_reference_or_head(head_file))
        old_head_file = contained_file(OLD_DATASET, original["videos"][HEAD_KEY]["path"])
        original_metrics = pixel_metrics(reference, decode_reference_or_head(old_head_file))
        require(corrected_metrics["psnr_db"] >= 30.0,
                f"Actual right-eye pixels failed PSNR>=30dB: {episode_id}; {corrected_metrics}")
        require(corrected_metrics["mse"] < original_metrics["mse"],
                f"New head is not closer than old wrong crop to independent right-eye pixels: {episode_id}")
        pixel_checks.append({"episode_index": index, "source_episode_id": episode_id,
                             "source_frame_index": anchors[0], "corrected": corrected_metrics,
                             "original_wrong_crop": original_metrics})
        expected_paths = {manifest["parquet"]["path"],
                          *(manifest["videos"][key]["path"] for key in WRIST_KEYS)}
        proofs = item["preserved_files"]
        require(len(proofs) == 3 and {proof["path"] for proof in proofs} == expected_paths,
                f"Incomplete preserved parquet/wrist proofs: {episode_id}")
        for proof in proofs:
            relative = proof["path"]
            current = contained_file(dataset, relative)
            old = contained_file(OLD_DATASET, relative)
            actual_sha, old_sha = digest(current), digest(old)
            require(actual_sha == old_sha == proof["sha256"] == proof["source_sha256"],
                    f"Head-only repair changed preserved file: {relative}")
            if relative == manifest["parquet"]["path"]:
                require(actual_sha == manifest["parquet"]["sha256"] == original["parquet"]["sha256"],
                        f"Preserved parquet manifest hash changed: {relative}")
            else:
                wrist_key = next(key for key in WRIST_KEYS if manifest["videos"][key]["path"] == relative)
                require(actual_sha == manifest["videos"][wrist_key]["sha256"]
                        == original["videos"][wrist_key]["sha256"],
                        f"Preserved wrist manifest hash changed: {relative}")
            preserved_count += 1
        frame_count += len(anchors)
        if (index + 1) % 10 == 0 or index == 84:
            print(f"[right-eye-smoke] verified {index + 1}/85 heads and preserved payloads", flush=True)
    require(frame_count == contract["total_frames"], "Repaired head frame inventory mismatch")
    return {"audit_sha256": digest(audit_path), "episodes": 85,
            "actual_raw_dimensions_all": [3840, 1200], "crop_xywh_all": [1920, 0, 1920, 1200],
            "head_output_dimensions_all": [640, 480], "head_frames": frame_count,
            "verified_preserved_files": preserved_count,
            "preserved_payload_policy": "Parquet and both wrist videos are byte-identical",
            "pixel_reference": "raw frame at CSV first anchor; independent crop=iw/2:ih:iw/2:0; "
                               "aspect-preserving scale and pad to 640x480",
            "pixel_min_psnr_db": min(item["corrected"]["psnr_db"] for item in pixel_checks),
            "pixel_max_mae": max(item["corrected"]["mae"] for item in pixel_checks),
            "pixel_checks": pixel_checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(OPENPI / "src"))
    import jax
    import numpy as np
    import pyarrow.parquet as pq
    from openpi import transforms
    from openpi.training import config as training_config, data_loader
    from openpi.shared import normalize

    require(all(device.platform == "cpu" for device in jax.devices()), "Smoke must use only CPU")
    cfg = training_config.get_config(args.config)
    factory = cfg.data
    require(cfg.name == EXPECTED_CONFIG, "Expected the independent no-state training config")
    for key, expected in {"batch_size": 64, "num_workers": 32, "num_train_steps": 10000,
                          "save_interval": 1000, "keep_period": 1000, "fsdp_devices": 8,
                          "resume": False, "overwrite": False}.items():
        require(getattr(cfg, key) == expected, f"Unexpected training {key}")
    require(cfg.model.pi05 is True and cfg.model.discrete_state_input is False
            and cfg.model.action_dim == 32 and cfg.model.action_horizon == 50,
            "Expected Pi05/H50/action32 with state conditioning disabled")
    require((cfg.policy_metadata or {}).get("state_input_mode") == "none"
            and (cfg.policy_metadata or {}).get("discrete_state_input") is False,
            "No-state policy audit metadata disagrees with the model config")
    model_path_evidence = no_state_model_paths(
        OPENPI / "src/openpi/models/pi0.py", OPENPI / "src/openpi/models/model.py")
    require(Path(factory.repo_id) == DATASET and factory.assets.asset_id == ASSET_ID
            and factory.default_prompt == PROMPT, "Wrong dataset, asset or task prompt")
    require(factory.use_head_camera and factory.head_crop_normalized is None
            and factory.head_crop_expected_aspect_ratio == 4 / 3,
            "Expected full 4:3 head image and both wrist images")
    require(factory.use_delta_eef_actions is False and factory.output_absolute_eef_actions is False,
            "Prechunked relative EEF actions must not be transformed again")
    contract_path, info_path = DATASET / "meta/umi_conversion.json", DATASET / "meta/info.json"
    contract, info = read_json(contract_path), read_json(info_path)
    require(contract["source"] == str(SOURCE) and contract["task"] == PROMPT, "Wrong raw source/task")
    selection_evidence = selection_check(contract)
    alignment_evidence = alignment_check(contract)
    repair_evidence = repair_check(read_json(BASE_DATASET / "meta/umi_conversion.json"), dataset=BASE_DATASET)
    from rotate_wrist_dataset import verify_dataset
    rotation_evidence = verify_dataset(DATASET, BASE_DATASET, decode=True)
    from wrist180_inference import self_test as inference_rotation_test
    rotation_evidence["inference_adapter_test"] = inference_rotation_test()
    require(cfg.policy_metadata.get("wrist_rotation_degrees") == {"left": 180, "right": 180}
            and cfg.policy_metadata.get("additional_training_wrist_rotation") is False
            and cfg.policy_metadata.get("training_wrist_rotation_source") == "encoded_dataset",
            "Wrist rotation metadata does not match the already-rotated dataset")
    norm_tool_path = Path("/home/dzq/data_deal/training_tools/compute_masked_norm_stats.py")
    norm_tool = load_module("task_v2_x2_masked_norm_validator", norm_tool_path)
    state_dim, horizon, episodes, frames, real_slots = norm_tool.validate_contract(
        DATASET, contract, info, expected_fps=10, expected_horizon=50, expected_task=PROMPT)
    require((state_dim, horizon, episodes) == (30, 50, 85), "Wrong dimensional contract")
    asset = Path(factory.assets.assets_dir) / ASSET_ID
    norm_path, audit_path = asset / "norm_stats.json", asset / "norm_stats_audit.json"
    require(digest(norm_path) == digest(OLD_ASSET / "norm_stats.json"),
            "Image-only repair changed numerical normalization statistics")
    audit = read_json(audit_path)
    expected_audit = {"schema": "umi-openpi-masked-norm-stats-audit", "status": "complete",
                      "asset_id": ASSET_ID, "dataset": str(DATASET),
                      "dataset_contract_sha256": digest(contract_path),
                      "dataset_info_sha256": digest(info_path),
                      "source_episode_ids_sha256": contract["source_episode_ids_sha256"],
                      "norm_stats_sha256": digest(norm_path),
                      "padding_policy": "all state rows; action[~action_is_pad] only",
                      "generator_sha256": digest(norm_tool_path), "output_episodes": 85,
                      "state_vectors": frames, "action_vectors": real_slots,
                      "action_horizon": 50, "state_dim": 30, "action_dim": 30}
    for key, value in expected_audit.items():
        require(audit.get(key) == value, f"Norm audit mismatch: {key}")
    recomputed_stats, recomputed_counts = norm_tool.compute(
        DATASET, contract, state_dim=state_dim, horizon=horizon,
        episodes=episodes, frames=frames, real_slots=real_slots)
    persisted_stats = normalize.deserialize_json(norm_path.read_text(encoding="utf-8"))
    for key in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            expected, actual = getattr(recomputed_stats[key], field), getattr(persisted_stats[key], field)
            require(np.asarray(actual).shape == (30,), f"Bad norm shape: {key}.{field}")
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12,
                                       err_msg=f"Masked norm stats mismatch: {key}.{field}")
    for key, value in recomputed_counts.items():
        require(audit.get(key) == value, f"Norm count/audit mismatch: {key}")

    adjacent_rows_checked = 0
    for manifest in contract["episodes"]:
        table = pq.read_table(DATASET / manifest["parquet"]["path"], columns=["observation.state", "action"])
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if len(states) > 1:
            np.testing.assert_allclose(actions[:-1, 0], states[1:], rtol=1e-5, atol=2e-6,
                                       err_msg=f"k=1 action/state inconsistency: {manifest['source_episode_id']}")
            adjacent_rows_checked += len(states) - 1

    data_cfg = factory.create(cfg.assets_dirs, cfg.model)
    require(tuple(data_cfg.action_sequence_keys) == () and data_cfg.sample_filter_key is None
            and data_cfg.action_padding_mask_key == "action_is_pad" and data_cfg.use_quantile_norm,
            "Loader must retain anchors, use prechunked masks and quantile normalization")
    pipeline = [*data_cfg.repack_transforms.inputs, *data_cfg.data_transforms.inputs, *data_cfg.model_transforms.inputs]
    require(not any(type(item).__name__.startswith(("Delta", "Absolute")) for item in pipeline),
            "Unexpected delta/absolute transform in the training input pipeline")
    token_steps = [item for item in data_cfg.model_transforms.inputs if isinstance(item, transforms.TokenizePrompt)]
    require(len(token_steps) == 1 and token_steps[0].discrete_state_input is False,
            "Expected one prompt-only tokenizer")
    tokenizer = token_steps[0].tokenizer
    before_model_transform = transforms.compose([
        *data_cfg.repack_transforms.inputs, *data_cfg.data_transforms.inputs,
        transforms.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm),
    ])
    model_input_transform = transforms.compose(data_cfg.model_transforms.inputs)
    raw = data_loader.create_torch_dataset(data_cfg, 50, cfg.model)
    transformed = data_loader.transform_dataset(raw, data_cfg)
    require(len(raw) == frames and len(transformed) == frames, "Loader has filtered observation rows")

    def normalize_expected(value: Any, key: str) -> Any:
        stats = persisted_stats[key]
        return (np.asarray(value, dtype=np.float32) - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0

    sample_evidence = []
    first_transformed = None
    for index in sorted({0, frames // 2, frames - 2, frames - 1}):
        source_sample, sample = raw[index], transformed[index]
        if index == 0:
            first_transformed = sample
        state, actions, mask = np.asarray(sample["state"]), np.asarray(sample["actions"]), np.asarray(sample["action_is_pad"])
        require(state.shape == (32,) and actions.shape == (50, 32), f"Wrong transformed shapes at {index}")
        require(mask.shape == (50,) and mask.dtype == np.dtype(bool), f"Wrong mask at {index}")
        require(np.isfinite(state).all() and np.isfinite(actions).all(), "Nonfinite transformed values")
        np.testing.assert_array_equal(state[30:], 0.0)
        np.testing.assert_array_equal(actions[:, 30:], 0.0)
        np.testing.assert_allclose(state[:30], normalize_expected(source_sample["observation.state"], "state"), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(actions[:, :30], normalize_expected(source_sample["action"], "actions"), rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(mask, np.asarray(source_sample["action_is_pad"]))
        require(set(sample["image"]) == IMAGE_KEYS and all(np.asarray(value).shape == (224, 224, 3)
                    for value in sample["image"].values()), "Wrong decoded/resized three-camera images")
        require(all(bool(value) for value in sample["image_mask"].values()), "A camera is masked out")

        # Prompt-only tokenization; the numeric state field remains in the API.
        clean_prompt = PROMPT.strip().replace("_", " ").replace("\n", " ")
        full_tokens = (tokenizer._tokenizer.encode(clean_prompt, add_bos=True)
                       + tokenizer._tokenizer.encode("\n"))
        require(len(full_tokens) <= cfg.model.max_token_len, f"Task prompt truncated at row {index}")
        token_ids, token_mask = tokenizer.tokenize(PROMPT, None)
        np.testing.assert_array_equal(sample["tokenized_prompt"], token_ids)
        np.testing.assert_array_equal(sample["tokenized_prompt_mask"], token_mask)
        np.testing.assert_array_equal(token_ids[token_mask], full_tokens)

        # Perturb an actual sample at the model-input boundary, then execute the
        # same configured model transforms as the real data loader. Re-tokenizing
        # only with a hardcoded None would not test this configuration path.
        before_model = before_model_transform(copy.deepcopy(source_sample))
        require(np.asarray(before_model["state"]).shape == (30,), "Wrong pre-tokenizer state shape")
        np.testing.assert_allclose(before_model["state"], state[:30], rtol=1e-6, atol=1e-6)
        original_input = model_input_transform(copy.deepcopy(before_model))
        alternate_input = copy.deepcopy(before_model)
        alternate_state = np.linspace(-0.875, 0.875, 30, dtype=np.float64)
        if np.array_equal(alternate_state, np.asarray(before_model["state"])):
            alternate_state += 0.125
        require(np.isfinite(alternate_state).all(), "State perturbation must be finite")
        alternate_input["state"] = alternate_state
        altered = model_input_transform(alternate_input)
        require(not np.array_equal(np.asarray(altered["state"]), np.asarray(original_input["state"])),
                "State perturbation did not reach the retained state field")
        for key in ("tokenized_prompt", "tokenized_prompt_mask"):
            np.testing.assert_array_equal(original_input[key], sample[key],
                                          err_msg=f"Real model transform mismatch: {key}")
            np.testing.assert_array_equal(altered[key], original_input[key],
                                          err_msg=f"State values leaked into prompt-only input: {key}")
        np.testing.assert_array_equal(altered["actions"], original_input["actions"])
        np.testing.assert_array_equal(altered["action_is_pad"], original_input["action_is_pad"])
        require(set(altered["image"]) == set(original_input["image"]) == IMAGE_KEYS
                and set(altered["image_mask"]) == set(original_input["image_mask"]) == IMAGE_KEYS,
                "State perturbation changed the three-camera input keys")
        for camera_key in sorted(IMAGE_KEYS):
            np.testing.assert_array_equal(altered["image"][camera_key], original_input["image"][camera_key],
                                          err_msg=f"State perturbation changed image: {camera_key}")
            np.testing.assert_array_equal(altered["image_mask"][camera_key], original_input["image_mask"][camera_key],
                                          err_msg=f"State perturbation changed image mask: {camera_key}")
        sample_evidence.append({"row": index, "state_token_dimensions": 0,
                                "finite_state_perturbation_preserves_tokens_and_mask": True,
                                "finite_state_perturbation_preserves_three_images_and_masks": True,
                                "finite_state_perturbation_preserves_actions_and_padding": True,
                                "prompt_token_count": len(full_tokens), "real_action_slots": int((~mask).sum())})
    require(sample_evidence[-2]["real_action_slots"] == 1 and sample_evidence[-1]["real_action_slots"] == 0,
            "Terminal mask does not preserve penultimate and terminal anchors")
    loader = data_loader.create_data_loader(dataclasses.replace(cfg, batch_size=1, num_workers=0),
                                           shuffle=False, num_batches=1, framework="jax")
    observation, batched_actions = next(iter(loader))
    require(np.asarray(observation.state).shape == (1, 32)
            and np.asarray(batched_actions).shape == (1, 50, 32)
            and np.asarray(observation.action_is_pad).shape == (1, 50)
            and np.asarray(observation.action_is_pad).dtype == np.dtype(bool), "Wrong real model batch")
    require(set(observation.images) == IMAGE_KEYS
            and all(np.asarray(value).shape == (1, 224, 224, 3) for value in observation.images.values()),
            "Wrong batched three-camera shapes")
    np.testing.assert_allclose(np.asarray(observation.state)[0], first_transformed["state"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(batched_actions)[0], first_transformed["actions"], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(observation.tokenized_prompt)[0], first_transformed["tokenized_prompt"])
    np.testing.assert_array_equal(np.asarray(observation.tokenized_prompt_mask)[0], first_transformed["tokenized_prompt_mask"])
    np.testing.assert_array_equal(np.asarray(observation.action_is_pad)[0], first_transformed["action_is_pad"])
    require(all(np.isfinite(np.asarray(value)).all() for value in observation.images.values()), "Nonfinite batched images")
    train = load_module("task_v2_x2_train_smoke", OPENPI / "scripts/train.py")
    loss = np.asarray([[1.0, 100.0, 3.0]], dtype=np.float32)
    loss_mask = np.asarray([[False, True, False]], dtype=bool)
    require(float(train.masked_action_loss_mean(loss, loss_mask)) == 2.0
            and float(train.masked_action_loss_mean(loss, np.ones_like(loss_mask))) == 0.0,
            "Action padding is not removed from loss correctly")
    print(json.dumps({"status": "complete", "config": args.config, "device": "cpu",
                      "selection": selection_evidence, "alignment": alignment_evidence,
                      "parent_right_eye_repair": repair_evidence, "wrist_rotation": rotation_evidence,
                      "dataset": str(DATASET), "frames": frames, "episodes": episodes,
                      "state_input_mode": "none", "discrete_state_input": False,
                      "state_values_used_for_conditioning": False,
                      "state_field_retained_for_interface": True,
                      "state_token_dimensions": 0, "model_path_audit": model_path_evidence,
                      "batch_state_shape": [1, 32], "batch_action_shape": [1, 50, 32],
                      "batch_mask_shape": [1, 50], "image_shape_each": [1, 224, 224, 3],
                      "action_sequence_keys": [], "second_delta_transform": False,
                      "adjacent_k1_rows_checked": adjacent_rows_checked, "samples": sample_evidence,
                      "masked_loss_example": 2.0, "all_pad_loss_example": 0.0,
                      "norm_vectors": {"state": frames, "actions_unpadded": real_slots},
                      "norm_stats_sha256": digest(norm_path), "norm_audit_sha256": digest(audit_path)},
                     ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
