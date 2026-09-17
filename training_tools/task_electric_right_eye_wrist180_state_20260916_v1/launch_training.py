#!/usr/bin/env python3
"""Validate a fresh UMI H50 with-state run and launch when all eight GPUs are idle.

The CPU smoke script must accept ``--config NAME`` and exit nonzero on failure.
This launcher never kills a process, resumes a run, overwrites a checkpoint, or
restarts failed training. ``--wait`` can be run under nohup to queue one run.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


OPENPI = Path("/home/dzq/openpi")
PYTHON = OPENPI / ".venv/bin/python"
GPUS = tuple(range(8))
MARKER = "LAUNCH_CONFIG_JSON="


def cpu_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               PYTHONPATH=str(OPENPI / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1")
    return env


def config_json(name: str) -> None:
    sys.path.insert(0, str(OPENPI / "src"))
    from openpi.training import config as training_config

    cfg = training_config.get_config(name)
    data = cfg.data
    value = {"name": cfg.name, "model_pi05": cfg.model.pi05,
             "action_dim": cfg.model.action_dim,
             "action_horizon": cfg.model.action_horizon,
             "discrete_state_input": cfg.model.discrete_state_input,
             "state_input_mode": cfg.policy_metadata.get("state_input_mode"),
             "state_values_used_for_conditioning": cfg.policy_metadata.get("state_values_used_for_conditioning"),
             "state_field_retained_for_interface": cfg.policy_metadata.get("state_field_retained_for_interface"),
             "wrist_rotation_degrees": cfg.policy_metadata.get("wrist_rotation_degrees"),
             "additional_training_wrist_rotation": cfg.policy_metadata.get("additional_training_wrist_rotation"),
             "training_wrist_rotation_source": cfg.policy_metadata.get("training_wrist_rotation_source"),
             "batch_size": cfg.batch_size, "num_workers": cfg.num_workers,
             "num_train_steps": cfg.num_train_steps,
             "save_interval": cfg.save_interval, "keep_period": cfg.keep_period,
             "fsdp_devices": cfg.fsdp_devices,
             "wandb_enabled": cfg.wandb_enabled,
             "resume": cfg.resume, "overwrite": cfg.overwrite,
             "data_class": type(data).__name__, "dataset": str(data.repo_id),
             "asset": str(Path(data.assets.assets_dir or cfg.assets_dirs)
                          / data.assets.asset_id),
             "task_prompt": data.default_prompt,
             "use_head_camera": data.use_head_camera,
             "head_crop_normalized": data.head_crop_normalized,
             "use_delta_eef_actions": data.use_delta_eef_actions,
             "output_absolute_eef_actions": data.output_absolute_eef_actions,
             "sample_filter_key": data.sample_filter_key,
             "action_padding_mask_key": data.action_padding_mask_key,
             "required_action_horizon": data.required_action_horizon,
             "base_params": str(cfg.weight_loader.params_path),
             "checkpoint_base_dir": str(cfg.checkpoint_base_dir)}
    print(MARKER + json.dumps(value, sort_keys=True))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validation(args: argparse.Namespace) -> dict[str, Any]:
    result = subprocess.run([str(PYTHON), str(Path(__file__).resolve()),
                             "--internal-config-json", args.config],
                            cwd=OPENPI, env=cpu_env(), text=True,
                            capture_output=True, check=True, timeout=180)
    lines = [line[len(MARKER):] for line in result.stdout.splitlines()
             if line.startswith(MARKER)]
    if len(lines) != 1:
        raise RuntimeError("Config inspection did not return exactly one JSON record")
    cfg = json.loads(lines[0])
    expected = {"name": args.config, "model_pi05": True, "action_dim": 32,
                "action_horizon": 50, "discrete_state_input": True,
                "state_input_mode": "pi05_discrete_tokens", "state_values_used_for_conditioning": True,
                "state_field_retained_for_interface": True,
                "wrist_rotation_degrees": {"left": 180, "right": 180},
                "additional_training_wrist_rotation": False,
                "training_wrist_rotation_source": "encoded_dataset",
                "batch_size": 64, "num_workers": 32, "num_train_steps": 20000,
                "save_interval": 1000, "keep_period": 1000, "fsdp_devices": 8,
                "wandb_enabled": False, "resume": False, "overwrite": False,
                "data_class": "LeRobotUmiPrechunkedEefRot6dHand30DataConfig",
                "use_head_camera": True, "head_crop_normalized": None,
                "use_delta_eef_actions": False, "output_absolute_eef_actions": False,
                "sample_filter_key": None, "action_padding_mask_key": "action_is_pad",
                "required_action_horizon": 50}
    errors = {key: {"actual": cfg.get(key), "expected": expected_value}
              for key, expected_value in expected.items()
              if cfg.get(key) != expected_value}
    if errors:
        raise ValueError("Training config mismatch: " + json.dumps(errors))
    dataset = Path(cfg["dataset"])
    asset = Path(cfg["asset"])
    if not dataset.is_absolute() or not dataset.is_dir() or ".building-" in dataset.name:
        raise ValueError(f"Final local dataset is missing: {dataset}")
    if list(dataset.parent.glob(f".{dataset.name}.building-*")):
        raise ValueError(f"Dataset conversion staging still exists for {dataset}")
    if not Path(cfg["base_params"]).is_dir():
        raise FileNotFoundError(cfg["base_params"])
    contract_path = dataset / "meta/umi_conversion.json"
    info_path = dataset / "meta/info.json"
    contract, info = read_json(contract_path), read_json(info_path)
    required = {"schema": "umi-folder-dual-hand-pose-lerobot", "status": "complete",
                "target": str(dataset), "task": cfg["task_prompt"], "fps": 10,
                "state_shape": [30], "action_shape": [50, 30],
                "source_snapshot_verified_before_publish": True,
                "one_source_one_output_episode": True}
    for key, value in required.items():
        if contract.get(key) != value:
            raise ValueError(f"Dataset contract {key}={contract.get(key)!r}, expected {value!r}")
    training = contract.get("training_contract", {})
    for key, value in {"action_is_prechunked": True, "action_horizon": 50,
                       "ordinary_future_row_slicing_allowed": False,
                       "apply_delta_transform_again": False,
                       "action_padding_mask_key": "action_is_pad",
                       "current_openpi_sample_filter_key": None}.items():
        if key not in training or training[key] != value:
            raise ValueError(f"Incompatible dataset training contract: {key}")
    if info.get("fps") != 10 or int(info.get("total_frames", 0)) <= 0:
        raise ValueError("Dataset info has invalid FPS/frame count")
    fingerprint_files = [contract_path, info_path, asset / "norm_stats.json",
                         asset / "norm_stats_audit.json", args.smoke_script,
                         Path(__file__).resolve(), OPENPI / "scripts/train.py",
                         OPENPI / "src/openpi/training/config.py",
                         OPENPI / "src/openpi/training/data_loader.py",
                         OPENPI / "src/openpi/models/pi0.py",
                         OPENPI / "src/openpi/models/pi0_config.py",
                         OPENPI / "src/openpi/models/tokenizer.py",
                         OPENPI / "src/openpi/models/model.py",
                         OPENPI / "src/openpi/policies/cx002_policy.py",
                         OPENPI / "src/openpi/shared/image_tools.py",
                         OPENPI / "src/openpi/transforms.py",
                         dataset / "meta/converter_electric.py",
                         dataset / "meta/publish_repaired_dataset.py",
                         dataset / "meta/episode_selection.txt",
                         Path("/mnt/data/dzq/umi_v2/data/task_electric/pass_sessions.txt"),
                         Path("/home/dzq/data_deal/training_tools/compute_masked_norm_stats.py")]
    # Freeze the selection proof used by this task's CPU smoke while queued.
    for name in ("pass_sessions.txt", "experiment.py", "converter_electric.py", "inspect.json",
                 "publish_repaired_dataset.py",
                 "wrist180_inference.py", "serve_wrist180.py"):
        dependency = args.smoke_script.parent / name
        if dependency.is_file():
            fingerprint_files.append(dependency)
    for manifest in contract.get("episodes", []):
        parquet = (dataset / manifest["parquet"]["path"]).resolve(strict=True)
        if dataset.resolve() not in parquet.parents:
            raise ValueError(f"Parquet escapes dataset: {parquet}")
        if digest(parquet) != manifest["parquet"]["sha256"]:
            raise ValueError(f"Dataset parquet hash changed: {parquet}")
        fingerprint_files.append(parquet)
        for video_record in manifest.get("videos", {}).values():
            video = (dataset / video_record["path"]).resolve(strict=True)
            if dataset.resolve() not in video.parents:
                raise ValueError(f"Video escapes dataset: {video}")
            if (video.stat().st_size != video_record["size_bytes"]
                    or digest(video) != video_record["sha256"]):
                raise ValueError(f"Dataset video hash/size changed: {video}")
            fingerprint_files.append(video)
    fingerprints = {str(path): digest(path) for path in fingerprint_files}
    return {"config": cfg, "total_episodes": info["total_episodes"],
            "total_frames": info["total_frames"], "fingerprints": fingerprints}


def active_openpi_training() -> list[dict[str, Any]]:
    """Detect the CPU initialization phase before a train PID claims its GPUs."""
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = entry.joinpath("cmdline").read_bytes().decode(
                "utf-8", errors="replace").split("\0")
            if str(OPENPI / "scripts/train.py") in argv or (
                "scripts/train.py" in argv and entry.joinpath("cwd").resolve() == OPENPI
            ):
                result.append({"pid": int(entry.name), "argv": [item for item in argv if item]})
        except (OSError, RuntimeError):
            continue
    return result


def gpu_snapshot() -> dict[str, Any]:
    def query(kind: str, fields: str) -> list[list[str]]:
        output = subprocess.run(["nvidia-smi", f"--query-{kind}={fields}",
                                 "--format=csv,noheader,nounits"],
                                text=True, capture_output=True, check=True, timeout=30).stdout
        return [[part.strip() for part in line.split(",")]
                for line in output.splitlines() if line.strip()]

    gpus = [{"index": int(index), "uuid": uuid, "memory_used_mib": int(memory),
             "utilization_percent": int(utilization)}
            for index, uuid, memory, utilization in
            query("gpu", "index,uuid,memory.used,utilization.gpu")]
    if tuple(gpu["index"] for gpu in gpus) != GPUS:
        raise RuntimeError(f"Expected exactly GPUs 0–7, got {gpus}")
    applications = query("compute-apps", "gpu_uuid,pid,used_gpu_memory")
    training_processes = active_openpi_training()
    idle = not applications and not training_processes and all(
        gpu["memory_used_mib"] <= 1024 and gpu["utilization_percent"] <= 5 for gpu in gpus)
    return {"idle": idle, "gpus": gpus, "compute_applications": applications,
            "active_openpi_train_processes": training_processes}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--internal-config-json":
        config_json(sys.argv[2])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--smoke-script", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--wait-timeout-hours", type=float, default=72)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    args = parser.parse_args()
    for name in (args.config, args.exp_name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            parser.error("config and exp-name must be single safe path components")
    if not args.smoke_script.is_absolute() or not args.smoke_script.is_file():
        parser.error("--smoke-script must name an existing absolute Python file")
    if args.poll_seconds < 5 or args.wait_timeout_hours <= 0 or not 0 < args.memory_fraction <= 0.90:
        parser.error("poll must be >=5s, timeout positive, and memory fraction in (0,0.90]")

    evidence = validation(args)
    log_dir = OPENPI / "logs"
    checkpoint = Path(evidence["config"]["checkpoint_base_dir"]) / args.config / args.exp_name
    log = log_dir / f"{args.exp_name}.log"
    receipt = log_dir / f"{args.exp_name}.launch.json"
    state = log_dir / f"{args.exp_name}.queue.json"
    smoke_log = log_dir / f"{args.exp_name}.smoke.log"
    conflicts = [str(path) for path in (checkpoint, log, receipt, state, smoke_log) if path.exists()]
    if conflicts:
        raise FileExistsError(f"Fresh run paths already exist: {conflicts}")
    snapshot = gpu_snapshot()
    if args.check_only:
        print(json.dumps({"status": "ready" if snapshot["idle"] else "prepared_gpu_busy",
                          "evidence": evidence, "gpu_snapshot": snapshot,
                          "checkpoint": str(checkpoint), "log": str(log)}, indent=2))
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f".{args.exp_name}.queue.lock").open("a+") as queue_lock:
        fcntl.flock(queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Recheck after acquiring the lock: a concurrent invocation may have won.
        if any(path.exists() for path in (checkpoint, log, receipt, state, smoke_log)):
            raise FileExistsError("Fresh run paths appeared before the queue lock was acquired")

        def status(phase: str, **details: Any) -> None:
            value = {"status": phase, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "queue_pid": os.getpid(), "config": args.config,
                     "exp_name": args.exp_name, "checkpoint": str(checkpoint),
                     "log": str(log), **details}
            atomic_json(state, value)
            print(json.dumps(value, ensure_ascii=False), flush=True)

        try:
            status("cpu_smoke", evidence=evidence)
            smoke_command = [str(PYTHON), str(args.smoke_script), "--config", args.config]
            with smoke_log.open("x", encoding="utf-8") as output:
                subprocess.run(smoke_command, cwd=OPENPI, env=cpu_env(),
                               stdout=output, stderr=subprocess.STDOUT,
                               check=True, timeout=1800)
            deadline = time.monotonic() + args.wait_timeout_hours * 3600
            with (log_dir / ".openpi_training_launch.lock").open("a+") as global_lock:
                while True:
                    snapshot = gpu_snapshot()
                    if snapshot["idle"]:
                        try:
                            fcntl.flock(global_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            snapshot = {**snapshot, "idle": False, "launch_lock_busy": True}
                        else:
                            stable = []
                            for _ in range(3):
                                stable.append(gpu_snapshot())
                                if not stable[-1]["idle"]:
                                    break
                                time.sleep(2)
                            if len(stable) == 3 and all(item["idle"] for item in stable):
                                break
                            fcntl.flock(global_lock.fileno(), fcntl.LOCK_UN)
                    status("waiting_gpu", gpu_snapshot=snapshot)
                    if not args.wait:
                        raise RuntimeError("All eight GPUs must be idle; use --wait to queue")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("GPU queue timed out")
                    time.sleep(args.poll_seconds)

                refreshed = validation(args)
                if refreshed != evidence:
                    raise RuntimeError("Config/data/norm/code changed after CPU smoke; refusing launch")
                if any(path.exists() for path in (checkpoint, log, receipt)):
                    raise FileExistsError("Training output appeared while waiting")
                final_gpu = gpu_snapshot()
                if not final_gpu["idle"]:
                    raise RuntimeError("GPU state changed immediately before launch")
                command = [str(PYTHON), str(OPENPI / "scripts/train.py"),
                           args.config, "--exp-name", args.exp_name]
                env = os.environ.copy()
                env.pop("JAX_PLATFORMS", None)
                env.pop("JAX_PLATFORM_NAME", None)
                env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           CUDA_VISIBLE_DEVICES=",".join(map(str, GPUS)),
                           XLA_PYTHON_CLIENT_PREALLOCATE="true",
                           XLA_PYTHON_CLIENT_MEM_FRACTION=str(args.memory_fraction),
                           PYTHONUNBUFFERED="1", PYTHONPATH=str(OPENPI / "src"))
                with log.open("x", encoding="utf-8") as output:
                    process = subprocess.Popen(command, cwd=OPENPI, env=env,
                                               stdin=subprocess.DEVNULL, stdout=output,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                raw_stat = Path(f"/proc/{process.pid}/stat").read_text()
                start_ticks = int(raw_stat[raw_stat.rfind(")") + 2:].split()[19])
                launch_receipt = {"schema": "umi-eight-gpu-training-launch", "mode": "fresh",
                                  "pid": process.pid, "process_start_ticks": start_ticks,
                                  "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                  "config": args.config, "exp_name": args.exp_name,
                                  "command": command, "cwd": str(OPENPI),
                                  "log": str(log), "checkpoint_root": str(checkpoint),
                                  "memory_fraction": args.memory_fraction,
                                  "gpu_indices": list(GPUS), "evidence": evidence,
                                  "cpu_smoke_command": smoke_command,
                                  "cpu_smoke_log": str(smoke_log),
                                  "prelaunch_gpu_snapshots": [*stable, final_gpu]}
                with receipt.open("x", encoding="utf-8") as stream:
                    json.dump(launch_receipt, stream, indent=2, ensure_ascii=False)
                    stream.write("\n")
                status("training_starting", pid=process.pid, receipt=str(receipt))
                # JAX may import/initialize on CPU for minutes. Hold the launch
                # lock until this exact process has claimed every target GPU.
                startup_deadline = time.monotonic() + 15 * 60
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Training exited during startup: {process.returncode}; see {log}")
                    current = gpu_snapshot()
                    target_uuids = {gpu["uuid"] for gpu in current["gpus"]}
                    claimed_uuids = {
                        row[0] for row in current["compute_applications"]
                        if len(row) >= 2 and row[1] == str(process.pid)
                    }
                    if target_uuids <= claimed_uuids:
                        status("training_process_running", pid=process.pid, receipt=str(receipt),
                               gpu_snapshot=current, all_eight_gpus_claimed=True)
                        return 0
                    if time.monotonic() >= startup_deadline:
                        status("startup_unconfirmed", pid=process.pid, receipt=str(receipt),
                               process_alive=True, gpu_snapshot=current,
                               error="Process is alive but did not claim all 8 GPUs within 15 minutes; "
                                     "inspect the exact PID and log before any manual retry. "
                                     "No process was killed and no restart was attempted.")
                        return 3
                    status("training_starting", pid=process.pid, receipt=str(receipt),
                           gpu_snapshot=current, claimed_gpu_count=len(claimed_uuids & target_uuids))
                    time.sleep(10)
        except Exception as error:
            status("failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
