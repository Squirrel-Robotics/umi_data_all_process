#!/usr/bin/env python3
"""Finish this explicit 85-episode conversion, normalize, and safely queue training.

No source data is changed and no process is stopped. Every stage fails closed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent
DATASET = Path('/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50')
ASSET = Path('/mnt/data/dzq/openpi/data/assets/umi_task_v2_x2_high_hand_pose_10hz_h50_masked_v1')
PROMPT = 'Put the two objects into the box.'
SELECTED_SHA = '10529d85a309173ca01348bb092ca36278d8064737656b88994baa90152f4b2b'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def process_state(pid: int, expected_start_ticks: int) -> str:
    path = Path(f'/proc/{pid}/stat')
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return 'exited'
    fields = raw[raw.rfind(')') + 2:].split()
    if int(fields[19]) != expected_start_ticks:
        raise RuntimeError('Conversion PID was reused; manual inspection required')
    return 'exited' if fields[0] == 'Z' else 'running'


def verify_dataset() -> dict:
    contract = read_json(DATASET / 'meta/umi_conversion.json')
    expected = {
        'schema': 'umi-folder-dual-hand-pose-lerobot', 'status': 'complete',
        'source': '/mnt/data/dzq/umi_v2/data/task_v2_x2', 'target': str(DATASET),
        'repo_id': 'dzq/task_v2_x2_high_lerobot_10hz_h50',
        'task': PROMPT, 'fps': 10, 'state_shape': [30], 'action_shape': [50, 30],
        'total_source_episodes': 85, 'total_output_episodes': 85,
        'total_frames': 9995,
        'one_source_one_output_episode': True,
        'source_snapshot_verified_before_publish': True,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f'Final dataset {key}: {contract.get(key)!r} != {value!r}')
    selected = ROOT / 'selected_85_episodes.txt'
    for path in (selected, DATASET / 'meta/episode_selection.txt'):
        if sha256(path) != SELECTED_SHA:
            raise ValueError(f'85-episode selection hash changed: {path}')
    ids = selected.read_text(encoding='utf-8').splitlines()
    ids_sha = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    selection = contract.get('episode_selection', {})
    if (selection.get('episode_list_sha256') != SELECTED_SHA
            or selection.get('selected_episode_ids') != ids
            or contract.get('source_episode_ids_sha256') != ids_sha
            or contract.get('output_episode_ids_sha256') != ids_sha):
        raise ValueError('Final contract does not bind the exact approved 85 episode IDs')
    if list(DATASET.parent.glob(f'.{DATASET.name}.building-*')):
        raise ValueError('Conversion staging remains; refusing normalization')
    return {'episodes': 85, 'frames': contract['total_frames'],
            'contract_sha256': sha256(DATASET / 'meta/umi_conversion.json')}


def status(phase: str, **details) -> None:
    target = ROOT / 'pipeline_status.json'
    temporary = ROOT / f'.pipeline_status.{os.getpid()}.tmp'
    value = {'status': phase, 'pid': os.getpid(),
             'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **details}
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    os.replace(temporary, target)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conversion-pid', required=True, type=int)
    parser.add_argument('--conversion-start-ticks', required=True, type=int)
    parser.add_argument('--conversion-timeout-hours', type=float, default=6)
    args = parser.parse_args()
    if (args.conversion_pid <= 0 or args.conversion_start_ticks <= 0
            or not math.isfinite(args.conversion_timeout_hours)
            or args.conversion_timeout_hours <= 0):
        parser.error('PID, start ticks and timeout must be positive')
    with (ROOT / '.finish_and_queue.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / 'pipeline_status.json').exists():
            raise FileExistsError('Pipeline status exists; inspect it instead of launching twice')
        try:
            if sha256(ROOT / 'selected_85_episodes.txt') != SELECTED_SHA:
                raise ValueError('Selected allowlist has changed')
            deadline = time.monotonic() + args.conversion_timeout_hours * 3600
            status('waiting_conversion', conversion_pid=args.conversion_pid)
            while process_state(args.conversion_pid, args.conversion_start_ticks) == 'running':
                if time.monotonic() > deadline:
                    raise TimeoutError('Conversion wait timed out; conversion was NOT stopped')
                time.sleep(30)
            evidence = verify_dataset()
            if ASSET.exists() or (ROOT / 'normalization.log').exists():
                raise FileExistsError('Norm asset/log already exists; manual verification required')
            status('normalizing', dataset=evidence)
            with (ROOT / 'normalization.log').open('x', encoding='utf-8') as output:
                subprocess.run(['bash', str(ROOT / 'normalize_dataset.sh')], cwd=ROOT,
                               stdout=output, stderr=subprocess.STDOUT,
                               check=True, timeout=1800)
            if verify_dataset() != evidence:
                raise RuntimeError('Dataset contract changed while normalization was running')
            audit = read_json(ASSET / 'norm_stats_audit.json')
            if (audit.get('status') != 'complete'
                    or audit.get('dataset_contract_sha256') != evidence['contract_sha256']
                    or audit.get('norm_stats_sha256') != sha256(ASSET / 'norm_stats.json')):
                raise ValueError('Normalization audit does not match the dataset and stats')
            status('cpu_validation_then_gpu_queue', dataset=evidence,
                   queue_status='/home/dzq/openpi/logs/task_v2_x2_high_10hz_h50_full_head_state_10k_b64_w32_8gpu_20260909.queue.json')
            subprocess.run(['bash', str(ROOT / 'start_training.sh')], cwd=ROOT, check=True)
            status('training_launcher_completed', dataset=evidence,
                   note='Inspect training .log for actual optimization progress; no automatic restart.')
            return 0
        except Exception as error:
            status('failed', error=f'{type(error).__name__}: {error}',
                   note='No training restart and no source-data deletion were performed.')
            raise


if __name__ == '__main__':
    raise SystemExit(main())
