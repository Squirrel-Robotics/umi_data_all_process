#!/usr/bin/env python3
"""Publish repaired right-eye data, verify it, and start a fresh 8-GPU run.

Every stage is fail-closed. Old data/checkpoints and other processes are untouched.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/dzq/openpi/.venv/bin/python'
OLD = Path('/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50')
DATASET = OLD.with_name(OLD.name + '_right_eye_v2')
ASSETS = Path('/mnt/data/dzq/openpi/data/assets')
ASSET_ID = 'umi_task_v2_x2_high_hand_pose_10hz_h50_right_eye_masked_v2'
PROMPT = 'Put the two objects into the box.'
SELECTED_SHA = '10529d85a309173ca01348bb092ca36278d8064737656b88994baa90152f4b2b'
OLD_CONTRACT_SHA = 'ffc68080ea965a2066877d86be7f022c45811aa8a2600872c1682b75d487aa41'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def status(phase: str, **details) -> None:
    path = ROOT / 'pipeline_status.json'
    temporary = ROOT / f'.pipeline_status.{os.getpid()}.tmp'
    value = {'status': phase, 'pipeline_pid': os.getpid(),
             'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **details}
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    os.replace(temporary, path)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def cpu_env():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1',
               PYTHONPATH='/home/dzq/openpi/src', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def validate_repaired_dataset():
    contract_file = DATASET / 'meta/umi_conversion.json'
    contract = read_json(contract_file)
    expected = {'schema': 'umi-folder-dual-hand-pose-lerobot', 'status': 'complete',
                'target': str(DATASET), 'repo_id': 'dzq/' + DATASET.name,
                'task': PROMPT, 'fps': 10, 'state_shape': [30], 'action_shape': [50, 30],
                'total_source_episodes': 85, 'total_output_episodes': 85, 'total_frames': 9995,
                'one_source_one_output_episode': True, 'source_snapshot_verified_before_publish': True}
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f'Dataset contract mismatch {key}: {contract.get(key)!r}')
    if digest(DATASET / 'meta/episode_selection.txt') != SELECTED_SHA:
        raise ValueError('Repaired data no longer contains the approved 85-episode list')
    if list(DATASET.parent.glob(f'.{DATASET.name}.building-*')):
        raise ValueError('Unfinished repair staging remains')
    repair = read_json(DATASET / 'meta/head_video_repair.json')
    if (repair.get('schema') != 'umi-head-video-right-eye-repair'
            or repair.get('status') != 'complete'
            or repair.get('source_dataset_contract_sha256') != OLD_CONTRACT_SHA
            or repair.get('source_dataset') != str(OLD)
            or repair.get('target_dataset') != str(DATASET)
            or len(repair.get('episodes', [])) != 85):
        raise ValueError('Repair provenance does not match this approved source dataset')
    return {'episodes': 85, 'frames': 9995, 'contract_sha256': digest(contract_file),
            'repair_audit_sha256': digest(DATASET / 'meta/head_video_repair.json')}


def main() -> int:
    with (ROOT / '.pipeline.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / 'pipeline_status.json').exists():
            raise FileExistsError('Pipeline status exists; inspect it instead of launching twice')
        try:
            if digest(OLD / 'meta/umi_conversion.json') != OLD_CONTRACT_SHA:
                raise ValueError('Original dataset contract changed')
            if digest(ROOT / 'selected_85_episodes.txt') != SELECTED_SHA:
                raise ValueError('The approved selection changed')
            if DATASET.exists() or (ASSETS / ASSET_ID).exists():
                raise FileExistsError('New dataset or normalization asset already exists')
            status('repairing_head_videos', target_dataset=str(DATASET),
                   source_dataset=str(OLD), original_files_preserved=True)
            repair_command = [PYTHON, str(ROOT / 'repair_head_videos.py'),
                              '--original-dataset', str(OLD), '--target', str(DATASET),
                              '--repo-id', 'dzq/' + DATASET.name, '--video-workers', '4',
                              '--confirm', 'REPAIR_HEAD_VIDEOS']
            with (ROOT / 'repair.log').open('x', encoding='utf-8') as output:
                subprocess.run(repair_command, cwd=ROOT, env=cpu_env(),
                               stdout=output, stderr=subprocess.STDOUT, check=True, timeout=6 * 3600)
            evidence = validate_repaired_dataset()
            status('normalizing', dataset=evidence)
            norm_command = [PYTHON, '/home/dzq/data_deal/training_tools/compute_masked_norm_stats.py',
                            '--dataset', str(DATASET), '--assets-base-dir', str(ASSETS),
                            '--asset-id', ASSET_ID, '--expected-fps', '10', '--expected-horizon', '50',
                            '--expected-task', PROMPT]
            with (ROOT / 'normalization.log').open('x', encoding='utf-8') as output:
                subprocess.run(norm_command, cwd=ROOT, env=cpu_env(), stdout=output,
                               stderr=subprocess.STDOUT, check=True, timeout=1800)
            if validate_repaired_dataset() != evidence:
                raise RuntimeError('Repaired dataset changed during normalization')
            status('cpu_validation_then_gpu_queue', dataset=evidence,
                   note='Queue releases only after real image/state/action/normalization validation.')
            subprocess.run(['bash', str(ROOT / 'start_training.sh')], cwd=ROOT, check=True)
            status('training_launcher_completed', dataset=evidence,
                   note='Training process claimed all eight GPUs; inspect training log for actual step progress.')
            return 0
        except Exception as error:
            status('failed', error=f'{type(error).__name__}: {error}',
                   note='No automatic retry, no original-data or checkpoint deletion.')
            raise


if __name__ == '__main__':
    raise SystemExit(main())
