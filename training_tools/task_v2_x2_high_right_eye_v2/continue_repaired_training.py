#!/usr/bin/env python3
"""Continue only the reviewed NFS renameat2/EINVAL publication recovery.

The original pipeline and failed status remain untouched. A separate status and
the original pipeline lock prevent overlapping or repeated continuations. The
unchanged training launcher guards its checkpoint/log/receipt/smoke paths and
performs CPU validation before claiming GPUs; this script never bypasses it.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import run_pipeline as pipeline


EXPECTED_PUBLISHER_SHA256 = 'be828bb3f91cca40a21734ed9fbf96caeef38a76c9129b51368a527edee88b96'
PUBLICATION_METHOD = 'exclusive_empty_directory_reservation_then_posix_atomic_rename'


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def status(phase: str, **details) -> None:
    target = pipeline.ROOT / 'continuation_status.json'
    temporary = pipeline.ROOT / f'.continuation_status.{os.getpid()}.tmp'
    value = {'status': phase, 'continuation_pid': os.getpid(),
             'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **details}
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, target)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def validate_known_failure() -> dict:
    """Accept only the exact terminal publication error for this target."""
    original_status_path = pipeline.ROOT / 'pipeline_status.json'
    original_status = pipeline.read_json(original_status_path)
    require(original_status.get('status') == 'failed',
            'Original pipeline must already be failed; a running/completed pipeline cannot be continued')
    original_error = original_status.get('error', '')
    require(isinstance(original_error, str) and 'CalledProcessError' in original_error
            and 'repair_head_videos.py' in original_error,
            'Original failure must belong to the repair subprocess, not normalization or training')
    repair_log_path = pipeline.ROOT / 'repair.log'
    text = repair_log_path.read_text(encoding='utf-8')
    header = 'Traceback (most recent call last):'
    traceback = text.rsplit(header, 1)[-1]
    lines = [line.strip() for line in traceback.splitlines() if line.strip()]
    terminal = f"OSError: [Errno 22] Invalid argument: '{pipeline.DATASET}'"
    require(header in text and bool(lines) and lines[-1] == terminal
            and 'in publish_no_replace' in traceback
            and 'publish_no_replace(staging, target)' in traceback
            and 'repair_head_videos.py' in traceback,
            'repair.log does not end with the exact approved renameat2 publication EINVAL failure')
    return {'original_pipeline_status_sha256': pipeline.digest(original_status_path),
            'repair_log_sha256': pipeline.digest(repair_log_path),
            'original_pipeline_pid': original_status.get('pipeline_pid')}


def validate_recovery() -> dict:
    failure = validate_known_failure()
    require(pipeline.digest(pipeline.OLD / 'meta/umi_conversion.json') == pipeline.OLD_CONTRACT_SHA,
            'Original dataset contract changed')
    require(pipeline.digest(pipeline.ROOT / 'selected_85_episodes.txt') == pipeline.SELECTED_SHA,
            'The approved selection changed')
    require(pipeline.DATASET.is_dir() and not pipeline.DATASET.is_symlink(),
            'The repaired dataset must exist as a real published target directory')
    dataset = pipeline.validate_repaired_dataset()
    marker_path = pipeline.DATASET / 'meta/head_video_publication.json'
    require(marker_path.is_file() and not marker_path.is_symlink(),
            'A publication marker at the final target is required; staging markers are not publication')
    marker = pipeline.read_json(marker_path)
    expected = {
        'schema': 'umi-head-video-publication', 'schema_version': 1,
        'status': 'complete', 'valid_only_at_target_dataset': True,
        'publication_method': PUBLICATION_METHOD,
        'source_dataset': str(pipeline.OLD), 'target_dataset': str(pipeline.DATASET),
        'source_dataset_contract_sha256': pipeline.OLD_CONTRACT_SHA,
        'repair_audit_sha256': dataset['repair_audit_sha256'],
        'published_contract_sha256': dataset['contract_sha256'],
        'publisher_script_sha256': EXPECTED_PUBLISHER_SHA256,
        'original_report_and_contract_unchanged': True,
        'total_episodes': 85, 'total_frames': 9995,
    }
    for key, value in expected.items():
        require(marker.get(key) == value,
                f'Publication marker mismatch {key}: {marker.get(key)!r}')
    for publisher in (pipeline.ROOT / 'publish_repaired_dataset.py',
                      pipeline.DATASET / 'meta/publish_repaired_dataset.py'):
        require(publisher.is_file() and not publisher.is_symlink(),
                f'A real reviewed publisher script is required: {publisher}')
        require(pipeline.digest(publisher) == EXPECTED_PUBLISHER_SHA256,
                f'Publisher script hash changed: {publisher}')
    return {**dataset, **failure,
            'publication_marker_sha256': pipeline.digest(marker_path),
            'publisher_script_sha256': EXPECTED_PUBLISHER_SHA256}


def verify_normalization(evidence: dict) -> None:
    asset = pipeline.ASSETS / pipeline.ASSET_ID
    audit = pipeline.read_json(asset / 'norm_stats_audit.json')
    require(audit.get('status') == 'complete'
            and audit.get('dataset_contract_sha256') == evidence['contract_sha256']
            and audit.get('norm_stats_sha256') == pipeline.digest(asset / 'norm_stats.json'),
            'Normalization did not produce matching complete audited stats')


def main() -> int:
    # This is the SAME lock held by run_pipeline.py throughout its lifetime.
    # Acquiring it non-blockingly also excludes a still-exiting failed pipeline.
    with (pipeline.ROOT / '.pipeline.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (pipeline.ROOT / 'continuation_status.json').exists():
            raise FileExistsError('Continuation status exists; inspect it instead of starting again')
        try:
            evidence = validate_recovery()
            asset = pipeline.ASSETS / pipeline.ASSET_ID
            norm_log = pipeline.ROOT / 'normalization.log'
            if asset.exists() or norm_log.exists():
                raise FileExistsError('Normalization asset/log already exists; no implicit retry is allowed')
            for script in ('start_training.sh', 'launch_training.py', 'smoke_training.py'):
                require((pipeline.ROOT / script).is_file(), f'Missing guarded training entry point: {script}')
            status('normalizing_after_verified_publication', evidence=evidence,
                   original_pipeline_status_preserved=True)
            command = [pipeline.PYTHON, '/home/dzq/data_deal/training_tools/compute_masked_norm_stats.py',
                       '--dataset', str(pipeline.DATASET), '--assets-base-dir', str(pipeline.ASSETS),
                       '--asset-id', pipeline.ASSET_ID, '--expected-fps', '10',
                       '--expected-horizon', '50', '--expected-task', pipeline.PROMPT]
            with norm_log.open('x', encoding='utf-8') as output:
                subprocess.run(command, cwd=pipeline.ROOT, env=pipeline.cpu_env(), stdout=output,
                               stderr=subprocess.STDOUT, check=True, timeout=1800)
            if validate_recovery() != evidence:
                raise RuntimeError('Dataset, publication evidence, or original failure changed during normalization')
            verify_normalization(evidence)
            status('cpu_validation_then_gpu_queue', evidence=evidence,
                   note='Unchanged launcher enforces fresh logs/checkpoints, CPU smoke and GPU ownership')
            subprocess.run(['bash', str(pipeline.ROOT / 'start_training.sh')],
                           cwd=pipeline.ROOT, check=True)
            status('training_launcher_completed', evidence=evidence,
                   note='Inspect training log for optimization progress; no automatic retry')
            return 0
        except Exception as error:
            status('failed', error=f'{type(error).__name__}: {error}',
                   note='Original failed status, data and checkpoints are preserved; no automatic retry')
            raise


if __name__ == '__main__':
    raise SystemExit(main())
