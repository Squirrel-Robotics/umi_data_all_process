#!/usr/bin/env python3
"""Create a new LeRobot dataset with each wrist rotated 180 degrees, no swapping.

Lossless H.264 encoding preserves the rotated decoded YUV420 pixels exactly.
Every frame is checked with framemd5 and presentation timestamps are unchanged.
The source dataset is read-only; failed staging is retained for inspection.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import repair_head_videos as video
from publish_repaired_dataset import reserve_and_rename

HEAD = 'observation.images.head_rgb'
WRISTS = ('observation.images.left_wrist_rgb', 'observation.images.right_wrist_rgb')
ALLOWED_META = {'meta/umi_conversion.json', 'meta/episodes_stats.jsonl', 'meta/stats.json'}
HELPER_SHA = '4e9c18c75770b8892a8a8141c7249972291e3de6c7866cfa9ebd5d05920bcc12'
PUBLICATION_SHA = 'be828bb3f91cca40a21734ed9fbf96caeef38a76c9129b51368a527edee88b96'


def require(test, message):
    if not test:
        raise ValueError(message)


def digest(path):
    return video.sha256(Path(path))


def read_json(path):
    return video.json_read(Path(path))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def frame_hashes(path, rotate=False):
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-filter_threads', '1',
               '-i', str(path), '-map', '0:v:0', '-an']
    if rotate:
        command += ['-vf', 'hflip,vflip']
    command += ['-vsync', '0', '-pix_fmt', 'yuv420p', '-c:v', 'rawvideo', '-threads', '1',
                '-f', 'framemd5', 'pipe:1']
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=600)
    rows = [line.split(',') for line in result.stdout.splitlines() if line and not line.startswith('#')]
    require(bool(rows) and all(len(row) == 6 for row in rows), f'Invalid decoded frame hashes: {path}')
    return [row[-1].strip() for row in rows]


def frame_times(path):
    command = ['ffprobe', '-v', 'error', '-threads', '1', '-select_streams', 'v:0',
               '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=600)
    return [item['best_effort_timestamp_time'] for item in json.loads(result.stdout)['frames']]


def rotate_video(source, target):
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-n', '-threads', '2', '-filter_threads', '1',
               '-i', str(source), '-map', '0:v:0', '-an', '-vf', 'hflip,vflip', '-vsync', '0',
               '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-crf', '0',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
    subprocess.run(command, check=True, timeout=1200)
    return command


def process_one(source, staging, episode, key):
    record = episode['videos'][key]
    old_path, new_path = video.inside(source, record['path']), video.inside(staging, record['path'])
    video.check_record(source, record)
    require(record['pix_fmt'] == 'yuv420p', 'Lossless rotation requires existing YUV420P videos')
    command = rotate_video(old_path, new_path)
    output = video.probe(new_path, count=True)
    require((output['codec_name'], output['pix_fmt'], output['width'], output['height'],
             int(output['nb_read_packets'])) == ('h264', 'yuv420p', record['width'], record['height'], record['frames']),
            f'Output format/frame count changed: {new_path}')
    from fractions import Fraction
    require(Fraction(output['avg_frame_rate']) == record['fps'], f'FPS changed: {new_path}')
    reference, actual = frame_hashes(old_path, rotate=True), frame_hashes(new_path)
    require(len(reference) == record['frames'] and actual == reference,
            f'Not an exact 180-degree rotation of every decoded YUV frame: {new_path}')
    times = frame_times(old_path)
    require(len(times) == record['frames'] and times == frame_times(new_path), f'Timestamps changed: {new_path}')
    stats = video.head_stats(new_path, record['frames'], record['width'], record['height'])
    require(digest(old_path) == record['sha256'], f'Source changed during processing: {old_path}')
    new_record = {**record, 'size_bytes': new_path.stat().st_size, 'sha256': digest(new_path)}
    proof = {'episode_index': episode['episode_index'], 'source_episode_id': episode['source_episode_id'],
             'camera_key': key, 'source_video': record, 'output_video': new_record,
             'rotation_degrees': 180, 'swapped_left_right': False, 'encoding': 'libx264 crf=0 yuv420p lossless',
             'decoded_yuv_frames_exact': True, 'checked_frames': len(actual),
             'rotated_frame_hashes_sha256': canonical_sha(actual), 'frame_times_sha256': canonical_sha(times),
             'presentation_timestamps_unchanged': True, 'command': command}
    return proof, stats


def verify_dataset(target, source, *, decode=True):
    """Release gate; source proof and all unchanged files must still match."""
    import numpy as np
    video.NP = np
    target, source = Path(target).resolve(), Path(source).resolve()
    contract = read_json(target / 'meta/umi_conversion.json')
    old = read_json(source / 'meta/umi_conversion.json')
    report_path = target / 'meta/wrist_rotation.json'
    report = read_json(report_path)
    require(report['schema'] == 'umi-wrist-rotation' and report['status'] == 'complete', 'Incomplete rotation report')
    require(report['source_dataset'] == str(source) and report['target_dataset'] == contract['target'], 'Wrong provenance paths')
    require(report['source_contract_sha256'] == digest(source / 'meta/umi_conversion.json'), 'Source contract changed')
    require(contract['wrist_rotation'] == {'path': 'meta/wrist_rotation.json', 'sha256': digest(report_path),
                                         'source_contract_sha256': report['source_contract_sha256']}, 'Report hash mismatch')
    expected = copy.deepcopy(old)
    expected.update(target=contract['target'], repo_id=contract['repo_id'], wrist_rotation=contract['wrist_rotation'])
    expected['video_preprocessing']['wrist_rotation_degrees'] = {key: 180 for key in WRISTS}
    require(len(report['videos']) == len(old['episodes']) * 2, 'Missing wrist videos')
    stats = [json.loads(line) for line in (target / 'meta/episodes_stats.jsonl').read_text().splitlines() if line]
    original_stats = [json.loads(line) for line in (source / 'meta/episodes_stats.jsonl').read_text().splitlines() if line]
    require(len(stats) == len(original_stats) == len(old['episodes']), 'Statistics episode count mismatch')
    records = {(item['episode_index'], item['camera_key']): item for item in report['videos']}
    require(len(records) == len(report['videos']), 'Duplicate rotation records')
    aggregate = {key: [] for key in WRISTS}
    for index, episode in enumerate(old['episodes']):
        require(episode['episode_index'] == index and stats[index]['episode_index'] == index, 'Episode order changed')
        for key in WRISTS:
            proof = records[index, key]
            require(proof['source_video'] == episode['videos'][key] and proof['source_episode_id'] == episode['source_episode_id'], 'Source video changed/swapped')
            require(proof['rotation_degrees'] == 180 and proof['swapped_left_right'] is False, 'Wrong image operation')
            rec = proof['output_video']
            require(rec['path'] == proof['source_video']['path'], 'Output camera identity changed')
            video.check_record(source, proof['source_video']); video.check_record(target, rec)
            expected['episodes'][index]['videos'][key] = rec
            if decode:
                reference, actual = frame_hashes(source / rec['path'], rotate=True), frame_hashes(target / rec['path'])
                require(actual == reference and len(actual) == rec['frames'], f'Pixel rotation mismatch: {index}/{key}')
                require(canonical_sha(actual) == proof['rotated_frame_hashes_sha256'], 'Frame-hash proof changed')
                times = frame_times(source / rec['path'])
                require(times == frame_times(target / rec['path']) and canonical_sha(times) == proof['frame_times_sha256'], 'Frame times changed')
                recomputed = video.head_stats(target / rec['path'], rec['frames'], rec['width'], rec['height'])
                for field, value in recomputed.items():
                    np.testing.assert_allclose(stats[index]['stats'][key][field], value, atol=1e-12, rtol=1e-12,
                                               err_msg=f'Image statistics mismatch: {index}/{key}/{field}')
            current_stats = stats[index]['stats'][key]
            aggregate[key].append(current_stats)
        require({k:v for k,v in stats[index]['stats'].items() if k not in WRISTS}
                == {k:v for k,v in original_stats[index]['stats'].items() if k not in WRISTS}, 'Head/numeric statistics changed')
        if decode and ((index + 1) % 10 == 0 or index + 1 == len(old['episodes'])):
            print(f'wrist release gate {index + 1}/{len(old["episodes"])} episodes', flush=True)
    require(expected == contract, 'Changes outside approved wrist-only transformation')
    for item in report['preserved_files']:
        p = video.inside(target, item['path']); original = video.inside(source, item['path'])
        require(digest(p) == digest(original) == item['sha256'], f'Preserved file changed: {p}')
    original_files = {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}
    wrist_paths = {e['videos'][k]['path'] for e in old['episodes'] for k in WRISTS}
    require({item['path'] for item in report['preserved_files']} == original_files - wrist_paths - ALLOWED_META,
            'Preservation inventory is incomplete')
    total_stats, old_total = read_json(target / 'meta/stats.json'), read_json(source / 'meta/stats.json')
    require({k:v for k,v in total_stats.items() if k not in WRISTS} == {k:v for k,v in old_total.items() if k not in WRISTS}, 'Non-wrist aggregate changed')
    for key in WRISTS:
        for field, value in video.aggregate_head(aggregate[key]).items():
            np.testing.assert_allclose(total_stats[key][field], value, atol=1e-12, rtol=1e-12)
    for name, sha in report['generator_files'].items():
        require(digest(target / 'meta' / name) == sha, f'Generator provenance changed: {name}')
    return {'status': 'complete', 'episodes': len(old['episodes']), 'frames': old['total_frames'],
            'wrist_videos': len(records), 'rotation_degrees_each': 180, 'swapped_left_right': False,
            'all_yuv_frames_verified': decode, 'preserved_file_count': len(report['preserved_files']),
            'source_contract_sha256': report['source_contract_sha256'], 'rotation_report_sha256': digest(report_path)}


def convert(args):
    import cv2
    import numpy as np
    cv2.setNumThreads(1); video.NP = np
    require(args.source.is_absolute() and args.target.is_absolute(), 'Use absolute dataset paths')
    source, target = args.source.resolve(), args.target.resolve()
    require(not args.source.is_symlink() and not args.target.is_symlink(), 'Dataset root symlink rejected')
    require(source.is_dir() and not target.exists(), 'Source missing or target already exists')
    require(source != target and not target.is_relative_to(source) and not source.is_relative_to(target), 'Overlapping paths')
    require(digest(source / 'meta/umi_conversion.json') == args.expected_source_sha256, 'Unexpected source contract')
    require(digest(Path(video.__file__)) == HELPER_SHA, 'Video helper changed')
    require(digest(Path(__file__).with_name('publish_repaired_dataset.py')) == PUBLICATION_SHA, 'Publication helper changed')
    old, info = read_json(source / 'meta/umi_conversion.json'), read_json(source / 'meta/info.json')
    require(old['status'] == 'complete' and old['target'] == str(source), 'Incomplete/mismatched source')
    require(old['total_episodes'] == info['total_episodes'] == len(old['episodes']), 'Episode count mismatch')
    require(old['total_frames'] == sum(e['output_rows'] for e in old['episodes']), 'Frame count mismatch')
    for episode in old['episodes']:
        video.check_record(source, episode['parquet'])
        for rec in episode['videos'].values(): video.check_record(source, rec)
    lock = target.parent / f'.{target.name}.rotation.lock'
    with lock.open('a+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not target.exists(), 'Target appeared before lock')
        staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.rotating-', dir=target.parent))
        print(f'staging={staging}', flush=True)
        wrists = {e['videos'][k]['path'] for e in old['episodes'] for k in WRISTS}
        preserved = video.copy_original(source, staging, wrists | ALLOWED_META)
        new = copy.deepcopy(old); new.update(target=str(target), repo_id='dzq/' + target.name)
        new['video_preprocessing']['wrist_rotation_degrees'] = {key: 180 for key in WRISTS}
        stat_rows = [json.loads(line) for line in (source / 'meta/episodes_stats.jsonl').read_text().splitlines() if line]
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_one, source, staging, e, k) for e in old['episodes'] for k in WRISTS]
            try:
                for completed, future in enumerate(as_completed(futures), 1):
                    proof, stats = future.result(); results.append(proof)
                    i, k = proof['episode_index'], proof['camera_key']
                    new['episodes'][i]['videos'][k] = proof['output_video']; stat_rows[i]['stats'][k] = stats
                    print(f'rotated+lossless-verified {completed}/{len(futures)}: episode={i} {k}', flush=True)
            except BaseException:
                for future in futures: future.cancel()
                raise
        all_stats = read_json(source / 'meta/stats.json')
        for key in WRISTS: all_stats[key] = video.aggregate_head([e['stats'][key] for e in stat_rows])
        video.json_write(staging / 'meta/stats.json', all_stats)
        with (staging / 'meta/episodes_stats.jsonl').open('x') as out:
            for row in stat_rows: out.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        generator_files = {}
        for path in (Path(__file__), Path(video.__file__), Path(__file__).with_name('publish_repaired_dataset.py')):
            dst = staging / 'meta' / path.name
            if dst.exists(): require(digest(dst) == digest(path), f'Conflicting provenance file {dst}')
            else: shutil.copy2(path, dst)
            generator_files[path.name] = digest(path)
        report = {'schema': 'umi-wrist-rotation', 'schema_version': 1, 'status': 'complete',
                  'source_dataset': str(source), 'target_dataset': str(target),
                  'source_contract_sha256': args.expected_source_sha256,
                  'rotation_degrees': 180, 'camera_keys': list(WRISTS), 'swapped_left_right': False,
                  'total_episodes': old['total_episodes'], 'total_frames': old['total_frames'],
                  'numeric_data_head_video_alignment_preserved': True,
                  'head_repair_report_describes_parent_dataset': str(source),
                  'generator_files': generator_files,
                  'preserved_files': [{'path': p, 'sha256': sha} for p, sha in sorted(preserved.items())],
                  'videos': sorted(results, key=lambda item: (item['episode_index'], item['camera_key']))}
        video.json_write(staging / 'meta/wrist_rotation.json', report)
        new['wrist_rotation'] = {'path': 'meta/wrist_rotation.json', 'sha256': digest(staging / 'meta/wrist_rotation.json'),
                                 'source_contract_sha256': args.expected_source_sha256}
        video.json_write(staging / 'meta/umi_conversion.json', new)
        verify_dataset(staging, source, decode=False)
        require(digest(source / 'meta/umi_conversion.json') == args.expected_source_sha256, 'Source changed before publish')
        result = reserve_and_rename(staging, target)
        print(json.dumps({'status': 'complete', 'dataset': str(target), 'videos': len(results),
                          'contract_sha256': digest(target / 'meta/umi_conversion.json'), **result}), flush=True)


def self_test():
    with tempfile.TemporaryDirectory(prefix='wrist180-test-') as directory:
        root = Path(directory); source, target = root/'source.mp4', root/'rotated.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x48:rate=10:duration=0.6',
                        '-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(source)],check=True,timeout=60)
        rotate_video(source, target)
        require(frame_hashes(source, True) == frame_hashes(target), 'Rotation not lossless')
        require(frame_hashes(source) != frame_hashes(target), 'Test cannot distinguish the rotation')
        require(frame_times(source) == frame_times(target), 'Timing changed')
        double = root/'double.mp4';rotate_video(target,double)
        require(frame_hashes(source) == frame_hashes(double), 'Double rotation not identity')
        from publish_repaired_dataset import self_test as publication_test
        publication_test()
    print('SELF_TEST_OK all-frame rotation, unchanged timestamps, double-rotation identity, safe publication')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--target',type=Path)
    parser.add_argument('--expected-source-sha256');parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--confirm',choices=['CREATE_ROTATED_COPY']);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test: return self_test()
    if not all((args.source,args.target,args.expected_source_sha256,args.confirm)) or not 1 <= args.workers <= 8:
        parser.error('source, target, expected-source-sha256, confirm and workers 1..8 required')
    convert(args)


if __name__=='__main__': main()
