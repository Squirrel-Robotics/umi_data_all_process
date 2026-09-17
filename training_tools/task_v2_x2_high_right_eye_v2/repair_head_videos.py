#!/usr/bin/env python3
"""Re-encode E6 right-eye videos without changing any existing sample selection.

Only the new target is written. Original data, numeric conversion provenance,
Parquet files, wrist videos, and original checkpoints are never modified.
Image statistics use the exact same 64 evenly spaced frames as the converter;
uint8 channel histograms avoid its expensive float64 whole-video reduction.
"""
from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from typing import Any

HEAD = "observation.images.head_rgb"
WRISTS = ("observation.images.left_wrist_rgb", "observation.images.right_wrist_rgb")
REQUIRED_CSVS = {
    "camera/hand_pose.csv", "camera/e6_rgb_stream_metainfo.csv", "sync/e6_rgb_timing.csv",
    "extensions/customer_camera/cam0/frames.csv", "extensions/customer_camera/cam1/frames.csv",
    "left_rx_packets.csv", "right_rx_packets.csv",
}
NP = None


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_read(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def json_write(path: Path, value: Any) -> None:
    # New staging tree only; atomic replace makes an interrupted report readable.
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def inside(root: Path, relative: str) -> Path:
    p = Path(relative)
    require(not p.is_absolute() and ".." not in p.parts and bool(p.parts), f"unsafe relative path: {relative}")
    candidate = root / p
    require(candidate.resolve().is_relative_to(root.resolve()), f"path escapes root: {candidate}")
    # Reject symlinks, including ancestor symlinks, to make copying unambiguous.
    for part in (candidate, *candidate.parents):
        if part == root.parent:
            break
        require(not part.is_symlink(), f"symlink not supported: {part}")
    return candidate


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def selected_rows(path: Path) -> tuple[list[int], list[dict[str, str]]]:
    rows = csv_rows(path)
    require(bool(rows), f"empty saved alignment: {path}")
    require({"is_output_anchor", "e6_source_row_index", "e6_frame_id", "e6_global_ts_ns"} <= rows[0].keys(),
            f"missing saved alignment fields: {path}")
    require(all(r["is_output_anchor"] in ("0", "1") for r in rows), f"bad anchor flag: {path}")
    anchors = [r for r in rows if r["is_output_anchor"] == "1"]
    indices = [int(r["e6_source_row_index"]) for r in anchors]
    require(bool(indices) and min(indices) >= 0 and all(a < b for a, b in zip(indices, indices[1:])),
            f"saved source indices are not positive monotonic: {path}")
    return indices, anchors


def index_hash(indices: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, indices)).encode("utf-8")).hexdigest()


def right_crop(width: int, height: int) -> list[int]:
    require(width > 0 and height > 0 and width % 4 == 0 and height % 2 == 0,
            f"stereo dimensions cannot be split safely for yuv420: {width}x{height}")
    return [width // 2, 0, width // 2, height]


def probe(path: Path, *, raw: bool = False, count: bool = False) -> dict[str, Any]:
    cmd = ["ffprobe", "-v", "error", "-threads", "4"]
    if raw:
        cmd += ["-f", "hevc"]
    if count:
        cmd += ["-count_packets"]
    cmd += ["-select_streams", "v:0", "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_packets,pix_fmt", "-of", "json", str(path)]
    value = json.loads(subprocess.check_output(cmd, text=True, timeout=120))
    require(len(value.get("streams", [])) == 1, f"need one video stream: {path}")
    return value["streams"][0]


def check_record(root: Path, record: dict[str, Any]) -> str:
    path = inside(root, record["path"])
    require(path.is_file(), f"missing file: {path}")
    require(path.stat().st_size == record["size_bytes"], f"file size changed: {path}")
    digest = sha256(path)
    require(digest == record["sha256"], f"SHA256 mismatch: {path}")
    return digest


def check_csv_hashes(episode: dict[str, Any]) -> None:
    base = Path(episode["source_episode"])
    hashes = episode["source_hashes"]
    require(REQUIRED_CSVS <= hashes.keys(), f"incomplete source CSV hash contract: {base}")
    for rel, expected in hashes.items():
        # The original converter records these two keys without their serial/ prefix.
        actual_rel = f"serial/{rel}" if rel in ("left_rx_packets.csv", "right_rx_packets.csv") else rel
        path = inside(base, actual_rel)
        require(sha256(path) == expected, f"source CSV changed since numeric conversion: {path}")


def channel_histogram_stats(hist: Any, sampled_frames: int) -> dict[str, Any]:
    np = NP
    require(hist.shape == (3, 256), "invalid channel histogram shape")
    totals = hist.sum(axis=1)
    require(sampled_frames > 0 and bool(np.all(totals > 0)) and bool(np.all(totals == totals[0])),
            "invalid channel histogram counts")
    level = np.arange(256, dtype=np.float64) / 255.0
    mean = (hist * level).sum(axis=1) / totals
    # Centered second moment avoids numerical cancellation for uniform images.
    variance = (hist * (level[None, :] - mean[:, None]) ** 2).sum(axis=1) / totals
    return {
        "min": np.array([level[np.flatnonzero(x)[0]] for x in hist])[:, None, None].tolist(),
        "max": np.array([level[np.flatnonzero(x)[-1]] for x in hist])[:, None, None].tolist(),
        "mean": mean[:, None, None].tolist(),
        "std": np.sqrt(variance)[:, None, None].tolist(),
        "count": [sampled_frames],
    }


def head_stats(path: Path, expected_frames: int, width: int, height: int) -> dict[str, Any]:
    import cv2
    np = NP
    wanted = set(np.linspace(0, expected_frames - 1, min(64, expected_frames)).round().astype(int).tolist())
    hist = np.zeros((3, 256), dtype=np.int64)
    cap = cv2.VideoCapture(str(path))
    require(cap.isOpened(), f"cannot decode: {path}")
    index = sampled = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            require(bgr.shape == (height, width, 3), f"decoded dimensions changed: {path}")
            if index in wanted:
                # cv2 uses BGR. Histograms in RGB exactly match converter cvtColor.
                for rgb_channel, bgr_channel in enumerate((2, 1, 0)):
                    hist[rgb_channel] += np.bincount(bgr[:, :, bgr_channel].ravel(), minlength=256)
                sampled += 1
            index += 1
    finally:
        cap.release()
    require(index == expected_frames and sampled == len(wanted),
            f"full decode count mismatch: {path}: {index}/{expected_frames}, sampled={sampled}")
    return channel_histogram_stats(hist, sampled)


def aggregate_head(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    # Same parallel-variance formula as installed LeRobot aggregate_feature_stats.
    np = NP
    values = [{k: np.asarray(v) for k, v in s.items()} for s in stats_list]
    means = np.stack([s["mean"] for s in values])
    variances = np.stack([s["std"] ** 2 for s in values])
    counts = np.stack([s["count"] for s in values])
    total_count = counts.sum(axis=0)
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, -1)
    mean = (means * counts).sum(axis=0) / total_count
    variance = ((variances + (means - mean) ** 2) * counts).sum(axis=0) / total_count
    return {"min": np.min(np.stack([s["min"] for s in values]), axis=0).tolist(),
            "max": np.max(np.stack([s["max"] for s in values]), axis=0).tolist(),
            "mean": mean.tolist(), "std": np.sqrt(variance).tolist(), "count": total_count.tolist()}


def preflight(original: Path, contract: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    require(contract["status"] == "complete", "original dataset is not complete")
    require(contract["schema"] == "umi-folder-dual-hand-pose-lerobot" and contract["schema_version"] == 1,
            "unexpected original conversion schema")
    require(Path(contract["target"]).resolve() == original, "original contract target mismatch")
    require(contract["fps"] == info["fps"] == 10 and contract["action_shape"] == [50, 30],
            "this repair expects the approved 10 Hz / H50 recipe")
    require(info["codebase_version"] == "v2.1" and info["features"][HEAD]["shape"] == [480, 640, 3],
            "unexpected original LeRobot video format")
    require(contract["video_preprocessing"]["resize_mode"] == "letterbox", "unexpected original resize mode")
    eps = contract["episodes"]
    ids = [e["source_episode_id"] for e in eps]
    require(len(ids) == len(set(ids)) == contract["total_episodes"] == info["total_episodes"], "episode count mismatch")
    require(ids == contract["episode_selection"]["selected_episode_ids"], "allowlist order mismatch")
    require(contract["one_source_one_output_episode"] is True, "repair does not accept split source episodes")
    require(hashlib.sha256("\n".join(ids).encode()).hexdigest() == contract["source_episode_ids_sha256"],
            "source ID digest mismatch")
    require(sum(e["output_rows"] for e in eps) == contract["total_frames"] == info["total_frames"], "frame count mismatch")
    selection = contract["episode_selection"]
    require(sha256(inside(original, selection["dataset_copy_path"])) == selection["episode_list_sha256"],
            "allowlist copy SHA256 mismatch")
    snapshot = copy.deepcopy(contract["source_snapshot"])
    snapshot_sha = snapshot.pop("sha256")
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    require(hashlib.sha256(canonical.encode()).hexdigest() == snapshot_sha, "original source snapshot corrupt")
    raw_size_by_path = {r["path"]: r["size_bytes"] for r in snapshot["required_files"]}
    plans = []
    for i, e in enumerate(eps):
        require(e["episode_index"] == i and e["source_episode_id"] == e["output_episode_id"], "episode identity mismatch")
        source = Path(e["source_episode"])
        require(source == Path(contract["source"]) / e["source_episode_id"], "source path/ID mismatch")
        check_csv_hashes(e)
        for rec in (e["parquet"], *e["videos"].values()):
            check_record(original, rec)
        grid_rel = f"meta/alignment/episode_{i:06d}/alignment_output_grid.csv"
        grid = inside(original, grid_rel)
        indices, anchors = selected_rows(grid)
        require(len(indices) == e["output_rows"] == e["videos"][HEAD]["frames"], "saved anchor count mismatch")
        require(indices[0] == e["output_first_e6_source_row"] and indices[-1] == e["output_last_e6_source_row"],
                "saved anchor endpoints mismatch")
        metainfo = csv_rows(source / "camera/e6_rgb_stream_metainfo.csv")
        timing = csv_rows(source / "sync/e6_rgb_timing.csv")
        require(len(metainfo) == len(timing) == e["e6_rgb_rows"], "raw timing/metadata count mismatch")
        for index, anchor in zip(indices, anchors):
            require(index < len(metainfo), "source row exceeds raw metadata")
            a, b = metainfo[index], timing[index]
            require(int(b["source_row_index"]) == index, "timing CSV row order changed")
            require(a["frame_id"] == b["frame_id"] == anchor["e6_frame_id"], "frame ID mismatch")
            require(b["mapped_5b_monotonic_ns"] == anchor["e6_global_ts_ns"], "saved timestamp mismatch")
            require(a["e6_mid_exposure_realtime_ns"] == anchor["hand_pose_timestamp_ns"], "pose timestamp mismatch")
        raw = inside(source, "camera/e6_rgb.h265")
        require(raw.stat().st_size == raw_size_by_path[f"{e['source_episode_id']}/camera/e6_rgb.h265"],
                f"raw stream size changed since conversion: {raw}")
        raw_hash = sha256(raw)
        raw_probe = probe(raw, raw=True)
        crop = right_crop(int(raw_probe["width"]), int(raw_probe["height"]))
        declaration_path = inside(source, "meta/e6_stream_session.json")
        declared = json_read(declaration_path)["video"]
        source_declaration = {
            "path": "meta/e6_stream_session.json", "sha256": sha256(declaration_path),
            "width": int(declared["width"]), "height": int(declared["height"]),
            "matches_decoded_dimensions": (int(declared["width"]), int(declared["height"]))
                == (int(raw_probe["width"]), int(raw_probe["height"])),
            "used_for_crop": False,
        }
        plans.append({"episode_index": i, "source_episode_id": e["source_episode_id"], "indices": indices,
                      "raw_path": str(raw), "raw_sha256": raw_hash, "input_probe": raw_probe,
                      "source_metadata_declaration": source_declaration,
                      "crop_xywh": crop, "alignment_output_grid_path": grid_rel,
                      "alignment_output_grid_sha256": sha256(grid), "old_episode": e})
        if (i + 1) % 10 == 0 or i + 1 == len(eps):
            print(f"preflight verified {i + 1}/{len(eps)} episodes (CSV hashes, data, videos, source rows)", flush=True)
    return plans


def encode_one(plan: dict[str, Any], staging: Path) -> dict[str, Any]:
    raw = Path(plan["raw_path"])
    require(sha256(raw) == plan["raw_sha256"], f"raw changed before encoding: {raw}")
    old = plan["old_episode"]
    rel = old["videos"][HEAD]["path"]
    target = inside(staging, rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), f"refusing to overwrite staged video: {target}")
    x, y, width, height = plan["crop_xywh"]
    select = "+".join(f"eq(n\\,{n})" for n in plan["indices"])
    vf = (f"select='{select}',crop={width}:{height}:{x}:{y},setpts=N/(10*TB),"
          "scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-threads", "4",
           "-filter_threads", "1", "-f", "hevc", "-i", str(raw), "-an", "-vf", vf,
           "-c:v", "libx264", "-threads", "4", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-r", "10", "-vsync", "cfr", "-movflags", "+faststart", str(target)]
    subprocess.run(cmd, check=True, timeout=1800)
    output = probe(target, count=True)
    require((output["width"], output["height"]) == (640, 480), f"encoded dimensions invalid: {target}")
    require(Fraction(output["avg_frame_rate"]) == 10 and int(output["nb_read_packets"]) == len(plan["indices"]),
            f"encoded rate/count invalid: {target}: {output}")
    require(output["codec_name"] == "h264" and output["pix_fmt"] == "yuv420p", f"encoded format invalid: {target}")
    stats = head_stats(target, len(plan["indices"]), 640, 480)
    require(sha256(raw) == plan["raw_sha256"], f"raw changed during encoding: {raw}")
    video = {"path": rel, "frames": len(plan["indices"]), "fps": 10, "width": 640, "height": 480,
             "codec": "h264", "pix_fmt": "yuv420p", "size_bytes": target.stat().st_size, "sha256": sha256(target)}
    return {"episode_index": plan["episode_index"], "video": video, "stats": stats, "command": cmd}


def copy_original(original: Path, staging: Path, excluded: set[str]) -> dict[str, str]:
    digests = {}
    for path in sorted(original.rglob("*")):
        rel = path.relative_to(original).as_posix()
        require(not path.is_symlink(), f"original dataset symlink not supported: {path}")
        dst = staging / rel
        if path.is_dir():
            dst.mkdir(exist_ok=True)
        elif path.is_file() and rel not in excluded:
            shutil.copy2(path, dst)
            original_digest = sha256(path)
            require(sha256(dst) == original_digest, f"copy hash mismatch: {rel}")
            require((path.stat().st_dev, path.stat().st_ino) != (dst.stat().st_dev, dst.stat().st_ino),
                    f"hardlink prohibited: {rel}")
            digests[rel] = original_digest
        elif not path.is_file():
            raise ValueError(f"unexpected dataset node: {path}")
    return digests


def publish_no_replace(staging: Path, target: Path) -> None:
    # renameat2(RENAME_NOREPLACE) also closes the existence-check race.
    require(sys.platform.startswith("linux"), "atomic publication requires Linux renameat2")
    libc = ctypes.CDLL(None, use_errno=True)
    fn = libc.renameat2
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    if fn(-100, os.fsencode(staging), -100, os.fsencode(target), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))
    parent_fd = os.open(target.parent, os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def repair(args: argparse.Namespace) -> None:
    original, target = args.original_dataset.resolve(), args.target.resolve()
    require(args.original_dataset.is_absolute() and args.target.is_absolute(), "dataset paths must be absolute")
    require(original.is_dir() and not args.original_dataset.is_symlink(), "original dataset missing or is a symlink")
    require(not args.target.is_symlink() and not target.exists(), f"target exists: {target}")
    require(not target.is_relative_to(original) and not original.is_relative_to(target), "overlapping dataset roots")
    require(target.parent.is_dir(), "target parent must already exist")
    lock_path = target.parent / f".{target.name}.repair.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        os.close(lock_fd)
        raise RuntimeError(f"repair already active: {lock_path}") from e
    staging = None
    try:
        require(not target.exists(), f"target exists: {target}")
        contract_path, info_path = original / "meta/umi_conversion.json", original / "meta/info.json"
        contract_sha, info_sha = sha256(contract_path), sha256(info_path)
        contract, info = json_read(contract_path), json_read(info_path)
        plans = preflight(original, contract, info)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.repair-", dir=target.parent))
        print(f"staging={staging}", flush=True)
        excluded = {e["videos"][HEAD]["path"] for e in contract["episodes"]}
        copy_digests = copy_original(original, staging, excluded)
        json_write(staging / "meta/head_video_repair.pending.json", {
            "status": "encoding", "source_dataset": str(original), "target_dataset": str(target),
            "source_dataset_contract_sha256": contract_sha,
        })
        results = {}
        with ThreadPoolExecutor(max_workers=args.video_workers) as pool:
            futures = {pool.submit(encode_one, p, staging): p for p in plans}
            try:
                for future in as_completed(futures):
                    result = future.result()
                    results[result["episode_index"]] = result
                    print(f"repaired head videos + statistics {len(results)}/{len(plans)}: "
                          f"{futures[future]['source_episode_id']}", flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        episode_stats_path = staging / "meta/episodes_stats.jsonl"
        episode_stats = [json.loads(line) for line in episode_stats_path.read_text().splitlines() if line.strip()]
        require([r["episode_index"] for r in episode_stats] == list(range(len(plans))), "episode stats order mismatch")
        old_episode_stats = copy.deepcopy(episode_stats)
        for r in episode_stats:
            r["stats"][HEAD] = results[r["episode_index"]]["stats"]
        # Rewrite only the head entry; numeric and wrist statistics are unchanged values.
        with episode_stats_path.open("w", encoding="utf-8") as f:
            for r in episode_stats:
                f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
        stats = json_read(staging / "meta/stats.json")
        old_stats = copy.deepcopy(stats)
        stats[HEAD] = aggregate_head([r["stats"][HEAD] for r in episode_stats])
        json_write(staging / "meta/stats.json", stats)
        report_eps = []
        for p in plans:
            i = p["episode_index"]
            e = contract["episodes"][i]
            preserved = []
            for rec in (e["parquet"], *(e["videos"][key] for key in WRISTS)):
                rel = rec["path"]
                source_sha = check_record(original, rec)
                new_sha = check_record(staging, rec)
                preserved.append({"path": rel, "sha256": new_sha, "source_sha256": source_sha,
                                  "size_bytes": rec["size_bytes"]})
            check_csv_hashes(e)
            require(sha256(Path(p["raw_path"])) == p["raw_sha256"], "raw hash changed before publication")
            declaration = p["source_metadata_declaration"]
            require(sha256(inside(Path(e["source_episode"]), declaration["path"])) == declaration["sha256"],
                    "source dimension declaration changed during repair")
            check_record(original, e["videos"][HEAD])
            e["videos"][HEAD] = results[i]["video"]
            report_eps.append({
                "episode_index": i, "source_episode_id": p["source_episode_id"], "input_probe": p["input_probe"],
                "crop_xywh": p["crop_xywh"], "source_raw_path": p["raw_path"],
                "source_metadata_declaration": declaration,
                "head_video": {**results[i]["video"], "source_raw_sha256": p["raw_sha256"]},
                "preserved_files": preserved, "frame_selection_origin": "existing_alignment_output_grid.is_output_anchor",
                "alignment_output_grid_path": p["alignment_output_grid_path"],
                "alignment_output_grid_sha256": p["alignment_output_grid_sha256"],
                "source_row_indices": p["indices"], "source_row_indices_sha256": index_hash(p["indices"]),
            })
        modified = {"meta/umi_conversion.json", "meta/stats.json", "meta/episodes_stats.jsonl"}
        for rel, digest in copy_digests.items():
            require(sha256(original / rel) == digest, f"original dataset changed during repair: {rel}")
            if rel not in modified:
                require(sha256(staging / rel) == digest, f"preserved artifact changed: {rel}")
        require(sha256(contract_path) == contract_sha and sha256(info_path) == info_sha, "original metadata changed")
        require({k: v for k, v in old_stats.items() if k != HEAD} == {k: v for k, v in stats.items() if k != HEAD},
                "numeric or wrist aggregate stats changed")
        for before, after in zip(old_episode_stats, episode_stats):
            require({k: v for k, v in before["stats"].items() if k != HEAD}
                    == {k: v for k, v in after["stats"].items() if k != HEAD}, "numeric or wrist episode stats changed")
        script_copy = staging / "meta/repair_head_videos.py"
        require(not script_copy.exists(), "repair provenance already exists in original")
        shutil.copy2(Path(__file__).resolve(), script_copy)
        report = {
            "schema": "umi-head-video-right-eye-repair", "schema_version": 1, "status": "complete",
            "source_dataset": str(original), "target_dataset": str(target), "repo_id": args.repo_id,
            "source_dataset_contract_sha256": contract_sha, "source_dataset_info_sha256": info_sha,
            "total_episodes": len(plans), "total_frames": contract["total_frames"], "fps": 10,
            "head_camera_eye": "right", "crop_rule": "right half of actual ffprobe-decoded stereo width",
            "output_width": 640, "output_height": 480, "resize_mode": "letterbox", "crf": 20,
            "video_workers": args.video_workers, "ffmpeg_codec_threads": 4, "ffmpeg_filter_threads": 1,
            "numeric_and_wrist_data_preserved_bit_identical": True,
            "numeric_and_wrist_statistics_preserved": True,
            "alignment_preserved_bit_identical": True,
            "no_episode_splitting_or_sample_replanning": True,
            "source_csv_hashes_verified_against_original_contract_before_and_after": True,
            "head_statistics": "same 64 evenly spaced decoded frames as original converter; RGB uint8 histograms, float64 moments",
            "generated_by": "meta/repair_head_videos.py", "repair_script_sha256": sha256(script_copy),
            "numeric_conversion_provenance": "meta/converter_strict.py remains original; only head images repaired",
            "original_converter_sha256": sha256(original / "meta/converter_strict.py"),
            "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "episodes": report_eps,
        }
        report_path = staging / "meta/head_video_repair.json"
        json_write(report_path, report)
        contract["target"], contract["repo_id"] = str(target), args.repo_id
        contract["video_preprocessing"]["head_rgb_source"] = "camera/e6_rgb.h265 (actual dimensions read using ffprobe)"
        contract["video_preprocessing"]["head_rgb_crop"] = "dynamic right half from decoded source dimensions; crop=iw/2:ih:iw/2:0"
        contract["video_preprocessing"]["head_rgb_actual_crops"] = [
            {"episode_index": p["episode_index"], "width": p["input_probe"]["width"],
             "height": p["input_probe"]["height"], "crop_xywh": p["crop_xywh"]} for p in plans]
        contract["head_video_repair"] = {"path": "meta/head_video_repair.json", "sha256": sha256(report_path),
                                          "original_dataset_contract_sha256": contract_sha}
        json_write(staging / "meta/umi_conversion.json", contract)
        json_write(staging / "meta/head_video_repair.pending.json", {"status": "complete", "report": "meta/head_video_repair.json"})
        # All immutable artifacts were verified; use a no-replace atomic directory rename.
        publish_no_replace(staging, target)
        print(f"COMPLETE target={target} episodes={len(plans)} frames={contract['total_frames']}", flush=True)
    except BaseException:
        if staging is not None:
            print(f"FAILED: original is unchanged; incomplete staging retained at {staging}", file=sys.stderr, flush=True)
        raise
    finally:
        os.close(lock_fd)


def self_test() -> None:
    np = NP
    require(right_crop(3840, 1200) == [1920, 0, 1920, 1200], "3840 crop test")
    require(right_crop(3200, 1200) == [1600, 0, 1600, 1200], "3200 crop test")
    for bad in ((3838, 1200), (3840, 1199), (0, 0)):
        try:
            right_crop(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid crop accepted")
    rng = np.random.default_rng(4)
    values = rng.integers(0, 256, (11, 24, 32, 3), dtype=np.uint8)
    hist = np.stack([np.bincount(values[..., c].ravel(), minlength=256) for c in range(3)])
    stats = channel_histogram_stats(hist, len(values))
    floats = values.astype(np.float64) / 255.0
    for key, reduction in (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std)):
        np.testing.assert_allclose(np.asarray(stats[key]).ravel(), reduction(floats, axis=(0, 1, 2)), atol=1e-12, rtol=1e-12)
    pooled = aggregate_head([stats, stats])
    for key in ("min", "max", "mean", "std"):
        np.testing.assert_allclose(pooled[key], stats[key], atol=1e-12)
    require(pooled["count"] == [22], "aggregate sample count")
    with tempfile.TemporaryDirectory(prefix="repair-head-self-test-") as temp:
        p = Path(temp) / "grid.csv"
        with p.open("w") as f:
            f.write("is_output_anchor,e6_source_row_index,e6_frame_id,e6_global_ts_ns\n0,0,8,100\n1,6,14,200\n1,12,20,300\n")
        indices, _ = selected_rows(p)
        require(indices == [6, 12], "saved anchor parse")
        require(index_hash(indices) == hashlib.sha256(b"6,12").hexdigest(), "index digest")
    print("SELF_TEST_OK crop, saved frame selection, histogram statistics, aggregation", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dataset", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--video-workers", type=int, default=4)
    parser.add_argument("--confirm", choices=["REPAIR_HEAD_VIDEOS"])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    global NP
    import numpy as np
    NP = np
    if args.self_test:
        self_test()
        return
    if not all((args.original_dataset, args.target, args.repo_id, args.confirm)):
        parser.error("--original-dataset --target --repo-id --confirm REPAIR_HEAD_VIDEOS are required")
    require(1 <= args.video_workers <= 4, "video workers must be in [1,4]")
    require(bool(args.repo_id.strip()) and "/" in args.repo_id, "expected namespace/repo repo-id")
    # Avoid hidden BLAS oversubscription in this CPU-only operation.
    import cv2
    cv2.setNumThreads(1)
    require(shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None, "ffmpeg and ffprobe required")
    repair(args)


if __name__ == "__main__":
    main()
