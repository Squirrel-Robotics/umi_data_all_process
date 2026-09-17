#!/usr/bin/env python3
"""Publish an already verified right-eye repair on filesystems lacking renameat2.

This recovery tool never re-encodes videos, changes numeric samples, deletes a
dataset, or modifies either original contract. It validates the completed repair
again, then atomically renames it over an exclusively created EMPTY reservation.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

FROZEN_REPAIR_SHA256 = "4e9c18c75770b8892a8a8141c7249972291e3de6c7866cfa9ebd5d05920bcc12"
HEAD = "observation.images.head_rgb"
WRISTS = ("observation.images.left_wrist_rgb", "observation.images.right_wrist_rgb")


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def identity(path: Path) -> tuple[int, int]:
    stat = path.lstat()
    return stat.st_dev, stat.st_ino


def no_symlink_tree(root: Path) -> None:
    require(root.is_dir() and not root.is_symlink(), f"need a real directory: {root}")
    for p in root.rglob("*"):
        require(not p.is_symlink() and (p.is_file() or p.is_dir()), f"unexpected node: {p}")


def json_create(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def validate(staging: Path, target: Path, original: Path) -> dict[str, Any]:
    import numpy as np
    import cv2
    cv2.setNumThreads(1)
    no_symlink_tree(staging)
    no_symlink_tree(original)
    pending = read_json(staging / "meta/head_video_repair.pending.json")
    require(pending == {"status": "complete", "report": "meta/head_video_repair.json"},
            "repair did not finish all encoding and QA; pending status is not complete")
    contract_path = staging / "meta/umi_conversion.json"
    report_path = staging / "meta/head_video_repair.json"
    info_path = staging / "meta/info.json"
    contract, report, info = map(read_json, (contract_path, report_path, info_path))
    old_contract_path, old_info_path = original / "meta/umi_conversion.json", original / "meta/info.json"
    old = read_json(old_contract_path)
    require(report["schema"] == "umi-head-video-right-eye-repair" and report["schema_version"] == 1
            and report["status"] == "complete", "unexpected repair report schema/status")
    require(report["source_dataset"] == str(original) and report["target_dataset"] == str(target),
            "repair source/target mismatch")
    require(contract["target"] == str(target) and contract["repo_id"] == report["repo_id"], "new contract identity mismatch")
    require(digest(old_contract_path) == report["source_dataset_contract_sha256"], "original contract changed")
    require(digest(old_info_path) == digest(info_path) == report["source_dataset_info_sha256"], "original/info copy changed")
    require(contract["head_video_repair"]["path"] == "meta/head_video_repair.json"
            and contract["head_video_repair"]["sha256"] == digest(report_path)
            and contract["head_video_repair"]["original_dataset_contract_sha256"] == digest(old_contract_path),
            "repair report / new contract hash linkage invalid")
    generator_path = staging / "meta/repair_head_videos.py"
    require(digest(generator_path) == report["repair_script_sha256"] == FROZEN_REPAIR_SHA256,
            "unexpected or modified repair generator; refusing to execute its validation helpers")
    require(digest(staging / "meta/converter_strict.py") == digest(original / "meta/converter_strict.py")
            == report["original_converter_sha256"], "original numeric converter provenance changed")
    # Execute ONLY the exact frozen generator just verified above. No import loader
    # is used, so this does not create __pycache__ files inside the staging tree.
    helpers = {"__name__": "verified_frozen_repair_validation", "__file__": str(generator_path)}
    exec(compile(generator_path.read_text(), str(generator_path), "exec"), helpers)
    helpers["NP"] = np
    require(old["status"] == contract["status"] == "complete", "dataset status invalid")
    require(old["total_episodes"] == contract["total_episodes"] == report["total_episodes"] == info["total_episodes"] == 85,
            "this recovery requires all approved 85 episodes")
    require(old["total_frames"] == contract["total_frames"] == report["total_frames"] == info["total_frames"] == 9995,
            "approved sample count changed")
    require(old["fps"] == contract["fps"] == report["fps"] == 10 and old["action_shape"] == [50, 30],
            "training recipe changed")
    require(info["codebase_version"] == "v2.1", "LeRobot version changed")
    expected = copy.deepcopy(old)
    expected["target"], expected["repo_id"] = str(target), report["repo_id"]
    # These are the ONLY permitted contract changes made by the frozen repair.
    expected["head_video_repair"] = contract["head_video_repair"]
    for key in ("head_rgb_source", "head_rgb_crop", "head_rgb_actual_crops"):
        expected["video_preprocessing"][key] = contract["video_preprocessing"][key]
    require(contract["video_preprocessing"]["head_rgb_crop"]
            == "dynamic right half from decoded source dimensions; crop=iw/2:ih:iw/2:0", "right-eye policy mismatch")
    require(len(report["episodes"]) == len(contract["episodes"]) == len(old["episodes"]) == 85, "episode manifests incomplete")
    stat_rows = [json.loads(x) for x in (staging / "meta/episodes_stats.jsonl").read_text().splitlines() if x.strip()]
    old_stat_rows = [json.loads(x) for x in (original / "meta/episodes_stats.jsonl").read_text().splitlines() if x.strip()]
    require([s["episode_index"] for s in stat_rows] == [s["episode_index"] for s in old_stat_rows] == list(range(85)),
            "episode stats identities invalid")
    head_stats = []
    preservation = []
    for i, (e, old_e, audit) in enumerate(zip(contract["episodes"], old["episodes"], report["episodes"])):
        require(e["episode_index"] == old_e["episode_index"] == audit["episode_index"] == i, "episode order changed")
        require(e["source_episode_id"] == old_e["source_episode_id"] == audit["source_episode_id"], "source identity changed")
        expected["episodes"][i]["videos"][HEAD] = e["videos"][HEAD]
        helpers["check_csv_hashes"](old_e)
        expected_preserved = [old_e["parquet"], *(old_e["videos"][k] for k in WRISTS)]
        require(len(audit["preserved_files"]) == 3, "missing Parquet/wrist preservation records")
        for rec, proof in zip(expected_preserved, audit["preserved_files"]):
            require(rec["path"] == proof["path"], "preservation path/order mismatch")
            source_sha = helpers["check_record"](original, rec)
            staged_sha = helpers["check_record"](staging, rec)
            require(source_sha == staged_sha == proof["source_sha256"] == proof["sha256"], "preservation SHA256 failed")
            require(identity(original / rec["path"]) != identity(staging / rec["path"]), "hardlink is not a preserved copy")
            preservation.append({"path": rec["path"], "sha256": staged_sha})
        helpers["check_record"](original, old_e["videos"][HEAD])
        staged_head_hash = helpers["check_record"](staging, e["videos"][HEAD])
        audit_head = dict(audit["head_video"])
        raw_sha = audit_head.pop("source_raw_sha256")
        require(audit_head == e["videos"][HEAD] and staged_head_hash == audit["head_video"]["sha256"], "head record mismatch")
        raw = Path(old_e["source_episode"]) / "camera/e6_rgb.h265"
        require(audit["source_raw_path"] == str(raw) and digest(raw) == raw_sha, "raw stream changed")
        fresh_probe = helpers["probe"](raw, raw=True)
        require((fresh_probe["width"], fresh_probe["height"]) == (3840, 1200)
                and fresh_probe == audit["input_probe"], "raw decoded stream format changed")
        require(audit["crop_xywh"] == helpers["right_crop"](fresh_probe["width"], fresh_probe["height"])
                == [1920, 0, 1920, 1200], "right-eye crop invalid")
        declaration = audit["source_metadata_declaration"]
        require(digest(Path(old_e["source_episode"]) / declaration["path"]) == declaration["sha256"], "source declaration changed")
        grid = staging / audit["alignment_output_grid_path"]
        require(digest(grid) == audit["alignment_output_grid_sha256"]
                == digest(original / audit["alignment_output_grid_path"]), "saved alignment changed")
        indices, _ = helpers["selected_rows"](grid)
        require(indices == audit["source_row_indices"] and len(indices) == e["output_rows"], "saved selection changed")
        require(helpers["index_hash"](indices) == audit["source_row_indices_sha256"], "frame selection digest invalid")
        output_probe = helpers["probe"](staging / e["videos"][HEAD]["path"], count=True)
        require((output_probe["width"], output_probe["height"], output_probe["avg_frame_rate"],
                 output_probe["codec_name"], output_probe["pix_fmt"], int(output_probe["nb_read_packets"]))
                == (640, 480, "10/1", "h264", "yuv420p", e["output_rows"]), "repaired head video format invalid")
        recomputed = helpers["head_stats"](staging / e["videos"][HEAD]["path"], e["output_rows"], 640, 480)
        for key, value in recomputed.items():
            np.testing.assert_allclose(stat_rows[i]["stats"][HEAD][key], value, atol=1e-12, rtol=1e-12,
                                       err_msg=f"head image stats mismatch episode {i}, {key}")
        before = {k: v for k, v in old_stat_rows[i]["stats"].items() if k != HEAD}
        after = {k: v for k, v in stat_rows[i]["stats"].items() if k != HEAD}
        require(before == after, "non-head episode statistics changed")
        head_stats.append(recomputed)
        if (i + 1) % 10 == 0 or i == 84:
            print(f"publication revalidation {i + 1}/85: hashes, old/new proof, crop, full head decode, statistics", flush=True)
    require(expected == contract, "new contract changes outside approved head repair")
    stats, old_stats = read_json(staging / "meta/stats.json"), read_json(original / "meta/stats.json")
    require({k: v for k, v in stats.items() if k != HEAD} == {k: v for k, v in old_stats.items() if k != HEAD},
            "non-head aggregate statistics changed")
    for key, value in helpers["aggregate_head"](head_stats).items():
        np.testing.assert_allclose(stats[HEAD][key], value, atol=1e-12, rtol=1e-12, err_msg="head aggregate mismatch")
    allowed_changes = {"meta/umi_conversion.json", "meta/stats.json", "meta/episodes_stats.jsonl"}
    head_paths = {e["videos"][HEAD]["path"] for e in old["episodes"]}
    for old_file in original.rglob("*"):
        if not old_file.is_file():
            continue
        rel = old_file.relative_to(original).as_posix()
        if rel not in allowed_changes | head_paths:
            require((staging / rel).is_file() and digest(old_file) == digest(staging / rel), f"preserved artifact changed: {rel}")
    manifest = [{"path": p.relative_to(staging).as_posix(), "size_bytes": p.stat().st_size, "sha256": digest(p)}
                for p in sorted(staging.rglob("*")) if p.is_file()]
    require(digest(old_contract_path) == report["source_dataset_contract_sha256"], "original changed during validation")
    return {"repair_report_sha256": digest(report_path), "repaired_contract_sha256": digest(contract_path),
            "original_contract_sha256": digest(old_contract_path), "repair_generator_sha256": FROZEN_REPAIR_SHA256,
            "total_episodes": 85, "total_frames": 9995, "preserved_data_files": preservation,
            "verified_staging_files": manifest}


def reserve_and_rename(staging: Path, target: Path, *, rename: Callable[..., Any] = os.rename) -> dict[str, Any]:
    """Never replace an existing target; remove only our own empty reservation."""
    require(staging.is_dir() and not staging.is_symlink(), "missing real staging directory")
    staging_identity = identity(staging)
    # mkdir, unlike rename, is atomic and fails if ANY existing target is present.
    os.mkdir(target, 0o700)
    reservation_identity = identity(target)
    renamed = False
    try:
        require(not target.is_symlink() and identity(target) == reservation_identity,
                "reservation was replaced; refusing rename")
        require(not any(target.iterdir()), "reservation is no longer empty; refusing rename")
        # POSIX atomic rename replaces an empty directory only. If another actor
        # puts files in the reservation, the kernel refuses with ENOTEMPTY.
        rename(staging, target)
        renamed = True
        require(identity(target) == staging_identity and not staging.exists(), "published inode mismatch")
        try:
            fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as e:
            if e.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
            print(f"NOTE parent-directory fsync unsupported ({e}); rename already completed", flush=True)
        return {"reservation_identity": list(reservation_identity), "published_identity": list(staging_identity)}
    finally:
        # Never delete staging, a dataset, a non-empty target, or another inode.
        if not renamed and target.exists() and not target.is_symlink() and identity(target) == reservation_identity:
            try:
                target.rmdir()
            except OSError as e:
                if e.errno not in (errno.ENOTEMPTY, errno.EEXIST, errno.ENOENT):
                    print(f"NOTE empty reservation could not be removed: {target}: {e}", file=sys.stderr)


def publish(args: argparse.Namespace) -> None:
    for p in (args.staging, args.target, args.original_dataset):
        require(p.is_absolute() and not p.is_symlink(), f"need absolute non-symlink path: {p}")
    staging, target, original = args.staging.resolve(), args.target.resolve(), args.original_dataset.resolve()
    require(staging.parent == target.parent and staging.name.startswith(f".{target.name}.repair-"), "unexpected staging path")
    require(target != original and not target.is_relative_to(original) and not original.is_relative_to(target), "overlapping original/target")
    require(not target.exists(), f"target already exists; refusing: {target}")
    lock = target.parent / f".{target.name}.repair.lock"
    lock_fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not target.exists(), f"target already exists; refusing: {target}")
        verified = validate(staging, target, original)
        publication = staging / "meta/head_video_publication.json"
        require(not publication.exists(), "publication record already exists; inspect previous recovery attempt before retry")
        publisher_copy = staging / "meta/publish_repaired_dataset.py"
        require(not publisher_copy.exists(), "publisher artifact already exists; inspect previous recovery attempt")
        shutil.copy2(Path(__file__).resolve(), publisher_copy)
        report = {"schema": "umi-head-video-publication", "schema_version": 1,
                  # This marker travels inside the atomic directory rename. A
                  # consumer must accept it only at target_dataset, never staging.
                  "status": "complete", "valid_only_at_target_dataset": True,
                  "publication_method": "exclusive_empty_directory_reservation_then_posix_atomic_rename",
                  "reason": "NFS renameat2(RENAME_NOREPLACE) returned EINVAL; existing completed repair reused",
                  "source_dataset": str(original), "staging_directory": str(staging), "target_dataset": str(target),
                  "validated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "publisher_script_sha256": digest(publisher_copy),
                  "source_dataset_contract_sha256": verified["original_contract_sha256"],
                  "repair_audit_sha256": verified["repair_report_sha256"],
                  "published_contract_sha256": verified["repaired_contract_sha256"],
                  "original_report_and_contract_unchanged": True, **verified}
        json_create(publication, report)
        result = reserve_and_rename(staging, target)
        # Do not rewrite the audit after publication: existence at target, the
        # retained inode, and this completion receipt establish successful rename.
        print(json.dumps({"status": "complete", "target": str(target), "publication_method": report["publication_method"],
                          "publication_record_sha256": digest(target / "meta/head_video_publication.json"), **result}), flush=True)
    finally:
        os.close(lock_fd)


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="safe-nfs-publication-test-") as root:
        base = Path(root)
        stage, target = base / "staging", base / "target"
        stage.mkdir()
        (stage / "payload").write_text("preserve me")
        result = reserve_and_rename(stage, target)
        require(not stage.exists() and (target / "payload").read_text() == "preserve me", "basic rename failed")
        require(identity(target) == tuple(result["published_identity"]), "inode identity not preserved")
        blocked = base / "blocked"
        blocked.mkdir()
        (blocked / "other-data").write_text("must not change")
        newstage = base / "newstage"
        newstage.mkdir()
        try:
            reserve_and_rename(newstage, blocked)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing target accepted")
        require((blocked / "other-data").read_text() == "must not change" and newstage.exists(), "existing target damaged")
        empty = base / "preexisting-empty"
        empty.mkdir()
        try:
            reserve_and_rename(newstage, empty)
        except FileExistsError:
            pass
        else:
            raise AssertionError("preexisting empty target accepted")
        require(empty.is_dir(), "preexisting empty target removed")
        failed = base / "failed-reservation"
        def fake_nfs_failure(src: Path, dst: Path) -> None:
            raise OSError(errno.EINVAL, "simulated unsupported atomic operation")
        try:
            reserve_and_rename(newstage, failed, rename=fake_nfs_failure)
        except OSError:
            pass
        else:
            raise AssertionError("simulated failure ignored")
        require(newstage.is_dir() and not failed.exists(), "failed empty reservation cleanup unsafe")
        raced = base / "raced-reservation"
        def concurrent_writer(src: Path, dst: Path) -> None:
            (dst / "other-writer").write_text("do not delete")
            os.rename(src, dst)
        try:
            reserve_and_rename(newstage, raced, rename=concurrent_writer)
        except OSError:
            pass
        else:
            raise AssertionError("nonempty reservation overwritten")
        require(newstage.is_dir() and (raced / "other-writer").read_text() == "do not delete", "concurrent files damaged")
    print("SELF_TEST_OK atomic rename; existing nonempty/empty targets; simulated NFS error; concurrent writer preservation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--original-dataset", type=Path)
    parser.add_argument("--confirm", choices=["PUBLISH_VERIFIED_REPAIR"])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.staging, args.target, args.original_dataset, args.confirm)):
        parser.error("--staging --target --original-dataset --confirm PUBLISH_VERIFIED_REPAIR are required")
    publish(args)


if __name__ == "__main__":
    main()
