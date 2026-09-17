#!/usr/bin/env python3
"""Single-user browser backend for reviewing complete UMI episodes.

Only complete, direct-child episode directories are ever exposed or mutated.
Rejecting an episode is recoverable: the directory is atomically moved to
``<dataset>/.data_choose_trash`` on the same filesystem and can be restored.
There is deliberately no permanent-delete endpoint.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import subprocess
import threading
import time
from typing import Any, Mapping

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.serving import make_server


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/mnt/data/dzq/umi/data/task_v1_new")
DEFAULT_STATE_DIR = SCRIPT_DIR / "runtime"
DEFAULT_CACHE_DIR = DEFAULT_STATE_DIR / "cache"
DEFAULT_DATABASE = DEFAULT_STATE_DIR / "review.sqlite3"
DEFAULT_TOKEN_FILE = DEFAULT_STATE_DIR / "review_token"
TRASH_DIRNAME = ".data_choose_trash"

API_VERSION = 1
CACHE_VERSION = 2
SOURCE_FPS = 60
PREVIEW_FPS = 15
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480

EPISODE_RE = re.compile(r"^\d{8}_\d{6}_\d+_\d+$")
TRASH_RE = re.compile(
    r"^(?P<episode>\d{8}_\d{6}_\d+_\d+)__"
    r"(?P<stamp>\d{8}T\d{6}\.\d{6}Z)__(?P<nonce>[0-9a-f]{8})$"
)


@dataclass(frozen=True)
class MediaSpec:
    key: str
    label: str
    relative_source: Path
    input_format: str
    crop: str | None = None


MEDIA_SPECS: dict[str, MediaSpec] = {
    "head": MediaSpec(
        key="head",
        label="头部相机（E6 右眼）",
        relative_source=Path("camera/e6_rgb.h265"),
        input_format="hevc",
        crop="crop=1600:1200:1600:0",
    ),
    "left_wrist": MediaSpec(
        key="left_wrist",
        label="左腕相机（cam0）",
        relative_source=Path("extensions/customer_camera/cam0/media.mjpeg"),
        input_format="mjpeg",
    ),
    "right_wrist": MediaSpec(
        key="right_wrist",
        label="右腕相机（cam1）",
        relative_source=Path("extensions/customer_camera/cam1/media.mjpeg"),
        input_format="mjpeg",
    ),
}

REQUIRED_EPISODE_FILES = (
    Path(".done"),
    Path("camera/hand_pose.csv"),
    MEDIA_SPECS["head"].relative_source,
    MEDIA_SPECS["left_wrist"].relative_source,
    MEDIA_SPECS["right_wrist"].relative_source,
)


class APIError(RuntimeError):
    """An expected error that should be returned as JSON."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def load_or_create_token(path: Path) -> str:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"unsafe token path: {path}")
        token = path.read_text(encoding="utf-8").strip()
        if len(token) < 24:
            raise RuntimeError(f"invalid token file: {path}")
        return token
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(token + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return token


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        written = os.write(descriptor, line)
        if written != len(line):
            raise OSError(f"short audit write: {written}/{len(line)}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def safe_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def count_csv_data_rows(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            next(reader, None)
            return sum(1 for row in reader if row)
    except OSError:
        return None


class DecisionStore:
    """Small SQLite store; one short-lived connection is used per operation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS episode_decisions (
                    episode_id TEXT PRIMARY KEY,
                    decision TEXT NOT NULL CHECK (decision IN ('keep', 'skip')),
                    reason TEXT,
                    trash_name TEXT UNIQUE,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _as_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def get(self, episode_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT episode_id, decision, reason, trash_name, updated_at "
                "FROM episode_decisions WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
        return self._as_dict(row)

    def all(self) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT episode_id, decision, reason, trash_name, updated_at "
                "FROM episode_decisions"
            ).fetchall()
        return {str(row["episode_id"]): dict(row) for row in rows}

    def upsert(
        self,
        episode_id: str,
        decision: str,
        reason: str | None,
        trash_name: str | None = None,
        updated_at: str | None = None,
    ) -> dict[str, Any]:
        timestamp = updated_at or utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO episode_decisions
                    (episode_id, decision, reason, trash_name, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(episode_id) DO UPDATE SET
                    decision = excluded.decision,
                    reason = excluded.reason,
                    trash_name = excluded.trash_name,
                    updated_at = excluded.updated_at
                """,
                (episode_id, decision, reason, trash_name, timestamp),
            )
        result = self.get(episode_id)
        assert result is not None
        return result

    def delete(self, episode_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM episode_decisions WHERE episode_id = ?", (episode_id,)
            )

    def restore_record(self, episode_id: str, record: dict[str, Any] | None) -> None:
        if record is None:
            self.delete(episode_id)
            return
        self.upsert(
            episode_id=episode_id,
            decision=str(record["decision"]),
            reason=record.get("reason"),
            trash_name=record.get("trash_name"),
            updated_at=str(record["updated_at"]),
        )


class MediaCache:
    """On-demand, bounded-concurrency H.264 preview cache."""

    def __init__(
        self,
        cache_dir: Path,
        ffmpeg: str,
        preview_fps: int,
        max_concurrent: int,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = ffmpeg
        self.preview_fps = preview_fps
        self.max_concurrent = max(1, max_concurrent)
        self.background_concurrency = max(1, self.max_concurrent - 1)
        self._slots = threading.Semaphore(self.max_concurrent)
        # Background prewarming may use at most N-1 of N FFmpeg slots.  With
        # the production default (N=2), one slot therefore remains available
        # when the browser requests an uncached episode during the first scan.
        self._background_slots = threading.Semaphore(self.background_concurrency)
        self._locks_guard = threading.Lock()
        self._locks: dict[tuple[str, str], threading.Lock] = {}
        self._state_guard = threading.Lock()
        self._states: dict[tuple[str, str], dict[str, Any]] = {}

    def _paths(self, episode_id: str, key: str) -> tuple[Path, Path]:
        target = self.cache_dir / f"{episode_id}.{key}.v{CACHE_VERSION}.mp4"
        metadata = self.cache_dir / f"{episode_id}.{key}.v{CACHE_VERSION}.json"
        return target, metadata

    @staticmethod
    def _signature(source: Path) -> dict[str, Any]:
        stat = source.stat()
        return {
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "source_inode": stat.st_ino,
            "source_device": stat.st_dev,
            "cache_version": CACHE_VERSION,
        }

    def _is_current(self, source: Path, target: Path, metadata: Path) -> bool:
        if not target.is_file() or target.stat().st_size == 0 or not metadata.is_file():
            return False
        saved = safe_json(metadata)
        try:
            return saved == self._signature(source)
        except OSError:
            return False

    def status(self, episode_id: str, spec: MediaSpec, source: Path) -> dict[str, Any]:
        target, metadata = self._paths(episode_id, spec.key)
        if source.is_file() and self._is_current(source, target, metadata):
            return {"status": "ready", "progress": 1.0, "error": None}
        with self._state_guard:
            state = dict(self._states.get((episode_id, spec.key), {}))
        # A remembered ``ready`` state is only an optimization hint.  Once the
        # source signature no longer matches, report it as pending so a later
        # rescan rebuilds the stale preview instead of trusting old memory.
        if state and state.get("status") != "ready":
            return {
                "status": state.get("status", "pending"),
                "progress": state.get("progress", 0.0),
                "error": state.get("error"),
            }
        if not source.is_file():
            return {"status": "missing", "progress": 0.0, "error": "source missing"}
        return {"status": "pending", "progress": 0.0, "error": None}

    def _set_state(self, episode_id: str, key: str, **state: Any) -> None:
        with self._state_guard:
            self._states[(episode_id, key)] = state

    def ensure(
        self,
        episode_id: str,
        spec: MediaSpec,
        source: Path,
        *,
        background: bool = False,
    ) -> Path:
        target, metadata = self._paths(episode_id, spec.key)
        lock_key = (episode_id, spec.key)
        with self._locks_guard:
            lock = self._locks.setdefault(lock_key, threading.Lock())

        with lock:
            if self._is_current(source, target, metadata):
                self._set_state(
                    episode_id, spec.key, status="ready", progress=1.0, error=None
                )
                return target
            self._set_state(
                episode_id, spec.key, status="building", progress=0.1, error=None
            )
            try:
                if background:
                    with self._background_slots:
                        with self._slots:
                            self._transcode(source, target, spec)
                else:
                    with self._slots:
                        self._transcode(source, target, spec)
                signature = self._signature(source)
                atomic_write_json(metadata, signature)
            except Exception as error:
                self._set_state(
                    episode_id,
                    spec.key,
                    status="error",
                    progress=0.0,
                    error=str(error)[:1000],
                )
                raise
            self._set_state(
                episode_id, spec.key, status="ready", progress=1.0, error=None
            )
            return target

    def _transcode(self, source: Path, target: Path, spec: MediaSpec) -> None:
        temporary = target.with_name(
            f".{target.stem}.{secrets.token_hex(4)}.tmp.mp4"
        )
        # Raw HEVC can carry decoder timestamps that jump by hundreds of
        # seconds (especially through CUVID).  Rebuild the timeline from frame
        # order before resampling so all three cameras stay near 13--16 s.
        filters: list[str] = [f"setpts=N/({SOURCE_FPS}*TB)"]
        if spec.crop:
            filters.append(spec.crop)
        filters.extend(
            [
                f"fps={self.preview_fps}",
                (
                    f"scale={PREVIEW_WIDTH}:{PREVIEW_HEIGHT}:"
                    "force_original_aspect_ratio=decrease"
                ),
                (
                    f"pad={PREVIEW_WIDTH}:{PREVIEW_HEIGHT}:"
                    "(ow-iw)/2:(oh-ih)/2:color=black"
                ),
            ]
        )
        def command(hardware_decode: bool) -> list[str]:
            decoder = (
                ["-hwaccel", "cuda", "-c:v", f"{spec.input_format}_cuvid"]
                if hardware_decode
                else []
            )
            return [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                *decoder,
                # Raw HEVC and concatenated MJPEG otherwise get misreported as 25 FPS.
                "-r",
                str(SOURCE_FPS),
                "-f",
                spec.input_format,
                "-i",
                str(source),
                "-an",
                "-vf",
                ",".join(filters),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "28",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(self.preview_fps),
                "-vsync",
                "cfr",
                "-movflags",
                "+faststart",
                "-threads",
                "2",
                str(temporary),
            ]
        try:
            errors: list[str] = []
            completed: subprocess.CompletedProcess[str] | None = None
            # CUVID makes 1920x1080 MJPEG review practical; CPU decode remains a
            # portable fallback for hosts without NVIDIA decoder support.
            for hardware_decode in (True, False):
                temporary.unlink(missing_ok=True)
                completed = subprocess.run(
                    command(hardware_decode),
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                if completed.returncode == 0:
                    break
                errors.append(completed.stderr.strip() or "ffmpeg failed")
            if completed is None or completed.returncode:
                raise RuntimeError("; ".join(errors)[-2000:])
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("ffmpeg created an empty preview")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


class ReviewService:
    def __init__(
        self,
        root: Path,
        database: Path,
        cache: MediaCache,
        token: str,
    ) -> None:
        original_root = root
        if original_root.is_symlink() or not original_root.is_dir():
            raise RuntimeError(f"dataset root is missing, not a directory, or a symlink: {root}")
        self.root = original_root.resolve(strict=True)
        self.trash = self.root / TRASH_DIRNAME
        self.audit_path = self.trash / "review_actions.jsonl"
        self.store = DecisionStore(database)
        self.cache = cache
        self.token = token
        self._mutation_lock = threading.Lock()

    @staticmethod
    def validate_episode_id(episode_id: str) -> str:
        if not EPISODE_RE.fullmatch(episode_id):
            raise APIError("invalid episode ID", 400)
        return episode_id

    def _safe_direct_directory(self, parent: Path, name: str) -> Path:
        candidate = parent / name
        if candidate.parent != parent or candidate.is_symlink():
            raise APIError("unsafe or symlinked directory", 400)
        if not candidate.is_dir():
            raise APIError(f"episode not found: {name}", 404)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise APIError(f"cannot resolve episode: {error}", 404) from error
        if resolved.parent != parent:
            raise APIError("episode must be a direct child of the dataset root", 400)
        return candidate

    @staticmethod
    def _is_safe_episode_file(episode: Path, relative: Path) -> bool:
        """Return true only for a regular file reached without any symlink."""
        if relative.is_absolute() or ".." in relative.parts:
            return False
        try:
            episode_real = episode.resolve(strict=True)
            current = episode
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    return False
            if not current.is_file():
                return False
            return current.resolve(strict=True).is_relative_to(episode_real)
        except OSError:
            return False

    @classmethod
    def _has_required_files(cls, episode: Path) -> bool:
        for relative in REQUIRED_EPISODE_FILES:
            if not cls._is_safe_episode_file(episode, relative):
                return False
        return True

    def validate_episode(self, episode_id: str) -> Path:
        self.validate_episode_id(episode_id)
        episode = self._safe_direct_directory(self.root, episode_id)
        if not self._has_required_files(episode):
            raise APIError("episode is incomplete or still being collected", 409)
        return episode

    def discover(self) -> list[Path]:
        episodes: list[Path] = []
        for path in self.root.iterdir():
            if (
                not EPISODE_RE.fullmatch(path.name)
                or path.is_symlink()
                or not path.is_dir()
            ):
                continue
            try:
                if path.resolve(strict=True).parent != self.root:
                    continue
            except OSError:
                continue
            if self._has_required_files(path):
                episodes.append(path)
        return sorted(episodes, key=lambda item: item.name)

    def _ensure_trash(self) -> Path:
        if self.trash.exists():
            if self.trash.is_symlink() or not self.trash.is_dir():
                raise APIError(f"unsafe trash directory: {self.trash}", 500)
        else:
            self.trash.mkdir(mode=0o775)
        if self.trash.resolve(strict=True).parent != self.root:
            raise APIError("trash must be a direct child of the dataset root", 500)
        return self.trash

    @staticmethod
    def _status(record: dict[str, Any] | None) -> str:
        if record is None:
            return "pending"
        if record.get("trash_name"):
            return "trashed"
        return "kept" if record.get("decision") == "keep" else "skipped"

    def summary(self, episode: Path, record: dict[str, Any] | None) -> dict[str, Any]:
        state = self._status(record)
        hand_pose = episode / "camera/hand_pose.csv"
        return {
            "id": episode.name,
            "episode_id": episode.name,
            "status": state,
            "state": state,
            "decision": record.get("decision") if record else None,
            "reason": record.get("reason") if record else None,
            "updated_at": record.get("updated_at") if record else None,
            "hand_pose_exists": hand_pose.is_file(),
            "hand_pose_rows": count_csv_data_rows(hand_pose),
            "version": API_VERSION,
        }

    def list_episodes(
        self,
        status: str,
        query: str,
        page: int,
        limit: int,
    ) -> dict[str, Any]:
        records = self.store.all()
        items = [self.summary(path, records.get(path.name)) for path in self.discover()]
        normalized = {
            "keep": "kept",
            "kept": "kept",
            "skip": "skipped",
            "skipped": "skipped",
            "pending": "pending",
            "all": "all",
            "": "all",
        }.get(status)
        if normalized is None:
            raise APIError(f"unknown status filter: {status}", 400)
        if normalized != "all":
            items = [item for item in items if item["status"] == normalized]
        if query:
            folded = query.casefold()
            items = [item for item in items if folded in item["episode_id"].casefold()]
        total = len(items)
        start = (page - 1) * limit
        selected = items[start : start + limit]
        return {
            "items": selected,
            "episodes": selected,
            "total": total,
            "page": page,
            "limit": limit,
            "version": API_VERSION,
        }

    def _head_timing(self, episode: Path) -> dict[str, Any]:
        path = episode / "camera/e6_rgb_stream_metainfo.csv"
        first: int | None = None
        last: int | None = None
        count = 0
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    raw = (
                        row.get("e6_mid_exposure_realtime_ns")
                        or row.get("e6_mid_exposure_boot_ns")
                    )
                    if not raw:
                        continue
                    try:
                        timestamp = int(raw)
                    except ValueError:
                        continue
                    first = timestamp if first is None else first
                    last = timestamp
                    count += 1
        except OSError:
            return {"frame_count": None, "duration_seconds": None, "source_fps": SOURCE_FPS}
        duration = None
        if first is not None and last is not None and count > 1 and last > first:
            duration = (last - first) / 1_000_000_000 + 1 / SOURCE_FPS
        return {
            "frame_count": count,
            "duration_seconds": round(duration, 3) if duration is not None else None,
            "source_fps": SOURCE_FPS,
        }

    @staticmethod
    def _quality(episode: Path) -> dict[str, Any]:
        customer = safe_json(
            episode / "extensions/customer_camera/summary.json"
        )
        health = safe_json(episode / "meta/health_summary.json")
        problem_codes = customer.get("problem_codes", [])
        if not isinstance(problem_codes, list):
            problem_codes = [str(problem_codes)]
        warnings = customer.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = [str(warnings)]
        customer_status = customer.get("status")
        health_status = health.get("status") or health.get("overall_status")
        has_warning = bool(problem_codes or warnings)
        if customer_status not in (None, "ok", "OK", "healthy"):
            has_warning = True
        if health_status not in (None, "ok", "OK", "healthy"):
            has_warning = True
        return {
            "customer_camera_status": customer_status,
            "health_status": health_status,
            "problem_codes": problem_codes,
            "warnings": warnings,
            "has_warning": has_warning,
        }

    def detail(self, episode_id: str) -> dict[str, Any]:
        episode = self.validate_episode(episode_id)
        record = self.store.get(episode_id)
        result = self.summary(episode, record)
        timing = self._head_timing(episode)
        media: dict[str, dict[str, Any]] = {}
        for key, spec in MEDIA_SPECS.items():
            source = episode / spec.relative_source
            media_state = self.cache.status(episode_id, spec, source)
            media[key] = {
                "key": key,
                "label": spec.label,
                "url": f"/api/media/{episode_id}/{key}",
                "source_exists": source.is_file() and not source.is_symlink(),
                **media_state,
            }
        states = [item["status"] for item in media.values()]
        if all(state == "ready" for state in states):
            aggregate = "ready"
        elif any(state == "error" for state in states):
            aggregate = "error"
        elif any(state == "building" for state in states):
            aggregate = "building"
        elif any(state == "ready" for state in states):
            aggregate = "partial"
        else:
            aggregate = "pending"
        errors = {
            key: item["error"] for key, item in media.items() if item.get("error")
        }
        result.update(
            {
                **timing,
                "preview_fps": self.cache.preview_fps,
                "hand_pose_rows": count_csv_data_rows(episode / "camera/hand_pose.csv"),
                "media": media,
                "media_status": aggregate,
                "media_progress": round(
                    sum(float(item["progress"]) for item in media.values()) / len(media), 3
                ),
                "media_error": errors or None,
                "quality": self._quality(episode),
                "required_files_complete": True,
            }
        )
        return result

    def stats(self) -> dict[str, int]:
        active = self.discover()
        active_names = {path.name for path in active}
        records = self.store.all()
        kept = sum(
            1
            for episode_id, record in records.items()
            if episode_id in active_names
            and record["decision"] == "keep"
            and not record.get("trash_name")
        )
        skipped = sum(
            1
            for episode_id, record in records.items()
            if episode_id in active_names
            and record["decision"] == "skip"
            and not record.get("trash_name")
        )
        trashed = len(self.list_trash())
        pending = max(0, len(active) - kept - skipped)
        return {
            "total": len(active) + trashed,
            "active": len(active),
            "pending": pending,
            "kept": kept,
            "keep": kept,
            "skipped": skipped,
            "skip": skipped,
            "trashed": trashed,
            "reviewed": kept + skipped + trashed,
        }

    def set_decision(
        self, episode_id: str, decision: str, reason: str | None
    ) -> dict[str, Any]:
        self.validate_episode(episode_id)
        if decision not in {"keep", "skip"}:
            raise APIError("decision must be keep, skip, or delete", 400)
        with self._mutation_lock:
            record = self.store.upsert(episode_id, decision, reason)
        return {
            "episode_id": episode_id,
            "decision": record["decision"],
            "status": self._status(record),
            "state": self._status(record),
            "reason": record.get("reason"),
            "updated_at": record["updated_at"],
        }

    def trash_episode(self, episode_id: str, reason: str | None) -> dict[str, Any]:
        with self._mutation_lock:
            source = self.validate_episode(episode_id)
            trash = self._ensure_trash()
            if source.stat().st_dev != trash.stat().st_dev:
                raise APIError("trash is not on the same filesystem; refusing non-atomic move", 500)
            previous = self.store.get(episode_id)
            trash_name = f"{episode_id}__{utc_stamp()}__{secrets.token_hex(4)}"
            target = trash / trash_name
            if target.exists() or target.is_symlink():
                raise APIError("trash destination collision", 409)
            moved = False
            try:
                os.replace(source, target)
                moved = True
                record = self.store.upsert(
                    episode_id, "skip", reason, trash_name=trash_name
                )
                append_jsonl(
                    self.audit_path,
                    {
                        "action": "trash",
                        "episode_id": episode_id,
                        "trash_name": trash_name,
                        "reason": reason,
                        "timestamp_utc": utc_now(),
                    },
                )
            except Exception:
                if moved and target.is_dir() and not source.exists():
                    os.replace(target, source)
                self.store.restore_record(episode_id, previous)
                try:
                    append_jsonl(
                        self.audit_path,
                        {
                            "action": "trash_rollback",
                            "episode_id": episode_id,
                            "trash_name": trash_name,
                            "timestamp_utc": utc_now(),
                        },
                    )
                except OSError:
                    pass
                raise
        return {
            "episode_id": episode_id,
            "decision": record["decision"],
            "status": "trashed",
            "state": "trashed",
            "trash_name": trash_name,
            "reason": reason,
            "updated_at": record["updated_at"],
        }

    def list_trash(self) -> list[dict[str, Any]]:
        if not self.trash.exists():
            return []
        if self.trash.is_symlink() or not self.trash.is_dir():
            raise APIError("unsafe trash directory", 500)
        records = self.store.all()
        items: list[dict[str, Any]] = []
        for path in self.trash.iterdir():
            match = TRASH_RE.fullmatch(path.name)
            if match is None or path.is_symlink() or not path.is_dir():
                continue
            try:
                if path.resolve(strict=True).parent != self.trash.resolve(strict=True):
                    continue
            except OSError:
                continue
            episode_id = match.group("episode")
            record = records.get(episode_id)
            items.append(
                {
                    "id": path.name,
                    "trash_name": path.name,
                    "episode_id": episode_id,
                    "deleted_at": match.group("stamp"),
                    "reason": record.get("reason") if record else None,
                    "status": "trashed",
                    "state": "trashed",
                }
            )
        return sorted(items, key=lambda item: item["trash_name"], reverse=True)

    def restore(self, trash_name: str) -> dict[str, Any]:
        match = TRASH_RE.fullmatch(trash_name)
        if match is None:
            raise APIError("invalid trash name", 400)
        episode_id = match.group("episode")
        with self._mutation_lock:
            trash = self._ensure_trash()
            source = self._safe_direct_directory(trash, trash_name)
            target = self.root / episode_id
            if target.exists() or target.is_symlink():
                raise APIError(f"restore destination already exists: {episode_id}", 409)
            if source.stat().st_dev != self.root.stat().st_dev:
                raise APIError("restore would not be atomic", 500)
            previous = self.store.get(episode_id)
            moved = False
            try:
                os.replace(source, target)
                moved = True
                if not self._has_required_files(target):
                    raise RuntimeError("restored episode no longer satisfies completeness contract")
                self.store.delete(episode_id)
                append_jsonl(
                    self.audit_path,
                    {
                        "action": "restore",
                        "episode_id": episode_id,
                        "trash_name": trash_name,
                        "timestamp_utc": utc_now(),
                    },
                )
            except Exception:
                if moved and target.is_dir() and not source.exists():
                    os.replace(target, source)
                self.store.restore_record(episode_id, previous)
                try:
                    append_jsonl(
                        self.audit_path,
                        {
                            "action": "restore_rollback",
                            "episode_id": episode_id,
                            "trash_name": trash_name,
                            "timestamp_utc": utc_now(),
                        },
                    )
                except OSError:
                    pass
                raise
        return {
            "episode_id": episode_id,
            "status": "pending",
            "state": "pending",
            "restored": True,
            "trash_name": trash_name,
        }


class PrecacheManager:
    """Thread-safe background builder for every valid episode preview.

    Discovery remains centralized in :meth:`ReviewService.discover`, while
    :meth:`MediaCache.ensure` provides cache validation, per-media locking,
    bounded FFmpeg concurrency, and atomic publication. Interactive media
    requests can therefore safely overlap this background worker.
    """

    def __init__(
        self,
        service: ReviewService,
        workers: int = 2,
        rescan_seconds: float = 0.0,
    ) -> None:
        if not 1 <= workers <= 4:
            raise ValueError("precache workers must be in [1, 4]")
        if not 0 <= rescan_seconds <= 3600:
            raise ValueError("precache rescan seconds must be in [0, 3600]")
        self.service = service
        self.workers = workers
        self.rescan_seconds = float(rescan_seconds)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._scheduler_thread: threading.Thread | None = None
        self._scheduler_event = threading.Event()
        self._auto_enabled = False
        self._next_scan_deadline: float | None = None
        self._next_scan_at: str | None = None
        self._run_number = 0
        self._running = False
        self._total_media = 0
        self._ready = 0
        self._completed = 0
        self._failed = 0
        self._current: set[str] = set()
        self._errors: list[dict[str, str]] = []
        self._started_at: str | None = None
        self._finished_at: str | None = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._total_media:
                percent = round(100.0 * self._completed / self._total_media, 1)
            elif self._started_at is not None and not self._running:
                percent = 100.0
            else:
                percent = 0.0
            return {
                "total_media": self._total_media,
                "ready": self._ready,
                "completed": self._completed,
                "failed": self._failed,
                "percent": percent,
                "running": self._running,
                "current": sorted(self._current),
                "errors": [dict(item) for item in self._errors],
                "workers": self.workers,
                "rescan_seconds": self.rescan_seconds,
                "auto_enabled": self._auto_enabled,
                "next_scan_at": self._next_scan_at,
                "run_number": self._run_number,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
            }

    def start(self) -> tuple[bool, dict[str, Any]]:
        """Start a fresh scan unless another scan is already running."""
        with self._lock:
            if self._running:
                started = False
            else:
                self._run_number += 1
                self._running = True
                self._total_media = 0
                self._ready = 0
                self._completed = 0
                self._failed = 0
                self._current.clear()
                self._errors.clear()
                self._started_at = utc_now()
                self._finished_at = None
                self._next_scan_deadline = None
                self._next_scan_at = None
                self._thread = threading.Thread(
                    target=self._run,
                    name=f"data-choose-precache-{self._run_number}",
                    daemon=True,
                )
                self._thread.start()
                started = True
        self._scheduler_event.set()
        return started, self.status()

    def enable_auto(self) -> bool:
        """Enable periodic rescans; starting the first scan remains explicit."""
        if self.rescan_seconds <= 0:
            return False
        with self._lock:
            if self._auto_enabled:
                return False
            self._auto_enabled = True
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name="data-choose-precache-scheduler",
                daemon=True,
            )
            self._scheduler_thread.start()
        self._scheduler_event.set()
        return True

    def shutdown(self) -> None:
        """Stop future automatic scans; an in-flight scan is allowed to finish."""
        with self._lock:
            self._auto_enabled = False
            self._next_scan_deadline = None
            self._next_scan_at = None
            scheduler = self._scheduler_thread
        self._scheduler_event.set()
        if scheduler is not None and scheduler is not threading.current_thread():
            scheduler.join(timeout=1.0)

    def _scheduler_loop(self) -> None:
        while True:
            with self._lock:
                if not self._auto_enabled:
                    return
                deadline = self._next_scan_deadline
            if deadline is None:
                self._scheduler_event.wait()
                self._scheduler_event.clear()
                continue
            timeout = max(0.0, deadline - time.monotonic())
            if self._scheduler_event.wait(timeout):
                self._scheduler_event.clear()
                continue
            # ``start`` is idempotent under its lock. If a manual run won the
            # race, that run will schedule the next deadline when it finishes.
            self.start()

    def _append_error(self, identity: str, error: BaseException) -> None:
        message = str(error).strip() or error.__class__.__name__
        with self._lock:
            self._errors.append({"media": identity, "error": message[:1000]})
            if len(self._errors) > 100:
                self._errors = self._errors[-100:]

    def _ensure_one(self, episode_id: str, media_key: str) -> Path:
        identity = f"{episode_id}/{media_key}"
        with self._lock:
            self._current.add(identity)
        try:
            # An episode may be quarantined after the initial snapshot. Check
            # the complete/non-symlink contract again immediately before work.
            episode = self.service.validate_episode(episode_id)
            spec = MEDIA_SPECS[media_key]
            if not self.service._is_safe_episode_file(episode, spec.relative_source):
                raise RuntimeError("media source became missing or unsafe")
            return self.service.cache.ensure(
                episode_id,
                spec,
                episode / spec.relative_source,
                background=True,
            )
        finally:
            with self._lock:
                self._current.discard(identity)

    def _run(self) -> None:
        try:
            episodes = self.service.discover()
            all_jobs = [
                (episode.name, media_key)
                for episode in episodes
                for media_key in MEDIA_SPECS
            ]
            pending: list[tuple[str, str]] = []
            already_ready = 0
            for episode_id, media_key in all_jobs:
                spec = MEDIA_SPECS[media_key]
                source = self.service.root / episode_id / spec.relative_source
                try:
                    cache_status = self.service.cache.status(
                        episode_id, spec, source
                    )
                except Exception:
                    # ``ensure`` is authoritative and will report a useful
                    # per-media failure without aborting the whole scan.
                    cache_status = {"status": "pending"}
                if cache_status.get("status") == "ready":
                    already_ready += 1
                else:
                    pending.append((episode_id, media_key))

            with self._lock:
                self._total_media = len(all_jobs)
                self._ready = already_ready
                self._completed = already_ready

            futures: dict[Future[Path], tuple[str, str]] = {}
            with ThreadPoolExecutor(
                max_workers=self.workers,
                thread_name_prefix="data-choose-precache-worker",
            ) as executor:
                for episode_id, media_key in pending:
                    future = executor.submit(self._ensure_one, episode_id, media_key)
                    futures[future] = (episode_id, media_key)
                for future in as_completed(futures):
                    episode_id, media_key = futures[future]
                    identity = f"{episode_id}/{media_key}"
                    try:
                        future.result()
                    except Exception as error:
                        with self._lock:
                            self._failed += 1
                            self._completed += 1
                        self._append_error(identity, error)
                    else:
                        with self._lock:
                            self._ready += 1
                            self._completed += 1
        except Exception as error:
            # A discovery or executor failure is visible through the status
            # endpoint and never escapes the daemon coordinator thread.
            self._append_error("precache", error)
            with self._lock:
                self._failed += 1
                if self._total_media and self._completed < self._total_media:
                    self._completed += 1
        finally:
            with self._lock:
                self._current.clear()
                self._running = False
                self._finished_at = utc_now()
                if self._auto_enabled and self.rescan_seconds > 0:
                    self._next_scan_deadline = time.monotonic() + self.rescan_seconds
                    self._next_scan_at = datetime.fromtimestamp(
                        time.time() + self.rescan_seconds, timezone.utc
                    ).isoformat()
                else:
                    self._next_scan_deadline = None
                    self._next_scan_at = None
            self._scheduler_event.set()


def parse_positive_int(name: str, default: int, maximum: int) -> int:
    raw = request.args.get(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise APIError(f"{name} must be an integer", 400) from error
    if value < 1 or value > maximum:
        raise APIError(f"{name} must be in [1, {maximum}]", 400)
    return value


def create_app(config: Mapping[str, Any] | None = None) -> Flask:
    app = Flask(
        __name__,
        static_folder=str(SCRIPT_DIR / "static"),
        static_url_path="/static",
    )
    app.config.from_mapping(
        DATA_ROOT=DEFAULT_DATA_ROOT,
        CACHE_DIR=DEFAULT_CACHE_DIR,
        STATE_DIR=DEFAULT_STATE_DIR,
        DATABASE=None,
        TOKEN_FILE=DEFAULT_TOKEN_FILE,
        REVIEW_TOKEN=None,
        FFMPEG="ffmpeg",
        PREVIEW_FPS=PREVIEW_FPS,
        FFMPEG_CONCURRENCY=2,
        AUTO_PRECACHE=False,
        PRECACHE_WORKERS=1,
        PRECACHE_RESCAN_SECONDS=0.0,
        MAX_CONTENT_LENGTH=64 * 1024,
    )
    if config:
        app.config.update(config)

    token = app.config.get("REVIEW_TOKEN")
    if not token:
        token = load_or_create_token(Path(app.config["TOKEN_FILE"]))
    if not isinstance(token, str) or len(token) < 8:
        raise RuntimeError("REVIEW_TOKEN must be a non-empty string")

    ffmpeg_concurrency = int(app.config["FFMPEG_CONCURRENCY"])
    if not 2 <= ffmpeg_concurrency <= 8:
        raise RuntimeError("FFMPEG_CONCURRENCY must be in [2, 8]")
    cache = MediaCache(
        cache_dir=Path(app.config["CACHE_DIR"]),
        ffmpeg=str(app.config["FFMPEG"]),
        preview_fps=int(app.config["PREVIEW_FPS"]),
        max_concurrent=ffmpeg_concurrency,
    )
    configured_database = app.config.get("DATABASE")
    database = (
        Path(configured_database)
        if configured_database
        else Path(app.config["STATE_DIR"]) / "review.sqlite3"
    )
    service = ReviewService(
        root=Path(app.config["DATA_ROOT"]),
        database=database,
        cache=cache,
        token=token,
    )
    precache_workers = int(app.config["PRECACHE_WORKERS"])
    if not 1 <= precache_workers <= 4:
        raise RuntimeError("PRECACHE_WORKERS must be in [1, 4]")
    if precache_workers >= ffmpeg_concurrency:
        raise RuntimeError(
            "PRECACHE_WORKERS must be smaller than FFMPEG_CONCURRENCY "
            "so an interactive slot remains available"
        )
    precache_rescan_seconds = float(app.config["PRECACHE_RESCAN_SECONDS"])
    if not 0 <= precache_rescan_seconds <= 3600:
        raise RuntimeError("PRECACHE_RESCAN_SECONDS must be in [0, 3600]")
    precache = PrecacheManager(
        service,
        workers=precache_workers,
        rescan_seconds=precache_rescan_seconds,
    )
    app.extensions["data_choose"] = service
    app.extensions["data_choose_precache"] = precache
    if bool(app.config["AUTO_PRECACHE"]):
        precache.enable_auto()
        precache.start()

    def supplied_token() -> str:
        return (
            request.headers.get("X-Review-Token")
            or request.args.get("token")
            or ""
        )

    @app.before_request
    def authorize_api() -> Response | None:
        # The HTML shell and health probe are harmless without a token.  Every
        # dataset API (including video reads) stays private; <video> elements
        # pass the token in their query string because they cannot set headers.
        if request.path.startswith("/api/"):
            provided = supplied_token()
            if not provided or not secrets.compare_digest(provided, service.token):
                return jsonify(error="missing or invalid review token"), 401
        return None

    @app.errorhandler(APIError)
    def handle_api_error(error: APIError) -> tuple[Response, int]:
        return jsonify(error=str(error)), error.status_code

    @app.errorhandler(subprocess.TimeoutExpired)
    def handle_timeout(error: subprocess.TimeoutExpired) -> tuple[Response, int]:
        return jsonify(error=f"ffmpeg timed out after {error.timeout} seconds"), 504

    @app.errorhandler(404)
    def handle_not_found(_error: Any) -> tuple[Response, int]:
        if request.path.startswith("/api/"):
            return jsonify(error="not found"), 404
        return jsonify(error="not found"), 404

    @app.get("/")
    def index() -> Response:
        index_path = SCRIPT_DIR / "static/index.html"
        if index_path.is_file():
            return send_file(index_path)
        return jsonify(
            name="UMI data chooser",
            api_version=API_VERSION,
            message="static/index.html has not been installed",
        )

    @app.get("/health")
    def health() -> Response:
        return jsonify(status="ok", root=str(service.root), version=API_VERSION)

    @app.get("/api/config")
    def api_config() -> Response:
        return jsonify(
            version=API_VERSION,
            root=str(service.root),
            preview_fps=service.cache.preview_fps,
            source_fps=SOURCE_FPS,
            auto_precache=bool(app.config["AUTO_PRECACHE"]),
            precache_workers=precache.workers,
            precache_rescan_seconds=precache.rescan_seconds,
            token_required=True,
            token_required_for_writes=True,
            episode_completeness=[str(path) for path in REQUIRED_EPISODE_FILES],
            cameras={
                key: {
                    "key": key,
                    "label": spec.label,
                    "source": str(spec.relative_source),
                    "crop": spec.crop,
                }
                for key, spec in MEDIA_SPECS.items()
            },
        )

    @app.get("/api/episodes")
    def episodes() -> Response:
        status = (
            request.args.get("state")
            or request.args.get("status")
            or "all"
        ).strip().lower()
        if status == "trashed":
            items = service.list_trash()
            return jsonify(
                items=items,
                episodes=items,
                total=len(items),
                page=1,
                limit=len(items),
                version=API_VERSION,
            )
        query = (request.args.get("q") or request.args.get("search") or "").strip()
        page = parse_positive_int("page", 1, 1_000_000)
        limit = parse_positive_int("limit", 100, 500)
        return jsonify(service.list_episodes(status, query, page, limit))

    @app.get("/api/stats")
    def stats() -> Response:
        payload = service.stats()
        payload["version"] = API_VERSION
        return jsonify(payload)

    @app.get("/api/precache")
    def precache_status() -> Response:
        return jsonify(precache.status())

    @app.post("/api/precache/start")
    def precache_start() -> Response:
        started, status = precache.start()
        return jsonify(started=started, **status)

    @app.get("/api/episodes/<episode_id>")
    def episode_detail(episode_id: str) -> Response:
        return jsonify(service.detail(episode_id))

    @app.get("/api/media/<episode_id>/<media_key>")
    def media(episode_id: str, media_key: str) -> Response:
        episode = service.validate_episode(episode_id)
        spec = MEDIA_SPECS.get(media_key)
        if spec is None:
            raise APIError(f"unknown media key: {media_key}", 404)
        source = episode / spec.relative_source
        if not service._is_safe_episode_file(episode, spec.relative_source):
            raise APIError("media source is missing or unsafe", 404)
        try:
            target = service.cache.ensure(episode_id, spec, source)
        except subprocess.TimeoutExpired:
            raise
        except Exception as error:
            raise APIError(f"preview generation failed: {error}", 500) from error
        response = send_file(
            target,
            mimetype="video/mp4",
            conditional=True,
            etag=True,
            max_age=3600,
        )
        response.headers["Accept-Ranges"] = "bytes"
        return response

    @app.post("/api/episodes/<episode_id>/decision")
    def decide(episode_id: str) -> Response:
        payload = request.get_json(silent=True)
        if payload is None:
            payload = request.form.to_dict()
        if not isinstance(payload, dict):
            raise APIError("request body must be a JSON object", 400)
        decision = str(payload.get("action") or payload.get("decision") or "").strip().lower()
        reason_value = payload.get("note")
        if reason_value is None:
            reason_value = payload.get("reason")
        if reason_value is None or str(reason_value).strip() == "":
            reason = None
        else:
            reason = str(reason_value).strip()
            if len(reason) > 500:
                raise APIError("reason must not exceed 500 characters", 400)
        if decision in {"delete", "reject", "trash", "quarantine"}:
            result = service.trash_episode(episode_id, reason)
        else:
            result = service.set_decision(episode_id, decision, reason)
        return jsonify(result=result, stats=service.stats(), **result)

    @app.get("/api/trash")
    def trash() -> Response:
        items = service.list_trash()
        return jsonify(items=items, total=len(items), version=API_VERSION)

    @app.post("/api/trash/<trash_name>/restore")
    def restore(trash_name: str) -> Response:
        result = service.restore(trash_name)
        return jsonify(result=result, stats=service.stats(), **result)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--preview-fps", type=int, default=PREVIEW_FPS)
    parser.add_argument("--ffmpeg-concurrency", type=int, default=2)
    parser.add_argument(
        "--no-precache",
        action="store_true",
        help="do not build missing media previews in the background at startup",
    )
    parser.add_argument("--precache-workers", type=int, default=1)
    parser.add_argument(
        "--precache-rescan-seconds",
        type=float,
        default=60.0,
        help="seconds after a completed run before rescanning; 0 disables rescans",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.root.is_symlink() or not args.root.is_dir():
        parser.error(f"dataset root does not exist, is not a directory, or is a symlink: {args.root}")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if not 1 <= args.preview_fps <= SOURCE_FPS:
        parser.error(f"--preview-fps must be in [1, {SOURCE_FPS}]")
    if not 2 <= args.ffmpeg_concurrency <= 8:
        parser.error("--ffmpeg-concurrency must be in [2, 8]")
    if not 1 <= args.precache_workers <= 4:
        parser.error("--precache-workers must be in [1, 4]")
    if args.precache_workers >= args.ffmpeg_concurrency:
        parser.error(
            "--precache-workers must be smaller than --ffmpeg-concurrency"
        )
    if not 0 <= args.precache_rescan_seconds <= 3600:
        parser.error("--precache-rescan-seconds must be in [0, 3600]")
    return args


def main() -> int:
    args = parse_args()
    auto_precache = not args.no_precache
    application = create_app(
        {
            "DATA_ROOT": args.root,
            "CACHE_DIR": args.cache,
            "STATE_DIR": args.state_dir,
            "TOKEN_FILE": args.token_file,
            "FFMPEG": args.ffmpeg,
            "PREVIEW_FPS": args.preview_fps,
            "FFMPEG_CONCURRENCY": args.ffmpeg_concurrency,
            # Production prewarming starts only after ``make_server`` has
            # successfully bound the port. This avoids spawning FFmpeg when a
            # duplicate process or unrelated service already owns the port.
            "AUTO_PRECACHE": False,
            "PRECACHE_WORKERS": args.precache_workers,
            "PRECACHE_RESCAN_SECONDS": args.precache_rescan_seconds,
        }
    )
    application.config["AUTO_PRECACHE"] = auto_precache
    service: ReviewService = application.extensions["data_choose"]
    precache: PrecacheManager = application.extensions["data_choose_precache"]
    server = make_server(args.host, args.port, application, threaded=True)
    print(f"dataset: {service.root}", flush=True)
    print(f"open: http://{args.host}:{args.port}/?token={service.token}", flush=True)
    if auto_precache:
        precache.enable_auto()
        precache.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        precache.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
