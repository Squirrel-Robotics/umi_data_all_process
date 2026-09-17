from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import CACHE_VERSION, MEDIA_SPECS, MediaCache, create_app  # noqa: E402


TOKEN = "test-review-token-0123456789"
EPISODE_A = "20260829_174440_126675865_137221"
EPISODE_B = "20260829_174457_588412161_137221"


def make_episode(root: Path, episode_id: str, *, complete: bool = True) -> Path:
    episode = root / episode_id
    (episode / "camera").mkdir(parents=True)
    (episode / "extensions/customer_camera/cam0").mkdir(parents=True)
    (episode / "extensions/customer_camera/cam1").mkdir(parents=True)
    (episode / "extensions/customer_camera").mkdir(parents=True, exist_ok=True)
    (episode / "meta").mkdir(parents=True)

    (episode / ".done").write_bytes(b"")
    (episode / "camera/hand_pose.csv").write_text(
        "timestamp,left_x,right_x\n1,0,0\n2,0.1,0.2\n", encoding="utf-8"
    )
    (episode / "camera/e6_rgb.h265").write_bytes(b"head-raw")
    (episode / "extensions/customer_camera/cam0/media.mjpeg").write_bytes(b"left-raw")
    (episode / "extensions/customer_camera/cam1/media.mjpeg").write_bytes(b"right-raw")
    (episode / "camera/e6_rgb_stream_metainfo.csv").write_text(
        "e6_mid_exposure_realtime_ns\n1000000000\n1016666667\n1033333334\n",
        encoding="utf-8",
    )
    (episode / "extensions/customer_camera/summary.json").write_text(
        json.dumps(
            {
                "status": "warning",
                "problem_codes": ["TEST_WARNING"],
                "warnings": ["synthetic test warning"],
            }
        ),
        encoding="utf-8",
    )
    (episode / "meta/health_summary.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    if not complete:
        (episode / ".done").unlink()
    return episode


@pytest.fixture()
def app_environment(tmp_path: Path):
    root = tmp_path / "dataset"
    root.mkdir()
    make_episode(root, EPISODE_A)
    make_episode(root, EPISODE_B)
    make_episode(root, "20260829_174518_398660877_137221", complete=False)
    collector = root / "collector_run_active"
    collector.mkdir()
    make_episode(collector, "20260829_174539_860486825_137221")

    outside = tmp_path / "outside"
    outside.mkdir()
    make_episode(outside, "20260829_174606_921024002_137221")
    symlink_id = "20260829_174624_941310438_137221"
    os.symlink(outside / "20260829_174606_921024002_137221", root / symlink_id)

    config = {
        "TESTING": True,
        "DATA_ROOT": root,
        "CACHE_DIR": tmp_path / "cache",
        "DATABASE": tmp_path / "state/review.sqlite3",
        "TOKEN_FILE": tmp_path / "state/token",
        "REVIEW_TOKEN": TOKEN,
        "FFMPEG_CONCURRENCY": 2,
    }
    application = create_app(config)
    return application, config, root, tmp_path


def auth_headers() -> dict[str, str]:
    return {"X-Review-Token": TOKEN}


def wait_for_precache(client, timeout: float = 3.0) -> dict:
    deadline = time.monotonic() + timeout
    latest: dict = {}
    while time.monotonic() < deadline:
        latest = client.get("/api/precache", headers=auth_headers()).get_json()
        if latest["run_number"] and not latest["running"]:
            return latest
        time.sleep(0.01)
    raise AssertionError(f"precache did not finish: {latest}")


def test_lists_only_complete_direct_non_symlink_episodes_and_reads_detail(app_environment):
    application, _config, _root, _tmp_path = app_environment
    client = application.test_client()

    response = client.get("/api/episodes", headers=auth_headers())
    assert response.status_code == 200
    payload = response.get_json()
    assert [item["episode_id"] for item in payload["items"]] == [EPISODE_A, EPISODE_B]
    assert payload["total"] == 2

    detail = client.get(f"/api/episodes/{EPISODE_A}", headers=auth_headers()).get_json()
    assert detail["required_files_complete"] is True
    assert detail["hand_pose_exists"] is True
    assert detail["hand_pose_rows"] == 2
    assert set(detail["media"]) == {"head", "left_wrist", "right_wrist"}
    assert detail["media"]["head"]["label"] == "头部相机（E6 右眼）"
    assert detail["quality"]["has_warning"] is True
    assert detail["quality"]["problem_codes"] == ["TEST_WARNING"]

    symlink = client.get(
        "/api/episodes/20260829_174624_941310438_137221",
        headers=auth_headers(),
    )
    assert symlink.status_code == 400


def test_rejects_intermediate_directory_symlink(tmp_path: Path):
    root = tmp_path / "dataset"
    root.mkdir()
    outside_camera = tmp_path / "outside_camera"
    outside_camera.mkdir()
    (outside_camera / "hand_pose.csv").write_text("frame\n1\n", encoding="utf-8")
    (outside_camera / "e6_rgb.h265").write_bytes(b"outside-head")

    episode_id = "20260829_174624_941310438_137221"
    episode = root / episode_id
    episode.mkdir()
    (episode / ".done").write_bytes(b"")
    os.symlink(outside_camera, episode / "camera")
    for camera in ("cam0", "cam1"):
        folder = episode / "extensions/customer_camera" / camera
        folder.mkdir(parents=True)
        (folder / "media.mjpeg").write_bytes(b"wrist")

    application = create_app(
        {
            "TESTING": True,
            "DATA_ROOT": root,
            "CACHE_DIR": tmp_path / "cache",
            "DATABASE": tmp_path / "state/review.sqlite3",
            "REVIEW_TOKEN": TOKEN,
        }
    )
    client = application.test_client()
    listed = client.get("/api/episodes", headers=auth_headers()).get_json()
    assert listed["items"] == []
    assert client.get(
        f"/api/episodes/{episode_id}", headers=auth_headers()
    ).status_code == 409


def test_keep_and_skip_require_token_and_persist_in_sqlite(app_environment):
    application, config, _root, _tmp_path = app_environment
    client = application.test_client()

    assert client.get("/api/episodes").status_code == 401
    assert client.get(f"/api/media/{EPISODE_A}/head").status_code == 401

    denied = client.post(
        f"/api/episodes/{EPISODE_A}/decision", json={"action": "keep"}
    )
    assert denied.status_code == 401

    kept = client.post(
        f"/api/episodes/{EPISODE_A}/decision",
        json={"action": "keep", "note": "画面正常"},
        headers=auth_headers(),
    )
    assert kept.status_code == 200
    assert kept.get_json()["state"] == "kept"

    skipped = client.post(
        f"/api/episodes/{EPISODE_B}/decision",
        json={"action": "skip", "note": "稍后复查"},
        headers=auth_headers(),
    )
    assert skipped.status_code == 200
    assert skipped.get_json()["state"] == "skipped"

    stats = client.get("/api/stats", headers=auth_headers()).get_json()
    assert stats["total"] == 2
    assert stats["pending"] == 0
    assert stats["kept"] == 1
    assert stats["skipped"] == 1
    assert stats["trashed"] == 0

    recreated = create_app(config)
    recreated_client = recreated.test_client()
    kept_items = recreated_client.get(
        "/api/episodes?state=kept", headers=auth_headers()
    ).get_json()["items"]
    skipped_items = recreated_client.get(
        "/api/episodes?state=skipped", headers=auth_headers()
    ).get_json()["items"]
    assert [item["episode_id"] for item in kept_items] == [EPISODE_A]
    assert [item["episode_id"] for item in skipped_items] == [EPISODE_B]


def test_quarantine_is_atomic_recoverable_and_audited(app_environment):
    application, _config, root, _tmp_path = app_environment
    client = application.test_client()

    response = client.post(
        f"/api/episodes/{EPISODE_A}/decision",
        json={"action": "quarantine", "note": "左腕遮挡"},
        headers=auth_headers(),
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["state"] == "trashed"
    trash_name = payload["trash_name"]
    trash = root / ".data_choose_trash"
    assert not (root / EPISODE_A).exists()
    assert (trash / trash_name / ".done").is_file()

    audit_rows = [
        json.loads(line)
        for line in (trash / "review_actions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert audit_rows[-1]["action"] == "trash"
    assert audit_rows[-1]["episode_id"] == EPISODE_A

    trash_payload = client.get("/api/trash", headers=auth_headers()).get_json()
    assert trash_payload["total"] == 1
    assert trash_payload["items"][0]["trash_name"] == trash_name

    denied = client.post(f"/api/trash/{trash_name}/restore")
    assert denied.status_code == 401
    restored = client.post(
        f"/api/trash/{trash_name}/restore", headers=auth_headers()
    )
    assert restored.status_code == 200
    assert restored.get_json()["state"] == "pending"
    assert (root / EPISODE_A / ".done").is_file()
    assert not (trash / trash_name).exists()
    assert client.get(
        f"/api/episodes/{EPISODE_A}", headers=auth_headers()
    ).get_json()["state"] == "pending"
    assert json.loads(
        (trash / "review_actions.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )["action"] == "restore"


def test_media_response_honors_http_range(app_environment, monkeypatch: pytest.MonkeyPatch):
    application, _config, _root, tmp_path = app_environment
    client = application.test_client()
    fake_video = tmp_path / "fake.mp4"
    fake_video.write_bytes(b"0123456789abcdef")
    service = application.extensions["data_choose"]
    monkeypatch.setattr(service.cache, "ensure", lambda *_args, **_kwargs: fake_video)

    response = client.get(
        f"/api/media/{EPISODE_A}/head",
        headers={**auth_headers(), "Range": "bytes=2-5"},
    )
    assert response.status_code == 206
    assert response.data == b"2345"
    assert response.headers["Content-Range"] == "bytes 2-5/16"
    assert response.headers["Accept-Ranges"] == "bytes"


def test_ffmpeg_command_uses_explicit_60fps_head_crop_and_atomic_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache = MediaCache(tmp_path / "cache", "ffmpeg-test", preview_fps=15, max_concurrent=1)
    source = tmp_path / "head.h265"
    source.write_bytes(b"synthetic-head")
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        Path(command[-1]).write_bytes(b"synthetic-mp4")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("app.subprocess.run", fake_run)
    target = cache.ensure(EPISODE_A, MEDIA_SPECS["head"], source)
    assert target.read_bytes() == b"synthetic-mp4"
    assert len(commands) == 1
    command = commands[0]
    input_index = command.index("-i")
    assert command[input_index - 4 : input_index] == ["-r", "60", "-f", "hevc"]
    filters = command[command.index("-vf") + 1]
    assert "setpts=N/(60*TB)" in filters
    assert "crop=1600:1200:1600:0" in filters
    assert "fps=15" in filters

    metadata = target.with_name(f"{EPISODE_A}.head.v{CACHE_VERSION}.json")
    assert metadata.is_file()
    assert not list(target.parent.glob("*.tmp.mp4"))
    assert cache.ensure(EPISODE_A, MEDIA_SPECS["head"], source) == target
    assert len(commands) == 1
    source.write_bytes(b"synthetic-head-was-updated")
    assert cache.status(EPISODE_A, MEDIA_SPECS["head"], source)["status"] == "pending"


def test_api_config_documents_camera_mapping(app_environment):
    application, _config, _root, _tmp_path = app_environment
    config = application.test_client().get(
        "/api/config", headers=auth_headers()
    ).get_json()
    assert config["source_fps"] == 60
    assert config["cameras"]["head"]["crop"] == "crop=1600:1200:1600:0"
    assert "cam0" in config["cameras"]["left_wrist"]["label"]
    assert "cam1" in config["cameras"]["right_wrist"]["label"]
    assert config["token_required_for_writes"] is True
    assert config["token_required"] is True


def test_precache_is_off_by_default_and_skips_current_cache(
    app_environment, monkeypatch: pytest.MonkeyPatch
):
    application, _config, _root, tmp_path = app_environment
    client = application.test_client()
    service = application.extensions["data_choose"]

    initial = client.get("/api/precache", headers=auth_headers()).get_json()
    assert initial == {
        "auto_enabled": False,
        "completed": 0,
        "current": [],
        "errors": [],
        "failed": 0,
        "finished_at": None,
        "next_scan_at": None,
        "percent": 0.0,
        "ready": 0,
        "rescan_seconds": 0.0,
        "run_number": 0,
        "running": False,
        "started_at": None,
        "total_media": 0,
        "workers": 1,
    }

    # Seed one genuinely current cache entry. The manager must not call
    # ``ensure`` for it, while still counting it as ready and completed.
    spec = MEDIA_SPECS["head"]
    source = service.root / EPISODE_A / spec.relative_source
    target, metadata = service.cache._paths(EPISODE_A, spec.key)
    target.write_bytes(b"fresh-cached-video")
    metadata.write_text(
        json.dumps(service.cache._signature(source)), encoding="utf-8"
    )

    calls: list[tuple[str, str, bool]] = []

    def fake_ensure(episode_id, media_spec, _source, *, background=False):
        calls.append((episode_id, media_spec.key, background))
        return tmp_path / f"{episode_id}.{media_spec.key}.mp4"

    monkeypatch.setattr(service.cache, "ensure", fake_ensure)
    assert client.post("/api/precache/start").status_code == 401
    started = client.post("/api/precache/start", headers=auth_headers())
    assert started.status_code == 200
    assert started.get_json()["started"] is True

    final = wait_for_precache(client)
    assert final["total_media"] == 6  # two complete episodes x three cameras
    assert final["ready"] == 6
    assert final["completed"] == 6
    assert final["failed"] == 0
    assert final["percent"] == 100.0
    assert final["errors"] == []
    assert len(calls) == 5
    assert (EPISODE_A, "head", True) not in calls
    assert {episode_id for episode_id, _key, _background in calls} == {
        EPISODE_A,
        EPISODE_B,
    }
    assert all(background for _episode_id, _key, background in calls)


def test_precache_configuration_reserves_an_interactive_ffmpeg_slot(
    tmp_path: Path,
):
    root = tmp_path / "dataset"
    root.mkdir()
    with pytest.raises(RuntimeError, match="interactive slot"):
        create_app(
            {
                "TESTING": True,
                "DATA_ROOT": root,
                "CACHE_DIR": tmp_path / "cache",
                "DATABASE": tmp_path / "state/review.sqlite3",
                "REVIEW_TOKEN": TOKEN,
                "FFMPEG_CONCURRENCY": 2,
                "PRECACHE_WORKERS": 2,
            }
        )


def test_precache_reports_media_failure_without_killing_service(
    app_environment, monkeypatch: pytest.MonkeyPatch
):
    application, _config, _root, tmp_path = app_environment
    client = application.test_client()
    service = application.extensions["data_choose"]

    def sometimes_fails(episode_id, media_spec, _source, *, background=False):
        assert background is True
        if episode_id == EPISODE_B and media_spec.key == "right_wrist":
            raise RuntimeError("synthetic ffmpeg failure")
        return tmp_path / f"{episode_id}.{media_spec.key}.mp4"

    monkeypatch.setattr(service.cache, "ensure", sometimes_fails)
    response = client.post("/api/precache/start", headers=auth_headers())
    assert response.status_code == 200
    final = wait_for_precache(client)

    assert final["total_media"] == 6
    assert final["ready"] == 5
    assert final["completed"] == 6
    assert final["failed"] == 1
    assert final["percent"] == 100.0
    assert final["running"] is False
    assert final["current"] == []
    assert final["errors"] == [
        {
            "media": f"{EPISODE_B}/right_wrist",
            "error": "synthetic ffmpeg failure",
        }
    ]
    assert client.get("/health").status_code == 200


def test_periodic_precache_rescan_discovers_new_complete_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "dataset"
    root.mkdir()
    make_episode(root, EPISODE_A)
    calls: list[tuple[str, str]] = []

    def fake_ensure(_cache, episode_id, media_spec, _source, *, background=False):
        assert background is True
        calls.append((episode_id, media_spec.key))
        return tmp_path / f"{episode_id}.{media_spec.key}.mp4"

    monkeypatch.setattr(MediaCache, "ensure", fake_ensure)
    application = create_app(
        {
            "TESTING": True,
            "DATA_ROOT": root,
            "CACHE_DIR": tmp_path / "cache",
            "DATABASE": tmp_path / "state/review.sqlite3",
            "REVIEW_TOKEN": TOKEN,
            "AUTO_PRECACHE": True,
            "PRECACHE_WORKERS": 1,
            "PRECACHE_RESCAN_SECONDS": 0.15,
        }
    )
    manager = application.extensions["data_choose_precache"]

    def wait_until(predicate, timeout: float = 3.0):
        deadline = time.monotonic() + timeout
        latest = manager.status()
        while time.monotonic() < deadline:
            latest = manager.status()
            if predicate(latest):
                return latest
            time.sleep(0.01)
        raise AssertionError(f"periodic precache condition not reached: {latest}")

    try:
        first = wait_until(
            lambda state: state["run_number"] >= 1 and not state["running"]
        )
        assert first["total_media"] == 3
        assert first["auto_enabled"] is True
        assert first["rescan_seconds"] == 0.15
        assert first["next_scan_at"] is not None
        # Waiting for the next scan is scheduler state, not an active run.
        assert first["running"] is False
        first_run_number = first["run_number"]

        make_episode(root, EPISODE_B)
        second = wait_until(
            lambda state: (
                state["run_number"] > first_run_number
                and not state["running"]
                and state["total_media"] == 6
            )
        )
        assert second["ready"] == 6
        assert second["completed"] == 6
        assert second["failed"] == 0
        assert any(episode_id == EPISODE_B for episode_id, _key in calls)
    finally:
        manager.shutdown()
    assert manager.status()["auto_enabled"] is False
    assert manager.status()["next_scan_at"] is None
