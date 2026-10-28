"""Config file parsing: defaults, BOM handling and error reporting."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from shotfinder.config import (
    FOLDERID_DOWNLOADS,
    FOLDERID_PICTURES,
    FOLDERID_SCREENSHOTS,
    FOLDERID_VIDEOS,
    Config,
    default_watched_folders,
)


def test_config_accepts_utf8_bom(tmp_path: Path) -> None:
    payload = '\ufeff{"watched_folders": ["C:/media"], "onnx_threads": 2}'
    (tmp_path / "config.json").write_text(payload, encoding="utf-8")

    config = Config.load(tmp_path)

    assert config.watched_folders == [Path("C:/media")]
    assert config.onnx_threads == 2


def test_malformed_config_logs_a_warning_with_path(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="shotfinder.config"):
        config = Config.load(tmp_path)

    assert config.data_dir == tmp_path
    assert "Could not read" in caplog.text
    assert str(path) in caplog.text
    assert "Expecting property name" in caplog.text


def test_min_visual_similarity_defaults_and_roundtrips(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path)
    # 0.20 is a garbage floor only; see docs/threshold-calibration.md.
    assert config.min_visual_similarity == 0.20

    config.min_visual_similarity = 0.31
    config.save()

    assert Config.load(tmp_path).min_visual_similarity == 0.31


def test_related_cap_and_video_turn_defaults_roundtrip(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path)
    assert config.max_visual_results == 8
    assert config.video_frames_per_turn == 30
    assert config.speech_max_minutes == 60

    config.max_visual_results = 3
    config.video_frames_per_turn = 12
    config.speech_max_minutes = 90
    config.save()

    loaded = Config.load(tmp_path)
    assert loaded.max_visual_results == 3
    assert loaded.video_frames_per_turn == 12
    assert loaded.speech_max_minutes == 90


def test_the_old_025_similarity_default_is_migrated(tmp_path: Path) -> None:
    # 0.25 was the default before calibration; it is not a user choice.
    (tmp_path / "config.json").write_text(
        json.dumps({"min_visual_similarity": 0.25}), encoding="utf-8"
    )

    assert Config.load(tmp_path).min_visual_similarity == 0.20


def test_a_tuned_similarity_is_kept(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"min_visual_similarity": 0.28}), encoding="utf-8"
    )

    assert Config.load(tmp_path).min_visual_similarity == 0.28


def test_hotkey_is_not_marked_custom_by_default(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path)
    config.save()

    loaded = Config.load(tmp_path)

    assert loaded.hotkey == "ctrl+alt+space"
    assert loaded.hotkey_is_custom is False


def test_a_hotkey_chosen_by_the_app_stays_automatic(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path)
    config.hotkey = "ctrl+shift+space"
    config.hotkey_is_custom = False
    config.save()

    loaded = Config.load(tmp_path)

    assert loaded.hotkey == "ctrl+shift+space"
    assert loaded.hotkey_is_custom is False


def test_an_explicit_hotkey_is_marked_custom(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"hotkey": "ctrl+alt+f"}), encoding="utf-8"
    )

    loaded = Config.load(tmp_path)

    assert loaded.hotkey == "ctrl+alt+f"
    assert loaded.hotkey_is_custom is True


def test_an_old_config_with_a_non_default_hotkey_is_treated_as_explicit(
    tmp_path: Path,
) -> None:
    # Written before hotkey_custom existed: a non-default combo came from the user.
    (tmp_path / "config.json").write_text(
        json.dumps({"hotkey": "alt+shift+s"}), encoding="utf-8"
    )

    assert Config.load(tmp_path).hotkey_is_custom is True


def test_cloud_only_indexing_is_off_by_default_and_roundtrips(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path)
    assert config.index_cloud_only_files is False

    config.index_cloud_only_files = True
    config.save()

    assert Config.load(tmp_path).index_cloud_only_files is True



def test_defaults_follow_the_windows_known_folder_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screenshots = tmp_path / "Immagini" / "Screenshots"
    pictures = tmp_path / "Immagini"
    videos = tmp_path / "Video"
    missing_downloads = tmp_path / "Download"
    for folder in (screenshots, pictures, videos):
        folder.mkdir(parents=True, exist_ok=True)

    resolved = {
        FOLDERID_SCREENSHOTS: screenshots,
        FOLDERID_PICTURES: pictures,
        FOLDERID_VIDEOS: videos,
        FOLDERID_DOWNLOADS: missing_downloads,
    }
    monkeypatch.setattr(
        "shotfinder.config.known_folder", lambda folder_id: resolved[folder_id]
    )

    assert default_watched_folders() == [screenshots, pictures, videos]


def test_legacy_default_folders_are_migrated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = [tmp_path / "Pictures", tmp_path / "Videos"]
    migrated = [tmp_path / "Immagini", tmp_path / "Video", tmp_path / "Download"]
    for folder in legacy + migrated:
        folder.mkdir()
    (tmp_path / "config.json").write_text(
        json.dumps({"watched_folders": [str(folder) for folder in legacy]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "shotfinder.config.legacy_default_watched_folders", lambda: list(legacy)
    )
    monkeypatch.setattr(
        "shotfinder.config.default_watched_folders", lambda: list(migrated)
    )

    config = Config.load(tmp_path)

    assert config.watched_folders == migrated


def test_custom_watched_folders_are_not_migrated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "projects" / "screens"
    custom.mkdir(parents=True)
    (tmp_path / "config.json").write_text(
        json.dumps({"watched_folders": [str(custom)]}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "shotfinder.config.legacy_default_watched_folders", lambda: [tmp_path / "Videos"]
    )
    monkeypatch.setattr(
        "shotfinder.config.default_watched_folders", lambda: [tmp_path / "Immagini"]
    )

    assert Config.load(tmp_path).watched_folders == [custom]


def test_missing_watched_folders_are_reported(tmp_path: Path) -> None:
    existing = tmp_path / "media"
    existing.mkdir()
    missing = tmp_path / "gone"

    config = Config(data_dir=tmp_path / "data", watched_folders=[existing, missing])

    assert config.missing_watched_folders() == [missing]


def test_cli_warns_about_missing_folders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    from shotfinder.cli import _warn_missing_folders

    missing = tmp_path / "gone"
    config = Config(data_dir=tmp_path / "data", watched_folders=[missing])

    with caplog.at_level(logging.WARNING, logger="shotfinder.cli"):
        _warn_missing_folders(config)

    assert str(missing) in capsys.readouterr().err
    assert str(missing) in caplog.text

