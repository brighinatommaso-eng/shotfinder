"""Config file parsing: defaults, BOM handling and error reporting."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from shotfinder.config import Config


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
