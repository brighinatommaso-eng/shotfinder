"""Shared fixtures: one synthetic video with a single scene cut."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from shotfinder.extractors.video import ffmpeg_path


@pytest.fixture(scope="session")
def scene_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """20 s of ``testsrc`` followed by 60 s of solid blue (one scene cut)."""
    target = tmp_path_factory.mktemp("video") / "scene.mp4"
    command = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=20:size=320x240:rate=10",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:duration=60:size=320x240:rate=10",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0",
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        str(target),
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    completed = subprocess.run(
        command,
        capture_output=True,
        creationflags=creationflags,
        timeout=120,
    )
    if completed.returncode != 0 or not target.is_file():
        pytest.skip(
            "could not generate a test video: "
            + completed.stderr.decode("utf-8", "replace").strip()
        )
    return target
