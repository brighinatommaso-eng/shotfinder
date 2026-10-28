"""Shared fixtures: synthetic videos for frame and speech tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from shotfinder.extractors.video import ffmpeg_path

SPOKEN_SENTENCE = "The quick brown fox jumps over the lazy dog"


def _creationflags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def _run(command: list[str], timeout: int = 180) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        capture_output=True,
        creationflags=_creationflags(),
        timeout=timeout,
    )


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


@pytest.fixture(scope="session")
def speech_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """12 s of ``testsrc`` muxed with an offline TTS recording of a known line.

    Windows SAPI is used because it works without a network; on other systems
    the speech tests are skipped rather than silently passing on silence.
    """
    if sys.platform != "win32":
        pytest.skip("offline TTS fixture is Windows-only")
    folder = tmp_path_factory.mktemp("speech")
    wav = folder / "spoken.wav"
    quoted = str(wav).replace("'", "''")
    sentence = SPOKEN_SENTENCE.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{quoted}'); "
        f"$s.Speak('{sentence}'); "
        "$s.Dispose()"
    )
    spoken = _run(["powershell", "-NoProfile", "-Command", script])
    if spoken.returncode != 0 or not wav.is_file():
        pytest.skip(
            "could not synthesise the speech fixture: "
            + spoken.stderr.decode("utf-8", "replace").strip()
        )

    target = folder / "speech.mp4"
    completed = _run(
        [
            ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=12:size=320x240:rate=10",
            "-i",
            str(wav),
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-shortest",
            str(target),
        ]
    )
    if completed.returncode != 0 or not target.is_file():
        pytest.skip(
            "could not mux the speech fixture: "
            + completed.stderr.decode("utf-8", "replace").strip()
        )
    return target
