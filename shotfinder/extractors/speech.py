"""Speech transcription with faster-whisper (Phase 5).

The model is loaded only from the local models directory: ``HF_HUB_OFFLINE`` is
set for the whole process so a missing or damaged file fails loudly instead of
silently pulling a different model from the network.

Audio is decoded by the bundled ffmpeg into a temporary 16 kHz mono WAV inside
the data directory. Whisper is faster on that format and the file is removed in
``finally`` even when decoding or transcription fails.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable

from .. import models
from .video import ffmpeg_path

# Set before faster-whisper (and its huggingface_hub dependency) is imported.
os.environ["HF_HUB_OFFLINE"] = "1"

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

ProgressCallback = Callable[[int], None]

_lock = threading.Lock()
_models: dict[tuple[str, int], object] = {}


class SpeechUnavailableError(RuntimeError):
    """Raised when the audio cannot be extracted or the model is missing."""


def _load_model(models_dir: str | Path, threads: int):
    """Return a cached WhisperModel loaded from the local directory."""
    if not models.speech_ready(models_dir):
        raise SpeechUnavailableError(
            f"speech model missing from {models.speech_model_dir(models_dir)}"
        )
    key = (str(Path(models_dir).resolve()), int(threads))
    with _lock:
        cached = _models.get(key)
        if cached is not None:
            return cached
        from faster_whisper import WhisperModel

        model = WhisperModel(
            str(models.speech_model_dir(models_dir)),
            device="cpu",
            compute_type="int8",
            cpu_threads=max(1, int(threads)),
            local_files_only=True,
        )
        _models[key] = model
        return model


def _extract_audio(video_path: str | Path, work_dir: str | Path) -> Path:
    """Write a 16 kHz mono WAV into *work_dir* and return its path."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix="speech-", suffix=".wav", dir=work_dir, delete=False
    )
    target = Path(handle.name)
    handle.close()
    command = [
        ffmpeg_path(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "wav",
        str(target),
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=3600,
            creationflags=creationflags,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        target.unlink(missing_ok=True)
        raise SpeechUnavailableError(f"could not extract audio: {exc}") from exc
    if completed.returncode != 0 or not target.is_file():
        target.unlink(missing_ok=True)
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise SpeechUnavailableError(
            f"ffmpeg could not extract audio from {video_path}: {detail}"
        )
    return target


def transcribe(
    video_path: str | Path,
    *,
    models_dir: str | Path,
    work_dir: str | Path,
    threads: int = 4,
    duration: float | None = None,
    progress: ProgressCallback | None = None,
) -> list[tuple[float, float, str]]:
    """Transcribe *video_path* into ``(start_s, end_s, text)`` segments.

    ``duration`` is only used to report progress; it comes from the same ffmpeg
    probe as the frame coverage so the percentage matches the video timeline.
    """
    wav = _extract_audio(video_path, work_dir)
    try:
        model = _load_model(models_dir, threads)
        segments, _info = model.transcribe(
            str(wav),
            language=None,
            beam_size=1,
            vad_filter=True,
        )
        rows: list[tuple[float, float, str]] = []
        for segment in segments:
            text = str(segment.text).strip()
            if not text:
                continue
            start = max(0.0, float(segment.start))
            end = max(start, float(segment.end))
            rows.append((start, end, text))
            if progress is not None and duration:
                percent = int(min(100.0, max(0.0, end / float(duration) * 100.0)))
                progress(percent)
        if progress is not None:
            progress(100)
        return rows
    finally:
        Path(wav).unlink(missing_ok=True)
