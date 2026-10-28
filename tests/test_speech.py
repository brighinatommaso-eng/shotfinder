"""Speech extraction: local model registry, transcript and temp-file cleanup."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from shotfinder import models
from shotfinder.extractors import speech

REPO_MODELS = Path(
    os.environ.get("SHOTFINDER_MODELS_DIR")
    or Path(__file__).resolve().parents[1] / "models"
)


@pytest.fixture(scope="module")
def models_dir() -> Path:
    if not models.speech_ready(REPO_MODELS):
        pytest.skip(
            f"whisper model not installed in {REPO_MODELS}; run "
            "`python -m shotfinder.cli models download`"
        )
    return REPO_MODELS


def test_speech_models_live_in_their_own_release_and_subfolder() -> None:
    assert models.SPEECH_MODELS
    for model in models.SPEECH_MODELS:
        assert model.release == models.SPEECH_RELEASE == "models-v2"
        assert model.directory == models.WHISPER_DIRECTORY
        assert model.relative_path.parent == Path(models.WHISPER_DIRECTORY)
        assert model.url.startswith(
            "https://github.com/brighinatommaso-eng/shotfinder/releases/download/models-v2/"
        )
    assert models.speech_model_dir("C:/models") == Path("C:/models/whisper-base")


def test_speech_ready_is_false_without_the_files(tmp_path: Path) -> None:
    assert models.speech_ready(tmp_path) is False
    assert [model.name for model in models.missing_files(tmp_path, models.SPEECH_MODELS)] == [
        model.name for model in models.SPEECH_MODELS
    ]


def test_hub_is_disabled_for_the_process() -> None:
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


@pytest.mark.slow
def test_transcript_contains_the_spoken_words(
    speech_video: Path, models_dir: Path, tmp_path: Path
) -> None:
    segments = speech.transcribe(
        speech_video,
        models_dir=models_dir,
        work_dir=tmp_path,
        threads=2,
    )

    text = " ".join(segment[2] for segment in segments).lower()
    assert "quick" in text
    assert "fox" in text
    assert segments
    assert all(end >= start >= 0 for start, end, _text in segments)
    assert list(tmp_path.glob("speech-*.wav")) == []


@pytest.mark.slow
def test_temp_wav_is_removed_even_when_transcription_fails(
    speech_video: Path,
    models_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_load(_models_dir, _threads):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(speech, "_load_model", broken_load)

    with pytest.raises(RuntimeError):
        speech.transcribe(
            speech_video,
            models_dir=models_dir,
            work_dir=tmp_path,
            threads=2,
        )

    assert list(tmp_path.glob("speech-*.wav")) == []
