"""Worker speech stage: eligibility, storage and failure recovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from shotfinder.config import Config
from shotfinder.index import Index
from shotfinder.worker import (
    SPEECH_NO_AUDIO_REASON,
    SPEECH_TOO_LONG_REASON,
    Indexer,
)


def _setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    has_audio: int = 1,
    duration: float = 100.0,
    max_minutes: int = 60,
) -> tuple[Indexer, Index, int]:
    config = Config(
        data_dir=tmp_path,
        speech_max_minutes=max_minutes,
        max_retries=0,
    )
    index = Index(config.db_path)
    path = tmp_path / "tutorial.mp4"
    path.write_bytes(b"fake video bytes")
    file_id, _ = index.upsert_file(path, "video")
    index.add_video_frames(
        file_id,
        [],
        next_ts=duration,
        kept=1,
        last_hash=None,
        coverage={
            "duration": duration,
            "interval": 5.0,
            "sampled": 1,
            "kept": 1,
            "last_ts": 0.0,
            "complete": 1,
            "has_audio": has_audio,
        },
    )
    index.mark_status(file_id, "done", stages_done="frames")
    monkeypatch.setattr("shotfinder.worker.models.speech_ready", lambda _dir: True)
    return Indexer(config, index), index, file_id


def _speech_rows(index: Index, file_id: int) -> list[tuple[str, float | None]]:
    rows = index.conn.execute(
        """
        SELECT content, ts FROM text_fts
         WHERE file_id = ? AND source = 'speech'
         ORDER BY ts
        """,
        (file_id,),
    ).fetchall()
    return [(row["content"], row["ts"]) for row in rows]


def test_speech_segments_are_stored_with_their_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexer, index, file_id = _setup(tmp_path, monkeypatch)
    calls: dict[str, object] = {}

    def fake_transcribe(path: Path, **kwargs):
        calls["path"] = path
        calls.update(kwargs)
        return [
            (0.0, 2.0, "Welcome to the tutorial"),
            (754.0, 758.0, "Now we solder the battery connector"),
        ]

    monkeypatch.setattr(
        "shotfinder.extractors.speech.transcribe", fake_transcribe
    )
    try:
        assert indexer._process_speech(index.get_file(file_id)) is None

        assert _speech_rows(index, file_id) == [
            ("Welcome to the tutorial", 0.0),
            ("Now we solder the battery connector", 754.0),
        ]
        coverage = index.video_coverage(file_id)
        assert coverage["speech"] == "done"
        assert coverage["speech_segments"] == 2
        row = index.get_file(file_id)
        assert row["status"] == "done"
        assert row["stages_done"] == "frames,speech"
        assert calls["path"] == Path(row["path"])
        assert calls["work_dir"] == indexer.config.data_dir
        assert indexer.stats["activity"] == ""
    finally:
        index.close()


def test_silent_videos_are_skipped_without_queueing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexer, index, file_id = _setup(tmp_path, monkeypatch, has_audio=0)
    try:
        indexer._maybe_enqueue_speech(file_id, index.get_file(file_id))

        coverage = index.video_coverage(file_id)
        assert coverage["speech"] == "skipped"
        assert coverage["speech_reason"] == SPEECH_NO_AUDIO_REASON
        assert indexer.stats["outstanding"] == 0
    finally:
        index.close()


def test_long_videos_are_skipped_by_the_duration_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexer, index, file_id = _setup(
        tmp_path, monkeypatch, duration=7200.0, max_minutes=60
    )
    try:
        indexer._maybe_enqueue_speech(file_id, index.get_file(file_id))

        coverage = index.video_coverage(file_id)
        assert coverage["speech"] == "skipped"
        assert coverage["speech_reason"] == SPEECH_TOO_LONG_REASON
        assert indexer.stats["outstanding"] == 0
    finally:
        index.close()


def test_speech_is_queued_behind_frames_and_images(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path)
    index = Index(config.db_path)
    indexer = Indexer(config, index)
    try:
        indexer._push_speech(1)
        indexer._push(2, "video")
        indexer._push(3, "image")

        items = [indexer._queue.get() for _ in range(3)]
        order = [(item[2], item[3]) for item in items]

        assert order == [(3, "media"), (2, "media"), (1, "speech")]
    finally:
        index.close()


def test_backfill_queues_only_eligible_videos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexer, index, file_id = _setup(tmp_path, monkeypatch)
    silent_path = tmp_path / "silent.mp4"
    silent_path.write_bytes(b"silent")
    silent_id, _ = index.upsert_file(silent_path, "video")
    index.add_video_frames(
        silent_id,
        [],
        next_ts=10.0,
        kept=1,
        last_hash=None,
        coverage={
            "duration": 10.0,
            "interval": 5.0,
            "sampled": 1,
            "kept": 1,
            "last_ts": 0.0,
            "complete": 1,
            "has_audio": 0,
        },
    )
    index.mark_status(silent_id, "done", stages_done="frames")
    try:
        assert indexer.backfill_speech() == 1

        _priority, _sequence, queued_id, stage = indexer._queue.get()
        assert (queued_id, stage) == (file_id, "speech")
        assert indexer.stats["outstanding"] == 1
    finally:
        index.close()


def test_a_failed_transcription_stays_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexer, index, file_id = _setup(tmp_path, monkeypatch)

    def broken_transcribe(*_args, **_kwargs):
        raise RuntimeError("ctranslate2 exploded")

    monkeypatch.setattr(
        "shotfinder.extractors.speech.transcribe", broken_transcribe
    )
    try:
        assert indexer._process_speech(index.get_file(file_id)) is None

        coverage = index.video_coverage(file_id)
        # NULL means "not done yet", so a restart retries this video.
        assert coverage["speech"] is None
        assert _speech_rows(index, file_id) == []
        assert indexer.stats["activity"] == ""
    finally:
        index.close()
