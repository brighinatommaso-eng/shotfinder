"""Worker video turns: per-turn budget, durable resume, atomic progress."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from PIL import Image

from shotfinder.config import Config
from shotfinder.extractors.video import Frame, FrameCursor
from shotfinder.index import Index
from shotfinder.worker import (
    VIDEO_MODELS_REASON,
    VIDEO_NOT_IMPLEMENTED_REASON,
    Indexer,
)

UNIT_VECTOR = [1.0] + [0.0] * 511


def _fake_frames(
    path: Path,
    cursor: FrameCursor,
    *,
    interval: float = 5.0,
    max_side: int = 448,
    max_frames: int | None = None,
) -> Iterator[Frame]:
    """Three frames at 0, 5 and 10 s; the generator mutates the cursor."""
    plan = [0.0, 5.0, 10.0]
    yielded = 0
    while cursor.kept < len(plan):
        if max_frames is not None and yielded >= max_frames:
            cursor.exhausted = False
            return
        timestamp = plan[cursor.kept]
        cursor.next_ts = timestamp + interval
        cursor.kept += 1
        cursor.last_hash = f"{cursor.kept:016x}"
        yielded += 1
        image = Image.new("RGB", (8, 8), (cursor.kept * 20, 0, 0))
        yield Frame(timestamp, image, cursor.kept)
    cursor.exhausted = True


def _fake_embed(images, **_kwargs) -> list[list[float]]:
    return [list(UNIT_VECTOR) for _ in images]


@pytest.fixture
def video_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = Config(data_dir=tmp_path, video_frames_per_turn=1, max_retries=0)
    index = Index(config.db_path)
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fake video bytes")
    file_id, _ = index.upsert_file(path, "video")
    monkeypatch.setattr("shotfinder.worker.models.models_ready", lambda _dir: True)
    monkeypatch.setattr("shotfinder.worker.iter_frames", _fake_frames)
    monkeypatch.setattr(
        "shotfinder.extractors.vision.embed_images", _fake_embed
    )
    indexer = Indexer(config, index)
    yield indexer, index, file_id
    index.close()


def _timestamps(index: Index, file_id: int) -> list[float]:
    rows = index.conn.execute(
        "SELECT ts FROM vec_meta WHERE file_id = ? ORDER BY ts", (file_id,)
    ).fetchall()
    return [float(row["ts"]) for row in rows]


def test_frames_accumulate_one_turn_at_a_time(video_setup) -> None:
    indexer, index, file_id = video_setup

    assert indexer._process_video(index.get_file(file_id)) == "video"
    progress = index.video_progress(file_id)
    assert progress is not None
    assert progress["kept"] == 1
    assert progress["next_ts"] == 5.0
    assert _timestamps(index, file_id) == [0.0]

    assert indexer._process_video(index.get_file(file_id)) == "video"
    assert _timestamps(index, file_id) == [0.0, 5.0]

    assert indexer._process_video(index.get_file(file_id)) is None
    row = index.get_file(file_id)
    assert row["status"] == "done"
    assert row["stages_done"] == "frames"
    assert index.video_progress(file_id) is None
    assert _timestamps(index, file_id) == [0.0, 5.0, 10.0]
    assert indexer.stats["processed"] == 1

    coverage = index.video_coverage(file_id)
    assert coverage is not None
    assert coverage["complete"] == 1
    assert coverage["sampled"] == 3
    assert coverage["kept"] == 3
    assert coverage["last_ts"] == 10.0


def test_progress_survives_a_restart(video_setup) -> None:
    indexer, index, file_id = video_setup
    assert indexer._process_video(index.get_file(file_id)) == "video"

    # A new worker (new process, same index) must resume after the first frame.
    restarted = Indexer(indexer.config, index)
    assert restarted._process_video(index.get_file(file_id)) == "video"
    assert restarted._process_video(index.get_file(file_id)) is None

    assert _timestamps(index, file_id) == [0.0, 5.0, 10.0]
    assert index.get_file(file_id)["status"] == "done"


def test_a_stale_turn_does_not_write_vectors_or_progress(video_setup) -> None:
    _indexer, index, file_id = video_setup

    assert (
        index.add_video_frames(
            file_id,
            [(1.0, UNIT_VECTOR)],
            next_ts=6.0,
            kept=1,
            last_hash="abc",
            expected_hash="not-the-hash",
        )
        is False
    )
    assert _timestamps(index, file_id) == []
    assert index.video_progress(file_id) is None

    assert index.add_video_frames(
        file_id,
        [(1.0, UNIT_VECTOR)],
        next_ts=6.0,
        kept=1,
        last_hash="abc",
        expected_hash=index.get_file(file_id)["hash"],
    )
    assert _timestamps(index, file_id) == [1.0]
    assert index.video_progress(file_id)["kept"] == 1


def test_old_phase4_skips_are_queued_again(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path)
    index = Index(config.db_path)
    path = tmp_path / "old.mp4"
    path.write_bytes(b"old video")
    file_id, _ = index.upsert_file(path, "video")
    index.mark_status(file_id, "skipped", VIDEO_NOT_IMPLEMENTED_REASON)
    try:
        reset = index.reset_skipped_videos(VIDEO_NOT_IMPLEMENTED_REASON)

        assert reset == [file_id]
        row = index.get_file(file_id)
        assert row["status"] == "pending"
        assert row["error"] is None
        assert row["stages_done"] == ""
    finally:
        index.close()


def test_videos_wait_for_models_then_backfill(
    video_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexer, index, file_id = video_setup
    monkeypatch.setattr("shotfinder.worker.models.models_ready", lambda _dir: False)

    assert indexer._process_video(index.get_file(file_id)) is None
    row = index.get_file(file_id)
    assert row["status"] == "skipped"
    assert row["error"] == VIDEO_MODELS_REASON
    assert indexer.stats["skipped"] == 1

    monkeypatch.setattr("shotfinder.worker.models.models_ready", lambda _dir: True)
    assert indexer.backfill_vision() == 1
    assert index.get_file(file_id)["status"] == "pending"
    assert indexer.stats["outstanding"] == 1


def test_images_are_dequeued_before_videos(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path)
    index = Index(config.db_path)
    indexer = Indexer(config, index)
    try:
        indexer._push(1, "video")
        indexer._push(2, "image")
        indexer._push(3, "other")

        order = [indexer._queue.get()[2] for _ in range(3)]

        assert order == [2, 1, 3]
    finally:
        index.close()
