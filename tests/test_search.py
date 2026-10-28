"""RRF fusion, the matches/related split and hybrid search wiring.

Fast: no ONNX models are loaded, so the visual path is exercised through
``combine`` and ``filter_vector_rows`` directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shotfinder.config import Config
from shotfinder.index import Index
from shotfinder.search import (
    MATCH_BOTH,
    MATCH_RECENT,
    MATCH_TEXT,
    MATCH_VISUAL,
    combine,
    distance_to_cosine,
    filter_vector_rows,
    search,
)


def _text(
    file_id: int, path: str = "file.png", ts: float | None = None, snippet: str = ""
) -> dict[str, object]:
    return {
        "file_id": file_id,
        "path": path,
        "kind": "image",
        "ts": ts,
        "snippet": snippet,
        "source": "ocr",
        "score": -1.0,
    }


def _vector(
    file_id: int,
    distance: float,
    path: str = "file.png",
    ts: float | None = None,
    kind: str = "image",
) -> dict[str, object]:
    return {
        "file_id": file_id,
        "path": path,
        "kind": kind,
        "ts": ts,
        "distance": distance,
    }


def test_file_in_both_lists_is_a_match_and_wins() -> None:
    text = [_text(1, "a.png"), _text(2, "b.png")]
    vectors = [_vector(2, 0.1, "b.png"), _vector(3, 0.2, "c.png")]

    results = combine(text, vectors)

    assert [row["file_id"] for row in results.matches] == [2, 1]
    assert [row["file_id"] for row in results.related] == [3]
    assert results.matches[0]["match_type"] == MATCH_BOTH
    assert results.matches[1]["match_type"] == MATCH_TEXT
    assert results.related[0]["match_type"] == MATCH_VISUAL
    assert results.matches[0]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert results.matches[0]["fts_rank"] == 2
    assert results.matches[0]["cosine"] == pytest.approx(distance_to_cosine(0.1))


def test_visual_only_hits_never_reach_the_match_list() -> None:
    vectors = [_vector(1, 0.2, "a.png"), _vector(2, 0.3, "b.png")]

    results = combine([], vectors)

    assert results.matches == []
    assert [row["file_id"] for row in results.related] == [1, 2]


def test_related_is_ordered_by_cosine_not_by_rrf() -> None:
    vectors = [
        _vector(1, _distance_for(0.22), "a.png"),
        _vector(2, _distance_for(0.40), "b.png"),
        _vector(3, _distance_for(0.31), "c.png"),
    ]

    results = combine([], vectors)

    assert [row["file_id"] for row in results.related] == [2, 3, 1]
    assert results.related[0]["cosine"] == pytest.approx(0.40)


def test_related_is_capped_but_matches_are_not() -> None:
    text = [_text(index, f"t{index}.png") for index in range(1, 13)]
    vectors = [
        _vector(index, _distance_for(0.30 - index / 100), f"v{index}.png")
        for index in range(1, 21)
    ]

    results = combine(text, vectors, max_related=8)

    assert len(results.matches) == 12
    assert len(results.related) == 8
    assert len(results.all()) == 20


def test_limit_applies_to_matches_only() -> None:
    text = [_text(1, "a.png"), _text(2, "b.png")]
    vectors = [_vector(9, 0.2, "z.png")]

    results = combine(text, vectors, limit=1)

    assert [row["file_id"] for row in results.matches] == [1]
    assert [row["file_id"] for row in results.related] == [9]


def test_duplicate_rows_do_not_inflate_score() -> None:
    text = [_text(1, "a.png"), _text(1, "a.png"), _text(1, "a.png"), _text(2, "b.png")]

    results = combine(text, [])

    assert [row["file_id"] for row in results.matches] == [1, 2]
    assert results.matches[0]["score"] == pytest.approx(1 / 61)
    assert results.matches[1]["score"] == pytest.approx(1 / 62)


def test_timed_text_owns_the_jump_point() -> None:
    text = [_text(9, "other.png"), _text(1, "video.mp4", ts=10.0)]
    vectors = [_vector(1, 0.1, "video.mp4", ts=5.0, kind="video")]

    results = combine(text, vectors)

    assert results.matches[0]["file_id"] == 1
    # A spoken line is what the user asked for, even when a frame ranks higher.
    assert results.matches[0]["ts"] == 10.0
    assert results.matches[0]["match_type"] == MATCH_BOTH


def test_video_related_keeps_the_best_frame_timestamp() -> None:
    # Rows arrive ordered by distance, so the first row per file is the best frame.
    vectors = [
        _vector(1, _distance_for(0.42), "clip.mp4", ts=120.0, kind="video"),
        _vector(1, _distance_for(0.30), "clip.mp4", ts=35.0, kind="video"),
        _vector(2, _distance_for(0.25), "other.mp4", ts=10.0, kind="video"),
    ]

    results = combine([], vectors)

    assert [row["file_id"] for row in results.related] == [1, 2]
    assert results.related[0]["ts"] == 120.0


def test_snippet_comes_from_the_best_text_row() -> None:
    text = [_text(1, "a.png", snippet="best"), _text(1, "a.png", snippet="worse")]

    results = combine(text, [])

    assert results.matches[0]["snippet"] == "best"


def test_combine_carries_hash_and_mtime_for_thumbnails() -> None:
    text = [{**_text(1, "a.png"), "hash": "abc", "mtime": 123.0}]
    vectors = [{**_vector(1, 0.1, "a.png"), "hash": "abc", "mtime": 123.0}]

    results = combine(text, vectors)

    assert results.matches[0]["hash"] == "abc"
    assert results.matches[0]["mtime"] == 123.0


def test_empty_inputs() -> None:
    results = combine([], [])

    assert not results
    assert results.matches == []
    assert results.related == []
    assert combine([_text(1)], [], limit=0).matches == []


def test_cosine_conversion() -> None:
    assert distance_to_cosine(0.0) == pytest.approx(1.0)
    assert distance_to_cosine(1.0) == pytest.approx(0.5)
    assert distance_to_cosine(2**0.5) == pytest.approx(0.0, abs=1e-9)


def _distance_for(cosine: float) -> float:
    return (2.0 * (1.0 - cosine)) ** 0.5


def test_vector_hits_below_the_garbage_floor_are_dropped() -> None:
    rows = [
        _vector(1, _distance_for(0.31)),
        _vector(2, _distance_for(0.24)),
        _vector(3, _distance_for(0.19)),
    ]

    kept = filter_vector_rows(rows, 0.20, margin=0.2)

    assert [row["file_id"] for row in kept] == [1, 2]


def test_relative_margin_drops_the_weak_tail() -> None:
    rows = [
        _vector(1, _distance_for(0.31)),
        _vector(2, _distance_for(0.27)),
        _vector(3, _distance_for(0.25)),
    ]

    # Best 0.31, margin 0.05 -> cutoff 0.26, so 0.25 goes.
    kept = filter_vector_rows(rows, 0.10)

    assert [row["file_id"] for row in kept] == [1, 2]


def test_no_visual_hits_when_everything_is_noise() -> None:
    rows = [_vector(1, _distance_for(0.24)), _vector(2, _distance_for(0.19))]

    assert filter_vector_rows(rows, 0.20) != []
    assert filter_vector_rows(rows, 0.30) == []
    assert filter_vector_rows([], 0.20) == []


def test_empty_query_returns_recent_files(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path)
    index = Index(config.db_path)
    try:
        path = tmp_path / "recent.png"
        path.write_bytes(b"recent")
        file_id, _ = index.upsert_file(path, "image")
        index.mark_status(file_id, "done", stages_done="ocr")

        results = search(config, index, "   ")
    finally:
        index.close()

    assert [row["file_id"] for row in results.matches] == [file_id]
    assert results.related == []
    assert results.matches[0]["match_type"] == MATCH_RECENT
    assert results.matches[0]["hash"]
    assert results.matches[0]["mtime"] is not None


def test_text_only_when_models_are_missing(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path)
    index = Index(config.db_path)
    try:
        path = tmp_path / "train.png"
        path.write_bytes(b"train")
        file_id, _ = index.upsert_file(path, "image")
        index.add_text_rows(file_id, ["The train to Milano departs at 14:32"], source="ocr")
        index.mark_status(file_id, "done", stages_done="ocr")

        results = search(config, index, "Milano")
    finally:
        index.close()

    assert [row["file_id"] for row in results.matches] == [file_id]
    assert results.matches[0]["match_type"] == MATCH_TEXT
    assert results.matches[0]["cosine"] is None
    assert results.matches[0]["fts_rank"] == 1


def test_filename_hit_is_a_match_not_a_related_suggestion(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path)
    index = Index(config.db_path)
    try:
        path = tmp_path / "Pulizia disco 2026-09-25 21-19-47.mp4"
        path.write_bytes(b"video")
        file_id, _ = index.upsert_file(path, "video")
        index.mark_status(file_id, "done", stages_done="frames")

        results = search(config, index, "pulizia disco")
    finally:
        index.close()

    assert [row["file_id"] for row in results.matches] == [file_id]
    assert results.matches[0]["match_type"] == MATCH_TEXT
    assert results.matches[0]["source"] == "name"
    assert results.related == []


def test_spoken_hit_keeps_its_timestamp_and_snippet(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path)
    index = Index(config.db_path)
    try:
        path = tmp_path / "tutorial.mp4"
        path.write_bytes(b"video")
        file_id, _ = index.upsert_file(path, "video")
        index.add_text_rows(
            file_id,
            [
                ("Welcome to this tutorial", 0.0),
                ("Now we solder the battery connector", 754.0),
            ],
            source="speech",
        )
        index.mark_status(file_id, "done", stages_done="frames,speech")

        results = search(config, index, "battery connector")
    finally:
        index.close()

    match = results.matches[0]
    assert match["file_id"] == file_id
    assert match["match_type"] == MATCH_TEXT
    assert match["source"] == "speech"
    assert match["ts"] == pytest.approx(754.0)
    assert "battery" in match["snippet"].lower()
