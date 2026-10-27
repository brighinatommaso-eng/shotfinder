"""RRF merge and hybrid search wiring (fast: no ONNX models loaded)."""

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
    distance_to_cosine,
    merge,
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


def test_file_in_both_lists_wins() -> None:
    text = [_text(1, "a.png"), _text(2, "b.png")]
    vectors = [_vector(2, 0.1, "b.png"), _vector(3, 0.2, "c.png")]

    results = merge(text, vectors)

    assert [row["file_id"] for row in results] == [2, 1, 3]
    assert results[0]["match_type"] == MATCH_BOTH
    assert results[1]["match_type"] == MATCH_TEXT
    assert results[2]["match_type"] == MATCH_VISUAL
    assert results[0]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert results[0]["fts_rank"] == 2
    assert results[0]["cosine"] == pytest.approx(distance_to_cosine(0.1))


def test_duplicate_rows_do_not_inflate_score() -> None:
    text = [_text(1, "a.png"), _text(1, "a.png"), _text(1, "a.png"), _text(2, "b.png")]

    results = merge(text, [])

    assert [row["file_id"] for row in results] == [1, 2]
    assert results[0]["score"] == pytest.approx(1 / 61)
    assert results[1]["score"] == pytest.approx(1 / 62)


def test_best_timestamp_follows_the_better_rank() -> None:
    text = [_text(9, "other.png"), _text(1, "video.mp4", ts=10.0)]
    vectors = [_vector(1, 0.1, "video.mp4", ts=5.0, kind="video")]

    results = merge(text, vectors)

    assert results[0]["file_id"] == 1
    assert results[0]["ts"] == 5.0
    assert results[0]["match_type"] == MATCH_BOTH


def test_snippet_comes_from_the_best_text_row() -> None:
    text = [_text(1, "a.png", snippet="best"), _text(1, "a.png", snippet="worse")]

    results = merge(text, [])

    assert results[0]["snippet"] == "best"


def test_limit_and_empty_inputs() -> None:
    assert merge([], []) == []
    assert merge([_text(1)], [], limit=0) == []

    results = merge([_text(1, "a.png"), _text(2, "b.png")], [], limit=1)
    assert len(results) == 1
    assert results[0]["file_id"] == 1


def test_cosine_conversion() -> None:
    assert distance_to_cosine(0.0) == pytest.approx(1.0)
    assert distance_to_cosine(1.0) == pytest.approx(0.5)
    assert distance_to_cosine(2**0.5) == pytest.approx(0.0, abs=1e-9)


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

    assert [row["file_id"] for row in results] == [file_id]
    assert results[0]["match_type"] == MATCH_RECENT


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

    assert [row["file_id"] for row in results] == [file_id]
    assert results[0]["match_type"] == MATCH_TEXT
    assert results[0]["cosine"] is None
    assert results[0]["fts_rank"] == 1
