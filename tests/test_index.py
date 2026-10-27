"""Tests for the SQLite index: insert, move-by-hash, cascade delete and FTS."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlite_vec

from shotfinder.index import Index, compute_file_hash


@pytest.fixture
def index(tmp_path: Path) -> Index:
    instance = Index(tmp_path / "index.db")
    yield instance
    instance.close()


def _make_file(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _count(index: Index, sql: str, parameters: tuple[object, ...] = ()) -> int:
    return int(index.conn.execute(sql, parameters).fetchone()[0])


def test_insert_and_search(index: Index, tmp_path: Path) -> None:
    path = _make_file(tmp_path / "screenshot.png", b"first file")
    file_id, changed = index.upsert_file(path, "image")
    assert changed is True

    row = index.get_file(file_id)
    assert row is not None
    assert row["path"] == str(path)
    assert row["status"] == "pending"
    assert row["stages_done"] == ""

    inserted = index.add_text_rows(
        file_id, ["The train to Milano departs at 14:32"], source="ocr"
    )
    assert inserted == 1
    index.mark_status(file_id, "done", stages_done="ocr")

    hits = index.search_text("train Milano")
    assert len(hits) == 1
    assert hits[0]["file_id"] == file_id
    assert hits[0]["source"] == "ocr"
    assert "Milano" in hits[0]["snippet"]

    assert index.search_text("tra mil")
    assert index.search_text("treno") == []
    assert index.search_text("   ") == []
    assert index.count_files() == 1
    assert index.count_pending() == 0


def test_unchanged_rescan_does_not_hash(
    index: Index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _make_file(tmp_path / "unchanged.png", b"stable bytes")
    file_id, _ = index.upsert_file(path, "image")

    calls: list[Path] = []

    def fail_if_called(candidate: str | Path) -> str:
        calls.append(Path(candidate))
        raise AssertionError("compute_file_hash should not run for an unchanged file")

    monkeypatch.setattr("shotfinder.index.compute_file_hash", fail_if_called)

    same_id, changed = index.upsert_file(path, "image")
    assert same_id == file_id
    assert changed is False
    assert calls == []


def test_move_by_hash_keeps_row_and_text(index: Index, tmp_path: Path) -> None:
    source = _make_file(tmp_path / "before.png", b"identical content")
    file_id, _ = index.upsert_file(source, "image")
    index.add_text_rows(file_id, ["unique searchable phrase"], source="ocr")
    index.mark_status(file_id, "done", stages_done="ocr")
    digest = index.get_file(file_id)["hash"]
    assert index.find_by_hash(digest)["id"] == file_id

    destination = tmp_path / "after.png"
    source.rename(destination)

    moved_id, changed = index.upsert_file(destination, "image")
    assert moved_id == file_id
    assert changed is False

    row = index.get_file(file_id)
    assert row is not None
    assert row["path"] == str(destination)
    assert row["status"] == "done"
    assert row["stages_done"] == "ocr"
    assert index.find_by_path(source) is None
    assert index.search_text("unique")[0]["file_id"] == file_id


def test_copy_with_same_hash_gets_its_own_row(index: Index, tmp_path: Path) -> None:
    source = _make_file(tmp_path / "original.png", b"same bytes")
    source_id, _ = index.upsert_file(source, "image")

    copy = _make_file(tmp_path / "copy.png", b"same bytes")
    copy_id, changed = index.upsert_file(copy, "image")

    assert changed is True
    assert copy_id != source_id
    assert index.get_file(source_id)["path"] == str(source)
    assert index.get_file(copy_id)["path"] == str(copy)


def test_modified_file_resets_stages_and_text(index: Index, tmp_path: Path) -> None:
    path = _make_file(tmp_path / "changing.png", b"version one")
    file_id, _ = index.upsert_file(path, "image")
    index.add_text_rows(file_id, ["old words"], source="ocr")
    index.mark_status(file_id, "done", stages_done="ocr")

    _make_file(path, b"version two, longer")
    same_id, changed = index.upsert_file(path, "image")

    assert same_id == file_id
    assert changed is True
    row = index.get_file(file_id)
    assert row["status"] == "pending"
    assert row["stages_done"] == ""
    assert row["error"] is None
    assert index.search_text("old") == []
    assert _count(index, "SELECT COUNT(*) FROM text_fts WHERE file_id = ?", (file_id,)) == 0


def test_delete_cascades_to_text_and_vectors(index: Index, tmp_path: Path) -> None:
    path = _make_file(tmp_path / "delete-me.png", b"delete me")
    file_id, _ = index.upsert_file(path, "image")
    index.add_text_rows(file_id, ["text that must disappear"], source="ocr")

    with index.conn as connection:
        cursor = connection.execute(
            "INSERT INTO vec_meta (file_id, ts) VALUES (?, NULL)", (file_id,)
        )
        vector_rowid = int(cursor.lastrowid)
        connection.execute(
            "INSERT INTO vec_items (rowid, embedding) VALUES (?, ?)",
            (vector_rowid, sqlite_vec.serialize_float32([0.0] * 512)),
        )

    assert _count(index, "SELECT COUNT(*) FROM text_fts WHERE file_id = ?", (file_id,)) == 1
    assert _count(index, "SELECT COUNT(*) FROM vec_meta WHERE file_id = ?", (file_id,)) == 1
    assert _count(index, "SELECT COUNT(*) FROM vec_items WHERE rowid = ?", (vector_rowid,)) == 1

    index.delete_file(file_id)

    assert index.get_file(file_id) is None
    assert index.find_by_hash(compute_file_hash(path)) is None
    assert _count(index, "SELECT COUNT(*) FROM text_fts WHERE file_id = ?", (file_id,)) == 0
    assert _count(index, "SELECT COUNT(*) FROM vec_meta WHERE file_id = ?", (file_id,)) == 0
    assert _count(index, "SELECT COUNT(*) FROM vec_items WHERE rowid = ?", (vector_rowid,)) == 0


def test_stale_extractor_result_is_rejected(index: Index, tmp_path: Path) -> None:
    path = _make_file(tmp_path / "race.png", b"original")
    file_id, _ = index.upsert_file(path, "image")

    assert (
        index.add_text_rows(
            file_id, ["stale text"], source="ocr", expected_hash="not-the-hash"
        )
        == 0
    )
    assert index.search_text("stale") == []
    assert index.mark_status(file_id, "done", expected_hash="not-the-hash") is False
    assert index.get_file(file_id)["status"] == "pending"


def test_recent_files_only_lists_done_files(index: Index, tmp_path: Path) -> None:
    first = _make_file(tmp_path / "first.png", b"first")
    second = _make_file(tmp_path / "second.png", b"second")
    first_id, _ = index.upsert_file(first, "image")
    second_id, _ = index.upsert_file(second, "image")
    index.mark_status(first_id, "done", stages_done="ocr")

    recent = index.recent_files(10)
    assert [row["file_id"] for row in recent] == [first_id]
    assert second_id not in {row["file_id"] for row in recent}
