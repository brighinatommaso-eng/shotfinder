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


def test_filename_tokens_drop_dates_and_camera_noise() -> None:
    from shotfinder.index import name_tokens

    assert name_tokens("Pulizia disco 2026-09-25 21-19-47.mp4") == [
        "pulizia",
        "disco",
    ]
    assert name_tokens("IMG_2024_ScreenRecording_2026-09-25 21-19-47.png") == [
        "screen",
        "recording",
    ]
    assert name_tokens("camelCaseVideo.mov") == ["camel", "case", "video"]
    assert name_tokens("DSC_0042.JPG") == []


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
    assert hits[0]["hash"] == row["hash"]
    assert hits[0]["mtime"] == row["mtime"]

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
    assert (
        _count(
            index,
            "SELECT COUNT(*) FROM text_fts WHERE file_id = ? AND source != 'name'",
            (file_id,),
        )
        == 0
    )


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

    assert (
        _count(
            index,
            "SELECT COUNT(*) FROM text_fts WHERE file_id = ? AND source != 'name'",
            (file_id,),
        )
        == 1
    )
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


def test_vision_backfill_lists_only_done_images_without_vectors(
    index: Index, tmp_path: Path
) -> None:
    done_ocr = _make_file(tmp_path / "ocr-only.png", b"ocr only")
    done_vision = _make_file(tmp_path / "embedded.png", b"embedded")
    pending = _make_file(tmp_path / "pending.png", b"pending")
    video = _make_file(tmp_path / "clip.mp4", b"video")

    ocr_id, _ = index.upsert_file(done_ocr, "image")
    vision_id, _ = index.upsert_file(done_vision, "image")
    pending_id, _ = index.upsert_file(pending, "image")
    video_id, _ = index.upsert_file(video, "video")

    index.mark_status(ocr_id, "done", stages_done="ocr")
    index.mark_status(vision_id, "done", stages_done="ocr,vision")
    index.mark_status(video_id, "done", stages_done="frames")

    backfill = index.next_vision_backfill(10)

    assert [row["id"] for row in backfill] == [ocr_id]
    assert pending_id not in {row["id"] for row in backfill}


def test_filename_words_are_searchable(index: Index, tmp_path: Path) -> None:
    path = _make_file(tmp_path / "Pulizia disco 2026-09-25 21-19-47.mp4", b"video")
    file_id, _ = index.upsert_file(path, "video")

    hits = index.search_text("pulizia disco")

    assert [hit["file_id"] for hit in hits] == [file_id]
    assert hits[0]["source"] == "name"
    assert hits[0]["ts"] is None
    assert "pulizia" in hits[0]["snippet"]


def test_rename_refreshes_the_name_row(index: Index, tmp_path: Path) -> None:
    old = _make_file(tmp_path / "alpha.mp4", b"same bytes")
    file_id, _ = index.upsert_file(old, "video")

    new = tmp_path / "beta.mp4"
    old.rename(new)
    moved_id, changed = index.upsert_file(new, "video")

    assert moved_id == file_id
    assert changed is False
    assert index.search_text("alpha") == []
    assert index.search_text("beta")[0]["file_id"] == file_id


def test_backfill_names_adds_rows_for_old_files(index: Index, tmp_path: Path) -> None:
    path = _make_file(tmp_path / "vecchio nome.png", b"image")
    file_id, _ = index.upsert_file(path, "image")
    with index.conn as connection:
        connection.execute("DELETE FROM text_fts WHERE file_id = ?", (file_id,))

    assert index.search_text("vecchio") == []
    assert index.backfill_names() == 1
    assert index.search_text("vecchio")[0]["file_id"] == file_id
    assert index.backfill_names() == 0


def _video_with_coverage(
    index: Index,
    path: Path,
    *,
    complete: int = 1,
    has_audio: int = 1,
    duration: float = 100.0,
    speech: str | None = None,
) -> int:
    path.write_bytes(path.name.encode())
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
            "complete": complete,
            "has_audio": has_audio,
            "speech": speech,
        },
    )
    index.mark_status(file_id, "done", stages_done="frames")
    return file_id


def test_speech_pending_skips_silent_long_and_finished_videos(
    index: Index, tmp_path: Path
) -> None:
    ready = _video_with_coverage(index, tmp_path / "ready.mp4")
    silent = _video_with_coverage(index, tmp_path / "silent.mp4", has_audio=0)
    long = _video_with_coverage(
        index, tmp_path / "long.mp4", duration=7200.0
    )
    done = _video_with_coverage(
        index, tmp_path / "done.mp4", speech="done"
    )
    unfinished = _video_with_coverage(
        index, tmp_path / "partial.mp4", complete=0
    )

    assert index.next_speech_pending(3600) == [ready]
    # 0 disables the duration cap, but not the audio or state filters.
    assert index.next_speech_pending(0) == [ready, long]
    assert silent not in index.next_speech_pending(0)
    assert done not in index.next_speech_pending(0)
    assert unfinished not in index.next_speech_pending(0)

    # Indexed before coverage existed: no row yet, but frames are done.
    legacy = _make_file(tmp_path / "legacy.mp4", b"legacy")
    legacy_id, _ = index.upsert_file(legacy, "video")
    index.mark_status(legacy_id, "done", stages_done="frames")
    assert legacy_id in index.next_speech_pending(3600)


def test_truncated_videos_reset_only_the_frames_stage(
    index: Index, tmp_path: Path
) -> None:
    path = _make_file(tmp_path / "long.mp4", b"long video")
    file_id, _ = index.upsert_file(path, "video")
    index.add_video_frames(
        file_id,
        [(0.0, [0.0] * 512)],
        next_ts=1500.0,
        kept=300,
        last_hash=None,
        coverage={
            "duration": 7200.0,
            "interval": 24.0,
            "sampled": 300,
            "kept": 300,
            "last_ts": 7176.0,
            "complete": 0,
            "has_audio": 1,
            "speech": "done",
            "speech_segments": 4,
        },
    )
    index.add_text_rows(file_id, [("keep this line", 12.0)], source="speech")
    index.mark_status(file_id, "done", stages_done="frames,speech")

    assert index.reset_truncated_videos() == [file_id]

    row = index.get_file(file_id)
    assert row["status"] == "pending"
    assert row["stages_done"] == "speech"
    coverage = index.video_coverage(file_id)
    assert coverage["complete"] == 0
    assert coverage["duration"] is None
    assert coverage["speech"] == "done"
    # Speech text survives: only frames are re-sampled.
    assert index.search_text("keep")[0]["file_id"] == file_id
    # While it is pending it is not reset a second time.
    assert index.reset_truncated_videos() == []


def test_legacy_videos_with_300_vectors_are_treated_as_truncated(
    index: Index, tmp_path: Path
) -> None:
    path = _make_file(tmp_path / "legacy-long.mp4", b"legacy long")
    file_id, _ = index.upsert_file(path, "video")
    index.add_video_frames(
        file_id,
        [(float(i), [0.0] * 512) for i in range(300)],
        next_ts=1500.0,
        kept=300,
        last_hash=None,
    )
    index.mark_status(file_id, "done", stages_done="frames")

    assert index.reset_truncated_videos() == [file_id]
    assert index.video_progress(file_id) is None
    assert index.get_file(file_id)["status"] == "pending"
