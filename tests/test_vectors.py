"""Vector insert, KNN and cascade behaviour on a temp DB (no models needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shotfinder.index import Index


@pytest.fixture
def index(tmp_path: Path) -> Index:
    instance = Index(tmp_path / "index.db")
    yield instance
    instance.close()


def _vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=512).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _file(index: Index, tmp_path: Path, name: str) -> int:
    path = tmp_path / name
    path.write_bytes(name.encode("ascii"))
    file_id, _ = index.upsert_file(path, "image")
    return file_id


def _count(index: Index, sql: str, parameters: tuple[object, ...] = ()) -> int:
    return int(index.conn.execute(sql, parameters).fetchone()[0])


def test_add_and_search_vectors(index: Index, tmp_path: Path) -> None:
    first = _file(index, tmp_path, "first.png")
    second = _file(index, tmp_path, "second.png")
    third = _file(index, tmp_path, "third.png")

    query = _vector(1)
    near = query * 0.999 + _vector(9) * 0.01
    near = near / np.linalg.norm(near)

    assert index.add_vectors(first, [(None, query)]) == 1
    assert index.add_vectors(second, [(None, _vector(2))]) == 1
    assert index.add_vectors(third, [(None, near)]) == 1

    hits = index.search_vectors(query, k=3)
    assert [hit["file_id"] for hit in hits] == [first, third, second]
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-5)
    assert hits[1]["distance"] < hits[2]["distance"]
    assert 1.0 - hits[1]["distance"] ** 2 / 2 > 0.99
    assert hits[0]["path"] == str(tmp_path / "first.png")
    assert hits[0]["kind"] == "image"


def test_add_vectors_replaces_previous_rows(index: Index, tmp_path: Path) -> None:
    file_id = _file(index, tmp_path, "video-frame.png")
    assert index.add_vectors(file_id, [(1.0, _vector(1)), (2.0, _vector(2))]) == 2
    assert _count(index, "SELECT COUNT(*) FROM vec_meta WHERE file_id = ?", (file_id,)) == 2

    assert index.add_vectors(file_id, [(3.0, _vector(3))]) == 1
    assert _count(index, "SELECT COUNT(*) FROM vec_meta WHERE file_id = ?", (file_id,)) == 1
    assert _count(index, "SELECT COUNT(*) FROM vec_items") == 1

    hit = index.search_vectors(_vector(3), k=1)[0]
    assert hit["file_id"] == file_id
    assert hit["ts"] == 3.0


def test_stale_hash_rejects_vectors(index: Index, tmp_path: Path) -> None:
    file_id = _file(index, tmp_path, "race.png")
    assert (
        index.add_vectors(file_id, [(None, _vector(1))], expected_hash="not-the-hash")
        == 0
    )
    assert _count(index, "SELECT COUNT(*) FROM vec_meta WHERE file_id = ?", (file_id,)) == 0
    assert index.search_vectors(_vector(1), k=5) == []


def test_vectors_are_deleted_with_the_file(index: Index, tmp_path: Path) -> None:
    file_id = _file(index, tmp_path, "delete-me.png")
    index.add_vectors(file_id, [(None, _vector(1))])
    assert _count(index, "SELECT COUNT(*) FROM vec_meta WHERE file_id = ?", (file_id,)) == 1

    index.delete_file(file_id)

    assert _count(index, "SELECT COUNT(*) FROM vec_meta WHERE file_id = ?", (file_id,)) == 0
    assert _count(index, "SELECT COUNT(*) FROM vec_items") == 0


def test_vector_dimension_is_validated(index: Index, tmp_path: Path) -> None:
    file_id = _file(index, tmp_path, "wrong-size.png")
    with pytest.raises(ValueError):
        index.add_vectors(file_id, [(None, [0.0] * 10)])
    with pytest.raises(ValueError):
        index.search_vectors([0.0] * 10)
