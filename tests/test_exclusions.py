"""Tests for the exclusion rules: directory names, project markers, tiny images."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from shotfinder.config import Config
from shotfinder.exclusions import (
    MIN_IMAGE_DIMENSION,
    Exclusions,
    has_project_marker,
    is_small_image,
)
from shotfinder.index import Index
from shotfinder.watcher import iter_media_files
from shotfinder.worker import Indexer


def _image(path: Path, size: tuple[int, int] = (200, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (20, 40, 60)).save(path)
    return path


def _config(tmp_path: Path, *roots: Path) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        watched_folders=list(roots) or [tmp_path],
    )


def test_excluded_names_match_any_component_case_insensitively(
    tmp_path: Path,
) -> None:
    root = tmp_path / "media"
    exclusions = Exclusions(_config(tmp_path, root), [root])

    assert exclusions.excludes_file(root / "node_modules" / "pkg" / "icon.png")
    assert exclusions.excludes_file(root / "Downloads" / "unity" / "LIBRARY" / "x.png")
    assert exclusions.excludes_file(root / "AppData" / "Local" / "x.png")
    assert not exclusions.excludes_file(root / "holidays" / "beach.png")
    assert not exclusions.excludes_file(root / "beach.png")


def test_watched_root_is_exempt_from_name_and_marker_rules(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    (root / ".git").mkdir()
    exclusions = Exclusions(_config(tmp_path, root), [root])

    assert not exclusions.excludes_dir(root)
    assert not exclusions.excludes_file(root / "photo.png")


def test_project_marker_directories_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "media"
    root.mkdir()
    for marker in ("package.json", "pyproject.toml", "app.sln", "app.csproj"):
        project = root / f"project-{marker}"
        project.mkdir()
        (project / marker).write_text("", encoding="utf-8")
        assert has_project_marker(project)
        assert Exclusions(_config(tmp_path, root), [root]).excludes_dir(project)

    unity = root / "unity"
    (unity / "Assets").mkdir(parents=True)
    (unity / "ProjectSettings").mkdir()
    assert has_project_marker(unity)
    assert not has_project_marker(root / "holidays")


def test_scan_skips_caches_and_projects_but_keeps_media(tmp_path: Path) -> None:
    root = tmp_path / "media"
    _image(root / "beach.png")
    _image(root / "Library" / "cache.png")
    _image(root / "node_modules" / "pkg" / "icon.png")
    _image(root / "unity" / "Assets" / "logo.png")
    (root / "unity" / "ProjectSettings").mkdir(parents=True)
    _image(root / "holidays" / "sea.png")

    found = {path.relative_to(root).as_posix() for path in iter_media_files(_config(tmp_path, root))}

    assert found == {"beach.png", "holidays/sea.png"}


def test_small_images_are_detected_without_decoding_pixels(tmp_path: Path) -> None:
    tiny = _image(tmp_path / "tiny.png", (64, 200))
    edge = _image(tmp_path / "edge.png", (MIN_IMAGE_DIMENSION, MIN_IMAGE_DIMENSION))
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")

    assert is_small_image(tiny)
    assert not is_small_image(edge)
    assert not is_small_image(broken)


def test_purge_removes_excluded_and_tiny_rows_only_once(tmp_path: Path) -> None:
    root = tmp_path / "media"
    normal = _image(root / "keep.png")
    junk = _image(root / "node_modules" / "pkg" / "icon.png")
    tiny = _image(root / "tiny.png", (64, 64))
    skipped = _image(root / "already-skipped.png", (64, 64))

    config = _config(tmp_path, root)
    index = Index(config.db_path)
    try:
        normal_id, _ = index.upsert_file(normal, "image")
        junk_id, _ = index.upsert_file(junk, "image")
        tiny_id, _ = index.upsert_file(tiny, "image")
        skipped_id, _ = index.upsert_file(skipped, "image")
        for file_id in (normal_id, junk_id, tiny_id):
            index.mark_status(file_id, "done", stages_done="ocr")
        index.mark_status(skipped_id, "skipped", "image too small (64x64)")

        indexer = Indexer(config, index)

        assert indexer.purge_excluded() == 2
        assert index.get_file(normal_id) is not None
        assert index.get_file(junk_id) is None
        assert index.get_file(tiny_id) is None
        assert index.get_file(skipped_id) is not None
        assert indexer.stats["purged"] == 2

        # A second pass must not report the same rows again.
        assert indexer.purge_excluded() == 0
    finally:
        index.close()


def test_worker_marks_tiny_images_skipped_before_ocr(tmp_path: Path) -> None:
    root = tmp_path / "media"
    tiny = _image(root / "tiny.png", (100, 90))
    config = _config(tmp_path, root)
    index = Index(config.db_path)
    try:
        file_id, _ = index.upsert_file(tiny, "image")
        indexer = Indexer(config, index)

        indexer._process_image(index.get_file(file_id))

        row = index.get_file(file_id)
        assert row["status"] == "skipped"
        assert "too small" in row["error"]
        assert indexer.stats["skipped"] == 1
    finally:
        index.close()
