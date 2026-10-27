"""Thumbnail cache: content-hash keying, reuse and broken sources."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from shotfinder.ui.thumbnails import THUMBNAIL_SIZE, ensure_thumbnail, thumbnail_path


def _image(path: Path, size: tuple[int, int] = (800, 600)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (30, 120, 200)).save(path)
    return path


def test_cache_path_is_keyed_by_the_content_hash(tmp_path: Path) -> None:
    assert thumbnail_path(tmp_path, "abc123") == tmp_path / "abc123.jpg"
    assert thumbnail_path(tmp_path, "abc123", 128) == tmp_path / "abc123.jpg"


def test_thumbnail_is_created_at_256_pixels(tmp_path: Path) -> None:
    source = _image(tmp_path / "wide.png", (1200, 400))
    cache = tmp_path / "thumbs"

    target = ensure_thumbnail(source, "hash-one", cache)

    assert target == cache / "hash-one.jpg"
    assert target is not None and target.is_file()
    with Image.open(target) as thumbnail:
        assert max(thumbnail.size) == THUMBNAIL_SIZE
        assert thumbnail.format == "JPEG"


def test_cached_thumbnail_is_reused_without_reading_the_source(tmp_path: Path) -> None:
    source = _image(tmp_path / "photo.png")
    cache = tmp_path / "thumbs"
    first = ensure_thumbnail(source, "same-hash", cache)
    assert first is not None
    first_bytes = first.read_bytes()

    # A moved/copied file keeps the same hash: the cache must not regenerate.
    source.unlink()
    second = ensure_thumbnail(source, "same-hash", cache)

    assert second == first
    assert second.read_bytes() == first_bytes


def test_different_hashes_get_different_files(tmp_path: Path) -> None:
    first_source = _image(tmp_path / "a.png", (300, 300))
    second_source = _image(tmp_path / "b.png", (300, 300))
    cache = tmp_path / "thumbs"

    first = ensure_thumbnail(first_source, "hash-a", cache)
    second = ensure_thumbnail(second_source, "hash-b", cache)

    assert first is not None and second is not None
    assert first.name == "hash-a.jpg"
    assert second.name == "hash-b.jpg"
    assert first != second


def test_broken_source_returns_none_and_leaves_no_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"this is not an image")
    cache = tmp_path / "thumbs"

    assert ensure_thumbnail(broken, "broken-hash", cache) is None
    assert not (cache / "broken-hash.jpg").exists()


def test_missing_source_returns_none(tmp_path: Path) -> None:
    assert ensure_thumbnail(tmp_path / "missing.png", "gone", tmp_path / "thumbs") is None


def test_video_thumbnail_uses_the_matched_frame(
    scene_video: Path, tmp_path: Path
) -> None:
    cache = tmp_path / "thumbs"

    target = ensure_thumbnail(scene_video, "clip-30", cache, ts=30)

    assert target == cache / "clip-30.jpg"
    assert target is not None and target.is_file()
    with Image.open(target) as thumbnail:
        red, _green, blue = thumbnail.convert("RGB").getpixel(
            (thumbnail.width // 2, thumbnail.height // 2)
        )
    assert blue > 150
    assert blue > red + 80


def test_video_without_a_timestamp_uses_the_first_frame(
    scene_video: Path, tmp_path: Path
) -> None:
    cache = tmp_path / "thumbs"

    target = ensure_thumbnail(scene_video, "clip-first", cache)

    assert target is not None and target.is_file()
    with Image.open(target) as thumbnail:
        red, _green, blue = thumbnail.convert("RGB").getpixel(
            (thumbnail.width // 2, thumbnail.height // 2)
        )
    assert not (blue > 150 and blue > red + 80)
