"""Thumbnail cache: one JPEG per indexed content hash (and video frame).

The cache key is the index hash (xxhash of the first 1 MB plus file size), so a
moved or copied file reuses its thumbnail and a changed file gets a new key.
For a video the key also carries the frame timestamp, because the tile shows
the best matching frame rather than the first one. Generation is safe to call
from a worker thread.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

from ..config import VIDEO_EXTENSIONS

log = logging.getLogger(__name__)

THUMBNAIL_SIZE = 256
JPEG_QUALITY = 85


def thumbnail_path(
    cache_dir: str | Path, file_hash: str, size: int = THUMBNAIL_SIZE
) -> Path:
    """Return the cache path for one key (``<key>.jpg``)."""
    return Path(cache_dir) / f"{file_hash}.jpg"


def ensure_thumbnail(
    source: str | Path,
    file_hash: str,
    cache_dir: str | Path,
    size: int = THUMBNAIL_SIZE,
    *,
    ts: float | None = None,
) -> Path | None:
    """Create the cached thumbnail if needed; return its path or ``None``.

    ``ts`` selects a video frame; without it a video falls back to its first
    frame. Writes through a unique temporary file and an atomic replace, so
    concurrent requests for the same key cannot leave a half-written JPEG.
    """
    target = thumbnail_path(cache_dir, file_hash, size)
    if target.is_file() and target.stat().st_size > 0:
        return target
    cache = Path(cache_dir)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        is_video = Path(source).suffix.lower() in VIDEO_EXTENSIONS
        if ts is not None or is_video:
            from ..extractors.video import extract_frame

            image = extract_frame(source, 0.0 if ts is None else ts, max_side=size)
            if image is None:
                return None
            _write_jpeg(image, target, cache, file_hash)
            return target
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            _write_jpeg(image, target, cache, file_hash)
    except Exception:
        log.warning("Could not build a thumbnail for %s", source, exc_info=True)
        return None
    return target


def _write_jpeg(image: Image.Image, target: Path, cache: Path, key: str) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=cache, prefix=f".{key}-", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        image.save(temporary, "JPEG", quality=JPEG_QUALITY, optimize=True)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
