"""Thumbnail cache: one JPEG per indexed content hash.

The cache key is the index hash (xxhash of the first 1 MB plus file size), so a
moved or copied file reuses its thumbnail and a changed file gets a new key.
Generation is pure Pillow code and safe to call from a worker thread.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

log = logging.getLogger(__name__)

THUMBNAIL_SIZE = 256
JPEG_QUALITY = 85


def thumbnail_path(
    cache_dir: str | Path, file_hash: str, size: int = THUMBNAIL_SIZE
) -> Path:
    """Return the cache path for one content hash (``<hash>.jpg``)."""
    return Path(cache_dir) / f"{file_hash}.jpg"


def ensure_thumbnail(
    source: str | Path,
    file_hash: str,
    cache_dir: str | Path,
    size: int = THUMBNAIL_SIZE,
) -> Path | None:
    """Create the cached thumbnail if needed; return its path or ``None``.

    Writes through a unique temporary file and an atomic replace, so concurrent
    requests for the same hash cannot leave a half-written JPEG behind.
    """
    target = thumbnail_path(cache_dir, file_hash, size)
    if target.is_file() and target.stat().st_size > 0:
        return target
    cache = Path(cache_dir)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            descriptor, temporary = tempfile.mkstemp(
                dir=cache, prefix=f".{file_hash}-", suffix=".tmp"
            )
            os.close(descriptor)
            try:
                image.save(temporary, "JPEG", quality=JPEG_QUALITY, optimize=True)
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
    except Exception:
        log.warning("Could not build a thumbnail for %s", source, exc_info=True)
        return None
    return target
