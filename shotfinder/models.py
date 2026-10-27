"""Model registry, first-run download and local resolution (Phase 2).

Only :mod:`urllib` is used for downloads, so the runtime keeps a small
dependency footprint. Files are written to ``<name>.part``, resumed with an
HTTP Range request, verified against a pinned SHA-256 and then atomically
renamed into place.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

IMAGE_MODEL_FILENAME = "clip_image_fp32.onnx"
TEXT_MODEL_FILENAME = "clip_text_int8.onnx"
TOKENIZER_FILENAME = "tokenizer.json"

# Release assets: https://github.com/brighinatommaso-eng/shotfinder/releases/tag/models-v1
# SHOTFINDER_MODELS_DIR bypasses downloads entirely.
MODEL_BASE_URL = (
    "https://github.com/brighinatommaso-eng/shotfinder/releases/download/models-v1"
)

_CHUNK = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ModelFile:
    """One downloadable artifact."""

    name: str
    filename: str
    sha256: str
    size: int

    @property
    def url(self) -> str:
        return f"{MODEL_BASE_URL}/{self.filename}"


MODELS: tuple[ModelFile, ...] = (
    ModelFile(
        "clip_image",
        IMAGE_MODEL_FILENAME,
        "01e3a47fbef27e2eedc4e2b4171efa830ed89e2a628d5aaf1c33e5a040415486",
        351_583_544,
    ),
    ModelFile(
        "clip_text",
        TEXT_MODEL_FILENAME,
        "99f73e83c4a9221f00f091a3da292747e011c8fdc43d491c60ee9b4d4e168234",
        135_510_667,
    ),
    ModelFile(
        "tokenizer",
        TOKENIZER_FILENAME,
        "bf1b59b7b11c95f194f51708d918eea378e09d05f84c0e1656dc5180e8117088",
        2_919_362,
    ),
)

ProgressCallback = Callable[[ModelFile, int, int], None]


class ModelDownloadError(RuntimeError):
    """Raised when a model cannot be downloaded or fails verification."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_present(models_dir: Path, model: ModelFile) -> bool:
    """Cheap check: existence plus exact size. Download does the real hash check."""
    path = models_dir / model.filename
    try:
        return path.is_file() and path.stat().st_size == model.size
    except OSError:
        return False


def missing_files(models_dir: str | Path) -> list[ModelFile]:
    directory = Path(models_dir)
    return [model for model in MODELS if not _looks_present(directory, model)]


def models_ready(models_dir: str | Path) -> bool:
    """True when every registered model is present with the expected size."""
    return not missing_files(models_dir)


def _fetch(model: ModelFile, part: Path, progress: ProgressCallback | None) -> None:
    downloaded = part.stat().st_size if part.exists() else 0
    if downloaded > model.size:
        part.unlink(missing_ok=True)
        downloaded = 0
    if downloaded == model.size:
        return

    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    request = urllib.request.Request(model.url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and downloaded:
            # The partial file is not usable; start over once.
            part.unlink(missing_ok=True)
            return _fetch(model, part, progress)
        raise ModelDownloadError(
            f"Could not download {model.filename}: HTTP {exc.code} from {model.url}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ModelDownloadError(
            f"Could not download {model.filename}: {exc.reason} ({model.url})"
        ) from exc

    with response:
        if downloaded and getattr(response, "status", 200) != 206:
            # The server ignored the Range header; restart from zero.
            downloaded = 0
        mode = "ab" if downloaded else "wb"
        with part.open(mode) as handle:
            while chunk := response.read(_CHUNK):
                handle.write(chunk)
                downloaded += len(chunk)
                if progress is not None:
                    progress(model, min(downloaded, model.size), model.size)


def download_model(
    model: ModelFile,
    models_dir: str | Path,
    progress: ProgressCallback | None = None,
) -> Path:
    """Download one model into *models_dir*, resuming a previous attempt."""
    directory = Path(models_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / model.filename
    part = directory / (model.filename + ".part")

    _fetch(model, part, progress)

    actual = part.stat().st_size if part.exists() else 0
    if actual != model.size:
        part.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"{model.filename}: expected {model.size} bytes, got {actual}"
        )
    digest = sha256_file(part)
    if digest != model.sha256:
        part.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"{model.filename}: sha256 mismatch (expected {model.sha256}, got {digest})"
        )
    os.replace(part, target)
    log.info("Downloaded %s (%d bytes)", target, model.size)
    return target


def download_models(
    models_dir: str | Path, progress: ProgressCallback | None = None
) -> list[Path]:
    """Ensure every registered model exists locally; returns their paths."""
    directory = Path(models_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model in MODELS:
        if _looks_present(directory, model):
            paths.append(directory / model.filename)
            continue
        paths.append(download_model(model, directory, progress))
    return paths
