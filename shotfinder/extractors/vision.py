"""CLIP image and text embeddings (Phase 2).

Pure functions over Pillow images and strings: this module never touches the
database. ONNX sessions are created lazily and cached per ``(models_dir,
threads)`` pair, so repeated calls reuse the same runtime.

The image graph is fp32 and the text graph is int8; neither normalizes its
output, so both public functions L2-normalize here.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from tokenizers import Tokenizer

from ..models import IMAGE_MODEL_FILENAME, TEXT_MODEL_FILENAME, TOKENIZER_FILENAME

IMAGE_SIZE = 224
VECTOR_DIMENSIONS = 512
MAX_TEXT_TOKENS = 128
PAD_TOKEN_ID = 0

# CLIP ViT-B/32 preprocessing constants.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class VisionUnavailableError(RuntimeError):
    """Raised when the ONNX models are missing or cannot be loaded."""


_SESSION_LOCK = threading.Lock()
_SESSIONS: dict[
    tuple[str, int], tuple[ort.InferenceSession, ort.InferenceSession, Tokenizer]
] = {}


def session_options(threads: int, *, allow_spinning: bool = False) -> ort.SessionOptions:
    """Runtime options shared by the app and tools/bench_vision.py."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, int(threads))
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry(
        "session.intra_op.allow_spinning", "1" if allow_spinning else "0"
    )
    return options


def create_session(
    path: str | Path, *, threads: int, allow_spinning: bool = False
) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(path),
        session_options(threads, allow_spinning=allow_spinning),
        providers=["CPUExecutionProvider"],
    )


def _load(
    models_dir: str | Path, threads: int
) -> tuple[ort.InferenceSession, ort.InferenceSession, Tokenizer]:
    directory = Path(models_dir)
    required = (IMAGE_MODEL_FILENAME, TEXT_MODEL_FILENAME, TOKENIZER_FILENAME)
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise VisionUnavailableError(
            f"Vision models missing from {directory}: {', '.join(missing)}. "
            "Run `python -m shotfinder.cli models download` or set "
            "SHOTFINDER_MODELS_DIR to a folder with the exported files."
        )

    image_session = create_session(directory / IMAGE_MODEL_FILENAME, threads=threads)
    text_session = create_session(directory / TEXT_MODEL_FILENAME, threads=threads)
    tokenizer = Tokenizer.from_file(str(directory / TOKENIZER_FILENAME))
    tokenizer.enable_truncation(max_length=MAX_TEXT_TOKENS)
    tokenizer.enable_padding(
        pad_id=PAD_TOKEN_ID, pad_token="[PAD]", length=MAX_TEXT_TOKENS
    )
    return image_session, text_session, tokenizer


def _sessions(
    models_dir: str | Path, threads: int
) -> tuple[ort.InferenceSession, ort.InferenceSession, Tokenizer]:
    key = (str(Path(models_dir)), max(1, int(threads)))
    with _SESSION_LOCK:
        cached = _SESSIONS.get(key)
        if cached is None:
            cached = _load(models_dir, threads)
            _SESSIONS[key] = cached
        return cached


def _prepare_image(image: Image.Image) -> Image.Image:
    """Orient the image and let libjpeg decode JPEGs at a reduced scale.

    ``draft`` only has an effect before the pixels are loaded, so it runs first.
    A 1/8-scale decode of a 12 MP photo still leaves ~500 px on the short side,
    well above the 224 px CLIP input, while cutting decode work substantially.
    """
    if image.format == "JPEG":
        image.draft("RGB", (IMAGE_SIZE * 2, IMAGE_SIZE * 2))
    # Phone photos carry orientation in EXIF; honour it like the OCR stage.
    return ImageOps.exif_transpose(image).convert("RGB")


def preprocess_images(images: Sequence[Image.Image]) -> np.ndarray:
    """CLIP ViT-B/32 preprocessing: RGB, shortest side 224 bicubic, center crop."""
    if not images:
        return np.zeros((0, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    mean = np.asarray(CLIP_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(CLIP_STD, dtype=np.float32).reshape(3, 1, 1)
    batch = np.empty((len(images), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    for index, image in enumerate(images):
        image = _prepare_image(image)
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError("image has an empty size")
        scale = IMAGE_SIZE / min(width, height)
        resized = image.resize(
            (max(IMAGE_SIZE, round(width * scale)), max(IMAGE_SIZE, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
        left = (resized.width - IMAGE_SIZE) // 2
        top = (resized.height - IMAGE_SIZE) // 2
        crop = resized.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
        array = np.asarray(crop, dtype=np.float32) / 255.0
        batch[index] = (array.transpose(2, 0, 1) - mean) / std
    return batch


def _normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def embed_images(
    images: Sequence[Image.Image],
    *,
    models_dir: str | Path,
    threads: int = 1,
    batch_size: int = 1,
) -> np.ndarray:
    """Return L2-normalized ``[N, 512]`` image embeddings."""
    if not images:
        return np.zeros((0, VECTOR_DIMENSIONS), dtype=np.float32)

    image_session, _, _ = _sessions(models_dir, threads)
    size = max(1, int(batch_size))
    chunks: list[np.ndarray] = []
    for start in range(0, len(images), size):
        pixels = preprocess_images(images[start : start + size])
        chunks.append(
            image_session.run(["image_embeds"], {"pixel_values": pixels})[0]
        )
    return _normalize(np.concatenate(chunks, axis=0))


def embed_text(
    text: str, *, models_dir: str | Path, threads: int = 1
) -> np.ndarray:
    """Return one L2-normalized ``[512]`` embedding for a query."""
    text = text.strip()
    if not text:
        raise ValueError("text must not be empty")

    _, text_session, tokenizer = _sessions(models_dir, threads)
    encoding = tokenizer.encode(text)
    input_ids = np.asarray([encoding.ids], dtype=np.int64)
    attention_mask = np.asarray([encoding.attention_mask], dtype=np.int64)
    vector = text_session.run(
        ["text_embeds"],
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )[0]
    return _normalize(vector)[0]
