"""CLIP embeddings: shape, normalization and a beach-vs-screenshot ranking.

Slow: loads the exported ONNX models. Skipped when they are not present.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from shotfinder import models
from shotfinder.extractors.vision import embed_images, embed_text

MODELS_DIR = Path(
    os.environ.get("SHOTFINDER_MODELS_DIR")
    or (Path(__file__).resolve().parent.parent / "models")
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not models.models_ready(MODELS_DIR),
        reason=f"vision models not exported in {MODELS_DIR}",
    ),
]


def _beach_sunset(size: int = 256) -> Image.Image:
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)
    horizon = int(size * 0.62)
    for y in range(horizon):
        t = y / max(1, horizon - 1)
        draw.line([(0, y), (size, y)], fill=(int(250 - 30 * t), int(150 + 70 * t), int(90 + 80 * t)))
    for y in range(horizon, size):
        t = (y - horizon) / max(1, size - horizon - 1)
        draw.line([(0, y), (size, y)], fill=(int(20 + 30 * t), int(90 + 40 * t), int(140 + 50 * t)))
    radius = size // 10
    cx, cy = int(size * 0.72), int(size * 0.40)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(255, 235, 150))
    return image


def _code_screenshot(size: int = 256) -> Image.Image:
    image = Image.new("RGB", (size, size), (252, 252, 252))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, size, 24], fill=(40, 44, 52))
    lines = [
        "def main(argv):",
        "    total = 0",
        "    for path in argv:",
        "        total += len(path)",
        "    return total",
    ]
    y = 44
    for line in lines:
        draw.text((14, y), line, fill=(60, 60, 70))
        y += 22
    return image


def test_embeddings_are_normalized_and_beach_ranks_above_code() -> None:
    images = [_beach_sunset(), _code_screenshot()]

    vectors = embed_images(images, models_dir=MODELS_DIR, threads=1, batch_size=1)
    assert vectors.shape == (2, 512)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    query = embed_text("spiaggia al tramonto", models_dir=MODELS_DIR, threads=1)
    assert query.shape == (512,)
    assert query.dtype == np.float32
    assert abs(float(np.linalg.norm(query)) - 1.0) < 1e-5

    beach = float(vectors[0] @ query)
    code = float(vectors[1] @ query)
    assert beach > code


def test_batching_matches_single_image_calls() -> None:
    images = [_beach_sunset(), _code_screenshot()]

    single = embed_images(images, models_dir=MODELS_DIR, threads=1, batch_size=1)
    batched = embed_images(images, models_dir=MODELS_DIR, threads=1, batch_size=2)

    similarity = np.sum(single * batched, axis=1)
    assert np.all(similarity > 0.999999)


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValueError):
        embed_text("   ", models_dir=MODELS_DIR, threads=1)
