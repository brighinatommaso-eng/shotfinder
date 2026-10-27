"""OCR test: generate a known-text PNG with Pillow and read it back."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from shotfinder.extractors.ocr import ocr_image

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows.Media.Ocr is only available on Windows"
)

SENTENCE = "ShotFinder reads this text 12345"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    fonts_dir = Path(r"C:\Windows\Fonts")
    for name in ("arial.ttf", "calibri.ttf", "segoeui.ttf"):
        candidate = fonts_dir / name
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def _make_fixture(path: Path) -> Path:
    image = Image.new("RGB", (1500, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 120), SENTENCE, fill="black", font=_font(72))
    image.save(path)
    return path


def test_ocr_finds_known_sentence(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path / "sentence.png")
    original = fixture.read_bytes()

    lines = ocr_image(fixture)

    compact = "".join(" ".join(lines).lower().split())
    assert "shotfinder" in compact
    assert "12345" in compact
    # The extractor must never modify the source media.
    assert fixture.read_bytes() == original
