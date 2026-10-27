"""Windows.Media.Ocr: image path -> list of text lines."""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

log = logging.getLogger(__name__)


class OCRError(RuntimeError):
    """Base class for OCR failures."""


class OCRUnavailableError(OCRError):
    """Raised when Windows OCR cannot run (missing winrt packages or no recognizer)."""


def _load_winrt() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from winrt.windows.globalization import Language
        from winrt.windows.graphics.imaging import BitmapDecoder
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream
    except ImportError as exc:
        raise OCRUnavailableError(
            "Windows OCR support is not installed; install the winrt packages for this platform."
        ) from exc
    return Language, BitmapDecoder, OcrEngine, DataWriter, InMemoryRandomAccessStream


def _make_engine(ocr_engine: Any, language_type: Any) -> Any:
    engine = ocr_engine.try_create_from_user_profile_languages()
    if engine is not None:
        return engine

    try:
        engine = ocr_engine.try_create_from_language(language_type("en-US"))
    except Exception:
        engine = None
    if engine is not None:
        return engine

    available = [language.language_tag for language in ocr_engine.available_recognizer_languages]
    if available:
        # Last resort: an installed recognizer is better than no OCR at all.
        try:
            engine = ocr_engine.try_create_from_language(language_type(available[0]))
        except Exception:
            engine = None
        if engine is not None:
            log.warning("en-US OCR recognizer missing; using %s", available[0])
            return engine

    raise OCRUnavailableError(
        "No Windows OCR recognizer is installed. Add a language with OCR support in "
        "Windows Settings > Time & language > Language & region. Available: "
        + (", ".join(available) if available else "none")
    )


def _encode_for_ocr(path: Path, max_dimension: int) -> bytes:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if max_dimension > 0 and longest > max_dimension:
            scale = max_dimension / longest
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def ocr_image(path: str | Path, *, max_dimension: int = 1600) -> list[str]:
    """Return the non-empty text lines visible in *path*.

    The image is downscaled (longest side) before OCR. Raises
    :class:`OCRUnavailableError` when Windows OCR cannot be created and
    :class:`OCRError` when recognition fails.
    """
    path = Path(path)
    language_type, decoder_type, ocr_engine, writer_type, stream_type = _load_winrt()
    engine = _make_engine(ocr_engine, language_type)
    payload = _encode_for_ocr(path, max_dimension)

    async def recognise() -> Any:
        stream = stream_type()
        writer = writer_type(stream)
        writer.write_bytes(payload)
        await writer.store_async()
        writer.detach_stream()
        stream.seek(0)
        decoder = await decoder_type.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        return await engine.recognize_async(bitmap)

    try:
        result = asyncio.run(recognise())
    except OCRError:
        raise
    except Exception as exc:
        raise OCRError(f"Windows OCR failed for {path}: {exc}") from exc

    return [
        line.text.strip()
        for line in result.lines
        if line.text and line.text.strip()
    ]
