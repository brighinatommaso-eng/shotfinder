"""Render the painted app icon to a multi-size Windows ``.ico``.

The tray and window icons are drawn with QPainter at runtime, so the build
needs one bitmap copy for Explorer, the taskbar and the executable's resource.
Each size is rendered by Qt itself (not downscaled) and packed into a single
ICO file; Windows accepts PNG-compressed entries since Vista.
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
TARGET = ROOT / "tools" / "shotfinder.ico"


def _png_frames() -> list[tuple[int, bytes]]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(ROOT))
    from PySide6.QtWidgets import QApplication

    from shotfinder.ui.icons import app_icon

    QApplication([])
    frames: list[tuple[int, bytes]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for size in SIZES:
            path = Path(tmp) / f"{size}.png"
            app_icon(size).pixmap(size, size).save(str(path), "PNG")
            frames.append((size, path.read_bytes()))
    return frames


def _write_ico(frames: list[tuple[int, bytes]], target: Path) -> None:
    header = struct.pack("<HHH", 0, 1, len(frames))
    offset = len(header) + 16 * len(frames)
    entries = bytearray()
    payload = bytearray()
    for size, png in frames:
        entries += struct.pack(
            "<BBBBHHII",
            0 if size >= 256 else size,
            0 if size >= 256 else size,
            0,
            0,
            1,
            32,
            len(png),
            offset,
        )
        payload += png
        offset += len(png)
    target.write_bytes(header + bytes(entries) + bytes(payload))


def main() -> int:
    frames = _png_frames()
    _write_ico(frames, TARGET)
    print(f"Wrote {TARGET} ({TARGET.stat().st_size:,} bytes, {len(frames)} sizes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
