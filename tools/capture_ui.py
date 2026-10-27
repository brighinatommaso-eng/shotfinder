"""Capture the Phase 3/4 screenshots into docs/screenshots (dev tool).

Usage::

    python tools/capture_ui.py

Runs the real search window against the real index, grabs the empty state,
results (including video tiles with timestamps) and an image preview, then
opens a video result and reports the player state. Requires the CLIP models to
show visual results; set SHOTFINDER_MODELS_DIR if they are not in ./models.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("SHOTFINDER_MODELS_DIR", str(ROOT / "models"))

from PySide6.QtMultimedia import QMediaPlayer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from shotfinder.config import Config  # noqa: E402
from shotfinder.index import Index  # noqa: E402
from shotfinder.search import SearchResults  # noqa: E402
from shotfinder.ui import theme  # noqa: E402
from shotfinder.ui.search_window import SearchWindow  # noqa: E402

OUTPUT = ROOT / "docs" / "screenshots"
NO_MATCH_QUERY = "xyzzy plugh"
RESULT_QUERY = "spider robot"
VIDEO_QUERY = "windows disk cleanup dialog"


def pump(app: QApplication, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def wait_for(app: QApplication, predicate, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    app.processEvents()
    return predicate()


def search(app: QApplication, window: SearchWindow, query: str) -> None:
    window._input.setText(query)
    window._debounce.start()
    ready = wait_for(
        app,
        lambda: window._last_query == query
        and window._session.query.strip() == query
        and window._session.is_current(window._session.generation),
    )
    if not ready:
        raise SystemExit(f"Search for {query!r} did not finish")


def thumbnails_ready(window: SearchWindow) -> bool:
    visible = window._tiles[:6]
    return bool(visible) and all(tile._pixmap is not None for tile in visible)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    theme.install(app)
    config = Config.load()
    index = Index(config.db_path)
    window = SearchWindow(config, index)
    try:
        window.show_window()
        wait_for(app, lambda: window._last_count > 0)

        search(app, window, NO_MATCH_QUERY)
        # Visual-only suggestions above the garbage floor still come back for a
        # nonsense query, so render the genuine no-results state directly.
        window._render(SearchResults())
        pump(app, 0.4)
        window.grab().save(str(OUTPUT / "empty-state.png"))
        print(f"empty state: {window._last_count} result(s) -> empty-state.png")

        search(app, window, RESULT_QUERY)
        if window._last_count == 0:
            raise SystemExit(
                "The result query returned nothing; index more media or check the models"
            )
        wait_for(app, lambda: thumbnails_ready(window), timeout=20)
        pump(app, 0.4)
        window.grab().save(str(OUTPUT / "results.png"))
        print(f"results: {window._last_count} result(s) -> results.png")

        window._select(0)
        window._open_preview()
        preview = window._preview
        if preview is None:
            raise SystemExit("Preview window was not created")
        wait_for(app, lambda: preview._image is not None, timeout=20)
        pump(app, 0.4)
        preview.grab().save(str(OUTPUT / "preview.png"))
        print("preview -> preview.png")

        preview.close()
        search(app, window, VIDEO_QUERY)
        video_index = next(
            (
                index
                for index, tile in enumerate(window._tiles)
                if tile.result.get("kind") == "video"
            ),
            None,
        )
        if video_index is None:
            raise SystemExit("The video query returned no video result")
        window._select(video_index)
        window._open_preview()
        preview = window._preview
        if preview is None:
            raise SystemExit("Video preview window was not created")
        wait_for(
            app,
            lambda: preview._fallback
            or preview._player.mediaStatus()
            in (
                QMediaPlayer.MediaStatus.LoadedMedia,
                QMediaPlayer.MediaStatus.BufferedMedia,
            ),
            timeout=20,
        )
        pump(app, 1.2)
        # A QVideoWidget paints through a native surface, so QWidget.grab()
        # cannot capture it. Verify the player state instead of shipping a
        # black rectangle; the result tiles carry the video frame + timestamp.
        status = preview._player.mediaStatus()
        print(
            "video preview: fallback="
            f"{preview._fallback} status={status.name} "
            f"position={preview._player.position()} ms"
        )
    finally:
        if window._preview is not None:
            window._preview.close()
        window.close()
        index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
