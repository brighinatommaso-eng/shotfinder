"""Capture the Phase 3 screenshots into docs/screenshots (dev tool).

Usage::

    python tools/capture_ui.py

Runs the real search window against the real index, grabs three states
(no results, results, preview) and writes PNGs. Requires the CLIP models to
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

from PySide6.QtWidgets import QApplication  # noqa: E402

from shotfinder.config import Config  # noqa: E402
from shotfinder.index import Index  # noqa: E402
from shotfinder.ui import theme  # noqa: E402
from shotfinder.ui.search_window import SearchWindow  # noqa: E402

OUTPUT = ROOT / "docs" / "screenshots"
NO_MATCH_QUERY = "xyzzy plugh"
RESULT_QUERY = "spider robot"


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
    finally:
        if window._preview is not None:
            window._preview.close()
        window.close()
        index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
