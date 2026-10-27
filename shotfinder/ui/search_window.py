"""Frameless Spotlight-style search window (Phase 3)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QRectF,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QCursor,
    QFontMetrics,
    QGuiApplication,
    QImage,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import models
from ..config import Config
from ..index import Index
from ..search import (
    MATCH_BOTH,
    MATCH_RECENT,
    MATCH_TEXT,
    MATCH_VISUAL,
    SearchResults,
    search,
)
from . import icons
from .formatting import clean_snippet, folder_label, format_date, format_timestamp
from .preview import PreviewWindow
from .search_session import SearchSession
from .theme import Palette, is_dark, palette_for
from .thumbnails import ensure_thumbnail

log = logging.getLogger(__name__)

WINDOW_WIDTH = 860
WINDOW_HEIGHT = 640
COLUMNS = 3
TILE_WIDTH = 244
TILE_HEIGHT = 224
THUMBNAIL_SIZE = (220, 124)
THUMBNAIL_RADIUS = 8
DEBOUNCE_MS = 250
RESULT_LIMIT = 60


class _SearchSignals(QObject):
    finished = Signal(int, object, str)


class SearchTask(QRunnable):
    """Run one hybrid search away from the GUI thread."""

    def __init__(
        self,
        config: Config,
        index: Index,
        query: str,
        generation: int,
        signals: _SearchSignals,
    ) -> None:
        super().__init__()
        self._config = config
        self._index = index
        self._query = query
        self._generation = generation
        self.signals = signals

    def run(self) -> None:
        try:
            rows = search(self._config, self._index, self._query, limit=RESULT_LIMIT)
        except Exception as exc:  # noqa: BLE001 - surfaced in the status line
            log.exception("Search failed for %r", self._query)
            self.signals.finished.emit(self._generation, SearchResults(), str(exc))
        else:
            self.signals.finished.emit(self._generation, rows, "")


class _ThumbnailSignals(QObject):
    done = Signal(str, str)


class ThumbnailTask(QRunnable):
    """Build one cached thumbnail; the cache itself is hash-keyed."""

    def __init__(
        self,
        source: str,
        file_hash: str,
        cache_dir: Path,
        key: str,
        signals: _ThumbnailSignals,
        *,
        ts: float | None = None,
    ) -> None:
        super().__init__()
        self._source = source
        self._hash = file_hash
        self._cache_dir = cache_dir
        self._key = key
        self._ts = ts
        self.signals = signals

    def run(self) -> None:
        path = ensure_thumbnail(
            self._source, self._hash, self._cache_dir, ts=self._ts
        )
        self.signals.done.emit(self._key, "" if path is None else str(path))


def _rounded_cover(source: QPixmap, size: tuple[int, int], radius: int) -> QPixmap:
    width, height = size
    target = QPixmap(width, height)
    target.fill(Qt.GlobalColor.transparent)
    painter = QPainter(target)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, width, height), radius, radius)
    painter.setClipPath(path)
    scaled = source.scaled(
        width,
        height,
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )
    x = max(0, (scaled.width() - width) // 2)
    y = max(0, (scaled.height() - height) // 2)
    painter.drawPixmap(0, 0, scaled, x, y, width, height)
    painter.end()
    return target


class ElidedLabel(QLabel):
    """A QLabel that elides in the middle instead of clipping."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setToolTip(text)

    def set_full_text(self, text: str) -> None:
        self._full_text = text
        self.setToolTip(text)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        metrics = QFontMetrics(self.font())
        text = metrics.elidedText(
            self._full_text, Qt.TextElideMode.ElideMiddle, self.width()
        )
        painter.drawText(
            self.rect(),
            int(self.alignment() | Qt.AlignmentFlag.AlignVCenter),
            text,
        )


class ResultTile(QFrame):
    """One result: thumbnail, name, match badge and OCR snippet."""

    clicked = Signal(object)
    activated = Signal(object)

    def __init__(
        self, result: dict[str, Any], palette: Palette, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.result = result
        self._palette = palette
        self._pixmap: QPixmap | None = None
        self._selected = False
        self.setFixedSize(TILE_WIDTH, TILE_HEIGHT)
        self.setProperty("selected", "false")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(str(result.get("path", "")))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._thumb = QLabel(self)
        self._thumb.setFixedSize(*THUMBNAIL_SIZE)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._thumb)

        self._timestamp: QLabel | None = None
        if result.get("kind") == "video" and result.get("ts") is not None:
            stamp = format_timestamp(float(result["ts"]))
            self._timestamp = QLabel(stamp, self._thumb)
            self._timestamp.setObjectName("Timestamp")
            self._timestamp.ensurePolished()
            self._timestamp.adjustSize()
            self._timestamp.move(
                THUMBNAIL_SIZE[0] - self._timestamp.width() - 6,
                THUMBNAIL_SIZE[1] - self._timestamp.height() - 6,
            )
            self._timestamp.raise_()
            self.setToolTip(f"{result.get('path', '')}\nFrame at {stamp}")

        name = ElidedLabel(Path(str(result.get("path", ""))).name, self)
        name.setObjectName("TileName")
        name.setFixedHeight(17)
        layout.addWidget(name)

        meta = QHBoxLayout()
        meta.setContentsMargins(0, 0, 0, 0)
        meta.setSpacing(6)
        self._badge = QLabel(self)
        self._badge.setObjectName("Badge")
        match_type = str(result.get("match_type") or "")
        badge_text = {
            MATCH_TEXT: "TEXT",
            MATCH_VISUAL: "VISUAL",
            MATCH_BOTH: "BOTH",
            MATCH_RECENT: "RECENT",
        }.get(match_type, "TEXT")
        badge_kind = {
            MATCH_TEXT: "text",
            MATCH_VISUAL: "visual",
            MATCH_BOTH: "both",
            MATCH_RECENT: "recent",
        }.get(match_type, "text")
        self._badge.setText(badge_text)
        self._badge.setProperty("kind", badge_kind)
        meta.addWidget(self._badge)
        date = QLabel(format_date(result.get("mtime")), self)
        date.setObjectName("TileMeta")
        meta.addWidget(date)
        meta.addStretch(1)
        layout.addLayout(meta)

        self._snippet = QLabel(self)
        self._snippet.setObjectName("TileSnippet")
        self._snippet.setWordWrap(True)
        self._snippet.setFixedHeight(30)
        self._snippet.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        if match_type == MATCH_RECENT:
            text = folder_label(str(result.get("path", "")))
        elif match_type in (MATCH_TEXT, MATCH_BOTH):
            text = clean_snippet(result.get("snippet")) or "Text match"
        else:
            text = "Matched by appearance"
        self._snippet.setText(text)
        layout.addWidget(self._snippet)

        self._render_placeholder()

    # -- content -------------------------------------------------------------

    @property
    def thumbnail_key(self) -> str:
        key = str(self.result.get("hash") or "")
        timestamp = self.thumbnail_timestamp
        if key and timestamp is not None:
            return f"{key}-{int(timestamp)}"
        return key

    @property
    def thumbnail_timestamp(self) -> float | None:
        """The frame to show for a video; ``0`` when no frame matched."""
        if self.result.get("kind") != "video":
            return None
        ts = self.result.get("ts")
        return float(ts) if ts is not None else 0.0

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._thumb.setPixmap(_rounded_cover(pixmap, THUMBNAIL_SIZE, THUMBNAIL_RADIUS))

    def _render_placeholder(self) -> None:
        extension = Path(str(self.result.get("path", ""))).suffix.lstrip(".") or "IMG"
        self._thumb.setPixmap(
            icons.placeholder(THUMBNAIL_SIZE, extension, self._palette.dark)
        )

    # -- selection -----------------------------------------------------------

    def set_selected(self, selected: bool) -> None:
        if selected == self._selected:
            return
        self._selected = selected
        self.setProperty("selected", "true" if selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self)


class SearchWindow(QWidget):
    """The window shown by the hotkey, the tray icon and a second launch."""

    def __init__(
        self, config: Config, index: Index, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._index = index
        self._palette = palette_for(is_dark(QGuiApplication.instance()))
        self._session = SearchSession()
        self._pool = QThreadPool.globalInstance()
        self._search_signals = _SearchSignals(self)
        self._search_signals.finished.connect(self._on_search_finished)
        self._thumbnail_signals = _ThumbnailSignals(self)
        self._thumbnail_signals.done.connect(self._on_thumbnail_ready)
        self._tasks: dict[int, SearchTask] = {}
        self._thumbnails: dict[str, QPixmap] = {}
        self._thumbnails_inflight: set[str] = set()
        self._tiles: list[ResultTile] = []
        self._section: QLabel | None = None
        self._selected = -1
        self._was_active = False
        self._preview: PreviewWindow | None = None
        self._last_query = ""
        self._last_count = 0

        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("ShotFinder")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self._build_ui()

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._start_search)
        self._input.textChanged.connect(lambda _text: self._debounce.start())

        self._status_reset = QTimer(self)
        self._status_reset.setSingleShot(True)
        self._status_reset.timeout.connect(self._update_status)

    # -- construction --------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        root = QFrame(self)
        root.setObjectName("Root")
        outer.addWidget(root)

        shadow = QGraphicsDropShadowEffect(root)
        shadow.setBlurRadius(48)
        shadow.setOffset(0, 10)
        shadow.setColor(self._palette.shadow)
        root.setGraphicsEffect(shadow)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 18, 20, 14)
        layout.setSpacing(12)

        search_row = QHBoxLayout()
        search_row.setContentsMargins(0, 0, 0, 0)
        search_row.setSpacing(10)
        self._search_icon = QLabel(root)
        self._search_icon.setFixedSize(24, 24)
        self._search_icon.setPixmap(icons.glyph("search", self._palette.muted, 22))
        search_row.addWidget(self._search_icon)

        self._input = QLineEdit(root)
        self._input.setObjectName("SearchInput")
        self._input.setPlaceholderText("Search what is in your screenshots, photos and videos")
        self._input.setClearButtonEnabled(False)
        self._input.installEventFilter(self)
        search_row.addWidget(self._input, 1)

        hint = QLabel("ESC", root)
        hint.setObjectName("Hint")
        search_row.addWidget(hint)
        layout.addLayout(search_row)

        separator = QFrame(root)
        separator.setObjectName("Separator")
        separator.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(separator)

        self._stack = QStackedWidget(root)
        layout.addWidget(self._stack, 1)

        self._scroll = QScrollArea(root)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._scroll.installEventFilter(self)
        host = QWidget()
        self._grid = QGridLayout(host)
        self._grid.setContentsMargins(2, 2, 8, 2)
        self._grid.setSpacing(12)
        self._grid.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self._scroll.setWidget(host)
        self._stack.addWidget(self._scroll)

        empty = QWidget(root)
        empty_layout = QVBoxLayout(empty)
        empty_layout.setContentsMargins(20, 20, 20, 20)
        empty_layout.setSpacing(6)
        empty_layout.addStretch(1)
        self._empty_icon = QLabel(empty)
        self._empty_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(self._empty_icon)
        self._empty_title = QLabel(empty)
        self._empty_title.setObjectName("EmptyTitle")
        self._empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(self._empty_title)
        self._empty_body = QLabel(empty)
        self._empty_body.setObjectName("EmptyBody")
        self._empty_body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_body.setWordWrap(True)
        self._empty_body.setFixedWidth(480)
        self._empty_body.setMinimumHeight(40)
        empty_layout.addWidget(self._empty_body, alignment=Qt.AlignmentFlag.AlignHCenter)
        empty_layout.addStretch(1)
        self._stack.addWidget(empty)

        self._status = QLabel(root)
        self._status.setObjectName("Status")
        layout.addWidget(self._status)

        self._refresh_empty_icon()

    def _refresh_empty_icon(self) -> None:
        self._empty_icon.setPixmap(icons.glyph("search", self._palette.faint, 52))
        self._search_icon.setPixmap(icons.glyph("search", self._palette.muted, 22))

    # -- window lifecycle ----------------------------------------------------

    def show_window(self) -> None:
        """Show, centre on the cursor's screen and load results."""
        self._center_on_cursor_screen()
        self.show()
        self.raise_()
        self.activateWindow()
        self._input.setFocus()
        self._input.selectAll()
        if self._input.text().strip():
            self._debounce.start()
        else:
            self._start_search()

    def _center_on_cursor_screen(self) -> None:
        screen = QGuiApplication.screenAt(QCursor.pos())
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        self.move(geometry.center() - QPoint(self.width() // 2, self.height() // 2))

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.type() == QEvent.Type.ActivationChange:
            if self.isActiveWindow():
                self._was_active = True
            elif self._was_active:
                self._was_active = False
                self.hide()
        elif event.type() == QEvent.Type.PaletteChange and hasattr(self, "_empty_icon"):
            self._palette = palette_for(is_dark(QGuiApplication.instance()))
            self._refresh_empty_icon()
        super().changeEvent(event)

    # -- keyboard ------------------------------------------------------------

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if event.type() == QEvent.Type.KeyPress and watched in (
            self._input,
            self._scroll,
        ):
            if self._handle_key(event, input_focused=watched is self._input):
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override
        if self._handle_key(event, input_focused=False):
            return
        super().keyPressEvent(event)

    def _handle_key(self, event: QKeyEvent, *, input_focused: bool) -> bool:
        key = event.key()
        control = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if key == Qt.Key.Key_Escape:
            self.hide()
            return True
        if key == Qt.Key.Key_Up:
            self._move_vertical(-1)
            return True
        if key == Qt.Key.Key_Down:
            self._move_vertical(1)
            return True
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right) and not input_focused:
            self._move_horizontal(-1 if key == Qt.Key.Key_Left else 1)
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if control:
                self._reveal_selected()
            else:
                self._open_preview()
            return True
        if key == Qt.Key.Key_C and control:
            if input_focused and self._input.hasSelectedText():
                return False
            self._copy_selected()
            return True
        if not input_focused and key == Qt.Key.Key_Backspace:
            self._input.setFocus()
            self._input.backspace()
            return True
        if (
            not input_focused
            and not (event.modifiers() & (Qt.KeyboardModifier.ControlModifier
                                          | Qt.KeyboardModifier.AltModifier
                                          | Qt.KeyboardModifier.MetaModifier))
            and event.text().isprintable()
            and event.text()
        ):
            # Typing while the result grid has focus returns to the query.
            self._input.setFocus()
            self._input.insert(event.text())
            return True
        return False

    # -- selection -----------------------------------------------------------

    def _select(self, index: int, *, scroll: bool = True) -> None:
        if not self._tiles:
            return
        index = max(0, min(index, len(self._tiles) - 1))
        for position, tile in enumerate(self._tiles):
            tile.set_selected(position == index)
        self._selected = index
        if scroll:
            self._scroll.ensureWidgetVisible(self._tiles[index], 0, 8)

    def _move_vertical(self, direction: int) -> None:
        if not self._tiles:
            return
        if self._selected < 0:
            self._select(0)
            return
        self._select(self._selected + direction * COLUMNS)

    def _move_horizontal(self, direction: int) -> None:
        if not self._tiles or self._selected < 0:
            return
        column = self._selected % COLUMNS
        if not 0 <= column + direction < COLUMNS:
            return
        self._select(self._selected + direction)

    def _current(self) -> ResultTile | None:
        if 0 <= self._selected < len(self._tiles):
            return self._tiles[self._selected]
        return None

    def _on_tile_clicked(self, tile: ResultTile) -> None:
        self._select(self._tiles.index(tile), scroll=False)
        self._scroll.setFocus()

    def _on_tile_activated(self, tile: ResultTile) -> None:
        self._select(self._tiles.index(tile), scroll=False)
        self._open_preview()

    # -- actions -------------------------------------------------------------

    def _open_preview(self) -> None:
        tile = self._current()
        if tile is None:
            return
        if self._preview is None:
            self._preview = PreviewWindow(self._config)
        self.hide()
        self._preview.show_result(tile.result)

    def _reveal_selected(self) -> None:
        tile = self._current()
        if tile is None:
            return
        path = str(tile.result.get("path", ""))
        try:
            subprocess.Popen(["explorer", "/select,", path])
        except OSError as exc:
            log.warning("Could not open Explorer for %s: %s", path, exc)
            self._flash_status("Could not open Explorer")

    def _copy_selected(self) -> None:
        tile = self._current()
        if tile is None:
            return
        path = str(tile.result.get("path", ""))
        if tile.result.get("kind") != "image":
            self._flash_status("Only images can be copied")
            return
        image = QImage(path)
        if image.isNull():
            self._flash_status("Could not read this image")
            return
        QGuiApplication.clipboard().setImage(image)
        self._flash_status("Image copied to the clipboard")

    # -- search --------------------------------------------------------------

    def _start_search(self) -> None:
        query = self._input.text().strip()
        generation = self._session.begin(query)
        self._set_status("Searching…")
        task = SearchTask(
            self._config, self._index, query, generation, self._search_signals
        )
        self._tasks[generation] = task
        self._pool.start(task)

    def _on_search_finished(
        self, generation: int, results: SearchResults, error: str
    ) -> None:
        self._tasks.pop(generation, None)
        if not self._session.is_current(generation):
            # A newer query is already in flight; this answer is stale.
            return
        if error:
            self._set_status(f"Search failed: {error}")
            return
        self._render(results)

    def _render(self, results: SearchResults) -> None:
        self._last_query = self._session.query.strip()
        self._last_count = len(results.all())
        self._clear_tiles()
        self._selected = -1

        if not results:
            self._show_empty_state()
            self._update_status()
            return

        self._stack.setCurrentWidget(self._scroll)
        for index, result in enumerate(results.matches):
            self._add_tile(result, index // COLUMNS, index % COLUMNS)

        if results.related:
            header_row = (len(results.matches) + COLUMNS - 1) // COLUMNS
            self._section = QLabel("Related by appearance · approximate", self)
            self._section.setObjectName("SectionHeader")
            self._grid.addWidget(self._section, header_row, 0, 1, COLUMNS)
            for index, result in enumerate(results.related):
                self._add_tile(
                    result,
                    header_row + 1 + index // COLUMNS,
                    index % COLUMNS,
                )

        self._select(0, scroll=False)
        self._request_thumbnails()
        self._update_status()

    def _add_tile(self, result: dict[str, Any], row: int, column: int) -> None:
        tile = ResultTile(result, self._palette)
        tile.clicked.connect(self._on_tile_clicked)
        tile.activated.connect(self._on_tile_activated)
        self._grid.addWidget(tile, row, column)
        self._tiles.append(tile)

    def _show_empty_state(self) -> None:
        self._stack.setCurrentIndex(1)
        if self._last_query:
            self._empty_title.setText("No matches")
            self._empty_body.setText(
                f'Nothing in your library matches "{self._last_query}". '
                "Try fewer words, or describe what the image shows."
            )
        else:
            self._empty_title.setText("Nothing indexed yet")
            self._empty_body.setText(
                "Run `python -m shotfinder.cli index` to scan your folders, "
                "then search again."
            )

    def _clear_tiles(self) -> None:
        if self._section is not None:
            self._grid.removeWidget(self._section)
            self._section.setParent(None)
            self._section.deleteLater()
            self._section = None
        for tile in self._tiles:
            self._grid.removeWidget(tile)
            tile.setParent(None)
            tile.deleteLater()
        self._tiles.clear()

    def _request_thumbnails(self) -> None:
        cache_dir = self._config.thumbs_dir
        for tile in self._tiles:
            key = tile.thumbnail_key
            if not key:
                continue
            cached = self._thumbnails.get(key)
            if cached is not None:
                tile.set_thumbnail(cached)
                continue
            if key in self._thumbnails_inflight:
                continue
            self._thumbnails_inflight.add(key)
            task = ThumbnailTask(
                str(tile.result.get("path", "")),
                key,
                cache_dir,
                key,
                self._thumbnail_signals,
                ts=tile.thumbnail_timestamp,
            )
            self._pool.start(task)

    def _on_thumbnail_ready(self, key: str, path: str) -> None:
        self._thumbnails_inflight.discard(key)
        if not path:
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        self._thumbnails[key] = pixmap
        for tile in self._tiles:
            if tile.thumbnail_key == key:
                tile.set_thumbnail(pixmap)

    # -- status --------------------------------------------------------------

    def _update_status(self) -> None:
        if self._last_query:
            noun = "result" if self._last_count == 1 else "results"
            text = f"{self._last_count} {noun}"
            related = sum(
                1 for tile in self._tiles if tile.result.get("match_type") == MATCH_VISUAL
            )
            if related:
                text += f" · {related} approximate"
        else:
            noun = "recent file" if self._last_count == 1 else "recent files"
            text = f"{self._last_count} {noun}"
        text += "   ·   Enter preview   ·   Ctrl+Enter Explorer   ·   Ctrl+C copy"
        if not models.models_ready(self._config.models_dir):
            text += "   ·   visual search off (models missing)"
        self._set_status(text)

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def _flash_status(self, text: str) -> None:
        self._set_status(text)
        self._status_reset.start(2200)
