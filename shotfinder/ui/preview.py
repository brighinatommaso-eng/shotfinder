"""Image preview: fit-to-window viewer with metadata (Phase 3).

Videos are not decoded in-app yet, so a video result shows a placeholder and an
"open externally" button instead of a player.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QUrl, Signal
from PySide6.QtGui import (
    QCursor,
    QDesktopServices,
    QGuiApplication,
    QImage,
    QImageReader,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..config import Config
from .formatting import format_date
from .theme import is_dark, palette_for

log = logging.getLogger(__name__)

MAX_IMAGE_SIDE = 2400


class _ImageSignals(QObject):
    done = Signal(int, object)


class ImageLoadTask(QRunnable):
    """Decode one image in a worker thread, already downscaled for the screen."""

    def __init__(
        self, path: str, token: int, maximum: QSize, signals: _ImageSignals
    ) -> None:
        super().__init__()
        self._path = path
        self._token = token
        self._maximum = maximum
        self.signals = signals

    def run(self) -> None:
        reader = QImageReader(self._path)
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid():
            reader.setScaledSize(
                size.scaled(self._maximum, Qt.AspectRatioMode.KeepAspectRatio)
            )
        image = reader.read()
        self.signals.done.emit(self._token, image)


class PreviewWindow(QWidget):
    """Frameless preview shown by Enter in the search window."""

    def __init__(self, config: Config, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._palette = palette_for(is_dark(QGuiApplication.instance()))
        self._image: QImage | None = None
        self._token = 0
        self._current: dict[str, Any] | None = None
        self._pool = QThreadPool.globalInstance()
        self._signals = _ImageSignals(self)
        self._signals.done.connect(self._on_image_loaded)

        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("ShotFinder preview")
        self.setMinimumSize(560, 420)
        self.resize(980, 720)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)
        root = QFrame(self)
        root.setObjectName("Root")
        outer.addWidget(root)

        shadow = QGraphicsDropShadowEffect(root)
        shadow.setBlurRadius(48)
        shadow.setOffset(0, 12)
        shadow.setColor(self._palette.shadow)
        root.setGraphicsEffect(shadow)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        self._name = QLabel(root)
        self._name.setObjectName("PreviewName")
        header.addWidget(self._name, 1)
        close = QToolButton(root)
        close.setObjectName("CloseButton")
        close.setText("✕")
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.setToolTip("Close (Esc)")
        close.clicked.connect(self.close)
        header.addWidget(close)
        layout.addLayout(header)

        self._image_label = QLabel(root)
        self._image_label.setObjectName("PreviewImage")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(400, 260)
        layout.addWidget(self._image_label, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(12)
        details = QVBoxLayout()
        details.setContentsMargins(0, 0, 0, 0)
        details.setSpacing(2)
        self._path = QLabel(root)
        self._path.setObjectName("PreviewPath")
        self._path.setWordWrap(True)
        self._path.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        details.addWidget(self._path)
        self._date = QLabel(root)
        self._date.setObjectName("PreviewDate")
        details.addWidget(self._date)
        footer.addLayout(details, 1)

        self._external = QPushButton("Open externally", root)
        self._external.setObjectName("OpenExternal")
        self._external.setCursor(Qt.CursorShape.PointingHandCursor)
        self._external.clicked.connect(self._open_externally)
        self._external.hide()
        footer.addWidget(self._external)
        layout.addLayout(footer)

    # -- content -------------------------------------------------------------

    def show_result(self, result: dict[str, Any]) -> None:
        self._current = result
        self._token += 1
        token = self._token
        path = Path(str(result.get("path", "")))
        self._name.setText(path.name)
        self._name.setToolTip(str(path))
        self._path.setText(str(path))
        self._date.setText(format_date(result.get("mtime")))
        self._image = None
        self._image_label.setPixmap(QPixmap())

        is_video = result.get("kind") == "video"
        self._external.setVisible(is_video)
        if is_video:
            self._image_label.setText(
                "Video preview is not available yet.\nOpen it in your player."
            )
        else:
            self._image_label.setText("Loading…")
            self._position()
            self.show()
            self.raise_()
            self.activateWindow()
            self._pool.start(
                ImageLoadTask(
                    str(path),
                    token,
                    QSize(MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
                    self._signals,
                )
            )
            return

        self._position()
        self.show()
        self.raise_()
        self.activateWindow()

    def _position(self) -> None:
        screen = QGuiApplication.screenAt(QCursor.pos())
        if screen is None:
            screen = QGuiApplication.screenAt(self.frameGeometry().center())
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        width = min(1080, max(560, geometry.width() - 160))
        height = min(820, max(420, geometry.height() - 160))
        self.resize(width, height)
        self.move(
            geometry.x() + (geometry.width() - width) // 2,
            geometry.y() + (geometry.height() - height) // 2,
        )

    def _on_image_loaded(self, token: int, image: QImage) -> None:
        if token != self._token:
            return
        if image is None or image.isNull():
            self._image_label.setText("Could not load this image.")
            return
        self._image = image
        self._render_image()

    def _render_image(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image)
        target = self._image_label.size()
        if pixmap.width() > target.width() or pixmap.height() > target.height():
            pixmap = pixmap.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._image_label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._render_image()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Ignore any image still decoding for the closed result.
        self._token += 1
        super().closeEvent(event)

    def _open_externally(self) -> None:
        if self._current is None:
            return
        path = str(self._current.get("path", ""))
        if not Path(path).exists():
            self._image_label.setText("This file is no longer on disk.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
