"""Preview: fit-to-window image viewer and an in-app video player (Phase 4).

Videos open in ``QMediaPlayer`` at the matched frame (two seconds earlier) and
are controlled with Space and the arrow keys. When the Windows backend cannot
decode a codec, the window shows a still frame and an "Open externally" button
instead of a black rectangle.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image
from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QCursor,
    QDesktopServices,
    QGuiApplication,
    QImage,
    QImageReader,
    QPixmap,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..config import Config
from ..extractors.video import extract_frame
from .formatting import format_date, format_timestamp
from .theme import is_dark, palette_for

log = logging.getLogger(__name__)

MAX_IMAGE_SIDE = 2400
PREVIEW_FRAME_SIDE = 1600
SEEK_BACK_SECONDS = 2.0
SEEK_STEP_MS = 5000


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


class _FrameSignals(QObject):
    done = Signal(int, object)


class VideoFrameTask(QRunnable):
    """Extract one frame for the codec fallback without blocking the GUI."""

    def __init__(
        self,
        path: str,
        timestamp: float,
        token: int,
        signals: _FrameSignals,
    ) -> None:
        super().__init__()
        self._path = path
        self._timestamp = timestamp
        self._token = token
        self.signals = signals

    def run(self) -> None:
        try:
            image = extract_frame(
                self._path, self._timestamp, max_side=PREVIEW_FRAME_SIDE
            )
        except Exception:
            log.warning("Could not extract a preview frame from %s", self._path)
            image = None
        self.signals.done.emit(self._token, image)


def _pil_to_qimage(image: Image.Image) -> QImage:
    rgb = image.convert("RGB")
    payload = rgb.tobytes("raw", "RGB")
    return QImage(
        payload,
        rgb.width,
        rgb.height,
        rgb.width * 3,
        QImage.Format.Format_RGB888,
    ).copy()


def _vlc_executable(suffix: str) -> str | None:
    """Return the default player's exe only when it is VLC.

    ``--start-time`` is a VLC option; passing it to another player would make it
    fail, so the start time is only forwarded in that one case.
    """
    if os.name != "nt":
        return None
    import winreg

    progid: str | None = None
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            "Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\FileExts"
            f"\\{suffix}\\UserChoice",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "ProgId")
            progid = str(value)
    except OSError:
        pass

    subkeys = [r"Applications\vlc.exe\shell\open\command"]
    if progid:
        subkeys.insert(0, rf"{progid}\shell\open\command")
    for subkey in subkeys:
        try:
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, subkey) as key:
                command, _ = winreg.QueryValueEx(key, "")
        except OSError:
            continue
        if "vlc" not in str(command).lower():
            continue
        match = re.match(r'^\s*"([^"]+)"', str(command))
        executable = match.group(1) if match else str(command).split()[0]
        if Path(executable).is_file():
            return executable
    return None


class PreviewWindow(QWidget):
    """Frameless preview shown by Enter in the search window."""

    def __init__(self, config: Config, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._palette = palette_for(is_dark(QGuiApplication.instance()))
        self._image: QImage | None = None
        self._token = 0
        self._frame_token = 0
        self._current: dict[str, Any] | None = None
        self._is_video = False
        self._fallback = False
        self._video_path = ""
        self._video_ts = 0.0
        self._pending_seek_ms = 0
        self._seek_applied = False
        self._pause_after_seek_pending = False
        self._pool = QThreadPool.globalInstance()
        self._signals = _ImageSignals(self)
        self._signals.done.connect(self._on_image_loaded)
        self._frame_signals = _FrameSignals(self)
        self._frame_signals.done.connect(self._on_frame_loaded)

        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("ShotFinder preview")
        self.setMinimumSize(560, 420)
        self.resize(980, 720)
        self._build_ui()

        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._audio.setVolume(0.8)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video)
        self._player.mediaStatusChanged.connect(self._on_media_status)
        self._player.errorOccurred.connect(self._on_player_error)
        self._pause_timer = QTimer(self)
        self._pause_timer.setSingleShot(True)
        self._pause_timer.timeout.connect(self._pause_after_seek)

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

        self._stack = QStackedWidget(root)
        layout.addWidget(self._stack, 1)

        self._image_label = QLabel(root)
        self._image_label.setObjectName("PreviewImage")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(400, 260)
        self._stack.addWidget(self._image_label)

        self._video = QVideoWidget(root)
        self._video.setMinimumSize(400, 260)
        self._stack.addWidget(self._video)

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
        self._frame_token += 1
        self._stop_video()
        path = Path(str(result.get("path", "")))
        self._name.setText(path.name)
        self._name.setToolTip(str(path))
        self._path.setText(str(path))
        self._date.setText(format_date(result.get("mtime")))
        self._image = None
        self._image_label.setPixmap(QPixmap())

        is_video = result.get("kind") == "video"
        self._external.setVisible(is_video)
        self._external.setText("Open externally")
        if is_video:
            self._start_video(path, float(result.get("ts") or 0.0))
        else:
            self._is_video = False
            self._stack.setCurrentWidget(self._image_label)
            self._image_label.setText("Loading…")
            self._pool.start(
                ImageLoadTask(
                    str(path),
                    self._token,
                    QSize(MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
                    self._signals,
                )
            )

        self._position()
        self.show()
        self.raise_()
        self.activateWindow()

    def _start_video(self, path: Path, timestamp: float) -> None:
        self._is_video = True
        self._fallback = False
        self._video_path = str(path)
        self._video_ts = max(0.0, timestamp)
        self._pending_seek_ms = max(
            0, int((self._video_ts - SEEK_BACK_SECONDS) * 1000)
        )
        self._seek_applied = False
        self._pause_after_seek_pending = True
        self._stack.setCurrentWidget(self._video)
        self._image_label.setPixmap(QPixmap())
        self._image_label.setText("Loading video…")
        self._player.setSource(QUrl.fromLocalFile(str(path)))

    def _stop_video(self) -> None:
        self._is_video = False
        self._pause_after_seek_pending = False
        self._pause_timer.stop()
        self._player.stop()
        self._player.setSource(QUrl())

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

    # -- images --------------------------------------------------------------

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

    # -- video ---------------------------------------------------------------

    def _on_media_status(self, status: QMediaPlayer.MediaStatus) -> None:
        if not self._is_video or self._fallback:
            return
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            self._show_fallback()
        elif status in (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        ):
            self._apply_seek()

    def _on_player_error(
        self, _error: QMediaPlayer.Error, message: str
    ) -> None:
        if not self._is_video or self._fallback:
            return
        log.info("Video backend rejected %s: %s", self._video_path, message)
        self._show_fallback()

    def _apply_seek(self) -> None:
        if self._seek_applied:
            return
        self._seek_applied = True
        self._player.setPosition(self._pending_seek_ms)
        # A paused player may not paint until it has been started once; start
        # and pause shortly after so the matched frame is visible.
        self._player.play()
        self._pause_timer.start(140)

    def _pause_after_seek(self) -> None:
        if not self._pause_after_seek_pending:
            return
        self._pause_after_seek_pending = False
        if (
            self._is_video
            and not self._fallback
            and self._player.playbackState()
            == QMediaPlayer.PlaybackState.PlayingState
        ):
            self._player.pause()

    def _show_fallback(self) -> None:
        if self._fallback:
            return
        self._fallback = True
        self._pause_after_seek_pending = False
        self._pause_timer.stop()
        self._player.stop()
        self._stack.setCurrentWidget(self._image_label)
        self._image_label.setPixmap(QPixmap())
        self._image_label.setText(
            "This codec is not supported in-app.\nShowing the matched frame instead."
        )
        self._external.setText(
            f"Open externally from {format_timestamp(self._video_ts)}"
        )
        self._frame_token += 1
        self._pool.start(
            VideoFrameTask(
                self._video_path,
                self._video_ts,
                self._frame_token,
                self._frame_signals,
            )
        )

    def _on_frame_loaded(self, token: int, image: Image.Image | None) -> None:
        if token != self._frame_token:
            return
        if image is None:
            self._image_label.setText("Could not decode a frame from this video.")
            return
        self._image = _pil_to_qimage(image)
        self._render_image()

    def _toggle_play(self) -> None:
        if not self._is_video or self._fallback:
            return
        self._pause_after_seek_pending = False
        self._pause_timer.stop()
        if (
            self._player.playbackState()
            == QMediaPlayer.PlaybackState.PlayingState
        ):
            self._player.pause()
        elif self._player.mediaStatus() not in (
            QMediaPlayer.MediaStatus.NoMedia,
            QMediaPlayer.MediaStatus.InvalidMedia,
        ):
            self._player.play()

    def _seek_relative(self, delta_ms: int) -> None:
        if not self._is_video or self._fallback:
            return
        target = max(0, self._player.position() + delta_ms)
        duration = self._player.duration()
        if duration > 0:
            target = min(target, max(0, duration - 100))
        self._player.setPosition(target)

    # -- window events -------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._render_image()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        if event.key() == Qt.Key.Key_Space:
            self._toggle_play()
            return
        if event.key() == Qt.Key.Key_Left:
            self._seek_relative(-SEEK_STEP_MS)
            return
        if event.key() == Qt.Key.Key_Right:
            self._seek_relative(SEEK_STEP_MS)
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Ignore any image still decoding for the closed result.
        self._token += 1
        self._frame_token += 1
        self._stop_video()
        super().closeEvent(event)

    def _open_externally(self) -> None:
        if self._current is None:
            return
        path = str(self._current.get("path", ""))
        if not Path(path).exists():
            self._image_label.setText("This file is no longer on disk.")
            return
        if self._current.get("kind") == "video":
            seconds = int(max(0.0, float(self._current.get("ts") or 0.0)))
            executable = _vlc_executable(Path(path).suffix.lower())
            if executable is not None:
                creationflags = (
                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    if os.name == "nt"
                    else 0
                )
                try:
                    subprocess.Popen(
                        [executable, f"--start-time={seconds}", path],
                        creationflags=creationflags,
                    )
                    return
                except OSError as exc:
                    log.warning("Could not start VLC: %s", exc)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
