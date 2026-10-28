"""System tray icon, live status and menu (Phase 3)."""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from ..worker import Indexer
from . import icons
from .hotkey import pretty_hotkey

log = logging.getLogger(__name__)


class IndexerReporter(QObject):
    """Turns the worker's progress callback into a queued Qt signal.

    The callback runs on the indexing thread; this object lives on the GUI
    thread, so Qt delivers ``progress`` as a queued connection and no widget is
    ever touched from the worker.
    """

    progress = Signal(dict)

    def report(self, stats: dict[str, Any]) -> None:
        self.progress.emit(dict(stats))


class TrayIcon(QObject):
    """One tray icon with a status tooltip and the app menu."""

    open_search = Signal()
    toggle_pause = Signal()
    download_models = Signal()
    open_data_folder = Signal()
    quit_requested = Signal()

    def __init__(self, indexer: Indexer, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._stats: dict[str, Any] = {}
        self._paused = indexer.paused
        self._models_missing = False
        self._download_text: str | None = None
        self._hotkey: str | None = None
        self._notification_action: str | None = None

        self._tray = QSystemTrayIcon(icons.app_icon(), self)
        self._tray.setToolTip("ShotFinder")
        self._tray.activated.connect(self._on_activated)
        self._tray.messageClicked.connect(self._on_message_clicked)

        self._menu = QMenu()
        self._hotkey_action = self._menu.addAction("Search: unavailable")
        self._hotkey_action.setEnabled(False)
        self._open_action = self._menu.addAction("Open search")
        self._open_action.triggered.connect(self.open_search.emit)
        self._menu.addSeparator()
        self._pause_action = self._menu.addAction("Pause indexing")
        self._pause_action.triggered.connect(self.toggle_pause.emit)
        self._models_action = self._menu.addAction("Download models")
        self._models_action.triggered.connect(self.download_models.emit)
        self._data_action = self._menu.addAction("Open data folder")
        self._data_action.triggered.connect(self.open_data_folder.emit)
        self._menu.addSeparator()
        self._quit_action = self._menu.addAction("Quit")
        self._quit_action.triggered.connect(self.quit_requested.emit)

        self._tray.setContextMenu(self._menu)
        self._models_action.setVisible(False)
        self._refresh()

    # -- lifecycle -----------------------------------------------------------

    def show(self) -> None:
        self._tray.show()

    def hide(self) -> None:
        self._tray.hide()

    @property
    def available(self) -> bool:
        return QSystemTrayIcon.isSystemTrayAvailable()

    # -- state ---------------------------------------------------------------

    def set_stats(self, stats: dict[str, Any]) -> None:
        self._stats = stats
        self._refresh()

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self._refresh()

    def set_models_missing(self, missing: bool) -> None:
        self._models_missing = missing
        self._models_action.setVisible(missing)
        self._refresh()

    def set_download_status(self, text: str | None) -> None:
        self._download_text = text
        self._refresh()

    def set_hotkey(self, spec: str | None) -> None:
        """Show which combo opens the window (``None`` when none registered)."""
        self._hotkey = spec
        self._refresh()

    def status_text(self) -> str:
        """The short state shown in the tooltip."""
        if self._download_text:
            return self._download_text
        activity = str(self._stats.get("activity") or "")
        if activity:
            percent = self._stats.get("activity_percent")
            if percent is None:
                return activity
            return f"{activity} ({int(percent)}%)"
        if self._paused:
            return "Paused"
        outstanding = int(self._stats.get("outstanding", 0))
        if outstanding > 0:
            done = (
                int(self._stats.get("processed", 0))
                + int(self._stats.get("skipped", 0))
                + int(self._stats.get("failed", 0))
            )
            total = done + outstanding
            return f"Indexing {done:,}/{total:,}"
        if self._models_missing:
            return "Models missing"
        return "Idle"

    def _refresh(self) -> None:
        hotkey = (
            f"Search: {pretty_hotkey(self._hotkey)}"
            if self._hotkey
            else "Search: unavailable"
        )
        self._hotkey_action.setText(hotkey)
        self._tray.setToolTip(f"ShotFinder\n{self.status_text()}\n{hotkey}")
        self._pause_action.setText(
            "Resume indexing" if self._paused else "Pause indexing"
        )

    def notify(
        self,
        title: str,
        message: str,
        *,
        warning: bool = False,
        action: str | None = None,
    ) -> None:
        self._notification_action = action
        icon = (
            QSystemTrayIcon.MessageIcon.Warning
            if warning
            else QSystemTrayIcon.MessageIcon.Information
        )
        self._tray.showMessage(title, message, icon, 8000)

    # -- events --------------------------------------------------------------

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.open_search.emit()

    def _on_message_clicked(self) -> None:
        action = self._notification_action
        self._notification_action = None
        if action == "download_models":
            self.download_models.emit()
