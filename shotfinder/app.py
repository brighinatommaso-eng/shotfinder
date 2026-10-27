"""Tray application wiring: single instance, indexer, hotkey, window, tray."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import QApplication

from . import models
from .config import HOTKEY_CANDIDATES, Config
from .index import Index
from .ipc import SingleInstance
from .logs import setup_logging
from .ui import icons, theme
from .ui.hotkey import GlobalHotkey, pretty_hotkey, register_first
from .ui.search_window import SearchWindow
from .ui.tray import IndexerReporter, TrayIcon
from .watcher import Watcher
from .worker import Indexer

log = logging.getLogger(__name__)


class _ScanSignals(QObject):
    finished = Signal(int)


class ScanTask(QRunnable):
    """Walk the watched folders without blocking the first paint."""

    def __init__(self, watcher: Watcher, signals: _ScanSignals) -> None:
        super().__init__()
        self._watcher = watcher
        self.signals = signals

    def run(self) -> None:
        try:
            found = self._watcher.scan()
        except Exception:
            log.exception("Initial scan failed")
            found = 0
        self.signals.finished.emit(found)


class _DownloadSignals(QObject):
    progress = Signal(str, int, int)
    finished = Signal(object)
    failed = Signal(str)


class DownloadTask(QRunnable):
    """Run the existing model downloader off the GUI thread."""

    def __init__(self, models_dir: Path, signals: _DownloadSignals) -> None:
        super().__init__()
        self._models_dir = models_dir
        self.signals = signals

    def run(self) -> None:
        def report(model: models.ModelFile, done: int, total: int) -> None:
            self.signals.progress.emit(model.filename, int(done), int(total))

        try:
            paths = models.download_models(self._models_dir, progress=report)
        except Exception as exc:  # noqa: BLE001 - shown in the tray
            log.exception("Model download failed")
            self.signals.failed.emit(str(exc))
        else:
            self.signals.finished.emit([str(path) for path in paths])


class ShotFinderApp:
    """Owns the long-lived objects and connects them."""

    def __init__(
        self, app: QApplication, config: Config, instance: SingleInstance
    ) -> None:
        self.app = app
        self.config = config
        self.instance = instance
        self.index = Index(config.db_path)
        self.reporter = IndexerReporter(app)
        self.indexer = Indexer(config, self.index, on_progress=self.reporter.report)
        self.window = SearchWindow(config, self.index)
        self.tray = TrayIcon(self.indexer)
        self.hotkey: GlobalHotkey | None = None
        self.watcher = Watcher(config, self.indexer)
        self._scan_signals = _ScanSignals(app)
        self._scan_signals.finished.connect(self._on_scan_finished)
        self._scan_task: ScanTask | None = None
        self._download_task: DownloadTask | None = None
        self._download_signals: _DownloadSignals | None = None
        self._shutdown_done = False
        self._wire()

    def _wire(self) -> None:
        self.reporter.progress.connect(self.tray.set_stats)
        self.tray.open_search.connect(self.show_search)
        self.tray.toggle_pause.connect(self.toggle_pause)
        self.tray.download_models.connect(self.start_download)
        self.tray.open_data_folder.connect(self.open_data_folder)
        self.tray.quit_requested.connect(self.app.quit)
        self.app.aboutToQuit.connect(self.shutdown)

    def run(self) -> int:
        if not self.instance.listen(self._on_second_instance):
            log.warning("Could not listen for a second launch; continuing")
        self.indexer.start()
        self.watcher.start()
        self._scan_task = ScanTask(self.watcher, self._scan_signals)
        QThreadPool.globalInstance().start(self._scan_task)
        self.tray.show()
        if not self.tray.available:
            log.warning("No system tray available; showing the search window")
            self.window.show_window()
        self._register_hotkey()
        QTimer.singleShot(1200, self._check_models)
        return self.app.exec()

    # -- signals -------------------------------------------------------------

    def _on_second_instance(self, _message: str) -> None:
        self.show_search()

    def _on_scan_finished(self, found: int) -> None:
        log.info("Initial scan found %d media file(s)", found)

    def show_search(self) -> None:
        self.window.show_window()

    def toggle_pause(self) -> None:
        if self.indexer.paused:
            self.indexer.resume()
        else:
            self.indexer.pause()
        self.tray.set_paused(self.indexer.paused)

    def open_data_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.config.data_dir)))

    def _register_hotkey(self) -> None:
        config = self.config
        preferred = config.hotkey
        candidates = [preferred] + [
            combo for combo in HOTKEY_CANDIDATES if combo != preferred
        ]
        hotkey, failed = register_first(candidates)
        self.hotkey = hotkey
        if hotkey is None:
            log.warning("No global hotkey could be registered: %s", ", ".join(failed))
            self.tray.set_hotkey(None)
            if self.tray.available:
                self.tray.notify(
                    "Hotkey unavailable",
                    f"{pretty_hotkey(preferred)} and every fallback are already "
                    "used by other apps. Open ShotFinder from the tray icon.",
                    warning=True,
                )
            return

        hotkey.activated.connect(self.show_search)
        self.tray.set_hotkey(hotkey.spec)
        log.info("Global hotkey %s registered", hotkey.spec)
        if hotkey.spec == preferred:
            return

        if config.hotkey_is_custom:
            # The user chose this combo; do not rewrite their config behind
            # their back, and make sure they know the session is using another.
            log.warning(
                "Configured hotkey %s is taken; using %s for this session",
                preferred,
                hotkey.spec,
            )
            if self.tray.available:
                self.tray.notify(
                    "Hotkey taken",
                    f"{pretty_hotkey(preferred)} is already used by another app. "
                    f"Using {pretty_hotkey(hotkey.spec)} until you change the "
                    "hotkey in config.json.",
                    warning=True,
                )
            return

        config.hotkey = hotkey.spec
        config.save()
        if self.tray.available:
            self.tray.notify(
                "Search hotkey",
                f"ShotFinder now opens with {pretty_hotkey(hotkey.spec)} "
                f"({pretty_hotkey(preferred)} was already taken).",
            )

    # -- models --------------------------------------------------------------

    def _check_models(self) -> None:
        if models.models_ready(self.config.models_dir):
            self.tray.set_models_missing(False)
            return
        self.tray.set_models_missing(True)
        self.tray.notify(
            "Models missing",
            "Visual search needs the CLIP models (about 490 MB). "
            "Click this notification or use the tray menu to download them. "
            "Text search works meanwhile.",
            action="download_models",
        )

    def start_download(self) -> None:
        if self._download_task is not None:
            return
        signals = _DownloadSignals(self.app)
        signals.progress.connect(self._on_download_progress)
        signals.finished.connect(self._on_download_finished)
        signals.failed.connect(self._on_download_failed)
        task = DownloadTask(self.config.models_dir, signals)
        self._download_task = task
        self._download_signals = signals
        self.tray.set_download_status("Starting download…")
        QThreadPool.globalInstance().start(task)

    def _on_download_progress(self, filename: str, done: int, total: int) -> None:
        percent = int(done * 100 / total) if total else 0
        self.tray.set_download_status(f"Downloading {filename} {percent}%")

    def _on_download_finished(self, paths: list[str]) -> None:
        self._download_task = None
        self._download_signals = None
        self.tray.set_download_status(None)
        self.tray.set_models_missing(False)
        self.tray.notify(
            "Models ready",
            "Visual search is now on. Existing images will be embedded in the background.",
        )
        self.indexer.prepare_vision()
        queued = self.indexer.backfill_vision()
        log.info("Queued %d image(s) for vision backfill", queued)

    def _on_download_failed(self, error: str) -> None:
        self._download_task = None
        self._download_signals = None
        self.tray.set_download_status(None)
        missing = not models.models_ready(self.config.models_dir)
        self.tray.set_models_missing(missing)
        self.tray.notify("Model download failed", error, warning=True)

    # -- shutdown ------------------------------------------------------------

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        if self.hotkey is not None:
            self.hotkey.unregister()
        self.tray.hide()
        self.watcher.stop()
        self.indexer.stop()
        self.instance.close()
        QThreadPool.globalInstance().waitForDone(5000)
        self.index.close()


def main(argv: list[str] | None = None) -> int:
    config = Config.load()
    setup_logging(config)
    config.save()

    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("ShotFinder")
    app.setApplicationDisplayName("ShotFinder")
    app.setOrganizationName("ShotFinder")
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(icons.app_icon())
    font = QFont("Segoe UI Variable Text")
    font.setPointSizeF(10.0)
    app.setFont(font)
    theme.install(app)

    instance = SingleInstance()
    if instance.notify_running():
        log.info("Another instance is running; asked it to show the window")
        return 0

    controller = ShotFinderApp(app, config, instance)
    return controller.run()
