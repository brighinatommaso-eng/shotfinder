"""Tray application wiring: single instance, indexer, hotkey, window, tray."""

from __future__ import annotations

import logging
import os
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
from .ui.settings import SettingsDialog
from .ui.tray import IndexerReporter, TrayIcon
from .ui.welcome import WelcomeDialog
from .watcher import Watcher
from .worker import Indexer

log = logging.getLogger(__name__)


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.path.abspath(os.fspath(left))) == os.path.normcase(
        os.path.abspath(os.fspath(right))
    )


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
        self,
        app: QApplication,
        config: Config,
        instance: SingleInstance,
        *,
        first_run: bool = False,
    ) -> None:
        self.app = app
        self.config = config
        self.instance = instance
        self.first_run = first_run
        self.index = Index(config.db_path)
        self.reporter = IndexerReporter(app)
        self.indexer = Indexer(config, self.index, on_progress=self.reporter.report)
        self.window = SearchWindow(config, self.index)
        self.tray = TrayIcon(self.indexer)
        self.hotkey: GlobalHotkey | None = None
        self.watcher = Watcher(config, self.indexer)
        self.settings: SettingsDialog | None = None
        self._scan_signals = _ScanSignals(app)
        self._scan_signals.finished.connect(self._on_scan_finished)
        self._scan_task: ScanTask | None = None
        self._scan_running = False
        self._rescan_after_scan = False
        self._download_task: DownloadTask | None = None
        self._download_signals: _DownloadSignals | None = None
        self._download_attempt = 0
        self._download_retry_scheduled = False
        self._shutdown_done = False
        self._wire()

    def _wire(self) -> None:
        self.reporter.progress.connect(self.tray.set_stats)
        self.tray.open_search.connect(self.show_search)
        self.tray.toggle_pause.connect(self.toggle_pause)
        self.tray.download_models.connect(self.start_download)
        self.tray.open_data_folder.connect(self.open_data_folder)
        self.tray.open_settings.connect(self.show_settings)
        self.tray.quit_requested.connect(self.app.quit)
        self.app.aboutToQuit.connect(self.shutdown)

    def run(self) -> int:
        if self.first_run:
            self._show_welcome()
        if not self.instance.listen(self._on_second_instance):
            log.warning("Could not listen for a second launch; continuing")
        self.indexer.start()
        self.watcher.start()
        self._start_scan()
        self.tray.show()
        if not self.tray.available:
            log.warning("No system tray available; showing the search window")
            self.window.show_window()
        self._register_hotkey()
        if self.first_run:
            self.tray.notify(
                "ShotFinder is running",
                "Use the hotkey or the tray icon to search. The models are "
                "downloading in the background; text search works now.",
            )
            QTimer.singleShot(300, self.start_download)
        else:
            QTimer.singleShot(1200, self._check_models)
        return self.app.exec()

    def _show_welcome(self) -> None:
        dialog = WelcomeDialog(self.config)
        dialog.exec()
        self.config.watched_folders = dialog.selected_folders()
        self.config.save()
        log.info(
            "First run: watching %s",
            ", ".join(str(folder) for folder in self.config.watched_folders)
            or "(no folders)",
        )

    # -- signals -------------------------------------------------------------

    def _on_second_instance(self, _message: str) -> None:
        self.show_search()

    def _start_scan(self) -> None:
        if self._scan_running:
            return
        self._scan_running = True
        self._scan_task = ScanTask(self.watcher, self._scan_signals)
        QThreadPool.globalInstance().start(self._scan_task)

    def _on_scan_finished(self, found: int) -> None:
        self._scan_running = False
        log.info("Initial scan found %d media file(s)", found)
        if self._rescan_after_scan:
            self._rescan_after_scan = False
            self._start_scan()

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
        if self._download_task is not None or self._download_retry_scheduled:
            return
        vision = models.models_ready(self.config.models_dir)
        speech = models.speech_ready(self.config.models_dir)
        if vision and speech:
            self.tray.set_models_missing(False)
            return
        self.tray.set_models_missing(True)
        if not vision:
            message = (
                "Visual search needs the CLIP models and transcription needs "
                "Whisper (about 640 MB). Click this notification or use the "
                "tray menu to download them. Text search works meanwhile."
            )
        else:
            message = (
                "Speech search needs the Whisper model (about 140 MB). Click "
                "this notification or use the tray menu to download it."
            )
        self.tray.notify(
            "Models missing",
            message,
            action="download_models",
        )

    def start_download(self) -> None:
        """Manual download from the tray; starts a fresh retry budget."""
        if self._download_task is not None:
            return
        self._download_attempt = 0
        self._download_retry_scheduled = False
        self._start_download_task()

    def _start_download_task(self) -> None:
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
        self._download_attempt = 0
        self._download_retry_scheduled = False
        self.tray.set_download_status(None)
        self.tray.set_models_missing(False)
        self.tray.notify(
            "Models ready",
            "Visual search and transcription are now on. Existing files will "
            "be processed in the background.",
        )
        self.indexer.prepare_vision()
        queued = self.indexer.backfill_vision()
        log.info("Queued %d image(s) for vision backfill", queued)
        speech = self.indexer.backfill_speech()
        log.info("Queued %d video(s) for speech", speech)

    def _on_download_failed(self, error: str) -> None:
        self._download_task = None
        self._download_signals = None
        self._download_attempt += 1
        missing = not models.models_ready(self.config.models_dir)
        self.tray.set_models_missing(missing)
        if self._download_attempt <= 3:
            delay = 5 * (2 ** (self._download_attempt - 1))
            self._download_retry_scheduled = True
            self.tray.set_download_status(
                f"Download failed; retrying in {delay} s"
            )
            log.warning(
                "Model download failed (%s); retry %d/3 in %d s",
                error,
                self._download_attempt,
                delay,
            )
            QTimer.singleShot(delay * 1000, self._retry_download)
            return
        self._download_retry_scheduled = False
        self.tray.set_download_status(None)
        self.tray.notify(
            "Model download failed",
            f"{error}\n\nText search still works. Use the tray menu to try the "
            "download again later.",
            warning=True,
            action="download_models",
        )

    def _retry_download(self) -> None:
        self._download_retry_scheduled = False
        if self._download_task is not None:
            return
        if models.models_ready(self.config.models_dir) and models.speech_ready(
            self.config.models_dir
        ):
            self.tray.set_download_status(None)
            self.tray.set_models_missing(False)
            return
        self._start_download_task()

    # -- settings ------------------------------------------------------------

    def show_settings(self) -> None:
        if self.settings is None:
            self.settings = SettingsDialog(self)
        self.settings.refresh()
        self.settings.show()
        self.settings.raise_()
        self.settings.activateWindow()

    def index_stats(self) -> dict[str, int]:
        return self.index.stats()

    def add_watched_folder(self, folder: Path) -> None:
        self.config.watched_folders.append(folder)
        self.config.save()
        self.watcher.restart()
        self._start_scan()

    def remove_watched_folder(self, folder: Path) -> None:
        self.config.watched_folders = [
            item
            for item in self.config.watched_folders
            if not _same_path(item, folder)
        ]
        self.config.save()
        removed = self.index.delete_under(folder)
        log.info("Removed %d row(s) under %s", removed, folder)
        self.watcher.restart()

    def set_video_frames(self, enabled: bool) -> None:
        self.config.index_video_frames = enabled
        self.config.save()
        if enabled:
            self.indexer.resume_video_frames()

    def set_speech(self, enabled: bool) -> None:
        self.config.index_speech = enabled
        self.config.save()
        if enabled:
            self.indexer.backfill_speech()

    def set_speech_max_minutes(self, minutes: int) -> None:
        self.config.speech_max_minutes = max(0, int(minutes))
        self.config.save()
        self.indexer.apply_speech_limit()

    def begin_hotkey_capture(self) -> None:
        if self.hotkey is not None:
            self.hotkey.unregister()
            self.hotkey = None
        self.tray.set_hotkey(None)

    def apply_hotkey(self, spec: str) -> str | None:
        """Register and save *spec*; returns an error message when Windows refuses."""
        try:
            hotkey = GlobalHotkey(spec)
        except ValueError as exc:
            return str(exc)
        if not hotkey.register():
            self._restore_hotkey()
            return "That combination is already used by another app."
        if self.hotkey is not None:
            self.hotkey.unregister()
        hotkey.activated.connect(self.show_search)
        self.hotkey = hotkey
        self.config.hotkey = spec
        self.config.hotkey_is_custom = True
        self.config.save()
        self.tray.set_hotkey(spec)
        log.info("Global hotkey changed to %s", spec)
        return None

    def end_hotkey_capture(self) -> None:
        if self.hotkey is None:
            self._register_hotkey()

    def _restore_hotkey(self) -> None:
        if self.hotkey is None:
            self._register_hotkey()

    def rebuild_index(self) -> None:
        self.indexer.pause()
        self.indexer.clear_queue()
        self.index.clear_all()
        self._rescan_after_scan = True
        self._start_scan()
        self.indexer.resume()
        self.tray.set_paused(False)
        log.info("Index rebuilt from scratch")

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
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--selftest":
        from .selftest import run_selftest

        return run_selftest(args[1] if len(args) == 2 else None)

    config = Config.load()
    first_run = not config.config_path.is_file()
    setup_logging(config)
    if not first_run:
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

    controller = ShotFinderApp(app, config, instance, first_run=first_run)
    return controller.run()
