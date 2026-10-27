"""Filesystem watcher: initial scan plus live create/modify/delete/move events."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterator, Protocol

from watchdog.events import FileSystemEvent, FileSystemEventHandler, FileMovedEvent
from watchdog.observers import Observer

from .config import Config

log = logging.getLogger(__name__)


class Sink(Protocol):
    """What the watcher needs from the indexing worker."""

    def enqueue(self, path: Path) -> bool: ...

    def remove(self, path: Path) -> None: ...


def _norm(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _is_under(path: str | Path, root: str | Path) -> bool:
    try:
        return os.path.commonpath([_norm(path), _norm(root)]) == _norm(root)
    except ValueError:
        # Different drives.
        return False


def _unique_roots(folders: list[Path]) -> list[Path]:
    """Drop missing and nested roots so each file is scanned once."""
    roots: list[Path] = []
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            log.warning("Watched folder does not exist: %s", folder)
            continue
        if any(_is_under(folder, root) for root in roots):
            continue
        roots = [root for root in roots if not _is_under(root, folder)]
        roots.append(folder)
    return roots


def iter_media_files(config: Config) -> Iterator[Path]:
    """Yield supported media files below the watched folders, skipping the data directory."""

    def on_error(error: OSError) -> None:
        log.warning("Could not scan %s: %s", getattr(error, "filename", "?"), error)

    for root in _unique_roots(config.watched_folders):
        for dirpath, dirnames, filenames in os.walk(
            root, onerror=on_error, followlinks=False
        ):
            current = Path(dirpath)
            dirnames[:] = [
                name
                for name in dirnames
                if not _is_under(current / name, config.data_dir)
            ]
            for filename in filenames:
                path = current / filename
                if config.is_supported(path):
                    yield path


class _Handler(FileSystemEventHandler):
    def __init__(self, config: Config, sink: Sink, roots: list[Path]) -> None:
        self._config = config
        self._sink = sink
        self._roots = roots
        self._recent: dict[str, tuple[int, int, float]] = {}
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def _watched(self, path: Path) -> bool:
        if _is_under(path, self._config.data_dir):
            return False
        return any(_is_under(path, root) for root in self._roots)

    def _maybe_enqueue(self, raw_path: str | Path) -> None:
        path = Path(raw_path)
        if not self._config.is_supported(path) or not self._watched(path):
            return
        try:
            stat = path.stat()
        except OSError:
            return
        if not path.is_file():
            return

        key = _norm(path)
        signature = (stat.st_mtime_ns, stat.st_size)
        now = time.monotonic()
        previous = self._recent.get(key)
        if (
            previous is not None
            and previous[0] == signature[0]
            and previous[1] == signature[1]
            and now - previous[2] < 2.0
        ):
            return
        self._recent[key] = (signature[0], signature[1], now)
        if len(self._recent) > 4096:
            cutoff = now - 60.0
            self._recent = {
                entry: value for entry, value in self._recent.items() if value[2] >= cutoff
            }
        try:
            self._sink.enqueue(path)
        except FileNotFoundError:
            return
        except OSError:
            # Windows briefly locks a file during rename/save. Retry once.
            log.warning(
                "Could not read %s yet; retrying in %.0f s",
                path,
                self._config.retry_delay_seconds,
            )
            self._schedule_retry(path)
        except Exception:
            log.exception("Could not enqueue %s", path)

    def _schedule_retry(self, path: Path) -> None:
        if self._closed:
            return
        timer = threading.Timer(
            self._config.retry_delay_seconds, self._retry_enqueue, args=(path,)
        )
        timer.daemon = True
        timer.start()

    def _retry_enqueue(self, path: Path) -> None:
        if self._closed:
            return
        try:
            self._sink.enqueue(path)
        except FileNotFoundError:
            return
        except OSError:
            log.warning("Could not enqueue %s after retry", path, exc_info=True)
        except Exception:
            log.exception("Could not enqueue %s after retry", path)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._maybe_enqueue(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._maybe_enqueue(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._config.is_supported(path) and self._watched(path):
            try:
                self._sink.remove(path)
            except Exception:
                log.exception("Could not remove %s", path)

    def on_moved(self, event: FileMovedEvent) -> None:
        if event.is_directory:
            return
        source = Path(event.src_path)
        destination = Path(event.dest_path)
        if self._config.is_supported(destination) and self._watched(destination):
            # The worker finds the old row by hash and updates its path.
            self._maybe_enqueue(destination)
        elif self._config.is_supported(source) and self._watched(source):
            # Moved outside the watched folders (or to an unsupported extension).
            try:
                self._sink.remove(source)
            except Exception:
                log.exception("Could not remove %s", source)


class Watcher:
    """Initial full scan and a watchdog observer for live changes."""

    def __init__(self, config: Config, sink: Sink) -> None:
        self._config = config
        self._sink = sink
        self._observer: Observer | None = None
        self._handler: _Handler | None = None

    def scan(self) -> int:
        """Walk every watched folder and enqueue each supported file."""
        found = 0
        for path in iter_media_files(self._config):
            found += 1
            try:
                self._sink.enqueue(path)
            except Exception:
                log.exception("Could not enqueue %s", path)
        log.info("Initial scan found %d media file(s)", found)
        return found

    def start(self) -> None:
        roots = _unique_roots(self._config.watched_folders)
        if not roots:
            log.warning("No existing watched folders; live watching disabled")
            return
        handler = _Handler(self._config, self._sink, roots)
        observer = Observer()
        for root in roots:
            observer.schedule(handler, str(root), recursive=True)
        observer.start()
        self._observer = observer
        self._handler = handler
        log.info("Watching %d folder(s): %s", len(roots), ", ".join(map(str, roots)))

    def stop(self) -> None:
        observer = self._observer
        handler = self._handler
        if handler is not None:
            handler.close()
        self._handler = None
        if observer is None:
            return
        observer.stop()
        observer.join(timeout=5.0)
        self._observer = None
