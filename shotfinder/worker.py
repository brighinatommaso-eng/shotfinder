"""Low-priority background worker: pulls the queue and runs extractors."""

from __future__ import annotations

import heapq
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import psutil
from PIL import Image

from . import models
from .config import Config
from .extractors.ocr import OCRUnavailableError, ocr_image
from .index import Index

log = logging.getLogger(__name__)


class Indexer:
    """Owns the indexing thread and its in-memory queue.

    SQLite remains the source of truth: pending rows are recovered on start, and
    every stage writes its result before the next stage begins.
    """

    def __init__(self, config: Config, index: Index) -> None:
        self.config = config
        self.index = index
        self._queue: queue.Queue[int] = queue.Queue()
        self._retries: list[tuple[float, int, int]] = []
        self._attempts: dict[int, int] = {}
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._outstanding = 0
        self._stats = {"enqueued": 0, "processed": 0, "skipped": 0, "failed": 0}
        self._ocr_error: str | None = None
        self._vision_missing_logged = False

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._load_vision_runtime()
        self._recover_pending()
        self._recover_vision()
        self._thread = threading.Thread(
            target=self._run, name="shotfinder-worker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=10.0)
            self._thread = None

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until the queue, retries and the current file are all done."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._outstanding > 0:
                if deadline is None:
                    self._cond.wait(1.0)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    @property
    def stats(self) -> dict[str, int]:
        with self._cond:
            return {
                **self._stats,
                "outstanding": self._outstanding,
                "queued": self._queue.qsize(),
                "retries": len(self._retries),
            }

    # -- queue ---------------------------------------------------------------

    def enqueue(self, path: str | Path) -> bool:
        """Insert/refresh a path in the index and queue it when work is needed."""
        path = Path(path)
        kind = self.config.kind_for(path)
        if kind is None:
            return False
        file_id, needs_indexing = self.index.upsert_file(path, kind)
        if needs_indexing:
            with self._cond:
                self._stats["enqueued"] += 1
            self._push(file_id)
        return needs_indexing

    def remove(self, path: str | Path) -> None:
        row = self.index.find_by_path(path)
        if row is not None:
            self.index.delete_file(int(row["id"]))

    def _push(self, file_id: int) -> None:
        with self._cond:
            self._outstanding += 1
        self._queue.put(file_id)

    def _recover_pending(self) -> None:
        rows = self.index.next_pending(limit=-1)
        for row in rows:
            self._push(int(row["id"]))
        if rows:
            log.info("Recovered %d pending file(s) from the index", len(rows))

    def _load_vision_runtime(self) -> None:
        """Import onnxruntime from the main thread before any OCR runs.

        Loading the ORT native module inside the worker thread after winrt OCR
        has initialised COM crashes the process with an access violation, so the
        import has to happen here. Sessions themselves stay lazy.
        """
        if not models.models_ready(self.config.models_dir):
            return
        try:
            from .extractors import vision  # noqa: F401
        except Exception:
            log.warning(
                "Could not load the ONNX runtime; images will be OCR-only",
                exc_info=True,
            )

    def _recover_vision(self) -> None:
        """Queue images indexed before the vision models were available."""
        if not models.models_ready(self.config.models_dir):
            return
        rows = self.index.next_vision_backfill(limit=-1)
        for row in rows:
            self._push(int(row["id"]))
        if rows:
            log.info("Queued %d image(s) for vision backfill", len(rows))

    # -- worker thread -------------------------------------------------------

    def _run(self) -> None:
        self._lower_priority()
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            if self._paused.is_set():
                self._stop.wait(0.2)
                continue
            self._dispatch_due_retries()
            try:
                file_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._should_throttle():
                self._stop.wait(self.config.cpu_pause_seconds)
                if self._stop.is_set():
                    # Leave the row pending; the next start recovers it.
                    with self._cond:
                        self._outstanding = max(0, self._outstanding - 1)
                        if self._outstanding == 0:
                            self._cond.notify_all()
                    break
            try:
                self._process(file_id)
                self._attempts.pop(file_id, None)
            except Exception as exc:
                self._handle_failure(file_id, exc)
            finally:
                with self._cond:
                    self._outstanding = max(0, self._outstanding - 1)
                    if self._outstanding == 0:
                        self._cond.notify_all()

    def _lower_priority(self) -> None:
        try:
            process = psutil.Process()
            priority = getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", None)
            process.nice(priority if priority is not None else 10)
        except Exception:
            log.warning("Could not lower the worker process priority", exc_info=True)

    def _should_throttle(self) -> bool:
        threshold = self.config.cpu_pause_threshold
        if threshold <= 0:
            return False
        try:
            cpu = psutil.cpu_percent(interval=None)
        except Exception:
            return False
        if cpu > threshold:
            log.info(
                "CPU at %.0f%% (> %.0f%%); pausing indexing for %.0f s",
                cpu,
                threshold,
                self.config.cpu_pause_seconds,
            )
            return True
        return False

    def _dispatch_due_retries(self) -> None:
        now = time.monotonic()
        while self._retries and self._retries[0][0] <= now:
            _, file_id, _ = heapq.heappop(self._retries)
            self._queue.put(file_id)

    def _handle_failure(self, file_id: int, exc: Exception) -> None:
        attempt = self._attempts.get(file_id, 0) + 1
        self._attempts[file_id] = attempt
        if attempt <= self.config.max_retries:
            due = time.monotonic() + self.config.retry_delay_seconds
            heapq.heappush(self._retries, (due, file_id, attempt))
            with self._cond:
                self._outstanding += 1
            log.warning(
                "Indexing file %s failed (%s); retry %d/%d in %.0f s",
                file_id,
                exc,
                attempt,
                self.config.max_retries,
                self.config.retry_delay_seconds,
            )
            return

        self._attempts.pop(file_id, None)
        try:
            self.index.mark_status(file_id, "error", str(exc))
        except Exception:
            log.exception("Could not mark file %s as error", file_id)
        with self._cond:
            self._stats["failed"] += 1
        log.exception("Giving up on file %s", file_id)

    def _process(self, file_id: int) -> None:
        row = self.index.get_file(file_id)
        if row is None:
            return
        stages = {stage for stage in (row["stages_done"] or "").split(",") if stage}
        # Backfill rows are already 'done'; only skip them when vision is present.
        if row["status"] == "done" and "vision" in stages:
            return
        if row["status"] not in ("pending", "done"):
            return
        path = Path(row["path"])
        if not path.is_file():
            # The watcher may have moved the file while this item was queued.
            current = self.index.get_file(file_id)
            if current is None:
                return
            if current["path"] != row["path"]:
                if current["status"] == "pending":
                    self._push(file_id)
                return
            if not Path(current["path"]).is_file():
                self.index.delete_file(file_id)
            return

        kind = row["kind"]
        if kind == "video":
            self.index.mark_status(
                file_id, "skipped", "video indexing is not implemented yet (Phase 4)"
            )
            self._count("skipped")
            return
        if kind != "image":
            self.index.mark_status(file_id, "skipped", f"unsupported kind: {kind}")
            self._count("skipped")
            return
        self._process_image(row)

    def _process_image(self, row: dict[str, Any]) -> None:
        file_id = int(row["id"])
        path = Path(row["path"])
        content_hash = row["hash"]
        stages = {stage for stage in (row["stages_done"] or "").split(",") if stage}

        if "ocr" not in stages:
            if self._ocr_error is not None:
                self.index.mark_status(file_id, "skipped", self._ocr_error)
                self._count("skipped")
                return
            try:
                lines = ocr_image(path, max_dimension=self.config.ocr_max_dimension)
            except OCRUnavailableError as exc:
                self._ocr_error = str(exc)
                log.error("Windows OCR is unavailable: %s", exc)
                self.index.mark_status(file_id, "skipped", str(exc))
                self._count("skipped")
                return
            if lines:
                count = self.index.add_text_rows(
                    file_id, lines, source="ocr", expected_hash=content_hash
                )
                if count == 0:
                    # The file changed while OCR ran; its queued refresh wins.
                    return
                log.debug("OCR %s: %d line(s)", path, count)
            else:
                log.debug("OCR %s: no text found", path)
            stages.add("ocr")

        if "vision" not in stages:
            if not models.models_ready(self.config.models_dir):
                if not self._vision_missing_logged:
                    log.info(
                        "Vision models not installed in %s; indexing images with OCR only",
                        self.config.models_dir,
                    )
                    self._vision_missing_logged = True
            else:
                try:
                    from .extractors.vision import embed_images

                    with Image.open(path) as image:
                        vectors = embed_images(
                            [image],
                            models_dir=self.config.models_dir,
                            threads=self.config.onnx_threads,
                            batch_size=self.config.vision_batch_size,
                        )
                except Exception as exc:
                    # OCR results stay valid; a later start retries the vision stage.
                    log.warning("Vision embedding failed for %s: %s", path, exc)
                else:
                    written = self.index.add_vectors(
                        file_id, [(None, vectors[0])], expected_hash=content_hash
                    )
                    if written == 0:
                        # The file changed while embedding; its queued refresh wins.
                        return
                    stages.add("vision")
                    log.debug("Vision %s: embedded", path)

        if not self.index.mark_status(
            file_id,
            "done",
            None,
            stages_done=",".join(sorted(stages)),
            expected_hash=content_hash,
        ):
            return
        self._count("processed")

    def _count(self, key: str) -> None:
        with self._cond:
            self._stats[key] += 1
