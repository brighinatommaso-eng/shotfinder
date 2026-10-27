"""Low-priority background worker: pulls the queue and runs extractors."""

from __future__ import annotations

import heapq
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

import psutil
from PIL import Image

from . import models
from .config import Config
from .exclusions import (
    CLOUD_ONLY_REASON,
    MIN_IMAGE_DIMENSION,
    Exclusions,
    is_cloud_only,
    is_small_image,
)
from .extractors.ocr import OCRUnavailableError, ocr_image
from .extractors.video import (
    MAX_KEPT_FRAMES,
    FrameCursor,
    VideoUnavailableError,
    iter_frames,
)
from .index import Index

log = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, int]], None]

VIDEO_NOT_IMPLEMENTED_REASON = "video indexing is not implemented yet (Phase 4)"
VIDEO_MODELS_REASON = "vision models missing; install them to index videos"
VIDEO_EMPTY_REASON = "no video frames could be decoded"


def _priority(kind: str) -> int:
    """Images first, then videos, then anything unexpected."""
    return 0 if kind == "image" else 1 if kind == "video" else 2


class Indexer:
    """Owns the indexing thread and its in-memory queue.

    SQLite remains the source of truth: pending rows are recovered on start, and
    every stage writes its result before the next stage begins.
    """

    def __init__(
        self,
        config: Config,
        index: Index,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self.index = index
        self._on_progress = on_progress
        # (priority, sequence, file_id); the sequence keeps FIFO within a kind.
        self._queue: queue.PriorityQueue[tuple[int, int, int]] = queue.PriorityQueue()
        self._sequence = 0
        self._retries: list[tuple[float, int, int, int]] = []
        self._attempts: dict[int, int] = {}
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._outstanding = 0
        self._stats = {
            "enqueued": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "purged": 0,
        }
        self._ocr_error: str | None = None
        self._vision_missing_logged = False

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.purge_excluded()
        self._reset_unimplemented_videos()
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

    def _notify_progress(self) -> None:
        """Report a stats snapshot; runs on whichever thread changed the queue."""
        callback = self._on_progress
        if callback is None:
            return
        try:
            callback(self.stats)
        except Exception:
            log.debug("Progress callback failed", exc_info=True)

    # -- queue ---------------------------------------------------------------

    def enqueue(self, path: str | Path) -> bool:
        """Insert/refresh a path in the index and queue it when work is needed."""
        path = Path(path)
        kind = self.config.kind_for(path)
        if kind is None:
            return False
        if not self.config.index_cloud_only_files and is_cloud_only(path):
            # Hashing would trigger a OneDrive download. Record the placeholder
            # and let a later scan re-check it once the user hydrates it.
            self.index.mark_cloud_only(path, kind)
            return False
        file_id, needs_indexing = self.index.upsert_file(path, kind)
        if needs_indexing:
            with self._cond:
                self._stats["enqueued"] += 1
            self._push(file_id, kind)
        return needs_indexing

    def remove(self, path: str | Path) -> None:
        row = self.index.find_by_path(path)
        if row is not None:
            self.index.delete_file(int(row["id"]))

    def _next_sequence(self) -> int:
        with self._cond:
            self._sequence += 1
            return self._sequence

    def _push(self, file_id: int, kind: str) -> None:
        with self._cond:
            self._outstanding += 1
        self._queue.put((_priority(kind), self._next_sequence(), file_id))
        self._notify_progress()

    def _requeue(self, file_id: int, kind: str) -> None:
        """Put a partly processed video back without changing the outstanding count."""
        self._queue.put((_priority(kind), self._next_sequence(), file_id))
        self._notify_progress()

    def _recover_pending(self) -> None:
        rows = self.index.next_pending(limit=-1)
        for row in rows:
            self._push(int(row["id"]), str(row["kind"]))
        if rows:
            log.info("Recovered %d pending file(s) from the index", len(rows))

    def _reset_unimplemented_videos(self) -> None:
        """Videos skipped by Phase 3 must be indexed now that frames exist."""
        file_ids = self.index.reset_skipped_videos(VIDEO_NOT_IMPLEMENTED_REASON)
        for file_id in file_ids:
            self._push(file_id, "video")
        if file_ids:
            log.info("Queued %d video(s) previously skipped", len(file_ids))

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
        count = self.backfill_vision()
        if count:
            log.info("Queued %d image(s) for vision backfill", count)

    def backfill_vision(self) -> int:
        """Queue images without visual embeddings and videos skipped for models."""
        rows = self.index.next_vision_backfill(limit=-1)
        for row in rows:
            self._push(int(row["id"]), "image")
        videos = self.index.reset_skipped_videos(VIDEO_MODELS_REASON)
        for file_id in videos:
            self._push(file_id, "video")
        return len(rows) + len(videos)

    def prepare_vision(self) -> None:
        """Import the ONNX runtime from the calling (main) thread."""
        self._load_vision_runtime()

    def purge_excluded(self) -> int:
        """Delete rows that are now outside the media scope.

        Directory exclusions always apply. A tiny image is purged only when it
        was already indexed as done; rows marked 'skipped' are left alone so the
        next scan does not recreate and re-skip them forever.
        """
        exclusions = Exclusions(self.config, self.config.watched_folders)
        removed = 0
        for row in self.index.all_files():
            path = Path(row["path"])
            excluded = exclusions.excludes_file(path)
            if not excluded and row["kind"] == "image" and row["status"] != "skipped":
                excluded = is_small_image(path)
            if not excluded:
                continue
            self.index.delete_file(int(row["id"]))
            removed += 1
        with self._cond:
            self._stats["purged"] = removed
        if removed:
            log.info("Purged %d excluded file(s) from the index", removed)
        return removed

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
                _, _, file_id = self._queue.get(timeout=0.5)
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
            requeue: str | None = None
            try:
                requeue = self._process(file_id)
                if requeue is None:
                    self._attempts.pop(file_id, None)
            except Exception as exc:
                self._handle_failure(file_id, exc)
            finally:
                if requeue is not None and not self._stop.is_set():
                    self._requeue(file_id, requeue)
                else:
                    with self._cond:
                        self._outstanding = max(0, self._outstanding - 1)
                        if self._outstanding == 0:
                            self._cond.notify_all()
                    self._notify_progress()

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
            _, _, file_id, priority = heapq.heappop(self._retries)
            self._queue.put((priority, self._next_sequence(), file_id))

    def _handle_failure(self, file_id: int, exc: Exception) -> None:
        attempt = self._attempts.get(file_id, 0) + 1
        self._attempts[file_id] = attempt
        if attempt <= self.config.max_retries:
            due = time.monotonic() + self.config.retry_delay_seconds
            row = self.index.get_file(file_id)
            priority = _priority(str(row["kind"])) if row is not None else 2
            heapq.heappush(
                self._retries,
                (due, self._next_sequence(), file_id, priority),
            )
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

    def _process(self, file_id: int) -> str | None:
        """Run one turn for *file_id*; return a kind when it must be requeued."""
        row = self.index.get_file(file_id)
        if row is None:
            return None
        stages = {stage for stage in (row["stages_done"] or "").split(",") if stage}
        kind = str(row["kind"])
        if row["status"] == "done":
            # Only an image waiting for its vision embedding is still work.
            if not (kind == "image" and "vision" not in stages):
                return None
        elif row["status"] != "pending":
            return None
        path = Path(row["path"])
        if not path.is_file():
            # The watcher may have moved the file while this item was queued.
            current = self.index.get_file(file_id)
            if current is None:
                return None
            if current["path"] != row["path"]:
                if current["status"] == "pending":
                    self._push(file_id, str(current["kind"]))
                return None
            if not Path(current["path"]).is_file():
                self.index.delete_file(file_id)
            return None

        if kind == "video":
            return self._process_video(row)
        if kind != "image":
            self.index.mark_status(file_id, "skipped", f"unsupported kind: {kind}")
            self._count("skipped")
            return None
        self._process_image(row)
        return None

    def _process_video(self, row: dict[str, Any]) -> str | None:
        """Sample one budget of frames, embed them and persist the cursor."""
        file_id = int(row["id"])
        path = Path(row["path"])
        content_hash = row["hash"]

        if not models.models_ready(self.config.models_dir):
            if self.index.mark_status(
                file_id,
                "skipped",
                VIDEO_MODELS_REASON,
                expected_hash=content_hash,
            ):
                self._count("skipped")
            if not self._vision_missing_logged:
                log.info(
                    "Vision models not installed in %s; videos wait for them",
                    self.config.models_dir,
                )
                self._vision_missing_logged = True
            return None

        if not self.config.index_cloud_only_files and is_cloud_only(path):
            # Dehydrated since it was queued; decoding would pull it from the
            # cloud. Leave it for a later scan once the user hydrates it.
            self.index.mark_status(
                file_id, "skipped", CLOUD_ONLY_REASON, expected_hash=content_hash
            )
            self._count("skipped")
            return None

        progress = self.index.video_progress(file_id)
        if progress is None:
            cursor = FrameCursor()
        else:
            cursor = FrameCursor(
                next_ts=float(progress["next_ts"]),
                kept=int(progress["kept"]),
                last_hash=progress["last_hash"],
            )
        budget = max(1, int(self.config.video_frames_per_turn))
        failure: str | None = None
        try:
            frames = list(iter_frames(path, cursor, max_frames=budget))
        except VideoUnavailableError as exc:
            frames = []
            failure = str(exc)
        except Exception as exc:  # noqa: BLE001 - reported per file
            frames = []
            failure = f"video decoding failed: {exc}"
            log.warning("Video decoding failed for %s", path, exc_info=True)

        if frames:
            from .extractors.vision import embed_images

            vectors = embed_images(
                [frame.image for frame in frames],
                models_dir=self.config.models_dir,
                threads=self.config.onnx_threads,
                batch_size=self.config.vision_batch_size,
            )
            if len(vectors) != len(frames):
                raise RuntimeError(
                    f"expected {len(frames)} vectors, got {len(vectors)}"
                )
            items = [
                (frame.timestamp, vector)
                for frame, vector in zip(frames, vectors)
            ]
        else:
            items = []

        if not self.index.add_video_frames(
            file_id,
            items,
            next_ts=cursor.next_ts,
            kept=cursor.kept,
            last_hash=cursor.last_hash,
            expected_hash=content_hash,
        ):
            # The file changed while ffmpeg was running; its queued refresh wins.
            return None

        if frames:
            log.debug(
                "Video %s: kept %d frame(s), %d total, through %.0fs",
                path.name,
                len(frames),
                cursor.kept,
                cursor.next_ts,
            )

        if failure:
            if cursor.kept:
                log.warning("Video %s decoded partially: %s", path.name, failure)
                self._finish_video(row, cursor)
            else:
                self.index.clear_video_progress(file_id)
                self.index.mark_status(
                    file_id, "skipped", failure, expected_hash=content_hash
                )
                self._count("skipped")
            return None

        if cursor.kept >= MAX_KEPT_FRAMES or cursor.exhausted:
            if cursor.kept == 0:
                self.index.clear_video_progress(file_id)
                self.index.mark_status(
                    file_id,
                    "skipped",
                    VIDEO_EMPTY_REASON,
                    expected_hash=content_hash,
                )
                self._count("skipped")
            else:
                self._finish_video(row, cursor)
            return None
        return "video"

    def _finish_video(self, row: dict[str, Any], cursor: FrameCursor) -> None:
        file_id = int(row["id"])
        stages = {stage for stage in (row["stages_done"] or "").split(",") if stage}
        stages.add("frames")
        if self.index.mark_status(
            file_id,
            "done",
            None,
            stages_done=",".join(sorted(stages)),
            expected_hash=row["hash"],
        ):
            self.index.clear_video_progress(file_id)
            self._count("processed")
        log.debug(
            "Video %s done: %d kept frame(s)", Path(row["path"]).name, cursor.kept
        )

    def _process_image(self, row: dict[str, Any]) -> None:
        file_id = int(row["id"])
        path = Path(row["path"])
        content_hash = row["hash"]
        stages = {stage for stage in (row["stages_done"] or "").split(",") if stage}

        if not self.config.index_cloud_only_files and is_cloud_only(path):
            # The file was hydrated when queued and dehydrated since. Opening
            # it now would pull it from the cloud.
            self.index.mark_status(
                file_id, "skipped", CLOUD_ONLY_REASON, expected_hash=content_hash
            )
            self._count("skipped")
            log.debug("Skipped %s: cloud-only placeholder", path)
            return

        try:
            with Image.open(path) as probe:
                width, height = probe.size
        except Exception:
            width = height = 0
        if 0 < min(width, height) < MIN_IMAGE_DIMENSION:
            self.index.mark_status(
                file_id,
                "skipped",
                f"image too small ({width}x{height})",
                expected_hash=content_hash,
            )
            self._count("skipped")
            log.debug("Skipped %s: %dx%d is below %d px", path, width, height, MIN_IMAGE_DIMENSION)
            return

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
        self._notify_progress()
