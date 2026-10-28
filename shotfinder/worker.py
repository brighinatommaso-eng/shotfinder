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
    probe_media,
    sampling_interval,
)
from .index import Index

log = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]

VIDEO_NOT_IMPLEMENTED_REASON = "video indexing is not implemented yet (Phase 4)"
VIDEO_MODELS_REASON = "vision models missing; install them to index videos"
VIDEO_EMPTY_REASON = "no video frames could be decoded"
SPEECH_NO_AUDIO_REASON = "no audio stream"
SPEECH_TOO_LONG_REASON = "longer than speech_max_minutes"

STAGE_MEDIA = "media"
STAGE_SPEECH = "speech"


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
        # (priority, sequence, file_id, stage); the sequence keeps FIFO within a
        # kind and the priority keeps speech behind every image and video frame.
        self._queue: queue.PriorityQueue[tuple[int, int, int, str]] = (
            queue.PriorityQueue()
        )
        self._sequence = 0
        self._retries: list[tuple[float, int, int, int, str]] = []
        self._attempts: dict[int, int] = {}
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._outstanding = 0
        self._activity: tuple[str, int | None] | None = None
        self._stats = {
            "enqueued": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "purged": 0,
        }
        self._ocr_error: str | None = None
        self._vision_missing_logged = False
        self._speech_missing_logged = False

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.purge_excluded()
        self._reset_unimplemented_videos()
        self._reset_truncated_videos()
        self._load_vision_runtime()
        self._backfill_names()
        self._recover_pending()
        self._recover_vision()
        self._recover_speech()
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
    def stats(self) -> dict[str, Any]:
        with self._cond:
            activity = self._activity
            return {
                **self._stats,
                "outstanding": self._outstanding,
                "queued": self._queue.qsize(),
                "retries": len(self._retries),
                "activity": activity[0] if activity else "",
                "activity_percent": activity[1] if activity else None,
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
        self._queue.put(
            (_priority(kind), self._next_sequence(), file_id, STAGE_MEDIA)
        )
        self._notify_progress()

    def _push_speech(self, file_id: int) -> None:
        """Queue a transcription behind every image and video frame."""
        with self._cond:
            self._outstanding += 1
        self._queue.put((2, self._next_sequence(), file_id, STAGE_SPEECH))
        self._notify_progress()

    def _requeue(self, file_id: int, kind: str) -> None:
        """Put a partly processed video back without changing the outstanding count."""
        self._queue.put(
            (_priority(kind), self._next_sequence(), file_id, STAGE_MEDIA)
        )
        self._notify_progress()

    def _set_activity(self, label: str | None, percent: int | None = None) -> None:
        with self._cond:
            self._activity = (label, percent) if label else None
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

    def _reset_truncated_videos(self) -> None:
        """Re-sample videos whose frames stopped before the end of the file."""
        file_ids = self.index.reset_truncated_videos()
        for file_id in file_ids:
            self._push(file_id, "video")
        if file_ids:
            log.info("Queued %d video(s) with truncated frame coverage", len(file_ids))

    def _backfill_names(self) -> None:
        count = self.index.backfill_names()
        if count:
            log.info("Added searchable filenames for %d file(s)", count)

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

    def _recover_speech(self) -> None:
        """Queue videos whose frames are done but speech has not run yet."""
        if not models.speech_ready(self.config.models_dir):
            return
        count = self.backfill_speech()
        if count:
            log.info("Queued %d video(s) for speech", count)

    def backfill_vision(self) -> int:
        """Queue images without visual embeddings and videos skipped for models."""
        rows = self.index.next_vision_backfill(limit=-1)
        for row in rows:
            self._push(int(row["id"]), "image")
        videos = self.index.reset_skipped_videos(VIDEO_MODELS_REASON)
        for file_id in videos:
            self._push(file_id, "video")
        return len(rows) + len(videos)

    def backfill_speech(self) -> int:
        """Queue videos that finished frames while the speech model was missing."""
        if not models.speech_ready(self.config.models_dir):
            return 0
        max_seconds = max(0, self.config.speech_max_minutes) * 60
        rows = self.index.next_speech_pending(max_seconds, limit=-1)
        for file_id in rows:
            self._push_speech(file_id)
        return len(rows)

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
                _, _, file_id, stage = self._queue.get(timeout=0.5)
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
                requeue = self._process(file_id, stage)
                if requeue is None:
                    self._attempts.pop(file_id, None)
            except Exception as exc:
                self._handle_failure(file_id, exc, stage)
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
            _, _, file_id, priority, stage = heapq.heappop(self._retries)
            self._queue.put((priority, self._next_sequence(), file_id, stage))

    def _handle_failure(
        self, file_id: int, exc: Exception, stage: str = STAGE_MEDIA
    ) -> None:
        attempt = self._attempts.get(file_id, 0) + 1
        self._attempts[file_id] = attempt
        if attempt <= self.config.max_retries:
            due = time.monotonic() + self.config.retry_delay_seconds
            row = self.index.get_file(file_id)
            priority = _priority(str(row["kind"])) if row is not None else 2
            if stage == STAGE_SPEECH:
                priority = 2
            heapq.heappush(
                self._retries,
                (due, self._next_sequence(), file_id, priority, stage),
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

    def _process(self, file_id: int, stage: str = STAGE_MEDIA) -> str | None:
        """Run one turn for *file_id*; return a kind when it must be requeued."""
        row = self.index.get_file(file_id)
        if row is None:
            return None
        if stage == STAGE_SPEECH:
            return self._process_speech(row)
        stages = {name for name in (row["stages_done"] or "").split(",") if name}
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

        coverage = self.index.video_coverage(file_id)
        if coverage is not None and coverage["duration"] is not None:
            duration = float(coverage["duration"])
            interval = float(coverage["interval"])
            has_audio = (
                None if coverage["has_audio"] is None else bool(coverage["has_audio"])
            )
        else:
            info = probe_media(path)
            duration = info.duration
            interval = sampling_interval(duration)
            has_audio = info.has_audio

        budget = max(1, int(self.config.video_frames_per_turn))
        failure: str | None = None
        try:
            frames = list(
                iter_frames(path, cursor, interval=interval, max_frames=budget)
            )
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

        sampled = int(round(cursor.next_ts / interval)) if interval > 0 else 0
        last_ts = cursor.next_ts - interval if sampled else None
        reached_end = cursor.exhausted or (
            duration is not None
            and last_ts is not None
            and last_ts >= duration - interval
        )
        complete = reached_end or cursor.kept >= MAX_KEPT_FRAMES
        if failure is not None and cursor.kept:
            # A decode error, not the frame budget: retrying the same bytes
            # every start would never get further.
            complete = True
        if not self.index.add_video_frames(
            file_id,
            items,
            next_ts=cursor.next_ts,
            kept=cursor.kept,
            last_hash=cursor.last_hash,
            expected_hash=content_hash,
            coverage={
                "duration": duration,
                "interval": interval,
                "sampled": sampled,
                "kept": cursor.kept,
                "last_ts": last_ts,
                "complete": int(complete),
                "has_audio": None if has_audio is None else int(has_audio),
            },
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

        if cursor.exhausted or cursor.kept >= MAX_KEPT_FRAMES:
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
            self._maybe_enqueue_speech(file_id, row)
        log.debug(
            "Video %s done: %d kept frame(s)", Path(row["path"]).name, cursor.kept
        )

    def _maybe_enqueue_speech(self, file_id: int, row: dict[str, Any]) -> None:
        """Queue transcription once frames are done, unless it cannot help."""
        coverage = self.index.video_coverage(file_id)
        if coverage is None or coverage["speech"] is not None:
            return
        content_hash = row["hash"]
        has_audio = coverage["has_audio"]
        if has_audio is not None and not int(has_audio):
            self.index.set_speech_state(
                file_id,
                "skipped",
                reason=SPEECH_NO_AUDIO_REASON,
                expected_hash=content_hash,
            )
            return
        duration = coverage["duration"]
        max_seconds = max(0, int(self.config.speech_max_minutes)) * 60
        if duration is not None and max_seconds and float(duration) > max_seconds:
            self.index.set_speech_state(
                file_id,
                "skipped",
                reason=SPEECH_TOO_LONG_REASON,
                expected_hash=content_hash,
            )
            return
        if not models.speech_ready(self.config.models_dir):
            if not self._speech_missing_logged:
                log.info(
                    "Speech model not installed in %s; videos stay searchable by frames",
                    self.config.models_dir,
                )
                self._speech_missing_logged = True
            return
        self._push_speech(file_id)

    def _process_speech(self, row: dict[str, Any]) -> None:
        """Transcribe one video and store its segments; no per-segment resume."""
        file_id = int(row["id"])
        path = Path(row["path"])
        content_hash = row["hash"]
        coverage = self.index.video_coverage(file_id)
        if coverage is None:
            # Indexed before coverage was recorded: probe once so the audio and
            # duration checks below still apply.
            info = probe_media(path)
            self.index.record_probe(
                file_id,
                duration=info.duration,
                has_audio=info.has_audio,
                complete=1,
                expected_hash=content_hash,
            )
            coverage = self.index.video_coverage(file_id)
        if (
            coverage is None
            or not int(coverage["complete"] or 0)
            or coverage["speech"] is not None
        ):
            return None
        if row["status"] != "done":
            return None
        if not models.speech_ready(self.config.models_dir):
            return None
        if coverage["has_audio"] is not None and not int(coverage["has_audio"]):
            self.index.set_speech_state(
                file_id,
                "skipped",
                reason=SPEECH_NO_AUDIO_REASON,
                expected_hash=content_hash,
            )
            return None
        duration = coverage["duration"]
        max_seconds = max(0, int(self.config.speech_max_minutes)) * 60
        if duration is not None and max_seconds and float(duration) > max_seconds:
            self.index.set_speech_state(
                file_id,
                "skipped",
                reason=SPEECH_TOO_LONG_REASON,
                expected_hash=content_hash,
            )
            return None

        name = path.name
        self._set_activity(f"Transcribing {name}", 0)
        started = time.monotonic()
        try:
            from .extractors.speech import transcribe

            segments = transcribe(
                path,
                models_dir=self.config.models_dir,
                work_dir=self.config.data_dir,
                threads=self.config.onnx_threads,
                duration=None if duration is None else float(duration),
                progress=lambda percent: self._set_activity(
                    f"Transcribing {name}", percent
                ),
            )
        except Exception as exc:  # noqa: BLE001 - a restart retries this video
            log.warning("Speech failed for %s: %s", path, exc)
            return None
        finally:
            self._set_activity(None)

        written = self.index.add_text_rows(
            file_id,
            [(text, start) for start, _end, text in segments],
            source="speech",
            expected_hash=content_hash,
        )
        if segments and written == 0:
            # The file changed while whisper was running.
            return None
        if not self.index.set_speech_state(
            file_id,
            "done",
            segments=len(segments),
            expected_hash=content_hash,
        ):
            return None
        current = self.index.get_file(file_id)
        if current is None:
            return None
        stages = {
            stage for stage in (current["stages_done"] or "").split(",") if stage
        }
        stages.add("speech")
        self.index.mark_status(
            file_id,
            str(current["status"]),
            current["error"],
            stages_done=",".join(sorted(stages)),
            expected_hash=content_hash,
        )
        log.info("Speech %s: %d segment(s)", path.name, len(segments))
        elapsed = time.monotonic() - started
        if duration:
            log.info(
                "Speech %s took %.1f s for %.0f s of audio (%.1fx realtime)",
                path.name,
                elapsed,
                float(duration),
                float(duration) / elapsed if elapsed else 0.0,
            )
        return None

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
