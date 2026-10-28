"""Headless end-to-end self-test for the packaged build.

``ShotFinder.exe --selftest <data_dir>`` runs before any ``QApplication`` or
tray icon exists. It imports every native dependency, downloads the models into
*data_dir*, indexes the bundled fixtures and runs one text, one visual and one
spoken query. The JSON report is written to ``<data_dir>/selftest-report.json``
and printed to stdout when the executable has a console.

Exit codes: 0 when no check failed (Windows OCR may be ``unavailable``), 1 when
a check failed, 2 when the data directory argument is missing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import __version__, models
from .config import Config
from .index import Index
from .logs import setup_logging
from .search import search
from .watcher import Watcher
from .worker import Indexer

log = logging.getLogger(__name__)

FIXTURE_NAMES = ("invoice.png", "pattern.png", "talk.mp4")
EXPECTED_OCR_TOKENS = ("ACME", "INVOICE")
INDEX_TIMEOUT_SECONDS = 900

TEXT_QUERY = "ACME INVOICE"
VISUAL_QUERY = "a colorful test pattern with bars"
# Whisper is an int8 model and a different CPU can spell a word differently
# ("quick" -> "quik" on the CI runner). The spoken query therefore uses a word
# from the transcript that was actually indexed, not a hard-coded phrase.
MIN_SPEECH_WORD_LENGTH = 4


def _longest_word(rows: list[list[Any]]) -> str | None:
    """Return the longest alphabetic word in the indexed speech rows."""
    words: list[str] = []
    for _ts, content in rows:
        words.extend(
            word
            for word in re.findall(r"[^\W\d_]+", str(content), flags=re.UNICODE)
            if len(word) >= MIN_SPEECH_WORD_LENGTH
        )
    return max(words, key=len, default=None)



class _Unavailable(RuntimeError):
    """A check that cannot run in this environment; only OCR may use this."""


def _emit(payload: dict[str, Any]) -> None:
    stream = sys.stdout
    if stream is None:
        return
    try:
        stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        stream.flush()
    except (OSError, ValueError):
        pass


def _emit_error(text: str) -> None:
    stream = sys.stderr
    if stream is None:
        return
    try:
        print(text, file=stream)
        stream.flush()
    except (OSError, ValueError):
        pass


class _Selftest:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.fixtures_source = Path(__file__).resolve().parent / "selftest_fixtures"
        self.fixtures_dir = self.data_dir / "fixtures"
        self.config: Config | None = None
        self.index: Index | None = None
        self.ocr_status = "not-run"
        self.checks: list[dict[str, Any]] = []
        self._download_percent: dict[str, int] = {}
        self._last_progress_at = 0.0
        self._last_progress: tuple[Any, Any] = (None, None)

    # -- driver --------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        self._run_check("setup", self._setup)
        if self.config is not None:
            steps: tuple[tuple[str, Callable[[], Any]], ...] = (
                ("onnxruntime", self._check_onnxruntime),
                ("ctranslate2", self._check_ctranslate2),
                ("faster_whisper_vad", self._check_vad),
                ("sqlite_vec", self._check_sqlite_vec),
                ("winrt_ocr", self._check_ocr),
                ("qt_multimedia", self._check_qt_multimedia),
                ("models", self._check_models),
                ("vision", self._check_vision),
                ("fixtures", self._check_fixtures),
                ("index", self._check_index),
                ("query_text", self._check_text_query),
                ("query_visual", self._check_visual_query),
                ("query_speech", self._check_speech_query),
            )
            for name, step in steps:
                self._run_check(name, step)
        if self.index is not None:
            try:
                self.index.close()
            except Exception:
                log.warning("Could not close the selftest index", exc_info=True)

        failed = [check["name"] for check in self.checks if check["status"] == "fail"]
        unavailable = [
            check["name"] for check in self.checks if check["status"] == "unavailable"
        ]
        report: dict[str, Any] = {
            "ok": not failed,
            "version": __version__,
            "started": started_at.isoformat(timespec="seconds"),
            "duration_seconds": round(time.perf_counter() - started, 2),
            "data_dir": str(self.data_dir),
            "models_dir": str(self.config.models_dir) if self.config else None,
            "frozen": bool(getattr(sys, "frozen", False)),
            "python": sys.version.split()[0],
            "failed": failed,
            "unavailable": unavailable,
            "checks": self.checks,
        }
        report_path = self.data_dir / "selftest-report.json"
        report["report_path"] = str(report_path)
        try:
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            report["report_write_error"] = str(exc)
            log.error("Could not write %s: %s", report_path, exc)
        return report

    def _run_check(self, name: str, func: Callable[[], Any]) -> None:
        started = time.perf_counter()
        try:
            detail = func()
        except _Unavailable as exc:
            entry: dict[str, Any] = {"status": "unavailable", "detail": str(exc)}
            log.warning("Selftest %s: unavailable (%s)", name, exc)
        except Exception as exc:  # noqa: BLE001 - every failure belongs in the report
            entry = {
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }
            log.exception("Selftest %s failed", name)
        else:
            entry = {"status": "pass", "detail": detail}
            log.info("Selftest %s: pass", name)
        entry["name"] = name
        entry["seconds"] = round(time.perf_counter() - started, 2)
        self.checks.append(entry)

    # -- setup ---------------------------------------------------------------

    def _setup(self) -> dict[str, Any]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("index.db", "index.db-wal", "index.db-shm"):
            (self.data_dir / suffix).unlink(missing_ok=True)

        config = Config(data_dir=self.data_dir)
        config.watched_folders = [self.fixtures_dir]
        config.index_video_frames = True
        config.index_speech = True
        # Failures must surface immediately instead of waiting for a retry, and
        # a busy CI machine must not throttle the worker to a halt.
        config.max_retries = 0
        config.cpu_pause_threshold = 0.0
        config.save()
        self.config = config
        setup_logging(config)

        log.info(
            "Selftest %s starting in %s (frozen=%s, python=%s)",
            __version__,
            self.data_dir,
            bool(getattr(sys, "frozen", False)),
            sys.version.split()[0],
        )
        return {
            "version": __version__,
            "data_dir": str(self.data_dir),
            "models_dir": str(config.models_dir),
            "frozen": bool(getattr(sys, "frozen", False)),
            "python": sys.version.split()[0],
            "executable": sys.executable,
        }

    # -- native imports ------------------------------------------------------

    def _check_onnxruntime(self) -> dict[str, Any]:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
        if "CPUExecutionProvider" not in providers:
            raise RuntimeError(
                f"CPUExecutionProvider missing from onnxruntime: {providers}"
            )
        return {"version": ort.__version__, "providers": providers}

    def _check_ctranslate2(self) -> dict[str, Any]:
        import ctranslate2

        return {"version": ctranslate2.__version__}

    def _check_vad(self) -> dict[str, Any]:
        import faster_whisper
        from faster_whisper.utils import get_assets_path

        asset = Path(get_assets_path()) / "silero_vad_v6.onnx"
        if not asset.is_file():
            raise RuntimeError(f"faster-whisper VAD asset missing: {asset}")
        return {
            "version": faster_whisper.__version__,
            "vad_asset": str(asset),
            "vad_bytes": asset.stat().st_size,
        }

    def _check_sqlite_vec(self) -> dict[str, Any]:
        import sqlite_vec

        loadable = Path(sqlite_vec.loadable_path())
        candidates = (loadable, loadable.with_suffix(".dll"))
        if not any(path.is_file() for path in candidates):
            raise RuntimeError(f"sqlite-vec extension missing: {loadable}")
        connection = sqlite3.connect(":memory:")
        try:
            connection.enable_load_extension(True)
            sqlite_vec.load(connection)
            connection.enable_load_extension(False)
            version = connection.execute("SELECT vec_version()").fetchone()[0]
            connection.execute("CREATE VIRTUAL TABLE t USING vec0(v float[3])")
            connection.execute("INSERT INTO t(rowid, v) VALUES (1, '[1, 0, 0]')")
            distance = connection.execute(
                "SELECT distance FROM t WHERE v MATCH '[1, 0, 0]' AND k = 1"
            ).fetchone()[0]
        finally:
            connection.close()
        return {
            "version": version,
            "loadable_path": str(loadable),
            "distance": float(distance),
        }

    def _check_ocr(self) -> dict[str, Any]:
        from .extractors.ocr import OCRUnavailableError, ocr_image

        # A missing winrt package is a packaging failure, not an environment
        # limitation: import the modules directly before asking for a recognizer.
        try:
            from winrt.windows.globalization import Language  # noqa: F401
            from winrt.windows.graphics.imaging import BitmapDecoder  # noqa: F401
            from winrt.windows.media.ocr import OcrEngine  # noqa: F401
            from winrt.windows.storage.streams import (  # noqa: F401
                DataWriter,
                InMemoryRandomAccessStream,
            )
        except ImportError as exc:
            raise RuntimeError(f"winrt OCR modules are missing: {exc}") from exc

        image = self.fixtures_source / "invoice.png"
        try:
            lines = ocr_image(image)
        except OCRUnavailableError as exc:
            self.ocr_status = "unavailable"
            raise _Unavailable(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - a broken recognizer is a failure
            raise RuntimeError(f"Windows OCR failed: {exc}") from exc

        text = "\n".join(lines).upper()
        missing = [token for token in EXPECTED_OCR_TOKENS if token not in text]
        if missing:
            raise RuntimeError(
                f"OCR read {lines!r}; expected {EXPECTED_OCR_TOKENS}, missing {missing}"
            )
        self.ocr_status = "pass"
        return {"lines": lines}

    def _check_qt_multimedia(self) -> dict[str, Any]:
        from PySide6.QtCore import QCoreApplication, QPluginLoader

        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication([])

        import PySide6

        base = Path(PySide6.__file__).resolve().parent
        plugin_dir = base / "plugins" / "multimedia"
        if not plugin_dir.is_dir():
            plugin_dir = base / "Qt" / "plugins" / "multimedia"
        if not plugin_dir.is_dir():
            raise RuntimeError(f"PySide6 multimedia plugin directory missing: {plugin_dir}")

        loaded: list[str] = []
        errors: dict[str, str] = {}
        for dll in sorted(plugin_dir.glob("*.dll")):
            loader = QPluginLoader(str(dll))
            if loader.isLoaded() or loader.load():
                loaded.append(dll.name)
            else:
                errors[dll.name] = loader.errorString()
        if not loaded:
            raise RuntimeError(
                f"no Qt multimedia plugin loaded from {plugin_dir}: {errors}"
            )

        from PySide6.QtMultimedia import QMediaPlayer

        player = QMediaPlayer()
        if player.error() != QMediaPlayer.Error.NoError:
            raise RuntimeError(f"QMediaPlayer failed: {player.errorString()}")
        return {
            "plugin_dir": str(plugin_dir),
            "loaded": loaded,
            "failed": errors,
            "player": "ok",
        }

    # -- models and index ----------------------------------------------------

    def _check_models(self) -> dict[str, Any]:
        config = self._require_config()
        models_dir = config.models_dir

        def progress(model: models.ModelFile, done: int, total: int) -> None:
            percent = int(done * 100 / total) if total else 0
            last = self._download_percent.get(model.filename, -1)
            if percent >= last + 10 or done == total:
                self._download_percent[model.filename] = percent
                log.info("Downloading %s: %d%%", model.filename, percent)

        started = time.perf_counter()
        paths = models.download_models(models_dir, progress=progress)
        missing = models.missing_files(models_dir, models.ALL_MODELS)
        if missing:
            raise RuntimeError(
                "models missing after download: "
                + ", ".join(model.filename for model in missing)
            )
        # Cached files only get a size check in download_models; hash them here
        # so a corrupted CI cache cannot produce a false green run.
        for model in models.ALL_MODELS:
            path = model.path_in(models_dir)
            digest = models.sha256_file(path)
            if digest != model.sha256:
                raise RuntimeError(
                    f"{model.filename}: sha256 mismatch "
                    f"(expected {model.sha256}, got {digest})"
                )
        total_bytes = sum(path.stat().st_size for path in paths)
        return {
            "models_dir": str(models_dir),
            "files": len(paths),
            "bytes": total_bytes,
            "seconds": round(time.perf_counter() - started, 2),
        }

    def _check_vision(self) -> dict[str, Any]:
        from PIL import Image

        from .extractors.vision import VECTOR_DIMENSIONS, embed_images, embed_text

        config = self._require_config()
        text = embed_text(
            "a quick brown fox", models_dir=config.models_dir, threads=config.onnx_threads
        )
        with Image.open(self.fixtures_source / "pattern.png") as image:
            vectors = embed_images(
                [image],
                models_dir=config.models_dir,
                threads=config.onnx_threads,
                batch_size=config.vision_batch_size,
            )
        if text.shape != (VECTOR_DIMENSIONS,):
            raise RuntimeError(f"text embedding shape {text.shape}, expected {(VECTOR_DIMENSIONS,)}")
        if vectors.shape != (1, VECTOR_DIMENSIONS):
            raise RuntimeError(
                f"image embedding shape {vectors.shape}, expected {(1, VECTOR_DIMENSIONS)}"
            )
        return {
            "dimensions": VECTOR_DIMENSIONS,
            "text_norm": round(float((text**2).sum() ** 0.5), 6),
            "image_norm": round(float((vectors[0] ** 2).sum() ** 0.5), 6),
        }

    def _check_fixtures(self) -> dict[str, Any]:
        if not self.fixtures_source.is_dir():
            raise RuntimeError(f"bundled fixtures are missing: {self.fixtures_source}")
        missing = [
            name for name in FIXTURE_NAMES if not (self.fixtures_source / name).is_file()
        ]
        if missing:
            raise RuntimeError(f"bundled fixtures are incomplete: {missing}")
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        sizes: dict[str, int] = {}
        for name in FIXTURE_NAMES:
            target = self.fixtures_dir / name
            shutil.copy2(self.fixtures_source / name, target)
            sizes[name] = target.stat().st_size
        return {"source": str(self.fixtures_source), "target": str(self.fixtures_dir), "sizes": sizes}

    def _check_index(self) -> dict[str, Any]:
        config = self._require_config()
        self.index = Index(config.db_path)
        indexer = Indexer(config, self.index, on_progress=self._on_index_progress)
        watcher = Watcher(config, indexer)
        indexer.start()
        try:
            found = watcher.scan()
            idle = indexer.wait_idle(timeout=INDEX_TIMEOUT_SECONDS)
        finally:
            indexer.stop()
            stats = indexer.stats

        rows = self.index.all_files()
        by_name = {Path(str(row["path"])).name: row for row in rows}
        problems: list[str] = []
        if found != len(FIXTURE_NAMES):
            problems.append(f"scan found {found} file(s), expected {len(FIXTURE_NAMES)}")
        if len(rows) != len(FIXTURE_NAMES):
            problems.append(f"index has {len(rows)} row(s), expected {len(FIXTURE_NAMES)}")
        if not idle:
            problems.append(f"indexer did not finish within {INDEX_TIMEOUT_SECONDS}s")
        if stats.get("failed"):
            problems.append(f"indexer reported {stats['failed']} failed file(s)")

        for name in FIXTURE_NAMES:
            row = by_name.get(name)
            if row is None:
                problems.append(f"{name}: not indexed")
                continue
            stages = {stage for stage in (row["stages_done"] or "").split(",") if stage}
            if row["status"] == "error":
                problems.append(f"{name}: error: {row['error']}")
                continue
            if name == "talk.mp4":
                if row["status"] != "done":
                    problems.append(f"{name}: status {row['status']}: {row['error']}")
                if "frames" not in stages:
                    problems.append(f"{name}: frames stage missing")
                if "speech" not in stages:
                    problems.append(f"{name}: speech stage missing: {row['error']}")
                continue
            if self.ocr_status == "pass":
                if row["status"] != "done":
                    problems.append(f"{name}: status {row['status']}: {row['error']}")
                if "ocr" not in stages:
                    problems.append(f"{name}: OCR stage missing")
                if "vision" not in stages:
                    problems.append(f"{name}: vision stage missing")
            else:
                # Without a recognizer the worker skips images before vision.
                # That is an environment limitation, not a packaging failure.
                if row["status"] == "done" and "vision" in stages:
                    continue
                if row["status"] == "skipped" and row["error"]:
                    continue
                problems.append(f"{name}: status {row['status']}: {row['error']}")

        speech = self._speech_diagnostics()
        if problems:
            raise RuntimeError(f"{'; '.join(problems)}; speech={speech!r}")
        return {
            "found": found,
            "rows": len(rows),
            "stats": stats,
            "speech": speech,
            "files": {
                name: {
                    "status": by_name[name]["status"],
                    "stages": by_name[name]["stages_done"],
                    "error": by_name[name]["error"],
                }
                for name in FIXTURE_NAMES
                if name in by_name
            },
        }

    def _speech_diagnostics(self) -> dict[str, Any]:
        """What the speech stage actually produced, for a failing CI run."""
        index = self._require_index()
        row = index.find_by_path(self.fixtures_dir / "talk.mp4")
        if row is None:
            return {"indexed": False}
        coverage = index.video_coverage(int(row["id"])) or {}
        segments = index.conn.execute(
            "SELECT ts, content FROM text_fts WHERE file_id = ? AND source = 'speech' "
            "ORDER BY ts",
            (int(row["id"]),),
        ).fetchall()
        return {
            "indexed": True,
            "speech": coverage.get("speech"),
            "speech_reason": coverage.get("speech_reason"),
            "speech_segments": coverage.get("speech_segments"),
            "has_audio": coverage.get("has_audio"),
            "duration": coverage.get("duration"),
            "rows": [[row[0], row[1]] for row in segments],
        }

    def _on_index_progress(self, stats: dict[str, Any]) -> None:
        now = time.monotonic()
        activity = (stats.get("activity"), stats.get("activity_percent"))
        if now - self._last_progress_at < 5.0 and activity == self._last_progress:
            return
        self._last_progress_at = now
        self._last_progress = activity
        log.info(
            "Indexing: processed=%s skipped=%s failed=%s outstanding=%s activity=%s",
            stats.get("processed"),
            stats.get("skipped"),
            stats.get("failed"),
            stats.get("outstanding"),
            stats.get("activity") or "-",
        )

    # -- queries -------------------------------------------------------------

    def _check_text_query(self) -> dict[str, Any]:
        if self.ocr_status != "pass":
            raise _Unavailable(
                "Windows OCR is unavailable; the OCR text query cannot be verified"
            )
        hit = self._search_hit(TEXT_QUERY, "invoice.png")
        if hit["match_type"] not in ("text", "both"):
            raise RuntimeError(f"{TEXT_QUERY!r} matched {hit['match_type']}, expected text")
        snippet = (hit.get("snippet") or "").upper()
        missing = [token for token in EXPECTED_OCR_TOKENS if token not in snippet]
        if missing:
            raise RuntimeError(
                f"{TEXT_QUERY!r} matched invoice.png but its snippet {snippet!r} "
                f"is missing {missing}"
            )
        return {
            "query": TEXT_QUERY,
            "path": hit["path"],
            "match_type": hit["match_type"],
            "snippet": hit.get("snippet"),
        }

    def _check_visual_query(self) -> dict[str, Any]:
        config = self._require_config()
        index = self._require_index()
        row = index.find_by_path(self.fixtures_dir / "pattern.png")
        stages = {
            stage for stage in ((row["stages_done"] if row else "") or "").split(",") if stage
        }
        if "vision" not in stages:
            raise _Unavailable(
                "pattern.png was not vision-indexed (Windows OCR unavailable)"
            )
        hit = self._search_hit(VISUAL_QUERY, "pattern.png")
        if hit["match_type"] not in ("visual", "both"):
            raise RuntimeError(
                f"{VISUAL_QUERY!r} matched {hit['match_type']}, expected visual"
            )
        if hit.get("cosine") is None:
            raise RuntimeError(f"{VISUAL_QUERY!r} matched pattern.png without a cosine")
        return {
            "query": VISUAL_QUERY,
            "path": hit["path"],
            "match_type": hit["match_type"],
            "cosine": hit.get("cosine"),
        }

    def _check_speech_query(self) -> dict[str, Any]:
        speech = self._speech_diagnostics()
        rows = speech.get("rows") or []
        if not rows:
            raise RuntimeError(
                "the speech stage produced no timed segments; "
                f"speech={speech!r}"
            )
        query = _longest_word(rows)
        if query is None:
            raise RuntimeError(
                f"the speech stage produced no searchable word; speech={speech!r}"
            )
        try:
            hit = self._search_hit(query, "talk.mp4")
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}; speech={speech!r}") from exc
        if hit["match_type"] not in ("text", "both"):
            raise RuntimeError(
                f"{query!r} matched {hit['match_type']}, expected speech text; "
                f"speech={speech!r}"
            )
        if hit.get("ts") is None or hit.get("source") != "speech":
            raise RuntimeError(
                f"{query!r} matched talk.mp4 without a speech timestamp "
                f"(source={hit.get('source')!r}); speech={speech!r}"
            )
        snippet = (hit.get("snippet") or "").lower()
        if query.lower() not in snippet:
            raise RuntimeError(
                f"{query!r} matched talk.mp4 but its snippet {snippet!r} "
                f"does not contain the word; speech={speech!r}"
            )
        return {
            "query": query,
            "path": hit["path"],
            "match_type": hit["match_type"],
            "ts": hit.get("ts"),
            "source": hit.get("source"),
            "snippet": hit.get("snippet"),
            "segments": len(rows),
            "transcript": [content for _ts, content in rows],
        }

    def _search_hit(self, query: str, expected_name: str) -> dict[str, Any]:
        config = self._require_config()
        index = self._require_index()
        results = search(config, index, query)
        for hit in results.all():
            if Path(str(hit["path"])).name == expected_name:
                return hit
        raise RuntimeError(
            f"{query!r} did not return {expected_name}; got "
            f"{[Path(str(hit['path'])).name for hit in results.all()]}"
        )

    # -- helpers -------------------------------------------------------------

    def _require_config(self) -> Config:
        if self.config is None:
            raise RuntimeError("setup did not complete")
        return self.config

    def _require_index(self) -> Index:
        if self.index is None:
            raise RuntimeError("the index was not built")
        return self.index


def run_selftest(data_dir: str | os.PathLike[str] | None) -> int:
    """Run the self-test and return the process exit code."""
    if not data_dir:
        payload = {
            "ok": False,
            "error": "usage: ShotFinder.exe --selftest <data_dir>",
        }
        _emit(payload)
        _emit_error(str(payload["error"]))
        return 2

    # The packaged exe must not need a desktop; Qt is only used to prove the
    # multimedia plugin loads. Models always go to <data_dir>/models so a
    # developer override cannot make CI pass with the wrong files.
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ.pop("SHOTFINDER_MODELS_DIR", None)

    directory = Path(data_dir).expanduser()
    report = _Selftest(directory).run()
    _emit(report)
    return 0 if report["ok"] else 1
