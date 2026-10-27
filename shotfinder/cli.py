"""Command line interface: index media or search the local index."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import models
from .config import Config
from .index import Index
from .logs import setup_logging
from .search import search
from .watcher import Watcher
from .worker import Indexer

log = logging.getLogger(__name__)


def _config(args: argparse.Namespace) -> Config:
    config = Config.load(args.data_dir)
    setup_logging(config)
    return config


def _format_ts(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _warn_missing_folders(config: Config) -> None:
    for folder in config.missing_watched_folders():
        log.warning("Watched folder does not exist: %s", folder)
        print(f"Warning: watched folder does not exist: {folder}", file=sys.stderr)


def _cmd_index(args: argparse.Namespace) -> int:
    config = _config(args)
    config.save()
    _warn_missing_folders(config)

    print(f"Data directory: {config.data_dir}")
    print(f"Config: {config.config_path}")
    print("Watched folders:")
    if config.watched_folders:
        for folder in config.watched_folders:
            suffix = "" if folder.is_dir() else "  (missing)"
            print(f"  {folder}{suffix}")
    else:
        print("  (none - add folders to config.json)")

    index = Index(config.db_path)
    indexer = Indexer(config, index)
    watcher = Watcher(config, indexer)
    indexer.start()
    try:
        found = watcher.scan()
        print(f"Found {found} media file(s); processing...")
        last_line = ""
        while not indexer.wait_idle(timeout=0.5):
            stats = indexer.stats
            line = (
                f"processed {stats['processed']}  skipped {stats['skipped']}  "
                f"errors {stats['failed']}  outstanding {stats['outstanding']}"
            )
            if line != last_line:
                print(f"\r{line}", end="", flush=True)
                last_line = line
        if last_line:
            print()
    finally:
        indexer.stop()
        index.close()

    stats = indexer.stats
    print(
        f"Done: {stats['processed']} indexed, {stats['skipped']} skipped, "
        f"{stats['failed']} error(s); purged {stats['purged']} excluded file(s)."
    )
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    config = _config(args)
    _warn_missing_folders(config)
    query = " ".join(args.query).strip()
    index = Index(config.db_path)
    try:
        rows = search(config, index, query, limit=args.limit)
    finally:
        index.close()

    if query and not models.models_ready(config.models_dir):
        print(
            "Vision models not installed; showing text matches only. "
            "Run `python -m shotfinder.cli models download`.",
            file=sys.stderr,
        )

    if not rows:
        print("No results.")
        return 0

    for row in rows:
        timestamp = row.get("ts")
        prefix = f"[{_format_ts(float(timestamp))}] " if timestamp is not None else ""
        print(f"[{row['match_type']}] {prefix}{row['path']}")
        snippet = " ".join((row.get("snippet") or "").split())
        if snippet:
            print(f"    {snippet}")
        if args.explain:
            fts_rank = row.get("fts_rank")
            cosine = row.get("cosine")
            rank_text = "-" if fts_rank is None else str(fts_rank)
            cosine_text = "-" if cosine is None else f"{float(cosine):.4f}"
            print(f"    fts_rank={rank_text} cosine={cosine_text}")
    return 0


def _cmd_models_download(args: argparse.Namespace) -> int:
    config = _config(args)
    print(f"Models directory: {config.models_dir}")
    current = {"filename": ""}

    def progress(model: models.ModelFile, done: int, total: int) -> None:
        if model.filename != current["filename"]:
            if current["filename"]:
                print()
            current["filename"] = model.filename
        percent = (done * 100 // total) if total else 0
        print(
            f"\r{model.filename}: {done / 1048576:.1f}/{total / 1048576:.1f} MB "
            f"({percent}%)",
            end="",
            flush=True,
        )

    try:
        paths = models.download_models(config.models_dir, progress=progress)
    except models.ModelDownloadError as exc:
        print(f"\nError: {exc}")
        return 1
    if current["filename"]:
        print()
    for path in paths:
        print(f"  {path}")
    print(
        "Models ready. Run `python -m shotfinder.cli index` to backfill visual embeddings."
    )
    return 0


def _cmd_models_status(args: argparse.Namespace) -> int:
    config = _config(args)
    print(f"Models directory: {config.models_dir}")
    ready = True
    for model in models.MODELS:
        path = config.models_dir / model.filename
        if path.is_file():
            actual = path.stat().st_size
            if actual == model.size:
                state = f"ok ({actual / 1048576:.1f} MB)"
            else:
                state = f"size mismatch: {actual} bytes, expected {model.size}"
                ready = False
        else:
            state = "missing"
            ready = False
        print(f"  {model.filename:26} {state}")
    if ready:
        print("All models present.")
    else:
        print("Run `python -m shotfinder.cli models download`.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the ShotFinder data directory.",
    )

    parser = argparse.ArgumentParser(
        prog="python -m shotfinder.cli",
        description="Index and search local screenshots, photos and videos.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "index",
        parents=[common],
        help="Scan watched folders and process the index queue.",
    )
    search = subparsers.add_parser(
        "search",
        parents=[common],
        help="Hybrid text + visual search (or list recent files when empty).",
    )
    search.add_argument("query", nargs="*", help="Words to search for.")
    search.add_argument(
        "--limit", type=int, default=40, help="Maximum number of results (default: 40)."
    )
    search.add_argument(
        "--explain",
        action="store_true",
        help="Print the raw FTS rank and CLIP cosine similarity per result.",
    )

    models_parser = subparsers.add_parser(
        "models",
        help="Download or inspect the ONNX vision models.",
    )
    models_sub = models_parser.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser(
        "download",
        parents=[common],
        help="Download missing models into the data directory.",
    )
    models_sub.add_parser(
        "status",
        parents=[common],
        help="Show which models are present.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page when output is piped.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "index":
            return _cmd_index(args)
        if args.command == "search":
            return _cmd_search(args)
        if args.command == "models":
            if args.models_command == "download":
                return _cmd_models_download(args)
            if args.models_command == "status":
                return _cmd_models_status(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        log.exception("Command failed")
        print(f"Error: {exc}")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
