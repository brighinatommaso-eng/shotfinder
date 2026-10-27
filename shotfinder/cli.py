"""Command line interface: index media or search the local index."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .index import Index
from .logs import setup_logging
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


def _cmd_index(args: argparse.Namespace) -> int:
    config = _config(args)
    config.save()

    print(f"Data directory: {config.data_dir}")
    print(f"Config: {config.config_path}")
    print("Watched folders:")
    if config.watched_folders:
        for folder in config.watched_folders:
            print(f"  {folder}")
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
        f"{stats['failed']} error(s)."
    )
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    config = _config(args)
    query = " ".join(args.query).strip()
    index = Index(config.db_path)
    try:
        if query:
            rows = index.search_text(query, limit=args.limit)
        else:
            rows = index.recent_files(limit=args.limit)
    finally:
        index.close()

    if not rows:
        print("No results.")
        return 0

    for row in rows:
        timestamp = row.get("ts")
        prefix = f"[{_format_ts(float(timestamp))}] " if timestamp is not None else ""
        print(f"{prefix}{row['path']}")
        snippet = " ".join((row.get("snippet") or "").split())
        if snippet:
            print(f"    {snippet}")
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
        help="Search indexed text (or list recent files when the query is empty).",
    )
    search.add_argument("query", nargs="*", help="Words to search for.")
    search.add_argument(
        "--limit", type=int, default=40, help="Maximum number of results (default: 40)."
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
