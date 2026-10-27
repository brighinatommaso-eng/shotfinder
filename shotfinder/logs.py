"""Rotating file logging in the ShotFinder data directory."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import Config

_HANDLER_MARKER = "_shotfinder_rotating_file"


def setup_logging(config: Config, level: int = logging.INFO) -> Path:
    """Attach one rotating file handler to the root logger and return its path."""
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    path = config.logs_dir / "shotfinder.log"
    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            handler.setLevel(level)
            return path

    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    setattr(handler, _HANDLER_MARKER, True)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )
    handler.setLevel(level)
    root.setLevel(level)
    root.addHandler(handler)
    return path
