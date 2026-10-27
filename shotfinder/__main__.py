"""Entry point for the ShotFinder tray app (Phase 3)."""

from __future__ import annotations

import logging

from .config import Config
from .logs import setup_logging


def main() -> int:
    config = Config.load()
    setup_logging(config)
    logging.getLogger(__name__).info("Tray app requested, but Phase 3 is not implemented yet.")
    print("ShotFinder tray app is not implemented yet (Phase 3).")
    print('Use "python -m shotfinder.cli index" or "python -m shotfinder.cli search <query>".')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
