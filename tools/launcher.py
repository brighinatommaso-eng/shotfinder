"""Frozen entry point.

PyInstaller runs this file as a plain script, so the package-relative import in
``shotfinder/__main__.py`` cannot be used here.
"""

from __future__ import annotations

import sys

from shotfinder.app import main

if __name__ == "__main__":
    sys.exit(main())
