"""Directory and image filters shared by the scanner, worker and purge pass.

Build trees, package caches and version-control metadata contain thousands of
icons that are not the user's media. A directory is skipped when its name is in
``exclude_dir_names`` or when it looks like a project root (``.git``,
``package.json``, a solution/project file, Unity's ``Assets`` +
``ProjectSettings``, ``pyproject.toml``). Watched folders themselves are exempt
from the project-marker rule.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from PIL import Image

from .config import Config

MIN_IMAGE_DIMENSION = 128


def _norm(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def has_project_marker(directory: str | Path) -> bool:
    """True when *directory* looks like a source tree rather than media."""
    try:
        names = {entry.name for entry in Path(directory).iterdir()}
    except OSError:
        return False
    if ".git" in names or "package.json" in names or "pyproject.toml" in names:
        return True
    if "Assets" in names and "ProjectSettings" in names:
        return True
    return any(name.endswith((".sln", ".csproj")) for name in names)


class Exclusions:
    """Decides whether a directory or file is outside the media scope."""

    def __init__(self, config: Config, roots: Iterable[str | Path]) -> None:
        self._names = {name.lower() for name in config.exclude_dir_names}
        self._roots = {_norm(root) for root in roots}
        self._cache: dict[str, bool] = {}

    def excludes_dir(self, directory: str | Path) -> bool:
        directory = Path(directory)
        key = _norm(directory)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if key in self._roots:
            result = False
        elif directory.name.lower() in self._names:
            result = True
        else:
            result = has_project_marker(directory)
        self._cache[key] = result
        return result

    def excludes_file(self, path: str | Path) -> bool:
        """True when any ancestor is excluded; the watched root itself is not."""
        for parent in Path(path).parents:
            if _norm(parent) in self._roots:
                return False
            if self.excludes_dir(parent):
                return True
        return False


def is_small_image(path: str | Path, minimum: int = MIN_IMAGE_DIMENSION) -> bool:
    """Read the image header only: True when either side is below *minimum*."""
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        return False
    return min(width, height) < minimum
