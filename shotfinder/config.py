"""Application paths, watched folders and tunables (dataclass + JSON file)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

APP_NAME = "ShotFinder"
DEFAULT_HOTKEY = "ctrl+alt+space"
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".heic"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})


def default_data_dir() -> Path:
    """Return %LOCALAPPDATA%\\ShotFinder (overridable for tests and portable runs)."""
    override = os.environ.get("SHOTFINDER_DATA_DIR")
    if override:
        return Path(override).expanduser()
    local_appdata = os.environ.get("LOCALAPPDATA")
    base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
    return base / APP_NAME


def default_watched_folders() -> list[Path]:
    """Return the first-run folder selection from section 5.6, keeping only existing folders."""
    home = Path.home()
    candidates = (
        home / "Pictures" / "Screenshots",
        home / "Pictures",
        home / "Videos",
        home / "Downloads",
    )
    folders: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key in seen or not candidate.is_dir():
            continue
        seen.add(key)
        folders.append(candidate)
    return folders


def _normalise_extensions(values: Any, fallback: frozenset[str]) -> set[str]:
    if not isinstance(values, (list, tuple, set)):
        return set(fallback)
    extensions: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        extension = value.strip().lower()
        if not extension.startswith("."):
            extension = "." + extension
        extensions.add(extension)
    return extensions or set(fallback)


@dataclass(slots=True)
class Config:
    """Runtime configuration; serialised to <data_dir>/config.json."""

    data_dir: Path = field(default_factory=default_data_dir)
    watched_folders: list[Path] = field(default_factory=default_watched_folders)
    image_extensions: set[str] = field(default_factory=lambda: set(IMAGE_EXTENSIONS))
    video_extensions: set[str] = field(default_factory=lambda: set(VIDEO_EXTENSIONS))
    hotkey: str = DEFAULT_HOTKEY
    ocr_max_dimension: int = 1600
    cpu_pause_threshold: float = 70.0
    cpu_pause_seconds: float = 10.0
    retry_delay_seconds: float = 30.0
    max_retries: int = 1
    onnx_threads: int = field(
        default_factory=lambda: max(1, min(4, (os.cpu_count() or 2) // 2))
    )
    vision_batch_size: int = 1

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser()
        self.watched_folders = [Path(folder).expanduser() for folder in self.watched_folders]
        self.image_extensions = {
            extension.lower() if extension.startswith(".") else "." + extension.lower()
            for extension in self.image_extensions
        }
        self.video_extensions = {
            extension.lower() if extension.startswith(".") else "." + extension.lower()
            for extension in self.video_extensions
        }

    @property
    def config_path(self) -> Path:
        return self.data_dir / "config.json"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.db"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def models_dir(self) -> Path:
        """ONNX models live beside the index unless SHOTFINDER_MODELS_DIR overrides."""
        override = os.environ.get("SHOTFINDER_MODELS_DIR")
        if override:
            return Path(override).expanduser()
        return self.data_dir / "models"

    def kind_for(self, path: str | Path) -> str | None:
        suffix = Path(path).suffix.lower()
        if suffix in self.image_extensions:
            return "image"
        if suffix in self.video_extensions:
            return "video"
        return None

    def is_supported(self, path: str | Path) -> bool:
        return self.kind_for(path) is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "watched_folders": [str(folder) for folder in self.watched_folders],
            "image_extensions": sorted(self.image_extensions),
            "video_extensions": sorted(self.video_extensions),
            "hotkey": self.hotkey,
            "ocr_max_dimension": self.ocr_max_dimension,
            "cpu_pause_threshold": self.cpu_pause_threshold,
            "cpu_pause_seconds": self.cpu_pause_seconds,
            "retry_delay_seconds": self.retry_delay_seconds,
            "max_retries": self.max_retries,
            "onnx_threads": self.onnx_threads,
            "vision_batch_size": self.vision_batch_size,
        }

    def save(self) -> None:
        """Write config.json atomically, creating the data directory if needed."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        target = self.config_path
        temporary = target.with_name(target.name + ".tmp")
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, data_dir: str | Path | None = None) -> Config:
        """Load config.json, falling back to defaults when it is missing or malformed."""
        directory = Path(data_dir).expanduser() if data_dir is not None else default_data_dir()
        config = cls(data_dir=directory)
        path = config.config_path
        if not path.is_file():
            return config
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            log.warning("Could not read %s: %s; using defaults", path, exc)
            return config
        if not isinstance(raw, dict):
            log.warning("%s does not contain a JSON object; using defaults", path)
            return config

        folders = raw.get("watched_folders")
        if isinstance(folders, list):
            config.watched_folders = [
                Path(folder).expanduser()
                for folder in folders
                if isinstance(folder, str) and folder.strip()
            ]
        config.image_extensions = _normalise_extensions(raw.get("image_extensions"), IMAGE_EXTENSIONS)
        config.video_extensions = _normalise_extensions(raw.get("video_extensions"), VIDEO_EXTENSIONS)

        hotkey = raw.get("hotkey")
        if isinstance(hotkey, str) and hotkey.strip():
            config.hotkey = hotkey.strip()

        for name in ("ocr_max_dimension", "max_retries"):
            value = raw.get(name)
            if isinstance(value, int) and value >= 0:
                setattr(config, name, value)
        for name in ("onnx_threads", "vision_batch_size"):
            value = raw.get(name)
            if isinstance(value, int) and value >= 1:
                setattr(config, name, value)
        for name in ("cpu_pause_threshold", "cpu_pause_seconds", "retry_delay_seconds"):
            value = raw.get(name)
            if isinstance(value, (int, float)) and value >= 0:
                setattr(config, name, float(value))
        return config
