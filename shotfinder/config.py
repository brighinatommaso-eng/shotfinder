"""Application paths, watched folders and tunables (dataclass + JSON file)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

APP_NAME = "ShotFinder"
DEFAULT_HOTKEY = "ctrl+alt+space"
# Tried in order when the configured combo is already owned by another app.
HOTKEY_CANDIDATES = (
    DEFAULT_HOTKEY,
    "ctrl+shift+space",
    "ctrl+alt+f",
    "ctrl+shift+f12",
    "alt+shift+s",
)
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".heic"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})

# Windows Known Folder IDs (FOLDERID_*). OneDrive redirection and localised
# folder names are handled by the shell, so never guess "Pictures" by name.
FOLDERID_PICTURES = "{33E28130-4E1E-4676-835A-98395C3BC3BB}"
FOLDERID_SCREENSHOTS = "{B7BEDE81-DF94-4682-A7D8-57A52620B86F}"
FOLDERID_VIDEOS = "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}"
FOLDERID_DOWNLOADS = "{374DE290-123F-4565-9164-39C4925E467B}"
DEFAULT_WATCHED_FOLDER_IDS = (
    FOLDERID_SCREENSHOTS,
    FOLDERID_PICTURES,
    FOLDERID_VIDEOS,
    FOLDERID_DOWNLOADS,
)

if os.name == "nt":  # pragma: no cover - exercised on Windows only
    import ctypes
    import uuid
    from ctypes import wintypes

    class _GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    _shell32 = ctypes.windll.shell32
    _shell32.SHGetKnownFolderPath.argtypes = [
        ctypes.POINTER(_GUID),
        wintypes.DWORD,
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_wchar_p),
    ]
    _shell32.SHGetKnownFolderPath.restype = ctypes.c_long
    _ole32 = ctypes.windll.ole32
    _ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    _ole32.CoTaskMemFree.restype = None
DEFAULT_EXCLUDED_DIR_NAMES = frozenset(
    {
        "Library",
        "PackageCache",
        "node_modules",
        ".git",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        "build",
        "dist",
        "obj",
        "bin",
        ".gradle",
        ".idea",
        ".vs",
        "Temp",
        "$RECYCLE.BIN",
        "AppData",
    }
)


def default_data_dir() -> Path:
    """Return %LOCALAPPDATA%\\ShotFinder (overridable for tests and portable runs)."""
    override = os.environ.get("SHOTFINDER_DATA_DIR")
    if override:
        return Path(override).expanduser()
    local_appdata = os.environ.get("LOCALAPPDATA")
    base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
    return base / APP_NAME


def _existing_unique(candidates: Iterable[Path]) -> list[Path]:
    """Keep the existing folders, in order, without duplicates."""
    folders: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(os.fspath(candidate)))
        if key in seen or not candidate.is_dir():
            continue
        seen.add(key)
        folders.append(candidate)
    return folders


def known_folder(folder_id: str) -> Path | None:
    """Resolve a Windows Known Folder ID through SHGetKnownFolderPath."""
    if os.name != "nt":
        return None
    guid = _GUID.from_buffer_copy(uuid.UUID(folder_id).bytes_le)
    raw = ctypes.c_wchar_p()
    try:
        result = _shell32.SHGetKnownFolderPath(
            ctypes.byref(guid), 0, None, ctypes.byref(raw)
        )
        if result != 0 or not raw.value:
            log.warning(
                "Could not resolve known folder %s (HRESULT 0x%08X)",
                folder_id,
                result & 0xFFFFFFFF,
            )
            return None
        return Path(raw.value)
    finally:
        if raw:
            _ole32.CoTaskMemFree(ctypes.cast(raw, ctypes.c_void_p))


def default_watched_folders() -> list[Path]:
    """Resolve the first-run folders through the Windows Known Folder API."""
    return _existing_unique(
        folder
        for folder in (
            known_folder(folder_id) for folder_id in DEFAULT_WATCHED_FOLDER_IDS
        )
        if folder is not None
    )


def legacy_default_watched_folders() -> list[Path]:
    """The hardcoded English names shipped before the Known Folder API.

    Only used to recognise configs that were saved with those wrong defaults.
    """
    home = Path.home()
    return _existing_unique(
        (
            home / "Pictures" / "Screenshots",
            home / "Pictures",
            home / "Videos",
            home / "Downloads",
        )
    )


def _same_folders(left: Iterable[Path], right: Iterable[Path]) -> bool:
    def keys(folders: Iterable[Path]) -> set[str]:
        return {
            os.path.normcase(os.path.abspath(os.fspath(folder))) for folder in folders
        }

    return keys(left) == keys(right)


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


def _normalise_dir_names(values: Any) -> set[str]:
    """An explicit empty list disables name-based exclusions entirely."""
    if not isinstance(values, (list, tuple, set)):
        return set(DEFAULT_EXCLUDED_DIR_NAMES)
    return {value.strip() for value in values if isinstance(value, str) and value.strip()}


@dataclass(slots=True)
class Config:
    """Runtime configuration; serialised to <data_dir>/config.json."""

    data_dir: Path = field(default_factory=default_data_dir)
    watched_folders: list[Path] = field(default_factory=default_watched_folders)
    image_extensions: set[str] = field(default_factory=lambda: set(IMAGE_EXTENSIONS))
    video_extensions: set[str] = field(default_factory=lambda: set(VIDEO_EXTENSIONS))
    hotkey: str = DEFAULT_HOTKEY
    # True only when the value came from the user, so a failed user choice is
    # reported instead of being silently replaced by a fallback.
    hotkey_is_custom: bool = False
    ocr_max_dimension: int = 1600
    cpu_pause_threshold: float = 70.0
    cpu_pause_seconds: float = 10.0
    retry_delay_seconds: float = 30.0
    max_retries: int = 1
    onnx_threads: int = field(
        default_factory=lambda: max(1, min(4, (os.cpu_count() or 2) // 2))
    )
    vision_batch_size: int = 1
    min_visual_similarity: float = 0.20
    max_visual_results: int = 8
    video_frames_per_turn: int = 30
    # Both stages can be switched off from the settings window; the rest of the
    # index keeps working.
    index_video_frames: bool = True
    index_speech: bool = True
    # Videos longer than this are not transcribed; frames still index them.
    speech_max_minutes: int = 60
    index_cloud_only_files: bool = False
    exclude_dir_names: set[str] = field(
        default_factory=lambda: set(DEFAULT_EXCLUDED_DIR_NAMES)
    )

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
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

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

    def missing_watched_folders(self) -> list[Path]:
        """Configured folders that do not exist (never skipped silently)."""
        return [folder for folder in self.watched_folders if not folder.is_dir()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "watched_folders": [str(folder) for folder in self.watched_folders],
            "image_extensions": sorted(self.image_extensions),
            "video_extensions": sorted(self.video_extensions),
            "hotkey": self.hotkey,
            "hotkey_custom": self.hotkey_is_custom,
            "ocr_max_dimension": self.ocr_max_dimension,
            "cpu_pause_threshold": self.cpu_pause_threshold,
            "cpu_pause_seconds": self.cpu_pause_seconds,
            "retry_delay_seconds": self.retry_delay_seconds,
            "max_retries": self.max_retries,
            "onnx_threads": self.onnx_threads,
            "vision_batch_size": self.vision_batch_size,
            "min_visual_similarity": self.min_visual_similarity,
            "max_visual_results": self.max_visual_results,
            "video_frames_per_turn": self.video_frames_per_turn,
            "index_video_frames": self.index_video_frames,
            "index_speech": self.index_speech,
            "speech_max_minutes": self.speech_max_minutes,
            "index_cloud_only_files": self.index_cloud_only_files,
            "exclude_dir_names": sorted(self.exclude_dir_names),
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
            if config.watched_folders and _same_folders(
                config.watched_folders, legacy_default_watched_folders()
            ):
                migrated = default_watched_folders()
                if migrated and not _same_folders(config.watched_folders, migrated):
                    log.info(
                        "Migrating watched folders from the old English defaults to %s",
                        ", ".join(str(folder) for folder in migrated),
                    )
                    config.watched_folders = migrated
        config.image_extensions = _normalise_extensions(raw.get("image_extensions"), IMAGE_EXTENSIONS)
        config.video_extensions = _normalise_extensions(raw.get("video_extensions"), VIDEO_EXTENSIONS)
        config.exclude_dir_names = _normalise_dir_names(raw.get("exclude_dir_names"))

        hotkey = raw.get("hotkey")
        if isinstance(hotkey, str) and hotkey.strip():
            config.hotkey = hotkey.strip()
        custom = raw.get("hotkey_custom")
        if isinstance(custom, bool):
            config.hotkey_is_custom = custom
        elif config.hotkey.lower() != DEFAULT_HOTKEY:
            # Configs written before hotkey_custom existed: a non-default combo
            # can only have come from the user.
            config.hotkey_is_custom = True

        for name in ("ocr_max_dimension", "max_retries", "speech_max_minutes"):
            value = raw.get(name)
            if isinstance(value, int) and value >= 0:
                setattr(config, name, value)
        for name in (
            "onnx_threads",
            "vision_batch_size",
            "max_visual_results",
            "video_frames_per_turn",
        ):
            value = raw.get(name)
            if isinstance(value, int) and value >= 1:
                setattr(config, name, value)
        for name in ("cpu_pause_threshold", "cpu_pause_seconds", "retry_delay_seconds"):
            value = raw.get(name)
            if isinstance(value, (int, float)) and value >= 0:
                setattr(config, name, float(value))
        similarity = raw.get("min_visual_similarity")
        if isinstance(similarity, (int, float)) and 0.0 <= float(similarity) <= 1.0:
            if abs(float(similarity) - 0.25) < 1e-9:
                # 0.25 was the old default before calibration showed that real
                # hits and noise overlap there; keep the new garbage floor.
                log.info(
                    "Migrating min_visual_similarity from the old 0.25 default to %.2f",
                    config.min_visual_similarity,
                )
            else:
                config.min_visual_similarity = float(similarity)
        for name in ("index_video_frames", "index_speech", "index_cloud_only_files"):
            value = raw.get(name)
            if isinstance(value, bool):
                setattr(config, name, value)
        return config
