"""First-run welcome dialog: folders, privacy and the one-time model download."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..config import (
    DEFAULT_WATCHED_FOLDER_IDS,
    FOLDERID_DOWNLOADS,
    FOLDERID_PICTURES,
    FOLDERID_SCREENSHOTS,
    FOLDERID_VIDEOS,
    Config,
    known_folder,
)

_LABELS = {
    FOLDERID_SCREENSHOTS: "Screenshots",
    FOLDERID_PICTURES: "Pictures",
    FOLDERID_VIDEOS: "Videos",
    FOLDERID_DOWNLOADS: "Downloads",
}


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.path.abspath(os.fspath(left))) == os.path.normcase(
        os.path.abspath(os.fspath(right))
    )


class WelcomeDialog(QDialog):
    """Shown once, before the first scan, when config.json does not exist."""

    def __init__(self, config: Config, parent: object | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._boxes: list[tuple[QCheckBox, Path]] = []
        self.setObjectName("AppDialog")
        self.setWindowTitle("Welcome to ShotFinder")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumWidth(580)
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 22)
        outer.setSpacing(10)

        title = QLabel("Welcome to ShotFinder", self)
        title.setObjectName("DialogTitle")
        outer.addWidget(title)

        body = QLabel(
            "Find any screenshot, photo or video by the text inside it or by "
            "what it shows. ShotFinder runs in the tray and indexes your "
            "folders in the background.",
            self,
        )
        body.setObjectName("DialogBody")
        body.setWordWrap(True)
        outer.addWidget(body)

        privacy = QLabel(
            "Everything stays on this PC. No accounts, no uploads, no telemetry.",
            self,
        )
        privacy.setObjectName("DialogBody")
        privacy.setWordWrap(True)
        outer.addWidget(privacy)

        header = QLabel("FOLDERS TO WATCH", self)
        header.setObjectName("SectionHeader")
        outer.addWidget(header)

        watched = list(self._config.watched_folders)
        for folder_id in DEFAULT_WATCHED_FOLDER_IDS:
            folder = known_folder(folder_id)
            if folder is None or not folder.is_dir():
                continue
            label = _LABELS.get(folder_id, folder.name)
            box = QCheckBox(label, self)
            box.setChecked(any(_same_path(folder, item) for item in watched))
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(10)
            row.addWidget(box)
            path = QLabel(str(folder), self)
            path.setObjectName("DialogNote")
            path.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            row.addWidget(path, 1)
            outer.addLayout(row)
            self._boxes.append((box, folder))

        if not self._boxes:
            note = QLabel(
                "No standard folders were found. You can add folders later in "
                "Settings.",
                self,
            )
            note.setObjectName("DialogNote")
            note.setWordWrap(True)
            outer.addWidget(note)

        download = QLabel(
            "Visual search and speech need about 635 MB of models, downloaded "
            "once. Text and OCR search work immediately while they download.",
            self,
        )
        download.setObjectName("DialogNote")
        download.setWordWrap(True)
        outer.addWidget(download)

        outer.addStretch(1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addStretch(1)
        start = QPushButton("Start", self)
        start.setObjectName("PrimaryButton")
        start.setDefault(True)
        start.clicked.connect(self.accept)
        footer.addWidget(start)
        outer.addLayout(footer)

    def selected_folders(self) -> list[Path]:
        return [folder for box, folder in self._boxes if box.isChecked()]
