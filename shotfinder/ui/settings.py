"""Settings window: folders, indexing stages, hotkey and index maintenance."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .. import startup
from .hotkey import pretty_hotkey

log = logging.getLogger(__name__)

_MODIFIER_KEYS = {
    Qt.Key.Key_Control,
    Qt.Key.Key_Shift,
    Qt.Key.Key_Alt,
    Qt.Key.Key_Meta,
}


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.path.abspath(os.fspath(left))) == os.path.normcase(
        os.path.abspath(os.fspath(right))
    )


def _format_size(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


class HotkeyRecorder(QLineEdit):
    """Read-only field that captures one modifier combo at a time."""

    captured = Signal(str)
    editing_changed = Signal(bool)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText("Press a shortcut")
        self.setProperty("recording", "false")
        self.setToolTip("Click here, then press a combination such as Ctrl+Alt+Space")

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.key() in _MODIFIER_KEYS:
            return
        parts: list[str] = []
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            parts.append("ctrl")
        if modifiers & Qt.KeyboardModifier.AltModifier:
            parts.append("alt")
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            parts.append("shift")
        if modifiers & Qt.KeyboardModifier.MetaModifier:
            parts.append("win")
        if not parts:
            self.setText("Add Ctrl, Alt or Shift")
            return
        name = (
            QKeySequence(event.key())
            .toString(QKeySequence.SequenceFormat.PortableText)
            .lower()
        )
        if not name:
            return
        spec = "+".join(parts + [name])
        self.setText(pretty_hotkey(spec))
        self.captured.emit(spec)

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._set_recording(True)
        self.editing_changed.emit(True)
        super().focusInEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._set_recording(False)
        self.editing_changed.emit(False)
        super().focusOutEvent(event)

    def _set_recording(self, recording: bool) -> None:
        self.setProperty("recording", "true" if recording else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        if recording:
            self.selectAll()


class SettingsDialog(QDialog):
    """The tray's "Settings…" window; every change applies immediately."""

    def __init__(self, controller: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._config = controller.config
        self.setObjectName("AppDialog")
        self.setWindowTitle("ShotFinder settings")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumWidth(640)
        self._build()
        self.refresh()

    # -- construction --------------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 22, 24, 20)
        outer.setSpacing(10)

        title = QLabel("Settings", self)
        title.setObjectName("DialogTitle")
        outer.addWidget(title)

        subtitle = QLabel(
            "ShotFinder runs in the tray and keeps the index on this PC.",
            self,
        )
        subtitle.setObjectName("DialogBody")
        outer.addWidget(subtitle)

        folders_header = QLabel("WATCHED FOLDERS", self)
        folders_header.setObjectName("SectionHeader")
        outer.addWidget(folders_header)

        self._folders = QListWidget(self)
        self._folders.setMinimumHeight(110)
        self._folders.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._folders.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self._folders.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        outer.addWidget(self._folders)

        folder_buttons = QHBoxLayout()
        folder_buttons.setContentsMargins(0, 0, 0, 0)
        add = QPushButton("Add folder…", self)
        add.clicked.connect(self._on_add_folder)
        remove = QPushButton("Remove", self)
        remove.clicked.connect(self._on_remove_folder)
        folder_buttons.addWidget(add)
        folder_buttons.addWidget(remove)
        folder_buttons.addStretch(1)
        outer.addLayout(folder_buttons)

        indexing_header = QLabel("INDEXING", self)
        indexing_header.setObjectName("SectionHeader")
        outer.addWidget(indexing_header)

        self._frames = QCheckBox(
            "Index video frames (visual search inside videos)", self
        )
        self._frames.toggled.connect(self._on_frames_toggled)
        outer.addWidget(self._frames)

        self._speech = QCheckBox("Transcribe speech in videos", self)
        self._speech.toggled.connect(self._on_speech_toggled)
        outer.addWidget(self._speech)

        minutes_row = QHBoxLayout()
        minutes_row.setContentsMargins(0, 0, 0, 0)
        minutes_row.setSpacing(8)
        minutes_label = QLabel("Skip speech on videos longer than", self)
        minutes_label.setObjectName("DialogBody")
        self._minutes = QSpinBox(self)
        self._minutes.setRange(0, 600)
        self._minutes.setSuffix(" min")
        self._minutes.setSpecialValueText("No limit")
        self._minutes.valueChanged.connect(self._on_minutes_changed)
        minutes_row.addWidget(minutes_label)
        minutes_row.addWidget(self._minutes)
        minutes_row.addStretch(1)
        outer.addLayout(minutes_row)

        hotkey_header = QLabel("SEARCH HOTKEY", self)
        hotkey_header.setObjectName("SectionHeader")
        outer.addWidget(hotkey_header)

        hotkey_row = QHBoxLayout()
        hotkey_row.setContentsMargins(0, 0, 0, 0)
        hotkey_row.setSpacing(10)
        self._recorder = HotkeyRecorder(self)
        self._recorder.setFixedWidth(230)
        self._recorder.captured.connect(self._on_hotkey_captured)
        self._recorder.editing_changed.connect(self._on_hotkey_editing)
        hotkey_row.addWidget(self._recorder)
        self._hotkey_status = QLabel(self)
        self._hotkey_status.setObjectName("DialogStatus")
        hotkey_row.addWidget(self._hotkey_status, 1)
        outer.addLayout(hotkey_row)

        hotkey_note = QLabel(
            "Click the field and press a combination. Windows checks that it is "
            "free before it is saved.",
            self,
        )
        hotkey_note.setObjectName("DialogNote")
        hotkey_note.setWordWrap(True)
        outer.addWidget(hotkey_note)

        self._startup = QCheckBox("Start ShotFinder with Windows", self)
        self._startup.toggled.connect(self._on_startup_toggled)
        outer.addWidget(self._startup)

        stats_header = QLabel("INDEX", self)
        stats_header.setObjectName("SectionHeader")
        outer.addWidget(stats_header)

        stats = QGridLayout()
        stats.setContentsMargins(0, 0, 0, 0)
        stats.setHorizontalSpacing(24)
        self._stat_values: dict[str, QLabel] = {}
        for column, (key, label_text) in enumerate(
            (
                ("files", "Files"),
                ("videos", "Videos"),
                ("vectors", "Frames"),
                ("db_bytes", "Index size"),
            )
        ):
            value = QLabel("–", self)
            value.setObjectName("StatValue")
            label = QLabel(label_text, self)
            label.setObjectName("StatLabel")
            stats.addWidget(value, 0, column)
            stats.addWidget(label, 1, column)
            self._stat_values[key] = value
        outer.addLayout(stats)

        rebuild_row = QHBoxLayout()
        rebuild_row.setContentsMargins(0, 0, 0, 0)
        rebuild_row.setSpacing(10)
        rebuild = QPushButton("Rebuild index", self)
        rebuild.clicked.connect(self._on_rebuild)
        rebuild_row.addWidget(rebuild)
        rebuild_note = QLabel(
            "Deletes the index and scans the watched folders again.", self
        )
        rebuild_note.setObjectName("DialogNote")
        rebuild_row.addWidget(rebuild_note, 1)
        outer.addLayout(rebuild_row)

        outer.addStretch(1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        self._status = QLabel("", self)
        self._status.setObjectName("DialogStatus")
        footer.addWidget(self._status, 1)
        close = QPushButton("Close", self)
        close.setObjectName("PrimaryButton")
        close.clicked.connect(self.close)
        footer.addWidget(close)
        outer.addLayout(footer)

    # -- refresh -------------------------------------------------------------

    def refresh(self) -> None:
        self._refresh_folders()
        self._refresh_controls()
        self._refresh_stats()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.refresh()
        super().showEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._controller.end_hotkey_capture()
        super().closeEvent(event)

    def _refresh_folders(self) -> None:
        self._folders.clear()
        for folder in self._config.watched_folders:
            notes: list[str] = []
            if not Path(folder).is_dir():
                notes.append("(missing)")
            cloud = self._controller.index.cloud_only_count_under(folder)
            if cloud:
                notes.append(f"{cloud} cloud-only")
            text = str(folder)
            if notes:
                text += "  ·  " + " · ".join(notes)
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, str(folder))
            item.setToolTip(str(folder))
            self._folders.addItem(item)

    def _refresh_controls(self) -> None:
        for widget in (self._frames, self._speech, self._minutes, self._startup):
            widget.blockSignals(True)
        self._frames.setChecked(bool(self._config.index_video_frames))
        self._speech.setChecked(bool(self._config.index_speech))
        self._minutes.setValue(int(self._config.speech_max_minutes))
        self._startup.setChecked(startup.is_enabled())
        for widget in (self._frames, self._speech, self._minutes, self._startup):
            widget.blockSignals(False)
        self._recorder.setText(pretty_hotkey(self._config.hotkey))

    def _refresh_stats(self) -> None:
        stats = self._controller.index_stats()
        self._stat_values["files"].setText(f"{stats['files']:,}")
        self._stat_values["videos"].setText(f"{stats['videos']:,}")
        self._stat_values["vectors"].setText(f"{stats['vectors']:,}")
        self._stat_values["db_bytes"].setText(_format_size(stats["db_bytes"]))

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    # -- folders -------------------------------------------------------------

    def _on_add_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Add a folder to watch", str(Path.home())
        )
        if not chosen:
            return
        folder = Path(chosen)
        if any(_same_path(folder, watched) for watched in self._config.watched_folders):
            self._set_status("That folder is already watched.")
            return
        self._controller.add_watched_folder(folder)
        self._refresh_folders()
        self._set_status(f"Watching {folder}")

    def _on_remove_folder(self) -> None:
        item = self._folders.currentItem()
        if item is None:
            self._set_status("Select a folder first.")
            return
        folder = Path(str(item.data(Qt.ItemDataRole.UserRole)))
        answer = QMessageBox.question(
            self,
            "Remove folder",
            f"Stop watching {folder}?\n\n"
            "Its files will be removed from the index.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._controller.remove_watched_folder(folder)
        self._refresh_folders()
        self._refresh_stats()
        self._set_status(f"Removed {folder}")

    # -- indexing ------------------------------------------------------------

    def _on_frames_toggled(self, enabled: bool) -> None:
        self._controller.set_video_frames(enabled)
        self._set_status(
            "Video frames on; new videos will be indexed."
            if enabled
            else "Video frames off; existing video results stay searchable."
        )

    def _on_speech_toggled(self, enabled: bool) -> None:
        self._controller.set_speech(enabled)
        self._set_status(
            "Speech on; videos will be transcribed in the background."
            if enabled
            else "Speech off; existing transcripts stay searchable."
        )

    def _on_minutes_changed(self, value: int) -> None:
        self._controller.set_speech_max_minutes(value)
        self._set_status(
            "No speech limit." if value == 0 else f"Speech limit: {value} min."
        )

    # -- hotkey --------------------------------------------------------------

    def _on_hotkey_editing(self, editing: bool) -> None:
        if editing:
            self._controller.begin_hotkey_capture()
            self._hotkey_status.setText("Press the combination you want…")
            return
        self._controller.end_hotkey_capture()
        self._recorder.setText(pretty_hotkey(self._config.hotkey))

    def _on_hotkey_captured(self, spec: str) -> None:
        error = self._controller.apply_hotkey(spec)
        if error:
            self._hotkey_status.setText(error)
            self._recorder.setText(pretty_hotkey(self._config.hotkey))
            return
        self._hotkey_status.setText(f"Saved: {pretty_hotkey(spec)}")

    # -- Windows startup -----------------------------------------------------

    def _on_startup_toggled(self, enabled: bool) -> None:
        if startup.set_enabled(enabled):
            self._set_status(
                "ShotFinder will start with Windows."
                if enabled
                else "ShotFinder will not start with Windows."
            )
            return
        self._startup.blockSignals(True)
        self._startup.setChecked(startup.is_enabled())
        self._startup.blockSignals(False)
        self._set_status("Could not update the Windows startup entry.")

    # -- maintenance ---------------------------------------------------------

    def _on_rebuild(self) -> None:
        answer = QMessageBox.question(
            self,
            "Rebuild index",
            "This deletes the whole index and scans the watched folders again.\n\n"
            "It can take a while for a large library. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._controller.rebuild_index()
        self._refresh_stats()
        self._set_status("Rebuilding the index…")
        QTimer.singleShot(1500, self._refresh_stats)
