"""Windows 11 style theme: palette, stylesheet and colour-scheme tracking.

The window is frameless with a translucent rounded surface instead of a real
DWM Mica backdrop, so it keeps its shape over any desktop and works on Windows
10 as well. Everything else follows the Fluent metrics: Segoe UI Variable,
12 px corners, 8 px radii on controls and a single accent colour.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication


@dataclass(frozen=True, slots=True)
class Palette:
    dark: bool
    window_top: str
    window_bottom: str
    solid_window: str
    surface_hover: str
    surface_selected: str
    border: str
    border_strong: str
    text: str
    muted: str
    faint: str
    accent: str
    accent_soft: str
    badge_text: str
    badge_text_bg: str
    badge_visual: str
    badge_visual_bg: str
    badge_both: str
    badge_both_bg: str
    badge_recent: str
    badge_recent_bg: str
    badge_spoken: str
    badge_spoken_bg: str
    scroll: str
    scroll_hover: str
    shadow: QColor


DARK = Palette(
    dark=True,
    window_top="rgba(38, 38, 38, 250)",
    window_bottom="rgba(28, 28, 28, 250)",
    solid_window="#242424",
    surface_hover="rgba(255, 255, 255, 20)",
    surface_selected="rgba(76, 194, 255, 34)",
    border="rgba(255, 255, 255, 24)",
    border_strong="rgba(76, 194, 255, 140)",
    text="#F5F5F5",
    muted="#ABABAB",
    faint="#858585",
    accent="#4CC2FF",
    accent_soft="rgba(76, 194, 255, 38)",
    badge_text="#9CDCFE",
    badge_text_bg="rgba(156, 220, 254, 30)",
    badge_visual="#C8A6FF",
    badge_visual_bg="rgba(200, 166, 255, 30)",
    badge_both="#7FE3C0",
    badge_both_bg="rgba(127, 227, 192, 30)",
    badge_recent="#B9B9B9",
    badge_recent_bg="rgba(255, 255, 255, 20)",
    badge_spoken="#FFD27F",
    badge_spoken_bg="rgba(255, 210, 127, 28)",
    scroll="rgba(255, 255, 255, 56)",
    scroll_hover="rgba(255, 255, 255, 96)",
    shadow=QColor(0, 0, 0, 110),
)

LIGHT = Palette(
    dark=False,
    window_top="rgba(250, 250, 250, 252)",
    window_bottom="rgba(242, 242, 242, 252)",
    solid_window="#F7F7F7",
    surface_hover="rgba(0, 0, 0, 12)",
    surface_selected="rgba(0, 103, 192, 26)",
    border="rgba(0, 0, 0, 22)",
    border_strong="rgba(0, 103, 192, 130)",
    text="#1A1A1A",
    muted="#5C5C5C",
    faint="#8A8A8A",
    accent="#0067C0",
    accent_soft="rgba(0, 103, 192, 24)",
    badge_text="#005A9E",
    badge_text_bg="rgba(0, 103, 192, 18)",
    badge_visual="#6B3FA0",
    badge_visual_bg="rgba(107, 63, 160, 16)",
    badge_both="#0B6B4F",
    badge_both_bg="rgba(11, 107, 79, 16)",
    badge_recent="#5C5C5C",
    badge_recent_bg="rgba(0, 0, 0, 14)",
    badge_spoken="#7A5200",
    badge_spoken_bg="rgba(122, 82, 0, 16)",
    scroll="rgba(0, 0, 0, 44)",
    scroll_hover="rgba(0, 0, 0, 80)",
    shadow=QColor(0, 0, 0, 60),
)

_STYLESHEET = """
* {
    font-family: "Segoe UI Variable Text", "Segoe UI", sans-serif;
    outline: none;
}
#Root {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 %(window_top)s, stop: 1 %(window_bottom)s);
    border: 1px solid %(border)s;
    border-radius: 14px;
}
#SearchRow { background: transparent; }
#SearchInput {
    background: transparent;
    border: none;
    color: %(text)s;
    font-size: 20px;
    padding: 0;
    selection-background-color: %(accent_soft)s;
    selection-color: %(text)s;
}
#Hint { color: %(faint)s; font-size: 10px; letter-spacing: 1px; }
#Separator { background: %(border)s; border: none; min-height: 1px; max-height: 1px; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px 2px 2px 0; }
QScrollBar::handle:vertical { background: %(scroll)s; border-radius: 3px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: %(scroll_hover)s; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
ResultTile { background: transparent; border: 1px solid transparent; border-radius: 10px; }
ResultTile:hover { background: %(surface_hover)s; }
ResultTile[selected="true"] {
    background: %(surface_selected)s;
    border: 1px solid %(border_strong)s;
}
#TileName { color: %(text)s; font-size: 12px; font-weight: 600; }
#TileMeta { color: %(muted)s; font-size: 10px; }
#TileSnippet { color: %(faint)s; font-size: 10px; }
#SectionHeader {
    color: %(faint)s;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 10px 2px 0 2px;
}
#Timestamp {
    background: rgba(0, 0, 0, 150);
    color: #FFFFFF;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 9px;
    font-weight: 600;
}
#Badge {
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 8px;
    font-weight: 700;
    letter-spacing: 1px;
}
#Badge[kind="text"] { color: %(badge_text)s; background: %(badge_text_bg)s; }
#Badge[kind="visual"] { color: %(badge_visual)s; background: %(badge_visual_bg)s; }
#Badge[kind="both"] { color: %(badge_both)s; background: %(badge_both_bg)s; }
#Badge[kind="recent"] { color: %(badge_recent)s; background: %(badge_recent_bg)s; }
#Badge[kind="spoken"] { color: %(badge_spoken)s; background: %(badge_spoken_bg)s; }
#Status { color: %(faint)s; font-size: 10px; }
#EmptyTitle { color: %(text)s; font-size: 17px; font-weight: 600; }
#EmptyBody { color: %(muted)s; font-size: 12px; }
#PreviewName { color: %(text)s; font-size: 14px; font-weight: 600; }
#PreviewPath { color: %(muted)s; font-size: 11px; }
#PreviewDate { color: %(faint)s; font-size: 11px; }
#PreviewImage { color: %(muted)s; font-size: 12px; }
#CloseButton {
    background: transparent;
    border: none;
    color: %(muted)s;
    font-size: 15px;
    border-radius: 6px;
    padding: 3px 8px;
}
#CloseButton:hover { background: %(surface_hover)s; color: %(text)s; }
#OpenExternal {
    background: %(accent_soft)s;
    color: %(accent)s;
    border: 1px solid %(border_strong)s;
    border-radius: 6px;
    padding: 7px 16px;
    font-size: 12px;
    font-weight: 600;
}
#OpenExternal:hover { background: %(surface_selected)s; }
QMenu {
    background: %(solid_window)s;
    color: %(text)s;
    border: 1px solid %(border)s;
    border-radius: 8px;
    padding: 4px;
}
QMenu::item { padding: 6px 28px 6px 12px; border-radius: 5px; }
QMenu::item:selected { background: %(surface_hover)s; }
QMenu::item:disabled { color: %(faint)s; }
QMenu::separator { height: 1px; background: %(border)s; margin: 4px 8px; }
QToolTip {
    background: %(solid_window)s;
    color: %(text)s;
    border: 1px solid %(border)s;
    padding: 4px 6px;
}
"""


def palette_for(dark: bool) -> Palette:
    return DARK if dark else LIGHT


def is_dark(app: QGuiApplication | None = None) -> bool:
    """Follow the Windows app theme; unknown falls back to light."""
    application = app or QGuiApplication.instance()
    if application is None:
        return False
    hints = application.styleHints()
    scheme = hints.colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return True
    if scheme == Qt.ColorScheme.Light:
        return False
    return False


def stylesheet(palette: Palette) -> str:
    return _STYLESHEET % {
        field.name: getattr(palette, field.name) for field in fields(palette)
    }


def install(app: QGuiApplication) -> Palette:
    """Apply the theme now and keep following Windows when it changes."""
    palette = palette_for(is_dark(app))
    app.setStyleSheet(stylesheet(palette))

    def follow_scheme(_scheme: Qt.ColorScheme) -> None:
        app.setStyleSheet(stylesheet(palette_for(is_dark(app))))

    app.styleHints().colorSchemeChanged.connect(follow_scheme)
    return palette
