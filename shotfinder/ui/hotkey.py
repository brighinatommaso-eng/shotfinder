"""Global hotkey through Win32 ``RegisterHotKey`` and a native event filter.

No keyboard hook and no extra thread: Windows posts ``WM_HOTKEY`` to the GUI
thread's message queue, and Qt delivers it to this filter before the event
loop sees it. Registration is best-effort; when another app owns the combo the
caller can fall back to the tray menu.
"""

from __future__ import annotations

import ctypes
import logging
import os
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter, QCoreApplication, QObject, Signal

log = logging.getLogger(__name__)

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312

HOTKEY_ID = 0x5346  # "SF"

_MODIFIERS = {
    "ctrl": MOD_CONTROL,
    "control": MOD_CONTROL,
    "alt": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN,
    "windows": MOD_WIN,
    "super": MOD_WIN,
    "meta": MOD_WIN,
}

_KEYS = {
    "space": 0x20,
    "enter": 0x0D,
    "return": 0x0D,
    "tab": 0x09,
    "esc": 0x1B,
    "escape": 0x1B,
    "backspace": 0x08,
    "delete": 0x2E,
    "del": 0x2E,
    "insert": 0x2D,
    "ins": 0x2D,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pgup": 0x21,
    "pagedown": 0x22,
    "pgdn": 0x22,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
}
_KEYS.update({f"f{number}": 0x6F + number for number in range(1, 25)})


def parse_hotkey(value: str) -> tuple[int, int]:
    """Parse ``ctrl+alt+space`` into ``(modifier_mask, virtual_key)``."""
    modifiers = 0
    key: int | None = None
    for raw in value.split("+"):
        part = raw.strip().lower()
        if not part:
            continue
        if part in _MODIFIERS:
            modifiers |= _MODIFIERS[part]
        elif part in _KEYS:
            key = _KEYS[part]
        elif len(part) == 1 and part.isalnum():
            key = ord(part.upper())
        else:
            raise ValueError(f"Unknown key or modifier: {raw.strip()!r}")
    if key is None:
        raise ValueError(f"No key in hotkey {value!r}")
    if modifiers == 0:
        raise ValueError(f"Hotkey {value!r} needs at least one modifier")
    return modifiers, key


class GlobalHotkey(QObject, QAbstractNativeEventFilter):
    """A single global hotkey that emits :attr:`activated`."""

    activated = Signal()

    def __init__(self, spec: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._spec = spec
        self._registered = False
        self._modifiers, self._virtual_key = parse_hotkey(spec)
        self._application: QCoreApplication | None = None
        self._user32 = None

    @property
    def spec(self) -> str:
        return self._spec

    @property
    def registered(self) -> bool:
        return self._registered

    def register(self) -> bool:
        """Register the hotkey; ``False`` when the OS refuses (combo taken)."""
        if os.name != "nt":
            log.info("Global hotkeys are only supported on Windows")
            return False
        if self._registered:
            return True
        user32 = ctypes.windll.user32
        user32.RegisterHotKey.argtypes = [
            wintypes.HWND,
            ctypes.c_int,
            wintypes.UINT,
            wintypes.UINT,
        ]
        user32.RegisterHotKey.restype = wintypes.BOOL
        user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.UnregisterHotKey.restype = wintypes.BOOL
        if not user32.RegisterHotKey(
            None, HOTKEY_ID, self._modifiers | MOD_NOREPEAT, self._virtual_key
        ):
            log.warning("Could not register the global hotkey %s", self._spec)
            return False
        application = QCoreApplication.instance()
        if application is None:
            user32.UnregisterHotKey(None, HOTKEY_ID)
            log.warning("No Qt application; cannot install the hotkey filter")
            return False
        self._user32 = user32
        self._application = application
        application.installNativeEventFilter(self)
        self._registered = True
        return True

    def unregister(self) -> None:
        if not self._registered:
            return
        application = self._application
        if application is not None:
            application.removeNativeEventFilter(self)
        if self._user32 is not None:
            self._user32.UnregisterHotKey(None, HOTKEY_ID)
        self._registered = False
        self._application = None
        self._user32 = None

    def nativeEventFilter(self, event_type, message):  # noqa: N802 - Qt override
        if not self._registered:
            return False, 0
        name = (
            bytes(event_type.data())
            if hasattr(event_type, "data")
            else bytes(event_type)
        )
        if name not in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
            return False, 0
        try:
            msg = wintypes.MSG.from_address(int(message))
        except (TypeError, ValueError):
            return False, 0
        if msg.message == WM_HOTKEY and int(msg.wParam) == HOTKEY_ID:
            self.activated.emit()
            return True, 0
        return False, 0
