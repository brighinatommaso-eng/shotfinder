"""Start-with-Windows support through the per-user Run key (no admin needed)."""

from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger(__name__)

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "ShotFinder"


def command() -> str:
    """The command Windows should run at sign-in."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    return f'"{sys.executable}" -m shotfinder'


def is_enabled() -> bool:
    if os.name != "nt":
        return False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            value, _kind = winreg.QueryValueEx(key, VALUE_NAME)
    except OSError:
        return False
    return bool(value)


def set_enabled(enabled: bool) -> bool:
    """Add or remove the Run entry; returns whether the change was written."""
    if os.name != "nt":
        return False
    import winreg

    try:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            if enabled:
                winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command())
            else:
                try:
                    winreg.DeleteValue(key, VALUE_NAME)
                except FileNotFoundError:
                    pass
    except OSError:
        log.warning("Could not update the Windows startup entry", exc_info=True)
        return False
    log.info("Start with Windows %s", "enabled" if enabled else "disabled")
    return True
