"""Hotkey parsing and fallback ordering (registration is Windows-only)."""

from __future__ import annotations

import pytest

from shotfinder.config import HOTKEY_CANDIDATES
from shotfinder.ui.hotkey import (
    MOD_ALT,
    MOD_CONTROL,
    MOD_SHIFT,
    MOD_WIN,
    parse_hotkey,
    pretty_hotkey,
)


def test_default_combo() -> None:
    assert parse_hotkey("ctrl+alt+space") == (MOD_CONTROL | MOD_ALT, 0x20)


def test_case_and_spaces_are_ignored() -> None:
    assert parse_hotkey(" Ctrl + Shift + F9 ") == (MOD_CONTROL | MOD_SHIFT, 0x78)


def test_win_key_alias() -> None:
    assert parse_hotkey("win+alt+a") == (MOD_WIN | MOD_ALT, ord("A"))


def test_digit_key() -> None:
    assert parse_hotkey("ctrl+1") == (MOD_CONTROL, ord("1"))


@pytest.mark.parametrize(
    "value",
    ["", "ctrl+alt", "space", "ctrl+alt+notakey", "ctrl+++alt"],
)
def test_invalid_combos_raise(value: str) -> None:
    with pytest.raises(ValueError):
        parse_hotkey(value)


def test_fallback_order_starts_with_the_documented_default() -> None:
    assert HOTKEY_CANDIDATES[0] == "ctrl+alt+space"
    assert HOTKEY_CANDIDATES == (
        "ctrl+alt+space",
        "ctrl+shift+space",
        "ctrl+alt+f",
        "ctrl+shift+f12",
        "alt+shift+s",
    )


def test_every_fallback_candidate_parses() -> None:
    for combo in HOTKEY_CANDIDATES:
        modifiers, key = parse_hotkey(combo)
        assert modifiers and key


def test_pretty_hotkey_for_menus() -> None:
    assert pretty_hotkey("ctrl+alt+space") == "Ctrl+Alt+Space"
    assert pretty_hotkey("ctrl+shift+f12") == "Ctrl+Shift+F12"
