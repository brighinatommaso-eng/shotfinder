"""Snippet and date formatting for the result tiles."""

from __future__ import annotations

from datetime import datetime

from shotfinder.ui.formatting import clean_snippet, folder_label, format_date


def test_fts_match_markers_are_removed() -> None:
    assert clean_snippet("6 Legs [Spider] [Robot] Kit —") == "6 Legs Spider Robot Kit —"


def test_elision_and_whitespace_are_collapsed() -> None:
    assert clean_snippet("…line one\n\n  line   two…") == "line one line two"


def test_long_snippets_are_capped() -> None:
    result = clean_snippet("x" * 300, limit=20)

    assert len(result) == 20
    assert result.endswith("…")


def test_missing_snippet_is_empty() -> None:
    assert clean_snippet(None) == ""
    assert clean_snippet("") == ""


def test_date_is_formatted_or_blank() -> None:
    moment = datetime(2026, 3, 9, 14, 5).timestamp()

    assert format_date(moment) == "09 Mar 2026 · 14:05"
    assert format_date(None) == ""
    assert format_date(float("nan")) == ""


def test_folder_label_falls_back_to_the_full_parent() -> None:
    assert folder_label(r"C:\Users\tomma\Downloads\Hexapod\shot.jpg") == "Hexapod"
