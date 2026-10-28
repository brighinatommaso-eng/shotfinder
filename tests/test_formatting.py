"""Snippet and date formatting for the result tiles."""

from __future__ import annotations

from datetime import datetime

from shotfinder.ui.formatting import (
    clean_snippet,
    folder_label,
    format_date,
    format_timestamp,
    highlight_snippet,
)


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


def test_video_timestamps_use_minutes_and_hours() -> None:
    assert format_timestamp(134) == "02:14"
    assert format_timestamp(3725) == "1:02:05"
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(-4) == "00:00"
    assert format_timestamp(None) == ""


def test_highlight_wraps_matches_and_escapes_text() -> None:
    rendered = highlight_snippet("Solder the [battery] <connector>")

    assert "<span" in rendered
    assert "battery" in rendered
    assert "&lt;connector&gt;" in rendered
    assert "[" not in rendered and "]" not in rendered


def test_highlight_closes_an_open_span_when_truncated() -> None:
    rendered = highlight_snippet("[battery connector] and more", limit=8)

    assert rendered.count("<span") == rendered.count("</span>")
    assert rendered.endswith("…")


def test_highlight_of_an_empty_snippet_is_empty() -> None:
    assert highlight_snippet(None) == ""
    assert highlight_snippet("") == ""
