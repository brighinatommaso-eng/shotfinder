"""Text helpers shared by the result tiles and the preview window."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def format_date(timestamp: float | None) -> str:
    if timestamp is None:
        return ""
    try:
        moment = datetime.fromtimestamp(float(timestamp))
    except (OverflowError, OSError, ValueError):
        return ""
    return moment.strftime("%d %b %Y · %H:%M")


def format_timestamp(seconds: float | None) -> str:
    """``134`` -> ``02:14`` (hours appear only when needed)."""
    if seconds is None:
        return ""
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def clean_snippet(snippet: str | None, limit: int = 160) -> str:
    """Turn an FTS snippet into display text.

    ``snippet()`` wraps every matched term in ``[...]`` and elides the rest with
    ``…``; neither is useful on a tile, so both are dropped.
    """
    if not snippet:
        return ""
    text = snippet.replace("[", "").replace("]", "").replace("\u2026", " ")
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def folder_label(path: str) -> str:
    parent = Path(path).parent
    return parent.name or str(parent)
