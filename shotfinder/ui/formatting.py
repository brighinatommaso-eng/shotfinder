"""Text helpers shared by the result tiles and the preview window."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

_HIGHLIGHT_OPEN = '<span style="background-color: rgba(76, 194, 255, 0.30);">'


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


def highlight_snippet(snippet: str | None, limit: int = 160) -> str:
    """Render an FTS snippet as rich text with the matched words highlighted.

    ``snippet()`` marks each match with ``[...]``; those markers become spans.
    The result is HTML-escaped, and a truncated snippet closes any open span.
    """
    if not snippet:
        return ""
    text = " ".join(snippet.replace("\u2026", " ").split())
    parts: list[str] = []
    visible = 0
    open_span = False
    truncated = False
    for character in text:
        if character == "[":
            if visible >= limit:
                truncated = True
                break
            parts.append(_HIGHLIGHT_OPEN)
            open_span = True
        elif character == "]":
            if open_span:
                parts.append("</span>")
                open_span = False
        else:
            if visible >= limit:
                truncated = True
                break
            parts.append(html.escape(character))
            visible += 1
    if open_span:
        parts.append("</span>")
    if truncated:
        parts.append("…")
    return "".join(parts)


def folder_label(path: str) -> str:
    parent = Path(path).parent
    return parent.name or str(parent)
