"""Hybrid text + visual search with reciprocal rank fusion (Phase 2).

Text and visual evidence are not equally trustworthy. OCR hits are literal,
while CLIP cosine similarity on the shipped model overlaps between noise and
real hits (see ``docs/threshold-calibration.md``). The two are therefore never
merged into one ranked list: :class:`SearchResults.matches` holds text and
both-hits ordered by RRF, and :class:`SearchResults.related` holds visual-only
hits ordered by cosine, capped and presented as suggestions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import models
from .config import Config
from .index import Index

log = logging.getLogger(__name__)

RRF_K = 60
VECTOR_K = 50
DEFAULT_LIMIT = 40
DEFAULT_MAX_RELATED = 8
RELATIVE_SIMILARITY_MARGIN = 0.05

MATCH_TEXT = "text"
MATCH_VISUAL = "visual"
MATCH_BOTH = "both"
MATCH_RECENT = "recent"


@dataclass(slots=True)
class SearchResults:
    """Two ranked groups: confident matches, then approximate suggestions."""

    matches: list[dict[str, Any]] = field(default_factory=list)
    related: list[dict[str, Any]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.matches or self.related)

    def all(self) -> list[dict[str, Any]]:
        """Matches first, then related, in display order."""
        return self.matches + self.related


def distance_to_cosine(distance: float) -> float:
    """Convert sqlite-vec L2 distance to cosine (stored vectors are unit length)."""
    return 1.0 - (distance * distance) / 2.0


def filter_vector_rows(
    rows: Sequence[dict[str, Any]],
    min_similarity: float,
    *,
    margin: float = RELATIVE_SIMILARITY_MARGIN,
) -> list[dict[str, Any]]:
    """Drop weak vector hits before they are ranked.

    The absolute floor is a garbage floor, not a relevance classifier: it only
    removes vectors that are essentially unrelated to the query. The relative
    margin then removes a long tail that is only slightly weaker than the best
    hit, so one strong match is not diluted by many mediocre ones.
    """
    if not rows:
        return []
    cosines = [distance_to_cosine(float(row["distance"])) for row in rows]
    cutoff = max(float(min_similarity), max(cosines) - margin)
    return [row for row, cosine in zip(rows, cosines) if cosine >= cutoff]


def _first_per_file(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the best-ranked row per file (input is already ordered)."""
    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        file_id = int(row["file_id"])
        if file_id in seen:
            continue
        seen.add(file_id)
        unique.append(row)
    return unique


def _finalise(entry: dict[str, Any]) -> dict[str, Any]:
    result = dict(entry)
    text = bool(result.pop("_text"))
    visual = bool(result.pop("_visual"))
    result.pop("_best_rank", None)
    result.pop("_timed_rank", None)
    if text and visual:
        result["match_type"] = MATCH_BOTH
    elif text:
        result["match_type"] = MATCH_TEXT
    else:
        result["match_type"] = MATCH_VISUAL
    return result


def combine(
    text_rows: Sequence[dict[str, Any]],
    vector_rows: Sequence[dict[str, Any]],
    *,
    k: int = RRF_K,
    limit: int = DEFAULT_LIMIT,
    max_related: int = DEFAULT_MAX_RELATED,
) -> SearchResults:
    """Split fused results into confident matches and visual suggestions.

    Each list contributes ``1 / (k + rank)`` per unique file to the RRF score
    used to order ``matches``. Rows are deduplicated first, so ten matching OCR
    lines do not outvote ten files. ``related`` is ordered by cosine instead,
    because RRF ranks would present an approximate hit as if it were certain.
    The timestamp of the best hit is kept as the video jump point.
    """
    groups: dict[int, dict[str, Any]] = {}

    def group(file_id: int) -> dict[str, Any]:
        return groups.setdefault(
            file_id,
            {
                "file_id": file_id,
                "path": "",
                "kind": "",
                "hash": "",
                "mtime": None,
                "ts": None,
                "snippet": "",
                "source": None,
                "score": 0.0,
                "match_type": MATCH_TEXT,
                "fts_rank": None,
                "cosine": None,
                "_text": False,
                "_visual": False,
                "_best_rank": None,
                "_timed_rank": None,
            },
        )

    for rank, row in enumerate(_first_per_file(text_rows), start=1):
        entry = group(int(row["file_id"]))
        entry["path"] = row["path"]
        entry["kind"] = row["kind"]
        entry["hash"] = row.get("hash") or entry["hash"]
        entry["mtime"] = row.get("mtime", entry["mtime"])
        entry["score"] += 1.0 / (k + rank)
        entry["_text"] = True
        best = entry["fts_rank"] is None or rank < entry["fts_rank"]
        if best:
            entry["fts_rank"] = rank
        if row.get("ts") is not None and (
            entry["_timed_rank"] is None or rank < entry["_timed_rank"]
        ):
            # A spoken line is a better jump point and snippet than the
            # filename row, even when the name matches more strongly.
            entry["_timed_rank"] = rank
            entry["ts"] = row.get("ts")
            entry["snippet"] = row.get("snippet") or ""
            entry["source"] = row.get("source")
        elif best and entry["_timed_rank"] is None:
            entry["snippet"] = row.get("snippet") or ""
            entry["source"] = row.get("source")
        if entry["_best_rank"] is None or rank < entry["_best_rank"]:
            entry["_best_rank"] = rank

    for rank, row in enumerate(_first_per_file(vector_rows), start=1):
        entry = group(int(row["file_id"]))
        entry["path"] = row["path"]
        entry["kind"] = row["kind"]
        entry["hash"] = row.get("hash") or entry["hash"]
        entry["mtime"] = row.get("mtime", entry["mtime"])
        entry["score"] += 1.0 / (k + rank)
        entry["_visual"] = True
        cosine = distance_to_cosine(float(row["distance"]))
        if entry["cosine"] is None or cosine > entry["cosine"]:
            entry["cosine"] = cosine
        if entry["_timed_rank"] is None and (
            entry["_best_rank"] is None or rank < entry["_best_rank"]
        ):
            entry["ts"] = row.get("ts")
        if entry["_best_rank"] is None or rank < entry["_best_rank"]:
            entry["_best_rank"] = rank

    matches: list[dict[str, Any]] = []
    related: list[dict[str, Any]] = []
    for entry in groups.values():
        (matches if entry["_text"] else related).append(_finalise(entry))

    matches.sort(
        key=lambda item: (-item["score"], item["path"].lower(), item["file_id"])
    )
    related.sort(
        key=lambda item: (-item["cosine"], item["path"].lower(), item["file_id"])
    )
    return SearchResults(
        matches=matches[: max(0, limit)],
        related=related[: max(0, max_related)],
    )


def search(
    config: Config,
    index: Index,
    query: str,
    limit: int = DEFAULT_LIMIT,
) -> SearchResults:
    """Hybrid search; an empty query returns the most recent indexed files."""
    query = query.strip()
    if not query:
        rows = index.recent_files(limit=limit)
        for row in rows:
            row["match_type"] = MATCH_RECENT
            row["fts_rank"] = None
            row["cosine"] = None
        return SearchResults(matches=rows)

    text_rows = index.search_text(query, limit=max(limit, DEFAULT_LIMIT))
    vector_rows: list[dict[str, Any]] = []
    if models.models_ready(config.models_dir):
        try:
            from .extractors.vision import embed_text

            vector = embed_text(
                query,
                models_dir=config.models_dir,
                threads=config.onnx_threads,
            )
        except Exception as exc:
            log.warning("Visual search unavailable: %s", exc)
        else:
            vector_rows = filter_vector_rows(
                index.search_vectors(vector, k=VECTOR_K),
                config.min_visual_similarity,
            )
    return combine(
        text_rows,
        vector_rows,
        limit=limit,
        max_related=config.max_visual_results,
    )
