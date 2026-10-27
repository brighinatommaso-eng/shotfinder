"""Hybrid text + vector search with reciprocal rank fusion (Phase 2)."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from . import models
from .config import Config
from .index import Index

log = logging.getLogger(__name__)

RRF_K = 60
VECTOR_K = 50
DEFAULT_LIMIT = 40
RELATIVE_SIMILARITY_MARGIN = 0.05

MATCH_TEXT = "text"
MATCH_VISUAL = "visual"
MATCH_BOTH = "both"
MATCH_RECENT = "recent"


def distance_to_cosine(distance: float) -> float:
    """Convert sqlite-vec L2 distance to cosine (stored vectors are unit length)."""
    return 1.0 - (distance * distance) / 2.0


def filter_vector_rows(
    rows: Sequence[dict[str, Any]],
    min_similarity: float,
    *,
    margin: float = RELATIVE_SIMILARITY_MARGIN,
) -> list[dict[str, Any]]:
    """Drop weak vector hits before they reach RRF.

    The absolute floor removes noise that has no visual relation to the query.
    The relative margin then removes a long tail that is only slightly weaker
    than the best hit, so one strong match is not diluted by many mediocre ones.
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


def merge(
    text_rows: Sequence[dict[str, Any]],
    vector_rows: Sequence[dict[str, Any]],
    *,
    k: int = RRF_K,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Reciprocal rank fusion, grouped by ``file_id``.

    Each list contributes ``1 / (k + rank)`` per unique file. Rows are
    deduplicated first, so ten matching OCR lines do not outvote ten files. The
    timestamp of the highest-ranked hit is kept as the video jump point.
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
                "score": 0.0,
                "match_type": MATCH_TEXT,
                "fts_rank": None,
                "cosine": None,
                "_text": False,
                "_visual": False,
                "_best_rank": None,
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
        if entry["fts_rank"] is None or rank < entry["fts_rank"]:
            entry["fts_rank"] = rank
            entry["snippet"] = row.get("snippet") or ""
        if entry["_best_rank"] is None or rank < entry["_best_rank"]:
            entry["_best_rank"] = rank
            entry["ts"] = row.get("ts")

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
        if entry["_best_rank"] is None or rank < entry["_best_rank"]:
            entry["_best_rank"] = rank
            entry["ts"] = row.get("ts")

    ordered = sorted(
        groups.values(),
        key=lambda item: (-item["score"], item["path"].lower(), item["file_id"]),
    )
    results: list[dict[str, Any]] = []
    for entry in ordered[: max(0, limit)]:
        entry = dict(entry)
        entry["match_type"] = (
            MATCH_BOTH
            if entry["_text"] and entry["_visual"]
            else MATCH_TEXT
            if entry["_text"]
            else MATCH_VISUAL
        )
        entry.pop("_text", None)
        entry.pop("_visual", None)
        entry.pop("_best_rank", None)
        results.append(entry)
    return results


def search(
    config: Config,
    index: Index,
    query: str,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Hybrid search; an empty query returns the most recent indexed files."""
    query = query.strip()
    if not query:
        rows = index.recent_files(limit=limit)
        for row in rows:
            row["match_type"] = MATCH_RECENT
            row["fts_rank"] = None
            row["cosine"] = None
        return rows

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
    return merge(text_rows, vector_rows, limit=limit)
