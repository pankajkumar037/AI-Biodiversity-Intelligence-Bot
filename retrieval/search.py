"""Vector + text search over chunks, fused with RRF, with a filter fallback ladder."""
from __future__ import annotations

from typing import Any

import config
from core import db, llm

PROJECTION = {
    "text": 1, "doc_id": 1, "doc_title": 1, "heading_path": 1, "topic": 1,
    "content_role": 1, "practices": 1, "metrics": 1, "climate_zones": 1,
    "land_use": 1, "evidence_level": 1, "evidence_weight": 1, "context_only": 1,
    "page_start": 1, "page_end": 1, "claims": 1, "n_claims": 1,
}


def zone_filter_values(site_zone: str | None) -> list[str]:
    """Site zone plus the generic zones, so well-tagged chunks are never lost."""
    zones = list(config.ALWAYS_ALLOWED_ZONES)
    if site_zone and site_zone not in zones:
        zones.insert(0, site_zone)
    return zones


def build_filter(site_zone: str | None, practice: str | None, level: int,
                 allow_context_only: bool) -> dict[str, Any]:
    """Filter for a rung of the fallback ladder. Level 0 zone+practice, 1 practice, 2 none."""
    clauses: list[dict] = []
    if not allow_context_only:
        clauses.append({"context_only": {"$eq": False}})
    if level <= 1 and practice:
        clauses.append({"practices": {"$in": [practice]}})
    if level == 0 and site_zone:
        clauses.append({"climate_zones": {"$in": zone_filter_values(site_zone)}})
    if not clauses:
        return {}
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def vector_search(query: str, mongo_filter: dict[str, Any],
                  limit: int = config.VECTOR_LIMIT) -> list[dict]:
    """$vectorSearch on vector_index. Query vector is RETRIEVAL_QUERY, 768d, normalised."""
    stage: dict[str, Any] = {
        "index": config.VECTOR_INDEX,
        "path": "embedding",
        "queryVector": llm.embed_query(query),
        "numCandidates": config.VECTOR_NUM_CANDIDATES,
        "limit": limit,
    }
    if mongo_filter:
        stage["filter"] = mongo_filter
    pipeline = [
        {"$vectorSearch": stage},
        {"$project": {**PROJECTION, "score": {"$meta": "vectorSearchScore"}}},
    ]
    return list(db.chunks().aggregate(pipeline))


def text_search(query: str, limit: int = config.TEXT_LIMIT) -> list[dict]:
    """$search on text_index. The index maps only text/topic/doc_title, so no filters here."""
    pipeline = [
        {"$search": {
            "index": config.TEXT_INDEX,
            "text": {"query": query, "path": ["text", "topic", "doc_title"]},
        }},
        {"$limit": limit},
        {"$project": {**PROJECTION, "score": {"$meta": "searchScore"}}},
    ]
    return list(db.chunks().aggregate(pipeline))


def _passes(chunk: dict, site_zone: str | None, practice: str | None, level: int,
            allow_context_only: bool) -> bool:
    """Apply the same filter to text hits, which the text index cannot filter itself."""
    if not allow_context_only and chunk.get("context_only"):
        return False
    if level <= 1 and practice and practice not in (chunk.get("practices") or []):
        return False
    if level == 0 and site_zone:
        zones = chunk.get("climate_zones") or []
        if not set(zones) & set(zone_filter_values(site_zone)):
            return False
    return True


def rrf_merge(result_lists: list[list[dict]], k: int = config.RRF_K) -> list[dict]:
    """Reciprocal rank fusion. Returns chunks carrying an rrf_score."""
    fused: dict[str, dict] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results):
            chunk_id = chunk["_id"]
            entry = fused.setdefault(chunk_id, {**chunk, "rrf_score": 0.0})
            entry["rrf_score"] += 1.0 / (k + rank + 1)
    return sorted(fused.values(), key=lambda c: c["rrf_score"], reverse=True)


def hybrid_search(query: str, site_zone: str | None = None, practice: str | None = None,
                  level: int = 0, allow_context_only: bool = False,
                  limit: int = config.VECTOR_LIMIT) -> list[dict]:
    """One rung: vector + text, filtered consistently, fused with RRF."""
    mongo_filter = build_filter(site_zone, practice, level, allow_context_only)
    vector_hits = vector_search(query, mongo_filter, limit)
    text_hits = [
        chunk for chunk in text_search(query, limit)
        if _passes(chunk, site_zone, practice, level, allow_context_only)
    ]
    return rrf_merge([vector_hits, text_hits])


def search_with_fallback(query: str, site_zone: str | None = None,
                         practice: str | None = None, allow_context_only: bool = False,
                         min_results: int = 3) -> tuple[list[dict], int]:
    """Walk the ladder until a rung returns enough chunks. Returns (chunks, filter_level)."""
    for level in (0, 1, 2):
        if level == 0 and not site_zone:
            continue
        if level == 1 and not practice:
            continue
        results = hybrid_search(query, site_zone, practice, level, allow_context_only)
        if len(results) >= min_results:
            return results, level
    results = hybrid_search(query, site_zone, practice, 2, allow_context_only)
    return results, 2
