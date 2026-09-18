"""Explainable scoring: rrf_score * evidence_weight * context_match."""
from __future__ import annotations

import config


def context_match(chunk: dict, site_zone: str | None, filter_level: int) -> float:
    """How well a chunk's climate tagging fits this site."""
    if filter_level > 0:
        return config.CONTEXT_MATCH_FALLBACK
    zones = chunk.get("climate_zones") or []
    if site_zone and site_zone in zones:
        return config.CONTEXT_MATCH_EXACT
    if set(zones) & set(config.ALWAYS_ALLOWED_ZONES):
        return config.CONTEXT_MATCH_GENERIC
    return config.CONTEXT_MATCH_FALLBACK


def score_chunks(chunks: list[dict], site_zone: str | None,
                 filter_level: int) -> list[dict]:
    """Attach final_score and its factors, then sort. No model in the request path."""
    scored = []
    for chunk in chunks:
        evidence_weight = chunk.get("evidence_weight")
        if evidence_weight is None:
            evidence_weight = config.EVIDENCE_WEIGHT.get(
                chunk.get("evidence_level", ""), 0.7
            )
        match = context_match(chunk, site_zone, filter_level)
        rrf = chunk.get("rrf_score", 0.0)
        scored.append({
            **chunk,
            "context_match": match,
            "evidence_weight": evidence_weight,
            "final_score": rrf * evidence_weight * match,
        })
    return sorted(scored, key=lambda c: c["final_score"], reverse=True)
