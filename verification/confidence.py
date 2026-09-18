"""confidence = evidence * context * agreement * data_quality, always with a breakdown."""
from __future__ import annotations

from typing import Any

import config
from core.schemas import (
    ConfidenceBreakdown,
    ConfidenceReport,
    EvidenceItem,
    SiteProfile,
)

GENERIC_ZONES = {"global", "unstated"}


def evidence_factor(items: list[EvidenceItem]) -> float:
    """Mean source strength of the chunks actually cited."""
    if not items:
        return 0.3
    weights = [
        config.EVIDENCE_WEIGHT.get(item.evidence_level or "", 0.7) for item in items
    ]
    return sum(weights) / len(weights)


def context_factor(items: list[EvidenceItem], site_zone: str | None,
                   filter_level: int) -> float:
    """How well the cited evidence fits this site, after any filter fallback."""
    penalty = config.FILTER_LEVEL_PENALTY.get(filter_level, 0.75)
    if not items or not site_zone:
        return 0.6 * penalty
    matches = 0
    for item in items:
        zones = set(item.climate_zones)
        if site_zone in zones:
            matches += 1
        elif zones & GENERIC_ZONES:
            matches += 0.7
    return min(1.0, matches / len(items)) * penalty


def agreement_factor(supporting: int, contradicting: int) -> float:
    """Supporting versus contradicting chunks found by the risk queries."""
    total = supporting + contradicting
    if total == 0:
        return 0.6
    return max(0.2, supporting / total)


def data_quality_factor(profile: SiteProfile, required: list[str] | None = None) -> float:
    """Share of the decisive fields the user actually stated, not inferred."""
    fields = required or ["soc_percent", "rainfall_mm", "land_use", "cropping_system",
                          "climate_zone"]
    if not fields:
        return 0.5
    score = 0.0
    for name in fields:
        field = profile.fields.get(name)
        if field is None:
            continue
        score += {"user": 1.0, "soilgrids": 0.8, "nasa_power": 0.8,
                  "nominatim": 0.8, "inferred": 0.4}.get(field.source.value, 0.5)
    return max(0.2, score / len(fields))


def band(value: float) -> str:
    if value >= config.CONFIDENCE_HIGH:
        return "high"
    if value >= config.CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def compute(items: list[EvidenceItem], profile: SiteProfile, site_zone: str | None,
            filter_level: int, supporting: int, contradicting: int,
            extra_penalty: float = 1.0) -> ConfidenceReport:
    """Multiply the four factors and report each one alongside the result."""
    breakdown = ConfidenceBreakdown(
        evidence=round(evidence_factor(items), 3),
        context=round(context_factor(items, site_zone, filter_level), 3),
        agreement=round(agreement_factor(supporting, contradicting), 3),
        data_quality=round(data_quality_factor(profile), 3),
    )
    value = (
        breakdown.evidence * breakdown.context
        * breakdown.agreement * breakdown.data_quality
        * extra_penalty
    )
    value = round(min(1.0, max(0.0, value)), 3)
    return ConfidenceReport(value=value, band=band(value), breakdown=breakdown)


def count_support(items: list[EvidenceItem]) -> tuple[int, int]:
    """Split cited chunks into supporting and risk/constraint evidence."""
    contradicting = sum(
        1 for item in items if item.content_role in ("risk", "constraint")
    )
    return len(items) - contradicting, contradicting


def summarise(profile: SiteProfile) -> dict[str, Any]:
    """Per-field provenance, used by the renderer and the trace."""
    return {
        name: {"value": field.value, "source": field.source.value,
               "uncertainty": field.uncertainty}
        for name, field in profile.fields.items()
    }
