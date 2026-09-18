"""Turn a diagnosis into targeted queries, then run them. Never uses the raw user message."""
from __future__ import annotations

import config
from core.schemas import EvidenceItem, PlannedQuery
from retrieval import assemble as assemble_mod
from retrieval import rerank, search

# Readable query text per practice id. Search works on words, not identifiers.
PRACTICE_TEXT = {
    "cover_crops": "cover crops",
    "mulching": "residue mulching",
    "crop_rotation": "crop rotation",
    "intercropping": "cereal legume intercropping",
    "no_till": "no-till",
    "reduced_tillage": "reduced tillage",
    "manure": "farmyard manure",
    "compost": "compost application",
    "integrated_nutrient_mgmt": "integrated nutrient management",
    "irrigation": "irrigation",
    "terracing": "terracing",
    "check_dams": "check dams",
    "shelterbelts": "shelterbelts windbreaks",
    "hedges_buffer_strips": "hedgerows field margin buffer strips",
    "grassland_restoration": "grassland restoration",
    "rotational_grazing": "rotational grazing",
    "crop_livestock_integration": "crop livestock integration",
    "conservation_agriculture": "conservation agriculture",
    "agroforestry_agrisilvicultural": "agroforestry trees with crops",
    "agroforestry_silvopastoral": "silvopastoral agroforestry trees with pasture",
    "agroforestry_agrosilvopastoral": "agrosilvopastoral agroforestry",
    "gypsum_amendment": "gypsum amendment sodic soil",
    "water_harvesting": "rainwater harvesting",
}

# What to probe for when asking "how could this practice go wrong here?".
RISK_TERMS = {
    "cover_crops": "water competition reduced cash crop yield dry climate",
    "intercropping": "competition for water and light yield penalty",
    "agroforestry_agrisilvicultural": "tree crop water competition shading dryland",
    "agroforestry_silvopastoral": "tree pasture competition dryland",
    "agroforestry_agrosilvopastoral": "tree crop livestock competition dryland",
    "irrigation": "salinization waterlogging dryland",
    "no_till": "weed pressure yield penalty",
    "reduced_tillage": "weed pressure compaction",
    "rotational_grazing": "overgrazing degradation",
    "shelterbelts": "land taken out of production water use",
    "mulching": "residue availability competing uses",
}
DEFAULT_RISK_TERM = "drawbacks constraints trade-offs limitations"

FLAG_QUERY_TEXT = {
    "low_SOC": "restoring soil organic carbon in degraded cropland",
    "water_limited": "soil moisture conservation under low rainfall",
    "monoculture": "crop diversification biodiversity monoculture",
    "low_habitat_diversity": "field margin vegetation habitat diversity farmland",
    "alkaline_soil": "alkaline sodic soil management",
    "acidic_soil": "acidic soil management liming",
    "low_natural_cover": "habitat fragmentation natural vegetation cover farmland",
    "biodiversity_decline": "farmland biodiversity decline drivers",
    "pollinator_decline": "pollinator decline drivers agrochemicals habitat loss farmland",
    "high_pesticide_use": "pesticide effects on pollinators and soil organisms",
    "residue_burning": "crop residue burning soil organic carbon loss",
    "recent_clearing": "land conversion deforestation habitat loss biodiversity",
    "pollution_exposure": "soil pollution sources effects on soil biota and crops",
    "erosion_observed": "soil erosion control vegetation cover cropland",
}

# Queries about the problem the user raised go in before practice queries, so
# the evidence block always carries something about what they actually asked.
PROBLEM_FIRST = ("pollinator_decline", "high_pesticide_use", "biodiversity_decline",
                 "residue_burning", "recent_clearing", "pollution_exposure",
                 "erosion_observed")


def practice_text(practice_id: str) -> str:
    """Readable search phrase for a practice id."""
    return PRACTICE_TEXT.get(practice_id, practice_id.replace("_", " "))


def plan(flags: list[str], practices: list[str], paths: list[dict],
         site_zone: str | None = None) -> list[PlannedQuery]:
    """Build 6-10 queries: support and risk per practice, plus path and flag queries."""
    zone = (site_zone or "").replace("_", " ")
    queries: list[PlannedQuery] = []

    for flag in PROBLEM_FIRST:
        if flag in flags and flag in FLAG_QUERY_TEXT:
            queries.append(PlannedQuery(
                q=f"{FLAG_QUERY_TEXT[flag]} {zone}".strip(), purpose="support",
                climate_zone=site_zone,
            ))

    practice_budget = 4 if len(queries) <= 1 else 3
    for practice_id in practices[:practice_budget]:
        text = practice_text(practice_id)
        queries.append(PlannedQuery(
            q=f"{text} soil organic carbon biodiversity {zone}".strip(),
            purpose="support", practice_id=practice_id, climate_zone=site_zone,
        ))
        risk_term = RISK_TERMS.get(practice_id, DEFAULT_RISK_TERM)
        queries.append(PlannedQuery(
            q=f"{text} {risk_term} {zone}".strip(),
            purpose="risk", practice_id=practice_id, climate_zone=site_zone,
        ))

    for path in paths[:2]:
        nodes = path.get("path", [])
        if len(nodes) < 2:
            continue
        readable = " ".join(node.replace("_", " ") for node in nodes)
        queries.append(PlannedQuery(
            q=f"{readable} {zone}".strip(), purpose="path",
            path_id=path.get("id"), climate_zone=site_zone,
        ))

    for flag in flags:
        text = FLAG_QUERY_TEXT.get(flag)
        if flag in PROBLEM_FIRST:
            continue
        if text and len(queries) < config.MAX_QUERIES:
            queries.append(PlannedQuery(
                q=f"{text} {zone}".strip(), purpose="support", climate_zone=site_zone,
            ))

    if not queries:
        queries.append(PlannedQuery(
            q="soil organic carbon biodiversity land management",
            purpose="background", climate_zone=site_zone,
        ))
    return queries[:config.MAX_QUERIES]


def run_plan(queries: list[PlannedQuery], site_zone: str | None = None,
             limit: int = config.EVIDENCE_BLOCK_SIZE
             ) -> tuple[list[EvidenceItem], list[dict], int]:
    """Run every planned query, pool the hits and assemble the evidence block.

    Returns (evidence items, per-query trace records, worst filter level used).
    """
    pooled: dict[str, dict] = {}
    traces: list[dict] = []
    worst_level = 0

    for query in queries:
        allow_context = query.purpose == "background"
        hits, level = search.search_with_fallback(
            query.q, site_zone, query.practice_id, allow_context_only=allow_context
        )
        scored = rerank.score_chunks(hits, site_zone, level)
        worst_level = max(worst_level, level)

        for chunk in scored:
            existing = pooled.get(chunk["_id"])
            if existing is None or chunk["final_score"] > existing["final_score"]:
                pooled[chunk["_id"]] = chunk

        traces.append({
            "q": query.q,
            "purpose": query.purpose,
            "practice_id": query.practice_id,
            "filter_level": level,
            "n_candidates": len(hits),
            "kept": [chunk["_id"] for chunk in scored[:5]],
        })

    ordered = sorted(pooled.values(), key=lambda c: c["final_score"], reverse=True)
    items = assemble_mod.assemble(ordered, limit=limit)
    return items, traces, worst_level
