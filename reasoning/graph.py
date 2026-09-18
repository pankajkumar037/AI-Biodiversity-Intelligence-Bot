"""The causal variable graph: load it once, and decide when a conditional edge applies."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import networkx as nx

import config
from core import db

# The extracted graph carries a few spellings of the same node. Normalise to one vocabulary.
NODE_ALIASES = {
    "soil_erosion": "erosion",
    "crop yield": "crop_yield",
    "soil moisture": "soil_moisture",
    "fungi abundance": "soil_biota",
    "pollinator_diversity": "pollinators",
    "pollinator_habitat": "habitat_diversity",
    "soil_fertility": "nutrient_availability",
    "nutrient_cycling": "nutrient_availability",
    "water_availability": "soil_moisture",
}

# Climate words as they actually appear in extracted conditions, per site zone.
ZONE_KEYWORDS = {
    "arid": ["arid", "dryland", "drylands", "dryclimate", "dryclimates", "desert", "dryyears"],
    "semi-arid": ["semiarid", "dryland", "drylands", "dryclimate", "dryclimates", "dryyears",
                  "dryseason"],
    "dry_sub_humid": ["drysubhumid", "subhumid", "dryland", "drylands"],
    "humid": ["humid", "wet", "highrainfall", "coolwet"],
    "temperate": ["temperate", "coolwet"],
    "tropical": ["tropical", "subtropical"],
}

LAND_USE_KEYWORDS = ["cropland", "grassland", "pasture", "orchard", "forest", "arable"]


def normalise_node(name: str) -> str:
    """One spelling per concept, matching the metric vocabulary where they overlap."""
    cleaned = (name or "").strip().replace(" ", "_")
    return NODE_ALIASES.get(cleaned, NODE_ALIASES.get(name.strip(), cleaned))


def _squash(text: str) -> str:
    return text.lower().replace("-", "").replace(" ", "").replace("_", "")


def condition_holds(condition: str | None, profile: dict[str, Any]) -> bool:
    """Does a free-text edge condition apply to this site?

    Unconditional edges always hold. Otherwise we match climate and land-use words.
    A condition we cannot interpret returns False, so the engine never claims an
    effect on evidence that may not apply here.
    """
    if not condition:
        return True

    text = _squash(condition)
    zone = profile.get("climate_zone")

    # "semiarid" contains "arid", so consume the longer word before testing the shorter.
    stripped = text.replace("semiarid", "")
    mentions_semiarid = "semiarid" in text
    mentions_arid = "arid" in stripped
    mentions_any_zone = mentions_semiarid or mentions_arid or any(
        keyword in text
        for keywords in ZONE_KEYWORDS.values()
        for keyword in keywords
        if keyword not in ("arid",)
    )

    if mentions_any_zone:
        if not zone:
            return False
        keywords = ZONE_KEYWORDS.get(zone, [])
        if mentions_semiarid and "semiarid" in keywords:
            return True
        if mentions_arid and "arid" in keywords:
            return True
        return any(keyword in text for keyword in keywords if keyword != "arid")

    land_use = _squash(str(profile.get("land_use") or ""))
    for keyword in LAND_USE_KEYWORDS:
        if keyword in text:
            return bool(land_use) and keyword in land_use

    return False


def edge_weight(edge: dict) -> float:
    """Strength of an edge, nudged up by how many documents support it."""
    strength = config.STRENGTH_WEIGHT.get(edge.get("strength") or "unstated", 0.5)
    support = min(int(edge.get("support_count") or 1), 5)
    return strength * (0.8 + 0.04 * support)


@lru_cache(maxsize=1)
def load_graph() -> nx.MultiDiGraph:
    """Every edge, normalised. Propagation later restricts itself to traversable ones."""
    graph = nx.MultiDiGraph()
    for edge in db.variable_graph().find({}):
        source = normalise_node(edge["from_node"])
        target = normalise_node(edge["to_node"])
        if not source or not target:
            continue
        graph.add_edge(
            source, target,
            key=edge["_id"],
            sign=int(edge.get("sign", 0)),
            effect=edge.get("effect"),
            condition=edge.get("condition"),
            strength=edge.get("strength"),
            support_count=int(edge.get("support_count") or 1),
            traversable=bool(edge.get("traversable")),
            docs=edge.get("docs") or [],
            units=edge.get("units") or [],
            mechanisms=edge.get("mechanisms") or [],
            weight=edge_weight(edge),
        )
    return graph


def out_edges(graph: nx.MultiDiGraph, node: str, profile: dict[str, Any],
              traversable_only: bool = True) -> list[tuple[str, dict]]:
    """Applicable outgoing edges: traversable, non-zero sign, condition satisfied."""
    if node not in graph:
        return []
    applicable = []
    for _, target, data in graph.out_edges(node, data=True):
        if traversable_only and not data["traversable"]:
            continue
        if data["sign"] == 0:
            continue
        if not condition_holds(data["condition"], profile):
            continue
        applicable.append((target, data))
    return applicable


def in_edges(graph: nx.MultiDiGraph, node: str, profile: dict[str, Any],
             traversable_only: bool = True) -> list[tuple[str, dict]]:
    """Applicable incoming edges, used when tracing a symptom back to its causes."""
    if node not in graph:
        return []
    applicable = []
    for source, _, data in graph.in_edges(node, data=True):
        if traversable_only and not data["traversable"]:
            continue
        if data["sign"] == 0:
            continue
        if not condition_holds(data["condition"], profile):
            continue
        applicable.append((source, data))
    return applicable


def risk_edges(graph: nx.MultiDiGraph, node: str,
               profile: dict[str, Any]) -> list[tuple[str, dict]]:
    """Negative effects of a practice that apply here, including non-traversable edges.

    NOTE: risks are read from the full graph on purpose. Several well-evidenced
    trade-offs (cover crops drying semi-arid soils, for one) are marked
    non-traversable, and silently dropping them would hide exactly the caveats
    a recommendation needs to carry.
    """
    if node not in graph:
        return []
    found = []
    for _, target, data in graph.out_edges(node, data=True):
        if data["sign"] >= 0:
            continue
        if not condition_holds(data["condition"], profile):
            continue
        found.append((target, data))
    return found


def clear_cache() -> None:
    """Drop the cached graph. Used by tests."""
    load_graph.cache_clear()
