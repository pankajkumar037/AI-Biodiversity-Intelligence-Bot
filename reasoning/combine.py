"""Candidate selection, hard filters, risk penalties and synergy pairing."""
from __future__ import annotations

import math
import re
from typing import Any

import config
from core.schemas import PracticeCard
from knowledge import loader
from reasoning import causal
from reasoning import graph as graph_mod

_NUMERIC_RULE = re.compile(r"^(\w+)\s*(<=|>=|<|>|==)\s*([\d.]+)$")
_ZONE_RULE = re.compile(r"^climate_zone\s+in\s*\[(.+)\]$")


def risk_applies(condition: str, profile: dict[str, Any], constraints: list[str]) -> bool:
    """Evaluate a soft-risk condition. Anything we cannot parse does not fire."""
    text = (condition or "").strip()
    if text == "always":
        return True
    if text.startswith("constraint "):
        return text.split(" ", 1)[1].strip() in constraints

    zone_match = _ZONE_RULE.match(text)
    if zone_match:
        zones = [zone.strip() for zone in zone_match.group(1).split(",")]
        return profile.get("climate_zone") in zones

    numeric = _NUMERIC_RULE.match(text)
    if numeric:
        field, operator, raw = numeric.groups()
        value = profile.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        threshold = float(raw)
        return {
            "<": value < threshold, "<=": value <= threshold,
            ">": value > threshold, ">=": value >= threshold,
            "==": value == threshold,
        }[operator]
    return False


def blocking_constraint(card: PracticeCard, profile: dict[str, Any],
                        constraints: list[str]) -> str | None:
    """Why this practice is impossible here, or None when it is allowed."""
    hard = card.hard_constraints

    for constraint in hard.incompatible_constraints:
        if constraint in constraints:
            return constraint

    land_use = profile.get("land_use")
    if land_use and hard.land_use_in and land_use not in hard.land_use_in:
        return f"land use {land_use} not suitable"

    rainfall = profile.get("rainfall_mm")
    if isinstance(rainfall, (int, float)) and not isinstance(rainfall, bool):
        if hard.min_rainfall_mm is not None and rainfall < hard.min_rainfall_mm:
            return f"needs at least {hard.min_rainfall_mm:.0f} mm rainfall"
        if hard.max_rainfall_mm is not None and rainfall > hard.max_rainfall_mm:
            return f"unsuited above {hard.max_rainfall_mm:.0f} mm rainfall"
    return None


def score_candidate(card: PracticeCard, flags: list[str], profile: dict[str, Any],
                    constraints: list[str]) -> dict:
    """Suitability for this site, with the risks that were applied and why."""
    graph = graph_mod.load_graph()
    practice_id = card.practice_id.value
    effects, paths = causal.propagate(graph, practice_id, profile)
    benefit = causal.benefit_for_flags(effects, flags)

    addressed = sorted(set(card.addresses_flags) & set(flags))
    coverage = len(addressed) / len(flags) if flags else 0.0

    risks_applied = [
        {
            "risk": risk.risk,
            "penalty": risk.penalty,
            "mitigation": risk.mitigation,
            "condition": risk.condition,
        }
        for risk in card.soft_risks
        if risk_applies(risk.condition, profile, constraints)
    ]
    penalty = min(0.8, sum(risk["penalty"] for risk in risks_applied))

    base = 0.5 * coverage + 0.5 * math.tanh(max(benefit, 0.0))
    suitability = round(max(0.0, base * (1.0 - penalty)), 4)

    negative = {node: round(value, 4) for node, value in effects.items() if value < 0}

    return {
        "practice_id": practice_id,
        "name": card.name,
        "suitability": suitability,
        "base_score": round(base, 4),
        "penalty": round(penalty, 4),
        "coverage": round(coverage, 4),
        "benefit": round(benefit, 4),
        "addresses": addressed,
        "net_effects": {node: round(value, 4) for node, value in effects.items()},
        "harms": negative,
        "risks_applied": risks_applied,
        "downgraded": bool(risks_applied),
        "time_horizon": card.time_horizon,
        "requires_first": card.requires_first,
        "impacted_metrics": [metric.value for metric in card.impacted_metrics],
        "mechanism": card.mechanism,
        "evidence_chunk_ids": card.evidence_chunk_ids,
        "paths": paths[:6],
    }


def candidates(flags: list[str], profile: dict[str, Any], constraints: list[str],
               cards: dict[str, PracticeCard] | None = None
               ) -> tuple[list[dict], list[dict]]:
    """Score every card that addresses a flag. Returns (ranked candidates, excluded)."""
    catalogue = cards if cards is not None else loader.practice_cards()
    ranked: list[dict] = []
    excluded: list[dict] = []

    for card in catalogue.values():
        if not set(card.addresses_flags) & set(flags):
            continue
        reason = blocking_constraint(card, profile, constraints)
        if reason:
            excluded.append({
                "practice_id": card.practice_id.value,
                "name": card.name,
                "reason": reason,
            })
            continue
        ranked.append(score_candidate(card, flags, profile, constraints))

    ranked.sort(key=lambda item: item["suitability"], reverse=True)
    return ranked, excluded


def pair_candidates(ranked: list[dict], flags: list[str],
                    top_n: int = config.TOP_CANDIDATES_TO_PAIR) -> list[dict]:
    """Pair the strongest candidates, rewarding a partner that offsets a harm."""
    targets = causal.flag_targets(flags)
    combos: list[dict] = []
    pool = ranked[:top_n]

    for i, first in enumerate(pool):
        for second in pool[i + 1:]:
            synergies = []
            conflicts = []

            for node in first["harms"]:
                partner_value = second["net_effects"].get(node, 0.0)
                if partner_value > 0:
                    synergies.append(
                        f"{second['name']} raises {node}, which {first['name']} lowers"
                    )
            for node in second["harms"]:
                partner_value = first["net_effects"].get(node, 0.0)
                if partner_value > 0:
                    synergies.append(
                        f"{first['name']} raises {node}, which {second['name']} lowers"
                    )
            for node, desired in targets.items():
                a = first["net_effects"].get(node, 0.0) * desired
                b = second["net_effects"].get(node, 0.0) * desired
                if a < 0 and b < 0:
                    conflicts.append(f"both push {node} the wrong way")

            synergy_bonus = config.SYNERGY_BONUS if synergies else 0.0
            conflict_penalty = config.CONFLICT_PENALTY * len(conflicts)
            score = (
                first["suitability"] + second["suitability"]
                + synergy_bonus - conflict_penalty
            )

            combos.append({
                "combo": [first["practice_id"], second["practice_id"]],
                "names": [first["name"], second["name"]],
                "score": round(score, 4),
                "synergy": synergies[0] if synergies else None,
                "conflicts": conflicts,
                "net_effects": _merge_effects(first["net_effects"], second["net_effects"]),
                "variables": sorted(
                    set(first["impacted_metrics"]) | set(second["impacted_metrics"])
                ),
            })

    combos.sort(key=lambda item: item["score"], reverse=True)
    return combos


def _merge_effects(first: dict[str, float], second: dict[str, float]) -> dict[str, float]:
    merged = dict(first)
    for node, value in second.items():
        merged[node] = round(merged.get(node, 0.0) + value, 4)
    return merged
