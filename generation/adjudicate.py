"""The single reasoning LLM call: explain and challenge the engine's ranking."""
from __future__ import annotations

import json
from typing import Any

import config
from core import llm
from core.schemas import Adjudication, EvidenceItem
from generation import prompts
from retrieval import assemble as assemble_mod


def build_paths(root_causes: list[dict],
                candidates: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    """One registry of path ids across root causes and effect paths.

    Ids must be unique across the whole dossier: if a root cause and a candidate
    both owned a "P1", a citation could not be checked against either.
    """
    registry: list[dict] = []
    per_candidate: dict[str, list[str]] = {}
    counter = 1

    for cause in root_causes[:6]:
        registry.append({
            "id": f"P{counter}", "kind": "root_cause",
            "path": cause["path"], "effect": cause.get("effect"),
        })
        counter += 1

    for candidate in candidates[:6]:
        ids = []
        for path in candidate["paths"]:
            registry.append({
                "id": f"P{counter}", "kind": "effect",
                "practice": candidate["practice_id"],
                "path": path["path"], "weight": path["weight"],
            })
            ids.append(f"P{counter}")
            counter += 1
        per_candidate[candidate["practice_id"]] = ids

    return registry, per_candidate


def build_dossier(profile: dict[str, Any], flags: list[str], patterns: list[dict],
                  root_causes: list[dict], leverage: list[dict], candidates: list[dict],
                  combos: list[dict], excluded: list[dict], plan: list[dict]) -> dict:
    """The compact, engine-produced picture the model is asked to explain."""
    registry, per_candidate = build_paths(root_causes, candidates)
    return {
        "site": profile,
        "diagnosis": flags,
        "patterns": [
            {"id": pattern["rule_id"], "name": pattern["name"], "loop": pattern.get("loop", [])}
            for pattern in patterns
        ],
        "paths": registry,
        "leverage_ranking": [
            {"node": item["node"], "score": item["score"], "reaches": item["reaches"]}
            for item in leverage[:5]
        ],
        "candidates": [
            {
                "practice_id": candidate["practice_id"],
                "name": candidate["name"],
                "suitability": candidate["suitability"],
                "addresses": candidate["addresses"],
                "impacted_metrics": candidate["impacted_metrics"],
                "net_effects": candidate["net_effects"],
                "mechanism": candidate["mechanism"],
                "downgraded": candidate["downgraded"],
                "risks_applied": candidate["risks_applied"],
                "time_horizon": candidate["time_horizon"],
                "path_ids": per_candidate.get(candidate["practice_id"], []),
            }
            for candidate in candidates[:6]
        ],
        "best_combinations": combos[:3],
        "excluded": excluded,
        "sequence": plan,
        "note": "Suitability and net_effects are ranking scores, not real-world effect sizes.",
    }


def path_ids(candidates: list[dict], root_causes: list[dict]) -> set[str]:
    """Every path id the model is allowed to cite."""
    registry, _ = build_paths(root_causes, candidates)
    return {path["id"] for path in registry}


def adjudicate(dossier: dict, evidence: list[EvidenceItem],
               feedback: str | None = None, previous: Adjudication | None = None,
               max_recommendations: int = 3) -> Adjudication:
    """One Flash call with thinking on. Returns the structured adjudication."""
    user = prompts.ADJUDICATE_USER_V1.format(
        dossier=json.dumps(dossier, indent=2, default=str),
        evidence=assemble_mod.render_evidence_block(evidence),
        max_recommendations=max_recommendations,
    )
    if feedback and previous is not None:
        user = user + "\n\n" + prompts.RETRY_FEEDBACK_V1.format(
            failures=feedback,
            previous=previous.model_dump_json(indent=2),
        )

    result = llm.generate_json(
        system=prompts.ADJUDICATE_SYSTEM_V1,
        user=user,
        schema=Adjudication,
        model=config.REASONING_MODEL,
        temperature=config.ADJUDICATE_TEMPERATURE,
    )
    return result
