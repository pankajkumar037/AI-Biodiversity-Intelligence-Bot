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
            "observed_on_site": bool(cause.get("observed")),
        })
        counter += 1

    for candidate in candidates:
        ids = []
        for path in candidate["paths"]:
            # Direction only. A numeric weight here gets copied out as an "estimate".
            registry.append({
                "id": f"P{counter}", "kind": "effect",
                "practice": candidate["practice_id"],
                "path": path["path"],
                "direction": "raises" if path["weight"] > 0 else "lowers",
            })
            ids.append(f"P{counter}")
            counter += 1
        per_candidate[candidate["practice_id"]] = ids

    return registry, per_candidate


def dossier_candidates(candidates: list[dict], top_n: int = 6,
                       extra_risky: int = 2) -> list[dict]:
    """The best options, plus the most heavily penalised ones.

    Showing only the winners hides the interesting half of the reasoning. A practice
    the user has just asked about can rank low precisely because it carries a large
    risk, and that risk is the thing worth explaining.
    """
    shortlist = list(candidates[:top_n])
    chosen = {candidate["practice_id"] for candidate in shortlist}
    penalised = sorted(
        (c for c in candidates[top_n:] if c["penalty"] > 0),
        key=lambda c: c["penalty"], reverse=True,
    )
    for candidate in penalised[:extra_risky]:
        if candidate["practice_id"] not in chosen:
            shortlist.append(candidate)
    return shortlist


def evidence_by_practice(evidence: list[EvidenceItem]) -> dict[str, list[str]]:
    """practice_id -> the S-labels of retrieved chunks tagged with that practice.

    Telling the model which chunks are about which practice is not citing for it:
    it still chooses, and V7 still judges whether the text supports the mechanism.
    """
    index: dict[str, list[str]] = {}
    for item in evidence:
        for practice in item.practices:
            index.setdefault(practice, []).append(item.label)
    return index


def build_dossier(profile: dict[str, Any], flags: list[str], patterns: list[dict],
                  root_causes: list[dict], leverage: list[dict], candidates: list[dict],
                  combos: list[dict], excluded: list[dict], plan: list[dict],
                  evidence: list[EvidenceItem] | None = None, asks: str = "none",
                  problem_flags: list[str] | None = None) -> dict:
    """The compact, engine-produced picture the model is asked to explain."""
    candidates = dossier_candidates(candidates)
    registry, per_candidate = build_paths(root_causes, candidates)
    labels_for = evidence_by_practice(evidence or [])
    return {
        "question": asks,
        "stated_problem": list(problem_flags or []),
        "site": profile,
        "diagnosis": flags,
        "patterns": [
            {"id": pattern["rule_id"], "name": pattern["name"], "loop": pattern.get("loop", [])}
            for pattern in patterns
        ],
        "paths": registry,
        "leverage_ranking": [
            {"rank": rank, "node": item["node"], "reaches": item["reaches"]}
            for rank, item in enumerate(leverage[:5], start=1)
        ],
        "candidates": [
            {
                "rank": rank,
                "practice_id": candidate["practice_id"],
                "name": candidate["name"],
                "addresses": candidate["addresses"],
                "impacted_metrics": candidate["impacted_metrics"],
                # Qualitative on purpose: the model explains direction, never magnitude.
                "raises": sorted(n for n, v in candidate["net_effects"].items() if v > 0),
                "lowers": sorted(n for n, v in candidate["net_effects"].items() if v < 0),
                "mechanism": candidate["mechanism"],
                "downgraded": candidate["downgraded"],
                "risks_applied": [
                    {"risk": r["risk"], "mitigation": r["mitigation"]}
                    for r in candidate["risks_applied"]
                ],
                "time_horizon": candidate["time_horizon"],
                "path_ids": per_candidate.get(candidate["practice_id"], []),
                "evidence_labels": labels_for.get(candidate["practice_id"], []),
            }
            for rank, candidate in enumerate(candidates, start=1)
        ],
        "best_combinations": [
            {"combo": c["combo"], "synergy": c["synergy"], "conflicts": c["conflicts"]}
            for c in combos[:3]
        ],
        "excluded": excluded,
        "sequence": [
            {"order": s["order"], "practice_id": s["practice_id"],
             "time_horizon": s["time_horizon"], "requires_first": s["requires_first"]}
            for s in plan
        ],
        "note": ("Candidates are listed in the engine's rank order. There are no effect "
                 "sizes anywhere in this dossier; the only numbers you may report are "
                 "on CLAIMS lines in the evidence block."),
    }


def path_ids(candidates: list[dict], root_causes: list[dict]) -> set[str]:
    """Every path id the model is allowed to cite.

    Built from the same shortlist the dossier shows, so verification and the prompt
    always agree on which ids exist.
    """
    registry, _ = build_paths(root_causes, dossier_candidates(candidates))
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
