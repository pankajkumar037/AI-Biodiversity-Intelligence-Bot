"""One function per node. Each takes the state and returns a partial update."""
from __future__ import annotations

import time
from typing import Any

import config
from core.schemas import (
    Adjudication,
    EvidenceItem,
    FieldSource,
    Intent,
)
from generation import adjudicate as adjudicate_mod
from generation import render as render_mod
from graph_flow.state import REPLACE, AgentState, as_fields, profile_values, to_profile
from intake import extract, geo, normalize
from knowledge import loader
from reasoning import causal, combine, diagnose, sequence, voi
from reasoning import graph as graph_mod
from retrieval import assemble as assemble_mod
from retrieval import planner, rerank, search
from verification import checks
from verification import confidence as confidence_mod

CRITICAL_SLOTS = ["soc_percent", "rainfall_mm", "land_use", "cropping_system"]


# ── input ──────────────────────────────────────────────────────────────
def intake(state: AgentState) -> dict[str, Any]:
    """Extract stated facts from the user's message into draft values."""
    message = (state.get("message") or "").strip()
    turn = int(state.get("turn", 0))
    if not message:
        return {}

    try:
        draft = extract.extract_draft(message)
    except RuntimeError as exc:
        return {"warnings": [f"Could not read values from your message: {exc}"]}

    values, derived, warnings, rejected = normalize.normalise_draft(draft)
    update: dict[str, Any] = {
        "profile": {
            **as_fields(values, FieldSource.user, turn),
            **as_fields(derived, FieldSource.inferred, turn, uncertainty=0.6),
        },
        "trace": {"intake": {"extracted": values, "derived": derived, "rejected": rejected}},
    }
    if warnings or rejected:
        update["warnings"] = warnings + rejected
    if draft.constraints:
        update["user_constraints"] = draft.constraints
    if draft.audience:
        update["audience"] = draft.audience.value
    return update


def normalize_node(state: AgentState) -> dict[str, Any]:
    """Fill what can be inferred cheaply, marked as inference rather than fact."""
    values = profile_values(state.get("profile", {}))
    inferred = normalize.infer_missing(values)
    if not inferred:
        return {}
    return {
        "profile": as_fields(inferred, FieldSource.inferred, int(state.get("turn", 0)),
                             uncertainty=0.5),
        "trace": {"inferred": inferred},
    }


def geo_enrich(state: AgentState) -> dict[str, Any]:
    """Fill soil and climate from public APIs when we have a location. Optional."""
    values = profile_values(state.get("profile", {}))
    lat, lon = values.get("lat"), values.get("lon")
    place = None
    how = "coordinates given"

    if (lat is None or lon is None) and values.get("place_name"):
        located = geo.geocode(str(values["place_name"]))
        if located:
            lat, lon = located["lat"], located["lon"]
            place = located.get("display_name")
            how = f"geocoded from '{values['place_name']}'"

    if lat is None or lon is None:
        return {}

    lat, lon = round(float(lat), 4), round(float(lon), 4)
    if place is None:
        place = geo.reverse_geocode(lat, lon)
    enriched = geo.enrich(lat, lon)
    update: dict[str, dict] = {}
    turn = int(state.get("turn", 0))

    soil = enriched.get("soil") or {}
    if soil:
        update.update(as_fields(soil, FieldSource.soilgrids, turn, uncertainty=0.4))
    climate = enriched.get("climate") or {}
    if climate:
        update.update(as_fields(
            {"rainfall_mm": climate.get("rainfall_mm")}, FieldSource.nasa_power,
            turn, uncertainty=0.3,
        ))

    geo_trace = {"lat": lat, "lon": lon, "place": place, "how": how,
                 "soil": soil, "climate": climate}
    if not update:
        return {"trace": {"geo": {**geo_trace, "result": "no data"}}}
    return {
        "profile": {**as_fields({"lat": lat, "lon": lon}, FieldSource.nominatim, turn),
                    **update},
        "trace": {"geo": geo_trace},
    }


def route_intent(state: AgentState) -> dict[str, Any]:
    """Classify the turn before anything is read into the profile.

    A general question on turn one used to be forced to new_info and then mined for
    site values it never contained. Every turn is classified now. This node also
    puts the baseline profile back if the previous turn was a what-if, so the
    hypothetical values never leak into the next question.
    """
    message = (state.get("message") or "").strip()
    update: dict[str, Any] = {}

    restore = state.get("restore_profile")
    if restore is not None:
        update["profile"] = {REPLACE: restore}
        update["restore_profile"] = None

    if not message:
        return {**update, "intent": Intent.new_info.value, "asks": "none"}

    intent, constraint, asks = extract.classify_intent(message)
    update.update({"intent": intent.value, "asks": asks,
                   "trace": {"intent": intent.value, "asks": asks}})
    if constraint:
        update["user_constraints"] = [constraint]

    if intent == Intent.what_if:
        # Snapshot the standing answer and the full baseline profile. The turn runs
        # on baseline + the hypothetical change; the baseline is restored next turn.
        baseline_profile = restore if restore is not None else dict(state.get("profile", {}))
        update["what_if_baseline"] = {
            "flags": list(state.get("flags", [])),
            "practices": [
                {"practice_id": candidate["practice_id"],
                 "name": candidate["name"],
                 "suitability": candidate["suitability"],
                 "downgraded": candidate["downgraded"]}
                for candidate in state.get("candidates", [])[:6]
            ],
            "profile": {name: field.get("value") for name, field in baseline_profile.items()},
            "constraints": list(state.get("user_constraints", [])),
        }
        update["restore_profile"] = baseline_profile
    return update


def slot_check(state: AgentState) -> dict[str, Any]:
    """Ask at most one question, chosen by which answer could change the advice."""
    values = profile_values(state.get("profile", {}))
    constraints = list(state.get("user_constraints", []))

    known = [slot for slot in CRITICAL_SLOTS if values.get(slot) is not None]
    if len(known) >= 2:
        return {"question": None}

    ranked = voi.rank_slots(values, constraints)
    if not ranked:
        return {"question": None}

    best = ranked[0]
    return {
        "question": render_mod.render_question(best["question"], best["reason"]),
        "trace": {"slot_check": {"asked": best["slot"], "ranking": ranked[:4]}},
    }


# ── diagnosis and reasoning ────────────────────────────────────────────
def diagnose_node(state: AgentState) -> dict[str, Any]:
    """Thresholds and compound rules. No LLM has any say here."""
    values = profile_values(state.get("profile", {}))
    flags, fired = diagnose.diagnose(values)
    names = diagnose.flag_names(flags)
    patterns = diagnose.detect_patterns(names)

    problem = [name for name in names if name in causal.PROBLEM_FLAGS]
    return {
        "flags": names,
        "problem_flags": problem,
        "flag_details": [flag.model_dump() for flag in flags],
        "fired_rules": fired,
        "patterns": [pattern.model_dump() for pattern in patterns],
        "trace": {"diagnosis": {
            "flags": names,
            "problem_flags": problem,
            "fired_rules": fired,
            "patterns": [pattern.model_dump() for pattern in patterns],
        }},
    }


def root_cause(state: AgentState) -> dict[str, Any]:
    """Trace each flagged metric back to the drivers present on this site."""
    values = profile_values(state.get("profile", {}))
    graph = graph_mod.load_graph()
    targets = list(causal.flag_targets(state.get("flags", [])))
    causes = causal.root_causes(graph, targets, values)[:8]
    return {
        "root_causes": causes,
        "trace": {"root_causes": [
            {"id": cause["id"], "path": cause["path"], "observed": cause["observed"]}
            for cause in causes
        ]},
    }


def leverage_node(state: AgentState) -> dict[str, Any]:
    """Rank nodes by how much of the diagnosed problem they reach."""
    values = profile_values(state.get("profile", {}))
    graph = graph_mod.load_graph()
    ranking = causal.leverage(graph, state.get("flags", []), values)
    return {"leverage": ranking, "trace": {"leverage": ranking[:5]}}


def candidates_node(state: AgentState) -> dict[str, Any]:
    """Score every practice that addresses a flag, and say why others were dropped."""
    values = profile_values(state.get("profile", {}))
    constraints = list(state.get("user_constraints", []))
    ranked, excluded = combine.candidates(state.get("flags", []), values, constraints)

    rejected = set(state.get("rejected_practices", []))
    if rejected:
        excluded = excluded + [
            {"practice_id": candidate["practice_id"], "name": candidate["name"],
             "reason": "you ruled this out earlier"}
            for candidate in ranked if candidate["practice_id"] in rejected
        ]
        ranked = [c for c in ranked if c["practice_id"] not in rejected]

    return {
        "candidates": ranked,
        "excluded": excluded,
        "trace": {
            "candidates": [
                {"practice_id": candidate["practice_id"],
                 "name": candidate["name"],
                 "suitability": candidate["suitability"],
                 "coverage": candidate["coverage"],
                 "benefit": candidate["benefit"],
                 "penalty": candidate["penalty"],
                 "net_effects": candidate["net_effects"],
                 "risks_applied": candidate["risks_applied"],
                 "downgraded": candidate["downgraded"],
                 "paths": [
                     {"path": path["path"], "weight": path["weight"],
                      "condition": path["condition"]}
                     for path in candidate["paths"][:4]
                 ]}
                for candidate in ranked[:6]
            ],
            "excluded": excluded,
        },
    }


def combine_node(state: AgentState) -> dict[str, Any]:
    """Pair the strongest candidates so one can offset the other's risk."""
    combos = combine.pair_candidates(state.get("candidates", []), state.get("flags", []))
    return {"combos": combos, "trace": {"combinations": combos[:3]}}


def sequence_node(state: AgentState) -> dict[str, Any]:
    """Order the shortlist by prerequisites and map it onto time horizons."""
    shortlist = [candidate["practice_id"] for candidate in state.get("candidates", [])[:4]]
    plan = sequence.sequence(shortlist)
    return {"plan": plan, "trace": {"sequence": plan}}


# ── retrieval ──────────────────────────────────────────────────────────
def plan_queries(state: AgentState) -> dict[str, Any]:
    """Build support, risk and path queries. The user's words are never the query."""
    values = profile_values(state.get("profile", {}))
    practices = [candidate["practice_id"] for candidate in state.get("candidates", [])[:4]]
    queries = planner.plan(
        flags=state.get("flags", []),
        practices=practices,
        paths=state.get("root_causes", []),
        site_zone=values.get("climate_zone"),
    )
    return {"queries": [query.model_dump() for query in queries]}


def retrieve(state: AgentState) -> dict[str, Any]:
    """Run the plan, score the hits and assemble a labelled evidence block."""
    from core.schemas import PlannedQuery

    values = profile_values(state.get("profile", {}))
    started = time.perf_counter()
    queries = [PlannedQuery.model_validate(query) for query in state.get("queries", [])]

    items, traces, filter_level = planner.run_plan(queries, values.get("climate_zone"))
    elapsed = int((time.perf_counter() - started) * 1000)

    return {
        "evidence": [item.model_dump() for item in items],
        "filter_level": filter_level,
        "trace": {
            "queries": traces,
            "evidence": [
                {"label": item.label, "chunk_id": item.chunk_id,
                 "doc_title": item.doc_title, "page": item.page_start,
                 "score": item.score, "evidence_level": item.evidence_level}
                for item in items
            ],
            "timing_ms": {**(state.get("trace", {}).get("timing_ms") or {}),
                          "retrieve": elapsed},
        },
    }


# ── adjudication and verification ──────────────────────────────────────
def _verify_context(state: AgentState) -> checks.VerifyContext:
    items = [EvidenceItem.model_validate(item) for item in state.get("evidence", [])]
    candidates = {c["practice_id"]: c for c in state.get("candidates", [])}
    risk_practices = {
        query.get("practice_id")
        for query in state.get("trace", {}).get("queries", [])
        if query.get("purpose") == "risk" and query.get("practice_id")
    }
    return checks.VerifyContext(
        evidence={item.label: item for item in items},
        path_ids=adjudicate_mod.path_ids(state.get("candidates", []),
                                         state.get("root_causes", [])),
        profile=profile_values(state.get("profile", {})),
        risk_practices=risk_practices,
        candidates=candidates,
        excluded_practices={item["practice_id"] for item in state.get("excluded", [])},
        problem_flags=set(state.get("problem_flags", [])),
    )


def adjudicate(state: AgentState) -> dict[str, Any]:
    """The one reasoning call. Retries carry targeted feedback about what failed."""
    items = [EvidenceItem.model_validate(item) for item in state.get("evidence", [])]
    dossier = adjudicate_mod.build_dossier(
        profile=profile_values(state.get("profile", {})),
        flags=state.get("flags", []),
        patterns=state.get("patterns", []),
        root_causes=state.get("root_causes", []),
        leverage=state.get("leverage", []),
        candidates=state.get("candidates", []),
        combos=state.get("combos", []),
        excluded=state.get("excluded", []),
        plan=state.get("plan", []),
        evidence=items,
        asks=state.get("asks") or "none",
        problem_flags=state.get("problem_flags", []),
    )

    previous = state.get("adjudication")
    started = time.perf_counter()
    try:
        result = adjudicate_mod.adjudicate(
            dossier, items,
            feedback=state.get("verify_feedback"),
            previous=Adjudication.model_validate(previous) if previous else None,
        )
    except RuntimeError as exc:
        return {
            "adjudication": None,
            "warnings": [f"The reasoning model could not be reached: {exc}"],
        }
    elapsed = int((time.perf_counter() - started) * 1000)

    return {
        "adjudication": result.model_dump(),
        "verify_attempts": int(state.get("verify_attempts", 0)) + 1,
        "trace": {"timing_ms": {**(state.get("trace", {}).get("timing_ms") or {}),
                                "adjudicate": elapsed}},
    }


def verify(state: AgentState) -> dict[str, Any]:
    """Run V1-V10. Failures either trigger a retry or, once spent, a degraded answer."""
    raw = state.get("adjudication")
    if raw is None:
        return {"verification": {"attempts": int(state.get("verify_attempts", 0)),
                                 "checks": {}, "passed": False}}

    adjudication = checks.normalise(Adjudication.model_validate(raw))
    ctx = _verify_context(state)
    failures, warnings, status, groups = checks.run_checks(adjudication, ctx)
    attempts = int(state.get("verify_attempts", 0))

    if not failures:
        adjudication, unsupported = _drop_unsupported_steps(adjudication, ctx)
        return {
            "adjudication": adjudication.model_dump(),
            "verification": {"attempts": attempts, "checks": status, "passed": True,
                             "warnings": warnings + unsupported, "stripped_numbers": 0,
                             "dropped_unsupported_steps": len(unsupported)},
            "verify_feedback": None,
            "trace": {"verification": {"attempts": attempts, "checks": status,
                                       "warnings": warnings,
                                       "unsupported_steps": unsupported}},
        }

    # V7 is a judgement call, not a hard fact check, so it earns one retry and then
    # softens instead of spending the whole retry budget on wording.
    hard = [
        failure for name, found in groups.items() if name != "V7" for failure in found
    ]
    soft = groups.get("V7", [])
    retry = (hard and attempts <= config.MAX_VERIFY_RETRIES) or (soft and attempts < 2)

    if retry:
        # Deduplicate and cap: forty near-identical lines bury the instruction.
        unique = list(dict.fromkeys(hard or soft))
        shown = unique[:12]
        if len(unique) > len(shown):
            shown.append(f"...and {len(unique) - len(shown)} more of the same kind")
        return {
            "verify_feedback": "\n".join(f"- {failure}" for failure in shown),
            "verification": {"attempts": attempts, "checks": status, "passed": False,
                             "warnings": warnings},
        }

    degraded, counts = checks.degrade(adjudication, ctx)
    degraded, unsupported = _drop_unsupported_steps(degraded, ctx)
    counts["dropped_unsupported_steps"] = len(unsupported)
    _, post_warnings, post_status, _ = checks.run_checks(degraded, ctx,
                                                         judge_mechanisms=False)
    update: dict[str, Any] = {
        "adjudication": degraded.model_dump(),
        "verify_feedback": None,
        "verification": {
            "attempts": attempts, "checks": {**status, **post_status}, "passed": False,
            "degraded": True, "warnings": warnings + post_warnings, **counts,
        },
        "trace": {"verification": {
            "attempts": attempts, "checks": status, "degraded": True,
            "unresolved": failures, **counts,
        }},
    }
    if soft:
        update["warnings"] = [
            "The cited passages support part but not all of the mechanism wording for "
            "at least one recommendation; the trace lists which."
        ]
    if counts["stripped_numbers"]:
        update.setdefault("warnings", []).append(
            f"{counts['stripped_numbers']} figure(s) had no matching claim in the cited "
            f"chunk and were removed rather than shown unverified."
        )
    return update


def _drop_unsupported_steps(adjudication: Adjudication, ctx) -> tuple[Adjudication, list[str]]:
    """Judge evidence-only causal steps and remove the ones the cited text does not state."""
    from verification import judge as judge_mod

    indices, reasons = judge_mod.unsupported_steps(adjudication, ctx)
    if not indices:
        return adjudication, []
    return checks.drop_steps(adjudication, indices), reasons


def should_retry(state: AgentState) -> str:
    """Conditional edge: back to the reasoning call, or on to the answer."""
    if state.get("verify_feedback"):
        return "adjudicate"
    return "render"


# ── output ─────────────────────────────────────────────────────────────
def render(state: AgentState) -> dict[str, Any]:
    """Assemble the final answer and the confidence breakdown."""
    from core.schemas import Audience, Flag, Pattern

    profile = to_profile(state.get("profile", {}))
    items = [EvidenceItem.model_validate(item) for item in state.get("evidence", [])]
    raw = state.get("adjudication")
    adjudication = Adjudication.model_validate(raw) if raw else Adjudication()

    cited_labels = {
        label
        for recommendation in adjudication.recommendations
        for label in (list(recommendation.mechanism_sources)
                      + [estimate.source for estimate in recommendation.estimates]
                      + [risk.source for risk in recommendation.risks])
    }
    cited = [item for item in items if item.label in cited_labels] or items
    supporting, contradicting = confidence_mod.count_support(cited)

    verification = state.get("verification", {})
    penalty = 0.85 if verification.get("degraded") else 1.0
    report = confidence_mod.compute(
        items=cited,
        profile=profile,
        site_zone=profile.value("climate_zone"),
        filter_level=int(state.get("filter_level", 0)),
        supporting=supporting,
        contradicting=contradicting,
        extra_penalty=penalty,
    )

    comparison = None
    baseline = state.get("what_if_baseline")
    if baseline:
        comparison = _what_if_comparison(baseline, state)

    audience = state.get("audience")
    answer = render_mod.render(
        profile=profile,
        flags=[Flag.model_validate(flag) for flag in state.get("flag_details", [])],
        fired_rules=state.get("fired_rules", []),
        patterns=[Pattern.model_validate(p) for p in state.get("patterns", [])],
        adjudication=adjudication,
        candidates=state.get("candidates", []),
        excluded=state.get("excluded", []),
        plan=state.get("plan", []),
        evidence=items,
        confidence=report,
        audience=Audience(audience) if audience else None,
        root_causes=state.get("root_causes", []),
        asks=state.get("asks") or "none",
        problem_flags=state.get("problem_flags", []),
        comparison=comparison,
    )

    warnings = state.get("warnings", [])
    if warnings:
        answer += "\nNOTES\n" + "\n".join(f"    {note}" for note in warnings) + "\n"

    trace: dict[str, Any] = {
        "profile": confidence_mod.summarise(profile),
        "confidence": report.model_dump(),
    }
    if comparison:
        trace["what_if"] = {
            "baseline_state": baseline.get("profile", {}),
            "hypothetical_state": profile_values(state.get("profile", {})),
            "change": comparison.get("change", {}),
            "moved": comparison.get("moved", []),
            "unchanged": comparison.get("unchanged", []),
            "constraints": comparison.get("constraints", []),
        }
    return {"answer": answer, "confidence": report.model_dump(), "trace": trace}


def _what_if_comparison(baseline: dict, state: AgentState) -> dict[str, Any]:
    """Baseline versus hypothetical as data: what moved, what stayed, what still binds."""
    before = {item["practice_id"]: item for item in baseline.get("practices", [])}
    after = {c["practice_id"]: c for c in state.get("candidates", [])[:6]}

    old_values = baseline.get("profile", {})
    new_values = profile_values(state.get("profile", {}))
    change = {
        name: (old_values.get(name), value)
        for name, value in new_values.items()
        if old_values.get(name) != value and name not in ("lat", "lon")
    }
    unchanged = sorted(
        name for name, value in new_values.items()
        if old_values.get(name) == value and name not in ("lat", "lon", "place_name")
    )

    moved: list[str] = []
    old_flags, new_flags = set(baseline.get("flags", [])), set(state.get("flags", []))
    if old_flags - new_flags:
        moved.append(f"resolved: {', '.join(sorted(old_flags - new_flags))}")
    if new_flags - old_flags:
        moved.append(f"newly flagged: {', '.join(sorted(new_flags - old_flags))}")

    for practice_id, candidate in after.items():
        previous = before.get(practice_id)
        if previous is None:
            moved.append(f"{candidate['name']}: now in play at {candidate['suitability']:.2f}")
        elif previous["downgraded"] and not candidate["downgraded"]:
            moved.append(
                f"{candidate['name']}: no longer downgraded "
                f"({previous['suitability']:.2f} -> {candidate['suitability']:.2f}), "
                f"the condition behind its risk no longer holds"
            )
        elif not previous["downgraded"] and candidate["downgraded"]:
            risk = candidate["risks_applied"][0]["risk"] if candidate["risks_applied"] else ""
            moved.append(
                f"{candidate['name']}: now downgraded "
                f"({previous['suitability']:.2f} -> {candidate['suitability']:.2f}) - {risk}"
            )
        elif abs(previous["suitability"] - candidate["suitability"]) >= 0.05:
            moved.append(
                f"{candidate['name']}: {previous['suitability']:.2f} -> {candidate['suitability']:.2f}"
            )
    for practice_id, previous in before.items():
        if practice_id not in after:
            moved.append(f"{previous['name']}: no longer applies")

    return {
        "change": change,
        "moved": moved,
        "unchanged": unchanged,
        "constraints": list(baseline.get("constraints", [])),
    }


def ask(state: AgentState) -> dict[str, Any]:
    """Terminal node for a turn that asks a question instead of answering."""
    return {"answer": state.get("question") or "Could you tell me more about the site?"}


def concept(state: AgentState) -> dict[str, Any]:
    """Answer a general question about a practice from the corpus, with no site reasoning.

    The query is built from practice and metric words found in the message, never
    from the raw message, and the passages are quoted rather than paraphrased, so
    a concept answer carries the same citations as a recommendation.
    """
    message = (state.get("message") or "").lower()
    cards = loader.practice_cards()

    matched = [
        practice_id for practice_id in cards
        if planner.practice_text(practice_id) in message
        or practice_id.replace("_", " ") in message
    ]
    metrics = [
        metric for metric in ("soil organic carbon", "soil moisture", "erosion",
                              "pollinators", "species richness", "habitat")
        if metric in message
    ]

    if matched:
        query = f"{planner.practice_text(matched[0])} {' '.join(metrics)} mechanism effects"
        practice = matched[0]
    elif metrics:
        query = f"{' '.join(metrics)} land management mechanism"
        practice = None
    else:
        return {"answer": (
            "Tell me which practice or which measure you are asking about and I will "
            "quote what the sources say about it."
        )}

    hits, level = search.search_with_fallback(
        query.strip(), None, practice, allow_context_only=True
    )
    items = assemble_mod.assemble(rerank.score_chunks(hits, None, level), limit=3)
    if not items:
        return {"answer": "I could not find anything on that in the indexed sources."}

    lines = [f"What the sources say about {planner.practice_text(practice) if practice else query.strip()}:", ""]
    for item in items:
        page = f", p{item.page_start}" if item.page_start is not None else ""
        lines.append(f"    {item.text.strip()}")
        lines.append(f"        — {item.doc_title}{page} [{item.label}]")
        lines.append("")
    lines.append("This is general evidence, not advice for your site. "
                 "Tell me about the land and I will work through what applies there.")

    return {
        "answer": "\n".join(lines),
        "evidence": [item.model_dump() for item in items],
        "trace": {"concept": {"query": query.strip(), "filter_level": level,
                              "kept": [item.chunk_id for item in items]}},
    }


def out_of_scope(state: AgentState) -> dict[str, Any]:
    """Polite redirect, so an unrelated question never reaches the pipeline."""
    return {"answer": (
        "That is outside what I can help with. I work on soil, land use, climate "
        "and biodiversity for a specific site. Tell me about your land and I will "
        "look at what the evidence supports."
    )}
