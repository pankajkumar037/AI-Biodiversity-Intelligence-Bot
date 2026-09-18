"""V1-V10. Pure Python except V7, which delegates to the judge."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import config
from core.schemas import Adjudication, EvidenceItem, Recommendation

UNIT_ALIASES = {"%": "percent", "pct": "percent", "percentage": "percent"}
GENERIC_ZONES = {"global", "unstated"}


@dataclass
class VerifyContext:
    """Everything the checks need to judge an answer, gathered by the caller."""

    evidence: dict[str, EvidenceItem]
    path_ids: set[str] = field(default_factory=set)
    profile: dict[str, Any] = field(default_factory=dict)
    risk_practices: set[str] = field(default_factory=set)
    candidates: dict[str, dict] = field(default_factory=dict)
    excluded_practices: set[str] = field(default_factory=set)
    problem_flags: set[str] = field(default_factory=set)


def _unit(value: str | None) -> str:
    """'Mg/ha/yr', 'Mg_ha_yr' and 'mg ha yr' are one unit; '%' is 'percent'."""
    text = (value or "").strip().lower()
    text = UNIT_ALIASES.get(text, text)
    return re.sub(r"[^a-z0-9]", "", text)


_LABEL = re.compile(r"^[\[\s(]*([SP]\d+)[\]\s)]*$", re.IGNORECASE)


def _clean_label(raw: str) -> str:
    """'[s3]' -> 'S3'. Anything else is returned stripped, to fail V2 honestly."""
    match = _LABEL.match(raw or "")
    return match.group(1).upper() if match else (raw or "").strip()


def _split_labels(values: list[str]) -> list[str]:
    """Models sometimes pack 'S1, S2' into one string."""
    out: list[str] = []
    for value in values:
        for part in re.split(r"[,;/]+", value or ""):
            cleaned = _clean_label(part)
            if cleaned and cleaned not in out:
                out.append(cleaned)
    return out


def normalise(adjudication: Adjudication) -> Adjudication:
    """Fix right-id-wrong-field mistakes before checking. Never invents a citation.

    A reasoning step that lists path ids under sources gets them moved to path_id.
    Labels are unbracketed and upper-cased. Nothing else changes: a path id left in
    a mechanism source is still a V2 failure, because it is not evidence.
    """
    steps = []
    for step in adjudication.reasoning_chain:
        sources = _split_labels(step.sources)
        path_ids = [s for s in sources if s.startswith("P")]
        evidence = [s for s in sources if s.startswith("S")]
        path_id = _clean_label(step.path_id) if step.path_id else None
        if path_id is None and path_ids:
            path_id = path_ids[0]
        steps.append(step.model_copy(update={"sources": evidence, "path_id": path_id}))

    recommendations = []
    for recommendation in adjudication.recommendations:
        cited = _split_labels(recommendation.mechanism_sources)
        paths = _split_labels(recommendation.mechanism_paths)
        paths += [label for label in cited if label.startswith("P") and label not in paths]
        recommendations.append(recommendation.model_copy(update={
            "mechanism_sources": [label for label in cited if label.startswith("S")],
            "mechanism_paths": [label for label in paths if label.startswith("P")],
            "estimates": [e.model_copy(update={"source": _clean_label(e.source)})
                          for e in recommendation.estimates],
            "risks": [r.model_copy(update={"source": _clean_label(r.source)})
                      for r in recommendation.risks],
        }))
    return adjudication.model_copy(update={
        "reasoning_chain": steps, "recommendations": recommendations,
    })


def _labels(recommendation: Recommendation) -> list[str]:
    labels = list(recommendation.mechanism_sources)
    labels += [estimate.source for estimate in recommendation.estimates]
    labels += [risk.source for risk in recommendation.risks]
    return [label for label in labels if label]


def _claim_supports(claim: dict, metric: str, unit: str,
                    value: float | None, low: float | None, high: float | None) -> bool:
    """Does one extracted claim back this number, on the same metric and unit?"""
    if str(claim.get("metric", "")).lower() != metric.lower():
        return False
    if _unit(claim.get("unit")) != unit:
        return False

    claim_value = claim.get("value")
    claim_low = claim.get("value_low")
    claim_high = claim.get("value_high")

    if value is not None:
        if claim_value is not None and abs(float(claim_value) - value) <= 0.005 * max(1.0, abs(value)):
            return True
        if claim_low is not None and claim_high is not None:
            if float(claim_low) <= value <= float(claim_high):
                return True
        return False

    if low is not None and high is not None:
        if claim_value is not None and low <= float(claim_value) <= high:
            return True
        if claim_low is not None and claim_high is not None:
            return abs(float(claim_low) - low) < 1e-6 and abs(float(claim_high) - high) < 1e-6
    return False


def check_v1_structure(adjudication: Adjudication) -> list[str]:
    """Schema already parsed; here we reject answers that are structurally empty."""
    failures = []
    if not adjudication.recommendations:
        failures.append("V1: no recommendations were produced")
    for recommendation in adjudication.recommendations:
        if not recommendation.action.strip():
            failures.append(f"V1: {recommendation.practice_id.value} has an empty action")
        if not recommendation.mechanism.strip():
            failures.append(f"V1: {recommendation.practice_id.value} has an empty mechanism")
    return failures


def check_v2_sources_exist(adjudication: Adjudication, ctx: VerifyContext) -> list[str]:
    failures = []
    for recommendation in adjudication.recommendations:
        for label in _labels(recommendation):
            if label not in ctx.evidence:
                failures.append(
                    f"V2: {recommendation.practice_id.value} cites {label}, "
                    f"which is not in the evidence block"
                )
    return failures


def check_v3_numbers_grounded(adjudication: Adjudication, ctx: VerifyContext) -> list[str]:
    failures = []
    for recommendation in adjudication.recommendations:
        for estimate in recommendation.estimates:
            if estimate.value is None and estimate.low is None and estimate.high is None:
                continue
            item = ctx.evidence.get(estimate.source)
            if item is None:
                failures.append(f"V3: {estimate.source} is not in the evidence block")
                continue
            supported = any(
                _claim_supports(claim, estimate.metric.value, _unit(estimate.unit),
                                estimate.value, estimate.low, estimate.high)
                for claim in item.claims
            )
            if not supported:
                failures.append(
                    f"V3: {estimate.source} has no claim for "
                    f"{estimate.metric.value} = {estimate.value or [estimate.low, estimate.high]} "
                    f"{estimate.unit or ''}. Use a claim from that chunk or drop the number."
                )
    return failures


def check_v4_direction(adjudication: Adjudication, ctx: VerifyContext) -> list[str]:
    failures = []
    for recommendation in adjudication.recommendations:
        candidate = ctx.candidates.get(recommendation.practice_id.value, {})
        effects = candidate.get("net_effects", {})
        for estimate in recommendation.estimates:
            metric = estimate.metric.value
            item = ctx.evidence.get(estimate.source)
            claim_directions = {
                str(claim.get("direction", "")).lower()
                for claim in (item.claims if item else [])
                if str(claim.get("metric", "")).lower() == metric.lower()
            }
            if estimate.direction in claim_directions:
                continue
            effect = effects.get(metric)
            if effect is not None:
                if (effect > 0 and estimate.direction == "increase") or \
                   (effect < 0 and estimate.direction == "decrease"):
                    continue
            if estimate.direction == "mixed" and claim_directions:
                continue
            failures.append(
                f"V4: {recommendation.practice_id.value} says {metric} will "
                f"{estimate.direction}, which neither {estimate.source} nor the graph supports"
            )
    return failures


def check_v5_climate_fit(adjudication: Adjudication, ctx: VerifyContext) -> list[str]:
    """A warning, not a failure: it lowers confidence rather than forcing a retry."""
    site_zone = ctx.profile.get("climate_zone")
    if not site_zone:
        return []
    warnings = []
    for recommendation in adjudication.recommendations:
        for label in _labels(recommendation):
            item = ctx.evidence.get(label)
            if item is None:
                continue
            zones = set(item.climate_zones)
            if zones and site_zone not in zones and not (zones & GENERIC_ZONES):
                warnings.append(
                    f"V5: {label} is tagged {sorted(zones)}, not {site_zone}"
                )
    return warnings


def check_v6_mechanism_and_risk_query(adjudication: Adjudication,
                                      ctx: VerifyContext) -> list[str]:
    failures = []
    for recommendation in adjudication.recommendations:
        practice = recommendation.practice_id.value
        if not recommendation.mechanism_sources:
            failures.append(f"V6: {practice} states a mechanism with no source")
        if ctx.risk_practices and practice not in ctx.risk_practices:
            failures.append(f"V6: no risk query was run for {practice}")
    return failures


def check_v8_generic_language(adjudication: Adjudication) -> list[str]:
    failures = []
    for recommendation in adjudication.recommendations:
        text = f"{recommendation.action} {recommendation.mechanism}".lower()
        for phrase in config.GENERIC_PHRASES:
            if phrase in text:
                failures.append(
                    f"V8: {recommendation.practice_id.value} uses the generic phrase "
                    f"'{phrase}'. Say what changes, for which metric, on this site."
                )
    return failures


def check_mechanism_paths_exist(adjudication: Adjudication, ctx: VerifyContext) -> list[str]:
    failures = []
    for recommendation in adjudication.recommendations:
        for path_id in recommendation.mechanism_paths:
            if path_id not in ctx.path_ids:
                failures.append(
                    f"V9: {recommendation.practice_id.value} cites path {path_id}, "
                    f"which is not in the dossier"
                )
    return failures


def check_v9_steps_anchored(adjudication: Adjudication, ctx: VerifyContext) -> list[str]:
    failures = []
    for step in adjudication.reasoning_chain:
        anchored = step.path_id in ctx.path_ids if step.path_id else False
        anchored = anchored or any(source in ctx.evidence for source in step.sources)
        if not anchored:
            failures.append(f"V9: reasoning step '{step.claim[:60]}' cites nothing that exists")
    return failures


def check_v10_multi_variable(adjudication: Adjudication) -> list[str]:
    failures = []
    for recommendation in adjudication.recommendations:
        variables = {metric.value for metric in recommendation.variables_considered}
        if len(variables) < config.MIN_VARIABLES_PER_RECOMMENDATION:
            failures.append(
                f"V10: {recommendation.practice_id.value} connects {len(variables)} "
                f"variables, needs at least {config.MIN_VARIABLES_PER_RECOMMENDATION}"
            )
    return failures


def check_practices_allowed(adjudication: Adjudication, ctx: VerifyContext) -> list[str]:
    failures = []
    for recommendation in adjudication.recommendations:
        practice = recommendation.practice_id.value
        if practice in ctx.excluded_practices:
            failures.append(f"V1: {practice} was excluded for this site and cannot be recommended")
        elif ctx.candidates and practice not in ctx.candidates:
            failures.append(f"V1: {practice} is not in the candidate list")
    return failures


def check_addresses_problem(adjudication: Adjudication, ctx: VerifyContext) -> list[str]:
    """A recommendation must address something the user said is wrong, when they said anything."""
    if not ctx.problem_flags:
        return []
    failures = []
    for recommendation in adjudication.recommendations:
        candidate = ctx.candidates.get(recommendation.practice_id.value, {})
        if not set(candidate.get("addresses", [])) & ctx.problem_flags:
            failures.append(
                f"V1: {recommendation.practice_id.value} addresses none of the stated "
                f"problem ({', '.join(sorted(ctx.problem_flags))}). Recommend something "
                f"that does, or leave it out."
            )
    return failures


def run_checks(adjudication: Adjudication, ctx: VerifyContext, judge_mechanisms: bool = True
               ) -> tuple[list[str], list[str], dict[str, str], dict[str, list[str]]]:
    """Run every check. Returns (failures, warnings, per-check status, failures by check)."""
    status: dict[str, str] = {}
    failures: list[str] = []

    groups = {
        "V1": check_v1_structure(adjudication) + check_practices_allowed(adjudication, ctx)
              + check_addresses_problem(adjudication, ctx),
        "V2": check_v2_sources_exist(adjudication, ctx),
        "V3": check_v3_numbers_grounded(adjudication, ctx),
        "V4": check_v4_direction(adjudication, ctx),
        "V6": check_v6_mechanism_and_risk_query(adjudication, ctx),
        "V8": check_v8_generic_language(adjudication),
        "V9": check_v9_steps_anchored(adjudication, ctx)
              + check_mechanism_paths_exist(adjudication, ctx),
        "V10": check_v10_multi_variable(adjudication),
    }

    if judge_mechanisms:
        from verification import judge as judge_mod
        groups["V7"] = judge_mod.check_v7_mechanism_supported(adjudication, ctx)

    for name, found in groups.items():
        status[name] = "pass" if not found else "fail"
        failures.extend(found)

    warnings = check_v5_climate_fit(adjudication, ctx)
    status["V5"] = "pass" if not warnings else "warn"
    return failures, warnings, status, groups


def drop_steps(adjudication: Adjudication, indices: set[int]) -> Adjudication:
    """Remove reasoning steps by position. Used for steps the judge found unsupported."""
    kept = [step for i, step in enumerate(adjudication.reasoning_chain) if i not in indices]
    return adjudication.model_copy(update={"reasoning_chain": kept})


def degrade(adjudication: Adjudication,
            ctx: VerifyContext) -> tuple[Adjudication, dict[str, int]]:
    """Last resort: strip what cannot be verified rather than blocking the answer."""
    counts = {"stripped_numbers": 0, "dropped_steps": 0, "dropped_recommendations": 0,
              "dropped_estimates": 0}
    kept_recommendations = []

    for recommendation in adjudication.recommendations:
        if recommendation.practice_id.value in ctx.excluded_practices:
            counts["dropped_recommendations"] += 1
            continue
        if ctx.problem_flags:
            candidate = ctx.candidates.get(recommendation.practice_id.value, {})
            if not set(candidate.get("addresses", [])) & ctx.problem_flags:
                counts["dropped_recommendations"] += 1
                continue
        if len({metric.value for metric in recommendation.variables_considered}) < \
                config.MIN_VARIABLES_PER_RECOMMENDATION:
            counts["dropped_recommendations"] += 1
            continue

        estimates = []
        for estimate in recommendation.estimates:
            item = ctx.evidence.get(estimate.source)
            if item is None:
                # An estimate whose source is not a real evidence label cites nothing
                # a reader could check, so it is dropped rather than shown.
                counts["dropped_estimates"] += 1
                continue
            has_number = not (estimate.value is None and estimate.low is None
                              and estimate.high is None)
            supported = item is not None and any(
                _claim_supports(claim, estimate.metric.value, _unit(estimate.unit),
                                estimate.value, estimate.low, estimate.high)
                for claim in item.claims
            )
            if has_number and not supported:
                estimate = estimate.model_copy(
                    update={"value": None, "low": None, "high": None, "unit": None}
                )
                counts["stripped_numbers"] += 1
            estimates.append(estimate)

        recommendation = recommendation.model_copy(update={
            "estimates": estimates,
            "mechanism_sources": [
                label for label in recommendation.mechanism_sources if label in ctx.evidence
            ],
            "risks": [risk for risk in recommendation.risks if risk.source in ctx.evidence],
        })
        kept_recommendations.append(recommendation)

    kept_steps = []
    for step in adjudication.reasoning_chain:
        anchored = (step.path_id in ctx.path_ids if step.path_id else False) or \
            any(source in ctx.evidence for source in step.sources)
        if anchored:
            kept_steps.append(step)
        else:
            counts["dropped_steps"] += 1

    return adjudication.model_copy(update={
        "recommendations": kept_recommendations,
        "reasoning_chain": kept_steps,
    }), counts
