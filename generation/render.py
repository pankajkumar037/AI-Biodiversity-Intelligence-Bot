"""Verified output to the final answer text. Pure Python: no model touches this.

Order is fixed: site, diagnosis, causes, what changed, evidence, recommendation,
trade-offs, exclusions, confidence. The question the user asked decides the lead.
"""
from __future__ import annotations

from typing import Any

from core.schemas import (
    Adjudication,
    Audience,
    ConfidenceReport,
    EvidenceItem,
    Recommendation,
    SiteProfile,
)

ARROW = {"increase": "up", "decrease": "down", "no_change": "unchanged", "mixed": "mixed"}
FLAG_TEXT = {
    "low_SOC": "soil organic carbon is low",
    "water_limited": "water is limiting",
    "monoculture": "a single-crop system",
    "biodiversity_decline": "biodiversity is declining",
    "pollinator_decline": "pollinators are declining",
    "high_pesticide_use": "pesticide use is high",
    "residue_burning": "residues are burned",
    "recent_clearing": "land was recently cleared",
    "pollution_exposure": "a pollution source is nearby",
    "erosion_observed": "erosion is visible",
    "low_natural_cover": "little natural cover nearby",
    "low_habitat_diversity": "habitat diversity is low",
    "bare_fallow": "bare fallow",
    "acidic_soil": "the soil is acidic",
    "alkaline_soil": "the soil is alkaline",
    "sodic_risk": "sodicity risk",
    "dry_subhumid": "dry sub-humid",
}


def _site_line(profile: SiteProfile) -> str:
    parts = []
    for name, field in profile.fields.items():
        parts.append(f"{name.replace('_', ' ')} {field.value} ({field.source.value})")
    return " | ".join(parts) if parts else "no site values supplied"


def _diagnosis_line(flags: list, patterns: list) -> str:
    names = " | ".join(f"{flag.flag} [{flag.rule_id}]" for flag in flags) or "no flags raised"
    if patterns:
        names += " | patterns: " + ", ".join(
            f"{pattern.name} [{pattern.rule_id}]" for pattern in patterns
        )
    return names


def _citation(item: EvidenceItem) -> str:
    page = f", p{item.page_start}" if item.page_start is not None else ""
    return f"{item.doc_title}{page}"


def _estimate_line(recommendation: Recommendation) -> str | None:
    lines = []
    for estimate in recommendation.estimates:
        metric = estimate.metric.value
        direction = ARROW.get(estimate.direction, estimate.direction)
        if estimate.value is not None:
            number = f"{estimate.value:g}{' ' + estimate.unit if estimate.unit else ''}"
        elif estimate.low is not None and estimate.high is not None:
            unit = f" {estimate.unit}" if estimate.unit else ""
            number = f"{estimate.low:g}-{estimate.high:g}{unit}"
        else:
            number = "no quantified evidence found"
        timeframe = (
            f" over {estimate.timeframe_years:g} years" if estimate.timeframe_years else ""
        )
        lines.append(f"{metric} {direction}: {number}{timeframe} [{estimate.source}]")
    return "; ".join(lines) if lines else None


def _cited_labels(adjudication: Adjudication) -> set[str]:
    labels: set[str] = set()
    for recommendation in adjudication.recommendations:
        labels.update(recommendation.mechanism_sources)
        labels.update(estimate.source for estimate in recommendation.estimates)
        labels.update(risk.source for risk in recommendation.risks)
    for step in adjudication.reasoning_chain:
        labels.update(step.sources)
    return labels


# ── sections ───────────────────────────────────────────────────────────
def causes_section(flags: list, patterns: list, root_causes: list[dict],
                   adjudication: Adjudication, problem_flags: list[str]) -> list[str]:
    """Why the site looks the way it does: observed drivers first, then the model's causal steps."""
    lines = ["CAUSES"]
    # Two edges can share a path (different documents, same link); show it once.
    seen: set[tuple] = set()
    unique = []
    for cause in root_causes:
        key = tuple(cause["path"])
        if key not in seen:
            seen.add(key)
            unique.append(cause)
    observed = [cause for cause in unique if cause.get("observed")]
    inferred = [cause for cause in unique if not cause.get("observed")]

    stated = [FLAG_TEXT.get(flag, flag.replace("_", " ")) for flag in problem_flags]
    if stated:
        lines.append(f"    You reported: {'; '.join(stated)}.")

    for cause in observed[:4]:
        chain = " -> ".join(cause["path"])
        lines.append(f"    (observed on this site) {chain} [{cause['id']}]")
    for cause in inferred[:3]:
        chain = " -> ".join(cause["path"])
        lines.append(f"    (possible driver, not confirmed here) {chain} [{cause['id']}]")

    for pattern in patterns:
        loop = " -> ".join(pattern.loop) if pattern.loop else ""
        lines.append(f"    (pattern) {pattern.name.replace('_', ' ')}: {loop} [{pattern.rule_id}]")

    for step in adjudication.reasoning_chain:
        if step.type in ("cause", "effect"):
            anchor = step.path_id or ", ".join(step.sources) or "-"
            lines.append(f"    ({step.type}) {step.claim} [{anchor}]")

    if len(lines) == 1:
        lines.append("    No causal chain in the graph reaches the flagged metrics for this site.")
    return lines


def evidence_section(evidence: list[EvidenceItem], cited: set[str]) -> list[str]:
    """The chunks the answer rests on, with a line of the text each, before the advice."""
    items = [item for item in evidence if item.label in cited] or evidence[:4]
    if not items:
        return []
    lines = ["EVIDENCE"]
    for item in sorted(items, key=lambda x: int(x.label[1:])):
        snippet = " ".join(item.text.split())
        if len(snippet) > 160:
            snippet = snippet[:157].rstrip() + "..."
        lines.append(f"    [{item.label}] {_citation(item)}: {snippet}")
    return lines


def order_note(adjudication: Adjudication, candidates: list[dict], plan: list[dict],
               excluded: list[dict]) -> str | None:
    """Say why the first recommendation is not the top-ranked candidate, when it is not."""
    if not adjudication.recommendations or not candidates:
        return None
    first = adjudication.recommendations[0].practice_id.value
    top = candidates[0]
    if top["practice_id"] == first:
        return None

    name = top["name"]
    reasons = []
    excluded_reason = next(
        (item["reason"] for item in excluded if item["practice_id"] == top["practice_id"]), None
    )
    if excluded_reason:
        reasons.append(f"it was excluded ({excluded_reason})")
    step = next((s for s in plan if s["practice_id"] == top["practice_id"]), None)
    if step and step.get("requires_first"):
        needs = ", ".join(step["requires_first"])
        reasons.append(f"it needs {needs} in place first")
    if top.get("downgraded") and top.get("risks_applied"):
        reasons.append(f"it carries a trade-off here: {top['risks_applied'][0]['risk']}")
    if not adjudication.agrees_with_ranking and adjudication.disagreement_reason:
        reasons.append(f"the review disagreed with the ranking: {adjudication.disagreement_reason}")
    if not reasons:
        reasons.append("the sequence puts a prerequisite ahead of it")
    return (f"    Order: {name} ranks first on suitability ({top['suitability']:.2f}) but is "
            f"not recommended first because {'; '.join(reasons)}.")


def _recommendation_block(index: int, recommendation: Recommendation,
                          plan_by_practice: dict[str, dict],
                          audience: Audience | None) -> list[str]:
    practice = recommendation.practice_id.value
    step = plan_by_practice.get(practice)
    lines = [f"[{index}] {recommendation.action}"]
    sources = ", ".join(recommendation.mechanism_sources) or "no source"
    paths = f" · {', '.join(recommendation.mechanism_paths)}" if recommendation.mechanism_paths else ""
    lines.append(f"    Why: {recommendation.mechanism} [{sources}{paths}]")
    lines.append(
        "    Variables: " + " | ".join(metric.value for metric in recommendation.variables_considered)
    )
    lines.append(
        "    Improves: " + " | ".join(metric.value for metric in recommendation.impacted_metrics)
    )
    estimate = _estimate_line(recommendation)
    lines.append(f"    Estimate: {estimate}" if estimate
                 else "    Estimate: direction only, no quantified evidence found")
    horizon = step["time_horizon"] if step else recommendation.time_horizon
    order = f", step {step['order']} in the sequence" if step else ""
    lines.append(f"    Horizon: {horizon}{order}")
    for risk in recommendation.risks:
        lines.append(f"    Risk: {risk.description} -> {risk.mitigation} [{risk.source}]")
    if audience == Audience.researcher:
        lines.append(f"    practice_id: {practice}")
    return lines


def tradeoffs_section(adjudication: Adjudication, candidates: list[dict]) -> list[str]:
    """Downgraded practices and the model's trade-off, synergy and conflict steps."""
    recommended = {r.practice_id.value for r in adjudication.recommendations}
    lines = ["TRADE-OFFS"]
    for candidate in candidates:
        if not candidate["downgraded"] or candidate["practice_id"] in recommended:
            continue
        risk = candidate["risks_applied"][0]
        lines.append(
            f"    {candidate['name']}: {risk['risk']} "
            f"(score {candidate['suitability']:.2f} after a {risk['penalty']:.2f} penalty)"
        )
        if risk.get("mitigation"):
            lines.append(f"        would need: {risk['mitigation']}")
        if len(lines) > 9:
            break
    for step in adjudication.reasoning_chain:
        if step.type in ("tradeoff", "synergy", "conflict"):
            anchor = step.path_id or ", ".join(step.sources) or "-"
            lines.append(f"    ({step.type}) {step.claim} [{anchor}]")
    return lines if len(lines) > 1 else []


def what_if_section(comparison: dict[str, Any] | None) -> list[str]:
    """Baseline versus hypothetical: what moved, what did not, what still binds."""
    if not comparison:
        return []
    lines = ["WHAT CHANGED"]
    change = comparison.get("change") or {}
    if change:
        lines.append("    Hypothetical: " + "; ".join(
            f"{name} {old} -> {new}" for name, (old, new) in change.items()
        ))
    for line in comparison.get("moved", []):
        lines.append(f"    {line}")
    if not comparison.get("moved"):
        lines.append("    Nothing in the ranking moved.")
    unchanged = comparison.get("unchanged", [])
    if unchanged:
        lines.append(f"    Unchanged: {', '.join(unchanged)}")
    constraints = comparison.get("constraints", [])
    if constraints:
        lines.append(f"    Still in force: {', '.join(constraints)}")
    lines.append("    The baseline is restored for your next question; this change is not kept.")
    return lines


def render(profile: SiteProfile, flags: list, fired_rules: list[str], patterns: list,
           adjudication: Adjudication, candidates: list[dict], excluded: list[dict],
           plan: list[dict], evidence: list[EvidenceItem], confidence: ConfidenceReport,
           audience: Audience | None = None, root_causes: list[dict] | None = None,
           asks: str = "none", problem_flags: list[str] | None = None,
           comparison: dict[str, Any] | None = None) -> str:
    """Assemble the answer. Every number and citation here already passed verification."""
    plan_by_practice = {step["practice_id"]: step for step in plan}
    cited = _cited_labels(adjudication)
    problem_flags = problem_flags or []
    root_causes = root_causes or []

    lines: list[str] = []
    lines.append(f"SITE      {_site_line(profile)}")
    lines.append(f"DIAGNOSIS {_diagnosis_line(flags, patterns)}")
    lines.append("")

    # A "why" gets its causes before anything else; a what-if gets its comparison first.
    causes = causes_section(flags, patterns, root_causes, adjudication, problem_flags)
    changed = what_if_section(comparison)
    if asks == "what_if" and changed:
        lines += changed + [""] + causes + [""]
    else:
        lines += causes + [""]
        if changed:
            lines += changed + [""]

    lines += evidence_section(evidence, cited) + [""]

    if adjudication.recommendations:
        lines.append("RECOMMENDED")
        note = order_note(adjudication, candidates, plan, excluded)
        if note:
            lines.append(note)
        for index, recommendation in enumerate(adjudication.recommendations, start=1):
            lines.extend(_recommendation_block(index, recommendation, plan_by_practice, audience))
            lines.append("")
    else:
        lines.append("RECOMMENDED  nothing could be verified for this site")
        lines.append("")

    tradeoffs = tradeoffs_section(adjudication, candidates)
    if tradeoffs:
        lines += tradeoffs + [""]

    if excluded:
        lines.append("EXCLUDED")
        for item in excluded:
            lines.append(f"    {item['name']}: {item['reason']}")
        lines.append("")

    if not adjudication.agrees_with_ranking and adjudication.disagreement_reason:
        lines.append(f"REVIEW    {adjudication.disagreement_reason}")
        lines.append("")

    breakdown = confidence.breakdown
    lines.append(
        f"CONFIDENCE {confidence.band} ({confidence.value:.2f}) = "
        f"evidence {breakdown.evidence:.2f} x context {breakdown.context:.2f} x "
        f"agreement {breakdown.agreement:.2f} x data quality {breakdown.data_quality:.2f}"
    )
    return "\n".join(lines).rstrip() + "\n"


def render_question(question: str, reason: str | None = None) -> str:
    """The one clarifying question a turn may ask, with why it matters."""
    if reason:
        return f"{question}\n\nWhy this matters: {reason}"
    return question


def payload(adjudication: Adjudication, confidence: ConfidenceReport) -> dict[str, Any]:
    """Structured half of the response, alongside the rendered text."""
    return {
        "recommendations": [r.model_dump() for r in adjudication.recommendations],
        "reasoning_chain": [step.model_dump() for step in adjudication.reasoning_chain],
        "agrees_with_ranking": adjudication.agrees_with_ranking,
        "disagreement_reason": adjudication.disagreement_reason,
        "confidence": confidence.model_dump(),
    }
