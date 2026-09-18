"""Verified output to the final answer text. Pure Python: no model touches this."""
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


def _site_line(profile: SiteProfile) -> str:
    parts = []
    for name, field in profile.fields.items():
        parts.append(f"{name.replace('_', ' ')} {field.value} ({field.source.value})")
    return " | ".join(parts) if parts else "no site values supplied"


def _diagnosis_line(flags: list, fired: list[str], patterns: list) -> str:
    names = " | ".join(
        f"{flag.flag} [{flag.rule_id}]" for flag in flags
    ) or "no flags raised"
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
            f" over {estimate.timeframe_years:g} years"
            if estimate.timeframe_years else ""
        )
        lines.append(f"{metric} {direction}: {number}{timeframe} [{estimate.source}]")
    return "; ".join(lines) if lines else None


def _recommendation_block(index: int, recommendation: Recommendation,
                          plan_by_practice: dict[str, dict],
                          audience: Audience | None) -> list[str]:
    practice = recommendation.practice_id.value
    step = plan_by_practice.get(practice)
    lines = [f"[{index}] {recommendation.action}"]
    lines.append(
        f"    Why: {recommendation.mechanism} "
        f"[{', '.join(recommendation.mechanism_sources) or 'no source'}]"
    )
    lines.append(
        "    Variables: "
        + " | ".join(metric.value for metric in recommendation.variables_considered)
    )
    lines.append(
        "    Improves: "
        + " | ".join(metric.value for metric in recommendation.impacted_metrics)
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


def render(profile: SiteProfile, flags: list, fired_rules: list[str], patterns: list,
           adjudication: Adjudication, candidates: list[dict], excluded: list[dict],
           plan: list[dict], evidence: list[EvidenceItem],
           confidence: ConfidenceReport, audience: Audience | None = None) -> str:
    """Assemble the answer. Every number and citation here already passed verification."""
    plan_by_practice = {step["practice_id"]: step for step in plan}
    evidence_by_label = {item.label: item for item in evidence}
    recommended = {
        recommendation.practice_id.value
        for recommendation in adjudication.recommendations
    }

    lines: list[str] = []
    lines.append(f"SITE      {_site_line(profile)}")
    lines.append(f"DIAGNOSIS {_diagnosis_line(flags, fired_rules, patterns)}")
    lines.append("")

    if adjudication.recommendations:
        lines.append("RECOMMENDED")
        for index, recommendation in enumerate(adjudication.recommendations, start=1):
            lines.extend(_recommendation_block(index, recommendation,
                                               plan_by_practice, audience))
            lines.append("")
    else:
        lines.append("RECOMMENDED  nothing could be verified for this site")
        lines.append("")

    downgraded = [
        candidate for candidate in candidates
        if candidate["downgraded"] and candidate["practice_id"] not in recommended
    ]
    if downgraded:
        lines.append("DOWNGRADED")
        for candidate in downgraded[:4]:
            risk = candidate["risks_applied"][0]
            lines.append(
                f"    {candidate['name']}: {risk['risk']} "
                f"(score {candidate['suitability']:.2f} after a {risk['penalty']:.2f} penalty)"
            )
            if risk.get("mitigation"):
                lines.append(f"        would need: {risk['mitigation']}")
        lines.append("")

    if excluded:
        lines.append("EXCLUDED")
        for item in excluded:
            lines.append(f"    {item['name']}: {item['reason']}")
        lines.append("")

    if adjudication.reasoning_chain:
        lines.append("REASONING")
        for step in adjudication.reasoning_chain:
            anchor = step.path_id or ", ".join(step.sources) or "-"
            lines.append(f"    ({step.type}) {step.claim} [{anchor}]")
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
    lines.append("")

    cited = sorted({
        label
        for recommendation in adjudication.recommendations
        for label in (list(recommendation.mechanism_sources)
                      + [estimate.source for estimate in recommendation.estimates]
                      + [risk.source for risk in recommendation.risks])
        if label in evidence_by_label
    }, key=lambda label: int(label[1:]) if label[1:].isdigit() else 0)

    if cited:
        lines.append("SOURCES")
        for label in cited:
            lines.append(f"    [{label}] {_citation(evidence_by_label[label])}")

    return "\n".join(lines).rstrip() + "\n"


def render_question(question: str, reason: str | None = None) -> str:
    """The one clarifying question a turn may ask, with why it matters."""
    if reason:
        return f"{question}\n\nWhy this matters: {reason}"
    return question


def payload(adjudication: Adjudication, confidence: ConfidenceReport) -> dict[str, Any]:
    """Structured half of the response, alongside the rendered text."""
    return {
        "recommendations": [
            recommendation.model_dump()
            for recommendation in adjudication.recommendations
        ],
        "reasoning_chain": [step.model_dump() for step in adjudication.reasoning_chain],
        "agrees_with_ranking": adjudication.agrees_with_ranking,
        "disagreement_reason": adjudication.disagreement_reason,
        "confidence": confidence.model_dump(),
    }
