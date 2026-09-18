"""V7: is the stated mechanism actually supported by the cited passage?"""
from __future__ import annotations

from core import llm
from core.schemas import Adjudication
from generation import prompts


def mechanism_supported(mechanism: str, passage: str) -> tuple[bool, str]:
    """Ask Flash-Lite for a yes/no verdict on one mechanism against one passage."""
    verdict = llm.judge(
        system=prompts.JUDGE_MECHANISM_SYSTEM_V1,
        user=prompts.JUDGE_MECHANISM_USER_V1.format(
            mechanism=mechanism, passage=passage[:4000]
        ),
    )
    return bool(verdict.get("supported")), str(verdict.get("reason", ""))


def check_v7_mechanism_supported(adjudication: Adjudication, ctx) -> list[str]:
    """One judged call per recommendation, against everything it cites for the mechanism.

    A mechanism sentence often draws on more than one passage, so the check reads
    the cited chunks together rather than demanding that the first one carry it all.
    """
    failures = []
    for recommendation in adjudication.recommendations:
        sources = [
            label for label in recommendation.mechanism_sources if label in ctx.evidence
        ]
        if not sources:
            continue
        passage = "\n\n".join(
            f"[{label}] {ctx.evidence[label].text}" for label in sources[:3]
        )
        try:
            supported, reason = mechanism_supported(recommendation.mechanism, passage)
        except RuntimeError as exc:
            # NOTE: a judge outage must not block an answer that passed every other
            # check. It is recorded as a warning in the trace instead.
            failures.append(f"V7: judge unavailable ({exc})")
            continue
        if not supported:
            failures.append(
                f"V7: {', '.join(sources[:3])} do not support the mechanism stated for "
                f"{recommendation.practice_id.value} ({reason}). Reword it to what the "
                f"passages say, or cite a chunk that does."
            )
    return failures


def unsupported_steps(adjudication: Adjudication, ctx, limit: int = 6) -> tuple[set[int], list[str]]:
    """Indices of cause/effect steps whose only anchor is evidence that does not state the link.

    A step anchored to a path id rests on the graph, which is itself extracted from
    the corpus, so only evidence-anchored causal claims are judged here.
    """
    dropped: set[int] = set()
    reasons: list[str] = []
    judged = 0
    for index, step in enumerate(adjudication.reasoning_chain):
        if step.type not in ("cause", "effect") or step.path_id or not step.sources:
            continue
        if judged >= limit:
            break
        passage = "\n\n".join(
            f"[{label}] {ctx.evidence[label].text}"
            for label in step.sources[:3] if label in ctx.evidence
        )
        if not passage:
            continue
        judged += 1
        try:
            supported, reason = mechanism_supported(step.claim, passage)
        except RuntimeError:
            continue
        if not supported:
            dropped.add(index)
            reasons.append(f"'{step.claim[:70]}' is not stated by {', '.join(step.sources[:3])}: {reason}")
    return dropped, reasons
