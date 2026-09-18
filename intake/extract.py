"""Free text to a site draft, and follow-up intent classification. Flash-Lite only."""
from __future__ import annotations

import config
from core import llm
from core.schemas import Intent, SiteProfileDraft
from generation import prompts


def extract_draft(message: str) -> SiteProfileDraft:
    """Pull stated site facts out of a user message. Never invents a value."""
    result = llm.generate_json(
        system=prompts.EXTRACT_SYSTEM_V1,
        user=message,
        schema=SiteProfileDraft,
        model=config.FAST_MODEL,
        temperature=0.0,
    )
    return result


def classify_intent(message: str) -> tuple[Intent, str | None]:
    """Route a follow-up. Returns (intent, constraint slug when intent is constraint)."""
    try:
        verdict = llm.judge(system=prompts.INTENT_SYSTEM_V1, user=message)
    except RuntimeError:
        # NOTE: if the classifier is unavailable, treat the turn as new information
        # rather than refusing to answer.
        return Intent.new_info, None

    raw = str(verdict.get("intent", "new_info"))
    try:
        intent = Intent(raw)
    except ValueError:
        intent = Intent.new_info

    constraint = verdict.get("constraint")
    if constraint in (None, "null", ""):
        constraint = None
    return intent, constraint
