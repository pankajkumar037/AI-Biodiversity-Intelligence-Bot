"""The graph state and its reducers. The LLM never writes the profile directly."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

from core.schemas import SOURCE_PRIORITY, FieldSource, SiteField, SiteProfile


def merge_profile(old: dict[str, dict], new: dict[str, dict]) -> dict[str, dict]:
    """Keep the value from the stronger source; a newer user value replaces an older one."""
    merged = dict(old or {})
    for name, field in (new or {}).items():
        current = merged.get(name)
        if current is None:
            merged[name] = field
            continue
        new_rank = SOURCE_PRIORITY.get(field.get("source", "inferred"), 1)
        old_rank = SOURCE_PRIORITY.get(current.get("source", "inferred"), 1)
        if new_rank > old_rank:
            merged[name] = field
        elif new_rank == old_rank and field.get("turn", 0) >= current.get("turn", 0):
            merged[name] = field
    return merged


def merge_trace(old: dict[str, Any], new: dict[str, Any] | None) -> dict[str, Any]:
    """The trace is built up a section at a time as the pipeline runs; None clears it.

    A trace belongs to one turn. Left in the checkpoint, last turn's diagnosis and
    verification would show under an explain turn that ran neither.
    """
    if new is None:
        return {}
    merged = dict(old or {})
    merged.update(new)
    return merged


def add_unique(old: list[str], new: list[str]) -> list[str]:
    """Append without duplicating, for constraints and rejected practices."""
    merged = list(old or [])
    for item in new or []:
        if item not in merged:
            merged.append(item)
    return merged


def reset_or_extend(old: list[str], new: list[str] | None) -> list[str]:
    """Like add_unique, but None clears the list.

    Warnings belong to a turn. Without an explicit reset they would accumulate in
    the checkpoint and a note about turn 2 would reappear under turn 5's answer.
    """
    if new is None:
        return []
    return add_unique(old, new)


class AgentState(TypedDict, total=False):
    """Everything one turn needs. Nodes return partial updates to this."""

    messages: Annotated[list, add_messages]
    profile: Annotated[dict[str, dict], merge_profile]
    user_constraints: Annotated[list[str], add_unique]
    rejected_practices: Annotated[list[str], add_unique]
    trace: Annotated[dict[str, Any], merge_trace]
    warnings: Annotated[list[str], reset_or_extend]

    session_id: str
    turn: int
    audience: str | None
    intent: str | None
    message: str

    flags: list[str]
    flag_details: list[dict]
    fired_rules: list[str]
    patterns: list[dict]
    root_causes: list[dict]
    leverage: list[dict]
    candidates: list[dict]
    excluded: list[dict]
    combos: list[dict]
    plan: list[dict]

    queries: list[dict]
    evidence: list[dict]
    filter_level: int

    adjudication: dict | None
    verification: dict
    verify_attempts: int
    verify_feedback: str | None
    confidence: dict | None

    question: str | None
    answer: str
    skip_reasoning: bool
    what_if_baseline: dict | None


def to_profile(state_profile: dict[str, dict]) -> SiteProfile:
    """Graph state to the validated profile object."""
    fields = {
        name: SiteField(
            value=field.get("value"),
            source=FieldSource(field.get("source", "inferred")),
            turn=int(field.get("turn", 0)),
            uncertainty=field.get("uncertainty"),
        )
        for name, field in (state_profile or {}).items()
    }
    return SiteProfile(fields=fields)


def profile_values(state_profile: dict[str, dict]) -> dict[str, Any]:
    """Flat name -> value view, which is what the rule engine and graph consume."""
    return {name: field.get("value") for name, field in (state_profile or {}).items()}


def as_fields(values: dict[str, Any], source: FieldSource, turn: int = 0,
              uncertainty: float | None = None) -> dict[str, dict]:
    """Wrap plain values as profile fields carrying their provenance."""
    return {
        name: {"value": value, "source": source.value, "turn": turn,
               "uncertainty": uncertainty}
        for name, value in values.items()
        if value is not None
    }
