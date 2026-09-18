"""Value of information: ask only the question that could change the answer."""
from __future__ import annotations

from typing import Any

from reasoning import combine, diagnose

# Plausible low/medium/high values per slot, used to simulate what the answer becomes.
SIMULATED_VALUES: dict[str, list[Any]] = {
    "soc_percent": [0.3, 0.6, 1.1],
    "rainfall_mm": [250.0, 550.0, 1100.0],
    "cropping_system": ["monoculture", "rotation", "mixed"],
    "land_use": ["cropland", "grassland", "orchard"],
    "ph": [5.5, 7.0, 8.6],
    "natural_cover_percent": [5.0, 20.0, 45.0],
}

QUESTIONS: dict[str, tuple[str, str]] = {
    "soc_percent": (
        "Roughly how much organic carbon does your soil have? "
        "Low, medium or high is fine, or the percentage if you have a soil test.",
        "Organic carbon decides whether the priority is building carbon or protecting "
        "what is already there.",
    ),
    "rainfall_mm": (
        "About how much rain does the land get in a year? "
        "Low, medium or high, or millimetres if you know them.",
        "Rainfall decides whether a practice like cover crops helps your crop or "
        "competes with it for water.",
    ),
    "cropping_system": (
        "Do you grow one crop continuously, or do you rotate or mix crops?",
        "A single-species system carries diversity and pest penalties that a rotation "
        "does not, and it changes which practices are worth doing.",
    ),
    "land_use": (
        "Is this cropland, grassland or pasture, or an orchard?",
        "Land use rules whole families of practice in or out before anything else.",
    ),
    "ph": (
        "Do you know the soil pH, or whether the soil is acidic or alkaline?",
        "pH decides which legumes and amendments will work on this soil.",
    ),
    "natural_cover_percent": (
        "Is there much natural vegetation nearby, such as trees, scrub or "
        "uncultivated margins?",
        "Nearby habitat is what pollinators and soil life recolonise from, so it "
        "changes how quickly biodiversity can recover.",
    ),
}


def _top_practice(values: dict[str, Any], constraints: list[str]) -> str | None:
    """The practice the engine would recommend first, given these values."""
    flags, _ = diagnose.diagnose(values)
    names = diagnose.flag_names(flags)
    if not names:
        return None
    ranked, _ = combine.candidates(names, values, constraints)
    return ranked[0]["practice_id"] if ranked else None


def rank_slots(values: dict[str, Any], constraints: list[str],
               slots: list[str] | None = None) -> list[dict]:
    """Score each missing slot by how often filling it changes the top recommendation."""
    candidate_slots = slots or list(SIMULATED_VALUES)
    baseline = _top_practice(values, constraints)
    scored: list[dict] = []

    for slot in candidate_slots:
        if values.get(slot) is not None:
            continue
        outcomes = []
        for simulated in SIMULATED_VALUES.get(slot, []):
            trial = dict(values)
            trial[slot] = simulated
            outcomes.append(_top_practice(trial, constraints))

        if not outcomes:
            continue
        flips = sum(1 for outcome in outcomes if outcome != baseline)
        distinct = len({outcome for outcome in outcomes if outcome is not None})
        flip_score = round(flips / len(outcomes), 3)
        if flip_score == 0 and distinct <= 1:
            continue

        question, reason = QUESTIONS.get(slot, (f"What is {slot}?", ""))
        scored.append({
            "slot": slot,
            "flip_score": flip_score,
            "distinct_outcomes": distinct,
            "question": question,
            "reason": reason,
        })

    scored.sort(key=lambda item: (item["flip_score"], item["distinct_outcomes"]),
                reverse=True)
    return scored


def next_question(values: dict[str, Any], constraints: list[str]) -> dict | None:
    """The single most decision-relevant question, or None when nothing would change."""
    ranked = rank_slots(values, constraints)
    return ranked[0] if ranked else None
