"""Load the hand-written knowledge files. Cached, because they never change at runtime."""
from __future__ import annotations

import json
from functools import lru_cache

import config
from core.schemas import PracticeCard

# NOTE: loaders live here rather than in each consumer so the JSON is parsed once
# and every module sees the same objects.


def _read(path) -> list | dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def thresholds() -> list[dict]:
    """Metric band and categorical rules."""
    return _read(config.THRESHOLDS_PATH)


@lru_cache(maxsize=1)
def compound_rules() -> list[dict]:
    """Interacting problem patterns."""
    return _read(config.COMPOUND_RULES_PATH)


@lru_cache(maxsize=1)
def regional_defaults() -> list[dict]:
    """Zone-level fallback values, each carrying its source."""
    return _read(config.REGIONAL_DEFAULTS_PATH)


@lru_cache(maxsize=1)
def practice_cards() -> dict[str, PracticeCard]:
    """practice_id -> validated card, read from knowledge/practices/*.json."""
    cards: dict[str, PracticeCard] = {}
    if not config.PRACTICES_DIR.exists():
        return cards
    for path in sorted(config.PRACTICES_DIR.glob("*.json")):
        card = PracticeCard.model_validate(_read(path))
        cards[card.practice_id.value] = card
    return cards


def defaults_for_zone(zone: str | None) -> dict:
    """Regional defaults for a climate zone, or an empty dict when unknown."""
    if not zone:
        return {}
    for entry in regional_defaults():
        if entry["zone"] == zone:
            return entry["defaults"]
    return {}
