"""Threshold bands and compound rules to flags. Pure Python, no LLM."""
from __future__ import annotations

from typing import Any

from core.schemas import Flag, Pattern
from knowledge import loader


def _in_band(value: float, band: dict) -> bool:
    low = band.get("min", float("-inf"))
    high = band.get("max", float("inf"))
    return low <= value < high


def diagnose(profile: dict[str, Any],
             thresholds: list[dict] | None = None) -> tuple[list[Flag], list[str]]:
    """Apply every rule to a flat name -> value profile. Returns (flags, fired rule ids)."""
    rules = thresholds if thresholds is not None else loader.thresholds()
    flags: list[Flag] = []
    fired: list[str] = []

    for rule in rules:
        value = profile.get(rule["metric"])
        if value is None:
            continue

        if "bands" in rule:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            for band in rule["bands"]:
                if not _in_band(float(value), band):
                    continue
                if band.get("flag"):
                    flags.append(Flag(
                        flag=band["flag"], rule_id=rule["rule_id"],
                        metric=rule["metric"], value=value, band=band.get("label"),
                    ))
                fired.append(rule["rule_id"])
                break

        elif "match" in rule:
            if value is None or str(value).strip() == "":
                continue
            for option in rule["match"]:
                wildcard = option["equals"] == "*"
                if not wildcard and str(value).lower() != str(option["equals"]).lower():
                    continue
                if option.get("flag"):
                    flags.append(Flag(
                        flag=option["flag"], rule_id=rule["rule_id"],
                        metric=rule["metric"], value=value, band=option.get("label"),
                    ))
                fired.append(rule["rule_id"])
                break

    return flags, fired


def detect_patterns(flag_names: list[str],
                    rules: list[dict] | None = None) -> list[Pattern]:
    """Find interacting problem patterns among the flags that fired."""
    compound = rules if rules is not None else loader.compound_rules()
    present = set(flag_names)
    return [
        Pattern(
            rule_id=rule["rule_id"], name=rule["name"], implies=rule["implies"],
            loop=rule.get("loop", []), flags=rule["if_all"],
        )
        for rule in compound
        if set(rule["if_all"]) <= present
    ]


def flag_names(flags: list[Flag]) -> list[str]:
    """Unique flag names, order preserved."""
    seen: list[str] = []
    for flag in flags:
        if flag.flag not in seen:
            seen.append(flag.flag)
    return seen
