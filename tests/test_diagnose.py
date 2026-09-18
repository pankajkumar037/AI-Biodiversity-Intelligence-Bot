"""Diagnosis must be exact and deterministic — it is the layer the LLM never touches."""
from __future__ import annotations

from reasoning import diagnose


def test_demo_site_raises_the_three_expected_flags():
    profile = {"soc_percent": 0.3, "rainfall_mm": 400, "cropping_system": "monoculture"}
    flags, fired = diagnose.diagnose(profile)
    names = diagnose.flag_names(flags)
    assert "low_SOC" in names
    assert "water_limited" in names
    assert "monoculture" in names
    assert "T-SOC-01" in fired
    assert "T-RAIN-01" in fired


def test_missing_values_fire_no_rules():
    flags, fired = diagnose.diagnose({"soc_percent": None, "ph": None})
    assert flags == []
    assert fired == []


def test_band_edges_are_half_open():
    low, _ = diagnose.diagnose({"soc_percent": 0.49})
    assert diagnose.flag_names(low) == ["low_SOC"]
    medium, fired = diagnose.diagnose({"soc_percent": 0.5})
    assert diagnose.flag_names(medium) == []
    assert "T-SOC-01" in fired


def test_healthy_soil_raises_no_soil_flag():
    flags, _ = diagnose.diagnose({"soc_percent": 1.2, "ph": 7.0})
    assert diagnose.flag_names(flags) == []


def test_alkaline_and_sodic_bands():
    alkaline, _ = diagnose.diagnose({"ph": 8.2})
    assert diagnose.flag_names(alkaline) == ["alkaline_soil"]
    sodic, _ = diagnose.diagnose({"ph": 9.0})
    assert diagnose.flag_names(sodic) == ["sodic_risk"]


def test_carbon_water_trap_pattern_fires():
    patterns = diagnose.detect_patterns(["low_SOC", "water_limited", "monoculture"])
    ids = [pattern.rule_id for pattern in patterns]
    assert "C-01" in ids
    trap = next(pattern for pattern in patterns if pattern.rule_id == "C-01")
    assert trap.implies == "reinforcing_loop"
    assert trap.loop[0] == "SOC"


def test_pattern_needs_every_flag():
    assert diagnose.detect_patterns(["low_SOC"]) == []


def test_every_threshold_rule_declares_a_source():
    from knowledge import loader
    for rule in loader.thresholds():
        assert rule.get("source"), f"{rule['rule_id']} has no source"
