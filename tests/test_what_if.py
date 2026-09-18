"""A what-if turn has to say what moved, what stayed, and what still binds."""
from __future__ import annotations

from generation.render import what_if_section
from graph_flow.nodes import _what_if_comparison


def _candidate(practice_id: str, name: str, suitability: float, downgraded: bool,
               risk: str | None = None) -> dict:
    return {
        "practice_id": practice_id, "name": name, "suitability": suitability,
        "downgraded": downgraded,
        "risks_applied": [{"risk": risk, "penalty": 0.25, "mitigation": "x",
                           "condition": "y"}] if risk else [],
    }


BASELINE = {
    "flags": ["low_SOC", "water_limited", "monoculture"],
    "profile": {"soc_percent": 0.3, "rainfall_mm": 400, "crop": "wheat"},
    "constraints": ["leased_land_no_trees"],
    "practices": [
        {"practice_id": "cover_crops", "name": "Cover cropping",
         "suitability": 0.56, "downgraded": True},
        {"practice_id": "mulching", "name": "Mulching",
         "suitability": 0.73, "downgraded": False},
    ],
}


def _state(flags=None, profile=None, candidates=None) -> dict:
    fields = {k: {"value": v, "source": "user"} for k, v in (profile or {}).items()}
    return {"flags": flags if flags is not None else BASELINE["flags"],
            "profile": fields, "candidates": candidates or []}


def test_a_resolved_flag_is_reported():
    comparison = _what_if_comparison(BASELINE, _state(
        flags=["low_SOC", "monoculture"],
        candidates=[_candidate("mulching", "Mulching", 0.73, False)]))
    assert "resolved: water_limited" in comparison["moved"]


def test_a_newly_flagged_problem_is_reported():
    comparison = _what_if_comparison(BASELINE, _state(
        flags=["low_SOC", "water_limited", "monoculture", "sodic_risk"]))
    assert "newly flagged: sodic_risk" in comparison["moved"]


def test_lifting_a_risk_is_explained_not_just_scored():
    comparison = _what_if_comparison(BASELINE, _state(
        candidates=[_candidate("cover_crops", "Cover cropping", 0.81, False)]))
    line = next(m for m in comparison["moved"] if m.startswith("Cover cropping"))
    assert "no longer downgraded" in line and "0.56 -> 0.81" in line


def test_a_newly_downgraded_practice_names_its_risk():
    comparison = _what_if_comparison(BASELINE, _state(
        candidates=[_candidate("mulching", "Mulching", 0.40, True,
                               risk="residues are needed for fodder")]))
    line = next(m for m in comparison["moved"] if m.startswith("Mulching"))
    assert "now downgraded" in line and "residues are needed for fodder" in line


def test_a_practice_that_enters_the_ranking_is_reported():
    comparison = _what_if_comparison(BASELINE, _state(
        candidates=[_candidate("irrigation", "Supplemental irrigation", 0.31, True,
                               risk="raises soil salinity in dryland areas")]))
    assert any("now in play at 0.31" in m for m in comparison["moved"])


def test_the_hypothetical_change_and_unchanged_fields_are_separated():
    comparison = _what_if_comparison(BASELINE, _state(
        profile={"soc_percent": 0.3, "rainfall_mm": 400, "crop": "wheat", "irrigation": "drip"}))
    assert comparison["change"] == {"irrigation": (None, "drip")}
    assert comparison["unchanged"] == ["crop", "rainfall_mm", "soc_percent"]


def test_constraints_still_in_force_are_carried():
    comparison = _what_if_comparison(BASELINE, _state())
    assert comparison["constraints"] == ["leased_land_no_trees"]


def test_a_change_that_moves_nothing_says_so():
    comparison = _what_if_comparison(BASELINE, _state(candidates=[
        _candidate("cover_crops", "Cover cropping", 0.56, True),
        _candidate("mulching", "Mulching", 0.73, False),
    ]))
    assert comparison["moved"] == []
    text = "\n".join(what_if_section(comparison))
    assert "Nothing in the ranking moved" in text


def test_a_practice_that_drops_out_entirely_is_reported():
    comparison = _what_if_comparison(BASELINE, _state(candidates=[]))
    assert "Cover cropping: no longer applies" in comparison["moved"]
    assert "Mulching: no longer applies" in comparison["moved"]


def test_the_section_states_the_baseline_is_restored():
    text = "\n".join(what_if_section({"change": {"irrigation": (None, "drip")},
                                      "moved": [], "unchanged": ["crop"],
                                      "constraints": ["no_livestock"]}))
    assert text.startswith("WHAT CHANGED")
    assert "irrigation None -> drip" in text
    assert "Unchanged: crop" in text
    assert "Still in force: no_livestock" in text
    assert "baseline is restored" in text
