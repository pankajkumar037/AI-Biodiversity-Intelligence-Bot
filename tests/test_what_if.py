"""A what-if turn has to say what moved and why, not just re-answer."""
from __future__ import annotations

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
    "profile": {"soc_percent": 0.3, "rainfall_mm": 400},
    "practices": [
        {"practice_id": "cover_crops", "name": "Cover cropping",
         "suitability": 0.56, "downgraded": True},
        {"practice_id": "mulching", "name": "Mulching",
         "suitability": 0.73, "downgraded": False},
    ],
}


def test_a_resolved_flag_is_reported():
    state = {
        "flags": ["low_SOC", "monoculture"],
        "profile": {"soc_percent": {"value": 0.3, "source": "user"},
                    "rainfall_mm": {"value": 400, "source": "user"}},
        "candidates": [_candidate("mulching", "Mulching", 0.73, False)],
    }
    text = _what_if_comparison(BASELINE, state)
    assert "resolved: water_limited" in text


def test_a_newly_flagged_problem_is_reported():
    state = {
        "flags": ["low_SOC", "water_limited", "monoculture", "sodic_risk"],
        "profile": {},
        "candidates": [],
    }
    text = _what_if_comparison(BASELINE, state)
    assert "newly flagged: sodic_risk" in text


def test_lifting_a_risk_is_explained_not_just_scored():
    state = {
        "flags": BASELINE["flags"],
        "profile": {},
        "candidates": [_candidate("cover_crops", "Cover cropping", 0.81, False)],
    }
    text = _what_if_comparison(BASELINE, state)
    assert "no longer downgraded" in text
    assert "0.56 -> 0.81" in text
    assert "condition behind its risk no longer holds" in text


def test_a_newly_downgraded_practice_names_its_risk():
    state = {
        "flags": BASELINE["flags"],
        "profile": {},
        "candidates": [_candidate("mulching", "Mulching", 0.40, True,
                                  risk="residues are needed for fodder")],
    }
    text = _what_if_comparison(BASELINE, state)
    assert "now downgraded" in text
    assert "residues are needed for fodder" in text


def test_a_practice_that_enters_the_ranking_is_reported():
    state = {
        "flags": BASELINE["flags"],
        "profile": {},
        "candidates": [_candidate("irrigation", "Supplemental irrigation", 0.31, True,
                                  risk="raises soil salinity in dryland areas")],
    }
    text = _what_if_comparison(BASELINE, state)
    assert "now in play at 0.31" in text


def test_changed_inputs_are_listed():
    state = {
        "flags": BASELINE["flags"],
        "profile": {"soc_percent": {"value": 0.3, "source": "user"},
                    "rainfall_mm": {"value": 400, "source": "user"},
                    "irrigation": {"value": "drip", "source": "user"}},
        "candidates": [],
    }
    text = _what_if_comparison(BASELINE, state)
    assert "irrigation: None -> drip" in text


def test_a_change_that_moves_nothing_says_so():
    unchanged = {
        "flags": BASELINE["flags"],
        "profile": {},
        "candidates": [
            _candidate("cover_crops", "Cover cropping", 0.56, True),
            _candidate("mulching", "Mulching", 0.73, False),
        ],
    }
    text = _what_if_comparison(BASELINE, unchanged)
    assert "nothing in the ranking moved" in text


def test_a_practice_that_drops_out_entirely_is_reported():
    state = {"flags": BASELINE["flags"], "profile": {}, "candidates": []}
    text = _what_if_comparison(BASELINE, state)
    assert "Cover cropping: no longer applies" in text
    assert "Mulching: no longer applies" in text


def test_small_score_wobbles_are_not_reported_as_changes():
    state = {
        "flags": BASELINE["flags"],
        "profile": {},
        "candidates": [_candidate("mulching", "Mulching", 0.75, False)],
    }
    text = _what_if_comparison(BASELINE, state)
    assert "Mulching" not in text
