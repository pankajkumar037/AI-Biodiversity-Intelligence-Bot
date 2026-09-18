"""State reducers decide what survives a turn, so the merge rules are tested directly."""
from __future__ import annotations

from graph_flow.state import add_unique, merge_profile, merge_trace, reset_or_extend

USER = {"value": 0.3, "source": "user", "turn": 2}
INFERRED = {"value": 0.9, "source": "inferred", "turn": 3}
API = {"value": 0.45, "source": "soilgrids", "turn": 1}


def test_a_user_value_beats_an_api_estimate():
    assert merge_profile({"soc_percent": API}, {"soc_percent": USER})["soc_percent"] == USER


def test_an_inferred_value_never_overwrites_a_user_value():
    # Even though the inferred field is from a later turn.
    merged = merge_profile({"soc_percent": USER}, {"soc_percent": INFERRED})
    assert merged["soc_percent"] == USER


def test_a_newer_user_value_replaces_an_older_one():
    newer = {"value": 0.8, "source": "user", "turn": 5}
    merged = merge_profile({"soc_percent": USER}, {"soc_percent": newer})
    assert merged["soc_percent"] == newer


def test_an_api_value_fills_an_empty_slot():
    assert merge_profile({}, {"ph": API})["ph"] == API


def test_constraints_never_duplicate():
    assert add_unique(["leased_land_no_trees"], ["leased_land_no_trees"]) == \
        ["leased_land_no_trees"]
    assert add_unique(["a"], ["b", "a"]) == ["a", "b"]


def test_warnings_clear_on_a_new_turn():
    assert reset_or_extend(["last turn's note"], None) == []
    assert reset_or_extend([], ["this turn's note"]) == ["this turn's note"]


def test_trace_sections_accumulate_and_overwrite_by_key():
    merged = merge_trace({"diagnosis": {"flags": ["low_SOC"]}}, {"queries": [1, 2]})
    assert set(merged) == {"diagnosis", "queries"}
    assert merge_trace({"a": 1}, {"a": 2})["a"] == 2


def test_a_replace_marker_starts_the_list_over():
    from graph_flow.state import REPLACE
    assert add_unique(["leased_land_no_trees"], [REPLACE, "no_livestock"]) == ["no_livestock"]
    assert add_unique(["leased_land_no_trees"], [REPLACE]) == []


def test_a_replace_marker_swaps_the_whole_profile():
    from graph_flow.state import REPLACE
    fresh = {"land_use": {"value": "grassland", "source": "user"}}
    assert merge_profile({"crop": USER}, {REPLACE: fresh}) == fresh
