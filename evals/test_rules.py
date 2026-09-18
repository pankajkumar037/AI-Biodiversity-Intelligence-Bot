"""Deterministic checks over the rule engine for every test case that has a profile.

These need the Atlas graph, so they skip when the database is unreachable. The
offline unit tests in tests/ are the ones CI relies on.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CASES_DIR = Path(__file__).parent / "test_cases"


def _load_cases() -> list[dict]:
    cases = []
    for path in sorted(CASES_DIR.glob("TC-*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if "profile" in case:
            cases.append(case)
    return cases


CASES = _load_cases()


@pytest.fixture(scope="module")
def engine():
    """Import the engine and prove the graph loads, or skip the module."""
    try:
        from reasoning import combine, diagnose
        from reasoning import graph as graph_mod
        graph_mod.load_graph()
    except Exception as exc:  # noqa: BLE001 - any connection failure means skip
        pytest.skip(f"database unavailable: {exc}")
    return diagnose, combine


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_expected_flags_fire(case, engine):
    diagnose, _ = engine
    flags, _ = diagnose.diagnose(case["profile"])
    names = set(diagnose.flag_names(flags))

    for expected in case["expected"].get("flags", []):
        assert expected in names, f"{case['id']}: expected flag {expected}, got {names}"
    for forbidden in case["expected"].get("must_not_flag", []):
        assert forbidden not in names, f"{case['id']}: {forbidden} should not fire"


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_expected_patterns_fire(case, engine):
    diagnose, _ = engine
    flags, _ = diagnose.diagnose(case["profile"])
    patterns = {p.rule_id for p in diagnose.detect_patterns(diagnose.flag_names(flags))}
    for expected in case["expected"].get("patterns", []):
        assert expected in patterns, f"{case['id']}: expected pattern {expected}"


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_hard_constraints_exclude(case, engine):
    diagnose, combine = engine
    flags, _ = diagnose.diagnose(case["profile"])
    names = diagnose.flag_names(flags)
    _, excluded = combine.candidates(names, case["profile"], case.get("constraints", []))
    excluded_ids = {item["practice_id"] for item in excluded}

    for expected in case["expected"].get("must_exclude", []):
        assert expected in excluded_ids, (
            f"{case['id']}: {expected} should have been excluded, got {excluded_ids}"
        )


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_risky_practices_are_downgraded(case, engine):
    diagnose, combine = engine
    flags, _ = diagnose.diagnose(case["profile"])
    names = diagnose.flag_names(flags)
    ranked, _ = combine.candidates(names, case["profile"], case.get("constraints", []))
    by_id = {candidate["practice_id"]: candidate for candidate in ranked}

    for expected in case["expected"].get("must_downgrade", []):
        assert expected in by_id, f"{case['id']}: {expected} was not even a candidate"
        assert by_id[expected]["downgraded"], (
            f"{case['id']}: {expected} should carry a risk penalty here"
        )

    for practice in case["expected"].get("must_not_downgrade_for_water_competition", []):
        if practice not in by_id:
            continue
        risks = " ".join(risk["risk"].lower() for risk in by_id[practice]["risks_applied"])
        assert "water" not in risks or "competition" not in risks, (
            f"{case['id']}: {practice} should not carry the dryland water risk here"
        )


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_a_downgraded_practice_ranks_below_a_clean_one(case, engine):
    """The trade-off has to actually cost the practice its place in the ranking."""
    diagnose, combine = engine
    downgraded = case["expected"].get("must_downgrade", [])
    wanted = case["expected"].get("must_recommend_any", [])
    if not downgraded or not wanted:
        pytest.skip("case does not compare a downgraded practice with a clean one")

    flags, _ = diagnose.diagnose(case["profile"])
    ranked, _ = combine.candidates(diagnose.flag_names(flags), case["profile"],
                                   case.get("constraints", []))
    order = {candidate["practice_id"]: i for i, candidate in enumerate(ranked)}

    best_wanted = min((order[p] for p in wanted if p in order), default=None)
    assert best_wanted is not None, f"{case['id']}: none of {wanted} were candidates"
    for practice in downgraded:
        if practice in order:
            assert order[practice] > best_wanted, (
                f"{case['id']}: {practice} outranks every practice in {wanted}"
            )
