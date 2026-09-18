"""The checks decide what a user is allowed to see, so each one is pinned down."""
from __future__ import annotations

import pytest

from core.schemas import (
    Adjudication,
    Estimate,
    EvidenceItem,
    ReasoningStep,
    Recommendation,
    Risk,
)
from generation import adjudicate as adjudicate_mod
from verification import checks

CLAIM = {
    "metric": "SOC", "direction": "increase", "value": 7.3, "unit": "percent",
    "evidence_span": "average SOC increase over each project was 7.3%",
}


def _evidence(**overrides) -> EvidenceItem:
    base = {
        "label": "S1", "chunk_id": "joshi2023_agj__p12", "doc_id": "joshi2023_agj",
        "doc_title": "Joshi et al. 2023", "text": "Cover crops raised SOC.",
        "page_start": 6, "evidence_level": "meta_analysis",
        "climate_zones": ["global"], "practices": ["cover_crops"], "claims": [CLAIM],
    }
    base.update(overrides)
    return EvidenceItem.model_validate(base)


def _ctx(**overrides) -> checks.VerifyContext:
    item = _evidence()
    base = {
        "evidence": {"S1": item},
        "path_ids": {"P1"},
        "profile": {"climate_zone": "semi-arid"},
        "risk_practices": {"cover_crops"},
        "candidates": {"cover_crops": {"net_effects": {"SOC": 0.8}}},
        "excluded_practices": set(),
    }
    base.update(overrides)
    return checks.VerifyContext(**base)


def _recommendation(**overrides) -> Recommendation:
    base = {
        "practice_id": "cover_crops",
        "action": "Sow a legume cover crop after the wheat harvest.",
        "mechanism": "Roots and residue return carbon to the soil.",
        "mechanism_sources": ["S1"],
        "impacted_metrics": ["SOC"],
        "variables_considered": ["SOC", "soil_moisture", "species_richness"],
        "estimates": [],
        "time_horizon": "medium",
        "risks": [],
    }
    base.update(overrides)
    return Recommendation.model_validate(base)


def _adjudication(recommendation: Recommendation | None = None,
                  steps: list[ReasoningStep] | None = None) -> Adjudication:
    return Adjudication(
        reasoning_chain=steps or [],
        recommendations=[recommendation or _recommendation()],
    )


# ── V2 ─────────────────────────────────────────────────────────────────
def test_v2_rejects_an_invented_source():
    bad = _recommendation(mechanism_sources=["S9"])
    failures = checks.check_v2_sources_exist(_adjudication(bad), _ctx())
    assert failures and "S9" in failures[0]


# ── V3 ─────────────────────────────────────────────────────────────────
def test_v3_accepts_a_number_that_matches_the_claim():
    good = _recommendation(estimates=[Estimate(
        metric="SOC", direction="increase", value=7.3, unit="percent", source="S1")])
    assert checks.check_v3_numbers_grounded(_adjudication(good), _ctx()) == []


def test_v3_rejects_a_number_the_claim_does_not_support():
    bad = _recommendation(estimates=[Estimate(
        metric="SOC", direction="increase", value=25.0, unit="percent", source="S1")])
    failures = checks.check_v3_numbers_grounded(_adjudication(bad), _ctx())
    assert failures and "25.0" in failures[0]


def test_v3_rejects_the_right_number_on_the_wrong_metric():
    bad = _recommendation(estimates=[Estimate(
        metric="crop_yield", direction="increase", value=7.3, unit="percent",
        source="S1")])
    assert checks.check_v3_numbers_grounded(_adjudication(bad), _ctx())


def test_v3_rejects_the_right_number_in_the_wrong_unit():
    bad = _recommendation(estimates=[Estimate(
        metric="SOC", direction="increase", value=7.3, unit="t/ha", source="S1")])
    assert checks.check_v3_numbers_grounded(_adjudication(bad), _ctx())


def test_v3_ignores_a_direction_only_estimate():
    fine = _recommendation(estimates=[Estimate(
        metric="SOC", direction="increase", source="S1")])
    assert checks.check_v3_numbers_grounded(_adjudication(fine), _ctx()) == []


def test_percent_sign_is_the_same_unit_as_percent():
    good = _recommendation(estimates=[Estimate(
        metric="SOC", direction="increase", value=7.3, unit="%", source="S1")])
    assert checks.check_v3_numbers_grounded(_adjudication(good), _ctx()) == []


# ── V4 ─────────────────────────────────────────────────────────────────
def test_v4_rejects_a_flipped_direction():
    bad = _recommendation(estimates=[Estimate(
        metric="SOC", direction="decrease", value=7.3, unit="percent", source="S1")])
    failures = checks.check_v4_direction(_adjudication(bad), _ctx())
    assert failures and "decrease" in failures[0]


# ── V6, V8, V9, V10 ────────────────────────────────────────────────────
def test_v6_requires_a_mechanism_source():
    bad = _recommendation(mechanism_sources=[])
    assert checks.check_v6_mechanism_and_risk_query(_adjudication(bad), _ctx())


def test_v6_requires_a_risk_query_for_the_practice():
    ctx = _ctx(risk_practices={"mulching"})
    assert checks.check_v6_mechanism_and_risk_query(_adjudication(), ctx)


def test_v8_catches_generic_advice():
    bad = _recommendation(action="Adopt sustainable practices on the farm.")
    failures = checks.check_v8_generic_language(_adjudication(bad))
    assert failures and "sustainable practices" in failures[0]


def test_v9_strips_a_step_that_cites_nothing_real():
    steps = [
        ReasoningStep(claim="grounded", type="cause", path_id="P1"),
        ReasoningStep(claim="floating", type="cause", path_id="P99"),
    ]
    failures = checks.check_v9_steps_anchored(_adjudication(steps=steps), _ctx())
    assert len(failures) == 1 and "floating" in failures[0]


def test_v10_requires_three_variables():
    bad = _recommendation(variables_considered=["SOC", "soil_moisture"])
    failures = checks.check_v10_multi_variable(_adjudication(bad))
    assert failures and "2 variables" in failures[0]


def test_an_excluded_practice_cannot_be_recommended():
    ctx = _ctx(excluded_practices={"cover_crops"})
    assert checks.check_practices_allowed(_adjudication(), ctx)


# ── degradation ────────────────────────────────────────────────────────
def test_degrade_strips_an_ungrounded_number_but_keeps_the_direction():
    bad = _recommendation(estimates=[Estimate(
        metric="SOC", direction="increase", value=25.0, unit="percent", source="S1")])
    degraded, counts = checks.degrade(_adjudication(bad), _ctx())
    estimate = degraded.recommendations[0].estimates[0]
    assert counts["stripped_numbers"] == 1
    assert estimate.value is None
    assert estimate.direction == "increase"


def test_degrade_drops_an_estimate_citing_a_path_id_instead_of_a_source():
    bad = _recommendation(estimates=[Estimate(
        metric="SOC", direction="increase", source="P8")])
    degraded, counts = checks.degrade(_adjudication(bad), _ctx())
    assert counts["dropped_estimates"] == 1
    assert degraded.recommendations[0].estimates == []


def test_degrade_drops_a_recommendation_with_too_few_variables():
    bad = _recommendation(variables_considered=["SOC"])
    degraded, counts = checks.degrade(_adjudication(bad), _ctx())
    assert counts["dropped_recommendations"] == 1
    assert degraded.recommendations == []


def test_degrade_removes_unanchored_reasoning_steps():
    steps = [
        ReasoningStep(claim="keep", type="cause", sources=["S1"]),
        ReasoningStep(claim="drop", type="cause", sources=["S9"]),
    ]
    degraded, counts = checks.degrade(_adjudication(steps=steps), _ctx())
    assert counts["dropped_steps"] == 1
    assert [step.claim for step in degraded.reasoning_chain] == ["keep"]


def test_degrade_drops_a_risk_citing_a_source_that_does_not_exist():
    bad = _recommendation(risks=[
        Risk(description="water competition", mitigation="terminate early", source="S9")
    ])
    degraded, _ = checks.degrade(_adjudication(bad), _ctx())
    assert degraded.recommendations[0].risks == []


# ── dossier shortlist ──────────────────────────────────────────────────
def _candidate(practice_id: str, suitability: float, penalty: float) -> dict:
    return {"practice_id": practice_id, "suitability": suitability,
            "penalty": penalty, "paths": []}


def test_dossier_keeps_a_heavily_penalised_practice_outside_the_top_six():
    candidates = [_candidate(f"p{i}", 0.9 - i * 0.1, 0.0) for i in range(6)]
    candidates.append(_candidate("irrigation", 0.08, 0.35))
    shortlist = adjudicate_mod.dossier_candidates(candidates)
    assert "irrigation" in {c["practice_id"] for c in shortlist}


def test_dossier_does_not_pad_with_unpenalised_also_rans():
    candidates = [_candidate(f"p{i}", 0.9 - i * 0.1, 0.0) for i in range(8)]
    assert len(adjudicate_mod.dossier_candidates(candidates)) == 6


@pytest.mark.parametrize("n", [0, 1, 6, 7])
def test_path_ids_are_unique(n):
    candidates = [
        {"practice_id": f"p{i}", "suitability": 0.5, "penalty": 0.0,
         "paths": [{"path": ["a", "b"], "weight": 0.5}]}
        for i in range(n)
    ]
    causes = [{"id": "P1", "path": ["x", "y"], "effect": "decrease"}]
    registry, _ = adjudicate_mod.build_paths(causes, candidates)
    ids = [entry["id"] for entry in registry]
    assert len(ids) == len(set(ids))
