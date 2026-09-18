"""The answer's section order is a requirement, so it is pinned here."""
from __future__ import annotations

from core.schemas import (
    Adjudication,
    ConfidenceBreakdown,
    ConfidenceReport,
    EvidenceItem,
    Flag,
    Pattern,
    ReasoningStep,
    Recommendation,
    SiteProfile,
)
from generation import render
from reasoning import causal
from verification import checks


def _evidence(label: str, practice: str) -> EvidenceItem:
    return EvidenceItem.model_validate({
        "label": label, "chunk_id": f"c-{label}", "doc_id": "d", "doc_title": "Doc",
        "text": "Field margins supply flowering resources for pollinators.",
        "page_start": 3, "evidence_level": "assessment", "climate_zones": ["global"],
        "practices": [practice], "claims": [],
    })


def _recommendation(practice: str, source: str = "S1") -> Recommendation:
    return Recommendation.model_validate({
        "practice_id": practice, "action": "Plant a margin.", "mechanism": "Margins feed bees.",
        "mechanism_sources": [source], "impacted_metrics": ["pollinators"],
        "variables_considered": ["pollinators", "habitat_diversity", "species_richness"],
        "estimates": [], "time_horizon": "medium", "risks": [],
    })


def _render(asks="none", comparison=None, adjudication=None, candidates=None, plan=None):
    adjudication = adjudication or Adjudication(
        reasoning_chain=[
            ReasoningStep(claim="pesticides suppress pollinators", type="cause", path_id="P1"),
            ReasoningStep(claim="margins offset that", type="synergy", sources=["S1"]),
        ],
        recommendations=[_recommendation("hedges_buffer_strips")],
    )
    confidence = ConfidenceReport(value=0.6, band="medium", breakdown=ConfidenceBreakdown(
        evidence=1, context=0.8, agreement=0.75, data_quality=1))
    return render.render(
        profile=SiteProfile(), flags=[Flag(flag="pollinator_decline", rule_id="T-POL-01")],
        fired_rules=["T-POL-01"],
        patterns=[Pattern(rule_id="C-02", name="simplified_system", implies="x",
                          loop=["habitat_diversity", "species_richness"])],
        adjudication=adjudication,
        candidates=candidates or [{"practice_id": "hedges_buffer_strips", "name": "Margins",
                                   "suitability": 0.7, "downgraded": False, "risks_applied": []}],
        excluded=[], plan=plan or [], evidence=[_evidence("S1", "hedges_buffer_strips")],
        confidence=confidence, root_causes=[
            {"id": "P1", "path": ["pesticide_use", "pollinators"], "observed": True},
            {"id": "P2", "path": ["habitat_loss", "pollinators"], "observed": False},
        ], asks=asks, problem_flags=["pollinator_decline"], comparison=comparison,
    )


def _order(text: str) -> list[str]:
    heads = ["SITE", "DIAGNOSIS", "CAUSES", "WHAT CHANGED", "EVIDENCE", "RECOMMENDED",
             "TRADE-OFFS", "EXCLUDED", "CONFIDENCE"]
    found = []
    for index, line in enumerate(text.splitlines()):
        for head in heads:
            if line.startswith(head):
                found.append((index, head))
    return [head for _, head in sorted(found)]


def test_sections_follow_the_required_order():
    assert _order(_render()) == ["SITE", "DIAGNOSIS", "CAUSES", "EVIDENCE", "RECOMMENDED",
                                 "TRADE-OFFS", "CONFIDENCE"]


def test_a_what_if_leads_with_the_comparison():
    text = _render(asks="what_if", comparison={"change": {"irrigation": (None, "drip")},
                                               "moved": ["x"], "unchanged": [], "constraints": []})
    order = _order(text)
    assert order.index("WHAT CHANGED") < order.index("CAUSES") < order.index("EVIDENCE")


def test_causes_put_observed_drivers_before_possible_ones_and_name_the_report():
    text = _render()
    causes = text[text.index("CAUSES"):text.index("EVIDENCE")]
    assert "You reported: pollinators are declining." in causes
    assert causes.index("(observed on this site) pesticide_use -> pollinators") < \
        causes.index("(possible driver, not confirmed here) habitat_loss -> pollinators")
    assert "(cause) pesticides suppress pollinators [P1]" in causes


def test_tradeoff_steps_land_in_the_tradeoffs_section_not_causes():
    text = _render()
    causes = text[text.index("CAUSES"):text.index("EVIDENCE")]
    assert "margins offset that" not in causes
    assert "(synergy) margins offset that [S1]" in text[text.index("TRADE-OFFS"):]


def test_evidence_comes_before_the_recommendation_with_a_snippet():
    text = _render()
    evidence = text[text.index("EVIDENCE"):text.index("RECOMMENDED")]
    assert "[S1] Doc, p3: Field margins supply flowering resources" in evidence


def test_order_note_explains_a_top_candidate_that_is_not_first():
    candidates = [
        {"practice_id": "agroforestry_agrisilvicultural", "name": "Agroforestry",
         "suitability": 0.8, "downgraded": True,
         "risks_applied": [{"risk": "tree-crop water competition", "penalty": 0.2}]},
        {"practice_id": "hedges_buffer_strips", "name": "Margins", "suitability": 0.7,
         "downgraded": False, "risks_applied": []},
    ]
    plan = [{"practice_id": "agroforestry_agrisilvicultural", "order": 2,
             "time_horizon": "long", "requires_first": ["mulching"]}]
    text = _render(candidates=candidates, plan=plan)
    note = next(line for line in text.splitlines() if line.strip().startswith("Order:"))
    assert "Agroforestry ranks first on suitability (0.80)" in note
    assert "needs mulching in place first" in note
    assert "tree-crop water competition" in note


def test_no_order_note_when_the_top_candidate_is_recommended_first():
    assert "Order:" not in _render()


def test_a_recommendation_that_ignores_the_stated_problem_fails_v1():
    ctx = checks.VerifyContext(
        evidence={"S1": _evidence("S1", "mulching")}, problem_flags={"pollinator_decline"},
        candidates={"mulching": {"addresses": ["low_SOC"], "net_effects": {}}},
    )
    failures = checks.check_addresses_problem(
        Adjudication(recommendations=[_recommendation("mulching")]), ctx)
    assert failures and "addresses none of the stated problem" in failures[0]


def test_problem_flags_are_the_user_stated_ones():
    assert "pollinator_decline" in causal.PROBLEM_FLAGS
    assert "low_SOC" not in causal.PROBLEM_FLAGS


def test_observed_drivers_come_from_the_profile():
    present = causal.observed_drivers({"pesticide_use": "high", "cropping_system": "monoculture"})
    assert {"pesticide_use", "monoculture"} <= present
    assert causal.observed_drivers({"pesticide_use": "low"}) == set()
