"""Control flow is decided by code, so the routing functions are tested on their own."""
from __future__ import annotations

from graph_flow import nodes
from graph_flow.build import _route_after_intent, _route_after_slot_check


def test_out_of_scope_never_reaches_the_pipeline():
    assert _route_after_intent({"intent": "out_of_scope"}) == "out_of_scope"


def test_a_concept_question_skips_site_reasoning():
    assert _route_after_intent({"intent": "concept"}) == "concept"



def test_new_info_and_constraint_run_the_pipeline():
    assert _route_after_intent({"intent": "new_info"}) == "slot_check"
    assert _route_after_intent({"intent": "constraint"}) == "slot_check"
    assert _route_after_intent({"intent": "what_if"}) == "slot_check"


def test_a_pending_question_ends_the_turn():
    assert _route_after_slot_check({"question": "How much rain?"}) == "ask"
    assert _route_after_slot_check({"question": None}) == "diagnose"


def test_verification_failure_sends_the_answer_back_for_another_pass():
    assert nodes.should_retry({"verify_feedback": "- V3: ..."}) == "adjudicate"
    assert nodes.should_retry({"verify_feedback": None}) == "render"


def test_out_of_scope_answer_redirects_without_advice():
    answer = nodes.out_of_scope({})["answer"]
    assert "soil" in answer.lower()


def test_a_different_land_use_or_place_is_a_new_site():
    from graph_flow.nodes import _new_site_reason
    assert _new_site_reason({"land_use": "cropland", "crop": "wheat"}, {"land_use": "grassland"})
    assert _new_site_reason({"place_name": "Sangrur"}, {"place_name": "Jodhpur"})
    assert _new_site_reason({"land_use": "cropland"}, {"land_use": "Cropland"}) is None
    assert _new_site_reason({"land_use": "cropland"}, {"soc_percent": 0.4}) is None
    assert _new_site_reason({}, {"land_use": "grassland"}) is None


def test_a_message_that_describes_a_site_from_scratch_is_a_new_site():
    from graph_flow.nodes import _new_site_reason
    stored = {"place_name": "Rajasthan", "lat": 26.81, "lon": 73.77, "climate_zone": "semi-arid",
              "land_use": "cropland", "soc_percent": 0.5}
    fresh = {"soc_percent": 0.3, "rainfall_mm": 350.0, "crop": "wheat", "land_use": "cropland"}
    assert _new_site_reason(stored, fresh, "new_info")
    # A single added fact builds on the standing site.
    assert _new_site_reason(stored, {"soc_percent": 0.3}, "new_info") is None
    assert _new_site_reason(stored, {"soc_percent": 0.3, "rainfall_mm": 350.0}, "new_info") is None
    # What-ifs and constraints never reset, however much they restate.
    assert _new_site_reason(stored, fresh, "what_if") is None
    assert _new_site_reason(stored, fresh, "constraint") is None
