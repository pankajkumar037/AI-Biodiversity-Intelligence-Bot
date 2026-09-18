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
