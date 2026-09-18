"""Assemble the LangGraph. Code decides the flow; the model only fills in content."""
from __future__ import annotations

import contextlib
from functools import lru_cache
from typing import Any

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph

import config
from core import db
from core.schemas import Intent
from graph_flow import nodes
from graph_flow.state import AgentState

REASONING_CHAIN = [
    ("diagnose", nodes.diagnose_node),
    ("root_cause", nodes.root_cause),
    ("leverage", nodes.leverage_node),
    ("candidates", nodes.candidates_node),
    ("combine", nodes.combine_node),
    ("sequence", nodes.sequence_node),
    ("plan_queries", nodes.plan_queries),
    ("retrieve", nodes.retrieve),
]


def _add_reasoning_chain(builder: StateGraph) -> None:
    """diagnose -> ... -> retrieve -> adjudicate -> verify -> render, with a retry loop."""
    for name, function in REASONING_CHAIN:
        builder.add_node(name, function)
    builder.add_node("adjudicate", nodes.adjudicate)
    builder.add_node("verify", nodes.verify)
    builder.add_node("render", nodes.render)

    for (name, _), (next_name, _) in zip(REASONING_CHAIN, REASONING_CHAIN[1:], strict=False):
        builder.add_edge(name, next_name)
    builder.add_edge("retrieve", "adjudicate")
    builder.add_edge("adjudicate", "verify")
    builder.add_conditional_edges(
        "verify", nodes.should_retry,
        {"adjudicate": "adjudicate", "render": "render"},
    )
    builder.add_edge("render", END)


@lru_cache(maxsize=1)
def analysis_graph():
    """Single-shot pipeline for /analyze: a profile in, a verified analysis out."""
    builder = StateGraph(AgentState)
    _add_reasoning_chain(builder)
    builder.add_edge(START, "diagnose")
    return builder.compile()


def _route_after_intent(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == Intent.out_of_scope.value:
        return "out_of_scope"
    if intent == Intent.explain.value and state.get("adjudication"):
        return "explain"
    if intent == Intent.concept.value:
        return "concept"
    return "slot_check"


def _route_after_slot_check(state: AgentState) -> str:
    return "ask" if state.get("question") else "diagnose"


@lru_cache(maxsize=1)
def chat_graph():
    """Conversational pipeline for /chat, checkpointed per session thread."""
    builder = StateGraph(AgentState)

    builder.add_node("intake", nodes.intake)
    builder.add_node("normalize", nodes.normalize_node)
    builder.add_node("geo_enrich", nodes.geo_enrich)
    builder.add_node("intent", nodes.route_intent)
    builder.add_node("slot_check", nodes.slot_check)
    builder.add_node("ask", nodes.ask)
    builder.add_node("explain", nodes.explain)
    builder.add_node("concept", nodes.concept)
    builder.add_node("out_of_scope", nodes.out_of_scope)
    _add_reasoning_chain(builder)

    # Intent first. A concept or out-of-scope turn must never reach intake, or a
    # general question gets mined for site values it does not contain.
    builder.add_edge(START, "intent")
    builder.add_conditional_edges("intent", _route_after_intent, {
        "out_of_scope": "out_of_scope",
        "explain": "explain",
        "concept": "concept",
        "slot_check": "intake",
    })
    builder.add_edge("intake", "normalize")
    builder.add_edge("normalize", "geo_enrich")
    builder.add_edge("geo_enrich", "slot_check")
    builder.add_conditional_edges("slot_check", _route_after_slot_check, {
        "ask": "ask", "diagnose": "diagnose",
    })
    builder.add_edge("ask", END)
    builder.add_edge("explain", END)
    builder.add_edge("concept", END)
    builder.add_edge("out_of_scope", END)

    checkpointer = MongoDBSaver(
        db.client(), db_name=config.DB_NAME, ttl=config.CHECKPOINT_TTL_SECONDS
    )
    return builder.compile(checkpointer=checkpointer)


def mermaid() -> str:
    """The compiled chat graph as a diagram, for the README and the submission."""
    return chat_graph().get_graph().draw_mermaid()


def run_analysis(values: dict[str, Any], constraints: list[str] | None = None,
                 audience: str | None = None) -> dict[str, Any]:
    """Run the single-shot pipeline over already-structured site values."""
    from core.schemas import FieldSource
    from graph_flow.state import as_fields

    state: dict[str, Any] = {
        "profile": as_fields(values, FieldSource.user, 0),
        "user_constraints": list(constraints or []),
        "audience": audience,
        "turn": 1,
        "trace": {},
        "verify_attempts": 0,
    }
    return analysis_graph().invoke(state)


def next_turn(session_id: str) -> int:
    """Turn number for the next message in this session."""
    stored = db.sessions().find_one({"_id": session_id}, {"turn": 1})
    return int(stored.get("turn", 0)) + 1 if stored else 1


def save_session(session_id: str, result: dict[str, Any]) -> None:
    """Persist site memory and this turn's trace alongside the checkpoint."""
    turn = int(result.get("turn", 1))
    adjudication = result.get("adjudication") or {}
    history = [
        {
            "turn": turn,
            "practice_id": recommendation.get("practice_id"),
            "status": "recommended",
        }
        for recommendation in adjudication.get("recommendations", [])
    ]
    history += [
        {"turn": turn, "practice_id": item.get("practice_id"),
         "status": "excluded", "reason": item.get("reason")}
        for item in result.get("excluded", [])
    ]

    db.sessions().update_one(
        {"_id": session_id},
        {
            "$set": {
                "turn": turn,
                "profile": result.get("profile", {}),
                "user_constraints": result.get("user_constraints", []),
                f"traces.{turn}": result.get("trace", {}),
            },
            "$push": {"recommendation_history": {"$each": history}},
        },
        upsert=True,
    )


def load_session(session_id: str) -> dict[str, Any]:
    """Profile, constraints and recommendation history for a session."""
    stored = db.sessions().find_one({"_id": session_id}, {"traces": 0})
    if not stored:
        return {"session_id": session_id, "exists": False, "profile": {},
                "user_constraints": [], "recommendation_history": []}
    return {
        "session_id": session_id,
        "exists": True,
        "turn": stored.get("turn", 0),
        "profile": stored.get("profile", {}),
        "user_constraints": stored.get("user_constraints", []),
        "recommendation_history": stored.get("recommendation_history", []),
    }


def load_trace(session_id: str, turn: int) -> dict[str, Any] | None:
    """The stored reasoning trace for one turn, or None."""
    stored = db.sessions().find_one({"_id": session_id}, {f"traces.{turn}": 1})
    if not stored:
        return None
    return (stored.get("traces") or {}).get(str(turn))


def reset_session(session_id: str) -> None:
    """Drop the site memory and the checkpointed thread for this session."""
    db.sessions().delete_one({"_id": session_id})
    # NOTE: an absent thread is the normal case when a session never ran.
    with contextlib.suppress(Exception):
        chat_graph().checkpointer.delete_thread(session_id)


def _turn_state(session_id: str, message: str, profile_patch: dict[str, Any] | None,
                turn: int) -> dict[str, Any]:
    """Fresh per-turn state. Anything left in the checkpoint from last turn would
    let one turn's question, warnings or what-if comparison resurface under the next."""
    from core.schemas import FieldSource
    from graph_flow.state import as_fields

    state: dict[str, Any] = {
        "message": message,
        "session_id": session_id,
        "turn": turn,
        "verify_attempts": 0,
        "verify_feedback": None,
        "question": None,
        "warnings": None,
        "what_if_baseline": None,
    }
    if profile_patch:
        state["profile"] = as_fields(profile_patch, FieldSource.user, turn)
    return state


def run_chat(session_id: str, message: str, profile_patch: dict[str, Any] | None = None,
             turn: int = 1) -> dict[str, Any]:
    """Run one conversational turn against the checkpointed thread."""
    return chat_graph().invoke(
        _turn_state(session_id, message, profile_patch, turn),
        config={"configurable": {"thread_id": session_id}},
    )


# Keys worth sending to a client as each node finishes. Evidence text and full
# candidate objects are large; the trace sections already summarise them.
STREAMED_KEYS = ("trace", "question", "intent", "flags", "warnings", "verify_attempts",
                 "verify_feedback", "filter_level")


def _node_event(node: str, update: dict[str, Any], elapsed_ms: int) -> dict[str, Any]:
    payload = {key: update[key] for key in STREAMED_KEYS if key in update}
    return {"type": "node", "node": node, "elapsed_ms": elapsed_ms, "update": payload}


def stream_chat(session_id: str, message: str, profile_patch: dict[str, Any] | None = None,
                turn: int = 1):
    """Yield one event per node as the turn runs, then the finished result.

    LangGraph's update stream is what makes the reasoning visible while it happens:
    the client sees diagnose finish before retrieval starts, not a spinner.
    """
    import time

    graph = chat_graph()
    config = {"configurable": {"thread_id": session_id}}
    started = time.perf_counter()
    last = started

    for event in graph.stream(_turn_state(session_id, message, profile_patch, turn),
                              config=config, stream_mode="updates"):
        now = time.perf_counter()
        for node, update in event.items():
            yield _node_event(node, update or {}, int((now - last) * 1000))
        last = now

    result = graph.get_state(config).values
    result["turn"] = turn
    save_session(session_id, result)
    yield {"type": "done", "elapsed_ms": int((time.perf_counter() - started) * 1000),
           "result": result}


def stream_analysis(values: dict[str, Any], constraints: list[str] | None = None,
                    audience: str | None = None):
    """Same as stream_chat for the single-shot pipeline."""
    import time

    from core.schemas import FieldSource
    from graph_flow.state import as_fields

    state: dict[str, Any] = {
        "profile": as_fields(values, FieldSource.user, 0),
        "user_constraints": list(constraints or []),
        "audience": audience,
        "turn": 1,
        "trace": {},
        "verify_attempts": 0,
    }
    started = time.perf_counter()
    last = started
    final: dict[str, Any] = dict(state)

    for event in analysis_graph().stream(state, stream_mode="updates"):
        now = time.perf_counter()
        for node, update in event.items():
            yield _node_event(node, update or {}, int((now - last) * 1000))
            _merge_into(final, update or {})
        last = now

    yield {"type": "done", "elapsed_ms": int((time.perf_counter() - started) * 1000),
           "result": final}


def _merge_into(state: dict[str, Any], update: dict[str, Any]) -> None:
    """Apply the same reducers the graph uses, for the uncheckpointed analysis graph."""
    from graph_flow.state import add_unique, merge_profile, merge_trace, reset_or_extend

    for key, value in update.items():
        if key == "trace":
            state["trace"] = merge_trace(state.get("trace", {}), value)
        elif key == "profile":
            state["profile"] = merge_profile(state.get("profile", {}), value)
        elif key in ("user_constraints", "rejected_practices"):
            state[key] = add_unique(state.get(key, []), value)
        elif key == "warnings":
            state[key] = reset_or_extend(state.get(key, []), value)
        else:
            state[key] = value
