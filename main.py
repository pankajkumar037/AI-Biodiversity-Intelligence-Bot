"""FastAPI entry point. Routes only — all logic lives in the packages."""
from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core import db
from core.schemas import AnalyzeRequest, ChatRequest, SearchRequest
from graph_flow import build
from retrieval import assemble, rerank, search

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

app = FastAPI(title="Darukaa Biodiversity Intelligence", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


HEARTBEAT_SECONDS = 15


def _sse(events: Iterator[dict]) -> Iterator[str]:
    """Server-sent events framing with a heartbeat. Errors become a final event.

    The pipeline can go quiet for a minute inside one node. Hosting proxies drop a
    response that sends nothing for that long, so the generator runs in a thread and
    a comment line goes out whenever it has been silent for HEARTBEAT_SECONDS.
    """
    trace_id = str(uuid.uuid4())
    queue: Queue = Queue()
    done = object()

    def pump() -> None:
        try:
            for event in events:
                queue.put(event)
        except Exception as exc:  # noqa: BLE001 - reported to the client as an event
            queue.put({"type": "error", "error": type(exc).__name__, "message": str(exc)})
        finally:
            queue.put(done)

    Thread(target=pump, daemon=True).start()
    while True:
        try:
            event = queue.get(timeout=HEARTBEAT_SECONDS)
        except Empty:
            yield ": ping\n\n"
            continue
        if event is done:
            return
        yield f"data: {json.dumps({'trace_id': trace_id, **event}, default=str)}\n\n"


def _chat_payload(result: dict, session_id: str, turn: int) -> dict:
    return {
        "session_id": session_id,
        "turn": turn,
        "intent": result.get("intent"),
        "answer": result.get("answer", ""),
        "question": result.get("question"),
        "recommendations": (result.get("adjudication") or {}).get("recommendations", []),
        "confidence": result.get("confidence"),
        "trace": result.get("trace", {}),
    }


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    """Return a JSON message, never a stack trace."""
    return JSONResponse(
        status_code=500,
        content={
            "trace_id": str(uuid.uuid4()),
            "error": type(exc).__name__,
            "message": str(exc),
        },
    )


@app.get("/health")
def health() -> dict:
    """Database ping, collection counts and Atlas index status."""
    return {"trace_id": str(uuid.uuid4()), **db.ping()}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """One conversational turn against a checkpointed session thread."""
    turn = build.next_turn(request.session_id)
    result = build.run_chat(
        session_id=request.session_id,
        message=request.message,
        profile_patch=request.profile_patch,
        turn=turn,
        audience=request.audience.value if request.audience else None,
    )
    build.save_session(request.session_id, result)
    return {"trace_id": str(uuid.uuid4()), **_chat_payload(result, request.session_id, turn)}


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """The same turn as /chat, but one event per pipeline node as it finishes."""
    turn = build.next_turn(request.session_id)

    def events() -> Iterator[dict]:
        for event in build.stream_chat(request.session_id, request.message,
                                       request.profile_patch, turn,
                                       request.audience.value if request.audience else None):
            if event["type"] == "done":
                event = {**event, "result": _chat_payload(event["result"],
                                                          request.session_id, turn)}
            yield event

    return StreamingResponse(_sse(events()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/session/{session_id}")
def session(session_id: str) -> dict:
    """Stored profile, constraints and recommendation history for a session."""
    return {"trace_id": str(uuid.uuid4()), **build.load_session(session_id)}


@app.post("/session/{session_id}/reset")
def reset_session(session_id: str) -> dict:
    """Clear the conversation thread and the stored site memory."""
    build.reset_session(session_id)
    return {"trace_id": str(uuid.uuid4()), "session_id": session_id, "reset": True}


@app.get("/trace/{session_id}/{turn}", response_model=None)
def trace(session_id: str, turn: int) -> dict | JSONResponse:
    """Full reasoning trace for one turn, for the UI panel."""
    stored = build.load_trace(session_id, turn)
    if stored is None:
        return JSONResponse(
            status_code=404,
            content={"trace_id": str(uuid.uuid4()),
                     "message": f"no trace stored for turn {turn} of {session_id}"},
        )
    return {"trace_id": str(uuid.uuid4()), "session_id": session_id,
            "turn": turn, "trace": stored}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    """Structured site profile in, full verified analysis out. No conversation."""
    result = build.run_analysis(
        values=request.profile,
        constraints=request.constraints,
        audience=request.audience.value if request.audience else None,
    )
    return {
        "trace_id": str(uuid.uuid4()),
        "answer": result.get("answer", ""),
        "recommendations": (result.get("adjudication") or {}).get("recommendations", []),
        "confidence": result.get("confidence"),
        "trace": result.get("trace", {}),
    }


@app.post("/analyze/stream")
def analyze_stream(request: AnalyzeRequest) -> StreamingResponse:
    """Streamed twin of /analyze."""

    def events() -> Iterator[dict]:
        for event in build.stream_analysis(
            request.profile, request.constraints,
            request.audience.value if request.audience else None,
        ):
            if event["type"] == "done":
                result = event["result"]
                event = {**event, "result": {
                    "answer": result.get("answer", ""),
                    "recommendations": (result.get("adjudication") or {}).get(
                        "recommendations", []),
                    "confidence": result.get("confidence"),
                    "trace": result.get("trace", {}),
                }}
            yield event

    return StreamingResponse(_sse(events()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/search")
def debug_search(request: SearchRequest) -> dict:
    """Debug route: raw hybrid retrieval for one query, with scores and filter level."""
    hits, level = search.search_with_fallback(
        request.q, request.climate_zone, request.practice
    )
    scored = rerank.score_chunks(hits, request.climate_zone, level)
    items = assemble.assemble(scored, limit=request.limit)
    return {
        "trace_id": str(uuid.uuid4()),
        "query": request.q,
        "filter_level": level,
        "n_candidates": len(hits),
        "results": [item.model_dump() for item in items],
    }


@app.get("/knowledge/stats")
def knowledge_stats() -> dict:
    """Corpus size: chunks, edges, practice drafts and indexed documents."""
    chunks = db.chunks()
    doc_ids = sorted(chunks.distinct("doc_id"))
    return {
        "trace_id": str(uuid.uuid4()),
        "chunks": chunks.count_documents({}),
        "edges": db.variable_graph().count_documents({}),
        "traversable_edges": db.variable_graph().count_documents({"traversable": True}),
        "practice_drafts": db.practices_draft().count_documents({}),
        "documents": doc_ids,
        "n_documents": len(doc_ids),
    }


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


# The UI is plain files served from the same origin, so no CORS dance during the demo.
if FRONTEND_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")
