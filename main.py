"""FastAPI entry point. Routes only — all logic lives in the packages."""
from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core import db
from core.schemas import SearchRequest
from retrieval import assemble, rerank, search

app = FastAPI(title="Darukaa Biodiversity Intelligence", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
