"""The app must import and build its route table without a database or an API key.

A route whose return annotation FastAPI cannot turn into a response model raises
at import time, not on the first request, so this is worth catching in CI.
"""
from __future__ import annotations

from main import app

EXPECTED = {
    ("GET", "/health"),
    ("POST", "/chat"),
    ("POST", "/analyze"),
    ("POST", "/search"),
    ("GET", "/session/{session_id}"),
    ("POST", "/session/{session_id}/reset"),
    ("GET", "/trace/{session_id}/{turn}"),
    ("GET", "/knowledge/stats"),
}


def _routes() -> set[tuple[str, str]]:
    found = set()
    for route in app.routes:
        for method in getattr(route, "methods", set()) - {"HEAD", "OPTIONS"}:
            found.add((method, route.path))
    return found


def test_every_documented_route_is_registered():
    missing = EXPECTED - _routes()
    assert not missing, f"missing routes: {sorted(missing)}"


def test_cors_is_open_for_the_demo():
    assert any(
        "CORSMiddleware" in str(middleware.cls) for middleware in app.user_middleware
    )
