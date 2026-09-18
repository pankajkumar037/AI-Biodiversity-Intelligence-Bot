"""Gemini wrappers. Three functions, retry with backoff, dev-time prompt cache."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel

import config

_client: genai.Client | None = None
_cache: dict[str, Any] = {}


def client() -> genai.Client:
    """Return the shared Gemini client, constructing it on first use."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.gemini_api_key())
    return _client


def _key(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def _with_retry(call, label: str):
    """Run call(), retrying with exponential backoff. Raises on final failure."""
    last: Exception | None = None
    for attempt in range(config.LLM_RETRIES):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - surfaced after the final attempt
            last = exc
            if attempt == config.LLM_RETRIES - 1:
                break
            time.sleep(min(30.0, config.LLM_BACKOFF_BASE_SECONDS * 2**attempt))
    raise RuntimeError(f"{label} failed after {config.LLM_RETRIES} attempts: {last}")


def generate_json(system: str, user: str, schema: type[BaseModel], model: str,
                  temperature: float = 0.0) -> BaseModel:
    """Call Gemini with a response schema and return the parsed model."""
    cache_key = _key("gen", model, str(temperature), system, user, schema.__name__)
    if cache_key in _cache:
        return _cache[cache_key]

    def call():
        response = client().models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
            ),
        )
        parsed = response.parsed
        if parsed is None:
            parsed = schema.model_validate_json(response.text)
        return parsed

    result = _with_retry(call, f"generate_json({model})")
    _cache[cache_key] = result
    return result


def embed_query(text: str) -> list[float]:
    """Embed a query: RETRIEVAL_QUERY, 768 dims, L2-normalised by us."""
    cache_key = _key("embed", text)
    if cache_key in _cache:
        return _cache[cache_key]

    def call():
        response = client().models.embed_content(
            model=config.EMBED_MODEL,
            contents=[text],
            config=types.EmbedContentConfig(
                task_type=config.EMBED_TASK_QUERY,
                output_dimensionality=config.EMBED_DIM,
            ),
        )
        vector = np.asarray(response.embeddings[0].values, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        # Truncated Matryoshka vectors are not pre-normalised.
        return (vector / norm).tolist() if norm else vector.tolist()

    result = _with_retry(call, "embed_query")
    _cache[cache_key] = result
    return result


def judge(system: str, user: str) -> dict:
    """Small JSON verdict from Flash-Lite at temperature 0."""
    cache_key = _key("judge", system, user)
    if cache_key in _cache:
        return _cache[cache_key]

    def call():
        response = client().models.generate_content(
            model=config.FAST_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        return json.loads(response.text)

    result = _with_retry(call, "judge")
    _cache[cache_key] = result
    return result
