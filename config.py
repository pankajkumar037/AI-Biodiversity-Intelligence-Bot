"""Single source of configuration: environment, model names and tunables."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ── paths ──────────────────────────────────────────────────────────────
KNOWLEDGE_DIR = ROOT / "knowledge"
PRACTICES_DIR = KNOWLEDGE_DIR / "practices"
THRESHOLDS_PATH = KNOWLEDGE_DIR / "thresholds.json"
COMPOUND_RULES_PATH = KNOWLEDGE_DIR / "compound_rules.json"
REGIONAL_DEFAULTS_PATH = KNOWLEDGE_DIR / "regional_defaults.json"

# ── database ───────────────────────────────────────────────────────────
DB_NAME = "darukaa"
VECTOR_INDEX = "vector_index"
TEXT_INDEX = "text_index"
CHECKPOINT_TTL_SECONDS = 86400


def mongodb_uri() -> str:
    """Build the Atlas URI, substituting the <db_password> placeholder."""
    uri = (os.getenv("MONGODB_URI") or "").strip()
    password = (os.getenv("db_password") or "").strip()
    if not uri:
        raise RuntimeError("MONGODB_URI missing from .env")
    if not password:
        raise RuntimeError("db_password missing from .env")
    return uri.replace("<db_password>", quote_plus(password))


def gemini_api_key() -> str:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing from .env")
    return key


# ── models ─────────────────────────────────────────────────────────────
# Flash-Lite does extraction, classification, query rewriting and judging.
# Flash does the single reasoning call. Never hardcode these elsewhere.
FAST_MODEL = "gemini-2.5-flash-lite"
REASONING_MODEL = "gemini-2.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
EMBED_TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
EMBED_TASK_QUERY = "RETRIEVAL_QUERY"

LLM_RETRIES = 4
LLM_BACKOFF_BASE_SECONDS = 2.0
ADJUDICATE_TEMPERATURE = 0.2

# ── retrieval ──────────────────────────────────────────────────────────
VECTOR_NUM_CANDIDATES = 200
VECTOR_LIMIT = 30
TEXT_LIMIT = 30
RRF_K = 60
MAX_CHUNKS_PER_DOC = 2
EVIDENCE_BLOCK_SIZE = 12
MIN_QUERIES = 6
MAX_QUERIES = 10

# Climate filters must always admit these, or well-tagged chunks are lost.
ALWAYS_ALLOWED_ZONES = ["global", "unstated"]

# Fallback ladder: level -> confidence multiplier applied to the context factor.
FILTER_LEVEL_PENALTY = {0: 1.00, 1: 0.90, 2: 0.75}
CONTEXT_MATCH_EXACT = 1.00
CONTEXT_MATCH_GENERIC = 0.85
CONTEXT_MATCH_FALLBACK = 0.70

# ── reasoning ──────────────────────────────────────────────────────────
PROPAGATE_MAX_HOPS = 3
PROPAGATE_DECAY = 0.7
ROOT_CAUSE_MAX_HOPS = 3
TOP_CANDIDATES_TO_PAIR = 5
SYNERGY_BONUS = 0.20
CONFLICT_PENALTY = 0.25
MIN_EDGE_SUPPORT = 2

STRENGTH_WEIGHT = {"strong": 1.0, "moderate": 0.7, "weak": 0.4, "unstated": 0.5}

# ── verification and confidence ────────────────────────────────────────
MAX_VERIFY_RETRIES = 2
MIN_VARIABLES_PER_RECOMMENDATION = 3
CONFIDENCE_HIGH = 0.70
CONFIDENCE_MEDIUM = 0.45

EVIDENCE_WEIGHT = {
    "assessment": 1.0,
    "meta_analysis": 1.0,
    "field_study": 0.9,
    "manual": 0.8,
    "review": 0.75,
    "case_study": 0.7,
    "web_context": 0.3,
}

# Phrases that signal advice with no metric attached (check V8).
GENERIC_PHRASES = [
    "sustainable practices",
    "improve soil health",
    "best practices",
    "eco-friendly",
    "environmentally friendly",
    "holistic approach",
    "boost biodiversity",
    "good agricultural practices",
]
