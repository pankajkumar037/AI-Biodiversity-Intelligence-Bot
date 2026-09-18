"""
Load the extracted corpus into MongoDB Atlas.

Usage:  python dbingestion/ingest.py           # full load
        python dbingestion/ingest.py --dry     # no writes, no API calls
        python dbingestion/ingest.py --reset   # drop collections first

.env (project root):
    GEMINI_API_KEY=...
    db_password=...
    MONGODB_URI=mongodb+srv://pankaj_37:<db_password>@cluster0.rjjui.mongodb.net/?appName=Cluster0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

# ─────────────────────────────── config ────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
load_dotenv(ROOT / ".env")

DB_NAME = "darukaa"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 2.0
MIN_LEN = 80
TERSE_OK_DOCS = ("fao2021", "handa", "keerthika", "tewari")

DOC_TITLES = {
    "fao2021_recarb_v3": "FAO Recarbonizing Global Soils Vol. 3",
    "fao2021_recarb_v4": "FAO Recarbonizing Global Soils Vol. 4",
    "ipcc2019_srccl": "IPCC Climate Change and Land (SRCCL)",
    "ipbes2018_ldr": "IPBES Land Degradation and Restoration",
    "ipbes2016_pollinators": "IPBES Pollinators and Pollination",
    "handa2019_agroforestry_models": "Successful Agroforestry Models for India",
    "joshi2023_agj": "Joshi et al. 2023, Agronomy Journal",
    "keerthika2026_agroforestry_semiarid": "Keerthika et al. 2026, Frontiers in Agronomy",
    "tewari_agroforestry_hot_arid": "Tewari et al., Agroforestry in Hot Arid Environments",
    "fao2018_soil_pollution": "FAO Soil Pollution: A Hidden Reality",
    "fao2019_sowbfa": "FAO State of the World's Biodiversity for Food and Agriculture",
    "fao_gsocseq_intro": "FAO GSOCseq Introduction",
    "fsi2023_isfr_v1": "India State of Forest Report 2023 Vol. 1",
    "fsi2023_isfr_v2": "India State of Forest Report 2023 Vol. 2",
}
EVIDENCE_WEIGHT = {
    "assessment": 1.0, "meta_analysis": 1.0, "field_study": 0.9,
    "manual": 0.8, "case_study": 0.7, "review": 0.75, "web_context": 0.3,
}
EVIDENCE_LEVEL = {
    "fao2021_recarb_v3": "manual", "fao2021_recarb_v4": "case_study",
    "ipcc2019_srccl": "assessment", "ipbes2018_ldr": "assessment",
    "ipbes2016_pollinators": "assessment", "handa2019_agroforestry_models": "manual",
    "joshi2023_agj": "meta_analysis", "keerthika2026_agroforestry_semiarid": "field_study",
    "tewari_agroforestry_hot_arid": "review", "fao2018_soil_pollution": "assessment",
    "fao2019_sowbfa": "assessment", "fao_gsocseq_intro": "manual",
    "fsi2023_isfr_v1": "assessment", "fsi2023_isfr_v2": "assessment",
}
CONTEXT_ONLY = {"fsi2023_isfr_v1", "fsi2023_isfr_v2", "fao_gsocseq_intro"}


# ─────────────────────────────── helpers ───────────────────────────────
def mongo_uri() -> str:
    uri = os.getenv("MONGODB_URI", "").strip()
    pw = os.getenv("db_password", "").strip()
    if not uri:
        sys.exit("MONGODB_URI missing from .env")
    if not pw:
        sys.exit("db_password missing from .env")
    return uri.replace("<db_password>", quote_plus(pw))


def find_dir(name: str) -> Path:
    hit = next((p for p in DATA.rglob(name) if p.is_dir()), None)
    if not hit:
        sys.exit(f"{name}/ not found under data/")
    return hit


def find_file(name: str) -> Path | None:
    return next((p for p in DATA.rglob(name) if "_v10" not in str(p)), None)


def keep_passage(p: dict, doc_id: str) -> bool:
    t = (p.get("text") or "").strip()
    if len(t) >= MIN_LEN:
        return True
    role = p.get("content_role", "")
    if doc_id.startswith(TERSE_OK_DOCS) and role in {"constraint", "risk", "evidence", "mechanism"}:
        return True
    if role == "description" and t.lower().endswith(("dryland", "drylands")):
        return True
    return False


def context_header(doc_id: str, heading_path: list[str]) -> str:
    title = DOC_TITLES.get(doc_id, doc_id)
    path = " > ".join(h for h in (heading_path or []) if h)
    return f"[{title}{' > ' + path if path else ''}]"


def load_units(ext_dir: Path) -> list[dict]:
    units = []
    for f in sorted(ext_dir.rglob("*.json")):
        if "_v10" in str(f) or "_failures" in str(f):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if "_meta" in d:
            units.append(d)
    return units


# ───────────────────────────── build chunks ────────────────────────────
def build_chunks(units: list[dict], manifest: dict) -> list[dict]:
    chunks, seen_text, stats = [], set(), Counter()

    for u in units:
        meta = u["_meta"]
        doc_id, unit_id = meta["doc_id"], meta["unit_id"]
        mrec = manifest.get(unit_id, {})
        level = EVIDENCE_LEVEL.get(doc_id, "manual")

        claims_by_passage = defaultdict(list)
        for c in u.get("claims", []):
            claims_by_passage[c.get("passage_id")].append(c)

        for p in u["passages"]:
            text = (p.get("text") or "").strip()
            if not keep_passage(p, doc_id):
                stats["filtered"] += 1
                continue
            key = (doc_id, text)
            if key in seen_text:
                stats["duplicate"] += 1
                continue
            seen_text.add(key)

            header = context_header(doc_id, p.get("heading_path", []))
            embed_text = f"{header}\n{text}"

            zones = p.get("climate_zones") or mrec.get("climate_zones") or ["unstated"]
            practices = p.get("practices") or mrec.get("practices") or []

            chunks.append({
                "_id": p["passage_id"],
                "text": text,
                "embed_text": embed_text,
                "content_hash": hashlib.sha256(embed_text.encode()).hexdigest()[:16],
                "doc_id": doc_id,
                "unit_id": unit_id,
                "doc_title": DOC_TITLES.get(doc_id, doc_id),
                "heading_path": p.get("heading_path", []),
                "topic": p.get("topic"),
                "content_role": p.get("content_role"),
                "practices": practices,
                "metrics": p.get("metrics", []),
                "climate_zones": zones,
                "land_use": p.get("land_use", []),
                "region": mrec.get("region"),
                "evidence_level": level,
                "evidence_weight": EVIDENCE_WEIGHT.get(level, 0.7),
                "context_only": doc_id in CONTEXT_ONLY,
                "page_start": p.get("page_start"),
                "page_end": p.get("page_end"),
                "claims": claims_by_passage.get(p["passage_id"], []),
                "n_claims": len(claims_by_passage.get(p["passage_id"], [])),
                "prompt_type": meta.get("prompt_type"),
                "model": meta.get("model"),
            })
            stats["kept"] += 1

    print(f"  chunks: kept {stats['kept']}, filtered {stats['filtered']}, "
          f"deduped {stats['duplicate']}")
    return chunks


# ───────────────────────────── embeddings ──────────────────────────────
def embed_batch(client: genai.Client, texts: list[str], task_type: str,
                retries: int = 4) -> list[list[float]]:
    for attempt in range(retries):
        try:
            r = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBED_DIM,
                ),
            )
            out = []
            for e in r.embeddings:
                v = np.asarray(e.values, dtype=np.float32)
                n = np.linalg.norm(v)
                out.append((v / n).tolist() if n else v.tolist())
            return out
        except Exception as exc:
            wait = min(60, 5 * 2 ** attempt)
            print(f"    embed attempt {attempt+1} failed ({type(exc).__name__}), "
                  f"waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("embedding failed after retries")


def embed_chunks(client: genai.Client, chunks: list[dict], existing: dict[str, str]) -> int:
    todo = [c for c in chunks if existing.get(c["_id"]) != c["content_hash"]]
    print(f"  embedding {len(todo)} of {len(chunks)} chunks "
          f"({len(chunks)-len(todo)} unchanged)")
    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i:i + BATCH_SIZE]
        vecs = embed_batch(client, [c["embed_text"] for c in batch], "RETRIEVAL_DOCUMENT")
        for c, v in zip(batch, vecs):
            c["embedding"] = v
        print(f"    {min(i+BATCH_SIZE, len(todo))}/{len(todo)}", flush=True)
        time.sleep(SLEEP_BETWEEN_BATCHES)
    return len(todo)


# ────────────────────────────── writing ────────────────────────────────
def upsert(coll, docs: list[dict], label: str) -> None:
    if not docs:
        return
    ops = [UpdateOne({"_id": d["_id"]}, {"$set": d}, upsert=True) for d in docs]
    try:
        res = coll.bulk_write(ops, ordered=False)
        print(f"  {label}: upserted {res.upserted_count}, modified {res.modified_count}")
    except BulkWriteError as e:
        print(f"  {label}: bulk write errors: {e.details.get('nErrors')}")


def build_graph_docs(path: Path, scope_label: str) -> list[dict]:
    edges = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    for e in edges:
        key = f"{e['from_node']}|{e['effect']}|{e['to_node']}|{e.get('condition') or ''}"
        docs.append({
            "_id": hashlib.sha1(key.encode()).hexdigest()[:20],
            "from_node": e["from_node"],
            "to_node": e["to_node"],
            "effect": e["effect"],
            "sign": 1 if e["effect"] == "increase" else (-1 if e["effect"] == "decrease" else 0),
            "condition": e.get("condition"),
            "support_count": e.get("support_count", 1),
            "strength": e.get("strength", "unstated"),
            "scope": e.get("scope", "core"),
            "traversable": scope_label == "traversable",
            "docs": e.get("docs", []),
            "units": e.get("units", []),
            "mechanisms": e.get("mechanisms", []),
            "source": "extracted",
        })
    return docs


def build_practice_drafts(units: list[dict]) -> list[dict]:
    out = []
    for u in units:
        pp = u.get("practice_profile")
        if not pp:
            continue
        meta = u["_meta"]
        out.append({
            "_id": f"draft__{meta['unit_id']}",
            "unit_id": meta["unit_id"],
            "doc_id": meta["doc_id"],
            "reviewed": False,
            **pp,
        })
    return out


INDEX_NOTE = f"""
Create these two indexes in Atlas (Atlas Search tab -> Create Index -> JSON editor),
on database "{DB_NAME}", collection "chunks":

  1) Vector Search index, name: vector_index
{json.dumps({
  "fields": [
    {"type": "vector", "path": "embedding", "numDimensions": EMBED_DIM, "similarity": "cosine"},
    {"type": "filter", "path": "climate_zones"},
    {"type": "filter", "path": "practices"},
    {"type": "filter", "path": "metrics"},
    {"type": "filter", "path": "evidence_level"},
    {"type": "filter", "path": "content_role"},
    {"type": "filter", "path": "context_only"},
    {"type": "filter", "path": "doc_id"},
  ]}, indent=2)}

  2) Atlas Search index, name: text_index
{json.dumps({"mappings": {"dynamic": False, "fields": {
    "text": {"type": "string"},
    "topic": {"type": "string"},
    "doc_title": {"type": "string"},
}}}, indent=2)}
"""


# ─────────────────────────────── main ──────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="no writes, no embedding calls")
    ap.add_argument("--reset", action="store_true", help="drop collections first")
    args = ap.parse_args()

    print("=" * 66)
    print("DARUKAA INGESTION")
    print("=" * 66)

    ext_dir = find_dir("extracted")
    units = load_units(ext_dir)
    print(f"  loaded {len(units)} extracted units")

    manifest = {}
    for name in ("p0_units_manifest.json", "p1_units_manifest.json"):
        f = find_file(name)
        if f:
            for r in json.loads(f.read_text(encoding="utf-8")):
                manifest[r["unit_id"]] = r
    print(f"  manifest entries: {len(manifest)}")

    chunks = build_chunks(units, manifest)
    drafts = build_practice_drafts(units)

    g_trav = find_file("variable_graph_traversable.json")
    g_full = find_file("variable_graph.json")
    graph_docs = build_graph_docs(g_trav, "traversable") if g_trav else []
    trav_ids = {d["_id"] for d in graph_docs}
    if g_full:
        for d in build_graph_docs(g_full, "full"):
            if d["_id"] not in trav_ids:
                graph_docs.append(d)
    print(f"  graph edges: {len(graph_docs)} ({len(trav_ids)} traversable)")
    print(f"  practice drafts: {len(drafts)}")

    if args.dry:
        print("\n  --dry: stopping before Mongo and embeddings")
        print(INDEX_NOTE)
        return

    client_db = MongoClient(mongo_uri(), serverSelectionTimeoutMS=20000)
    client_db.admin.command("ping")
    db = client_db[DB_NAME]
    print(f"\n  connected to Atlas, db '{DB_NAME}'")

    if args.reset:
        for c in ("chunks", "variable_graph", "practices_draft"):
            db[c].drop()
        print("  collections dropped")

    existing = {d["_id"]: d.get("content_hash")
                for d in db.chunks.find({}, {"content_hash": 1})}

    gclient = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    n = embed_chunks(gclient, chunks, existing)

    to_write = [c for c in chunks if "embedding" in c]
    upsert(db.chunks, to_write, "chunks")
    upsert(db.variable_graph, graph_docs, "variable_graph")
    upsert(db.practices_draft, drafts, "practices_draft")

    db.meta.update_one(
        {"_id": "ingestion"},
        {"$set": {
            "embed_model": EMBED_MODEL, "embed_dim": EMBED_DIM,
            "n_chunks": db.chunks.count_documents({}),
            "n_edges": db.variable_graph.count_documents({}),
            "n_drafts": db.practices_draft.count_documents({}),
            "embedded_this_run": n,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }}, upsert=True)

    print("\n" + "=" * 66)
    print(f"  chunks in db:  {db.chunks.count_documents({})}")
    print(f"  graph edges:   {db.variable_graph.count_documents({})}")
    print(f"  drafts:        {db.practices_draft.count_documents({})}")
    print("=" * 66)
    print(INDEX_NOTE)


if __name__ == "__main__":
    main()
