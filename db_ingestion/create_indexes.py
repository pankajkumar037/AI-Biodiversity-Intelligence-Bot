import os
import time
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
uri = os.getenv("MONGODB_URI").replace("<db_password>", quote_plus(os.getenv("db_password")))
db = MongoClient(uri)["darukaa"]

existing = {ix["name"] for ix in db.chunks.list_search_indexes()}
print("existing:", existing or "none")

if "vector_index" not in existing:
    db.chunks.create_search_index(SearchIndexModel(
        name="vector_index",
        type="vectorSearch",
        definition={"fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 768, "similarity": "cosine"},
            {"type": "filter", "path": "climate_zones"},
            {"type": "filter", "path": "practices"},
            {"type": "filter", "path": "metrics"},
            {"type": "filter", "path": "evidence_level"},
            {"type": "filter", "path": "content_role"},
            {"type": "filter", "path": "context_only"},
            {"type": "filter", "path": "doc_id"},
        ]},
    ))
    print("vector_index requested")

if "text_index" not in existing:
    db.chunks.create_search_index(SearchIndexModel(
        name="text_index",
        type="search",
        definition={"mappings": {"dynamic": False, "fields": {
            "text": {"type": "string"},
            "topic": {"type": "string"},
            "doc_title": {"type": "string"},
        }}},
    ))
    print("text_index requested")

print("\nwaiting for indexes to become queryable...")
for _ in range(40):
    ixs = list(db.chunks.list_search_indexes())
    states = {ix["name"]: (ix.get("status"), ix.get("queryable")) for ix in ixs}
    print("  ", states, flush=True)
    if len(ixs) >= 2 and all(q for _, q in states.values()):
        print("\nboth indexes are queryable")
        break
    time.sleep(15)
else:
    print("\nstill building — re-run this script or check the Atlas UI")
