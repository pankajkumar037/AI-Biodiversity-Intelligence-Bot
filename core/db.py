"""MongoDB connection and collection handles. No query logic lives here."""
from __future__ import annotations

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

import config

_client: MongoClient | None = None


def client() -> MongoClient:
    """Return the shared client, connecting on first use."""
    global _client
    if _client is None:
        _client = MongoClient(config.mongodb_uri(), serverSelectionTimeoutMS=20000)
    return _client


def database() -> Database:
    return client()[config.DB_NAME]


def chunks() -> Collection:
    return database()["chunks"]


def variable_graph() -> Collection:
    return database()["variable_graph"]


def practices_draft() -> Collection:
    return database()["practices_draft"]


def sessions() -> Collection:
    return database()["sessions"]


def ping() -> dict:
    """Health probe: connectivity, collection counts and search index state."""
    database().command("ping")
    counts = {
        "chunks": chunks().count_documents({}),
        "variable_graph": variable_graph().count_documents({}),
        "practices_draft": practices_draft().count_documents({}),
    }
    indexes = [
        {
            "name": index.get("name"),
            "type": index.get("type"),
            "status": index.get("status"),
            "queryable": index.get("queryable"),
        }
        for index in chunks().list_search_indexes()
    ]
    return {"ok": True, "database": config.DB_NAME, "counts": counts, "indexes": indexes}
