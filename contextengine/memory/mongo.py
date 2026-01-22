import math
from typing import List

from pymongo import MongoClient, ASCENDING

from contextengine.models import ContextUnit
from contextengine.memory.base import MemoryStore


def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-9)


class MongoMemoryStore(MemoryStore):
    def __init__(
        self,
        uri: str,
        db_name: str = "contextengine",
        collection_name: str = "context_units",
    ):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.col = self.db[collection_name]

        # Index for session-based retrieval
        self.col.create_index(
            [("session_id", ASCENDING), ("created_at", ASCENDING)]
        )

    def add(self, items: List[ContextUnit]) -> None:
        if not items:
            return

        docs = []
        for item in items:
            docs.append(
                {
                    "_id": item.id,
                    "session_id": item.session_id,
                    "role": item.role,
                    "content": item.content,
                    "embedding": None,  # added later
                    "metadata": item.metadata,
                    "created_at": item.created_at,
                }
            )

        self.col.insert_many(docs)

    def attach_embedding(self, unit_id: str, embedding: list):
        self.col.update_one(
            {"_id": unit_id},
            {"$set": {"embedding": embedding}}
        )

    def search(
        self,
        session_id: str,
        query_embedding: list,
        limit: int,
        min_score: float,
    ) -> List[ContextUnit]:

        docs = self.col.find(
            {"session_id": session_id, "embedding": {"$ne": None}}
        )

        scored = []

        for d in docs:
            score = cosine_similarity(query_embedding, d["embedding"])
            if score >= min_score:
                scored.append((d, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        units = []
        for d, _ in scored[:limit]:
            units.append(
                ContextUnit(
                    id=d["_id"],
                    role=d["role"],
                    content=d["content"],
                    session_id=d["session_id"],
                    metadata=d.get("metadata", {}),
                    created_at=d["created_at"],
                )
            )

        return units

    def get_recent(self, session_id: str, limit: int) -> List[ContextUnit]:
        docs = (
            self.col.find({"session_id": session_id})
            .sort("created_at", -1)
            .limit(limit)
        )

        units = []
        for d in reversed(list(docs)):
            units.append(
                ContextUnit(
                    id=d["_id"],
                    role=d["role"],
                    content=d["content"],
                    session_id=d["session_id"],
                    metadata=d.get("metadata", {}),
                    created_at=d["created_at"],
                )
            )

        return units

    def clear_session(self, session_id: str) -> None:
        self.col.delete_many({"session_id": session_id})
