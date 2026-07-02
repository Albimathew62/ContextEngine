import math
from typing import List
from contextengine.models import ContextUnit
from contextengine.memory.base import MemoryStore


from contextengine.utils import cosine_similarity


class InMemoryStore(MemoryStore):
    def __init__(self):
        self._items: List[ContextUnit] = []
        self._embeddings: dict[str, list] = {}

    def add(self, items: List[ContextUnit]) -> None:
        for item in items:
            self._items.append(item)

    def attach_embedding(self, unit_id: str, embedding: list):
        self._embeddings[unit_id] = embedding

    def search(
        self,
        session_id: str,
        query_embedding: list,
        limit: int,
        min_score: float
    ) -> List[ContextUnit]:

        scored = []

        for item in self._items:
            if item.session_id != session_id:
                continue

            emb = self._embeddings.get(item.id)
            if not emb:
                continue

            score = cosine_similarity(query_embedding, emb)
            if score >= min_score:
                scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored[:limit]]

    def get_recent(self, session_id: str, limit: int) -> List[ContextUnit]:
        items = [i for i in self._items if i.session_id == session_id]
        items.sort(key=lambda x: x.created_at)
        return items[-limit:]

    def clear_session(self, session_id: str) -> None:
        self._items = [i for i in self._items if i.session_id != session_id]
        self._embeddings = {
            k: v for k, v in self._embeddings.items()
            if any(i.id == k for i in self._items)
        }
