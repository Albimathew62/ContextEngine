import math
from typing import List, Dict, Optional
from contextengine.models import ContextUnit
from contextengine.memory.base import MemoryStore
from contextengine.encoding.base import Encoder
from contextengine.config import ContextConfig


class ContextEngine:

    def _estimate_tokens(self, text: str) -> int:
        if self.config.token_estimator == "words":
            return len(text.split())
        # default: chars (≈ tokens / 4 heuristic)
        return max(1, len(text) // 4)

    def _count_context_tokens(self, messages: list) -> int:
        return sum(self._estimate_tokens(m["content"]) for m in messages)
    def __init__(
        self,
        store: MemoryStore,
        encoder: Encoder,
        config: ContextConfig
    ):
        self.store = store
        self.encoder = encoder
        self.config = config

    # ---------------- ADD MEMORY ----------------

    def add_input(self, content: str, metadata: Optional[Dict] = None):
        if not self.config.store_inputs:
            return

        unit = ContextUnit(
            role="user",
            content=content,
            session_id=self.config.session_id,
            metadata=metadata or {}
        )

        self._store_units([unit])

    def add_output(self, content: str, metadata: Optional[Dict] = None):
        if not self.config.store_outputs:
            return

        unit = ContextUnit(
            role="assistant",
            content=content,
            session_id=self.config.session_id,
            metadata=metadata or {}
        )

        self._store_units([unit])

    def store_interaction(self, input: str, output: str):
        units = []

        if self.config.store_inputs:
            units.append(
                ContextUnit(
                    role="user",
                    content=input,
                    session_id=self.config.session_id
                )
            )

        if self.config.store_outputs:
            units.append(
                ContextUnit(
                    role="assistant",
                    content=output,
                    session_id=self.config.session_id
                )
            )

        self._store_units(units)

    def _store_units(self, units: List[ContextUnit]):
        texts = [u.content for u in units]
        embeddings = self.encoder.encode(texts)

        self.store.add(units)

        for unit, emb in zip(units, embeddings):
            if hasattr(self.store, "attach_embedding"):
                self.store.attach_embedding(unit.id, emb)

    # ---------------- RETRIEVE CONTEXT ----------------

    import math
    from typing import Optional, Dict, List

    def get_context(self, query: Optional[str] = None) -> List[Dict]:
        session_id = self.config.session_id
        max_units = self.config.max_context_units

        # ---------- STEP 1: Decide split ----------
        semantic_k = math.ceil(max_units * 0.6)
        recent_k = math.ceil(max_units * 0.4)

        semantic_units = []
        recent_units = []

        # ---------- STEP 2: Semantic retrieval (PRIMARY) ----------
        if query:
            query_emb = self.encoder.encode([query])[0]
            semantic_units = self.store.search(
                session_id=session_id,
                query_embedding=query_emb,
                limit=semantic_k,
                min_score=self.config.similarity_threshold
            )

        # ---------- STEP 3: Recent retrieval ----------
        recent_units = self.store.get_recent(
            session_id=session_id,
            limit=recent_k
        )

        # ---------- STEP 4: Merge + dedupe ----------
        merged = {}
        for unit in semantic_units + recent_units:
            merged[unit.id] = unit

        units = list(merged.values())

        # ---------- STEP 5: Sort chronologically ----------
        units.sort(key=lambda u: u.created_at)

        # ---------- STEP 6: Final trim ----------
        units = units[-max_units:]

        # ---------- STEP 7: Token budget trimming ----------
        messages = [
            {"role": u.role, "content": u.content}
            for u in units
        ]

        while messages and self._count_context_tokens(messages) > self.config.max_tokens:
            # remove oldest message first
            messages.pop(0)

        return messages

    # ---------------- UTIL ----------------

    def inspect_memory(self) -> List[ContextUnit]:
        return self.store.get_recent(
            session_id=self.config.session_id,
            limit=10_000
        )

    def clear_session(self):
        self.store.clear_session(self.config.session_id)
