import asyncio
from typing import List, Dict, Optional

from contextengine.engine import ContextEngine
from contextengine.memory.base import MemoryStore
from contextengine.encoding.base import Encoder
from contextengine.config import ContextConfig
from contextengine.models import ContextUnit


class AsyncContextEngine:
    """
    Async wrapper around ContextEngine.

    Exposes the same API as ContextEngine but with async/await support.
    Compatible with FastAPI, aiohttp, and any asyncio-based application.

    Usage:
        engine = AsyncContextEngine(
            store=InMemoryStore(),
            encoder=SentenceTransformerEncoder(),
            config=ContextConfig(session_id="user_123")
        )

        await engine.store_interaction(
            input="My name is Albi",
            output="Nice to meet you, Albi."
        )

        context = await engine.get_context(query="What is my name?")
    """

    def __init__(
        self,
        store: MemoryStore,
        encoder: Encoder,
        config: ContextConfig
    ):
        self._engine = ContextEngine(
            store=store,
            encoder=encoder,
            config=config
        )

    async def store_interaction(self, input: str, output: str) -> None:
        await asyncio.to_thread(
            self._engine.store_interaction,
            input,
            output
        )

    async def add_input(self, content: str, metadata: Optional[Dict] = None) -> None:
        await asyncio.to_thread(
            self._engine.add_input,
            content,
            metadata
        )

    async def add_output(self, content: str, metadata: Optional[Dict] = None) -> None:
        await asyncio.to_thread(
            self._engine.add_output,
            content,
            metadata
        )

    async def get_context(self, query: Optional[str] = None) -> List[Dict]:
        return await asyncio.to_thread(
            self._engine.get_context,
            query
        )

    async def inspect_memory(self) -> List[ContextUnit]:
        return await asyncio.to_thread(
            self._engine.inspect_memory
        )

    async def clear_session(self) -> None:
        await asyncio.to_thread(
            self._engine.clear_session
        )