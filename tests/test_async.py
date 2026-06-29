import pytest
import asyncio
from contextengine import (
    AsyncContextEngine,
    ContextConfig,
    InMemoryStore,
    SentenceTransformerEncoder,
)


@pytest.fixture
def async_engine():
    return AsyncContextEngine(
        store=InMemoryStore(),
        encoder=SentenceTransformerEncoder(),
        config=ContextConfig(session_id="async_test")
    )


@pytest.mark.asyncio
async def test_async_store_and_retrieve(async_engine):
    await async_engine.store_interaction(
        input="My name is Albi",
        output="Nice to meet you Albi"
    )
    context = await async_engine.get_context(query="name")
    contents = [c["content"] for c in context]
    assert any("Albi" in c for c in contents)


@pytest.mark.asyncio
async def test_async_session_isolation():
    store = InMemoryStore()
    encoder = SentenceTransformerEncoder()

    engine1 = AsyncContextEngine(
        store=store,
        encoder=encoder,
        config=ContextConfig(session_id="async_user1")
    )
    engine2 = AsyncContextEngine(
        store=store,
        encoder=encoder,
        config=ContextConfig(session_id="async_user2")
    )

    await engine1.store_interaction(
        input="user1 secret",
        output="noted"
    )

    ctx2 = await engine2.get_context(query="secret")
    assert ctx2 == []


@pytest.mark.asyncio
async def test_async_clear_session(async_engine):
    await async_engine.store_interaction(
        input="temporary memory",
        output="will be cleared"
    )
    await async_engine.clear_session()
    memory = await async_engine.inspect_memory()
    assert memory == []


@pytest.mark.asyncio
async def test_async_concurrent_store():
    """Two coroutines storing simultaneously should not interfere."""
    store = InMemoryStore()
    encoder = SentenceTransformerEncoder()

    engine = AsyncContextEngine(
        store=store,
        encoder=encoder,
        config=ContextConfig(session_id="concurrent_test")
    )

    await asyncio.gather(
        engine.store_interaction(input="message one", output="response one"),
        engine.store_interaction(input="message two", output="response two"),
    )

    memory = await engine.inspect_memory()
    assert len(memory) == 4  # 2 inputs + 2 outputs