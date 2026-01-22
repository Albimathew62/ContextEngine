from contextengine import (
    ContextEngine,
    ContextConfig,
    InMemoryStore,
    SentenceTransformerEncoder,
)


def test_session_isolation():
    store = InMemoryStore()
    encoder = SentenceTransformerEncoder()

    engine1 = ContextEngine(
        store=store,
        encoder=encoder,
        config=ContextConfig(session_id="user1")
    )

    engine2 = ContextEngine(
        store=store,
        encoder=encoder,
        config=ContextConfig(session_id="user2")
    )

    engine1.store_interaction(
        input="User1 secret",
        output="Noted"
    )

    ctx2 = engine2.get_context(query="secret")

    assert ctx2 == []
