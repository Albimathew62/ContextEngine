from contextengine import (
    ContextEngine,
    ContextConfig,
    InMemoryStore,
    SentenceTransformerEncoder,
)


def test_hybrid_retrieval_keeps_flow():
    engine = ContextEngine(
        store=InMemoryStore(),
        encoder=SentenceTransformerEncoder(),
        config=ContextConfig(
            session_id="test",
            max_context_units=4
        )
    )

    engine.store_interaction(
        input="My name is Albi",
        output="Nice to meet you Albi"
    )

    engine.store_interaction(
        input="What is my name?",
        output="Your name is Albi"
    )

    context = engine.get_context(query="name")

    roles = [c["role"] for c in context]

    # must preserve chronological order
    assert roles[0] == "user"
    assert roles[-1] == "assistant"

    # must contain the question + answer
    contents = [c["content"] for c in context]
    assert "My name is Albi" in contents
    assert "Your name is Albi" in contents
