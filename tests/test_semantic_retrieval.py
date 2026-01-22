from contextengine import (
    ContextEngine,
    ContextConfig,
    InMemoryStore,
    SentenceTransformerEncoder,
)


def test_semantic_retrieval():
    engine = ContextEngine(
        store=InMemoryStore(),
        encoder=SentenceTransformerEncoder(),
        config=ContextConfig(session_id="test")
    )

    engine.store_interaction(
        input="I live in Kerala",
        output="Kerala is a beautiful place"
    )

    engine.store_interaction(
        input="I work as a software engineer",
        output="That's a great career"
    )

    context = engine.get_context(query="Where do I live?")

    contents = [c["content"] for c in context]

    assert any("Kerala" in c for c in contents)
