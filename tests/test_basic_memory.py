from contextengine import (
    ContextEngine,
    ContextConfig,
    InMemoryStore,
    SentenceTransformerEncoder,
)


def test_store_interaction_basic(engine):
    engine.store_interaction(
        input="My name is Albi",
        output="Nice to meet you Albi"
    )

    memory = engine.inspect_memory()

    assert len(memory) == 2
    assert memory[0].role == "user"
    assert memory[1].role == "assistant"
    memory = engine.inspect_memory()

    assert len(memory) == 2
    assert memory[0].role == "user"
    assert memory[1].role == "assistant"
