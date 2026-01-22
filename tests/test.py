from contextengine import (
    ContextEngine,
    ContextConfig,
    InMemoryStore,
    SentenceTransformerEncoder,
)

engine = ContextEngine(
    store=InMemoryStore(),
    encoder=SentenceTransformerEncoder(),
    config=ContextConfig(
        session_id="trim_test",
        max_tokens=20  # VERY small on purpose
    )
)

engine.store_interaction(
    input="This is a very long message that should be trimmed",
    output="Another long response that will push token limit"
)

engine.store_interaction(
    input="Short question?",
    output="Short answer."
)

ctx = engine.get_context(query="question")
print(ctx)
