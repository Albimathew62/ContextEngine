import pytest

from contextengine import ContextEngine, ContextConfig, InMemoryStore, SentenceTransformerEncoder
from contextengine.memory.mongo import MongoMemoryStore


@pytest.fixture(params=["inmemory", "mongo"])
def engine(request):
    encoder = SentenceTransformerEncoder()

    if request.param == "inmemory":
        store = InMemoryStore()
    else:
        store = MongoMemoryStore(
            uri="mongodb://localhost:27017",
            db_name="ce_test"
        )

    config = ContextConfig(session_id="test_session")

    engine = ContextEngine(
        store=store,
        encoder=encoder,
        config=config
    )

    # 🔥 important: clear session before each test
    engine.clear_session()

    return engine
