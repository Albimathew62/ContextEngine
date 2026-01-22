from contextengine import ContextEngine, ContextConfig, SentenceTransformerEncoder
from contextengine.memory.mongo import MongoMemoryStore

store = MongoMemoryStore(
    uri="mongodb://localhost:27017",
    db_name="ce_test"
)

engine = ContextEngine(
    store=store,
    encoder=SentenceTransformerEncoder(),
    config=ContextConfig(session_id="mongo_test")
)

# 🔥 CLEAR previous memory
engine.clear_session()

engine.store_interaction(
    input="I live in Kerala",
    output="Kerala is in India"
)

ctx = engine.get_context(query="Where do I live?")
print(ctx)
