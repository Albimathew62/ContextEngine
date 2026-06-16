# ContextEngine

> A lightweight, vendor-agnostic memory layer for LLM applications.

[![PyPI version](https://img.shields.io/pypi/v/contextengine-ai.svg)](https://pypi.org/project/contextengine-ai/)
[![Python](https://img.shields.io/pypi/pyversions/contextengine-ai.svg)](https://pypi.org/project/contextengine-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://img.shields.io/pypi/dm/contextengine-ai.svg)](https://pypi.org/project/contextengine-ai/)

---

## The problem

Every LLM API call is stateless. You manually stitch together conversation history, guess when context overflows, and rebuild this logic in every project.

ContextEngine is the infrastructure layer that solves this — once.

---

## What it does

ContextEngine sits between your application and any LLM API. It handles:

- **Storing** every interaction automatically
- **Retrieving** the most relevant context using semantic search
- **Trimming** history to stay within your token budget
- **Isolating** memory per session or user

No framework lock-in. No vendor dependency. Drop it in, configure once, forget about context management.

---

## Install

```bash
# Core — no database dependency
pip install contextengine-ai

# With MongoDB support
pip install contextengine-ai[mongo]

# With .env support
pip install contextengine-ai[env]
```

> The PyPI package is `contextengine-ai`. The import is `contextengine`.

---

## Quick start

```python
from contextengine import (
    ContextEngine,
    ContextConfig,
    InMemoryStore,
    SentenceTransformerEncoder,
)

engine = ContextEngine(
    store=InMemoryStore(),
    encoder=SentenceTransformerEncoder(),
    config=ContextConfig(session_id="user_123")
)

# Store an interaction
engine.store_interaction(
    input="My name is Albi",
    output="Nice to meet you, Albi."
)

# Retrieve relevant context for a new query
context = engine.get_context(query="What is my name?")
print(context)
# → [{"role": "user", "content": "My name is Albi"}, ...]
```

Pass `context` directly to your LLM call. That's it.

---

## Persistent memory with MongoDB

```python
from contextengine import ContextEngine, ContextConfig, SentenceTransformerEncoder
from contextengine.memory.mongo import MongoMemoryStore

engine = ContextEngine(
    store=MongoMemoryStore(
        uri="mongodb://localhost:27017",
        db_name="contextengine"
    ),
    encoder=SentenceTransformerEncoder(),
    config=ContextConfig(session_id="user_42")
)

engine.store_interaction(
    input="I live in Kerala",
    output="Kerala is in southern India."
)

context = engine.get_context(query="Where do I live?")
```

Switch from `InMemoryStore` to `MongoMemoryStore` and nothing else changes. Same API, same behavior.

---

## Token-safe trimming

ContextEngine enforces a token budget on every retrieval. No overflows, no surprises.

```python
ContextConfig(
    session_id="chat",
    max_tokens=2048,
    token_estimator="chars"  # or "words"
)
```

Oldest messages are dropped first. Message order is preserved. No vendor tokenizer required.

---

## Optional `.env` support

```python
from contextengine import load_env
import os

load_env()  # loads .env if present, no-op otherwise

mongo_uri = os.getenv("MONGO_URI")
```

All config is still passed explicitly — `.env` is a convenience, not a requirement.

---

## Architecture

```
Your App
   │
   ▼
ContextEngine
   ├── ContextConfig   — session ID, token budget, estimator
   ├── Encoder         — SentenceTransformerEncoder (pluggable)
   └── MemoryStore     — InMemoryStore | MongoMemoryStore | (FAISS soon)
           │
           ▼
      store_interaction(input, output)
      get_context(query)  →  hybrid retrieval (semantic + recent)
           │
           ▼
     Token-trimmed context list → pass to any LLM API
```

### Core types

| Type | Description |
|---|---|
| `ContextUnit` | Atomic memory unit: `(role, content, session_id, metadata, timestamp)` |
| `ContextConfig` | Session config: ID, token budget, estimator |
| `MemoryStore` | Pluggable storage interface |
| `Encoder` | Pluggable embedding interface |

---

## Storage backends

| Backend | Use case | Status |
|---|---|---|
| `InMemoryStore` | Development, testing | ✅ Available |
| `MongoMemoryStore` | Production persistence | ✅ Available |
| FAISS / vector DBs | High-scale semantic search | 🔜 Roadmap |

---

## How it compares

| | ContextEngine | LangChain Memory | Mem0 |
|---|---|---|---|
| Vendor-agnostic | ✅ | ✅ | ✅ |
| Pip-installable standalone | ✅ | ❌ (part of LC) | ✅ |
| No framework required | ✅ | ❌ | ⚠️ |
| Pluggable storage | ✅ | ⚠️ | ✅ |
| Lightweight install | ✅ | ❌ | ⚠️ |
| Deterministic & inspectable | ✅ | ⚠️ | ⚠️ |

ContextEngine is infrastructure. It doesn't include agents, prompts, chains, or clients — by design.

---

## What ContextEngine is not

ContextEngine intentionally excludes:

- UI or chat interfaces
- Agent or orchestration frameworks
- Prompt template systems
- LLM clients or wrappers
- Workflow engines

If you need those, pair ContextEngine with the tool of your choice. It works with anything.

---

## Testing

```bash
pytest
```

Tests run against both `InMemoryStore` and `MongoMemoryStore`. Identical behavior is guaranteed across backends.

---

## Roadmap

- [ ] FAISS and vector DB backends (Chroma, Qdrant)
- [ ] Async support (`async/await` compatible API)
- [ ] Accurate token estimation via `tiktoken`
- [ ] Policy-based trimming (summarize instead of drop)
- [ ] Optional interaction summarization
- [ ] Streaming support
- [ ] GitHub Actions CI

---

## Contributing

Contributions, issues, and feature requests are welcome.

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m 'Add your feature'`
4. Push and open a Pull Request

→ [Open an issue](https://github.com/Albimathew62/contextengine/issues)

---

## Links

| | |
|---|---|
| PyPI | https://pypi.org/project/contextengine-ai/ |
| GitHub | https://github.com/Albimathew62/contextengine |
| Issues | https://github.com/Albimathew62/contextengine/issues |

---

## License

MIT © [Albi Mathew](https://github.com/Albimathew62)
