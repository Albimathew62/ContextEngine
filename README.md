# ContextEngine

**ContextEngine** is a lightweight, vendor-agnostic, pip-installable **context & memory engine** for LLM applications.

It adds **state and memory** to otherwise stateless LLM API calls by automatically storing, retrieving, and trimming past interactions.

> This is infrastructure — not a chatbot framework.

---

## ✨ Why ContextEngine?

Most LLM APIs are **stateless**:
- Every API call forgets the past
- Developers manually manage history
- Context windows overflow unpredictably

**ContextEngine solves this** by providing:
- Automatic memory storage (input + output)
- Semantic-first retrieval (real memory)
- Hybrid context (memory + conversation flow)
- Token-safe context trimming
- Pluggable storage (In-memory, MongoDB, more later)
- Zero vendor lock-in

---

## 🚀 Features

- ✅ Auto-save interactions by default
- ✅ Semantic memory (embedding-based)
- ✅ Hybrid retrieval (semantic + recent)
- ✅ Session-based isolation
- ✅ Token-budget aware context trimming
- ✅ InMemoryStore (dev / testing)
- ✅ MongoMemoryStore (production persistence)
- ✅ Fully vendor-agnostic
- ✅ Deterministic & inspectable behavior

---

## 📦 Installation

### Core (no database dependency)

```bash
pip install contextengine-ai
````

### Optional extras

```bash
pip install contextengine-ai[mongo,env]
```

> **Note**
> The PyPI package name is **`contextengine-ai`**, but the Python import remains:
>
> ```python
> import contextengine
> ```

ContextEngine keeps all database and environment helpers **optional**.

---

## 🧠 Core Concepts

### ContextUnit

The atomic unit of memory:

```text
(role, content, session_id, metadata, timestamp)
```

### Session

A user-defined identifier that isolates memory:

```python
session_id="user_123"
```

### MemoryStore

Pluggable backend for persistence:

* InMemoryStore
* MongoMemoryStore
* (FAISS / others later)

---

## 🔧 Quick Start (In-Memory)

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
    config=ContextConfig(session_id="demo")
)

engine.store_interaction(
    input="My name is Albi",
    output="Nice to meet you Albi"
)

context = engine.get_context(query="What is my name?")
print(context)
```

---

## 🗄️ Using MongoDB (Persistent Memory)

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
    output="Kerala is in India"
)

context = engine.get_context(query="Where do I live?")
print(context)
```

---

## ✂️ Token-Safe Context Trimming

ContextEngine automatically enforces a token budget.

```python
ContextConfig(
    session_id="chat",
    max_tokens=2048,
    token_estimator="chars"  # or "words"
)
```

* Oldest messages are removed first
* Order is preserved
* No vendor-specific tokenizers required

---

## 🌱 Optional `.env` Support

ContextEngine does **not** require `.env`, but supports it as a convenience.

```python
from contextengine import load_env
import os

load_env()  # loads .env if present

mongo_uri = os.getenv("MONGO_URI")
```

All configuration is still passed **explicitly**.

---

## 🧪 Testing

```bash
pytest
```

Tests run against:

* InMemoryStore
* MongoMemoryStore

Same behavior guaranteed.

---

## 🎯 Non-Goals

ContextEngine intentionally does **NOT** include:

* UI
* Agent frameworks
* Workflow orchestration
* Prompt templates
* LLM clients

It is a **memory layer**, not an application framework.

---

## 🧭 Roadmap

* FAISS / vector DB backend
* Custom token estimators
* Policy-based trimming
* Optional summaries
* Streaming support

---

## 📦 Project Links

* 📦 PyPI: [https://pypi.org/project/contextengine-ai/](https://pypi.org/project/contextengine-ai/)
* 🧠 GitHub: [https://github.com/Albimathew62/contextengine](https://github.com/Albimathew62/contextengine)

---

## 📜 License

MIT License

---

## 👤 Author

Built independently as an open-source infrastructure project by **Albi Mathew**.

````