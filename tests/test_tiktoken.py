import pytest
from contextengine.utils.tokens import TiktokenEstimator


def test_tiktoken_count_is_accurate():
    tiktoken = pytest.importorskip("tiktoken")  # skip if not installed
    estimator = TiktokenEstimator(model="gpt-4")
    # "My name is Albi" = 5 tokens in cl100k_base
    assert estimator.count("My name is Albi") == 5


def test_tiktoken_unknown_model_falls_back_to_base_encoding():
    pytest.importorskip("tiktoken")
    estimator = TiktokenEstimator(model="unknown-model-xyz")
    # should not crash, should return a positive count
    assert estimator.count("hello world") > 0


def test_tiktoken_fallback_without_library(monkeypatch):
    import sys
    # Simulate tiktoken not being installed
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    import importlib
    import contextengine.utils.tokens as t
    importlib.reload(t)

    estimator = t.TiktokenEstimator(model="gpt-4")
    # should fall back to chars//4
    result = estimator.count("hello")
    assert result >= 1


def test_engine_uses_tiktoken_estimator():
    pytest.importorskip("tiktoken")
    from contextengine import ContextEngine, ContextConfig, InMemoryStore, SentenceTransformerEncoder

    engine = ContextEngine(
        store=InMemoryStore(),
        encoder=SentenceTransformerEncoder(),
        config=ContextConfig(
            session_id="tiktoken_test",
            token_estimator="tiktoken",
            tiktoken_model="gpt-4",
            max_tokens=2048
        )
    )

    engine.store_interaction(
        input="My name is Albi",
        output="Nice to meet you Albi"
    )

    context = engine.get_context(query="name")
    assert len(context) > 0