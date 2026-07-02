from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ContextConfig:
    session_id: str

    # Memory behavior
    auto_save: bool = True
    store_inputs: bool = True
    store_outputs: bool = True

    # Retrieval behavior
    max_context_units: int = 10
    similarity_threshold: float = 0.3

    # Token control
    max_tokens: int = 2048
    token_estimator: str = "chars"  # "chars" | "words" | "tiktoken"
    tiktoken_model: str = "gpt-4"   # used only when token_estimator="tiktoken"

    # Optional metadata
    metadata: Optional[Dict[str, Any]] = None