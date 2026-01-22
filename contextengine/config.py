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
    token_estimator: str = "chars"  # "chars" | "words"

    metadata: Optional[Dict[str, Any]] = None

    # Optional metadata
    metadata: Optional[Dict[str, Any]] = None
