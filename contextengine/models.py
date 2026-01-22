from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any
import uuid


@dataclass
class ContextUnit:
    role: str                    # "user", "assistant", "system", "tool"
    content: str
    session_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
