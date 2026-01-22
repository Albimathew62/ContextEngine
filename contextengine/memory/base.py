from abc import ABC, abstractmethod
from typing import List
from contextengine.models import ContextUnit


class MemoryStore(ABC):

    @abstractmethod
    def add(self, items: List[ContextUnit]) -> None:
        ...

    @abstractmethod
    def search(
        self,
        session_id: str,
        query_embedding: list,
        limit: int,
        min_score: float
    ) -> List[ContextUnit]:
        ...

    @abstractmethod
    def get_recent(
        self,
        session_id: str,
        limit: int
    ) -> List[ContextUnit]:
        ...

    @abstractmethod
    def clear_session(self, session_id: str) -> None:
        ...
