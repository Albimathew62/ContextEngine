from abc import ABC, abstractmethod
from typing import List


class Encoder(ABC):

    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        ...
