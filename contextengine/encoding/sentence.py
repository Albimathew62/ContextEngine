from sentence_transformers import SentenceTransformer
from contextengine.encoding.base import Encoder


class SentenceTransformerEncoder(Encoder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts):
        return self.model.encode(texts).tolist()
