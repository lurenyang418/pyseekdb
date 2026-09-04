"""Small test doubles shared by unit and integration tests."""

from __future__ import annotations

from typing import Any

from pyseekdb.client.embedding_function import Documents, EmbeddingFunction, Embeddings


class StubEmbeddingFunction(EmbeddingFunction):
    """Deterministic dense embedding function for tests."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def __call__(self, documents: Documents) -> Embeddings:
        if isinstance(documents, str):
            documents = [documents]
        return [[0.0] * self.dimension for _ in documents]

    def embed_query(self, documents: Documents) -> Embeddings:
        return self(documents)

    @staticmethod
    def name() -> str:
        return "test_stub_embedding_function"

    def get_config(self) -> dict[str, Any]:
        return {"dimension": self.dimension}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> StubEmbeddingFunction:
        return StubEmbeddingFunction(dimension=config.get("dimension", 384))
