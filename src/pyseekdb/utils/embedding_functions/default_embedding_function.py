import sys
import warnings
from typing import Any, Self

from pyseekdb.client.embedding_function import Documents, EmbeddingFunction, Embeddings


class DefaultEmbeddingFunction(EmbeddingFunction[Documents]):
    _MODEL_NAME = "all-MiniLM-L6-v2"
    _HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    _DIMENSION = 384

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        preferred_providers: list[str] | None = None,
    ):
        if model_name != self._MODEL_NAME:
            raise ValueError(f"Currently only '{self._MODEL_NAME}' is supported, got '{model_name}'")
        if preferred_providers:
            warnings.warn(
                "preferred_providers is deprecated and will be removed in a future version. "
                "Use the preferred_providers argument of OnnxEmbeddingFunction instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.model_name = self._MODEL_NAME
        if sys.version_info >= (3, 14):
            from pyseekdb.utils.embedding_functions.sentence_transformer_embedding_function import (
                SentenceTransformerEmbeddingFunction,
            )

            self._backend = SentenceTransformerEmbeddingFunction(model_name=self._MODEL_NAME)
        else:
            from pyseekdb.utils.embedding_functions import OnnxEmbeddingFunction

            self._backend = OnnxEmbeddingFunction(
                model_name=self._MODEL_NAME,
                hf_model_id=self._HF_MODEL_ID,
                dimension=self._DIMENSION,
                preferred_providers=preferred_providers,
            )

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    def __call__(self, documents: Documents) -> Embeddings:
        return self._backend(documents)

    @staticmethod
    def name() -> str:
        return "default"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(_config: dict[str, Any]) -> Self:
        return DefaultEmbeddingFunction()

    def __repr__(self) -> str:
        return f"DefaultEmbeddingFunction(model_name='{self.model_name}')"


_default_embedding_function: DefaultEmbeddingFunction | None = None


def get_default_embedding_function() -> DefaultEmbeddingFunction:
    global _default_embedding_function
    if _default_embedding_function is None:
        _default_embedding_function = DefaultEmbeddingFunction()
    return _default_embedding_function
