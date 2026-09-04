"""
Schema and index configuration for collection creation.

The Schema class provides fine-grained control over index configuration,
including dense vector index (HNSW), sparse vector index, and fulltext index.

Schema is the single configuration object accepted by ``create_collection``.
Its own ``embedding_function`` parameter can be used to attach a dense embedding
function.

"""

from __future__ import annotations

from typing import Any

from .configuration import (
    FulltextIndexConfig,
    HNSWConfiguration,
    IVFConfiguration,
    SparseVectorIndexConfig,
    VectorIndexConfig,
)
from .embedding_function import EmbeddingFunction


class Schema:
    """
    Schema configuration for collection creation.

    Schema provides fine-grained control over indexes and their parameters.
    Use Schema's ``embedding_function`` parameter for dense embedding configuration.

    Standard collections use a 384-dimensional HNSW index with cosine distance
    when ``vector_index`` is omitted. An omitted ``fulltext_index`` uses the IK
    analyzer, and ``sparse_vector_index`` is optional.

    Args:
        vector_index: HNSW configuration for dense vector index (optional).
        sparse_vector_index: Sparse vector index configuration (optional).
        fulltext_index: Fulltext index configuration (optional).
        embedding_function: Dense embedding function (optional). If provided with
            ``vector_index``, this is associated with the dense vector index.

    """

    def __init__(
        self,
        vector_index: VectorIndexConfig | HNSWConfiguration | IVFConfiguration | None = None,
        sparse_vector_index: SparseVectorIndexConfig | None = None,
        fulltext_index: FulltextIndexConfig | None = None,
        embedding_function: EmbeddingFunction | None = None,
    ):
        """Init.

        If ``vector_index`` is given as an ``HNSWConfiguration``/``IVFConfiguration``,
        ``embedding_function=`` is forwarded to the resulting ``VectorIndexConfig``.
        When a pre-built ``VectorIndexConfig`` has no embedding function, the same
        argument attaches one without mutating the caller's configuration object.
        """
        if isinstance(vector_index, VectorIndexConfig):
            self.vector_index = self._merge_embedding_function(vector_index, embedding_function)
        elif isinstance(vector_index, HNSWConfiguration):
            self.vector_index = VectorIndexConfig(hnsw=vector_index, embedding_function=embedding_function)
        elif isinstance(vector_index, IVFConfiguration):
            self.vector_index = VectorIndexConfig(ivf=vector_index, embedding_function=embedding_function)
        elif vector_index is None:
            # Standard collection creation resolves the default HNSW dimension when no EF is set.
            self.vector_index = VectorIndexConfig(embedding_function=embedding_function)
        else:
            raise TypeError(
                f"Unsupported vector index configuration type: {type(vector_index).__name__}. "
                f"Expected VectorIndexConfig, HNSWConfiguration, IVFConfiguration, or None."
            )
        self.sparse_vector_index = sparse_vector_index
        self.fulltext_index = fulltext_index

    @staticmethod
    def _merge_embedding_function(
        config: VectorIndexConfig,
        embedding_function: EmbeddingFunction | None,
    ) -> VectorIndexConfig:
        """Attach a supplied EF to a pre-built vector config without silently discarding it."""
        if embedding_function is None or config.embedding_function is embedding_function:
            return config
        if config.embedding_function is not None:
            raise ValueError(
                "VectorIndexConfig already has an embedding function; pass the same instance or omit "
                "Schema(embedding_function=...)."
            )
        return VectorIndexConfig(
            ivf=config.ivf,
            hnsw=config.hnsw,
            embedding_function=embedding_function,
        )

    def create_index(self, config: Any, embedding_function: EmbeddingFunction | None = None) -> Schema:
        """
        Add an index configuration to this schema.

        Supports method chaining for fluent API usage.

        Args:
            config: Index configuration object. It may be a
                ``VectorIndexConfig``, ``HNSWConfiguration``, ``IVFConfiguration``,
                ``SparseVectorIndexConfig``, or ``FulltextIndexConfig``.
            embedding_function: Optional dense embedding function forwarded to
                ``VectorIndexConfig`` for HNSW/IVF configurations, or attached to a
                pre-built ``VectorIndexConfig`` when it has no EF already.

        Returns:
            This Schema instance (for chaining).

        Raises:
            TypeError: If config is not a recognized index configuration type.

        """
        if isinstance(config, HNSWConfiguration):
            self.vector_index = VectorIndexConfig(hnsw=config, embedding_function=embedding_function)
        elif isinstance(config, IVFConfiguration):
            self.vector_index = VectorIndexConfig(ivf=config, embedding_function=embedding_function)
        elif isinstance(config, VectorIndexConfig):
            self.vector_index = self._merge_embedding_function(config, embedding_function)
        elif isinstance(config, SparseVectorIndexConfig):
            self.sparse_vector_index = config
        elif isinstance(config, FulltextIndexConfig):
            self.fulltext_index = config
        else:
            raise TypeError(
                f"Unsupported index configuration type: {type(config).__name__}. "
                f"Expected VectorIndexConfig, HNSWConfiguration, IVFConfiguration, SparseVectorIndexConfig, or FulltextIndexConfig."
            )
        return self

    def __repr__(self) -> str:
        """Repr."""
        parts = []
        if self.vector_index is not None:
            parts.append(f"vector_index={self.vector_index}")
        if self.sparse_vector_index is not None:
            parts.append(f"sparse_vector_index={self.sparse_vector_index}")
        if self.fulltext_index is not None:
            parts.append(f"fulltext_index={self.fulltext_index}")
        return f"Schema({', '.join(parts)})"
