"""Public client-side API for pyseekdb."""

from .async_client import AsyncClient
from .async_collection import AsyncCollection
from .client import Client
from .collection import Collection
from .configuration import (
    BengProperties,
    FulltextIndexConfig,
    HNSWConfiguration,
    IKMode,
    IKProperties,
    IVFConfiguration,
    IVFIndexLib,
    IVFIndexType,
    Ngram2Properties,
    NgramProperties,
    SpaceProperties,
    SparseVectorIndexConfig,
    VectorIndexConfig,
)
from .embedding_function import EmbeddingFunction, register_embedding_function
from .namespace import Namespace
from .query_types import QueryHint
from .schema import Schema
from .sparse_embedding_function import (
    SparseEmbeddingFunction,
    SparseEmbeddingFunctionRegistry,
    SparseVector,
    register_sparse_embedding_function,
)
from .types import K
from .version import Version

__all__ = [
    "AsyncClient",
    "AsyncCollection",
    "BengProperties",
    "Client",
    "Collection",
    "EmbeddingFunction",
    "FulltextIndexConfig",
    "HNSWConfiguration",
    "IKMode",
    "IKProperties",
    "IVFConfiguration",
    "IVFIndexLib",
    "IVFIndexType",
    "K",
    "Namespace",
    "Ngram2Properties",
    "NgramProperties",
    "QueryHint",
    "Schema",
    "SpaceProperties",
    "SparseEmbeddingFunction",
    "SparseEmbeddingFunctionRegistry",
    "SparseVector",
    "SparseVectorIndexConfig",
    "VectorIndexConfig",
    "Version",
    "register_embedding_function",
    "register_sparse_embedding_function",
]
