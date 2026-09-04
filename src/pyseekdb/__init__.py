"""
pyseekdb - Python SDK for seekdb and OceanBase vector search.

A unified client for connecting to a seekdb Server or OceanBase Server via pymysql.
Provides collection-first APIs for vector, full-text, and hybrid retrieval.

Since pyseekdb 2.0, embedded mode and bundled embedding function implementations
have been removed. Users must run a seekdb/OceanBase server and supply their own
``EmbeddingFunction`` (via the :class:`~pyseekdb.client.embedding_function.EmbeddingFunction`
protocol) when creating collections with a dense vector index.

Examples:

Remote server mode (seekdb Server) - Collection management:

.. code-block:: python

    import pyseekdb
    from pyseekdb import HNSWConfiguration, Schema, VectorIndexConfig
    client = pyseekdb.Client(
        host='localhost',
        port=2881,
        tenant="sys",
        database="test",
        user="root",
        password="pass",
    )
    collection = client.get_or_create_collection(
        "my_collection",
        schema=Schema(vector_index=VectorIndexConfig(hnsw=HNSWConfiguration(dimension=384))),
    )

Remote server mode (OceanBase Server) - Collection management:

.. code-block:: python

    import pyseekdb
    from pyseekdb import HNSWConfiguration, Schema, VectorIndexConfig
    client = pyseekdb.Client(
        host='localhost',
        port=2881,
        tenant="test",
        database="test",
        user="root",
        password="pass",
    )
    collection = client.get_or_create_collection(
        "my_collection",
        schema=Schema(vector_index=VectorIndexConfig(hnsw=HNSWConfiguration(dimension=384))),
    )

Database provisioning is intentionally outside the SDK. Use your deployment or
DBA tooling to create the database before constructing ``Client``.
"""

import importlib.metadata

from .client import (
    AsyncClient,
    AsyncCollection,
    BengProperties,
    Client,
    Collection,
    EmbeddingFunction,
    FulltextIndexConfig,
    HNSWConfiguration,
    IKMode,
    IKProperties,
    IVFConfiguration,
    IVFIndexLib,
    IVFIndexType,
    K,
    Namespace,
    Ngram2Properties,
    NgramProperties,
    QueryHint,
    Schema,
    SpaceProperties,
    SparseEmbeddingFunction,
    SparseEmbeddingFunctionRegistry,
    SparseVector,
    SparseVectorIndexConfig,
    VectorIndexConfig,
    Version,
    register_embedding_function,
    register_sparse_embedding_function,
)

try:
    __version__ = importlib.metadata.version("pyseekdb")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.1.dev1"

__author__ = "OceanBase <open_oceanbase@oceanbase.com>"

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
