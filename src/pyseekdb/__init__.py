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

Admin client - Database management:

.. code-block:: python

    import pyseekdb
    admin = pyseekdb.AdminClient(host="localhost", port=2881, user="root", password="pass")
    admin.create_database("new_db")
    databases = admin.list_databases()
"""

import importlib.metadata

from .client import (
    AdminAPI,
    AdminClient,
    BaseClient,
    BaseConnection,
    BengProperties,
    Client,
    Database,
    EmbeddingFunction,
    FulltextIndexConfig,
    HNSWConfiguration,
    IKMode,
    IKProperties,
    IVFConfiguration,
    IVFIndexLib,
    IVFIndexType,
    K,
    Ngram2Properties,
    NgramProperties,
    RemoteServerClient,
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
from .client.collection import Collection
from .client.namespace import Namespace

try:
    __version__ = importlib.metadata.version("pyseekdb")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.1.dev1"

__author__ = "OceanBase <open_oceanbase@oceanbase.com>"

__all__ = [
    "AdminAPI",
    "AdminClient",
    "BaseClient",
    "BaseConnection",
    "BengProperties",
    "Client",
    "Collection",
    "Database",
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
    "RemoteServerClient",
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
