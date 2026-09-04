"""
pyseekdb client module

Provides client and admin factory functions with strict separation:

Collection Management (Client):
- Client() - Factory for remote server mode (seekdb Server or OceanBase Server)
- Returns: _ClientProxy (collection operations only)

Database Management (AdminAPI):
- AdminClient() - Factory for remote server mode (seekdb Server or OceanBase Server)
- Returns: _AdminClientProxy (database operations only)

All factories use the underlying RemoteServerClient implementation, which connects to a
seekdb/OceanBase server over pymysql. Embedded mode (pylibseekdb) was removed in pyseekdb 2.0.
"""

import logging
import os
from typing import Any

from .admin_client import AdminAPI, _AdminClientProxy, _ClientProxy
from .base_connection import BaseConnection
from .client_base import BaseClient
from .client_seekdb_server import RemoteServerClient
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
from .database import Database
from .embedding_function import (
    EmbeddingFunction,
    register_embedding_function,
)
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

logger = logging.getLogger(__name__)


def _resolve_password(password: str) -> str:
    """Get password from env if not provided (keeps existing behavior)."""
    return password or os.environ.get("SEEKDB_PASSWORD", "")


def _create_server_client(
    *,
    host: str | None,
    port: int | None,
    tenant: str,
    database: str,
    user: str | None,
    password: str,
    is_admin: bool,
    **kwargs: Any,
) -> BaseClient:
    """
    Create the underlying remote server client (single change point).

    Shared between ``Client()`` and ``AdminClient()``. Embedded mode was removed in pyseekdb 2.0;
    ``host`` is required.
    """
    if host is None:
        raise ValueError(
            "`host=` is required: embedded mode was removed in pyseekdb 2.0. "
            "Provide host/port (and user/password) to connect to a remote seekdb/OceanBase server."
        )

    if port is None:
        port = 2881
    if user is None:
        user = "root"
    if is_admin:
        logger.debug(f"Creating remote server admin client: {user}@{tenant}@{host}:{port}")
    else:
        logger.debug(f"Creating remote server client: {user}@{tenant}@{host}:{port}/{database}")
    return RemoteServerClient(
        host=host,
        port=port,
        tenant=tenant,
        database=database,
        user=user,
        password=password,
        **kwargs,
    )


__all__ = [
    "AdminAPI",
    "AdminClient",
    "BaseClient",
    "BaseConnection",
    "BengProperties",
    "Client",
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
    "Ngram2Properties",
    "NgramProperties",
    "QueryHint",
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


def Client(
    host: str | None = None,
    port: int | None = None,
    tenant: str = "sys",
    database: str = "test",
    user: str | None = None,
    password: str = "",  # Can be retrieved from SEEKDB_PASSWORD environment variable
    **kwargs,
) -> _ClientProxy:
    """
    Remote server client factory (returns ClientProxy for collection operations only).

    Connects to a remote seekdb Server or OceanBase Server over pymysql. Since pyseekdb 2.0
    embedded mode has been removed; ``host`` (and typically ``port``, ``user``, ``password``) is
    required.

    Returns a ClientProxy that only exposes collection operations.
    For database management, use AdminClient().

    Args:
        host: server address (required).
        port: server port (default 2881).
        tenant: tenant name (default "sys" for seekdb Server, "test" for OceanBase).
        database: database name.
        user: username (without tenant suffix).
        password: password. If not provided, will be retrieved from SEEKDB_PASSWORD environment variable.
        **kwargs: other parameters.

    Returns:
        _ClientProxy: A proxy that only exposes collection operations.

    Examples:
        >>> # Remote server mode (seekdb Server)
        >>> from pyseekdb import HNSWConfiguration, Schema, VectorIndexConfig
        >>> client = Client(
        ...     host='localhost',
        ...     port=2881,
        ...     tenant="sys",
        ...     database="test",
        ...     user="root",
        ...     password="pass",
        ... )
        >>> collection = client.get_or_create_collection(
        ...     "my_collection",
        ...     schema=Schema(vector_index=VectorIndexConfig(hnsw=HNSWConfiguration(dimension=384))),
        ... )

        >>> # Remote server mode (OceanBase Server)
        >>> from pyseekdb import HNSWConfiguration, Schema, VectorIndexConfig
        >>> client = Client(
        ...     host='localhost',
        ...     port=2881,
        ...     tenant="test",
        ...     database="test",
        ...     user="root",
        ...     password="pass",
        ... )
    """
    password = _resolve_password(password)
    server = _create_server_client(
        host=host,
        port=port,
        tenant=tenant,
        database=database,
        user=user,
        password=password,
        is_admin=False,
        **kwargs,
    )

    # Return ClientProxy (only exposes collection operations)
    return _ClientProxy(server=server)


def AdminClient(
    host: str | None = None,
    port: int | None = None,
    tenant: str = "sys",
    user: str | None = None,
    password: str = "",  # Can be retrieved from SEEKDB_PASSWORD environment variable
    **kwargs,
) -> _AdminClientProxy:
    """
    Remote server admin client factory (proxy pattern).

    Connects to a remote seekdb Server or OceanBase Server over pymysql. Since pyseekdb 2.0
    embedded mode has been removed; ``host`` (and typically ``port``, ``user``, ``password``) is
    required.

    Returns a lightweight AdminClient proxy that only exposes database operations.
    For collection management, use Client().

    Args:
        host: server address (required).
        port: server port (default 2881).
        tenant: tenant name (default "sys" for seekdb Server, "test" for OceanBase).
        user: username (without tenant suffix).
        password: password. If not provided, will be retrieved from SEEKDB_PASSWORD environment variable.
        **kwargs: other parameters.

    Returns:
        _AdminClientProxy: A proxy that only exposes database operations.

    Examples:
        >>> # Remote server mode (seekdb Server)
        >>> admin = AdminClient(
        ...     host='localhost',
        ...     port=2881,
        ...     tenant="sys",
        ...     user="root",
        ...     password="pass",
        ... )
        >>> admin.create_database("new_db")

        >>> # Remote server mode (OceanBase Server)
        >>> admin = AdminClient(
        ...     host='localhost',
        ...     port=2881,
        ...     tenant="test",
        ...     user="root",
        ...     password="pass",
        ... )
    """
    password = _resolve_password(password)
    # Keep existing semantics: admin operations always use system database.
    server = _create_server_client(
        host=host,
        port=port,
        tenant=tenant,
        database="information_schema",
        user=user,
        password=password,
        is_admin=True,
        **kwargs,
    )

    # Return AdminClient proxy (only exposes database operations)
    return _AdminClientProxy(server=server)
