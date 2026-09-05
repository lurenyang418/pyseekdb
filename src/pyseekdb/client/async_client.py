"""Asynchronous client backed by an aiomysql connection pool."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import uuid
from typing import Any

from .async_collection import AsyncCollection
from .capabilities import BackendCapabilities, _first_result_value, parse_backend_identity
from .collection_lifecycle import (
    COLLECTION_CREATION_TOKEN_KEY,
    COLLECTION_STATE_CREATING,
    COLLECTION_STATE_FAILED,
    COLLECTION_STATE_READY,
    collection_state,
    parse_collection_settings,
)
from .configuration import (
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_VECTOR_DIMENSION,
    MAX_HNSW_VECTOR_DIMENSION,
    HNSWConfiguration,
)
from .embedding_function import EmbeddingFunction, EmbeddingFunctionRegistry
from .fork import build_drop_database_sql
from .kernel_errors import maybe_reraise_friendly_kernel_error
from .meta_info import CollectionFieldNames, CollectionNames
from .query_builder import (
    build_fulltext_index_sql,
    build_select_clause,
    build_vector_index_sql,
    build_where_clause,
    embed_texts,
    embedding_to_hexstring,
    normalize_collection_batch,
    normalize_include_fields,
    normalize_query_embeddings,
    process_get_row,
    process_query_row,
)
from .schema import Schema
from .sql_utils import _query_hint_to_sql, is_query_sql
from .types import _NOT_PROVIDED, K
from .validators import (
    _quote_sql_identifier,
    _validate_collection_name,
    _validate_database_name,
    _validate_include,
    _validate_pagination,
)

try:
    import aiomysql
except ImportError:  # pragma: no cover - exercised when the optional extra is absent
    aiomysql = None

logger = logging.getLogger(__name__)

_COLLECTION_READY_TIMEOUT_SECONDS = 30.0
_COLLECTION_READY_POLL_INTERVAL_SECONDS = 0.05


class AsyncClient:
    """Asynchronous client bound to one existing remote database.

    The client uses an ``aiomysql`` connection pool and never delegates network
    work to a thread. Install the optional dependency with ``pyseekdb[async]``.
    Async namespace collections are intentionally deferred; standard collections,
    database fork, and collection fork are supported in this first API version.
    """

    def __init__(
        self,
        host: str,
        port: int = 2881,
        tenant: str = "sys",
        database: str = "test",
        user: str = "root",
        password: str = "",
        charset: str = "utf8mb4",
        min_pool_size: int = 1,
        max_pool_size: int = 10,
        **kwargs: Any,
    ) -> None:
        if host is None:
            raise ValueError(
                "`host=` is required: embedded mode was removed in pyseekdb 2.0. "
                "Provide host/port (and user/password) to connect to a remote seekdb/OceanBase server."
            )
        if min_pool_size < 1 or max_pool_size < min_pool_size:
            raise ValueError("AsyncClient requires 1 <= min_pool_size <= max_pool_size")
        _validate_database_name(database)
        self.host = host
        self.port = port
        self.tenant = tenant
        self.database = database
        self.user = user
        self.password = password or os.environ.get("SEEKDB_PASSWORD", "")
        self.charset = charset
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.kwargs = kwargs
        self.full_user = f"{user}@{tenant}"
        self._pool: Any = None
        self._pool_lock: asyncio.Lock | None = None
        self._backend_capabilities: BackendCapabilities | None = None

    def _require_driver(self) -> Any:
        """Return aiomysql or raise an installation hint."""
        if aiomysql is None:
            raise ImportError("AsyncClient requires aiomysql; install it with `pip install pyseekdb[async]`")
        return aiomysql

    async def _ensure_pool(self) -> Any:
        """Lazily create and return the connection pool."""
        driver = self._require_driver()
        if self._pool_lock is None:
            self._pool_lock = asyncio.Lock()
        async with self._pool_lock:
            if self._pool is None or getattr(self._pool, "closed", False):
                connection_kwargs = dict(self.kwargs)
                connection_kwargs.pop("db", None)
                connection_kwargs.pop("database", None)
                self._pool = await driver.create_pool(
                    host=self.host,
                    port=self.port,
                    user=self.full_user,
                    password=self.password,
                    db=self.database,
                    charset=self.charset,
                    cursorclass=driver.DictCursor,
                    autocommit=True,
                    minsize=self.min_pool_size,
                    maxsize=self.max_pool_size,
                    **connection_kwargs,
                )
                logger.info("Connected to remote server pool: %s:%s/%s", self.host, self.port, self.database)
        return self._pool

    async def _execute(self, sql: str, params: list[Any] | tuple[Any, ...] | None = None) -> Any:
        """Execute SQL with a fresh pooled connection and cursor."""
        if os.environ.get("PYSEEKDB_PRINT_SQL", "").lower() in ("1", "true", "yes"):
            print(f"[pyseekdb SQL] {sql} -- params={params}", flush=True)
        pool = await self._ensure_pool()
        try:
            async with pool.acquire() as connection, connection.cursor() as cursor:
                if params is None:
                    await cursor.execute(sql)
                else:
                    await cursor.execute(sql, tuple(params))
                if cursor.description is not None or is_query_sql(sql):
                    return await cursor.fetchall()
                return None
        except Exception as exc:
            maybe_reraise_friendly_kernel_error(exc)
            raise

    async def _execute_transaction(self, statements: list[tuple[str, list[Any] | tuple[Any, ...]]]) -> None:
        """Execute a group of DML statements atomically on one pooled connection."""
        if not statements:
            raise ValueError("A transaction requires at least one statement")

        pool = await self._ensure_pool()
        try:
            async with pool.acquire() as connection:
                try:
                    await connection.begin()
                    async with connection.cursor() as cursor:
                        for sql, params in statements:
                            await cursor.execute(sql, tuple(params))
                    await connection.commit()
                except BaseException:
                    with contextlib.suppress(BaseException):
                        await connection.rollback()
                    raise
        except Exception as exc:
            maybe_reraise_friendly_kernel_error(exc)
            raise

    async def ping(self) -> bool:
        """Check whether the remote server responds."""
        rows = await self._execute("SELECT 1 AS pyseekdb_ping")
        return bool(rows and rows[0].get("pyseekdb_ping") == 1)

    def is_connected(self) -> bool:
        """Return whether an async pool has been created and remains open."""
        return self._pool is not None and not getattr(self._pool, "closed", False)

    async def close(self) -> None:
        """Close the pool and wait for all pooled connections to finish."""
        if self._pool_lock is None:
            self._pool_lock = asyncio.Lock()
        async with self._pool_lock:
            if self._pool is not None:
                pool = self._pool
                pool.close()
                await pool.wait_closed()
                self._pool = None

    async def __aenter__(self) -> AsyncClient:
        """Enter an async context manager without opening a connection eagerly."""
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close pooled resources when leaving an async context."""
        await self.close()

    def __repr__(self) -> str:
        """Return a debug-friendly representation."""
        status = "connected" if self.is_connected() else "disconnected"
        return f"<AsyncClient {self.full_user}@{self.host}:{self.port}/{self.database} status={status}>"

    async def detect_db_type_and_version(self) -> tuple[str, Any]:
        """Detect the connected server type and version."""
        version_rows = await self._execute("SELECT version() AS version")
        version_value = _first_result_value(version_rows, "version")
        if version_value and "seekdb" in version_value.lower():
            return parse_backend_identity(version_value, None)

        ob_version_rows = await self._execute("SELECT ob_version() AS ob_version")
        ob_version_value = _first_result_value(ob_version_rows, "ob_version")
        return parse_backend_identity(version_value, ob_version_value)

    async def get_backend_capabilities(self) -> BackendCapabilities:
        """Detect and cache backend capabilities for this async client."""
        if self._backend_capabilities is None:
            backend, version = await self.detect_db_type_and_version()
            self._backend_capabilities = BackendCapabilities(backend=backend, version=version)
        return self._backend_capabilities

    async def supports_fork_database(self) -> bool:
        """Return whether database fork is supported by the connected backend."""
        capabilities = await self.get_backend_capabilities()
        return capabilities.supports_fork_database

    async def _create_catalog_if_not_exists(self) -> None:
        """Create the SDK collection catalog in the bound database."""
        await self._execute(
            """CREATE TABLE IF NOT EXISTS `sdk_collections` (
                collection_id CHAR(32) PRIMARY KEY DEFAULT (replace(uuid(), '-', '')),
                collection_name STRING,
                settings JSON COMMENT "Generated by SDK, don't modify",
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_sdk_coll_name (collection_name)
            ) COMMENT='Settings of collections created by SDK' ORGANIZATION INDEX;"""
        )

    async def _get_collection_id(self, name: str) -> str:
        """Resolve a collection name to its catalog ID."""
        await self._create_catalog_if_not_exists()
        rows = await self._execute(
            "SELECT collection_id FROM `sdk_collections` WHERE collection_name = %s "
            "ORDER BY created_at, collection_id LIMIT 1",
            [name],
        )
        if not rows:
            raise ValueError(f"Collection not found: '{name}'")
        row = rows[0]
        return str(row.get("collection_id") or row.get("COLLECTION_ID") or row[0])

    async def _get_collection_catalog_row(self, name: str, *, ensure_catalog: bool = True) -> dict[str, Any] | None:
        """Fetch one collection catalog row by its validated name."""
        if ensure_catalog:
            await self._create_catalog_if_not_exists()
        rows = await self._execute(
            "SELECT collection_id, collection_name, settings FROM `sdk_collections` WHERE collection_name = %s LIMIT 1",
            [name],
        )
        return rows[0] if rows else None

    @classmethod
    def _collection_state(cls, row: dict[str, Any]) -> str:
        """Read the persisted lifecycle state, treating pre-state rows as ready."""
        return collection_state(row.get("settings") or row.get("SETTINGS"))

    async def _wait_for_collection_ready(self, name: str) -> dict[str, Any]:
        """Wait for a concurrent asynchronous collection create to finish."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _COLLECTION_READY_TIMEOUT_SECONDS
        ensure_catalog = True
        while True:
            row = await self._get_collection_catalog_row(name, ensure_catalog=ensure_catalog)
            ensure_catalog = False
            if row is None:
                raise ValueError(f"Collection not found: '{name}'")
            state = self._collection_state(row)
            if state == COLLECTION_STATE_READY:
                return row
            if state == COLLECTION_STATE_FAILED:
                raise ValueError(f"Collection '{name}' is in a failed creation state; retry creation")
            if state != COLLECTION_STATE_CREATING:
                raise ValueError(f"Collection '{name}' has unknown lifecycle state '{state}'")
            if loop.time() >= deadline:
                raise ValueError(
                    f"Collection '{name}' is still being created after {_COLLECTION_READY_TIMEOUT_SECONDS:g} seconds"
                )
            await asyncio.sleep(_COLLECTION_READY_POLL_INTERVAL_SECONDS)

    @staticmethod
    def _parse_settings(value: Any) -> dict[str, Any]:
        return parse_collection_settings(value)

    async def _describe_dimension(self, table_name: str) -> int:
        """Read a vector dimension from an existing physical collection table."""
        rows = await self._execute(f"DESCRIBE {_quote_sql_identifier(table_name)}")
        for row in rows:
            field = row.get("Field") or row.get("field")
            field_type = str(row.get("Type") or row.get("type") or "")
            if field == CollectionFieldNames.EMBEDDING:
                match = re.search(r"vector\((\d+)\)", field_type, re.IGNORECASE)
                if match:
                    return int(match.group(1))
        return DEFAULT_VECTOR_DIMENSION

    async def _describe_distance(self, table_name: str) -> str | None:
        """Read a vector distance metric from an existing collection table."""
        rows = await self._execute(f"SHOW CREATE TABLE {_quote_sql_identifier(table_name)}")
        if not rows:
            return None
        row = rows[0]
        if isinstance(row, dict):
            create_stmt = row.get("Create Table") or row.get("create table") or ""
        elif isinstance(row, (tuple, list)):
            create_stmt = row[1] if len(row) > 1 else ""
        else:
            create_stmt = str(row)
        match = re.search(r"with\s*\([^)]*distance\s*=\s*(['\"]?)(\w+)\1", str(create_stmt), re.IGNORECASE)
        if not match:
            return None
        distance = match.group(2).lower()
        return (
            {"ip": "inner_product"}.get(distance, distance)
            if distance in {"ip", "l2", "cosine", "inner_product"}
            else None
        )

    async def _collection_from_row(self, row: dict[str, Any]) -> AsyncCollection:
        name = row.get("collection_name") or row.get("COLLECTION_NAME")
        collection_id = str(row.get("collection_id") or row.get("COLLECTION_ID"))
        settings = self._parse_settings(row.get("settings") or row.get("SETTINGS"))
        if settings.get("use_namespace"):
            return AsyncCollection(
                client=self,
                name=name,
                collection_id=collection_id,
                dimension=settings.get("dimension"),
                distance=settings.get("distance"),
                use_namespace=True,
            )
        dimension = settings.get("dimension") or await self._describe_dimension(
            CollectionNames.table_name(collection_id)
        )
        distance = settings.get("distance")
        if distance is None:
            distance = await self._describe_distance(CollectionNames.table_name(collection_id))
        embedding_function = None
        ef_info = settings.get("embedding_function")
        if ef_info:
            ef_class = EmbeddingFunctionRegistry.get_class(ef_info["name"])
            if ef_class is not None:
                embedding_function = ef_class.build_from_config(ef_info.get("properties", {}))
        has_sparse_vector_index = "sparse_vector_index" in settings and settings["sparse_vector_index"] is not None
        return AsyncCollection(
            client=self,
            name=name,
            collection_id=collection_id,
            dimension=dimension,
            embedding_function=embedding_function,
            distance=distance or DEFAULT_DISTANCE_METRIC,
            has_sparse_vector_index=has_sparse_vector_index,
        )

    async def create_collection(
        self,
        name: str,
        schema: Schema | None = None,
        use_namespace: bool = False,
        partition_count: int | None = None,
    ) -> AsyncCollection:
        """Create a standard collection asynchronously."""
        _validate_collection_name(name)
        if use_namespace:
            raise NotImplementedError(
                "Async namespace collections are not implemented yet; use the synchronous Client instead"
            )
        if partition_count is not None:
            raise ValueError("partition_count is only supported for namespace-enabled collections")
        if schema is None:
            schema = Schema()
        if not isinstance(schema, Schema):
            raise TypeError(f"schema must be a Schema instance, got {type(schema).__name__}")
        if schema.sparse_vector_index is not None:
            raise NotImplementedError("Async sparse vector collections are not implemented yet")
        vector_config = schema.vector_index
        if vector_config.ivf is not None:
            raise ValueError("Async standard collections require an HNSW vector index")
        embedding_function = vector_config.embedding_function
        if embedding_function is _NOT_PROVIDED:
            raise ValueError("Invalid Schema: vector_index.embedding_function must be an EmbeddingFunction or None")
        hnsw_config = vector_config.hnsw
        if hnsw_config is None:
            dimension = (
                self._embedding_dimension(embedding_function)
                if embedding_function is not None
                else DEFAULT_VECTOR_DIMENSION
            )
            hnsw_config = HNSWConfiguration(dimension=dimension, distance=DEFAULT_DISTANCE_METRIC)
        elif embedding_function is not None:
            dimension = self._embedding_dimension(embedding_function)
            if dimension != hnsw_config.dimension:
                raise ValueError(
                    f"Schema dimension ({hnsw_config.dimension}) doesn't match embedding function dimension ({dimension})"
                )
        if hnsw_config.dimension < 1 or hnsw_config.dimension > MAX_HNSW_VECTOR_DIMENSION:
            raise ValueError(f"Dimension must be between 1 and {MAX_HNSW_VECTOR_DIMENSION}")

        await self._create_catalog_if_not_exists()
        if await self.has_collection(name):
            raise ValueError(f"Collection '{name}' already exists")
        settings: dict[str, Any] = {
            "version": 2,
            "dimension": hnsw_config.dimension,
            "distance": hnsw_config.distance,
            "state": COLLECTION_STATE_CREATING,
            "creation_token": uuid.uuid4().hex,
        }
        if embedding_function is not None and EmbeddingFunction.support_persistence(embedding_function):
            settings["embedding_function"] = {
                "name": embedding_function.name(),
                "properties": embedding_function.get_config(),
            }
        collection_id: str | None = None
        insert_attempted = False
        try:
            insert_attempted = True
            await self._execute(
                "INSERT INTO `sdk_collections` (collection_name, settings) VALUES (%s, %s)",
                [name, json.dumps(settings, ensure_ascii=False)],
            )
            collection_id = await self._get_collection_id(name)
            table_name = CollectionNames.table_name(collection_id)
            fulltext_sql = build_fulltext_index_sql(schema.fulltext_index)
            sql = f"""CREATE TABLE IF NOT EXISTS {_quote_sql_identifier(table_name)} (
                _id varbinary(512) PRIMARY KEY NOT NULL,
                document string,
                embedding vector({hnsw_config.dimension}),
                metadata json,
                FULLTEXT INDEX idx_fts(document) {fulltext_sql},
                VECTOR INDEX idx_vec (embedding) {build_vector_index_sql(hnsw_config)}
            ) ORGANIZATION = HEAP;"""
            await self._execute(sql)
            ready_settings = dict(settings)
            ready_settings["state"] = COLLECTION_STATE_READY
            ready_settings.pop(COLLECTION_CREATION_TOKEN_KEY, None)
            await self._execute(
                "UPDATE `sdk_collections` SET settings = %s WHERE collection_name = %s AND collection_id = %s "
                "AND JSON_UNQUOTE(JSON_EXTRACT(settings, '$.creation_token')) = %s "
                "AND JSON_UNQUOTE(JSON_EXTRACT(settings, '$.state')) = %s",
                [
                    json.dumps(ready_settings, ensure_ascii=False),
                    name,
                    collection_id,
                    settings[COLLECTION_CREATION_TOKEN_KEY],
                    COLLECTION_STATE_CREATING,
                ],
            )
            published = await self._get_collection_catalog_row(name, ensure_catalog=False)
            if (
                published is None
                or str(published.get("collection_id") or published.get("COLLECTION_ID")) != str(collection_id)
                or self._collection_state(published) != COLLECTION_STATE_READY
            ):
                raise RuntimeError(  # noqa: TRY301
                    f"Collection '{name}' creation ownership was lost before it became ready"
                )
        except BaseException:
            if insert_attempted:
                with contextlib.suppress(BaseException):
                    await asyncio.shield(
                        self._cleanup_failed_collection(name, collection_id, settings["creation_token"])
                    )
            raise
        return AsyncCollection(
            client=self,
            name=name,
            collection_id=collection_id,
            dimension=hnsw_config.dimension,
            embedding_function=embedding_function,
            distance=hnsw_config.distance,
        )

    async def _cleanup_failed_collection(self, name: str, collection_id: str | None, creation_token: str) -> bool:
        """Best-effort cleanup of resources owned by a failed async create."""
        row = await self._get_collection_catalog_row(name, ensure_catalog=False)
        if row is None:
            return True
        settings = self._parse_settings(row.get("settings") or row.get("SETTINGS"))
        if settings.get(COLLECTION_CREATION_TOKEN_KEY) != creation_token:
            return False
        row_collection_id = str(row.get("collection_id") or row.get("COLLECTION_ID") or "")
        if collection_id is not None and row_collection_id and str(collection_id) != row_collection_id:
            return False
        collection_id = row_collection_id or collection_id
        if collection_id:
            table_name = CollectionNames.table_name(collection_id)
            try:
                await asyncio.shield(self._execute(f"DROP TABLE IF EXISTS {_quote_sql_identifier(table_name)}"))
            except BaseException:
                logger.warning("Failed to drop incomplete collection table '%s'", table_name, exc_info=True)
                return False
        delete_sql = (
            "DELETE FROM `sdk_collections` WHERE collection_name = %s AND collection_id = %s "
            "AND JSON_UNQUOTE(JSON_EXTRACT(settings, '$.creation_token')) = %s"
        )
        try:
            await asyncio.shield(self._execute(delete_sql, [name, collection_id, creation_token]))
        except BaseException:
            logger.warning("Failed to delete incomplete collection catalog row '%s'", name, exc_info=True)
            return False
        return True

    async def cleanup_incomplete_collection(self, name: str) -> None:
        """Explicitly remove a stale or failed collection creation.

        Callers must first ensure that no live creator is still building the
        collection.  A timeout alone is not sufficient evidence that the
        creator has stopped, so this method is never called automatically.
        """
        _validate_collection_name(name)
        row = await self._get_collection_catalog_row(name)
        if row is None:
            raise ValueError(f"Collection '{name}' does not exist")
        settings = self._parse_settings(row.get("settings") or row.get("SETTINGS"))
        state = collection_state(settings)
        if state not in {COLLECTION_STATE_CREATING, COLLECTION_STATE_FAILED}:
            raise ValueError(f"Collection '{name}' is not an incomplete creation")
        creation_token = settings.get(COLLECTION_CREATION_TOKEN_KEY)
        if not creation_token:
            raise ValueError(f"Collection '{name}' has no verifiable creation owner; clean it up with DBA tooling")
        collection_id = str(row.get("collection_id") or row.get("COLLECTION_ID") or "") or None
        if not await self._cleanup_failed_collection(name, collection_id, creation_token):
            raise RuntimeError(f"Failed to clean up incomplete collection '{name}'")

    @staticmethod
    def _embedding_dimension(embedding_function: EmbeddingFunction) -> int:
        if hasattr(embedding_function, "dimension"):
            return int(embedding_function.dimension)
        values = embedding_function(["seekdb"])
        if not values or not values[0]:
            raise ValueError("Embedding function returned no values")
        return len(values[0])

    async def get_collection(self, name: str, embedding_function: Any = _NOT_PROVIDED) -> AsyncCollection:
        """Get an existing collection asynchronously."""
        _validate_collection_name(name)
        row = await self._wait_for_collection_ready(name)
        collection = await self._collection_from_row(row)
        if embedding_function is not _NOT_PROVIDED:
            collection._embedding_function = embedding_function
        return collection

    async def has_collection(self, name: str) -> bool:
        """Return whether a collection exists in the bound database."""
        _validate_collection_name(name)
        row = await self._get_collection_catalog_row(name)
        return row is not None and self._collection_state(row) == COLLECTION_STATE_READY

    async def list_collections(self) -> list[AsyncCollection]:
        """List standard and namespace collection handles."""
        await self._create_catalog_if_not_exists()
        rows = await self._execute("SELECT collection_id, collection_name, settings FROM `sdk_collections`")
        return [
            await self._collection_from_row(row)
            for row in rows
            if self._collection_state(row) == COLLECTION_STATE_READY
        ]

    async def get_or_create_collection(
        self,
        name: str,
        schema: Schema | None = None,
        use_namespace: bool = False,
    ) -> AsyncCollection:
        """Get a collection or create it if absent."""
        _validate_collection_name(name)
        embedding_function = schema.vector_index.embedding_function if schema is not None else _NOT_PROVIDED
        if await self.has_collection(name):
            return await self.get_collection(name, embedding_function=embedding_function)
        try:
            return await self.create_collection(name, schema=schema, use_namespace=use_namespace)
        except Exception as exc:
            if "already exists" in str(exc).lower() or "duplicate" in str(exc).lower():
                try:
                    return await self.get_collection(name, embedding_function=embedding_function)
                except ValueError as recovery_exc:
                    if str(recovery_exc).startswith("Collection not found:"):
                        return await self.create_collection(name, schema=schema, use_namespace=use_namespace)
                    raise
            raise

    async def delete_collection(self, name: str) -> None:
        """Delete a standard collection and its SDK metadata."""
        _validate_collection_name(name)
        row = await self._get_collection_catalog_row(name)
        if row is not None and self._collection_state(row) in {COLLECTION_STATE_CREATING, COLLECTION_STATE_FAILED}:
            raise ValueError(
                f"Collection '{name}' is incomplete; call cleanup_incomplete_collection() "
                "after confirming the creator has stopped"
            )
        collection = await self.get_collection(name)
        if collection.use_namespace:
            raise NotImplementedError("Async namespace collection deletion is not implemented yet")
        await self._execute(f"DROP TABLE IF EXISTS {_quote_sql_identifier(CollectionNames.table_name(collection.id))}")
        await self._execute("DELETE FROM `sdk_collections` WHERE collection_name = %s", [name])

    async def count_collection(self) -> int:
        """Return the number of collection catalog entries."""
        await self._create_catalog_if_not_exists()
        rows = await self._execute("SELECT COUNT(*) AS count FROM `sdk_collections`")
        value = rows[0].get("count", rows[0].get("COUNT(*)"))
        return int(value)

    async def _collection_add(
        self,
        collection: AsyncCollection,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None,
        metadatas: dict | list[dict] | None,
        documents: str | list[str] | None,
        **kwargs: Any,
    ) -> None:
        id_list, embedding_list, metadata_list, document_list = normalize_collection_batch(
            ids, embeddings, metadatas, documents, require_values=True
        )
        if embedding_list is None and document_list is not None:
            embedding_list = (
                list(collection.embedding_function(document_list)) if collection.embedding_function else None
            )
            if embedding_list is None:
                raise ValueError("Documents require an embedding function")
        if embedding_list is None and document_list is None:
            raise ValueError("Add requires embeddings or documents")
        table = _quote_sql_identifier(CollectionNames.table_name(collection.id))
        statements: list[tuple[str, list[Any]]] = []
        for index, record_id in enumerate(id_list):
            vector_sql = "NULL" if embedding_list is None else embedding_to_hexstring(embedding_list[index])
            statements.append((
                f"INSERT INTO {table} (_id, document, embedding, metadata) VALUES (CAST(%s AS BINARY), %s, {vector_sql}, %s)",
                [
                    record_id,
                    document_list[index] if document_list is not None else None,
                    json.dumps(metadata_list[index], ensure_ascii=False) if metadata_list is not None else None,
                ],
            ))
        await self._execute_transaction(statements)

    async def _collection_update(
        self,
        collection: AsyncCollection,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None,
        metadatas: dict | list[dict] | None,
        documents: str | list[str] | None,
        **kwargs: Any,
    ) -> None:
        id_list, embedding_list, metadata_list, document_list = normalize_collection_batch(
            ids, embeddings, metadatas, documents, require_values=True
        )
        if embedding_list is None and document_list is not None:
            embedding_list = (
                list(collection.embedding_function(document_list)) if collection.embedding_function else None
            )
            if embedding_list is None:
                raise ValueError("Documents require an embedding function")
        table = _quote_sql_identifier(CollectionNames.table_name(collection.id))
        statements: list[tuple[str, list[Any]]] = []
        for index, record_id in enumerate(id_list):
            assignments: list[str] = []
            params: list[Any] = []
            if document_list is not None:
                assignments.append("document = %s")
                params.append(document_list[index])
            if embedding_list is not None:
                assignments.append(f"embedding = {embedding_to_hexstring(embedding_list[index])}")
            if metadata_list is not None:
                assignments.append("metadata = %s")
                params.append(json.dumps(metadata_list[index], ensure_ascii=False))
            statements.append((
                f"UPDATE {table} SET {', '.join(assignments)} WHERE _id = CAST(%s AS BINARY)",
                [*params, record_id],
            ))
        await self._execute_transaction(statements)

    async def _collection_upsert(self, collection: AsyncCollection, **kwargs: Any) -> None:
        ids, embeddings, metadatas, documents = normalize_collection_batch(
            kwargs.pop("ids"),
            kwargs.pop("embeddings"),
            kwargs.pop("metadatas"),
            kwargs.pop("documents"),
            require_values=True,
        )
        if embeddings is None and documents is not None:
            if collection.embedding_function is None:
                raise ValueError("Documents require an embedding function")
            embeddings = list(collection.embedding_function(documents))

        table = _quote_sql_identifier(CollectionNames.table_name(collection.id))
        statements: list[tuple[str, list[Any]]] = []
        for index, record_id in enumerate(ids):
            embedding_sql = "NULL" if embeddings is None else embedding_to_hexstring(embeddings[index])
            update_clauses: list[str] = []
            if documents is not None:
                update_clauses.append("document = VALUES(document)")
            if embeddings is not None:
                update_clauses.append("embedding = VALUES(embedding)")
            if metadatas is not None:
                update_clauses.append("metadata = VALUES(metadata)")
            if not update_clauses:
                update_clauses.append("_id = _id")
            statements.append((
                f"INSERT INTO {table} (_id, document, embedding, metadata) "
                f"VALUES (CAST(%s AS BINARY), %s, {embedding_sql}, %s) "
                f"ON DUPLICATE KEY UPDATE {', '.join(update_clauses)}",
                [
                    record_id,
                    documents[index] if documents is not None else None,
                    json.dumps(metadatas[index], ensure_ascii=False) if metadatas is not None else None,
                ],
            ))
        await self._execute_transaction(statements)

    def _build_where(
        self,
        ids: str | list[str] | None,
        where: dict[str, Any] | None,
        where_document: dict[str, Any] | None,
    ) -> tuple[str, list[Any]]:
        id_list = None if ids is None else ([ids] if isinstance(ids, str) else list(ids))
        where_clause, params = build_where_clause(where, where_document, id_list)
        return where_clause.removeprefix("WHERE ") or "1=1", params

    async def _collection_delete(self, collection: AsyncCollection, **kwargs: Any) -> None:
        ids = kwargs.pop("ids")
        where = kwargs.pop("where")
        where_document = kwargs.pop("where_document")
        if ids is not None and not ids:
            raise ValueError("ids must not be empty")
        if ids is None and not where and not where_document:
            raise ValueError("At least one of ids, where, or where_document must be provided")
        where_sql, params = self._build_where(ids, where, where_document)
        await self._execute(
            f"DELETE FROM {_quote_sql_identifier(CollectionNames.table_name(collection.id))} WHERE {where_sql}",
            params,
        )

    @staticmethod
    def _include_fields(include: list[str] | None) -> set[str]:
        _validate_include(include)
        return set(normalize_include_fields(include))

    async def _collection_get(self, collection: AsyncCollection, **kwargs: Any) -> dict[str, Any]:
        ids = kwargs.pop("ids")
        where = kwargs.pop("where")
        where_document = kwargs.pop("where_document")
        limit_value = kwargs.pop("limit")
        offset_value = kwargs.pop("offset")
        _validate_pagination(limit_value, offset_value)
        limit = 100 if limit_value is None else limit_value
        offset = 0 if offset_value is None else offset_value
        include = kwargs.pop("include")
        query_hint = kwargs.pop("query_hint")
        fields = self._include_fields(include)
        include_fields = dict.fromkeys(fields, True)
        select = build_select_clause(include_fields)
        where_sql, params = self._build_where(ids, where, where_document)
        hint = _query_hint_to_sql(query_hint, CollectionNames.table_name(collection.id))
        rows = await self._execute(
            f"SELECT {hint} {select} FROM {_quote_sql_identifier(CollectionNames.table_name(collection.id))} "
            f"WHERE {where_sql} LIMIT %s OFFSET %s",
            [*params, limit, offset],
        )
        result: dict[str, Any] = {"ids": []}
        if "documents" in fields or "document" in fields:
            result["documents"] = []
        if "metadatas" in fields or "metadata" in fields:
            result["metadatas"] = []
        if "embeddings" in fields or "embedding" in fields:
            result["embeddings"] = []
        for row in rows:
            processed = process_get_row(row, include_fields)
            result["ids"].append(processed["id"])
            if "documents" in result:
                result["documents"].append(processed["document"])
            if "metadatas" in result:
                result["metadatas"].append(processed["metadata"] or {})
            if "embeddings" in result:
                result["embeddings"].append(processed["embedding"])
        return result

    async def _collection_query(self, collection: AsyncCollection, **kwargs: Any) -> dict[str, Any]:
        query_embeddings = kwargs.pop("query_embeddings")
        query_texts = kwargs.pop("query_texts")
        n_results = kwargs.pop("n_results")
        where = kwargs.pop("where")
        where_document = kwargs.pop("where_document")
        include = kwargs.pop("include")
        query_key = kwargs.pop("query_key")
        query_hint = kwargs.pop("query_hint")
        if query_key is not None and (query_key is K.SPARSE_EMBEDDING or query_key == K.SPARSE_EMBEDDING.name):
            raise NotImplementedError("Async sparse vector queries are not implemented yet")
        embedding_function = collection.embedding_function
        if query_embeddings is None and query_texts is not None:
            if embedding_function is None:
                raise ValueError("query_texts requires an embedding function")
            query_embeddings = embed_texts(query_texts, embedding_function)
        if query_embeddings is None:
            raise ValueError("Provide query_embeddings or query_texts")
        vectors = normalize_query_embeddings(query_embeddings)
        fields = self._include_fields(include)
        include_fields = dict.fromkeys(fields, True)
        select = build_select_clause(include_fields)
        where_sql, params = self._build_where(None, where, where_document)
        distance = kwargs.pop("distance", collection.distance or DEFAULT_DISTANCE_METRIC)
        distance_func = {"l2": "l2_distance", "cosine": "cosine_distance", "inner_product": "inner_product"}.get(
            distance, "l2_distance"
        )
        table = _quote_sql_identifier(CollectionNames.table_name(collection.id))
        all_results = {"ids": [], "distances": []}
        if "documents" in fields or "document" in fields:
            all_results["documents"] = []
        if "metadatas" in fields or "metadata" in fields:
            all_results["metadatas"] = []
        if "embeddings" in fields or "embedding" in fields:
            all_results["embeddings"] = []
        hint = _query_hint_to_sql(query_hint, CollectionNames.table_name(collection.id))
        for vector in vectors:
            vector_sql = embedding_to_hexstring(vector)
            rows = await self._execute(
                f"SELECT {hint} {select}, {distance_func}(embedding, {vector_sql}) AS distance "
                f"FROM {table} WHERE {where_sql} ORDER BY {distance_func}(embedding, {vector_sql}) APPROXIMATE LIMIT %s",
                [*params, n_results],
            )
            result_ids: list[str] = []
            result_distances: list[float] = []
            result_documents: list[Any] = []
            result_metadatas: list[Any] = []
            result_embeddings: list[Any] = []
            for row in rows:
                processed = process_query_row(row, include_fields)
                result_ids.append(processed["_id"])
                result_distances.append(processed["distance"])
                if "documents" in all_results:
                    result_documents.append(processed.get("document"))
                if "metadatas" in all_results:
                    result_metadatas.append(processed.get("metadata") or {})
                if "embeddings" in all_results:
                    result_embeddings.append(processed.get("embedding"))
            all_results["ids"].append(result_ids)
            all_results["distances"].append(result_distances)
            if "documents" in all_results:
                all_results["documents"].append(result_documents)
            if "metadatas" in all_results:
                all_results["metadatas"].append(result_metadatas)
            if "embeddings" in all_results:
                all_results["embeddings"].append(result_embeddings)
        return all_results

    async def _collection_count(self, collection: AsyncCollection) -> int:
        rows = await self._execute(
            f"SELECT COUNT(*) AS count FROM {_quote_sql_identifier(CollectionNames.table_name(collection.id))}"
        )
        value = rows[0].get("count", rows[0].get("COUNT(*)"))
        return int(value)

    async def _collection_fork(self, collection: AsyncCollection, forked_name: str) -> AsyncCollection:
        """Fork a standard collection using the seekdb FORK TABLE statement."""
        _validate_collection_name(forked_name)
        capabilities = await self.get_backend_capabilities()
        if not capabilities.supports_fork_table:
            raise ValueError("Fork collection is not supported by this backend (requires seekdb >= 1.1.0)")
        if await self.has_collection(forked_name):
            raise ValueError(f"Collection '{forked_name}' already exists")
        rows = await self._execute(
            "SELECT settings FROM `sdk_collections` WHERE collection_name = %s",
            [collection.name],
        )
        source_settings = self._parse_settings((rows[0].get("settings") or rows[0].get("SETTINGS")) if rows else {})
        creation_token = uuid.uuid4().hex
        settings = dict(source_settings)
        settings["state"] = COLLECTION_STATE_CREATING
        settings[COLLECTION_CREATION_TOKEN_KEY] = creation_token
        forked_id: str | None = None
        insert_attempted = False
        try:
            insert_attempted = True
            await self._execute(
                "INSERT INTO `sdk_collections` (collection_name, settings) VALUES (%s, %s)",
                [forked_name, json.dumps(settings, ensure_ascii=False)],
            )
            forked_id = await self._get_collection_id(forked_name)
            await self._execute(
                f"FORK TABLE {_quote_sql_identifier(CollectionNames.table_name(collection.id))} "
                f"TO {_quote_sql_identifier(CollectionNames.table_name(forked_id))}"
            )
            ready_settings = dict(settings)
            ready_settings["state"] = COLLECTION_STATE_READY
            ready_settings.pop(COLLECTION_CREATION_TOKEN_KEY, None)
            await self._execute(
                "UPDATE `sdk_collections` SET settings = %s WHERE collection_name = %s AND collection_id = %s "
                "AND JSON_UNQUOTE(JSON_EXTRACT(settings, '$.creation_token')) = %s "
                "AND JSON_UNQUOTE(JSON_EXTRACT(settings, '$.state')) = %s",
                [
                    json.dumps(ready_settings, ensure_ascii=False),
                    forked_name,
                    forked_id,
                    creation_token,
                    COLLECTION_STATE_CREATING,
                ],
            )
            published = await self._get_collection_catalog_row(forked_name, ensure_catalog=False)
            if (
                published is None
                or str(published.get("collection_id") or published.get("COLLECTION_ID")) != str(forked_id)
                or self._collection_state(published) != COLLECTION_STATE_READY
            ):
                raise RuntimeError(  # noqa: TRY301
                    f"Collection '{forked_name}' creation ownership was lost before it became ready"
                )
        except BaseException:
            if insert_attempted:
                with contextlib.suppress(BaseException):
                    await asyncio.shield(self._cleanup_failed_collection(forked_name, forked_id, creation_token))
            raise
        return await self.get_collection(forked_name, embedding_function=collection.embedding_function)

    async def refresh_index(self) -> None:
        """Refresh vector indexes when supported by the backend."""
        capabilities = await self.get_backend_capabilities()
        if capabilities.supports_refresh_index:
            await self._execute("CALL dbms_index_manager.refresh();")

    async def fork_database(self, destination_name: str) -> AsyncClient:
        """Fork this client's database and return an async client for the copy."""
        _validate_database_name(destination_name)
        if destination_name == self.database:
            raise ValueError("Fork destination must differ from the source database")
        if not await self.supports_fork_database():
            raise ValueError("Fork database is not supported by this backend (requires seekdb >= 1.2.0)")
        try:
            await self._execute(
                f"FORK DATABASE {_quote_sql_identifier(self.database)} TO {_quote_sql_identifier(destination_name)}"
            )
        except Exception as exc:
            args = getattr(exc, "args", ())
            if args and isinstance(args[0], int) and args[0] == 1007:
                raise ValueError(f"Database '{destination_name}' already exists") from exc
            if "already exists" in str(exc).lower() and "database" in str(exc).lower():
                raise ValueError(f"Database '{destination_name}' already exists") from exc
            raise
        forked = type(self)(
            host=self.host,
            port=self.port,
            tenant=self.tenant,
            database=destination_name,
            user=self.user,
            password=self.password,
            charset=self.charset,
            min_pool_size=self.min_pool_size,
            max_pool_size=self.max_pool_size,
            **self.kwargs,
        )
        forked._fork_parent = self
        return forked

    async def destroy(self) -> None:
        """Destroy this client's forked database and close its pool.

        Only clients returned by :meth:`fork_database` can be destroyed this way;
        ordinary database provisioning and deletion remain outside the SDK.
        """
        parent = getattr(self, "_fork_parent", None)
        if parent is None:
            raise ValueError("Only a client returned by fork_database() can destroy its database")
        await self.close()
        await parent._execute(build_drop_database_sql(self.database))
        self._fork_parent = None


__all__ = ["AsyncClient"]
