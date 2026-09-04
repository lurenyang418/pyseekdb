"""Asynchronous client backed by an aiomysql connection pool."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from typing import Any

from .async_collection import AsyncCollection
from .capabilities import BackendCapabilities
from .configuration import (
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_VECTOR_DIMENSION,
    MAX_HNSW_VECTOR_DIMENSION,
    HNSWConfiguration,
)
from .embedding_function import EmbeddingFunction, EmbeddingFunctionRegistry
from .filters import FilterBuilder
from .kernel_errors import maybe_reraise_friendly_kernel_error
from .meta_info import CollectionFieldNames, CollectionNames
from .query_builder import (
    build_fulltext_index_sql,
    build_vector_index_sql,
    embedding_to_hexstring,
)
from .schema import Schema
from .sql_utils import _query_hint_to_sql, is_query_sql
from .types import _NOT_PROVIDED, K
from .validators import _quote_sql_identifier, _validate_collection_name, _validate_database_name, _validate_include

try:
    import aiomysql
except ImportError:  # pragma: no cover - exercised when the optional extra is absent
    aiomysql = None

logger = logging.getLogger(__name__)


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
        self._backend_capabilities: BackendCapabilities | None = None

    def _require_driver(self) -> Any:
        """Return aiomysql or raise an installation hint."""
        if aiomysql is None:
            raise ImportError("AsyncClient requires aiomysql; install it with `pip install pyseekdb[async]`")
        return aiomysql

    async def _ensure_pool(self) -> Any:
        """Lazily create and return the connection pool."""
        driver = self._require_driver()
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

    async def ping(self) -> bool:
        """Check whether the remote server responds."""
        rows = await self._execute("SELECT 1 AS pyseekdb_ping")
        return bool(rows and rows[0].get("pyseekdb_ping") == 1)

    def is_connected(self) -> bool:
        """Return whether an async pool has been created and remains open."""
        return self._pool is not None and not getattr(self._pool, "closed", False)

    async def close(self) -> None:
        """Close the pool and wait for all pooled connections to finish."""
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
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
        rows = await self._execute("SELECT version() AS version")
        version_value = rows[0].get("version") if rows else None
        version_text = str(version_value or "").strip()
        if "seekdb" in version_text.lower():
            match = re.search(r"seekdb[-\s]v?(\d+(?:\.\d+){2,3})", version_text, re.IGNORECASE)
            if match:
                from .version import Version

                return "seekdb", Version(match.group(1))

        rows = await self._execute("SELECT ob_version() AS ob_version")
        version_value = rows[0].get("ob_version") if rows else None
        version_text = str(version_value or "").strip()
        if version_text:
            from .version import Version

            parts = re.findall(r"\d+", version_text)
            if len(parts) >= 3:
                return "oceanbase", Version(".".join(parts[:4]))
        raise ValueError("Unable to detect database type or server version")

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

    @staticmethod
    def _parse_settings(value: Any) -> dict[str, Any]:
        if isinstance(value, str):
            return json.loads(value) if value else {}
        return value or {}

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
        embedding_function = None
        ef_info = settings.get("embedding_function")
        if ef_info:
            ef_class = EmbeddingFunctionRegistry.get_class(ef_info["name"])
            if ef_class is not None:
                embedding_function = ef_class.build_from_config(ef_info.get("properties", {}))
        return AsyncCollection(
            client=self,
            name=name,
            collection_id=collection_id,
            dimension=dimension,
            embedding_function=embedding_function,
            distance=settings.get("distance", DEFAULT_DISTANCE_METRIC),
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
        }
        if embedding_function is not None and EmbeddingFunction.support_persistence(embedding_function):
            settings["embedding_function"] = {
                "name": embedding_function.name(),
                "properties": embedding_function.get_config(),
            }
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
        try:
            await self._execute(sql)
        except Exception:
            with contextlib.suppress(Exception):
                await self._execute("DELETE FROM `sdk_collections` WHERE collection_name = %s", [name])
            raise
        return AsyncCollection(
            client=self,
            name=name,
            collection_id=collection_id,
            dimension=hnsw_config.dimension,
            embedding_function=embedding_function,
            distance=hnsw_config.distance,
        )

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
        await self._create_catalog_if_not_exists()
        rows = await self._execute(
            "SELECT collection_id, collection_name, settings FROM `sdk_collections` WHERE collection_name = %s",
            [name],
        )
        if not rows:
            raise ValueError(f"Collection not found: '{name}'")
        collection = await self._collection_from_row(rows[0])
        if embedding_function is not _NOT_PROVIDED:
            collection._embedding_function = embedding_function
        return collection

    async def has_collection(self, name: str) -> bool:
        """Return whether a collection exists in the bound database."""
        _validate_collection_name(name)
        await self._create_catalog_if_not_exists()
        rows = await self._execute("SELECT 1 FROM `sdk_collections` WHERE collection_name = %s LIMIT 1", [name])
        return bool(rows)

    async def list_collections(self) -> list[AsyncCollection]:
        """List standard and namespace collection handles."""
        await self._create_catalog_if_not_exists()
        rows = await self._execute("SELECT collection_id, collection_name, settings FROM `sdk_collections`")
        return [await self._collection_from_row(row) for row in rows]

    async def get_or_create_collection(
        self,
        name: str,
        schema: Schema | None = None,
        use_namespace: bool = False,
    ) -> AsyncCollection:
        """Get a collection or create it if absent."""
        if await self.has_collection(name):
            return await self.get_collection(name)
        try:
            return await self.create_collection(name, schema=schema, use_namespace=use_namespace)
        except Exception as exc:
            if "already exists" in str(exc).lower() or "duplicate" in str(exc).lower():
                return await self.get_collection(name)
            raise

    async def delete_collection(self, name: str) -> None:
        """Delete a standard collection and its SDK metadata."""
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

    @staticmethod
    def _normalize_batch(
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None,
        metadatas: dict | list[dict] | None,
        documents: str | list[str] | None,
    ) -> tuple[list[str], list[list[float]] | None, list[dict] | None, list[str] | None]:
        id_list = [ids] if isinstance(ids, str) else list(ids)
        if not id_list:
            raise ValueError("ids must not be empty")
        document_list = [documents] if isinstance(documents, str) else documents
        metadata_list = [metadatas] if isinstance(metadatas, dict) else metadatas
        embedding_list = embeddings
        if embedding_list is not None and (not embedding_list or not isinstance(embedding_list[0], list)):
            embedding_list = [embedding_list]  # type: ignore[list-item]
        for label, values in (
            ("documents", document_list),
            ("metadatas", metadata_list),
            ("embeddings", embedding_list),
        ):
            if values is not None and len(values) != len(id_list):
                raise ValueError(f"Number of {label} ({len(values)}) does not match number of ids ({len(id_list)})")
        if embedding_list is None and document_list is None and metadata_list is None:
            raise ValueError("Provide embeddings, documents, or metadatas")
        return id_list, embedding_list, metadata_list, document_list

    async def _collection_add(
        self,
        collection: AsyncCollection,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None,
        metadatas: dict | list[dict] | None,
        documents: str | list[str] | None,
        **kwargs: Any,
    ) -> None:
        id_list, embedding_list, metadata_list, document_list = self._normalize_batch(
            ids, embeddings, metadatas, documents
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
        for index, record_id in enumerate(id_list):
            vector_sql = "NULL" if embedding_list is None else embedding_to_hexstring(embedding_list[index])
            await self._execute(
                f"INSERT INTO {table} (_id, document, embedding, metadata) VALUES (CAST(%s AS BINARY), %s, {vector_sql}, %s)",
                [
                    record_id,
                    document_list[index] if document_list else None,
                    json.dumps(metadata_list[index], ensure_ascii=False) if metadata_list else None,
                ],
            )

    async def _collection_update(
        self,
        collection: AsyncCollection,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None,
        metadatas: dict | list[dict] | None,
        documents: str | list[str] | None,
        **kwargs: Any,
    ) -> None:
        id_list, embedding_list, metadata_list, document_list = self._normalize_batch(
            ids, embeddings, metadatas, documents
        )
        if embedding_list is None and document_list is not None:
            embedding_list = (
                list(collection.embedding_function(document_list)) if collection.embedding_function else None
            )
            if embedding_list is None:
                raise ValueError("Documents require an embedding function")
        table = _quote_sql_identifier(CollectionNames.table_name(collection.id))
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
            await self._execute(
                f"UPDATE {table} SET {', '.join(assignments)} WHERE _id = CAST(%s AS BINARY)",
                [*params, record_id],
            )

    async def _collection_upsert(self, collection: AsyncCollection, **kwargs: Any) -> None:
        ids, embeddings, metadatas, documents = self._normalize_batch(
            kwargs.pop("ids"), kwargs.pop("embeddings"), kwargs.pop("metadatas"), kwargs.pop("documents")
        )
        for index, record_id in enumerate(ids):
            exists = await self._collection_get(collection, ids=record_id, include=[])
            operation = self._collection_update if exists["ids"] else self._collection_add
            await operation(
                collection,
                ids=record_id,
                embeddings=embeddings[index] if embeddings else None,
                metadatas=metadatas[index] if metadatas else None,
                documents=documents[index] if documents else None,
                **kwargs,
            )

    def _build_where(
        self,
        ids: str | list[str] | None,
        where: dict[str, Any] | None,
        where_document: dict[str, Any] | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if ids is not None:
            id_list = [ids] if isinstance(ids, str) else list(ids)
            clauses.append("_id IN (" + ", ".join("CAST(%s AS BINARY)" for _ in id_list) + ")")
            params.extend(id_list)
        if where:
            clause, values = FilterBuilder.build_metadata_filter(where)
            clauses.append(clause)
            params.extend(values)
        if where_document:
            clause, values = FilterBuilder.build_document_filter(where_document)
            clauses.append(clause)
            params.extend(values)
        return (" AND ".join(clauses) if clauses else "1=1"), params

    async def _collection_delete(self, collection: AsyncCollection, **kwargs: Any) -> None:
        ids = kwargs.pop("ids")
        where = kwargs.pop("where")
        where_document = kwargs.pop("where_document")
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
        if include is None:
            return {"documents", "metadatas"}
        return set(include)

    async def _collection_get(self, collection: AsyncCollection, **kwargs: Any) -> dict[str, Any]:
        ids = kwargs.pop("ids")
        where = kwargs.pop("where")
        where_document = kwargs.pop("where_document")
        limit_value = kwargs.pop("limit")
        offset_value = kwargs.pop("offset")
        limit = 100 if limit_value is None else limit_value
        offset = 0 if offset_value is None else offset_value
        include = kwargs.pop("include")
        query_hint = kwargs.pop("query_hint")
        fields = self._include_fields(include)
        select = ["_id"]
        if "documents" in fields or "document" in fields:
            select.append("document")
        if "metadatas" in fields or "metadata" in fields:
            select.append("metadata")
        if "embeddings" in fields or "embedding" in fields:
            select.append("embedding")
        where_sql, params = self._build_where(ids, where, where_document)
        hint = _query_hint_to_sql(query_hint, CollectionNames.table_name(collection.id))
        rows = await self._execute(
            f"SELECT {hint} {', '.join(select)} FROM {_quote_sql_identifier(CollectionNames.table_name(collection.id))} "
            f"WHERE {where_sql} LIMIT %s OFFSET %s",
            [*params, limit, offset],
        )
        result: dict[str, Any] = {"ids": []}
        if include is None or "documents" in fields or "document" in fields:
            result["documents"] = []
        if include is None or "metadatas" in fields or "metadata" in fields:
            result["metadatas"] = []
        if "embeddings" in fields or "embedding" in fields:
            result["embeddings"] = []
        for row in rows:
            result["ids"].append(self._decode_id(row.get("_id")))
            if "documents" in result:
                result["documents"].append(row.get("document"))
            if "metadatas" in result:
                result["metadatas"].append(self._parse_value(row.get("metadata")) or {})
            if "embeddings" in result:
                result["embeddings"].append(self._parse_value(row.get("embedding")))
        return result

    @staticmethod
    def _decode_id(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @staticmethod
    def _parse_value(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return value
        return value

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
            query_embeddings = embedding_function([query_texts] if isinstance(query_texts, str) else query_texts)
        if query_embeddings is None:
            raise ValueError("Provide query_embeddings or query_texts")
        vectors = (
            [query_embeddings]
            if query_embeddings and isinstance(query_embeddings[0], (int, float))
            else query_embeddings
        )
        fields = self._include_fields(include)
        select = ["_id"]
        if include is None or "documents" in fields or "document" in fields:
            select.append("document")
        if include is None or "metadatas" in fields or "metadata" in fields:
            select.append("metadata")
        if "embeddings" in fields or "embedding" in fields:
            select.append("embedding")
        where_sql, params = self._build_where(None, where, where_document)
        distance = kwargs.pop("distance", collection.distance or DEFAULT_DISTANCE_METRIC)
        distance_func = {"l2": "l2_distance", "cosine": "cosine_distance", "inner_product": "inner_product"}.get(
            distance, "l2_distance"
        )
        table = _quote_sql_identifier(CollectionNames.table_name(collection.id))
        all_results = {"ids": [], "distances": []}
        if include is None or "documents" in fields or "document" in fields:
            all_results["documents"] = []
        if include is None or "metadatas" in fields or "metadata" in fields:
            all_results["metadatas"] = []
        if "embeddings" in fields or "embedding" in fields:
            all_results["embeddings"] = []
        hint = _query_hint_to_sql(query_hint, CollectionNames.table_name(collection.id))
        for vector in vectors:
            vector_sql = embedding_to_hexstring(vector)
            rows = await self._execute(
                f"SELECT {hint} {', '.join(select)}, {distance_func}(embedding, {vector_sql}) AS distance "
                f"FROM {table} WHERE {where_sql} ORDER BY {distance_func}(embedding, {vector_sql}) APPROXIMATE LIMIT %s",
                [*params, n_results],
            )
            result_ids: list[str] = []
            result_distances: list[float] = []
            result_documents: list[Any] = []
            result_metadatas: list[Any] = []
            result_embeddings: list[Any] = []
            for row in rows:
                result_ids.append(self._decode_id(row.get("_id")))
                result_distances.append(float(row.get("distance")))
                if "documents" in all_results:
                    result_documents.append(row.get("document"))
                if "metadatas" in all_results:
                    result_metadatas.append(self._parse_value(row.get("metadata")) or {})
                if "embeddings" in all_results:
                    result_embeddings.append(self._parse_value(row.get("embedding")))
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
        settings = rows[0].get("settings") if rows else "{}"
        await self._execute(
            "INSERT INTO `sdk_collections` (collection_name, settings) VALUES (%s, %s)",
            [forked_name, settings],
        )
        forked_id = await self._get_collection_id(forked_name)
        try:
            await self._execute(
                f"FORK TABLE {_quote_sql_identifier(CollectionNames.table_name(collection.id))} "
                f"TO {_quote_sql_identifier(CollectionNames.table_name(forked_id))}"
            )
        except Exception:
            with contextlib.suppress(Exception):
                await self._execute("DELETE FROM `sdk_collections` WHERE collection_name = %s", [forked_name])
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
        return type(self)(
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


__all__ = ["AsyncClient"]
