"""Synchronous collection catalog and lifecycle operations."""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from pymysql.converters import escape_string

from .collection import Collection
from .collection_lifecycle import (
    COLLECTION_CREATION_TOKEN_KEY,
    COLLECTION_STATE_CREATING,
    COLLECTION_STATE_FAILED,
    COLLECTION_STATE_READY,
    collection_state,
    parse_collection_settings,
)
from .configuration import DEFAULT_DISTANCE_METRIC, LOGIC_DATA_TABLE_LOB_INROW_THRESHOLD
from .embedding_function import Documents as EmbeddingDocuments
from .embedding_function import EmbeddingFunction, EmbeddingFunctionRegistry
from .meta_info import CollectionNames, NamespaceCollectionNames
from .query_builder import (
    build_default_ltable_schema as _build_default_ltable_schema,
)
from .query_builder import (
    build_fulltext_index_sql as _get_fulltext_index_sql,
)
from .query_builder import (
    build_ivf_vector_index_sql as _get_ivf_vector_index_sql,
)
from .schema import Schema, SparseVectorIndexConfig
from .sparse_embedding_function import SparseEmbeddingFunction, SparseEmbeddingFunctionRegistry
from .types import _NOT_PROVIDED, _NotProvided
from .types import K as FieldKey
from .validators import _quote_sql_identifier, _validate_collection_name, _validate_database_name

EmbeddingFunctionParam = EmbeddingFunction[EmbeddingDocuments] | None | _NotProvided

logger = logging.getLogger(__name__)

_COLLECTION_READY_TIMEOUT_SECONDS = 30.0
_COLLECTION_READY_POLL_INTERVAL_SECONDS = 0.05


def _extract_collection_id_from_sdk_row(row: Any) -> str:
    """Extract the collection_id from an sdk_collections row (dict/tuple/scalar)."""
    if isinstance(row, dict):
        collection_id = row.get("COLLECTION_ID") or row.get("collection_id") or ""
    elif isinstance(row, (tuple, list)):
        collection_id = row[0] if len(row) > 0 else ""
    elif isinstance(row, bytes):
        collection_id = row.decode("utf-8", errors="replace")
    elif isinstance(row, str):
        collection_id = row
    else:
        collection_id = ""
    return str(collection_id or "")


def _is_collection_conflict_error(exc: BaseException) -> bool:
    """Whether the exception chain describes a collection/table already-exists conflict."""
    current: BaseException | None = exc
    while current is not None:
        message = str(current).lower()
        object_already_exists = re.search(
            r"\b(?:collection|table)\b\s+[`'\"][^`'\"]+[`'\"]\s+already exists\b",
            message,
        )
        if "code=1050" in message or object_already_exists:
            return True
        if type(current).__name__ == "SeekdbError" and "already exists" in message:
            return True
        current = current.__cause__
    return False


def _is_collection_not_ready_error(exc: BaseException) -> bool:
    """Whether the exception can be resolved by waiting for collection creation."""
    current: BaseException | None = exc
    while current is not None:
        message = str(current).lower()
        if "being created by another client" in message or "still being created" in message:
            return True
        if "already has catalog metadata but is not ready for a new create" in message:
            return True
        if (
            "failed to get collection" in message
            and "table" in message
            and ("not found" in message or "not exists" in message)
        ):
            return True
        current = current.__cause__
    return False


def _is_namespace_catalog_conflict_error(exc: BaseException) -> bool:
    """Whether the exception indicates a duplicate sdk_namespaces/sdk_ltables unique-key conflict (1062)."""
    current: BaseException | None = exc
    while current is not None:
        message = str(current).lower()
        if "duplicate entry" in message and (
            "uk_sdk_ns_coll_name" in message or "uk_sdk_lt_coll_ns_name" in message or "code=1062" in message
        ):
            return True
        if type(current).__name__ == "IntegrityError" and "1062" in message:
            return True
        current = current.__cause__
    return False


def _is_sdk_collection_catalog_conflict_error(exc: BaseException) -> bool:
    """Whether the exception indicates a duplicate sdk_collections collection_name conflict (1062)."""
    current: BaseException | None = exc
    while current is not None:
        message = str(current).lower()
        if "duplicate entry" in message and (
            "uk_sdk_coll_name" in message
            or "idx_name" in message
            or "collection_name" in message
            or "code=1062" in message
        ):
            return True
        if type(current).__name__ == "IntegrityError" and "1062" in message:
            return True
        current = current.__cause__
    return False


def _is_duplicate_key_error(exc: BaseException) -> bool:
    """Return whether an exception in the cause chain is a duplicate-key error."""
    current: BaseException | None = exc
    while current is not None:
        message = str(current).lower()
        if "duplicate entry" in message or "duplicate key" in message or "code=1062" in message or "(1062," in message:
            return True
        if type(current).__name__ == "IntegrityError" and "1062" in message:
            return True
        current = current.__cause__
    return False


def _is_missing_table_error(exc: BaseException) -> bool:
    """Return whether an exception indicates that a cleanup target is absent."""
    message = str(exc).lower()
    return (
        "table doesn't exist" in message
        or "table does not exist" in message
        or "table not exist" in message
        or "code=1146" in message
        or "(1146," in message
    )


def _reraise_unless_unique_index_exists(exc: BaseException) -> None:
    """Re-raise unless the exception indicates the unique index is already present."""
    message = str(exc).lower()
    if (
        "already exists" in message
        or "duplicate key name" in message
        or "code=1061" in message
        or ("1061" in message and "duplicate" in message)
    ):
        return
    raise exc


_DEFAULT_PARTITION_COUNT = 1000


@dataclass
class _CollectionMeta:
    """
    Collection metadata in sdk_collections table.
    """

    collection_id: str
    collection_name: str
    settings: str | None

    @staticmethod
    def from_row(row: Any) -> _CollectionMeta:
        """Construct a _CollectionMeta from a catalog row (dict returned by the server)."""
        if isinstance(row, dict):
            collection_id = row["COLLECTION_ID"]
            collection_name = row["COLLECTION_NAME"]
            settings = row["SETTINGS"]
        elif isinstance(row, (tuple, list)):
            # Defensive: some drivers may return positional tuples
            collection_id = row[0] if len(row) > 0 else ""
            collection_name = row[1] if len(row) > 1 else ""
            settings = row[2] if len(row) > 2 else ""
        else:
            raise TypeError(f"Unsupported sdk_collections row type: {type(row).__name__}")
        return _CollectionMeta(collection_id=collection_id, collection_name=collection_name, settings=settings)


class CollectionCatalogMixin:
    """Provide collection metadata, catalog, and lifecycle operations."""

    def _get_embedding_function_dimension(self, embedding_function: EmbeddingFunction) -> int:
        """Get the dimension from an embedding function."""
        try:
            if hasattr(embedding_function, "dimension"):
                dim = embedding_function.dimension
                logger.debug(f"Using embedding function dimension: {dim}")
                return dim
            else:
                test_embeddings = embedding_function.__call__("seekdb")
                if test_embeddings and len(test_embeddings) > 0:
                    dim = len(test_embeddings[0])
                    logger.info(f"Calculated embedding function dimension: {dim}")
                    return dim
                else:
                    raise ValueError(  # noqa: TRY301
                        "Embedding function returned empty result when called with 'seekdb'"
                    )
        except Exception as e:
            raise ValueError(
                f"Failed to get dimension from embedding function: {e}. "
                f"Please ensure the embedding function has a 'dimension' attribute or can be called with a string input."
            ) from e

    def _create_sdk_collections_if_not_exists(self) -> None:
        """Create the sdk_collections catalog table if it does not already exist."""
        try:
            self._use_catalog_database()
            sdk_coll = self._qtable(CollectionNames.sdk_collections_table_name())
            scp = self._stg_cache_policy_clause()
            create_table_sql = f"""CREATE TABLE IF NOT EXISTS {sdk_coll} (
                collection_id CHAR(32) PRIMARY KEY DEFAULT (replace(uuid(), '-', '')),
                collection_name STRING,
                settings JSON COMMENT "Generated by SDK, don't modify",
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_sdk_coll_name (collection_name)
            ) COMMENT='Settings of collections created by SDK' ORGANIZATION INDEX {scp};"""
            self._execute(create_table_sql)
            try:
                self._execute(f"CREATE UNIQUE INDEX uk_sdk_coll_name ON {sdk_coll} (collection_name)")
            except Exception as exc:
                _reraise_unless_unique_index_exists(exc)
        except Exception as e:
            raise ValueError(f"Failed to create sdk_collections table: {e}") from e

    def _create_collection_meta(
        self,
        collection_name: str,
        embedding_function,
        sparse_vector_index_config: SparseVectorIndexConfig | None = None,
        *,
        dimension: int | None = None,
        distance: str | None = None,
        lifecycle_state: str | None = None,
        creation_token: str | None = None,
    ) -> dict[str, Any]:
        """Insert collection metadata into sdk_collections, resolving unique-key conflicts idempotently."""
        settings = {"version": 2}
        if dimension is not None:
            settings["dimension"] = dimension
        if distance is not None:
            settings["distance"] = distance
        if embedding_function is not None and EmbeddingFunction.support_persistence(embedding_function):
            settings["embedding_function"] = {
                "name": embedding_function.name(),
                "properties": embedding_function.get_config(),
            }

        # Persist sparse vector index config
        if sparse_vector_index_config is not None:
            sparse_settings = {}
            source_key = sparse_vector_index_config.source_key
            if source_key is not None:
                # Convert FieldKey to string for serialization
                sparse_settings["source_key"] = source_key.name if hasattr(source_key, "name") else str(source_key)
            sparse_ef = sparse_vector_index_config.embedding_function
            if sparse_ef is not None and SparseEmbeddingFunction.support_persistence(sparse_ef):
                sparse_settings["embedding_function"] = {
                    "name": sparse_ef.name(),
                    "properties": sparse_ef.get_config(),
                }
            settings["sparse_vector_index"] = sparse_settings

        if lifecycle_state is not None:
            settings["state"] = lifecycle_state
        if creation_token is not None:
            settings[COLLECTION_CREATION_TOKEN_KEY] = creation_token

        settings_str = json.dumps(settings, ensure_ascii=False)

        self._create_sdk_collections_if_not_exists()
        inserted = False
        try:
            collection_id = self._get_collection_id(collection_name)
        except ValueError:
            insert_sql = (
                f"INSERT INTO `{CollectionNames.sdk_collections_table_name()}` "
                "(COLLECTION_NAME, SETTINGS) VALUES (%s, %s)"
            )
            try:
                self._execute(insert_sql, [collection_name, settings_str])
            except Exception as exc:
                if not _is_sdk_collection_catalog_conflict_error(exc):
                    raise
                conn_getter = getattr(self, "_ensure_connection", None)
                if conn_getter is not None:
                    with contextlib.suppress(Exception):
                        conn_getter().rollback()
            else:
                inserted = True
            # If the insert succeeded this reads its generated id. If another
            # creator won the race, it reads that row instead.
            collection_id = self._get_collection_id(collection_name)

        # A creator must never reuse a catalog row that another creator has
        # already claimed.  This check closes the race where an async creator
        # inserts its ``creating`` row between the caller's existence check and
        # this metadata operation.
        if not inserted and lifecycle_state == COLLECTION_STATE_CREATING:
            existing_meta = self._resolve_collection_metadata_from_sdk_collections(collection_name)
            existing_state = collection_state(existing_meta.settings if existing_meta else None)
            if existing_state == COLLECTION_STATE_CREATING:
                raise ValueError(
                    f"Collection '{collection_name}' is being created by another client; retry after it is ready"
                )
            if lifecycle_state == COLLECTION_STATE_CREATING and existing_meta is not None:
                raise ValueError(
                    f"Collection '{collection_name}' already has catalog metadata but is not ready for a new create"
                )

        return {
            "collection_id": collection_id,
            "table_name": CollectionNames.table_name(collection_id),
        }

    # ==================== Namespace Catalog Methods ====================

    def _catalog_database(self) -> str:
        """Database that holds sdk_* catalog tables (must match OB bg scheduler scan target)."""
        db = getattr(self, "database", None)
        if not db:
            raise ValueError("client database is not configured")
        _validate_database_name(db)
        return db

    def _qtable(self, table: str) -> str:
        """Fully-qualified catalog table: `{database}`.`{table}`."""
        db = _quote_sql_identifier(self._catalog_database())
        table_quoted = _quote_sql_identifier(table)
        return f"{db}.{table_quoted}"

    def _use_catalog_database(self) -> None:
        """Align session with pymysql database= so PL (DROP_NAMESPACE) uses the same DB."""
        self._execute(f"USE {_quote_sql_identifier(self._catalog_database())}")

    def _ensure_namespace_catalogs(self) -> None:
        """Create the sdk_namespaces and sdk_ltables catalog tables and their unique indexes."""
        self._use_catalog_database()
        ns_namespaces_q = self._qtable(NamespaceCollectionNames.sdk_namespaces_table())
        ns_ltables_q = self._qtable(NamespaceCollectionNames.sdk_ltables_table())
        scp = self._stg_cache_policy_clause()
        ns_namespaces_sql = f"""CREATE TABLE IF NOT EXISTS {ns_namespaces_q} (
            namespace_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
            collection_id CHAR(32) NOT NULL,
            namespace_name VARCHAR(256) NOT NULL,
            created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
            updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
            info JSON,
            PRIMARY KEY (namespace_id),
            UNIQUE KEY uk_sdk_ns_coll_name (collection_id, namespace_name),
            KEY idx_sdk_ns_by_collection (collection_id)
        ) COMMENT='Namespace catalog' ORGANIZATION INDEX {scp};"""
        ns_ltables_sql = f"""CREATE TABLE IF NOT EXISTS {ns_ltables_q} (
            ltable_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
            collection_id CHAR(32) NOT NULL,
            namespace_id BIGINT UNSIGNED NOT NULL,
            ltable_name VARCHAR(256) NOT NULL DEFAULT 'default',
            created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
            updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
            info JSON,
            PRIMARY KEY (ltable_id),
            UNIQUE KEY uk_sdk_lt_coll_ns_name (collection_id, namespace_id, ltable_name),
            KEY idx_sdk_lt_by_ns (collection_id, namespace_id)
        ) COMMENT='LTable catalog' ORGANIZATION INDEX {scp};"""
        namespaces_stats_sql = f"""CREATE TABLE IF NOT EXISTS {self._qtable(NamespaceCollectionNames.sdk_namespaces_stats_table())} (
            collection_id CHAR(32) NOT NULL COMMENT 'collection id',
            namespace_id BIGINT UNSIGNED NOT NULL COMMENT 'namespace internal id',
            ltable_id BIGINT UNSIGNED NOT NULL COMMENT 'logic table internal id, 0 means namespace summary',
            estimated_rows BIGINT NOT NULL DEFAULT 0 COMMENT 'estimated row count',
            average_row_size BIGINT NOT NULL DEFAULT 0 COMMENT 'average row size in bytes',
            row_limit BIGINT NOT NULL DEFAULT -1 COMMENT 'row count limit, -1 means unlimited',
            size_limit BIGINT NOT NULL DEFAULT -1 COMMENT 'storage size limit in bytes, -1 means unlimited',
            last_estimate_time TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT 'last estimate time',
            included_index BOOL NOT NULL DEFAULT FALSE COMMENT 'whether stats include index data',
            PRIMARY KEY (namespace_id, ltable_id, included_index),
            KEY idx_sdk_ns_stat_by_collection (collection_id)
        ) COMMENT='Logic table row count and storage size statistics' DEFAULT CHARSET=utf8mb4 ORGANIZATION INDEX
        PARTITION BY KEY(namespace_id) PARTITIONS 8;"""
        self._execute(ns_namespaces_sql)
        self._execute(ns_ltables_sql)
        self._execute(namespaces_stats_sql)
        try:
            self._execute(
                f"CREATE UNIQUE INDEX uk_sdk_ns_coll_name ON {ns_namespaces_q} (collection_id, namespace_name)"
            )
        except Exception as exc:
            _reraise_unless_unique_index_exists(exc)
        try:
            self._execute(
                f"CREATE UNIQUE INDEX uk_sdk_lt_coll_ns_name ON {ns_ltables_q} "
                f"(collection_id, namespace_id, ltable_name)"
            )
        except Exception as exc:
            _reraise_unless_unique_index_exists(exc)

    def _rollback_connection_if_supported(self) -> None:
        """Roll back the current connection transaction if the backend supports it."""
        conn_getter = getattr(self, "_ensure_connection", None)
        if conn_getter is not None:
            with contextlib.suppress(Exception):
                conn_getter().rollback()

    def _create_ns_collection_meta(self, collection_name: str, settings: dict) -> dict:
        """Insert namespace collection metadata, resolving unique-key conflicts idempotently."""
        self._create_sdk_collections_if_not_exists()
        settings_str = json.dumps(settings, ensure_ascii=False)
        sdk_coll = self._qtable(CollectionNames.sdk_collections_table_name())
        insert_sql = f"INSERT INTO {sdk_coll} (collection_name, settings) VALUES (%s, %s)"
        try:
            self._execute(insert_sql, [collection_name, settings_str])
        except Exception as exc:
            if not _is_sdk_collection_catalog_conflict_error(exc):
                raise
            conn_getter = getattr(self, "_ensure_connection", None)
            if conn_getter is not None:
                with contextlib.suppress(Exception):
                    conn_getter().rollback()
        rows = self._execute(
            f"SELECT collection_id FROM {sdk_coll} WHERE collection_name = %s",
            [collection_name],
        )
        collection_id = str(rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]["collection_id"])
        self._set_session_ns_context(collection_id=collection_id)
        return {"collection_id": collection_id, "collection_name": collection_name}

    def _get_ns_collection_meta(self, collection_name: str) -> dict | None:
        """Fetch namespace collection metadata from the catalog."""
        try:
            rows = self._execute(
                f"SELECT collection_id, collection_name, settings "
                f"FROM {self._qtable(CollectionNames.sdk_collections_table_name())} "
                "WHERE collection_name = %s",
                [collection_name],
            )
        except Exception as exc:
            if _is_missing_table_error(exc):
                return None
            raise
        if not rows:
            return None
        row = rows[0]
        if isinstance(row, (list, tuple)):
            settings = json.loads(row[2]) if row[2] else {}
        else:
            settings = json.loads(row["settings"]) if row.get("settings") else {}
        if not settings.get("use_namespace"):
            return None
        if isinstance(row, (list, tuple)):
            return {
                "collection_id": str(row[0]),
                "collection_name": row[1],
                "settings": settings,
            }
        return {
            "collection_id": str(row["collection_id"]),
            "collection_name": row["collection_name"],
            "settings": settings,
        }

    def _has_ns_collection(self, collection_name: str) -> bool:
        """Return whether a namespace collection with the given name exists."""
        return self._get_ns_collection_meta(collection_name) is not None

    def _ns_collection_exists_by_id(self, collection_id: str) -> bool:
        """Whether a namespace-enabled collection with this id still exists.

        Used to reject namespace operations on a stale Collection handle whose
        underlying collection was deleted (the in-memory handle keeps its old id).
        """
        collection_id_escaped = escape_string(str(collection_id))
        rows = self._execute(
            f"SELECT collection_id FROM {self._qtable(CollectionNames.sdk_collections_table_name())} "
            f"WHERE collection_id = '{collection_id_escaped}'"
        )
        return bool(rows)

    def _delete_ns_collection_meta(self, collection_name: str) -> None:
        """Delete namespace collection metadata from the catalog."""
        meta = self._get_ns_collection_meta(collection_name)
        if meta is None:
            raise ValueError(f"Namespace collection '{collection_name}' not found")
        collection_id = meta["collection_id"]
        collection_id_escaped = escape_string(collection_id)
        self._execute(
            f"DELETE FROM `{CollectionNames.sdk_collections_table_name()}` "
            f"WHERE collection_id = '{collection_id_escaped}'"
        )
        cleanup_errors: list[tuple[str, Exception]] = []
        for table_name in (
            NamespaceCollectionNames.sdk_ltables_table(),
            NamespaceCollectionNames.sdk_namespaces_table(),
            NamespaceCollectionNames.sdk_namespaces_stats_table(),
        ):
            try:
                self._execute(f"DELETE FROM `{table_name}` WHERE collection_id = '{collection_id_escaped}'")
            except Exception as exc:
                # Catalog cleanup is best effort because recovery can encounter
                # partially-created tables. Missing tables are expected; other
                # failures are reported after all cleanup attempts finish.
                logger.warning("Failed to clean namespace catalog table %s", table_name, exc_info=True)
                if not _is_missing_table_error(exc):
                    cleanup_errors.append((table_name, exc))
        self._cleanup_namespace_physical_tables(collection_id)
        if cleanup_errors:
            failed_tables = ", ".join(table_name for table_name, _ in cleanup_errors)
            raise RuntimeError(f"Failed to clean namespace catalog table(s): {failed_tables}") from cleanup_errors[0][1]

    def _create_namespace_physical_tables(
        self,
        collection_id: str,
        dimension: int,
        ivf_config=None,
        fulltext_config=None,
        is_shared_storage: bool = False,
        partition_count: int = _DEFAULT_PARTITION_COUNT,
        cleanup_on_error: bool = True,
    ) -> None:
        """Create the physical tables backing a namespace collection."""
        tg_name = NamespaceCollectionNames.tablegroup_name(collection_id)
        data_table = NamespaceCollectionNames.data_table_name(collection_id)
        kv_table = NamespaceCollectionNames.kv_data_table_name(collection_id)
        schema_table = NamespaceCollectionNames.logic_schema_table_name(collection_id)

        index_parts = ["SEARCH INDEX idx_json(data_content)"]
        if fulltext_config is not None:
            fulltext_clause = _get_fulltext_index_sql(fulltext_config)
            index_parts.insert(0, f"FULLTEXT INDEX idx_fts(document) {fulltext_clause}")
        if ivf_config is not None:
            vector_index_sql = _get_ivf_vector_index_sql(ivf_config)
            index_parts.append(f"VECTOR INDEX idx_vec(embedding) {vector_index_sql}")
        index_sql = ",\n                ".join(index_parts)
        partition_clause = f"PARTITION BY KEY(namespace_id) PARTITIONS {partition_count}"

        # All CREATEs use IF NOT EXISTS so this is idempotent: a fresh create builds
        # everything, while a resume (after a crash left some tables behind) skips the
        # existing ones and only fills in the gaps.
        try:
            self._execute(f"CREATE TABLEGROUP IF NOT EXISTS `{tg_name}` SHARDING='ADAPTIVE'")

            data_sql = f"""CREATE TABLE IF NOT EXISTS `{data_table}` (
                namespace_id BIGINT UNSIGNED NOT NULL,
                ltable_id BIGINT UNSIGNED NOT NULL,
                document LONGTEXT,
                embedding VECTOR({dimension}),
                data_content JSON NOT NULL,
                created_by VARCHAR(64) DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                {index_sql}
            ) TABLEGROUP=`{tg_name}` COMMENT='逻辑表主数据' DEFAULT CHARSET=utf8mb4 ORGANIZATION HEAP IS_LOGIC_TABLE = TRUE LOB_INROW_THRESHOLD={LOGIC_DATA_TABLE_LOB_INROW_THRESHOLD}
            {partition_clause}"""

            self._execute(data_sql)

            if is_shared_storage:
                hot_table = NamespaceCollectionNames.hot_table_name(collection_id)
                self._execute(f"""CREATE TABLE IF NOT EXISTS `{hot_table}` (
                    namespace_id BIGINT UNSIGNED NOT NULL,
                    last_access_time TIMESTAMP(6) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    PRIMARY KEY(namespace_id)
                ) TABLEGROUP=`{tg_name}` COMMENT='热点/TTL附属表' DEFAULT CHARSET=utf8mb4 ORGANIZATION INDEX
                {partition_clause}""")

            self._execute(f"""CREATE TABLE IF NOT EXISTS `{kv_table}` (
                namespace_id BIGINT UNSIGNED NOT NULL,
                kv_key VARBINARY(1024) NOT NULL,
                kv_value LONGBLOB NOT NULL,
                PRIMARY KEY(namespace_id, kv_key)
            ) TABLEGROUP=`{tg_name}` COMMENT='索引与映射KV表' DEFAULT CHARSET=utf8mb4 ORGANIZATION INDEX LOB_INROW_THRESHOLD=786432
            {partition_clause}""")

            self._execute(f"""CREATE TABLE IF NOT EXISTS `{schema_table}` (
                namespace_id BIGINT UNSIGNED NOT NULL,
                ltable_id BIGINT UNSIGNED NOT NULL,
                schema_content JSON NOT NULL,
                created_by VARCHAR(64) DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY(namespace_id, ltable_id)
            ) TABLEGROUP=`{tg_name}` COMMENT='LTable schema定义' DEFAULT CHARSET=utf8mb4 ORGANIZATION INDEX
            {partition_clause}""")

        except Exception:
            if cleanup_on_error:
                self._cleanup_namespace_physical_tables(collection_id)
            raise

    def _cleanup_namespace_physical_tables(self, collection_id: str) -> None:
        """Drop the physical tables backing a namespace collection."""
        for suffix_fn in [
            NamespaceCollectionNames.data_table_name,
            NamespaceCollectionNames.logic_schema_table_name,
            NamespaceCollectionNames.kv_data_table_name,
            NamespaceCollectionNames.hot_table_name,
        ]:
            try:
                self._execute(f"DROP TABLE IF EXISTS `{suffix_fn(collection_id)}`")
            except Exception:
                logger.warning("Failed to clean namespace physical table %s", suffix_fn(collection_id), exc_info=True)
        try:
            self._execute(f"DROP TABLEGROUP IF EXISTS `{NamespaceCollectionNames.tablegroup_name(collection_id)}`")
        except Exception:
            logger.warning(
                "Failed to clean namespace tablegroup %s",
                NamespaceCollectionNames.tablegroup_name(collection_id),
                exc_info=True,
            )

    def _table_exists(self, table_name: str) -> bool:
        """Whether `table_name` exists in the current (catalog) database."""
        name_escaped = escape_string(table_name)
        rows = self._execute(
            f"SELECT 1 FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = '{name_escaped}'"
        )
        return bool(rows)

    def _tablegroup_exists(self, tablegroup_name: str) -> bool:
        """Whether `tablegroup_name` exists in the current OceanBase tenant."""
        name_escaped = escape_string(tablegroup_name)
        rows = self._execute(f"SELECT 1 FROM oceanbase.DBA_OB_TABLEGROUPS WHERE TABLEGROUP_NAME = '{name_escaped}'")
        return bool(rows)

    def _ns_missing_physical_resources(self, collection_id: str, is_shared_storage: bool) -> list[str]:
        """Return expected tablegroup/tables that are absent for this namespace collection."""
        resources: list[tuple[str, bool]] = [
            (NamespaceCollectionNames.tablegroup_name(collection_id), True),
            (NamespaceCollectionNames.data_table_name(collection_id), False),
            (NamespaceCollectionNames.kv_data_table_name(collection_id), False),
            (NamespaceCollectionNames.logic_schema_table_name(collection_id), False),
        ]
        if is_shared_storage:
            resources.append((NamespaceCollectionNames.hot_table_name(collection_id), False))
        missing: list[str] = []
        for resource_name, is_tablegroup in resources:
            exists = self._tablegroup_exists(resource_name) if is_tablegroup else self._table_exists(resource_name)
            if not exists:
                missing.append(resource_name)
        return missing

    def _is_incomplete_ns_collection(self, name: str) -> bool:
        """Whether `name` is a namespace collection whose catalog row exists but
        whose physical tables are not all present (e.g. creation was interrupted
        by a crash). Such a collection can be finished by re-running create.
        """
        meta = self._get_ns_collection_meta(name)
        if meta is None:
            return False
        is_ss = meta.get("settings", {}).get("storage_mode") == "ss"
        self._use_catalog_database()
        return len(self._ns_missing_physical_resources(meta["collection_id"], is_ss)) > 0

    def _resolve_namespace_ltable_id(self, collection_id: str, namespace_id: int) -> int:
        """Resolve the default ltable_id for (collection_id, namespace_id) from
        sdk_ltables. Cached per (collection_id, namespace_id) on the client
        instance to avoid the extra round-trip on every DML/DQL call.

        The cache is also seeded by `_create_ns_namespace_meta` and
        `_get_ns_namespace_meta` once they have read/created the row.
        """
        key = (str(collection_id), int(namespace_id))
        cache = getattr(self, "_ns_ltable_id_cache", None)
        if cache is None:
            cache = {}
            self._ns_ltable_id_cache = cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        coll_id_escaped = escape_string(str(collection_id))
        rows = self._execute(
            f"SELECT ltable_id FROM {self._qtable(NamespaceCollectionNames.sdk_ltables_table())} "
            f"WHERE collection_id = '{coll_id_escaped}' AND namespace_id = {int(namespace_id)} "
            f"AND ltable_name = 'default' LIMIT 1"
        )
        if not rows:
            raise ValueError(
                f"No default ltable found for collection_id={collection_id}, namespace_id={namespace_id} in sdk_ltables"
            )
        row = rows[0]
        lt_id = int(row[0] if isinstance(row, (list, tuple)) else row["ltable_id"])
        cache[key] = lt_id
        return lt_id

    def _cache_namespace_ltable_id(self, collection_id: str, namespace_id: int, ltable_id: int) -> None:
        """Cache the resolved logical-table id for a namespace."""
        key = (str(collection_id), int(namespace_id))
        cache = getattr(self, "_ns_ltable_id_cache", None)
        if cache is None:
            cache = {}
            self._ns_ltable_id_cache = cache
        cache[key] = int(ltable_id)

    def _set_session_ns_context(
        self,
        collection_id: str | None = None,
        namespace_id: int | None = None,
        ltable_id: int | None = None,
    ) -> None:
        """Set session variables identifying the active namespace context."""
        if collection_id is not None:
            self._execute(f"SET @collection_id = '{escape_string(str(collection_id))}'")
        if namespace_id is not None:
            self._execute(f"SET @namespace_id = {int(namespace_id)}")
        if ltable_id is not None:
            self._execute(f"SET @ltable_id = {int(ltable_id)}")

    def _fetch_ns_namespace_id(self, collection_id: str, namespace_name: str) -> int:
        """Fetch the namespace id for a collection/namespace pair from the catalog."""
        namespace_name_escaped = escape_string(namespace_name)
        collection_id_escaped = escape_string(collection_id)
        ns_table = self._qtable(NamespaceCollectionNames.sdk_namespaces_table())
        rows = self._execute(
            f"SELECT namespace_id FROM {ns_table} "
            f"WHERE collection_id = '{collection_id_escaped}' AND namespace_name = '{namespace_name_escaped}'"
        )
        if not rows:
            raise ValueError(
                f"Namespace '{namespace_name}' not found for collection_id={collection_id} in sdk_namespaces"
            )
        return int(rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]["namespace_id"])

    def _fetch_ns_ltable_id(
        self,
        collection_id: str,
        namespace_id: int,
        ltable_name: str = "default",
    ) -> int:
        """Fetch the logical-table id for a namespace from the catalog."""
        collection_id_escaped = escape_string(collection_id)
        ltable_name_escaped = escape_string(ltable_name)
        lt_table = self._qtable(NamespaceCollectionNames.sdk_ltables_table())
        lt_rows = self._execute(
            f"SELECT ltable_id FROM {lt_table} "
            f"WHERE collection_id = '{collection_id_escaped}' AND namespace_id = {int(namespace_id)} "
            f"AND ltable_name = '{ltable_name_escaped}'"
        )
        if not lt_rows:
            raise ValueError(
                f"LTable '{ltable_name}' not found for collection_id={collection_id}, "
                f"namespace_id={namespace_id} in sdk_ltables"
            )
        return int(lt_rows[0][0] if isinstance(lt_rows[0], (list, tuple)) else lt_rows[0]["ltable_id"])

    def _insert_ns_namespace_catalog_row(
        self,
        collection_id: str,
        namespace_name: str,
        *,
        idempotent: bool,
    ) -> int:
        """Insert a row into the sdk_namespaces catalog table."""
        namespace_name_escaped = escape_string(namespace_name)
        collection_id_escaped = escape_string(collection_id)
        ns_table = self._qtable(NamespaceCollectionNames.sdk_namespaces_table())
        try:
            self._execute(
                f"INSERT INTO {ns_table} "
                f"(collection_id, namespace_name) VALUES ('{collection_id_escaped}', '{namespace_name_escaped}')"
            )
        except Exception as exc:
            if idempotent and _is_namespace_catalog_conflict_error(exc):
                self._rollback_connection_if_supported()
            else:
                raise
        return self._fetch_ns_namespace_id(collection_id, namespace_name)

    def _insert_ns_ltable_catalog_row(
        self,
        collection_id: str,
        namespace_id: int,
        *,
        idempotent: bool,
        ltable_name: str = "default",
    ) -> int:
        """Insert a row into the sdk_ltables catalog table."""
        collection_id_escaped = escape_string(collection_id)
        ltable_name_escaped = escape_string(ltable_name)
        lt_table = self._qtable(NamespaceCollectionNames.sdk_ltables_table())
        try:
            self._execute(
                f"INSERT INTO {lt_table} "
                f"(collection_id, namespace_id, ltable_name) "
                f"VALUES ('{collection_id_escaped}', {int(namespace_id)}, '{ltable_name_escaped}')"
            )
        except Exception as exc:
            if idempotent and _is_namespace_catalog_conflict_error(exc):
                self._rollback_connection_if_supported()
            else:
                raise
        return self._fetch_ns_ltable_id(collection_id, namespace_id, ltable_name)

    def _resolve_ns_ltable_index_layout(self, collection_id: str) -> tuple[bool, bool]:
        """Infer which optional indexes exist for a namespace collection."""
        collection_id_escaped = escape_string(collection_id)
        rows = self._execute(
            f"SELECT settings FROM {self._qtable(CollectionNames.sdk_collections_table_name())} "
            f"WHERE collection_id = '{collection_id_escaped}'"
        )
        settings: dict[str, Any] = {}
        if rows:
            raw = rows[0]["settings"] if isinstance(rows[0], dict) else rows[0][0]
            settings = json.loads(raw) if raw else {}
        has_ivf = settings.get("dense_index_type") == "ivf"
        if "has_fulltext_index" in settings:
            has_fulltext = bool(settings["has_fulltext_index"])
        else:
            data_table = NamespaceCollectionNames.data_table_name(collection_id)
            if self._table_exists(data_table):
                index_rows = self._execute(f"SHOW INDEX FROM `{data_table}`")
                index_names = {(row.get("Key_name") if isinstance(row, dict) else row[2]) for row in (index_rows or [])}
                has_fulltext = "idx_fts" in index_names
            else:
                has_fulltext = False
        return has_fulltext, has_ivf

    def _finalize_ns_namespace_meta(
        self,
        collection_id: str,
        namespace_name: str,
        namespace_id: int,
        ltable_id: int,
    ) -> dict:
        """Finalize namespace metadata after catalog rows and physical tables are created."""
        self._cache_namespace_ltable_id(collection_id, namespace_id, ltable_id)
        schema_table = self._qtable(NamespaceCollectionNames.logic_schema_table_name(collection_id))
        has_fulltext, has_ivf = self._resolve_ns_ltable_index_layout(collection_id)
        schema_content = json.dumps(_build_default_ltable_schema(has_fulltext=has_fulltext, has_ivf=has_ivf))
        try:
            self._execute(
                f"INSERT INTO {schema_table} (namespace_id, ltable_id, schema_content) "
                f"VALUES ({namespace_id}, {ltable_id}, '{escape_string(schema_content)}')"
            )
        except Exception as exc:
            if not _is_duplicate_key_error(exc):
                raise
        self._set_session_ns_context(namespace_id=namespace_id, ltable_id=ltable_id)
        return {"namespace_id": str(namespace_id), "namespace_name": namespace_name, "ltable_id": str(ltable_id)}

    def _create_ns_namespace_meta(self, collection_id: str, namespace_name: str) -> dict:
        """Create namespace metadata, resolving unique-key conflicts idempotently."""
        if self._get_ns_namespace_meta(collection_id, namespace_name) is not None:
            raise ValueError(f"Namespace '{namespace_name}' already exists")
        try:
            ns_id = self._insert_ns_namespace_catalog_row(collection_id, namespace_name, idempotent=False)
            lt_id = self._insert_ns_ltable_catalog_row(collection_id, ns_id, idempotent=False)
        except Exception as exc:
            if _is_namespace_catalog_conflict_error(exc):
                self._rollback_connection_if_supported()
                raise ValueError(f"Namespace '{namespace_name}' already exists") from exc
            raise
        return self._finalize_ns_namespace_meta(collection_id, namespace_name, ns_id, lt_id)

    def _get_or_create_ns_namespace_meta(self, collection_id: str, namespace_name: str) -> dict:
        """Get existing namespace metadata or create it if absent."""
        meta = self._get_ns_namespace_meta(collection_id, namespace_name)
        if meta is not None:
            if meta.get("ltable_id") is not None:
                return meta
            lt_id = self._insert_ns_ltable_catalog_row(collection_id, int(meta["namespace_id"]), idempotent=True)
            return self._finalize_ns_namespace_meta(collection_id, namespace_name, int(meta["namespace_id"]), lt_id)
        ns_id = self._insert_ns_namespace_catalog_row(collection_id, namespace_name, idempotent=True)
        lt_id = self._insert_ns_ltable_catalog_row(collection_id, ns_id, idempotent=True)
        return self._finalize_ns_namespace_meta(collection_id, namespace_name, ns_id, lt_id)

    def _get_ns_namespace_meta(self, collection_id: str, namespace_name: str) -> dict | None:
        """Fetch namespace metadata from the catalog."""
        namespace_name_escaped = escape_string(namespace_name)
        collection_id_escaped = escape_string(collection_id)
        ns_table = self._qtable(NamespaceCollectionNames.sdk_namespaces_table())
        lt_table = self._qtable(NamespaceCollectionNames.sdk_ltables_table())
        rows = self._execute(
            f"SELECT n.namespace_id AS namespace_id, n.namespace_name AS namespace_name, "
            f"l.ltable_id AS ltable_id "
            f"FROM {ns_table} n "
            f"LEFT JOIN {lt_table} l "
            f"ON l.collection_id = n.collection_id "
            f"AND l.namespace_id = n.namespace_id "
            f"AND l.ltable_name = 'default' "
            f"WHERE n.collection_id = '{collection_id_escaped}' "
            f"AND n.namespace_name = '{namespace_name_escaped}'"
        )
        if not rows:
            return None
        row = rows[0]
        if isinstance(row, (list, tuple)):
            ns_id = str(row[0])
            ns_name = row[1]
            lt_raw = row[2] if len(row) > 2 else None
        else:
            ns_id = str(row["namespace_id"])
            ns_name = row["namespace_name"]
            lt_raw = row.get("ltable_id")
        lt_id = int(lt_raw) if lt_raw is not None else None
        if lt_id is not None:
            self._cache_namespace_ltable_id(collection_id, int(ns_id), lt_id)
        self._set_session_ns_context(namespace_id=int(ns_id), ltable_id=lt_id)
        meta: dict = {"namespace_id": ns_id, "namespace_name": ns_name}
        if lt_id is not None:
            meta["ltable_id"] = str(lt_id)
        return meta

    def _has_ns_namespace(self, collection_id: str, namespace_name: str) -> bool:
        """Return whether a namespace with the given name exists."""
        return self._get_ns_namespace_meta(collection_id, namespace_name) is not None

    def _ns_namespace_exists_by_id(self, collection_id: str, namespace_id: str) -> bool:
        """Whether a namespace with this id is still live in the catalog.

        Used to reject DML/DQL on a stale Namespace handle whose namespace (or
        whole collection) was deleted. delete_namespace soft-deletes by renaming
        the row to '__recyclebin_<name>_<id>' (kernel async cleanup follows), so
        a recyclebin-prefixed row counts as gone; a deleted collection removes
        the row outright. Underlying data may linger after either, so we trust
        the catalog, not the data table.
        """
        collection_id_escaped = escape_string(str(collection_id))
        rows = self._execute(
            f"SELECT namespace_id FROM {self._qtable(NamespaceCollectionNames.sdk_namespaces_table())} "
            f"WHERE collection_id = '{collection_id_escaped}' AND namespace_id = {int(namespace_id)} "
            f"AND LEFT(namespace_name, 13) <> '__recyclebin_'"
        )
        return bool(rows)

    def _delete_ns_namespace_meta(self, collection_id: str, namespace_name: str) -> None:
        """Delete namespace metadata from the catalog."""
        meta = self._get_ns_namespace_meta(collection_id, namespace_name)
        if meta is None:
            raise ValueError(f"Namespace '{namespace_name}' not found")
        ns_id = meta["namespace_id"]
        lt_id_raw = meta.get("ltable_id")
        lt_id = int(lt_id_raw) if lt_id_raw is not None else None
        collection_id_escaped = escape_string(collection_id)
        # PL reads session database_name; must match where catalog tables live.
        self._use_catalog_database()
        self._set_session_ns_context(collection_id=collection_id, namespace_id=int(ns_id), ltable_id=lt_id)
        self._execute(f"CALL DBMS_LOGIC_TABLE.DROP_NAMESPACE('{collection_id_escaped}', {ns_id})")

    def _list_ns_namespaces(self, collection_id: str) -> list[dict]:
        """List namespaces registered for a collection."""
        collection_id_escaped = escape_string(collection_id)
        rows = self._execute(
            f"SELECT namespace_id, namespace_name FROM {self._qtable(NamespaceCollectionNames.sdk_namespaces_table())} "
            f"WHERE collection_id = '{collection_id_escaped}' "
            f"AND LEFT(namespace_name, 13) <> '__recyclebin_' "
            f"ORDER BY namespace_id"
        )
        results = []
        for row in rows:
            if isinstance(row, (list, tuple)):
                results.append({"namespace_id": str(row[0]), "namespace_name": row[1]})
            else:
                results.append({"namespace_id": str(row["namespace_id"]), "namespace_name": row["namespace_name"]})
        return results

    # ==================== End Namespace Catalog Methods ====================

    def get_collection(self, name: str, embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED) -> Collection:
        """Get an existing collection by name."""
        _validate_collection_name(name)
        ns_meta = self._get_ns_collection_meta(name)
        if ns_meta is not None:
            self._assert_ns_collection_ready(ns_meta)
            return self._build_ns_collection_from_meta(ns_meta, embedding_function)
        return self._get_collection(name, embedding_function)

    def _assert_ns_collection_ready(self, meta: dict) -> None:
        """Raise when a namespace collection is incomplete without mutating it."""
        self._use_catalog_database()
        settings = meta.get("settings", {})
        is_shared_storage = settings.get("storage_mode") == "ss"
        missing = self._ns_missing_physical_resources(meta["collection_id"], is_shared_storage)
        if missing:
            raise ValueError(
                f"Namespace collection '{meta['collection_name']}' is not ready; "
                f"missing physical resources: {', '.join(missing)}. "
                "Retry create_collection() to resume creation."
            )

    def _build_ns_collection_from_meta(self, meta: dict, embedding_function=_NOT_PROVIDED) -> Collection:
        """Build a namespace Collection facade from catalog metadata."""
        settings = meta.get("settings", {})
        dimension = settings.get("dimension")
        distance = settings.get("distance", DEFAULT_DISTANCE_METRIC)
        partition_count = settings.get("partition_count")
        ef = None
        if embedding_function is not _NOT_PROVIDED:
            ef = embedding_function
        elif "embedding_function" in settings:
            ef_info = settings["embedding_function"]
            ef_class = EmbeddingFunctionRegistry.get_class(ef_info["name"])
            if ef_class is None:
                raise ValueError(f"Embedding function class '{ef_info['name']}' not found")
            ef = ef_class.build_from_config(ef_info.get("properties", {}))
        return Collection(
            client=self,
            name=meta["collection_name"],
            collection_id=meta["collection_id"],
            dimension=dimension,
            embedding_function=ef,
            distance=distance,
            use_namespace=True,
            partition_count=partition_count,
            has_vector_index=settings.get("dense_index_type") == "ivf",
        )

    def _resolve_collection_metadata_from_sdk_collections(self, collection_name: str) -> _CollectionMeta | None:
        """
        Resolve collection metadata information from sdk_collections table
        """
        try:
            query_sql = (
                f"SELECT COLLECTION_ID, COLLECTION_NAME, SETTINGS "
                f"FROM `{CollectionNames.sdk_collections_table_name()}` WHERE COLLECTION_NAME = %s"
            )
            rows = self._execute(query_sql, [collection_name])
            if rows:
                return _CollectionMeta.from_row(rows[0])

        except Exception as e:
            raise ValueError(f"Failed to resolve collection metadata from sdk_collections table: {e}") from e
        return None

    def _mark_collection_ready(self, name: str, collection_id: str, creation_token: str) -> None:
        """Publish a synchronously created collection only while owning its token."""
        meta = self._resolve_collection_metadata_from_sdk_collections(name)
        if meta is None or str(meta.collection_id) != str(collection_id):
            raise RuntimeError(f"Collection '{name}' creation ownership was lost before it became ready")
        settings = parse_collection_settings(meta.settings)
        if settings.get(COLLECTION_CREATION_TOKEN_KEY) != creation_token:
            raise RuntimeError(f"Collection '{name}' creation ownership was lost before it became ready")

        settings["state"] = COLLECTION_STATE_READY
        settings.pop(COLLECTION_CREATION_TOKEN_KEY, None)
        update_sql = (
            f"UPDATE {self._qtable(CollectionNames.sdk_collections_table_name())} SET settings = %s "
            "WHERE collection_name = %s AND collection_id = %s "
            "AND JSON_UNQUOTE(JSON_EXTRACT(settings, '$.creation_token')) = %s "
            "AND JSON_UNQUOTE(JSON_EXTRACT(settings, '$.state')) = %s"
        )
        self._execute(
            update_sql,
            [json.dumps(settings, ensure_ascii=False), name, collection_id, creation_token, COLLECTION_STATE_CREATING],
        )

        published = self._resolve_collection_metadata_from_sdk_collections(name)
        if published is None or collection_state(published.settings) != COLLECTION_STATE_READY:
            raise RuntimeError(f"Collection '{name}' could not be published as ready")

    def _cleanup_failed_collection(self, name: str, collection_id: str | None, creation_token: str) -> None:
        """Best-effort cleanup of a failed synchronous collection create."""
        try:
            meta = self._resolve_collection_metadata_from_sdk_collections(name)
        except Exception:
            logger.warning("Failed to inspect catalog while cleaning collection '%s'", name, exc_info=True)
            return
        if meta is None or (collection_id is not None and str(meta.collection_id) != str(collection_id)):
            return
        settings = parse_collection_settings(meta.settings)
        if settings.get(COLLECTION_CREATION_TOKEN_KEY) != creation_token:
            return

        try:
            collection_id = str(meta.collection_id)
            self._execute(f"DROP TABLE IF EXISTS {_quote_sql_identifier(CollectionNames.table_name(collection_id))}")
        except Exception:
            # Keep the catalog row when the physical cleanup failed so an
            # operator can retry explicitly instead of leaving an orphan table.
            logger.warning("Failed to drop incomplete collection table '%s'", collection_id, exc_info=True)
            return

        delete_sql = (
            f"DELETE FROM {self._qtable(CollectionNames.sdk_collections_table_name())} "
            "WHERE collection_name = %s AND collection_id = %s "
            "AND JSON_UNQUOTE(JSON_EXTRACT(settings, '$.creation_token')) = %s"
        )
        with contextlib.suppress(Exception):
            self._execute(delete_sql, [name, collection_id, creation_token])

    def cleanup_incomplete_collection(self, name: str) -> None:
        """Explicitly remove a stale or failed collection creation.

        This operation is intentionally explicit.  Callers must first ensure
        that no live creator is still building the collection; a timeout alone
        is not sufficient evidence that the creator has stopped.
        """
        _validate_collection_name(name)
        meta = self._resolve_collection_metadata_from_sdk_collections(name)
        if meta is None:
            raise ValueError(f"Collection '{name}' does not exist")
        settings = parse_collection_settings(meta.settings)
        state = collection_state(settings)
        if state not in {COLLECTION_STATE_CREATING, COLLECTION_STATE_FAILED}:
            raise ValueError(f"Collection '{name}' is not an incomplete creation")
        creation_token = settings.get(COLLECTION_CREATION_TOKEN_KEY)
        if not creation_token:
            raise ValueError(f"Collection '{name}' has no verifiable creation owner; clean it up with DBA tooling")
        self._cleanup_failed_collection(name, str(meta.collection_id), creation_token)
        if self._resolve_collection_metadata_from_sdk_collections(name) is not None:
            raise RuntimeError(f"Failed to clean up incomplete collection '{name}'")

    def _resolve_collection_metadata_from_table(self, table_name: str, collection_name: str) -> dict[str, Any]:
        """
        Resolve collection metadata information from collection table (not sdk_collections table)
        """
        metadata = {
            "dimension": None,
            "distance": None,
        }
        # Check if table exists by describing it
        try:
            table_info = self._execute(f"DESCRIBE `{table_name}`")
            if not table_info or len(table_info) == 0:
                raise ValueError(  # noqa: TRY301
                    f"Collection ('{collection_name}') not found: Table('{table_name}') not exists"
                )
        except Exception as e:
            # If DESCRIBE fails, check if it's because table doesn't exist
            error_msg = str(e).lower()
            if "doesn't exist" in error_msg or "not found" in error_msg:
                raise ValueError(f"Collection ('{collection_name}') not found: Table('{table_name}') not exists") from e
            raise

        # Extract dimension from embedding column
        for row in table_info:
            # Handle both dict and tuple formats
            if isinstance(row, dict):
                field_name = row.get("Field", row.get("field", ""))
                field_type = row.get("Type", row.get("type", ""))
            elif isinstance(row, (tuple, list)):
                field_name = row[0] if len(row) > 0 else ""
                field_type = row[1] if len(row) > 1 else ""
            else:
                continue

            if field_name == "embedding" and "vector" in str(field_type).lower():
                # Extract dimension from vector(dimension) format
                match = re.search(r"vector\s*\(\s*(\d+)\s*\)", str(field_type), re.IGNORECASE)
                if match:
                    metadata["dimension"] = int(match.group(1))
                break

        # Extract distance from CREATE TABLE statement
        try:
            create_table_result = self._execute(f"SHOW CREATE TABLE `{table_name}`")
            if create_table_result and len(create_table_result) > 0:
                # Handle both dict and tuple formats
                if isinstance(create_table_result[0], dict):
                    create_stmt = create_table_result[0].get(
                        "Create Table", create_table_result[0].get("create table", "")
                    )
                elif isinstance(create_table_result[0], (tuple, list)):
                    # CREATE TABLE statement is usually in the second column
                    create_stmt = create_table_result[0][1] if len(create_table_result[0]) > 1 else ""
                else:
                    create_stmt = str(create_table_result[0])

                # Extract distance from VECTOR INDEX ... with(distance=..., ...)
                # Pattern: VECTOR INDEX ... with(distance=l2, ...) or with(distance='l2', ...)
                # Match: with(distance=value, ...) where value can be l2, cosine, inner_product, or ip
                distance_match = re.search(
                    r'with\s*\([^)]*distance\s*=\s*([\'"]?)(\w+)\1',
                    create_stmt,
                    re.IGNORECASE,
                )
                if distance_match:
                    distance = metadata["distance"] = distance_match.group(2).lower()
                    # Normalize distance values
                    if distance == "ip":
                        distance = "inner_product"
                    elif distance in ["l2", "cosine", "inner_product"]:
                        pass
                    else:
                        # Unknown distance, default to None
                        logger.warning(
                            f"Unknown distance value '{distance}' in CREATE TABLE statement, defaulting to None"
                        )
                        metadata["distance"] = None
        except Exception as e:
            # If SHOW CREATE TABLE fails, log warning but continue
            logger.warning(f"Failed to get CREATE TABLE statement for '{table_name}': {e}")

        return metadata

    def _resolve_embedding_function(self, settings: str | None) -> EmbeddingFunction[EmbeddingDocuments] | None:
        """Resolve the embedding function to use for a collection."""
        if not settings:
            return None
        settings_json = json.loads(settings)
        ef_settings = settings_json.get("embedding_function", {})
        ef_name = ef_settings.get("name", "")
        if not ef_name:
            return None
        embedding_function_class = EmbeddingFunctionRegistry.get_class(ef_name)
        if not embedding_function_class:
            raise ValueError(f"Embedding function class '{ef_name}' not found")
        return embedding_function_class.build_from_config(ef_settings.get("properties", {}))

    def _resolve_sparse_vector_index_config(self, settings: str | None) -> SparseVectorIndexConfig | None:
        """Restore SparseVectorIndexConfig from persisted settings JSON."""
        if not settings:
            return None
        settings_json = json.loads(settings)
        sparse_settings = settings_json.get("sparse_vector_index")
        if not sparse_settings:
            return None

        # Resolve sparse embedding function (required)
        sparse_ef = None
        ef_info = sparse_settings.get("embedding_function")
        if not ef_info:
            raise ValueError("Sparse vector index settings missing required embedding_function configuration")

        ef_name = ef_info.get("name", "")
        if not ef_name:
            raise ValueError("Sparse vector index settings missing embedding_function name")

        sparse_ef_class = SparseEmbeddingFunctionRegistry.get_class(ef_name)
        if sparse_ef_class is None:
            raise ValueError(f"Sparse embedding function class '{ef_name}' not found in registry")
        sparse_ef = sparse_ef_class.build_from_config(ef_info.get("properties", {}))

        # Resolve source_key
        source_key_str = sparse_settings.get("source_key")
        source_key = FieldKey.DOCUMENT
        if source_key_str:
            if source_key_str == "#document" or source_key_str == FieldKey.DOCUMENT.name:
                source_key = FieldKey.DOCUMENT
            else:
                source_key = source_key_str

        return SparseVectorIndexConfig(
            embedding_function=sparse_ef,
            source_key=source_key,
        )

    def _validate_embedding_function(
        self,
        embedding_function: EmbeddingFunction | None,
        embedding_function_persistence: EmbeddingFunction | None,
    ) -> EmbeddingFunction[EmbeddingDocuments] | None:
        """
        Validate embedding function

        Args:
            embedding_function: Embedding function user provided
            embedding_function_persistence: Embedding function restored from table metadata

        Returns:
            Persisted or supplied embedding function, or None when the collection has no dense EF.
        """

        if embedding_function_persistence is not None and embedding_function is not _NOT_PROVIDED:
            if embedding_function is None or embedding_function_persistence.name() != embedding_function.name():
                raise ValueError(
                    "Both embedding function from parameter (not _NOT_PROVIDED, default value) and embedding function from persistence provided."
                )
            else:
                return embedding_function_persistence
        if embedding_function is _NOT_PROVIDED:
            if embedding_function_persistence is not None:
                return embedding_function_persistence
            return None
        else:
            return embedding_function

    def _get_collection(self, name: str, embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED) -> Collection:
        """Fetch a collection using the SDK catalog layout."""
        _validate_collection_name(name)
        collection_meta = self._resolve_collection_metadata_from_sdk_collections(name)
        if not collection_meta or not collection_meta.collection_id:
            raise ValueError(f"Collection '{name}' does not exist")

        state = collection_state(collection_meta.settings)
        if state == COLLECTION_STATE_CREATING:
            raise ValueError(f"Collection '{name}' is still being created; retry after it is ready")
        if state == COLLECTION_STATE_FAILED:
            raise ValueError(f"Collection '{name}' is in a failed creation state; clean it up before retrying")

        try:
            embedding_function_persistence = self._resolve_embedding_function(collection_meta.settings)
            embedding_function = self._validate_embedding_function(embedding_function, embedding_function_persistence)
            metadata = self._resolve_collection_metadata_from_table(
                CollectionNames.table_name(collection_meta.collection_id), name
            )

            # Resolve sparse vector index config from persisted settings
            sparse_vector_index_config = self._resolve_sparse_vector_index_config(collection_meta.settings)

            return Collection(
                client=self,
                name=name,
                collection_id=collection_meta.collection_id,
                embedding_function=embedding_function,
                dimension=metadata["dimension"],
                distance=metadata["distance"],
                sparse_vector_index_config=sparse_vector_index_config,
            )
        except Exception as e:
            raise ValueError(f"Failed to get collection: {e}") from e

    def delete_collection(self, name: str) -> None:
        """Delete a collection.

        Args:
            name: The name of the collection to delete.

        Raises:
            ValueError: If the collection does not exist.

        Examples:
            >>> client.delete_collection("my_collection")
        """
        _validate_collection_name(name)
        if self._has_ns_collection(name):
            self._delete_ns_collection_meta(name)
            logger.debug(f"Deleted namespace collection '{name}'")
            return
        self._delete_collection(name)
        logger.debug(f"✅ Successfully deleted collection '{name}' from sdk_collections table")

    def _delete_collection(self, name: str) -> None:
        """
        Delete a collection (user-facing API)

        Args:
            name: Collection name
        """
        collection = self._get_collection(name)
        if not collection:
            raise ValueError(f"Collection '{name}' does not exist")
        drop_table_sql = f"DROP TABLE `{CollectionNames.table_name(collection.id)}`"
        query_sql = f"DELETE FROM `{CollectionNames.sdk_collections_table_name()}` WHERE COLLECTION_NAME = %s"
        self._execute(drop_table_sql)
        self._execute(query_sql, [name])
        logger.debug(f"✅ Successfully deleted collection '{name}' from sdk_collections table")

    def list_collections(self) -> list[Collection]:
        """List all collections in the database.

        Returns:
            A list of ``Collection`` objects.

        Examples:
            >>> collections = client.list_collections()
            >>> for col in collections:
            ...     print(col.name)
        """
        collections = self._list_ns_collections()
        collections.extend(self._list_collections())
        return collections

    def _list_ns_collections(self) -> list[Collection]:
        """List namespace-enabled collections."""
        result = []
        try:
            sdk_table = CollectionNames.sdk_collections_table_name()
            check_sql = f"SHOW TABLES LIKE '{sdk_table}'"
            check_result = self._execute(check_sql)
            if not check_result:
                return result
            rows = self._execute(f"SELECT collection_id, collection_name, settings FROM `{sdk_table}`")
            for row in rows:
                try:
                    if isinstance(row, dict):
                        settings = json.loads(row["settings"]) if row.get("settings") else {}
                    else:
                        settings = json.loads(row[2]) if row[2] else {}
                    if not settings.get("use_namespace"):
                        continue
                    if isinstance(row, dict):
                        meta = {
                            "collection_id": str(row["collection_id"]),
                            "collection_name": row["collection_name"],
                            "settings": settings,
                        }
                    else:
                        meta = {
                            "collection_id": str(row[0]),
                            "collection_name": row[1],
                            "settings": settings,
                        }
                    result.append(self._build_ns_collection_from_meta(meta))
                except Exception as e:
                    logger.warning(f"Failed to build namespace collection from row: {e}")
        except Exception:
            logger.debug("Failed to list namespace collections from catalog", exc_info=True)
        return result

    def _list_collections(self) -> list[Collection]:
        """List standard collections using the SDK catalog layout."""
        collections = []
        try:
            # Detect if the sdk_collections table exists before querying it
            sdk_collections_table = CollectionNames.sdk_collections_table_name()
            has_sdk_collections = False
            try:
                check_table_sql = f"SHOW TABLES LIKE '{sdk_collections_table}'"
                check_result = self._execute(check_table_sql)
                if check_result:
                    # Table exists (SHOW TABLES LIKE returns at least one row if exists)
                    has_sdk_collections = True
            except Exception:
                has_sdk_collections = False

            if has_sdk_collections:
                query_sql = f"SELECT COLLECTION_ID, COLLECTION_NAME, SETTINGS FROM {sdk_collections_table}"
                rows = self._execute(query_sql)
                for row in rows:
                    collection_name = ""
                    try:
                        if isinstance(row, dict):
                            collection_id = str(row.get("COLLECTION_ID") or row.get("collection_id") or "")
                            collection_name = row.get("COLLECTION_NAME") or row.get("collection_name", "")
                            settings_raw = row.get("SETTINGS") or row.get("settings")
                        elif isinstance(row, (tuple, list)):
                            collection_id = str(row[0]) if len(row) > 0 else ""
                            collection_name = row[1] if len(row) > 1 else ""
                            settings_raw = row[2] if len(row) > 2 else None
                        else:
                            collection_id = ""
                            collection_name = str(row)
                            settings_raw = None
                        settings = None
                        if settings_raw:
                            try:
                                settings = json.loads(settings_raw) if isinstance(settings_raw, str) else settings_raw
                                if isinstance(settings, dict) and collection_state(settings) != COLLECTION_STATE_READY:
                                    continue
                                if isinstance(settings, dict) and settings.get("use_namespace"):
                                    continue
                            except (json.JSONDecodeError, TypeError):
                                settings = None
                        if collection_id and isinstance(settings, dict):
                            embedding_function = self._resolve_embedding_function(
                                settings_raw if isinstance(settings_raw, str) else json.dumps(settings)
                            )
                            metadata = self._resolve_collection_metadata_from_table(
                                CollectionNames.table_name(collection_id), collection_name
                            )
                            sparse_config = self._resolve_sparse_vector_index_config(
                                settings_raw if isinstance(settings_raw, str) else json.dumps(settings)
                            )
                            collection = Collection(
                                client=self,
                                name=collection_name,
                                collection_id=collection_id,
                                embedding_function=embedding_function,
                                dimension=metadata["dimension"],
                                distance=metadata["distance"],
                                sparse_vector_index_config=sparse_config,
                            )
                        else:
                            collection = self.get_collection(collection_name)
                        collections.append(collection)
                    except Exception as e:
                        logger.warning(
                            f"Failed to get collection. The data may be corrupted. The collection name: '{collection_name}': {e}"
                        )
                        continue

        except Exception as e:
            raise ValueError(f"Failed to list collections: {e}") from e
        return collections

    def count_collection(self) -> int:
        """Count the total number of collections.

        Returns:
            The number of collections.

        Examples:
            >>> count = client.count_collection()
            >>> print(f"Database has {count} collections")
        """
        collections = self.list_collections()
        return len(collections)

    def has_collection(self, name: str) -> bool:
        """Check if a collection exists.

        Args:
            name: The name of the collection to check.

        Returns:
            True if the collection exists, False otherwise.

        Examples:
            >>> if client.has_collection("my_collection"):
            ...     print("Collection exists!")
        """
        _validate_collection_name(name)
        return self._has_ns_collection(name) or self._has_collection(name)

    def _collection_table_exists(self, table_name: str) -> bool:
        """Return whether the physical table for a collection exists."""
        try:
            table_info = self._execute(f"DESCRIBE `{table_name}`")
            return table_info is not None and len(table_info) > 0
        except Exception:
            return False

    def _has_collection(self, name: str) -> bool:
        """Return whether a standard collection exists in the SDK catalog."""
        try:
            query_sql = (
                f"SELECT collection_id, settings FROM {self._qtable(CollectionNames.sdk_collections_table_name())} "
                "WHERE collection_name = %s "
                f"ORDER BY created_at, collection_id LIMIT 1"
            )
            rows = self._execute(query_sql, [name])
            if not rows or len(rows) == 0:
                return False

            row = rows[0]
            if isinstance(row, dict):
                settings = row.get("SETTINGS") or row.get("settings")
            elif isinstance(row, (tuple, list)):
                settings = row[1] if len(row) > 1 else None
            else:
                settings = None
            if collection_state(settings) != COLLECTION_STATE_READY:
                return False

            collection_id = _extract_collection_id_from_sdk_row(row)
            if not collection_id:
                return False

            return self._collection_table_exists(CollectionNames.table_name(collection_id))
        except Exception:
            return False

    def get_or_create_collection(
        self,
        name: str,
        schema: Schema | None = None,
        use_namespace: bool = False,
    ) -> Collection:
        """Get a collection if it exists, otherwise create it.

        Args:
            name: The name of the collection.
            schema: Schema configuration for fine-grained control of dense, sparse,
                full-text, and embedding-function settings.
            use_namespace: If True, create a namespace-enabled collection. Defaults to False.

        Returns:
            The existing or newly created ``Collection`` object.

        Raises:
            ValueError: If the schema/embedding function combination is invalid (e.g., dimension mismatch).

        Examples:
            >>> schema = Schema(
            ...     vector_index=VectorIndexConfig(hnsw=HNSWConfiguration(dimension=384))
            ... )
            >>> collection = client.get_or_create_collection("my_collection", schema=schema)
        """
        _validate_collection_name(name)
        embedding_function = schema.vector_index.embedding_function if schema is not None else _NOT_PROVIDED

        if self.has_collection(name):
            if use_namespace and self._is_incomplete_ns_collection(name):
                return self.create_collection(
                    name=name,
                    schema=schema,
                    use_namespace=use_namespace,
                )
            self._assert_get_or_create_namespace_mode_matches(name, use_namespace)
            return self.get_collection(name, embedding_function=embedding_function)

        try:
            return self.create_collection(
                name=name,
                schema=schema,
                use_namespace=use_namespace,
            )
        except Exception as exc:
            if _is_collection_conflict_error(exc) or _is_collection_not_ready_error(exc):
                return self._get_or_resume_existing_collection(
                    name,
                    schema=schema,
                    use_namespace=use_namespace,
                )
            raise

    def _get_or_resume_existing_collection(
        self,
        name: str,
        *,
        schema: Schema | None,
        use_namespace: bool,
    ) -> Collection:
        """Return an existing collection or resume an incomplete namespace-enabled one."""
        if use_namespace:
            if self._get_ns_collection_meta(name) is None:
                raise ValueError(f"Collection '{name}' conflicted during create but namespace metadata is missing")
            if self._is_incomplete_ns_collection(name):
                return self.create_collection(
                    name=name,
                    schema=schema,
                    use_namespace=use_namespace,
                )
        self._assert_get_or_create_namespace_mode_matches(name, use_namespace)
        embedding_function = schema.vector_index.embedding_function if schema is not None else _NOT_PROVIDED
        try:
            return self.get_collection(name, embedding_function=embedding_function)
        except ValueError as exc:
            if not _is_collection_not_ready_error(exc):
                raise
            return self._wait_for_collection_ready(name, embedding_function=embedding_function)

    def _wait_for_collection_ready(
        self,
        name: str,
        *,
        embedding_function: EmbeddingFunctionParam = _NOT_PROVIDED,
    ) -> Collection:
        """Wait for another creator to publish a collection before returning it."""
        deadline = time.monotonic() + _COLLECTION_READY_TIMEOUT_SECONDS
        last_error: ValueError | None = None
        while True:
            try:
                return self.get_collection(name, embedding_function=embedding_function)
            except ValueError as exc:
                if not _is_collection_not_ready_error(exc):
                    raise
                last_error = exc

            if time.monotonic() >= deadline:
                raise ValueError(
                    f"Collection '{name}' is still being created after {_COLLECTION_READY_TIMEOUT_SECONDS:g} seconds"
                ) from last_error
            time.sleep(_COLLECTION_READY_POLL_INTERVAL_SECONDS)

    def _assert_get_or_create_namespace_mode_matches(self, name: str, use_namespace: bool) -> None:
        """Reject get_or_create when an existing collection's namespace mode differs."""
        existing_use_namespace = self._get_ns_collection_meta(name) is not None
        if existing_use_namespace == use_namespace:
            return
        kind = "namespace-enabled" if existing_use_namespace else "standard"
        raise ValueError(
            f"Collection '{name}' already exists as a {kind} collection "
            f"(use_namespace={existing_use_namespace}), but get_or_create_collection was called with "
            f"use_namespace={use_namespace}. Delete the collection or use a different name."
        )

    def _get_collection_table_name(self, collection_id: str | None, collection_name: str) -> str:
        """Return the v2 physical table name for a collection."""
        if not collection_id:
            collection_id = self._get_collection_id(collection_name)
        return CollectionNames.table_name(collection_id)

    def _fork_table_enabled(self) -> bool:
        """Return whether table fork is enabled on the backend."""
        return self.backend_capabilities.supports_fork_table

    def _refresh_enabled(self) -> bool:
        """Return whether index refresh is enabled on the backend."""
        return self.backend_capabilities.supports_refresh_index

    def refresh_index(self) -> None:
        """
        Flush async vector index build tasks when supported.

        For unsupported database versions, this method is a no-op to keep
        collection-level API calls backward compatible.
        """
        if not self._refresh_enabled():
            return

        self._execute("CALL dbms_index_manager.refresh();")

    def _get_collection_id(self, collection_name: str) -> str:
        """Resolve the collection id for a collection name."""
        collection_id_query_sql = (
            f"SELECT collection_id FROM {self._qtable(CollectionNames.sdk_collections_table_name())} "
            "WHERE collection_name = %s "
            f"ORDER BY created_at, collection_id LIMIT 1"
        )
        collection_id_query_result = self._execute(collection_id_query_sql, [collection_name])
        if not collection_id_query_result or len(collection_id_query_result) == 0:
            raise ValueError(f"Collection not found: '{collection_name}'")
        collection_id = _extract_collection_id_from_sdk_row(collection_id_query_result[0])
        if not collection_id:
            raise ValueError(f"Collection not found: '{collection_name}'")
        return collection_id

    def _collection_fork(self, collection: Collection, forked_name: str) -> None:
        """Fork a standard collection and publish it after the physical fork finishes."""
        if not self._fork_table_enabled():
            raise ValueError("Fork is not enabled for this database")
        _validate_collection_name(forked_name)
        if self.has_collection(forked_name):
            raise ValueError(f"Collection '{forked_name}' already exists")

        self._create_sdk_collections_if_not_exists()
        source_meta = self._resolve_collection_metadata_from_sdk_collections(collection.name)
        if source_meta is None:
            raise ValueError(f"Collection '{collection.name}' does not exist")
        source_table_name = self._get_collection_table_name(collection.id, collection.name)
        settings = parse_collection_settings(source_meta.settings)
        settings["state"] = COLLECTION_STATE_CREATING
        creation_token = uuid.uuid4().hex
        settings[COLLECTION_CREATION_TOKEN_KEY] = creation_token
        forked_id: str | None = None
        insert_attempted = False
        try:
            insert_attempted = True
            self._execute(
                f"INSERT INTO {self._qtable(CollectionNames.sdk_collections_table_name())} "
                "(collection_name, settings) VALUES (%s, %s)",
                [forked_name, json.dumps(settings, ensure_ascii=False)],
            )
            forked_id = self._get_collection_id(forked_name)
            self._execute(
                f"FORK TABLE {_quote_sql_identifier(source_table_name)} "
                f"TO {_quote_sql_identifier(CollectionNames.table_name(forked_id))}"
            )
            self._mark_collection_ready(forked_name, forked_id, creation_token)
        except Exception as exc:
            if insert_attempted:
                with contextlib.suppress(Exception):
                    self._cleanup_failed_collection(forked_name, forked_id, creation_token)
            raise ValueError(f"Failed to fork collection: {exc}") from exc
        logger.debug("Successfully forked collection '%s' to '%s'", collection.name, forked_name)
