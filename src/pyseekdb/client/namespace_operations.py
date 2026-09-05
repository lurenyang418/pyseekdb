"""Synchronous namespace collection operations."""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import Any

from pymysql.converters import escape_string

from .configuration import DEFAULT_DISTANCE_METRIC, DEFAULT_VECTOR_DIMENSION
from .document_query_builder import where_document_knn_prefilterable
from .embedding_function import Documents as EmbeddingDocuments
from .embedding_function import EmbeddingFunction
from .filters import FilterBuilder
from .kernel_errors import namespace_kernel_error_guard
from .meta_info import NamespaceCollectionNames, NamespaceFieldNames
from .query_builder import embedding_to_hexstring as _embedding_to_hexstring
from .query_builder import parse_embedding_value as _parse_embedding_value
from .query_types import QueryHint
from .sql_utils import _query_hint_to_sql
from .validators import (
    _MAX_N_RESULTS,
    _MAX_NAMESPACE_BATCH_SIZE,
    _validate_include,
    _validate_n_results,
    _validate_namespace_explicit_embedding_dimensions,
    _validate_pagination,
    _validate_record_ids,
)

logger = logging.getLogger(__name__)

_NS_DATA_CONTENT_ID_EXPR = "JSON_UNQUOTE(JSON_EXTRACT(data_content, '$.id'))"


class NamespaceOperationsMixin:
    """Provide namespace catalog, DML, DQL, and hybrid-search operations."""

    @staticmethod
    def _rewrite_where_for_ns(where: dict[str, Any] | None) -> dict[str, Any] | None:
        """Rewrite a WHERE clause to target namespace-scoped columns."""
        if where is None:
            return None
        rewritten = {}
        for k, v in where.items():
            if k in ("$and", "$or"):
                rewritten[k] = [NamespaceOperationsMixin._rewrite_where_for_ns(sub) for sub in v]
            elif k == "$not":
                rewritten[k] = NamespaceOperationsMixin._rewrite_where_for_ns(v)
            else:
                rewritten[f"metadata.{k}"] = v
        return rewritten

    @staticmethod
    def _append_namespace_filter(
        where_clause: str,
        params: list[Any],
        namespace_id: int,
        ltable_id: int,
    ) -> tuple[str, list[Any]]:
        """Append the namespace id filter to a WHERE clause."""
        ns_cond = f"namespace_id = {namespace_id} AND ltable_id = {ltable_id}"
        if not where_clause:
            return f"WHERE {ns_cond}", params
        if where_clause.strip().upper().startswith("WHERE"):
            inner = where_clause.strip()[5:].strip()
            return f"WHERE {ns_cond} AND ({inner})", params
        return f"WHERE {ns_cond} AND ({where_clause})", params

    @staticmethod
    def _validate_namespace_explicit_embeddings_if_needed(
        embeddings: list[list[float]] | None,
        *,
        explicit_embeddings: bool,
        has_vector_index: bool,
        collection_dimension: int | None,
    ) -> None:
        """Validate user-supplied embeddings match the collection VECTOR column dimension."""
        if not explicit_embeddings or not embeddings:
            return
        expected = collection_dimension if collection_dimension is not None else DEFAULT_VECTOR_DIMENSION
        _validate_namespace_explicit_embedding_dimensions(
            embeddings,
            expected_dimension=expected,
            has_vector_index=has_vector_index,
        )

    @staticmethod
    def _warn_explicit_embeddings_override_embedding_function(
        *,
        operation: str,
        explicit_embeddings: bool,
        has_documents: bool,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None,
    ) -> None:
        """Log when explicit embeddings take priority over embedding_function."""
        if explicit_embeddings and has_documents and embedding_function is not None:
            logger.warning(
                "%s: explicit embeddings provided together with documents while an "
                "embedding_function is configured; using explicit embeddings and "
                "not calling embedding_function.",
                operation,
            )

    def _count_namespace_records_by_id(
        self,
        table_name: str,
        namespace_id: int,
        ltable_id: int,
        record_id: str,
    ) -> int:
        """Count namespace records matching the given id."""
        id_expr = _NS_DATA_CONTENT_ID_EXPR
        sql = (
            f"SELECT COUNT(*) AS cnt FROM `{table_name}` "
            f"WHERE namespace_id = {int(namespace_id)} AND ltable_id = {int(ltable_id)} "
            f"AND {id_expr} = %s"
        )
        conn = self._ensure_connection()
        use_ctx = self._use_context_manager_for_cursor()
        rows = self._execute_query_with_cursor(conn, sql, [record_id], use_ctx)
        if not rows:
            return 0
        row = rows[0]
        if isinstance(row, dict):
            return int(row.get("cnt", 0))
        if isinstance(row, (tuple, list)):
            return int(row[0])
        return int(row)

    def _delete_namespace_records_by_id(
        self,
        table_name: str,
        namespace_id: int,
        ltable_id: int,
        record_id: str,
    ) -> None:
        """Delete namespace records matching the given id."""
        id_expr = _NS_DATA_CONTENT_ID_EXPR
        sql = (
            f"DELETE FROM `{table_name}` "
            f"WHERE namespace_id = {int(namespace_id)} AND ltable_id = {int(ltable_id)} "
            f"AND {id_expr} = %s"
        )
        conn = self._ensure_connection()
        use_ctx = self._use_context_manager_for_cursor()
        if use_ctx:
            with conn.cursor() as cursor:
                cursor.execute(sql, [record_id])
        else:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, [record_id])
            finally:
                cursor.close()

    def _reconcile_namespace_duplicate_records(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        ltable_id: int,
        table_name: str,
        ids: list[str],
        documents: list[str] | None,
        metadatas: list[dict] | None,
        embeddings: list[list[float]] | None,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None,
        **kwargs: Any,
    ) -> None:
        """Collapse concurrent upsert races to a single row per business id.

        Namespace business IDs currently live inside ``data_content`` and are
        not backed by a unique table constraint. This is therefore a
        best-effort post-write reconciliation step, not an atomic upsert
        primitive; making it atomic requires a schema/index design change.
        """
        if not ids or collection_id is None:
            return
        ns_id = int(namespace_id)
        for i, record_id in enumerate(ids):
            doc_val = documents[i] if documents and i < len(documents) else None
            meta_val = metadatas[i] if metadatas and i < len(metadatas) else None
            emb_val = embeddings[i] if embeddings and i < len(embeddings) else None
            for attempt in range(120):
                duplicate_count = self._count_namespace_records_by_id(table_name, ns_id, ltable_id, record_id)
                if duplicate_count <= 1:
                    break
                self._delete_namespace_records_by_id(table_name, ns_id, ltable_id, record_id)
                if self._count_namespace_records_by_id(table_name, ns_id, ltable_id, record_id) == 0:
                    self._namespace_add(
                        collection_id=collection_id,
                        collection_name=collection_name,
                        namespace_id=namespace_id,
                        namespace_name=namespace_name,
                        ids=[record_id],
                        embeddings=[emb_val] if emb_val is not None else None,
                        metadatas=[meta_val] if meta_val is not None else None,
                        documents=[doc_val] if doc_val is not None else None,
                        embedding_function=embedding_function,
                        **kwargs,
                    )
                if attempt < 119:
                    time.sleep(0.05 * min(attempt + 1, 10))
            else:
                raise ValueError(f"Failed to reconcile duplicate namespace rows for record_id={record_id!r}")

    @namespace_kernel_error_guard
    def _namespace_add(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None = None,
        **kwargs,
    ) -> None:
        """Add records to a namespace collection."""
        has_vector_index = kwargs.pop("has_vector_index", True)
        collection_dimension = kwargs.pop("collection_dimension", None)
        explicit_embeddings = embeddings is not None
        if isinstance(ids, str):
            ids = [ids]
        _validate_record_ids(ids)
        if not ids:
            raise ValueError("ids must not be empty")
        if len(ids) > _MAX_NAMESPACE_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(ids)} exceeds maximum allowed {_MAX_NAMESPACE_BATCH_SIZE} records per request."
            )
        if isinstance(documents, str):
            documents = [documents]
        if metadatas is not None and isinstance(metadatas, dict):
            metadatas = [metadatas]
        if (
            embeddings is not None
            and isinstance(embeddings, list)
            and len(embeddings) > 0
            and not isinstance(embeddings[0], list)
        ):
            embeddings = [embeddings]

        self._warn_explicit_embeddings_override_embedding_function(
            operation="namespace.add",
            explicit_embeddings=explicit_embeddings,
            has_documents=bool(documents),
            embedding_function=embedding_function,
        )

        if not embeddings and documents:
            if embedding_function is not None:
                embeddings = embedding_function(documents)
            else:
                raise ValueError(
                    "Documents provided but no embeddings and no embedding function. "
                    "Either:\n"
                    "  1. Provide embeddings directly when calling add(), or\n"
                    "  2. Provide embedding_function to auto-generate embeddings from documents."
                )
        elif not embeddings and not documents and not metadatas:
            raise ValueError(
                "Neither embeddings, documents, nor metadatas provided. "
                "Please provide at least one of:\n"
                "  1. embeddings directly,\n"
                "  2. documents with embedding_function to generate embeddings, or\n"
                "  3. metadatas for metadata-only add."
            )

        num_items = len(ids)
        if documents and len(documents) != num_items:
            raise ValueError(f"Number of documents ({len(documents)}) does not match number of ids ({num_items})")
        if metadatas and len(metadatas) != num_items:
            raise ValueError(f"Number of metadatas ({len(metadatas)}) does not match number of ids ({num_items})")
        if embeddings and len(embeddings) != num_items:
            raise ValueError(f"Number of embeddings ({len(embeddings)}) does not match number of ids ({num_items})")

        self._validate_namespace_explicit_embeddings_if_needed(
            embeddings,
            explicit_embeddings=explicit_embeddings,
            has_vector_index=has_vector_index,
            collection_dimension=collection_dimension,
        )

        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        table_name = NamespaceCollectionNames.data_table_name(collection_id)
        ns_id = int(namespace_id)

        values_list = []
        for i in range(num_items):
            doc_val = documents[i] if documents else None
            doc_sql = f"'{escape_string(doc_val)}'" if doc_val is not None else "NULL"

            vec_val = embeddings[i] if embeddings else None
            vec_sql = "NULL" if vec_val is None else _embedding_to_hexstring(vec_val)

            meta_val = metadatas[i] if metadatas else None
            data_content = {"id": ids[i]}
            if meta_val is not None:
                data_content["metadata"] = meta_val
            dc_json = json.dumps(data_content, ensure_ascii=False)
            dc_sql = f"'{escape_string(dc_json)}'"

            values_list.append(f"({ns_id}, {ltable_id}, {doc_sql}, {vec_sql}, {dc_sql})")

        columns = (
            f"{NamespaceFieldNames.NAMESPACE_ID}, {NamespaceFieldNames.LTABLE_ID}, "
            f"{NamespaceFieldNames.DOCUMENT}, {NamespaceFieldNames.EMBEDDING}, {NamespaceFieldNames.DATA_CONTENT}"
        )
        sql = f"INSERT INTO `{table_name}` ({columns}) VALUES {','.join(values_list)}"
        self._execute(sql)

    @namespace_kernel_error_guard
    def _namespace_update(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None = None,
        **kwargs,
    ) -> None:
        """Update existing records in a namespace collection."""
        has_vector_index = kwargs.pop("has_vector_index", True)
        collection_dimension = kwargs.pop("collection_dimension", None)
        explicit_embeddings = embeddings is not None
        if isinstance(ids, str):
            ids = [ids]
        _validate_record_ids(ids)
        if not ids:
            raise ValueError("ids must not be empty")
        if len(ids) > _MAX_NAMESPACE_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(ids)} exceeds maximum allowed {_MAX_NAMESPACE_BATCH_SIZE} records per request."
            )
        if isinstance(documents, str):
            documents = [documents]
        if metadatas is not None and isinstance(metadatas, dict):
            metadatas = [metadatas]
        if (
            embeddings is not None
            and isinstance(embeddings, list)
            and len(embeddings) > 0
            and not isinstance(embeddings[0], list)
        ):
            embeddings = [embeddings]

        self._warn_explicit_embeddings_override_embedding_function(
            operation="namespace.update",
            explicit_embeddings=explicit_embeddings,
            has_documents=bool(documents),
            embedding_function=embedding_function,
        )

        if not embeddings and documents:
            # embeddings not provided but documents are provided, check for embedding_function
            if embedding_function is not None:
                embeddings = embedding_function(documents)
            else:
                raise ValueError(
                    "Documents provided but no embeddings and no embedding function. "
                    "Either:\n"
                    "  1. Provide embeddings directly when calling update(), or\n"
                    "  2. Provide embedding_function to auto-generate embeddings from documents."
                )
        elif not embeddings and not metadatas:
            raise ValueError(
                "Neither embeddings nor documents nor metadatas provided. "
                "Please provide at least one of embeddings, documents with an embedding_function, or metadatas."
            )

        self._validate_namespace_explicit_embeddings_if_needed(
            embeddings,
            explicit_embeddings=explicit_embeddings,
            has_vector_index=has_vector_index,
            collection_dimension=collection_dimension,
        )

        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        table_name = NamespaceCollectionNames.data_table_name(collection_id)
        ns_id = int(namespace_id)

        id_expr = _NS_DATA_CONTENT_ID_EXPR
        active_ids = []
        for i, record_id in enumerate(ids):
            has_update = (
                (documents and i < len(documents) and documents[i] is not None)
                or (embeddings and i < len(embeddings) and embeddings[i] is not None)
                or (metadatas and i < len(metadatas) and metadatas[i] is not None)
            )
            if has_update:
                active_ids.append((i, record_id))

        if not active_ids:
            return

        conn = self._ensure_connection()
        use_context_manager = self._use_context_manager_for_cursor()

        # VECTOR columns do not support CASE-WHEN in OceanBase, so embedding
        # updates are executed per-row while document/metadata use batch CASE-WHEN.
        emb_ids = []
        for i, record_id in active_ids:
            if embeddings and i < len(embeddings) and embeddings[i] is not None:
                emb_ids.append((i, record_id))

        if emb_ids:
            for i, record_id in emb_ids:
                vec_sql = _embedding_to_hexstring(embeddings[i])
                sql = (
                    f"UPDATE `{table_name}` SET embedding = {vec_sql} "
                    f"WHERE namespace_id = {ns_id} AND ltable_id = {ltable_id} "
                    f"AND {id_expr} = %s"
                )
                if use_context_manager:
                    with conn.cursor() as cursor:
                        cursor.execute(sql, [record_id])
                else:
                    cursor = conn.cursor()
                    try:
                        cursor.execute(sql, [record_id])
                    finally:
                        cursor.close()

        doc_case_parts = []
        meta_case_parts = []
        params = []
        has_doc = False
        has_meta = False
        batch_ids = []

        for i, record_id in active_ids:
            if documents and i < len(documents) and documents[i] is not None:
                has_doc = True
                doc_case_parts.append(f"WHEN {id_expr} = %s THEN %s")
                params.extend([record_id, documents[i]])
                if record_id not in batch_ids:
                    batch_ids.append(record_id)
            if metadatas and i < len(metadatas) and metadatas[i] is not None:
                has_meta = True
                meta_json = json.dumps(metadatas[i], ensure_ascii=False)
                meta_case_parts.append(
                    f"WHEN {id_expr} = %s THEN JSON_SET(data_content, '$.metadata', CAST(%s AS JSON))"
                )
                params.extend([record_id, meta_json])
                if record_id not in batch_ids:
                    batch_ids.append(record_id)

        if has_doc or has_meta:
            set_clauses = []
            if has_doc:
                set_clauses.append(f"document = CASE {' '.join(doc_case_parts)} ELSE document END")
            if has_meta:
                set_clauses.append(f"data_content = CASE {' '.join(meta_case_parts)} ELSE data_content END")

            id_placeholders = ", ".join(["%s"] * len(batch_ids))
            params.extend(batch_ids)

            sql = (
                f"UPDATE `{table_name}` SET {', '.join(set_clauses)} "
                f"WHERE namespace_id = {ns_id} AND ltable_id = {ltable_id} "
                f"AND {id_expr} IN ({id_placeholders})"
            )
            if use_context_manager:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
            else:
                cursor = conn.cursor()
                try:
                    cursor.execute(sql, params)
                finally:
                    cursor.close()

    @namespace_kernel_error_guard
    def _namespace_upsert(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None = None,
        **kwargs,
    ) -> None:
        """Insert or update records in a namespace collection."""
        has_vector_index = kwargs.pop("has_vector_index", True)
        collection_dimension = kwargs.pop("collection_dimension", None)
        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        if isinstance(ids, str):
            ids = [ids]
        _validate_record_ids(ids)
        if not ids:
            raise ValueError("ids must not be empty")
        if len(ids) > _MAX_NAMESPACE_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(ids)} exceeds maximum allowed {_MAX_NAMESPACE_BATCH_SIZE} records per request."
            )
        if isinstance(documents, str):
            documents = [documents]
        if metadatas is not None and isinstance(metadatas, dict):
            metadatas = [metadatas]
        if (
            embeddings is not None
            and isinstance(embeddings, list)
            and len(embeddings) > 0
            and not isinstance(embeddings[0], list)
        ):
            embeddings = [embeddings]

        table_name = NamespaceCollectionNames.data_table_name(collection_id)
        ns_id = int(namespace_id)

        existing_ids = set()
        id_expr = _NS_DATA_CONTENT_ID_EXPR
        id_placeholders = ", ".join(["%s"] * len(ids))
        check_sql = (
            f"SELECT JSON_EXTRACT(data_content, '$.id') AS rid FROM `{table_name}` "
            f"WHERE namespace_id = {ns_id} AND ltable_id = {ltable_id} "
            f"AND {id_expr} IN ({id_placeholders})"
        )
        conn = self._ensure_connection()
        use_ctx = self._use_context_manager_for_cursor()
        rows = self._execute_query_with_cursor(conn, check_sql, list(ids), use_ctx)
        for row in rows:
            rid_raw = row.get("rid") if isinstance(row, dict) else row[0]
            rid = json.loads(rid_raw) if isinstance(rid_raw, str) else rid_raw
            if rid is not None:
                existing_ids.add(str(rid))

        add_indices = []
        update_indices = []
        for i, rid in enumerate(ids):
            if rid in existing_ids:
                update_indices.append(i)
            else:
                add_indices.append(i)

        if add_indices:
            add_ids = [ids[i] for i in add_indices]
            add_docs = [documents[i] for i in add_indices] if documents else None
            add_metas = [metadatas[i] for i in add_indices] if metadatas else None
            add_embs = [embeddings[i] for i in add_indices] if embeddings else None
            self._namespace_add(
                collection_id=collection_id,
                collection_name=collection_name,
                namespace_id=namespace_id,
                namespace_name=namespace_name,
                ids=add_ids,
                embeddings=add_embs,
                metadatas=add_metas,
                documents=add_docs,
                embedding_function=embedding_function,
                has_vector_index=has_vector_index,
                collection_dimension=collection_dimension,
                **kwargs,
            )

        if update_indices:
            upd_ids = [ids[i] for i in update_indices]
            upd_docs = [documents[i] for i in update_indices] if documents else None
            upd_metas = [metadatas[i] for i in update_indices] if metadatas else None
            upd_embs = [embeddings[i] for i in update_indices] if embeddings else None
            self._namespace_update(
                collection_id=collection_id,
                collection_name=collection_name,
                namespace_id=namespace_id,
                namespace_name=namespace_name,
                ids=upd_ids,
                embeddings=upd_embs,
                metadatas=upd_metas,
                documents=upd_docs,
                embedding_function=embedding_function,
                has_vector_index=has_vector_index,
                collection_dimension=collection_dimension,
                **kwargs,
            )

        self._reconcile_namespace_duplicate_records(
            collection_id=collection_id,
            collection_name=collection_name,
            namespace_id=namespace_id,
            namespace_name=namespace_name,
            ltable_id=ltable_id,
            table_name=table_name,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            embedding_function=embedding_function,
            **kwargs,
        )

    @namespace_kernel_error_guard
    def _namespace_delete(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        ids: str | list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Delete records from a namespace collection."""
        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        if ids is None and where is None and where_document is None:
            raise ValueError("At least one of ids, where, or where_document must be provided")

        table_name = NamespaceCollectionNames.data_table_name(collection_id)
        ns_id = int(namespace_id)

        conditions = []
        params = []

        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
            _validate_record_ids(ids)
            if not ids:
                raise ValueError("ids must not be empty")
            id_placeholders = " OR ".join([f"{_NS_DATA_CONTENT_ID_EXPR} = %s"] * len(ids))
            conditions.append(f"({id_placeholders})")
            params.extend(ids)

        if where is not None:
            rewritten = self._rewrite_where_for_ns(where)
            meta_clause, meta_params = FilterBuilder.build_metadata_filter(rewritten, "data_content")
            if meta_clause:
                conditions.append(meta_clause)
                params.extend(meta_params)

        if where_document is not None:
            doc_clause, doc_params = FilterBuilder.build_document_filter(where_document, "document")
            if doc_clause:
                conditions.append(doc_clause)
                params.extend(doc_params)

        user_where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        where_clause, params = self._append_namespace_filter(user_where, params, ns_id, ltable_id)
        sql = f"DELETE FROM `{table_name}` {where_clause}"
        if params:
            conn = self._ensure_connection()
            use_context_manager = self._use_context_manager_for_cursor()
            if use_context_manager:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
            else:
                cursor = conn.cursor()
                try:
                    cursor.execute(sql, params)
                finally:
                    cursor.close()
        else:
            self._execute(sql)

    @namespace_kernel_error_guard
    def _namespace_query(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        query_embeddings: list[float] | list[list[float]] | None = None,
        query_texts: str | list[str] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run a vector query against a namespace collection."""
        _validate_n_results(n_results)
        _validate_include(include)
        has_vector_index = kwargs.pop("has_vector_index", True)
        collection_dimension = kwargs.pop("collection_dimension", kwargs.pop("dimension", None))
        explicit_query_embeddings = query_embeddings is not None
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None = kwargs.get("embedding_function")
        distance = kwargs.get("distance", DEFAULT_DISTANCE_METRIC)

        self._warn_explicit_embeddings_override_embedding_function(
            operation="namespace.query",
            explicit_embeddings=explicit_query_embeddings,
            has_documents=query_texts is not None,
            embedding_function=embedding_function,
        )

        if query_embeddings is None:
            if query_texts is not None:
                if embedding_function is not None:
                    query_embeddings = self._embed_texts(query_texts, embedding_function=embedding_function)
                else:
                    raise ValueError("query_texts provided but no embedding_function.")
            else:
                raise ValueError("Neither query_embeddings nor query_texts provided.")

        query_embeddings = self._normalize_query_embeddings(query_embeddings)
        if not query_embeddings:
            raise ValueError("query_embeddings must not be empty")
        self._validate_namespace_explicit_embeddings_if_needed(
            query_embeddings,
            explicit_embeddings=explicit_query_embeddings,
            has_vector_index=has_vector_index,
            collection_dimension=collection_dimension,
        )

        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        include_fields = self._normalize_include_fields(include)

        # Logical-table vector search must use hybrid_search DSL; direct SQL vector
        # index scans on namespace partitions are not supported on OceanBase.
        all_ids: list[list[Any]] = []
        all_documents: list[list[Any]] = []
        all_metadatas: list[list[Any]] = []
        all_embeddings: list[list[Any]] = []
        all_distances: list[list[float]] = []

        hybrid_kwargs = {k: v for k, v in kwargs.items() if k not in ("embedding_function", "distance", "dimension")}

        for query_vector in query_embeddings:
            knn_cfg: dict[str, Any] = {
                "query_embeddings": query_vector,
                "n_results": n_results,
            }
            if where is not None:
                knn_cfg["where"] = where

            post_filter_wd: dict[str, Any] | str | None = None
            hybrid_include = include
            if where_document is not None:
                if where_document_knn_prefilterable(where_document):
                    knn_cfg["where_document"] = where_document
                else:
                    post_filter_wd = where_document
                    knn_cfg["n_results"] = min(max(n_results * 5, n_results), _MAX_N_RESULTS)
                    if include is None:
                        hybrid_include = ["documents", "metadatas"]
                    elif not {"documents", "document"} & {*(include or [])}:
                        hybrid_include = ["documents", *include]

            batch = self._namespace_hybrid_search(
                collection_id=collection_id,
                collection_name=collection_name,
                namespace_id=namespace_id,
                namespace_name=namespace_name,
                query=None,
                knn=knn_cfg,
                n_results=knn_cfg["n_results"],
                include=hybrid_include,
                embedding_function=embedding_function,
                distance=distance,
                dimension=collection_dimension,
                **hybrid_kwargs,
            )

            if post_filter_wd is not None:
                batch = self._post_filter_namespace_query_result(
                    batch,
                    post_filter_wd,
                    n_results=n_results,
                )

            batch_ids = batch.get("ids") or [[]]
            all_ids.append(batch_ids[0] if batch_ids else [])
            all_distances.append((batch.get("distances") or [[]])[0])

            if "documents" in include_fields or "document" in include_fields or include is None:
                batch_docs = batch.get("documents") or [[]]
                all_documents.append(batch_docs[0] if batch_docs else [])
            if "metadatas" in include_fields or "metadata" in include_fields or include is None:
                batch_meta = batch.get("metadatas") or [[]]
                all_metadatas.append(batch_meta[0] if batch_meta else [])
            if "embeddings" in include_fields or "embedding" in include_fields:
                batch_emb = batch.get("embeddings") or [[]]
                all_embeddings.append(batch_emb[0] if batch_emb else [])

        result: dict[str, Any] = {"ids": all_ids, "distances": all_distances}
        if "documents" in include_fields or "document" in include_fields or include is None:
            result["documents"] = all_documents
        if "metadatas" in include_fields or "metadata" in include_fields or include is None:
            result["metadatas"] = all_metadatas
        if "embeddings" in include_fields or "embedding" in include_fields:
            result["embeddings"] = all_embeddings
        return result

    @namespace_kernel_error_guard
    def _namespace_get(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        ids: str | list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Fetch records from a namespace collection by id/where/pagination."""
        _validate_pagination(limit, offset)
        _validate_include(include)
        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        include_fields = self._normalize_include_fields(include)
        table_name = NamespaceCollectionNames.data_table_name(collection_id)
        ns_id = int(namespace_id)

        select_parts = ["JSON_EXTRACT(data_content, '$.id') AS record_id"]
        if include_fields.get("documents") or include_fields.get("document") or include is None:
            select_parts.append("document")
        if include_fields.get("metadatas") or include_fields.get("metadata") or include is None:
            select_parts.append("JSON_EXTRACT(data_content, '$.metadata') AS metadata")
        if include_fields.get("embeddings") or include_fields.get("embedding"):
            select_parts.append("embedding")

        user_conditions = []
        params = []

        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
            id_conds = []
            for rid in ids:
                id_escaped = escape_string(rid)
                id_conds.append(f"{_NS_DATA_CONTENT_ID_EXPR} = '{id_escaped}'")
            user_conditions.append(f"({' OR '.join(id_conds)})")

        if where is not None:
            rewritten = self._rewrite_where_for_ns(where)
            meta_clause, meta_params = FilterBuilder.build_metadata_filter(rewritten, "data_content")
            if meta_clause:
                user_conditions.append(meta_clause)
                params.extend(meta_params)

        if where_document is not None:
            doc_clause, doc_params = FilterBuilder.build_document_filter(where_document, "document")
            if doc_clause:
                user_conditions.append(doc_clause)
                params.extend(doc_params)

        user_where = f"WHERE {' AND '.join(user_conditions)}" if user_conditions else ""
        where_str, params = self._append_namespace_filter(user_where, params, ns_id, ltable_id)
        select_clause = ", ".join(select_parts)
        sql = f"SELECT {select_clause} FROM `{table_name}` {where_str}"
        if limit is None and offset is not None:
            limit = 100
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
            if offset is not None:
                sql += f" OFFSET {int(offset)}"

        conn = self._ensure_connection()
        use_context_manager = self._use_context_manager_for_cursor()
        rows = self._execute_query_with_cursor(conn, sql, params if params else [], use_context_manager)

        result_ids = []
        result_documents = []
        result_metadatas = []
        result_embeddings = []

        for row in rows:
            if isinstance(row, dict):
                rid_raw = row.get("record_id")
                rid = json.loads(rid_raw) if isinstance(rid_raw, str) else rid_raw
                result_ids.append(rid)
                if "documents" in include_fields or "document" in include_fields or include is None:
                    result_documents.append(row.get("document"))
                if "metadatas" in include_fields or "metadata" in include_fields or include is None:
                    meta_raw = row.get("metadata")
                    if isinstance(meta_raw, str):
                        meta_raw = json.loads(meta_raw)
                    result_metadatas.append(meta_raw or {})
                if "embeddings" in include_fields or "embedding" in include_fields:
                    emb = _parse_embedding_value(row.get("embedding"))
                    result_embeddings.append(emb)
            elif isinstance(row, (list, tuple)):
                idx = 0
                rid_raw = row[idx]
                idx += 1
                rid = json.loads(rid_raw) if isinstance(rid_raw, str) else rid_raw
                result_ids.append(rid)
                if "documents" in include_fields or "document" in include_fields or include is None:
                    result_documents.append(row[idx])
                    idx += 1
                if "metadatas" in include_fields or "metadata" in include_fields or include is None:
                    meta_raw = row[idx]
                    idx += 1
                    if isinstance(meta_raw, str):
                        meta_raw = json.loads(meta_raw)
                    result_metadatas.append(meta_raw or {})
                if "embeddings" in include_fields or "embedding" in include_fields:
                    emb = row[idx]
                    idx += 1
                    emb = _parse_embedding_value(emb)
                    result_embeddings.append(emb)

        result = {"ids": result_ids}
        if "documents" in include_fields or "document" in include_fields or include is None:
            result["documents"] = result_documents
        if "metadatas" in include_fields or "metadata" in include_fields or include is None:
            result["metadatas"] = result_metadatas
        if "embeddings" in include_fields or "embedding" in include_fields:
            result["embeddings"] = result_embeddings
        return result

    @namespace_kernel_error_guard
    def _namespace_count(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        **kwargs,
    ) -> int:
        """Count records in a namespace collection."""
        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        table_name = NamespaceCollectionNames.data_table_name(collection_id)
        ns_id = int(namespace_id)
        where_clause, _ = self._append_namespace_filter("", [], ns_id, ltable_id)
        sql = f"SELECT COUNT(*) AS cnt FROM `{table_name}` {where_clause}"
        conn = self._ensure_connection()
        use_context_manager = self._use_context_manager_for_cursor()
        rows = self._execute_query_with_cursor(conn, sql, [], use_context_manager)
        if not rows:
            return 0
        row = rows[0]
        if isinstance(row, dict):
            return row.get("cnt", 0)
        elif isinstance(row, (tuple, list)):
            return row[0] if len(row) > 0 else 0
        return int(row) if row else 0

    @namespace_kernel_error_guard
    def _namespace_peek(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        limit: int = 10,
        **kwargs,
    ) -> dict[str, Any]:
        """Return a small sample of records from a namespace collection."""
        return self._namespace_get(
            collection_id=collection_id,
            collection_name=collection_name,
            namespace_id=namespace_id,
            namespace_name=namespace_name,
            limit=limit,
            offset=0,
            include=["documents", "metadatas", "embeddings"],
        )

    def _adapt_search_parm_for_ns(self, search_parm: dict[str, Any], ns_id: int, lt_id: int) -> dict[str, Any]:
        """Adapt search parameters to the namespace-scoped table layout."""
        ns_filter = [
            {"term": {"namespace_id": ns_id}},
            {"term": {"ltable_id": int(lt_id)}},
        ]

        def _rewrite_field_refs(obj):
            """Rewrite field references to their namespace-scoped equivalents."""
            if isinstance(obj, dict):
                new_dict = {}
                for k, v in obj.items():
                    new_key = k
                    if k == "_id":
                        new_key = "data_content.id"
                    elif isinstance(k, str):
                        prefix = "(JSON_EXTRACT(metadata, '$."
                        suffix = "'))"
                        if k.startswith(prefix) and k.endswith(suffix) and len(k) > len(prefix) + len(suffix):
                            inner = k[len(prefix) : -len(suffix)]
                            new_key = f"data_content.metadata.{inner}"
                    new_dict[new_key] = _rewrite_field_refs(v)
                return new_dict
            elif isinstance(obj, list):
                return [_rewrite_field_refs(item) for item in obj]
            return obj

        search_parm = _rewrite_field_refs(search_parm)

        if "_source" in search_parm:
            needs_data_content = False
            new_source = []
            for field in search_parm["_source"]:
                if field in ("_id", "metadata"):
                    needs_data_content = True
                else:
                    new_source.append(field)
            if needs_data_content:
                new_source.insert(0, "data_content")
            search_parm["_source"] = new_source

        def _inject_filter_into_knn(node):
            """Inject the namespace filter into a KNN search expression."""
            if isinstance(node, dict):
                if "filter" in node:
                    existing = node["filter"]
                    if isinstance(existing, list):
                        node["filter"] = existing + ns_filter
                    else:
                        node["filter"] = [existing, *ns_filter]
                else:
                    node["filter"] = list(ns_filter)
            return node

        # Scoring (full-text) leaf queries may live in a `must` clause; scalar
        # leaf queries (term/terms/range/json/array) MUST go into `filter`. The
        # kernel rejects a scalar query inside must/should of a scoring bool with
        # `OB_NOT_SUPPORTED: scalar term query in must/should clause`, and a
        # top-level bool query is treated as scoring by default.
        _scoring_leaf_keys = {"query_string", "match", "multi_match", "match_phrase"}

        def _inject_filter_into_query(node):
            """Inject the namespace filter into a query expression."""
            if not isinstance(node, dict):
                return node
            if "bool" in node:
                bool_node = node["bool"]
                if "filter" in bool_node:
                    existing = bool_node["filter"]
                    if isinstance(existing, list):
                        bool_node["filter"] = existing + ns_filter
                    else:
                        bool_node["filter"] = [existing, *ns_filter]
                else:
                    bool_node["filter"] = list(ns_filter)
                return node
            if _scoring_leaf_keys & node.keys():
                return {"bool": {"must": [node], "filter": list(ns_filter)}}
            return {"bool": {"filter": [node, *ns_filter]}}

        if "query" in search_parm:
            q = search_parm["query"]
            if isinstance(q, list):
                search_parm["query"] = [_inject_filter_into_query(item) for item in q]
            else:
                search_parm["query"] = _inject_filter_into_query(q)

        if "knn" in search_parm:
            knn = search_parm["knn"]
            if isinstance(knn, list):
                for item in knn:
                    _inject_filter_into_knn(item)
            else:
                _inject_filter_into_knn(knn)

        if "query" not in search_parm and "knn" not in search_parm:
            search_parm["query"] = {"bool": {"filter": list(ns_filter)}}

        return search_parm

    @namespace_kernel_error_guard
    def _namespace_hybrid_search(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        query: dict[str, Any] | None = None,
        knn: dict[str, Any] | None = None,
        rank: dict[str, Any] | None = None,
        n_results: int = 10,
        include: list[str] | None = None,
        query_hint: QueryHint | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run a hybrid (vector + fulltext) search against a namespace collection."""
        _validate_n_results(n_results)
        _validate_include(include)
        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        conn = self._ensure_connection()
        table_name = NamespaceCollectionNames.data_table_name(collection_id)
        ns_id = int(namespace_id)

        dimension = kwargs.pop("dimension", None)
        search_parm = self._build_search_parm(
            query,
            knn,
            rank,
            n_results,
            include=include,
            dimension=dimension,
            **kwargs,
        )
        search_parm = self._adapt_search_parm_for_ns(search_parm, ns_id, ltable_id)

        search_parm.pop("_source", None)

        if "knn" in search_parm:
            if query_hint is None:
                query_hint = QueryHint(vector_index=True)
            elif query_hint.vector_index is None:
                query_hint = QueryHint(
                    parallel=query_hint.parallel,
                    query_timeout=query_hint.query_timeout,
                    vector_index=True,
                )

        search_parm_json = json.dumps(search_parm, ensure_ascii=False)
        use_context_manager = self._use_context_manager_for_cursor()

        escaped_params = search_parm_json.replace("'", "''")

        hint_sql = _query_hint_to_sql(query_hint, table_name=table_name) or ""
        hybrid_sql = (
            f"SELECT {hint_sql + ' ' if hint_sql else ''}* FROM hybrid_search(TABLE `{table_name}`, '{escaped_params}')"
        )
        result_rows = self._execute_query_with_cursor(conn, hybrid_sql, [], use_context_manager)
        if not result_rows:
            return {
                "ids": [[]],
                "distances": [[]],
                "metadatas": [[]],
                "documents": [[]],
                "embeddings": [[]],
            }
        return self._transform_ns_hybrid_result(result_rows, include)

    def _transform_ns_hybrid_result(
        self, result_rows: list[dict[str, Any]], include: list[str] | None
    ) -> dict[str, Any]:
        """Transform raw hybrid-search rows into the public result shape."""
        if not result_rows:
            return {
                "ids": [[]],
                "distances": [[]],
                "metadatas": [[]],
                "documents": [[]],
                "embeddings": [[]],
            }

        ids = []
        distances = []
        metadatas = []
        documents = []
        embeddings = []

        for row in result_rows:
            dc_raw = row.get("data_content") or row.get("DATA_CONTENT")
            dc = {}
            if isinstance(dc_raw, str):
                with contextlib.suppress(json.JSONDecodeError):
                    dc = json.loads(dc_raw)
            elif isinstance(dc_raw, dict):
                dc = dc_raw

            row_id = dc.get("id")
            if row_id is None:
                for key in ("id", "_id", "ID"):
                    if key in row and row[key] is not None:
                        row_id = row[key]
                        break
            row_id = self._convert_id_from_bytes(row_id)
            ids.append(row_id)

            distances.append(self._hybrid_row_score(row))

            if include is None or "metadatas" in include or "metadata" in include:
                meta = dc.get("metadata")
                if meta is None:
                    meta = row.get("metadata") or row.get("METADATA")
                if isinstance(meta, str):
                    with contextlib.suppress(json.JSONDecodeError):
                        meta = json.loads(meta)
                metadatas.append(meta or {})
            else:
                metadatas.append(None)

            if include is None or "documents" in include or "document" in include:
                documents.append(row.get("document") or row.get("DOCUMENT"))
            else:
                documents.append(None)

            if include and ("embeddings" in include or "embedding" in include):
                emb = _parse_embedding_value(row.get("embedding") or row.get("EMBEDDING"))
                embeddings.append(emb)
            else:
                embeddings.append(None)

        result = {"ids": [ids], "distances": [distances]}
        if include is None or "documents" in include or "document" in include:
            result["documents"] = [documents]
        if include is None or "metadatas" in include or "metadata" in include:
            result["metadatas"] = [metadatas]
        if include and ("embeddings" in include or "embedding" in include):
            result["embeddings"] = [embeddings]
        return result

    def _namespace_prewarm(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        **kwargs,
    ) -> None:
        """Prewarm the namespace logical table to reduce first-query latency."""
        raise NotImplementedError("prewarm is not supported in this client mode")
