"""Synchronous standard collection and search operations."""

from __future__ import annotations

import contextlib
import json
import logging
import re
from typing import Any

from pymysql.converters import escape_string

from .configuration import DEFAULT_DISTANCE_METRIC
from .document_query_builder import (
    _pure_must_not_clauses as _pure_must_not_clauses_fragment,
)
from .document_query_builder import (
    build_document_hybrid_expression,
    doc_matches_where_document,
    document_expr_as_knn_filter,
    where_document_knn_prefilterable,
)
from .embedding_function import Documents as EmbeddingDocuments
from .embedding_function import EmbeddingFunction
from .meta_info import CollectionFieldNames
from .query_builder import (
    build_select_clause as _build_select_clause_fragment,
)
from .query_builder import (
    build_where_clause as _build_where_clause_fragment,
)
from .query_builder import (
    convert_id_from_bytes as _convert_id_from_bytes_fragment,
)
from .query_builder import (
    convert_id_to_sql as _convert_id_to_sql_fragment,
)
from .query_builder import (
    embed_texts as _embed_texts_fragment,
)
from .query_builder import (
    embedding_to_hexstring as _embedding_to_hexstring,
)
from .query_builder import (
    normalize_collection_batch as _normalize_collection_batch_fragment,
)
from .query_builder import (
    normalize_include_fields as _normalize_include_fields_fragment,
)
from .query_builder import (
    normalize_query_embeddings as _normalize_query_embeddings_fragment,
)
from .query_builder import (
    parse_embedding_value as _parse_embedding_value_fragment,
)
from .query_builder import (
    parse_row_value as _parse_row_value_fragment,
)
from .query_builder import (
    process_get_row as _process_get_row_fragment,
)
from .query_builder import (
    process_query_row as _process_query_row_fragment,
)
from .query_types import QueryHint
from .schema import SparseVectorIndexConfig
from .sparse_embedding_function import SparseVector, _sparse_vector_to_sql
from .sql_utils import _query_hint_to_sql
from .types import K as FieldKey
from .validators import _validate_include, _validate_n_results, _validate_pagination

_QUOTED_JSON_EXTRACT_EXPRESSION_PATTERN = re.compile(
    r"`(?P<expression>\s*\(*\s*JSON_EXTRACT\s*\([^`]*\)\s*\)*\s*)`",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)


def _unquote_json_extract_expressions(query_sql: str) -> str:
    """Unquote JSON_EXTRACT expressions without touching adjacent identifiers."""
    return _QUOTED_JSON_EXTRACT_EXPRESSION_PATTERN.sub(r"\g<expression>", query_sql)


class CollectionOperationsMixin:
    """Provide standard collection DML, DQL, and search operations."""

    def _generate_sparse_embeddings(
        self,
        sparse_config: SparseVectorIndexConfig,
        documents: list[str] | None,
        metadatas: list[dict] | None,
        num_items: int,
    ) -> list[SparseVector | None]:
        """
        Generate sparse embeddings based on SparseVectorIndexConfig.

        Returns a list of SparseVector (or None) for each item.
        """
        sparse_ef = sparse_config.embedding_function
        if sparse_ef is None:
            raise ValueError("Sparse embedding function is not provided")

        source_type, metadata_key = sparse_config.resolve_source_key()

        # Gather source texts
        source_texts = []
        for i in range(num_items):
            if source_type == "document":
                text = documents[i] if documents and i < len(documents) else None
                if text is None:
                    raise ValueError(
                        f"Sparse vector index is configured to generate from document field, "
                        f"but document at index {i} is None."
                    )
                if not isinstance(text, str):
                    raise TypeError(
                        f"Sparse vector index source_key refers to document field, "
                        f"but value at index {i} is not a string: {type(text).__name__}"
                    )
                source_texts.append(text)
            elif source_type == "metadata":
                meta = metadatas[i] if metadatas and i < len(metadatas) else None
                if meta is None:
                    raise ValueError(
                        f"Sparse vector index is configured to generate from metadata['{metadata_key}'], "
                        f"but metadata at index {i} is None."
                    )
                text = meta.get(metadata_key)
                if text is None:
                    raise ValueError(
                        f"Sparse vector index is configured to generate from metadata['{metadata_key}'], "
                        f"but metadata['{metadata_key}'] at index {i} is None."
                    )
                if not isinstance(text, str):
                    raise TypeError(
                        f"Sparse vector index source_key refers to metadata['{metadata_key}'], "
                        f"but value at index {i} is not a string: {type(text).__name__}"
                    )
                source_texts.append(text)
            else:
                raise ValueError(f"Invalid source type: {source_type}")

        # Generate sparse embeddings
        logger.debug(f"Generating sparse embeddings for {len(source_texts)} items")
        try:
            sparse_vectors = sparse_ef(source_texts)
        except Exception as e:
            raise ValueError(f"Failed to generate sparse embeddings: {e}") from e
        else:
            if len(sparse_vectors) != num_items:
                raise ValueError(
                    f"Sparse embedding function returned {len(sparse_vectors)} vectors, expected {num_items}."
                )
            logger.debug(f"✅ Successfully generated {len(sparse_vectors)} sparse embeddings")
            return sparse_vectors

    def _collection_add(
        self,
        collection_id: str | None,
        collection_name: str,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None = None,
        **kwargs,
    ) -> None:
        """
        [Internal] Add data to collection - Common SQL-based implementation

        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs
            embeddings: Single embedding or list of embeddings (optional)
            metadatas: Single metadata dict or list of metadata dicts (optional)
            documents: Single document or list of documents (optional)
            embedding_function: EmbeddingFunction instance to convert documents to embeddings.
                               Required if documents provided but embeddings not provided.
                               Must implement __call__ method that accepts Documents
                               and returns Embeddings (List[List[float]]).
            **kwargs: Additional parameters
        """
        logger.debug(f"Adding data to collection '{collection_name}'")

        explicit_embeddings = embeddings is not None
        ids, embeddings, metadatas, documents = _normalize_collection_batch_fragment(
            ids, embeddings, metadatas, documents
        )

        self._warn_explicit_embeddings_override_embedding_function(
            operation="collection.add",
            explicit_embeddings=explicit_embeddings,
            has_documents=bool(documents),
            embedding_function=embedding_function,
        )

        # Handle vector generation logic:
        # 1. If embeddings are provided, use them directly without embedding
        # 2. If embeddings are not provided but documents are provided:
        #    - If embedding_function is provided, use it to generate embeddings from documents
        #    - If embedding_function is not provided, raise an error
        # 3. If neither embeddings nor documents are provided, raise an error
        # An omitted embedding function is normalized to None when the collection has an
        # explicit dense dimension; documents without explicit embeddings still require an EF.

        if embeddings:
            # embeddings provided, use them directly without embedding
            pass
        elif documents:
            # embeddings not provided but documents are provided, check for embedding_function
            if embedding_function is not None:
                logger.debug(f"Generating embeddings for {len(documents)} documents using embedding function")
                try:
                    embeddings = embedding_function(documents)
                except Exception as e:
                    logger.exception("Failed to generate embeddings")
                    raise ValueError(f"Failed to generate embeddings from documents: {e}") from e
            else:
                raise ValueError(
                    "Documents provided but no embeddings and no embedding function. "
                    "Either:\n"
                    "  1. Provide embeddings directly when calling add(), or\n"
                    "  2. Provide embedding_function to auto-generate embeddings from documents."
                )
        else:
            # Neither embeddings nor documents provided, raise an error
            raise ValueError(
                "Neither embeddings nor documents provided. "
                "Please provide either:\n"
                "  1. embeddings directly, or\n"
                "  2. documents with embedding_function to generate embeddings."
            )

        num_items = len(ids)

        # Get table name
        table_name = self._get_collection_table_name(collection_id, collection_name)

        # Handle sparse embeddings generation
        sparse_config = kwargs.get("sparse_vector_index_config")
        sparse_embeddings = None
        has_sparse = sparse_config is not None
        if has_sparse:
            sparse_embeddings = self._generate_sparse_embeddings(sparse_config, documents, metadatas, num_items)

        # Build INSERT SQL
        values_list = []
        for i in range(num_items):
            # Process ID - support any string format
            id_val = ids[i] if ids else None
            if id_val:
                if not isinstance(id_val, str):
                    id_val = str(id_val)
                id_sql = self._convert_id_to_sql(id_val)
            else:
                raise ValueError("ids must be provided for add operation")

            # Process document
            doc_val = documents[i] if documents else None
            if doc_val is not None:
                # Use pymysql's escape_string for safe escaping
                doc_val_escaped = escape_string(doc_val)
                doc_sql = f"'{doc_val_escaped}'"
            else:
                doc_sql = "NULL"

            # Process metadata
            meta_val = metadatas[i] if metadatas else None
            if meta_val is not None:
                # Convert to JSON string and escape using pymysql's escape_string
                meta_json = json.dumps(meta_val, ensure_ascii=False)
                meta_json_escaped = escape_string(meta_json)
                meta_sql = f"'{meta_json_escaped}'"
            else:
                meta_sql = "NULL"

            # Process vector
            vec_val = embeddings[i] if embeddings else None
            vec_sql = "NULL" if vec_val is None else _embedding_to_hexstring(vec_val)

            # Process sparse vector
            if has_sparse:
                sparse_val = sparse_embeddings[i] if sparse_embeddings else None
                sparse_sql = "NULL" if sparse_val is None else _sparse_vector_to_sql(sparse_val)
                values_list.append(f"({id_sql}, {doc_sql}, {meta_sql}, {vec_sql}, {sparse_sql})")
            else:
                values_list.append(f"({id_sql}, {doc_sql}, {meta_sql}, {vec_sql})")

        # Build column list
        columns = f"{CollectionFieldNames.ID}, {CollectionFieldNames.DOCUMENT}, {CollectionFieldNames.METADATA}, {CollectionFieldNames.EMBEDDING}"
        if has_sparse:
            columns += f", {CollectionFieldNames.SPARSE_EMBEDDING}"

        # Build final SQL
        sql = f"""INSERT INTO `{table_name}` ({columns})
                 VALUES {",".join(values_list)}"""

        logger.debug(f"Executing SQL: {sql}")
        self._execute(sql)
        logger.debug(f"✅ Successfully added {num_items} item(s) to collection '{collection_name}'")

    def _collection_update(
        self,
        collection_id: str | None,
        collection_name: str,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None = None,
        **kwargs,
    ) -> None:
        """
        [Internal] Update data in collection - Common SQL-based implementation

        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs to update
            embeddings: New embeddings (optional)
            metadatas: New metadata (optional)
            documents: New documents (optional)
            embedding_function: EmbeddingFunction instance to convert documents to embeddings.
                               Required if documents provided but embeddings not provided.
                               Must implement __call__ method that accepts Documents
                               and returns Embeddings (List[List[float]]).
            **kwargs: Additional parameters
        """
        logger.debug(f"Updating data in collection '{collection_name}'")

        explicit_embeddings = embeddings is not None
        ids, embeddings, metadatas, documents = _normalize_collection_batch_fragment(
            ids, embeddings, metadatas, documents
        )

        self._warn_explicit_embeddings_override_embedding_function(
            operation="collection.update",
            explicit_embeddings=explicit_embeddings,
            has_documents=bool(documents),
            embedding_function=embedding_function,
        )

        # Handle vector generation logic:
        # 1. If embeddings are provided, use them directly without embedding
        # 2. If embeddings are not provided but documents are provided:
        #    - If embedding_function is provided, use it to generate embeddings from documents
        #    - If embedding_function is not provided, raise an error
        # 3. If neither embeddings nor documents are provided:
        #    - If metadatas are provided, allow update (metadata-only update)
        #    - If metadatas are not provided, raise an error

        if embeddings:
            # embeddings provided, use them directly without embedding
            pass
        elif documents:
            # embeddings not provided but documents are provided, check for embedding_function
            if embedding_function is not None:
                logger.debug(f"Generating embeddings for {len(documents)} documents using embedding function")
                try:
                    embeddings = embedding_function(documents)
                except Exception as e:
                    logger.exception("Failed to generate embeddings")
                    raise ValueError(f"Failed to generate embeddings from documents: {e}") from e
            else:
                raise ValueError(
                    "Documents provided but no embeddings and no embedding function. "
                    "Either:\n"
                    "  1. Provide embeddings directly when calling update(), or\n"
                    "  2. Provide embedding_function to auto-generate embeddings from documents."
                )
        elif not metadatas:
            # Neither embeddings nor documents nor metadatas provided, raise an error
            raise ValueError(
                "Neither embeddings nor documents nor metadatas provided. "
                "Please provide at least one of:\n"
                "  1. embeddings directly, or\n"
                "  2. documents with embedding_function to generate embeddings, or\n"
                "  3. metadatas to update metadata only."
            )

        # Get table name
        table_name = self._get_collection_table_name(collection_id, collection_name)

        # Handle sparse embeddings generation
        sparse_config = kwargs.get("sparse_vector_index_config")
        sparse_embeddings = None
        if sparse_config is not None:
            source_type, _ = sparse_config.resolve_source_key()
            should_generate = (source_type == "document" and documents) or (source_type == "metadata" and metadatas)
            if should_generate:
                sparse_embeddings = self._generate_sparse_embeddings(sparse_config, documents, metadatas, len(ids))

        # Update each item
        for i in range(len(ids)):
            # Process ID - support any string format
            id_val = ids[i]
            if not isinstance(id_val, str):
                id_val = str(id_val)
            id_sql = self._convert_id_to_sql(id_val)

            # Build SET clause
            set_clauses = []

            if documents:
                doc_val = documents[i]
                if doc_val is not None:
                    doc_val_escaped = escape_string(doc_val)
                    set_clauses.append(f"{CollectionFieldNames.DOCUMENT} = '{doc_val_escaped}'")

            if metadatas:
                meta_val = metadatas[i]
                if meta_val is not None:
                    meta_json = json.dumps(meta_val, ensure_ascii=False)
                    meta_json_escaped = escape_string(meta_json)
                    set_clauses.append(f"{CollectionFieldNames.METADATA} = '{meta_json_escaped}'")

            if embeddings:
                vec_val = embeddings[i]
                if vec_val is not None:
                    vec_str = "[" + ",".join(map(str, vec_val)) + "]"
                    set_clauses.append(f"{CollectionFieldNames.EMBEDDING} = '{vec_str}'")

            # Handle sparse embedding update
            if sparse_embeddings and sparse_embeddings[i] is not None:
                sparse_sql = _sparse_vector_to_sql(sparse_embeddings[i])
                set_clauses.append(f"{CollectionFieldNames.SPARSE_EMBEDDING} = {sparse_sql}")

            if not set_clauses:
                continue

            # Build UPDATE SQL
            sql = f"UPDATE `{table_name}` SET {', '.join(set_clauses)} WHERE {CollectionFieldNames.ID} = {id_sql}"

            logger.debug(f"Executing SQL: {sql}")
            self._execute(sql)

        logger.debug(f"✅ Successfully updated {len(ids)} item(s) in collection '{collection_name}'")

    def _collection_upsert(
        self,
        collection_id: str | None,
        collection_name: str,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None = None,
        **kwargs,
    ) -> None:
        """
        [Internal] Insert or update data in collection - Common SQL-based implementation

        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs
            embeddings: embeddings (optional)
            metadatas: Metadata (optional)
            documents: Documents (optional)
            embedding_function: EmbeddingFunction instance to convert documents to embeddings.
                               Required if documents provided but embeddings not provided.
                               Must implement __call__ method that accepts Documents
                               and returns Embeddings (List[List[float]]).
            **kwargs: Additional parameters
        """
        logger.debug(f"Upserting data in collection '{collection_name}'")

        ids, embeddings, metadatas, documents = _normalize_collection_batch_fragment(
            ids, embeddings, metadatas, documents
        )

        # Handle vector generation logic:
        # 1. If embeddings are provided, use them directly without embedding
        # 2. If embeddings are not provided but documents are provided:
        #    - If embedding_function is provided, use it to generate embeddings from documents
        #    - If embedding_function is not provided, raise an error
        # 3. If neither embeddings nor documents are provided:
        #    - If metadatas are provided, allow upsert (metadata-only upsert)
        #    - If metadatas are not provided, raise an error

        if embeddings:
            # embeddings provided, use them directly without embedding
            pass
        elif documents:
            # embeddings not provided but documents are provided, check for embedding_function
            if embedding_function is not None:
                logger.debug(f"Generating embeddings for {len(documents)} documents using embedding function")
                try:
                    embeddings = embedding_function(documents)
                except Exception as e:
                    logger.exception("Failed to generate embeddings")
                    raise ValueError(f"Failed to generate embeddings from documents: {e}") from e
            else:
                raise ValueError(
                    "Documents provided but no embeddings and no embedding function. "
                    "Either:\n"
                    "  1. Provide embeddings directly when calling upsert(), or\n"
                    "  2. Provide embedding_function to auto-generate embeddings from documents."
                )
        elif not metadatas:
            # Neither embeddings nor documents nor metadatas provided, raise an error
            raise ValueError(
                "Neither embeddings nor documents nor metadatas provided. "
                "Please provide at least one of:\n"
                "  1. embeddings directly, or\n"
                "  2. documents with embedding_function to generate embeddings, or\n"
                "  3. metadatas to update metadata only."
            )

        # Get table name
        table_name = self._get_collection_table_name(collection_id, collection_name)

        # Handle sparse embeddings generation
        sparse_config = kwargs.get("sparse_vector_index_config")
        sparse_embeddings = None
        if sparse_config is not None:
            source_type, _ = sparse_config.resolve_source_key()
            should_generate = (source_type == "document" and documents) or (source_type == "metadata" and metadatas)
            if should_generate:
                sparse_embeddings = self._generate_sparse_embeddings(sparse_config, documents, metadatas, len(ids))

        # Use an atomic INSERT ... ON DUPLICATE KEY UPDATE for each item. The
        # update assignments include only fields supplied by the caller, which
        # preserves the partial-update semantics without a read-before-write.
        for i in range(len(ids)):
            id_val = ids[i]
            if not isinstance(id_val, str):
                id_val = str(id_val)
            id_sql = self._convert_id_to_sql(id_val)
            doc_val = documents[i] if documents else None
            meta_val = metadatas[i] if metadatas else None
            vec_val = embeddings[i] if embeddings else None

            doc_sql = "NULL"
            if doc_val is not None:
                doc_sql = f"'{escape_string(doc_val)}'"

            meta_sql = "NULL"
            if meta_val is not None:
                meta_json = json.dumps(meta_val, ensure_ascii=False)
                meta_sql = f"'{escape_string(meta_json)}'"

            vec_sql = "NULL" if vec_val is None else _embedding_to_hexstring(vec_val)
            columns = [
                CollectionFieldNames.ID,
                CollectionFieldNames.DOCUMENT,
                CollectionFieldNames.METADATA,
                CollectionFieldNames.EMBEDDING,
            ]
            values = [id_sql, doc_sql, meta_sql, vec_sql]
            update_clauses = []

            if doc_val is not None:
                update_clauses.append(f"{CollectionFieldNames.DOCUMENT} = {doc_sql}")
            if meta_val is not None:
                update_clauses.append(f"{CollectionFieldNames.METADATA} = {meta_sql}")
            if vec_val is not None:
                update_clauses.append(f"{CollectionFieldNames.EMBEDDING} = {vec_sql}")

            if sparse_embeddings and sparse_embeddings[i] is not None:
                sparse_sql = _sparse_vector_to_sql(sparse_embeddings[i])
                columns.append(CollectionFieldNames.SPARSE_EMBEDDING)
                values.append(sparse_sql)
                update_clauses.append(f"{CollectionFieldNames.SPARSE_EMBEDDING} = {sparse_sql}")

            if not update_clauses:
                update_clauses.append(f"{CollectionFieldNames.ID} = {CollectionFieldNames.ID}")

            sql = (
                f"INSERT INTO `{table_name}` ({', '.join(columns)}) VALUES ({', '.join(values)}) "
                f"ON DUPLICATE KEY UPDATE {', '.join(update_clauses)}"
            )
            logger.debug("Executing SQL: %s", sql)
            self._execute(sql)

        logger.debug(f"✅ Successfully upserted {len(ids)} item(s) in collection '{collection_name}'")

    def _collection_delete(
        self,
        collection_id: str | None,
        collection_name: str,
        ids: str | list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        [Internal] Delete data from collection - Common SQL-based implementation

        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs to delete (optional)
            where: Filter condition on metadata (optional)
            where_document: Filter condition on documents (optional)
            **kwargs: Additional parameters
        """
        logger.debug(f"Deleting data from collection '{collection_name}'")

        # Validate that at least one filter is provided
        if not ids and not where and not where_document:
            raise ValueError("At least one of ids, where, or where_document must be provided")

        # Normalize ids to list
        id_list = None
        if ids is not None:
            id_list = [ids] if isinstance(ids, str) else ids

        # Get table name
        table_name = self._get_collection_table_name(collection_id, collection_name)

        # Build WHERE clause
        where_clause, params = self._build_where_clause(where, where_document, id_list)

        # Build DELETE SQL
        sql = f"DELETE FROM `{table_name}` {where_clause}"

        logger.debug(f"Executing SQL: {sql}")
        logger.debug(f"Parameters: {params}")

        # Execute DELETE using parameterized query
        conn = self._ensure_connection()
        use_context_manager = self._use_context_manager_for_cursor()
        self._execute_query_with_cursor(conn, sql, params, use_context_manager)

        logger.debug(f"✅ Successfully deleted data from collection '{collection_name}'")

    # -------------------- DQL Operations --------------------
    # Note: _collection_query() and _collection_get() are implemented below with common SQL-based logic

    def _normalize_query_embeddings(
        self, query_embeddings: list[float] | list[list[float]] | None
    ) -> list[list[float]]:
        """Normalize query embeddings to a list of vectors."""
        return _normalize_query_embeddings_fragment(query_embeddings)

    def _normalize_include_fields(self, include: list[str] | None) -> dict[str, bool]:
        """Normalize the optional result-field list."""
        return _normalize_include_fields_fragment(include)

    def _embed_texts(
        self,
        texts: str | list[str],
        embedding_function: EmbeddingFunction[EmbeddingDocuments] | None = None,
        **kwargs,
    ) -> list[list[float]]:
        """Generate vectors through the configured embedding function."""
        return _embed_texts_fragment(texts, embedding_function, **kwargs)

    def _build_select_clause(self, include_fields: dict[str, bool]) -> str:
        """Build the collection SELECT field list."""
        return _build_select_clause_fragment(include_fields)

    def _build_where_clause(
        self,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        id_list: list[str] | None = None,
    ) -> tuple[str, list[Any]]:
        """Build a parameterized WHERE clause for collection queries."""
        return _build_where_clause_fragment(where, where_document, id_list)

    def _parse_row_value(self, value: Any) -> Any:
        """Decode JSON text returned by a database row when possible."""
        return _parse_row_value_fragment(value)

    def _convert_id_to_sql(self, id_val: str) -> str:
        """Build an escaped CAST expression for a binary record ID."""
        return _convert_id_to_sql_fragment(id_val)

    def _convert_id_from_bytes(self, record_id: Any) -> str:
        """Convert a database binary ID to its string representation."""
        return _convert_id_from_bytes_fragment(record_id)

    def _process_query_row(self, row: dict[str, Any], include_fields: dict[str, bool]) -> dict[str, Any]:
        """Normalize one vector-query result row."""
        return _process_query_row_fragment(row, include_fields)

    def _process_get_row(self, row: dict[str, Any], include_fields: dict[str, bool]) -> dict[str, Any]:
        """Normalize one standard collection get-result row."""
        return _process_get_row_fragment(row, include_fields)

    # -------------------- DQL Operations (Common Implementation) --------------------

    def _collection_query(
        self,
        collection_id: str | None,
        collection_name: str,
        query_embeddings: list[float] | list[list[float]] | None = None,
        query_texts: str | list[str] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
        query_key: FieldKey | None = None,
        query_hint: QueryHint | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        [Internal] Query collection by vector similarity - Common SQL-based implementation

        Args:
            collection_id: Collection ID
            collection_name: Collection name
            query_embeddings: Query vector(s) (preferred)
            query_texts: Query text(s) - will be embedded if provided (preferred)
            n_results: Number of results (default: 10)
            where: Metadata filter
            where_document: Document filter
            include: Fields to include
            **kwargs: Additional parameters, including:
                embedding_function: EmbeddingFunction instance to convert query_texts to embeddings.
                                   Required if query_texts is provided and collection doesn't have
                                   an embedding_function set. Must implement __call__ method that
                                   accepts Documents and returns Embeddings (List[List[float]]).
                distance: Distance metric to use for similarity calculation (e.g., 'l2', 'cosine', 'inner_product').
                         Defaults to 'l2' if not provided.

        Returns:
            Dict with keys:
            - ids: List[List[str]] - List of ID lists, one list per query
            - documents: Optional[List[List[str]]] - List of document lists, one list per query
            - metadatas: Optional[List[List[Dict]]] - List of metadata lists, one list per query
            - embeddings: Optional[List[List[List[float]]]] - List of embedding lists, one list per query
            - distances: Optional[List[List[float]]] - List of distance lists, one list per query
        """
        _validate_n_results(n_results)
        _validate_include(include)
        logger.debug(f"Querying collection '{collection_name}' with n_results={n_results}")
        conn = self._ensure_connection()

        # Convert collection name to table name
        table_name = self._get_collection_table_name(collection_id, collection_name)

        # Check if this is a sparse vector query
        sparse_config = kwargs.get("sparse_vector_index_config")
        is_sparse_query = query_key is not None and (
            query_key is FieldKey.SPARSE_EMBEDDING
            or query_key == FieldKey.SPARSE_EMBEDDING.name
            or (hasattr(query_key, "name") and query_key.name == "#sparse_embedding")
        )

        if is_sparse_query:
            return self._collection_query_sparse(
                conn=conn,
                table_name=table_name,
                query_embeddings=query_embeddings,
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include,
                sparse_config=sparse_config,
                collection_name=collection_name,
                query_hint=query_hint,
                **kwargs,
            )

        # ===== Dense vector query path =====
        # Handle vector generation logic:
        # 1. If query_embeddings are provided, use them directly without embedding
        # 2. If query_embeddings are not provided but query_texts are provided:
        #    - If embedding_function is provided, use it to generate embeddings from query_texts
        #    - If embedding_function is not provided, raise an error
        # 3. If neither query_embeddings nor query_texts are provided, raise an error

        embedding_function = kwargs.get("embedding_function")

        if query_embeddings is not None:
            # Query embeddings provided, use them directly without embedding
            pass
        elif query_texts is not None:
            # Query embeddings not provided but query_texts are provided, check for embedding_function
            if embedding_function is not None:
                logger.debug("Embedding query texts...")
                query_embeddings = self._embed_texts(query_texts, embedding_function=embedding_function)
            else:
                raise ValueError(
                    "query_texts provided but no query_embeddings and no embedding_function. "
                    "Either:\n"
                    "  1. Provide query_embeddings directly, or\n"
                    "  2. Provide embedding_function to auto-generate embeddings from query_texts."
                )
        else:
            # Neither query_embeddings nor query_texts provided, raise an error
            raise ValueError(
                "Neither query_embeddings nor query_texts provided. "
                "Please provide either:\n"
                "  1. query_embeddings directly, or\n"
                "  2. query_texts with embedding_function to generate embeddings."
            )

        # Normalize query embeddings to list of lists
        query_embeddings = self._normalize_query_embeddings(query_embeddings)
        if not query_embeddings:
            raise ValueError("query_embeddings must not be empty")

        # Normalize include fields
        include_fields = self._normalize_include_fields(include)

        # Build SELECT clause
        select_clause = self._build_select_clause(include_fields)

        # Build WHERE clause from filters
        where_clause, params = self._build_where_clause(where, where_document)

        # Get distance metric from kwargs, default to DEFAULT_DISTANCE_METRIC if not provided
        distance = kwargs.get("distance", DEFAULT_DISTANCE_METRIC)

        # Map distance metric to SQL function name
        distance_function_map = {
            "l2": "l2_distance",
            "cosine": "cosine_distance",
            "inner_product": "inner_product",
        }

        # Get the distance function name, default to 'l2_distance' if distance is not recognized
        distance_func = distance_function_map.get(distance, "l2_distance")

        if distance not in distance_function_map:
            logger.warning(f"Unknown distance metric '{distance}', defaulting to 'l2_distance'")

        use_context_manager = self._use_context_manager_for_cursor()

        # Collect results for each query vector separately
        all_ids = []
        all_documents = []
        all_metadatas = []
        all_embeddings = []
        all_distances = []

        for query_vector in query_embeddings:
            # Convert vector to string format for SQL
            vector_str = _embedding_to_hexstring(query_vector)

            # Build query hint
            hint_sql = _query_hint_to_sql(query_hint, table_name=table_name)

            # Build SQL query with vector distance calculation
            # Reference: SELECT id, vec FROM t2 ORDER BY l2_distance(vec, '[0.1, 0.2, 0.3]') APPROXIMATE LIMIT 5;
            # Need to include distance in SELECT for result processing
            # Use the appropriate distance function based on the index configuration
            sql = f"""
                SELECT {hint_sql} {select_clause},
                       {distance_func}(embedding, {vector_str}) AS distance
                FROM `{table_name}`
                {where_clause}
                ORDER BY {distance_func}(embedding, {vector_str})
                APPROXIMATE
                LIMIT %s
            """

            # Execute query
            query_params = [*params, n_results]
            logger.debug(f"Executing SQL: {sql}")
            logger.debug(f"Parameters: {query_params}")

            rows = self._execute_query_with_cursor(conn, sql, query_params, use_context_manager)

            # Collect results for this query vector
            query_ids = []
            query_documents = []
            query_metadatas = []
            query_embeddings = []
            query_distances = []

            for row in rows:
                result_item = self._process_query_row(row, include_fields)
                query_ids.append(result_item.get("_id"))

                if "documents" in include_fields or include is None:
                    query_documents.append(result_item.get("document"))

                if "metadatas" in include_fields or include is None:
                    query_metadatas.append(result_item.get("metadata") or {})

                if "embeddings" in include_fields:
                    query_embeddings.append(result_item.get("embedding"))

                query_distances.append(result_item.get("distance"))

            all_ids.append(query_ids)
            if "documents" in include_fields or include is None:
                all_documents.append(query_documents)
            if "metadatas" in include_fields or include is None:
                all_metadatas.append(query_metadatas)
            if "embeddings" in include_fields:
                all_embeddings.append(query_embeddings)
            all_distances.append(query_distances)

        # Build result dictionary in chromadb format
        result = {"ids": all_ids, "distances": all_distances}

        if "documents" in include_fields or include is None:
            result["documents"] = all_documents

        if "metadatas" in include_fields or include is None:
            result["metadatas"] = all_metadatas

        if "embeddings" in include_fields:
            result["embeddings"] = all_embeddings

        logger.debug(
            f"✅ Query completed for '{collection_name}' with {len(query_embeddings)} vectors, returning {len(all_ids)} result lists"
        )
        return result

    def _collection_query_sparse(
        self,
        conn,
        table_name: str,
        query_embeddings: list[float] | list[list[float]] | None = None,
        query_texts: str | list[str] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
        sparse_config=None,
        collection_name: str = "",
        query_hint=None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        [Internal] Query collection by sparse vector similarity.

        Supports:
        1. Text-based sparse vector query via query_texts + sparse embedding function

        Args:
            conn: Database connection
            table_name: Table name
            query_embeddings: Not supported for sparse query path. Use query_texts instead.
            query_texts: Query text(s) to be converted to sparse vectors
            n_results: Number of results
            where: Metadata filter
            where_document: Document filter
            include: Fields to include
            sparse_config: SparseVectorIndexConfig instance
            collection_name: Collection name (for logging)
        """
        logger.debug(f"Sparse vector query on collection '{collection_name}'")

        # Resolve sparse query vectors
        sparse_query_vectors: list[SparseVector] = []

        if query_embeddings is not None:
            raise ValueError(
                "For sparse vector queries, query_embeddings is not supported. "
                "Please provide query_texts and use the configured sparse embedding function."
            )
        elif query_texts is not None:
            # Generate sparse vectors from query texts using sparse embedding function
            if sparse_config is None or sparse_config.embedding_function is None:
                raise ValueError(
                    "query_texts provided for sparse vector query but no sparse embedding function is configured."
                )
            sparse_ef = sparse_config.embedding_function
            # Normalize query_texts to list
            if isinstance(query_texts, str):
                query_texts = [query_texts]
            logger.debug(f"Generating sparse embeddings for {len(query_texts)} query texts...")
            sparse_vectors = sparse_ef(query_texts)
            sparse_query_vectors = sparse_vectors
        else:
            raise ValueError(
                "Neither query_embeddings nor query_texts provided for sparse vector query. "
                "Please provide query_texts with a configured sparse embedding function."
            )

        if not sparse_query_vectors:
            raise ValueError("No sparse query vectors resolved.")

        hint_sql = _query_hint_to_sql(query_hint, table_name=table_name)

        # Normalize include fields
        include_fields = self._normalize_include_fields(include)

        # Build SELECT clause
        select_clause = self._build_select_clause(include_fields)

        # Build WHERE clause from filters
        where_clause, params = self._build_where_clause(where, where_document)

        # Sparse vector queries always use inner_product distance
        distance_func = "inner_product"

        use_context_manager = self._use_context_manager_for_cursor()

        # Collect results for each sparse query vector separately
        all_ids = []
        all_documents = []
        all_metadatas = []
        all_embeddings = []
        all_distances = []

        for sv in sparse_query_vectors:
            # Convert sparse vector to SQL string format
            sv_sql = _sparse_vector_to_sql(sv)

            # Build SQL query with sparse vector distance calculation
            sql = f"""
                SELECT {hint_sql} {select_clause},
                       {distance_func}(sparse_embedding, {sv_sql}) AS distance
                FROM `{table_name}`
                {where_clause}
                ORDER BY {distance_func}(sparse_embedding, {sv_sql})
                APPROXIMATE
                LIMIT %s
            """.strip()

            # Execute query
            query_params = [*params, n_results]
            logger.debug(f"Executing sparse SQL: {sql}")
            logger.debug(f"Parameters: {query_params}")

            rows = self._execute_query_with_cursor(conn, sql, query_params, use_context_manager)

            # Collect results for this query vector
            query_ids = []
            query_documents = []
            query_metadatas = []
            query_embeddings_list = []
            query_distances = []

            for row in rows:
                result_item = self._process_query_row(row, include_fields)
                query_ids.append(result_item.get("_id"))

                if "documents" in include_fields or include is None:
                    query_documents.append(result_item.get("document"))

                if "metadatas" in include_fields or include is None:
                    query_metadatas.append(result_item.get("metadata") or {})

                if "embeddings" in include_fields:
                    query_embeddings_list.append(result_item.get("embedding"))

                query_distances.append(result_item.get("distance"))

            all_ids.append(query_ids)
            if "documents" in include_fields or include is None:
                all_documents.append(query_documents)
            if "metadatas" in include_fields or include is None:
                all_metadatas.append(query_metadatas)
            if "embeddings" in include_fields:
                all_embeddings.append(query_embeddings_list)
            all_distances.append(query_distances)

        # Build result dictionary in chromadb format
        result = {"ids": all_ids, "distances": all_distances}

        if "documents" in include_fields or include is None:
            result["documents"] = all_documents

        if "metadatas" in include_fields or include is None:
            result["metadatas"] = all_metadatas

        if "embeddings" in include_fields:
            result["embeddings"] = all_embeddings

        logger.debug(
            f"Sparse query completed for '{collection_name}' with {len(sparse_query_vectors)} vectors, "
            f"returning {len(all_ids)} result lists"
        )
        return result

    def _collection_get(
        self,
        collection_id: str | None,
        collection_name: str,
        ids: str | list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: list[str] | None = None,
        query_hint: QueryHint | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        [Internal] Get data from collection by IDs or filters - Common SQL-based implementation

        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs (optional)
            where: Filter condition on metadata (optional)
            where_document: Filter condition on documents (optional)
            limit: Maximum number of results (optional)
            offset: Number of results to skip (optional)
            include: Fields to include in results (optional)
            query_hint: Query optimization hints for database execution (optional)
            **kwargs: Additional parameters

        Returns:
            Dict with keys:
            - ids: List[str] - List of IDs
            - documents: Optional[List[str]] - List of documents
            - metadatas: Optional[List[Dict]] - List of metadata dictionaries
            - embeddings: Optional[List[List[float]]] - List of embeddings
        """
        _validate_pagination(limit, offset)
        _validate_include(include)
        logger.debug(f"Getting data from collection '{collection_name}'")
        conn = self._ensure_connection()

        # Convert collection name to table name
        table_name = self._get_collection_table_name(collection_id, collection_name)

        # Set defaults
        if limit is None:
            limit = 100
        if offset is None:
            offset = 0

        # Normalize ids to list
        id_list = None
        if ids is not None:
            id_list = [ids] if isinstance(ids, str) else ids

        # Note: get() now returns dict format (not QueryResult)
        # Normalize include fields (default includes documents and metadatas)
        include_fields = self._normalize_include_fields(include)

        # Build SELECT clause - always include _id
        select_clause = self._build_select_clause(include_fields)

        use_context_manager = self._use_context_manager_for_cursor()

        # Build WHERE clause from filters
        where_clause, params = self._build_where_clause(where, where_document, id_list)

        # Build query hint
        hint_sql = _query_hint_to_sql(query_hint, table_name=table_name)

        # Build SQL query
        sql = f"""
            SELECT {hint_sql} {select_clause}
            FROM `{table_name}`
            {where_clause}
            LIMIT %s OFFSET %s
        """

        # Execute query
        query_params = [*params, limit, offset]
        logger.debug(f"Executing SQL: {sql}")
        logger.debug(f"Parameters: {query_params}")

        rows = self._execute_query_with_cursor(conn, sql, query_params, use_context_manager)

        # Build result dictionary in chromadb format
        result_ids = []
        result_documents = []
        result_metadatas = []
        result_embeddings = []

        for row in rows:
            processed_row = self._process_get_row(row, include_fields)
            result_ids.append(processed_row["id"])

            if "documents" in include_fields or include is None:
                result_documents.append(processed_row["document"])

            if "metadatas" in include_fields or include is None:
                result_metadatas.append(processed_row["metadata"] or {})

            if "embeddings" in include_fields:
                result_embeddings.append(processed_row["embedding"])

        # Build result dictionary
        result = {"ids": result_ids}

        if "documents" in include_fields or include is None:
            result["documents"] = result_documents

        if "metadatas" in include_fields or include is None:
            result["metadatas"] = result_metadatas

        if "embeddings" in include_fields:
            result["embeddings"] = result_embeddings

        logger.debug(f"✅ Get completed for '{collection_name}', found {len(result_ids)} results")
        return result

    def _collection_hybrid_search(
        self,
        collection_id: str | None,
        collection_name: str,
        query: dict[str, Any] | None = None,
        knn: dict[str, Any] | None = None,
        rank: dict[str, Any] | None = None,
        n_results: int = 10,
        include: list[str] | None = None,
        query_hint: QueryHint | None = None,
        dimension: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        [Internal] Hybrid search combining full-text search and vector similarity search - Common SQL-based implementation

        Supports:
        1. Scalar query (metadata filtering only)
        2. Full-text search (with optional metadata filtering)
        3. Vector search (with optional metadata filtering)
        4. Scalar + vector search (with optional metadata filtering)

        Args:
            collection_id: Collection ID
            collection_name: Collection name
            query: Full-text search configuration dict with:
                - where_document: Document filter conditions (e.g., {"$contains": "text"})
                - where: Metadata filter conditions (e.g., {"page": {"$gte": 5}})
            knn: Vector search configuration dict with:
                - query_texts: Query text(s) to be embedded (optional if query_embeddings provided)
                - query_embeddings: Query vector(s) (optional if query_texts provided)
                - where: Metadata filter conditions (optional)
                - n_results: Number of results for vector search (optional)
            rank: Ranking configuration dict (e.g., {"rrf": {"rank_window_size": 60, "rank_constant": 60}})
            n_results: Final number of results to return after ranking (default: 10)
            include: Fields to include in results (optional)
            dimension: Collection vector dimension for validating query_embeddings (optional)
            **kwargs: Additional parameters, including:
                embedding_function: EmbeddingFunction instance to convert query_texts in knn to embeddings.
                                   Required if knn.query_texts is provided and collection doesn't have
                                   an embedding_function set. Must implement __call__ method that
                                   accepts Documents and returns Embeddings (List[List[float]]).

        Returns:
            Dict with keys (query-compatible format):
            - ids: List[List[str]] - List of ID lists (one list for hybrid search result)
            - documents: Optional[List[List[str]]] - List of document lists (if included)
            - metadatas: Optional[List[List[Dict]]] - List of metadata lists (if included)
            - embeddings: Optional[List[List[List[float]]]] - List of embedding lists (if included)
            - distances: Optional[List[List[float]]] - List of distance lists
        """
        _validate_n_results(n_results)
        _validate_include(include)
        logger.debug(f"Hybrid search in collection '{collection_name}' with n_results={n_results}")
        conn = self._ensure_connection()

        # Build table name
        table_name = self._get_collection_table_name(collection_id, collection_name)

        # Build search_parm JSON
        search_parm = self._build_search_parm(
            query,
            knn,
            rank,
            n_results,
            include=include,
            dimension=dimension,
            **kwargs,
        )

        # Convert search_parm to JSON string
        search_parm_json = json.dumps(search_parm, ensure_ascii=False)

        # Use variable binding to avoid datatype issues
        use_context_manager = self._use_context_manager_for_cursor()

        # Set the search_parm variable first (use safe escaping)
        escaped_params = escape_string(search_parm_json)
        set_sql = f"SET @search_parm = '{escaped_params}'"
        logger.debug(f"Setting search_parm: {set_sql}")
        logger.debug(f"Search parm JSON: {search_parm_json}")

        # Execute SET statement
        self._execute_query_with_cursor(conn, set_sql, [], use_context_manager)

        # Get SQL query from DBMS_HYBRID_SEARCH.GET_SQL
        get_sql_table_name = escape_string(table_name)
        get_sql_query = (
            f"SELECT DBMS_HYBRID_SEARCH.GET_SQL('{get_sql_table_name}', @search_parm) as query_sql FROM dual"
        )
        logger.debug(f"Getting SQL query: {get_sql_query}")

        rows = self._execute_query_with_cursor(conn, get_sql_query, [], use_context_manager)

        if not rows or not rows[0].get("query_sql"):
            logger.warning("No SQL query returned from GET_SQL")
            return {
                "ids": [[]],
                "distances": [[]],
                "metadatas": [[]],
                "documents": [[]],
                "embeddings": [[]],
            }

        # Get the SQL query string
        query_sql = rows[0]["query_sql"]
        if isinstance(query_sql, str):
            # Remove any surrounding quotes if present
            query_sql = query_sql.strip().strip("'\"")

        # OB's GET_SQL can wrap JSON_EXTRACT expressions in backticks, which
        # turns them into literal column names. Unquote only those complete
        # expressions; ordinary identifiers around them must remain quoted.
        query_sql = _unquote_json_extract_expressions(query_sql)

        # Add query hint to the generated SQL
        hint_sql = _query_hint_to_sql(query_hint, table_name=table_name)
        if hint_sql and query_sql[:6].upper() == "SELECT":
            # Insert hint after SELECT keyword
            query_sql = f"SELECT {hint_sql} {query_sql[6:]}"

        logger.debug(f"Executing query SQL: {query_sql}")

        # Execute the returned SQL query
        result_rows = self._execute_query_with_cursor(conn, query_sql, [], use_context_manager)

        # Transform SQL query results to standard format
        return self._transform_sql_result(result_rows, include)

    def _build_search_parm(
        self,
        query: dict[str, Any] | list[dict[str, Any]] | None,
        knn: dict[str, Any] | list[dict[str, Any]] | None,
        rank: dict[str, Any] | None,
        n_results: int,
        include: list[str] | None = None,
        dimension: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Build search_parm JSON from query, knn, and rank parameters

        Args:
            query: Full-text search configuration dict or list of dicts
            knn: Vector search configuration dict or list of dicts
            rank: Ranking configuration dict
            n_results: Final number of results to return
            include: Fields requested by the SDK caller. Used to infer the minimal OceanBase GET_SQL
                `_source` allowlist to avoid returning large unused columns (e.g. `embedding`).
            dimension: Collection dimension for validating query_embeddings (optional)
            **kwargs: Additional parameters, including:
                embedding_function: EmbeddingFunction instance to convert query_texts in knn to embeddings.
                                   Required if knn.query_texts is provided. Must implement __call__
                                   method that accepts Documents and returns Embeddings (List[List[float]]).

        Returns:
            search_parm dictionary
        """
        search_parm = {}

        # Build query part (full-text search or scalar query)
        query_expr_list: list[dict[str, Any]] = []
        if query:
            query_items = query if isinstance(query, list) else [query]
            for query_item in query_items:
                query_expr = self._build_query_expression(query_item)
                if query_expr:
                    query_expr_list.append(query_expr)
        if query_expr_list:
            search_parm["query"] = query_expr_list if len(query_expr_list) > 1 else query_expr_list[0]

        # Build knn part (vector search)
        knn_expr_list: list[dict[str, Any]] = []
        if knn:
            knn_items = knn if isinstance(knn, list) else [knn]
            for knn_item in knn_items:
                knn_expr = self._build_knn_expression(knn_item, dimension=dimension, **kwargs)
                if not knn_expr:
                    continue
                if isinstance(knn_expr, list):
                    knn_expr_list.extend(knn_expr)
                else:
                    knn_expr_list.append(knn_expr)
        if knn_expr_list:
            search_parm["knn"] = knn_expr_list if len(knn_expr_list) > 1 else knn_expr_list[0]

        if n_results is not None:
            search_parm["size"] = n_results

        # Build rank part
        if rank:
            search_parm["rank"] = rank

        # Always infer a minimal `_source` allowlist from include to reduce response payload.
        search_parm["_source"] = self._build_source_fields(include)

        return search_parm

    @staticmethod
    def _positive_clause_for_must_not(must_not_clauses: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a type-independent positive filter for a must_not-only bool."""
        # The bool query requires one positive leaf, but using the negated
        # metadata field here is incorrect for strings, booleans, and missing
        # fields. Every collection row has an _id, so it is the universal leaf.
        return {"exists": {"field": CollectionFieldNames.ID}}

    @staticmethod
    def _hoist_must_not_from_filters(
        filter_conditions: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split a filter clause list into positive filters and hoisted must_not leaves."""
        positive: list[dict[str, Any]] = []
        negative: list[dict[str, Any]] = []
        for cond in filter_conditions:
            clauses = _pure_must_not_clauses_fragment(cond)
            if clauses is not None:
                negative.extend(clauses)
            else:
                positive.append(cond)
        return positive, negative

    def _build_query_expression(self, query: dict[str, Any]) -> dict[str, Any] | None:
        """
        Build query expression from query dict

        Supports:
        - Scalar query (metadata filtering only): query.range or query.term
        - Full-text search: query.query_string
        - Full-text search with metadata filtering: query.bool with must and filter
        """
        where_document = query.get("where_document")
        where = query.get("where")
        boost = query.get("boost")

        # Case 1: Scalar query (metadata filtering only, no full-text search)
        if not where_document and where:
            filter_conditions = self._build_metadata_filter_for_search_parm(where)
            if filter_conditions:
                # Wrap scalar conditions in a (non-scoring) `filter` clause: a
                # top-level bool is scoring by default and the kernel rejects scalar
                # term/range queries inside must/should of a scoring bool
                # (`scalar ... query in must/should clause not supported`). Negation
                # conditions ($not/$ne/$nin) come back as pure `{"bool": {"must_not"}}`
                # nodes; a bool with only must_not is rejected with `bool query ...
                # should have at least one positive clause`, so hoist their must_not
                # clauses onto the outer bool (which gains a positive `filter` once the
                # namespace filter is injected) instead of nesting them standalone.
                positive, negative = self._hoist_must_not_from_filters(filter_conditions)
                bool_q: dict[str, Any] = {}
                if positive:
                    bool_q["filter"] = positive
                if negative:
                    bool_q["must_not"] = negative
                return {"bool": bool_q}

        # Case 2: Full-text search (with or without metadata filtering)
        if where_document:
            # Build document query using query_string
            doc_query = self._build_document_query(where_document, boost=boost)
            if doc_query:
                filter_conditions = self._build_metadata_filter_for_search_parm(where)
                pos_filters, meta_must_not = self._hoist_must_not_from_filters(filter_conditions)
                doc_must_not = _pure_must_not_clauses_fragment(doc_query)
                must_not_all = list(meta_must_not)
                if doc_must_not is not None:
                    must_not_all.extend(doc_must_not)

                if not filter_conditions and doc_must_not is None:
                    return doc_query

                bool_q: dict[str, Any] = {}
                if doc_must_not is None:
                    bool_q["must"] = [doc_query]
                if pos_filters:
                    bool_q["filter"] = pos_filters
                if must_not_all:
                    bool_q["must_not"] = must_not_all
                return {"bool": bool_q}

        return None

    def _build_document_query(
        self, where_document: dict[str, Any], boost: float | None = None
    ) -> dict[str, Any] | None:
        """
        Build document query from where_document condition using query_string

        Args:
            where_document: Document filter conditions
            boost: Optional weight for this document query

        Returns:
            query_string query dict
        """
        if not where_document:
            return None

        def _with_boost(expr: dict[str, Any] | None) -> dict[str, Any] | None:
            """Apply field boosting to a document query expression."""
            if boost is None or not expr:
                return expr

            def _apply_boost(target: Any) -> None:
                """Apply a boost factor to a single field expression."""
                if not isinstance(target, dict):
                    return
                if "query_string" in target and isinstance(target["query_string"], dict):
                    target["query_string"]["boost"] = boost
                    return
                bool_clause = target.get("bool")
                if isinstance(bool_clause, dict):
                    for key in ("must", "should", "must_not", "filter"):
                        clause = bool_clause.get(key)
                        if isinstance(clause, list):
                            for item in clause:
                                _apply_boost(item)
                        elif isinstance(clause, dict):
                            _apply_boost(clause)

            _apply_boost(expr)
            return expr

        return _with_boost(build_document_hybrid_expression(where_document, boost=boost))

    def _build_metadata_filter_for_search_parm(self, where: dict[str, Any] | None) -> list[dict[str, Any]]:
        """
        Build metadata filter conditions for search_parm using JSON_EXTRACT format

        Args:
            where: Metadata filter conditions

        Returns:
            List of filter conditions in search_parm format
            Format: {"term": {"(JSON_EXTRACT(metadata, '$.field_name'))": "value"}}
            or {"range": {"(JSON_EXTRACT(metadata, '$.field_name'))": {"gte": 30, "lte": 90}}}
        """
        if not where:
            return []

        return self._build_metadata_filter_conditions(where)

    def _build_search_parm_field_name(self, key: str) -> str:
        """
        Build field name used in search_parm filters.

        Uses ``JSON_EXTRACT``-wrapped keys for ``DBMS_HYBRID_SEARCH.GET_SQL`` (collection path).
        Namespace ``hybrid_search(TABLE ...)`` rewrites these to ``data_content.metadata.*`` DSL
        keys in ``_adapt_search_parm_for_ns``.
        """
        if key == "#id" or key == CollectionFieldNames.ID:
            return CollectionFieldNames.ID
        return f"(JSON_EXTRACT(metadata, '$.{key}'))"

    def _build_metadata_filter_conditions(self, condition: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Recursively build metadata filter conditions from nested dictionary

        Args:
            condition: Filter condition dictionary

        Returns:
            List of filter conditions
        """
        if not condition:
            return []

        result = []

        # Handle logical operators
        if "$and" in condition:
            must_conditions = []
            for sub_condition in condition["$and"]:
                sub_filters = self._build_metadata_filter_conditions(sub_condition)
                must_conditions.extend(sub_filters)
            if must_conditions:
                # Scalar conditions must be ANDed via a (non-scoring) `filter`
                # clause, not `must`: the kernel rejects scalar term/range queries
                # inside must/should with `scalar ... query in must/should clause
                # not supported`. Hoist must_not-only leaves ($ne/$nin/$not) so they
                # are not nested as standalone bools inside `filter`.
                positive, negative = self._hoist_must_not_from_filters(must_conditions)
                bool_q: dict[str, Any] = {}
                if positive:
                    bool_q["filter"] = positive
                elif negative:
                    bool_q["filter"] = [self._positive_clause_for_must_not(negative)]
                if negative:
                    bool_q["must_not"] = negative
                result.append({"bool": bool_q})
            return result

        if "$or" in condition:
            should_conditions = []
            for sub_condition in condition["$or"]:
                sub_filters = self._build_metadata_filter_conditions(sub_condition)
                should_conditions.extend(sub_filters)
            if should_conditions:
                # `minimum_should_match: 1` makes this an explicit OR. In a
                # non-scoring (filter) context the kernel does not reliably apply the
                # implicit "at least one should" default, which otherwise yields an
                # intermittent `1210 Invalid argument`.
                result.append({"bool": {"should": should_conditions, "minimum_should_match": 1}})
            return result

        if "$not" in condition:
            not_filters = self._build_metadata_filter_conditions(condition["$not"])
            if not_filters:
                result.append({"bool": {"must_not": not_filters}})
            return result

        # Handle field conditions
        for key, value in condition.items():
            if key in ["$and", "$or", "$not"]:
                continue

            # Build field name with JSON_EXTRACT format (or _id for special key)
            field_name = self._build_search_parm_field_name(key)

            if isinstance(value, dict):
                # Handle comparison operators
                range_conditions = {}
                term_value = None

                for op, op_value in value.items():
                    if op == "$eq":
                        term_value = op_value
                    elif op == "$ne":
                        # $ne should be in must_not
                        result.append({"bool": {"must_not": [{"term": {field_name: op_value}}]}})
                    elif op == "$lt":
                        range_conditions["lt"] = op_value
                    elif op == "$lte":
                        range_conditions["lte"] = op_value
                    elif op == "$gt":
                        range_conditions["gt"] = op_value
                    elif op == "$gte":
                        range_conditions["gte"] = op_value
                    elif op == "$in":
                        # For $in, use terms query to match any value in list
                        if isinstance(op_value, (list, tuple)) and len(op_value) > 0:
                            result.append({"terms": {field_name: list(op_value)}})
                    elif op == "$nin" and isinstance(op_value, (list, tuple)) and len(op_value) > 0:
                        # For $nin, use must_not with terms query
                        result.append({"bool": {"must_not": [{"terms": {field_name: list(op_value)}}]}})

                if range_conditions:
                    result.append({"range": {field_name: range_conditions}})
                elif term_value is not None:
                    result.append({"term": {field_name: term_value}})
            else:
                # Direct equality
                result.append({"term": {field_name: value}})

        return result

    def _build_knn_expression(
        self, knn: dict[str, Any], dimension: int | None = None, **kwargs
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Build knn expression from knn dict

        Args:
            knn: Vector search configuration dict with:
                - query_texts: Query text(s) to be embedded (optional if query_embeddings provided)
                - query_embeddings: Query vector(s) (optional if query_texts provided)
                - where: Metadata filter conditions (optional)
                - n_results: Number of results for vector search (optional)
                - boost: Optional weight for this knn search route
            **kwargs: Additional parameters, including:
                embedding_function: EmbeddingFunction instance to convert query_texts to embeddings.
                                   Required if query_texts is provided. Must implement __call__
                                   method that accepts Documents and returns Embeddings (List[List[float]]).
            dimension: Optional collection dimension for validating embeddings

        Returns:
            knn expression dict (or list of dicts when multiple query vectors) with optional filter
        """
        query_texts = knn.get("query_texts")
        query_embeddings = knn.get("query_embeddings")
        where = knn.get("where")
        where_document = knn.get("where_document")
        n_results = knn.get("n_results", 10)
        _validate_n_results(n_results)
        boost = knn.get("boost")

        embedding_function = kwargs.get("embedding_function")

        self._warn_explicit_embeddings_override_embedding_function(
            operation="hybrid_search.knn",
            explicit_embeddings=query_embeddings is not None,
            has_documents=query_texts is not None,
            embedding_function=embedding_function,
        )

        vectors: list[list[float]] = []
        if query_embeddings is not None:
            vectors = _normalize_query_embeddings_fragment(query_embeddings)
        elif query_texts is not None:
            if embedding_function is not None:
                try:
                    texts = query_texts if isinstance(query_texts, list) else [query_texts]
                    embeddings = self._embed_texts(texts, embedding_function=embedding_function)
                    if embeddings and len(embeddings) > 0:
                        vectors = embeddings
                except Exception as e:
                    logger.exception("Failed to generate embeddings from query_texts")
                    raise ValueError(f"Failed to generate embeddings from query_texts: {e}") from e
            else:
                raise ValueError(
                    "knn.query_texts provided but no knn.query_embeddings and no embedding_function. "
                    "Either:\n"
                    "  1. Provide knn.query_embeddings directly, or\n"
                    "  2. Provide embedding_function to auto-generate embeddings from knn.query_texts."
                )
        else:
            raise ValueError(
                "knn requires either query_embeddings or query_texts. "
                "Please provide either:\n"
                "  1. knn.query_embeddings directly, or\n"
                "  2. knn.query_texts with embedding_function to generate embeddings."
            )

        if not vectors:
            return None

        if dimension is not None:
            for vec in vectors:
                if len(vec) != dimension:
                    raise ValueError(f"Embedding dimension mismatch: expected {dimension}, got {len(vec)}")

        # Build knn expressions (one per vector)
        knn_exprs: list[dict[str, Any]] = []
        filter_conditions = self._build_metadata_filter_for_search_parm(where)
        pos_filters, must_not_clauses = self._hoist_must_not_from_filters(filter_conditions)
        knn_filter: list[dict[str, Any]] | None = None
        if must_not_clauses:
            bool_filter: dict[str, Any] = {
                "must_not": must_not_clauses,
                "filter": pos_filters or [self._positive_clause_for_must_not(must_not_clauses)],
            }
            knn_filter = [{"bool": bool_filter}]
        elif pos_filters:
            knn_filter = pos_filters

        if where_document is not None and where_document_knn_prefilterable(where_document):
            doc_filter = document_expr_as_knn_filter(build_document_hybrid_expression(where_document))
            if doc_filter is not None:
                knn_filter = [doc_filter] if knn_filter is None else [*knn_filter, doc_filter]

        for vector in vectors:
            expr = {"field": "embedding", "k": n_results, "query_vector": vector}
            if boost is not None:
                expr["boost"] = boost

            if knn_filter is not None:
                expr["filter"] = knn_filter

            knn_exprs.append(expr)

        return knn_exprs if len(knn_exprs) > 1 else knn_exprs[0]

    def _post_filter_namespace_query_result(
        self,
        result: dict[str, Any],
        where_document: dict[str, Any] | str,
        *,
        n_results: int,
    ) -> dict[str, Any]:
        """Drop hybrid_search rows that violate a where_document predicate."""
        ids_groups = result.get("ids") or []
        if not ids_groups:
            return result

        filtered: dict[str, Any] = {"ids": []}
        if result.get("distances") is not None:
            filtered["distances"] = []
        for optional_key in ("documents", "metadatas", "embeddings"):
            if optional_key in result:
                filtered[optional_key] = []

        for qi, ids in enumerate(ids_groups):
            kept_indices: list[int] = []
            docs_group = (result.get("documents") or [[]])[qi] if result.get("documents") else None
            for idx, _doc_id in enumerate(ids):
                if not docs_group or idx >= len(docs_group) or docs_group[idx] is None:
                    continue
                doc_text = str(docs_group[idx])
                if doc_matches_where_document(doc_text, where_document):
                    kept_indices.append(idx)
                if len(kept_indices) >= n_results:
                    break

            filtered["ids"].append([ids[i] for i in kept_indices])
            if result.get("distances"):
                dist_groups = result.get("distances")
                if dist_groups and qi < len(dist_groups):
                    group = dist_groups[qi]
                    filtered["distances"].append([group[i] for i in kept_indices if i < len(group)])
            for optional_key in ("documents", "metadatas", "embeddings"):
                if optional_key in filtered:
                    groups = result.get(optional_key)
                    if groups and qi < len(groups):
                        group = groups[qi]
                        filtered[optional_key].append([group[i] for i in kept_indices if i < len(group)])

        return filtered

    def _build_source_fields(self, include: list[str] | None) -> list[str]:
        """
        Infer OceanBase GET_SQL `_source` allowlist from include.
        """
        if include is None:
            requested = {"documents", "metadatas"}
        else:
            if not isinstance(include, list) or not all(isinstance(item, str) for item in include):
                raise TypeError("include must be a List[str] or None")
            requested = {item.lower() for item in include}

        source = ["_id"]

        if {"documents", "document"} & requested:
            source.append("document")
        if {"metadatas", "metadata"} & requested:
            source.append("metadata")
        if {"embeddings", "embedding"} & requested:
            source.append("embedding")

        return source

    def _hybrid_row_score(self, row: dict[str, Any]) -> float:
        """Extract relevance score from a hybrid_search SQL row (OB uses ``__score``)."""
        for key in (
            "_distance",
            "distance",
            "_score",
            "score",
            "__score",
            "DISTANCE",
            "_DISTANCE",
            "SCORE",
            "__SCORE",
        ):
            val = row.get(key)
            if val is not None:
                return float(val)
        return 0.0

    def _transform_sql_result(self, result_rows: list[dict[str, Any]], include: list[str] | None) -> dict[str, Any]:
        """
        Transform SQL query results to standard format (query-compatible format)

        Args:
            result_rows: List of row dictionaries from SQL query
            include: Fields to include in results (optional)

        Returns:
            Standard format dictionary with ids, distances, metadatas, documents, embeddings
            in query-compatible format (List[List[...]] for consistency with query method)
        """

        ids = []
        distances = []
        metadatas = []
        documents = []
        include_embeddings = bool(include and {"embeddings", "embedding"} & set(include))
        embeddings: list[Any] | None = [] if include_embeddings else None

        for row in result_rows:
            # Extract id (handle different column names and fallbacks)
            row_id = None
            for key in ("id", "_id", "ID", "Id", "_ID"):
                if key in row and row.get(key) is not None:
                    row_id = row.get(key)
                    break
            if row_id is None:
                for key in row:
                    if isinstance(key, str) and key.lower().endswith("id") and row.get(key) is not None:
                        row_id = row.get(key)
                        break
            row_id = self._convert_id_from_bytes(row_id)
            ids.append(row_id)

            distances.append(self._hybrid_row_score(row))

            # Extract metadata
            if include is None or "metadatas" in include or "metadata" in include:
                metadata = row.get("metadata") or row.get("METADATA")
                # Parse JSON string if needed
                if isinstance(metadata, str):
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        metadata = json.loads(metadata)
                metadatas.append(metadata or {})
            else:
                metadatas.append(None)

            # Extract document
            if include is None or "documents" in include or "document" in include:
                document = row.get("document") or row.get("DOCUMENT")
                documents.append(document)
            else:
                documents.append(None)

            # Extract embedding
            if include_embeddings and embeddings is not None:
                embedding = row.get("embedding") or row.get("EMBEDDING")
                embedding = _parse_embedding_value_fragment(embedding)
                embeddings.append(embedding)

        # Return in query-compatible format (List[List[...]])
        result = {"ids": [ids], "distances": [distances]}

        if include is None or "documents" in include or "document" in include:
            result["documents"] = [documents]

        if include is None or "metadatas" in include or "metadata" in include:
            result["metadatas"] = [metadatas]

        if include_embeddings and embeddings is not None:
            result["embeddings"] = [embeddings]

        return result

    # -------------------- Collection Info --------------------

    def _collection_count(self, collection_id: str | None, collection_name: str) -> int:
        """
        [Internal] Get the number of items in collection - Common SQL-based implementation

        Args:
            collection_id: Collection ID
            collection_name: Collection name

        Returns:
            Item count
        """
        logger.debug(f"Counting items in collection '{collection_name}'")
        conn = self._ensure_connection()

        # Convert collection name to table name
        table_name = self._get_collection_table_name(collection_id, collection_name)

        # Execute COUNT query
        sql = f"SELECT COUNT(*) as cnt FROM `{table_name}`"
        logger.debug(f"Executing SQL: {sql}")

        use_context_manager = self._use_context_manager_for_cursor()
        rows = self._execute_query_with_cursor(conn, sql, [], use_context_manager)

        if not rows:
            count = 0
        else:
            # Extract count from result
            row = rows[0]
            if isinstance(row, dict):
                count = row.get("cnt", 0)
            elif isinstance(row, (tuple, list)):
                count = row[0] if len(row) > 0 else 0
            else:
                count = int(row) if row else 0

        logger.debug(f"✅ Collection '{collection_name}' has {count} items")
        return count
