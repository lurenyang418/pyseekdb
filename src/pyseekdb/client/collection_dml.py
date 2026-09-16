"""Shared collection write preparation and SQL builders.

The synchronous and asynchronous clients intentionally keep their own
connection and transaction handling. This module contains the state-free
parts of collection writes so both execution layers use the same input and
SQL semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from .meta_info import CollectionFieldNames
from .query_builder import embedding_to_hexstring, normalize_collection_batch
from .sparse_embedding_function import SparseVector, _sparse_vector_to_sql

CollectionWriteOperation = Literal["add", "update", "upsert"]


@dataclass(frozen=True)
class CollectionWriteBatch:
    """Normalized records ready for a collection write operation."""

    ids: list[str]
    embeddings: list[list[float]] | None
    metadatas: list[dict] | None
    documents: list[str] | None


def prepare_collection_write_batch(
    ids: str | list[str],
    embeddings: list[float] | list[list[float]] | None,
    metadatas: dict | list[dict] | None,
    documents: str | list[str] | None,
    *,
    operation: CollectionWriteOperation,
    embedding_function: Any = None,
) -> CollectionWriteBatch:
    """Normalize a write batch and generate dense vectors when necessary.

    ``EmbeddingFunction`` is deliberately typed as ``Any`` here: the helper
    is shared by both clients and only requires the callable protocol. The
    generated result is validated before SQL is assembled, which keeps a
    malformed embedding function from causing a partially executed batch.
    """
    id_list, embedding_list, metadata_list, document_list = normalize_collection_batch(
        ids, embeddings, metadatas, documents
    )

    if operation not in {"add", "update", "upsert"}:
        raise ValueError(f"Unsupported collection write operation: {operation}")

    if embedding_list is None and document_list is not None:
        if embedding_function is None:
            raise ValueError("Documents require an embedding function")
        try:
            generated_embeddings = embedding_function(document_list)
            embedding_list = list(generated_embeddings)
        except Exception as exc:
            raise ValueError(f"Failed to generate embeddings from documents: {exc}") from exc
        if len(embedding_list) != len(id_list):
            raise ValueError(f"Embedding function returned {len(embedding_list)} vectors, expected {len(id_list)}")

    has_payload = embedding_list is not None or document_list is not None or metadata_list is not None
    if not has_payload or (operation == "add" and embedding_list is None and document_list is None):
        if operation == "add":
            raise ValueError("Add requires embeddings or documents")
        raise ValueError(f"{operation.capitalize()} requires embeddings, documents, or metadatas")

    return CollectionWriteBatch(
        ids=id_list,
        embeddings=embedding_list,
        metadatas=metadata_list,
        documents=document_list,
    )


def build_collection_upsert_statement(
    table_sql: str,
    record_id: str,
    document: str | None,
    embedding: list[float] | None,
    metadata: dict | None,
    sparse_embedding: SparseVector | dict[int, float] | None = None,
) -> tuple[str, list[Any]]:
    """Build one parameterized collection ``INSERT ... ON DUPLICATE KEY``.

    Vector values remain safe SQL literals because they are serialized from
    numeric values by the shared vector serializer. IDs, documents, and
    metadata use driver parameters in both clients.
    """
    vector_sql = "NULL" if embedding is None else embedding_to_hexstring(embedding)
    columns = [
        CollectionFieldNames.ID,
        CollectionFieldNames.DOCUMENT,
        CollectionFieldNames.EMBEDDING,
        CollectionFieldNames.METADATA,
    ]
    values = ["CAST(%s AS BINARY)", "%s", vector_sql, "%s"]
    params: list[Any] = [
        record_id,
        document,
        json.dumps(metadata, ensure_ascii=False) if metadata is not None else None,
    ]
    update_clauses: list[str] = []

    if document is not None:
        update_clauses.append(f"{CollectionFieldNames.DOCUMENT} = VALUES({CollectionFieldNames.DOCUMENT})")
    if embedding is not None:
        update_clauses.append(f"{CollectionFieldNames.EMBEDDING} = VALUES({CollectionFieldNames.EMBEDDING})")
    if metadata is not None:
        update_clauses.append(f"{CollectionFieldNames.METADATA} = VALUES({CollectionFieldNames.METADATA})")

    if sparse_embedding is not None:
        columns.append(CollectionFieldNames.SPARSE_EMBEDDING)
        values.append(_sparse_vector_to_sql(sparse_embedding))
        update_clauses.append(
            f"{CollectionFieldNames.SPARSE_EMBEDDING} = VALUES({CollectionFieldNames.SPARSE_EMBEDDING})"
        )

    if not update_clauses:
        update_clauses.append(f"{CollectionFieldNames.ID} = {CollectionFieldNames.ID}")

    sql = (
        f"INSERT INTO {table_sql} ({', '.join(columns)}) VALUES ({', '.join(values)}) "  # noqa: S608 - table fragment is validated by callers
        f"ON DUPLICATE KEY UPDATE {', '.join(update_clauses)}"
    )
    return sql, params
