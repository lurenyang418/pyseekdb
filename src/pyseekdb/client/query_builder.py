"""Pure SQL fragments and query configuration builders.

This module intentionally has no connection or client state so synchronous and
asynchronous clients can share the same SQL configuration semantics.
"""

import json
import struct
from typing import Any

from pymysql.converters import escape_string

from .configuration import FulltextIndexConfig, HNSWConfiguration, IVFConfiguration
from .filters import FilterBuilder
from .schema import SparseVectorIndexConfig


def build_default_ltable_schema(*, has_fulltext: bool = False, has_ivf: bool = False) -> dict[str, Any]:
    """Return the logical-table schema matching the indexes provisioned."""
    index_info: list[dict[str, Any]] = [
        {"index_seq": 0, "index_type": "PRIMARY", "indexed_columns": []},
        {"index_seq": 1, "index_type": "SEARCH_INDEX", "indexed_columns": [1]},
    ]
    next_seq = 2
    if has_fulltext:
        index_info.append({"index_seq": next_seq, "index_type": "FULLTEXT", "indexed_columns": [2]})
        next_seq += 1
    if has_ivf:
        index_info.append({"index_seq": next_seq, "index_type": "IVF", "indexed_columns": [3]})
    return {
        "col_info": [
            {"col_idx": 1, "col_name": "metadata", "col_type": "JSON"},
            {"col_idx": 2, "col_name": "content", "col_type": "TEXT"},
            {"col_idx": 3, "col_name": "embedding", "col_type": "VECTOR"},
        ],
        "index_info": index_info,
    }


def build_fulltext_index_sql(fulltext_config: FulltextIndexConfig | None = None) -> str:
    """Generate a FULLTEXT INDEX SQL clause."""
    if fulltext_config is None:
        return "WITH PARSER ik"

    properties = fulltext_config.properties or {}
    if not properties:
        return f"WITH PARSER {fulltext_config.analyzer}"

    property_parts = []
    for key, value in properties.items():
        rendered = f"'{value}'" if isinstance(value, str) else str(value)
        property_parts.append(f"{key}={rendered}")
    return f"WITH PARSER {fulltext_config.analyzer} PARSER_PROPERTIES=({', '.join(property_parts)})"


def build_vector_index_sql(hnsw_config: HNSWConfiguration) -> str:
    """Generate a dense HNSW vector-index SQL clause."""
    property_parts = []
    for key, value in (hnsw_config.properties or {}).items():
        rendered = f"'{value}'" if isinstance(value, str) else str(value)
        property_parts.append(f"{key}={rendered}")
    optional_fields = (
        ("M", hnsw_config.M),
        ("ef_construction", hnsw_config.ef_construction),
        ("ef_search", hnsw_config.ef_search),
        ("extra_info_max_size", hnsw_config.extra_info_max_size),
        ("refine_k", hnsw_config.refine_k),
        ("refine_type", hnsw_config.refine_type),
        ("bq_bits_query", hnsw_config.bq_bits_query),
        ("bq_use_fht", hnsw_config.bq_use_fht),
    )
    for key, value in optional_fields:
        if value is not None:
            rendered = (
                f"'{value}'"
                if isinstance(value, str)
                else str(value).lower()
                if isinstance(value, bool)
                else str(value)
            )
            property_parts.append(f"{key}={rendered}")
    properties_str = f", {', '.join(property_parts)}" if property_parts else ""
    return f"WITH (DISTANCE={hnsw_config.distance}, TYPE={hnsw_config.type}, LIB={hnsw_config.lib}{properties_str})"


def build_ivf_vector_index_sql(ivf_config: IVFConfiguration) -> str:
    """Generate an IVF vector-index SQL clause."""
    property_parts = []
    for key, value in (ivf_config.properties or {}).items():
        rendered = f"'{value}'" if isinstance(value, str) else str(value)
        property_parts.append(f"{key}={rendered}")
    if ivf_config.centroids_fresh_mode is not None:
        property_parts.append(f"centroids_fresh_mode={ivf_config.centroids_fresh_mode}")
    properties_str = f", {', '.join(property_parts)}" if property_parts else ""
    return (
        f"WITH (DISTANCE={ivf_config.distance}, TYPE={ivf_config.type.upper()}, "
        f"LIB={ivf_config.lib.upper()}{properties_str})"
    )


def build_sparse_vector_index_sql(sparse_config: SparseVectorIndexConfig) -> str:
    """Generate a sparse vector-index SQL clause."""
    parts = [
        f"DISTANCE={sparse_config.distance}",
        f"TYPE={sparse_config.type}",
        f"LIB={sparse_config.lib}",
    ]
    optional_fields = (
        ("prune", sparse_config.prune),
        ("refine", sparse_config.refine),
        ("drop_ratio_build", sparse_config.drop_ratio_build),
        ("drop_ratio_search", sparse_config.drop_ratio_search),
        ("refine_k", sparse_config.refine_k),
    )
    for key, value in optional_fields:
        if value is not None:
            rendered = str(value).lower() if isinstance(value, bool) else str(value)
            parts.append(f"{key}={rendered}")
    for key, value in (sparse_config.properties or {}).items():
        rendered = f"'{value}'" if isinstance(value, str) else str(value)
        parts.append(f"{key}={rendered}")
    return f"WITH ({', '.join(parts)})"


def embedding_to_hexstring(embedding: list[float]) -> str:
    """Serialize a dense embedding as a compact SQL hex literal."""
    if not embedding:
        return ""
    binary = struct.pack(f"<{len(embedding)}f", *embedding)
    return f"X'{binary.hex()}'"


def normalize_query_embeddings(
    query_embeddings: list[float] | list[list[float]] | None,
) -> list[list[float]]:
    """Normalize a single vector or a vector batch to a list of vectors."""
    if query_embeddings is None:
        return []
    if query_embeddings and isinstance(query_embeddings[0], (int, float)):
        return [query_embeddings]  # type: ignore[list-item]
    return query_embeddings


def normalize_include_fields(include: list[str] | None) -> dict[str, bool]:
    """Normalize the optional result-field list used by collection queries."""
    if include is None:
        return {"documents": True, "metadatas": True}
    return dict.fromkeys(include, True)


def embed_texts(
    texts: str | list[str],
    embedding_function: Any = None,
    **kwargs: Any,
) -> list[list[float]]:
    """Generate vectors with a synchronous embedding function."""
    del kwargs
    if embedding_function is None:
        raise NotImplementedError(
            "Text embedding is not implemented. "
            "Please provide query_embeddings directly or set embedding_function in collection."
        )
    return embedding_function([texts] if isinstance(texts, str) else texts)


def build_select_clause(include_fields: dict[str, bool]) -> str:
    """Build the standard collection SELECT field list."""
    select_fields = ["_id"]
    if include_fields.get("embeddings") or include_fields.get("embedding"):
        select_fields.append("embedding")
    if include_fields.get("documents") or include_fields.get("document"):
        select_fields.append("document")
    if include_fields.get("metadatas") or include_fields.get("metadata"):
        select_fields.append("metadata")
    return ", ".join(select_fields)


def build_where_clause(
    where: dict[str, Any] | None = None,
    where_document: dict[str, Any] | None = None,
    id_list: list[str] | None = None,
) -> tuple[str, list[Any]]:
    """Build a parameterized WHERE clause for standard collection queries."""
    where_clauses: list[str] = []
    params: list[Any] = []
    if id_list:
        processed_ids = ["CAST(%s AS BINARY)" for _ in id_list]
        where_clauses.append(f"_id IN ({','.join(processed_ids)})")
        params.extend(str(value) if not isinstance(value, str) else value for value in id_list)
    if where:
        meta_clause, meta_params = FilterBuilder.build_metadata_filter(where, "metadata")
        if meta_clause:
            where_clauses.append(meta_clause)
            params.extend(meta_params)
    if where_document:
        doc_clause, doc_params = FilterBuilder.build_document_filter(where_document, "document")
        if doc_clause:
            where_clauses.append(doc_clause)
            params.extend(doc_params)
    return (f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""), params


def parse_row_value(value: Any) -> Any:
    """Decode JSON text returned by a database row when possible."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def convert_id_to_sql(id_value: str) -> str:
    """Build a safely escaped CAST expression for a binary record ID."""
    if not isinstance(id_value, str):
        id_value = str(id_value)
    return f"CAST('{escape_string(id_value)}' AS BINARY)"


def convert_id_to_sql_with_parameters(id_value: str) -> tuple[str, str]:
    """Build a parameterized CAST expression for a binary record ID."""
    return "CAST(%s AS BINARY)", id_value


def convert_id_from_bytes(record_id: Any) -> str | None:
    """Convert a database binary ID to its string representation."""
    if record_id is None:
        return None
    if isinstance(record_id, str):
        return record_id
    if isinstance(record_id, bytes):
        try:
            return record_id.decode("utf-8")
        except UnicodeDecodeError:
            return record_id.hex()
    return str(record_id)


def process_query_row(row: dict[str, Any], include_fields: dict[str, bool]) -> dict[str, Any]:
    """Normalize one vector-query result row."""
    del include_fields
    result_item: dict[str, Any] = {"_id": convert_id_from_bytes(row["_id"])}
    if "document" in row and row["document"] is not None:
        result_item["document"] = row["document"]
    if "embedding" in row and row["embedding"] is not None:
        result_item["embedding"] = parse_row_value(row["embedding"])
    if "metadata" in row and row["metadata"] is not None:
        result_item["metadata"] = parse_row_value(row["metadata"])
    if "distance" in row:
        result_item["distance"] = float(row["distance"])
    return result_item


def process_get_row(row: dict[str, Any], include_fields: dict[str, bool]) -> dict[str, Any]:
    """Normalize one standard collection get-result row."""
    record_id = convert_id_from_bytes(row["_id"])
    document = (
        row["document"]
        if (include_fields.get("documents") or include_fields.get("document")) and "document" in row
        else None
    )
    metadata = (
        parse_row_value(row["metadata"])
        if (include_fields.get("metadatas") or include_fields.get("metadata")) and row.get("metadata") is not None
        else None
    )
    embedding = (
        parse_row_value(row["embedding"])
        if (include_fields.get("embeddings") or include_fields.get("embedding")) and row.get("embedding") is not None
        else None
    )
    return {"id": record_id, "document": document, "embedding": embedding, "metadata": metadata}
