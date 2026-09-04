"""Pure SQL fragments and query configuration builders.

This module intentionally has no connection or client state so synchronous and
asynchronous clients can share the same SQL configuration semantics.
"""

import struct
from typing import Any

from .configuration import FulltextIndexConfig, HNSWConfiguration, IVFConfiguration
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
