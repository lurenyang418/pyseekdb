"""
Integration tests for optional namespace collection indexes.

Product behavior (Agent Database):
- ``schema.vector_index.ivf`` unset → no VECTOR INDEX
- ``schema.fulltext_index`` unset → no FULLTEXT INDEX
- SEARCH INDEX on ``data_content`` is always created

These tests verify DDL, settings metadata, and query behavior for:
1. fulltext only
2. vector (IVF) only
3. search index only (no fulltext, no vector)
"""

from __future__ import annotations

import json
import time
from typing import Any

import pymysql.err
import pytest
from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT

from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import FulltextIndexConfig, VectorIndexConfig
from pyseekdb.client.meta_info import NamespaceCollectionNames
from pyseekdb.client.schema import Schema


def _schema_fts_only() -> Schema:
    """Schema fts only."""
    return Schema(
        vector_index=VectorIndexConfig(embedding_function=None),
        fulltext_index=FulltextIndexConfig(analyzer="ik"),
    )


def _schema_vector_only() -> Schema:
    """Schema vector only."""
    return Schema(
        vector_index=VectorIndexConfig(
            ivf=IVFConfiguration(dimension=3, distance="l2", centroids_fresh_mode="spfresh"),
            embedding_function=None,
        ),
    )


def _schema_search_only() -> Schema:
    """Schema search only."""
    return Schema(vector_index=VectorIndexConfig(embedding_function=None))


def _index_names(client: Any, collection_id: str) -> set[str]:
    """Index names."""
    table = NamespaceCollectionNames.data_table_name(collection_id)
    rows = client._server._execute(f"SHOW INDEX FROM `{table}`")
    return {(r.get("Key_name") if isinstance(r, dict) else r[2]) for r in (rows or [])}


def _collection_settings(client: Any, collection_id: str) -> dict:
    """Collection settings."""
    rows = client._server._execute(f"SELECT settings FROM sdk_collections WHERE collection_id = '{collection_id}'")
    raw = rows[0]["settings"] if isinstance(rows[0], dict) else rows[0][0]
    return json.loads(raw) if raw else {}


def _create_collection(client: Any, label: str, schema: Schema) -> Any:
    """Create collection."""
    name = f"test_ns_opt_idx_{label}_{int(time.time() * 1000)}"
    return client.create_collection(
        name=name,
        schema=schema,
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )


def _unit_vector(dimension: int, axis: int = 0) -> list[float]:
    """Unit vector."""
    vec = [0.0] * dimension
    vec[axis % dimension] = 1.0
    return vec


def _seed_namespace(ns: Any, dimension: int) -> None:
    """Seed namespace."""
    ns.add(
        ids=["d1", "d2"],
        embeddings=[_unit_vector(dimension, 0), _unit_vector(dimension, 1)],
        documents=["machine learning guide", "python database tutorial"],
        metadatas=[{"category": "AI", "score": 90}, {"category": "Programming", "score": 80}],
    )


class TestNamespaceOptionalIndexes:
    """DDL + query matrix for optional VECTOR / FULLTEXT indexes."""

    def test_fts_only_indexes_and_queries(self, db_client):
        """Test fts only indexes and queries."""
        collection = _create_collection(db_client, "fts_only", _schema_fts_only())
        try:
            keys = _index_names(db_client, collection.id)
            settings = _collection_settings(db_client, collection.id)
            assert keys == {"idx_fts", "idx_json"}
            assert "dense_index_type" not in settings

            ns = collection.create_namespace("ns")
            _seed_namespace(ns, collection.dimension)
            time.sleep(1)

            result = ns.hybrid_search(
                query={"where_document": {"$contains": "machine"}},
                n_results=5,
                include=["documents"],
            )
            assert result["ids"] and result["ids"][0]

            with pytest.raises(pymysql.err.Error, match="knn search without vector index not supported"):
                ns.query(query_embeddings=_unit_vector(collection.dimension), n_results=3)

            with pytest.raises(pymysql.err.Error, match="knn search without vector index not supported"):
                ns.hybrid_search(
                    knn={"query_embeddings": _unit_vector(collection.dimension), "n_results": 3},
                    n_results=3,
                )
        finally:
            db_client.delete_collection(name=collection.name)

    def test_vector_only_indexes_and_queries(self, db_client):
        """Test vector only indexes and queries."""
        collection = _create_collection(db_client, "vec_only", _schema_vector_only())
        try:
            keys = _index_names(db_client, collection.id)
            settings = _collection_settings(db_client, collection.id)
            assert keys == {"idx_json", "idx_vec"}
            assert settings.get("dense_index_type") == "ivf"
            assert settings.get("centroids_fresh_mode") == "spfresh"

            ns = collection.create_namespace("ns")
            _seed_namespace(ns, collection.dimension)
            time.sleep(1)

            result = ns.query(query_embeddings=_unit_vector(3), n_results=3)
            assert result["ids"] and result["ids"][0]

            with pytest.raises(ValueError, match="Embedding dimension mismatch: expected 3"):
                ns.add(ids="bad_dim", embeddings=[1.0, 2.0])

            knn = ns.hybrid_search(
                knn={"query_embeddings": _unit_vector(3), "n_results": 3},
                n_results=3,
            )
            assert knn["ids"] and knn["ids"][0]

            with pytest.raises(pymysql.err.OperationalError, match="FULLTEXT"):
                ns.hybrid_search(
                    query={"where_document": {"$contains": "machine"}},
                    n_results=5,
                )
        finally:
            db_client.delete_collection(name=collection.name)

    def test_search_only_indexes_and_queries(self, db_client):
        """Test search only indexes and queries."""
        collection = _create_collection(db_client, "search_only", _schema_search_only())
        try:
            keys = _index_names(db_client, collection.id)
            settings = _collection_settings(db_client, collection.id)
            assert keys == {"idx_json"}
            assert "dense_index_type" not in settings

            ns = collection.create_namespace("ns")
            _seed_namespace(ns, collection.dimension)
            time.sleep(1)

            scalar = ns.hybrid_search(
                query={"where": {"category": "AI"}},
                n_results=10,
                include=["metadatas"],
            )
            ids = scalar["ids"][0] if scalar["ids"] else []
            assert ids == ["d1"]

            got = ns.get(where={"category": "Programming"})
            assert sorted(got["ids"]) == ["d2"]

            with pytest.raises(pymysql.err.OperationalError, match="FULLTEXT"):
                ns.hybrid_search(
                    query={"where_document": {"$contains": "machine"}},
                    n_results=5,
                )

            with pytest.raises(pymysql.err.Error, match="knn search without vector index not supported"):
                ns.query(query_embeddings=_unit_vector(collection.dimension), n_results=3)

            with pytest.raises(pymysql.err.Error, match="knn search without vector index not supported"):
                ns.hybrid_search(
                    knn={"query_embeddings": _unit_vector(collection.dimension), "n_results": 3},
                    n_results=3,
                )
        finally:
            db_client.delete_collection(name=collection.name)

    def test_search_only_rejects_non_default_explicit_embeddings(self, db_client):
        """Without VECTOR INDEX, explicit embeddings must match VECTOR(384)."""
        collection = _create_collection(db_client, "search_dim", _schema_search_only())
        try:
            assert collection.dimension == 384
            assert collection.has_vector_index is False
            ns = collection.create_namespace("ns")

            with pytest.raises(ValueError, match="384-dimensional"):
                ns.add(ids="bad", embeddings=[1.0, 2.0, 3.0])

            ns.add(ids="ok", embeddings=[0.0] * 384)
            got = ns.get(ids="ok", include=["embeddings"])
            assert len(got["embeddings"][0]) == 384
        finally:
            db_client.delete_collection(name=collection.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-k", "oceanbase"])
