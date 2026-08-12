"""
Namespace query integration tests.
Tests vector similarity query, metadata filtering, include control, and multi-namespace isolation.
"""

import time

import pytest
from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT

from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import FulltextIndexConfig, VectorIndexConfig
from pyseekdb.client.schema import Schema


class TestNamespaceQuery:
    """TestNamespaceQuery class."""

    def _setup(self, client, suffix=""):
        """Setup."""
        name = f"test_ns_q_{int(time.time() * 1000)}{suffix}"
        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=IVFConfiguration(dimension=3, distance="l2", centroids_fresh_mode="spfresh"),
                embedding_function=None,
            ),
            fulltext_index=FulltextIndexConfig(analyzer="ik"),
        )
        collection = client.create_collection(
            name=name,
            schema=schema,
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        return collection

    def _insert_data(self, ns):
        """Insert data."""
        ns.add(
            ids=["q1", "q2", "q3", "q4", "q5"],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            documents=[
                "Machine learning basics",
                "Python programming guide",
                "OceanBase distributed database",
                "Advanced ML algorithms",
                "Data science with Python",
            ],
            metadatas=[
                {"category": "AI", "score": 95},
                {"category": "Programming", "score": 88},
                {"category": "Database", "score": 92},
                {"category": "AI", "score": 90},
                {"category": "Data Science", "score": 85},
            ],
        )

    def test_basic_vector_query(self, db_client):
        """Test basic vector query."""
        collection = self._setup(db_client)
        ns = collection.create_namespace("qns")
        try:
            self._insert_data(ns)
            result = ns.query(query_embeddings=[1.0, 0.0, 0.0], n_results=3)
            assert result is not None
            assert "ids" in result
            assert len(result["ids"]) > 0
            assert len(result["ids"][0]) <= 3
        finally:
            db_client.delete_collection(name=collection.name)

    def test_query_with_metadata_filter(self, db_client):
        """Test query with metadata filter."""
        collection = self._setup(db_client)
        ns = collection.create_namespace("qns_meta")
        try:
            self._insert_data(ns)
            result = ns.query(
                query_embeddings=[1.0, 0.0, 0.0],
                n_results=5,
                where={"category": "AI"},
            )
            assert result is not None
            assert len(result["ids"][0]) > 0
            if result.get("metadatas"):
                for meta in result["metadatas"][0]:
                    assert meta["category"] == "AI"
        finally:
            db_client.delete_collection(name=collection.name)

    def test_query_with_score_filter(self, db_client):
        """Test query with score filter."""
        collection = self._setup(db_client)
        ns = collection.create_namespace("qns_score")
        try:
            self._insert_data(ns)
            result = ns.query(
                query_embeddings=[1.0, 0.0, 0.0],
                n_results=5,
                where={"score": {"$gte": 90}},
            )
            assert result is not None
            assert len(result["ids"][0]) > 0
        finally:
            db_client.delete_collection(name=collection.name)

    def test_query_with_include(self, db_client):
        """Test query with include."""
        collection = self._setup(db_client)
        ns = collection.create_namespace("qns_inc")
        try:
            self._insert_data(ns)
            result = ns.query(
                query_embeddings=[1.0, 0.0, 0.0],
                n_results=3,
                include=["documents", "metadatas"],
            )
            assert "documents" in result
            assert "metadatas" in result
            assert len(result["documents"][0]) == len(result["ids"][0])
            assert len(result["metadatas"][0]) == len(result["ids"][0])
        finally:
            db_client.delete_collection(name=collection.name)

    def test_query_include_embeddings(self, db_client):
        """Test query include embeddings."""
        collection = self._setup(db_client)
        ns = collection.create_namespace("qns_emb")
        try:
            self._insert_data(ns)
            result = ns.query(
                query_embeddings=[1.0, 0.0, 0.0],
                n_results=2,
                include=["embeddings"],
            )
            assert "embeddings" in result
            if result["embeddings"] and result["embeddings"][0]:
                assert len(result["embeddings"][0][0]) == 3
        finally:
            db_client.delete_collection(name=collection.name)

    def test_query_rejects_invalid_include(self, db_client):
        """Invalid include fields must fail before executing the query."""
        collection = self._setup(db_client)
        ns = collection.create_namespace("qns_bad_inc")
        try:
            self._insert_data(ns)
            with pytest.raises(ValueError, match="Invalid include field"):
                ns.query(
                    query_embeddings=[1.0, 0.0, 0.0],
                    n_results=3,
                    include=["invalid_field_name"],
                )
        finally:
            db_client.delete_collection(name=collection.name)

    def test_multi_namespace_isolation(self, db_client):
        """Test multi namespace isolation."""
        collection = self._setup(db_client, suffix="_iso")
        ns_a = collection.create_namespace("tenant_a")
        ns_b = collection.create_namespace("tenant_b")
        try:
            ns_a.add(
                ids=["a1"],
                embeddings=[[1.0, 0.0, 0.0]],
                documents=["Data from tenant A"],
                metadatas=[{"owner": "A"}],
            )
            ns_b.add(
                ids=["b1"],
                embeddings=[[0.0, 1.0, 0.0]],
                documents=["Data from tenant B"],
                metadatas=[{"owner": "B"}],
            )

            result_a = ns_a.query(query_embeddings=[1.0, 0.0, 0.0], n_results=10)
            result_b = ns_b.query(query_embeddings=[1.0, 0.0, 0.0], n_results=10)

            ids_a = result_a["ids"][0] if result_a["ids"] else []
            ids_b = result_b["ids"][0] if result_b["ids"] else []

            assert "a1" in ids_a
            assert "b1" not in ids_a
            assert "b1" in ids_b
            assert "a1" not in ids_b
        finally:
            db_client.delete_collection(name=collection.name)

    def test_same_id_different_namespaces(self, db_client):
        """Test same id different namespaces."""
        collection = self._setup(db_client, suffix="_sameid")
        ns_x = collection.create_namespace("ns_x")
        ns_y = collection.create_namespace("ns_y")
        try:
            ns_x.add(ids="shared_id", embeddings=[1.0, 0.0, 0.0], metadatas={"src": "X"})
            ns_y.add(ids="shared_id", embeddings=[0.0, 1.0, 0.0], metadatas={"src": "Y"})

            res_x = ns_x.get(ids="shared_id", include=["metadatas"])
            res_y = ns_y.get(ids="shared_id", include=["metadatas"])

            assert len(res_x["ids"]) == 1
            assert res_x["metadatas"][0]["src"] == "X"
            assert len(res_y["ids"]) == 1
            assert res_y["metadatas"][0]["src"] == "Y"
        finally:
            db_client.delete_collection(name=collection.name)

    # ==================== hybrid_search tests ====================

    def test_hybrid_search_fulltext_only(self, db_client):
        """Test hybrid search fulltext only."""
        collection = self._setup(db_client)
        ns = collection.create_namespace("qns_hs_ft")
        try:
            self._insert_data(ns)
            time.sleep(1)
            result = ns.hybrid_search(
                query={"where_document": {"$contains": "machine learning"}},
                n_results=3,
                include=["documents", "metadatas"],
            )
            assert result is not None
            assert "ids" in result
            assert len(result["ids"]) > 0
            assert len(result["ids"][0]) > 0
            if "documents" in result and result["documents"][0]:
                for doc in result["documents"][0]:
                    if doc:
                        assert "machine" in doc.lower() or "learning" in doc.lower()
        finally:
            db_client.delete_collection(name=collection.name)

    def test_hybrid_search_vector_only(self, db_client):
        """Test hybrid search vector only."""
        collection = self._setup(db_client)
        ns = collection.create_namespace("qns_hs_vec")
        try:
            self._insert_data(ns)
            time.sleep(1)
            result = ns.hybrid_search(
                knn={"query_embeddings": [1.0, 0.0, 0.0], "n_results": 3},
                n_results=3,
                include=["documents"],
            )
            assert result is not None
            assert "ids" in result
            assert "distances" in result
            assert len(result["ids"]) > 0
            assert len(result["ids"][0]) > 0
            for dist in result["distances"][0]:
                assert dist >= 0
        finally:
            db_client.delete_collection(name=collection.name)

    def test_hybrid_search_combined(self, db_client):
        """Test hybrid search combined."""
        collection = self._setup(db_client)
        ns = collection.create_namespace("qns_hs_comb")
        try:
            self._insert_data(ns)
            time.sleep(1)
            result = ns.hybrid_search(
                query={"where_document": {"$contains": "machine"}, "n_results": 5},
                knn={"query_embeddings": [1.0, 0.0, 0.0], "n_results": 5},
                rank={"rrf": {"rank_window_size": 60, "rank_constant": 60}},
                n_results=3,
                include=["documents", "metadatas"],
            )
            assert result is not None
            assert "ids" in result
            assert len(result["ids"]) > 0
            assert len(result["ids"][0]) > 0
        finally:
            db_client.delete_collection(name=collection.name)

    def test_hybrid_search_namespace_isolation(self, db_client):
        """Test hybrid search namespace isolation."""
        collection = self._setup(db_client, suffix="_hs_iso")
        ns_a = collection.create_namespace("hs_tenant_a")
        ns_b = collection.create_namespace("hs_tenant_b")
        try:
            ns_a.add(
                ids=["ha1"],
                embeddings=[[1.0, 0.0, 0.0]],
                documents=["Machine learning from tenant A"],
                metadatas=[{"owner": "A"}],
            )
            ns_b.add(
                ids=["hb1"],
                embeddings=[[0.0, 1.0, 0.0]],
                documents=["Python programming from tenant B"],
                metadatas=[{"owner": "B"}],
            )
            time.sleep(1)

            result_a = ns_a.hybrid_search(
                knn={"query_embeddings": [1.0, 0.0, 0.0], "n_results": 10},
                n_results=10,
                include=["documents"],
            )
            result_b = ns_b.hybrid_search(
                knn={"query_embeddings": [1.0, 0.0, 0.0], "n_results": 10},
                n_results=10,
                include=["documents"],
            )

            ids_a = result_a["ids"][0] if result_a["ids"] else []
            ids_b = result_b["ids"][0] if result_b["ids"] else []

            assert "ha1" in ids_a
            assert "hb1" not in ids_a
            assert "hb1" in ids_b
            assert "ha1" not in ids_b
        finally:
            db_client.delete_collection(name=collection.name)

    def test_hybrid_search_with_data_content_filter(self, db_client):
        """
        Namespace hybrid_search applies metadata / id filters on JSON under ``data_content``
        (SDK maps ``metadata.*`` / ``_id`` to ``data_content.metadata.*`` / ``data_content.id``
        for OceanBase hybrid_search field syntax).
        """
        collection = self._setup(db_client)
        ns = collection.create_namespace("qns_hs_dc")
        try:
            self._insert_data(ns)
            time.sleep(1)

            # Full-text + filter on data_content.metadata.category
            result_ft = ns.hybrid_search(
                query={
                    "where_document": {"$contains": "Python"},
                    "where": {"category": "Programming"},
                    "n_results": 10,
                },
                n_results=10,
                include=["documents", "metadatas"],
            )
            ids_ft = result_ft["ids"][0] if result_ft["ids"] else []
            assert ids_ft == ["q2"], ids_ft
            if result_ft.get("metadatas") and result_ft["metadatas"][0]:
                for meta in result_ft["metadatas"][0]:
                    if meta:
                        assert meta["category"] == "Programming"
        finally:
            db_client.delete_collection(name=collection.name)

    def test_hybrid_search_with_data_content_filter_knn_branch(self, db_client):
        """KNN + metadata / id filters on ``data_content`` (same field mapping as full-text branch)."""
        collection = self._setup(db_client)
        ns = collection.create_namespace("qns_hs_dc_knn")
        try:
            self._insert_data(ns)
            time.sleep(1)
            result_knn = ns.hybrid_search(
                knn={
                    "query_embeddings": [1.0, 0.0, 0.0],
                    "where": {
                        "$and": [
                            {"score": {"$gte": 90}},
                            {"#id": "q1"},
                        ]
                    },
                    "n_results": 10,
                },
                n_results=10,
                include=["metadatas"],
            )
            ids_knn = result_knn["ids"][0] if result_knn["ids"] else []
            assert ids_knn == ["q1"], ids_knn
            if result_knn.get("metadatas") and result_knn["metadatas"][0]:
                assert result_knn["metadatas"][0][0]["score"] >= 90
        finally:
            db_client.delete_collection(name=collection.name)

    def test_hybrid_search_scalar_only_metadata_filter(self, db_client):
        """
        Scalar-only hybrid_search (only ``where``, no ``where_document`` and no ``knn``).

        Regression: the namespace filter injection used to wrap the scalar leaf query in a
        ``must`` clause, which the kernel rejects with
        ``OB_NOT_SUPPORTED: scalar term query in must/should clause`` because a top-level
        bool query is scoring by default. Scalar leaves must go into ``filter``.
        """
        collection = self._setup(db_client)
        ns = collection.create_namespace("qns_hs_scalar")
        try:
            self._insert_data(ns)
            time.sleep(1)

            # term filter
            result_term = ns.hybrid_search(
                query={"where": {"category": "AI"}},
                n_results=10,
                include=["metadatas"],
            )
            ids_term = sorted(result_term["ids"][0]) if result_term["ids"] else []
            assert ids_term == ["q1", "q4"], ids_term
            if result_term.get("metadatas") and result_term["metadatas"][0]:
                for meta in result_term["metadatas"][0]:
                    if meta:
                        assert meta["category"] == "AI"

            # range filter
            result_range = ns.hybrid_search(
                query={"where": {"score": {"$gte": 90}}},
                n_results=10,
                include=["metadatas"],
            )
            ids_range = sorted(result_range["ids"][0]) if result_range["ids"] else []
            assert ids_range == ["q1", "q3", "q4"], ids_range
        finally:
            db_client.delete_collection(name=collection.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
