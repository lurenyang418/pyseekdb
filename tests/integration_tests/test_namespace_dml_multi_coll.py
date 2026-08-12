"""
Namespace DML integration tests: multiple collections x 1 namespace each.
Verifies collection-level isolation via get.
"""

import pytest
from namespace_dml_helpers import (
    assert_get_absent,
    assert_get_present,
    cleanup,
    create_ns_collection,
)


class TestNamespaceDMLMultiColl:
    """TestNamespaceDMLMultiColl class."""

    def test_same_namespace_name_across_collections(self, db_client):
        """Test same namespace name across collections."""
        coll_a = create_ns_collection(db_client, suffix="_mc_a")
        coll_b = create_ns_collection(db_client, suffix="_mc_b")
        ns_a = coll_a.create_namespace("tenant")
        ns_b = coll_b.create_namespace("tenant")
        ns_a.prewarm()
        ns_b.prewarm()
        try:
            ns_a.add(
                ids="doc1",
                embeddings=[1.0, 0.0, 0.0],
                documents="Data in coll A",
                metadatas={"coll": "A"},
            )
            ns_b.add(
                ids="doc1",
                embeddings=[0.0, 1.0, 0.0],
                documents="Data in coll B",
                metadatas={"coll": "B"},
            )

            assert_get_present(ns_a, "doc1", documents="Data in coll A", metadatas={"coll": "A"})
            assert_get_present(ns_b, "doc1", documents="Data in coll B", metadatas={"coll": "B"})
        finally:
            cleanup(db_client, coll_a, coll_b)

    def test_update_delete_isolated_by_collection(self, db_client):
        """Test update delete isolated by collection."""
        coll_a = create_ns_collection(db_client, suffix="_mc_upd_a")
        coll_b = create_ns_collection(db_client, suffix="_mc_upd_b")
        ns_a = coll_a.create_namespace("tenant")
        ns_b = coll_b.create_namespace("tenant")
        ns_a.prewarm()
        ns_b.prewarm()
        try:
            ns_a.add(ids="doc1", embeddings=[1.0, 0.0, 0.0], documents="Original A")
            ns_b.add(ids="doc1", embeddings=[0.0, 1.0, 0.0], documents="Original B")

            ns_a.update(ids="doc1", embeddings=[0.5, 0.5, 0.5], documents="Updated A", metadatas={"updated": True})
            assert_get_present(ns_a, "doc1", documents="Updated A", metadatas={"updated": True})
            assert_get_present(ns_b, "doc1", documents="Original B")

            ns_a.delete(ids="doc1")
            assert_get_absent(ns_a, "doc1")
            assert_get_present(ns_b, "doc1", documents="Original B")
        finally:
            cleanup(db_client, coll_a, coll_b)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
