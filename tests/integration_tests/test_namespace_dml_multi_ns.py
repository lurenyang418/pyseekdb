"""
Namespace DML integration tests: 1 collection x multiple namespaces.
Verifies add/update/upsert/delete isolation via get.
"""

import pytest
from namespace_dml_helpers import (
    assert_get_absent,
    assert_get_present,
    cleanup,
    create_ns_collection,
)


class TestNamespaceDMLMultiNs:
    """TestNamespaceDMLMultiNs class."""

    def test_add_isolation(self, db_client):
        """Test add isolation."""
        collection = create_ns_collection(db_client, suffix="_mns_add")
        ns_a = collection.create_namespace("ns_a")
        ns_b = collection.create_namespace("ns_b")
        ns_a.prewarm()
        ns_b.prewarm()
        try:
            ns_a.add(ids="a1", embeddings=[1.0, 0.0, 0.0], metadatas={"owner": "A"})
            ns_b.add(ids="b1", embeddings=[0.0, 1.0, 0.0], metadatas={"owner": "B"})

            assert_get_present(ns_a, "a1", metadatas={"owner": "A"})
            assert_get_present(ns_b, "b1", metadatas={"owner": "B"})
            assert_get_absent(ns_a, "b1")
            assert_get_absent(ns_b, "a1")
        finally:
            cleanup(db_client, collection)

    def test_same_id_different_namespaces(self, db_client):
        """Test same id different namespaces."""
        collection = create_ns_collection(db_client, suffix="_mns_sameid")
        ns_x = collection.create_namespace("ns_x")
        ns_y = collection.create_namespace("ns_y")
        ns_x.prewarm()
        ns_y.prewarm()
        try:
            ns_x.add(ids="shared", embeddings=[1.0, 0.0, 0.0], metadatas={"src": "X"})
            ns_y.add(ids="shared", embeddings=[0.0, 1.0, 0.0], metadatas={"src": "Y"})

            assert_get_present(ns_x, "shared", metadatas={"src": "X"})
            assert_get_present(ns_y, "shared", metadatas={"src": "Y"})
        finally:
            cleanup(db_client, collection)

    def test_update_does_not_affect_other_ns(self, db_client):
        """Test update does not affect other ns."""
        collection = create_ns_collection(db_client, suffix="_mns_upd")
        ns_a = collection.create_namespace("ns_a")
        ns_b = collection.create_namespace("ns_b")
        ns_a.prewarm()
        ns_b.prewarm()
        doc_id = "shared_upd"
        try:
            ns_a.add(ids=doc_id, embeddings=[1.0, 0.0, 0.0], documents="A original")
            ns_b.add(ids=doc_id, embeddings=[0.0, 1.0, 0.0], documents="B original")
            assert_get_present(ns_a, doc_id, documents="A original")
            assert_get_present(ns_b, doc_id, documents="B original")

            ns_a.update(ids=doc_id, embeddings=[0.5, 0.5, 0.5], documents="A updated", metadatas={"updated": True})

            assert_get_present(ns_a, doc_id, documents="A updated", metadatas={"updated": True})
            assert_get_present(ns_b, doc_id, documents="B original")
        finally:
            cleanup(db_client, collection)

    def test_delete_only_target_ns(self, db_client):
        """Test delete only target ns."""
        collection = create_ns_collection(db_client, suffix="_mns_del")
        ns_a = collection.create_namespace("ns_a")
        ns_b = collection.create_namespace("ns_b")
        ns_a.prewarm()
        ns_b.prewarm()
        doc_id = "shared_del"
        try:
            ns_a.add(ids=doc_id, embeddings=[1.0, 0.0, 0.0])
            ns_b.add(ids=doc_id, embeddings=[0.0, 1.0, 0.0])
            assert_get_present(ns_a, doc_id)
            assert_get_present(ns_b, doc_id)

            ns_a.delete(ids=doc_id)

            assert_get_absent(ns_a, doc_id)
            assert_get_present(ns_b, doc_id)
        finally:
            cleanup(db_client, collection)

    def test_upsert_isolation(self, db_client):
        """Test upsert isolation."""
        collection = create_ns_collection(db_client, suffix="_mns_ups")
        ns_a = collection.create_namespace("ns_a")
        ns_b = collection.create_namespace("ns_b")
        ns_a.prewarm()
        ns_b.prewarm()
        try:
            ns_a.add(ids="a_only", embeddings=[1.0, 0.0, 0.0], metadatas={"ns": "a"})
            ns_b.upsert(ids="b_only", embeddings=[0.0, 1.0, 0.0], metadatas={"ns": "b"})

            assert_get_present(ns_a, "a_only", metadatas={"ns": "a"})
            assert_get_absent(ns_a, "b_only")
            assert_get_present(ns_b, "b_only", metadatas={"ns": "b"})
            assert_get_absent(ns_b, "a_only")
        finally:
            cleanup(db_client, collection)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
