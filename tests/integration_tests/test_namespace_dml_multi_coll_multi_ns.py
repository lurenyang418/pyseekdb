"""
Namespace DML integration tests: multiple collections x multiple namespaces.
Verifies cross-quadrant isolation via get.
"""

import pytest
from namespace_dml_helpers import (
    assert_get_absent,
    assert_get_present,
    cleanup,
    create_ns_collection,
)


class TestNamespaceDMLMultiCollMultiNs:
    """TestNamespaceDMLMultiCollMultiNs class."""

    def _setup_quadrants(self, db_client):
        """Setup quadrants."""
        coll_1 = create_ns_collection(db_client, suffix="_mcmn_c1")
        coll_2 = create_ns_collection(db_client, suffix="_mcmn_c2")
        quadrants = {
            "coll_1": coll_1,
            "coll_2": coll_2,
            "c1_x": coll_1.create_namespace("ns_x"),
            "c1_y": coll_1.create_namespace("ns_y"),
            "c2_x": coll_2.create_namespace("ns_x"),
            "c2_y": coll_2.create_namespace("ns_y"),
        }
        for key in ("c1_x", "c1_y", "c2_x", "c2_y"):
            quadrants[key].prewarm()
        return quadrants

    def test_cross_quadrant_isolation(self, db_client):
        """Test cross quadrant isolation."""
        q = self._setup_quadrants(db_client)
        marker_id = "marker_id"
        try:
            q["c1_x"].add(
                ids=marker_id,
                embeddings=[1.0, 0.0, 0.0],
                metadatas={"cell": "c1_x"},
            )
            assert_get_present(q["c1_x"], marker_id, metadatas={"cell": "c1_x"})
            assert_get_absent(q["c1_y"], marker_id)
            assert_get_absent(q["c2_x"], marker_id)
            assert_get_absent(q["c2_y"], marker_id)

            q["c2_y"].add(
                ids=marker_id,
                embeddings=[0.0, 1.0, 0.0],
                metadatas={"cell": "c2_y"},
            )
            assert_get_present(q["c1_x"], marker_id, metadatas={"cell": "c1_x"})
            assert_get_present(q["c2_y"], marker_id, metadatas={"cell": "c2_y"})
            assert_get_absent(q["c1_y"], marker_id)
            assert_get_absent(q["c2_x"], marker_id)
        finally:
            cleanup(db_client, q["coll_1"], q["coll_2"])

    def test_delete_in_one_quadrant(self, db_client):
        """Test delete in one quadrant."""
        q = self._setup_quadrants(db_client)
        doc_id = "shared_quad_del"
        try:
            for ns in (q["c1_x"], q["c1_y"], q["c2_x"], q["c2_y"]):
                ns.add(ids=doc_id, embeddings=[1.0, 2.0, 3.0], metadatas={"present": True})

            for ns in (q["c1_x"], q["c1_y"], q["c2_x"], q["c2_y"]):
                assert_get_present(ns, doc_id, metadatas={"present": True})

            q["c1_x"].delete(ids=doc_id)

            assert_get_absent(q["c1_x"], doc_id)
            for ns in (q["c1_y"], q["c2_x"], q["c2_y"]):
                result = ns.get(ids=doc_id, include=["metadatas"])
                assert doc_id in result["ids"], f"expected {doc_id!r} in {ns.name}, got {result['ids']}"
                idx = result["ids"].index(doc_id)
                assert result["metadatas"][idx]["present"] is True
        finally:
            cleanup(db_client, q["coll_1"], q["coll_2"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
