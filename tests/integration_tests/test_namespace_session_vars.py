"""
Integration tests for namespace session variable propagation.
Verifies that collection_id, namespace_id, and ltable_id are set
as OceanBase session user variables (@collection_id, @namespace_id, @ltable_id)
during namespace collection and namespace creation/retrieval.
"""

import time

import pytest
from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT

from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import VectorIndexConfig
from pyseekdb.client.schema import Schema


class TestNamespaceSessionVars:
    """TestNamespaceSessionVars class."""

    def _create_ns_collection(self, client):
        """Create ns collection."""
        name = f"test_ns_sessvar_{int(time.time() * 1000)}"
        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=IVFConfiguration(dimension=3, distance="l2", centroids_fresh_mode="spfresh"),
                embedding_function=None,
            ),
        )
        return client.create_collection(
            name=name,
            schema=schema,
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )

    def _query_session_vars(self, client):
        """Query session vars."""
        rows = client._server._execute("SELECT @collection_id AS cid, @namespace_id AS nsid, @ltable_id AS ltid")
        row = rows[0]
        if isinstance(row, (list, tuple)):
            return {"cid": row[0], "nsid": row[1], "ltid": row[2]}
        return {"cid": row.get("cid"), "nsid": row.get("nsid"), "ltid": row.get("ltid")}

    def test_session_collection_id_after_create(self, db_client):
        """Test session collection id after create."""
        collection = self._create_ns_collection(db_client)
        try:
            vars_ = self._query_session_vars(db_client)
            assert vars_["cid"] is not None, "@collection_id should be set after create_collection"
            assert str(vars_["cid"]) == str(collection.id)
        finally:
            db_client.delete_collection(name=collection.name)

    def test_session_ns_ids_after_create_namespace(self, db_client):
        """Test session ns ids after create namespace."""
        collection = self._create_ns_collection(db_client)
        try:
            ns = collection.create_namespace("sess_ns")
            vars_ = self._query_session_vars(db_client)
            assert vars_["nsid"] is not None, "@namespace_id should be set after create_namespace"
            assert int(vars_["nsid"]) == int(ns.namespace_id)
            assert vars_["ltid"] is not None, "@ltable_id should be set after create_namespace"
            assert int(vars_["ltid"]) > 0, "@ltable_id should be a positive integer"
        finally:
            db_client.delete_collection(name=collection.name)

    def test_session_ns_id_after_get_namespace(self, db_client):
        """Test session ns id after get namespace."""
        collection = self._create_ns_collection(db_client)
        try:
            ns1 = collection.create_namespace("sess_get_ns")
            ns1_id = int(ns1.namespace_id)
            ns1_ltid = int(self._query_session_vars(db_client)["ltid"])

            ns2 = collection.create_namespace("sess_get_ns2")
            ns2_id = int(ns2.namespace_id)
            ns2_ltid = int(self._query_session_vars(db_client)["ltid"])

            # After creating ns2, session should have ns2's id and its own @ltable_id
            vars_ = self._query_session_vars(db_client)
            assert int(vars_["nsid"]) == ns2_id
            assert int(vars_["ltid"]) == ns2_ltid

            # Now get_namespace(ns1) should update session to ns1's id AND ns1's @ltable_id;
            # otherwise kernel-side ops would observe ns2's stale @ltable_id while ns1 is in use.
            ns1_again = collection.get_namespace("sess_get_ns")
            vars_ = self._query_session_vars(db_client)
            assert int(vars_["nsid"]) == ns1_id
            assert int(vars_["ltid"]) == ns1_ltid
            assert int(ns1_again.namespace_id) == ns1_id
        finally:
            db_client.delete_collection(name=collection.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
