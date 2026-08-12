"""
Namespace prewarm integration tests.

- embedded mode: prewarm raises ValueError
- oceanbase SS mode: prewarm inserts hot_table record, repeat prewarm updates last_access_time
"""

import time

import pytest

from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import VectorIndexConfig
from pyseekdb.client.meta_info import NamespaceCollectionNames
from pyseekdb.client.schema import Schema


class TestNamespacePrewarm:
    """TestNamespacePrewarm class."""

    def _create_ns_collection_and_namespace(self, client):
        """Create ns collection and namespace."""
        from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT

        name = f"test_ns_pw_{int(time.time() * 1000)}"
        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=IVFConfiguration(dimension=3, distance="cosine", centroids_fresh_mode="spfresh"),
                embedding_function=None,
            ),
        )
        collection = client.create_collection(
            name=name,
            schema=schema,
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        namespace = collection.create_namespace("pw_ns")
        return collection, namespace

    def _query_hot_table(self, client, collection, namespace_id):
        """Query hot table."""
        coll_id = collection.id
        hot_table = NamespaceCollectionNames.hot_table_name(coll_id)
        rows = client._server._execute(
            f"SELECT namespace_id, last_access_time FROM `{hot_table}` WHERE namespace_id = {namespace_id}"
        )
        return rows

    def test_prewarm_inserts_hot_table_record(self, oceanbase_client):
        """First prewarm should insert a record into hot_table."""
        collection, namespace = self._create_ns_collection_and_namespace(oceanbase_client)
        try:
            rows_before = self._query_hot_table(oceanbase_client, collection, namespace._namespace_id)
            assert len(rows_before) == 0, "hot_table should be empty before prewarm"

            namespace.prewarm()

            rows_after = self._query_hot_table(oceanbase_client, collection, namespace._namespace_id)
            assert len(rows_after) == 1, "hot_table should have exactly one record after prewarm"
        finally:
            oceanbase_client.delete_collection(name=collection.name)

    def test_prewarm_repeat_updates_access_time(self, oceanbase_client):
        """Second prewarm should update last_access_time (not insert a new row)."""
        collection, namespace = self._create_ns_collection_and_namespace(oceanbase_client)
        try:
            namespace.prewarm()
            rows1 = self._query_hot_table(oceanbase_client, collection, namespace._namespace_id)
            assert len(rows1) == 1
            ts1 = rows1[0]["last_access_time"] if isinstance(rows1[0], dict) else rows1[0][1]

            time.sleep(2)

            namespace.prewarm()
            rows2 = self._query_hot_table(oceanbase_client, collection, namespace._namespace_id)
            assert len(rows2) == 1, "repeat prewarm should not create a second row"
            ts2 = rows2[0]["last_access_time"] if isinstance(rows2[0], dict) else rows2[0][1]

            assert ts2 > ts1, f"last_access_time should increase after repeat prewarm: {ts1} -> {ts2}"
        finally:
            oceanbase_client.delete_collection(name=collection.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
