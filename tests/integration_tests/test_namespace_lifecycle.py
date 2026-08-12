"""
Namespace lifecycle integration tests.
Tests collection creation with use_namespace=True, namespace CRUD, and collection deletion.
"""

import contextlib
import time

import pytest
from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT

import pyseekdb
from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import VectorIndexConfig
from pyseekdb.client.schema import Schema


class TestNamespaceLifecycle:
    """TestNamespaceLifecycle class."""

    def _create_ns_collection(self, client, suffix=""):
        """Create ns collection."""
        name = f"test_ns_lc_{int(time.time() * 1000)}{suffix}"
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
        return collection

    def test_create_namespace_collection(self, db_client):
        """Test create namespace collection."""
        collection = self._create_ns_collection(db_client)
        try:
            assert collection.use_namespace is True
            assert collection.dimension == 3
            assert collection.name.startswith("test_ns_lc_")
        finally:
            db_client.delete_collection(name=collection.name)

    def test_get_collection_preserves_namespace_flag(self, db_client):
        """Test get collection preserves namespace flag."""
        collection = self._create_ns_collection(db_client)
        try:
            retrieved = db_client.get_collection(collection.name)
            assert retrieved.use_namespace is True
            assert retrieved.dimension == collection.dimension
        finally:
            db_client.delete_collection(name=collection.name)

    def test_get_collection_restores_embedding_function(self, db_client):
        """Test get collection restores embedding function."""
        from pyseekdb import DefaultEmbeddingFunction

        ef = DefaultEmbeddingFunction()
        name = f"test_ns_ef_{int(time.time() * 1000)}"
        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=IVFConfiguration(dimension=ef.dimension, distance="cosine", centroids_fresh_mode="spfresh"),
                embedding_function=ef,
            ),
        )
        db_client.create_collection(
            name=name,
            schema=schema,
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            reopened = db_client.get_collection(name)
            assert reopened.use_namespace is True
            assert reopened.embedding_function is not None
            assert reopened.embedding_function.name() == "default"
        finally:
            db_client.delete_collection(name=name)

    def test_create_and_get_namespace(self, db_client):
        """Test create and get namespace."""
        collection = self._create_ns_collection(db_client)
        try:
            ns = collection.create_namespace("ns_a")
            assert ns.name == "ns_a"
            assert ns.namespace_id is not None

            ns2 = collection.get_namespace("ns_a")
            assert ns2.name == "ns_a"
            assert ns2.namespace_id == ns.namespace_id
        finally:
            db_client.delete_collection(name=collection.name)

    def test_get_or_create_namespace(self, db_client):
        """Test get or create namespace."""
        collection = self._create_ns_collection(db_client)
        try:
            ns1 = collection.get_or_create_namespace("ns_goc")
            assert ns1.name == "ns_goc"

            ns2 = collection.get_or_create_namespace("ns_goc")
            assert ns2.namespace_id == ns1.namespace_id
        finally:
            db_client.delete_collection(name=collection.name)

    def test_has_namespace(self, db_client):
        """Test has namespace."""
        collection = self._create_ns_collection(db_client)
        try:
            assert collection.has_namespace("nonexistent") is False
            collection.create_namespace("ns_check")
            assert collection.has_namespace("ns_check") is True
        finally:
            db_client.delete_collection(name=collection.name)

    def test_list_namespaces(self, db_client):
        """Test list namespaces."""
        collection = self._create_ns_collection(db_client)
        try:
            collection.create_namespace("ns_x")
            collection.create_namespace("ns_y")
            ns_list = collection.list_namespaces()
            names = {ns.name for ns in ns_list}
            assert "ns_x" in names
            assert "ns_y" in names
            assert len(ns_list) >= 2
        finally:
            db_client.delete_collection(name=collection.name)

    def test_delete_namespace(self, db_client):
        """Test delete namespace."""
        collection = self._create_ns_collection(db_client)
        try:
            collection.create_namespace("ns_del")
            assert collection.has_namespace("ns_del") is True

            collection.delete_namespace("ns_del")
            assert collection.has_namespace("ns_del") is False
        finally:
            db_client.delete_collection(name=collection.name)

    def test_get_nonexistent_namespace_raises(self, db_client):
        """Test get nonexistent namespace raises."""
        collection = self._create_ns_collection(db_client)
        try:
            with pytest.raises(ValueError):
                collection.get_namespace("does_not_exist")
        finally:
            db_client.delete_collection(name=collection.name)

    def test_create_duplicate_namespace_raises_friendly_error(self, db_client):
        """Duplicate namespace creation should surface a clear SDK error."""
        collection = self._create_ns_collection(db_client)
        try:
            collection.create_namespace("dup_ns")
            with pytest.raises(ValueError, match="already exists"):
                collection.create_namespace("dup_ns")
        finally:
            db_client.delete_collection(name=collection.name)

    def test_prewarm_after_namespace_deleted_raises_friendly_error(self, db_client):
        """Prewarm on a deleted namespace should not leak raw kernel error codes."""
        collection = self._create_ns_collection(db_client)
        ns = collection.create_namespace("prewarm_del")
        try:
            collection.delete_namespace("prewarm_del")
            with pytest.raises(ValueError, match=r"no longer exists|being dropped"):
                ns.prewarm()
        finally:
            db_client.delete_collection(name=collection.name)

    def test_delete_collection_cleans_namespaces(self, db_client):
        """Test delete collection cleans namespaces."""
        collection = self._create_ns_collection(db_client)
        coll_name = collection.name
        collection.create_namespace("ns_cleanup")
        db_client.delete_collection(name=coll_name)

        assert db_client.has_collection(coll_name) is False

    def test_create_resumes_incomplete_collection(self, oceanbase_client):
        """A namespace collection whose physical tables were partially lost (e.g. a
        crash mid-create) is finished by re-running create_collection, instead of
        being blocked by 'already exists'. A complete collection still errors."""
        from pyseekdb.client.meta_info import NamespaceCollectionNames

        client = oceanbase_client
        srv = client._server
        name = f"test_ns_resume_{int(time.time() * 1000)}"
        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=IVFConfiguration(dimension=3, distance="cosine", centroids_fresh_mode="spfresh"),
                embedding_function=None,
            ),
        )
        collection = client.create_collection(name=name, schema=schema, use_namespace=True, partition_count=4)
        try:
            cid = collection.id
            # Simulate an interrupted create: drop one physical table.
            srv._use_catalog_database()
            srv._execute(f"DROP TABLE IF EXISTS `{NamespaceCollectionNames.kv_data_table_name(cid)}`")
            assert srv._is_incomplete_ns_collection(name) is True

            # Re-create resumes (same id, rebuilds the missing table) instead of raising.
            resumed = client.create_collection(name=name, schema=schema, use_namespace=True, partition_count=4)
            assert resumed.id == cid
            assert srv._is_incomplete_ns_collection(name) is False

            # Data path works after resume.
            ns = resumed.create_namespace("ns_x")
            ns.add(ids="d1", embeddings=[1.0, 2.0, 3.0])
            assert ns.count() == 1

            # A complete collection still rejects a duplicate create.
            with pytest.raises(ValueError, match="already exists"):
                client.create_collection(name=name, schema=schema, use_namespace=True, partition_count=4)
        finally:
            client.delete_collection(name=name)

    def test_get_collection_purges_broken_ns_collection(self, oceanbase_client):
        """get_collection on a namespace collection with missing physical tables
        should treat it as non-existent and purge catalog leftovers."""
        from pyseekdb.client.meta_info import NamespaceCollectionNames

        client = oceanbase_client
        srv = client._server
        name = f"test_ns_broken_get_{int(time.time() * 1000)}"
        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=IVFConfiguration(dimension=3, distance="cosine", centroids_fresh_mode="spfresh"),
                embedding_function=None,
            ),
        )
        collection = client.create_collection(name=name, schema=schema, use_namespace=True, partition_count=4)
        cid = collection.id
        try:
            srv._use_catalog_database()
            srv._execute(f"DROP TABLE IF EXISTS `{NamespaceCollectionNames.kv_data_table_name(cid)}`")
            assert srv._is_incomplete_ns_collection(name) is True

            with pytest.raises(ValueError, match="does not exist"):
                client.get_collection(name)

            assert client.has_collection(name) is False
            assert srv._get_ns_collection_meta(name) is None
            assert not srv._table_exists(NamespaceCollectionNames.data_table_name(cid))
            assert not srv._table_exists(NamespaceCollectionNames.kv_data_table_name(cid))
            assert not srv._table_exists(NamespaceCollectionNames.logic_schema_table_name(cid))
            assert not srv._tablegroup_exists(NamespaceCollectionNames.tablegroup_name(cid))
        finally:
            with contextlib.suppress(Exception):
                client.delete_collection(name=name)

    def test_namespace_ops_blocked_after_collection_deleted(self, db_client):
        """Test namespace ops blocked after collection deleted."""
        collection = self._create_ns_collection(db_client)
        db_client.delete_collection(name=collection.name)

        with pytest.raises(ValueError, match="no longer exists"):
            collection.create_namespace("demo")
        with pytest.raises(ValueError, match="no longer exists"):
            collection.get_or_create_namespace("demo")
        with pytest.raises(ValueError, match="no longer exists"):
            collection.list_namespaces()
        with pytest.raises(ValueError, match="no longer exists"):
            collection.has_namespace("demo")

    def test_collection_data_api_blocked_when_namespace_enabled(self, db_client):
        """Test collection data api blocked when namespace enabled."""
        collection = self._create_ns_collection(db_client)
        try:
            with pytest.raises(ValueError, match="namespace enabled"):
                collection.add(ids="1", embeddings=[1.0, 2.0, 3.0])
        finally:
            db_client.delete_collection(name=collection.name)

    def test_namespace_on_non_namespace_collection_raises(self, db_client):
        """Test namespace on non namespace collection raises."""
        name = f"test_nons_{int(time.time() * 1000)}"
        collection = db_client.create_collection(
            name=name,
            configuration=pyseekdb.HNSWConfiguration(dimension=3),
            embedding_function=None,
        )
        try:
            with pytest.raises(ValueError, match="not enabled"):
                collection.create_namespace("ns1")
        finally:
            try:
                db_client.delete_collection(name=name)
            except (ValueError, RuntimeError):
                db_client._server._execute(f"DROP TABLE IF EXISTS `{name}`")
                db_client._server._execute(f"DELETE FROM `sdk_collections` WHERE COLLECTION_NAME = '{name}'")

    def test_create_namespace_collection_with_hnsw_raises(self, db_client):
        """Test create namespace collection with hnsw raises."""
        from pyseekdb.client.configuration import HNSWConfiguration

        name = f"test_ns_hnsw_{int(time.time() * 1000)}"
        schema = Schema(
            vector_index=VectorIndexConfig(
                hnsw=HNSWConfiguration(dimension=3),
                embedding_function=None,
            ),
        )
        with pytest.raises(ValueError, match="does not support HNSW"):
            db_client.create_collection(name=name, schema=schema, use_namespace=True)

    def test_ss_mode_creates_hot_table(self, oceanbase_client):
        """Verify hot_table is created when _is_shared_storage_mode returns True (SS mode)."""
        import json
        from unittest.mock import patch

        from pyseekdb.client.meta_info import NamespaceCollectionNames

        client = oceanbase_client

        with patch.object(type(client._server), "_is_shared_storage_mode", return_value=True):
            name = f"test_ns_ss_{int(time.time() * 1000)}"
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

        try:
            collection_id = collection.id

            # Verify settings has storage_mode=ss
            rows = client._server._execute(
                f"SELECT settings FROM sdk_collections WHERE collection_id = '{collection_id}'"
            )
            settings = json.loads(rows[0]["settings"])
            assert settings["storage_mode"] == "ss", f"Expected ss, got {settings.get('storage_mode')}"

            # Verify hot_table was actually created
            hot_table = NamespaceCollectionNames.hot_table_name(collection_id)
            rows = client._server._execute(f"DESCRIBE `{hot_table}`")
            assert rows is not None and len(rows) > 0, "hot_table should exist in SS mode"

            col_names = {r["Field"] for r in rows}
            assert "namespace_id" in col_names
            assert "last_access_time" in col_names
        finally:
            client.delete_collection(name=name)

    def test_custom_collection_partition_count(self, oceanbase_client):
        """Verify create_collection(partition_count=...) controls the PARTITIONS clause
        and is exposed via collection.partition_count."""
        import re

        from pyseekdb.client.meta_info import NamespaceCollectionNames

        name = f"test_ns_lc_pc_{int(time.time() * 1000)}"
        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=IVFConfiguration(dimension=3, distance="cosine", centroids_fresh_mode="spfresh"),
                embedding_function=None,
            ),
        )
        collection = oceanbase_client.create_collection(name=name, schema=schema, use_namespace=True, partition_count=4)
        try:
            assert collection.partition_count == 4
            # Reopened handle restores partition_count from settings.
            assert oceanbase_client.get_collection(name).partition_count == 4

            data_table = NamespaceCollectionNames.data_table_name(collection.id)
            rows = oceanbase_client._server._execute(f"SHOW CREATE TABLE `{data_table}`")
            create_sql = rows[0].get("Create Table", "") if rows else ""
            partitions = re.findall(r"partition `p\d+`", create_sql)
            assert len(partitions) == 4, f"Expected 4 partitions, found {len(partitions)}: {create_sql[-300:]}"
        finally:
            oceanbase_client.delete_collection(name=collection.name)

    def test_partition_count_rejected_for_non_namespace(self, db_client):
        """Test partition count rejected for non namespace."""
        name = f"test_nons_pc_{int(time.time() * 1000)}"
        with pytest.raises(ValueError, match="partition_count is only supported"):
            db_client.create_collection(
                name=name,
                configuration=pyseekdb.HNSWConfiguration(dimension=3),
                embedding_function=None,
                partition_count=4,
            )
        assert not db_client.has_collection(name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
