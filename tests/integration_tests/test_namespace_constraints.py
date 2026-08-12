"""
Namespace constraint integration tests.

Covers two SDK-level constraints for use_namespace=True collections:
1. Minimum LakeBase version: namespace-enabled collections require LakeBase (OceanBase Database AI) >= 4.6.1.
2. Vector index type: only IVF_FLAT is currently supported; ivf_sq8 / ivf_pq are
   rejected at the SDK layer before any DDL is issued.
"""

import time

import pytest
from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT

from pyseekdb import IVFConfiguration
from pyseekdb.client.client_base import NAMESPACE_MIN_LAKEBASE_VERSION, NAMESPACE_MIN_OB_VERSION
from pyseekdb.client.configuration import VectorIndexConfig
from pyseekdb.client.schema import Schema
from pyseekdb.client.version import Version


def _make_schema(ivf_type: str) -> Schema:
    """Make schema."""
    return Schema(
        vector_index=VectorIndexConfig(
            ivf=IVFConfiguration(dimension=3, distance="cosine", type=ivf_type, centroids_fresh_mode="spfresh"),
            embedding_function=None,
        ),
    )


def _unique_name(suffix: str) -> str:
    """Unique name."""
    return f"test_ns_constraint_{int(time.time() * 1000)}{suffix}"


class TestNamespaceIvfTypeConstraint:
    """TestNamespaceIvfTypeConstraint class."""

    @pytest.mark.parametrize("ivf_type", ["ivf_pq", "ivf_sq8"])
    def test_non_ivf_flat_rejected(self, oceanbase_client, ivf_type):
        """Non-ivf_flat IVF types must be rejected at the SDK layer."""
        name = _unique_name(f"_{ivf_type}")
        with pytest.raises(ValueError, match="only supports IVF index type 'ivf_flat'"):
            oceanbase_client.create_collection(name=name, schema=_make_schema(ivf_type), use_namespace=True)
        # Rejection happens before any DDL: no collection metadata is left behind.
        assert not oceanbase_client.has_collection(name)

    def test_ivf_flat_accepted(self, oceanbase_client):
        """Positive control: ivf_flat namespace collection creation succeeds."""
        name = _unique_name("_ivf_flat")
        collection = oceanbase_client.create_collection(
            name=name,
            schema=_make_schema("ivf_flat"),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            assert collection.use_namespace is True
            assert collection.dimension == 3
        finally:
            oceanbase_client.delete_collection(name=name)


class TestNamespaceIvfDimensionConstraint:
    """Namespace IVF dimension must stay within SDK max (logic_data_table LOB in-row threshold)."""

    def test_dimension_2049_accepted(self, oceanbase_client):
        """Dimensions above default LOB threshold succeed with logic_data_table DDL fix."""
        name = _unique_name("_dim_2049")
        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=IVFConfiguration(dimension=2049, distance="cosine", centroids_fresh_mode="spfresh"),
                embedding_function=None,
            ),
        )
        collection = oceanbase_client.create_collection(
            name=name,
            schema=schema,
            use_namespace=True,
            partition_count=1,
        )
        try:
            assert collection.dimension == 2049
        finally:
            oceanbase_client.delete_collection(name=name)

    def test_dimension_4096_accepted(self, oceanbase_client):
        """Positive control: max supported IVF dimension succeeds."""
        name = _unique_name("_dim_4096")
        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=IVFConfiguration(dimension=4096, distance="cosine", centroids_fresh_mode="spfresh"),
                embedding_function=None,
            ),
        )
        collection = oceanbase_client.create_collection(
            name=name,
            schema=schema,
            use_namespace=True,
            partition_count=1,
        )
        try:
            assert collection.dimension == 4096
        finally:
            oceanbase_client.delete_collection(name=name)

    def test_dimension_4097_rejected_at_sdk(self, oceanbase_client):
        """Dimensions above 4096 are rejected before OB DDL."""
        name = _unique_name("_dim_4097")
        with pytest.raises(ValueError, match="between 1 and 4096"):
            oceanbase_client.create_collection(
                name=name,
                schema=Schema(
                    vector_index=VectorIndexConfig(
                        ivf=IVFConfiguration(
                            dimension=4097,
                            distance="cosine",
                            centroids_fresh_mode="spfresh",
                        ),
                        embedding_function=None,
                    ),
                ),
                use_namespace=True,
            )
        assert not oceanbase_client.has_collection(name)


class TestNamespaceMinVersionConstraint:
    """TestNamespaceMinVersionConstraint class."""

    def test_connected_lakebase_meets_min_version(self, oceanbase_client):
        """The kernel under test must be LakeBase >= 4.6.1, and creation succeeds."""
        db_type, version = oceanbase_client._server.detect_db_type_and_version()
        assert db_type.lower() == "oceanbase"
        assert oceanbase_client._server._is_lakebase_cluster() is True
        assert version >= NAMESPACE_MIN_LAKEBASE_VERSION, (
            f"connected LakeBase version {version} < required {NAMESPACE_MIN_LAKEBASE_VERSION}"
        )

        name = _unique_name("_ver_ok")
        collection = oceanbase_client.create_collection(
            name=name,
            schema=_make_schema("ivf_flat"),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            assert collection.use_namespace is True
        finally:
            oceanbase_client.delete_collection(name=name)

    def test_standard_oceanbase_rejected(self, oceanbase_client, monkeypatch):
        """Standard OceanBase (without Database AI) must be rejected even when version is new enough."""
        server = oceanbase_client._server
        monkeypatch.setattr(type(server), "_is_lakebase_cluster", lambda self: False)
        name = _unique_name("_not_lakebase")
        with pytest.raises(ValueError, match="only supported on LakeBase"):
            oceanbase_client.create_collection(name=name, schema=_make_schema("ivf_flat"), use_namespace=True)
        assert not oceanbase_client.has_collection(name)

    def test_old_lakebase_version_rejected(self, oceanbase_client, monkeypatch):
        """Simulate an older LakeBase kernel: namespace creation must fail with a clear error."""
        server = oceanbase_client._server
        monkeypatch.setattr(type(server), "_is_lakebase_cluster", lambda self: True)
        monkeypatch.setattr(
            type(server),
            "detect_db_type_and_version",
            lambda self: ("oceanbase", Version("4.6.0.0")),
        )
        name = _unique_name("_ver_old")
        with pytest.raises(ValueError, match=r"requires LakeBase version >= 4\.6\.1"):
            oceanbase_client.create_collection(name=name, schema=_make_schema("ivf_flat"), use_namespace=True)
        # restore before checking leftovers
        monkeypatch.undo()
        assert not oceanbase_client.has_collection(name)

    def test_min_version_constant(self):
        """Test min version constant."""
        assert Version("4.6.1.0") == NAMESPACE_MIN_LAKEBASE_VERSION == NAMESPACE_MIN_OB_VERSION
