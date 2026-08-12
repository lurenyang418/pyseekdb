"""
Integration tests for namespace public API compatibility (create / get_or_create).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

import pyseekdb
from pyseekdb.client.configuration import HNSWConfiguration, SparseVectorIndexConfig, VectorIndexConfig
from pyseekdb.client.schema import Schema
from tests.integration_tests.namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT, ns_schema


class TestNamespacePublicAPIIntegration:
    """End-to-end checks for namespace create/get_or_create guardrails."""

    def test_create_without_schema_raises_clear_error(self, oceanbase_client):
        """Omitting schema must fail before building default HNSW."""
        name = f"test_ns_no_schema_{int(time.time() * 1000)}"
        with pytest.raises(ValueError, match="requires an explicit Schema with an IVF vector index"):
            oceanbase_client.create_collection(name=name, use_namespace=True)

    def test_create_with_sparse_vector_raises(self, oceanbase_client):
        """Sparse vector schema is rejected for namespace collections."""
        name = f"test_ns_sparse_{int(time.time() * 1000)}"
        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=pyseekdb.IVFConfiguration(dimension=3, distance="l2", centroids_fresh_mode="spfresh"),
                embedding_function=None,
            ),
            sparse_vector_index=SparseVectorIndexConfig(embedding_function=MagicMock()),
        )
        with pytest.raises(ValueError, match="does not support SparseVectorIndexConfig"):
            oceanbase_client.create_collection(
                name=name,
                schema=schema,
                use_namespace=True,
                partition_count=NAMESPACE_TEST_PARTITION_COUNT,
            )

    def test_get_or_create_mode_mismatch(self, oceanbase_client):
        """Standard and namespace collections with the same name are incompatible."""
        name = f"test_ns_mode_{int(time.time() * 1000)}"
        oceanbase_client.create_collection(
            name=name,
            configuration=HNSWConfiguration(dimension=3),
            embedding_function=None,
        )
        try:
            with pytest.raises(ValueError, match="already exists as a standard collection"):
                oceanbase_client.get_or_create_collection(name=name, schema=ns_schema(), use_namespace=True)
        finally:
            oceanbase_client.delete_collection(name=name)

    def test_get_or_create_reuses_matching_namespace_collection(self, oceanbase_client):
        """get_or_create with matching namespace mode returns the same handle."""
        name = f"test_ns_goc_{int(time.time() * 1000)}"
        created = oceanbase_client.create_collection(
            name=name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        try:
            reused = oceanbase_client.get_or_create_collection(
                name=name,
                schema=ns_schema(),
                use_namespace=True,
                partition_count=NAMESPACE_TEST_PARTITION_COUNT,
            )
            assert reused.id == created.id
            assert reused.use_namespace is True
        finally:
            oceanbase_client.delete_collection(name=name)
