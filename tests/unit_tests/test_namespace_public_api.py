"""
Public API tests for namespace collection creation and get_or_create behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from pyseekdb import IVFConfiguration  # noqa: E402
from pyseekdb.client.client_base import BaseClient  # noqa: E402
from pyseekdb.client.configuration import (  # noqa: E402
    HNSWConfiguration,
    SparseVectorIndexConfig,
    VectorIndexConfig,
)
from pyseekdb.client.schema import Schema  # noqa: E402
from pyseekdb.client.types import _NOT_PROVIDED  # noqa: E402
from tests.unit_tests.test_namespace import FakeClient  # noqa: E402


class TestNamespaceCreateCollectionPublicAPI:
    """create_collection(use_namespace=True) validation via public entry points."""

    @staticmethod
    def _client() -> FakeClient:
        client = FakeClient()
        client.create_collection = BaseClient.create_collection.__get__(client, BaseClient)
        return client

    def test_create_without_schema_raises_clear_error(self):
        """Omitting schema must not fall through to default HNSW."""
        client = self._client()
        with pytest.raises(ValueError, match="requires an explicit Schema with an IVF vector index"):
            client.create_collection(name="ns_coll", use_namespace=True)

    def test_create_with_explicit_hnsw_schema_raises(self):
        """Explicit HNSW schema is rejected with an IVF-oriented message."""
        client = self._client()
        schema = Schema(
            vector_index=VectorIndexConfig(
                hnsw=HNSWConfiguration(dimension=3),
                embedding_function=None,
            ),
        )
        with pytest.raises(ValueError, match="does not support HNSW"):
            client.create_collection(name="ns_coll", schema=schema, use_namespace=True)

    def test_create_with_sparse_vector_schema_raises(self):
        """Sparse vector index in schema is rejected for namespace collections."""
        client = self._client()
        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=IVFConfiguration(dimension=3, centroids_fresh_mode="spfresh"),
                embedding_function=None,
            ),
            sparse_vector_index=SparseVectorIndexConfig(embedding_function=MagicMock()),
        )
        with pytest.raises(ValueError, match="does not support SparseVectorIndexConfig"):
            client.create_collection(name="ns_coll", schema=schema, use_namespace=True)


class TestNamespaceGetOrCreatePublicAPI:
    """get_or_create_collection namespace mode compatibility."""

    @staticmethod
    def _bind_helpers(client):
        client._assert_get_or_create_namespace_mode_matches = (
            BaseClient._assert_get_or_create_namespace_mode_matches.__get__(client, BaseClient)
        )
        client.get_or_create_collection = BaseClient.get_or_create_collection.__get__(client, BaseClient)

    def test_mode_mismatch_standard_then_namespace(self):
        """Existing standard collection cannot be fetched as namespace."""
        client = MagicMock(spec=BaseClient)
        self._bind_helpers(client)
        client.has_collection.return_value = True
        client._is_incomplete_ns_collection.return_value = False
        client._get_ns_collection_meta.return_value = None

        with pytest.raises(ValueError, match="already exists as a standard collection"):
            client.get_or_create_collection("items", use_namespace=True)

        client.get_collection.assert_not_called()

    def test_mode_mismatch_namespace_then_standard(self):
        """Existing namespace collection cannot be fetched as standard."""
        client = MagicMock(spec=BaseClient)
        self._bind_helpers(client)
        client.has_collection.return_value = True
        client._is_incomplete_ns_collection.return_value = False
        client._get_ns_collection_meta.return_value = {
            "collection_id": "1",
            "collection_name": "items",
            "settings": {"use_namespace": True},
        }

        with pytest.raises(ValueError, match="already exists as a namespace-enabled collection"):
            client.get_or_create_collection("items", use_namespace=False)

        client.get_collection.assert_not_called()

    def test_matching_namespace_mode_returns_existing(self):
        """Same namespace mode reuses the existing collection."""
        client = MagicMock(spec=BaseClient)
        self._bind_helpers(client)
        client.has_collection.return_value = True
        client._is_incomplete_ns_collection.return_value = False
        client._get_ns_collection_meta.return_value = {
            "collection_id": "1",
            "collection_name": "items",
            "settings": {"use_namespace": True},
        }
        existing = object()
        client.get_collection.return_value = existing

        result = client.get_or_create_collection("items", use_namespace=True)

        assert result is existing
        client.get_collection.assert_called_once_with("items", embedding_function=_NOT_PROVIDED)
