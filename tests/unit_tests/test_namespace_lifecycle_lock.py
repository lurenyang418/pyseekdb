"""Unit tests for namespace catalog concurrency without GET_LOCK."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from pyseekdb.client.client_base import BaseClient  # noqa: E402


class TestNamespaceLifecycleWithoutGetLock:
    """TestNamespaceLifecycleWithoutGetLock class."""

    def test_has_namespace_queries_catalog_directly(self):
        """Test has namespace queries catalog directly."""
        client = MagicMock(spec=BaseClient)
        client._get_ns_namespace_meta.return_value = None

        assert BaseClient._has_ns_namespace(client, "cid", "ns1") is False
        client._get_ns_namespace_meta.assert_called_once_with("cid", "ns1")

    def test_create_namespace_raises_when_catalog_row_exists(self):
        """Test create namespace raises when catalog row exists."""
        client = MagicMock(spec=BaseClient)
        client._get_ns_namespace_meta.return_value = {
            "namespace_id": "42",
            "namespace_name": "ns1",
            "ltable_id": "7",
        }

        with pytest.raises(ValueError, match="already exists"):
            BaseClient._create_ns_namespace_meta(client, "cid", "ns1")

        client._insert_ns_namespace_catalog_row.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
