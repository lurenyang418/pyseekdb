"""
Unit tests for concurrent-safe get_or_create_namespace helpers.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from pyseekdb.client.client_base import (  # noqa: E402
    BaseClient,
    _is_namespace_catalog_conflict_error,
)


class TestNamespaceCatalogConflictDetection:
    """TestNamespaceCatalogConflictDetection class."""

    def test_detects_integrity_error_on_sdk_namespaces(self):
        """Test detects integrity error on sdk namespaces."""

        class IntegrityError(Exception):
            """IntegrityError class."""

            pass

        exc = IntegrityError("(1062, \"Duplicate entry 'cid-ns_shared' for key 'uk_sdk_ns_coll_name'\")")
        assert _is_namespace_catalog_conflict_error(exc)

    def test_detects_integrity_error_on_sdk_ltables(self):
        """Test detects integrity error on sdk ltables."""

        class IntegrityError(Exception):
            """IntegrityError class."""

            pass

        exc = IntegrityError("(1062, \"Duplicate entry 'cid-1-default' for key 'uk_sdk_lt_coll_ns_name'\")")
        assert _is_namespace_catalog_conflict_error(exc)

    def test_ignores_unrelated_errors(self):
        """Test ignores unrelated errors."""
        assert not _is_namespace_catalog_conflict_error(ValueError("invalid namespace name"))


class TestGetOrCreateNamespaceMetaRecovery:
    """TestGetOrCreateNamespaceMetaRecovery class."""

    def test_idempotent_insert_reuses_existing_namespace(self):
        """Test idempotent insert reuses existing namespace."""
        client = MagicMock(spec=BaseClient)
        client._get_ns_namespace_meta.return_value = None
        client._insert_ns_namespace_catalog_row.return_value = 42
        client._insert_ns_ltable_catalog_row.return_value = 7
        client._finalize_ns_namespace_meta.return_value = {
            "namespace_id": "42",
            "namespace_name": "ns_shared",
            "ltable_id": "7",
        }

        result = BaseClient._get_or_create_ns_namespace_meta(client, "cid", "ns_shared")

        assert result["namespace_id"] == "42"
        client._insert_ns_namespace_catalog_row.assert_called_once_with("cid", "ns_shared", idempotent=True)
        client._insert_ns_ltable_catalog_row.assert_called_once_with("cid", 42, idempotent=True)

    def test_existing_namespace_without_ltable_creates_default_ltable(self):
        """Test existing namespace without ltable creates default ltable."""
        client = MagicMock(spec=BaseClient)
        client._get_ns_namespace_meta.return_value = {
            "namespace_id": "42",
            "namespace_name": "ns_shared",
        }
        client._insert_ns_ltable_catalog_row.return_value = 7
        client._finalize_ns_namespace_meta.return_value = {
            "namespace_id": "42",
            "namespace_name": "ns_shared",
            "ltable_id": "7",
        }

        result = BaseClient._get_or_create_ns_namespace_meta(client, "cid", "ns_shared")

        assert result["ltable_id"] == "7"
        client._insert_ns_namespace_catalog_row.assert_not_called()
        client._insert_ns_ltable_catalog_row.assert_called_once_with("cid", 42, idempotent=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
