"""
Unit tests for concurrent-safe get_or_create_collection helpers.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from pyseekdb.client.client_base import BaseClient  # noqa: E402
from pyseekdb.client.collection_catalog import (  # noqa: E402
    _is_collection_conflict_error,
    _is_sdk_collection_catalog_conflict_error,
)
from pyseekdb.client.types import _NOT_PROVIDED  # noqa: E402


class TestCollectionCatalogConflictDetection:
    """TestCollectionCatalogConflictDetection class."""

    def test_detects_integrity_error_on_sdk_collections(self):
        """Test detects integrity error on sdk collections."""

        class IntegrityError(Exception):
            """IntegrityError class."""

            pass

        exc = IntegrityError("(1062, \"Duplicate entry 'my_coll' for key 'uk_sdk_coll_name'\")")
        assert _is_sdk_collection_catalog_conflict_error(exc)

    def test_ignores_unrelated_errors(self):
        """Test ignores unrelated errors."""
        assert not _is_sdk_collection_catalog_conflict_error(ValueError("invalid dimension"))


class TestCollectionCatalogInsertRecovery:
    """TestCollectionCatalogInsertRecovery class."""

    def test_insert_conflict_reuses_existing_collection_id(self):
        """Test insert conflict reuses existing collection id."""
        client = MagicMock(spec=BaseClient)
        client._get_collection_id.side_effect = [ValueError("not found"), "existing_id"]
        conn = MagicMock()
        client._ensure_connection.return_value = conn

        class IntegrityError(Exception):
            """IntegrityError class."""

            pass

        def execute_side_effect(sql, _params=None):
            """Execute side effect."""
            if "INSERT INTO" in sql:
                raise IntegrityError("(1062, \"Duplicate entry 'items' for key 'uk_sdk_coll_name'\")")
            return []

        client._execute.side_effect = execute_side_effect

        result = BaseClient._create_collection_meta(client, "items", None)

        assert result["collection_id"] == "existing_id"
        conn.rollback.assert_called_once()
        assert client._get_collection_id.call_count == 2

    def test_existing_catalog_row_is_reused_without_insert(self):
        """Test existing catalog row is reused without insert."""
        client = MagicMock(spec=BaseClient)
        client._get_collection_id.return_value = "existing_id"

        result = BaseClient._create_collection_meta(client, "items", None)

        assert result["collection_id"] == "existing_id"
        insert_calls = [call for call in client._execute.call_args_list if "INSERT INTO" in str(call)]
        assert not insert_calls

    def test_sync_create_does_not_reuse_async_creating_catalog_row(self):
        """A synchronous creator must not steal an asynchronous creator's ID."""
        client = MagicMock(spec=BaseClient)
        client._get_collection_id.return_value = "async_id"
        client._resolve_collection_metadata_from_sdk_collections.return_value = MagicMock(
            collection_id="async_id",
            settings='{"state": "creating", "creation_token": "owner-token"}',
        )

        with pytest.raises(ValueError, match="being created by another client"):
            BaseClient._create_collection_meta(
                client,
                "items",
                None,
                lifecycle_state="creating",
                creation_token="sync-token",  # noqa: S106
            )

    def test_sync_collection_fork_uses_lifecycle_state(self):
        """The synchronous facade must publish a fork only after FORK TABLE succeeds."""
        client = MagicMock(spec=BaseClient)
        client._fork_table_enabled.return_value = True
        client.has_collection.return_value = False
        client._get_collection_table_name.return_value = "c$v2$source_id"
        client._get_collection_id.return_value = "forked_id"
        client._resolve_collection_metadata_from_sdk_collections.return_value = MagicMock(
            collection_id="source_id",
            settings='{"version": 2, "dimension": 3, "distance": "l2", "state": "ready"}',
        )
        source = MagicMock(id="source_id", name="source")

        BaseClient._collection_fork(client, source, "forked")

        insert_call = next(call for call in client._execute.call_args_list if "INSERT INTO" in call.args[0])
        settings = json.loads(insert_call.args[1][1])
        assert settings["state"] == "creating"
        assert "creation_token" in settings
        fork_call = next(call for call in client._execute.call_args_list if "FORK TABLE" in call.args[0])
        assert "c$v2$source_id" in fork_call.args[0]
        client._mark_collection_ready.assert_called_once_with("forked", "forked_id", settings["creation_token"])


class TestCollectionConflictDetection:
    """TestCollectionConflictDetection class."""

    def test_detects_value_error_for_existing_collection(self):
        """Test detects value error for existing collection."""
        assert _is_collection_conflict_error(ValueError("Collection 'items' already exists"))

    def test_detects_seekdb_table_exists_error(self):
        """Test detects seekdb table exists error."""

        class SeekdbError(Exception):
            """SeekdbError class."""

            pass

        exc = SeekdbError("Table 'c$v2$abc' already exists failed: code=1050")
        assert _is_collection_conflict_error(exc)

    def test_ignores_unrelated_errors(self):
        """Test ignores unrelated errors."""
        assert not _is_collection_conflict_error(ValueError("invalid dimension"))

    def test_ignores_unrelated_already_exists_message(self):
        """Test does not classify an unrelated already-exists message as a collection conflict."""
        assert not _is_collection_conflict_error(ValueError("metadata table reference already exists"))

    def test_ignores_metadata_failure_without_conflict_cause(self):
        """Test ignores metadata failure without conflict cause."""
        assert not _is_collection_conflict_error(
            ValueError("Failed to create collection metadata: Collection not found: 'items'")
        )

    def test_detects_conflict_in_cause_chain(self):
        """Test detects conflict in cause chain."""
        inner = Exception("Table 'c$v2$abc' already exists failed: code=1050")
        outer = ValueError("Failed to create collection metadata: duplicate entry")
        outer.__cause__ = inner
        assert _is_collection_conflict_error(outer)


class TestGetOrCreateCollectionRecovery:
    """TestGetOrCreateCollectionRecovery class."""

    @staticmethod
    def _bind_resume_helper(client):
        """Bind resume helper."""
        client._get_or_resume_existing_collection = BaseClient._get_or_resume_existing_collection.__get__(
            client, BaseClient
        )

    def test_returns_existing_collection_after_create_conflict(self):
        """Test returns existing collection after create conflict."""
        client = MagicMock(spec=BaseClient)
        self._bind_resume_helper(client)
        client.has_collection.return_value = False
        existing = object()
        client.get_collection.return_value = existing
        client.create_collection.side_effect = ValueError("Collection 'items' already exists")

        result = BaseClient.get_or_create_collection(client, "items")

        assert result is existing
        client.get_collection.assert_called_once_with("items", embedding_function=_NOT_PROVIDED)

    def test_retries_get_after_wrapped_table_conflict(self):
        """Test retries get after wrapped table conflict."""
        client = MagicMock(spec=BaseClient)
        self._bind_resume_helper(client)
        client.has_collection.return_value = False
        existing = object()
        client.get_collection.return_value = existing
        inner = Exception("Table 'c$v2$abc' already exists failed: code=1050")
        outer = ValueError("Failed to create collection metadata: duplicate entry")
        outer.__cause__ = inner
        client.create_collection.side_effect = outer

        result = BaseClient.get_or_create_collection(client, "items")

        assert result is existing
        client.get_collection.assert_called_once_with("items", embedding_function=_NOT_PROVIDED)

    def test_waits_for_collection_owned_by_another_creator(self):
        """A concurrent get_or_create waits instead of treating creating as absent."""
        client = MagicMock(spec=BaseClient)
        self._bind_resume_helper(client)
        existing = object()
        client.has_collection.return_value = False
        client.create_collection.side_effect = ValueError(
            "Collection 'items' is being created by another client; retry after it is ready"
        )
        client.get_collection.return_value = existing

        result = BaseClient.get_or_create_collection(client, "items")

        assert result is existing
        client.get_collection.assert_called_once_with("items", embedding_function=_NOT_PROVIDED)

    def test_resume_helper_waits_when_get_observes_creating_state(self):
        """The recovery path polls when the catalog is still in creating state."""
        client = MagicMock(spec=BaseClient)
        self._bind_resume_helper(client)
        existing = object()
        client.get_collection.side_effect = ValueError("Collection 'items' is still being created")
        client._wait_for_collection_ready = MagicMock(return_value=existing)

        result = BaseClient._get_or_resume_existing_collection(
            client,
            "items",
            schema=None,
            use_namespace=False,
        )

        assert result is existing
        client._wait_for_collection_ready.assert_called_once_with("items", embedding_function=_NOT_PROVIDED)

    def test_conflict_on_namespace_collection_resumes_incomplete_handle(self):
        """Test conflict on namespace collection resumes incomplete handle."""
        client = MagicMock(spec=BaseClient)
        self._bind_resume_helper(client)
        resumed = object()
        client.has_collection.return_value = False
        client._get_ns_collection_meta.return_value = {
            "collection_id": "abc",
            "collection_name": "items",
            "settings": {"use_namespace": True},
        }
        client._is_incomplete_ns_collection.return_value = True
        client.create_collection.side_effect = [
            ValueError("Collection 'items' already exists"),
            resumed,
        ]

        result = BaseClient.get_or_create_collection(client, "items", use_namespace=True)

        assert result is resumed
        assert client.create_collection.call_count == 2
        client.get_collection.assert_not_called()

    def test_conflict_on_namespace_collection_without_metadata_reraises(self):
        """Test conflict on namespace collection without metadata reraises."""
        client = MagicMock(spec=BaseClient)
        self._bind_resume_helper(client)
        client.has_collection.return_value = False
        client._get_ns_collection_meta.return_value = None
        client.create_collection.side_effect = ValueError("Collection 'items' already exists")

        with pytest.raises(ValueError, match="namespace metadata is missing"):
            BaseClient.get_or_create_collection(client, "items", use_namespace=True)

    def test_resumes_incomplete_namespace_collection_when_present(self):
        """Test resumes incomplete namespace collection when present."""
        client = MagicMock(spec=BaseClient)
        resumed = object()
        client.has_collection.return_value = True
        client._is_incomplete_ns_collection.return_value = True
        client.create_collection.return_value = resumed

        result = BaseClient.get_or_create_collection(client, "items", use_namespace=True)

        assert result is resumed
        client.create_collection.assert_called_once()
        client.get_collection.assert_not_called()

    def test_does_not_mask_unrelated_create_errors(self):
        """Test does not mask unrelated create errors."""
        client = MagicMock(spec=BaseClient)
        client.has_collection.return_value = False
        client.create_collection.side_effect = ValueError("invalid dimension")

        with pytest.raises(ValueError, match="invalid dimension"):
            BaseClient.get_or_create_collection(client, "items")


class TestListNsNamespacesRecyclebinFilter:
    """TestListNsNamespacesRecyclebinFilter class."""

    def test_sql_excludes_recyclebin_rows(self):
        """Test sql excludes recyclebin rows."""
        client = MagicMock(spec=BaseClient)
        client._qtable.return_value = "`sdk_namespaces`"
        client._execute.return_value = [
            ("1", "active_ns"),
        ]

        result = BaseClient._list_ns_namespaces(client, "coll_1")

        assert result == [{"namespace_id": "1", "namespace_name": "active_ns"}]
        sql = client._execute.call_args[0][0]
        assert "__recyclebin_" in sql
        assert "LEFT(namespace_name, 13) <> '__recyclebin_'" in sql


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
