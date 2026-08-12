"""
Unit tests for namespace upsert duplicate-record reconciliation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from pyseekdb.client.client_base import BaseClient  # noqa: E402


class TestNamespaceUpsertReconcile:
    """TestNamespaceUpsertReconcile class."""

    def test_reconcile_skips_when_single_row_exists(self):
        """Test reconcile skips when single row exists."""
        client = MagicMock(spec=BaseClient)
        client._count_namespace_records_by_id.return_value = 1

        BaseClient._reconcile_namespace_duplicate_records(
            client,
            collection_id="c" * 32,
            collection_name="items",
            namespace_id="7",
            namespace_name="race_ns",
            ltable_id=9,
            table_name="logic_data_table",
            ids=["same_new_id"],
            documents=["doc"],
            metadatas=[{"client": 0}],
            embeddings=[[1.0, 2.0, 3.0]],
            embedding_function=None,
        )

        client._delete_namespace_records_by_id.assert_not_called()
        client._namespace_add.assert_not_called()

    def test_reconcile_collapses_duplicate_rows(self):
        """Test reconcile collapses duplicate rows."""
        client = MagicMock(spec=BaseClient)
        client._count_namespace_records_by_id.side_effect = [4, 0, 1]

        BaseClient._reconcile_namespace_duplicate_records(
            client,
            collection_id="c" * 32,
            collection_name="items",
            namespace_id="7",
            namespace_name="race_ns",
            ltable_id=9,
            table_name="logic_data_table",
            ids=["same_new_id"],
            documents=["winner"],
            metadatas=[{"client": 2}],
            embeddings=[[1.0, 2.0, 3.0]],
            embedding_function=None,
        )

        client._delete_namespace_records_by_id.assert_called_once_with("logic_data_table", 7, 9, "same_new_id")
        client._namespace_add.assert_called_once()
        add_kwargs = client._namespace_add.call_args.kwargs
        assert add_kwargs["ids"] == ["same_new_id"]
        assert add_kwargs["documents"] == ["winner"]
        assert add_kwargs["metadatas"] == [{"client": 2}]
        assert add_kwargs["embeddings"] == [[1.0, 2.0, 3.0]]

    def test_reconcile_raises_when_retries_exhausted(self):
        """Test reconcile raises when retries exhausted."""
        client = MagicMock(spec=BaseClient)
        client._count_namespace_records_by_id.return_value = 4

        with (
            patch("pyseekdb.client.client_base.time.sleep"),
            pytest.raises(ValueError, match="Failed to reconcile duplicate namespace rows"),
        ):
            BaseClient._reconcile_namespace_duplicate_records(
                client,
                collection_id="c" * 32,
                collection_name="items",
                namespace_id="7",
                namespace_name="race_ns",
                ltable_id=9,
                table_name="logic_data_table",
                ids=["same_new_id"],
                documents=["doc"],
                metadatas=None,
                embeddings=None,
                embedding_function=None,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
