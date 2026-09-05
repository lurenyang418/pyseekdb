"""Unit tests for synchronous collection operations."""

from unittest.mock import MagicMock

import pytest

from pyseekdb.client.collection_operations import CollectionOperationsMixin


def test_collection_upsert_uses_atomic_write_without_read_before_write() -> None:
    client = CollectionOperationsMixin()
    client._get_collection_table_name = MagicMock(return_value="collection_table")
    client._execute = MagicMock()
    client._collection_get = MagicMock(side_effect=AssertionError("upsert must not read each record"))

    client._collection_upsert(
        collection_id="collection-id",
        collection_name="items",
        ids=["one", "two"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["first", "second"],
        metadatas=[{"rank": 1}, {"rank": 2}],
    )

    assert client._execute.call_count == 2
    for call in client._execute.call_args_list:
        sql = call.args[0]
        assert "ON DUPLICATE KEY UPDATE" in sql
        assert "INSERT INTO `collection_table`" in sql


def test_collection_upsert_rejects_empty_ids() -> None:
    client = CollectionOperationsMixin()

    with pytest.raises(ValueError, match="ids must not be empty"):
        client._collection_upsert(
            collection_id="collection-id",
            collection_name="items",
            ids=[],
            metadatas=[{"rank": 1}],
        )


def test_collection_upsert_supports_metadata_only() -> None:
    client = CollectionOperationsMixin()
    client._get_collection_table_name = MagicMock(return_value="collection_table")
    client._execute = MagicMock()

    client._collection_upsert(
        collection_id="collection-id",
        collection_name="items",
        ids=["one"],
        metadatas=[{"rank": 1}],
    )

    sql = client._execute.call_args.args[0]
    assert "embedding = NULL" not in sql
    assert r"\"rank\"" in sql
    assert "metadata =" in sql
    assert "ON DUPLICATE KEY UPDATE" in sql


def test_collection_upsert_includes_sparse_embedding() -> None:
    client = CollectionOperationsMixin()
    client._get_collection_table_name = MagicMock(return_value="collection_table")
    client._execute = MagicMock()
    client._generate_sparse_embeddings = MagicMock(return_value=[{1: 0.5}])
    sparse_config = MagicMock()
    sparse_config.resolve_source_key.return_value = ("document", None)

    client._collection_upsert(
        collection_id="collection-id",
        collection_name="items",
        ids=["one"],
        documents=["first"],
        embeddings=[[1.0, 0.0]],
        sparse_vector_index_config=sparse_config,
    )

    sql = client._execute.call_args.args[0]
    assert "sparse_embedding" in sql
