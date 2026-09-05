"""Unit tests for the optional asynchronous client API."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import pyseekdb
from pyseekdb.client.async_collection import AsyncCollection
from pyseekdb.client.capabilities import BackendCapabilities
from pyseekdb.client.version import Version


class _AsyncLease:
    def __init__(self, value):
        self.value = value
        self.exited = False

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *_):
        self.exited = True


class _FailingCursor:
    description = None

    def __init__(self):
        self.exited = False

    async def execute(self, *_):
        raise RuntimeError("query failed")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.exited = True
        return None


class _Pool:
    closed = False

    def __init__(self):
        self.connection_lease = None
        self.cursor_lease = None
        self.cursor_obj = _FailingCursor()
        self.close_called = False
        self.wait_closed_called = False

    def acquire(self):
        self.connection_lease = _AsyncLease(self)
        return self.connection_lease

    def cursor(self):
        self.cursor_lease = _AsyncLease(self.cursor_obj)
        return self.cursor_lease

    def close(self):
        self.close_called = True
        self.closed = True

    async def wait_closed(self):
        self.wait_closed_called = True


def test_async_api_is_exported_and_connection_is_lazy() -> None:
    client = pyseekdb.AsyncClient(host="localhost", database="test")

    assert pyseekdb.AsyncClient is not None
    assert pyseekdb.AsyncCollection is AsyncCollection
    assert not client.is_connected()
    assert "disconnected" in repr(client)


def test_async_backend_detection_uses_shared_version_parser() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    client._execute = AsyncMock(return_value=[{"version": "seekdb-v1.3.0.0"}])

    backend, version = asyncio.run(client.detect_db_type_and_version())

    assert backend == "seekdb"
    assert version == Version("1.3.0.0")
    client._execute.assert_awaited_once_with("SELECT version() AS version")


def test_async_get_uses_provided_limit_and_offset_once() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    client._execute = AsyncMock(return_value=[])
    collection = AsyncCollection(client, "items", "collection_id", dimension=3)

    result = asyncio.run(
        client._collection_get(
            collection,
            ids=None,
            where=None,
            where_document=None,
            limit=7,
            offset=2,
            include=[],
            query_hint=None,
        )
    )

    assert result == {"ids": []}
    client._execute.assert_awaited_once()
    assert client._execute.await_args.args[1][-2:] == [7, 2]


def test_async_upsert_uses_atomic_write_without_read_before_write() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    client._execute_transaction = AsyncMock()
    collection = AsyncCollection(client, "items", "collection_id", dimension=3)

    asyncio.run(
        client._collection_upsert(
            collection,
            ids=["one", "two"],
            embeddings=None,
            metadatas=[{"rank": 1}, {"rank": 2}],
            documents=None,
        )
    )

    client._execute_transaction.assert_awaited_once()
    statements = client._execute_transaction.await_args.args[0]
    assert len(statements) == 2
    for sql, _params in statements:
        assert "ON DUPLICATE KEY UPDATE" in sql
        assert "SELECT" not in sql


def test_async_count_preserves_zero() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    client._execute = AsyncMock(return_value=[{"count": 0}])
    collection = AsyncCollection(client, "items", "collection_id", dimension=3)

    assert asyncio.run(client._collection_count(collection)) == 0


def test_async_add_can_generate_embeddings_from_documents() -> None:
    class StubEmbeddingFunction:
        dimension = 3

        def __call__(self, documents: list[str]) -> list[list[float]]:
            return [[float(len(document)), 0.0, 1.0] for document in documents]

    client = pyseekdb.AsyncClient(host="localhost")
    client._execute_transaction = AsyncMock()
    collection = AsyncCollection(
        client,
        "items",
        "collection_id",
        dimension=3,
        embedding_function=StubEmbeddingFunction(),
    )

    asyncio.run(client._collection_add(collection, ids="one", embeddings=None, metadatas=None, documents="hello"))

    client._execute_transaction.assert_awaited_once()
    statements = client._execute_transaction.await_args.args[0]
    assert len(statements) == 1
    assert "INSERT INTO" in statements[0][0]
    assert statements[0][1][0] == "one"


def test_async_delete_rejects_empty_ids() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    client._execute = AsyncMock()
    collection = AsyncCollection(client, "items", "collection_id", dimension=3)

    with pytest.raises(ValueError, match="ids must not be empty"):
        asyncio.run(
            client._collection_delete(
                collection,
                ids=[],
                where=None,
                where_document=None,
            )
        )

    client._execute.assert_not_awaited()


def test_async_transaction_rolls_back_when_a_statement_fails() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    connection = MagicMock()
    connection.begin = AsyncMock()
    connection.commit = AsyncMock()
    connection.rollback = AsyncMock()
    cursor = MagicMock()
    cursor.execute = AsyncMock(side_effect=[None, RuntimeError("second statement failed")])
    connection.cursor.return_value = _AsyncLease(cursor)
    pool = MagicMock(closed=False)
    pool.acquire.return_value = _AsyncLease(connection)
    client._pool = pool

    with pytest.raises(RuntimeError, match="second statement failed"):
        asyncio.run(client._execute_transaction([("INSERT 1", []), ("INSERT 2", [])]))

    connection.begin.assert_awaited_once()
    connection.commit.assert_not_awaited()
    connection.rollback.assert_awaited_once()


def test_async_create_marks_catalog_ready_only_after_physical_table_creation() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    client._create_catalog_if_not_exists = AsyncMock()
    client.has_collection = AsyncMock(return_value=False)
    client._get_collection_id = AsyncMock(return_value="collection_id")
    client._get_collection_catalog_row = AsyncMock(
        return_value={
            "collection_id": "collection_id",
            "collection_name": "items",
            "settings": {"state": "ready", "dimension": 384},
        }
    )
    client._execute = AsyncMock(return_value=None)

    collection = asyncio.run(client.create_collection("items"))

    assert collection.id == "collection_id"
    insert_params = client._execute.await_args_list[0].args[1]
    update_params = client._execute.await_args_list[-1].args[1]
    assert json.loads(insert_params[1])["state"] == "creating"
    assert json.loads(update_params[0])["state"] == "ready"
    assert "UPDATE `sdk_collections`" in client._execute.await_args_list[-1].args[0]


def test_async_fork_publishes_ready_only_after_fork_table_finishes() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    client.get_backend_capabilities = AsyncMock(
        return_value=BackendCapabilities(backend="seekdb", version=Version("1.2.0"))
    )
    client.has_collection = AsyncMock(return_value=False)
    client._get_collection_id = AsyncMock(return_value="forked_id")
    client._get_collection_catalog_row = AsyncMock(
        return_value={
            "collection_id": "forked_id",
            "collection_name": "forked",
            "settings": {"state": "ready", "dimension": 3, "distance": "l2"},
        }
    )
    client._execute = AsyncMock(
        side_effect=[
            [{"settings": json.dumps({"version": 2, "dimension": 3, "distance": "l2", "state": "ready"})}],
            None,
            None,
            None,
        ]
    )
    source = AsyncCollection(client, "source", "source_id", dimension=3, distance="l2")

    result = asyncio.run(client._collection_fork(source, "forked"))

    assert result.id == "forked_id"
    inserted_settings = json.loads(client._execute.await_args_list[1].args[1][1])
    assert inserted_settings["state"] == "creating"
    assert "creation_token" in inserted_settings
    assert "FORK TABLE" in client._execute.await_args_list[2].args[0]


def test_async_fork_cancellation_cleans_owned_catalog() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    client.get_backend_capabilities = AsyncMock(
        return_value=BackendCapabilities(backend="seekdb", version=Version("1.2.0"))
    )
    client.has_collection = AsyncMock(return_value=False)
    client._get_collection_id = AsyncMock(return_value="forked_id")
    client._execute = AsyncMock(
        side_effect=[
            [{"settings": json.dumps({"version": 2, "dimension": 3, "state": "ready"})}],
            None,
            asyncio.CancelledError(),
        ]
    )
    client._cleanup_failed_collection = AsyncMock(return_value=True)
    source = AsyncCollection(client, "source", "source_id", dimension=3)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(client._collection_fork(source, "forked"))

    client._cleanup_failed_collection.assert_awaited_once()
    assert client._cleanup_failed_collection.await_args.args[0:2] == ("forked", "forked_id")


def test_async_cleanup_incomplete_collection_requires_creation_token() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    client._get_collection_catalog_row = AsyncMock(
        return_value={
            "collection_id": "collection_id",
            "collection_name": "items",
            "settings": {"state": "creating", "creation_token": "stale-token"},
        }
    )
    client._cleanup_failed_collection = AsyncMock(return_value=True)

    asyncio.run(client.cleanup_incomplete_collection("items"))

    client._cleanup_failed_collection.assert_awaited_once_with("items", "collection_id", "stale-token")


def test_async_loaded_sparse_collection_rejects_writes() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    client._execute = AsyncMock(return_value=[])

    collection = asyncio.run(
        client._collection_from_row({
            "collection_id": "collection_id",
            "collection_name": "items",
            "settings": {"dimension": 3, "sparse_vector_index": {}},
        })
    )

    assert collection.has_sparse_vector_index
    with pytest.raises(NotImplementedError, match="sparse vector collections"):
        asyncio.run(collection.add(ids="one", embeddings=[1.0, 2.0, 3.0]))


def test_async_get_waits_for_a_collection_created_by_another_client() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    ready_collection = AsyncCollection(client, "items", "collection_id", dimension=3, distance="l2")
    client._get_collection_catalog_row = AsyncMock(
        side_effect=[
            {"collection_id": "collection_id", "collection_name": "items", "settings": {"state": "creating"}},
            {
                "collection_id": "collection_id",
                "collection_name": "items",
                "settings": {"state": "ready", "dimension": 3, "distance": "l2"},
            },
        ]
    )
    client._collection_from_row = AsyncMock(return_value=ready_collection)

    result = asyncio.run(client.get_collection("items"))

    assert result is ready_collection
    assert client._get_collection_catalog_row.await_count == 2


def test_async_get_recovers_legacy_distance_from_table_definition() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    client._execute = AsyncMock(
        return_value=[{"Create Table": "VECTOR INDEX idx_vec (embedding) WITH (DISTANCE=l2, TYPE=HNSW)"}]
    )

    collection = asyncio.run(
        client._collection_from_row({
            "collection_id": "collection_id",
            "collection_name": "items",
            "settings": {"dimension": 3},
        })
    )

    assert collection.distance == "l2"


def test_async_get_or_create_passes_schema_embedding_function_to_existing_collection() -> None:
    class EmbeddingFunctionStub:
        dimension = 3

    embedding_function = EmbeddingFunctionStub()
    client = pyseekdb.AsyncClient(host="localhost")
    client.has_collection = AsyncMock(return_value=True)
    existing = object()
    client.get_collection = AsyncMock(return_value=existing)

    result = asyncio.run(
        client.get_or_create_collection(
            "items",
            schema=pyseekdb.Schema(embedding_function=embedding_function),
        )
    )

    assert result is existing
    client.get_collection.assert_awaited_once_with("items", embedding_function=embedding_function)


def test_async_fork_database_returns_client_bound_to_destination() -> None:
    client = pyseekdb.AsyncClient(host="localhost", database="source", user="root")
    client._execute = AsyncMock(return_value=None)
    client.get_backend_capabilities = AsyncMock(
        return_value=BackendCapabilities(backend="seekdb", version=Version("1.2.0"))
    )

    forked = asyncio.run(client.fork_database("destination"))

    assert isinstance(forked, pyseekdb.AsyncClient)
    assert forked.database == "destination"
    assert forked._fork_parent is client
    client._execute.assert_awaited_once()
    assert "FORK DATABASE `source` TO `destination`" in client._execute.await_args.args[0]


def test_only_async_forked_client_can_destroy_its_database() -> None:
    client = pyseekdb.AsyncClient(host="localhost", database="source")

    async def scenario() -> None:
        with pytest.raises(ValueError, match="Only a client returned by fork_database"):
            await client.destroy()

        client.get_backend_capabilities = AsyncMock(
            return_value=BackendCapabilities(backend="seekdb", version=Version("1.2.0"))
        )
        client._execute = AsyncMock(return_value=None)
        forked = await client.fork_database("destination")
        await forked.destroy()

        assert client._execute.await_args_list[-1].args[0] == "DROP DATABASE IF EXISTS `destination`"
        assert forked._fork_parent is None

    asyncio.run(scenario())


def test_async_execute_releases_pool_lease_when_cursor_fails() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    pool = _Pool()
    client._pool = pool

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="query failed"):
            await client._execute("SELECT 1")
        await client.close()

    asyncio.run(scenario())

    assert pool.connection_lease is not None
    assert pool.connection_lease.exited
    assert pool.cursor_lease is not None
    assert pool.cursor_lease.exited
    assert pool.close_called
    assert pool.wait_closed_called
    assert not client.is_connected()


def test_async_context_manager_closes_pool() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    pool = _Pool()
    client._pool = pool

    async def scenario() -> None:
        async with client:
            assert client.is_connected()

    asyncio.run(scenario())

    assert pool.close_called
    assert pool.wait_closed_called
    assert not client.is_connected()


def test_async_pool_is_initialized_once_for_concurrent_first_use() -> None:
    client = pyseekdb.AsyncClient(host="localhost")
    driver = MagicMock()
    driver.DictCursor = object()
    pool = MagicMock(closed=False)

    async def create_pool(**_kwargs):
        await asyncio.sleep(0)
        return pool

    driver.create_pool = AsyncMock(side_effect=create_pool)
    client._require_driver = MagicMock(return_value=driver)

    async def scenario() -> None:
        first, second = await asyncio.gather(client._ensure_pool(), client._ensure_pool())
        assert first is pool
        assert second is pool

    asyncio.run(scenario())

    assert client._require_driver.call_count == 2
    assert driver.create_pool.await_count == 1
