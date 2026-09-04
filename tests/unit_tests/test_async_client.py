"""Unit tests for the optional asynchronous client API."""

import asyncio
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
    client._execute = AsyncMock(return_value=None)
    collection = AsyncCollection(
        client,
        "items",
        "collection_id",
        dimension=3,
        embedding_function=StubEmbeddingFunction(),
    )

    asyncio.run(client._collection_add(collection, ids="one", embeddings=None, metadatas=None, documents="hello"))

    client._execute.assert_awaited_once()
    assert "INSERT INTO" in client._execute.await_args.args[0]
    assert client._execute.await_args.args[1][0] == "one"


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
