"""Unit tests for the optional asynchronous client API."""

import asyncio
from unittest.mock import AsyncMock

import pyseekdb
from pyseekdb.client.async_collection import AsyncCollection
from pyseekdb.client.capabilities import BackendCapabilities
from pyseekdb.client.version import Version


def test_async_api_is_exported_and_connection_is_lazy() -> None:
    client = pyseekdb.AsyncClient(host="localhost", database="test")

    assert pyseekdb.AsyncClient is not None
    assert pyseekdb.AsyncCollection is AsyncCollection
    assert not client.is_connected()
    assert "disconnected" in repr(client)


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
    client._execute.assert_awaited_once()
    assert "FORK DATABASE `source` TO `destination`" in client._execute.await_args.args[0]
