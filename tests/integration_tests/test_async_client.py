"""Integration coverage for the optional asynchronous seekdb client."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time

import pytest

import pyseekdb

pytestmark = pytest.mark.asyncio


def _new_name(prefix: str) -> str:
    """Return a valid unique SQL identifier for the test server."""
    return f"{prefix}_{os.getpid()}_{int(time.time() * 1000)}"


def _server_client() -> pyseekdb.AsyncClient:
    """Build an async client from the same server settings as sync fixtures."""
    return pyseekdb.AsyncClient(
        host=os.environ.get("SERVER_HOST", "127.0.0.1"),
        port=int(os.environ.get("SERVER_PORT", "2881")),
        tenant="sys",
        database=os.environ.get("SERVER_DATABASE", "test"),
        user=os.environ.get("SERVER_USER", "root"),
        password=os.environ.get("SERVER_PASSWORD", ""),
    )


async def test_async_server_crud_query_concurrency_and_collection_fork() -> None:
    """Exercise pooled CRUD, concurrent queries, and collection fork on seekdb."""
    client = _server_client()
    collection_name = _new_name("async_collection")
    forked_name = f"{collection_name}_fork"
    try:
        assert await client.ping()
        collection = await client.create_collection(
            collection_name,
            schema=pyseekdb.Schema(
                vector_index=pyseekdb.HNSWConfiguration(dimension=3, distance="l2"),
            ),
        )
        await collection.add(
            ids=["one", "two"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            documents=["first", "second"],
            metadatas=[{"rank": 1}, {"rank": 2}],
        )
        assert await collection.count() == 2
        assert client._pool is not None
        pool = client._pool

        result = await collection.query(query_embeddings=[1.0, 0.0, 0.0], n_results=1)
        assert result["ids"][0] == ["one"]

        concurrent_results = await asyncio.gather(
            *(collection.query(query_embeddings=[1.0, 0.0, 0.0], n_results=1) for _ in range(4))
        )
        assert all(item["ids"][0] == ["one"] for item in concurrent_results)
        assert client._pool is pool

        forked = await collection.fork(forked_name)
        try:
            assert await forked.count() == 2
        finally:
            await client.delete_collection(forked_name)
    finally:
        with contextlib.suppress(Exception):
            await client.delete_collection(collection_name)
        await client.close()


async def test_async_server_database_fork_returns_bound_client() -> None:
    """Exercise FORK DATABASE and async lifecycle on a seekdb >= 1.2 server."""
    client = _server_client()
    destination = _new_name("async_database_fork")
    forked = None
    try:
        forked = await client.fork_database(destination)
        assert forked.database == destination
        assert await forked.ping()
    finally:
        if forked is not None:
            await forked.destroy()
        else:
            with contextlib.suppress(Exception):
                await client._execute(f"DROP DATABASE IF EXISTS `{destination}`")
        await client.close()
