"""
Namespace integration tests for connection lifecycle and multi-client access.

Verifies that namespace data persists across:
1. Disconnect + lazy reconnect on the same Client instance
2. A separate Client connecting to the same collection/namespace
3. Cross-client read/write visibility on shared namespace data
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any

from namespace_dml_helpers import cleanup, create_ns_collection

import pyseekdb


def _new_oceanbase_client() -> Any:
    """Second pymysql-backed client (separate TCP connection)."""
    client = pyseekdb.Client(
        host=os.environ.get("OB_HOST", "localhost"),
        port=int(os.environ.get("OB_PORT", "11202")),
        tenant=os.environ.get("OB_TENANT", "mysql"),
        database=os.environ.get("OB_DATABASE", "test"),
        user=os.environ.get("OB_USER", "root"),
        password=os.environ.get("OB_PASSWORD", ""),
    )
    result = client._server._execute("SELECT 1 AS test")
    assert result and result[0].get("test") == 1
    return client


class TestNamespaceReconnectAndMultiClient:
    """TestNamespaceReconnectAndMultiClient class."""

    def test_same_client_reconnect_reads_namespace_data(self, oceanbase_client):
        """After _cleanup(), the next SDK call should reconnect and read prior rows."""
        client = oceanbase_client
        collection = create_ns_collection(client, suffix="_reconn")
        coll_name = collection.name
        try:
            ns = collection.create_namespace("persist_ns")
            ns.add(
                ids="doc1",
                embeddings=[[1.0, 0.0, 0.0]],
                documents=["persist after reconnect"],
                metadatas={"tag": "before_disconnect"},
            )

            assert client._server.is_connected()
            client._server._cleanup()
            assert not client._server.is_connected()

            collection2 = client.get_collection(coll_name)
            assert client._server.is_connected()

            ns2 = collection2.get_namespace("persist_ns")
            res = ns2.get(ids="doc1", include=["documents", "metadatas"])
            assert res["ids"] == ["doc1"]
            assert res["documents"][0] == "persist after reconnect"
            assert res["metadatas"][0]["tag"] == "before_disconnect"

            query = ns2.query(query_embeddings=[1.0, 0.0, 0.0], n_results=3, include=["metadatas"])
            assert "doc1" in query["ids"][0]
        finally:
            cleanup(client, collection)

    def test_new_client_reads_existing_namespace(self, oceanbase_client):
        """A fresh Client should see data written by another connection."""
        client_a = oceanbase_client
        client_b = _new_oceanbase_client()
        collection = create_ns_collection(client_a, suffix="_newcli")
        coll_name = collection.name
        try:
            ns_a = collection.create_namespace("shared_read")
            ns_a.add(
                ids="shared1",
                embeddings=[[0.0, 1.0, 0.0]],
                documents=["written by client A"],
                metadatas={"writer": "A"},
            )

            coll_b = client_b.get_collection(coll_name)
            assert coll_b.use_namespace is True
            assert coll_b.id == collection.id

            ns_b = coll_b.get_namespace("shared_read")
            res = ns_b.get(ids="shared1", include=["documents", "metadatas"])
            assert res["ids"] == ["shared1"]
            assert res["documents"][0] == "written by client A"
            assert res["metadatas"][0]["writer"] == "A"
        finally:
            cleanup(client_a, collection)
            with contextlib.suppress(Exception):
                client_b._server._cleanup()

    def test_two_clients_read_write_same_namespace(self, oceanbase_client):
        """Writes from client B should be visible to client A on the same namespace."""
        client_a = oceanbase_client
        client_b = _new_oceanbase_client()
        collection = create_ns_collection(client_a, suffix="_multi")
        coll_name = collection.name
        try:
            ns_a = collection.create_namespace("shared_rw")
            ns_a.add(
                ids="from_a",
                embeddings=[[1.0, 0.0, 0.0]],
                documents=["seed from A"],
                metadatas={"src": "A"},
            )

            coll_b = client_b.get_collection(coll_name)
            ns_b = coll_b.get_namespace("shared_rw")

            got_b = ns_b.get(ids="from_a", include=["metadatas"])
            assert got_b["ids"] == ["from_a"]
            assert got_b["metadatas"][0]["src"] == "A"

            ns_b.add(
                ids="from_b",
                embeddings=[[0.0, 1.0, 0.0]],
                documents=["written by B"],
                metadatas={"src": "B"},
            )

            # Small pause so count/get on A observes B's insert on a separate session.
            time.sleep(0.2)
            got_a = ns_a.get(where={"src": "B"}, include=["documents"])
            assert got_a["ids"] == ["from_b"]
            assert got_a["documents"][0] == "written by B"

            assert ns_a.count() == 2
            assert ns_b.count() == 2
        finally:
            cleanup(client_a, collection)
            with contextlib.suppress(Exception):
                client_b._server._cleanup()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s", "-k", "oceanbase"])
