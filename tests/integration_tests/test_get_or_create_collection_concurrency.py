"""
Integration tests for concurrent-safe get_or_create_collection.
"""

from __future__ import annotations

import contextlib
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from pymysql.converters import escape_string

import pyseekdb


def _make_oceanbase_client():
    """Make oceanbase client."""
    import os

    return pyseekdb.Client(
        host=os.environ.get("OB_HOST", "127.0.0.1"),
        port=int(os.environ.get("OB_PORT", "10902")),
        tenant=os.environ.get("OB_TENANT", "mysql"),
        database=os.environ.get("OB_DATABASE", "test"),
        user=os.environ.get("OB_USER", "root"),
        password=os.environ.get("OB_PASSWORD", ""),
    )


def _count_sdk_collections(client, collection_name: str) -> int:
    """Count sdk collections."""
    name_escaped = escape_string(collection_name)
    rows = client._server._execute(
        f"SELECT COUNT(*) AS cnt FROM sdk_collections WHERE collection_name = '{name_escaped}'"
    )
    row = rows[0]
    return int(row["cnt"] if isinstance(row, dict) else row[0])


def _has_unique_name_index(client) -> bool:
    """Has unique name index."""
    rows = client._server._execute("SHOW INDEX FROM sdk_collections")
    for row in rows:
        key_name = row["Key_name"] if isinstance(row, dict) else row[2]
        if key_name == "uk_sdk_coll_name":
            return True
    return False


class TestGetOrCreateCollectionConcurrencyOceanBase:
    """TestGetOrCreateCollectionConcurrencyOceanBase class."""

    def test_concurrent_get_or_create_collection_is_idempotent(self, oceanbase_client):
        """8 independent clients race on get_or_create_collection with the same name."""
        owner = oceanbase_client
        collection_name = f"goc_conc_{uuid.uuid4().hex[:12]}"
        barrier = threading.Barrier(8)
        results: dict[int, dict] = {}
        errors: list[str] = []

        def _worker(thread_id: int) -> None:
            """Worker."""
            client = _make_oceanbase_client()
            try:
                barrier.wait(timeout=30)
                coll = client.get_or_create_collection(
                    collection_name,
                    configuration=pyseekdb.HNSWConfiguration(dimension=3, distance="cosine"),
                    embedding_function=None,
                )
                results[thread_id] = {
                    "collection_id": str(coll.id),
                    "name": coll.name,
                }
            except Exception as exc:
                errors.append(f"thread {thread_id}: {exc!r}")
            finally:
                if hasattr(client, "close"):
                    client.close()

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_worker, i) for i in range(8)]
            for fut in as_completed(futures, timeout=120):
                fut.result()

        try:
            assert not errors, f"concurrent get_or_create_collection failed: {errors}"
            assert len(results) == 8, results
            coll_ids = {item["collection_id"] for item in results.values()}
            assert len(coll_ids) == 1, f"expected one collection_id, got {coll_ids}"
            assert _count_sdk_collections(owner, collection_name) == 1
            assert _has_unique_name_index(owner)
        finally:
            with contextlib.suppress(Exception):
                owner.delete_collection(name=collection_name)
