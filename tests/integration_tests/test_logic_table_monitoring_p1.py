"""
Logic table monitoring P1 integration tests.

Includes concurrent get_or_create_namespace idempotency coverage.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT, ns_schema

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


def _count_sdk_namespaces(client, collection_id: str, namespace_name: str) -> int:
    """Count sdk namespaces."""
    rows = client._server._execute(
        "SELECT COUNT(*) AS cnt FROM sdk_namespaces "
        f"WHERE collection_id = '{collection_id}' AND namespace_name = '{namespace_name}'"
    )
    row = rows[0]
    return int(row["cnt"] if isinstance(row, dict) else row[0])


def _count_sdk_ltables(client, collection_id: str, namespace_id: int) -> int:
    """Count sdk ltables."""
    rows = client._server._execute(
        "SELECT COUNT(*) AS cnt FROM sdk_ltables "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {int(namespace_id)} "
        "AND ltable_name = 'default'"
    )
    row = rows[0]
    return int(row["cnt"] if isinstance(row, dict) else row[0])


class TestLogicTableMonitoringP1:
    """TestLogicTableMonitoringP1 class."""

    def test_concurrent_get_or_create_same_namespace_is_idempotent(self, oceanbase_client):
        """8 threads with independent clients race on the same namespace name."""
        owner = oceanbase_client
        collection_name = f"ltmon_p1_{uuid.uuid4().hex[:12]}"
        collection = owner.create_collection(
            name=collection_name,
            schema=ns_schema(),
            use_namespace=True,
            partition_count=NAMESPACE_TEST_PARTITION_COUNT,
        )
        coll_id = collection.id
        namespace_name = "ns_shared"
        barrier = threading.Barrier(8)
        results: dict[int, dict] = {}
        errors: list[str] = []

        def _worker(thread_id: int) -> None:
            """Worker."""
            client = _make_oceanbase_client()
            try:
                coll = client.get_collection(collection_name)
                barrier.wait(timeout=30)
                ns = coll.get_or_create_namespace(namespace_name)
                results[thread_id] = {
                    "namespace_id": str(ns.namespace_id),
                    "name": ns.name,
                }
            except Exception as exc:
                errors.append(f"thread {thread_id}: {exc!r}")
            finally:
                if hasattr(client, "close"):
                    client.close()

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_worker, i) for i in range(8)]
            for fut in as_completed(futures, timeout=60):
                fut.result()

        try:
            assert not errors, f"concurrent get_or_create_namespace failed: {errors}"
            assert len(results) == 8, results
            ns_ids = {item["namespace_id"] for item in results.values()}
            assert len(ns_ids) == 1, f"expected one namespace_id, got {ns_ids}"
            ns_id = next(iter(ns_ids))
            assert _count_sdk_namespaces(owner, coll_id, namespace_name) == 1
            assert _count_sdk_ltables(owner, coll_id, int(ns_id)) == 1
        finally:
            owner.delete_collection(name=collection_name)
