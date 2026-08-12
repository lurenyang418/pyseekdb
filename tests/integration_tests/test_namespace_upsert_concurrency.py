"""Concurrent namespace.upsert() repro for duplicate business ids."""

from __future__ import annotations

import contextlib
import os
import threading
import time

from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT, cleanup, ns_schema

import pyseekdb


def _unique_name(prefix: str) -> str:
    """Unique name."""
    return f"{prefix}_{time.time_ns()}"


def _new_oceanbase_client():
    """New oceanbase client."""
    return pyseekdb.Client(
        host=os.environ.get("OB_HOST", "127.0.0.1"),
        port=int(os.environ.get("OB_PORT", "10902")),
        tenant=os.environ.get("OB_TENANT", "mysql"),
        database=os.environ.get("OB_DATABASE", "test"),
        user=os.environ.get("OB_USER", "root"),
        password=os.environ.get("OB_PASSWORD", ""),
    )


def test_multi_client_concurrent_upsert_same_new_id_should_keep_single_record(
    oceanbase_client,
):
    """Test multi client concurrent upsert same new id should keep single record."""
    collection = oceanbase_client.create_collection(
        name=_unique_name("test_ns_upsert_race_repro"),
        schema=ns_schema(),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )
    clients = [_new_oceanbase_client() for _ in range(4)]
    try:
        ns = collection.create_namespace("race_ns")
        namespaces = [client.get_collection(collection.name).get_namespace("race_ns") for client in clients]
        barrier = threading.Barrier(len(namespaces))
        errors: list[Exception] = []

        def _upsert_once(target_ns, idx: int) -> None:
            """Upsert once."""
            try:
                barrier.wait(timeout=30)
                target_ns.upsert(
                    ids="same_new_id",
                    embeddings=[1.0, 2.0, 3.0],
                    documents=f"written by client {idx}",
                    metadatas={"client": idx},
                )
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_upsert_once, args=(target_ns, idx)) for idx, target_ns in enumerate(namespaces)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert all(not thread.is_alive() for thread in threads), "upsert worker thread timed out"

        assert errors == []
        result = ns.get(ids="same_new_id", include=["documents", "metadatas"])
        assert result["ids"] == ["same_new_id"]
        assert ns.count() == 1
    finally:
        for client in clients:
            with contextlib.suppress(Exception):
                client.close()
        cleanup(oceanbase_client, collection)
