"""test namespace drop validation module."""

import contextlib
import threading
import time

import pytest
from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT

import pyseekdb
from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import VectorIndexConfig
from pyseekdb.client.meta_info import NamespaceCollectionNames
from pyseekdb.client.schema import Schema

RECYCLEBIN_PREFIX = "__recyclebin_"

# LTABLE_BG scans every 30s; poll slightly beyond two intervals.
BG_POLL_TIMEOUT_SEC = 65.0
BG_POLL_INTERVAL_SEC = 1.0


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def _create_namespace(collection, name: str):
    """Create namespace."""
    ns = collection.create_namespace(name)
    ns.prewarm()
    return ns


def _make_collection(client, suffix: str = ""):
    """Create a namespace-mode collection with an IVF index (so we get a full
    set of physical tables: logic_data / kv_data / logic_schema / hot_table)."""
    name = f"test_ns_drop_{int(time.time() * 1000)}{suffix}"
    schema = Schema(
        vector_index=VectorIndexConfig(
            ivf=IVFConfiguration(dimension=3, distance="cosine", centroids_fresh_mode="spfresh"),
            embedding_function=None,
        ),
    )
    return client.create_collection(
        name=name,
        schema=schema,
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )


def _execute(client, sql: str):
    """Execute."""
    return client._server._execute(sql)


def _catalog_table(client, table: str) -> str:
    """Qualified catalog table in the client's configured database (default: test)."""
    db = client._server.database
    return f"`{db}`.`{table}`"


def _is_ss_mode(client) -> bool:
    """Best-effort detect whether the connected OB cluster is in SS mode."""
    try:
        rows = _execute(
            client,
            "SHOW PARAMETERS LIKE 'enable_logservice'",
        )
        for r in rows:
            value = (r.get("value") or r.get("VALUE") or "").lower()
            if value in ("true", "1", "on"):
                return True
    except Exception:
        pass
    return False


def _fetch_namespace_name(client, collection_id: str, namespace_id: int):
    """Fetch namespace name."""
    ns_table = _catalog_table(client, "sdk_namespaces")
    rows = _execute(
        client,
        f"SELECT namespace_name FROM {ns_table} "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {namespace_id}",
    )
    if not rows:
        return None
    return rows[0].get("namespace_name") or rows[0].get("NAMESPACE_NAME")


def _count_ltables(client, collection_id: str, namespace_id: int) -> int:
    """Count ltables."""
    lt_table = _catalog_table(client, "sdk_ltables")
    rows = _execute(
        client,
        f"SELECT COUNT(*) AS c FROM {lt_table} "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {namespace_id}",
    )
    return int(rows[0]["c"] if "c" in rows[0] else rows[0]["C"])


def _count_logic_schema_rows(client, collection_id: str, namespace_id: int) -> int:
    """Count logic schema rows."""
    tbl = NamespaceCollectionNames.logic_schema_table_name(collection_id)
    db = client._server.database
    rows = _execute(
        client,
        f"SELECT COUNT(*) AS c FROM `{db}`.`{tbl}` WHERE namespace_id = {namespace_id}",
    )
    return int(rows[0]["c"] if "c" in rows[0] else rows[0]["C"])


def _count_hot_table_rows(client, collection_id: str, namespace_id: int) -> int:
    """Count hot table rows."""
    tbl = NamespaceCollectionNames.hot_table_name(collection_id)
    db = client._server.database
    try:
        rows = _execute(
            client,
            f"SELECT COUNT(*) AS c FROM `{db}`.`{tbl}` WHERE namespace_id = {namespace_id}",
        )
    except Exception:
        return -1  # table missing
    return int(rows[0]["c"] if "c" in rows[0] else rows[0]["C"])


def _count_logic_data_rows(client, collection_id: str, namespace_id: int, ltable_id: int | None = None) -> int:
    """Count logic data rows."""
    tbl = NamespaceCollectionNames.data_table_name(collection_id)
    db = client._server.database
    if ltable_id is not None:
        where = f"namespace_id = {namespace_id} AND ltable_id = {ltable_id}"
    else:
        where = f"namespace_id = {namespace_id}"
    rows = _execute(
        client,
        f"SELECT COUNT(*) AS c FROM `{db}`.`{tbl}` WHERE {where}",
    )
    return int(rows[0]["c"] if "c" in rows[0] else rows[0]["C"])


def _count_kv_data_rows(client, collection_id: str, namespace_id: int) -> int:
    """Count kv data rows."""
    tbl = NamespaceCollectionNames.kv_data_table_name(collection_id)
    db = client._server.database
    try:
        rows = _execute(
            client,
            f"SELECT COUNT(*) AS c FROM `{db}`.`{tbl}` WHERE namespace_id = {namespace_id}",
        )
    except Exception:
        return -1
    return int(rows[0]["c"] if "c" in rows[0] else rows[0]["C"])


def _wait_until(
    predicate, timeout_sec: float = BG_POLL_TIMEOUT_SEC, interval_sec: float = BG_POLL_INTERVAL_SEC, desc: str = ""
):
    """Wait until."""
    deadline = time.time() + timeout_sec
    last_exc = None
    while time.time() < deadline:
        try:
            if predicate():
                return
        except Exception as exc:
            last_exc = exc
        time.sleep(interval_sec)
    msg = f"timeout after {timeout_sec}s waiting for {desc!r}"
    if last_exc is not None:
        raise AssertionError(msg) from last_exc
    raise AssertionError(msg)


def _seed_kv_data(client, collection_id: str, namespace_id: int, count: int = 5):
    """Seed rows in <collection>_kv_data_table (primary target of LTABLE_BG ns delete)."""
    tbl = NamespaceCollectionNames.kv_data_table_name(collection_id)
    db = client._server.database
    values = []
    for i in range(count):
        key_hex = f"{i + 1:064x}"
        values.append(f"({namespace_id}, X'{key_hex}', X'00')")
    _execute(
        client,
        f"INSERT INTO `{db}`.`{tbl}` (namespace_id, kv_key, kv_value) VALUES {', '.join(values)}",
    )


def _seed_logic_data(client, collection_id: str, namespace_id: int, ltable_id: int, count: int = 3):
    """Seed logic data."""
    tbl = NamespaceCollectionNames.data_table_name(collection_id)
    db = client._server.database
    values = []
    for i in range(count):
        values.append(
            f"({namespace_id}, {ltable_id}, 'doc_{i}', "
            f"X'0000803f0000000000000000', "
            f'\'{{"id": "id_{i}", "metadata": {{"k": "v"}}}}\')'
        )
    _execute(
        client,
        f"INSERT INTO `{db}`.`{tbl}` (namespace_id, ltable_id, document, embedding, data_content) "
        f"VALUES {', '.join(values)}",
    )


def _fetch_ltable_id(client, collection_id: str, namespace_id: int) -> int | None:
    """Fetch ltable id."""
    lt_table = _catalog_table(client, "sdk_ltables")
    rows = _execute(
        client,
        f"SELECT ltable_id FROM {lt_table} "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {namespace_id} "
        f"ORDER BY ltable_id LIMIT 1",
    )
    if not rows:
        return None
    return int(rows[0]["ltable_id"] if "ltable_id" in rows[0] else rows[0]["LTABLE_ID"])


def _hot_table_exists(client, collection_id: str) -> bool:
    """Hot table exists."""
    tbl = NamespaceCollectionNames.hot_table_name(collection_id)
    db = client._server.database
    try:
        rows = _execute(
            client,
            f"SELECT 1 AS ok FROM information_schema.tables WHERE table_schema = '{db}' AND table_name = '{tbl}'",
        )
        return bool(rows)
    except Exception:
        return False


def _ensure_hot_table(client, collection_id: str) -> str | None:
    """SDK only creates hot_table in SS mode; on SN create it for BG cleanup tests.

    Returns None on success, or an error message string.
    """
    if _hot_table_exists(client, collection_id):
        return None
    tbl = NamespaceCollectionNames.hot_table_name(collection_id)
    db = client._server.database
    tg = NamespaceCollectionNames.tablegroup_name(collection_id)
    ddl_with_tg = (
        f"CREATE TABLE IF NOT EXISTS `{db}`.`{tbl}` ("
        f"  namespace_id BIGINT UNSIGNED NOT NULL,"
        f"  last_access_time TIMESTAMP(6) NOT NULL,"
        f"  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
        f"  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,"
        f"  PRIMARY KEY(namespace_id)"
        f") TABLEGROUP=`{tg}` COMMENT='热点/TTL附属表' DEFAULT CHARSET=utf8mb4 "
        f"PARTITION BY KEY(namespace_id) PARTITIONS 1000"
    )
    ddl_plain = (
        f"CREATE TABLE IF NOT EXISTS `{db}`.`{tbl}` ("
        f"  namespace_id BIGINT UNSIGNED NOT NULL,"
        f"  last_access_time TIMESTAMP(6) NOT NULL,"
        f"  PRIMARY KEY(namespace_id)"
        f") COMMENT='热点/TTL附属表' DEFAULT CHARSET=utf8mb4 "
        f"PARTITION BY KEY(namespace_id) PARTITIONS 1000"
    )
    last_err = ""
    for ddl in (ddl_with_tg, ddl_plain):
        try:
            _execute(client, ddl)
            if _hot_table_exists(client, collection_id):
                return None
        except Exception as exc:
            last_err = str(exc)
    return last_err or "hot_table not visible after CREATE"


def _seed_hot_table(client, collection_id: str, namespace_id: int):
    """Insert a hot_table row (table must exist — use _ensure_hot_table on SN first)."""
    if _ensure_hot_table(client, collection_id) is not None:
        return False
    tbl = NamespaceCollectionNames.hot_table_name(collection_id)
    db = client._server.database
    try:
        _execute(
            client,
            f"INSERT INTO `{db}`.`{tbl}` (namespace_id, last_access_time) VALUES ({namespace_id}, NOW(6))",
        )
        return True
    except Exception:
        return False


def _count_active_ltables(client, collection_id: str, namespace_id: int) -> int:
    """sdk_ltables rows whose name is not a recyclebin entry (BG skip guard)."""
    lt_table = _catalog_table(client, "sdk_ltables")
    rows = _execute(
        client,
        f"SELECT COUNT(*) AS c FROM {lt_table} "
        f"WHERE collection_id = '{collection_id}' AND namespace_id = {namespace_id} "
        f"AND ltable_name NOT LIKE '{RECYCLEBIN_PREFIX}%'",
    )
    return int(rows[0]["c"] if "c" in rows[0] else rows[0]["C"])


def _insert_active_ltable_row(
    client,
    collection_id: str,
    namespace_id: int,
    ltable_name: str,
    ltable_id: int,
):
    """Insert active ltable row."""
    lt_table = _catalog_table(client, "sdk_ltables")
    _execute(
        client,
        f"INSERT INTO {lt_table} "
        f"(ltable_id, collection_id, namespace_id, ltable_name) "
        f"VALUES ({ltable_id}, '{collection_id}', {namespace_id}, '{ltable_name}')",
    )


def _fetch_namespace_drop_history(client, collection_id: str, namespace_id: int):
    """Fetch namespace drop history."""
    try:
        return _execute(
            client,
            "SELECT operation_type, operation_status, total_deleted_rows "
            "FROM oceanbase.__all_virtual_agent_drop_history "
            f"WHERE collection_id = '{collection_id}' "
            f"  AND namespace_id = {namespace_id} AND ltable_id = 0 "
            "ORDER BY finish_time DESC LIMIT 5",
        )
    except Exception:
        return None


def _drop_namespace_via_pl(client, collection_id: str, namespace_id: int):
    """Call DBMS_LOGIC_TABLE.DROP_NAMESPACE directly (raw SQL, bypassing
    SDK's pre-existence guard)."""
    _execute(
        client,
        f"CALL DBMS_LOGIC_TABLE.DROP_NAMESPACE('{collection_id}', {namespace_id})",
    )


def _second_oceanbase_client():
    """Second pymysql-backed client (separate TCP connection) for concurrency tests."""
    import os

    client = pyseekdb.Client(
        host=os.environ.get("OB_HOST", "localhost"),
        port=int(os.environ.get("OB_PORT", "11202")),
        tenant=os.environ.get("OB_TENANT", "mysql"),
        database=os.environ.get("OB_DATABASE", "test"),
        user=os.environ.get("OB_USER", "root"),
        password=os.environ.get("OB_PASSWORD", ""),
    )
    result = client._server._execute("SELECT 1 as test")
    assert result and result[0].get("test") == 1
    return client


# --------------------------------------------------------------------------- #
# tests                                                                       #
# --------------------------------------------------------------------------- #


class TestDropNamespaceCatalogValidation:
    """DROP_NAMESPACE sync transaction + LTABLE_BG async namespace deletion."""

    # ------------------------------------------------------------- #
    # 1. Happy path: rename + cleanup all relevant rows              #
    # ------------------------------------------------------------- #
    def test_drop_namespace_renames_and_cleans_catalog(self, oceanbase_client):
        """Test drop namespace renames and cleans catalog."""
        client = oceanbase_client
        is_ss = _is_ss_mode(client)
        collection = _make_collection(client)
        coll_id = collection.id
        try:
            ns = _create_namespace(collection, "ns_validate")
            ns_id = int(ns.namespace_id)

            # Pre-conditions
            orig_name = _fetch_namespace_name(client, coll_id, ns_id)
            assert orig_name == "ns_validate"
            assert _count_ltables(client, coll_id, ns_id) >= 1, "create_namespace should have inserted a default ltable"
            assert _count_logic_schema_rows(client, coll_id, ns_id) >= 1, (
                "create_namespace should have inserted a logic_schema row"
            )
            if is_ss:
                assert _count_hot_table_rows(client, coll_id, ns_id) == 1

            # Seed logic_data rows — sync DROP must not touch them; LTABLE_BG ns DAG
            # deletes kv_data (see async tests), not logic_data_table.
            lt_id = _fetch_ltable_id(client, coll_id, ns_id)
            assert lt_id is not None, "should have a default ltable"
            _seed_logic_data(client, coll_id, ns_id, lt_id, count=3)
            assert _count_logic_data_rows(client, coll_id, ns_id, lt_id) == 3

            # Action
            _drop_namespace_via_pl(client, coll_id, ns_id)

            # Post-conditions: synchronous phase
            new_name = _fetch_namespace_name(client, coll_id, ns_id)
            assert new_name is not None, "namespace row must still exist (renamed, not deleted)"
            assert new_name.startswith(RECYCLEBIN_PREFIX), (
                f"namespace must be renamed to __recyclebin_ prefix, got {new_name!r}"
            )
            assert "ns_validate" in new_name, f"original name should be embedded in recyclebin name, got {new_name!r}"
            assert _count_ltables(client, coll_id, ns_id) == 0, "sdk_ltables rows must be deleted"
            assert _count_logic_schema_rows(client, coll_id, ns_id) == 0, "logic_schema_table rows must be deleted"
            if is_ss:
                assert _count_hot_table_rows(client, coll_id, ns_id) == 0, "hot_table row must be deleted in SS mode"

            # Logic data rows still exist — async background task handles physical cleanup.
            assert _count_logic_data_rows(client, coll_id, ns_id, lt_id) == 3, (
                "logic_data rows must survive the sync drop phase"
            )
        finally:
            client.delete_collection(name=collection.name)

    # ------------------------------------------------------------- #
    # 1b. Async cleanup: wait 32s, verify physical data is gone      #
    # ------------------------------------------------------------- #
    def test_drop_namespace_async_cleanup(self, oceanbase_client):
        """LTABLE_BG NAMESPACE_DELETE: batch-delete kv_data, remove sdk_namespaces row."""
        client = oceanbase_client
        collection = _make_collection(client)
        coll_id = collection.id
        try:
            ns = _create_namespace(collection, "ns_async")
            ns_id = int(ns.namespace_id)
            lt_id = _fetch_ltable_id(client, coll_id, ns_id)
            assert lt_id is not None

            kv_before = max(_count_kv_data_rows(client, coll_id, ns_id), 0)
            _seed_kv_data(client, coll_id, ns_id, count=5)
            _seed_logic_data(client, coll_id, ns_id, lt_id, count=3)
            kv_after_seed = _count_kv_data_rows(client, coll_id, ns_id)
            assert kv_after_seed > kv_before, (
                f"kv_data should have rows for BG to delete, before={kv_before} after={kv_after_seed}"
            )
            assert _count_logic_data_rows(client, coll_id, ns_id, lt_id) == 3

            _drop_namespace_via_pl(client, coll_id, ns_id)

            recycled_name = _fetch_namespace_name(client, coll_id, ns_id)
            assert recycled_name is not None
            assert recycled_name.startswith(RECYCLEBIN_PREFIX)
            assert _count_kv_data_rows(client, coll_id, ns_id) == kv_after_seed
            assert _count_ltables(client, coll_id, ns_id) == 0

            def _bg_done():
                """Bg done."""
                return (
                    _fetch_namespace_name(client, coll_id, ns_id) is None
                    and _count_kv_data_rows(client, coll_id, ns_id) == 0
                )

            _wait_until(_bg_done, desc="namespace mapping and kv_data removed by LTABLE_BG")

        finally:
            client.delete_collection(name=collection.name)

    def test_async_skipped_while_active_ltable_remains(self, oceanbase_client):
        """BG must not delete sdk_namespaces while a non-recyclebin sdk_ltables row exists."""
        client = oceanbase_client
        collection = _make_collection(client)
        coll_id = collection.id
        blocker_lt_id = 9_000_001
        try:
            ns = _create_namespace(collection, "ns_block")
            ns_id = int(ns.namespace_id)
            _drop_namespace_via_pl(client, coll_id, ns_id)
            assert _fetch_namespace_name(client, coll_id, ns_id).startswith(RECYCLEBIN_PREFIX)
            assert _count_ltables(client, coll_id, ns_id) == 0

            # Simulate a stuck active ltable row (sync phase normally clears these).
            _insert_active_ltable_row(
                client,
                coll_id,
                ns_id,
                "active_ltable_blocker",
                blocker_lt_id,
            )
            assert _count_active_ltables(client, coll_id, ns_id) == 1

            # Wait one BG scan period; scheduler must skip while active ltable exists.
            time.sleep(35)
            assert _fetch_namespace_name(client, coll_id, ns_id) is not None, (
                "recyclebin namespace row must remain while active ltable exists"
            )

            _execute(
                client,
                f"DELETE FROM {_catalog_table(client, 'sdk_ltables')} WHERE ltable_id = {blocker_lt_id}",
            )
            assert _count_active_ltables(client, coll_id, ns_id) == 0

            _wait_until(
                lambda: _fetch_namespace_name(client, coll_id, ns_id) is None,
                desc="namespace row removed after blocker ltable deleted",
            )
        finally:
            client.delete_collection(name=collection.name)

    def test_async_no_kv_data_fast_cleanup(self, oceanbase_client):
        """Recyclebin namespace with zero kv rows should still be removed by BG."""
        client = oceanbase_client
        collection = _make_collection(client)
        coll_id = collection.id
        try:
            ns = _create_namespace(collection, "ns_empty_kv")
            ns_id = int(ns.namespace_id)
            assert _count_kv_data_rows(client, coll_id, ns_id) == 0

            _drop_namespace_via_pl(client, coll_id, ns_id)
            assert _fetch_namespace_name(client, coll_id, ns_id).startswith(RECYCLEBIN_PREFIX)

            _wait_until(
                lambda: _fetch_namespace_name(client, coll_id, ns_id) is None,
                desc="empty-kv recyclebin namespace removed",
            )
        finally:
            client.delete_collection(name=collection.name)

    def test_async_hot_table_cleaned_on_non_ss(self, oceanbase_client):
        """When hot_table exists, sync DROP (SN) leaves it; LTABLE_BG deletes it."""
        client = oceanbase_client
        if _is_ss_mode(client):
            pytest.skip("SS mode deletes hot_table in synchronous DROP_NAMESPACE")
        collection = _make_collection(client)
        coll_id = collection.id
        hot_err = _ensure_hot_table(client, coll_id)
        if hot_err is not None:
            pytest.fail(f"hot_table setup failed: {hot_err}")
        try:
            ns = _create_namespace(collection, "ns_hot_async")
            ns_id = int(ns.namespace_id)
            assert _count_hot_table_rows(client, coll_id, ns_id) == 1

            _drop_namespace_via_pl(client, coll_id, ns_id)
            assert _fetch_namespace_name(client, coll_id, ns_id).startswith(RECYCLEBIN_PREFIX)
            assert _count_hot_table_rows(client, coll_id, ns_id) == 1, "sync DROP must not remove hot_table on non-SS"

            def _hot_and_ns_gone():
                """Hot and ns gone."""
                return (
                    _count_hot_table_rows(client, coll_id, ns_id) == 0
                    and _fetch_namespace_name(client, coll_id, ns_id) is None
                )

            _wait_until(_hot_and_ns_gone, desc="hot_table and namespace removed by BG")
        finally:
            client.delete_collection(name=collection.name)

    def test_async_drop_history_recorded(self, oceanbase_client):
        """After successful BG delete, audit row appears in agent drop history."""
        client = oceanbase_client
        collection = _make_collection(client)
        coll_id = collection.id
        try:
            ns = _create_namespace(collection, "ns_history")
            ns_id = int(ns.namespace_id)
            _seed_kv_data(client, coll_id, ns_id, count=2)
            _drop_namespace_via_pl(client, coll_id, ns_id)

            def _bg_done():
                """Bg done."""
                return _fetch_namespace_name(client, coll_id, ns_id) is None

            _wait_until(_bg_done, desc="namespace removed for history check")

            history = _fetch_namespace_drop_history(client, coll_id, ns_id)
            if history is None:
                pytest.skip("__all_virtual_agent_drop_history not available")
            assert len(history) >= 1, "expected at least one history row"
            row = history[0]
            op = row.get("operation_type") or row.get("OPERATION_TYPE")
            status = row.get("operation_status") or row.get("OPERATION_STATUS")
            assert op == "NAMESPACE_DELETE", f"unexpected operation_type: {op!r}"
            assert status == "COMPLETED", f"unexpected operation_status: {status!r}"
        finally:
            client.delete_collection(name=collection.name)

    # ------------------------------------------------------------- #
    # 2. Namespace-name rename format                                #
    # ------------------------------------------------------------- #
    def test_recyclebin_name_format(self, oceanbase_client):
        """Test recyclebin name format."""
        client = oceanbase_client
        collection = _make_collection(client)
        try:
            ns = _create_namespace(collection, "fmt_ns")
            ns_id = int(ns.namespace_id)
            _drop_namespace_via_pl(client, collection.id, ns_id)
            new_name = _fetch_namespace_name(client, collection.id, ns_id)
            # Expected format: __recyclebin_<orig>_<ts_us>
            assert new_name.startswith("__recyclebin_fmt_ns_")
            ts_suffix = new_name[len("__recyclebin_fmt_ns_") :]
            assert ts_suffix.isdigit() and len(ts_suffix) >= 16, f"trailing timestamp_us looks malformed: {new_name!r}"
        finally:
            client.delete_collection(name=collection.name)

    # ------------------------------------------------------------- #
    # 3. Multiple ltables under one namespace all get deleted        #
    # ------------------------------------------------------------- #
    def test_multiple_ltables_all_deleted(self, oceanbase_client):
        """Test multiple ltables all deleted."""
        client = oceanbase_client
        collection = _make_collection(client)
        try:
            ns = _create_namespace(collection, "ns_multi")
            ns_id = int(ns.namespace_id)
            # Insert two extra ltable rows directly so we can verify bulk delete.
            lt_table = _catalog_table(client, "sdk_ltables")
            _execute(
                client,
                f"INSERT INTO {lt_table} (collection_id, namespace_id, ltable_name) "
                f"VALUES ('{collection.id}', {ns_id}, 'extra_lt_a'),"
                f"       ('{collection.id}', {ns_id}, 'extra_lt_b')",
            )
            assert _count_ltables(client, collection.id, ns_id) >= 3

            _drop_namespace_via_pl(client, collection.id, ns_id)

            assert _count_ltables(client, collection.id, ns_id) == 0
        finally:
            client.delete_collection(name=collection.name)

    # ------------------------------------------------------------- #
    # 4. Atomic rollback when an in-transaction DELETE fails         #
    # ------------------------------------------------------------- #
    def test_atomic_rollback_on_logic_schema_missing(self, oceanbase_client):
        """If we pre-DROP <coll>_logic_schema_table the in-trans DELETE on it
        fails with OB_TABLE_NOT_EXIST, which must roll back the namespace
        rename and the sdk_ltables delete that ran earlier in the same
        transaction."""
        client = oceanbase_client
        collection = _make_collection(client)
        coll_id = collection.id
        schema_tbl = NamespaceCollectionNames.logic_schema_table_name(coll_id)
        schema_tbl_q = f"`{client._server.database}`.`{schema_tbl}`"
        try:
            ns = _create_namespace(collection, "ns_rollback")
            ns_id = int(ns.namespace_id)
            orig_name = _fetch_namespace_name(client, coll_id, ns_id)
            assert orig_name == "ns_rollback"
            ltable_cnt_before = _count_ltables(client, coll_id, ns_id)
            assert ltable_cnt_before >= 1

            # Sabotage: drop the logic_schema_table so step 6 inside the
            # DROP_NAMESPACE transaction will fail with OB_TABLE_NOT_EXIST.
            _execute(client, f"DROP TABLE {schema_tbl_q}")

            with pytest.raises(Exception):
                _drop_namespace_via_pl(client, coll_id, ns_id)

            # Everything before the failed step must have been rolled back.
            after_name = _fetch_namespace_name(client, coll_id, ns_id)
            assert after_name == "ns_rollback", f"namespace_name must be rolled back to original, got {after_name!r}"
            assert _count_ltables(client, coll_id, ns_id) == ltable_cnt_before, "sdk_ltables delete must be rolled back"
        finally:
            # Recreate the schema table (empty) so cleanup_namespace_physical_tables
            # / delete_collection runs cleanly.
            with contextlib.suppress(Exception):
                _execute(
                    client,
                    f"CREATE TABLE IF NOT EXISTS {schema_tbl_q} ("
                    f"  namespace_id BIGINT UNSIGNED NOT NULL,"
                    f"  ltable_id BIGINT UNSIGNED NOT NULL,"
                    f"  schema_content JSON NOT NULL,"
                    f"  PRIMARY KEY (namespace_id, ltable_id)) "
                    f"PARTITION BY KEY(namespace_id) PARTITIONS 8",
                )
            client.delete_collection(name=collection.name)

    # ------------------------------------------------------------- #
    # 5. Idempotent: second drop on the same ns is a no-op success   #
    # ------------------------------------------------------------- #
    def test_drop_namespace_is_idempotent(self, oceanbase_client):
        """Test drop namespace is idempotent."""
        client = oceanbase_client
        collection = _make_collection(client)
        coll_id = collection.id
        try:
            ns = _create_namespace(collection, "ns_idem")
            ns_id = int(ns.namespace_id)
            _drop_namespace_via_pl(client, coll_id, ns_id)
            first_name = _fetch_namespace_name(client, coll_id, ns_id)
            assert first_name.startswith(RECYCLEBIN_PREFIX)

            # Second call must succeed without raising and must NOT re-rename
            # (the row name stays exactly the same — no double prefix).
            _drop_namespace_via_pl(client, coll_id, ns_id)
            second_name = _fetch_namespace_name(client, coll_id, ns_id)
            assert second_name == first_name, (
                f"second DROP_NAMESPACE must be no-op, got {second_name!r} vs {first_name!r}"
            )
            # And LEFT(name, 26) must not be '__recyclebin___recyclebin_'
            assert not second_name.startswith("__recyclebin___recyclebin_")
        finally:
            client.delete_collection(name=collection.name)

    # ------------------------------------------------------------- #
    # 6. Concurrent drop from two SDK connections                    #
    # ------------------------------------------------------------- #
    def test_concurrent_drop_from_two_sdks(self, oceanbase_client):
        """Two independent SDK connections race to DROP_NAMESPACE on the same
        (collection_id, namespace_id).

        Expected: exactly one transaction performs the rename + deletes; the
        other observes the __recyclebin_ row inside its own short trans and
        commits an empty trans (no exception, no double prefix).  Both calls
        return success."""
        client_a = oceanbase_client
        client_b = _second_oceanbase_client()

        collection = _make_collection(client_a, suffix="_conc")
        coll_id = collection.id
        try:
            ns = _create_namespace(collection, "ns_concurrent")
            ns_id = int(ns.namespace_id)

            results = {}
            barrier = threading.Barrier(2)

            def _worker(tag, c):
                """Worker."""
                try:
                    barrier.wait(timeout=10)
                    _drop_namespace_via_pl(c, coll_id, ns_id)
                    results[tag] = "ok"
                except Exception as exc:
                    results[tag] = f"err: {exc!r}"

            ta = threading.Thread(target=_worker, args=("A", client_a))
            tb = threading.Thread(target=_worker, args=("B", client_b))
            ta.start()
            tb.start()
            ta.join(timeout=30)
            tb.join(timeout=30)

            # Both calls must succeed (one does the work, the other no-ops).
            assert results.get("A") == "ok", f"A failed: {results.get('A')!r}"
            assert results.get("B") == "ok", f"B failed: {results.get('B')!r}"

            final_name = _fetch_namespace_name(client_a, coll_id, ns_id)
            assert final_name.startswith(RECYCLEBIN_PREFIX)
            assert "ns_concurrent" in final_name
            # The pattern must be __recyclebin_<orig>_<ts> — never doubled.
            assert not final_name.startswith("__recyclebin___recyclebin_"), (
                f"concurrent drops produced double-prefixed name: {final_name!r}"
            )
            # ltables and logic_schema must be empty (one of the trans cleared them).
            assert _count_ltables(client_a, coll_id, ns_id) == 0
            assert _count_logic_schema_rows(client_a, coll_id, ns_id) == 0
        finally:
            try:
                client_a.delete_collection(name=collection.name)
            finally:
                if hasattr(client_b, "close"):
                    with contextlib.suppress(Exception):
                        client_b.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
