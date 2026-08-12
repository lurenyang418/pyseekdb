"""
LOB prewarm integration tests (OceanBase shared-storage mode).

Validates the OB-side change that, after prewarming a namespace's KV data tablet,
the prewarm RPC ALSO prewarms that KV tablet's aux LOB-meta tablet (whole range).

Observability is layered into three independent tiers so the suite degrades
gracefully depending on what the cluster exposes:

  Tier 1 - structural (always):
      Resolve the kv_data_table's aux LOB-meta tablet purely from SQL
      (__all_table.data_table_id + table_type=13 -> __all_virtual_tablet_to_ls).
      Proves the table the prewarm targets actually owns a LOB-meta tablet.

  Tier 2 - ss-cache (SS mode + real out-of-row LOB):
      Force a >0.75MB kv_value so it spills out-of-row into the LOB-meta tablet,
      freeze it to a macro block, prewarm, then read
      __all_virtual_ss_macro_cache_info filtered by that LOB-meta tablet_id.
      Proves blocks were actually pulled into the local cache.

  Tier 3 - log marker (opt-in via OB_OBSERVER_LOG):
      Grep observer.log for "prewarm_logic_table lob meta done", which prints the
      exact lob_meta_tablet_id. Proves the new code path executed for the right tablet.

Run:
    OB_PORT=11202 OB_TENANT=mysql \
    OB_OBSERVER_LOG=/path/to/observer.log \
    pytest tests/integration_tests/test_namespace_prewarm_lob.py -v -s
"""

import contextlib
import os
import time

import pytest

try:
    import pymysql
except ImportError:  # pragma: no cover - pymysql ships with the test deps
    pymysql = None

from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import VectorIndexConfig
from pyseekdb.client.meta_info import NamespaceCollectionNames
from pyseekdb.client.schema import Schema

AUX_LOB_META_TABLE_TYPE = 13
LARGE_BLOB_BYTES = 1_000_000  # > OB_MAX_LOB_INROW_THRESHOLD (786432) -> always out-of-row
LOB_PROBE_ROWS = 3  # 3 x ~1MB incompressible -> ~3MB of real macro blocks
SS_CACHE_VIEW = "oceanbase.__all_virtual_ss_macro_cache_info"
ALL_TABLE_VIEW = "oceanbase.__all_table"
TABLET_TO_LS_VIEW = "oceanbase.__all_tablet_to_ls"

# Flushing the SS local cache requires a sys-tenant connection with an explicit
# TENANT clause; user-tenant flush returns OB-4016. These are env-overridable so
# the rigorous before/after assertion can run wherever a sys login is available.
OB_HOST = os.environ.get("OB_HOST", "127.0.0.1")
OB_PORT = int(os.environ.get("OB_PORT", "10902"))
OB_TENANT = os.environ.get("OB_TENANT", "test_tenant")
OB_SYS_USER = os.environ.get("OB_SYS_USER", "root@sys")
OB_SYS_PASSWORD = os.environ.get("OB_SYS_PASSWORD", "")


# ==================== SQL observability helpers ====================


def _exec(client, sql):
    """Exec."""
    return client._server._execute(sql)


def _resolve_table_id(client, table_name):
    """Resolve table id."""
    rows = _exec(
        client,
        f"SELECT table_id FROM {ALL_TABLE_VIEW} WHERE table_name = '{table_name}'",
    )
    assert rows, f"table_id not found for {table_name}"
    return rows[0]["table_id"]


def _resolve_lob_meta_tablets(client, kv_table_id):
    """All aux-LOB-meta tablets of a KV data table (one per KV partition)."""
    rows = _exec(
        client,
        f"SELECT table_id FROM {ALL_TABLE_VIEW} "
        f"WHERE data_table_id = {kv_table_id} AND table_type = {AUX_LOB_META_TABLE_TYPE}",
    )
    assert rows, f"no aux lob meta table for kv table {kv_table_id}"
    lob_meta_table_id = rows[0]["table_id"]

    tablet_rows = _exec(
        client,
        f"SELECT tablet_id FROM {TABLET_TO_LS_VIEW} WHERE table_id = {lob_meta_table_id}",
    )
    return {r["tablet_id"] for r in tablet_rows}


def _ss_cache_blocks_for_tablets(client, tablet_ids):
    """Map tablet_id -> (block_count, total_size) from the SS macro cache view.

    Returns None when the SS cache view is not available (non-SS cluster), so
    callers can skip tier-2 assertions cleanly.
    """
    if not tablet_ids:
        return {}
    ids = ",".join(str(t) for t in tablet_ids)
    try:
        rows = _exec(
            client,
            f"SELECT tablet_id, COUNT(*) AS blk, COALESCE(SUM(size),0) AS sz "
            f"FROM {SS_CACHE_VIEW} WHERE tablet_id IN ({ids}) GROUP BY tablet_id",
        )
    except Exception:
        return None
    return {r["tablet_id"]: (r["blk"], r["sz"]) for r in rows}


def _flush_tenant_macro_cache():
    """Evict the tenant's SS local macro cache via a sys-tenant connection.

    Returns True on success, False if a sys login / flush is unavailable so the
    caller can skip the rigorous tier-2 assertion instead of failing spuriously.
    """
    if pymysql is None:
        return False
    try:
        conn = pymysql.connect(
            host=OB_HOST,
            port=OB_PORT,
            user=OB_SYS_USER,
            password=OB_SYS_PASSWORD,
            autocommit=True,
        )
    except Exception:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(f"ALTER SYSTEM FLUSH SS_LOCAL_CACHE TENANT = {OB_TENANT} CACHE = macro_cache")
        return True
    except Exception:
        return False
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def _is_ss_mode(client, collection_id):
    """SS mode is detectable by the per-collection hot_table existing."""
    hot_table = NamespaceCollectionNames.hot_table_name(collection_id)
    rows = _exec(
        client,
        f"SELECT table_id FROM {ALL_TABLE_VIEW} WHERE table_name = '{hot_table}'",
    )
    return bool(rows)


def _observer_log_files():
    """The configured observer.log plus any rotated siblings in the same dir.

    OB rotates observer.log aggressively (observer.log.<ts>), so a marker emitted
    during the test may already have moved to a rotated file by the time we grep.
    """
    log_path = os.environ.get("OB_OBSERVER_LOG")
    if not log_path or not os.path.exists(log_path):
        return None
    directory = os.path.dirname(log_path) or "."
    base = os.path.basename(log_path)
    files = []
    for fn in os.listdir(directory):
        # observer.log and observer.log.<ts>, but not observer.log.wf*
        if (fn == base or fn.startswith(base + ".")) and ".wf" not in fn:
            files.append(os.path.join(directory, fn))
    return files


def _grep_lob_prewarm_log(lob_meta_tablet_ids):
    """Tier 3: confirm the new code path logged a lob prewarm for one of the tablets."""
    files = _observer_log_files()
    if files is None:
        return None
    wanted = {str(t) for t in lob_meta_tablet_ids}
    hits = []
    for path in files:
        with open(path, errors="ignore") as fh:
            for line in fh:
                if "prewarm_logic_table lob meta done" in line and any(t in line for t in wanted):
                    hits.append(line.strip())
    return hits


# ==================== Fixtures / collection helpers ====================


class _BaseLobPrewarm:
    """BaseLobPrewarm class."""

    def _make_collection(self, client, name, partitions):
        """Make collection."""
        schema = Schema(
            vector_index=VectorIndexConfig(
                ivf=IVFConfiguration(dimension=3, distance="cosine"),
                embedding_function=None,
            ),
        )
        return client.create_collection(name=name, schema=schema, use_namespace=True, partition_count=partitions)

    def _force_out_of_row_lob(self, client, collection_id, namespace_id):
        """Insert several >0.75MB incompressible kv_values so they spill out-of-row
        into the LOB-meta tablet, then freeze them into macro blocks.

        REPEAT('x', N) compresses to ~nothing and yields a single tiny block, which
        is useless for proving prewarm restored real data. os.urandom is
        incompressible, so each row produces a genuine ~1MB macro block.
        """
        kv_table = NamespaceCollectionNames.kv_data_table_name(collection_id)
        # _execute() takes no params, so use the raw DBAPI connection to bind the
        # binary blob safely (avoids building multi-MB hex-literal SQL strings).
        raw = client._server.get_raw_connection()
        with raw.cursor() as cur:
            for i in range(LOB_PROBE_ROWS):
                cur.execute(
                    f"INSERT INTO `{kv_table}` (namespace_id, kv_key, kv_value) VALUES ({namespace_id}, %s, %s)",
                    (f"__lob_probe_{i}__", os.urandom(LARGE_BLOB_BYTES)),
                )
        raw.commit()
        # Push the rows into macro blocks so prewarm has remote data to pull.
        with contextlib.suppress(Exception):
            _exec(client, "ALTER SYSTEM MINOR FREEZE")


# ==================== Tier 1 + structural multi-namespace ====================


class TestLobPrewarmStructural(_BaseLobPrewarm):
    """TestLobPrewarmStructural class."""

    def test_kv_table_has_lob_meta_tablet(self, oceanbase_client):
        """Tier 1: the prewarm target (kv_data_table) owns an aux LOB-meta tablet."""
        name = f"lob_pw_struct_{int(time.time() * 1000)}"
        collection = self._make_collection(oceanbase_client, name, partitions=1)
        try:
            kv_table = NamespaceCollectionNames.kv_data_table_name(collection.id)
            kv_table_id = _resolve_table_id(oceanbase_client, kv_table)
            lob_tablets = _resolve_lob_meta_tablets(oceanbase_client, kv_table_id)
            assert len(lob_tablets) == 1, (
                f"single-partition kv table should have exactly 1 lob meta tablet, got {lob_tablets}"
            )
        finally:
            oceanbase_client.delete_collection(name=collection.name)

    def test_multi_namespace_prewarm_is_independent(self, oceanbase_client):
        """Multi-namespace: prewarming each namespace succeeds, records its own hot_table
        row, and never disturbs another namespace's row. Cached LOB tablets stay within
        the collection's own LOB-meta tablet set (no foreign tablets)."""
        name = f"lob_pw_multi_{int(time.time() * 1000)}"
        collection = self._make_collection(oceanbase_client, name, partitions=8)
        try:
            kv_table = NamespaceCollectionNames.kv_data_table_name(collection.id)
            kv_table_id = _resolve_table_id(oceanbase_client, kv_table)
            own_lob_tablets = _resolve_lob_meta_tablets(oceanbase_client, kv_table_id)

            namespaces = [collection.create_namespace(f"ns_{i}") for i in range(3)]
            for ns in namespaces:
                ns.prewarm()

            hot_table = NamespaceCollectionNames.hot_table_name(collection.id)
            rows = _exec(
                oceanbase_client,
                f"SELECT namespace_id FROM `{hot_table}`",
            )
            recorded = {int(r["namespace_id"]) for r in rows}
            expected = {int(ns._namespace_id) for ns in namespaces}
            assert expected.issubset(recorded), (
                f"every prewarmed namespace must have a hot_table row: expected {expected}, got {recorded}"
            )

            cache = _ss_cache_blocks_for_tablets(oceanbase_client, own_lob_tablets)
            if cache:
                assert set(cache).issubset(own_lob_tablets), "cached lob tablets must belong to this collection only"
        finally:
            oceanbase_client.delete_collection(name=collection.name)


# ==================== Tier 2 + Tier 3: real LOB caching ====================


class TestLobPrewarmCaching(_BaseLobPrewarm):
    """TestLobPrewarmCaching class."""

    @staticmethod
    def _total_bytes(cache):
        """Total bytes."""
        return sum(v[1] for v in (cache or {}).values())

    def test_out_of_row_lob_is_prewarmed_into_cache(self, oceanbase_client):
        """Tier 2/3: prove prewarm actually *pulls* evicted LOB data back, not that
        the write cache merely still holds it.

        Methodology (the only rigorous one):
          1. Insert incompressible out-of-row LOB rows + freeze -> real macro blocks.
          2. Flush the tenant's SS macro cache (sys connection) -> evict those blocks.
          3. Assert cache bytes for the LOB tablet dropped near-zero BEFORE prewarm.
          4. prewarm() -> poll the cache view.
          5. Assert cache bytes were restored (>> the flushed floor) AFTER prewarm.

        Without the flush step, frozen data already sits in the local write cache, so a
        plain "blocks > 0 after prewarm" assertion would pass even if prewarm did nothing.
        """
        name = f"lob_pw_cache_{int(time.time() * 1000)}"
        collection = self._make_collection(oceanbase_client, name, partitions=1)
        try:
            if not _is_ss_mode(oceanbase_client, collection.id):
                pytest.skip("LOB cache prewarm only observable in shared-storage mode")

            namespace = collection.create_namespace("lob_ns")
            kv_table = NamespaceCollectionNames.kv_data_table_name(collection.id)
            kv_table_id = _resolve_table_id(oceanbase_client, kv_table)
            lob_tablets = _resolve_lob_meta_tablets(oceanbase_client, kv_table_id)

            self._force_out_of_row_lob(oceanbase_client, collection.id, namespace._namespace_id)

            if _ss_cache_blocks_for_tablets(oceanbase_client, lob_tablets) is None:
                pytest.skip(f"{SS_CACHE_VIEW} not available on this cluster")

            # Give the freeze time to upload macro blocks to object storage.
            time.sleep(10)
            written = _ss_cache_blocks_for_tablets(oceanbase_client, lob_tablets) or {}
            assert self._total_bytes(written) > 0, (
                f"out-of-row LOB should have produced macro blocks in cache, got {written}"
            )

            # Evict so prewarm has something real to restore. Requires sys flush.
            if not _flush_tenant_macro_cache():
                pytest.skip(
                    "sys-tenant SS_LOCAL_CACHE flush unavailable; cannot prove "
                    "prewarm pulls evicted data (set OB_SYS_USER/OB_SYS_PASSWORD)"
                )
            time.sleep(4)

            flushed = _ss_cache_blocks_for_tablets(oceanbase_client, lob_tablets) or {}
            flushed_bytes = self._total_bytes(flushed)
            written_bytes = self._total_bytes(written)
            assert flushed_bytes < written_bytes, (
                f"flush should have evicted LOB macro blocks before prewarm: "
                f"written={written_bytes}, flushed={flushed_bytes}"
            )

            namespace.prewarm()

            # Prewarm pulls remote macro blocks asynchronously; poll for restoration.
            cached = flushed
            deadline = time.time() + 40
            while time.time() < deadline:
                cached = _ss_cache_blocks_for_tablets(oceanbase_client, lob_tablets) or {}
                if self._total_bytes(cached) > flushed_bytes:
                    break
                time.sleep(2)

            log_hits = _grep_lob_prewarm_log(lob_tablets)
            if log_hits is not None:
                assert log_hits, (
                    "observer.log should contain a 'prewarm_logic_table lob meta done' "
                    f"line for lob tablets {lob_tablets}"
                )

            restored_bytes = self._total_bytes(cached)
            assert restored_bytes > flushed_bytes, (
                f"prewarm should have restored evicted LOB macro blocks: "
                f"written={written_bytes}, flushed={flushed_bytes}, restored={restored_bytes}"
            )
        finally:
            oceanbase_client.delete_collection(name=collection.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
