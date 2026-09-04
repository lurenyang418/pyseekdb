"""Integration tests for the seekdb-specific database fork API."""

from __future__ import annotations

import contextlib
import logging
import time

import pytest

import pyseekdb

logger = logging.getLogger(__name__)


class TestDatabaseFork:
    """Test ``Client.fork_database()`` against configured server backends."""

    @staticmethod
    def _unique_name(prefix: str) -> str:
        return f"{prefix}_{int(time.time() * 1000)}"

    @staticmethod
    def _drop_database(client: pyseekdb.Client, name: str) -> None:
        """Clean up a fork through the raw SQL escape hatch used by the test."""
        with contextlib.suppress(Exception):
            client._execute(f"DROP DATABASE IF EXISTS `{name}`")

    @staticmethod
    def _require_fork(client: pyseekdb.Client) -> None:
        try:
            supported = client.supports_fork_database
        except Exception:
            logger.exception("Failed to check if fork is enabled")
            supported = False
        if not supported:
            pytest.skip("Fork database is not enabled for this database")

    def test_fork_database_returns_bound_client(self, db_client: pyseekdb.Client):
        """Forking the bound database returns a directly usable target client."""
        self._require_fork(db_client)
        destination_name = self._unique_name("test_forkdb_dst")
        forked = None

        try:
            forked = db_client.fork_database(destination_name)

            assert isinstance(forked, pyseekdb.Client)
            assert forked.database == destination_name
            assert forked.host == db_client.host
            assert forked.tenant == db_client.tenant
            assert forked.ping()
        finally:
            if forked is not None:
                forked.destroy()
            else:
                self._drop_database(db_client, destination_name)

    def test_fork_database_destination_already_exists(self, db_client: pyseekdb.Client):
        """Forking to an existing database raises a clear error."""
        self._require_fork(db_client)
        destination_name = self._unique_name("test_forkdb_dup")

        try:
            db_client._execute(f"CREATE DATABASE `{destination_name}`")
            with pytest.raises(ValueError, match="already exists"):
                db_client.fork_database(destination_name)
        finally:
            self._drop_database(db_client, destination_name)

    def test_fork_database_preserves_table_data_and_is_independent(self, db_client: pyseekdb.Client):
        """A fork contains source data and subsequent writes remain independent."""
        self._require_fork(db_client)
        destination_name = self._unique_name("test_forkdb_data_dst")
        table_name = self._unique_name("fork_table")
        forked = None

        try:
            db_client._execute(f"CREATE TABLE `{table_name}` (id INT PRIMARY KEY, val VARCHAR(100))")
            db_client._execute(f"INSERT INTO `{table_name}` VALUES (1, 'original')")
            forked = db_client.fork_database(destination_name)

            forked._execute(f"INSERT INTO `{table_name}` VALUES (2, 'forked_only')")
            source_rows = db_client._execute(f"SELECT COUNT(*) AS cnt FROM `{table_name}`")
            destination_rows = forked._execute(f"SELECT COUNT(*) AS cnt FROM `{table_name}`")

            assert source_rows[0]["cnt"] == 1
            assert destination_rows[0]["cnt"] == 2
        finally:
            if forked is not None:
                forked.destroy()
            else:
                self._drop_database(db_client, destination_name)
            with contextlib.suppress(Exception):
                db_client._execute(f"DROP TABLE IF EXISTS `{table_name}`")

    def test_fork_database_multiple_times(self, db_client: pyseekdb.Client):
        """The same bound source database can be forked more than once."""
        self._require_fork(db_client)
        destination_names = [
            self._unique_name("test_forkdb_multi_a"),
            self._unique_name("test_forkdb_multi_b"),
        ]
        forked_clients = []

        try:
            forked_clients = [db_client.fork_database(name) for name in destination_names]
            assert [client.database for client in forked_clients] == destination_names
        finally:
            for client in forked_clients:
                client.destroy()
            for name in destination_names[len(forked_clients) :]:
                self._drop_database(db_client, name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
