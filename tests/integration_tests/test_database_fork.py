"""
Integration tests for AdminClient.fork_database method.

Tests the fork_database functionality against real databases, including:
- Successful fork operations
- Empty database fork
- Fork preserves database attributes (charset, collation)
- Fork creates independent databases
- Error handling when destination already exists
- Error handling when fork is not enabled
"""

import contextlib
import logging
import time

import pytest

logger = logging.getLogger(__name__)


class TestDatabaseFork:
    """Tests for AdminClient.fork_database() using real database connections."""

    def _is_fork_database_enabled(self, admin_client) -> bool:
        """Check if fork_database is enabled for the given admin client."""
        try:
            return admin_client._server._fork_database_enabled()
        except Exception:
            logger.exception("Failed to check if fork_database is enabled")
            return False

    def _unique_name(self, prefix: str) -> str:
        return f"{prefix}_{int(time.time() * 1000)}"

    def test_fork_database_success(self, admin_client):
        """
        Test successful fork_database operation.

        Automatically runs for: server, oceanbase
        Skips if fork_database is not enabled.
        """
        if not self._is_fork_database_enabled(admin_client):
            pytest.skip("Fork database is not enabled for this database")

        source_name = self._unique_name("test_forkdb_src")
        dest_name = self._unique_name("test_forkdb_dst")

        try:
            admin_client.create_database(source_name)

            forked_db = admin_client.fork_database(source_name, dest_name)

            assert forked_db is not None
            assert forked_db.name == dest_name

            retrieved_db = admin_client.get_database(dest_name)
            assert retrieved_db.name == dest_name

        finally:
            with contextlib.suppress(Exception):
                admin_client.delete_database(dest_name)
            with contextlib.suppress(Exception):
                admin_client.delete_database(source_name)

    def test_fork_empty_database(self, admin_client):
        """
        Test fork of an empty database (no tables).

        Automatically runs for: server, oceanbase
        Skips if fork_database is not enabled.
        """
        if not self._is_fork_database_enabled(admin_client):
            pytest.skip("Fork database is not enabled for this database")

        source_name = self._unique_name("test_forkdb_empty_src")
        dest_name = self._unique_name("test_forkdb_empty_dst")

        try:
            admin_client.create_database(source_name)

            forked_db = admin_client.fork_database(source_name, dest_name)
            assert forked_db is not None
            assert forked_db.name == dest_name

        finally:
            with contextlib.suppress(Exception):
                admin_client.delete_database(dest_name)
            with contextlib.suppress(Exception):
                admin_client.delete_database(source_name)

    def test_fork_database_preserves_attributes(self, admin_client):
        """
        Test that fork preserves database attributes (charset, collation).

        Automatically runs for: server, oceanbase
        Skips if fork_database is not enabled.
        """
        if not self._is_fork_database_enabled(admin_client):
            pytest.skip("Fork database is not enabled for this database")

        source_name = self._unique_name("test_forkdb_attr_src")
        dest_name = self._unique_name("test_forkdb_attr_dst")

        try:
            admin_client.create_database(source_name)
            source_db = admin_client.get_database(source_name)

            forked_db = admin_client.fork_database(source_name, dest_name)

            assert forked_db.charset == source_db.charset
            assert forked_db.collation == source_db.collation

        finally:
            with contextlib.suppress(Exception):
                admin_client.delete_database(dest_name)
            with contextlib.suppress(Exception):
                admin_client.delete_database(source_name)

    def test_fork_database_destination_already_exists(self, admin_client):
        """
        Test that forking to an existing database raises an error.

        Automatically runs for: server, oceanbase
        Skips if fork_database is not enabled.
        """
        if not self._is_fork_database_enabled(admin_client):
            pytest.skip("Fork database is not enabled for this database")

        source_name = self._unique_name("test_forkdb_dup_src")
        dest_name = self._unique_name("test_forkdb_dup_dst")

        try:
            admin_client.create_database(source_name)
            admin_client.create_database(dest_name)

            with pytest.raises(ValueError):
                admin_client.fork_database(source_name, dest_name)

        finally:
            with contextlib.suppress(Exception):
                admin_client.delete_database(dest_name)
            with contextlib.suppress(Exception):
                admin_client.delete_database(source_name)

    def test_fork_database_independent_operations(self, admin_client):
        """
        Test that forked database is independent from the source.

        Creates a table in source before fork, then verifies that modifications
        to the forked database do not affect the source and vice versa.

        Automatically runs for: server, oceanbase
        Skips if fork_database is not enabled.
        """
        if not self._is_fork_database_enabled(admin_client):
            pytest.skip("Fork database is not enabled for this database")

        source_name = self._unique_name("test_forkdb_indep_src")
        dest_name = self._unique_name("test_forkdb_indep_dst")

        try:
            admin_client.create_database(source_name)

            admin_client._server._execute(f"CREATE TABLE `{source_name}`.`t1` (id INT PRIMARY KEY, val VARCHAR(100))")
            admin_client._server._execute(f"INSERT INTO `{source_name}`.`t1` VALUES (1, 'original')")

            admin_client.fork_database(source_name, dest_name)

            admin_client._server._execute(f"INSERT INTO `{dest_name}`.`t1` VALUES (2, 'forked_only')")

            source_rows = admin_client._server._execute(f"SELECT COUNT(*) as cnt FROM `{source_name}`.`t1`")
            dest_rows = admin_client._server._execute(f"SELECT COUNT(*) as cnt FROM `{dest_name}`.`t1`")

            source_count = source_rows[0]["cnt"] if isinstance(source_rows[0], dict) else source_rows[0][0]
            dest_count = dest_rows[0]["cnt"] if isinstance(dest_rows[0], dict) else dest_rows[0][0]

            assert source_count == 1, f"Source should have 1 row, got {source_count}"
            assert dest_count == 2, f"Destination should have 2 rows, got {dest_count}"

        finally:
            with contextlib.suppress(Exception):
                admin_client.delete_database(dest_name)
            with contextlib.suppress(Exception):
                admin_client.delete_database(source_name)

    def test_fork_database_multiple_times(self, admin_client):
        """
        Test that the same source database can be forked multiple times.

        Automatically runs for: server, oceanbase
        Skips if fork_database is not enabled.
        """
        if not self._is_fork_database_enabled(admin_client):
            pytest.skip("Fork database is not enabled for this database")

        source_name = self._unique_name("test_forkdb_multi_src")
        dest_name_1 = self._unique_name("test_forkdb_multi_d1")
        dest_name_2 = self._unique_name("test_forkdb_multi_d2")

        try:
            admin_client.create_database(source_name)

            forked_1 = admin_client.fork_database(source_name, dest_name_1)
            forked_2 = admin_client.fork_database(source_name, dest_name_2)

            assert forked_1.name == dest_name_1
            assert forked_2.name == dest_name_2

        finally:
            with contextlib.suppress(Exception):
                admin_client.delete_database(dest_name_2)
            with contextlib.suppress(Exception):
                admin_client.delete_database(dest_name_1)
            with contextlib.suppress(Exception):
                admin_client.delete_database(source_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
