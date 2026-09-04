"""Tests for the public synchronous client boundary."""

from unittest.mock import MagicMock, patch

import pytest

import pyseekdb
from pyseekdb.client.fork import execute_database_fork


def test_client_is_a_concrete_database_bound_client() -> None:
    """The public Client is not a proxy and does not expose database CRUD."""
    client = pyseekdb.Client(host="localhost", database="demo")

    assert type(client) is pyseekdb.Client
    assert client.database == "demo"
    assert not hasattr(client, "create_database")
    assert not hasattr(client, "get_database")
    assert not hasattr(client, "delete_database")
    assert not hasattr(client, "list_databases")
    assert "AdminClient" not in pyseekdb.__all__


def test_client_fork_returns_client_bound_to_destination() -> None:
    """Forking returns a directly usable client for the new database."""
    client = pyseekdb.Client(
        host="localhost",
        port=2881,
        tenant="sys",
        database="source",
        user="root",
        password="secret",  # noqa: S106
        connect_timeout=3,
    )

    with patch("pyseekdb.client.sync_client.execute_database_fork") as execute:
        forked = client.fork_database("destination")

    execute.assert_called_once_with(client, "destination")
    assert type(forked) is pyseekdb.Client
    assert forked.database == "destination"
    assert forked.host == client.host
    assert forked.port == client.port
    assert forked.tenant == client.tenant
    assert forked.user == client.user
    assert forked.password == client.password
    assert forked.kwargs == client.kwargs


def test_only_forked_client_can_destroy_its_database() -> None:
    client = pyseekdb.Client(host="localhost", database="source")

    with pytest.raises(ValueError, match="Only a client returned by fork_database"):
        client.destroy()

    with patch("pyseekdb.client.sync_client.execute_database_fork"):
        forked = client.fork_database("destination")
    client._execute = MagicMock()

    forked.destroy()

    client._execute.assert_called_once_with("DROP DATABASE IF EXISTS `destination`")
    assert forked._fork_parent is None


def test_execute_database_fork_uses_bound_database() -> None:
    """The low-level fork operation always uses the client's source database."""

    class FakeClient:
        database = "source"
        supports_fork_database = True

        def __init__(self) -> None:
            self.sql = None

        def _execute(self, sql: str) -> None:
            self.sql = sql

    client = FakeClient()

    execute_database_fork(client, "destination")

    assert client.sql == "FORK DATABASE `source` TO `destination`"


def test_execute_database_fork_rejects_unsupported_backend() -> None:
    class FakeClient:
        database = "source"
        supports_fork_database = False

    with pytest.raises(ValueError, match=r"requires seekdb >= 1\.2\.0"):
        execute_database_fork(FakeClient(), "destination")


@pytest.mark.parametrize("destination", ["", "bad-name", "source"])
def test_execute_database_fork_rejects_invalid_destination(destination: str) -> None:
    """Invalid or identical fork destinations fail before SQL execution."""

    class FakeClient:
        database = "source"
        supports_fork_database = True

        def _execute(self, sql: str) -> None:
            raise AssertionError(f"SQL should not execute: {sql}")

    with pytest.raises((TypeError, ValueError)):
        execute_database_fork(FakeClient(), destination)
