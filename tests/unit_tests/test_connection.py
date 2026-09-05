"""Unit tests for the synchronous connection/executor mixin."""

from unittest.mock import MagicMock, patch

from pyseekdb.client.client import Client


def test_client_uses_extracted_mysql_connection_mixin() -> None:
    connection = MagicMock()
    connection.open = True
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.description = [("pyseekdb_ping",)]
    cursor.fetchall.return_value = [{"pyseekdb_ping": 1}]

    client = Client(host="localhost", database="demo")
    client._use_catalog_database = MagicMock()

    with patch("pyseekdb.client.connection.pymysql.connect", return_value=connection) as connect:
        assert client.ping()

    connect.assert_called_once()
    assert connect.call_args.kwargs["database"] == "demo"
    assert connect.call_args.kwargs["user"] == "root@sys"
    client._use_catalog_database.assert_called_once()

    client.close()
    connection.close.assert_called_once()
    assert not client.is_connected()


def test_connection_executor_supports_parameterized_sql() -> None:
    connection = MagicMock()
    connection.open = True
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.description = [("value",)]
    cursor.fetchall.return_value = [{"value": 1}]

    client = Client(host="localhost", database="demo")
    client._use_catalog_database = MagicMock()

    with patch("pyseekdb.client.connection.pymysql.connect", return_value=connection):
        assert client._execute("SELECT %s AS value", [1]) == [{"value": 1}]

    cursor.execute.assert_called_once_with("SELECT %s AS value", (1,))
