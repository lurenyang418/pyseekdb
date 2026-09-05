"""Synchronous MySQL connection and SQL execution support."""

from __future__ import annotations

import logging
import os
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from .kernel_errors import maybe_reraise_friendly_kernel_error
from .sql_utils import is_query_sql

logger = logging.getLogger(__name__)


class MySQLConnectionMixin:
    """Provide lazy PyMySQL connection and cursor execution methods.

    The mixin deliberately knows nothing about collections. It expects the
    concrete client to provide ``_use_catalog_database()`` when a connection
    is established, which keeps catalog/session policy in the client layer.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2881,
        tenant: str = "sys",
        database: str = "test",
        user: str = "root",
        password: str = "",
        charset: str = "utf8mb4",
        **kwargs: Any,
    ) -> None:
        self.host = host
        self.port = port
        self.tenant = tenant
        self.database = database
        self.user = user
        self.password = password
        self.charset = charset
        self.kwargs = kwargs
        self.full_user = f"{user}@{tenant}"
        self._connection: pymysql.Connection | None = None
        logger.debug("Initialize MySQL client: %s@%s:%s/%s", self.full_user, self.host, self.port, self.database)

    def _ensure_connection(self) -> pymysql.Connection:
        """Ensure a PyMySQL connection is established."""
        if self._connection is None or not self._connection.open:
            self._connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.full_user,
                password=self.password,
                database=self.database,
                charset=self.charset,
                cursorclass=DictCursor,
                autocommit=True,
                **self.kwargs,
            )
            logger.info("Connected to remote server: %s:%s/%s", self.host, self.port, self.database)
            try:
                self._use_catalog_database()
            except Exception as exc:
                logger.warning("Failed to initialize catalog database on connect: %s", exc)
        return self._connection

    def _cleanup(self) -> None:
        """Close the connection and release its resources."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("Connection closed: %s:%s/%s", self.host, self.port, self.database)

    def is_connected(self) -> bool:
        """Return whether the lazy connection is currently open."""
        return self._connection is not None and self._connection.open

    def get_raw_connection(self) -> pymysql.Connection:
        """Return the underlying connection, opening it if necessary."""
        return self._ensure_connection()

    def _normalize_row(self, row: Any, cursor_description: Any | None = None) -> dict[str, Any]:
        """Normalize a cursor row to a mapping when column metadata is available."""
        if isinstance(row, dict):
            return row
        if cursor_description is not None:
            return {column[0]: row[index] for index, column in enumerate(cursor_description)}
        return dict(row) if hasattr(row, "_asdict") else row

    def _execute_query_with_cursor(
        self,
        conn: pymysql.Connection,
        sql: str,
        params: list[Any],
        use_context_manager: bool = True,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized query and normalize its result rows."""
        if os.environ.get("PYSEEKDB_PRINT_SQL", "").lower() in ("1", "true", "yes"):
            print(f"[pyseekdb SQL] {sql}  -- params={params}", flush=True)
        try:
            if use_context_manager:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    if not self._should_fetch_results(cursor, sql):
                        return []
                    return [self._normalize_row(row, cursor.description) for row in cursor.fetchall()]

            cursor = conn.cursor()
            try:
                cursor.execute(sql, params)
                if not self._should_fetch_results(cursor, sql):
                    return []
                return [self._normalize_row(row, cursor.description) for row in cursor.fetchall()]
            finally:
                cursor.close()
        except Exception as exc:
            maybe_reraise_friendly_kernel_error(exc)
            raise

    def _use_context_manager_for_cursor(self) -> bool:
        """Return whether cursor context managers should be used."""
        return True

    def _should_fetch_results(self, cursor: Any, sql: str) -> bool:
        """Return whether a statement is expected to yield result rows."""
        description = getattr(cursor, "description", None)
        return description is not None or is_query_sql(sql)

    def _execute(self, sql: str, params: list[Any] | tuple[Any, ...] | None = None) -> Any:
        """Execute a SQL statement and return rows for query statements."""
        if os.environ.get("PYSEEKDB_PRINT_SQL", "").lower() in ("1", "true", "yes"):
            print(f"[pyseekdb SQL] {sql} -- params={params}", flush=True)
        conn = self._ensure_connection()
        try:
            if self._use_context_manager_for_cursor():
                with conn.cursor() as cursor:
                    if params is None:
                        cursor.execute(sql)
                    else:
                        cursor.execute(sql, tuple(params))
                    return cursor.fetchall() if self._should_fetch_results(cursor, sql) else None

            cursor = conn.cursor()
            try:
                if params is None:
                    cursor.execute(sql)
                else:
                    cursor.execute(sql, tuple(params))
                return cursor.fetchall() if self._should_fetch_results(cursor, sql) else None
            finally:
                cursor.close()
        except Exception as exc:
            maybe_reraise_friendly_kernel_error(exc)
            raise


__all__ = ["MySQLConnectionMixin"]
