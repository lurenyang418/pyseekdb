"""Public synchronous client for remote seekdb and OceanBase servers."""

from __future__ import annotations

import os

from .client_base import BaseClient
from .connection import MySQLConnectionMixin
from .fork import build_drop_database_sql, execute_database_fork
from .kernel_errors import namespace_kernel_error_guard


class Client(MySQLConnectionMixin, BaseClient):
    """Synchronous client bound to one existing remote database.

    ``Client`` manages collections in the database supplied at construction time.
    Database provisioning is intentionally left to deployment/DBA tooling.  The
    seekdb-specific :meth:`fork_database` operation is exposed as an advanced
    capability and returns a client already bound to the forked database.
    """

    def __init__(
        self,
        host: str,
        port: int = 2881,
        tenant: str = "sys",
        database: str = "test",
        user: str = "root",
        password: str = "",
        charset: str = "utf8mb4",
        **kwargs,
    ) -> None:
        if host is None:
            raise ValueError(
                "`host=` is required: embedded mode was removed in pyseekdb 2.0. "
                "Provide host/port (and user/password) to connect to a remote seekdb/OceanBase server."
            )
        super().__init__(
            host=host,
            port=port,
            tenant=tenant,
            database=database,
            user=user,
            password=password or os.environ.get("SEEKDB_PASSWORD", ""),
            charset=charset,
            **kwargs,
        )

    @namespace_kernel_error_guard
    def _namespace_prewarm(
        self,
        collection_id: str | None,
        collection_name: str,
        namespace_id: str,
        namespace_name: str,
        **kwargs,
    ) -> None:
        """Prewarm the namespace logical table to reduce first-query latency."""
        ltable_id = self._resolve_namespace_ltable_id(collection_id, namespace_id)
        self._use_catalog_database()
        self._set_session_ns_context(
            collection_id=collection_id,
            namespace_id=int(namespace_id),
            ltable_id=ltable_id,
        )
        sql = f"CALL DBMS_LOGIC_TABLE.PREWARM('{collection_id}', {namespace_id})"
        self._execute(sql)

    def fork_database(self, destination_name: str) -> Client:
        """Fork this client's database and return a client bound to the copy.

        The source database is always ``self.database``.  The destination must not
        exist and is created only when the connected seekdb backend supports the
        ``FORK DATABASE`` statement.
        """
        execute_database_fork(self, destination_name)
        forked = type(self)(
            host=self.host,
            port=self.port,
            tenant=self.tenant,
            database=destination_name,
            user=self.user,
            password=self.password,
            charset=self.charset,
            **self.kwargs,
        )
        forked._fork_parent = self
        return forked

    def destroy(self) -> None:
        """Destroy this client's forked database and close its connection.

        Only clients returned by :meth:`fork_database` can be destroyed this way;
        ordinary database provisioning and deletion remain outside the SDK.
        """
        parent = getattr(self, "_fork_parent", None)
        if parent is None:
            raise ValueError("Only a client returned by fork_database() can destroy its database")
        self.close()
        parent._execute(build_drop_database_sql(self.database))
        self._fork_parent = None

    def __repr__(self) -> str:
        """Return the developer-readable representation."""
        status = "connected" if self.is_connected() else "disconnected"
        return f"<{type(self).__name__} {self.full_user}@{self.host}:{self.port}/{self.database} status={status}>"


__all__ = ["Client"]
