"""Public synchronous client for remote seekdb and OceanBase servers."""

from __future__ import annotations

import os

from .client_seekdb_server import RemoteServerClient
from .fork import execute_database_fork


class Client(RemoteServerClient):
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

    def fork_database(self, destination_name: str) -> Client:
        """Fork this client's database and return a client bound to the copy.

        The source database is always ``self.database``.  The destination must not
        exist and is created only when the connected seekdb backend supports the
        ``FORK DATABASE`` statement.
        """
        execute_database_fork(self, destination_name)
        return type(self)(
            host=self.host,
            port=self.port,
            tenant=self.tenant,
            database=destination_name,
            user=self.user,
            password=self.password,
            charset=self.charset,
            **self.kwargs,
        )


__all__ = ["Client"]
