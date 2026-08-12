"""
Remote server mode client - based on pymysql
Supports both seekdb Server and OceanBase Server
"""

import logging
from collections.abc import Sequence

import pymysql
from pymysql.cursors import DictCursor

from .admin_client import DEFAULT_TENANT
from .client_base import BaseClient
from .database import Database
from .kernel_errors import namespace_kernel_error_guard

logger = logging.getLogger(__name__)


class RemoteServerClient(BaseClient):
    """Remote server mode client (connecting via pymysql, lazy loading)

    Supports both seekdb Server and OceanBase Server.
    Uses user@tenant format for authentication.
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
        **kwargs,
    ):
        """
        Initialize remote server mode client (no immediate connection)

        Args:
            host: server address
            port: server port (default 2881)
            tenant: tenant name (default "sys" for seekdb Server, "test" for OceanBase)
            database: database name
            user: username (without tenant suffix)
            password: password
            charset: charset (default "utf8mb4")
            **kwargs: other pymysql connection parameters
        """
        self.host = host
        self.port = port
        self.tenant = tenant
        self.database = database
        self.user = user
        self.password = password
        self.charset = charset
        self.kwargs = kwargs

        # Remote server username format: user@tenant
        self.full_user = f"{user}@{tenant}"
        self._connection = None

        logger.debug(f"Initialize RemoteServerClient: {self.full_user}@{self.host}:{self.port}/{self.database}")

    # ==================== Connection Management ====================

    def _ensure_connection(self) -> pymysql.Connection:
        """Ensure connection is established (internal method)"""
        if self._connection is None or not self._connection.open:
            self._connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.full_user,  # Remote server format: user@tenant
                password=self.password,
                database=self.database,
                charset=self.charset,
                cursorclass=DictCursor,
                autocommit=True,
                **self.kwargs,
            )
            logger.info(f"✅ Connected to remote server: {self.host}:{self.port}/{self.database}")
            try:
                self._use_catalog_database()
            except Exception as exc:
                logger.warning("Failed to initialize catalog database on connect: %s", exc)

        return self._connection

    def _cleanup(self):
        """Internal cleanup method: close connection)"""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info(f"Connection closed: {self.host}:{self.port}/{self.database}")

    def is_connected(self) -> bool:
        """Check connection status"""
        return self._connection is not None and self._connection.open

    def get_raw_connection(self) -> pymysql.Connection:
        """Get raw connection object"""
        return self._ensure_connection()

    @property
    def mode(self) -> str:
        """Return the client mode identifier."""
        return "RemoteServerClient"

    # ==================== Collection Management (framework) ====================

    # create_collection is inherited from BaseClient - no override needed
    # get_collection is inherited from BaseClient - no override needed
    # delete_collection is inherited from BaseClient - no override needed
    # list_collections is inherited from BaseClient - no override needed
    # has_collection is inherited from BaseClient - no override needed

    # ==================== Collection Internal Operations ====================
    # These methods are called by Collection objects

    # -------------------- DML Operations --------------------
    # _collection_add is inherited from BaseClient
    # _collection_update is inherited from BaseClient
    # _collection_upsert is inherited from BaseClient
    # _collection_delete is inherited from BaseClient

    # -------------------- DQL Operations --------------------
    # Note: _collection_query() and _collection_get() use base class implementation

    # _collection_hybrid_search is inherited from BaseClient

    # -------------------- Collection Info --------------------

    # _collection_count is inherited from BaseClient - no override needed

    # ==================== Database Management ====================

    def create_database(self, name: str, tenant: str = DEFAULT_TENANT) -> None:
        """
        Create database (remote server has tenant concept, uses client's tenant)

        Args:
            name: database name
            tenant: tenant name (if different from client tenant, will use client tenant)

        Note:
            Remote server has multi-tenant architecture. Database is scoped to client's tenant.
        """
        return super().create_database(name=name, tenant=tenant)

    def get_database(self, name: str, tenant: str = DEFAULT_TENANT) -> Database:
        """
        Get database object (remote server has tenant concept, uses client's tenant)

        Args:
            name: database name
            tenant: tenant name (if different from client tenant, will use client tenant)

        Returns:
            Database object with tenant information

        Note:
            Remote server has multi-tenant architecture. Database is scoped to client's tenant.
        """
        return super().get_database(name=name, tenant=tenant)

    def delete_database(self, name: str, tenant: str = DEFAULT_TENANT) -> None:
        """
        Delete database (remote server has tenant concept, uses client's tenant)

        Args:
            name: database name
            tenant: tenant name (if different from client tenant, will use client tenant)

        Note:
            Remote server has multi-tenant architecture. Database is scoped to client's tenant.
        """
        return super().delete_database(name=name, tenant=tenant)

    def list_databases(
        self,
        limit: int | None = None,
        offset: int | None = None,
        tenant: str = DEFAULT_TENANT,
    ) -> Sequence[Database]:
        """
        List all databases (remote server has tenant concept, uses client's tenant)

        Args:
            limit: maximum number of results to return
            offset: number of results to skip
            tenant: tenant name (if different from client tenant, will use client tenant)

        Returns:
            Sequence of Database objects with tenant information

        Note:
            Remote server has multi-tenant architecture. Lists databases in client's tenant.
        """
        return super().list_databases(limit=limit, offset=offset, tenant=tenant)

    def _database_tenant(self, tenant: str) -> str | None:
        """Return the tenant associated with the active database."""
        if tenant != self.tenant and tenant != DEFAULT_TENANT:
            logger.warning(
                f"Specified tenant '{tenant}' differs from client tenant '{self.tenant}', using client tenant"
            )
        return self.tenant

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

    def __repr__(self):
        """Return the developer-readable representation."""
        status = "connected" if self.is_connected() else "disconnected"
        return f"<RemoteServerClient {self.full_user}@{self.host}:{self.port}/{self.database} status={status}>"
