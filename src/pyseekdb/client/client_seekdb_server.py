"""
Remote server mode client - based on pymysql
Supports both seekdb Server and OceanBase Server
"""

import logging

from .client_base import BaseClient
from .connection import MySQLConnectionMixin
from .kernel_errors import namespace_kernel_error_guard

logger = logging.getLogger(__name__)


class RemoteServerClient(MySQLConnectionMixin, BaseClient):
    """Remote server mode client (connecting via pymysql, lazy loading)

    Supports both seekdb Server and OceanBase Server.
    Uses user@tenant format for authentication.
    """

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
        return f"<{type(self).__name__} {self.full_user}@{self.host}:{self.port}/{self.database} status={status}>"
