"""
Base connection interface definition
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseConnection(ABC):
    """
    Abstract base class for connection management.
    Defines unified connection interface for all clients.
    """

    # ==================== Connection Management ====================

    @abstractmethod
    def _ensure_connection(self) -> Any:
        """Ensure connection is established (internal method)"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check connection status"""
        pass

    @abstractmethod
    def _cleanup(self):
        """Internal cleanup method to close connection and release resources"""
        pass

    def close(self) -> None:
        """Close the client connection and release owned resources."""
        self._cleanup()

    def ping(self) -> bool:
        """Return whether the server responds to a lightweight health check."""
        result = self._execute("SELECT 1 AS pyseekdb_ping")
        if not result:
            return False

        row = result[0]
        if isinstance(row, dict):
            return row.get("pyseekdb_ping") == 1
        if isinstance(row, (tuple, list)):
            return bool(row) and row[0] == 1
        return False

    @abstractmethod
    def _execute(self, sql: str) -> Any:
        """Execute SQL statement (basic functionality)"""
        pass

    @abstractmethod
    def get_raw_connection(self) -> Any:
        """Get raw connection object"""
        pass

    # ==================== Context Manager ====================

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support: automatic resource cleanup"""
        self.close()

    def __del__(self):
        """Destructor: ensure connection is closed to prevent resource leaks"""
        try:
            if hasattr(self, "_connection"):
                self.close()
        except Exception as exc:
            # Ignore all exceptions in destructor
            # Avoid issues during interpreter shutdown
            logger.debug("Failed to cleanup connection in destructor: %s", exc)
