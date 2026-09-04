"""Backend capability metadata and detection for server-side feature checks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from .version import Version

logger = logging.getLogger(__name__)

NAMESPACE_MIN_LAKEBASE_VERSION = Version("4.6.1.0")
_LAKEBASE_VERSION_MARKER = "database ai"


def is_lakebase_version_string(version_str: str) -> bool:
    """Return whether a version string identifies an OceanBase Database AI cluster."""
    return _LAKEBASE_VERSION_MARKER in version_str.lower()


def _first_result_value(result: Any, key: str) -> str | None:
    """Extract and normalize the first value from a DB-API result set."""
    if not result:
        return None
    row = result[0]
    if isinstance(row, dict):
        value = row.get(key) or row.get(key.upper())
    elif isinstance(row, (tuple, list)) and row:
        value = row[0]
    else:
        value = row
    return str(value).strip() if value is not None else None


def _query_value(client: Any, sql: str, key: str) -> str | None:
    """Execute a scalar metadata query and normalize its first value."""
    try:
        result = client._execute(sql)
    except Exception as exc:
        logger.debug("Failed to execute %s: %s", sql, exc)
        return None
    return _first_result_value(result, key)


def _extract_seekdb_version(version_str: str) -> str | None:
    """Extract a three- or four-part seekdb version from a server banner."""
    for pattern in (
        r"seekdb[-\s]v?(\d+\.\d+\.\d+\.\d+)",
        r"seekdb[-\s]v?(\d+\.\d+\.\d+)",
    ):
        match = re.search(pattern, version_str, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _truncate(value: Any, length: int = 20) -> str:
    """Keep diagnostic values short when constructing detection errors."""
    if value is None:
        return "None"
    value_str = str(value)
    return value_str[:length] + ("..." if len(value_str) > length else "")


def parse_backend_identity(version_value: Any, ob_version_value: Any) -> tuple[str, Version]:
    """Parse seekdb/OceanBase identity from the two server version banners."""
    version_result = str(version_value).strip() if version_value is not None else None
    if version_result and re.search(r"seekdb", version_result, re.IGNORECASE):
        seekdb_version = _extract_seekdb_version(version_result)
        if seekdb_version:
            return "seekdb", Version(seekdb_version)
        raise ValueError(f"Detected seekdb in version string, but failed to extract version: {version_result}")

    ob_version = str(ob_version_value).strip() if ob_version_value is not None else None
    if ob_version:
        try:
            return "oceanbase", Version(ob_version)
        except ValueError as exc:
            parts = re.findall(r"\d+", ob_version)
            if len(parts) >= 3:
                return "oceanbase", Version(".".join(parts[:4]))
            raise ValueError(f"Unable to parse OceanBase version: {ob_version}") from exc

    raise ValueError(
        f"Unable to detect database type. version()={_truncate(version_result)}, ob_version()={_truncate(ob_version)}"
    )


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Features advertised by the connected backend and server version."""

    backend: str
    version: Version

    @property
    def is_seekdb(self) -> bool:
        """Whether the connected server identifies itself as seekdb."""
        return self.backend.lower() == "seekdb"

    @property
    def supports_fork_table(self) -> bool:
        """Whether collection/table forking is available."""
        return self.is_seekdb and self.version >= Version("1.1.0.0")

    @property
    def supports_fork_database(self) -> bool:
        """Whether database forking is available."""
        return self.is_seekdb and self.version >= Version("1.2.0.0")

    @property
    def supports_refresh_index(self) -> bool:
        """Whether explicit vector-index refresh is available."""
        return self.is_seekdb and self.version >= Version("1.3.0.0")


class BackendCapabilitiesMixin:
    """Detect backend identity and derive capabilities from the active connection."""

    def _validate_ob_database_type(self) -> None:
        """Validate that the backend is LakeBase and meets the namespace minimum version."""
        db_type, version = self.detect_db_type_and_version()
        if db_type.lower() != "oceanbase":
            raise ValueError("use_namespace=True is only supported on LakeBase (OceanBase Database AI)")
        if not self._is_lakebase_cluster():
            raise ValueError(
                "use_namespace=True is only supported on LakeBase (OceanBase Database AI); "
                "the connected cluster is standard OceanBase"
            )
        if version < NAMESPACE_MIN_LAKEBASE_VERSION:
            raise ValueError(
                f"use_namespace=True requires LakeBase version >= {NAMESPACE_MIN_LAKEBASE_VERSION}, "
                f"current version is {version}"
            )

    def _is_lakebase_cluster(self) -> bool:
        """Return whether the connected OceanBase cluster is LakeBase."""
        cached = getattr(self, "_lakebase_cluster", None)
        if cached is not None:
            return cached
        result = False
        try:
            rows = self._execute("SELECT version() AS version")
            if rows:
                row = rows[0]
                if isinstance(row, dict):
                    version_str = row.get("version") or row.get("VERSION") or ""
                elif isinstance(row, (tuple, list)) and row:
                    version_str = row[0]
                else:
                    version_str = str(row)
                result = is_lakebase_version_string(str(version_str))
        except Exception:
            result = False
        self._lakebase_cluster = result
        return result

    def _is_shared_storage_mode(self) -> bool:
        """Return whether the OceanBase deployment runs in shared-storage mode."""
        cached = getattr(self, "_shared_storage", None)
        if cached is not None:
            return cached
        result = False
        try:
            rows = self._execute("SELECT VALUE FROM oceanbase.GV$OB_PARAMETERS WHERE name = 'ob_startup_mode'")
            if rows:
                val = rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]["VALUE"]
                result = str(val).upper() == "SHARED_STORAGE"
        except Exception:
            result = False
        self._shared_storage = result
        return result

    def _stg_cache_policy_clause(self) -> str:
        """Return the shared-storage catalog cache policy clause when applicable."""
        if self._is_shared_storage_mode():
            return 'STORAGE_CACHE_POLICY = (GLOBAL = "hot")'
        return ""

    def detect_db_type_and_version(self) -> tuple[str, Version]:
        """Detect the connected backend type and version."""
        self._ensure_connection()
        version_result = _query_value(self, "SELECT version() as version", "version")
        if version_result and "seekdb" in version_result.lower():
            return parse_backend_identity(version_result, None)
        ob_version = _query_value(self, "SELECT ob_version() as ob_version", "ob_version")
        return parse_backend_identity(version_result, ob_version)

    @property
    def backend_capabilities(self) -> BackendCapabilities:
        """Return capabilities detected for the connected backend."""
        cached = getattr(self, "_backend_capabilities", None)
        if cached is None:
            backend, version = self.detect_db_type_and_version()
            cached = BackendCapabilities(backend=backend, version=version)
            self._backend_capabilities = cached
        return cached


__all__ = [
    "NAMESPACE_MIN_LAKEBASE_VERSION",
    "BackendCapabilities",
    "BackendCapabilitiesMixin",
    "is_lakebase_version_string",
    "parse_backend_identity",
]
