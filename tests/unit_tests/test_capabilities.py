"""Unit tests for backend detection and feature capability boundaries."""

from pyseekdb.client.capabilities import BackendCapabilities, BackendCapabilitiesMixin
from pyseekdb.client.version import Version


def test_backend_capabilities_are_version_gated() -> None:
    old = BackendCapabilities(backend="seekdb", version=Version("1.1.0"))
    current = BackendCapabilities(backend="seekdb", version=Version("1.3.0"))
    oceanbase = BackendCapabilities(backend="oceanbase", version=Version("9.9.9"))

    assert old.supports_fork_table
    assert not old.supports_fork_database
    assert not old.supports_refresh_index
    assert current.supports_fork_database
    assert current.supports_refresh_index
    assert not oceanbase.supports_fork_table


def test_capability_mixin_detects_seekdb_without_connection_duplication() -> None:
    class FakeClient(BackendCapabilitiesMixin):
        def __init__(self) -> None:
            self.queries: list[str] = []

        def _ensure_connection(self) -> None:
            self.queries.append("connect")

        def _execute(self, sql: str) -> list[dict[str, str]]:
            self.queries.append(sql)
            return [{"version": "seekdb-v1.3.0.0"}] if "version()" in sql else []

    client = FakeClient()

    backend, version = client.detect_db_type_and_version()

    assert backend == "seekdb"
    assert version == Version("1.3.0.0")
    assert client.queries == ["connect", "SELECT version() as version"]
    assert client.backend_capabilities.supports_refresh_index


def test_capability_mixin_exposes_database_fork_support() -> None:
    class FakeClient(BackendCapabilitiesMixin):
        _backend_capabilities = BackendCapabilities(backend="seekdb", version=Version("1.2.0"))

    assert FakeClient().supports_fork_database
