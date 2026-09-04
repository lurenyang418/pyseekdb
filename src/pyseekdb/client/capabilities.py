"""Backend capability metadata for server-side feature checks."""

from dataclasses import dataclass

from .version import Version


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
