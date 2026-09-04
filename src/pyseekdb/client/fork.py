"""Database fork operations shared by the synchronous client."""

from typing import TYPE_CHECKING

from .validators import _quote_sql_identifier, _validate_database_name

if TYPE_CHECKING:
    from .client_base import BaseClient


def build_drop_database_sql(database_name: str) -> str:
    """Build the explicit SQL used to destroy a forked database."""
    _validate_database_name(database_name)
    return f"DROP DATABASE IF EXISTS {_quote_sql_identifier(database_name)}"


def execute_database_fork(client: "BaseClient", destination_name: str) -> None:
    """Fork the client's current database into ``destination_name``.

    Database provisioning and deletion remain outside pyseekdb's client API.  This
    helper only performs the seekdb-specific branch operation for an existing source
    database and lets the public client bind to the new database afterwards.
    """
    source_name = client.database
    _validate_database_name(source_name)
    _validate_database_name(destination_name)
    if source_name == destination_name:
        raise ValueError("Fork destination must differ from the source database")
    if not client.supports_fork_database:
        raise ValueError("Fork database is not supported by this backend (requires seekdb >= 1.2.0)")

    sql = f"FORK DATABASE {_quote_sql_identifier(source_name)} TO {_quote_sql_identifier(destination_name)}"
    try:
        client._execute(sql)
    except Exception as exc:
        args = getattr(exc, "args", ())
        if args and isinstance(args[0], int) and args[0] == 1007:
            raise ValueError(f"Database '{destination_name}' already exists") from exc

        message = str(exc).lower()
        if ("database exists" in message or "already exists" in message) and "database" in message:
            raise ValueError(f"Database '{destination_name}' already exists") from exc
        raise


__all__ = ["build_drop_database_sql", "execute_database_fork"]
