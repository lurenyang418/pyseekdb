# 2. Fork Database

`Client.fork_database()` is an advanced seekdb capability. It forks the database
bound to the current client and returns a new client already bound to the target
database:

```python
import pyseekdb

client = pyseekdb.Client(
    host="127.0.0.1",
    port=2881,
    tenant="sys",
    database="agent_state",
    user="root",
    password="",
)

sandbox = client.fork_database("agent_sandbox_42")
collection = sandbox.get_collection("memory")

# When the experiment is finished, destroy only this forked database.
sandbox.destroy()
```

The source database is always `client.database`; database creation, deletion, and
authorization remain outside pyseekdb. The destination database must not already
exist. Forking requires a seekdb server with database-fork support (seekdb 1.2.0
or newer), source-level `SELECT` permission, and destination `CREATE` permission.

`destroy()` is available only on a client returned by `fork_database()`. It closes
that branch client and asks the original client to execute `DROP DATABASE IF EXISTS`;
the caller therefore needs permission to drop the branch. An ordinary `Client`
cannot destroy an arbitrary database. Calling `close()` alone only releases the
branch connection and does not delete data.

The fork does not copy source database grants. Routine, trigger, sequence, and
other non-table objects may not be copied. Database fork is not enabled for a
generic OceanBase Server connection until compatibility has been verified.

## DIFF / MERGE scope

seekdb also provides `DIFF TABLE` and `MERGE TABLE` for comparing or applying
changes between table branches. pyseekdb 2.0 deliberately does not wrap these
statements yet: their table mapping, conflict strategy, and vector/async-index
compatibility need a separate tested API. Use a MySQL-compatible driver or DBA
workflow for them for now; `Client` only exposes the tested fork and branch
cleanup operations.

When the connected backend does not support the operation, `fork_database()` raises
`ValueError` with an explanation instead of returning a partially usable database
object.
