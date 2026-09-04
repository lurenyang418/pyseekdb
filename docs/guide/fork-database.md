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
```

The source database is always `client.database`; database creation, deletion, and
authorization remain outside pyseekdb. The destination database must not already
exist. Forking requires a seekdb server with database-fork support (seekdb 1.2.0
or newer), source-level `SELECT` permission, and destination `CREATE` permission.

The fork does not copy source database grants. Routine, trigger, sequence, and
other non-table objects may not be copied. Database fork is not enabled for a
generic OceanBase Server connection until compatibility has been verified.

When the connected backend does not support the operation, `fork_database()` raises
`ValueError` with an explanation instead of returning a partially usable database
object.
