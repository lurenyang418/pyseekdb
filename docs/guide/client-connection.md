# 1. Client Connection

The `Client` class connects to a remote seekdb Server or OceanBase Server over pymysql.
The `host` parameter (and typically `port`, `user`, `password`) is required; embedded
mode was removed in pyseekdb 2.0.

## 1.1 Remote Server Client

Connect to a remote server (supports both seekdb Server and OceanBase Server):

```python
import pyseekdb

# Create remote server client (seekdb Server)
client = pyseekdb.Client(
    host="127.0.0.1",      # Server host (required)
    port=2881,              # Server port (default: 2881)
    database="demo",        # Database name
    user="root",            # Username (default: "root")
    password=""             # Password (can be retrieved from SEEKDB_PASSWORD environment variable)
)

# Create remote server client (OceanBase Server)
client = pyseekdb.Client(
    host="127.0.0.1",      # Server host (required)
    port=2881,              # Server port (default: 2881)
    tenant="sys",          # Tenant name (default: sys)
    database="demo",       # Database name
    user="root",           # Username (default: "root")
    password=""             # Password (can be retrieved from SEEKDB_PASSWORD environment variable)
)
```

**Note:** If the `password` parameter is not provided (empty string), the client will automatically retrieve it from the `SEEKDB_PASSWORD` environment variable. This is useful for keeping passwords out of your code:

```bash
export SEEKDB_PASSWORD="your_password"
```

```python
# Password will be automatically retrieved from SEEKDB_PASSWORD environment variable
client = pyseekdb.Client(
    host="127.0.0.1",
    port=2881,
    database="demo",
    user="root"
    # password parameter omitted - will use SEEKDB_PASSWORD from environment
)
```

## 1.2 Client Methods and Properties

| Method / Property     | Description                                                    |
|-----------------------|----------------------------------------------------------------|
| `create_collection()`  | Create a new collection (see Collection Management)            |
| `get_collection()`    | Get an existing collection object                              |
| `delete_collection()` | Delete a collection                                            |
| `list_collections()`  | List all collections in the current database                   |
| `has_collection()`    | Check if a collection exists                                   |
| `get_or_create_collection()` | Get an existing collection or create it if it doesn't exist |
| `count_collection()`  | Count the number of collections in the current database         |
| `fork_database()`     | Fork the current seekdb database and return a client for the branch |

The client is bound to the database supplied at construction time. The database
must already exist; provision it with deployment or DBA tooling before creating
the client.
