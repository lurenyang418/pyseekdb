# Asynchronous Client

The optional `AsyncClient` uses an `aiomysql` connection pool. Install the extra
before using it:

```bash
pip install -U "pyseekdb[async]"
# or
uv add "pyseekdb[async]"
```

The async client is lazy: creating it does not open a connection. Use it as an
async context manager so the pool is closed reliably:

```python
import pyseekdb

async with pyseekdb.AsyncClient(
    host="127.0.0.1",
    port=2881,
    tenant="sys",
    database="demo",
    user="root",
    password="",
) as client:
    collection = await client.get_collection("my_collection")
    results = await collection.query(query_embeddings=[[0.1, 0.2, 0.3]], n_results=3)
    print(results["ids"])
```

`AsyncClient` supports remote standard-collection management, CRUD, dense vector
queries, collection fork, and database fork. Namespace collections and sparse
vector operations remain synchronous-only for now. A client returned by
`fork_database()` also supports `await forked.destroy()` for explicit branch
cleanup; `close()` only releases the connection pool. Database provisioning and
authorization are still handled by deployment or DBA tooling. Embedding functions
use the synchronous `EmbeddingFunction` protocol for now and are called directly,
so a slow embedding provider can block the event loop.
