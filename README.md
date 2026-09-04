# pyseekdb

pyseekdb is a Python SDK for seekdb and OceanBase AI search. It connects to a remote
seekdb Server or OceanBase Server over pymysql and exposes a collection-first API for
vector, full-text, and hybrid retrieval. For advanced database operations, you can use
MySQL-compatible drivers to run SQL against seekdb and OceanBase.

Since pyseekdb 2.0, embedded mode and bundled embedding function implementations
have been removed. You must run a seekdb/OceanBase server and supply your own
`EmbeddingFunction` (via the `EmbeddingFunction` protocol) when creating
collections with a dense vector index.

Key features:

- **Remote Server Connection**: Connects to a seekdb Server or OceanBase Server
- **Vector Operations**: Efficient vector similarity search
- **Hybrid Search**: Combine vector and full-text search
- **Pluggable Embedding Functions**: Use your own `EmbeddingFunction` implementation
- **Collection Management**: Easy collection (table) creation and management
- **Database Branching**: Fork an existing seekdb database into an isolated branch

## Documentation

- Docs home: https://docs.seekdb.ai
- SDK guide: https://docs.seekdb.ai/seekdb/pyseekdb-sdk-get-started
- User guide: https://docs.seekdb.ai/seekdb/deploy-overview
- API reference: https://docs.seekdb.ai/seekdb/api-overview
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)

## Installation

```bash
pip install -U pyseekdb
# or with uv
uv add pyseekdb

# Optional asynchronous client
pip install -U "pyseekdb[async]"
```

## Quick Start

```python
import pyseekdb
from your_app.embedding_functions import MyDenseEmbeddingFunction

client = pyseekdb.Client(
    host="127.0.0.1",
    port=2881,
    tenant="sys",
    database="demo",
    user="root",
    password="",
)
collection = client.get_or_create_collection(
    "my_collection",
    schema=pyseekdb.Schema(
        vector_index=pyseekdb.VectorIndexConfig(
            embedding_function=MyDenseEmbeddingFunction(),
        )
    ),
)

collection.add(
    ids=["doc1", "doc2"],
    documents=["Hello world", "pyseekdb quick start"],
    metadatas=[{"tag": "hello"}, {"tag": "demo"}],
)

# Refresh the index to make the added documents searchable
collection.refresh_index()

results = collection.query(query_texts=["hello"], n_results=3)
print(results["ids"][0])
```

For full usage, connection modes, collection management, and operations, see the
[User Guide](https://oceanbase.github.io/pyseekdb/guide/).

## License

This package is licensed under Apache 2.0. See [LICENSE](LICENSE).
