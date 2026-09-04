# pyseekdb

Welcome to pyseekdb's documentation.

pyseekdb is a Python SDK for seekdb and OceanBase AI search. Since pyseekdb 2.0 it
connects to a remote seekdb Server or OceanBase Server over pymysql and exposes a
collection-first API for vector, full-text, and hybrid retrieval. For advanced database
operations, you can use MySQL-compatible drivers to run SQL against seekdb and OceanBase.

## Features

- **Remote Server Connection**: Connects to a seekdb Server or OceanBase Server via pymysql.
- **Vector Operations**: Efficient vector similarity search.
- **Hybrid Search**: Combine vector and full-text search.
- **Pluggable Embedding Functions**: Bring your own `EmbeddingFunction` implementation.
- **Collection Management**: Easy collection (table) creation and management.
- **Database Branching**: Fork an existing seekdb database into an isolated branch.

## Installation

```bash
pip install -U pyseekdb
# or with uv
uv add pyseekdb
```

## Quick Start

Remote Server Mode

```python
import pyseekdb
from your_app.embedding_functions import MyDenseEmbeddingFunction
from pyseekdb import Schema, VectorIndexConfig

client = pyseekdb.Client(
    host="localhost",
    port=2881,
    tenant="sys",
    database="test",
    user="root",
    password="pass",
)
collection = client.get_or_create_collection(
    "my_collection",
    schema=Schema(
        vector_index=VectorIndexConfig(
            embedding_function=MyDenseEmbeddingFunction(),
        )
    ),
)
```

The database must already exist before creating a client. Use your deployment or
DBA tooling, or a MySQL-compatible driver, to provision databases. pyseekdb does
not expose database-wide CRUD operations.

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

guide/index
api/index
```
