# 3. Collection (Table) Management

Collections are the primary data structures in pyseekdb. Each collection can
store documents, vector embeddings, metadata, and full-text search indexes.

Since pyseekdb 2.0, collection creation uses one `Schema` object. Embedding
function implementations are not bundled; provide your own implementation or
set an explicit vector dimension and pass embeddings directly.

## 3.1 Creating a Collection

```python
import pyseekdb
from pyseekdb import HNSWConfiguration, Schema, VectorIndexConfig
from your_app.embedding_functions import MyDenseEmbeddingFunction

client = pyseekdb.Client(host="127.0.0.1", port=2881, database="test")

# Explicit dimension; callers provide embeddings to add/update/upsert.
collection = client.create_collection(
    "my_collection",
    schema=Schema(vector_index=HNSWConfiguration(dimension=384, distance="cosine")),
)

# Store an embedding function in the schema.
ef = MyDenseEmbeddingFunction(model_name="all-MiniLM-L6-v2")
schema = Schema(
    vector_index=VectorIndexConfig(
        hnsw=HNSWConfiguration(dimension=384, distance="cosine"),
        embedding_function=ef,
    )
)
collection = client.create_collection("my_collection", schema=schema)
```

The schema can also configure a full-text parser:

```python
from pyseekdb import FulltextIndexConfig, HNSWConfiguration, Schema, VectorIndexConfig

schema = Schema(
    vector_index=VectorIndexConfig(
        hnsw=HNSWConfiguration(dimension=384, distance="cosine"),
    ),
    fulltext_index=FulltextIndexConfig(analyzer="ik"),
)
collection = client.create_collection("my_collection", schema=schema)
```

Supported analyzers include `ik` (the default), `space`, `ngram`, `ngram2`,
and `beng`. Analyzer-specific options go in
`FulltextIndexConfig.properties`.

`get_or_create_collection` uses the same `schema` parameter:

```python
collection = client.get_or_create_collection("my_collection", schema=schema)
```

If `schema` is omitted for a standard collection, pyseekdb uses the default
384-dimensional HNSW schema. Namespace-enabled collections require an
explicit IVF schema; see [Namespace Collections](namespace.md).

## 3.2 Getting a Collection

```python
collection = client.get_collection("my_collection")

if client.has_collection("my_collection"):
    collection = client.get_collection("my_collection")

# A retrieval-time embedding function is useful when the function is not
# persisted in the collection metadata.
collection = client.get_collection("my_collection", embedding_function=ef)
```

The `embedding_function` argument to `get_collection` only supplies the
runtime function used by collection operations; creation settings belong in
`Schema`.

## 3.3 Listing and Deleting Collections

```python
for collection in client.list_collections():
    print(collection.name, collection.dimension)

print(client.count_collection())
client.delete_collection("my_collection")
```

## 3.4 Collection Properties

Each `Collection` object exposes:

- `name`: Collection name
- `id`: Unique collection identifier
- `dimension`: Dense vector dimension, if configured
- `embedding_function`: Runtime embedding function, if available
- `distance`: Distance metric used by the vector index
- `metadata`: Collection metadata
