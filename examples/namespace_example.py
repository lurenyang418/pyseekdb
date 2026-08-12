"""
Namespace collection example (LakeBase 4.6.1.0+).

Demonstrates:
1. Creating a namespace-enabled collection with an explicit IVF schema
2. Creating a namespace and adding documents with record ids
3. Vector query inside the namespace
"""

import pyseekdb
from pyseekdb import FulltextIndexConfig, IVFConfiguration, Schema, VectorIndexConfig

# Connect to LakeBase / OceanBase (adjust host/port/credentials)
client = pyseekdb.Client(
    host="127.0.0.1",
    port=2881,
    database="test",
    user="root@test",
    password="",
)

schema = Schema(
    vector_index=VectorIndexConfig(
        ivf=IVFConfiguration(dimension=3, distance="l2", centroids_fresh_mode="spfresh"),
        embedding_function=None,
    ),
    fulltext_index=FulltextIndexConfig(analyzer="ik"),
)

collection_name = "demo_namespace_collection"
collection = client.create_collection(
    name=collection_name,
    schema=schema,
    use_namespace=True,
    partition_count=4,
)

ns = collection.get_or_create_namespace("demo")

# Record ids are per-document keys inside this namespace (see docs/guide/namespace.md).
ns.add(
    ids=["item_alpha", "item_beta"],
    embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    documents=["alpha product", "beta product"],
    metadatas=[{"tier": "a"}, {"tier": "b"}],
)

results = ns.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=2, include=["documents", "metadatas"])
print("Top match:", results["ids"][0][0], results["documents"][0][0])

collection.delete_namespace("demo")
client.delete_collection(collection_name)
