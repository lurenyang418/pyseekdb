"""
Integration tests: ns.query where_document operators (1.6.10).

Vector query with document pre-filters applied via knn.filter.
"""

from __future__ import annotations

import time

from namespace_dml_helpers import cleanup

from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import FulltextIndexConfig, VectorIndexConfig
from pyseekdb.client.schema import Schema


def _schema() -> Schema:
    """Namespace collection schema for where_document query tests."""
    return Schema(
        vector_index=VectorIndexConfig(
            ivf=IVFConfiguration(dimension=3, distance="l2", centroids_fresh_mode="spfresh"),
            embedding_function=None,
        ),
        fulltext_index=FulltextIndexConfig(analyzer="ik"),
    )


def _create_ns(client, suffix: str):
    """Create a namespace-enabled test collection."""
    name = f"test_ns_qb_{int(time.time() * 1000)}{suffix}"
    return client.create_collection(
        name=name,
        schema=_schema(),
        use_namespace=True,
        partition_count=4,
    )


def _seed(ns) -> None:
    """Load deterministic corpus for where_document operator checks."""
    ns.add(
        ids=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        embeddings=[
            [1.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [1.0, 0.4, 0.0],
            [1.0, 0.6, 0.0],
            [1.0, 0.8, 0.0],
            [0.8, 1.0, 0.0],
            [0.6, 1.0, 0.0],
            [0.4, 1.0, 0.0],
            [0.2, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        documents=[
            "machine learning AI research",
            "Python programming language",
            "OceanBase distributed database",
            "deep learning neural networks",
            "data science machine learning",
            "web development Python Django",
            "cloud computing AWS",
            "database SQL optimization",
            "container Docker Kubernetes",
            "microservices architecture",
            "blockchain distributed ledger",
        ],
        metadatas=[
            {"cat": "AI", "score": 95},
            {"cat": "Prog", "score": 88},
            {"cat": "DB", "score": 92},
            {"cat": "AI", "score": 90},
            {"cat": "DS", "score": 85},
            {"cat": "Prog", "score": 75},
            {"cat": "Cloud", "score": 70},
            {"cat": "DB", "score": 80},
            {"cat": "Cloud", "score": 65},
            {"cat": "Arch", "score": 60},
            {"cat": "DB", "score": 55},
        ],
    )


class TestQueryWhereDocumentOperators:
    """1.6.10: where_document $not_contains / $and / $or / $regex / nested combos."""

    def test_where_document_contains_baseline(self, db_client):
        """$contains baseline: single operator works under ns.query."""
        collection = _create_ns(db_client, "_wd_contains")
        try:
            ns = collection.create_namespace("ns_wd_contains")
            _seed(ns)
            time.sleep(1)

            result = ns.query(
                query_embeddings=[1.0, 0.0, 0.0],
                where_document={"$contains": "machine"},
                n_results=10,
                include=["documents"],
            )
            docs = result.get("documents", [[]])[0] or []
            assert docs, "expected at least one hit for $contains 'machine'"
            for doc in docs:
                assert "machine" in doc.lower(), f"$contains 'machine' returned doc without 'machine': {doc}"
        finally:
            cleanup(db_client, collection)

    def test_where_document_not_contains(self, db_client):
        """$not_contains excludes documents containing the keyword."""
        collection = _create_ns(db_client, "_wd_not_ctn")
        try:
            ns = collection.create_namespace("ns_wd_not_ctn")
            _seed(ns)
            time.sleep(1)

            result = ns.query(
                query_embeddings=[1.0, 0.0, 0.0],
                where_document={"$not_contains": "machine"},
                n_results=10,
                include=["documents"],
            )
            docs = result.get("documents", [[]])[0] or []
            assert docs, "expected at least one hit for $not_contains 'machine'"
            violations = [d for d in docs if "machine" in d.lower()]
            assert not violations, (
                f"$not_contains 'machine' should exclude docs with 'machine', got violations: {violations}"
            )
        finally:
            cleanup(db_client, collection)

    def test_where_document_and(self, db_client):
        """$and: multiple $contains must all match."""
        collection = _create_ns(db_client, "_wd_and")
        try:
            ns = collection.create_namespace("ns_wd_and")
            _seed(ns)
            time.sleep(1)

            result = ns.query(
                query_embeddings=[1.0, 0.0, 0.0],
                where_document={"$and": [{"$contains": "machine"}, {"$contains": "learning"}]},
                n_results=10,
                include=["documents"],
            )
            docs = result.get("documents", [[]])[0] or []
            assert docs, "expected at least one hit for $and machine+learning"
            violations = [d for d in docs if not ("machine" in d.lower() and "learning" in d.lower())]
            assert not violations, f"$and should require both 'machine' and 'learning', got violations: {violations}"
        finally:
            cleanup(db_client, collection)

    def test_where_document_or(self, db_client):
        """$or: any $contains child may match."""
        collection = _create_ns(db_client, "_wd_or")
        try:
            ns = collection.create_namespace("ns_wd_or")
            _seed(ns)
            time.sleep(1)

            result = ns.query(
                query_embeddings=[1.0, 0.0, 0.0],
                where_document={"$or": [{"$contains": "blockchain"}, {"$contains": "Kubernetes"}]},
                n_results=10,
                include=["documents"],
            )
            docs = result.get("documents", [[]])[0] or []
            assert docs, "expected at least one hit for $or blockchain|Kubernetes"
            violations = [d for d in docs if not ("blockchain" in d.lower() or "kubernetes" in d.lower())]
            assert not violations, f"$or should accept 'blockchain' or 'Kubernetes', got violations: {violations}"
        finally:
            cleanup(db_client, collection)

    def test_where_document_regex(self, db_client):
        """$regex filters documents by regexp on the document column."""
        collection = _create_ns(db_client, "_wd_regex")
        try:
            ns = collection.create_namespace("ns_wd_regex")
            _seed(ns)
            time.sleep(1)

            result = ns.query(
                query_embeddings=[1.0, 0.0, 0.0],
                where_document={"$regex": r"machine.*learning"},
                n_results=10,
                include=["documents"],
            )
            docs = result.get("documents", [[]])[0] or []
            assert docs, "expected at least one hit for $regex machine.*learning"
            violations = [d for d in docs if "machine" not in d.lower() or "learning" not in d.lower()]
            assert not violations, f"$regex should match docs with machine+learning, got violations: {violations}"
        finally:
            cleanup(db_client, collection)

    def test_where_document_and_or_nested(self, db_client):
        """Nested $and($or): (machine|database) AND learning."""
        collection = _create_ns(db_client, "_wd_and_or")
        try:
            ns = collection.create_namespace("ns_wd_and_or")
            _seed(ns)
            time.sleep(1)

            result = ns.query(
                query_embeddings=[1.0, 0.0, 0.0],
                where_document={
                    "$and": [
                        {
                            "$or": [
                                {"$contains": "machine"},
                                {"$contains": "database"},
                            ]
                        },
                        {"$contains": "learning"},
                    ]
                },
                n_results=10,
                include=["documents"],
            )
            docs = result.get("documents", [[]])[0] or []
            assert docs, "expected at least one hit for nested $and($or)"
            violations = [
                d for d in docs if not (("machine" in d.lower() or "database" in d.lower()) and "learning" in d.lower())
            ]
            assert not violations, (
                f"Nested $and($or) should filter (machine|database) + learning, got violations: {violations}"
            )
        finally:
            cleanup(db_client, collection)

    def test_where_document_or_and_nested(self, db_client):
        """Nested $or($and): (machine+learning) OR (distributed+database)."""
        collection = _create_ns(db_client, "_wd_or_and")
        try:
            ns = collection.create_namespace("ns_wd_or_and")
            _seed(ns)
            time.sleep(1)

            result = ns.query(
                query_embeddings=[1.0, 0.0, 0.0],
                where_document={
                    "$or": [
                        {
                            "$and": [
                                {"$contains": "machine"},
                                {"$contains": "learning"},
                            ]
                        },
                        {
                            "$and": [
                                {"$contains": "distributed"},
                                {"$contains": "database"},
                            ]
                        },
                    ]
                },
                n_results=10,
                include=["documents"],
            )
            docs = result.get("documents", [[]])[0] or []
            assert docs, "expected at least one hit for nested $or($and)"
            violations = [
                d
                for d in docs
                if not (
                    ("machine" in d.lower() and "learning" in d.lower())
                    or ("distributed" in d.lower() and "database" in d.lower())
                )
            ]
            assert not violations, (
                f"Nested $or($and) should filter (machine+learning) OR (distributed+database), "
                f"got violations: {violations}"
            )
        finally:
            cleanup(db_client, collection)
