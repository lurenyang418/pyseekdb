"""
Shared helpers for namespace DML integration tests.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any

from pyseekdb import IVFConfiguration
from pyseekdb.client.configuration import FulltextIndexConfig, VectorIndexConfig
from pyseekdb.client.schema import Schema

NAMESPACE_TEST_PARTITION_COUNT = 4

LARGE_DML_CORPUS_SIZE = 1000
DML_BATCH_SIZE = 100
# Wait for FULLTEXT index after bulk load (same as namespace_fts_helpers).
INDEX_SETTLE_SECONDS = 3

# Default sizes for P0 get/delete filter tests (single namespace).
FILTER_KEEP_COUNT = 300
FILTER_PURGE_COUNT = 200

# Multi-namespace orthogonality: 500 keep + 500 purge per namespace.
FILTER_MULTI_NS_KEEP = 500
FILTER_MULTI_NS_PURGE = 500


@dataclass(frozen=True)
class DmlRecord:
    """DmlRecord class."""

    doc_id: str
    embedding: list[float]
    document: str
    metadata: dict[str, Any]


def ns_schema() -> Schema:
    """Ns schema."""
    return Schema(
        vector_index=VectorIndexConfig(
            ivf=IVFConfiguration(dimension=3, distance="l2", centroids_fresh_mode="spfresh"),
            embedding_function=None,
        ),
        fulltext_index=FulltextIndexConfig(analyzer="ik"),
    )


def create_ns_collection(client: Any, suffix: str = "") -> Any:
    """Create ns collection."""
    name = f"test_ns_dml_{int(time.time() * 1000)}{suffix}"
    return client.create_collection(
        name=name,
        schema=ns_schema(),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )


def cleanup(client: Any, *collections: Any) -> None:
    """Cleanup."""
    for collection in collections:
        with contextlib.suppress(Exception):
            client.delete_collection(name=collection.name)


def build_large_dml_corpus(
    size: int = LARGE_DML_CORPUS_SIZE,
    *,
    id_prefix: str = "dml",
    ns_tag: str | None = None,
) -> list[DmlRecord]:
    """Deterministic corpus for count/peek scale tests."""
    records: list[DmlRecord] = []
    for i in range(size):
        doc_id = f"{id_prefix}_{i:04d}"
        embedding = [
            (i % 97) / 97.0,
            ((i * 3) % 89) / 89.0,
            ((i * 7) % 83) / 83.0,
        ]
        document = f"namespace dml document {id_prefix} index {i}"
        metadata: dict[str, Any] = {"seq": i, "id_prefix": id_prefix}
        if ns_tag is not None:
            metadata["ns_tag"] = ns_tag
        records.append(DmlRecord(doc_id, embedding, document, metadata))
    return records


def corpus_id_set(corpus: list[DmlRecord]) -> set[str]:
    """Corpus id set."""
    return {r.doc_id for r in corpus}


def _filter_embedding(index: int) -> list[float]:
    """Filter embedding."""
    return [
        (index % 97) / 97.0,
        ((index * 3) % 89) / 89.0,
        ((index * 7) % 83) / 83.0,
    ]


def build_filter_corpus(
    *,
    keep_count: int = FILTER_KEEP_COUNT,
    purge_count: int = FILTER_PURGE_COUNT,
    ai_in_keep: int = 120,
    python_in_keep: int = 50,
    id_prefix: str = "flt",
    ns_tag: str | None = None,
) -> tuple[list[DmlRecord], dict[str, int]]:
    """
    Corpus for conditional get/delete tests.

    - ``tag=keep``: stable rows; first ``ai_in_keep`` have ``category=AI``;
      first ``python_in_keep`` documents contain ``python``.
    - ``tag=purge``: rows whose documents contain ``obsolete`` (for ``where_document`` delete).
    """
    records: list[DmlRecord] = []
    idx = 0

    for i in range(keep_count):
        category = "AI" if i < ai_in_keep else ("Programming" if i % 2 == 0 else "Database")
        document = (
            f"python guide {id_prefix} keep index {i}"
            if i < python_in_keep
            else f"stable document {id_prefix} keep index {i}"
        )
        meta: dict[str, Any] = {"tag": "keep", "category": category, "seq": i}
        if ns_tag is not None:
            meta["ns_tag"] = ns_tag
        records.append(DmlRecord(f"{id_prefix}_keep_{i:04d}", _filter_embedding(idx), document, meta))
        idx += 1

    for i in range(purge_count):
        meta = {"tag": "purge", "category": "Database", "seq": i}
        if ns_tag is not None:
            meta["ns_tag"] = ns_tag
        records.append(
            DmlRecord(
                f"{id_prefix}_purge_{i:04d}",
                _filter_embedding(idx),
                f"obsolete payload {id_prefix} purge index {i}",
                meta,
            )
        )
        idx += 1

    ground_truth = {
        "total": keep_count + purge_count,
        "keep": keep_count,
        "purge": purge_count,
        "category_ai": ai_in_keep,
        "doc_python": python_in_keep,
        "doc_obsolete": purge_count,
    }
    return records, ground_truth


def assert_get_where_count(
    ns: Any,
    where: dict[str, Any],
    expected: int,
    *,
    limit: int | None = None,
    check_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``get(where=...)`` must return exactly ``expected`` rows (up to ``limit``)."""
    kwargs: dict[str, Any] = {"where": where, "include": ["metadatas", "documents"]}
    if limit is not None:
        kwargs["limit"] = limit
    result = ns.get(**kwargs)
    ids = result.get("ids") or []
    assert len(ids) == expected, f"expected {expected} ids, got {len(ids)}: {ids[:5]!r}..."
    if check_meta is not None and result.get("metadatas"):
        for meta in result["metadatas"]:
            assert meta is not None
            for key, value in check_meta.items():
                assert meta.get(key) == value, meta
    return result


def settle_fts_index() -> None:
    """Allow namespace logic-table FULLTEXT index to catch up after bulk insert."""
    time.sleep(INDEX_SETTLE_SECONDS)


def assert_get_where_document_count(
    ns: Any,
    where_document: dict[str, Any],
    expected: int,
    *,
    limit: int | None = None,
    substring: str | None = None,
    context: str = "",
) -> dict[str, Any]:
    """Assert get where document count."""
    kwargs: dict[str, Any] = {"where_document": where_document, "include": ["documents", "metadatas"]}
    if limit is not None:
        kwargs["limit"] = limit
    result = ns.get(**kwargs)
    ids = result.get("ids") or []
    prefix = f"{context}: " if context else ""
    if len(ids) != expected:
        docs = result.get("documents") or []
        raise AssertionError(
            f"{prefix}get(where_document={where_document!r}, limit={limit!r}) "
            f"expected {expected} row(s), got {len(ids)}. "
            f"Uses namespace DML document filter (LIKE for $contains, not hybrid_search FTS). "
            f"sample_ids={ids[:5]!r}, sample_documents={docs[:3]!r}"
        )
    if substring and result.get("documents"):
        for i, doc in enumerate(result["documents"]):
            if doc is None:
                raise AssertionError(f"{prefix}document at index {i} is None for id={ids[i]!r}")
            if substring not in doc.lower():
                raise AssertionError(f"{prefix}id={ids[i]!r} document={doc!r} does not contain substring {substring!r}")
    return result


def precheck_where_document_hits(
    ns: Any,
    where_document: dict[str, Any],
    expected_hits: int,
    *,
    context: str = "before delete",
) -> dict[str, Any]:
    """Ensure FTS-backed get sees rows before delete(where_document=...)."""
    probe_limit = min(expected_hits, 50) if expected_hits > 0 else 10
    return assert_get_where_document_count(
        ns,
        where_document,
        expected_hits,
        limit=expected_hits if expected_hits <= 1000 else probe_limit,
        context=context,
    )


def assert_count_after_document_delete(
    ns: Any,
    *,
    where_document: dict[str, Any],
    expected_count: int,
    total_before: int,
    expected_removed: int,
    pre_delete_hit_count: int,
) -> None:
    """Assert count after document delete."""
    actual = ns.count()
    if actual == expected_count:
        return
    raise AssertionError(
        f"after delete(where_document={where_document!r}): "
        f"count expected {expected_count}, got {actual} "
        f"(total_before={total_before}, expected_removed={expected_removed} rows). "
        f"Pre-delete get(where_document=...) returned {pre_delete_hit_count} hit(s). "
        "If pre-delete hits matched but count did not drop, suspect logic-table "
        "DELETE with MATCH(document) AGAINST on OceanBase."
    )


def load_filter_corpus(
    namespace: Any,
    corpus: list[DmlRecord],
    *,
    assert_count: bool = True,
    settle_fts: bool = True,
) -> None:
    """Load filter test corpus; optionally wait for FULLTEXT index (where_document paths)."""
    load_dml_corpus(namespace, corpus, assert_count=assert_count)
    if settle_fts:
        settle_fts_index()


def insert_dml_corpus_in_batches(
    namespace: Any,
    corpus: list[DmlRecord],
    batch_size: int = DML_BATCH_SIZE,
) -> None:
    """Insert dml corpus in batches."""
    for start in range(0, len(corpus), batch_size):
        chunk = corpus[start : start + batch_size]
        namespace.add(
            ids=[r.doc_id for r in chunk],
            embeddings=[r.embedding for r in chunk],
            documents=[r.document for r in chunk],
            metadatas=[r.metadata for r in chunk],
        )


def load_dml_corpus(
    namespace: Any,
    corpus: list[DmlRecord],
    *,
    assert_count: bool = True,
) -> None:
    """Load dml corpus."""
    insert_dml_corpus_in_batches(namespace, corpus)
    if assert_count:
        expected = len(corpus)
        actual = namespace.count()
        assert actual == expected, f"namespace {namespace.name!r} expected count {expected}, got {actual}"


def assert_peek_result(
    result: dict[str, Any],
    *,
    expected_len: int,
    allowed_ids: set[str] | None = None,
    ns_tag: str | None = None,
) -> None:
    """Assert peek result."""
    assert "ids" in result
    assert "documents" in result
    assert "metadatas" in result
    assert "embeddings" in result
    assert len(result["ids"]) == expected_len
    assert len(result["documents"]) == expected_len
    assert len(result["metadatas"]) == expected_len
    assert len(result["embeddings"]) == expected_len
    if allowed_ids is not None:
        assert set(result["ids"]).issubset(allowed_ids), (
            f"peek returned foreign ids: {set(result['ids']) - allowed_ids}"
        )
    if ns_tag is not None and result["metadatas"]:
        for meta in result["metadatas"]:
            assert meta is not None
            assert meta.get("ns_tag") == ns_tag, meta


def _default_include(
    documents: str | None,
    metadatas: dict | None,
    embeddings: list[float] | None,
    include: list[str] | None,
) -> list[str] | None:
    """Default include."""
    if include is not None:
        return include
    fields: list[str] = []
    if documents is not None:
        fields.append("documents")
    if metadatas is not None:
        fields.append("metadatas")
    if embeddings is not None:
        fields.append("embeddings")
    return fields or None


def assert_get_present(
    ns: Any,
    doc_id: str,
    *,
    documents: str | None = None,
    metadatas: dict | None = None,
    embeddings: list[float] | None = None,
    include: list[str] | None = None,
) -> dict[str, Any]:
    """Assert get present."""
    result = ns.get(ids=doc_id, include=_default_include(documents, metadatas, embeddings, include))
    indices = [i for i, rid in enumerate(result["ids"]) if rid == doc_id]
    assert indices, f"expected doc {doc_id!r}, got {result['ids']}"
    idx = indices[0]
    if documents is not None:
        assert result["documents"][idx] == documents
    if metadatas is not None:
        assert result["metadatas"][idx] == metadatas
    if embeddings is not None:
        assert result["embeddings"][idx] == embeddings
    return result


def assert_get_absent(ns: Any, doc_id: str) -> None:
    """Assert get absent."""
    result = ns.get(ids=doc_id)
    assert len(result["ids"]) == 0, f"expected doc {doc_id!r} absent, got {result['ids']}"
