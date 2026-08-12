"""
Namespace hybrid_search full-text tests: 2 collections x 2 loaded namespaces
plus one empty namespace per collection (use_namespace=True).

Verifies full-text correctness in each loaded namespace, cross-namespace data
isolation, and empty namespace isolation.

Run one case in isolation, e.g.::

    pytest tests/integration_tests/test_namespace_hybrid_search_fulltext_multi_coll_multi_ns.py \\
        -k "or_zpx_alp and oceanbase" -v -s
"""

from __future__ import annotations

import time

import pytest
from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT
from namespace_fts_helpers import (
    CORPUS_SIZE,
    MULTI_COLL_MULTI_NS_QUADRANT_KEYS,
    TOKEN_ZPX,
    CorpusRecord,
    assert_hybrid_search_no_hits,
    assert_not_contains_no_token_leak,
    build_large_fts_corpus,
    get_fts_case,
    insert_corpus_in_batches,
    ns_schema,
    run_hybrid_search_fts_case,
    teardown_multi_coll_multi_ns_fts,
)

_EMPTY_NAMESPACE_KEYS: tuple[str, ...] = ("c1_empty", "c2_empty")
_INDEX_SETTLE_SECONDS = 3


def _namespace_corpus(base_corpus: list[CorpusRecord], key: str) -> list[CorpusRecord]:
    """Give each namespace a distinct id/document/metadata marker."""
    return [
        CorpusRecord(
            doc_id=f"{key}_{record.doc_id}",
            document=f"[{key}] {record.document}",
            embedding=record.embedding,
            metadata={**record.metadata, "namespace_marker": key},
            rel_hint=record.rel_hint,
        )
        for record in base_corpus
    ]


def _setup_multi_coll_multi_ns_isolation_fts(db_client):
    """
    2 collections x 2 loaded namespaces, plus one empty namespace per collection.

    Every loaded namespace receives its own marked corpus, so a leaked row from a
    sibling namespace fails the existing corpus-based assertions.
    """
    ts = int(time.time() * 1000)
    coll_1 = db_client.create_collection(
        name=f"test_ns_hs_ft_mcmn_iso_{ts}_c1",
        schema=ns_schema(),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )
    coll_2 = db_client.create_collection(
        name=f"test_ns_hs_ft_mcmn_iso_{ts}_c2",
        schema=ns_schema(),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )
    ctx: dict[str, object] = {"coll_1": coll_1, "coll_2": coll_2}
    for coll_tag, collection in (("c1", coll_1), ("c2", coll_2)):
        for ns_suffix in ("x", "y"):
            ns = collection.create_namespace(f"ns_{ns_suffix}")
            ns.prewarm()
            ctx[f"{coll_tag}_{ns_suffix}"] = ns
        empty_ns = collection.create_namespace("ns_empty")
        empty_ns.prewarm()
        ctx[f"{coll_tag}_empty"] = empty_ns

    base_corpus = build_large_fts_corpus(CORPUS_SIZE)
    if len(base_corpus) <= 1000:
        raise ValueError(f"corpus must exceed 1000 rows, got {len(base_corpus)}")

    for key in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
        corpus = _namespace_corpus(base_corpus, key)
        namespace = ctx[key]
        insert_corpus_in_batches(namespace, corpus)
        ctx[key] = (corpus, namespace)

    time.sleep(_INDEX_SETTLE_SECONDS)
    return ctx


class TestNamespaceHybridSearchFulltextMultiCollMultiNs:
    """Large-scale hybrid_search FTS across 2 collections x 2 loaded namespaces."""

    def _run_fts_case_all_quadrants(self, db_client, case_name: str) -> None:
        """Run fts case all quadrants."""
        ctx = _setup_multi_coll_multi_ns_isolation_fts(db_client)
        try:
            case = get_fts_case(case_name)
            for key in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
                corpus, namespace = ctx[key]
                run_hybrid_search_fts_case(namespace, corpus, case)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    def test_cross_quadrant_fts_isolation(self, db_client):
        """Loaded namespaces must see only their own rows; empty namespaces return no hits."""
        ctx = _setup_multi_coll_multi_ns_isolation_fts(db_client)
        try:
            case = get_fts_case("contains_token_zpx")
            for key in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
                corpus, namespace = ctx[key]
                result = run_hybrid_search_fts_case(namespace, corpus, case)
                ids = result["ids"][0]
                assert ids
                assert all(doc_id.startswith(f"{key}_") for doc_id in ids), (
                    f"[{key}] hybrid_search leaked rows from another namespace: {ids[:10]!r}"
                )

            for key in _EMPTY_NAMESPACE_KEYS:
                namespace = ctx[key]
                assert_hybrid_search_no_hits(
                    namespace,
                    case.where_document,
                    n_results=case.n_results,
                )
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    def test_hybrid_search_fulltext_contains_token_zpx(self, db_client):
        """Test hybrid search fulltext contains token zpx."""
        self._run_fts_case_all_quadrants(db_client, "contains_token_zpx")

    def test_hybrid_search_fulltext_contains_token_alp(self, db_client):
        """Test hybrid search fulltext contains token alp."""
        self._run_fts_case_all_quadrants(db_client, "contains_token_alp")

    def test_hybrid_search_fulltext_contains_string_shorthand(self, db_client):
        """Test hybrid search fulltext contains string shorthand."""
        self._run_fts_case_all_quadrants(db_client, "string_shorthand_token_zpx")

    def test_hybrid_search_fulltext_not_contains(self, db_client):
        """Test hybrid search fulltext not contains."""
        case = get_fts_case("not_contains_token_zpx")
        ctx = _setup_multi_coll_multi_ns_isolation_fts(db_client)
        try:
            for key in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
                corpus, namespace = ctx[key]
                result = run_hybrid_search_fts_case(namespace, corpus, case)
                assert_not_contains_no_token_leak(corpus, result, TOKEN_ZPX)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    def test_hybrid_search_fulltext_and_zpx_alp(self, db_client):
        """Test hybrid search fulltext and zpx alp."""
        self._run_fts_case_all_quadrants(db_client, "and_zpx_alp")

    def test_hybrid_search_fulltext_and_multi_contains(self, db_client):
        """Test hybrid search fulltext and multi contains."""
        self._run_fts_case_all_quadrants(db_client, "and_multi_contains_phrase")

    def test_hybrid_search_fulltext_or_zpx_alp(self, db_client):
        """Test hybrid search fulltext or zpx alp."""
        self._run_fts_case_all_quadrants(db_client, "or_zpx_alp")

    def test_hybrid_search_fulltext_top1_contains_zpx(self, db_client):
        """Test hybrid search fulltext top1 contains zpx."""
        ctx = _setup_multi_coll_multi_ns_isolation_fts(db_client)
        try:
            for key in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
                _, namespace = ctx[key]
                top_result = namespace.hybrid_search(
                    query={"where_document": {"$contains": TOKEN_ZPX}, "n_results": 1},
                    n_results=1,
                    include=["documents"],
                )
                assert top_result["ids"][0][0] == f"{key}_zpx_top_5", (
                    f"[{key}] most relevant TOKEN_ZPX document must rank first, got {top_result['ids'][0]}"
                )
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
