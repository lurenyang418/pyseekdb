"""
Namespace triple-branch hybrid_search: 2 collections x 2 loaded namespaces.

Extends ``test_namespace_hybrid_search_triple_branch.py`` with multi-collection /
multi-namespace isolation and cross-quadrant correctness checks.

Run one case::

    pytest tests/integration_tests/test_namespace_hybrid_search_triple_branch_multi_coll_multi_ns.py \\
        -k "intersection_both_tokens and oceanbase" -v -s
"""

from __future__ import annotations

import time

import pytest
from namespace_dml_helpers import NAMESPACE_TEST_PARTITION_COUNT
from namespace_fts_helpers import (
    CORPUS_SIZE,
    MULTI_COLL_MULTI_NS_FTS_LOADED_QUADRANTS,
    MULTI_COLL_MULTI_NS_QUADRANT_KEYS,
    TOKEN_ZPX,
    CorpusRecord,
    VectorDistanceMetric,
    build_large_fts_corpus,
    insert_corpus_in_batches,
    ns_schema,
)
from namespace_hybrid_search_helpers import (
    TRIPLE_BRANCH_CASES,
    assert_hybrid_search_index_no_hits,
    assert_hybrid_triple_branch_no_hits,
    corpus_matches_triple_intersection,
    get_triple_branch_case,
    run_hybrid_triple_branch_case,
    run_triple_branch_case_on_quadrants,
    setup_multi_coll_multi_ns_fts,
    setup_multi_coll_multi_ns_fts_single_loaded,
    teardown_multi_coll_multi_ns_fts,
)

_EMPTY_NAMESPACE_KEYS: tuple[str, ...] = ("c1_empty", "c2_empty")
_INDEX_SETTLE_SECONDS = 3

_FTS_TRIPLE_CASE_NAMES: tuple[str, ...] = tuple(
    c.name for c in TRIPLE_BRANCH_CASES if c.verify in ("fts", "not_contains")
)
_SI_TRIPLE_CASE_NAMES: tuple[str, ...] = tuple(c.name for c in TRIPLE_BRANCH_CASES if c.verify == "search_index")
_KNN_TRIPLE_CASE_NAMES: tuple[str, ...] = tuple(c.name for c in TRIPLE_BRANCH_CASES if c.verify == "knn")
_INTERSECTION_CASE_NAMES: tuple[str, ...] = tuple(c.name for c in TRIPLE_BRANCH_CASES if c.verify == "intersection")
_RRF_CASE_NAMES: tuple[str, ...] = tuple(c.name for c in TRIPLE_BRANCH_CASES if c.verify == "rrf_fusion")


def _namespace_corpus(base_corpus: list[CorpusRecord], key: str) -> list[CorpusRecord]:
    """Namespace corpus."""
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


def _setup_multi_coll_multi_ns_isolation_triple(
    db_client,
    *,
    distance: VectorDistanceMetric = "l2",
):
    """2 collections x 2 loaded namespaces with distinct per-namespace corpus markers."""
    ts = int(time.time() * 1000)
    coll_1 = db_client.create_collection(
        name=f"test_ns_hs_tb_mcmn_iso_{distance}_{ts}_c1",
        schema=ns_schema(distance),
        use_namespace=True,
        partition_count=NAMESPACE_TEST_PARTITION_COUNT,
    )
    coll_2 = db_client.create_collection(
        name=f"test_ns_hs_tb_mcmn_iso_{distance}_{ts}_c2",
        schema=ns_schema(distance),
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


class TestNamespaceHybridSearchTripleBranchMultiCollMultiNs:
    """TestNamespaceHybridSearchTripleBranchMultiCollMultiNs class."""

    @pytest.mark.parametrize("vector_distance", ["l2", "cosine"])
    def test_cross_quadrant_triple_branch_isolation(self, db_client, vector_distance: VectorDistanceMetric):
        """Loaded namespaces return only their own rows; empty namespaces stay empty."""
        ctx = _setup_multi_coll_multi_ns_isolation_triple(db_client, distance=vector_distance)
        try:
            case = get_triple_branch_case("intersection_both_tokens")
            for key in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
                corpus, namespace = ctx[key]
                result = run_hybrid_triple_branch_case(
                    namespace,
                    corpus,
                    case,
                    distance_metric=vector_distance,
                )
                ids = result["ids"][0]
                assert ids
                assert all(doc_id.startswith(f"{key}_") for doc_id in ids), (
                    f"[{key}] triple-branch hybrid_search leaked rows: {ids[:10]!r}"
                )

            for key in _EMPTY_NAMESPACE_KEYS:
                namespace = ctx[key]
                assert_hybrid_triple_branch_no_hits(namespace, case)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    def test_cross_quadrant_search_index_branch_isolation(self, db_client):
        """Test cross quadrant search index branch isolation."""
        ctx = setup_multi_coll_multi_ns_fts_single_loaded(db_client, loaded_quadrant="c1_x")
        try:
            case = get_triple_branch_case("si_and_has_both_zpx_gte")
            corpus, ns_loaded = ctx["c1_x"]
            run_hybrid_triple_branch_case(ns_loaded, corpus, case)
            for key in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
                if key == "c1_x":
                    continue
                _, namespace = ctx[key]
                assert_hybrid_search_index_no_hits(namespace, case.where, n_results=case.n_results)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    @pytest.mark.parametrize("vector_distance", ["l2", "cosine"])
    def test_triple_branch_top1_fts_all_quadrants(self, db_client, vector_distance: VectorDistanceMetric):
        """Test triple branch top1 fts all quadrants."""
        ctx = _setup_multi_coll_multi_ns_isolation_triple(db_client, distance=vector_distance)
        try:
            for key in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
                _, namespace = ctx[key]
                top_result = namespace.hybrid_search(
                    query={
                        "where_document": {"$contains": TOKEN_ZPX},
                        "where": {"rel_hint": {"$gte": 0}},
                        "n_results": 1,
                    },
                    knn={
                        "query_embeddings": [1.0, 1.0, 0.0],
                        "where": {"rel_hint": {"$gte": 0}},
                        "n_results": 20,
                    },
                    n_results=1,
                    include=["documents"],
                )
                assert top_result["ids"][0][0] == f"{key}_zpx_top_5", (
                    f"[{key}] FTS top-1 must be zpx_top_5, got {top_result['ids'][0]}"
                )
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    @pytest.mark.parametrize("case_name", _FTS_TRIPLE_CASE_NAMES)
    def test_hybrid_search_triple_branch_fts_all_loaded_quadrants(self, db_client, case_name: str):
        """Test hybrid search triple branch fts all loaded quadrants."""
        ctx = setup_multi_coll_multi_ns_fts(db_client)
        try:
            run_triple_branch_case_on_quadrants(ctx, case_name)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    @pytest.mark.parametrize("case_name", _SI_TRIPLE_CASE_NAMES)
    def test_hybrid_search_triple_branch_search_index_all_loaded_quadrants(self, db_client, case_name: str):
        """Test hybrid search triple branch search index all loaded quadrants."""
        ctx = setup_multi_coll_multi_ns_fts(db_client)
        try:
            run_triple_branch_case_on_quadrants(ctx, case_name)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    @pytest.mark.parametrize("vector_distance", ["l2", "cosine"])
    @pytest.mark.parametrize("case_name", _KNN_TRIPLE_CASE_NAMES)
    def test_hybrid_search_triple_branch_knn_all_loaded_quadrants(
        self, db_client, vector_distance: VectorDistanceMetric, case_name: str
    ):
        """Test hybrid search triple branch knn all loaded quadrants."""
        ctx = setup_multi_coll_multi_ns_fts(db_client, distance=vector_distance)
        try:
            run_triple_branch_case_on_quadrants(ctx, case_name, distance_metric=vector_distance)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    @pytest.mark.parametrize("vector_distance", ["l2", "cosine"])
    @pytest.mark.parametrize("case_name", _INTERSECTION_CASE_NAMES)
    def test_hybrid_search_triple_branch_intersection_all_loaded_quadrants(
        self, db_client, vector_distance: VectorDistanceMetric, case_name: str
    ):
        """Test hybrid search triple branch intersection all loaded quadrants."""
        ctx = setup_multi_coll_multi_ns_fts(db_client, distance=vector_distance)
        try:
            run_triple_branch_case_on_quadrants(ctx, case_name, distance_metric=vector_distance)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    @pytest.mark.parametrize("vector_distance", ["l2", "cosine"])
    @pytest.mark.parametrize("case_name", _RRF_CASE_NAMES)
    def test_hybrid_search_triple_branch_rrf_all_loaded_quadrants(
        self, db_client, vector_distance: VectorDistanceMetric, case_name: str
    ):
        """Test hybrid search triple branch rrf all loaded quadrants."""
        ctx = setup_multi_coll_multi_ns_fts(db_client, distance=vector_distance)
        try:
            run_triple_branch_case_on_quadrants(ctx, case_name, distance_metric=vector_distance)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    @pytest.mark.parametrize("vector_distance", ["l2", "cosine"])
    def test_hybrid_search_triple_branch_intersection_subset_on_quadrants(
        self, db_client, vector_distance: VectorDistanceMetric
    ):
        """Every hit must lie in the triple-branch intersection on each loaded quadrant."""
        ctx = setup_multi_coll_multi_ns_fts(db_client, distance=vector_distance)
        try:
            case = get_triple_branch_case("intersection_zpx_gte40")
            for key in MULTI_COLL_MULTI_NS_FTS_LOADED_QUADRANTS:
                corpus, namespace = ctx[key]
                result = run_hybrid_triple_branch_case(
                    namespace,
                    corpus,
                    case,
                    distance_metric=vector_distance,
                )
                corpus_by_id = {rec.doc_id: rec for rec in corpus}
                for doc_id in result["ids"][0]:
                    assert corpus_matches_triple_intersection(corpus_by_id[doc_id], case), (
                        f"[{key}] id={doc_id!r} outside intersection"
                    )
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
