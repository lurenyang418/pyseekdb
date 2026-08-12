"""
Namespace hybrid_search tests: 2 collections x 2 namespaces.

Extends FTS multi-coll coverage with search-index, KNN, and combined scenarios.
"""

from __future__ import annotations

import pytest
from namespace_fts_helpers import (
    MULTI_COLL_MULTI_NS_QUADRANT_KEYS,
    VectorDistanceMetric,
    assert_hybrid_search_no_hits,
    get_fts_case,
    run_hybrid_search_fts_case,
    run_hybrid_search_fts_case_all_quadrants,
)
from namespace_hybrid_search_helpers import (
    assert_hybrid_search_index_no_hits,
    get_hybrid_combined_case,
    get_search_index_case,
    get_vector_knn_case,
    run_hybrid_combined_case,
    run_hybrid_knn_case,
    run_hybrid_search_index_case,
    run_knn_case_on_quadrants,
    run_search_index_case_on_quadrants,
    setup_multi_coll_multi_ns_fts,
    setup_multi_coll_multi_ns_fts_single_loaded,
    teardown_multi_coll_multi_ns_fts,
)


class TestNamespaceHybridSearchMultiCollMultiNs:
    """TestNamespaceHybridSearchMultiCollMultiNs class."""

    def test_cross_quadrant_search_index_isolation(self, db_client):
        """Only the loaded quadrant returns ``has_both`` rows."""
        ctx = setup_multi_coll_multi_ns_fts_single_loaded(db_client, loaded_quadrant="c1_x")
        try:
            case = get_search_index_case("and_has_both_zpx_gte")
            corpus, ns_loaded = ctx["c1_x"]
            run_hybrid_search_index_case(ns_loaded, corpus, case)
            for key in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
                if key == "c1_x":
                    continue
                _, namespace = ctx[key]
                assert_hybrid_search_index_no_hits(namespace, case.where, n_results=case.n_results)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    def test_cross_quadrant_fts_isolation(self, db_client):
        """Test cross quadrant fts isolation."""
        ctx = setup_multi_coll_multi_ns_fts_single_loaded(db_client, loaded_quadrant="c1_x")
        try:
            case = get_fts_case("contains_token_zpx")
            corpus, ns_loaded = ctx["c1_x"]
            run_hybrid_search_fts_case(ns_loaded, corpus, case)
            for key in MULTI_COLL_MULTI_NS_QUADRANT_KEYS:
                if key == "c1_x":
                    continue
                _, namespace = ctx[key]
                assert_hybrid_search_no_hits(namespace, case.where_document, n_results=case.n_results)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    @pytest.mark.parametrize(
        "case_name",
        [
            "contains_token_zpx",
            "not_contains_token_zpx",
            "and_zpx_alp",
            "or_zpx_alp",
            "contains_zpx_filter_has_both",
            "contains_zpx_filter_zpx_hint_gte_40",
        ],
    )
    def test_hybrid_search_fulltext_all_loaded_quadrants(self, db_client, case_name: str):
        """Test hybrid search fulltext all loaded quadrants."""
        ctx = setup_multi_coll_multi_ns_fts(db_client)
        try:
            run_hybrid_search_fts_case_all_quadrants(ctx, case_name)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    @pytest.mark.parametrize(
        "case_name",
        [
            "eq_zpx_hint_direct",
            "gte_zpx_hint_40",
            "in_alp_hint_values",
            "and_has_both_zpx_gte",
            "or_zpx_alp_hint_high",
            "id_in_zpx_tops",
        ],
    )
    def test_hybrid_search_search_index_all_loaded_quadrants(self, db_client, case_name: str):
        """Test hybrid search search index all loaded quadrants."""
        ctx = setup_multi_coll_multi_ns_fts(db_client)
        try:
            run_search_index_case_on_quadrants(ctx, case_name)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    @pytest.mark.parametrize("vector_distance", ["l2", "cosine"])
    @pytest.mark.parametrize("case_name", ["knn_global_top5", "knn_filter_has_both"])
    def test_hybrid_search_vector_all_loaded_quadrants(
        self, db_client, vector_distance: VectorDistanceMetric, case_name: str
    ):
        """Test hybrid search vector all loaded quadrants."""
        ctx = setup_multi_coll_multi_ns_fts(db_client, distance=vector_distance)
        try:
            run_knn_case_on_quadrants(ctx, case_name, distance_metric=vector_distance)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    def test_hybrid_search_fts_plus_search_index_multi_coll(self, db_client):
        """Test hybrid search fts plus search index multi coll."""
        ctx = setup_multi_coll_multi_ns_fts(db_client)
        try:
            for key in ("c1_x", "c2_y"):
                corpus, namespace = ctx[key]
                run_hybrid_search_fts_case(namespace, corpus, get_fts_case("contains_zpx_filter_has_both"))
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    @pytest.mark.parametrize("vector_distance", ["l2", "cosine"])
    def test_hybrid_search_combined_rrf_multi_coll(self, db_client, vector_distance: VectorDistanceMetric):
        """Test hybrid search combined rrf multi coll."""
        ctx = setup_multi_coll_multi_ns_fts(db_client, distance=vector_distance)
        try:
            case = get_hybrid_combined_case("fts_zpx_filter_gte_knn")
            for key in ("c1_x", "c2_y"):
                corpus, namespace = ctx[key]
                run_hybrid_combined_case(namespace, corpus, case, distance_metric=vector_distance)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)

    @pytest.mark.parametrize("vector_distance", ["l2", "cosine"])
    def test_hybrid_search_vector_plus_search_index_multi_coll(self, db_client, vector_distance: VectorDistanceMetric):
        """Test hybrid search vector plus search index multi coll."""
        ctx = setup_multi_coll_multi_ns_fts(db_client, distance=vector_distance)
        try:
            knn_case = get_vector_knn_case("knn_filter_has_both")
            for key in ("c1_x", "c2_y"):
                corpus, namespace = ctx[key]
                run_hybrid_knn_case(namespace, corpus, knn_case, distance_metric=vector_distance)
        finally:
            teardown_multi_coll_multi_ns_fts(db_client, ctx)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
