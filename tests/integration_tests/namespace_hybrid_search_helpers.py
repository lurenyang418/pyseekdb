"""
Helpers for namespace hybrid_search vector / search-index / combined integration tests.

Reuses the large deterministic corpus from ``namespace_fts_helpers`` so full-text,
metadata (JSON SEARCH INDEX), and vector branches share one ground-truth dataset.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Literal

from namespace_fts_helpers import (
    BATCH_SIZE,
    CORPUS_SIZE,
    INDEX_SETTLE_SECONDS,
    MULTI_COLL_MULTI_NS_FTS_LOADED_QUADRANTS,
    MULTI_COLL_MULTI_NS_QUADRANT_KEYS,
    TOKEN_ALP,
    TOKEN_ZPX,
    VECTOR_DISTANCE_METRICS,
    CorpusRecord,
    VectorDistanceMetric,
    assert_hybrid_fulltext_result,
    assert_not_contains_no_token_leak,
    build_large_fts_corpus,
    doc_matches_where_document,
    doc_matches_where_metadata,
    flat_hybrid_search_schema,
    get_fts_case,
    insert_corpus_into_collection,
    run_hybrid_search_fts_case,
    setup_fts_namespace_with_corpus,
    setup_large_fts_collection,
    setup_multi_coll_multi_ns_fts,
    setup_multi_coll_multi_ns_fts_single_loaded,
    teardown_large_fts_collection,
    teardown_multi_coll_multi_ns_fts,
)

# Fixed query vector for KNN tests (same dimension as corpus embeddings).
KNN_QUERY_VECTOR: list[float] = [1.0, 1.0, 0.0]

RRF_RANK = {"rrf": {"rank_window_size": 60, "rank_constant": 60}}

# Broad filters so a third branch stays active without narrowing the primary signal under test.
WHERE_DOCUMENT_UNIVERSAL: dict[str, str] = {"$contains": "document"}
WHERE_BROAD_METADATA: dict[str, Any] = {"rel_hint": {"$gte": 0}}
# Filler rows start after the deterministic FTS tier block (~43 rows).
WHERE_FILLER_SEQ: dict[str, Any] = {"seq": {"$gte": 43}}
# ``both_tokens`` row — use numeric fields for baseline-compatible filters.
# Boolean ``has_both`` is not filterable on flat collection (HNSW / GET_SQL) path.
WHERE_BOTH_TOKENS_NUMERIC: dict[str, Any] = {
    "$and": [{"zpx_hint": 5}, {"alp_hint": 5}],
}
WHERE_NOT_BOTH_TOKENS_NUMERIC: dict[str, Any] = {
    "$not": {"$and": [{"zpx_hint": 5}, {"alp_hint": 5}]},
}


@dataclass(frozen=True)
class SearchIndexQueryCase:
    """SearchIndexQueryCase class."""

    name: str
    where: dict[str, Any]
    n_results: int
    min_hits: int = 1
    exact_match_count: int | None = None


@dataclass(frozen=True)
class VectorKnnCase:
    """VectorKnnCase class."""

    name: str
    query_vector: list[float]
    n_results: int
    where: dict[str, Any] | None = None
    check_top1: bool = True


@dataclass(frozen=True)
class HybridCombinedCase:
    """Multi-branch hybrid_search (full-text + metadata + optional KNN)."""

    name: str
    n_results: int
    where_document: dict[str, Any] | str | None = None
    where: dict[str, Any] | None = None
    knn: dict[str, Any] | None = None
    use_rrf: bool = False
    check_fts_ranking: bool = True
    # When True, only assert non-empty results in corpus (RRF reorders across branches).
    rrf_smoke_only: bool = False


def corpus_matches_where(record: CorpusRecord, where: dict[str, Any]) -> bool:
    """Metadata / ``#id`` filter against a corpus row (hybrid_search ``query.where`` / ``knn.where``)."""
    if "$and" in where:
        return all(corpus_matches_where(record, sub) for sub in where["$and"])
    if "$or" in where:
        return any(corpus_matches_where(record, sub) for sub in where["$or"])
    if "$not" in where:
        return not corpus_matches_where(record, where["$not"])

    for key, value in where.items():
        if key in ("$and", "$or", "$not"):
            continue
        if key == "#id":
            if isinstance(value, dict):
                if "$in" in value and record.doc_id not in value["$in"]:
                    return False
                if "$nin" in value and record.doc_id in value["$nin"]:
                    return False
                if "$eq" in value and record.doc_id != value["$eq"]:
                    return False
                if "$ne" in value and record.doc_id == value["$ne"]:
                    return False
            elif record.doc_id != value:
                return False
            continue
        if not doc_matches_where_metadata(record.metadata, {key: value}):
            return False
    return True


def count_corpus_matches(corpus: list[CorpusRecord], where: dict[str, Any]) -> int:
    """Count corpus matches."""
    return sum(1 for rec in corpus if corpus_matches_where(rec, where))


def l2_squared(a: list[float], b: list[float]) -> float:
    """L2 squared."""
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))


def _vector_l2_norm(vec: list[float]) -> float:
    """Vector l2 norm."""
    return math.sqrt(sum(x * x for x in vec))


def knn_compare_distance(
    a: list[float],
    b: list[float],
    metric: VectorDistanceMetric = "l2",
) -> float:
    """Distance value used for KNN ground-truth ordering (smaller = nearer)."""
    if metric == "l2":
        return l2_squared(a, b)
    if metric == "cosine":
        norm_a = _vector_l2_norm(a)
        norm_b = _vector_l2_norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 2.0
        cos_sim = sum(x * y for x, y in zip(a, b, strict=False)) / (norm_a * norm_b)
        return 1.0 - cos_sim
    raise ValueError(f"unsupported vector distance metric: {metric!r}")


def ensure_shared_hybrid_search_collection(
    shared_cache: dict[str, dict[str, Any]],
    db_client: Any,
    request: Any,
    vector_distance: VectorDistanceMetric,
) -> dict[str, Any]:
    """Return or create a shared (db mode, distance metric) corpus + collection entry."""
    mode = request.node.callspec.params["db_client"] if request.node.callspec else "default"
    cache_key = f"{mode}:{vector_distance}"
    if cache_key not in shared_cache:
        corpus, collection = setup_large_fts_collection(db_client, distance=vector_distance)
        shared_cache[cache_key] = {
            "db_client": db_client,
            "corpus": corpus,
            "collection": collection,
            "vector_distance": vector_distance,
        }
    return shared_cache[cache_key]


def _knn_scored_rows(
    corpus: list[CorpusRecord],
    query_vector: list[float],
    where: dict[str, Any] | None = None,
    *,
    distance_metric: VectorDistanceMetric = "l2",
) -> list[tuple[str, float, int]]:
    """(doc_id, compare_distance, metadata seq) for rows matching *where*."""
    rows: list[tuple[str, float, int]] = []
    for rec in corpus:
        if where is not None and not corpus_matches_where(rec, where):
            continue
        seq = rec.metadata.get("seq", 0)
        if not isinstance(seq, int):
            seq = 0
        rows.append((
            rec.doc_id,
            knn_compare_distance(rec.embedding, query_vector, distance_metric),
            seq,
        ))
    return rows


def nearest_knn_doc_ids(
    corpus: list[CorpusRecord],
    query_vector: list[float],
    where: dict[str, Any] | None = None,
    *,
    distance_metric: VectorDistanceMetric = "l2",
    tol: float = 1e-9,
) -> set[str]:
    """Doc ids at minimum compare distance (ties allowed)."""
    scored = _knn_scored_rows(corpus, query_vector, where, distance_metric=distance_metric)
    if not scored:
        return set()
    min_d = min(d for _, d, _ in scored)
    return {doc_id for doc_id, d, _ in scored if math.isclose(d, min_d, abs_tol=tol)}


def expected_knn_ids(
    corpus: list[CorpusRecord],
    query_vector: list[float],
    n_results: int,
    where: dict[str, Any] | None = None,
    *,
    distance_metric: VectorDistanceMetric = "l2",
) -> list[str]:
    """Expected knn ids."""
    scored = _knn_scored_rows(corpus, query_vector, where, distance_metric=distance_metric)
    # Tie-break: distance, then seq (matches OB hybrid_search), then doc_id.
    scored.sort(key=lambda item: (item[1], item[2], item[0]))
    return [doc_id for doc_id, _, _ in scored[:n_results]]


def assert_hybrid_search_index_result(
    corpus: list[CorpusRecord],
    result: dict[str, Any],
    where: dict[str, Any],
    n_results: int,
    *,
    min_hits: int = 1,
    exact_match_count: int | None = None,
) -> None:
    """Assert hybrid search index result."""
    assert result is not None
    assert result.get("ids")
    ids = result["ids"][0]
    assert len(ids) <= n_results
    assert len(ids) >= min_hits, f"expected at least {min_hits} search-index hits"

    id_to_meta = {rec.doc_id: rec.metadata for rec in corpus}
    for doc_id in ids:
        assert doc_id in id_to_meta, f"unknown id {doc_id!r}"
        assert corpus_matches_where(next(rec for rec in corpus if rec.doc_id == doc_id), where), (
            f"id={doc_id!r} does not satisfy where={where!r}"
        )

    total_matches = count_corpus_matches(corpus, where)
    if exact_match_count is not None:
        assert total_matches == exact_match_count, (
            f"test case assumes {exact_match_count} corpus matches, got {total_matches}"
        )
    if exact_match_count is not None and exact_match_count <= n_results:
        assert len(ids) == exact_match_count
        expected_ids = {rec.doc_id for rec in corpus if corpus_matches_where(rec, where)}
        assert set(ids) == expected_ids
    elif len(ids) == n_results and total_matches > n_results:
        # No higher-priority scalar key — every returned row is valid; ensure we did not
        # miss matches only when the engine returned a full page and corpus is small.
        pass


def assert_hybrid_knn_result(
    corpus: list[CorpusRecord],
    result: dict[str, Any],
    query_vector: list[float],
    n_results: int,
    where: dict[str, Any] | None = None,
    *,
    distance_metric: VectorDistanceMetric = "l2",
    check_top1: bool = True,
    check_score_order: bool = True,
) -> None:
    """Assert hybrid_search KNN branch results against corpus ground truth.

    ``result["distances"]`` comes from OB ``__score`` via the client (higher = nearer),
    not raw vector distance — only score monotonicity is checked here. Id order uses
    corpus ``distance_metric`` ground truth.
    """
    assert result is not None
    assert result.get("ids")
    ids = result["ids"][0]
    scores = result.get("distances", [[]])[0] if result.get("distances") else []
    assert len(ids) <= n_results
    assert len(ids) > 0, "expected at least one KNN hit"

    corpus_by_id = {rec.doc_id: rec for rec in corpus}
    for doc_id in ids:
        rec = corpus_by_id[doc_id]
        if where is not None:
            assert corpus_matches_where(rec, where), f"id={doc_id!r} metadata does not satisfy knn.where={where!r}"

    if scores:
        assert len(scores) == len(ids)
        if check_score_order:
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1] or math.isclose(scores[i], scores[i + 1]), (
                    f"KNN scores should be non-increasing (higher = nearer, {distance_metric} index): {scores!r}"
                )

    expected = expected_knn_ids(corpus, query_vector, n_results, where, distance_metric=distance_metric)
    if len(ids) == len(expected):
        assert ids == expected, (
            f"KNN id order mismatch ({distance_metric} ground truth): got {ids!r} expected {expected!r}"
        )
    elif check_top1 and expected:
        nearest = nearest_knn_doc_ids(corpus, query_vector, where, distance_metric=distance_metric)
        assert ids[0] in nearest, (
            f"top-1 KNN must be among nearest neighbors ({distance_metric} tie set), "
            f"got {ids[0]!r} expected one of {sorted(nearest)[:8]!r}"
            f"{'…' if len(nearest) > 8 else ''}"
        )

    if len(ids) == n_results:
        scored = _knn_scored_rows(corpus, query_vector, where, distance_metric=distance_metric)
        worst_d = max(d for doc_id, d, _ in scored if doc_id in ids) if scored else float("inf")
        for doc_id, d, _ in scored:
            if doc_id not in ids and d < worst_d - 1e-9:
                raise AssertionError(f"closer match {doc_id!r} ({distance_metric}={d}) missing from top-{n_results}")


def run_hybrid_search_index_case(
    namespace: Any,
    corpus: list[CorpusRecord],
    case: SearchIndexQueryCase,
) -> dict[str, Any]:
    """Run hybrid search index case."""
    result = namespace.hybrid_search(
        query={"where": case.where, "n_results": case.n_results},
        n_results=case.n_results,
        include=["metadatas"],
    )
    assert_hybrid_search_index_result(
        corpus,
        result,
        case.where,
        case.n_results,
        min_hits=case.min_hits,
        exact_match_count=case.exact_match_count,
    )
    return result


def run_hybrid_knn_case(
    namespace: Any,
    corpus: list[CorpusRecord],
    case: VectorKnnCase,
    *,
    distance_metric: VectorDistanceMetric = "l2",
) -> dict[str, Any]:
    """Run hybrid knn case."""
    knn: dict[str, Any] = {
        "query_embeddings": case.query_vector,
        "n_results": case.n_results,
    }
    if case.where is not None:
        knn["where"] = case.where
    result = namespace.hybrid_search(
        knn=knn,
        n_results=case.n_results,
        include=["metadatas", "distances"],
    )
    assert_hybrid_knn_result(
        corpus,
        result,
        case.query_vector,
        case.n_results,
        case.where,
        distance_metric=distance_metric,
        check_top1=case.check_top1,
    )
    return result


def run_hybrid_combined_case(
    namespace: Any,
    corpus: list[CorpusRecord],
    case: HybridCombinedCase,
    *,
    distance_metric: VectorDistanceMetric = "l2",
) -> dict[str, Any]:
    """Run hybrid combined case."""
    query: dict[str, Any] | None = None
    if case.where_document is not None or case.where is not None:
        query = {"n_results": case.n_results}
        if case.where_document is not None:
            query["where_document"] = case.where_document
        if case.where is not None:
            query["where"] = case.where

    rank = RRF_RANK if case.use_rrf else None
    result = namespace.hybrid_search(
        query=query,
        knn=case.knn,
        rank=rank,
        n_results=case.n_results,
        include=["documents", "metadatas", "distances"],
    )

    corpus_by_id = {rec.doc_id: rec for rec in corpus}
    ids = result["ids"][0]
    assert len(ids) <= case.n_results
    assert len(ids) > 0, f"expected hybrid_search hits for case {case.name!r}"
    for doc_id in ids:
        assert doc_id in corpus_by_id, f"unknown id {doc_id!r}"

    if case.rrf_smoke_only:
        return result

    if case.knn is not None and case.where_document is None and query is None:
        vec = case.knn["query_embeddings"]
        assert_hybrid_knn_result(
            corpus,
            result,
            vec,
            case.n_results,
            case.knn.get("where"),
            distance_metric=distance_metric,
            check_top1=True,
        )
        return result

    if case.where_document is not None:
        assert_hybrid_fulltext_result(
            corpus,
            result,
            case.where_document,
            case.n_results,
            where=case.where,
            check_ranking=case.check_fts_ranking,
        )
    elif case.where is not None:
        assert_hybrid_search_index_result(corpus, result, case.where, case.n_results)
    return result


# --- Search-index-only cases (JSON SEARCH INDEX / metadata ``query.where``) ---

SEARCH_INDEX_CASES: list[SearchIndexQueryCase] = [
    SearchIndexQueryCase(
        name="eq_zpx_hint_direct",
        where={"zpx_hint": 50},
        n_results=5,
        exact_match_count=1,
    ),
    SearchIndexQueryCase(
        name="eq_zpx_hint_operator",
        where={"zpx_hint": {"$eq": 50}},
        n_results=5,
        exact_match_count=1,
    ),
    SearchIndexQueryCase(
        name="ne_zpx_hint_zero",
        where={"zpx_hint": {"$ne": 0}},
        n_results=20,
        min_hits=1,
    ),
    SearchIndexQueryCase(
        name="gte_zpx_hint_40",
        where={"zpx_hint": {"$gte": 40}},
        n_results=15,
        min_hits=2,
    ),
    SearchIndexQueryCase(
        name="lt_zpx_hint_10",
        where={"zpx_hint": {"$lt": 10}},
        n_results=20,
        min_hits=1,
    ),
    SearchIndexQueryCase(
        name="lte_alp_hint_8",
        where={"alp_hint": {"$lte": 8}},
        n_results=15,
        min_hits=1,
    ),
    SearchIndexQueryCase(
        name="gt_rel_hint_zero",
        where={"rel_hint": {"$gt": 0}},
        n_results=25,
        min_hits=5,
    ),
    SearchIndexQueryCase(
        name="in_alp_hint_values",
        where={"alp_hint": {"$in": [32, 24, 16, 8]}},
        n_results=10,
        min_hits=4,
    ),
    SearchIndexQueryCase(
        name="nin_alp_hint_high",
        where={"alp_hint": {"$nin": [32, 24, 16, 8]}},
        n_results=15,
        min_hits=1,
    ),
    SearchIndexQueryCase(
        name="and_has_both_zpx_gte",
        where={
            "$and": [
                {"has_both": True},
                {"zpx_hint": {"$gte": 5}},
            ],
        },
        n_results=5,
        exact_match_count=1,
    ),
    SearchIndexQueryCase(
        name="or_zpx_alp_hint_high",
        where={
            "$or": [
                {"zpx_hint": {"$gte": 50}},
                {"alp_hint": {"$gte": 32}},
            ],
        },
        n_results=10,
        min_hits=2,
    ),
    SearchIndexQueryCase(
        name="not_has_both",
        where={"$not": {"has_both": True}},
        n_results=20,
        min_hits=1,
    ),
    SearchIndexQueryCase(
        name="id_eq_both_tokens",
        where={"#id": "both_tokens"},
        n_results=5,
        exact_match_count=1,
    ),
    SearchIndexQueryCase(
        name="id_in_zpx_tops",
        where={"#id": {"$in": ["zpx_top_5", "zpx_top_4", "zpx_top_3"]}},
        n_results=5,
        exact_match_count=3,
    ),
]


def get_search_index_case(name: str) -> SearchIndexQueryCase:
    """Get search index case."""
    for case in SEARCH_INDEX_CASES:
        if case.name == name:
            return case
    raise KeyError(f"unknown search-index case: {name!r}")


# --- Pure KNN cases ---

VECTOR_KNN_CASES: list[VectorKnnCase] = [
    VectorKnnCase(
        name="knn_global_top5",
        query_vector=KNN_QUERY_VECTOR,
        n_results=5,
    ),
    VectorKnnCase(
        name="knn_filter_has_both",
        query_vector=KNN_QUERY_VECTOR,
        n_results=3,
        where={"has_both": True},
        check_top1=True,
    ),
    VectorKnnCase(
        name="knn_filter_zpx_hint_gte_40",
        query_vector=KNN_QUERY_VECTOR,
        n_results=5,
        where={"zpx_hint": {"$gte": 40}},
    ),
]


def get_vector_knn_case(name: str) -> VectorKnnCase:
    """Get vector knn case."""
    for case in VECTOR_KNN_CASES:
        if case.name == name:
            return case
    raise KeyError(f"unknown vector KNN case: {name!r}")


# --- Combined hybrid_search cases ---

HYBRID_COMBINED_CASES: list[HybridCombinedCase] = [
    # Full-text + search index (metadata on query branch)
    HybridCombinedCase(
        name="fts_contains_zpx_filter_has_both",
        where_document={"$contains": TOKEN_ZPX},
        where={"has_both": True},
        n_results=5,
    ),
    # Vector + search index (KNN branch filter only)
    HybridCombinedCase(
        name="knn_with_where_has_both",
        n_results=5,
        knn={
            "query_embeddings": KNN_QUERY_VECTOR,
            "where": {"has_both": True},
            "n_results": 10,
        },
    ),
    # Vector + full-text + RRF
    HybridCombinedCase(
        name="fts_or_zpx_alp_plus_knn",
        where_document={
            "$or": [
                {"$contains": TOKEN_ZPX},
                {"$contains": TOKEN_ALP},
            ],
        },
        knn={"query_embeddings": KNN_QUERY_VECTOR, "n_results": 15},
        n_results=10,
        use_rrf=True,
        rrf_smoke_only=True,
    ),
    # Vector + full-text + search index + RRF (smoke: fused ranking across three signals)
    HybridCombinedCase(
        name="fts_zpx_filter_gte_knn",
        where_document={"$contains": TOKEN_ZPX},
        where={"zpx_hint": {"$gte": 40}},
        knn={
            "query_embeddings": KNN_QUERY_VECTOR,
            "where": {"zpx_hint": {"$gte": 40}},
            "n_results": 15,
        },
        n_results=8,
        use_rrf=True,
        rrf_smoke_only=True,
    ),
    # Full-text + search index without vector (aligned with FTS case ``contains_zpx_filter_zpx_hint_and_alp_zero``)
    HybridCombinedCase(
        name="fts_zpx_filter_zpx_hint_and_alp_zero",
        where_document={"$contains": TOKEN_ZPX},
        where={
            "$and": [
                {"zpx_hint": {"$gte": 40}},
                {"alp_hint": 0},
            ],
        },
        n_results=10,
    ),
]


def get_hybrid_combined_case(name: str) -> HybridCombinedCase:
    """Get hybrid combined case."""
    for case in HYBRID_COMBINED_CASES:
        if case.name == name:
            return case
    raise KeyError(f"unknown combined hybrid case: {name!r}")


# --- Triple-branch hybrid_search (vector + full-text + search index) ---


@dataclass(frozen=True)
class HybridTripleBranchCase:
    """
    Namespace hybrid_search with ``query`` (FTS + search index) and ``knn`` all active.

    ``verify`` selects which branch supplies ground-truth assertions; the other branches
    use broad aligned filters so every signal participates without masking the operator
    under test.
    """

    name: str
    n_results: int
    verify: Literal["fts", "search_index", "knn", "intersection", "not_contains", "rrf_fusion"]
    where_document: dict[str, Any] | str | None = None
    where: dict[str, Any] | None = None
    knn: dict[str, Any] | None = None
    use_rrf: bool = False
    # Triple-branch always runs KNN; fused __score order need not match pure FTS tiers.
    check_fts_ranking: bool = False
    min_hits: int = 1
    exact_match_count: int | None = None


def _default_knn(
    n_results: int,
    *,
    where: dict[str, Any] | None = None,
    knn_n_results: int | None = None,
) -> dict[str, Any]:
    """Default knn."""
    return {
        "query_embeddings": KNN_QUERY_VECTOR,
        "n_results": knn_n_results or max(n_results * 2, 20),
        "where": where or WHERE_BROAD_METADATA,
    }


def corpus_matches_triple_intersection(
    record: CorpusRecord,
    case: HybridTripleBranchCase,
) -> bool:
    """Corpus matches triple intersection."""
    if case.where_document is not None and not doc_matches_where_document(record.document, case.where_document):
        return False
    if case.where is not None and not corpus_matches_where(record, case.where):
        return False
    if case.knn is not None:
        knn_where = case.knn.get("where")
        if knn_where is not None and not corpus_matches_where(record, knn_where):
            return False
    return True


def count_triple_intersection_matches(
    corpus: list[CorpusRecord],
    case: HybridTripleBranchCase,
) -> int:
    """Count triple intersection matches."""
    return sum(1 for rec in corpus if corpus_matches_triple_intersection(rec, case))


def _slice_hybrid_result(
    result: dict[str, Any],
    corpus: list[CorpusRecord],
    predicate,
) -> dict[str, Any]:
    """Slice hybrid result."""
    corpus_by_id = {rec.doc_id: rec for rec in corpus}
    ids = result["ids"][0]
    indices = [i for i, doc_id in enumerate(ids) if doc_id in corpus_by_id and predicate(corpus_by_id[doc_id])]
    sliced: dict[str, Any] = {"ids": [[ids[i] for i in indices]]}
    for key in ("distances", "documents", "metadatas"):
        rows = result.get(key)
        if rows and rows[0] is not None:
            sliced[key] = [[rows[0][i] for i in indices]]
    return sliced


def assert_hybrid_triple_branch_result(
    corpus: list[CorpusRecord],
    result: dict[str, Any],
    case: HybridTripleBranchCase,
    *,
    distance_metric: VectorDistanceMetric = "l2",
) -> None:
    """Assert hybrid triple branch result."""
    assert result is not None
    assert result.get("ids")
    ids = result["ids"][0]
    assert len(ids) <= case.n_results
    assert len(ids) >= case.min_hits, f"case {case.name!r}: expected at least {case.min_hits} hits, got {len(ids)}"

    corpus_by_id = {rec.doc_id: rec for rec in corpus}
    for doc_id in ids:
        assert doc_id in corpus_by_id, f"unknown id {doc_id!r}"

    if case.verify == "not_contains":
        assert case.where_document is not None
        assert_not_contains_no_token_leak(corpus, result, TOKEN_ZPX)
        if case.where is not None:
            for doc_id in ids:
                assert doc_matches_where_metadata(corpus_by_id[doc_id].metadata, case.where), (
                    f"id={doc_id!r} violates query.where={case.where!r}"
                )
        if case.knn and case.knn.get("where"):
            for doc_id in ids:
                assert corpus_matches_where(corpus_by_id[doc_id], case.knn["where"]), (
                    f"id={doc_id!r} violates knn.where={case.knn['where']!r}"
                )
        return

    if case.verify == "fts":
        assert case.where_document is not None

        def _fts_row(rec: CorpusRecord) -> bool:
            """Fts row."""
            if not doc_matches_where_document(rec.document, case.where_document):
                return False
            return not (case.where is not None and not doc_matches_where_metadata(rec.metadata, case.where))

        fts_result = _slice_hybrid_result(result, corpus, _fts_row)
        fts_ids = fts_result["ids"][0]
        assert len(fts_ids) >= 1, f"case {case.name!r}: expected at least one FTS-matching row in hybrid result"
        assert_hybrid_fulltext_result(
            corpus,
            fts_result,
            case.where_document,
            min(len(fts_ids), case.n_results),
            where=case.where,
            check_ranking=case.check_fts_ranking,
        )
        if case.knn and case.knn.get("where"):
            for doc_id in ids:
                assert corpus_matches_where(corpus_by_id[doc_id], case.knn["where"]), (
                    f"id={doc_id!r} violates knn.where={case.knn['where']!r}"
                )
        return

    if case.verify == "search_index":
        assert case.where is not None

        def _si_row(rec: CorpusRecord) -> bool:
            """Si row."""
            if not corpus_matches_where(rec, case.where):
                return False
            return not (
                case.where_document is not None and not doc_matches_where_document(rec.document, case.where_document)
            )

        si_result = _slice_hybrid_result(result, corpus, _si_row)
        si_ids = si_result["ids"][0]
        assert len(si_ids) >= case.min_hits, f"case {case.name!r}: expected at least {case.min_hits} search-index rows"
        assert_hybrid_search_index_result(
            corpus,
            si_result,
            case.where,
            min(len(si_ids), case.n_results),
            min_hits=case.min_hits,
            exact_match_count=case.exact_match_count,
        )
        if case.knn and case.knn.get("where"):
            for doc_id in ids:
                assert corpus_matches_where(corpus_by_id[doc_id], case.knn["where"]), (
                    f"id={doc_id!r} violates knn.where={case.knn['where']!r}"
                )
        return

    if case.verify == "knn":
        assert case.knn is not None
        knn_where = case.knn.get("where")

        def _knn_row(rec: CorpusRecord) -> bool:
            """Knn row."""
            if knn_where is not None and not corpus_matches_where(rec, knn_where):
                return False
            if case.where is not None and not doc_matches_where_metadata(rec.metadata, case.where):
                return False
            return not (
                case.where_document is not None and not doc_matches_where_document(rec.document, case.where_document)
            )

        knn_result = _slice_hybrid_result(result, corpus, _knn_row)
        knn_ids = knn_result["ids"][0]
        assert len(knn_ids) >= 1, f"case {case.name!r}: expected at least one KNN-eligible row in hybrid result"
        assert_hybrid_knn_result(
            corpus,
            knn_result,
            case.knn["query_embeddings"],
            min(len(knn_ids), case.n_results),
            knn_where,
            distance_metric=distance_metric,
            check_top1=True,
            check_score_order=False,
        )
        return

    if case.verify == "intersection":
        total = count_triple_intersection_matches(corpus, case)
        if case.exact_match_count is not None:
            assert total == case.exact_match_count, (
                f"case {case.name!r}: expected {case.exact_match_count} intersection matches, corpus has {total}"
            )
        for doc_id in ids:
            assert corpus_matches_triple_intersection(corpus_by_id[doc_id], case), (
                f"id={doc_id!r} outside triple-branch intersection for case {case.name!r}"
            )
        if case.exact_match_count is not None and case.exact_match_count <= case.n_results:
            expected_ids = {rec.doc_id for rec in corpus if corpus_matches_triple_intersection(rec, case)}
            assert set(ids) == expected_ids
        return

    if case.verify == "rrf_fusion":
        if case.where is not None:
            for doc_id in ids:
                assert doc_matches_where_metadata(corpus_by_id[doc_id].metadata, case.where), (
                    f"id={doc_id!r} violates query.where={case.where!r}"
                )
        if case.knn and case.knn.get("where"):
            for doc_id in ids:
                assert corpus_matches_where(corpus_by_id[doc_id], case.knn["where"]), (
                    f"id={doc_id!r} violates knn.where={case.knn['where']!r}"
                )
        intersection_ids = {rec.doc_id for rec in corpus if corpus_matches_triple_intersection(rec, case)}
        assert intersection_ids, f"case {case.name!r}: intersection must be non-empty for RRF fusion checks"
        returned = set(ids)
        assert returned & intersection_ids, (
            f"case {case.name!r}: RRF result must include at least one intersection hit, got {ids[:5]!r}"
        )
        if case.where_document is not None:
            fts_hits = {rec.doc_id for rec in corpus if doc_matches_where_document(rec.document, case.where_document)}
            assert returned & fts_hits, f"case {case.name!r}: RRF result must include at least one FTS hit"
        return

    raise ValueError(f"unsupported verify mode: {case.verify!r}")


def _build_triple_branch_hybrid_search_kwargs(
    case: HybridTripleBranchCase,
) -> dict[str, Any]:
    """Build triple branch hybrid search kwargs."""
    query: dict[str, Any] | None = None
    if case.where_document is not None or case.where is not None:
        query = {"n_results": case.n_results}
        if case.where_document is not None:
            query["where_document"] = case.where_document
        if case.where is not None:
            query["where"] = case.where
    return {
        "query": query,
        "knn": case.knn,
        "rank": RRF_RANK if case.use_rrf else None,
        "n_results": case.n_results,
        "include": ["documents", "metadatas", "distances"],
    }


def assert_flat_collection_baseline(collection: Any) -> None:
    """Runtime guard: baseline must be a flat collection (HNSW heap table path)."""
    if getattr(collection, "use_namespace", None) is not False:
        raise TypeError(
            f"collection baseline must have use_namespace=False, "
            f"got {getattr(collection, 'use_namespace', None)!r} on {collection!r}"
        )
    if not hasattr(collection, "hybrid_search"):
        raise TypeError(f"collection baseline must expose hybrid_search(), got {type(collection)!r}")


def execute_hybrid_triple_branch_on_collection(
    collection: Any,
    case: HybridTripleBranchCase,
) -> dict[str, Any]:
    """
    Triple-branch hybrid_search on flat collection.

    Path: ``collection.add`` (during setup) → ``collection.hybrid_search``
    → ``BaseClient._collection_hybrid_search`` → ``DBMS_HYBRID_SEARCH.GET_SQL``.
    Vector index: HNSW (``flat_hybrid_search_schema``).
    """
    assert_flat_collection_baseline(collection)
    return collection.hybrid_search(**_build_triple_branch_hybrid_search_kwargs(case))


def execute_hybrid_triple_branch_on_namespace(
    namespace: Any,
    case: HybridTripleBranchCase,
) -> dict[str, Any]:
    """
    Triple-branch hybrid_search on namespace.

    Path: ``namespace.add`` (during setup) → ``namespace.hybrid_search``
    → ``BaseClient._namespace_hybrid_search`` → ``hybrid_search(TABLE logic_data_table, ...)``.
    Vector index: IVF spfresh (``ns_schema``).
    """
    if not hasattr(namespace, "namespace_id"):
        raise TypeError(f"expected Namespace, got {type(namespace)!r}")
    return namespace.hybrid_search(**_build_triple_branch_hybrid_search_kwargs(case))


def execute_hybrid_triple_branch_search(
    target: Any,
    case: HybridTripleBranchCase,
) -> dict[str, Any]:
    """Dispatch to collection or namespace executor based on target type."""
    if getattr(target, "use_namespace", None) is False:
        return execute_hybrid_triple_branch_on_collection(target, case)
    if hasattr(target, "namespace_id"):
        return execute_hybrid_triple_branch_on_namespace(target, case)
    raise TypeError(
        f"unsupported hybrid_search target {target!r}: expected flat Collection (use_namespace=False) or Namespace"
    )


def setup_large_fts_flat_collection(
    db_client: Any,
    *,
    distance: VectorDistanceMetric = "l2",
) -> tuple[list[CorpusRecord], Any]:
    """Create flat collection (HNSW + FTS), load via ``collection.add``."""
    corpus = build_large_fts_corpus(CORPUS_SIZE)
    if len(corpus) <= 1000:
        raise ValueError(f"corpus must exceed 1000 rows, got {len(corpus)}")

    name = f"test_hs_tb_baseline_{distance}_{int(time.time() * 1000)}"
    collection = db_client.create_collection(name=name, schema=flat_hybrid_search_schema(distance), use_namespace=False)
    assert collection.use_namespace is False
    insert_corpus_into_collection(collection, corpus)
    expected = len(corpus)
    actual = collection.count()
    assert actual == expected, f"baseline collection {collection.name!r} expected {expected} rows, got {actual}"
    time.sleep(INDEX_SETTLE_SECONDS)
    return corpus, collection


def _try_assert_hybrid_triple_branch_result(
    corpus: list[CorpusRecord],
    result: dict[str, Any],
    case: HybridTripleBranchCase,
    *,
    distance_metric: VectorDistanceMetric = "l2",
) -> BaseException | None:
    """Try assert hybrid triple branch result."""
    try:
        assert_hybrid_triple_branch_result(corpus, result, case, distance_metric=distance_metric)
        return None
    except BaseException as exc:
        return exc


def assert_hybrid_triple_branch_vs_collection_baseline(
    corpus: list[CorpusRecord],
    case: HybridTripleBranchCase,
    namespace_result: dict[str, Any],
    collection_result: dict[str, Any],
    *,
    distance_metric: VectorDistanceMetric = "l2",
) -> None:
    """
    Compare namespace (IVF logic table) vs collection baseline (HNSW heap table).

    Collection baseline uses ``use_namespace=False``: data via ``collection.add``,
    search via ``collection.hybrid_search`` — the standard collection hybrid path.

    Because HNSW vs IVF may rank differently on KNN/RRF branches, **id list equality
    is not required** when both sides satisfy corpus ground truth. Comparison focuses
    on pass/fail attribution:

      - ``[test-case]`` — collection baseline also fails; likely expectation / case design issue
      - ``[observer/namespace]`` — collection passes ground truth but namespace fails
      - ``[test-case/baseline]`` — namespace passes but collection baseline fails
    """
    coll_err = _try_assert_hybrid_triple_branch_result(corpus, collection_result, case, distance_metric=distance_metric)
    ns_err = _try_assert_hybrid_triple_branch_result(corpus, namespace_result, case, distance_metric=distance_metric)
    coll_ids = collection_result["ids"][0]
    ns_ids = namespace_result["ids"][0]

    if coll_err is None and ns_err is None:
        # Both paths satisfy ground truth; HNSW vs IVF ranking may differ — success.
        return

    if coll_err is not None and ns_err is not None:
        if coll_ids == ns_ids:
            raise AssertionError(
                f"[test-case] case {case.name!r}: collection (use_namespace=false, HNSW) and "
                f"namespace (IVF) fail the same ground-truth checks with identical ids; "
                f"likely test expectation issue.\n"
                f"  ids={coll_ids[:10]!r}\n"
                f"  error={coll_err!r}"
            ) from coll_err
        raise AssertionError(
            f"[mixed] case {case.name!r}: collection and namespace both fail ground truth "
            f"and ids differ.\n"
            f"  collection_ids={coll_ids[:10]!r}\n"
            f"  namespace_ids={ns_ids[:10]!r}\n"
            f"  collection_error={coll_err!r}\n"
            f"  namespace_error={ns_err!r}"
        ) from ns_err

    if coll_err is None and ns_err is not None:
        raise AssertionError(
            f"[observer/namespace] case {case.name!r}: collection baseline (HNSW, "
            f"use_namespace=false) passes but namespace (IVF) fails.\n"
            f"  collection_ids={coll_ids!r}\n"
            f"  namespace_ids={ns_ids!r}\n"
            f"  namespace_error={ns_err!r}"
        ) from ns_err

    raise AssertionError(
        f"[test-case/baseline] case {case.name!r}: namespace (IVF) passes but collection "
        f"baseline (HNSW, use_namespace=false) fails; revisit case assumptions.\n"
        f"  collection_ids={coll_ids!r}\n"
        f"  namespace_ids={ns_ids!r}\n"
        f"  collection_error={coll_err!r}"
    ) from coll_err


def run_hybrid_triple_branch_case(
    namespace: Any,
    corpus: list[CorpusRecord],
    case: HybridTripleBranchCase,
    *,
    collection_baseline: Any | None = None,
    distance_metric: VectorDistanceMetric = "l2",
) -> dict[str, Any]:
    """Run hybrid triple branch case."""
    if collection_baseline is not None:
        assert_flat_collection_baseline(collection_baseline)
        try:
            collection_result = execute_hybrid_triple_branch_on_collection(collection_baseline, case)
        except BaseException as coll_exec_err:
            raise AssertionError(
                f"[test-case/baseline] case {case.name!r}: collection.hybrid_search "
                f"(use_namespace=false, HNSW/GET_SQL) raised: {coll_exec_err!r}"
            ) from coll_exec_err
        try:
            namespace_result = execute_hybrid_triple_branch_on_namespace(namespace, case)
        except BaseException as ns_exec_err:
            raise AssertionError(
                f"[observer/namespace] case {case.name!r}: collection.hybrid_search "
                f"succeeded but namespace.hybrid_search (IVF/logic table) raised: "
                f"{ns_exec_err!r}"
            ) from ns_exec_err
        assert_hybrid_triple_branch_vs_collection_baseline(
            corpus,
            case,
            namespace_result,
            collection_result,
            distance_metric=distance_metric,
        )
        return namespace_result

    result = execute_hybrid_triple_branch_on_namespace(namespace, case)
    assert_hybrid_triple_branch_result(corpus, result, case, distance_metric=distance_metric)
    return result


TRIPLE_BRANCH_CASES: list[HybridTripleBranchCase] = [
    # --- Full-text operators (FTS is primary; search index + KNN use broad filters) ---
    HybridTripleBranchCase(
        name="fts_contains_zpx",
        verify="fts",
        where_document={"$contains": TOKEN_ZPX},
        where=WHERE_BROAD_METADATA,
        knn=_default_knn(15),
        n_results=15,
    ),
    HybridTripleBranchCase(
        name="fts_contains_alp",
        verify="fts",
        where_document={"$contains": TOKEN_ALP},
        where=WHERE_BROAD_METADATA,
        knn=_default_knn(15),
        n_results=15,
    ),
    HybridTripleBranchCase(
        name="fts_string_shorthand",
        verify="fts",
        where_document=TOKEN_ZPX,
        where=WHERE_BROAD_METADATA,
        knn=_default_knn(15),
        n_results=15,
    ),
    HybridTripleBranchCase(
        name="fts_not_contains",
        verify="not_contains",
        where_document={"$not_contains": TOKEN_ZPX},
        where=WHERE_FILLER_SEQ,
        knn=_default_knn(20, where=WHERE_FILLER_SEQ),
        n_results=20,
    ),
    HybridTripleBranchCase(
        name="fts_and_zpx_alp",
        verify="fts",
        where_document={
            "$and": [
                {"$contains": TOKEN_ZPX},
                {"$contains": TOKEN_ALP},
            ],
        },
        where=WHERE_BROAD_METADATA,
        knn=_default_knn(10),
        n_results=10,
    ),
    HybridTripleBranchCase(
        name="fts_or_zpx_alp",
        verify="fts",
        where_document={
            "$or": [
                {"$contains": TOKEN_ZPX},
                {"$contains": TOKEN_ALP},
            ],
        },
        where=WHERE_BROAD_METADATA,
        knn=_default_knn(20),
        n_results=20,
    ),
    HybridTripleBranchCase(
        name="fts_and_multi_contains",
        verify="fts",
        where_document={
            "$and": [
                {"$contains": "Primary"},
                {"$contains": TOKEN_ZPX},
            ],
        },
        where=WHERE_BROAD_METADATA,
        knn=_default_knn(10),
        n_results=10,
    ),
    HybridTripleBranchCase(
        name="fts_contains_filter_has_both",
        verify="fts",
        where_document={"$contains": TOKEN_ZPX},
        where=WHERE_BOTH_TOKENS_NUMERIC,
        knn=_default_knn(5, where=WHERE_BOTH_TOKENS_NUMERIC),
        n_results=5,
    ),
    HybridTripleBranchCase(
        name="fts_contains_filter_zpx_hint_eq",
        verify="fts",
        where_document={"$contains": TOKEN_ZPX},
        where={"zpx_hint": 50},
        knn=_default_knn(5, where={"zpx_hint": 50}),
        n_results=5,
    ),
    HybridTripleBranchCase(
        name="fts_contains_filter_zpx_hint_gte",
        verify="fts",
        where_document={"$contains": TOKEN_ZPX},
        where={"zpx_hint": {"$gte": 40}},
        knn=_default_knn(10, where={"zpx_hint": {"$gte": 40}}),
        n_results=10,
    ),
    HybridTripleBranchCase(
        name="fts_and_zpx_alp_filter_has_both",
        verify="fts",
        where_document={
            "$and": [
                {"$contains": TOKEN_ZPX},
                {"$contains": TOKEN_ALP},
            ],
        },
        where=WHERE_BOTH_TOKENS_NUMERIC,
        knn=_default_knn(5, where=WHERE_BOTH_TOKENS_NUMERIC),
        n_results=5,
    ),
    HybridTripleBranchCase(
        name="fts_contains_filter_zpx_hint_and_alp_zero",
        verify="fts",
        where_document={"$contains": TOKEN_ZPX},
        where={
            "$and": [
                {"zpx_hint": {"$gte": 40}},
                {"alp_hint": 0},
            ],
        },
        knn=_default_knn(
            10,
            where={
                "$and": [
                    {"zpx_hint": {"$gte": 40}},
                    {"alp_hint": 0},
                ],
            },
        ),
        n_results=10,
    ),
    # --- Search-index operators (metadata is primary; universal FTS + broad KNN) ---
    HybridTripleBranchCase(
        name="si_eq_zpx_hint_direct",
        verify="search_index",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where={"zpx_hint": 50},
        knn=_default_knn(5, where={"zpx_hint": 50}),
        n_results=5,
        exact_match_count=1,
    ),
    HybridTripleBranchCase(
        name="si_eq_zpx_hint_operator",
        verify="search_index",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where={"zpx_hint": {"$eq": 50}},
        knn=_default_knn(5, where={"zpx_hint": {"$eq": 50}}),
        n_results=5,
        exact_match_count=1,
    ),
    HybridTripleBranchCase(
        name="si_ne_zpx_hint_zero",
        verify="search_index",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where={"zpx_hint": {"$ne": 0}},
        knn=_default_knn(20, where={"zpx_hint": {"$ne": 0}}),
        n_results=20,
        min_hits=1,
    ),
    HybridTripleBranchCase(
        name="si_gte_zpx_hint_40",
        verify="search_index",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where={"zpx_hint": {"$gte": 40}},
        knn=_default_knn(15, where={"zpx_hint": {"$gte": 40}}),
        n_results=15,
        min_hits=2,
    ),
    HybridTripleBranchCase(
        name="si_lt_zpx_hint_10",
        verify="search_index",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where={"zpx_hint": {"$lt": 10}},
        knn=_default_knn(20, where={"zpx_hint": {"$lt": 10}}),
        n_results=20,
        min_hits=1,
    ),
    HybridTripleBranchCase(
        name="si_lte_alp_hint_8",
        verify="search_index",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where={"alp_hint": {"$lte": 8}},
        knn=_default_knn(15, where={"alp_hint": {"$lte": 8}}),
        n_results=15,
        min_hits=1,
    ),
    HybridTripleBranchCase(
        name="si_gt_rel_hint_zero",
        verify="search_index",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where={"rel_hint": {"$gt": 0}},
        knn=_default_knn(25, where={"rel_hint": {"$gt": 0}}),
        n_results=25,
        min_hits=5,
    ),
    HybridTripleBranchCase(
        name="si_in_alp_hint_values",
        verify="search_index",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where={"alp_hint": {"$in": [32, 24, 16, 8]}},
        knn=_default_knn(10, where={"alp_hint": {"$in": [32, 24, 16, 8]}}),
        n_results=10,
        min_hits=4,
    ),
    HybridTripleBranchCase(
        name="si_nin_alp_hint_high",
        verify="search_index",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where={"alp_hint": {"$nin": [32, 24, 16, 8]}},
        knn=_default_knn(15, where={"alp_hint": {"$nin": [32, 24, 16, 8]}}),
        n_results=15,
        min_hits=1,
    ),
    HybridTripleBranchCase(
        name="si_and_has_both_zpx_gte",
        verify="search_index",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where={
            "$and": [
                {"zpx_hint": {"$gte": 5}},
                {"alp_hint": 5},
            ],
        },
        knn=_default_knn(
            5,
            where={
                "$and": [
                    {"zpx_hint": {"$gte": 5}},
                    {"alp_hint": 5},
                ],
            },
        ),
        n_results=5,
        exact_match_count=1,
    ),
    HybridTripleBranchCase(
        name="si_or_zpx_alp_hint_high",
        verify="search_index",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where={
            "$or": [
                {"zpx_hint": {"$gte": 50}},
                {"alp_hint": {"$gte": 32}},
            ],
        },
        knn=_default_knn(
            10,
            where={
                "$or": [
                    {"zpx_hint": {"$gte": 50}},
                    {"alp_hint": {"$gte": 32}},
                ],
            },
        ),
        n_results=10,
        min_hits=2,
    ),
    HybridTripleBranchCase(
        name="si_not_has_both",
        verify="search_index",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where=WHERE_NOT_BOTH_TOKENS_NUMERIC,
        knn=_default_knn(20, where=WHERE_NOT_BOTH_TOKENS_NUMERIC),
        n_results=20,
        min_hits=1,
    ),
    HybridTripleBranchCase(
        name="si_id_eq_both_tokens",
        verify="search_index",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where={"#id": "both_tokens"},
        knn=_default_knn(5, where={"#id": "both_tokens"}),
        n_results=5,
        exact_match_count=1,
    ),
    HybridTripleBranchCase(
        name="si_id_in_zpx_tops",
        verify="search_index",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where={"#id": {"$in": ["zpx_top_5", "zpx_top_4", "zpx_top_3"]}},
        knn=_default_knn(
            5,
            where={"#id": {"$in": ["zpx_top_5", "zpx_top_4", "zpx_top_3"]}},
        ),
        n_results=5,
        exact_match_count=3,
    ),
    # --- Vector / KNN operators (KNN primary; universal FTS + broad metadata) ---
    HybridTripleBranchCase(
        name="knn_global_top5",
        verify="knn",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where=WHERE_BROAD_METADATA,
        knn=_default_knn(5, knn_n_results=10),
        n_results=5,
    ),
    HybridTripleBranchCase(
        name="knn_filter_has_both",
        verify="knn",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where=WHERE_BOTH_TOKENS_NUMERIC,
        knn=_default_knn(3, where=WHERE_BOTH_TOKENS_NUMERIC, knn_n_results=10),
        n_results=3,
    ),
    HybridTripleBranchCase(
        name="knn_filter_zpx_hint_gte_40",
        verify="knn",
        where_document=WHERE_DOCUMENT_UNIVERSAL,
        where={"zpx_hint": {"$gte": 40}},
        knn=_default_knn(5, where={"zpx_hint": {"$gte": 40}}, knn_n_results=15),
        n_results=5,
    ),
    # --- Aligned intersection (all three branches share the same filter intent) ---
    HybridTripleBranchCase(
        name="intersection_zpx_gte40",
        verify="intersection",
        where_document={"$contains": TOKEN_ZPX},
        where={"zpx_hint": {"$gte": 40}},
        knn=_default_knn(10, where={"zpx_hint": {"$gte": 40}}),
        n_results=10,
        min_hits=2,
    ),
    HybridTripleBranchCase(
        name="intersection_both_tokens",
        verify="intersection",
        where_document={
            "$and": [
                {"$contains": TOKEN_ZPX},
                {"$contains": TOKEN_ALP},
            ],
        },
        where=WHERE_BOTH_TOKENS_NUMERIC,
        knn=_default_knn(5, where=WHERE_BOTH_TOKENS_NUMERIC),
        n_results=5,
        exact_match_count=1,
    ),
    HybridTripleBranchCase(
        name="intersection_id_zpx_top5",
        verify="intersection",
        where_document={"$contains": TOKEN_ZPX},
        where={"#id": "zpx_top_5"},
        knn=_default_knn(5, where={"#id": "zpx_top_5"}),
        n_results=5,
        exact_match_count=1,
    ),
    # --- RRF fusion across three branches ---
    HybridTripleBranchCase(
        name="rrf_fts_or_knn_si_gte",
        verify="rrf_fusion",
        where_document={
            "$or": [
                {"$contains": TOKEN_ZPX},
                {"$contains": TOKEN_ALP},
            ],
        },
        where={"rel_hint": {"$gte": 4}},
        knn=_default_knn(10, where={"rel_hint": {"$gte": 4}}, knn_n_results=15),
        n_results=10,
        use_rrf=True,
        min_hits=3,
    ),
    HybridTripleBranchCase(
        name="rrf_all_branches_zpx_gte40",
        verify="rrf_fusion",
        where_document={"$contains": TOKEN_ZPX},
        where={"zpx_hint": {"$gte": 40}},
        knn=_default_knn(8, where={"zpx_hint": {"$gte": 40}}, knn_n_results=15),
        n_results=8,
        use_rrf=True,
        min_hits=2,
    ),
]


def get_triple_branch_case(name: str) -> HybridTripleBranchCase:
    """Get triple branch case."""
    for case in TRIPLE_BRANCH_CASES:
        if case.name == name:
            return case
    raise KeyError(f"unknown triple-branch case: {name!r}")


def run_triple_branch_case_on_quadrants(
    ctx: dict[str, Any],
    case: HybridTripleBranchCase | str,
    quadrant_keys: tuple[str, ...] = MULTI_COLL_MULTI_NS_FTS_LOADED_QUADRANTS,
    *,
    collection_baseline: Any | None = None,
    distance_metric: VectorDistanceMetric = "l2",
) -> None:
    """Run triple branch case on quadrants."""
    tb_case = case if isinstance(case, HybridTripleBranchCase) else get_triple_branch_case(case)
    for key in quadrant_keys:
        corpus, namespace = ctx[key]
        run_hybrid_triple_branch_case(
            namespace,
            corpus,
            tb_case,
            collection_baseline=collection_baseline,
            distance_metric=distance_metric,
        )


def run_search_index_case_on_quadrants(
    ctx: dict[str, Any],
    case: SearchIndexQueryCase | str,
    quadrant_keys: tuple[str, ...] = MULTI_COLL_MULTI_NS_FTS_LOADED_QUADRANTS,
) -> None:
    """Run search index case on quadrants."""
    si_case = case if isinstance(case, SearchIndexQueryCase) else get_search_index_case(case)
    for key in quadrant_keys:
        corpus, namespace = ctx[key]
        run_hybrid_search_index_case(namespace, corpus, si_case)


def run_knn_case_on_quadrants(
    ctx: dict[str, Any],
    case: VectorKnnCase | str,
    quadrant_keys: tuple[str, ...] = MULTI_COLL_MULTI_NS_FTS_LOADED_QUADRANTS,
    *,
    distance_metric: VectorDistanceMetric = "l2",
) -> None:
    """Run knn case on quadrants."""
    knn_case = case if isinstance(case, VectorKnnCase) else get_vector_knn_case(case)
    for key in quadrant_keys:
        corpus, namespace = ctx[key]
        run_hybrid_knn_case(namespace, corpus, knn_case, distance_metric=distance_metric)


def assert_hybrid_search_index_no_hits(
    namespace: Any,
    where: dict[str, Any],
    *,
    n_results: int = 10,
) -> None:
    """Assert hybrid search index no hits."""
    result = namespace.hybrid_search(
        query={"where": where, "n_results": n_results},
        n_results=n_results,
        include=["metadatas"],
    )
    ids = result.get("ids", [[]])[0] if result.get("ids") else []
    assert len(ids) == 0, f"expected no search-index hits in namespace {namespace.name!r}, got {ids[:5]!r}"


def assert_hybrid_triple_branch_no_hits(
    namespace: Any,
    case: HybridTripleBranchCase,
) -> None:
    """Assert triple-branch hybrid_search returns no rows (namespace isolation)."""
    query: dict[str, Any] | None = None
    if case.where_document is not None or case.where is not None:
        query = {"n_results": case.n_results}
        if case.where_document is not None:
            query["where_document"] = case.where_document
        if case.where is not None:
            query["where"] = case.where
    result = namespace.hybrid_search(
        query=query,
        knn=case.knn,
        rank=RRF_RANK if case.use_rrf else None,
        n_results=case.n_results,
        include=["documents"],
    )
    ids = result.get("ids", [[]])[0] if result.get("ids") else []
    assert len(ids) == 0, f"expected no triple-branch hits in namespace {namespace.name!r}, got {ids[:5]!r}"


__all__ = [
    "BATCH_SIZE",
    "CORPUS_SIZE",
    "HYBRID_COMBINED_CASES",
    "INDEX_SETTLE_SECONDS",
    "KNN_QUERY_VECTOR",
    "MULTI_COLL_MULTI_NS_FTS_LOADED_QUADRANTS",
    "MULTI_COLL_MULTI_NS_QUADRANT_KEYS",
    "RRF_RANK",
    "SEARCH_INDEX_CASES",
    "TOKEN_ALP",
    "TOKEN_ZPX",
    "TRIPLE_BRANCH_CASES",
    "VECTOR_DISTANCE_METRICS",
    "VECTOR_KNN_CASES",
    "WHERE_BROAD_METADATA",
    "WHERE_DOCUMENT_UNIVERSAL",
    "WHERE_FILLER_SEQ",
    "HybridCombinedCase",
    "HybridTripleBranchCase",
    "SearchIndexQueryCase",
    "VectorDistanceMetric",
    "VectorKnnCase",
    "assert_flat_collection_baseline",
    "assert_hybrid_search_index_no_hits",
    "assert_hybrid_triple_branch_no_hits",
    "assert_hybrid_triple_branch_result",
    "assert_hybrid_triple_branch_vs_collection_baseline",
    "assert_not_contains_no_token_leak",
    "build_large_fts_corpus",
    "ensure_shared_hybrid_search_collection",
    "execute_hybrid_triple_branch_on_collection",
    "execute_hybrid_triple_branch_on_namespace",
    "execute_hybrid_triple_branch_search",
    "get_fts_case",
    "get_hybrid_combined_case",
    "get_search_index_case",
    "get_triple_branch_case",
    "get_vector_knn_case",
    "knn_compare_distance",
    "run_hybrid_combined_case",
    "run_hybrid_knn_case",
    "run_hybrid_search_fts_case",
    "run_hybrid_search_index_case",
    "run_hybrid_triple_branch_case",
    "run_knn_case_on_quadrants",
    "run_search_index_case_on_quadrants",
    "run_triple_branch_case_on_quadrants",
    "setup_fts_namespace_with_corpus",
    "setup_large_fts_collection",
    "setup_large_fts_flat_collection",
    "setup_multi_coll_multi_ns_fts",
    "setup_multi_coll_multi_ns_fts_single_loaded",
    "teardown_large_fts_collection",
    "teardown_multi_coll_multi_ns_fts",
]
