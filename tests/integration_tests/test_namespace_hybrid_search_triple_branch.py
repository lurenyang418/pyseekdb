"""
Namespace hybrid_search triple-branch integration tests.

Each case activates all three signals in one ``hybrid_search`` call:
  - full-text via ``query.where_document`` ($contains / $not_contains / $and / $or)
  - search index via ``query.where`` ($eq / $ne / $lt / $lte / $gt / $gte / $in / $nin / $and / $or / $not / #id)
  - vector via ``knn`` (KNN distance ordering + optional ``knn.where``)

Operator coverage is orthogonal: one branch is verified against corpus ground truth while
the other two stay active with broad or aligned filters. For ``verify="fts"``, hits and
filters are checked; fused FTS+KNN ``__score`` order is not asserted (see
``check_fts_ranking`` on ``HybridTripleBranchCase``).

Vector index distance: L2 and cosine (IVF spfresh).

Run one case::

    pytest tests/integration_tests/test_namespace_hybrid_search_triple_branch.py \\
        -k "fts_contains_zpx and oceanbase and l2" -v -s
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest
from namespace_hybrid_search_helpers import (
    TOKEN_ZPX,
    TRIPLE_BRANCH_CASES,
    VectorDistanceMetric,
    ensure_shared_hybrid_search_collection,
    get_triple_branch_case,
    run_hybrid_triple_branch_case,
    setup_fts_namespace_with_corpus,
    teardown_large_fts_collection,
)


@pytest.mark.parametrize("vector_distance", ["l2", "cosine"])
class TestNamespaceHybridSearchTripleBranch:
    """Large-scale namespace vector + full-text + search-index hybrid_search tests."""

    _shared_by_mode: ClassVar[dict[str, dict[str, Any]]] = {}

    @pytest.fixture(autouse=True)
    def _bind_shared_collection(
        self,
        db_client: Any,
        vector_distance: VectorDistanceMetric,
        request: pytest.FixtureRequest,
    ) -> None:
        """Bind shared collection."""
        entry = ensure_shared_hybrid_search_collection(self._shared_by_mode, db_client, request, vector_distance)
        self._corpus = entry["corpus"]
        self._collection = entry["collection"]
        self._vector_distance = vector_distance

    @classmethod
    def teardown_class(cls) -> None:
        """Teardown class."""
        for entry in cls._shared_by_mode.values():
            teardown_large_fts_collection(entry["db_client"], entry["collection"])
        cls._shared_by_mode.clear()

    def _new_namespace(self, case_name: str) -> Any:
        """New namespace."""
        return setup_fts_namespace_with_corpus(
            self._collection,
            self._corpus,
            namespace_name=f"ns_tb_{case_name}_{int(time.time() * 1000)}",
        )

    @pytest.mark.parametrize("case_name", [c.name for c in TRIPLE_BRANCH_CASES])
    def test_hybrid_search_triple_branch(self, db_client, case_name: str):
        """Test hybrid search triple branch."""
        case = get_triple_branch_case(case_name)
        namespace = self._new_namespace(case_name)
        run_hybrid_triple_branch_case(
            namespace,
            self._corpus,
            case,
            distance_metric=self._vector_distance,
        )

    def test_hybrid_search_triple_branch_fts_hits_with_knn_active(self, db_client):
        """FTS+KNN hybrid returns TOKEN_ZPX hits; fused order is not asserted."""
        namespace = self._new_namespace("fts_hits_knn_si")
        result = namespace.hybrid_search(
            query={
                "where_document": {"$contains": TOKEN_ZPX},
                "where": {"rel_hint": {"$gte": 0}},
                "n_results": 5,
            },
            knn={
                "query_embeddings": [1.0, 1.0, 0.0],
                "where": {"rel_hint": {"$gte": 0}},
                "n_results": 20,
            },
            n_results=5,
            include=["documents", "metadatas"],
        )
        ids = result["ids"][0]
        assert len(ids) > 0, "expected at least one hybrid FTS+KNN hit"
        docs = result.get("documents", [[]])[0]
        for doc_id, doc_text in zip(ids, docs, strict=True):
            assert TOKEN_ZPX.lower() in (doc_text or "").lower(), f"id={doc_id!r} must contain {TOKEN_ZPX!r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
