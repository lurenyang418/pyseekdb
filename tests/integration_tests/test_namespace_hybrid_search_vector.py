"""
Namespace hybrid_search pure vector (KNN) integration tests.

Validates KNN distance ordering and metadata filters on the ``knn`` branch.
Covers both L2 and cosine IVF index distance metrics.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest
from namespace_hybrid_search_helpers import (
    KNN_QUERY_VECTOR,
    VECTOR_KNN_CASES,
    VectorDistanceMetric,
    ensure_shared_hybrid_search_collection,
    expected_knn_ids,
    get_vector_knn_case,
    run_hybrid_knn_case,
    setup_fts_namespace_with_corpus,
    teardown_large_fts_collection,
)


@pytest.mark.parametrize("vector_distance", ["l2", "cosine"])
class TestNamespaceHybridSearchVector:
    """Large-scale namespace pure KNN hybrid_search tests."""

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
        ns_name = f"ns_knn_{case_name}_{int(time.time() * 1000)}"
        return setup_fts_namespace_with_corpus(
            self._collection,
            self._corpus,
            namespace_name=ns_name,
        )

    def _run_knn_case(self, case_name: str) -> None:
        """Run knn case."""
        namespace = self._new_namespace(case_name)
        run_hybrid_knn_case(
            namespace,
            self._corpus,
            get_vector_knn_case(case_name),
            distance_metric=self._vector_distance,
        )

    @pytest.mark.parametrize("case_name", [c.name for c in VECTOR_KNN_CASES])
    def test_hybrid_search_vector_knn_cases(self, db_client, case_name: str):
        """Test hybrid search vector knn cases."""
        self._run_knn_case(case_name)

    def test_hybrid_search_vector_top1_nearest(self, db_client):
        """Top-1 must be the global nearest neighbor for a fixed query vector."""
        namespace = self._new_namespace("top1_nearest")
        result = namespace.hybrid_search(
            knn={"query_embeddings": KNN_QUERY_VECTOR, "n_results": 1},
            n_results=1,
            include=["distances"],
        )
        assert (
            result["ids"][0][0]
            == expected_knn_ids(
                self._corpus,
                KNN_QUERY_VECTOR,
                1,
                distance_metric=self._vector_distance,
            )[0]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
