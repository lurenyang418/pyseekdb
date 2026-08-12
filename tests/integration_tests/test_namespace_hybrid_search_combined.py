"""
Namespace hybrid_search combined-branch tests.

Covers:
  - full-text + search index (metadata on ``query``)
  - vector + search index (``knn.where``)
  - vector + full-text (+ RRF smoke)
  - vector + full-text + search index (+ RRF smoke)

Vector branches run under both L2 and cosine IVF index metrics.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest
from namespace_hybrid_search_helpers import (
    HYBRID_COMBINED_CASES,
    VectorDistanceMetric,
    ensure_shared_hybrid_search_collection,
    get_hybrid_combined_case,
    run_hybrid_combined_case,
    setup_fts_namespace_with_corpus,
    teardown_large_fts_collection,
)


@pytest.mark.parametrize("vector_distance", ["l2", "cosine"])
class TestNamespaceHybridSearchCombined:
    """TestNamespaceHybridSearchCombined class."""

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
            namespace_name=f"ns_comb_{case_name}_{int(time.time() * 1000)}",
        )

    @pytest.mark.parametrize("case_name", [c.name for c in HYBRID_COMBINED_CASES])
    def test_hybrid_search_combined(self, db_client, case_name: str):
        """Test hybrid search combined."""
        case = get_hybrid_combined_case(case_name)
        namespace = self._new_namespace(case_name)
        run_hybrid_combined_case(namespace, self._corpus, case, distance_metric=self._vector_distance)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
