"""
Namespace hybrid_search full-text integration tests (large corpus, >1000 rows).

One collection is created per test class (per ``db_client`` param); each test method
only creates a fresh namespace and loads the shared corpus.

    pytest tests/integration_tests/test_namespace_hybrid_search_fulltext.py \\
        -k "filter_has_both and oceanbase" -v -s
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest
from namespace_fts_helpers import (
    TOKEN_ZPX,
    CorpusRecord,
    assert_not_contains_no_token_leak,
    get_fts_case,
    run_hybrid_search_fts_case,
    setup_fts_namespace_with_corpus,
    setup_large_fts_collection,
    teardown_large_fts_collection,
)


class TestNamespaceHybridSearchFulltext:
    """Large-scale namespace pure full-text hybrid_search tests."""

    # Keyed by db_client param (embedded / server / oceanbase).
    _shared_by_mode: ClassVar[dict[str, dict[str, Any]]] = {}

    @pytest.fixture(autouse=True)
    def _bind_shared_collection(self, db_client: Any, request: pytest.FixtureRequest) -> None:
        """Bind shared collection."""
        mode = request.node.callspec.params["db_client"] if request.node.callspec else "default"
        if mode not in self._shared_by_mode:
            corpus, collection = setup_large_fts_collection(db_client)
            self._shared_by_mode[mode] = {
                "db_client": db_client,
                "corpus": corpus,
                "collection": collection,
            }
        entry = self._shared_by_mode[mode]
        self._corpus: list[CorpusRecord] = entry["corpus"]
        self._collection = entry["collection"]
        self._db_client = db_client

    @classmethod
    def teardown_class(cls) -> None:
        """Teardown class."""
        for entry in cls._shared_by_mode.values():
            teardown_large_fts_collection(entry["db_client"], entry["collection"])
        cls._shared_by_mode.clear()

    def _new_namespace(self, case_name: str) -> Any:
        """New namespace."""
        ns_name = f"ns_{case_name}_{int(time.time() * 1000)}"
        return setup_fts_namespace_with_corpus(
            self._collection,
            self._corpus,
            namespace_name=ns_name,
        )

    def _run_fts_case(self, case_name: str) -> None:
        """Run fts case."""
        namespace = self._new_namespace(case_name)
        run_hybrid_search_fts_case(namespace, self._corpus, get_fts_case(case_name))

    def test_hybrid_search_fulltext_contains_token_zpx(self, db_client):
        """``where_document``: ``{\"$contains\": TOKEN_ZPX}``"""
        self._run_fts_case("contains_token_zpx")

    def test_hybrid_search_fulltext_contains_token_alp(self, db_client):
        """``where_document``: ``{\"$contains\": TOKEN_ALP}``"""
        self._run_fts_case("contains_token_alp")

    def test_hybrid_search_fulltext_contains_string_shorthand(self, db_client):
        """``where_document``: string shorthand (same as ``$contains``)"""
        self._run_fts_case("string_shorthand_token_zpx")

    def test_hybrid_search_fulltext_not_contains(self, db_client):
        """``where_document``: ``{\"$not_contains\": TOKEN_ZPX}``"""
        case = get_fts_case("not_contains_token_zpx")
        namespace = self._new_namespace("not_contains_token_zpx")
        result = run_hybrid_search_fts_case(namespace, self._corpus, case)
        assert_not_contains_no_token_leak(self._corpus, result, TOKEN_ZPX)

    def test_hybrid_search_fulltext_and_zpx_alp(self, db_client):
        """``where_document``: ``{\"$and\": [{\"$contains\": ...}, ...]}``"""
        self._run_fts_case("and_zpx_alp")

    def test_hybrid_search_fulltext_and_multi_contains(self, db_client):
        """``where_document``: ``{\"$and\": [{\"$contains\": \"Primary\"}, ...]}``"""
        self._run_fts_case("and_multi_contains_phrase")

    def test_hybrid_search_fulltext_or_zpx_alp(self, db_client):
        """``where_document``: ``{\"$or\": [{\"$contains\": ...}, ...]}``"""
        self._run_fts_case("or_zpx_alp")

    def test_hybrid_search_fulltext_top1_contains_zpx(self, db_client):
        """Top-1 relevance check for ``$contains`` on TOKEN_ZPX."""
        namespace = self._new_namespace("top1_contains_zpx")
        top_result = namespace.hybrid_search(
            query={"where_document": {"$contains": TOKEN_ZPX}, "n_results": 1},
            n_results=1,
            include=["documents"],
        )
        assert top_result["ids"][0][0] == "zpx_top_5", (
            f"most relevant TOKEN_ZPX document must rank first, got {top_result['ids'][0]}"
        )

    def test_hybrid_search_fulltext_contains_zpx_filter_has_both(self, db_client):
        """``where_document`` + ``where``: ``has_both`` metadata equality."""
        self._run_fts_case("contains_zpx_filter_has_both")

    def test_hybrid_search_fulltext_contains_zpx_filter_zpx_hint_eq(self, db_client):
        """``where_document`` + ``where``: ``zpx_hint`` scalar equality."""
        self._run_fts_case("contains_zpx_filter_zpx_hint_50")

    def test_hybrid_search_fulltext_contains_zpx_filter_zpx_hint_gte(self, db_client):
        """``where_document`` + ``where``: ``zpx_hint`` ``$gte`` range."""
        self._run_fts_case("contains_zpx_filter_zpx_hint_gte_40")

    def test_hybrid_search_fulltext_and_zpx_alp_filter_has_both(self, db_client):
        """``$and`` on ``where_document`` combined with ``has_both`` metadata filter."""
        self._run_fts_case("and_zpx_alp_filter_has_both")

    def test_hybrid_search_fulltext_contains_zpx_filter_zpx_hint_and_alp_zero(self, db_client):
        """``where_document`` + ``where`` ``$and``: ``zpx_hint`` range and ``alp_hint`` equality."""
        self._run_fts_case("contains_zpx_filter_zpx_hint_and_alp_zero")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
