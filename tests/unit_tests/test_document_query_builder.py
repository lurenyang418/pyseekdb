"""Unit tests for document_query_builder."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from pyseekdb.client.document_query_builder import (  # noqa: E402
    build_document_hybrid_expression,
    document_expr_as_knn_filter,
)


class TestBuildDocumentHybridExpression:
    """TestBuildDocumentHybridExpression class."""

    def test_not_contains_builds_must_not(self):
        """Test not contains builds must not."""
        expr = build_document_hybrid_expression({"$not_contains": "machine"})
        assert expr == {
            "bool": {
                "must_not": [{"query_string": {"fields": ["document"], "query": "machine"}}],
            }
        }

    def test_and_multi_word_uses_bool_composition(self):
        """Multi-word $contains terms must not use the single query_string fast path."""
        expr = build_document_hybrid_expression({
            "$and": [{"$contains": "foo bar"}, {"$contains": "baz"}],
        })
        assert expr == {
            "bool": {
                "must": [
                    {"query_string": {"fields": ["document"], "query": "foo bar"}},
                    {"query_string": {"fields": ["document"], "query": "baz"}},
                ]
            }
        }

    def test_and_or_nested(self):
        """Test and or nested."""
        expr = build_document_hybrid_expression({
            "$and": [
                {"$or": [{"$contains": "a"}, {"$contains": "b"}]},
                {"$contains": "c"},
            ]
        })
        assert expr == {
            "bool": {
                "must": [
                    {
                        "query_string": {
                            "fields": ["document"],
                            "query": "a b",
                            "default_operator": "or",
                        }
                    },
                    {
                        "query_string": {
                            "fields": ["document"],
                            "query": "c",
                        }
                    },
                ]
            }
        }

    def test_query_string_escapes_reserved_characters(self):
        """Reserved Lucene characters are escaped in query_string terms."""
        expr = build_document_hybrid_expression({"$contains": "C++"})
        assert expr == {
            "query_string": {
                "fields": ["document"],
                "query": r"C\+\+",
            }
        }

    def test_regex_leaf(self):
        """Test regex leaf."""
        expr = build_document_hybrid_expression({"$regex": r"machine.*learning"})
        assert expr == {"regexp": {"document": {"value": r"machine.*learning"}}}


class TestDocumentExprAsKnnFilter:
    """TestDocumentExprAsKnnFilter class."""

    def test_pure_must_not_adds_exists_filter(self):
        """Test pure must not adds exists filter."""
        expr = build_document_hybrid_expression({"$not_contains": "x"})
        wrapped = document_expr_as_knn_filter(expr)
        assert wrapped is not None
        assert wrapped["bool"]["must_not"] == [
            {"query_string": {"fields": ["document"], "query": "x"}},
        ]
        assert wrapped["bool"]["filter"] == [{"exists": {"field": "document"}}]

    def test_contains_wrapped_in_bool_must(self):
        """Test contains wrapped in bool must."""
        expr = build_document_hybrid_expression({"$contains": "hello"})
        wrapped = document_expr_as_knn_filter(expr)
        assert wrapped == {"bool": {"must": [expr]}}
