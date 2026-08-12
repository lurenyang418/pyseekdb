"""Unit tests for BaseClient._build_document_query DSL generation."""

from unittest.mock import patch

import pytest


@pytest.fixture()
def client():
    """Create a BaseClient subclass instance with mocked abstract methods."""
    from pyseekdb.client.client_base import BaseClient

    with patch.multiple(BaseClient, __abstractmethods__=set()):
        instance = BaseClient.__new__(BaseClient)
    return instance


class TestBuildDocumentQuery:
    """TestBuildDocumentQuery class."""

    def test_and_contains_has_default_operator_and(self, client):
        """Test and contains has default operator and."""
        where_document = {"$and": [{"$contains": "TOKENZPX"}, {"$contains": "TOKENALP"}]}
        result = client._build_document_query(where_document)
        assert result is not None
        qs = result["query_string"]
        assert qs["default_operator"] == "and"
        assert "TOKENZPX" in qs["query"]
        assert "TOKENALP" in qs["query"]
        assert qs["fields"] == ["document"]

    def test_or_contains_has_default_operator_or(self, client):
        """Test or contains has default operator or."""
        where_document = {"$or": [{"$contains": "TOKENZPX"}, {"$contains": "TOKENALP"}]}
        result = client._build_document_query(where_document)
        assert result is not None
        qs = result["query_string"]
        assert qs["default_operator"] == "or"
        assert "TOKENZPX" in qs["query"]
        assert "TOKENALP" in qs["query"]
        assert " OR " not in qs["query"]

    def test_single_contains_no_default_operator(self, client):
        """Test single contains no default operator."""
        where_document = {"$contains": "hello"}
        result = client._build_document_query(where_document)
        assert result is not None
        qs = result["query_string"]
        assert "default_operator" not in qs
        assert qs["query"] == "hello"

    def test_and_contains_with_boost(self, client):
        """Test and contains with boost."""
        where_document = {"$and": [{"$contains": "foo"}, {"$contains": "bar"}]}
        result = client._build_document_query(where_document, boost=2.0)
        assert result is not None
        qs = result["query_string"]
        assert qs["default_operator"] == "and"
        assert qs["boost"] == 2.0

    def test_and_contains_three_terms(self, client):
        """Test and contains three terms."""
        where_document = {
            "$and": [
                {"$contains": "alpha"},
                {"$contains": "beta"},
                {"$contains": "gamma"},
            ]
        }
        result = client._build_document_query(where_document)
        assert result is not None
        qs = result["query_string"]
        assert qs["default_operator"] == "and"
        assert qs["query"] == "alpha beta gamma"
