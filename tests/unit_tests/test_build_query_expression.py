"""Unit tests for BaseClient._build_query_expression DSL generation."""

from unittest.mock import patch

import pytest


@pytest.fixture()
def client():
    """Client."""
    from pyseekdb.client.client_base import BaseClient

    with patch.multiple(BaseClient, __abstractmethods__=set()):
        instance = BaseClient.__new__(BaseClient)
    return instance


class TestBuildQueryExpressionNotContains:
    """TestBuildQueryExpressionNotContains class."""

    def test_not_contains_with_metadata_filter_hoists_must_not(self, client):
        """$not_contains + where must not nest a must_not-only bool inside must."""
        where = {"seq": {"$gte": 43}}
        with patch.object(
            client,
            "_build_metadata_filter_for_search_parm",
            return_value=[{"range": {"data_content.metadata.seq": {"gte": 43}}}],
        ):
            expr = client._build_query_expression({
                "where_document": {"$not_contains": "TOKENZPX"},
                "where": where,
            })

        assert expr == {
            "bool": {
                "filter": [{"range": {"data_content.metadata.seq": {"gte": 43}}}],
                "must_not": [
                    {
                        "query_string": {
                            "fields": ["document"],
                            "query": "TOKENZPX",
                        }
                    }
                ],
            }
        }
        assert "must" not in expr["bool"]

    def test_not_contains_only_hoists_must_not_without_synthetic_filter(self, client):
        """Namespace path injects ns/lt filters later; no exists/match_all leaf here."""
        with patch.object(client, "_build_metadata_filter_for_search_parm", return_value=[]):
            expr = client._build_query_expression({
                "where_document": {"$not_contains": "TOKENZPX"},
            })

        assert expr == {
            "bool": {
                "must_not": [
                    {
                        "query_string": {
                            "fields": ["document"],
                            "query": "TOKENZPX",
                        }
                    }
                ],
            }
        }
        assert "filter" not in expr["bool"]

    def test_contains_with_metadata_filter_still_uses_must(self, client):
        """Test contains with metadata filter still uses must."""
        with patch.object(
            client,
            "_build_metadata_filter_for_search_parm",
            return_value=[{"term": {"data_content.metadata.seq": {"value": 1}}}],
        ):
            expr = client._build_query_expression({
                "where_document": {"$contains": "TOKENZPX"},
                "where": {"seq": 1},
            })

        assert "must" in expr["bool"]
        assert "query_string" in expr["bool"]["must"][0]
        assert "filter" in expr["bool"]


class TestBuildQueryExpressionMetadataNe:
    """TestBuildQueryExpressionMetadataNe class."""

    def test_ne_with_fts_hoists_must_not_from_filter(self, client):
        """$ne in where must not appear as a must_not-only bool inside filter."""
        with patch.object(
            client,
            "_build_metadata_filter_for_search_parm",
            return_value=[{"bool": {"must_not": [{"term": {"data_content.metadata.zpx_hint": 0}}]}}],
        ):
            expr = client._build_query_expression({
                "where_document": {"$contains": "document"},
                "where": {"zpx_hint": {"$ne": 0}},
            })

        assert expr == {
            "bool": {
                "must": [{"query_string": {"fields": ["document"], "query": "document"}}],
                "must_not": [{"term": {"data_content.metadata.zpx_hint": 0}}],
            }
        }
        filters = expr["bool"].get("filter", [])
        for item in filters:
            nested = item.get("bool", {}) if isinstance(item, dict) else {}
            assert set(nested.keys()) != {"must_not"}, nested

    def test_ne_with_not_contains_and_positive_where(self, client):
        """Test ne with not contains and positive where."""
        with patch.object(
            client,
            "_build_metadata_filter_for_search_parm",
            return_value=[
                {"bool": {"must_not": [{"term": {"data_content.metadata.zpx_hint": 0}}]}},
                {"range": {"data_content.metadata.seq": {"gte": 1}}},
            ],
        ):
            expr = client._build_query_expression({
                "where_document": {"$not_contains": "TOKENZPX"},
                "where": {"zpx_hint": {"$ne": 0}, "seq": {"$gte": 1}},
            })

        assert expr["bool"]["filter"] == [{"range": {"data_content.metadata.seq": {"gte": 1}}}]
        assert len(expr["bool"]["must_not"]) == 2
        assert "must" not in expr["bool"]
