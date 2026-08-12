"""Unit tests for KNN filter DSL generation with hoisted must_not."""

from unittest.mock import patch

import pytest


@pytest.fixture()
def client():
    """Client."""
    from pyseekdb.client.client_base import BaseClient

    with patch.multiple(BaseClient, __abstractmethods__=set()):
        instance = BaseClient.__new__(BaseClient)
    return instance


class TestBuildKnnFilterNe:
    """TestBuildKnnFilterNe class."""

    def test_ne_hoists_must_not_in_knn_filter(self, client):
        """Test ne hoists must not in knn filter."""
        with patch.object(
            client,
            "_build_metadata_filter_for_search_parm",
            return_value=[{"bool": {"must_not": [{"term": {"data_content.metadata.zpx_hint": 0}}]}}],
        ):
            knn_expr = client._build_knn_expression(
                {"query_embeddings": [1.0, 1.0, 0.0], "where": {"zpx_hint": {"$ne": 0}}, "n_results": 40},
                dimension=3,
            )

        assert knn_expr["filter"] == [
            {
                "bool": {
                    "filter": [{"range": {"data_content.metadata.zpx_hint": {"gte": -9223372036854775808}}}],
                    "must_not": [{"term": {"data_content.metadata.zpx_hint": 0}}],
                }
            }
        ]
