"""Unit tests for namespace explicit embedding dimension validation."""

import pytest

from pyseekdb.client.validators import (
    _validate_namespace_explicit_embedding_dimensions,
    _validate_namespace_no_index_explicit_embeddings,
)


class TestNoIndexExplicitEmbeddingDimension:
    def test_ignores_none_embeddings(self):
        _validate_namespace_no_index_explicit_embeddings(
            [None, [0.0] * 384],
            expected_dimension=384,
        )

    def test_accepts_matching_dimension(self):
        _validate_namespace_no_index_explicit_embeddings(
            [[0.0] * 384, [1.0] * 384],
            expected_dimension=384,
        )

    def test_rejects_wrong_dimension(self):
        with pytest.raises(ValueError, match="384-dimensional"):
            _validate_namespace_no_index_explicit_embeddings(
                [[1.0, 2.0, 3.0]],
                expected_dimension=384,
            )

    def test_reports_index_in_error(self):
        with pytest.raises(ValueError, match="index 1"):
            _validate_namespace_no_index_explicit_embeddings(
                [[0.0] * 384, [1.0, 2.0]],
                expected_dimension=384,
            )


class TestIvfExplicitEmbeddingDimension:
    def test_ignores_none_embeddings(self):
        _validate_namespace_explicit_embedding_dimensions(
            [None, [1.0, 0.0, 0.0]],
            expected_dimension=3,
            has_vector_index=True,
        )

    def test_rejects_wrong_dimension(self):
        with pytest.raises(ValueError, match="Embedding dimension mismatch: expected 3"):
            _validate_namespace_explicit_embedding_dimensions(
                [[1.0, 2.0]],
                expected_dimension=3,
                has_vector_index=True,
            )

    def test_accepts_matching_dimension(self):
        _validate_namespace_explicit_embedding_dimensions(
            [[1.0, 0.0, 0.0]],
            expected_dimension=3,
            has_vector_index=True,
        )
