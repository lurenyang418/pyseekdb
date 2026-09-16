"""Tests for collection write helpers shared by sync and async clients."""

import pytest

from pyseekdb.client.collection_dml import (
    build_collection_upsert_statement,
    prepare_collection_write_batch,
)


class _EmbeddingFunction:
    def __call__(self, documents: list[str]) -> list[list[float]]:
        return [[float(len(document)), 0.0] for document in documents]


def test_prepare_collection_write_batch_generates_and_validates_embeddings() -> None:
    batch = prepare_collection_write_batch(
        ["one", "two"],
        None,
        [{"rank": 1}, {"rank": 2}],
        ["a", "bb"],
        operation="add",
        embedding_function=_EmbeddingFunction(),
    )

    assert batch.ids == ["one", "two"]
    assert batch.embeddings == [[1.0, 0.0], [2.0, 0.0]]


def test_prepare_collection_write_batch_rejects_short_embedding_output() -> None:
    with pytest.raises(ValueError, match="returned 1 vectors, expected 2"):
        prepare_collection_write_batch(
            ["one", "two"],
            None,
            None,
            ["a", "bb"],
            operation="upsert",
            embedding_function=lambda _documents: [[1.0, 0.0]],
        )


def test_shared_upsert_builder_parameterizes_user_values() -> None:
    sql, params = build_collection_upsert_statement(
        "`c$v2$collection-id`",
        "id' OR '1'='1",
        "document'; DROP TABLE users; --",
        [1.0, 0.0],
        {"note": "metadata's value"},
    )

    assert "CAST(%s AS BINARY)" in sql
    assert "document'; DROP TABLE users; --" not in sql
    assert "metadata's value" not in sql
    assert params == [
        "id' OR '1'='1",
        "document'; DROP TABLE users; --",
        '{"note": "metadata\'s value"}',
    ]
    assert "ON DUPLICATE KEY UPDATE" in sql
