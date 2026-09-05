"""Unit tests for state-free SQL and result helpers shared by clients."""

import struct

import pytest

from pyseekdb.client.query_builder import (
    build_select_clause,
    build_where_clause,
    convert_id_from_bytes,
    convert_id_to_sql,
    embed_texts,
    normalize_collection_batch,
    normalize_include_fields,
    normalize_query_embeddings,
    parse_embedding_value,
    process_get_row,
    process_query_row,
)
from pyseekdb.client.validators import _validate_include


def test_query_configuration_helpers_normalize_shared_inputs() -> None:
    assert normalize_query_embeddings([1.0, 2.0]) == [[1.0, 2.0]]
    assert normalize_query_embeddings([[1.0], [2.0]]) == [[1.0], [2.0]]
    assert normalize_include_fields(None) == {"documents": True, "metadatas": True}
    assert build_select_clause({"documents": True, "embeddings": True}) == "_id, embedding, document"


def test_include_accepts_ids_as_an_always_returned_field() -> None:
    _validate_include(["ids", "documents", "metadatas"])


def test_collection_batch_helper_normalizes_sync_and_async_inputs() -> None:
    normalized = normalize_collection_batch(
        "one",
        [1.0, 2.0],
        {"rank": 1},
        "hello",
        require_values=True,
    )
    assert normalized == (["one"], [[1.0, 2.0]], [{"rank": 1}], ["hello"])

    with pytest.raises(ValueError, match="does not match number of ids"):
        normalize_collection_batch(["one", "two"], None, None, ["hello"])


def test_where_builder_keeps_ids_parameterized_and_combines_filters() -> None:
    clause, params = build_where_clause(
        {"rank": {"$gte": 2}},
        {"$contains": "hello"},
        ["one", 2],
    )

    assert clause.startswith("WHERE _id IN (CAST(%s AS BINARY),CAST(%s AS BINARY))")
    assert "JSON_EXTRACT(metadata" in clause
    assert "MATCH(document)" in clause
    assert params[:2] == ["one", "2"]
    assert params[-1] == "hello"


def test_id_and_result_helpers_decode_database_values() -> None:
    assert convert_id_from_bytes(b"one") == "one"
    assert convert_id_from_bytes(b"\xff") == "ff"
    assert convert_id_to_sql("one's") == "CAST('one\\'s' AS BINARY)"

    include = {"documents": True, "metadatas": True, "embeddings": True}
    get_row = process_get_row(
        {"_id": b"one", "document": "hello", "metadata": '{"rank": 1}', "embedding": "[1.0, 0.0]"},
        include,
    )
    assert get_row == {
        "id": "one",
        "document": "hello",
        "metadata": {"rank": 1},
        "embedding": [1.0, 0.0],
    }

    query_row = process_query_row(
        {"_id": b"one", "metadata": '{"rank": 1}', "distance": "0.25"},
        include,
    )
    assert query_row == {"_id": "one", "metadata": {"rank": 1}, "distance": 0.25}

    query_row_without_distance = process_query_row({"_id": b"one", "distance": None}, include)
    assert query_row_without_distance == {"_id": "one"}


def test_embedding_value_decodes_json_and_packed_float32_bytes() -> None:
    assert parse_embedding_value([1.0, 2.0]) == [1.0, 2.0]
    assert parse_embedding_value("[1.0, 2.0]") == [1.0, 2.0]
    assert parse_embedding_value(b"[1.0, 2.0]") == [1.0, 2.0]
    assert parse_embedding_value(struct.pack("<2f", 1.5, -2.0)) == pytest.approx([1.5, -2.0])
    assert parse_embedding_value(b"") == []


def test_embedding_value_rejects_invalid_packed_bytes() -> None:
    with pytest.raises(ValueError, match="not divisible by 4"):
        parse_embedding_value(b"bad")


def test_embed_texts_requires_a_function_and_normalizes_single_text() -> None:
    with pytest.raises(NotImplementedError, match="Text embedding is not implemented"):
        embed_texts("hello")

    assert embed_texts("hello", lambda texts: [[float(len(texts[0]))]]) == [[5.0]]
