"""Unit tests for state-free SQL and result helpers shared by clients."""

import pytest

from pyseekdb.client.query_builder import (
    build_select_clause,
    build_where_clause,
    convert_id_from_bytes,
    convert_id_to_sql,
    embed_texts,
    normalize_include_fields,
    normalize_query_embeddings,
    process_get_row,
    process_query_row,
)


def test_query_configuration_helpers_normalize_shared_inputs() -> None:
    assert normalize_query_embeddings([1.0, 2.0]) == [[1.0, 2.0]]
    assert normalize_query_embeddings([[1.0], [2.0]]) == [[1.0], [2.0]]
    assert normalize_include_fields(None) == {"documents": True, "metadatas": True}
    assert build_select_clause({"documents": True, "embeddings": True}) == "_id, embedding, document"


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


def test_embed_texts_requires_a_function_and_normalizes_single_text() -> None:
    with pytest.raises(NotImplementedError, match="Text embedding is not implemented"):
        embed_texts("hello")

    assert embed_texts("hello", lambda texts: [[float(len(texts[0]))]]) == [[5.0]]
