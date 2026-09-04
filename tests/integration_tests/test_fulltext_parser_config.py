"""Integration tests for full-text parser configuration."""

import contextlib
import time

import pytest

from pyseekdb import FulltextIndexConfig, HNSWConfiguration, Schema, VectorIndexConfig
from pyseekdb.client.meta_info import CollectionNames


class TestFulltextIndexConfig:
    """Test full-text parser configuration against supported backends."""

    @staticmethod
    def _schema(dimension: int, parser: str | None = None, params: dict | None = None) -> Schema:
        return Schema(
            vector_index=VectorIndexConfig(hnsw=HNSWConfiguration(dimension=dimension, distance="cosine")),
            fulltext_index=(FulltextIndexConfig(analyzer=parser, properties=params) if parser else None),
        )

    @staticmethod
    def _table_name(collection) -> str:
        return CollectionNames.table_name(collection.id)

    def _assert_parser(self, client, collection, parser: str, params: dict | None = None) -> None:
        table_name = self._table_name(collection)
        try:
            result = client._server._execute(f"SHOW CREATE TABLE `{table_name}`")
            assert result
            row = result[0]
            create_stmt = (
                row[1] if isinstance(row, (tuple, list)) else row.get("Create Table", row.get("create table", ""))
            )
            assert "FULLTEXT KEY" in create_stmt.upper()
            assert f"PARSER {parser}".upper() in create_stmt.upper()
            for key, value in (params or {}).items():
                assert key in create_stmt or str(value) in create_stmt
        finally:
            with contextlib.suppress(Exception):
                client.delete_collection(collection.name)

    def test_fulltext_parser(self, db_client):
        """Create collections using each supported parser and parser options."""
        for parser in ["ik", "space", "ngram", "ngram2", "beng"]:
            name = f"test_fulltext_{parser}_{int(time.time() * 1000)}"
            collection = db_client.create_collection(name=name, schema=self._schema(128, parser))
            self._assert_parser(db_client, collection, parser)

        name = f"test_fulltext_ngram_{int(time.time() * 1000)}"
        collection = db_client.create_collection(
            name=name,
            schema=self._schema(128, "ngram", {"ngram_token_size": 3}),
        )
        self._assert_parser(db_client, collection, "ngram", {"ngram_token_size": 3})

    def test_default_parser(self, db_client):
        """Use the default IK parser when no full-text configuration is supplied."""
        name = f"test_default_parser_{int(time.time() * 1000)}"
        collection = db_client.create_collection(name=name, schema=self._schema(128))
        self._assert_parser(db_client, collection, "ik")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
