"""Integration tests for :meth:`Collection.fork`."""

import contextlib
import logging
import time
import uuid

import pytest

import pyseekdb

logger = logging.getLogger(__name__)


class TestCollectionFork:
    """Test collection fork behavior against supported backends."""

    @staticmethod
    def _schema() -> pyseekdb.Schema:
        return pyseekdb.Schema(
            vector_index=pyseekdb.VectorIndexConfig(hnsw=pyseekdb.HNSWConfiguration(dimension=3, distance="l2"))
        )

    @staticmethod
    def _fork_enabled(client) -> bool:
        try:
            return client._server._fork_table_enabled()
        except Exception:
            logger.exception("Failed to check if fork is enabled")
            return False

    def _create_collection(self, client, suffix: str):
        name = f"test_fork_{suffix}_{int(time.time() * 1000)}"
        return name, client.get_or_create_collection(name=name, schema=self._schema())

    def test_fork_success(self, db_client):
        """A fork contains the original collection's data."""
        if not self._fork_enabled(db_client):
            pytest.skip("Fork is not enabled for this database")

        collection_name, collection = self._create_collection(db_client, "original")
        forked_name = f"{collection_name}_forked"
        test_ids = [str(uuid.uuid4()) for _ in range(3)]
        try:
            collection.add(
                ids=test_ids,
                embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                documents=["Document 1", "Document 2", "Document 3"],
                metadatas=[{"tag": "A"}, {"tag": "B"}, {"tag": "C"}],
            )

            forked = collection.fork(forked_name)

            assert forked.name == forked_name
            assert forked.dimension == collection.dimension
            assert forked.count() == collection.count() == 3
            data = forked.get(ids=test_ids)
            assert set(data["documents"]) == {"Document 1", "Document 2", "Document 3"}
            assert {metadata["tag"] for metadata in data["metadatas"]} == {"A", "B", "C"}
        finally:
            with contextlib.suppress(Exception):
                db_client.delete_collection(forked_name)
            with contextlib.suppress(Exception):
                db_client.delete_collection(collection_name)

    @pytest.mark.parametrize("invalid_name", ["invalid-name", ""])
    def test_fork_rejects_invalid_name(self, db_client, invalid_name):
        """Fork rejects names that cannot identify a collection."""
        if not self._fork_enabled(db_client):
            pytest.skip("Fork is not enabled for this database")

        collection_name, collection = self._create_collection(db_client, "invalid")
        try:
            with pytest.raises(ValueError, match="Invalid collection name"):
                collection.fork(invalid_name)
        finally:
            with contextlib.suppress(Exception):
                db_client.delete_collection(collection_name)

    def test_fork_preserves_original(self, db_client):
        """Changes to the fork do not modify the original collection."""
        if not self._fork_enabled(db_client):
            pytest.skip("Fork is not enabled for this database")

        collection_name, collection = self._create_collection(db_client, "independent")
        forked_name = f"{collection_name}_forked"
        original_id = str(uuid.uuid4())
        forked_id = str(uuid.uuid4())
        try:
            collection.add(
                ids=original_id,
                embeddings=[1.0, 2.0, 3.0],
                documents="Original document",
                metadatas={"source": "original"},
            )
            forked = collection.fork(forked_name)
            forked.add(
                ids=forked_id,
                embeddings=[4.0, 5.0, 6.0],
                documents="Forked document",
                metadatas={"source": "forked"},
            )

            assert collection.count() == 1
            assert forked.count() == 2
            assert forked_id not in collection.get()["ids"]
            assert original_id in forked.get()["ids"]
        finally:
            with contextlib.suppress(Exception):
                db_client.delete_collection(forked_name)
            with contextlib.suppress(Exception):
                db_client.delete_collection(collection_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
