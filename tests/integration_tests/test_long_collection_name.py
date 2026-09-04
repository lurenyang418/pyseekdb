"""
Integration tests for collection names exceeding 64 characters.
"""

import hashlib
from typing import ClassVar

import pytest

import pyseekdb


class Simple3DEmbeddingFunction:
    """Simple 3D embedding function for testing."""

    def __init__(self):
        self.dimension = 3

    def __call__(self, texts: str | list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for doc in texts:
            # Use a deterministic hash for testing
            hash_val = int(hashlib.md5(doc.encode(), usedforsecurity=False).hexdigest(), 16) % 1000
            embedding = [
                float((hash_val % 10) / 10.0),
                float(((hash_val // 10) % 10) / 10.0),
                float(((hash_val // 100) % 10) / 10.0),
            ]
            embeddings.append(embedding)
        return embeddings


class TestLongCollectionName:
    """Test collection operations with names > 64 characters using standard db_client."""

    LONG_NAMES: ClassVar[list[str]] = [
        "a" * 65,  # Boundary + 1
        "b" * 128,  # 128 chars
        "c" * 512,  # 512 chars (Max)
    ]

    def test_create_collection_with_long_name(self, db_client):
        """Test creating and deleting collections with long names."""
        for name in self.LONG_NAMES:
            try:
                collection = db_client.create_collection(
                    name=name, schema=pyseekdb.Schema(embedding_function=Simple3DEmbeddingFunction())
                )
                assert collection.name == name
                assert db_client.has_collection(name=name)
            finally:
                # Ensure cleanup even if assertion fails
                if db_client.has_collection(name=name):
                    db_client.delete_collection(name=name)

    @pytest.mark.xfail(reason="Issue #121: get_or_create uses raw name as identifier")
    def test_get_or_create_with_long_name(self, db_client):
        """Test get_or_create_collection with long names (Known Limitation)."""
        name = "x" * 100
        try:
            collection = db_client.get_or_create_collection(
                name=name, schema=pyseekdb.Schema(embedding_function=Simple3DEmbeddingFunction())
            )
            assert collection.name == name
        finally:
            if db_client.has_collection(name=name):
                db_client.delete_collection(name=name)

    def test_boundary_cases(self, db_client):
        """Test boundary cases for collection name lengths."""
        for length in [64, 65, 512]:
            name = "z" * length
            try:
                collection = db_client.create_collection(
                    name=name, schema=pyseekdb.Schema(embedding_function=Simple3DEmbeddingFunction())
                )
                assert len(collection.name) == length
            finally:
                if db_client.has_collection(name=name):
                    db_client.delete_collection(name=name)
