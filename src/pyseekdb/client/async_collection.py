"""Asynchronous collection facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .validators import _validate_n_results

if TYPE_CHECKING:
    from .embedding_function import EmbeddingFunction
    from .query_types import QueryHint
    from .schema import SparseVectorIndexConfig


class AsyncCollection:
    """A collection handle whose network operations are awaitable."""

    def __init__(
        self,
        client: Any,
        name: str,
        collection_id: str,
        dimension: int | None,
        embedding_function: EmbeddingFunction | None = None,
        distance: str | None = None,
        sparse_vector_index_config: SparseVectorIndexConfig | None = None,
        use_namespace: bool = False,
        **metadata: Any,
    ) -> None:
        self._client = client
        self._name = name
        self._id = collection_id
        self._dimension = dimension
        self._embedding_function = embedding_function
        self._distance = distance
        self._sparse_vector_index_config = sparse_vector_index_config
        self._use_namespace = use_namespace
        self._metadata = metadata

    @property
    def name(self) -> str:
        """Collection name."""
        return self._name

    @property
    def id(self) -> str:
        """Collection identifier."""
        return self._id

    @property
    def dimension(self) -> int | None:
        """Dense vector dimension."""
        return self._dimension

    @property
    def client(self) -> Any:
        """Associated :class:`AsyncClient`."""
        return self._client

    @property
    def metadata(self) -> dict[str, Any]:
        """Collection metadata."""
        return self._metadata

    @property
    def embedding_function(self) -> EmbeddingFunction | None:
        """Configured dense embedding function."""
        return self._embedding_function

    @property
    def distance(self) -> str | None:
        """Vector distance metric."""
        return self._distance

    @property
    def use_namespace(self) -> bool:
        """Whether this collection uses namespace storage."""
        return self._use_namespace

    def _guard_standard_collection(self) -> None:
        """Reject namespace collections until their async operations are implemented."""
        if self._use_namespace:
            raise NotImplementedError(
                "Async namespace collection operations are not implemented yet; "
                "use the synchronous Client for namespace collections."
            )

    def __repr__(self) -> str:
        """Return a debug-friendly representation."""
        return (
            f"AsyncCollection(name='{self._name}', dimension={self._dimension}, client={type(self._client).__name__})"
        )

    async def add(
        self,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Add records to the collection."""
        self._guard_standard_collection()
        await self._client._collection_add(
            self,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            **kwargs,
        )

    async def update(
        self,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Update existing records."""
        self._guard_standard_collection()
        await self._client._collection_update(
            self,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            **kwargs,
        )

    async def upsert(
        self,
        ids: str | list[str],
        embeddings: list[float] | list[list[float]] | None = None,
        metadatas: dict | list[dict] | None = None,
        documents: str | list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Insert records or update records with matching IDs."""
        self._guard_standard_collection()
        await self._client._collection_upsert(
            self,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            **kwargs,
        )

    async def delete(
        self,
        ids: str | list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Delete records by IDs or filters."""
        self._guard_standard_collection()
        await self._client._collection_delete(
            self,
            ids=ids,
            where=where,
            where_document=where_document,
            **kwargs,
        )

    async def query(
        self,
        query_embeddings: list[float] | list[list[float]] | None = None,
        query_texts: str | list[str] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
        query_key: Any | None = None,
        query_hint: QueryHint | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Query the collection by dense vector similarity."""
        self._guard_standard_collection()
        _validate_n_results(n_results)
        return await self._client._collection_query(
            self,
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
            query_key=query_key,
            query_hint=query_hint,
            **kwargs,
        )

    async def get(
        self,
        ids: str | list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: list[str] | None = None,
        query_hint: QueryHint | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Retrieve records by IDs or filters."""
        self._guard_standard_collection()
        return await self._client._collection_get(
            self,
            ids=ids,
            where=where,
            where_document=where_document,
            limit=limit,
            offset=offset,
            include=include,
            query_hint=query_hint,
            **kwargs,
        )

    async def count(self) -> int:
        """Return the number of records."""
        self._guard_standard_collection()
        return await self._client._collection_count(self)

    async def peek(self, limit: int = 10) -> dict[str, Any]:
        """Return a small preview of the collection."""
        self._guard_standard_collection()
        return await self.get(limit=limit, include=["documents", "metadatas", "embeddings"])

    async def fork(self, forked_name: str) -> AsyncCollection:
        """Fork this collection and return a handle to the copy."""
        self._guard_standard_collection()
        return await self._client._collection_fork(self, forked_name)

    async def refresh_index(self) -> None:
        """Refresh asynchronous vector-index build tasks when supported."""
        await self._client.refresh_index()
