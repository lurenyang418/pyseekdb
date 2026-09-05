"""
Base client interface definition
"""

import contextlib
import logging
import uuid

from pymysql.converters import escape_string

from .base_connection import BaseConnection
from .capabilities import BackendCapabilitiesMixin
from .collection import Collection
from .collection_catalog import _DEFAULT_PARTITION_COUNT, CollectionCatalogMixin
from .collection_lifecycle import COLLECTION_STATE_CREATING
from .collection_operations import CollectionOperationsMixin
from .configuration import (
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_VECTOR_DIMENSION,
    MAX_HNSW_VECTOR_DIMENSION,
    MAX_IVF_VECTOR_DIMENSION,
    HNSWConfiguration,
    IVFIndexType,
)
from .embedding_function import EmbeddingFunction
from .meta_info import CollectionFieldNames, CollectionNames
from .namespace_operations import NamespaceOperationsMixin
from .query_builder import (
    build_fulltext_index_sql as _get_fulltext_index_sql,
)
from .query_builder import (
    build_sparse_vector_index_sql as _get_sparse_vector_index_sql,
)
from .query_builder import (
    build_vector_index_sql as _get_vector_index_sql,
)
from .schema import Schema
from .types import _NOT_PROVIDED
from .validators import (
    _MAX_COLLECTION_NAME_LENGTH,  # noqa: F401 - kept as an internal compatibility alias
    _validate_collection_name,
)

logger = logging.getLogger(__name__)


class BaseClient(
    CollectionCatalogMixin,
    CollectionOperationsMixin,
    NamespaceOperationsMixin,
    BackendCapabilitiesMixin,
    BaseConnection,
):
    """
    Abstract base class for all clients.

    Design Pattern:
    1. Provides public collection management methods (create_collection, get_collection, etc.)
    2. Defines internal operation interfaces (_collection_* methods) called by Collection objects
    3. Subclasses implement all abstract methods to provide specific business logic

    Benefits of this design:
    - Collection object interface is unified regardless of which client created it
    - Different clients can have completely different underlying implementations (SQL/gRPC/REST)
    - Easy to extend with new client types

    This class is intentionally not a standalone concrete client: its shared
    operations require the connection contract supplied by a backend mixin
    (``_execute()``, ``_ensure_connection()``, and cursor helpers). Concrete
    clients provide that lifecycle while this class hosts the shared collection
    implementation.
    """

    # ==================== Collection Management (User-facing) ====================

    def create_collection(
        self,
        name: str,
        schema: Schema | None = None,
        use_namespace: bool = False,
        partition_count: int | None = None,
    ) -> "Collection":
        """Create a new collection.

        Args:
            name: The name of the collection to create. Must contain only alphanumeric
                characters or underscores.
            schema: Schema configuration. Defaults to ``Schema()``. The schema contains
                all dense, sparse, full-text, and embedding-function configuration.
            use_namespace: If True, create a namespace-enabled collection. Defaults to False.
            partition_count: Number of partitions for the namespace physical tables.
                Only valid when ``use_namespace=True``. Defaults to 1000 when not provided.
                Passing it for a non-namespace collection raises ``ValueError``.

        Returns:
            The created ``Collection`` object.

        Raises:
            ValueError: If the collection name is invalid, already exists, or if the
                schema/embedding function combination is invalid (e.g., dimension mismatch).
            TypeError: If the schema object is of an invalid type.

        Examples:
            Create a collection with an explicit schema:

            >>> schema = Schema(
            ...     vector_index=VectorIndexConfig(hnsw=HNSWConfiguration(dimension=128, distance="l2"))
            ... )
            >>> collection = client.create_collection("custom_config", schema=schema)
        """
        _validate_collection_name(name)
        if partition_count is not None and not use_namespace:
            raise ValueError(
                "partition_count is only supported for namespace-enabled collections (use_namespace=True)."
            )
        # Only fully initialized collections (metadata + physical table) count as existing.
        # Metadata without a table is treated as an incomplete create and repaired below.
        if self.has_collection(name) and not (use_namespace and self._is_incomplete_ns_collection(name)):
            raise ValueError(f"Collection '{name}' already exists")

        if schema is None and use_namespace:
            raise ValueError(
                "use_namespace=True requires an explicit Schema with an IVF vector index. "
                "When schema is omitted, create_collection builds the default non-namespace "
                "schema (HNSW), which namespace collections do not support. "
                "Pass schema=Schema(vector_index=VectorIndexConfig("
                "ivf=IVFConfiguration(dimension=..., distance=...)), ...). "
                "Or set use_namespace=False for a standard HNSW collection."
            )
        elif schema is None:
            schema = Schema()

        if not isinstance(schema, Schema):
            raise TypeError(f"schema must be a Schema instance, got {type(schema).__name__}")
        logger.debug(f"schema: {schema}")

        if use_namespace:
            return self._create_namespace_collection(name, schema, partition_count=partition_count)

        if schema.vector_index.ivf is not None:
            raise ValueError(
                "Standard collections currently support HNSW vector indexes only. "
                "Use use_namespace=True for IVF collections."
            )

        # Resolve HNSW configuration dimension if not set
        hnsw_config = schema.vector_index.hnsw
        dense_embedding_function = schema.vector_index.embedding_function
        if dense_embedding_function is _NOT_PROVIDED:
            raise ValueError(
                "Invalid Schema: `vector_index.embedding_function` must be an EmbeddingFunction, None, "
                "or omitted when the dense dimension is explicit."
            )
        if hnsw_config is None:
            if dense_embedding_function is None:
                # Schema() omits the optional wrapper configuration. Standard collections still
                # use the documented default dense schema, while callers provide embeddings manually.
                hnsw_config = HNSWConfiguration(dimension=DEFAULT_VECTOR_DIMENSION, distance=DEFAULT_DISTANCE_METRIC)
            else:
                actual_dimension = self._get_embedding_function_dimension(dense_embedding_function)
                hnsw_config = HNSWConfiguration(dimension=actual_dimension, distance=DEFAULT_DISTANCE_METRIC)
        else:
            # Validate dimension matches embedding function if available
            if dense_embedding_function is not None:
                actual_dimension = self._get_embedding_function_dimension(dense_embedding_function)
                if hnsw_config.dimension != actual_dimension:
                    raise ValueError(
                        f"Schema dimension ({hnsw_config.dimension}) doesn't match "
                        f"embedding function dimension ({actual_dimension})."
                    )

        dimension = hnsw_config.dimension
        if dimension < 1 or dimension > MAX_HNSW_VECTOR_DIMENSION:
            raise ValueError(f"Dimension must be between 1 and {MAX_HNSW_VECTOR_DIMENSION}, got {dimension}")

        # Extract fulltext parser configuration
        fulltext_index_clause = _get_fulltext_index_sql(schema.fulltext_index)

        # Sparse vector index
        sparse_vector_index_config = schema.sparse_vector_index
        sparse_field_sql = (
            f"{CollectionFieldNames.SPARSE_EMBEDDING} SPARSEVECTOR,\n" if sparse_vector_index_config else ""
        )
        sparse_index_sql = (
            f",\n            VECTOR INDEX idx_sparse ({CollectionFieldNames.SPARSE_EMBEDDING}) {_get_sparse_vector_index_sql(sparse_vector_index_config)}"
            if sparse_vector_index_config
            else ""
        )

        # Construct table name
        creation_token = uuid.uuid4().hex
        collection_meta = self._create_collection_meta(
            name,
            dense_embedding_function,
            sparse_vector_index_config=sparse_vector_index_config,
            dimension=dimension,
            distance=hnsw_config.distance,
            lifecycle_state=COLLECTION_STATE_CREATING,
            creation_token=creation_token,
        )
        collection_id = collection_meta.get("collection_id")
        table_name = collection_meta["table_name"]

        # Construct CREATE TABLE SQL statement with HEAP organization
        sql = f"""CREATE TABLE IF NOT EXISTS `{table_name}` (
            _id varbinary(512) PRIMARY KEY NOT NULL,
            document string,
            embedding vector({dimension}),
            {sparse_field_sql}metadata json,
            FULLTEXT INDEX idx_fts(document) {fulltext_index_clause},
            VECTOR INDEX idx_vec (embedding) {_get_vector_index_sql(hnsw_config)}{sparse_index_sql}
        ) ORGANIZATION = HEAP;"""

        # Execute SQL to create table and publish the catalog row only after
        # the physical resource exists.  The token prevents a concurrent
        # creator or explicit recovery operation from publishing this row.
        try:
            logger.debug(f"Creating table: {table_name} with SQL: {sql}")
            self._execute(sql)
            self._mark_collection_ready(name, collection_id, creation_token)
        except BaseException:
            with contextlib.suppress(BaseException):
                self._cleanup_failed_collection(name, collection_id, creation_token)
            raise

        # Create and return Collection object
        return Collection(
            client=self,
            name=name,
            collection_id=collection_id,
            dimension=dimension,
            embedding_function=schema.vector_index.embedding_function,
            distance=hnsw_config.distance,
            sparse_vector_index_config=sparse_vector_index_config,
        )

    def _create_namespace_collection(
        self, name: str, schema: Schema, partition_count: int | None = None
    ) -> "Collection":
        """Create a namespace-enabled collection and its catalog/physical tables."""
        dense_embedding_function = schema.vector_index.embedding_function
        ivf_config = schema.vector_index.ivf
        hnsw_config = schema.vector_index.hnsw

        if schema.sparse_vector_index is not None:
            raise ValueError(
                "use_namespace=True does not support SparseVectorIndexConfig yet. "
                "Remove sparse_vector_index from the schema or set use_namespace=False."
            )
        if hnsw_config is not None:
            raise ValueError(
                "use_namespace=True does not support HNSW. Namespace collections require an IVF "
                "schema: Schema(vector_index=VectorIndexConfig(ivf=IVFConfiguration(dimension=..., "
                "distance=...)), ...)."
            )
        if ivf_config is not None and ivf_config.type != IVFIndexType.IVF_FLAT.value:
            raise ValueError(
                f"use_namespace=True currently only supports IVF index type '{IVFIndexType.IVF_FLAT.value}', "
                f"got '{ivf_config.type}'"
            )
        self._validate_ob_database_type()

        # Resume an incomplete collection (meta row exists but physical tables are
        # missing, e.g. a crash interrupted creation): reuse its id/settings and
        # idempotently (re)build only the missing physical tables. Otherwise create
        # from scratch. Both paths are safe to call repeatedly.
        existing = self._get_ns_collection_meta(name)
        if existing is not None:
            collection_id = existing["collection_id"]
            settings = existing.get("settings", {})
            dimension = settings.get("dimension")
            distance = settings.get("distance", DEFAULT_DISTANCE_METRIC)
            pc = int(settings.get("partition_count", _DEFAULT_PARTITION_COUNT))
            is_ss = settings.get("storage_mode") == "ss"
            logger.info(
                f"Namespace collection '{name}' already has a catalog entry; resuming creation "
                f"(reusing collection_id={collection_id}, rebuilding any missing physical tables)."
            )
        else:
            pc = _DEFAULT_PARTITION_COUNT if partition_count is None else partition_count
            if pc < 1:
                raise ValueError("partition_count must be >= 1")
            if ivf_config is not None:
                dimension = ivf_config.dimension
                distance = ivf_config.distance
            else:
                if dense_embedding_function is not None:
                    dimension = self._get_embedding_function_dimension(dense_embedding_function)
                else:
                    dimension = DEFAULT_VECTOR_DIMENSION
                distance = DEFAULT_DISTANCE_METRIC

            is_ss = self._is_shared_storage_mode()
            settings = {
                "version": 2,
                "use_namespace": True,
                "storage_mode": "ss" if is_ss else "sn",
                "dimension": dimension,
                "distance": distance,
                "partition_count": pc,
            }
            if ivf_config is not None:
                settings["dense_index_type"] = "ivf"
                if ivf_config.centroids_fresh_mode is not None:
                    settings["centroids_fresh_mode"] = ivf_config.centroids_fresh_mode
            if schema.fulltext_index is not None:
                settings["has_fulltext_index"] = True
            if dense_embedding_function is not None and EmbeddingFunction.support_persistence(dense_embedding_function):
                settings["embedding_function"] = {
                    "name": dense_embedding_function.name(),
                    "properties": dense_embedding_function.get_config(),
                }

            collection_meta = self._create_ns_collection_meta(name, settings)
            collection_id = collection_meta["collection_id"]

        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError(f"dimension must be an integer, got {type(dimension).__name__}")
        if dimension < 1 or dimension > MAX_IVF_VECTOR_DIMENSION:
            raise ValueError(
                f"Dimension must be between 1 and {MAX_IVF_VECTOR_DIMENSION} for namespace "
                f"IVF collections, got {dimension}"
            )

        self._ensure_namespace_catalogs()

        try:
            self._create_namespace_physical_tables(
                collection_id=collection_id,
                dimension=dimension,
                ivf_config=ivf_config,
                fulltext_config=schema.fulltext_index,
                is_shared_storage=is_ss,
                partition_count=pc,
                # On a fresh create, roll back partial tables so a failure leaves no
                # trace. On resume, keep whatever already exists so a later retry can
                # finish the job.
                cleanup_on_error=existing is None,
            )
        except Exception:
            if existing is None:
                with contextlib.suppress(Exception):
                    collection_id_escaped = escape_string(collection_id)
                    self._execute(
                        f"DELETE FROM `{CollectionNames.sdk_collections_table_name()}` "
                        f"WHERE collection_id = '{collection_id_escaped}'"
                    )
            raise

        return Collection(
            client=self,
            name=name,
            collection_id=collection_id,
            dimension=dimension,
            embedding_function=dense_embedding_function,
            distance=distance,
            use_namespace=True,
            partition_count=pc,
            has_vector_index=settings.get("dense_index_type") == "ivf",
        )
