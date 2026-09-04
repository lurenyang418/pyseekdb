"""
Metadata information for collection fields and namespace catalog naming.
"""

from typing import ClassVar


class CollectionFieldNames:
    """Standard field names used in collection record payloads."""

    ID = "_id"
    DOCUMENT = "document"
    EMBEDDING = "embedding"
    SPARSE_EMBEDDING = "sparse_embedding"
    METADATA = "metadata"

    ALL_FIELDS: ClassVar[list[str]] = [ID, DOCUMENT, EMBEDDING, METADATA]


class CollectionNames:
    """Helpers for mapping between collection names and physical table names."""

    _PREFIX = "c$v2$"

    @staticmethod
    def table_name(collection_id: str) -> str:
        """Convert collection id to table name."""
        return f"{CollectionNames._PREFIX}{collection_id}"

    @staticmethod
    def sdk_collections_table_name() -> str:
        """Return the SDK catalog table that stores collection metadata."""
        return "sdk_collections"


class NamespaceCollectionNames:
    """Naming helpers for namespace-enabled collection physical tables."""

    _LOGIC_DATA_SUFFIX = "_logic_data_table"
    _HOT_SUFFIX = "_hot_table"
    _KV_DATA_SUFFIX = "_kv_data_table"
    _LOGIC_SCHEMA_SUFFIX = "_logic_schema_table"
    _TG_SUFFIX = "_tg"

    @staticmethod
    def sdk_namespaces_table() -> str:
        """Return the SDK catalog table that stores namespace metadata."""
        return "sdk_namespaces"

    @staticmethod
    def sdk_ltables_table() -> str:
        """Return the SDK catalog table that stores logic-table metadata."""
        return "sdk_ltables"

    @staticmethod
    def sdk_namespaces_stats_table() -> str:
        """Return the SDK catalog table that stores namespace statistics."""
        return "sdk_namespaces_stats"

    @staticmethod
    def data_table_name(collection_id: str) -> str:
        """Build the logic data table name for a namespace-enabled collection."""
        return f"{collection_id}{NamespaceCollectionNames._LOGIC_DATA_SUFFIX}"

    @staticmethod
    def hot_table_name(collection_id: str) -> str:
        """Build the hot data table name for a namespace-enabled collection."""
        return f"{collection_id}{NamespaceCollectionNames._HOT_SUFFIX}"

    @staticmethod
    def kv_data_table_name(collection_id: str) -> str:
        """Build the KV data table name for a namespace-enabled collection."""
        return f"{collection_id}{NamespaceCollectionNames._KV_DATA_SUFFIX}"

    @staticmethod
    def logic_schema_table_name(collection_id: str) -> str:
        """Build the logic schema table name for a namespace-enabled collection."""
        return f"{collection_id}{NamespaceCollectionNames._LOGIC_SCHEMA_SUFFIX}"

    @staticmethod
    def tablegroup_name(collection_id: str) -> str:
        """Build the table group name for a namespace-enabled collection."""
        return f"{collection_id}{NamespaceCollectionNames._TG_SUFFIX}"

    @staticmethod
    def is_ns_data_table(table_name: str) -> bool:
        """Return True if ``table_name`` is a namespace logic data table."""
        return table_name.endswith(NamespaceCollectionNames._LOGIC_DATA_SUFFIX)


class NamespaceFieldNames:
    """Standard field names used in namespace record payloads."""

    NAMESPACE_ID = "namespace_id"
    LTABLE_ID = "ltable_id"
    DOCUMENT = "document"
    EMBEDDING = "embedding"
    DATA_CONTENT = "data_content"
