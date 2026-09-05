"""
Unit tests for configuration classes
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pyseekdb import (  # noqa: E402
    FulltextIndexConfig,
    HNSWConfiguration,
    IKProperties,
    Ngram2Properties,
    NgramProperties,
    Schema,
    SpaceProperties,
)
from pyseekdb.client.client import Client  # noqa: E402
from pyseekdb.client.client_base import BaseClient  # noqa: E402
from pyseekdb.client.configuration import IVFConfiguration, VectorIndexConfig  # noqa: E402
from pyseekdb.client.embedding_function import EmbeddingFunction  # noqa: E402
from pyseekdb.client.query_builder import build_vector_index_sql as _get_vector_index_sql  # noqa: E402
from pyseekdb.client.types import _NOT_PROVIDED  # noqa: E402


class TestHNSWConfiguration:
    """Test HNSWConfiguration class"""

    def test_valid_configuration(self):
        """Test creating valid HNSWConfiguration"""
        config = HNSWConfiguration(dimension=128, distance="cosine", type="hnsw", lib="vsag")
        assert config.dimension == 128
        assert config.distance == "cosine"
        assert config.type == "hnsw"
        assert config.lib == "vsag"

    def test_default_distance(self):
        """Test default distance metric"""
        config = HNSWConfiguration()
        assert config.distance == "cosine"
        assert config.dimension == 384

    def test_invalid_dimension(self):
        """Test that invalid dimension raises ValueError"""
        with pytest.raises(ValueError, match="must be between"):
            HNSWConfiguration(dimension=0)

        with pytest.raises(ValueError, match="must be between"):
            HNSWConfiguration(dimension=-1)
        with pytest.raises(ValueError, match="must be between"):
            HNSWConfiguration(dimension=4097)

    def test_invalid_distance(self):
        """Test that invalid distance raises ValueError"""
        with pytest.raises(ValueError, match="distance must be one of"):
            HNSWConfiguration(dimension=128, distance="invalid")

    def test_invalid_type(self):
        """Test that invalid type raises ValueError"""
        with pytest.raises(ValueError, match="type must be one of"):
            HNSWConfiguration(type="invalid")

    def test_invalid_lib(self):
        """Test that invalid lib raises ValueError"""
        with pytest.raises(ValueError, match="lib must be one of"):
            HNSWConfiguration(lib="invalid")

    def test_properties_with_primitive_types(self):
        """Test properties with primitive value types"""
        config = HNSWConfiguration(
            dimension=128,
            distance="cosine",
            type="hnsw",
            lib="vsag",
            properties={
                "normalize": True,
                "quantization": "pq",
                "alpha": 0.75,
            },
        )

        assert config.properties["normalize"] is True
        assert config.properties["quantization"] == "pq"
        assert config.properties["alpha"] == 0.75

    def test_properties_invalid_type(self):
        """Test properties with invalid value types"""
        with pytest.raises(TypeError, match="properties must be a dictionary of string, int, float, or bool"):
            HNSWConfiguration(
                dimension=128,
                properties={
                    "m": 16,
                    "invalid": {"nested": "dict"},
                },
            )

    def test_properties_reserved_keywords(self):
        """Test that top-level keys in properties are removed with warning"""
        with pytest.warns(UserWarning, match="reserved keyword"):
            config = HNSWConfiguration(
                dimension=128,
                distance="cosine",
                properties={
                    "distance": "cosine",
                    "type": "hnsw",
                    "lib": "vsag",
                    "M": 32,
                },
            )
        assert "distance" not in {key.lower() for key in config.properties}
        assert "type" not in {key.lower() for key in config.properties}
        assert "lib" not in {key.lower() for key in config.properties}
        assert "m" not in {key.lower() for key in config.properties}

    def test_hnsw_numeric_ranges(self):
        with pytest.raises(ValueError, match="M must be between 5 and 128"):
            HNSWConfiguration(M=3)
        with pytest.raises(ValueError, match="ef_construction must be between 5 and 1000"):
            HNSWConfiguration(ef_construction=1001)
        with pytest.raises(ValueError, match="ef_search must be between 1 and 1000"):
            HNSWConfiguration(ef_search=0)
        with pytest.raises(ValueError, match="extra_info_max_size must be between 0 and 16384"):
            HNSWConfiguration(extra_info_max_size=17000)

    def test_hnsw_bq_properties(self):
        config = HNSWConfiguration(type="hnsw_bq", refine_k=4.0, refine_type="sq8", bq_bits_query=32, bq_use_fht=True)
        assert config.type == "hnsw_bq"
        assert config.refine_k == 4.0
        assert config.refine_type == "sq8"
        assert config.bq_bits_query == 32
        assert config.bq_use_fht is True

    def test_vector_index_sql_with_properties(self):
        """Test SQL generation includes properties"""
        config = HNSWConfiguration(
            dimension=128,
            distance="cosine",
            type="hnsw_sq",
            lib="vsag",
            M=16,
            ef_search=200,
            properties={
                "quantization": "pq",
            },
        )
        sql = _get_vector_index_sql(config)
        assert "DISTANCE=cosine" in sql
        assert "TYPE=hnsw_sq" in sql
        assert "LIB=vsag" in sql
        assert "M=16" in sql
        assert "ef_search=200" in sql
        assert "quantization='pq'" in sql

    def test_vector_index_sql_with_bq_fields(self):
        """Test SQL generation quotes string fields and formats bool fields"""
        config = HNSWConfiguration(
            dimension=128,
            type="hnsw_bq",
            refine_k=4.0,
            refine_type="sq8",
            bq_bits_query=32,
            bq_use_fht=True,
        )
        sql = _get_vector_index_sql(config)
        assert "refine_k=4.0" in sql
        assert "refine_type='sq8'" in sql
        assert "bq_bits_query=32" in sql
        assert "bq_use_fht=" in sql


class TestFulltextIndexConfig:
    """Test FulltextIndexConfig class"""

    def test_valid_parsers(self):
        """Test creating FulltextIndexConfig with valid parsers"""
        valid_parsers = ["ik", "space", "ngram", "ngram2", "beng"]
        for parser in valid_parsers:
            config = FulltextIndexConfig(analyzer=parser)
            assert config.analyzer == parser
            assert config.properties is None

    def test_default_parser(self):
        """Test default parser is 'ik'"""
        config = FulltextIndexConfig()
        assert config.analyzer == "ik"

    def test_parser_with_params(self):
        """Test parser with parameters"""
        config = FulltextIndexConfig(analyzer="ngram", properties={"ngram_token_size": 2})
        assert config.analyzer == "ngram"
        assert config.properties == {"ngram_token_size": 2}

    def test_unknown_analyzer_warns(self):
        with pytest.warns(UserWarning, match="Unknown analyzer"):
            config = FulltextIndexConfig(analyzer="jieba", properties={"token_size": 4})
        assert config.analyzer == "jieba"
        assert config.properties["token_size"] == 4

    def test_space_analyzer_param_validation(self):
        with pytest.raises(ValueError, match="max_token_size should not be less than min_token_size"):
            FulltextIndexConfig(analyzer="space", properties=SpaceProperties(min_token_size=16, max_token_size=10))

    def test_ngram_analyzer_param_validation(self):
        with pytest.raises(ValueError, match="ngram_token_size must be between 1 and 10"):
            FulltextIndexConfig(analyzer="ngram", properties=NgramProperties(ngram_token_size=11))

    def test_ngram2_analyzer_param_validation(self):
        with pytest.raises(ValueError, match="max_ngram_size should not be less than min_ngram_size"):
            FulltextIndexConfig(analyzer="ngram2", properties=Ngram2Properties(min_ngram_size=10, max_ngram_size=2))

    def test_ik_mode_validation(self):
        with pytest.raises(ValueError, match="ik_mode should be one of"):
            FulltextIndexConfig(analyzer="ik", properties=IKProperties(ik_mode="invalid"))

    def test_params_with_different_types(self):
        """Test params with different primitive types"""
        config = FulltextIndexConfig(
            analyzer="jieba",
            properties={
                "string_param": "value",
                "int_param": 42,
                "float_param": 3.14,
                "bool_param": True,
            },
        )
        assert config.properties["string_param"] == "value"
        assert config.properties["int_param"] == 42
        assert config.properties["float_param"] == 3.14
        assert config.properties["bool_param"] is True


class _StubEmbeddingFunction(EmbeddingFunction):
    """Tiny dense-vector EF for VectorIndexConfig validation tests."""

    def __call__(self, documents):  # type: ignore[override]
        docs = documents if isinstance(documents, list) else [documents]
        return [[0.0] * 4 for _ in docs]

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config):  # type: ignore[override]
        return _StubEmbeddingFunction()

    @staticmethod
    def name() -> str:
        return "stub"


class TestVectorIndexConfigEmbeddingRequired:
    """VectorIndexConfig no longer ships a default EF (since pyseekdb 2.0).

    An EF is only required when no explicit ``dimension=`` is supplied on the
    HNSW/IVF configuration; otherwise the dimension is unknown and the SDK
    cannot infer the vector size.
    """

    def test_no_hnsw_no_ivf_no_ef_is_ok(self):
        config = VectorIndexConfig()
        assert config.embedding_function is None
        assert config.hnsw is None
        assert config.ivf is None

    def test_hnsw_with_ef_is_ok(self):
        config = VectorIndexConfig(
            hnsw=HNSWConfiguration(dimension=4),
            embedding_function=_StubEmbeddingFunction(),
        )
        assert config.hnsw is not None
        assert config.embedding_function is not None

    def test_hnsw_with_explicit_dimension_no_ef_is_ok(self):
        """Explicit dimension removes the need for an EF."""
        config = VectorIndexConfig(hnsw=HNSWConfiguration(dimension=4))
        assert config.hnsw is not None
        assert config.embedding_function is None

    def test_ivf_with_explicit_dimension_no_ef_is_ok(self):
        """Explicit dimension removes the need for an EF."""
        config = VectorIndexConfig(ivf=IVFConfiguration(dimension=4))
        assert config.ivf is not None
        assert config.embedding_function is None

    def test_hnsw_without_dimension_or_ef_raises(self):
        # Replace the default dimension with None to simulate "no dimension known".
        hnsw = HNSWConfiguration()
        hnsw.dimension = None
        with pytest.raises(ValueError, match=r"requires an `embedding_function=`"):
            VectorIndexConfig(hnsw=hnsw)

    def test_ivf_without_dimension_or_ef_raises(self):
        ivf = IVFConfiguration()
        ivf.dimension = None
        with pytest.raises(ValueError, match=r"requires an `embedding_function=`"):
            VectorIndexConfig(ivf=ivf)

    def test_remote_client_ping_executes_healthcheck_query(self):
        client = Client.__new__(Client)
        client._execute = MagicMock(return_value=[{"pyseekdb_ping": 1}])

        assert client.ping()
        client._execute.assert_called_once_with("SELECT 1 AS pyseekdb_ping")

    def test_standard_schema_uses_default_dimension_without_ef(self):
        client = MagicMock(spec=BaseClient)
        client.has_collection.return_value = False
        client._create_collection_meta.return_value = {
            "collection_id": "collection-id",
            "table_name": "table-name",
        }

        collection = BaseClient.create_collection(client, "items", schema=Schema())

        assert collection.dimension == 384
        create_sql = client._execute.call_args.args[0]
        assert "embedding vector(384)" in create_sql

    def test_standard_collection_rejects_ivf_configuration(self):
        """IVF must not be silently replaced with the default HNSW index."""
        client = MagicMock(spec=BaseClient)
        client.has_collection.return_value = False
        schema = Schema(vector_index=IVFConfiguration(dimension=3, distance="l2"))

        with pytest.raises(ValueError, match="support HNSW vector indexes only"):
            BaseClient.create_collection(client, "items", schema=schema)

        client._create_collection_meta.assert_not_called()

    def test_standard_collection_metadata_persists_distance_and_dimension(self):
        """Reopening through another client must have the original vector settings."""
        client = MagicMock(spec=BaseClient)
        client._get_collection_id.side_effect = [ValueError("not found"), "collection-id"]
        client._execute.return_value = []

        BaseClient._create_collection_meta(client, "items", None, dimension=3, distance="l2")

        insert_call = next(call for call in client._execute.call_args_list if "INSERT INTO" in call.args[0])
        settings = json.loads(insert_call.args[1][1])
        assert settings["dimension"] == 3
        assert settings["distance"] == "l2"

    def test_missing_persisted_ef_is_optional_when_reopening(self):
        client = MagicMock(spec=BaseClient)

        assert BaseClient._validate_embedding_function(client, _NOT_PROVIDED, None) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
