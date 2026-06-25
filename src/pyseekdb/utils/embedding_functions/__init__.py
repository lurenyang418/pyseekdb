"""
Embedding function implementations for pyseekdb.

This module provides various embedding function implementations that can be used
with pyseekdb collections. Each implementation may require additional dependencies;
imports are guarded so the package remains usable without all extras installed.
"""

import importlib
import logging

logger = logging.getLogger(__name__)


def _safe_import(mod_name: str, cls_name: str):
    """Try to import a class from a sibling module, return None on failure."""
    try:
        mod = importlib.import_module(f".{mod_name}", __package__)
        return getattr(mod, cls_name)
    except ImportError as e:
        logger.debug(f"Optional embedding function {cls_name} not available: {e}")
        return None


# Dense embedding functions
DefaultEmbeddingFunction = _safe_import("default_embedding_function", "DefaultEmbeddingFunction")
OnnxEmbeddingFunction = _safe_import("onnx_embedding_function", "OnnxEmbeddingFunction")
SentenceTransformerEmbeddingFunction = _safe_import(
    "sentence_transformer_embedding_function", "SentenceTransformerEmbeddingFunction"
)

# OpenAI-compatible base
OpenAIBaseEmbeddingFunction = _safe_import("openai_base_embedding_function", "OpenAIBaseEmbeddingFunction")

# Provider-specific dense embedding functions
OpenAIEmbeddingFunction = _safe_import("openai_embedding_function", "OpenAIEmbeddingFunction")
QwenEmbeddingFunction = _safe_import("qwen_embedding_function", "QwenEmbeddingFunction")
MistralEmbeddingFunction = _safe_import("mistral_embedding_function", "MistralEmbeddingFunction")
MorphEmbeddingFunction = _safe_import("morph_embedding_function", "MorphEmbeddingFunction")
SiliconflowEmbeddingFunction = _safe_import("siliconflow_embedding_function", "SiliconflowEmbeddingFunction")
TencentHunyuanEmbeddingFunction = _safe_import("tencent_hunyuan_embedding_function", "TencentHunyuanEmbeddingFunction")
Text2VecEmbeddingFunction = _safe_import("text2vec_embedding_function", "Text2VecEmbeddingFunction")
OllamaEmbeddingFunction = _safe_import("ollama_embedding_function", "OllamaEmbeddingFunction")
VoyageaiEmbeddingFunction = _safe_import("voyageai_embedding_function", "VoyageaiEmbeddingFunction")
GoogleVertexEmbeddingFunction = _safe_import("google_vertex_embedding_function", "GoogleVertexEmbeddingFunction")
CohereEmbeddingFunction = _safe_import("cohere_embedding_function", "CohereEmbeddingFunction")
JinaEmbeddingFunction = _safe_import("jina_embedding_function", "JinaEmbeddingFunction")
AmazonBedrockEmbeddingFunction = _safe_import("amazon_bedrock_embedding_function", "AmazonBedrockEmbeddingFunction")
LiteLLMBaseEmbeddingFunction = _safe_import("litellm_base_embedding_function", "LiteLLMBaseEmbeddingFunction")

# Sparse embedding functions
BM25SparseEmbeddingFunction = _safe_import("bm25_sparse_embedding_function", "BM25SparseEmbeddingFunction")
HuggingFaceSparseEmbeddingFunction = _safe_import(
    "huggingface_sparse_embedding_function", "HuggingFaceSparseEmbeddingFunction"
)
