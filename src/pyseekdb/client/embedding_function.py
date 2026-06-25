"""
Embedding function interface and implementations

This module provides the EmbeddingFunction protocol and default implementations
for converting text documents to vector embeddings.
"""

import importlib
import logging
from abc import abstractmethod
from typing import (
    Any,
    ClassVar,
    Protocol,
    TypeVar,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

# Type variable for input types
D = TypeVar("D")

# Type aliases
Documents = str | list[str]
Embeddings = list[list[float]]
Embedding = list[float]


@runtime_checkable
class EmbeddingFunction(Protocol[D]):
    """
    Protocol for embedding functions that convert documents to vectors.

    This is similar to Chroma's EmbeddingFunction interface.
    Implementations should convert text documents to vector embeddings.

    Implementations should also provide:
    - `name()`: Static method that returns a unique name identifier for routing (not persisted in config)
    - `get_config()`: Instance method that returns a configuration dictionary
    - `build_from_config(config)`: Static method that restores an instance from config

    Example:
        >>> class MyEmbeddingFunction(EmbeddingFunction[Documents]):
        ...     @staticmethod
        ...     def name() -> str:
        ...         return "my_embedding_function"
        ...     def __call__(self, documents: Documents) -> Embeddings:
        ...         # Convert documents to embeddings
        ...         return [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        ...     def get_config(self) -> Dict[str, Any]:
        ...         return {...}  # Note: 'name' is not included
        ...     @staticmethod
        ...     def build_from_config(config: Dict[str, Any]) -> "MyEmbeddingFunction":
        ...         return MyEmbeddingFunction(...)
        >>>
        >>> ef = MyEmbeddingFunction()
        >>> embeddings = ef(["Hello", "World"])
        >>> config = ef.get_config()
        >>> restored_ef = MyEmbeddingFunction.build_from_config(config)
    """

    @abstractmethod
    def __call__(self, documents: D) -> Embeddings:
        """
        Convert input documents to embeddings.

        Args:
            documents: Documents to embed (can be a single string or list of strings)

        Returns:
            List of embedding vectors (list of floats)
        """
        ...

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Get the configuration dictionary for the embedding function.

        This method should return a dictionary that contains all the information
        needed to restore the embedding function after restart.

        Returns:
            Dictionary containing the embedding function's configuration.
            Note: The 'name' field is not included as it's handled by the upper layer for routing.
        """
        return NotImplemented

    @staticmethod
    def support_persistence(embedding_function: Any) -> bool:
        """
        Check if the embedding function supports persistence.
        """
        if embedding_function is None:
            return False
        if (
            not hasattr(embedding_function, "name")
            or not hasattr(embedding_function, "build_from_config")
            or not hasattr(embedding_function, "get_config")
        ):
            return False
        try:
            if embedding_function.get_config() is NotImplemented:
                return False
        except Exception:
            return False
        return True


def dimension_of(embedding_function: EmbeddingFunction[D]) -> int:
    """
    Get the dimension of the embeddings produced by the embedding function.
    """
    if hasattr(embedding_function, "dimension") and callable(getattr(embedding_function, "dimension", None)):
        return embedding_function.dimension()
    elif hasattr(embedding_function, "dimension"):
        return embedding_function.dimension
    else:
        # Fallback: if no dimension attribute, call the function to calculate dimension
        # This may trigger model initialization, but is necessary for custom embedding functions
        test_embeddings = embedding_function.__call__("seekdb")
        if test_embeddings and len(test_embeddings) > 0:
            return len(test_embeddings[0])
        else:
            raise ValueError("Embedding function returned empty result when called with 'seekdb'")


def get_default_embedding_function():
    """
    Get or create the default embedding function instance.

    The DefaultEmbeddingFunction is an optional feature. Requires the
    ``default-embedding`` extra to be installed: ``pip install pyseekdb[default-embedding]``

    Returns:
        DefaultEmbeddingFunction instance

    Raises:
        ImportError: If pyseekdb[default-embedding] is not installed.
    """
    try:
        from pyseekdb.utils.embedding_functions.default_embedding_function import (
            get_default_embedding_function as _impl,
        )
        return _impl()
    except (ImportError, ValueError) as e:
        raise ImportError(
            "DefaultEmbeddingFunction is not available. "
            "Install it with: pip install pyseekdb[default-embedding]"
        ) from e


class EmbeddingFunctionRegistry:
    """
    Registry for embedding function classes.

    This registry maps embedding function names (returned by their name() method)
    to their corresponding classes, allowing dynamic instantiation from persisted configurations.

    To register a custom embedding function, you have two options:

    Option 1 (Recommended): Use the @register_embedding_function decorator:
       >>> @register_embedding_function
       ... class MyCustomEmbeddingFunction(EmbeddingFunction[Documents]):
       ...     # ... implementation ...

    Option 2: Manually register the class:
       >>> EmbeddingFunctionRegistry.register(MyCustomEmbeddingFunction)

    Your embedding function class must implement:
       - __call__() to convert documents to embeddings
       - A static name() method that returns a unique identifier
       - get_config() to return configuration dictionary
       - A static build_from_config() to restore from configuration

    Example:
        >>> from pyseekdb.client.embedding_function import (
        ...     EmbeddingFunction, Documents, Embeddings, EmbeddingFunctionRegistry
        ... )
        >>> from typing import Dict, Any
        >>>
        >>> class MyCustomEmbeddingFunction(EmbeddingFunction[Documents]):
        ...     def __init__(self, model_name: str = "my-model", dimension: int = 128):
        ...         self.model_name = model_name
        ...         self._dimension = dimension
        ...
        ...     def __call__(self, input: Documents) -> Embeddings:
        ...         # Your embedding logic here
        ...         if isinstance(input, str):
        ...             input = [input]
        ...         # Return list of embedding vectors
        ...         return [[0.1] * self._dimension for _ in input]
        ...
        ...     @property
        ...     def dimension(self) -> int:
        ...         return self._dimension
        ...
        ...     @staticmethod
        ...     def name() -> str:
        ...         return "my_custom_embedding"
        ...
        ...     def get_config(self) -> Dict[str, Any]:
        ...         return {
        ...             "model_name": self.model_name,
        ...             "dimension": self._dimension,
        ...         }
        ...
        ...     @staticmethod
        ...     def build_from_config(config: Dict[str, Any]) -> "MyCustomEmbeddingFunction":
        ...         return MyCustomEmbeddingFunction(
        ...             model_name=config.get("model_name", "my-model"),
        ...             dimension=config.get("dimension", 128),
        ...         )
        >>>
        >>> # Register your custom embedding function
        >>> EmbeddingFunctionRegistry.register(MyCustomEmbeddingFunction)
        >>>
        >>> # Now you can use it when creating collections
        >>> import pyseekdb
        >>> client = pyseekdb.Client(path="./db")
        >>> ef = MyCustomEmbeddingFunction()
        >>> collection = client.create_collection("my_collection", embedding_function=ef)
        >>>
        >>> # When the collection is retrieved later, it will automatically restore
        >>> # the embedding function using the registry
        >>> collection2 = client.get_collection("my_collection")
    """

    _registry: ClassVar[dict[str, type]] = {}
    _initialized: ClassVar[bool] = False

    @classmethod
    def _initialize(cls) -> None:
        """Initialize the registry with built-in embedding functions."""
        if cls._initialized:
            return

        # Try to register DefaultEmbeddingFunction (may not be installed)
        try:
            from pyseekdb.utils.embedding_functions.default_embedding_function import (
                DefaultEmbeddingFunction,
            )

            cls._registry["default"] = DefaultEmbeddingFunction
        except ImportError:
            pass

        # Try to register optional embedding functions (each individually)
        _optional_efs: list[tuple[str, str, str]] = [
            ("sentence_transformer", "sentence_transformer_embedding_function", "SentenceTransformerEmbeddingFunction"),
            ("openai", "openai_embedding_function", "OpenAIEmbeddingFunction"),
            ("qwen", "qwen_embedding_function", "QwenEmbeddingFunction"),
            ("mistral", "mistral_embedding_function", "MistralEmbeddingFunction"),
            ("morph", "morph_embedding_function", "MorphEmbeddingFunction"),
            ("siliconflow", "siliconflow_embedding_function", "SiliconflowEmbeddingFunction"),
            ("tencent_hunyuan", "tencent_hunyuan_embedding_function", "TencentHunyuanEmbeddingFunction"),
            ("text2vec", "text2vec_embedding_function", "Text2VecEmbeddingFunction"),
            ("ollama", "ollama_embedding_function", "OllamaEmbeddingFunction"),
            ("voyageai", "voyageai_embedding_function", "VoyageaiEmbeddingFunction"),
            ("google_vertex", "google_vertex_embedding_function", "GoogleVertexEmbeddingFunction"),
            ("cohere", "cohere_embedding_function", "CohereEmbeddingFunction"),
            ("jina", "jina_embedding_function", "JinaEmbeddingFunction"),
            ("amazon_bedrock", "amazon_bedrock_embedding_function", "AmazonBedrockEmbeddingFunction"),
        ]
        for name, module_name, class_name in _optional_efs:
            try:
                module = importlib.import_module(f"pyseekdb.utils.embedding_functions.{module_name}")
                cls._registry[name] = getattr(module, class_name)
            except ImportError:
                pass

        cls._initialized = True

    @classmethod
    def register(cls, embedding_function_class: type) -> None:
        """
        Register an embedding function class.

        This method should be called before creating collections that use the custom
        embedding function. Once registered, the embedding function can be automatically
        restored from persisted collection metadata.

        Args:
            embedding_function_class: The embedding function class to register.
                                    Must implement:
                                    - A static name() method that returns a unique identifier
                                    - A get_config() instance method that returns configuration dict
                                    - A static build_from_config(config) method to restore instances

        Raises:
            ValueError: If the class doesn't have the required methods or if the name
                       is already registered to a different class.

        Example:
            >>> from pyseekdb.client.embedding_function import EmbeddingFunctionRegistry
            >>> EmbeddingFunctionRegistry.register(MyCustomEmbeddingFunction)
            >>>
            >>> # Verify registration
            >>> assert "my_custom_embedding" in EmbeddingFunctionRegistry.list_registered()
        """
        cls._initialize()

        if not hasattr(embedding_function_class, "name") or not hasattr(embedding_function_class, "build_from_config"):
            raise ValueError(
                f"Embedding function class {embedding_function_class.__name__} "
                f"must have a static name() method, static build_from_config() method"
            )

        name = embedding_function_class.name()
        if name in cls._registry and cls._registry[name] != embedding_function_class:
            raise ValueError(
                f"Embedding function name '{name}' is already registered to {cls._registry[name].__name__}"
            )

        cls._registry[name] = embedding_function_class
        logger.debug(f"Registered embedding function '{name}' -> {embedding_function_class.__name__}")

    @classmethod
    def get_class(cls, name: str) -> type | None:
        """
        Get an embedding function class by name.

        Args:
            name: The name identifier of the embedding function (as returned by its name() method).

        Returns:
            The embedding function class if found, None otherwise.
        """
        cls._initialize()
        return cls._registry.get(name)

    @classmethod
    def list_registered(cls) -> list[str]:
        """
        List all registered embedding function names.

        Returns:
            List of registered embedding function names.
        """
        cls._initialize()
        return list(cls._registry.keys())


T = TypeVar("T", bound=type)


def register_embedding_function(embedding_function_class: type[T]) -> type[T]:
    """
    Decorator to automatically register an embedding function class.

    This decorator can be used as a class decorator to automatically register
    an embedding function when the class is defined, eliminating the need to
    manually call EmbeddingFunctionRegistry.register().

    Args:
        embedding_function_class: The embedding function class to register.
                                Must implement:
                                - A static name() method that returns a unique identifier
                                - A get_config() instance method that returns configuration dict
                                - A static build_from_config(config) method to restore instances

    Returns:
        The same class (for use as a decorator).

    Raises:
        ValueError: If the class doesn't have the required methods or if the name
                   is already registered to a different class.

    Example:
        >>> from pyseekdb.client.embedding_function import (
        ...     EmbeddingFunction, Documents, Embeddings, register_embedding_function
        ... )
        >>> from typing import Dict, Any
        >>>
        >>> @register_embedding_function
        ... class MyCustomEmbeddingFunction(EmbeddingFunction[Documents]):
        ...     def __init__(self, model_name: str = "my-model"):
        ...         self.model_name = model_name
        ...
        ...     def __call__(self, input: list[str]|str) -> list[list[float]]:
        ...         # Your embedding logic
        ...         return [[0.1, 0.2, 0.3] for _ in (input if isinstance(input, list) else [input])]
        ...
        ...     @staticmethod
        ...     def name() -> str:
        ...         return "my_custom_embedding"
        ...
        ...     def get_config(self) -> Dict[str, Any]:
        ...         return {"model_name": self.model_name}
        ...
        ...     @staticmethod
        ...     def build_from_config(config: Dict[str, Any]) -> "MyCustomEmbeddingFunction":
        ...         return MyCustomEmbeddingFunction(model_name=config.get("model_name", "my-model"))
        >>>
        >>> # The class is now automatically registered!
        >>> # You can use it immediately when creating collections
        >>> import pyseekdb
        >>> client = pyseekdb.Client(path="./seekdb.db")
        >>> ef = MyCustomEmbeddingFunction()
        >>> collection = client.create_collection("my_collection", embedding_function=ef)
    """
    EmbeddingFunctionRegistry.register(embedding_function_class)
    return embedding_function_class
