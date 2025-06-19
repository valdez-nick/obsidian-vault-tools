"""
Model Adapters Package
Contains adapters for different model types and providers
"""

from .ollama_adapter import OllamaAdapter
from .transformer_adapter import TransformerAdapter
from .embedding_adapter import EmbeddingAdapter
from .classifier_adapter import ClassifierAdapter

__all__ = [
    "OllamaAdapter",
    "TransformerAdapter", 
    "EmbeddingAdapter",
    "ClassifierAdapter"
]