"""
Model management module for Obsidian Vault Tools Memory Service.

This module provides comprehensive model management for transformer models,
embeddings, and local AI models with automatic downloading, caching, and
device optimization support.

Key Features:
- Transformer model management (Gemma 2-2B, SmolLM2 variants)
- Embedding model management with multiple provider support
- Automatic model downloading and caching
- Device optimization (CPU/GPU/MPS)
- Model quantization and memory optimization
- Integration with Hugging Face transformers ecosystem
"""

from .model_manager import ModelManager
from .model_cache import ModelCache
from .model_config import ModelConfig, TransformerModelConfig, EmbeddingModelConfig
from .transformer_models import TransformerModelManager

__all__ = [
    'ModelManager',
    'ModelCache', 
    'ModelConfig',
    'TransformerModelConfig',
    'EmbeddingModelConfig',
    'TransformerModelManager'
]

# Version info
__version__ = "1.0.0"