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

from .model_manager import ModelManager, ModelInfo, ModelLoadError
from .model_cache import ModelCache, CacheEntry
from .model_config import (
    ModelConfig, 
    TransformerModelConfig, 
    EmbeddingModelConfig,
    ModelType,
    DeviceType,
    QuantizationType
)
from .transformer_models import TransformerModelManager, GenerationResult

__all__ = [
    # Core classes
    'ModelManager',
    'ModelCache', 
    'ModelConfig',
    'TransformerModelConfig',
    'EmbeddingModelConfig',
    'TransformerModelManager',
    
    # Data classes
    'ModelInfo',
    'CacheEntry',
    'GenerationResult',
    
    # Enums
    'ModelType',
    'DeviceType', 
    'QuantizationType',
    
    # Exceptions
    'ModelLoadError'
]

# Version info
__version__ = "2.0.0"