"""
Configuration management for model management system.

Provides configuration classes for transformer models, embedding models,
and general model management settings with 2025 state-of-the-art defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from enum import Enum


class ModelType(Enum):
    """Supported model types."""
    TRANSFORMER = "transformer"
    EMBEDDING = "embedding"
    QUANTIZED = "quantized"


class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


class QuantizationType(Enum):
    """Supported quantization types."""
    NONE = "none"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class ModelConfig:
    """Base configuration for model management."""
    
    # Base directories
    cache_dir: Optional[str] = None
    models_dir: Optional[str] = None
    
    # Device settings
    device: DeviceType = DeviceType.AUTO
    use_gpu_if_available: bool = True
    
    # Memory management
    max_memory_gb: Optional[float] = None
    low_memory_mode: bool = False
    
    # Performance settings
    batch_size: int = 1
    max_workers: int = 4
    
    # Caching settings
    enable_model_caching: bool = True
    cache_ttl_hours: int = 24
    auto_cleanup: bool = True
    
    def __post_init__(self):
        """Initialize default paths and validate configuration."""
        if self.cache_dir is None:
            self.cache_dir = str(
                Path.home() / ".cache" / "obsidian_vault_tools" / "models"
            )
        
        if self.models_dir is None:
            self.models_dir = str(
                Path.home() / ".local" / "share" / "obsidian_vault_tools" / "models"
            )
        
        # Ensure directories exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def optimize_for_device(self) -> None:
        """Optimize configuration based on available hardware."""
        if self.device == DeviceType.AUTO:
            try:
                import torch
                if torch.cuda.is_available() and self.use_gpu_if_available:
                    self.device = DeviceType.CUDA
                    # Increase batch size for GPU
                    self.batch_size = min(self.batch_size * 4, 32)
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = DeviceType.MPS
                    self.batch_size = min(self.batch_size * 2, 16)
                else:
                    self.device = DeviceType.CPU
            except ImportError:
                self.device = DeviceType.CPU
    
    def get_device_str(self) -> str:
        """Get device string for PyTorch/transformers."""
        if self.device == DeviceType.AUTO:
            self.optimize_for_device()
        return self.device.value


@dataclass
class TransformerModelConfig(ModelConfig):
    """Configuration for transformer models (2025 state-of-the-art)."""
    
    # Model selection (2025 recommendations)
    primary_model: str = "google/gemma-2-2b"  # State-of-the-art lightweight model
    fallback_models: List[str] = field(default_factory=lambda: [
        "HuggingFaceTB/SmolLM2-1.7B",
        "HuggingFaceTB/SmolLM2-360M",
        "HuggingFaceTB/SmolLM2-135M"
    ])
    
    # Model parameters
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Optimization settings
    quantization: QuantizationType = QuantizationType.FP16
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = False
    
    # Generation settings
    max_new_tokens: int = 512
    repetition_penalty: float = 1.1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # Memory optimization
    offload_to_cpu: bool = False
    device_map: str = "auto"
    
    def get_model_registry(self) -> Dict[str, Dict[str, Any]]:
        """Get registry of supported transformer models with metadata."""
        return {
            "google/gemma-2-2b": {
                "name": "Gemma 2-2B",
                "description": "Google's state-of-the-art 2B parameter model (2025)",
                "parameters": "2.6B",
                "memory_gb": 5.2,
                "context_length": 8192,
                "quantization_support": ["fp16", "int8", "int4"],
                "flash_attention": True,
                "recommended_batch_size": 1
            },
            "HuggingFaceTB/SmolLM2-1.7B": {
                "name": "SmolLM2 1.7B",
                "description": "Optimized lightweight model for edge devices",
                "parameters": "1.7B",
                "memory_gb": 3.4,
                "context_length": 8192,
                "quantization_support": ["fp16", "int8"],
                "flash_attention": True,
                "recommended_batch_size": 2
            },
            "HuggingFaceTB/SmolLM2-360M": {
                "name": "SmolLM2 360M",
                "description": "Ultra-lightweight model for resource-constrained environments",
                "parameters": "360M",
                "memory_gb": 1.4,
                "context_length": 8192,
                "quantization_support": ["fp16", "int8"],
                "flash_attention": False,
                "recommended_batch_size": 4
            },
            "HuggingFaceTB/SmolLM2-135M": {
                "name": "SmolLM2 135M",
                "description": "Minimal model for basic text generation",
                "parameters": "135M",
                "memory_gb": 0.6,
                "context_length": 2048,
                "quantization_support": ["fp16"],
                "flash_attention": False,
                "recommended_batch_size": 8
            }
        }
    
    def select_optimal_model(self, available_memory_gb: Optional[float] = None) -> str:
        """Select optimal model based on available system resources."""
        registry = self.get_model_registry()
        
        if available_memory_gb is None:
            # Try to detect available memory
            try:
                import psutil
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
            except ImportError:
                available_memory_gb = 8.0  # Conservative default
        
        # Filter models that fit in available memory
        suitable_models = []
        for model_id, info in registry.items():
            memory_needed = info["memory_gb"]
            if self.quantization != QuantizationType.NONE:
                memory_needed *= 0.5  # Rough estimate for quantization
            
            if memory_needed <= available_memory_gb * 0.8:  # Leave 20% buffer
                suitable_models.append((model_id, info))
        
        if not suitable_models:
            # Fallback to smallest model
            return "HuggingFaceTB/SmolLM2-135M"
        
        # Return the largest model that fits
        suitable_models.sort(key=lambda x: x[1]["memory_gb"], reverse=True)
        return suitable_models[0][0]


@dataclass
class EmbeddingModelConfig(ModelConfig):
    """Configuration for embedding models (extends embedding service config)."""
    
    # Model selection (2025 recommendations)
    primary_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    fallback_models: List[str] = field(default_factory=lambda: [
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    ])
    
    # OpenAI-compatible models
    openai_models: List[str] = field(default_factory=lambda: [
        "text-embedding-3-small",
        "text-embedding-3-large", 
        "text-embedding-ada-002"
    ])
    
    # Model parameters
    max_sequence_length: int = 512
    normalize_embeddings: bool = True
    batch_size: int = 32
    
    # Advanced settings
    trust_remote_code: bool = False
    use_auth_token: Optional[str] = None
    
    def get_embedding_registry(self) -> Dict[str, Dict[str, Any]]:
        """Get registry of supported embedding models with metadata."""
        return {
            "sentence-transformers/all-MiniLM-L6-v2": {
                "name": "All-MiniLM-L6-v2",
                "description": "General-purpose sentence embeddings",
                "dimensions": 384,
                "max_sequence_length": 256,
                "model_size_mb": 90,
                "performance_score": 0.85,
                "use_case": "general"
            },
            "sentence-transformers/all-MiniLM-L12-v2": {
                "name": "All-MiniLM-L12-v2", 
                "description": "Higher quality sentence embeddings",
                "dimensions": 384,
                "max_sequence_length": 256,
                "model_size_mb": 130,
                "performance_score": 0.88,
                "use_case": "general"
            },
            "sentence-transformers/paraphrase-MiniLM-L6-v2": {
                "name": "Paraphrase-MiniLM-L6-v2",
                "description": "Optimized for paraphrase detection",
                "dimensions": 384,
                "max_sequence_length": 128,
                "model_size_mb": 90,
                "performance_score": 0.82,
                "use_case": "paraphrase"
            },
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": {
                "name": "Multi-QA-MiniLM-L6-cos-v1",
                "description": "Optimized for question-answering tasks",
                "dimensions": 384,
                "max_sequence_length": 512,
                "model_size_mb": 90,
                "performance_score": 0.87,
                "use_case": "qa"
            },
            "text-embedding-3-small": {
                "name": "OpenAI Text Embedding 3 Small",
                "description": "OpenAI's latest small embedding model",
                "dimensions": 1536,
                "max_sequence_length": 8191,
                "model_size_mb": 0,  # API-based
                "performance_score": 0.92,
                "use_case": "general",
                "requires_api": True
            },
            "text-embedding-3-large": {
                "name": "OpenAI Text Embedding 3 Large",
                "description": "OpenAI's latest large embedding model",
                "dimensions": 3072,
                "max_sequence_length": 8191,
                "model_size_mb": 0,  # API-based
                "performance_score": 0.95,
                "use_case": "general",
                "requires_api": True
            }
        }
    
    def select_optimal_embedding_model(self, use_case: str = "general") -> str:
        """Select optimal embedding model based on use case and resources."""
        registry = self.get_embedding_registry()
        
        # Filter by use case
        suitable_models = [
            (model_id, info) for model_id, info in registry.items()
            if info["use_case"] == use_case or info["use_case"] == "general"
        ]
        
        if not suitable_models:
            suitable_models = list(registry.items())
        
        # Prefer local models unless API is explicitly requested
        local_models = [
            (model_id, info) for model_id, info in suitable_models
            if not info.get("requires_api", False)
        ]
        
        if local_models:
            # Sort by performance score
            local_models.sort(key=lambda x: x[1]["performance_score"], reverse=True)
            return local_models[0][0]
        
        # Fallback to API models if no local models available
        suitable_models.sort(key=lambda x: x[1]["performance_score"], reverse=True)
        return suitable_models[0][0]


# Default configuration instances
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRANSFORMER_CONFIG = TransformerModelConfig()
DEFAULT_EMBEDDING_CONFIG = EmbeddingModelConfig()