"""
Model Manager for Obsidian Vault Tools Memory Service.

Provides unified model lifecycle management for transformers, embeddings,
and quantized models with automatic downloading, caching, and optimization.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

try:
    from .model_config import ModelConfig, TransformerModelConfig, EmbeddingModelConfig, ModelType, DeviceType, QuantizationType
    from .model_cache import ModelCache
except ImportError:
    from model_config import ModelConfig, TransformerModelConfig, EmbeddingModelConfig, ModelType, DeviceType, QuantizationType
    from model_cache import ModelCache

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_id: str
    model_type: ModelType
    model_object: Any
    config: ModelConfig
    load_time: datetime
    last_used: datetime
    memory_usage_gb: float
    device: str
    is_quantized: bool = False
    quantization_type: Optional[QuantizationType] = None


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class ModelManager:
    """
    Unified model manager for all model types.
    
    Features:
    - Automatic model downloading and caching
    - Device-aware model loading (CPU/GPU/MPS)
    - Model quantization support
    - Memory-aware model management
    - Lazy loading and unloading
    - Thread-safe operations
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize model manager with configuration."""
        self.config = config or ModelConfig()
        self.config.optimize_for_device()
        
        # Initialize cache
        self.cache = ModelCache(self.config)
        
        # Model registry - tracks loaded models
        self._loaded_models: Dict[str, ModelInfo] = {}
        self._loading_lock = threading.Lock()
        self._usage_stats: Dict[str, Dict] = {}
        
        # Model type handlers
        self._model_handlers = {
            ModelType.TRANSFORMER: self._load_transformer_model,
            ModelType.EMBEDDING: self._load_embedding_model,
            ModelType.QUANTIZED: self._load_quantized_model
        }
        
        logger.info(f"Initialized ModelManager with device: {self.config.get_device_str()}")
    
    def _ensure_dependencies(self, model_type: ModelType) -> bool:
        """Check if required dependencies are available for model type."""
        try:
            if model_type in [ModelType.TRANSFORMER, ModelType.QUANTIZED]:
                import transformers
                import torch
                return True
            elif model_type == ModelType.EMBEDDING:
                import sentence_transformers
                return True
            return True
        except ImportError as e:
            logger.error(f"Dependencies not available for {model_type}: {e}")
            return False
    
    def load_model(
        self,
        model_id: str,
        model_type: ModelType,
        force_reload: bool = False,
        **kwargs
    ) -> ModelInfo:
        """
        Load a model with caching and optimization.
        
        Args:
            model_id: Model identifier (HuggingFace model name or path)
            model_type: Type of model to load
            force_reload: Force reload even if cached
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelInfo object with loaded model details
        """
        with self._loading_lock:
            # Check if already loaded
            if model_id in self._loaded_models and not force_reload:
                model_info = self._loaded_models[model_id]
                model_info.last_used = datetime.now()
                logger.debug(f"Using cached model: {model_id}")
                return model_info
            
            # Check dependencies
            if not self._ensure_dependencies(model_type):
                raise ModelLoadError(f"Dependencies not available for {model_type}")
            
            # Check cache for model files
            if not force_reload and self.cache.is_cached(model_id):
                logger.info(f"Model {model_id} found in cache")
            else:
                logger.info(f"Downloading model {model_id} to cache")
            
            try:
                # Load model using appropriate handler
                handler = self._model_handlers.get(model_type)
                if not handler:
                    raise ModelLoadError(f"No handler for model type: {model_type}")
                
                start_time = time.time()
                model_object = handler(model_id, **kwargs)
                load_time = time.time() - start_time
                
                # Estimate memory usage
                memory_usage = self._estimate_memory_usage(model_object, model_type)
                
                # Create model info
                model_info = ModelInfo(
                    model_id=model_id,
                    model_type=model_type,
                    model_object=model_object,
                    config=self.config,
                    load_time=datetime.now(),
                    last_used=datetime.now(),
                    memory_usage_gb=memory_usage,
                    device=self.config.get_device_str(),
                    is_quantized=kwargs.get('quantization') != QuantizationType.NONE,
                    quantization_type=kwargs.get('quantization', QuantizationType.NONE)
                )
                
                # Store in registry
                self._loaded_models[model_id] = model_info
                
                # Update cache
                self.cache.mark_used(model_id)
                
                # Track usage stats
                self._track_model_load(model_id, load_time, memory_usage)
                
                logger.info(
                    f"Loaded {model_type.value} model: {model_id} "
                    f"({memory_usage:.1f}GB) in {load_time:.2f}s"
                )
                
                return model_info
                
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                raise ModelLoadError(f"Failed to load model {model_id}: {e}")
    
    def _load_transformer_model(self, model_id: str, **kwargs) -> Any:
        """Load a transformer model with optimization."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Device mapping
            device_map = kwargs.get('device_map', 'auto')
            
            # Quantization settings
            quantization = kwargs.get('quantization', QuantizationType.FP16)
            
            # Load configuration
            model_kwargs = {
                'device_map': device_map,
                'trust_remote_code': kwargs.get('trust_remote_code', False),
                'cache_dir': self.config.cache_dir
            }
            
            # Apply quantization
            if quantization == QuantizationType.FP16:
                model_kwargs['torch_dtype'] = torch.float16
            elif quantization == QuantizationType.INT8:
                model_kwargs['load_in_8bit'] = True
            elif quantization == QuantizationType.INT4:
                model_kwargs['load_in_4bit'] = True
            
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                cache_dir=self.config.cache_dir,
                trust_remote_code=kwargs.get('trust_remote_code', False)
            )
            
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return {'model': model, 'tokenizer': tokenizer}
            
        except Exception as e:
            logger.error(f"Failed to load transformer model {model_id}: {e}")
            raise
    
    def _load_embedding_model(self, model_id: str, **kwargs) -> Any:
        """Load an embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer(
                model_id,
                cache_folder=self.config.cache_dir,
                device=self.config.get_device_str(),
                trust_remote_code=kwargs.get('trust_remote_code', False)
            )
            
            # Set model parameters
            max_seq_length = kwargs.get('max_sequence_length', 512)
            if hasattr(model, 'max_seq_length'):
                model.max_seq_length = max_seq_length
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_id}: {e}")
            raise
    
    def _load_quantized_model(self, model_id: str, **kwargs) -> Any:
        """Load a pre-quantized model."""
        # For now, treat as transformer with quantization
        kwargs['quantization'] = kwargs.get('quantization', QuantizationType.INT8)
        return self._load_transformer_model(model_id, **kwargs)
    
    def _estimate_memory_usage(self, model_object: Any, model_type: ModelType) -> float:
        """Estimate memory usage of loaded model in GB."""
        try:
            if model_type == ModelType.TRANSFORMER:
                # For transformer models, count parameters
                model = model_object.get('model') if isinstance(model_object, dict) else model_object
                if hasattr(model, 'num_parameters'):
                    params = model.num_parameters()
                    # Rough estimate: 4 bytes per parameter for fp32, 2 for fp16
                    return (params * 2) / (1024**3)  # Assuming fp16
                
            elif model_type == ModelType.EMBEDDING:
                # For embedding models, use a rough estimate
                return 0.5  # Most embedding models are < 500MB
            
            # Fallback estimate
            return 1.0
            
        except Exception:
            return 1.0  # Conservative fallback
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get a loaded model by ID."""
        model_info = self._loaded_models.get(model_id)
        if model_info:
            model_info.last_used = datetime.now()
        return model_info
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        if model_id not in self._loaded_models:
            return False
        
        try:
            model_info = self._loaded_models[model_id]
            
            # Clean up model object
            if hasattr(model_info.model_object, 'cpu'):
                model_info.model_object.cpu()
            
            del model_info.model_object
            del self._loaded_models[model_id]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info(f"Unloaded model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}")
            return False
    
    def unload_all_models(self) -> int:
        """Unload all models from memory."""
        count = 0
        model_ids = list(self._loaded_models.keys())
        
        for model_id in model_ids:
            if self.unload_model(model_id):
                count += 1
        
        return count
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage by model."""
        return {
            model_id: info.memory_usage_gb 
            for model_id, info in self._loaded_models.items()
        }
    
    def get_total_memory_usage(self) -> float:
        """Get total memory usage across all loaded models."""
        return sum(info.memory_usage_gb for info in self._loaded_models.values())
    
    def cleanup_unused_models(self, max_age_hours: int = 1) -> int:
        """Clean up models that haven't been used recently."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        count = 0
        
        model_ids_to_remove = [
            model_id for model_id, info in self._loaded_models.items()
            if info.last_used < cutoff_time
        ]
        
        for model_id in model_ids_to_remove:
            if self.unload_model(model_id):
                count += 1
        
        return count
    
    def _track_model_load(self, model_id: str, load_time: float, memory_usage: float):
        """Track model loading statistics."""
        if model_id not in self._usage_stats:
            self._usage_stats[model_id] = {
                'load_count': 0,
                'total_load_time': 0,
                'avg_load_time': 0,
                'memory_usage_gb': memory_usage,
                'first_loaded': datetime.now().isoformat(),
                'last_loaded': datetime.now().isoformat()
            }
        
        stats = self._usage_stats[model_id]
        stats['load_count'] += 1
        stats['total_load_time'] += load_time
        stats['avg_load_time'] = stats['total_load_time'] / stats['load_count']
        stats['last_loaded'] = datetime.now().isoformat()
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        loaded_models = {
            model_id: {
                'model_type': info.model_type.value,
                'device': info.device,
                'memory_usage_gb': info.memory_usage_gb,
                'load_time': info.load_time.isoformat(),
                'last_used': info.last_used.isoformat(),
                'is_quantized': info.is_quantized,
                'quantization_type': info.quantization_type.value if info.quantization_type else None
            }
            for model_id, info in self._loaded_models.items()
        }
        
        return {
            'loaded_models': loaded_models,
            'total_models': len(self._loaded_models),
            'total_memory_gb': self.get_total_memory_usage(),
            'device': self.config.get_device_str(),
            'cache_directory': self.config.cache_dir,
            'usage_statistics': self._usage_stats
        }
    
    def list_available_models(self, model_type: Optional[ModelType] = None) -> List[str]:
        """List available models for a given type."""
        if model_type == ModelType.TRANSFORMER:
            from .model_config import TransformerModelConfig
            config = TransformerModelConfig()
            registry = config.get_model_registry()
            return list(registry.keys())
        
        elif model_type == ModelType.EMBEDDING:
            from .model_config import EmbeddingModelConfig
            config = EmbeddingModelConfig()
            registry = config.get_embedding_registry()
            return list(registry.keys())
        
        else:
            # Return all available models
            transformer_config = TransformerModelConfig()
            embedding_config = EmbeddingModelConfig()
            
            return (
                list(transformer_config.get_model_registry().keys()) +
                list(embedding_config.get_embedding_registry().keys())
            )
    
    def recommend_model(
        self, 
        model_type: ModelType, 
        use_case: str = "general",
        max_memory_gb: Optional[float] = None
    ) -> str:
        """Recommend optimal model based on requirements."""
        if model_type == ModelType.TRANSFORMER:
            config = TransformerModelConfig()
            return config.select_optimal_model(max_memory_gb)
        
        elif model_type == ModelType.EMBEDDING:
            config = EmbeddingModelConfig()
            return config.select_optimal_embedding_model(use_case)
        
        else:
            # Default to embedding for general use
            config = EmbeddingModelConfig()
            return config.select_optimal_embedding_model(use_case)
    
    def cleanup(self):
        """Clean up all resources."""
        self.unload_all_models()
        if self.cache:
            self.cache.cleanup()
        
        logger.info("ModelManager cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()