"""
Configuration management for embedding service.

Provides model selection, performance tuning, and fallback strategies
for the embedding and vector storage system.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service and vector storage."""
    
    # Model Configuration
    primary_model: str = "multi-qa-MiniLM-L6-cos-v1"
    fallback_model: str = "all-MiniLM-L6-v2"
    model_cache_dir: Optional[str] = None
    
    # Performance Settings
    batch_size: int = 32
    max_sequence_length: int = 512
    device: str = "cpu"  # Can be "cuda" if available
    enable_model_caching: bool = True
    lazy_loading: bool = True
    
    # ChromaDB Settings
    persist_directory: Optional[str] = None
    collection_name: str = "obsidian_vault_embeddings"
    distance_metric: str = "cosine"  # "cosine", "l2", "ip"
    
    # Search Settings
    default_search_limit: int = 10
    similarity_threshold: float = 0.7
    
    # Fallback Configuration
    enable_fallback: bool = True
    fallback_timeout: float = 30.0
    max_retries: int = 3
    
    # Storage Settings
    metadata_fields: List[str] = field(default_factory=lambda: [
        "source", "type", "timestamp", "vault_path", "tags"
    ])
    
    def __post_init__(self):
        """Initialize default paths and validate configuration."""
        if self.model_cache_dir is None:
            self.model_cache_dir = str(
                Path.home() / ".cache" / "obsidian_vault_tools" / "models"
            )
        
        if self.persist_directory is None:
            self.persist_directory = str(
                Path.home() / ".local" / "share" / "obsidian_vault_tools" / "embeddings"
            )
        
        # Ensure directories exist
        os.makedirs(self.model_cache_dir, exist_ok=True)
        os.makedirs(self.persist_directory, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'EmbeddingConfig':
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'primary_model': self.primary_model,
            'fallback_model': self.fallback_model,
            'model_cache_dir': self.model_cache_dir,
            'batch_size': self.batch_size,
            'max_sequence_length': self.max_sequence_length,
            'device': self.device,
            'enable_model_caching': self.enable_model_caching,
            'lazy_loading': self.lazy_loading,
            'persist_directory': self.persist_directory,
            'collection_name': self.collection_name,
            'distance_metric': self.distance_metric,
            'default_search_limit': self.default_search_limit,
            'similarity_threshold': self.similarity_threshold,
            'enable_fallback': self.enable_fallback,
            'fallback_timeout': self.fallback_timeout,
            'max_retries': self.max_retries,
            'metadata_fields': self.metadata_fields
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available embedding models."""
        return [
            "multi-qa-MiniLM-L6-cos-v1",  # Optimized for semantic search
            "all-MiniLM-L6-v2",           # General purpose
            "all-mpnet-base-v2",          # High quality, larger
            "paraphrase-MiniLM-L6-v2",    # Paraphrase detection
            "all-distilroberta-v1"        # DistilRoBERTa based
        ]
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        try:
            # Check batch size
            if self.batch_size <= 0:
                raise ValueError("batch_size must be positive")
            
            # Check sequence length
            if self.max_sequence_length <= 0:
                raise ValueError("max_sequence_length must be positive")
            
            # Check distance metric
            valid_metrics = ["cosine", "l2", "ip"]
            if self.distance_metric not in valid_metrics:
                raise ValueError(f"distance_metric must be one of {valid_metrics}")
            
            # Check search parameters
            if self.default_search_limit <= 0:
                raise ValueError("default_search_limit must be positive")
            
            if not 0 <= self.similarity_threshold <= 1:
                raise ValueError("similarity_threshold must be between 0 and 1")
            
            return True
            
        except ValueError as e:
            print(f"Configuration validation error: {e}")
            return False
    
    def optimize_for_device(self) -> None:
        """Optimize configuration based on available hardware."""
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                # Increase batch size for GPU
                self.batch_size = min(self.batch_size * 2, 128)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
                self.batch_size = min(self.batch_size * 1.5, 64)
        except ImportError:
            # Torch not available, stay with CPU
            pass
    
    def get_chroma_settings(self) -> Dict:
        """Get ChromaDB-specific settings."""
        return {
            'persist_directory': self.persist_directory,
            'collection_name': self.collection_name,
            'metadata': {
                'hnsw:space': self.distance_metric,
                'hnsw:construction_ef': 200,
                'hnsw:M': 16
            }
        }


# Default configuration instance
DEFAULT_CONFIG = EmbeddingConfig()