"""
Embeddings module for Obsidian Vault Tools Memory Service.

This module provides vector embeddings and similarity search capabilities
using local models and ChromaDB storage.
"""

from .embedding_service import EmbeddingService
from .chroma_store import ChromaStore
from .config import EmbeddingConfig

__all__ = [
    'EmbeddingService',
    'ChromaStore', 
    'EmbeddingConfig'
]