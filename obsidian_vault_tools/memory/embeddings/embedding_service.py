"""
Embedding service for generating vector representations of text.

Provides local sentence-transformers integration with model caching,
batch processing, and fallback strategies.
"""

import logging
import time
from typing import List, Optional, Union, Dict, Tuple
from pathlib import Path
import numpy as np

try:
    from .config import EmbeddingConfig, DEFAULT_CONFIG
except ImportError:
    from config import EmbeddingConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Local embedding service using sentence-transformers.
    
    Features:
    - Multiple model support with fallbacks
    - Batch processing for efficiency
    - Model caching and lazy loading
    - CPU/GPU optimization
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize embedding service with configuration."""
        self.config = config or DEFAULT_CONFIG
        self.config.optimize_for_device()
        
        self._primary_model = None
        self._fallback_model = None
        self._model_loaded = False
        
        # Performance tracking
        self._embedding_times = []
        self._batch_sizes = []
        
        logger.info(f"Initialized EmbeddingService with device: {self.config.device}")
    
    def _ensure_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import sentence_transformers
            import torch
            return True
        except ImportError as e:
            logger.error(f"Required dependencies not available: {e}")
            return False
    
    def _load_model(self, model_name: str, is_fallback: bool = False):
        """Load a sentence transformer model with error handling."""
        if not self._ensure_dependencies():
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading {'fallback ' if is_fallback else ''}model: {model_name}")
            
            model = SentenceTransformer(
                model_name,
                cache_folder=self.config.model_cache_dir,
                device=self.config.device
            )
            
            # Set model parameters
            if hasattr(model, 'max_seq_length'):
                model.max_seq_length = self.config.max_sequence_length
            
            logger.info(f"Successfully loaded model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            if not is_fallback and self.config.enable_fallback:
                logger.info(f"Attempting fallback model: {self.config.fallback_model}")
                return self._load_model(self.config.fallback_model, is_fallback=True)
            raise
    
    def _get_model(self):
        """Get the active model, loading if necessary."""
        if not self.config.lazy_loading or not self._model_loaded:
            if self._primary_model is None:
                try:
                    self._primary_model = self._load_model(self.config.primary_model)
                    self._model_loaded = True
                    return self._primary_model
                except Exception:
                    if self.config.enable_fallback and self._fallback_model is None:
                        try:
                            self._fallback_model = self._load_model(
                                self.config.fallback_model, 
                                is_fallback=True
                            )
                            self._model_loaded = True
                            return self._fallback_model
                        except Exception as e:
                            logger.error(f"Failed to load any model: {e}")
                            raise
                    raise
        
        return self._primary_model or self._fallback_model
    
    def encode_text(
        self, 
        texts: Union[str, List[str]], 
        batch_size: Optional[int] = None,
        normalize_embeddings: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for input text(s).
        
        Args:
            texts: Single text string or list of texts
            batch_size: Override default batch size
            normalize_embeddings: Whether to normalize vectors
            show_progress: Show progress bar for large batches
            
        Returns:
            numpy array of embeddings
        """
        start_time = time.time()
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        # Use configured or provided batch size
        effective_batch_size = batch_size or self.config.batch_size
        
        try:
            model = self._get_model()
            
            # Generate embeddings
            embeddings = model.encode(
                texts,
                batch_size=effective_batch_size,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # Track performance
            elapsed_time = time.time() - start_time
            self._embedding_times.append(elapsed_time)
            self._batch_sizes.append(len(texts))
            
            logger.debug(
                f"Generated {len(texts)} embeddings in {elapsed_time:.2f}s "
                f"(batch_size={effective_batch_size})"
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def encode_batch(
        self, 
        text_batches: List[List[str]], 
        **kwargs
    ) -> List[np.ndarray]:
        """
        Process multiple batches of texts efficiently.
        
        Args:
            text_batches: List of text lists to process
            **kwargs: Additional arguments for encode_text
            
        Returns:
            List of embedding arrays
        """
        results = []
        
        for i, batch in enumerate(text_batches):
            logger.debug(f"Processing batch {i+1}/{len(text_batches)}")
            embeddings = self.encode_text(batch, **kwargs)
            results.append(embeddings)
        
        return results
    
    def compute_similarity(
        self, 
        embeddings1: np.ndarray, 
        embeddings2: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute similarity between embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings  
            metric: Similarity metric ("cosine", "dot", "euclidean")
            
        Returns:
            Similarity matrix
        """
        try:
            if metric == "cosine":
                # Cosine similarity
                norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
                norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
                normalized1 = embeddings1 / norm1
                normalized2 = embeddings2 / norm2
                return np.dot(normalized1, normalized2.T)
                
            elif metric == "dot":
                # Dot product
                return np.dot(embeddings1, embeddings2.T)
                
            elif metric == "euclidean":
                # Negative euclidean distance (higher = more similar)
                distances = np.cdist(embeddings1, embeddings2, metric='euclidean')
                return -distances
                
            else:
                raise ValueError(f"Unsupported similarity metric: {metric}")
                
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            raise
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray, 
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to a query.
        
        Args:
            query_embedding: Query vector
            candidate_embeddings: Candidate vectors
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = self.compute_similarity(
            query_embedding, 
            candidate_embeddings,
            metric="cosine"
        )[0]  # Get first row since query is single vector
        
        # Apply threshold if specified
        if threshold is not None:
            valid_indices = np.where(similarities >= threshold)[0]
            similarities = similarities[valid_indices]
        else:
            valid_indices = np.arange(len(similarities))
        
        # Get top-k results
        if len(similarities) == 0:
            return []
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            (valid_indices[idx], similarities[idx]) 
            for idx in top_indices
        ]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from the current model."""
        try:
            model = self._get_model()
            # Test with a simple sentence
            test_embedding = model.encode(["test"], convert_to_numpy=True)
            return test_embedding.shape[1]
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {e}")
            # Return common dimensions as fallbacks
            if "MiniLM-L6" in self.config.primary_model:
                return 384
            elif "mpnet-base" in self.config.primary_model:
                return 768
            else:
                return 384  # Most common for MiniLM models
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for the service."""
        if not self._embedding_times:
            return {}
        
        avg_time = np.mean(self._embedding_times)
        avg_batch_size = np.mean(self._batch_sizes)
        total_processed = sum(self._batch_sizes)
        
        return {
            'total_embeddings_generated': total_processed,
            'average_time_per_batch': avg_time,
            'average_batch_size': avg_batch_size,
            'embeddings_per_second': avg_batch_size / avg_time if avg_time > 0 else 0,
            'current_model': (
                self.config.primary_model if self._primary_model else 
                self.config.fallback_model if self._fallback_model else 
                "No model loaded"
            ),
            'device': self.config.device
        }
    
    def cleanup(self):
        """Clean up resources and models."""
        if self._primary_model is not None:
            del self._primary_model
            self._primary_model = None
        
        if self._fallback_model is not None:
            del self._fallback_model
            self._fallback_model = None
        
        self._model_loaded = False
        
        # Clear performance tracking
        self._embedding_times.clear()
        self._batch_sizes.clear()
        
        logger.info("EmbeddingService cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()