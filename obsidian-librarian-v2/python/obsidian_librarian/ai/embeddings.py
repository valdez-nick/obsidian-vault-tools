"""
Embedding service for semantic search and similarity.

Handles text embedding generation using various providers with caching
and batch processing for optimal performance.
"""

import asyncio
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import structlog

from .models import AIModelManager, AIProvider, ModelCapability
from ..database.base import DatabaseManager

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """Service for generating and managing text embeddings."""
    
    def __init__(
        self, 
        ai_manager: AIModelManager,
        database_manager: Optional[DatabaseManager] = None
    ):
        self.ai_manager = ai_manager
        self.db_manager = database_manager
        self._embedding_cache: Dict[str, List[float]] = {}
        self._batch_size = 100
        self._lock = asyncio.Lock()
    
    async def embed_text(
        self, 
        text: str, 
        model: Optional[str] = None,
        use_cache: bool = True
    ) -> List[float]:
        """Generate embedding for a single text."""
        # Create cache key
        cache_key = self._create_cache_key(text, model)
        
        # Check cache first
        if use_cache:
            cached = await self._get_cached_embedding(cache_key)
            if cached:
                return cached
        
        # Generate embedding
        embedding = await self._generate_embedding(text, model)
        
        # Cache result
        if use_cache and embedding:
            await self._cache_embedding(cache_key, embedding)
        
        return embedding
    
    async def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        use_cache: bool = True,
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        batch_size = batch_size or self._batch_size
        
        # Check cache for all texts
        results = []
        uncached_texts = []
        uncached_indices = []
        
        if use_cache:
            for i, text in enumerate(texts):
                cache_key = self._create_cache_key(text, model)
                cached = await self._get_cached_embedding(cache_key)
                if cached:
                    results.append(cached)
                else:
                    results.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            results = [None] * len(texts)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            logger.debug("Generating embeddings", count=len(uncached_texts))
            
            # Process in batches
            embeddings = []
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]
                batch_embeddings = await self._generate_batch_embeddings(batch, model)
                embeddings.extend(batch_embeddings)
            
            # Insert results and cache
            for i, embedding in enumerate(embeddings):
                original_index = uncached_indices[i]
                results[original_index] = embedding
                
                if use_cache and embedding:
                    cache_key = self._create_cache_key(uncached_texts[i], model)
                    await self._cache_embedding(cache_key, embedding)
        
        return results
    
    async def _generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding using available providers."""
        # Get best embedding model
        if not model:
            model_info = await self.ai_manager.get_best_model(ModelCapability.EMBEDDING)
            if not model_info:
                raise RuntimeError("No embedding models available")
            model = model_info.name
        
        # Determine provider and generate
        if model in self.ai_manager._models:
            provider = self.ai_manager._models[model].provider
            
            if provider == AIProvider.OPENAI:
                return await self._generate_openai_embedding(text, model)
            elif provider == AIProvider.OLLAMA:
                return await self._generate_ollama_embedding(text, model)
            elif provider == AIProvider.HUGGINGFACE:
                return await self._generate_hf_embedding(text, model)
        
        raise ValueError(f"Unsupported model for embedding: {model}")
    
    async def _generate_batch_embeddings(
        self, 
        texts: List[str], 
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if not model:
            model_info = await self.ai_manager.get_best_model(ModelCapability.EMBEDDING)
            if not model_info:
                raise RuntimeError("No embedding models available")
            model = model_info.name
        
        provider = self.ai_manager._models[model].provider
        
        if provider == AIProvider.OPENAI:
            return await self._generate_openai_batch_embeddings(texts, model)
        elif provider == AIProvider.OLLAMA:
            return await self._generate_ollama_batch_embeddings(texts, model)
        else:
            # Fallback to individual embeddings
            embeddings = []
            for text in texts:
                embedding = await self._generate_embedding(text, model)
                embeddings.append(embedding)
            return embeddings
    
    async def _generate_openai_embedding(self, text: str, model: str) -> List[float]:
        """Generate embedding using OpenAI."""
        client = self.ai_manager.get_client(AIProvider.OPENAI)
        if not client:
            raise RuntimeError("OpenAI client not available")
        
        try:
            # Get rate limiter
            rate_limiter = self.ai_manager.get_rate_limiter(AIProvider.OPENAI)
            async with rate_limiter:
                response = await client.embeddings.create(
                    model=model,
                    input=text,
                    encoding_format="float"
                )
                
                return response.data[0].embedding
                
        except Exception as e:
            logger.error("OpenAI embedding failed", model=model, error=str(e))
            raise
    
    async def _generate_openai_batch_embeddings(
        self, 
        texts: List[str], 
        model: str
    ) -> List[List[float]]:
        """Generate batch embeddings using OpenAI."""
        client = self.ai_manager.get_client(AIProvider.OPENAI)
        if not client:
            raise RuntimeError("OpenAI client not available")
        
        try:
            rate_limiter = self.ai_manager.get_rate_limiter(AIProvider.OPENAI)
            async with rate_limiter:
                response = await client.embeddings.create(
                    model=model,
                    input=texts,
                    encoding_format="float"
                )
                
                return [item.embedding for item in response.data]
                
        except Exception as e:
            logger.error("OpenAI batch embedding failed", model=model, error=str(e))
            raise
    
    async def _generate_ollama_embedding(self, text: str, model: str) -> List[float]:
        """Generate embedding using Ollama."""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.ai_manager.config.ollama_base_url}/api/embeddings"
                payload = {
                    "model": model,
                    "prompt": text
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["embedding"]
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Ollama embedding failed: {error_text}")
                        
        except Exception as e:
            logger.error("Ollama embedding failed", model=model, error=str(e))
            raise
    
    async def _generate_ollama_batch_embeddings(
        self, 
        texts: List[str], 
        model: str
    ) -> List[List[float]]:
        """Generate batch embeddings using Ollama."""
        # Ollama doesn't support batch embeddings, so process individually
        embeddings = []
        for text in texts:
            embedding = await self._generate_ollama_embedding(text, model)
            embeddings.append(embedding)
        return embeddings
    
    async def _generate_hf_embedding(self, text: str, model: str) -> List[float]:
        """Generate embedding using Hugging Face transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load model (this should be cached)
            sentence_model = SentenceTransformer(model)
            
            # Generate embedding
            embedding = sentence_model.encode([text])[0]
            return embedding.tolist()
            
        except Exception as e:
            logger.error("Hugging Face embedding failed", model=model, error=str(e))
            raise
    
    def _create_cache_key(self, text: str, model: Optional[str] = None) -> str:
        """Create cache key for text and model."""
        # Use text hash to avoid storing long text as keys
        text_hash = hashlib.md5(text.encode()).hexdigest()
        model_part = model or "default"
        return f"embed:{model_part}:{text_hash}"
    
    async def _get_cached_embedding(self, cache_key: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        # Check in-memory cache first
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Check database cache
        if self.db_manager and self.db_manager.cache:
            try:
                cached = await self.db_manager.cache.get(cache_key)
                if cached:
                    # Store in memory cache for faster access
                    self._embedding_cache[cache_key] = cached
                    return cached
            except Exception as e:
                logger.warning("Failed to get cached embedding", error=str(e))
        
        return None
    
    async def _cache_embedding(self, cache_key: str, embedding: List[float]) -> None:
        """Cache embedding."""
        # Store in memory cache
        self._embedding_cache[cache_key] = embedding
        
        # Store in database cache
        if self.db_manager and self.db_manager.cache:
            try:
                await self.db_manager.cache.set(
                    cache_key, 
                    embedding, 
                    ttl=86400  # 24 hours
                )
            except Exception as e:
                logger.warning("Failed to cache embedding", error=str(e))
    
    async def compute_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float],
        metric: str = "cosine"
    ) -> float:
        """Compute similarity between two embeddings."""
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same dimension")
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        elif metric == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(vec1 - vec2)
            return 1.0 / (1.0 + distance)
        
        elif metric == "dot":
            # Dot product
            return np.dot(vec1, vec2)
        
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    async def find_similar_embeddings(
        self, 
        query_embedding: List[float],
        candidate_embeddings: List[Tuple[str, List[float]]],
        top_k: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """Find most similar embeddings from candidates."""
        similarities = []
        
        for item_id, embedding in candidate_embeddings:
            similarity = await self.compute_similarity(query_embedding, embedding)
            if similarity >= threshold:
                similarities.append((item_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    async def cluster_embeddings(
        self, 
        embeddings: List[Tuple[str, List[float]]],
        num_clusters: int = 10,
        method: str = "kmeans"
    ) -> Dict[int, List[str]]:
        """Cluster embeddings into groups."""
        if len(embeddings) < num_clusters:
            # Not enough data for clustering
            return {0: [item_id for item_id, _ in embeddings]}
        
        try:
            from sklearn.cluster import KMeans, DBSCAN
            
            # Extract embeddings and IDs
            ids = [item_id for item_id, _ in embeddings]
            vectors = np.array([emb for _, emb in embeddings])
            
            if method == "kmeans":
                clusterer = KMeans(n_clusters=num_clusters, random_state=42)
                labels = clusterer.fit_predict(vectors)
            elif method == "dbscan":
                clusterer = DBSCAN(eps=0.5, min_samples=2)
                labels = clusterer.fit_predict(vectors)
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            # Group by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(ids[i])
            
            return clusters
            
        except ImportError:
            logger.warning("scikit-learn not available for clustering")
            return {0: [item_id for item_id, _ in embeddings]}
        except Exception as e:
            logger.error("Clustering failed", error=str(e))
            return {0: [item_id for item_id, _ in embeddings]}
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics."""
        stats = {
            "cache_size": len(self._embedding_cache),
            "providers_available": len(self.ai_manager._clients),
        }
        
        # Add model-specific stats
        embedding_models = await self.ai_manager.get_available_models(ModelCapability.EMBEDDING)
        stats["available_models"] = [model.name for model in embedding_models]
        stats["model_count"] = len(embedding_models)
        
        return stats
    
    async def clear_cache(self) -> None:
        """Clear embedding cache."""
        async with self._lock:
            self._embedding_cache.clear()
            
            if self.db_manager and self.db_manager.cache:
                try:
                    await self.db_manager.cache.clear_pattern("embed:*")
                except Exception as e:
                    logger.warning("Failed to clear database cache", error=str(e))
    
    async def warm_cache(self, texts: List[str], model: Optional[str] = None) -> None:
        """Pre-generate embeddings for a list of texts."""
        logger.info("Warming embedding cache", text_count=len(texts))
        
        # Generate embeddings in batches
        await self.embed_batch(texts, model=model, use_cache=True)
        
        logger.info("Embedding cache warmed", cached_count=len(self._embedding_cache))