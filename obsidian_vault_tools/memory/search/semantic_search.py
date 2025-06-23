"""
Semantic Search for Obsidian Vault Tools Memory Service.

Provides vector-based semantic similarity search with support for
context-aware queries, metadata filtering, and relevance explanations.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

try:
    from ..embeddings.embedding_service import EmbeddingService
    from ..embeddings.chroma_store import ChromaStore
    from ..embeddings.config import EmbeddingConfig, DEFAULT_CONFIG
except ImportError:
    from embeddings.embedding_service import EmbeddingService
    from embeddings.chroma_store import ChromaStore
    from embeddings.config import EmbeddingConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result with semantic similarity."""
    document_id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    relevance_explanation: str
    embedding_distance: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'document_id': self.document_id,
            'content': self.content,
            'similarity_score': self.similarity_score,
            'metadata': self.metadata,
            'relevance_explanation': self.relevance_explanation,
            'embedding_distance': self.embedding_distance,
            'rank': self.rank
        }


@dataclass
class SearchResultSet:
    """Collection of search results with metadata."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    embedding_time: float
    retrieval_time: float
    query_embedding: Optional[np.ndarray] = None
    filters_applied: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'query': self.query,
            'results': [r.to_dict() for r in self.results],
            'total_results': self.total_results,
            'search_time': self.search_time,
            'embedding_time': self.embedding_time,
            'retrieval_time': self.retrieval_time,
            'filters_applied': self.filters_applied,
            'timestamp': datetime.now().isoformat()
        }


class SemanticSearch:
    """
    Semantic search engine using vector embeddings.
    
    Features:
    - Vector-based similarity search
    - Context-aware query understanding
    - Metadata filtering and faceting
    - Multi-modal search (text + metadata)
    - Relevance explanation generation
    - Performance optimization
    - Configurable similarity thresholds
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize semantic search with configuration."""
        self.config = config or DEFAULT_CONFIG
        
        # Initialize services
        self.embedding_service = EmbeddingService(self.config)
        self.vector_store = ChromaStore(self.config)
        
        # Search configuration
        self.default_limit = self.config.default_search_limit
        self.similarity_threshold = self.config.similarity_threshold
        
        # Performance tracking
        self._search_stats = {
            'total_searches': 0,
            'total_search_time': 0,
            'avg_search_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Query embedding cache (for repeated queries)
        self._query_cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 100
        
        logger.info("Initialized SemanticSearch with vector embeddings")
    
    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict] = None,
        include_embeddings: bool = False,
        explain_relevance: bool = True
    ) -> SearchResultSet:
        """
        Perform semantic search for similar content.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            filters: Metadata filters to apply
            include_embeddings: Include embeddings in results
            explain_relevance: Generate relevance explanations
            
        Returns:
            SearchResultSet with ranked results
        """
        start_time = time.time()
        
        # Set defaults
        limit = limit or self.default_limit
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        try:
            # Generate query embedding
            embedding_start = time.time()
            query_embedding = self._get_query_embedding(query)
            embedding_time = time.time() - embedding_start
            
            # Search vector store
            retrieval_start = time.time()
            raw_results = self.vector_store.query_similar(
                query_embedding=query_embedding,
                n_results=limit * 2,  # Get more to account for filtering
                where=filters,
                include=["documents", "distances", "metadatas", "embeddings"] if include_embeddings else ["documents", "distances", "metadatas"]
            )
            retrieval_time = time.time() - retrieval_start
            
            # Process results
            processed_results = self._process_search_results(
                query=query,
                query_embedding=query_embedding,
                raw_results=raw_results,
                similarity_threshold=similarity_threshold,
                limit=limit,
                explain_relevance=explain_relevance
            )
            
            total_time = time.time() - start_time
            
            # Update statistics
            self._update_search_stats(total_time)
            
            result_set = SearchResultSet(
                query=query,
                results=processed_results,
                total_results=len(processed_results),
                search_time=total_time,
                embedding_time=embedding_time,
                retrieval_time=retrieval_time,
                query_embedding=query_embedding if include_embeddings else None,
                filters_applied=filters
            )
            
            logger.debug(f"Semantic search for '{query}' returned {len(processed_results)} results in {total_time:.3f}s")
            
            return result_set
            
        except Exception as e:
            logger.error(f"Semantic search failed for query '{query}': {e}")
            raise
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for query with caching."""
        # Check cache first
        if query in self._query_cache:
            self._search_stats['cache_hits'] += 1
            return self._query_cache[query]
        
        # Generate new embedding
        embedding = self.embedding_service.encode_text(query)
        
        # Cache management
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[query] = embedding
        self._search_stats['cache_misses'] += 1
        
        return embedding
    
    def _process_search_results(
        self,
        query: str,
        query_embedding: np.ndarray,
        raw_results: Dict,
        similarity_threshold: float,
        limit: int,
        explain_relevance: bool
    ) -> List[SearchResult]:
        """Process raw search results into SearchResult objects."""
        results = []
        
        # Extract results data
        ids = raw_results.get('ids', [[]])[0]
        documents = raw_results.get('documents', [[]])[0]
        distances = raw_results.get('distances', [[]])[0]
        metadatas = raw_results.get('metadatas', [[]])[0] or [{}] * len(ids)
        
        for i, (doc_id, content, distance, metadata) in enumerate(zip(ids, documents, distances, metadatas)):
            # Convert distance to similarity score
            # ChromaDB typically returns cosine distance, so similarity = 1 - distance
            similarity_score = max(0, 1 - distance)
            
            # Apply similarity threshold
            if similarity_score < similarity_threshold:
                continue
            
            # Generate relevance explanation
            relevance_explanation = ""
            if explain_relevance:
                relevance_explanation = self._generate_relevance_explanation(
                    query=query,
                    content=content,
                    similarity_score=similarity_score,
                    metadata=metadata
                )
            
            result = SearchResult(
                document_id=doc_id,
                content=content,
                similarity_score=similarity_score,
                metadata=metadata,
                relevance_explanation=relevance_explanation,
                embedding_distance=distance,
                rank=i + 1
            )
            
            results.append(result)
            
            # Stop if we have enough results
            if len(results) >= limit:
                break
        
        return results
    
    def _generate_relevance_explanation(
        self,
        query: str,
        content: str,
        similarity_score: float,
        metadata: Dict
    ) -> str:
        """Generate human-readable explanation of relevance."""
        explanations = []
        
        # Similarity score explanation
        if similarity_score >= 0.9:
            explanations.append("Very high semantic similarity")
        elif similarity_score >= 0.8:
            explanations.append("High semantic similarity")
        elif similarity_score >= 0.7:
            explanations.append("Good semantic similarity")
        elif similarity_score >= 0.6:
            explanations.append("Moderate semantic similarity")
        else:
            explanations.append("Lower semantic similarity")
        
        # Check for keyword matches
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        common_words = query_words & content_words
        
        if common_words:
            explanations.append(f"Shares keywords: {', '.join(list(common_words)[:3])}")
        
        # Check metadata relevance
        if metadata:
            if metadata.get('type') == 'note' and 'note' in query.lower():
                explanations.append("Matches content type")
            
            if 'tags' in metadata and isinstance(metadata['tags'], list):
                query_lower = query.lower()
                matching_tags = [tag for tag in metadata['tags'] if tag.lower() in query_lower]
                if matching_tags:
                    explanations.append(f"Relevant tags: {', '.join(matching_tags[:2])}")
        
        return "; ".join(explanations) if explanations else "Semantic similarity match"
    
    def search_similar_to_document(
        self,
        document_id: str,
        limit: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict] = None
    ) -> SearchResultSet:
        """
        Find documents similar to a specific document.
        
        Args:
            document_id: ID of reference document
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            filters: Metadata filters to apply
            
        Returns:
            SearchResultSet with similar documents
        """
        try:
            # Get the reference document
            ref_doc = self.vector_store.get_by_ids([document_id], include=["documents", "embeddings", "metadatas"])
            
            if not ref_doc.get('embeddings') or not ref_doc['embeddings'][0]:
                raise ValueError(f"No embedding found for document {document_id}")
            
            ref_embedding = np.array(ref_doc['embeddings'][0])
            ref_content = ref_doc.get('documents', [''])[0]
            
            # Use the document content as the "query" for explanation purposes
            query = f"Similar to: {ref_content[:100]}..."
            
            # Search using the reference embedding
            raw_results = self.vector_store.query_similar(
                query_embedding=ref_embedding,
                n_results=(limit or self.default_limit) + 1,  # +1 to account for self-match
                where=filters,
                include=["documents", "distances", "metadatas"]
            )
            
            # Process results (excluding the reference document itself)
            processed_results = self._process_search_results(
                query=query,
                query_embedding=ref_embedding,
                raw_results=raw_results,
                similarity_threshold=similarity_threshold or self.similarity_threshold,
                limit=limit or self.default_limit,
                explain_relevance=True
            )
            
            # Remove self-reference if present
            processed_results = [r for r in processed_results if r.document_id != document_id]
            
            return SearchResultSet(
                query=query,
                results=processed_results,
                total_results=len(processed_results),
                search_time=0,  # Not tracking timing for this method
                embedding_time=0,
                retrieval_time=0,
                filters_applied=filters
            )
            
        except Exception as e:
            logger.error(f"Similar document search failed for {document_id}: {e}")
            raise
    
    def multi_query_search(
        self,
        queries: List[str],
        limit: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict] = None,
        aggregation_method: str = "average"
    ) -> SearchResultSet:
        """
        Search using multiple queries and aggregate results.
        
        Args:
            queries: List of query strings
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            filters: Metadata filters to apply
            aggregation_method: How to combine scores ("average", "max", "weighted")
            
        Returns:
            SearchResultSet with aggregated results
        """
        if not queries:
            raise ValueError("At least one query is required")
        
        # Generate embeddings for all queries
        query_embeddings = []
        for query in queries:
            embedding = self._get_query_embedding(query)
            query_embeddings.append(embedding)
        
        # Combine embeddings based on aggregation method
        if aggregation_method == "average":
            combined_embedding = np.mean(query_embeddings, axis=0)
        elif aggregation_method == "max":
            combined_embedding = np.max(query_embeddings, axis=0)
        elif aggregation_method == "weighted":
            # Simple weighting - could be enhanced with query importance scores
            weights = [1.0 / len(queries)] * len(queries)
            combined_embedding = np.average(query_embeddings, axis=0, weights=weights)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
        
        # Search with combined embedding
        raw_results = self.vector_store.query_similar(
            query_embedding=combined_embedding,
            n_results=limit or self.default_limit,
            where=filters,
            include=["documents", "distances", "metadatas"]
        )
        
        # Process results
        combined_query = " | ".join(queries)
        processed_results = self._process_search_results(
            query=combined_query,
            query_embedding=combined_embedding,
            raw_results=raw_results,
            similarity_threshold=similarity_threshold or self.similarity_threshold,
            limit=limit or self.default_limit,
            explain_relevance=True
        )
        
        return SearchResultSet(
            query=combined_query,
            results=processed_results,
            total_results=len(processed_results),
            search_time=0,
            embedding_time=0,
            retrieval_time=0,
            filters_applied=filters
        )
    
    def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        Args:
            partial_query: Incomplete query text
            limit: Number of suggestions
            
        Returns:
            List of suggested query completions
        """
        # This is a simplified implementation
        # In practice, you might maintain a query log or use more sophisticated methods
        
        suggestions = []
        
        # Add common completions based on partial query
        common_completions = {
            "how": ["how to", "how does", "how can"],
            "what": ["what is", "what are", "what does"],
            "why": ["why does", "why is", "why should"],
            "when": ["when to", "when is", "when should"],
            "where": ["where is", "where to", "where can"]
        }
        
        partial_lower = partial_query.lower().strip()
        
        for prefix, completions in common_completions.items():
            if partial_lower.startswith(prefix):
                suggestions.extend([comp + partial_query[len(prefix):] for comp in completions])
        
        # Add domain-specific suggestions
        if any(word in partial_lower for word in ['note', 'document', 'file']):
            suggestions.extend([
                partial_query + " creation",
                partial_query + " management", 
                partial_query + " organization"
            ])
        
        return suggestions[:limit]
    
    def _update_search_stats(self, search_time: float):
        """Update search performance statistics."""
        self._search_stats['total_searches'] += 1
        self._search_stats['total_search_time'] += search_time
        self._search_stats['avg_search_time'] = (
            self._search_stats['total_search_time'] / self._search_stats['total_searches']
        )
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search performance statistics."""
        return {
            **self._search_stats,
            'query_cache_size': len(self._query_cache),
            'similarity_threshold': self.similarity_threshold,
            'default_limit': self.default_limit,
            'vector_store_stats': self.vector_store.get_collection_stats()
        }
    
    def clear_cache(self):
        """Clear query embedding cache."""
        self._query_cache.clear()
        logger.info("Cleared semantic search query cache")
    
    def cleanup(self):
        """Clean up search resources."""
        self.clear_cache()
        if self.embedding_service:
            self.embedding_service.cleanup()
        if self.vector_store:
            self.vector_store.cleanup()
        
        logger.info("SemanticSearch cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()