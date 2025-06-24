"""
Hybrid Search for Obsidian Vault Tools Memory Service.

Combines semantic vector search with traditional text search (TF-IDF, BM25)
for optimal retrieval performance across different query types.
"""

import logging
import time
import re
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import math
import numpy as np

try:
    from .semantic_search import SemanticSearch, SearchResult, SearchResultSet
    from .similarity import SimilarityCalculator, SimilarityMetric
    from ..embeddings.config import EmbeddingConfig, DEFAULT_CONFIG
except ImportError:
    from semantic_search import SemanticSearch, SearchResult, SearchResultSet
    from similarity import SimilarityCalculator, SimilarityMetric
    from embeddings.config import EmbeddingConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Hybrid search result combining semantic and text search."""
    document_id: str
    content: str
    semantic_score: float
    text_score: float
    combined_score: float
    metadata: Dict[str, Any]
    semantic_rank: int
    text_rank: int
    final_rank: int
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'document_id': self.document_id,
            'content': self.content,
            'semantic_score': self.semantic_score,
            'text_score': self.text_score,
            'combined_score': self.combined_score,
            'metadata': self.metadata,
            'semantic_rank': self.semantic_rank,
            'text_rank': self.text_rank,
            'final_rank': self.final_rank,
            'explanation': self.explanation
        }


class TextSearchEngine:
    """Traditional text search engine with TF-IDF and BM25 scoring."""
    
    def __init__(self):
        """Initialize text search engine."""
        self.documents: Dict[str, str] = {}
        self.document_metadata: Dict[str, Dict] = {}
        self.term_frequencies: Dict[str, Dict[str, int]] = {}
        self.document_frequencies: Dict[str, int] = {}
        self.document_lengths: Dict[str, int] = {}
        self.avg_document_length = 0.0
        self.total_documents = 0
        
        # BM25 parameters
        self.k1 = 1.2  # Term frequency saturation parameter
        self.b = 0.75  # Length normalization parameter
        
        logger.debug("Initialized TextSearchEngine")
    
    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None):
        """Add document to text search index."""
        self.documents[doc_id] = content
        self.document_metadata[doc_id] = metadata or {}
        
        # Tokenize and calculate term frequencies
        tokens = self._tokenize(content)
        term_freq = Counter(tokens)
        self.term_frequencies[doc_id] = dict(term_freq)
        self.document_lengths[doc_id] = len(tokens)
        
        # Update document frequencies
        unique_terms = set(tokens)
        for term in unique_terms:
            self.document_frequencies[term] = self.document_frequencies.get(term, 0) + 1
        
        # Update statistics
        self.total_documents = len(self.documents)
        self.avg_document_length = sum(self.document_lengths.values()) / self.total_documents
    
    def remove_document(self, doc_id: str):
        """Remove document from text search index."""
        if doc_id not in self.documents:
            return
        
        # Update document frequencies
        tokens = self._tokenize(self.documents[doc_id])
        unique_terms = set(tokens)
        for term in unique_terms:
            self.document_frequencies[term] -= 1
            if self.document_frequencies[term] <= 0:
                del self.document_frequencies[term]
        
        # Remove document data
        del self.documents[doc_id]
        del self.document_metadata[doc_id]
        del self.term_frequencies[doc_id]
        del self.document_lengths[doc_id]
        
        # Update statistics
        self.total_documents = len(self.documents)
        if self.total_documents > 0:
            self.avg_document_length = sum(self.document_lengths.values()) / self.total_documents
        else:
            self.avg_document_length = 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        # Simple tokenization - could be enhanced with stemming, stopword removal, etc.
        text = text.lower()
        # Keep alphanumeric and some special characters
        text = re.sub(r'[^a-z0-9\s\-_]', ' ', text)
        tokens = text.split()
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 1]
        return tokens
    
    def search_tfidf(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Search using TF-IDF scoring."""
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        scores = {}
        
        for doc_id in self.documents:
            score = 0.0
            doc_tf = self.term_frequencies[doc_id]
            doc_length = self.document_lengths[doc_id]
            
            for term in query_terms:
                if term in doc_tf:
                    # Term frequency
                    tf = doc_tf[term] / doc_length
                    
                    # Inverse document frequency
                    df = self.document_frequencies.get(term, 0)
                    if df > 0:
                        idf = math.log(self.total_documents / df)
                        score += tf * idf
            
            if score > 0:
                scores[doc_id] = score
        
        # Sort by score and return top results
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]
    
    def search_bm25(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25 scoring."""
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        scores = {}
        
        for doc_id in self.documents:
            score = 0.0
            doc_tf = self.term_frequencies[doc_id]
            doc_length = self.document_lengths[doc_id]
            
            for term in query_terms:
                if term in doc_tf:
                    # Term frequency in document
                    tf = doc_tf[term]
                    
                    # Document frequency
                    df = self.document_frequencies.get(term, 0)
                    if df > 0:
                        # IDF component
                        idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
                        
                        # BM25 term score
                        term_score = idf * (tf * (self.k1 + 1)) / (
                            tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_document_length))
                        )
                        score += term_score
            
            if score > 0:
                scores[doc_id] = score
        
        # Sort by score and return top results
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get text search index statistics."""
        return {
            'total_documents': self.total_documents,
            'total_terms': len(self.document_frequencies),
            'avg_document_length': self.avg_document_length,
            'total_tokens': sum(self.document_lengths.values())
        }


class HybridSearch:
    """
    Hybrid search engine combining semantic and text search.
    
    Features:
    - Semantic search via vector embeddings
    - Text search via TF-IDF and BM25
    - Intelligent score fusion strategies
    - Query type detection and routing
    - Performance optimization
    - Configurable weighting schemes
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize hybrid search engine."""
        self.config = config or DEFAULT_CONFIG
        
        # Initialize search engines
        self.semantic_search = SemanticSearch(self.config)
        self.text_search = TextSearchEngine()
        self.similarity_calc = SimilarityCalculator()
        
        # Fusion weights (can be adjusted based on query type)
        self.semantic_weight = 0.6
        self.text_weight = 0.4
        
        # Performance tracking
        self._search_stats = {
            'total_searches': 0,
            'semantic_only': 0,
            'text_only': 0,
            'hybrid': 0,
            'avg_search_time': 0.0
        }
        
        logger.info("Initialized HybridSearch with semantic and text engines")
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict] = None,
        embedding: Optional[np.ndarray] = None
    ):
        """
        Add document to both search indexes.
        
        Args:
            doc_id: Document identifier
            content: Document content
            metadata: Document metadata
            embedding: Pre-computed embedding (optional)
        """
        try:
            # Add to text search index
            self.text_search.add_document(doc_id, content, metadata)
            
            # Add to semantic search (vector store)
            if embedding is None:
                # Generate embedding
                embedding = self.semantic_search.embedding_service.encode_text(content)
            
            # Convert embedding to list for ChromaDB
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            self.semantic_search.vector_store.add_embeddings(
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata or {}],
                ids=[doc_id]
            )
            
            logger.debug(f"Added document {doc_id} to hybrid search index")
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            raise
    
    def remove_document(self, doc_id: str):
        """Remove document from both search indexes."""
        try:
            # Remove from text search
            self.text_search.remove_document(doc_id)
            
            # Remove from vector store
            self.semantic_search.vector_store.delete_embeddings([doc_id])
            
            logger.debug(f"Removed document {doc_id} from hybrid search index")
            
        except Exception as e:
            logger.error(f"Failed to remove document {doc_id}: {e}")
            raise
    
    def search(
        self,
        query: str,
        limit: int = 10,
        semantic_weight: Optional[float] = None,
        text_weight: Optional[float] = None,
        fusion_method: str = "weighted_sum",
        text_method: str = "bm25",
        filters: Optional[Dict] = None,
        explain_results: bool = True
    ) -> List[HybridResult]:
        """
        Perform hybrid search combining semantic and text search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            semantic_weight: Weight for semantic scores
            text_weight: Weight for text scores
            fusion_method: How to combine scores ("weighted_sum", "max", "harmonic")
            text_method: Text search method ("tfidf", "bm25")
            filters: Metadata filters for semantic search
            explain_results: Generate result explanations
            
        Returns:
            List of HybridResult objects
        """
        start_time = time.time()
        
        try:
            # Use configured weights if not specified
            sem_weight = semantic_weight if semantic_weight is not None else self.semantic_weight
            txt_weight = text_weight if text_weight is not None else self.text_weight
            
            # Normalize weights
            total_weight = sem_weight + txt_weight
            if total_weight > 0:
                sem_weight /= total_weight
                txt_weight /= total_weight
            
            # Perform semantic search
            semantic_results = self.semantic_search.search(
                query=query,
                limit=limit * 2,  # Get more to account for fusion
                filters=filters,
                explain_relevance=False
            )
            
            # Perform text search
            if text_method == "bm25":
                text_results = self.text_search.search_bm25(query, limit * 2)
            else:
                text_results = self.text_search.search_tfidf(query, limit * 2)
            
            # Combine results
            hybrid_results = self._fuse_results(
                query=query,
                semantic_results=semantic_results.results,
                text_results=text_results,
                semantic_weight=sem_weight,
                text_weight=txt_weight,
                fusion_method=fusion_method,
                explain_results=explain_results
            )
            
            # Limit and rank results
            final_results = hybrid_results[:limit]
            
            # Update rankings
            for i, result in enumerate(final_results):
                result.final_rank = i + 1
            
            # Update statistics
            search_time = time.time() - start_time
            self._update_search_stats('hybrid', search_time)
            
            logger.debug(f"Hybrid search for '{query}' returned {len(final_results)} results in {search_time:.3f}s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed for query '{query}': {e}")
            raise
    
    def _fuse_results(
        self,
        query: str,
        semantic_results: List[SearchResult],
        text_results: List[Tuple[str, float]],
        semantic_weight: float,
        text_weight: float,
        fusion_method: str,
        explain_results: bool
    ) -> List[HybridResult]:
        """Fuse semantic and text search results."""
        # Create lookup for semantic results
        semantic_lookup = {result.document_id: result for result in semantic_results}
        semantic_scores = {result.document_id: result.similarity_score for result in semantic_results}
        
        # Create lookup for text results
        text_lookup = {doc_id: score for doc_id, score in text_results}
        
        # Get all unique document IDs
        all_doc_ids = set(semantic_scores.keys()) | set(text_lookup.keys())
        
        # Normalize scores to [0, 1] range
        if semantic_scores:
            max_sem_score = max(semantic_scores.values())
            min_sem_score = min(semantic_scores.values())
            sem_range = max_sem_score - min_sem_score if max_sem_score != min_sem_score else 1.0
        else:
            max_sem_score = min_sem_score = sem_range = 1.0
        
        if text_lookup:
            max_txt_score = max(text_lookup.values())
            min_txt_score = min(text_lookup.values())
            txt_range = max_txt_score - min_txt_score if max_txt_score != min_txt_score else 1.0
        else:
            max_txt_score = min_txt_score = txt_range = 1.0
        
        fused_results = []
        
        for doc_id in all_doc_ids:
            # Get normalized scores
            sem_score = semantic_scores.get(doc_id, 0.0)
            txt_score = text_lookup.get(doc_id, 0.0)
            
            # Normalize to [0, 1]
            norm_sem_score = (sem_score - min_sem_score) / sem_range if sem_range > 0 else 0.0
            norm_txt_score = (txt_score - min_txt_score) / txt_range if txt_range > 0 else 0.0
            
            # Combine scores based on fusion method
            if fusion_method == "weighted_sum":
                combined_score = semantic_weight * norm_sem_score + text_weight * norm_txt_score
            elif fusion_method == "max":
                combined_score = max(norm_sem_score, norm_txt_score)
            elif fusion_method == "harmonic":
                if norm_sem_score > 0 and norm_txt_score > 0:
                    combined_score = 2 * norm_sem_score * norm_txt_score / (norm_sem_score + norm_txt_score)
                else:
                    combined_score = max(norm_sem_score, norm_txt_score)
            else:
                combined_score = semantic_weight * norm_sem_score + text_weight * norm_txt_score
            
            # Get document information
            if doc_id in semantic_lookup:
                semantic_result = semantic_lookup[doc_id]
                content = semantic_result.content
                metadata = semantic_result.metadata
                semantic_rank = semantic_result.rank
            else:
                # Document only found in text search
                content = self.text_search.documents.get(doc_id, "")
                metadata = self.text_search.document_metadata.get(doc_id, {})
                semantic_rank = len(semantic_results) + 1
            
            text_rank = next((i + 1 for i, (tid, _) in enumerate(text_results) if tid == doc_id), 
                           len(text_results) + 1)
            
            # Generate explanation
            explanation = ""
            if explain_results:
                explanation = self._generate_hybrid_explanation(
                    norm_sem_score, norm_txt_score, combined_score,
                    semantic_weight, text_weight, fusion_method
                )
            
            hybrid_result = HybridResult(
                document_id=doc_id,
                content=content,
                semantic_score=norm_sem_score,
                text_score=norm_txt_score,
                combined_score=combined_score,
                metadata=metadata,
                semantic_rank=semantic_rank,
                text_rank=text_rank,
                final_rank=0,  # Will be set after sorting
                explanation=explanation
            )
            
            fused_results.append(hybrid_result)
        
        # Sort by combined score
        fused_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return fused_results
    
    def _generate_hybrid_explanation(
        self,
        semantic_score: float,
        text_score: float,
        combined_score: float,
        semantic_weight: float,
        text_weight: float,
        fusion_method: str
    ) -> str:
        """Generate explanation for hybrid search result."""
        explanations = []
        
        # Score breakdown
        explanations.append(f"Semantic: {semantic_score:.3f}")
        explanations.append(f"Text: {text_score:.3f}")
        explanations.append(f"Combined: {combined_score:.3f}")
        
        # Fusion method
        if fusion_method == "weighted_sum":
            explanations.append(f"Weighted fusion ({semantic_weight:.1f}/{text_weight:.1f})")
        elif fusion_method == "max":
            explanations.append("Maximum score fusion")
        elif fusion_method == "harmonic":
            explanations.append("Harmonic mean fusion")
        
        # Dominant component
        if semantic_score > text_score * 1.5:
            explanations.append("Semantic-dominant match")
        elif text_score > semantic_score * 1.5:
            explanations.append("Text-dominant match")
        else:
            explanations.append("Balanced semantic-text match")
        
        return "; ".join(explanations)
    
    def search_semantic_only(
        self,
        query: str,
        limit: int = 10,
        **kwargs
    ) -> SearchResultSet:
        """Perform semantic-only search."""
        start_time = time.time()
        
        result = self.semantic_search.search(query, limit, **kwargs)
        
        search_time = time.time() - start_time
        self._update_search_stats('semantic_only', search_time)
        
        return result
    
    def search_text_only(
        self,
        query: str,
        limit: int = 10,
        method: str = "bm25"
    ) -> List[Tuple[str, float]]:
        """Perform text-only search."""
        start_time = time.time()
        
        if method == "bm25":
            results = self.text_search.search_bm25(query, limit)
        else:
            results = self.text_search.search_tfidf(query, limit)
        
        search_time = time.time() - start_time
        self._update_search_stats('text_only', search_time)
        
        return results
    
    def detect_query_type(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine optimal search strategy.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with query analysis and recommendations
        """
        analysis = {
            'query': query,
            'length': len(query.split()),
            'has_quotes': '"' in query,
            'has_operators': any(op in query.lower() for op in ['and', 'or', 'not']),
            'is_factual': any(word in query.lower() for word in ['what', 'when', 'where', 'who', 'how']),
            'is_conceptual': len(query.split()) > 5,
            'recommended_strategy': 'hybrid'
        }
        
        # Determine optimal strategy
        if analysis['has_quotes'] or analysis['has_operators']:
            analysis['recommended_strategy'] = 'text'
            analysis['recommended_weights'] = {'semantic': 0.3, 'text': 0.7}
        elif analysis['is_conceptual'] and analysis['length'] > 3:
            analysis['recommended_strategy'] = 'semantic'
            analysis['recommended_weights'] = {'semantic': 0.8, 'text': 0.2}
        elif analysis['is_factual']:
            analysis['recommended_strategy'] = 'hybrid'
            analysis['recommended_weights'] = {'semantic': 0.5, 'text': 0.5}
        else:
            analysis['recommended_strategy'] = 'hybrid'
            analysis['recommended_weights'] = {'semantic': 0.6, 'text': 0.4}
        
        return analysis
    
    def auto_search(self, query: str, limit: int = 10, **kwargs) -> List[HybridResult]:
        """
        Automatically determine and execute optimal search strategy.
        
        Args:
            query: Search query
            limit: Maximum results
            **kwargs: Additional search parameters
            
        Returns:
            Search results using optimal strategy
        """
        query_analysis = self.detect_query_type(query)
        strategy = query_analysis['recommended_strategy']
        weights = query_analysis.get('recommended_weights', {'semantic': 0.6, 'text': 0.4})
        
        logger.debug(f"Auto-search using {strategy} strategy for: {query}")
        
        if strategy == 'semantic':
            semantic_results = self.search_semantic_only(query, limit, **kwargs)
            # Convert to HybridResult format
            hybrid_results = []
            for i, result in enumerate(semantic_results.results):
                hybrid_result = HybridResult(
                    document_id=result.document_id,
                    content=result.content,
                    semantic_score=result.similarity_score,
                    text_score=0.0,
                    combined_score=result.similarity_score,
                    metadata=result.metadata,
                    semantic_rank=i + 1,
                    text_rank=0,
                    final_rank=i + 1,
                    explanation=f"Semantic-only search: {result.relevance_explanation}"
                )
                hybrid_results.append(hybrid_result)
            return hybrid_results
        
        elif strategy == 'text':
            text_results = self.search_text_only(query, limit)
            # Convert to HybridResult format
            hybrid_results = []
            for i, (doc_id, score) in enumerate(text_results):
                content = self.text_search.documents.get(doc_id, "")
                metadata = self.text_search.document_metadata.get(doc_id, {})
                hybrid_result = HybridResult(
                    document_id=doc_id,
                    content=content,
                    semantic_score=0.0,
                    text_score=score,
                    combined_score=score,
                    metadata=metadata,
                    semantic_rank=0,
                    text_rank=i + 1,
                    final_rank=i + 1,
                    explanation=f"Text-only search (BM25): {score:.3f}"
                )
                hybrid_results.append(hybrid_result)
            return hybrid_results
        
        else:  # hybrid
            return self.search(
                query=query,
                limit=limit,
                semantic_weight=weights['semantic'],
                text_weight=weights['text'],
                **kwargs
            )
    
    def _update_search_stats(self, search_type: str, search_time: float):
        """Update search statistics."""
        self._search_stats['total_searches'] += 1
        self._search_stats[search_type] += 1
        
        # Update average search time
        total_time = self._search_stats['avg_search_time'] * (self._search_stats['total_searches'] - 1)
        self._search_stats['avg_search_time'] = (total_time + search_time) / self._search_stats['total_searches']
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        return {
            'hybrid_stats': self._search_stats,
            'semantic_stats': self.semantic_search.get_search_stats(),
            'text_stats': self.text_search.get_document_stats(),
            'fusion_weights': {
                'semantic': self.semantic_weight,
                'text': self.text_weight
            }
        }
    
    def optimize_weights(
        self,
        test_queries: List[str],
        relevance_judgments: Dict[str, List[str]],
        weight_combinations: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Optimize fusion weights based on test queries and relevance judgments.
        
        Args:
            test_queries: List of test queries
            relevance_judgments: Dict mapping queries to relevant document IDs
            weight_combinations: List of (semantic_weight, text_weight) to test
            
        Returns:
            Optimization results with best weights
        """
        if weight_combinations is None:
            weight_combinations = [
                (1.0, 0.0), (0.8, 0.2), (0.6, 0.4), (0.5, 0.5),
                (0.4, 0.6), (0.2, 0.8), (0.0, 1.0)
            ]
        
        best_weights = (0.6, 0.4)
        best_score = 0.0
        results = []
        
        for sem_weight, txt_weight in weight_combinations:
            total_precision = 0.0
            total_recall = 0.0
            total_f1 = 0.0
            
            for query in test_queries:
                if query not in relevance_judgments:
                    continue
                
                relevant_docs = set(relevance_judgments[query])
                search_results = self.search(
                    query=query,
                    semantic_weight=sem_weight,
                    text_weight=txt_weight,
                    limit=20
                )
                
                retrieved_docs = set(result.document_id for result in search_results)
                
                # Calculate metrics
                if retrieved_docs:
                    precision = len(relevant_docs & retrieved_docs) / len(retrieved_docs)
                else:
                    precision = 0.0
                
                if relevant_docs:
                    recall = len(relevant_docs & retrieved_docs) / len(relevant_docs)
                else:
                    recall = 0.0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
            
            # Average metrics
            avg_precision = total_precision / len(test_queries)
            avg_recall = total_recall / len(test_queries)
            avg_f1 = total_f1 / len(test_queries)
            
            results.append({
                'semantic_weight': sem_weight,
                'text_weight': txt_weight,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1
            })
            
            # Use F1 score for optimization
            if avg_f1 > best_score:
                best_score = avg_f1
                best_weights = (sem_weight, txt_weight)
        
        return {
            'best_weights': {
                'semantic': best_weights[0],
                'text': best_weights[1]
            },
            'best_f1_score': best_score,
            'all_results': results
        }
    
    def cleanup(self):
        """Clean up hybrid search resources."""
        if self.semantic_search:
            self.semantic_search.cleanup()
        
        # Clear text search data
        self.text_search.documents.clear()
        self.text_search.document_metadata.clear()
        self.text_search.term_frequencies.clear()
        self.text_search.document_frequencies.clear()
        self.text_search.document_lengths.clear()
        
        logger.info("HybridSearch cleanup completed")