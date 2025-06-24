"""
Search module for Obsidian Vault Tools Memory Service.

This module provides hybrid search capabilities that combine semantic embeddings
with traditional text search methods for optimal retrieval performance.
"""

from .semantic_search import SemanticSearch, SearchResult, SearchResultSet
from .hybrid_search import HybridSearch, HybridResult, TextSearchEngine
from .ranking import RankingEngine, RankedResult, RankingCriteria, RankingStrategy, UserProfile
from .similarity import SimilarityCalculator, SimilarityResult, SimilarityMetric

__all__ = [
    # Main search engines
    'SemanticSearch',
    'HybridSearch', 
    'RankingEngine',
    'SimilarityCalculator',
    'TextSearchEngine',
    
    # Result classes
    'SearchResult',
    'SearchResultSet',
    'HybridResult',
    'RankedResult',
    'SimilarityResult',
    
    # Configuration classes
    'RankingCriteria',
    'UserProfile',
    
    # Enums
    'RankingStrategy',
    'SimilarityMetric'
]