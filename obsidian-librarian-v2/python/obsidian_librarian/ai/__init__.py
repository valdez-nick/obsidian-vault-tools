"""
AI module for Obsidian Librarian.

Provides intelligent content processing, embeddings, and language model integration
with support for multiple providers and local models.
"""

from .models import AIModelManager, AIConfig
from .embeddings import EmbeddingService
from .language_models import LanguageModelService
from .content_analyzer import ContentAnalyzer
from .query_processor import QueryProcessor
from .content_summarizer import ContentSummarizer

__all__ = [
    "AIModelManager",
    "AIConfig",
    "EmbeddingService", 
    "LanguageModelService",
    "ContentAnalyzer",
    "QueryProcessor",
    "ContentSummarizer",
]