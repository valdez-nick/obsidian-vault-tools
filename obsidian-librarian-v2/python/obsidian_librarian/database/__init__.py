"""
Database integration for Obsidian Librarian.

This module provides a unified interface for multiple database backends:
- DuckDB for analytics and reporting
- Qdrant for vector search and similarity
- Redis for caching and session management
"""

from .base import DatabaseManager, DatabaseConfig
from .analytics import AnalyticsDB
from .vector import VectorDB
from .cache import CacheDB

__all__ = [
    "DatabaseManager",
    "DatabaseConfig", 
    "AnalyticsDB",
    "VectorDB",
    "CacheDB",
]