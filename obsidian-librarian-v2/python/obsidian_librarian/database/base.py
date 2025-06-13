"""
Base database interfaces and configuration.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    
    # Analytics database (DuckDB)
    analytics_path: Path = field(default_factory=lambda: Path(".obsidian-librarian/analytics.db"))
    analytics_memory_limit: str = "1GB"
    analytics_threads: int = 4
    
    # Vector database (Qdrant)
    vector_url: str = "http://localhost:6333"
    vector_collection: str = "obsidian_notes"
    vector_dimension: int = 1536  # OpenAI embedding dimension
    vector_distance: str = "Cosine"
    vector_local_path: Optional[Path] = field(
        default_factory=lambda: Path(".obsidian-librarian/vector_db")
    )
    
    # Cache database (Redis)
    cache_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600  # 1 hour default
    cache_max_memory: str = "100mb"
    cache_local_fallback: bool = True
    cache_local_path: Optional[Path] = field(
        default_factory=lambda: Path(".obsidian-librarian/cache.db")
    )
    
    # Connection settings
    connection_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


class Database(Protocol):
    """Protocol for database implementations."""
    
    async def initialize(self) -> None:
        """Initialize the database connection."""
        ...
    
    async def close(self) -> None:
        """Close the database connection."""
        ...
    
    async def health_check(self) -> bool:
        """Check if the database is healthy."""
        ...


class AnalyticsDatabase(Database, Protocol):
    """Protocol for analytics database operations."""
    
    async def create_tables(self) -> None:
        """Create analytics tables."""
        ...
    
    async def insert_note_metrics(self, note_id: str, metrics: Dict[str, Any]) -> None:
        """Insert note metrics."""
        ...
    
    async def get_vault_stats(self) -> Dict[str, Any]:
        """Get vault statistics."""
        ...
    
    async def get_usage_patterns(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get usage patterns over time."""
        ...


class VectorDatabase(Database, Protocol):
    """Protocol for vector database operations."""
    
    async def create_collection(self) -> None:
        """Create vector collection."""
        ...
    
    async def upsert_embedding(
        self, 
        note_id: str, 
        embedding: List[float], 
        metadata: Dict[str, Any]
    ) -> None:
        """Insert or update note embedding."""
        ...
    
    async def search_similar(
        self, 
        embedding: List[float], 
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar notes."""
        ...
    
    async def delete_embedding(self, note_id: str) -> None:
        """Delete note embedding."""
        ...


class CacheDatabase(Database, Protocol):
    """Protocol for cache database operations."""
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        ...
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value."""
        ...
    
    async def delete(self, key: str) -> None:
        """Delete cached value."""
        ...
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        ...


class DatabaseManager:
    """Unified database manager for all database backends."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.analytics: Optional[AnalyticsDatabase] = None
        self.vector: Optional[VectorDatabase] = None
        self.cache: Optional[CacheDatabase] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all database connections."""
        if self._initialized:
            return
        
        logger.info("Initializing database manager")
        
        try:
            # Initialize analytics database
            from .analytics import AnalyticsDB
            self.analytics = AnalyticsDB(self.config)
            await self.analytics.initialize()
            logger.info("Analytics database initialized")
            
            # Initialize vector database
            from .vector import VectorDB
            self.vector = VectorDB(self.config)
            await self.vector.initialize()
            logger.info("Vector database initialized")
            
            # Initialize cache database
            from .cache import CacheDB
            self.cache = CacheDB(self.config)
            await self.cache.initialize()
            logger.info("Cache database initialized")
            
            self._initialized = True
            logger.info("All databases initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize databases", error=str(e))
            await self.close()
            raise
    
    async def close(self) -> None:
        """Close all database connections."""
        logger.info("Closing database connections")
        
        errors = []
        
        for db_name, db in [
            ("analytics", self.analytics),
            ("vector", self.vector), 
            ("cache", self.cache)
        ]:
            if db:
                try:
                    await db.close()
                    logger.debug(f"{db_name} database closed")
                except Exception as e:
                    logger.error(f"Error closing {db_name} database", error=str(e))
                    errors.append(e)
        
        self.analytics = None
        self.vector = None
        self.cache = None
        self._initialized = False
        
        if errors:
            raise RuntimeError(f"Errors occurred while closing databases: {errors}")
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all databases."""
        health = {}
        
        for db_name, db in [
            ("analytics", self.analytics),
            ("vector", self.vector),
            ("cache", self.cache)
        ]:
            if db:
                try:
                    health[db_name] = await db.health_check()
                except Exception as e:
                    logger.error(f"Health check failed for {db_name}", error=str(e))
                    health[db_name] = False
            else:
                health[db_name] = False
        
        return health
    
    async def migrate(self) -> None:
        """Run database migrations."""
        logger.info("Running database migrations")
        
        if self.analytics:
            await self.analytics.create_tables()
        
        if self.vector:
            await self.vector.create_collection()
        
        logger.info("Database migrations completed")
    
    async def backup(self, backup_path: Path) -> None:
        """Backup all databases."""
        logger.info("Starting database backup", backup_path=backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # TODO: Implement backup logic for each database
        logger.info("Database backup completed")
    
    async def restore(self, backup_path: Path) -> None:
        """Restore all databases from backup."""
        logger.info("Starting database restore", backup_path=backup_path)
        
        # TODO: Implement restore logic for each database
        logger.info("Database restore completed")
    
    # Context manager support
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ConnectionError(DatabaseError):
    """Database connection error."""
    pass


class QueryError(DatabaseError):
    """Database query error."""
    pass


class MigrationError(DatabaseError):
    """Database migration error."""
    pass