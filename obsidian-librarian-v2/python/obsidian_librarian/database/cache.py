"""
Cache database implementation using Redis with SQLite fallback.

Provides high-performance caching for research results, embeddings,
and session data with local fallback when Redis is unavailable.
"""

import asyncio
import json
# import pickle  # SECURITY: Removed pickle to prevent arbitrary code execution
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import aioredis
    AIOREDIS_AVAILABLE = True
except ImportError:
    AIOREDIS_AVAILABLE = False
    aioredis = None

import structlog

from .base import CacheDatabase, DatabaseConfig, ConnectionError, QueryError

logger = structlog.get_logger(__name__)


class CacheDB(CacheDatabase):
    """Redis-based cache with SQLite fallback."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.redis: Optional[aioredis.Redis] = None
        self.sqlite_conn: Optional[sqlite3.Connection] = None
        self.use_redis = True
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize cache database connections."""
        async with self._lock:
            # Try Redis first
            if await self._init_redis():
                logger.info("Cache initialized with Redis", url=self.config.cache_url)
            elif self.config.cache_local_fallback:
                await self._init_sqlite()
                logger.info("Cache initialized with SQLite fallback", 
                           path=self.config.cache_local_path)
            else:
                raise ConnectionError("No cache backend available")
    
    async def _init_redis(self) -> bool:
        """Initialize Redis connection."""
        if not AIOREDIS_AVAILABLE:
            logger.warning("aioredis not available, falling back to SQLite")
            return False
            
        try:
            self.redis = aioredis.from_url(
                self.config.cache_url,
                encoding="utf-8",
                decode_responses=False,  # We'll handle encoding ourselves
                socket_timeout=self.config.connection_timeout,
                socket_connect_timeout=self.config.connection_timeout,
            )
            
            # Test connection
            await self.redis.ping()
            
            # Configure Redis settings
            await self.redis.config_set("maxmemory", self.config.cache_max_memory)
            await self.redis.config_set("maxmemory-policy", "allkeys-lru")
            
            self.use_redis = True
            return True
            
        except Exception as e:
            logger.warning("Failed to connect to Redis", error=str(e))
            if self.redis:
                await self.redis.close()
                self.redis = None
            self.use_redis = False
            return False
    
    async def _init_sqlite(self) -> None:
        """Initialize SQLite fallback."""
        try:
            if self.config.cache_local_path:
                self.config.cache_local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create connection in a thread executor since sqlite3 is synchronous
                loop = asyncio.get_event_loop()
                self.sqlite_conn = await loop.run_in_executor(
                    None, 
                    sqlite3.connect, 
                    str(self.config.cache_local_path)
                )
                
                # Create tables
                await self._create_sqlite_tables()
                
        except Exception as e:
            logger.error("Failed to initialize SQLite cache", error=str(e))
            raise ConnectionError(f"SQLite cache initialization failed: {e}")
    
    async def _create_sqlite_tables(self) -> None:
        """Create SQLite cache tables."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS cache_entries (
            key TEXT PRIMARY KEY,
            value BLOB NOT NULL,
            expiry TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_cache_expiry ON cache_entries(expiry)
        """
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            lambda: self.sqlite_conn.execute(create_table_sql)
        )
        await loop.run_in_executor(
            None, 
            lambda: self.sqlite_conn.execute(create_index_sql)
        )
        await loop.run_in_executor(
            None, 
            lambda: self.sqlite_conn.commit()
        )
    
    async def close(self) -> None:
        """Close cache database connections."""
        async with self._lock:
            if self.redis:
                try:
                    await self.redis.close()
                    self.redis = None
                    logger.debug("Redis connection closed")
                except Exception as e:
                    logger.error("Error closing Redis connection", error=str(e))
            
            if self.sqlite_conn:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.sqlite_conn.close)
                    self.sqlite_conn = None
                    logger.debug("SQLite connection closed")
                except Exception as e:
                    logger.error("Error closing SQLite connection", error=str(e))
    
    async def health_check(self) -> bool:
        """Check cache database health."""
        try:
            if self.use_redis and self.redis:
                await self.redis.ping()
                return True
            elif self.sqlite_conn:
                # Test SQLite connection
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, 
                    lambda: self.sqlite_conn.execute("SELECT 1").fetchone()
                )
                return True
            return False
        except Exception:
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        try:
            if self.use_redis and self.redis:
                return await self._redis_get(key)
            elif self.sqlite_conn:
                return await self._sqlite_get(key)
            else:
                raise ConnectionError("No cache backend available")
        except Exception as e:
            logger.error("Cache get failed", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value."""
        try:
            if self.use_redis and self.redis:
                await self._redis_set(key, value, ttl)
            elif self.sqlite_conn:
                await self._sqlite_set(key, value, ttl)
            else:
                raise ConnectionError("No cache backend available")
        except Exception as e:
            logger.error("Cache set failed", key=key, error=str(e))
            raise QueryError(f"Cache set failed: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete cached value."""
        try:
            if self.use_redis and self.redis:
                await self.redis.delete(key)
            elif self.sqlite_conn:
                await self._sqlite_delete(key)
        except Exception as e:
            logger.error("Cache delete failed", key=key, error=str(e))
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            if self.use_redis and self.redis:
                return bool(await self.redis.exists(key))
            elif self.sqlite_conn:
                return await self._sqlite_exists(key)
            return False
        except Exception as e:
            logger.error("Cache exists check failed", key=key, error=str(e))
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        try:
            if self.use_redis and self.redis:
                keys = await self.redis.keys(pattern)
                if keys:
                    return await self.redis.delete(*keys)
                return 0
            elif self.sqlite_conn:
                return await self._sqlite_clear_pattern(pattern)
            return 0
        except Exception as e:
            logger.error("Cache pattern clear failed", pattern=pattern, error=str(e))
            return 0
    
    async def _redis_get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        data = await self.redis.get(key)
        if data is None:
            return None
        
        # SECURITY: Avoid pickle deserialization for security reasons
        # Only use JSON for serialization to prevent arbitrary code execution
        try:
            # Primary: Try JSON deserialization
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                # Fallback: Return as string
                return data.decode('utf-8', errors='ignore')
            except:
                # Last resort: Return raw bytes
                logger.warning("Failed to deserialize cache data, returning raw bytes")
                return data
    
    async def _redis_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis."""
        # SECURITY: Use JSON serialization instead of pickle to prevent code injection
        try:
            # Primary: Use JSON serialization
            data = json.dumps(value, default=str).encode('utf-8')
        except (TypeError, ValueError) as e:
            # Fallback: Convert to string
            logger.warning(f"Failed to JSON serialize value, using string representation: {e}")
            data = str(value).encode('utf-8')
        
        ttl = ttl or self.config.cache_ttl_seconds
        await self.redis.setex(key, ttl, data)
    
    async def _sqlite_get(self, key: str) -> Optional[Any]:
        """Get value from SQLite."""
        loop = asyncio.get_event_loop()
        
        def _get():
            cursor = self.sqlite_conn.cursor()
            cursor.execute(
                "SELECT value, expiry FROM cache_entries WHERE key = ?", 
                (key,)
            )
            result = cursor.fetchone()
            
            if not result:
                return None
            
            value_blob, expiry = result
            
            # Check expiry
            if expiry:
                expiry_dt = datetime.fromisoformat(expiry)
                if datetime.now() > expiry_dt:
                    # Expired, delete and return None
                    cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    self.sqlite_conn.commit()
                    return None
            
            # SECURITY: Deserialize value using JSON only
            try:
                return json.loads(value_blob.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    return value_blob.decode('utf-8', errors='ignore')
                except:
                    logger.warning("Failed to deserialize SQLite cache data")
                    return value_blob
        
        return await loop.run_in_executor(None, _get)
    
    async def _sqlite_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in SQLite."""
        loop = asyncio.get_event_loop()
        
        def _set():
            # SECURITY: Serialize value using JSON only
            try:
                value_blob = json.dumps(value, default=str).encode('utf-8')
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to JSON serialize for SQLite, using string: {e}")
                value_blob = str(value).encode('utf-8')
            
            # Calculate expiry
            expiry = None
            if ttl or self.config.cache_ttl_seconds:
                expiry_seconds = ttl or self.config.cache_ttl_seconds
                expiry = (datetime.now() + timedelta(seconds=expiry_seconds)).isoformat()
            
            cursor = self.sqlite_conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO cache_entries (key, value, expiry) 
                VALUES (?, ?, ?)
                """,
                (key, value_blob, expiry)
            )
            self.sqlite_conn.commit()
        
        await loop.run_in_executor(None, _set)
    
    async def _sqlite_delete(self, key: str) -> None:
        """Delete value from SQLite."""
        loop = asyncio.get_event_loop()
        
        def _delete():
            cursor = self.sqlite_conn.cursor()
            cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            self.sqlite_conn.commit()
        
        await loop.run_in_executor(None, _delete)
    
    async def _sqlite_exists(self, key: str) -> bool:
        """Check if key exists in SQLite."""
        loop = asyncio.get_event_loop()
        
        def _exists():
            cursor = self.sqlite_conn.cursor()
            cursor.execute(
                "SELECT 1 FROM cache_entries WHERE key = ? AND (expiry IS NULL OR expiry > ?)",
                (key, datetime.now().isoformat())
            )
            return cursor.fetchone() is not None
        
        return await loop.run_in_executor(None, _exists)
    
    async def _sqlite_clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern in SQLite."""
        loop = asyncio.get_event_loop()
        
        def _clear():
            cursor = self.sqlite_conn.cursor()
            # Convert shell-style pattern to SQL LIKE pattern
            sql_pattern = pattern.replace('*', '%').replace('?', '_')
            cursor.execute("DELETE FROM cache_entries WHERE key LIKE ?", (sql_pattern,))
            self.sqlite_conn.commit()
            return cursor.rowcount
        
        return await loop.run_in_executor(None, _clear)
    
    # High-level cache operations
    
    async def cache_research_result(
        self, 
        query: str, 
        result: Dict[str, Any], 
        ttl: int = 3600
    ) -> None:
        """Cache research result."""
        key = f"research:{hash(query)}"
        await self.set(key, result, ttl)
    
    async def get_cached_research(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached research result."""
        key = f"research:{hash(query)}"
        return await self.get(key)
    
    async def cache_embedding(
        self, 
        content_hash: str, 
        embedding: List[float], 
        ttl: int = 86400  # 24 hours
    ) -> None:
        """Cache content embedding."""
        key = f"embedding:{content_hash}"
        await self.set(key, embedding, ttl)
    
    async def get_cached_embedding(self, content_hash: str) -> Optional[List[float]]:
        """Get cached embedding."""
        key = f"embedding:{content_hash}"
        return await self.get(key)
    
    async def cache_search_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        ttl: int = 1800  # 30 minutes
    ) -> None:
        """Cache search results."""
        key = f"search:{hash(query)}"
        await self.set(key, results, ttl)
    
    async def get_cached_search(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        key = f"search:{hash(query)}"
        return await self.get(key)
    
    async def cache_vault_stats(
        self, 
        stats: Dict[str, Any], 
        ttl: int = 300  # 5 minutes
    ) -> None:
        """Cache vault statistics."""
        await self.set("vault:stats", stats, ttl)
    
    async def get_cached_vault_stats(self) -> Optional[Dict[str, Any]]:
        """Get cached vault statistics."""
        return await self.get("vault:stats")
    
    async def cleanup_expired(self) -> int:
        """Clean up expired cache entries."""
        if self.use_redis and self.redis:
            # Redis handles expiry automatically
            return 0
        elif self.sqlite_conn:
            loop = asyncio.get_event_loop()
            
            def _cleanup():
                cursor = self.sqlite_conn.cursor()
                cursor.execute(
                    "DELETE FROM cache_entries WHERE expiry IS NOT NULL AND expiry <= ?",
                    (datetime.now().isoformat(),)
                )
                self.sqlite_conn.commit()
                return cursor.rowcount
            
            count = await loop.run_in_executor(None, _cleanup)
            logger.info("Cleaned up expired cache entries", count=count)
            return count
        
        return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {"backend": "redis" if self.use_redis else "sqlite"}
        
        try:
            if self.use_redis and self.redis:
                info = await self.redis.info("memory")
                stats.update({
                    "memory_used": info.get("used_memory_human", "unknown"),
                    "keys": await self.redis.dbsize(),
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0),
                })
                
                if stats["hits"] + stats["misses"] > 0:
                    stats["hit_rate"] = stats["hits"] / (stats["hits"] + stats["misses"])
                
            elif self.sqlite_conn:
                loop = asyncio.get_event_loop()
                
                def _get_sqlite_stats():
                    cursor = self.sqlite_conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM cache_entries")
                    total_keys = cursor.fetchone()[0]
                    
                    cursor.execute(
                        "SELECT COUNT(*) FROM cache_entries WHERE expiry IS NOT NULL AND expiry <= ?",
                        (datetime.now().isoformat(),)
                    )
                    expired_keys = cursor.fetchone()[0]
                    
                    return total_keys, expired_keys
                
                total_keys, expired_keys = await loop.run_in_executor(None, _get_sqlite_stats)
                stats.update({
                    "total_keys": total_keys,
                    "expired_keys": expired_keys,
                    "active_keys": total_keys - expired_keys,
                })
        
        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            stats["error"] = str(e)
        
        return stats