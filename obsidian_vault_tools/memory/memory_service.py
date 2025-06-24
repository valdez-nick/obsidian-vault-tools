"""
Memory Service Core

A singleton service providing unified access to memory operations with:
- Async/sync dual interfaces for compatibility
- Caching layer for performance
- Thread-safe operations
- Integration with memory_client and memory_manager
"""

import asyncio
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from pathlib import Path
import json
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager

from ..mcp_tools.memory_client import get_memory_client, MemoryMCPClient, Entity, Relation
from ..intelligence.memory_manager import get_memory_manager, MemoryManager
from ..intelligence.memory_models import EntityType, RelationType

# Vector embeddings imports
try:
    from .embeddings.embedding_service import EmbeddingService
    from .embeddings.chroma_store import ChromaStore
    from .embeddings.config import EmbeddingConfig, DEFAULT_CONFIG
    from .search.semantic_search import SemanticSearch
    from .search.hybrid_search import HybridSearch
    from .search.ranking import RankingEngine
    from .models.model_manager import ModelManager
    VECTOR_EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Vector embeddings not available: {e}")
    VECTOR_EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)


# Custom exceptions
class MemoryServiceError(Exception):
    """Base exception for memory service errors"""
    pass


class MemoryServiceNotInitializedError(MemoryServiceError):
    """Raised when attempting to use the service before initialization"""
    pass


# Cache configuration
@dataclass
class CacheConfig:
    """Configuration for memory service caching"""
    max_size: int = 1000
    ttl_seconds: int = 300  # 5 minutes default
    enable_user_preferences: bool = True
    enable_vault_entities: bool = True
    enable_tool_recommendations: bool = True
    enable_research_interests: bool = True
    # Vector embeddings caching
    enable_vector_search_cache: bool = True
    enable_embedding_cache: bool = True
    vector_cache_ttl_seconds: int = 600  # 10 minutes for vector results


# Helper decorator for sync/async compatibility
def dual_interface(sync_method_name: str):
    """Decorator to create both async and sync versions of a method"""
    def decorator(async_method):
        @wraps(async_method)
        def wrapper(self, *args, **kwargs):
            # If called from async context, return the coroutine
            if asyncio.iscoroutinefunction(async_method):
                return async_method(self, *args, **kwargs)
            else:
                # Otherwise, run in the event loop
                return self._run_async(async_method(self, *args, **kwargs))
        
        # Store the sync version on the class
        wrapper._sync_name = sync_method_name
        return wrapper
    return decorator


class MemoryService:
    """
    Singleton memory service providing unified access to memory operations.
    
    Features:
    - Thread-safe singleton pattern
    - Async/sync dual interfaces
    - Multi-level caching with TTL
    - Automatic connection lifecycle management
    - Helper methods for common patterns
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Skip initialization if already done
        if self._initialized:
            return
        
        # Core components
        self.memory_client: Optional[MemoryMCPClient] = None
        self.memory_manager: Optional[MemoryManager] = None
        
        # Vector embeddings components (optional)
        self.vector_config: Optional[EmbeddingConfig] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.semantic_search: Optional[SemanticSearch] = None
        self.hybrid_search: Optional[HybridSearch] = None
        self.ranking_engine: Optional[RankingEngine] = None
        self.model_manager: Optional[ModelManager] = None
        self._vector_enabled = VECTOR_EMBEDDINGS_AVAILABLE
        
        # Configuration
        self.cache_config = CacheConfig()
        self.user_id = "default_user"
        self.vault_path: Optional[str] = None
        
        # Thread-safe state
        self._is_connected = False
        self._connection_lock = threading.Lock()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Cache storage with timestamps
        self._cache_data: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_lock = threading.Lock()
        
        # Statistics
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_requests": 0,
            "errors": 0
        }
        
        self._initialized = True
        logger.info("MemoryService singleton initialized")
    
    # Connection management
    
    async def initialize(
        self, 
        user_id: str = "default_user", 
        vault_path: Optional[str] = None,
        enable_vectors: bool = True,
        vector_config: Optional[EmbeddingConfig] = None
    ) -> bool:
        """
        Initialize the memory service with user and vault information.
        
        Args:
            user_id: User identifier for memory operations
            vault_path: Optional path to Obsidian vault
            enable_vectors: Whether to enable vector embeddings features
            vector_config: Custom vector embeddings configuration
            
        Returns:
            bool: True if initialization successful
        """
        with self._connection_lock:
            if self._is_connected:
                logger.info("MemoryService already initialized")
                return True
            
            try:
                self.user_id = user_id
                self.vault_path = vault_path
                
                # Initialize memory components
                self.memory_client = get_memory_client()
                self.memory_manager = get_memory_manager(user_id, vault_path)
                
                # Initialize the manager (which initializes the client)
                success = await self.memory_manager.initialize()
                
                # Initialize vector embeddings if available and enabled
                if success and enable_vectors and self._vector_enabled:
                    await self._initialize_vector_components(vector_config)
                
                if success:
                    self._is_connected = True
                    logger.info(f"MemoryService initialized for user: {user_id}")
                    if self._vector_enabled and enable_vectors:
                        logger.info("Vector embeddings enabled")
                else:
                    logger.error("Failed to initialize memory components")
                
                return success
                
            except Exception as e:
                logger.error(f"Error initializing MemoryService: {e}")
                self._stats["errors"] += 1
                return False
    
    def initialize_sync(
        self, 
        user_id: str = "default_user", 
        vault_path: Optional[str] = None,
        enable_vectors: bool = True,
        vector_config: Optional[EmbeddingConfig] = None
    ) -> bool:
        """Synchronous version of initialize"""
        return self._run_async(self.initialize(user_id, vault_path, enable_vectors, vector_config))
    
    async def ensure_connected(self) -> None:
        """Ensure the service is connected, raise error if not"""
        if not self._is_connected:
            raise MemoryServiceNotInitializedError(
                "MemoryService not initialized. Call initialize() first."
            )
    
    @asynccontextmanager
    async def connection(self):
        """Context manager for memory service connection"""
        await self.ensure_connected()
        try:
            yield self
        finally:
            pass  # Connection persists across contexts
    
    # Cache management
    
    def _cache_key(self, category: str, *args) -> str:
        """Generate cache key from category and arguments"""
        key_parts = [category] + [str(arg) for arg in args]
        return ":".join(key_parts)
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached data is still valid based on TTL"""
        age = datetime.now() - timestamp
        return age < timedelta(seconds=self.cache_config.ttl_seconds)
    
    def _get_cached(self, key: str, ttl_seconds: Optional[int] = None) -> Optional[Any]:
        """Get value from cache if valid"""
        with self._cache_lock:
            if key in self._cache_data:
                value, timestamp = self._cache_data[key]
                
                # Use custom TTL if provided
                if ttl_seconds is not None:
                    age = datetime.now() - timestamp
                    is_valid = age < timedelta(seconds=ttl_seconds)
                else:
                    is_valid = self._is_cache_valid(timestamp)
                
                if is_valid:
                    self._stats["cache_hits"] += 1
                    logger.debug(f"Cache hit for key: {key}")
                    return value
                else:
                    # Remove expired entry
                    del self._cache_data[key]
            
            self._stats["cache_misses"] += 1
            return None
    
    def _set_cached(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set value in cache with current timestamp"""
        with self._cache_lock:
            # Enforce max cache size (simple LRU by removing oldest)
            if len(self._cache_data) >= self.cache_config.max_size:
                oldest_key = min(self._cache_data.items(), key=lambda x: x[1][1])[0]
                del self._cache_data[oldest_key]
            
            # Store value with timestamp (TTL is checked during retrieval)
            self._cache_data[key] = (value, datetime.now())
            logger.debug(f"Cached value for key: {key} (TTL: {ttl_seconds or 'default'}s)")
    
    def clear_cache(self, category: Optional[str] = None) -> None:
        """Clear cache, optionally for a specific category"""
        with self._cache_lock:
            if category:
                # Clear only keys matching the category
                keys_to_remove = [k for k in self._cache_data.keys() if k.startswith(f"{category}:")]
                for key in keys_to_remove:
                    del self._cache_data[key]
                logger.info(f"Cleared {len(keys_to_remove)} cache entries for category: {category}")
            else:
                # Clear all cache
                self._cache_data.clear()
                logger.info("Cleared all cache entries")
    
    # User preference operations
    
    @dual_interface("record_preference_sync")
    async def record_preference(self, preference_name: str, preference_value: str, 
                              strength: str = "moderate") -> bool:
        """Record a user preference with caching"""
        await self.ensure_connected()
        self._stats["total_requests"] += 1
        
        try:
            result = await self.memory_manager.record_user_preference(
                preference_name, preference_value, strength
            )
            
            if result:
                # Invalidate user preferences cache
                self.clear_cache("user_preferences")
            
            return result
            
        except Exception as e:
            logger.error(f"Error recording preference: {e}")
            self._stats["errors"] += 1
            return False
    
    @dual_interface("get_preferences_sync")
    async def get_preferences(self, category: Optional[str] = None, 
                            use_cache: bool = True) -> Dict[str, Any]:
        """Get user preferences with caching"""
        await self.ensure_connected()
        self._stats["total_requests"] += 1
        
        # Check cache first
        cache_key = self._cache_key("user_preferences", category or "all")
        if use_cache and self.cache_config.enable_user_preferences:
            cached_value = self._get_cached(cache_key)
            if cached_value is not None:
                return cached_value
        
        try:
            preferences = await self.memory_manager.get_user_preferences(category)
            
            # Cache the result
            if self.cache_config.enable_user_preferences:
                self._set_cached(cache_key, preferences)
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error getting preferences: {e}")
            self._stats["errors"] += 1
            return {}
    
    # Action and tool tracking
    
    @dual_interface("track_action_sync")
    async def track_action(self, action_name: str, success: bool = True,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Track a user action"""
        await self.ensure_connected()
        self._stats["total_requests"] += 1
        
        try:
            # Create action entity
            action_entity = Entity(
                name=f"action_{action_name.replace(' ', '_')}",
                entity_type=EntityType.ACTION.value,
                observations=[
                    f"Performed on {datetime.now().isoformat()}",
                    f"Success: {success}"
                ]
            )
            
            if metadata:
                for key, value in metadata.items():
                    action_entity.observations.append(f"{key}: {value}")
            
            # Create entity and relation
            await self.memory_client.create_entities([action_entity])
            
            relation = Relation(
                from_entity=self.user_id,
                to_entity=action_entity.name,
                relation_type=RelationType.PERFORMS.value
            )
            await self.memory_client.create_relations([relation])
            
            return True
            
        except Exception as e:
            logger.error(f"Error tracking action: {e}")
            self._stats["errors"] += 1
            return False
    
    @dual_interface("track_tool_usage_sync")
    async def track_tool_usage(self, tool_name: str, category: str, 
                             success: bool, duration: Optional[float] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Track tool usage with caching invalidation"""
        await self.ensure_connected()
        self._stats["total_requests"] += 1
        
        try:
            result = await self.memory_manager.track_tool_usage(
                tool_name, category, success, duration, metadata
            )
            
            if result:
                # Invalidate tool recommendations cache
                self.clear_cache("tool_recommendations")
            
            return result
            
        except Exception as e:
            logger.error(f"Error tracking tool usage: {e}")
            self._stats["errors"] += 1
            return False
    
    @dual_interface("get_tool_recommendations_sync")
    async def get_tool_recommendations(self, context: Optional[str] = None,
                                     use_cache: bool = True) -> List[Dict[str, Any]]:
        """Get tool recommendations with caching"""
        await self.ensure_connected()
        self._stats["total_requests"] += 1
        
        # Check cache first
        cache_key = self._cache_key("tool_recommendations", context or "general")
        if use_cache and self.cache_config.enable_tool_recommendations:
            cached_value = self._get_cached(cache_key)
            if cached_value is not None:
                return cached_value
        
        try:
            recommendations = await self.memory_manager.get_tool_recommendations(context)
            
            # Cache the result
            if self.cache_config.enable_tool_recommendations:
                self._set_cached(cache_key, recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting tool recommendations: {e}")
            self._stats["errors"] += 1
            return []
    
    # Vault operations
    
    @dual_interface("track_vault_structure_sync")
    async def track_vault_structure(self, structure_data: Dict[str, Any]) -> bool:
        """Track vault structure changes"""
        await self.ensure_connected()
        self._stats["total_requests"] += 1
        
        try:
            result = await self.memory_manager.track_vault_structure(structure_data)
            
            if result:
                # Invalidate vault entities cache
                self.clear_cache("vault_entities")
            
            return result
            
        except Exception as e:
            logger.error(f"Error tracking vault structure: {e}")
            self._stats["errors"] += 1
            return False
    
    @dual_interface("get_vault_entities_sync")
    async def get_vault_entities(self, use_cache: bool = True) -> Dict[str, List[str]]:
        """Get vault entities with caching"""
        await self.ensure_connected()
        self._stats["total_requests"] += 1
        
        # Check cache first
        cache_key = self._cache_key("vault_entities", self.vault_path or "default")
        if use_cache and self.cache_config.enable_vault_entities:
            cached_value = self._get_cached(cache_key)
            if cached_value is not None:
                return cached_value
        
        try:
            entities = await self.memory_manager.get_vault_entities()
            
            # Cache the result
            if self.cache_config.enable_vault_entities:
                self._set_cached(cache_key, entities)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error getting vault entities: {e}")
            self._stats["errors"] += 1
            return {"notes": [], "folders": [], "tags": []}
    
    # Research tracking
    
    @dual_interface("track_research_topic_sync")
    async def track_research_topic(self, topic: str, domain: Optional[str] = None,
                                 confidence: float = 1.0) -> bool:
        """Track research topic interest"""
        await self.ensure_connected()
        self._stats["total_requests"] += 1
        
        try:
            result = await self.memory_manager.track_research_topic(topic, domain, confidence)
            
            if result:
                # Invalidate research interests cache
                self.clear_cache("research_interests")
            
            return result
            
        except Exception as e:
            logger.error(f"Error tracking research topic: {e}")
            self._stats["errors"] += 1
            return False
    
    @dual_interface("get_research_interests_sync")
    async def get_research_interests(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Get research interests with caching"""
        await self.ensure_connected()
        self._stats["total_requests"] += 1
        
        # Check cache first
        cache_key = self._cache_key("research_interests", self.user_id)
        if use_cache and self.cache_config.enable_research_interests:
            cached_value = self._get_cached(cache_key)
            if cached_value is not None:
                return cached_value
        
        try:
            interests = await self.memory_manager.get_research_interests()
            
            # Cache the result
            if self.cache_config.enable_research_interests:
                self._set_cached(cache_key, interests)
            
            return interests
            
        except Exception as e:
            logger.error(f"Error getting research interests: {e}")
            self._stats["errors"] += 1
            return []
    
    # Pattern analysis
    
    @dual_interface("get_user_patterns_sync")
    async def get_user_patterns(self, use_cache: bool = True) -> Dict[str, Any]:
        """Get comprehensive user patterns with caching"""
        await self.ensure_connected()
        self._stats["total_requests"] += 1
        
        # Check cache first
        cache_key = self._cache_key("user_patterns", self.user_id)
        if use_cache:
            cached_value = self._get_cached(cache_key)
            if cached_value is not None:
                return cached_value
        
        try:
            patterns = await self.memory_manager.get_user_patterns()
            
            # Cache the result
            self._set_cached(cache_key, patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting user patterns: {e}")
            self._stats["errors"] += 1
            return {
                "frequent_actions": [],
                "preferred_tools": [],
                "research_interests": [],
                "time_patterns": {},
                "productivity_metrics": {}
            }
    
    # Utility methods
    
    @dual_interface("get_stats_sync")
    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        await self.ensure_connected()
        
        # Get memory stats
        memory_stats = await self.memory_manager.get_memory_stats()
        
        # Calculate cache stats
        with self._cache_lock:
            cache_size = len(self._cache_data)
            cache_memory = sum(len(str(v[0])) for v in self._cache_data.values())
        
        hit_rate = 0.0
        if self._stats["cache_hits"] + self._stats["cache_misses"] > 0:
            hit_rate = self._stats["cache_hits"] / (self._stats["cache_hits"] + self._stats["cache_misses"])
        
        return {
            "service": {
                "total_requests": self._stats["total_requests"],
                "errors": self._stats["errors"],
                "error_rate": self._stats["errors"] / max(self._stats["total_requests"], 1),
                "cache_hits": self._stats["cache_hits"],
                "cache_misses": self._stats["cache_misses"],
                "cache_hit_rate": hit_rate,
                "cache_size": cache_size,
                "cache_memory_bytes": cache_memory
            },
            "memory": memory_stats
        }
    
    @dual_interface("export_data_sync")
    async def export_data(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Export all memory data"""
        await self.ensure_connected()
        return await self.memory_manager.export_memory_data(output_path)
    
    # Sync execution helper
    
    def _run_async(self, coro):
        """Run async coroutine in sync context"""
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in a running loop, schedule the coroutine
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                # If no loop is running, use asyncio.run
                return asyncio.run(coro)
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(coro)
    
    # Create sync versions of all dual interface methods
    def record_preference_sync(self, preference_name: str, preference_value: str, 
                             strength: str = "moderate") -> bool:
        """Synchronous version of record_preference"""
        return self._run_async(self.record_preference(preference_name, preference_value, strength))
    
    def get_preferences_sync(self, category: Optional[str] = None, 
                           use_cache: bool = True) -> Dict[str, Any]:
        """Synchronous version of get_preferences"""
        return self._run_async(self.get_preferences(category, use_cache))
    
    def track_action_sync(self, action_name: str, success: bool = True,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Synchronous version of track_action"""
        return self._run_async(self.track_action(action_name, success, metadata))
    
    def track_tool_usage_sync(self, tool_name: str, category: str, 
                            success: bool, duration: Optional[float] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Synchronous version of track_tool_usage"""
        return self._run_async(self.track_tool_usage(tool_name, category, success, duration, metadata))
    
    def get_tool_recommendations_sync(self, context: Optional[str] = None,
                                    use_cache: bool = True) -> List[Dict[str, Any]]:
        """Synchronous version of get_tool_recommendations"""
        return self._run_async(self.get_tool_recommendations(context, use_cache))
    
    def track_vault_structure_sync(self, structure_data: Dict[str, Any]) -> bool:
        """Synchronous version of track_vault_structure"""
        return self._run_async(self.track_vault_structure(structure_data))
    
    def get_vault_entities_sync(self, use_cache: bool = True) -> Dict[str, List[str]]:
        """Synchronous version of get_vault_entities"""
        return self._run_async(self.get_vault_entities(use_cache))
    
    def track_research_topic_sync(self, topic: str, domain: Optional[str] = None,
                                confidence: float = 1.0) -> bool:
        """Synchronous version of track_research_topic"""
        return self._run_async(self.track_research_topic(topic, domain, confidence))
    
    def get_research_interests_sync(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Synchronous version of get_research_interests"""
        return self._run_async(self.get_research_interests(use_cache))
    
    def get_user_patterns_sync(self, use_cache: bool = True) -> Dict[str, Any]:
        """Synchronous version of get_user_patterns"""
        return self._run_async(self.get_user_patterns(use_cache))
    
    def get_stats_sync(self) -> Dict[str, Any]:
        """Synchronous version of get_stats"""
        return self._run_async(self.get_stats())
    
    def export_data_sync(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous version of export_data"""
        return self._run_async(self.export_data(output_path))
    
    # Context managers for sync usage
    
    @contextmanager
    def sync_connection(self):
        """Synchronous context manager for memory service"""
        if not self._is_connected:
            raise MemoryServiceNotInitializedError(
                "MemoryService not initialized. Call initialize_sync() first."
            )
        try:
            yield self
        finally:
            pass
    
    # Cleanup
    
    async def close(self):
        """Close the memory service and cleanup resources"""
        if self._is_connected:
            try:
                await self.memory_manager.close()
                self._is_connected = False
                logger.info("MemoryService closed")
            except Exception as e:
                logger.error(f"Error closing MemoryService: {e}")
    
    def close_sync(self):
        """Synchronous version of close"""
        return self._run_async(self.close())
    
    # Vector embeddings initialization
    
    async def _initialize_vector_components(self, vector_config: Optional[EmbeddingConfig] = None):
        """Initialize vector embeddings components."""
        try:
            if not VECTOR_EMBEDDINGS_AVAILABLE:
                logger.warning("Vector embeddings not available - skipping initialization")
                return
            
            # Use provided config or default
            self.vector_config = vector_config or DEFAULT_CONFIG
            
            # Initialize core services
            self.embedding_service = EmbeddingService(self.vector_config)
            self.semantic_search = SemanticSearch(self.vector_config)
            self.hybrid_search = HybridSearch(self.vector_config)
            self.ranking_engine = RankingEngine()
            self.model_manager = ModelManager(self.vector_config)
            
            logger.info("Vector embeddings components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector components: {e}")
            self._vector_enabled = False
    
    # Vector search methods
    
    @dual_interface("semantic_search_sync")
    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector embeddings.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            filters: Optional metadata filters
            use_cache: Whether to use cached results
            
        Returns:
            List of search results with similarity scores
        """
        await self.ensure_connected()
        
        if not self._vector_enabled or not self.semantic_search:
            logger.warning("Vector embeddings not available - semantic search unavailable")
            return []
        
        self._stats["total_requests"] += 1
        
        # Check cache
        cache_key = self._cache_key("semantic_search", query, limit, similarity_threshold, str(filters))
        if use_cache and self.cache_config.enable_vector_search_cache:
            cached_value = self._get_cached(cache_key, ttl_seconds=self.cache_config.vector_cache_ttl_seconds)
            if cached_value is not None:
                self._stats["cache_hits"] += 1
                return cached_value
        
        try:
            # Perform semantic search
            search_results = self.semantic_search.search(
                query=query,
                limit=limit,
                similarity_threshold=similarity_threshold,
                filters=filters
            )
            
            # Convert to serializable format
            results = [result.to_dict() for result in search_results.results]
            
            # Cache results
            if self.cache_config.enable_vector_search_cache:
                self._set_cached(cache_key, results, ttl_seconds=self.cache_config.vector_cache_ttl_seconds)
            
            self._stats["cache_misses"] += 1
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            self._stats["errors"] += 1
            return []
    
    @dual_interface("hybrid_search_sync")
    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        semantic_weight: float = 0.6,
        text_weight: float = 0.4,
        fusion_method: str = "weighted_sum",
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and text search.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            semantic_weight: Weight for semantic similarity
            text_weight: Weight for text relevance
            fusion_method: Method to combine scores
            use_cache: Whether to use cached results
            
        Returns:
            List of hybrid search results
        """
        await self.ensure_connected()
        
        if not self._vector_enabled or not self.hybrid_search:
            logger.warning("Vector embeddings not available - hybrid search unavailable")
            return []
        
        self._stats["total_requests"] += 1
        
        # Check cache
        cache_key = self._cache_key("hybrid_search", query, limit, semantic_weight, text_weight, fusion_method)
        if use_cache and self.cache_config.enable_vector_search_cache:
            cached_value = self._get_cached(cache_key, ttl_seconds=self.cache_config.vector_cache_ttl_seconds)
            if cached_value is not None:
                self._stats["cache_hits"] += 1
                return cached_value
        
        try:
            # Perform hybrid search
            search_results = self.hybrid_search.search(
                query=query,
                limit=limit,
                semantic_weight=semantic_weight,
                text_weight=text_weight,
                fusion_method=fusion_method
            )
            
            # Convert to serializable format
            results = [result.to_dict() for result in search_results]
            
            # Cache results
            if self.cache_config.enable_vector_search_cache:
                self._set_cached(cache_key, results, ttl_seconds=self.cache_config.vector_cache_ttl_seconds)
            
            self._stats["cache_misses"] += 1
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            self._stats["errors"] += 1
            return []
    
    @dual_interface("add_document_to_index_sync")
    async def add_document_to_index(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add a document to the vector search index.
        
        Args:
            doc_id: Document identifier
            content: Document content
            metadata: Optional document metadata
            
        Returns:
            bool: True if successfully added
        """
        await self.ensure_connected()
        
        if not self._vector_enabled or not self.hybrid_search:
            logger.warning("Vector embeddings not available - cannot add to index")
            return False
        
        try:
            # Add to hybrid search index (both semantic and text)
            self.hybrid_search.add_document(doc_id, content, metadata)
            
            # Invalidate relevant caches
            self.clear_cache("semantic_search")
            self.clear_cache("hybrid_search")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_id} to index: {e}")
            self._stats["errors"] += 1
            return False
    
    @dual_interface("remove_document_from_index_sync")
    async def remove_document_from_index(self, doc_id: str) -> bool:
        """
        Remove a document from the vector search index.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            bool: True if successfully removed
        """
        await self.ensure_connected()
        
        if not self._vector_enabled or not self.hybrid_search:
            return False
        
        try:
            self.hybrid_search.remove_document(doc_id)
            
            # Invalidate relevant caches
            self.clear_cache("semantic_search")
            self.clear_cache("hybrid_search")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove document {doc_id} from index: {e}")
            self._stats["errors"] += 1
            return False
    
    @dual_interface("get_document_recommendations_sync")
    async def get_document_recommendations(
        self,
        document_id: str,
        limit: int = 5,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get document recommendations based on similarity to a given document.
        
        Args:
            document_id: Reference document ID
            limit: Maximum recommendations
            use_cache: Whether to use cached results
            
        Returns:
            List of recommended documents
        """
        await self.ensure_connected()
        
        if not self._vector_enabled or not self.semantic_search:
            return []
        
        cache_key = self._cache_key("doc_recommendations", document_id, limit)
        if use_cache and self.cache_config.enable_vector_search_cache:
            cached_value = self._get_cached(cache_key, ttl_seconds=self.cache_config.vector_cache_ttl_seconds)
            if cached_value is not None:
                return cached_value
        
        try:
            search_results = self.semantic_search.search_similar_to_document(
                document_id=document_id,
                limit=limit
            )
            
            results = [result.to_dict() for result in search_results.results]
            
            if self.cache_config.enable_vector_search_cache:
                self._set_cached(cache_key, results, ttl_seconds=self.cache_config.vector_cache_ttl_seconds)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get recommendations for document {document_id}: {e}")
            self._stats["errors"] += 1
            return []
    
    @dual_interface("get_vector_stats_sync")
    async def get_vector_stats(self) -> Dict[str, Any]:
        """
        Get vector embeddings statistics.
        
        Returns:
            Dictionary with vector system statistics
        """
        if not self._vector_enabled:
            return {'vector_embeddings_enabled': False}
        
        stats = {'vector_embeddings_enabled': True}
        
        try:
            if self.semantic_search:
                stats['semantic_search'] = self.semantic_search.get_search_stats()
            
            if self.hybrid_search:
                stats['hybrid_search'] = self.hybrid_search.get_search_stats()
            
            if self.ranking_engine:
                stats['ranking_engine'] = self.ranking_engine.get_ranking_stats()
            
            if self.model_manager:
                stats['model_manager'] = self.model_manager.get_model_stats()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get vector stats: {e}")
            return {'vector_embeddings_enabled': True, 'error': str(e)}
    
    def is_vector_enabled(self) -> bool:
        """Check if vector embeddings are enabled."""
        return self._vector_enabled and self.semantic_search is not None


# Global singleton instance
_memory_service_instance: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """
    Get the global memory service singleton instance.
    
    Returns:
        MemoryService: The singleton memory service instance
    """
    global _memory_service_instance
    if _memory_service_instance is None:
        _memory_service_instance = MemoryService()
    return _memory_service_instance


# Convenience functions for direct access

async def track_action(action_name: str, success: bool = True, 
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function to track an action"""
    service = get_memory_service()
    return await service.track_action(action_name, success, metadata)


async def track_tool_usage(tool_name: str, category: str, success: bool,
                         duration: Optional[float] = None, 
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function to track tool usage"""
    service = get_memory_service()
    return await service.track_tool_usage(tool_name, category, success, duration, metadata)


async def get_recommendations(context: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convenience function to get tool recommendations"""
    service = get_memory_service()
    return await service.get_tool_recommendations(context)


# Example usage patterns
if __name__ == "__main__":
    async def example_usage():
        """Example of using the memory service"""
        # Get service instance
        service = get_memory_service()
        
        # Initialize
        await service.initialize("example_user", "/path/to/vault")
        
        # Track an action
        await service.track_action("opened_note", success=True, metadata={"note": "Daily Notes"})
        
        # Track tool usage
        await service.track_tool_usage("note_creator", "creation", True, 1.5)
        
        # Get recommendations
        recommendations = await service.get_tool_recommendations()
        print(f"Tool recommendations: {recommendations}")
        
        # Get user patterns
        patterns = await service.get_user_patterns()
        print(f"User patterns: {patterns}")
        
        # Get stats
        stats = await service.get_stats()
        print(f"Service stats: {stats}")
        
        # Close
        await service.close()
    
    # Run example
    # asyncio.run(example_usage())