"""
Unit tests for database components.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from obsidian_librarian.database import DatabaseManager, DatabaseConfig
from obsidian_librarian.database.analytics import AnalyticsDB
from obsidian_librarian.database.cache import CacheDB
from obsidian_librarian.database.vector import VectorDB


class TestDatabaseConfig:
    """Test database configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DatabaseConfig()
        
        assert config.analytics_threads == 4
        assert config.vector_dimension == 1536
        assert config.cache_ttl_seconds == 3600
        assert config.connection_timeout == 30
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = DatabaseConfig(
            analytics_threads=8,
            vector_dimension=768,
            cache_ttl_seconds=7200
        )
        
        assert config.analytics_threads == 8
        assert config.vector_dimension == 768
        assert config.cache_ttl_seconds == 7200


class TestAnalyticsDB:
    """Test analytics database operations."""
    
    @pytest.fixture
    async def analytics_db(self):
        """Create analytics database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(analytics_path=Path(tmpdir) / "test.db")
            db = AnalyticsDB(config)
            await db.initialize()
            yield db
            await db.close()
    
    @pytest.mark.asyncio
    async def test_initialization(self, analytics_db):
        """Test database initialization."""
        assert analytics_db.connection is not None
        assert await analytics_db.health_check()
    
    @pytest.mark.asyncio
    async def test_note_metrics_insertion(self, analytics_db):
        """Test inserting note metrics."""
        metrics = {
            "file_path": "test.md",
            "title": "Test Note",
            "word_count": 100,
            "character_count": 500,
            "link_count": 3,
            "tag_count": 2,
        }
        
        await analytics_db.insert_note_metrics("test_note", metrics)
        
        # Verify insertion
        result = await analytics_db._execute_query(
            "SELECT title, word_count FROM note_metrics WHERE note_id = ?",
            ("test_note",)
        )
        
        assert len(result) == 1
        assert result[0][0] == "Test Note"
        assert result[0][1] == 100
    
    @pytest.mark.asyncio
    async def test_vault_stats(self, analytics_db):
        """Test vault statistics."""
        # Insert test data
        metrics = {
            "file_path": "note1.md",
            "word_count": 100,
            "link_count": 2,
        }
        await analytics_db.insert_note_metrics("note1", metrics)
        
        metrics = {
            "file_path": "note2.md", 
            "word_count": 200,
            "link_count": 1,
        }
        await analytics_db.insert_note_metrics("note2", metrics)
        
        # Get stats
        stats = await analytics_db.get_vault_stats()
        
        assert stats["total_notes"] == 2
        assert stats["total_words"] == 300
        assert stats["total_links"] == 3
    
    @pytest.mark.asyncio
    async def test_usage_event_recording(self, analytics_db):
        """Test recording usage events."""
        await analytics_db.record_usage_event(
            event_type="note_created",
            note_id="test_note",
            user_action="create",
            duration_ms=500
        )
        
        # Verify event was recorded
        result = await analytics_db._execute_query(
            "SELECT event_type, note_id FROM usage_events WHERE note_id = ?",
            ("test_note",)
        )
        
        assert len(result) == 1
        assert result[0][0] == "note_created"
        assert result[0][1] == "test_note"


class TestCacheDB:
    """Test cache database operations."""
    
    @pytest.fixture
    async def cache_db(self):
        """Create cache database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(
                cache_url="redis://invalid:6379/0",  # Force fallback to SQLite
                cache_local_fallback=True,
                cache_local_path=Path(tmpdir) / "cache.db"
            )
            db = CacheDB(config)
            await db.initialize()
            yield db
            await db.close()
    
    @pytest.mark.asyncio
    async def test_initialization(self, cache_db):
        """Test cache database initialization."""
        assert not cache_db.use_redis  # Should fall back to SQLite
        assert cache_db.sqlite_conn is not None
        assert await cache_db.health_check()
    
    @pytest.mark.asyncio
    async def test_basic_operations(self, cache_db):
        """Test basic cache operations."""
        # Set value
        await cache_db.set("test_key", "test_value")
        
        # Check existence
        assert await cache_db.exists("test_key")
        
        # Get value
        value = await cache_db.get("test_key")
        assert value == "test_value"
        
        # Delete value
        await cache_db.delete("test_key")
        assert not await cache_db.exists("test_key")
    
    @pytest.mark.asyncio
    async def test_complex_data_types(self, cache_db):
        """Test caching complex data types."""
        test_data = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "number": 42,
            "boolean": True,
        }
        
        await cache_db.set("complex_data", test_data)
        retrieved = await cache_db.get("complex_data")
        
        assert retrieved == test_data
    
    @pytest.mark.asyncio
    async def test_research_cache(self, cache_db):
        """Test research-specific cache operations."""
        query = "test query"
        result = {"findings": ["result1", "result2"]}
        
        await cache_db.cache_research_result(query, result)
        cached = await cache_db.get_cached_research(query)
        
        assert cached == result
    
    @pytest.mark.asyncio
    async def test_embedding_cache(self, cache_db):
        """Test embedding cache operations."""
        content_hash = "abc123"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        await cache_db.cache_embedding(content_hash, embedding)
        cached = await cache_db.get_cached_embedding(content_hash)
        
        assert cached == embedding


class TestVectorDB:
    """Test vector database operations."""
    
    @pytest.fixture
    async def vector_db(self):
        """Create vector database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(
                vector_url="http://invalid:6333",  # Force local mode
                vector_local_path=Path(tmpdir) / "vector_db",
                vector_dimension=384  # Smaller for testing
            )
            
            # Mock Qdrant client
            with patch('obsidian_librarian.database.vector.QdrantClient') as mock_client:
                mock_instance = AsyncMock()
                mock_client.return_value = mock_instance
                
                # Mock collection operations
                mock_instance.get_collections.return_value = []
                mock_instance.get_collection.side_effect = Exception("Collection not found")
                mock_instance.create_collection.return_value = None
                mock_instance.create_payload_index.return_value = None
                mock_instance.upsert.return_value = None
                mock_instance.search.return_value = []
                
                db = VectorDB(config)
                await db.initialize()
                yield db, mock_instance
                await db.close()
    
    @pytest.mark.asyncio
    async def test_initialization(self, vector_db):
        """Test vector database initialization."""
        db, mock_client = vector_db
        assert db.client is not None
        assert await db.health_check()
        
        # Verify collection creation was called
        mock_client.create_collection.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_embedding_upsert(self, vector_db):
        """Test upserting embeddings."""
        db, mock_client = vector_db
        
        embedding = [0.1] * 384  # Match dimension
        metadata = {
            "title": "Test Note",
            "content": "Test content",
            "tags": ["test"],
            "word_count": 10,
        }
        
        await db.upsert_embedding("test_note", embedding, metadata)
        
        # Verify upsert was called
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        points = call_args[0][1]  # Second argument is points list
        
        assert len(points) == 1
        assert points[0].id == "test_note"
        assert points[0].vector == embedding
        assert points[0].payload["title"] == "Test Note"
    
    @pytest.mark.asyncio
    async def test_similarity_search(self, vector_db):
        """Test similarity search."""
        db, mock_client = vector_db
        
        # Mock search results
        mock_result = MagicMock()
        mock_result.id = "similar_note"
        mock_result.score = 0.95
        mock_result.payload = {"title": "Similar Note"}
        mock_client.search.return_value = [mock_result]
        
        query_embedding = [0.2] * 384
        results = await db.search_similar(query_embedding, limit=5)
        
        assert len(results) == 1
        assert results[0]["note_id"] == "similar_note"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"]["title"] == "Similar Note"


class TestDatabaseManager:
    """Test database manager."""
    
    @pytest.fixture
    async def db_manager(self):
        """Create database manager for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(
                analytics_path=Path(tmpdir) / "analytics.db",
                vector_url="http://invalid:6333",
                vector_local_path=Path(tmpdir) / "vector_db",
                cache_url="redis://invalid:6379/0",
                cache_local_path=Path(tmpdir) / "cache.db",
                cache_local_fallback=True,
            )
            
            with patch('obsidian_librarian.database.vector.QdrantClient'):
                manager = DatabaseManager(config)
                await manager.initialize()
                yield manager
                await manager.close()
    
    @pytest.mark.asyncio
    async def test_initialization(self, db_manager):
        """Test database manager initialization."""
        assert db_manager.analytics is not None
        assert db_manager.vector is not None
        assert db_manager.cache is not None
        assert db_manager._initialized
    
    @pytest.mark.asyncio
    async def test_health_check(self, db_manager):
        """Test health check for all databases."""
        health = await db_manager.health_check()
        
        assert "analytics" in health
        assert "vector" in health
        assert "cache" in health
        
        # Analytics and cache should be healthy (SQLite-based)
        assert health["analytics"]
        assert health["cache"]
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test database manager as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(
                analytics_path=Path(tmpdir) / "analytics.db",
                cache_local_path=Path(tmpdir) / "cache.db",
                cache_local_fallback=True,
            )
            
            with patch('obsidian_librarian.database.vector.QdrantClient'):
                async with DatabaseManager(config) as manager:
                    assert manager._initialized
                    assert manager.analytics is not None
                
                # Should be closed after context exit
                assert not manager._initialized
                assert manager.analytics is None


class TestIntegration:
    """Integration tests for database components."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete database workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(
                analytics_path=Path(tmpdir) / "analytics.db",
                cache_local_path=Path(tmpdir) / "cache.db",
                cache_local_fallback=True,
                vector_dimension=384,
            )
            
            with patch('obsidian_librarian.database.vector.QdrantClient'):
                async with DatabaseManager(config) as db_manager:
                    # Test analytics
                    metrics = {
                        "file_path": "test.md",
                        "word_count": 100,
                        "link_count": 2,
                    }
                    await db_manager.analytics.insert_note_metrics("test_note", metrics)
                    
                    # Test cache
                    await db_manager.cache.set("test_key", "test_value")
                    cached_value = await db_manager.cache.get("test_key")
                    assert cached_value == "test_value"
                    
                    # Test vector (mocked)
                    embedding = [0.1] * 384
                    metadata = {"title": "Test"}
                    await db_manager.vector.upsert_embedding("test_note", embedding, metadata)
                    
                    # Test health
                    health = await db_manager.health_check()
                    assert health["analytics"]
                    assert health["cache"]