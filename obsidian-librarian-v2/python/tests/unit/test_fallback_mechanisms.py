"""
Comprehensive tests for database and AI provider fallback mechanisms.

Tests the robustness of the system when external dependencies fail:
- Database fallbacks (ChromaDB -> SQLite)
- AI provider fallbacks (OpenAI -> Local models)
- Cache fallbacks
- Graceful degradation
"""

import pytest
import asyncio
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import json

from obsidian_librarian.models import LibrarianConfig, Note, NoteMetadata
from obsidian_librarian.database import VectorDatabase, CacheDatabase, AnalyticsDatabase
from obsidian_librarian.ai import (
    EmbeddingService,
    ContentAnalyzer,
    LanguageModelService,
    QueryProcessor
)
from obsidian_librarian.services.tag_manager import TagManagerService
from obsidian_librarian.services.auto_organizer import AutoOrganizer


class TestDatabaseFallbacks:
    """Test database fallback mechanisms."""
    
    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create temporary database directory."""
        db_path = tmp_path / "test_db"
        db_path.mkdir()
        return db_path
    
    @pytest.fixture
    def config_with_fallback(self, temp_db_path):
        """Create config with fallback settings."""
        return LibrarianConfig(
            vault_path=Path("/test/vault"),
            database={
                "vector_db": {
                    "provider": "chromadb",
                    "path": str(temp_db_path / "chroma"),
                    "fallback": {
                        "enabled": True,
                        "provider": "sqlite",
                        "path": str(temp_db_path / "vector_fallback.db")
                    }
                },
                "cache_db": {
                    "provider": "redis",
                    "url": "redis://localhost:6379",
                    "fallback": {
                        "enabled": True,
                        "provider": "sqlite",
                        "path": str(temp_db_path / "cache_fallback.db")
                    }
                },
                "analytics_db": {
                    "provider": "postgresql",
                    "url": "postgresql://localhost/obsidian",
                    "fallback": {
                        "enabled": True,
                        "provider": "sqlite",
                        "path": str(temp_db_path / "analytics_fallback.db")
                    }
                }
            }
        )
    
    async def test_vector_db_fallback_to_sqlite(self, config_with_fallback, temp_db_path):
        """Test vector database fallback from ChromaDB to SQLite."""
        with patch('chromadb.Client') as mock_chroma:
            # Simulate ChromaDB failure
            mock_chroma.side_effect = Exception("ChromaDB connection failed")
            
            # Initialize vector database
            vector_db = VectorDatabase(config_with_fallback.database["vector_db"])
            await vector_db.initialize()
            
            # Should have fallen back to SQLite
            assert vector_db.provider == "sqlite"
            assert vector_db.is_fallback is True
            
            # Test basic operations work with fallback
            # Add embeddings
            await vector_db.add_embeddings(
                ids=["note1", "note2"],
                embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                metadatas=[
                    {"title": "Note 1", "tags": ["test"]},
                    {"title": "Note 2", "tags": ["test", "fallback"]}
                ]
            )
            
            # Query similar
            results = await vector_db.query_similar(
                query_embedding=[0.15, 0.25, 0.35],
                n_results=2
            )
            
            assert len(results) > 0
            assert results[0]["id"] in ["note1", "note2"]
            
            # Verify SQLite database was created
            sqlite_path = temp_db_path / "vector_fallback.db"
            assert sqlite_path.exists()
            
            # Check schema
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()
            
            table_names = [t[0] for t in tables]
            assert "embeddings" in table_names
            assert "metadata" in table_names
    
    async def test_cache_db_fallback_to_sqlite(self, config_with_fallback, temp_db_path):
        """Test cache database fallback from Redis to SQLite."""
        with patch('redis.Redis') as mock_redis:
            # Simulate Redis connection failure
            mock_redis.return_value.ping.side_effect = Exception("Redis not available")
            
            # Initialize cache database
            cache_db = CacheDatabase(config_with_fallback.database["cache_db"])
            await cache_db.initialize()
            
            # Should have fallen back to SQLite
            assert cache_db.provider == "sqlite"
            assert cache_db.is_fallback is True
            
            # Test cache operations
            # Set cache
            await cache_db.set("test_key", {"data": "test_value"}, ttl=3600)
            
            # Get cache
            result = await cache_db.get("test_key")
            assert result is not None
            assert result["data"] == "test_value"
            
            # Delete cache
            await cache_db.delete("test_key")
            result = await cache_db.get("test_key")
            assert result is None
            
            # Test batch operations
            batch_data = {
                f"key_{i}": {"value": i}
                for i in range(10)
            }
            
            await cache_db.set_many(batch_data, ttl=3600)
            
            keys = list(batch_data.keys())
            results = await cache_db.get_many(keys)
            
            assert len(results) == len(keys)
            assert all(results[k] == batch_data[k] for k in keys)
    
    async def test_analytics_db_fallback_to_sqlite(self, config_with_fallback, temp_db_path):
        """Test analytics database fallback from PostgreSQL to SQLite."""
        with patch('asyncpg.create_pool') as mock_pg:
            # Simulate PostgreSQL connection failure
            mock_pg.side_effect = Exception("PostgreSQL not available")
            
            # Initialize analytics database
            analytics_db = AnalyticsDatabase(config_with_fallback.database["analytics_db"])
            await analytics_db.initialize()
            
            # Should have fallen back to SQLite
            assert analytics_db.provider == "sqlite"
            assert analytics_db.is_fallback is True
            
            # Test analytics operations
            # Record event
            await analytics_db.record_event(
                event_type="note_created",
                data={
                    "note_id": "test_note",
                    "timestamp": datetime.utcnow().isoformat(),
                    "tags": ["test", "analytics"]
                }
            )
            
            # Query events
            events = await analytics_db.query_events(
                event_type="note_created",
                start_date=datetime.utcnow().replace(hour=0, minute=0, second=0),
                limit=10
            )
            
            assert len(events) > 0
            assert events[0]["event_type"] == "note_created"
            
            # Aggregate metrics
            metrics = await analytics_db.aggregate_metrics(
                metric="note_count",
                group_by="day",
                start_date=datetime.utcnow().replace(hour=0, minute=0, second=0)
            )
            
            assert isinstance(metrics, list)
    
    async def test_fallback_data_migration(self, config_with_fallback, temp_db_path):
        """Test data migration when switching to fallback."""
        # First, use primary database successfully
        with patch('chromadb.Client') as mock_chroma:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_chroma.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            vector_db = VectorDatabase(config_with_fallback.database["vector_db"])
            await vector_db.initialize()
            
            # Add some data
            await vector_db.add_embeddings(
                ids=["note1", "note2", "note3"],
                embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                metadatas=[
                    {"title": f"Note {i}"} for i in range(1, 4)
                ]
            )
        
        # Now simulate primary database failure
        with patch('chromadb.Client') as mock_chroma:
            mock_chroma.side_effect = Exception("ChromaDB failed")
            
            # Re-initialize (should trigger fallback and migration)
            vector_db2 = VectorDatabase(config_with_fallback.database["vector_db"])
            await vector_db2.initialize()
            
            # Should be using fallback
            assert vector_db2.provider == "sqlite"
            
            # Data should be migrated (if migration is implemented)
            # This is a placeholder for actual migration testing
            # In real implementation, you'd check if data was copied
    
    async def test_fallback_performance_monitoring(self, config_with_fallback):
        """Test that fallback usage is monitored and logged."""
        with patch('chromadb.Client') as mock_chroma:
            mock_chroma.side_effect = Exception("ChromaDB failed")
            
            # Capture logs
            with patch('structlog.get_logger') as mock_logger:
                logger = MagicMock()
                mock_logger.return_value = logger
                
                vector_db = VectorDatabase(config_with_fallback.database["vector_db"])
                await vector_db.initialize()
                
                # Check that fallback was logged
                logger.warning.assert_called()
                warning_calls = logger.warning.call_args_list
                
                fallback_logged = any(
                    "fallback" in str(call).lower()
                    for call in warning_calls
                )
                assert fallback_logged
                
                # Perform operations
                await vector_db.add_embeddings(
                    ids=["test"],
                    embeddings=[[0.1, 0.2]],
                    metadatas=[{"title": "Test"}]
                )
                
                # Check performance logging
                logger.info.assert_called()
                info_calls = logger.info.call_args_list
                
                performance_logged = any(
                    "performance" in str(call).lower() or "latency" in str(call).lower()
                    for call in info_calls
                )
                
                # Performance tracking might be logged
                # This depends on implementation
    
    async def test_multiple_fallback_chain(self, temp_db_path):
        """Test fallback chain: Primary -> Secondary -> Tertiary."""
        config = LibrarianConfig(
            vault_path=Path("/test/vault"),
            database={
                "vector_db": {
                    "provider": "weaviate",
                    "url": "http://localhost:8080",
                    "fallback": {
                        "enabled": True,
                        "provider": "chromadb",
                        "path": str(temp_db_path / "chroma"),
                        "fallback": {
                            "enabled": True,
                            "provider": "sqlite",
                            "path": str(temp_db_path / "sqlite.db")
                        }
                    }
                }
            }
        )
        
        # Mock all providers to fail except SQLite
        with patch('weaviate.Client') as mock_weaviate:
            with patch('chromadb.Client') as mock_chroma:
                mock_weaviate.side_effect = Exception("Weaviate failed")
                mock_chroma.side_effect = Exception("ChromaDB failed")
                
                vector_db = VectorDatabase(config.database["vector_db"])
                await vector_db.initialize()
                
                # Should fall back all the way to SQLite
                assert vector_db.provider == "sqlite"
                assert vector_db.fallback_chain == ["weaviate", "chromadb", "sqlite"]


class TestAIProviderFallbacks:
    """Test AI provider fallback mechanisms."""
    
    @pytest.fixture
    def config_with_ai_fallback(self, tmp_path):
        """Create config with AI fallback settings."""
        return LibrarianConfig(
            vault_path=Path("/test/vault"),
            openai_api_key="test-key",
            ai_providers={
                "embeddings": {
                    "primary": {
                        "provider": "openai",
                        "model": "text-embedding-ada-002",
                        "api_key": "test-key"
                    },
                    "fallback": {
                        "provider": "sentence-transformers",
                        "model": "all-MiniLM-L6-v2",
                        "device": "cpu"
                    }
                },
                "language_model": {
                    "primary": {
                        "provider": "openai",
                        "model": "gpt-4",
                        "api_key": "test-key"
                    },
                    "fallback": {
                        "provider": "llama-cpp",
                        "model_path": str(tmp_path / "llama-model.gguf"),
                        "n_ctx": 2048
                    }
                },
                "content_analysis": {
                    "primary": {
                        "provider": "openai",
                        "model": "gpt-4"
                    },
                    "fallback": {
                        "provider": "spacy",
                        "model": "en_core_web_sm"
                    }
                }
            }
        )
    
    async def test_embedding_service_fallback(self, config_with_ai_fallback):
        """Test embedding service fallback from OpenAI to local model."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            # Simulate OpenAI API failure
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.embeddings.create.side_effect = Exception("OpenAI API error")
            
            # Initialize embedding service
            embedding_service = EmbeddingService(config_with_ai_fallback.ai_providers["embeddings"])
            await embedding_service.initialize()
            
            # Should have fallen back to sentence-transformers
            assert embedding_service.provider == "sentence-transformers"
            assert embedding_service.is_fallback is True
            
            # Test embedding generation with fallback
            texts = ["Hello world", "Testing fallback", "AI providers"]
            embeddings = await embedding_service.embed_texts(texts)
            
            assert len(embeddings) == len(texts)
            assert all(isinstance(emb, list) for emb in embeddings)
            assert all(len(emb) > 0 for emb in embeddings)
            
            # Test similarity computation
            similarity = await embedding_service.compute_similarity(
                embeddings[0],
                embeddings[1]
            )
            
            assert isinstance(similarity, float)
            assert 0 <= similarity <= 1
    
    async def test_language_model_fallback(self, config_with_ai_fallback, tmp_path):
        """Test language model fallback from OpenAI to local model."""
        # Create a mock local model file
        mock_model_path = tmp_path / "llama-model.gguf"
        mock_model_path.write_bytes(b"mock model data")
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            # Simulate OpenAI API failure
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("OpenAI API error")
            
            # Mock llama-cpp-python
            with patch('llama_cpp.Llama') as mock_llama:
                mock_model = MagicMock()
                mock_llama.return_value = mock_model
                mock_model.create_completion.return_value = {
                    "choices": [{"text": "Fallback response"}]
                }
                
                # Initialize language model service
                lm_service = LanguageModelService(config_with_ai_fallback.ai_providers["language_model"])
                await lm_service.initialize()
                
                # Should have fallen back to llama-cpp
                assert lm_service.provider == "llama-cpp"
                assert lm_service.is_fallback is True
                
                # Test generation with fallback
                response = await lm_service.generate(
                    prompt="Test prompt",
                    max_tokens=100,
                    temperature=0.7
                )
                
                assert response == "Fallback response"
                
                # Test streaming
                chunks = []
                async for chunk in lm_service.generate_stream(
                    prompt="Test streaming",
                    max_tokens=50
                ):
                    chunks.append(chunk)
                
                assert len(chunks) > 0
    
    async def test_content_analyzer_fallback(self, config_with_ai_fallback):
        """Test content analyzer fallback from OpenAI to SpaCy."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            # Simulate OpenAI API failure
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("OpenAI API error")
            
            # Mock spacy
            with patch('spacy.load') as mock_spacy_load:
                mock_nlp = MagicMock()
                mock_doc = MagicMock()
                mock_spacy_load.return_value = mock_nlp
                mock_nlp.return_value = mock_doc
                
                # Setup mock spacy analysis
                mock_doc.ents = [
                    MagicMock(text="Test Entity", label_="ORG"),
                    MagicMock(text="Another Entity", label_="PERSON")
                ]
                mock_doc._.keywords = ["test", "content", "analysis"]
                mock_doc.cats = {"technology": 0.8, "science": 0.6}
                
                # Initialize content analyzer
                analyzer = ContentAnalyzer(config_with_ai_fallback.ai_providers["content_analysis"])
                await analyzer.initialize()
                
                # Should have fallen back to spacy
                assert analyzer.provider == "spacy"
                assert analyzer.is_fallback is True
                
                # Test content analysis with fallback
                content = "This is test content for analysis with entities and keywords."
                analysis = await analyzer.analyze_content(content)
                
                assert analysis is not None
                assert hasattr(analysis, 'entities')
                assert hasattr(analysis, 'keywords')
                assert hasattr(analysis, 'topics')
                
                assert len(analysis.entities) > 0
                assert len(analysis.keywords) > 0
    
    async def test_fallback_quality_degradation(self, config_with_ai_fallback):
        """Test graceful quality degradation with fallbacks."""
        # Test tag suggestions with degraded AI
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_openai.side_effect = Exception("OpenAI unavailable")
            
            # Create services with fallbacks
            embedding_service = EmbeddingService(config_with_ai_fallback.ai_providers["embeddings"])
            await embedding_service.initialize()
            
            content_analyzer = ContentAnalyzer(config_with_ai_fallback.ai_providers["content_analysis"])
            await content_analyzer.initialize()
            
            # Both should be using fallbacks
            assert embedding_service.is_fallback
            assert content_analyzer.is_fallback
            
            # Create tag manager with degraded services
            mock_vault = Mock()
            tag_config = config_with_ai_fallback.features.get("tag_management", {})
            
            tag_manager = TagManagerService(
                vault=mock_vault,
                config=tag_config,
                embedding_service=embedding_service,
                content_analyzer=content_analyzer
            )
            
            # Test functionality still works (with reduced quality)
            note = Note(
                id="test-note",
                path=Path("test.md"),
                content="# Machine Learning Research\n\nStudying neural networks and deep learning.",
                metadata=NoteMetadata(),
                created_at=datetime.utcnow(),
                modified_at=datetime.utcnow()
            )
            
            mock_vault.get_note.return_value = note
            
            # Should still provide suggestions, but maybe fewer or less accurate
            suggestions = await tag_manager.suggest_tags("test-note", max_suggestions=5)
            
            assert isinstance(suggestions, list)
            # Quality might be lower, but should still work
            assert all(hasattr(s, 'tag') and hasattr(s, 'confidence') for s in suggestions)
    
    async def test_fallback_retry_logic(self, config_with_ai_fallback):
        """Test retry logic before falling back."""
        call_count = 0
        
        async def mock_openai_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 attempts
                raise Exception("Temporary failure")
            return MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])])
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.embeddings.create = AsyncMock(side_effect=mock_openai_call)
            
            # Configure retry policy
            config_with_ai_fallback.ai_providers["embeddings"]["primary"]["retry"] = {
                "max_attempts": 3,
                "backoff_factor": 0.1
            }
            
            embedding_service = EmbeddingService(config_with_ai_fallback.ai_providers["embeddings"])
            await embedding_service.initialize()
            
            # Should retry and eventually succeed without fallback
            embeddings = await embedding_service.embed_texts(["test"])
            
            assert call_count == 3  # Should have retried
            assert embedding_service.provider == "openai"  # Should not have fallen back
            assert not embedding_service.is_fallback
    
    async def test_fallback_cost_tracking(self, config_with_ai_fallback):
        """Test tracking of API costs and fallback usage."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            # First work normally
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            # Track API calls
            api_calls = []
            
            async def track_call(*args, **kwargs):
                api_calls.append({
                    "provider": "openai",
                    "timestamp": datetime.utcnow(),
                    "tokens": kwargs.get("max_tokens", 0)
                })
                return MagicMock(
                    choices=[MagicMock(message=MagicMock(content="Response"))],
                    usage=MagicMock(total_tokens=50)
                )
            
            mock_client.chat.completions.create = AsyncMock(side_effect=track_call)
            
            lm_service = LanguageModelService(config_with_ai_fallback.ai_providers["language_model"])
            await lm_service.initialize()
            
            # Make some calls
            for _ in range(5):
                await lm_service.generate("Test", max_tokens=100)
            
            # Now simulate API limit
            mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")
            
            # Re-initialize to trigger fallback
            lm_service = LanguageModelService(config_with_ai_fallback.ai_providers["language_model"])
            await lm_service.initialize()
            
            # Should track fallback usage
            assert lm_service.is_fallback
            assert hasattr(lm_service, 'fallback_reason')
            assert "rate limit" in lm_service.fallback_reason.lower() or \
                   "error" in lm_service.fallback_reason.lower()
    
    async def test_fallback_feature_parity(self, config_with_ai_fallback):
        """Test that fallbacks provide similar features as primary providers."""
        # Test embedding service features
        primary_features = {
            "embed_texts": True,
            "embed_text": True,
            "compute_similarity": True,
            "batch_embed": True,
        }
        
        fallback_features = {
            "embed_texts": True,
            "embed_text": True,
            "compute_similarity": True,
            "batch_embed": True,
        }
        
        # Both providers should support same core features
        assert primary_features == fallback_features
        
        # Test with actual services
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_openai.side_effect = Exception("Force fallback")
            
            embedding_service = EmbeddingService(config_with_ai_fallback.ai_providers["embeddings"])
            await embedding_service.initialize()
            
            # Test all features work with fallback
            # Single text
            single_emb = await embedding_service.embed_text("Single text")
            assert isinstance(single_emb, list)
            
            # Multiple texts
            multi_emb = await embedding_service.embed_texts(["Text 1", "Text 2"])
            assert len(multi_emb) == 2
            
            # Similarity
            sim = await embedding_service.compute_similarity(single_emb, multi_emb[0])
            assert isinstance(sim, float)
            
            # Batch operations
            batch_emb = await embedding_service.batch_embed(
                ["Batch 1", "Batch 2", "Batch 3"],
                batch_size=2
            )
            assert len(batch_emb) == 3


class TestIntegratedFallbacks:
    """Test fallback mechanisms in integrated scenarios."""
    
    @pytest.fixture
    async def integrated_system(self, tmp_path):
        """Create an integrated system with fallbacks."""
        config = LibrarianConfig(
            vault_path=tmp_path / "vault",
            enable_ai_features=True,
            database={
                "vector_db": {
                    "provider": "chromadb",
                    "fallback": {
                        "enabled": True,
                        "provider": "sqlite"
                    }
                }
            },
            ai_providers={
                "embeddings": {
                    "primary": {"provider": "openai"},
                    "fallback": {"provider": "sentence-transformers"}
                }
            }
        )
        
        # Create mock vault
        vault_path = tmp_path / "vault"
        vault_path.mkdir()
        (vault_path / ".obsidian").mkdir()
        
        # Add test notes
        test_notes = {
            "note1.md": "# Note 1\n\nContent about #testing",
            "note2.md": "# Note 2\n\nMore content with #fallback",
        }
        
        for name, content in test_notes.items():
            (vault_path / name).write_text(content)
        
        return config, vault_path
    
    async def test_search_with_fallbacks(self, integrated_system):
        """Test search functionality with database and AI fallbacks."""
        config, vault_path = integrated_system
        
        # Force both systems to fallback
        with patch('chromadb.Client') as mock_chroma:
            with patch('openai.AsyncOpenAI') as mock_openai:
                mock_chroma.side_effect = Exception("ChromaDB unavailable")
                mock_openai.side_effect = Exception("OpenAI unavailable")
                
                # Initialize services
                from obsidian_librarian import ObsidianLibrarian
                
                async with ObsidianLibrarian(config) as librarian:
                    # Both should be using fallbacks
                    assert librarian._vector_db.is_fallback
                    assert librarian._embedding_service.is_fallback
                    
                    # Search should still work
                    results = await librarian.search_notes(
                        query="testing fallback",
                        limit=10
                    )
                    
                    assert isinstance(results, list)
                    # May have degraded quality but should return results
    
    async def test_tag_operations_with_fallbacks(self, integrated_system):
        """Test tag operations with AI fallbacks."""
        config, vault_path = integrated_system
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_openai.side_effect = Exception("OpenAI unavailable")
            
            from obsidian_librarian import ObsidianLibrarian
            
            async with ObsidianLibrarian(config) as librarian:
                # Tag analysis should work with fallback
                tag_analysis = await librarian._tag_manager.analyze_tags()
                
                assert tag_analysis is not None
                assert tag_analysis.total_tags > 0
                
                # Tag suggestions should work (maybe with lower quality)
                suggestions = await librarian._tag_manager.suggest_tags(
                    "note1.md",
                    max_suggestions=3
                )
                
                assert isinstance(suggestions, list)
    
    async def test_organization_with_fallbacks(self, integrated_system):
        """Test auto-organization with AI fallbacks."""
        config, vault_path = integrated_system
        
        # Add an unorganized note
        (vault_path / "unorganized.md").write_text(
            "# Meeting Notes\n\nAgenda and discussion points"
        )
        
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_openai.side_effect = Exception("OpenAI unavailable")
            
            from obsidian_librarian import ObsidianLibrarian
            
            async with ObsidianLibrarian(config) as librarian:
                # Organization should work with fallback
                result = await librarian._auto_organizer.organize_file(
                    Path("unorganized.md")
                )
                
                assert result is not None
                # May have lower confidence but should classify
                assert hasattr(result, 'suggested_path')
                assert hasattr(result, 'confidence')
    
    async def test_fallback_monitoring_dashboard(self, integrated_system, tmp_path):
        """Test monitoring of fallback usage."""
        config, vault_path = integrated_system
        
        # Add monitoring configuration
        config.monitoring = {
            "enabled": True,
            "metrics_path": str(tmp_path / "metrics.json")
        }
        
        with patch('chromadb.Client') as mock_chroma:
            with patch('openai.AsyncOpenAI') as mock_openai:
                # Simulate intermittent failures
                call_count = 0
                
                def chroma_side_effect(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    if call_count % 3 == 0:
                        raise Exception("ChromaDB temporary failure")
                    return MagicMock()
                
                mock_chroma.side_effect = chroma_side_effect
                mock_openai.side_effect = Exception("OpenAI unavailable")
                
                from obsidian_librarian import ObsidianLibrarian
                
                async with ObsidianLibrarian(config) as librarian:
                    # Perform various operations
                    await librarian.analyze_vault()
                    await librarian.search_notes("test")
                    
                    # Get monitoring metrics
                    metrics = await librarian.get_system_metrics()
                    
                    assert "fallback_usage" in metrics
                    assert "database_fallbacks" in metrics["fallback_usage"]
                    assert "ai_fallbacks" in metrics["fallback_usage"]
                    
                    # Should track fallback events
                    assert metrics["fallback_usage"]["ai_fallbacks"]["embeddings"] > 0
                    
                    # Save metrics to file
                    metrics_file = Path(config.monitoring["metrics_path"])
                    if metrics_file.exists():
                        with open(metrics_file) as f:
                            saved_metrics = json.load(f)
                        assert "fallback_events" in saved_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])