"""
Integration tests for AI components.

Tests the complete AI pipeline including models, embeddings, 
language models, and content analysis.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obsidian_librarian.ai.models import AIModelManager, AIConfig, AIProvider, ModelCapability
from obsidian_librarian.ai.embeddings import EmbeddingService  
from obsidian_librarian.ai.language_models import LanguageModelService, ChatRequest, ChatMessage
from obsidian_librarian.ai.content_analyzer import ContentAnalyzer, ContentType
from obsidian_librarian.ai.content_summarizer import ContentSummarizer, SummaryConfig, SummaryType
from obsidian_librarian.ai.query_processor import QueryProcessor, QueryIntent
from obsidian_librarian.models import Note, NoteMetadata


class TestAIModelManager:
    """Test AI model management."""
    
    @pytest.fixture
    def config(self):
        """Create test AI config."""
        return AIConfig(
            primary_provider=AIProvider.OPENAI,
            fallback_providers=[AIProvider.OLLAMA],
            temperature=0.7,
            max_tokens=1000,
        )
    
    @pytest.fixture
    def manager(self, config):
        """Create test AI manager."""
        return AIModelManager(config)
    
    def test_config_initialization(self, config):
        """Test configuration initialization."""
        assert config.primary_provider == AIProvider.OPENAI
        assert AIProvider.OLLAMA in config.fallback_providers
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
    
    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert manager.config is not None
        assert isinstance(manager._models, dict)
        assert isinstance(manager._clients, dict)
        assert manager._cost_tracker is not None
    
    @pytest.mark.asyncio
    async def test_model_discovery_mock(self, manager):
        """Test model discovery with mocked clients."""
        # Mock OpenAI client
        mock_openai_client = Mock()
        mock_models_response = Mock()
        mock_models_response.data = [
            Mock(id="gpt-4", owned_by="openai"),
            Mock(id="text-embedding-ada-002", owned_by="openai"),
        ]
        mock_openai_client.models.list = AsyncMock(return_value=mock_models_response)
        
        manager._clients[AIProvider.OPENAI] = mock_openai_client
        
        # Discover models
        await manager._discover_models()
        
        # Check discovered models
        assert len(manager._models) >= 2
        assert "gpt-4" in manager._models
        assert "text-embedding-ada-002" in manager._models
    
    @pytest.mark.asyncio
    async def test_get_best_model(self, manager):
        """Test getting best model for capability."""
        # Add mock models
        from obsidian_librarian.ai.models import ModelInfo
        
        manager._models["gpt-4"] = ModelInfo(
            name="gpt-4",
            provider=AIProvider.OPENAI,
            capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
            context_length=8192,
            input_cost_per_1k=0.03,
            output_cost_per_1k=0.06,
        )
        
        manager._models["gpt-3.5-turbo"] = ModelInfo(
            name="gpt-3.5-turbo",
            provider=AIProvider.OPENAI,
            capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
            context_length=4096,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
        )
        
        # Test getting best for quality
        best_quality = await manager.get_best_model(ModelCapability.CHAT, prefer_fast=False)
        assert best_quality.name == "gpt-4"
        
        # Test getting best for speed
        best_speed = await manager.get_best_model(ModelCapability.CHAT, prefer_fast=True)
        assert best_speed.name == "gpt-3.5-turbo"
    
    @pytest.mark.asyncio
    async def test_cost_estimation(self, manager):
        """Test cost estimation."""
        # Add mock model
        from obsidian_librarian.ai.models import ModelInfo
        
        manager._models["gpt-4"] = ModelInfo(
            name="gpt-4",
            provider=AIProvider.OPENAI,
            capabilities=[ModelCapability.CHAT],
            context_length=8192,
            input_cost_per_1k=0.03,
            output_cost_per_1k=0.06,
        )
        
        cost = await manager.get_cost_estimate("gpt-4", 1000, 500)
        expected = (1000/1000 * 0.03) + (500/1000 * 0.06)
        assert cost == expected


class TestEmbeddingService:
    """Test embedding service."""
    
    @pytest.fixture
    def ai_manager(self):
        """Create mock AI manager."""
        manager = Mock()
        manager.config = AIConfig()
        
        # Mock model info
        from obsidian_librarian.ai.models import ModelInfo
        manager.get_best_model = AsyncMock(return_value=ModelInfo(
            name="text-embedding-ada-002",
            provider=AIProvider.OPENAI,
            capabilities=[ModelCapability.EMBEDDING],
            context_length=8191,
            embedding_dimensions=1536,
        ))
        
        manager._models = {
            "text-embedding-ada-002": ModelInfo(
                name="text-embedding-ada-002",
                provider=AIProvider.OPENAI,
                capabilities=[ModelCapability.EMBEDDING],
                context_length=8191,
                embedding_dimensions=1536,
            )
        }
        
        return manager
    
    @pytest.fixture
    def embedding_service(self, ai_manager):
        """Create embedding service."""
        return EmbeddingService(ai_manager)
    
    def test_cache_key_generation(self, embedding_service):
        """Test cache key generation."""
        key1 = embedding_service._create_cache_key("test text", "model1")
        key2 = embedding_service._create_cache_key("test text", "model1")
        key3 = embedding_service._create_cache_key("test text", "model2")
        
        assert key1 == key2  # Same text and model
        assert key1 != key3  # Different model
    
    @pytest.mark.asyncio
    async def test_similarity_computation(self, embedding_service):
        """Test similarity computation."""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        emb3 = [1.0, 0.0, 0.0]
        
        # Test cosine similarity
        sim1 = await embedding_service.compute_similarity(emb1, emb2, "cosine")
        sim2 = await embedding_service.compute_similarity(emb1, emb3, "cosine")
        
        assert sim1 == 0.0  # Orthogonal vectors
        assert sim2 == 1.0  # Identical vectors
    
    @pytest.mark.asyncio
    async def test_find_similar_embeddings(self, embedding_service):
        """Test finding similar embeddings."""
        query_embedding = [1.0, 0.0, 0.0]
        candidates = [
            ("doc1", [1.0, 0.0, 0.0]),  # Identical
            ("doc2", [0.9, 0.1, 0.0]),  # Similar
            ("doc3", [0.0, 1.0, 0.0]),  # Different
        ]
        
        similar = await embedding_service.find_similar_embeddings(
            query_embedding, candidates, top_k=2, threshold=0.8
        )
        
        assert len(similar) == 2
        assert similar[0][0] == "doc1"  # Most similar first
        assert similar[1][0] == "doc2"


class TestLanguageModelService:
    """Test language model service."""
    
    @pytest.fixture
    def ai_manager(self):
        """Create mock AI manager."""
        manager = Mock()
        manager.config = AIConfig()
        
        # Mock model selection
        from obsidian_librarian.ai.models import ModelInfo
        manager.get_best_model = AsyncMock(return_value=ModelInfo(
            name="gpt-3.5-turbo",
            provider=AIProvider.OPENAI,
            capabilities=[ModelCapability.CHAT],
            context_length=4096,
        ))
        
        manager._models = {
            "gpt-3.5-turbo": ModelInfo(
                name="gpt-3.5-turbo",
                provider=AIProvider.OPENAI,
                capabilities=[ModelCapability.CHAT],
                context_length=4096,
            )
        }
        
        manager.check_model_availability = AsyncMock(return_value=True)
        manager.get_cost_estimate = AsyncMock(return_value=0.01)
        
        return manager
    
    @pytest.fixture
    def language_service(self, ai_manager):
        """Create language service."""
        return LanguageModelService(ai_manager)
    
    def test_cache_key_generation(self, language_service):
        """Test cache key generation."""
        request1 = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.7,
        )
        request2 = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.7,
        )
        request3 = ChatRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            temperature=0.7,
        )
        
        key1 = language_service._create_chat_cache_key(request1)
        key2 = language_service._create_chat_cache_key(request2)
        key3 = language_service._create_chat_cache_key(request3)
        
        assert key1 == key2  # Same request
        assert key1 != key3  # Different content


class TestContentAnalyzer:
    """Test content analyzer."""
    
    @pytest.fixture
    def mock_language_service(self):
        """Create mock language service."""
        service = Mock()
        
        # Mock responses for different analysis types
        async def mock_chat_completion(request):
            content = request.messages[0].content.lower()
            
            if "classify" in content:
                return Mock(text="article")
            elif "quality" in content:
                return Mock(text="Structure: 0.8\nClarity: 0.7\nCompleteness: 0.9\nRelevance: 0.8\nActionability: 0.6")
            elif "topics" in content or "extract" in content:
                return Mock(text="machine learning, artificial intelligence, deep learning")
            elif "sentiment" in content:
                return Mock(text="positive")
            elif "suggest" in content and "improvement" in content:
                return Mock(text="Add more examples\nImprove structure\nInclude references")
            elif "tags" in content:
                return Mock(text="ai, ml, technology, research")
            else:
                return Mock(text="test response")
        
        service.chat_completion = AsyncMock(side_effect=mock_chat_completion)
        return service
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        service = Mock()
        service.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
        return service
    
    @pytest.fixture
    def content_analyzer(self, mock_language_service, mock_embedding_service):
        """Create content analyzer."""
        return ContentAnalyzer(mock_language_service, mock_embedding_service)
    
    @pytest.fixture
    def sample_note(self):
        """Create sample note for testing."""
        return Note(
            id="test-note",
            path=Path("test.md"),
            content="# Machine Learning\n\nThis is an introduction to machine learning concepts.",
            content_hash="abc123",
            metadata=NoteMetadata(),
            links=[],
            tasks=[],
            word_count=10,
            file_size=100,
        )
    
    @pytest.mark.asyncio
    async def test_content_analysis(self, content_analyzer, sample_note):
        """Test comprehensive content analysis."""
        analysis = await content_analyzer.analyze_note(sample_note)
        
        assert analysis.note_id == "test-note"
        assert analysis.content_type == ContentType.ARTICLE
        assert 0.0 <= analysis.quality_score <= 1.0
        assert len(analysis.key_topics) > 0
        assert analysis.sentiment in ["positive", "negative", "neutral"]
        assert analysis.reading_time_minutes > 0
    
    @pytest.mark.asyncio
    async def test_batch_analysis(self, content_analyzer):
        """Test batch content analysis."""
        notes = [
            Note(
                id=f"note-{i}",
                path=Path(f"note{i}.md"),
                content=f"Content {i}",
                content_hash=f"hash{i}",
                metadata=NoteMetadata(),
                links=[], tasks=[], word_count=5, file_size=50,
            )
            for i in range(3)
        ]
        
        analyses = await content_analyzer.analyze_vault_content(notes, batch_size=2)
        
        assert len(analyses) == 3
        for analysis in analyses:
            assert analysis.note_id.startswith("note-")


class TestContentSummarizer:
    """Test content summarizer."""
    
    @pytest.fixture
    def mock_language_service(self):
        """Create mock language service."""
        service = Mock()
        
        async def mock_chat_completion(request):
            content = request.messages[0].content
            if "brief" in content.lower():
                return Mock(text="This is a brief summary.")
            elif "comprehensive" in content.lower() or "detailed" in content.lower():
                return Mock(text="This is a detailed summary with multiple points.")
            else:
                return Mock(text="This is a standard summary of the content.")
        
        service.chat_completion = AsyncMock(side_effect=mock_chat_completion)
        return service
    
    @pytest.fixture
    def content_summarizer(self, mock_language_service):
        """Create content summarizer."""
        return ContentSummarizer(mock_language_service)
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for summarization."""
        return """
        Machine learning is a method of data analysis that automates analytical model building. 
        It is a branch of artificial intelligence based on the idea that systems can learn from data, 
        identify patterns and make decisions with minimal human intervention.
        
        The process begins with observations or data, such as examples, direct experience, or instruction, 
        in order to look for patterns in data and make better decisions in the future based on the examples we provide.
        
        Machine learning algorithms build a mathematical model based on training data, in order to make 
        predictions or decisions without being explicitly programmed to do so.
        """
    
    @pytest.mark.asyncio
    async def test_extractive_summarization(self, content_summarizer, sample_text):
        """Test extractive summarization."""
        config = SummaryConfig(summary_type=SummaryType.EXTRACTIVE, max_length=150)
        result = await content_summarizer.summarize_detailed(sample_text, config)
        
        assert result.summary_type == SummaryType.EXTRACTIVE
        assert len(result.summary) <= 200  # Should be roughly within limit
        assert result.compression_ratio < 1.0
        assert len(result.key_points) > 0
    
    @pytest.mark.asyncio
    async def test_abstractive_summarization(self, content_summarizer, sample_text):
        """Test AI-powered abstractive summarization."""
        config = SummaryConfig(summary_type=SummaryType.ABSTRACTIVE, max_length=100)
        result = await content_summarizer.summarize_detailed(sample_text, config, ContentType.ARTICLE)
        
        assert result.summary_type == SummaryType.ABSTRACTIVE
        assert "summary" in result.summary.lower()
        assert result.compression_ratio < 1.0
    
    @pytest.mark.asyncio
    async def test_bullet_point_summarization(self, content_summarizer, sample_text):
        """Test bullet point summarization."""
        config = SummaryConfig(summary_type=SummaryType.BULLET_POINTS, max_length=200)
        result = await content_summarizer.summarize_detailed(sample_text, config)
        
        assert result.summary_type == SummaryType.BULLET_POINTS
        assert "•" in result.summary
        lines = result.summary.split('\n')
        assert len(lines) > 1
    
    @pytest.mark.asyncio
    async def test_batch_summarization(self, content_summarizer):
        """Test batch summarization."""
        texts = [
            "Short text one.",
            "Short text two with more content here.",
            "Short text three with even more content and details.",
        ]
        
        results = await content_summarizer.summarize_batch(texts)
        
        assert len(results) == 3
        for result in results:
            assert len(result.summary) > 0
            assert result.compression_ratio <= 1.0


class TestQueryProcessor:
    """Test query processor."""
    
    @pytest.fixture
    def mock_language_service(self):
        """Create mock language service."""
        service = Mock()
        
        async def mock_chat_completion(request):
            # Mock AI query processing response
            return Mock(text='{"intent": "search", "entities": [], "filters": {}, "search_terms": ["notes"], "confidence": 0.9}')
        
        service.chat_completion = AsyncMock(side_effect=mock_chat_completion)
        return service
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        service = Mock()
        return service
    
    @pytest.fixture
    def query_processor(self, mock_language_service, mock_embedding_service):
        """Create query processor."""
        return QueryProcessor(mock_language_service, mock_embedding_service)
    
    @pytest.mark.asyncio
    async def test_search_query_processing(self, query_processor):
        """Test search query processing."""
        result = await query_processor.process_query("find my notes about machine learning")
        
        assert result.intent == QueryIntent.SEARCH
        assert "machine" in result.search_terms or "learning" in result.search_terms
        assert result.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_create_query_processing(self, query_processor):
        """Test create query processing."""
        result = await query_processor.process_query("create a new note about AI")
        
        assert result.intent == QueryIntent.CREATE
        assert "ai" in result.search_terms or "AI" in result.search_terms
    
    @pytest.mark.asyncio
    async def test_research_query_processing(self, query_processor):
        """Test research query processing."""
        result = await query_processor.process_query("research the latest developments in neural networks")
        
        assert result.intent == QueryIntent.RESEARCH
        assert len(result.search_terms) > 0
    
    @pytest.mark.asyncio
    async def test_entity_extraction(self, query_processor):
        """Test entity extraction."""
        result = await query_processor.process_query('find notes with tag #ai and date "2024-01-15"')
        
        entities = result.entities
        assert len(entities) > 0
        
        # Should extract tag and date entities
        entity_types = [e.type for e in entities]
        assert "tag" in entity_types or "date" in entity_types


class TestAIIntegration:
    """Test complete AI pipeline integration."""
    
    @pytest.fixture
    def ai_config(self):
        """Create AI configuration for testing."""
        return AIConfig(
            primary_provider=AIProvider.OPENAI,
            temperature=0.7,
            max_tokens=1000,
            enable_caching=True,
        )
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, ai_config):
        """Test complete AI pipeline without external dependencies."""
        # Initialize components with mocks
        ai_manager = AIModelManager(ai_config)
        
        # Mock successful initialization
        ai_manager._clients = {AIProvider.OPENAI: Mock()}
        ai_manager._models = {
            "gpt-3.5-turbo": Mock(provider=AIProvider.OPENAI, capabilities=[ModelCapability.CHAT]),
            "text-embedding-ada-002": Mock(provider=AIProvider.OPENAI, capabilities=[ModelCapability.EMBEDDING]),
        }
        
        embedding_service = EmbeddingService(ai_manager)
        language_service = LanguageModelService(ai_manager)
        
        # Test that services are properly initialized
        assert embedding_service.ai_manager == ai_manager
        assert language_service.ai_manager == ai_manager
        
        # Test service statistics
        embedding_stats = await embedding_service.get_embedding_stats()
        language_stats = await language_service.get_service_stats()
        
        assert "cache_size" in embedding_stats
        assert "cache_size" in language_stats
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallbacks(self, ai_config):
        """Test error handling and fallback mechanisms."""
        ai_manager = AIModelManager(ai_config)
        
        # Test with no clients (should handle gracefully)
        embedding_service = EmbeddingService(ai_manager)
        
        # Should not raise exception even with no models
        stats = await embedding_service.get_embedding_stats()
        assert stats["model_count"] == 0
        
        # Test cache clearing
        await embedding_service.clear_cache()
        assert len(embedding_service._embedding_cache) == 0


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])