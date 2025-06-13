"""
Simple tests for AI components without full integration dependencies.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Test individual AI components in isolation
def test_ai_config():
    """Test AI configuration."""
    from obsidian_librarian.ai.models import AIConfig, AIProvider
    
    config = AIConfig(
        primary_provider=AIProvider.OPENAI,
        temperature=0.7,
        max_tokens=1000,
    )
    
    assert config.primary_provider == AIProvider.OPENAI
    assert config.temperature == 0.7
    assert config.max_tokens == 1000


def test_model_info():
    """Test ModelInfo class."""
    from obsidian_librarian.ai.models import ModelInfo, AIProvider, ModelCapability
    
    model = ModelInfo(
        name="gpt-4",
        provider=AIProvider.OPENAI,
        capabilities=[ModelCapability.CHAT, ModelCapability.TEXT_GENERATION],
        context_length=8192,
        input_cost_per_1k=0.03,
        output_cost_per_1k=0.06,
    )
    
    assert model.name == "gpt-4"
    assert model.provider == AIProvider.OPENAI
    assert ModelCapability.CHAT in model.capabilities
    assert model.context_length == 8192


def test_chat_message():
    """Test ChatMessage class.""" 
    from obsidian_librarian.ai.language_models import ChatMessage
    
    message = ChatMessage(
        role="user",
        content="Hello, world!",
    )
    
    assert message.role == "user"
    assert message.content == "Hello, world!"


def test_chat_request():
    """Test ChatRequest class."""
    from obsidian_librarian.ai.language_models import ChatRequest, ChatMessage
    
    messages = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi there!"),
    ]
    
    request = ChatRequest(
        messages=messages,
        temperature=0.7,
        max_tokens=100,
    )
    
    assert len(request.messages) == 2
    assert request.temperature == 0.7
    assert request.max_tokens == 100


def test_summary_config():
    """Test SummaryConfig class."""
    from obsidian_librarian.ai.content_summarizer import SummaryConfig, SummaryType
    
    config = SummaryConfig(
        max_length=300,
        summary_type=SummaryType.ABSTRACTIVE,
        preserve_structure=True,
    )
    
    assert config.max_length == 300
    assert config.summary_type == SummaryType.ABSTRACTIVE
    assert config.preserve_structure is True


def test_content_analyzer_without_services():
    """Test ContentAnalyzer initialization without services."""
    from obsidian_librarian.ai.content_analyzer import ContentAnalyzer
    
    # Should work without language and embedding services
    analyzer = ContentAnalyzer(None, None, None)
    assert analyzer.language_service is None
    assert analyzer.embedding_service is None


def test_content_summarizer_without_service():
    """Test ContentSummarizer initialization without language service."""
    from obsidian_librarian.ai.content_summarizer import ContentSummarizer
    
    # Should work without language service
    summarizer = ContentSummarizer(None)
    assert summarizer.language_service is None


def test_query_processor_without_services():
    """Test QueryProcessor initialization without services."""
    from obsidian_librarian.ai.query_processor import QueryProcessor
    
    # Should work without services
    processor = QueryProcessor(None, None)
    assert processor.language_service is None
    assert processor.embedding_service is None


@pytest.mark.asyncio
async def test_extractive_summarization():
    """Test extractive summarization without AI service."""
    from obsidian_librarian.ai.content_summarizer import ContentSummarizer, SummaryConfig, SummaryType
    
    summarizer = ContentSummarizer(None)
    
    sample_text = """
    Machine learning is a powerful tool for data analysis. It enables systems to learn from data automatically.
    The technology has many applications in various fields. Modern algorithms can process large datasets efficiently.
    Deep learning is a subset of machine learning that uses neural networks. These networks can recognize complex patterns.
    """
    
    config = SummaryConfig(
        summary_type=SummaryType.EXTRACTIVE,
        max_length=100,
    )
    
    result = await summarizer.summarize_detailed(sample_text, config)
    
    assert result.summary_type == SummaryType.EXTRACTIVE
    assert len(result.summary) > 0
    assert result.compression_ratio < 1.0
    assert result.word_count > 0


@pytest.mark.asyncio
async def test_bullet_point_summarization():
    """Test bullet point summarization."""
    from obsidian_librarian.ai.content_summarizer import ContentSummarizer, SummaryConfig, SummaryType
    
    summarizer = ContentSummarizer(None)
    
    sample_text = """
    # Machine Learning Guide
    
    ## Key Concepts
    - Supervised learning uses labeled data
    - Unsupervised learning finds patterns
    - Reinforcement learning learns through rewards
    
    ## Applications
    - Image recognition
    - Natural language processing
    - Recommendation systems
    """
    
    config = SummaryConfig(
        summary_type=SummaryType.BULLET_POINTS,
        max_length=200,
    )
    
    result = await summarizer.summarize_detailed(sample_text, config)
    
    assert result.summary_type == SummaryType.BULLET_POINTS
    assert "•" in result.summary
    assert len(result.summary.split('\n')) > 1


@pytest.mark.asyncio
async def test_rule_based_query_processing():
    """Test rule-based query processing."""
    from obsidian_librarian.ai.query_processor import QueryProcessor, QueryIntent
    
    processor = QueryProcessor(None, None)
    
    # Test search query
    result = await processor.process_query("find my notes about machine learning")
    assert result.intent == QueryIntent.SEARCH
    assert len(result.search_terms) > 0
    
    # Test create query
    result = await processor.process_query("create a new note about AI")
    assert result.intent == QueryIntent.CREATE
    
    # Test research query
    result = await processor.process_query("research neural networks online")
    assert result.intent == QueryIntent.RESEARCH


@pytest.mark.asyncio
async def test_entity_extraction():
    """Test entity extraction from queries."""
    from obsidian_librarian.ai.query_processor import QueryProcessor
    
    processor = QueryProcessor(None, None)
    
    # Test tag extraction
    result = await processor.process_query("find notes with #ai tag")
    tag_entities = [e for e in result.entities if e.type == "tag"]
    assert len(tag_entities) > 0
    assert "ai" in [e.text for e in tag_entities]
    
    # Test wiki link extraction
    result = await processor.process_query("show me [[Project Alpha]] notes")
    link_entities = [e for e in result.entities if e.type == "note_name"]
    assert len(link_entities) > 0
    assert "Project Alpha" in [e.text for e in link_entities]


def test_embedding_service_cache_key():
    """Test embedding service cache key generation."""
    from obsidian_librarian.ai.embeddings import EmbeddingService
    
    # Create service with mock manager
    mock_manager = Mock()
    mock_manager.config = Mock()
    service = EmbeddingService(mock_manager)
    
    # Test cache key generation
    key1 = service._create_cache_key("test text", "model1")
    key2 = service._create_cache_key("test text", "model1")
    key3 = service._create_cache_key("different text", "model1")
    
    assert key1 == key2  # Same input should generate same key
    assert key1 != key3  # Different input should generate different key


@pytest.mark.asyncio
async def test_embedding_similarity():
    """Test embedding similarity computation."""
    from obsidian_librarian.ai.embeddings import EmbeddingService
    
    mock_manager = Mock()
    mock_manager.config = Mock()
    service = EmbeddingService(mock_manager)
    
    # Test cosine similarity
    emb1 = [1.0, 0.0, 0.0]
    emb2 = [0.0, 1.0, 0.0]  # Orthogonal
    emb3 = [1.0, 0.0, 0.0]  # Identical
    
    sim1 = await service.compute_similarity(emb1, emb2, "cosine")
    sim2 = await service.compute_similarity(emb1, emb3, "cosine")
    
    assert abs(sim1 - 0.0) < 0.001  # Orthogonal vectors
    assert abs(sim2 - 1.0) < 0.001  # Identical vectors


def test_ai_manager_initialization():
    """Test AI manager initialization."""
    from obsidian_librarian.ai.models import AIModelManager, AIConfig
    
    config = AIConfig()
    manager = AIModelManager(config)
    
    assert manager.config == config
    assert isinstance(manager._models, dict)
    assert isinstance(manager._clients, dict)
    assert manager._cost_tracker is not None


@pytest.mark.asyncio
async def test_ai_manager_cost_estimation():
    """Test cost estimation."""
    from obsidian_librarian.ai.models import AIModelManager, AIConfig, ModelInfo, AIProvider, ModelCapability
    
    config = AIConfig()
    manager = AIModelManager(config)
    
    # Add a test model
    manager._models["test-model"] = ModelInfo(
        name="test-model",
        provider=AIProvider.OPENAI,
        capabilities=[ModelCapability.CHAT],
        context_length=4096,
        input_cost_per_1k=0.01,
        output_cost_per_1k=0.02,
    )
    
    # Test cost calculation
    cost = await manager.get_cost_estimate("test-model", 1000, 500)
    expected = (1000/1000 * 0.01) + (500/1000 * 0.02)
    assert abs(cost - expected) < 0.001


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])