"""
Unit tests for research service.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import aiohttp

from obsidian_librarian.services.research import (
    ResearchService, ResearchConfig, ResearchSession, ResearchResult
)
from obsidian_librarian.ai.query_processor import ProcessedQuery, QueryType, QueryIntent
from obsidian_librarian.models import Note


@pytest.fixture
def research_config():
    """Create test research configuration."""
    return ResearchConfig(
        max_concurrent_requests=5,
        request_timeout=10,
        enable_content_extraction=True,
        enable_summarization=False,
        max_results_per_source=10,
    )


@pytest.fixture
def mock_vault():
    """Create a mock vault."""
    vault = Mock()
    vault.path = "/test/vault"
    vault.create_note = AsyncMock(return_value="new_note_id")
    return vault


@pytest.fixture
def processed_query():
    """Create a test processed query."""
    return ProcessedQuery(
        original_text="machine learning algorithms",
        cleaned_text="machine learning algorithms",
        keywords=["machine", "learning", "algorithms"],
        key_phrases=["machine learning"],
        entities=["machine learning"],
        query_type=QueryType.TECHNICAL,
        intent=QueryIntent.RESEARCH,
        confidence=0.9,
        synonyms={"machine learning": ["ml", "ai"]},
        expanded_terms=["ml", "ai", "artificial intelligence"],
        related_concepts=["deep learning", "neural networks"],
        search_terms=["machine learning", "algorithms", "ml"],
        boost_terms=["official", "documentation"],
        filter_terms=["recent"],
        session_id="test_session_123",
    )


@pytest.mark.asyncio
async def test_research_service_initialization(mock_vault, research_config):
    """Test research service initialization."""
    service = ResearchService(mock_vault, research_config)
    
    assert service.vault == mock_vault
    assert service.config == research_config
    assert service.query_processor is not None
    assert service.content_summarizer is not None
    assert service.source_manager is not None


@pytest.mark.asyncio
async def test_research_session_creation(processed_query, research_config):
    """Test research session creation and management."""
    session = ResearchSession(processed_query, research_config)
    
    assert session.query == processed_query
    assert session.session_id == processed_query.session_id
    assert session.status == "starting"
    assert len(session.results) == 0
    assert len(session.processed_urls) == 0
    
    # Test adding results
    result = ResearchResult(
        url="https://example.com/ml",
        title="Machine Learning Guide",
        summary="A guide to ML",
        source="Example",
        quality_score=0.8,
    )
    
    session.add_result(result)
    assert len(session.results) == 1
    assert "https://example.com/ml" in session.processed_urls
    
    # Test duplicate prevention
    session.add_result(result)
    assert len(session.results) == 1  # Should not add duplicate
    
    # Test statistics
    stats = session.get_stats()
    assert stats['results_found'] == 1
    assert stats['unique_sources'] == 1
    assert stats['average_quality'] == 0.8


@pytest.mark.asyncio
async def test_research_with_mock_sources(mock_vault, research_config):
    """Test research operation with mocked sources."""
    service = ResearchService(mock_vault, research_config)
    
    # Mock the HTTP session
    async with service:
        assert service.http_session is not None
        
        # Mock query processor
        with patch.object(service.query_processor, 'process', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = ProcessedQuery(
                original_text="test query",
                cleaned_text="test query",
                keywords=["test", "query"],
                key_phrases=[],
                entities=[],
                query_type=QueryType.GENERAL,
                intent=QueryIntent.RESEARCH,
                confidence=0.8,
                synonyms={},
                expanded_terms=[],
                related_concepts=[],
                search_terms=["test", "query"],
                boost_terms=[],
                filter_terms=[],
                session_id="test_123",
            )
            
            # Mock source manager
            mock_source = AsyncMock()
            mock_source.search.return_value.__aiter__.return_value = [
                ResearchResult(
                    url="https://test.com/1",
                    title="Test Result 1",
                    summary="Summary 1",
                    source="TestSource",
                    quality_score=0.9,
                ),
                ResearchResult(
                    url="https://test.com/2",
                    title="Test Result 2",
                    summary="Summary 2",
                    source="TestSource",
                    quality_score=0.7,
                ),
            ]
            
            with patch.object(service.source_manager, 'select_sources', new_callable=AsyncMock) as mock_select:
                mock_select.return_value = [mock_source]
                
                # Run research
                results = []
                async for result in service.research("test query", max_results=5):
                    results.append(result)
                
                assert len(results) == 2
                assert results[0].url == "https://test.com/1"
                assert results[0].quality_score == 0.9
                assert results[1].url == "https://test.com/2"
                assert results[1].quality_score == 0.7


@pytest.mark.asyncio
async def test_content_extraction(mock_vault, research_config):
    """Test content extraction from research results."""
    service = ResearchService(mock_vault, research_config)
    
    result = ResearchResult(
        url="https://example.com/article",
        title="Test Article",
        summary="",
        source="Example",
        quality_score=0.5,
    )
    
    # Mock HTTP response
    mock_response = Mock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value="""
        <html>
        <body>
            <h1>Test Article</h1>
            <p>This is the main content of the article.</p>
            <p>It contains multiple paragraphs.</p>
        </body>
        </html>
    """)
    
    # Create mock session
    mock_session = Mock()
    mock_session.get = AsyncMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_response)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    
    service.http_session = mock_session
    
    await service._extract_content(result)
    
    assert result.content is not None
    assert len(result.content) > 100
    assert "Test Article" in result.content
    assert "main content" in result.content


@pytest.mark.asyncio
async def test_quality_score_calculation(mock_vault, research_config):
    """Test quality score calculation for research results."""
    service = ResearchService(mock_vault, research_config)
    
    # High quality result
    high_quality = ResearchResult(
        url="https://github.com/example/repo",
        title="Comprehensive Machine Learning Library",
        summary="A well-documented ML library",
        content="Detailed documentation with examples...",
        source="GitHub",
        quality_score=0.0,
    )
    
    score = service._calculate_quality_score(high_quality)
    assert score > 0.7  # Should be high quality
    
    # Low quality result
    low_quality = ResearchResult(
        url="https://random-site.com/page",
        title="ML",
        summary="",
        content="",
        source="Unknown",
        quality_score=0.0,
    )
    
    score = service._calculate_quality_score(low_quality)
    assert score < 0.5  # Should be low quality


@pytest.mark.asyncio
async def test_organize_results(mock_vault, research_config):
    """Test organizing research results into vault structure."""
    service = ResearchService(mock_vault, research_config)
    
    results = [
        ResearchResult(
            url="https://github.com/example/ml-lib",
            title="ML Library",
            summary="Machine learning library",
            source="GitHub",
            quality_score=0.9,
        ),
        ResearchResult(
            url="https://arxiv.org/abs/1234.5678",
            title="Deep Learning Paper",
            summary="Research on deep learning",
            source="ArXiv",
            quality_score=0.85,
        ),
    ]
    
    organized = await service.organize_results(results, "machine learning research")
    
    assert "by_topic" in organized
    assert "by_source" in organized
    assert "by_date" in organized
    
    # Verify vault create_note was called for organization
    assert mock_vault.create_note.called


@pytest.mark.asyncio
async def test_research_error_handling(mock_vault, research_config):
    """Test error handling in research operations."""
    service = ResearchService(mock_vault, research_config)
    
    # Mock query processor to raise error
    with patch.object(service.query_processor, 'process', new_callable=AsyncMock) as mock_process:
        mock_process.side_effect = Exception("Query processing failed")
        
        # Research should handle the error gracefully
        results = []
        async for result in service.research("test query"):
            results.append(result)
        
        assert len(results) == 0  # No results due to error


@pytest.mark.asyncio
async def test_concurrent_source_search(mock_vault, research_config):
    """Test concurrent searching across multiple sources."""
    service = ResearchService(mock_vault, research_config)
    
    # Create multiple mock sources with delays
    async def create_delayed_results(source_name, delay):
        await asyncio.sleep(delay)
        return [
            ResearchResult(
                url=f"https://{source_name}.com/1",
                title=f"{source_name} Result",
                summary=f"From {source_name}",
                source=source_name,
                quality_score=0.8,
            )
        ]
    
    # Mock multiple sources
    source1 = AsyncMock()
    source1.search.return_value.__aiter__.return_value = await create_delayed_results("Source1", 0.1)
    
    source2 = AsyncMock()
    source2.search.return_value.__aiter__.return_value = await create_delayed_results("Source2", 0.05)
    
    # Results should come in order of completion (Source2 first due to shorter delay)
    # This tests the concurrent processing behavior