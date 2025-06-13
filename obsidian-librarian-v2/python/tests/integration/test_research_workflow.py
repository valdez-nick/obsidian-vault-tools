"""
Integration tests for research workflow functionality.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio
import aiohttp

from obsidian_librarian.librarian import ObsidianLibrarian
from obsidian_librarian.models import (
    LibrarianConfig, ResearchQuery, ResearchResult, 
    ResearchSource, NoteMetadata
)
from obsidian_librarian.services.research import ResearchService
from obsidian_librarian.services.ai import AIService


@pytest.fixture
def research_vault(tmp_path):
    """Create a vault structure for research testing."""
    vault_path = tmp_path / "research_vault"
    vault_path.mkdir()
    
    # Create .obsidian directory
    (vault_path / ".obsidian").mkdir()
    
    # Create research structure
    dirs = [
        "Research",
        "Research/Papers",
        "Research/Web",
        "Research/Archive",
        "Sources",
        "References",
        "Templates"
    ]
    
    for dir_path in dirs:
        (vault_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create templates
    templates = {
        "Templates/research.md": """# {{title}}

## Summary
{{summary}}

## Key Points
{{key_points}}

## Source
- URL: {{url}}
- Type: {{source_type}}
- Date: {{date}}
- Quality Score: {{quality_score}}

## Notes


## References
{{references}}

---
Tags: #research #{{source_type}} {{tags}}
Created: {{created_date}}
""",
        "Templates/paper.md": """# {{title}}

## Abstract
{{abstract}}

## Authors
{{authors}}

## Key Contributions
{{contributions}}

## Methodology
{{methodology}}

## Results
{{results}}

## Citations
{{citations}}

---
Tags: #paper #research {{tags}}
Source: {{source}}
""",
        "Templates/web_article.md": """# {{title}}

## Summary
{{summary}}

## Main Points
{{main_points}}

## Quotes
{{quotes}}

## Source
- URL: {{url}}
- Author: {{author}}
- Published: {{published_date}}

---
Tags: #web #article {{tags}}
"""
    }
    
    for template_path, content in templates.items():
        (vault_path / template_path).write_text(content)
    
    # Create some existing research notes
    existing_notes = {
        "Research/existing_paper.md": """# Existing Research Paper

## Summary
This is an existing research paper about machine learning.

## Key Points
- Point 1
- Point 2

Tags: #research #ml #existing
""",
        "Sources/trusted_source.md": """# Trusted Source List

## Academic
- arxiv.org
- scholar.google.com
- pubmed.ncbi.nlm.nih.gov

## Technical
- github.com
- stackoverflow.com
- docs.python.org

## News
- nature.com
- sciencedaily.com
"""
    }
    
    for note_path, content in existing_notes.items():
        (vault_path / note_path).write_text(content)
    
    return vault_path


@pytest.fixture
def mock_ai_service():
    """Create a mock AI service."""
    ai_service = AsyncMock(spec=AIService)
    
    # Mock embeddings
    ai_service.get_embedding.return_value = [0.1] * 768
    
    # Mock summarization
    ai_service.summarize.return_value = "This is a summarized version of the content."
    
    # Mock key point extraction
    ai_service.extract_key_points.return_value = [
        "Key point 1",
        "Key point 2",
        "Key point 3"
    ]
    
    # Mock quality assessment
    ai_service.assess_quality.return_value = {
        "score": 0.85,
        "reasoning": "High quality content with good references"
    }
    
    return ai_service


@pytest.fixture
def mock_research_sources():
    """Mock external research sources."""
    sources = {
        "arxiv": [
            {
                "title": "Deep Learning for Natural Language Processing",
                "authors": ["John Doe", "Jane Smith"],
                "abstract": "This paper presents a novel approach...",
                "url": "https://arxiv.org/abs/2024.12345",
                "published": "2024-01-15",
                "citations": 42
            },
            {
                "title": "Transformer Architecture Improvements",
                "authors": ["Alice Johnson"],
                "abstract": "We propose several improvements...",
                "url": "https://arxiv.org/abs/2024.12346",
                "published": "2024-01-10",
                "citations": 28
            }
        ],
        "github": [
            {
                "title": "awesome-nlp",
                "description": "A curated list of NLP resources",
                "url": "https://github.com/user/awesome-nlp",
                "stars": 1500,
                "language": "Markdown",
                "topics": ["nlp", "machine-learning", "deep-learning"]
            }
        ],
        "web": [
            {
                "title": "Understanding Transformers: A Visual Guide",
                "url": "https://example.com/transformers-guide",
                "author": "Tech Blog",
                "published": "2024-01-01",
                "content": "Transformers have revolutionized NLP..."
            }
        ]
    }
    return sources


class TestResearchWorkflow:
    """Test complete research workflow from query to organization."""
    
    @pytest.mark.asyncio
    async def test_basic_research_flow(self, research_vault, mock_ai_service, mock_research_sources):
        """Test basic research workflow."""
        config = LibrarianConfig()
        
        with patch('obsidian_librarian.services.ai.AIService', return_value=mock_ai_service):
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(research_vault)
                
                # Mock external API calls
                with patch('aiohttp.ClientSession.get') as mock_get:
                    # Mock ArXiv response
                    mock_arxiv_response = AsyncMock()
                    mock_arxiv_response.status = 200
                    mock_arxiv_response.text.return_value = json.dumps({
                        "results": mock_research_sources["arxiv"]
                    })
                    
                    mock_get.return_value.__aenter__.return_value = mock_arxiv_response
                    
                    # Perform research
                    results = []
                    async for result in librarian.research(
                        session_id=session_id,
                        query="transformer architecture improvements",
                        sources=["arxiv", "github"],
                        max_results=10,
                        organize=True
                    ):
                        if result.get('type') == 'result':
                            results.append(result['data'])
                        elif result.get('type') == 'complete':
                            break
                    
                    assert len(results) > 0
                    
                    # Check results structure
                    first_result = results[0]
                    assert 'title' in first_result
                    assert 'source' in first_result
                    assert 'quality_score' in first_result
                    assert 'summary' in first_result
                    
                    # Check if notes were created
                    research_notes = list((research_vault / "Research").glob("**/*.md"))
                    assert len(research_notes) > 1  # More than just existing notes
    
    @pytest.mark.asyncio
    async def test_research_with_filtering(self, research_vault, mock_ai_service):
        """Test research with quality and relevance filtering."""
        config = LibrarianConfig(
            research_quality_threshold=0.8,
            research_relevance_threshold=0.7
        )
        
        research_service = ResearchService(config, mock_ai_service)
        
        # Mock search results with varying quality
        mock_results = [
            ResearchResult(
                title="High Quality Result",
                source="arxiv",
                url="https://arxiv.org/1",
                quality_score=0.9,
                relevance_score=0.85,
                summary="Excellent research"
            ),
            ResearchResult(
                title="Low Quality Result",
                source="web",
                url="https://example.com/2",
                quality_score=0.6,
                relevance_score=0.5,
                summary="Poor quality content"
            ),
            ResearchResult(
                title="Medium Quality Result",
                source="github",
                url="https://github.com/3",
                quality_score=0.75,
                relevance_score=0.8,
                summary="Decent content"
            )
        ]
        
        # Filter results
        filtered_results = await research_service.filter_results(
            results=mock_results,
            query="test query"
        )
        
        # Only high quality result should pass
        assert len(filtered_results) == 1
        assert filtered_results[0].title == "High Quality Result"
    
    @pytest.mark.asyncio
    async def test_research_deduplication(self, research_vault, mock_ai_service):
        """Test research result deduplication."""
        config = LibrarianConfig()
        research_service = ResearchService(config, mock_ai_service)
        
        # Create duplicate results
        mock_results = [
            ResearchResult(
                title="Deep Learning Paper",
                source="arxiv",
                url="https://arxiv.org/1",
                quality_score=0.9,
                summary="A paper about deep learning"
            ),
            ResearchResult(
                title="Deep Learning Paper",  # Same title
                source="web",
                url="https://example.com/1",
                quality_score=0.7,
                summary="A paper about deep learning"
            ),
            ResearchResult(
                title="Different Paper",
                source="github",
                url="https://github.com/2",
                quality_score=0.8,
                summary="Something different"
            )
        ]
        
        # Mock similarity check
        mock_ai_service.calculate_similarity.side_effect = [
            0.95,  # Very similar
            0.3    # Not similar
        ]
        
        deduplicated = await research_service.deduplicate_results(mock_results)
        
        # Should keep the higher quality duplicate and the different paper
        assert len(deduplicated) == 2
        assert deduplicated[0].source == "arxiv"  # Higher quality kept
        assert deduplicated[1].title == "Different Paper"
    
    @pytest.mark.asyncio
    async def test_research_organization(self, research_vault, mock_ai_service):
        """Test automatic organization of research results."""
        config = LibrarianConfig()
        
        with patch('obsidian_librarian.services.ai.AIService', return_value=mock_ai_service):
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(research_vault)
                
                # Create mock research results
                results = [
                    {
                        'title': 'Machine Learning Paper',
                        'source': 'arxiv',
                        'url': 'https://arxiv.org/ml',
                        'content': 'ML research content',
                        'metadata': {
                            'authors': ['Author 1'],
                            'date': '2024-01-01',
                            'citations': 10
                        }
                    },
                    {
                        'title': 'Web Article on AI',
                        'source': 'web',
                        'url': 'https://blog.com/ai',
                        'content': 'AI article content',
                        'metadata': {
                            'author': 'Blog Author',
                            'published': '2024-01-02'
                        }
                    },
                    {
                        'title': 'GitHub NLP Project',
                        'source': 'github',
                        'url': 'https://github.com/nlp',
                        'content': 'NLP project description',
                        'metadata': {
                            'stars': 100,
                            'language': 'Python'
                        }
                    }
                ]
                
                # Organize results
                organized_notes = await librarian.organize_research_results(
                    session_id=session_id,
                    results=results,
                    organization_strategy="source"
                )
                
                # Check organization structure
                assert (research_vault / "Research/Papers").exists()
                assert (research_vault / "Research/Web").exists()
                
                # Check if notes were created in correct directories
                paper_notes = list((research_vault / "Research/Papers").glob("*.md"))
                web_notes = list((research_vault / "Research/Web").glob("*.md"))
                
                assert len(paper_notes) >= 1
                assert len(web_notes) >= 1
                
                # Verify note content
                paper_content = paper_notes[0].read_text()
                assert "Machine Learning Paper" in paper_content
                assert "arxiv" in paper_content.lower()
    
    @pytest.mark.asyncio
    async def test_research_with_existing_notes(self, research_vault, mock_ai_service):
        """Test research that detects and links to existing notes."""
        config = LibrarianConfig()
        
        # Mock AI service to detect similarity with existing content
        mock_ai_service.calculate_similarity.return_value = 0.85  # High similarity
        
        with patch('obsidian_librarian.services.ai.AIService', return_value=mock_ai_service):
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(research_vault)
                
                # Research similar to existing content
                results = []
                async for result in librarian.research(
                    session_id=session_id,
                    query="machine learning research",
                    sources=["arxiv"],
                    max_results=5,
                    organize=True,
                    link_existing=True
                ):
                    if result.get('type') == 'result':
                        results.append(result['data'])
                    elif result.get('type') == 'similar_found':
                        # Should detect existing similar note
                        assert 'existing_note' in result
                        assert 'similarity_score' in result
                
                # New notes should link to existing ones
                new_notes = list((research_vault / "Research").glob("**/machine*.md"))
                if new_notes:
                    content = new_notes[0].read_text()
                    assert "[[existing_paper]]" in content or "Related:" in content


class TestResearchSources:
    """Test different research source integrations."""
    
    @pytest.mark.asyncio
    async def test_arxiv_source(self, mock_ai_service):
        """Test ArXiv research source."""
        from obsidian_librarian.services.sources import ArxivSource
        
        source = ArxivSource()
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text.return_value = """<?xml version="1.0" encoding="UTF-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom">
                <entry>
                    <title>Test Paper Title</title>
                    <summary>Test abstract content</summary>
                    <author><name>Test Author</name></author>
                    <link href="http://arxiv.org/abs/2024.00001"/>
                    <published>2024-01-01T00:00:00Z</published>
                </entry>
            </feed>"""
            
            mock_get.return_value.__aenter__.return_value = mock_response
            
            results = await source.search("test query", max_results=5)
            
            assert len(results) > 0
            assert results[0].title == "Test Paper Title"
            assert results[0].source == "arxiv"
            assert "Test Author" in str(results[0].metadata.get('authors', []))
    
    @pytest.mark.asyncio
    async def test_github_source(self, mock_ai_service):
        """Test GitHub research source."""
        from obsidian_librarian.services.sources import GitHubSource
        
        source = GitHubSource(api_token="test_token")
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "items": [
                    {
                        "name": "test-repo",
                        "full_name": "user/test-repo",
                        "description": "Test repository",
                        "html_url": "https://github.com/user/test-repo",
                        "stargazers_count": 100,
                        "language": "Python",
                        "topics": ["test", "example"]
                    }
                ]
            }
            
            mock_get.return_value.__aenter__.return_value = mock_response
            
            results = await source.search("test query", max_results=5)
            
            assert len(results) > 0
            assert results[0].title == "test-repo"
            assert results[0].source == "github"
            assert results[0].metadata['stars'] == 100
    
    @pytest.mark.asyncio
    async def test_web_scraping_source(self, mock_ai_service):
        """Test web scraping research source."""
        from obsidian_librarian.services.sources import WebScrapingSource
        
        source = WebScrapingSource()
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock search engine results
            mock_search_response = AsyncMock()
            mock_search_response.status = 200
            mock_search_response.text.return_value = """
            <html>
                <div class="result">
                    <a href="https://example.com/article">Test Article</a>
                    <p>Article description...</p>
                </div>
            </html>
            """
            
            # Mock article content
            mock_article_response = AsyncMock()
            mock_article_response.status = 200
            mock_article_response.text.return_value = """
            <html>
                <article>
                    <h1>Test Article</h1>
                    <p>Article content goes here...</p>
                </article>
            </html>
            """
            
            mock_get.side_effect = [
                AsyncMock(__aenter__=AsyncMock(return_value=mock_search_response)),
                AsyncMock(__aenter__=AsyncMock(return_value=mock_article_response))
            ]
            
            results = await source.search("test query", max_results=5)
            
            assert len(results) > 0
            assert results[0].source == "web"
    
    @pytest.mark.asyncio
    async def test_custom_source_integration(self, research_vault):
        """Test custom research source integration."""
        from obsidian_librarian.services.sources import CustomSource
        
        # Define custom source
        class TestCustomSource(CustomSource):
            async def search(self, query: str, max_results: int = 10) -> List[ResearchResult]:
                return [
                    ResearchResult(
                        title=f"Custom Result: {query}",
                        source="custom_test",
                        url="https://custom.source/1",
                        quality_score=0.9,
                        summary="Custom source result"
                    )
                ]
        
        config = LibrarianConfig()
        config.custom_sources = [TestCustomSource()]
        
        async with ObsidianLibrarian(config) as librarian:
            session_id = await librarian.create_session(research_vault)
            
            results = []
            async for result in librarian.research(
                session_id=session_id,
                query="test",
                sources=["custom_test"],
                max_results=5
            ):
                if result.get('type') == 'result':
                    results.append(result['data'])
            
            assert len(results) > 0
            assert results[0]['source'] == "custom_test"


class TestResearchTemplates:
    """Test research template application."""
    
    @pytest.mark.asyncio
    async def test_template_selection(self, research_vault, mock_ai_service):
        """Test automatic template selection based on source."""
        config = LibrarianConfig()
        
        with patch('obsidian_librarian.services.ai.AIService', return_value=mock_ai_service):
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(research_vault)
                
                # Test different source types
                test_cases = [
                    ("arxiv", "Templates/paper.md"),
                    ("github", "Templates/research.md"),
                    ("web", "Templates/web_article.md")
                ]
                
                for source, expected_template in test_cases:
                    template = await librarian.select_research_template(
                        session_id=session_id,
                        source=source,
                        content_type="research"
                    )
                    
                    assert template is not None
                    assert expected_template in str(template)
    
    @pytest.mark.asyncio
    async def test_template_variable_filling(self, research_vault, mock_ai_service):
        """Test filling template variables with research data."""
        config = LibrarianConfig()
        
        research_data = {
            'title': 'Test Research Paper',
            'summary': 'This is a test summary',
            'url': 'https://test.com/paper',
            'source_type': 'arxiv',
            'quality_score': 0.9,
            'authors': ['Author One', 'Author Two'],
            'date': '2024-01-01',
            'key_points': ['Point 1', 'Point 2'],
            'tags': ['#ml', '#nlp']
        }
        
        with patch('obsidian_librarian.services.ai.AIService', return_value=mock_ai_service):
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(research_vault)
                
                # Fill template
                filled_content = await librarian.fill_research_template(
                    session_id=session_id,
                    template_path=research_vault / "Templates/research.md",
                    data=research_data
                )
                
                # Verify all variables were replaced
                assert "{{title}}" not in filled_content
                assert "Test Research Paper" in filled_content
                assert "This is a test summary" in filled_content
                assert "https://test.com/paper" in filled_content
                assert "- Point 1" in filled_content
                assert "#ml #nlp" in filled_content


class TestResearchCaching:
    """Test research result caching."""
    
    @pytest.mark.asyncio
    async def test_result_caching(self, research_vault, mock_ai_service):
        """Test caching of research results."""
        config = LibrarianConfig(
            enable_research_cache=True,
            research_cache_ttl=3600  # 1 hour
        )
        
        call_count = 0
        
        async def mock_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return [
                ResearchResult(
                    title="Cached Result",
                    source="test",
                    url="https://test.com",
                    quality_score=0.8
                )
            ]
        
        with patch('obsidian_librarian.services.sources.TestSource.search', mock_search):
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(research_vault)
                
                # First search
                results1 = []
                async for result in librarian.research(
                    session_id=session_id,
                    query="test query",
                    sources=["test"],
                    max_results=5
                ):
                    if result.get('type') == 'result':
                        results1.append(result['data'])
                
                # Second search with same query
                results2 = []
                async for result in librarian.research(
                    session_id=session_id,
                    query="test query",
                    sources=["test"],
                    max_results=5
                ):
                    if result.get('type') == 'result':
                        results2.append(result['data'])
                
                # Should use cache, not call search again
                assert call_count == 1
                assert len(results1) == len(results2)
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, research_vault, mock_ai_service):
        """Test research cache invalidation."""
        config = LibrarianConfig(
            enable_research_cache=True,
            research_cache_ttl=1  # 1 second
        )
        
        with patch('obsidian_librarian.services.ai.AIService', return_value=mock_ai_service):
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(research_vault)
                
                # Clear cache
                await librarian.clear_research_cache(session_id)
                
                # Wait for TTL to expire
                await asyncio.sleep(1.1)
                
                # Cache should be invalidated
                # (Implementation would check cache state)


class TestResearchExport:
    """Test research result export functionality."""
    
    @pytest.mark.asyncio
    async def test_export_to_bibtex(self, research_vault, mock_ai_service):
        """Test exporting research results to BibTeX format."""
        results = [
            ResearchResult(
                title="Test Paper",
                source="arxiv",
                url="https://arxiv.org/abs/2024.00001",
                metadata={
                    'authors': ['Author One', 'Author Two'],
                    'year': 2024,
                    'journal': 'arXiv'
                }
            )
        ]
        
        config = LibrarianConfig()
        research_service = ResearchService(config, mock_ai_service)
        
        bibtex = await research_service.export_to_bibtex(results)
        
        assert "@article{" in bibtex
        assert "title = {Test Paper}" in bibtex
        assert "author = {Author One and Author Two}" in bibtex
        assert "year = {2024}" in bibtex
    
    @pytest.mark.asyncio
    async def test_export_to_markdown_table(self, research_vault, mock_ai_service):
        """Test exporting research results to markdown table."""
        results = [
            ResearchResult(
                title="Paper 1",
                source="arxiv",
                url="https://arxiv.org/1",
                quality_score=0.9,
                summary="Summary 1"
            ),
            ResearchResult(
                title="Paper 2",
                source="github",
                url="https://github.com/2",
                quality_score=0.8,
                summary="Summary 2"
            )
        ]
        
        config = LibrarianConfig()
        research_service = ResearchService(config, mock_ai_service)
        
        markdown_table = await research_service.export_to_markdown_table(results)
        
        assert "| Title | Source | Quality | URL |" in markdown_table
        assert "| Paper 1 | arxiv | 0.90 |" in markdown_table
        assert "| Paper 2 | github | 0.80 |" in markdown_table


if __name__ == "__main__":
    pytest.main([__file__, "-v"])