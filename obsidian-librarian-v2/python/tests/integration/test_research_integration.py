"""
Integration tests for Research Service.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from obsidian_librarian.services import ResearchService
from obsidian_librarian.models import ResearchConfig, ResearchQuery


@pytest.mark.integration
class TestResearchIntegration:
    """Test Research Service with mocked external dependencies."""
    
    @pytest.fixture
    async def research_service(self):
        """Create research service with test configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ResearchConfig(
                cache_dir=Path(tmpdir) / "cache",
                max_concurrent_requests=2,
                rate_limit_per_second=10,
                enable_caching=True,
            )
            
            service = ResearchService(config)
            await service.initialize()
            
            yield service
            
            await service.close()
    
    @pytest.mark.asyncio
    @patch('obsidian_librarian.services.research.WebScraper')
    @patch('obsidian_librarian.services.research.QueryProcessor')
    async def test_research_flow(self, mock_processor, mock_scraper, research_service):
        """Test complete research flow."""
        # Mock query processor
        mock_processor_instance = AsyncMock()
        mock_processor_instance.process_query.return_value = {
            "refined_query": "machine learning transformers",
            "search_terms": ["transformers", "BERT", "GPT"],
            "domains": ["arxiv.org", "github.com"],
        }
        mock_processor.return_value = mock_processor_instance
        
        # Mock web scraper
        mock_scraper_instance = AsyncMock()
        async def mock_scrape(url):
            if "arxiv" in url:
                return {
                    "url": url,
                    "title": "Attention Is All You Need",
                    "content": "The Transformer architecture...",
                    "metadata": {"authors": ["Vaswani et al."]},
                }
            else:
                return {
                    "url": url,
                    "title": "Transformers Library",
                    "content": "State-of-the-art NLP...",
                    "metadata": {"stars": 100000},
                }
        
        mock_scraper_instance.scrape.side_effect = mock_scrape
        mock_scraper.return_value = mock_scraper_instance
        
        # Execute research
        query = ResearchQuery(
            query="find transformer architecture papers",
            sources=["https://arxiv.org", "https://github.com"],
            max_results=5,
        )
        
        results = []
        async for result in research_service.research(
            query.query,
            query.sources,
            query.max_results
        ):
            results.append(result)
        
        assert len(results) > 0
        assert any("Attention Is All You Need" in r.title for r in results)
    
    @pytest.mark.asyncio
    async def test_concurrent_research(self, research_service):
        """Test concurrent research queries."""
        with patch('obsidian_librarian.services.research.WebScraper') as mock_scraper:
            # Mock scraper with delay
            mock_scraper_instance = AsyncMock()
            async def mock_scrape(url):
                await asyncio.sleep(0.1)  # Simulate network delay
                return {
                    "url": url,
                    "title": f"Result for {url}",
                    "content": "Test content",
                    "metadata": {},
                }
            
            mock_scraper_instance.scrape.side_effect = mock_scrape
            mock_scraper.return_value = mock_scraper_instance
            
            # Run multiple queries concurrently
            queries = [
                "machine learning",
                "deep learning",
                "neural networks",
            ]
            
            async def run_query(q):
                results = []
                async for result in research_service.research(q, max_results=3):
                    results.append(result)
                return results
            
            all_results = await asyncio.gather(*[run_query(q) for q in queries])
            
            assert len(all_results) == 3
            for results in all_results:
                assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_caching(self, research_service):
        """Test research result caching."""
        with patch('obsidian_librarian.services.research.WebScraper') as mock_scraper:
            call_count = 0
            
            mock_scraper_instance = AsyncMock()
            async def mock_scrape(url):
                nonlocal call_count
                call_count += 1
                return {
                    "url": url,
                    "title": "Cached Result",
                    "content": "This should be cached",
                    "metadata": {"call_count": call_count},
                }
            
            mock_scraper_instance.scrape.side_effect = mock_scrape
            mock_scraper.return_value = mock_scraper_instance
            
            # First query
            results1 = []
            async for result in research_service.research("test query", max_results=2):
                results1.append(result)
            
            initial_calls = call_count
            
            # Second identical query (should use cache)
            results2 = []
            async for result in research_service.research("test query", max_results=2):
                results2.append(result)
            
            # Cache should prevent additional calls
            assert call_count == initial_calls
            assert len(results1) == len(results2)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, research_service):
        """Test rate limiting functionality."""
        with patch('obsidian_librarian.services.research.WebScraper') as mock_scraper:
            request_times = []
            
            mock_scraper_instance = AsyncMock()
            async def mock_scrape(url):
                request_times.append(asyncio.get_event_loop().time())
                return {"url": url, "title": "Test", "content": "Test", "metadata": {}}
            
            mock_scraper_instance.scrape.side_effect = mock_scrape
            mock_scraper.return_value = mock_scraper_instance
            
            # Make many requests quickly
            urls = [f"https://example.com/{i}" for i in range(20)]
            
            start_time = asyncio.get_event_loop().time()
            tasks = []
            
            async for result in research_service.research("test", max_results=20):
                pass
            
            # Check that requests were rate limited
            # With 10 requests/second limit, 20 requests should take ~2 seconds
            total_time = asyncio.get_event_loop().time() - start_time
            assert total_time >= 1.5  # Allow some margin
    
    @pytest.mark.asyncio
    async def test_source_filtering(self, research_service):
        """Test source domain filtering."""
        with patch('obsidian_librarian.services.research.WebScraper') as mock_scraper:
            scraped_urls = []
            
            mock_scraper_instance = AsyncMock()
            async def mock_scrape(url):
                scraped_urls.append(url)
                return {
                    "url": url,
                    "title": f"Result from {url}",
                    "content": "Test",
                    "metadata": {},
                }
            
            mock_scraper_instance.scrape.side_effect = mock_scrape
            mock_scraper.return_value = mock_scraper_instance
            
            # Research with specific sources
            sources = ["https://arxiv.org", "https://github.com"]
            
            results = []
            async for result in research_service.research(
                "test query",
                sources=sources,
                max_results=5
            ):
                results.append(result)
            
            # Check that only specified domains were scraped
            for url in scraped_urls:
                assert any(source in url for source in sources)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, research_service):
        """Test error handling during research."""
        with patch('obsidian_librarian.services.research.WebScraper') as mock_scraper:
            mock_scraper_instance = AsyncMock()
            
            # Some URLs succeed, some fail
            async def mock_scrape(url):
                if "fail" in url:
                    raise Exception("Scraping failed")
                return {
                    "url": url,
                    "title": "Success",
                    "content": "Content",
                    "metadata": {},
                }
            
            mock_scraper_instance.scrape.side_effect = mock_scrape
            mock_scraper.return_value = mock_scraper_instance
            
            # Mix of good and bad URLs
            with patch('obsidian_librarian.services.research.search_web') as mock_search:
                mock_search.return_value = [
                    "https://good.com/1",
                    "https://fail.com/1",
                    "https://good.com/2",
                    "https://fail.com/2",
                ]
                
                results = []
                async for result in research_service.research("test", max_results=4):
                    results.append(result)
                
                # Should get results only from successful URLs
                assert len(results) == 2
                assert all("Success" in r.title for r in results)