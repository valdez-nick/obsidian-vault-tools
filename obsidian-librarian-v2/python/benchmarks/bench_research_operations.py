"""
Performance benchmarks for research operations.
"""

import asyncio
import time
import statistics
from pathlib import Path
import tempfile
from typing import List, Dict, Any
from unittest.mock import AsyncMock, patch

from obsidian_librarian.services import ResearchService
from obsidian_librarian.models import ResearchConfig


class ResearchBenchmark:
    """Benchmark research service operations."""
    
    def __init__(self):
        self.results: Dict[str, List[float]] = {}
    
    async def setup(self):
        """Set up research service for benchmarking."""
        self.tmpdir = tempfile.mkdtemp()
        
        config = ResearchConfig(
            cache_dir=Path(self.tmpdir) / "cache",
            max_concurrent_requests=10,
            rate_limit_per_second=100,  # High limit for benchmarking
            enable_caching=True,
        )
        
        self.service = ResearchService(config)
        await self.service.initialize()
        
        # Mock the web scraper for consistent benchmarking
        self.mock_scraper = AsyncMock()
        self.setup_mock_scraper()
    
    def setup_mock_scraper(self):
        """Set up mock scraper with realistic delays."""
        async def mock_scrape(url):
            # Simulate network delay
            await asyncio.sleep(0.05)  # 50ms
            
            return {
                "url": url,
                "title": f"Research Result: {url.split('/')[-1]}",
                "content": "Lorem ipsum " * 100,  # ~100 words
                "metadata": {
                    "author": "Test Author",
                    "date": "2024-01-01",
                    "relevance_score": 0.85,
                },
            }
        
        self.mock_scraper.scrape.side_effect = mock_scrape
    
    async def benchmark_single_research(self):
        """Benchmark single research queries."""
        print("\n=== Single Research Queries ===")
        
        queries = [
            "machine learning algorithms",
            "transformer architecture",
            "neural network optimization",
            "deep learning frameworks",
            "reinforcement learning",
        ]
        
        with patch('obsidian_librarian.services.research.WebScraper', return_value=self.mock_scraper):
            times = []
            
            for query in queries:
                start = time.perf_counter()
                
                results = []
                async for result in self.service.research(query, max_results=10):
                    results.append(result)
                
                end = time.perf_counter()
                elapsed = end - start
                times.append(elapsed)
                
                print(f"  Query '{query}': {len(results)} results in {elapsed:.2f}s")
            
            self.results["single_research"] = times
            print(f"Average single research: {statistics.mean(times):.2f}s")
    
    async def benchmark_concurrent_research(self):
        """Benchmark concurrent research queries."""
        print("\n=== Concurrent Research Queries ===")
        
        queries = [
            "python programming",
            "rust systems programming",
            "javascript frameworks",
            "golang concurrency",
            "java enterprise",
        ]
        
        with patch('obsidian_librarian.services.research.WebScraper', return_value=self.mock_scraper):
            # Run queries concurrently
            async def run_query(q):
                results = []
                async for result in self.service.research(q, max_results=5):
                    results.append(result)
                return results
            
            start = time.perf_counter()
            all_results = await asyncio.gather(*[run_query(q) for q in queries])
            end = time.perf_counter()
            
            total_results = sum(len(r) for r in all_results)
            elapsed = end - start
            
            print(f"5 concurrent queries: {total_results} total results in {elapsed:.2f}s")
            print(f"Throughput: {total_results/elapsed:.1f} results/second")
            
            self.results["concurrent_research"] = [elapsed]
    
    async def benchmark_caching(self):
        """Benchmark caching effectiveness."""
        print("\n=== Caching Performance ===")
        
        query = "artificial intelligence"
        
        with patch('obsidian_librarian.services.research.WebScraper', return_value=self.mock_scraper):
            # First run (cache miss)
            start = time.perf_counter()
            results1 = []
            async for result in self.service.research(query, max_results=20):
                results1.append(result)
            end = time.perf_counter()
            
            first_run = end - start
            print(f"First run (cache miss): {first_run:.2f}s")
            
            # Second run (cache hit)
            start = time.perf_counter()
            results2 = []
            async for result in self.service.research(query, max_results=20):
                results2.append(result)
            end = time.perf_counter()
            
            second_run = end - start
            print(f"Second run (cache hit): {second_run:.2f}s")
            print(f"Cache speedup: {first_run/second_run:.1f}x")
            
            self.results["cache_performance"] = [first_run, second_run]
    
    async def benchmark_rate_limiting(self):
        """Benchmark rate limiting impact."""
        print("\n=== Rate Limiting Impact ===")
        
        # Test with different rate limits
        rate_limits = [10, 50, 100]  # requests per second
        
        with patch('obsidian_librarian.services.research.WebScraper', return_value=self.mock_scraper):
            for limit in rate_limits:
                self.service.config.rate_limit_per_second = limit
                
                start = time.perf_counter()
                
                # Try to make 50 requests
                results = []
                async for result in self.service.research("test query", max_results=50):
                    results.append(result)
                
                end = time.perf_counter()
                elapsed = end - start
                
                actual_rate = len(results) / elapsed
                print(f"  Rate limit {limit}/s: achieved {actual_rate:.1f}/s in {elapsed:.2f}s")
    
    async def benchmark_large_scale_research(self):
        """Benchmark large-scale research operations."""
        print("\n=== Large Scale Research ===")
        
        with patch('obsidian_librarian.services.research.WebScraper', return_value=self.mock_scraper):
            # Research with many results
            result_counts = [50, 100, 200]
            
            for count in result_counts:
                start = time.perf_counter()
                
                results = []
                async for result in self.service.research("large scale test", max_results=count):
                    results.append(result)
                
                end = time.perf_counter()
                elapsed = end - start
                
                print(f"  {count} results: {elapsed:.2f}s ({count/elapsed:.1f} results/s)")
                
                self.results[f"large_scale_{count}"] = [elapsed]
    
    async def benchmark_memory_usage(self):
        """Benchmark memory usage during research."""
        print("\n=== Memory Usage ===")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        with patch('obsidian_librarian.services.research.WebScraper', return_value=self.mock_scraper):
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run large research
            results = []
            async for result in self.service.research("memory test", max_results=500):
                results.append(result)
            
            # Peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            print(f"Baseline memory: {baseline_memory:.1f} MB")
            print(f"Peak memory: {peak_memory:.1f} MB")
            print(f"Memory increase: {peak_memory - baseline_memory:.1f} MB")
            
            # Clear results
            results.clear()
            
            # After cleanup
            cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"After cleanup: {cleanup_memory:.1f} MB")
    
    async def benchmark_source_filtering(self):
        """Benchmark source filtering performance."""
        print("\n=== Source Filtering ===")
        
        source_configs = [
            (["arxiv.org"], "Single source"),
            (["arxiv.org", "github.com", "papers.nips.cc"], "Multiple sources"),
            (None, "No filtering"),
        ]
        
        with patch('obsidian_librarian.services.research.WebScraper', return_value=self.mock_scraper):
            for sources, desc in source_configs:
                start = time.perf_counter()
                
                results = []
                async for result in self.service.research(
                    "test query",
                    sources=sources,
                    max_results=20
                ):
                    results.append(result)
                
                end = time.perf_counter()
                elapsed = end - start
                
                print(f"  {desc}: {elapsed:.2f}s")
    
    async def run(self):
        """Run all benchmarks."""
        print(f"\n{'='*60}")
        print("Research Service Performance Benchmark")
        print(f"{'='*60}")
        
        await self.setup()
        
        await self.benchmark_single_research()
        await self.benchmark_concurrent_research()
        await self.benchmark_caching()
        await self.benchmark_rate_limiting()
        await self.benchmark_large_scale_research()
        await self.benchmark_memory_usage()
        await self.benchmark_source_filtering()
        
        print(f"\n{'='*60}")
        print("Benchmark complete!")
        
        # Cleanup
        await self.service.close()
        
        return self.results


async def main():
    """Run research benchmarks."""
    benchmark = ResearchBenchmark()
    await benchmark.run()


if __name__ == "__main__":
    asyncio.run(main())