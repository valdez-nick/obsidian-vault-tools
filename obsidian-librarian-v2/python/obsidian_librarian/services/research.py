"""
Research Assistant Service for intelligent content discovery and organization.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, AsyncGenerator
from urllib.parse import urlparse
import aiohttp
import structlog
from dataclasses import dataclass

from ..ai.query_processor import QueryProcessor
from ..ai.content_summarizer import ContentSummarizer
from ..sources import SourceManager, ResearchSource
from ..models import ResearchResult, ResearchQuery, Note
from ..vault import Vault

logger = structlog.get_logger(__name__)


@dataclass
class ResearchConfig:
    """Configuration for research operations."""
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    max_results_per_source: int = 50
    enable_content_extraction: bool = True
    enable_summarization: bool = True
    min_content_length: int = 100
    quality_threshold: float = 0.7


class ResearchSession:
    """Manages a single research session with progress tracking."""
    
    def __init__(self, query: ResearchQuery, config: ResearchConfig):
        self.query = query
        self.config = config
        self.session_id = query.session_id
        self.results: List[ResearchResult] = []
        self.processed_urls: Set[str] = set()
        self.start_time = datetime.utcnow()
        self.status = "starting"
        
    def add_result(self, result: ResearchResult) -> None:
        """Add a research result to this session."""
        if result.url not in self.processed_urls:
            self.results.append(result)
            self.processed_urls.add(result.url)
            
    def get_stats(self) -> Dict:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "query": self.query.text,
            "status": self.status,
            "results_found": len(self.results),
            "unique_sources": len(set(r.source for r in self.results)),
            "duration_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "average_quality": sum(r.quality_score for r in self.results) / len(self.results) if self.results else 0.0,
        }


class ResearchService:
    """
    Intelligent research assistant that discovers, extracts, and organizes 
    relevant content from the web based on natural language queries.
    """
    
    def __init__(
        self,
        vault: Vault,
        config: Optional[ResearchConfig] = None,
    ):
        self.vault = vault
        self.config = config or ResearchConfig()
        self.query_processor = QueryProcessor()
        self.content_summarizer = ContentSummarizer()
        self.source_manager = SourceManager()
        self.active_sessions: Dict[str, ResearchSession] = {}
        
        # Initialize HTTP session
        self.http_session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent_requests,
            limit_per_host=5,
        )
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        
        self.http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "Obsidian-Librarian/0.1.0 (Research Assistant)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.http_session:
            await self.http_session.close()
    
    async def research(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results: Optional[int] = None,
    ) -> AsyncGenerator[ResearchResult, None]:
        """
        Perform intelligent research based on a natural language query.
        
        Args:
            query: Natural language research query
            sources: Optional list of specific sources to search
            max_results: Maximum number of results to return
            
        Yields:
            ResearchResult objects as they are discovered
        """
        logger.info("Starting research", query=query, sources=sources)
        
        # Process the query to understand intent and extract keywords
        processed_query = await self.query_processor.process(query)
        
        # Create research session
        session = ResearchSession(processed_query, self.config)
        self.active_sessions[session.session_id] = session
        session.status = "searching"
        
        try:
            # Get appropriate sources for this query
            target_sources = await self.source_manager.select_sources(
                processed_query, 
                preferred_sources=sources
            )
            
            logger.info(
                "Selected sources for research",
                sources=[s.name for s in target_sources],
                query_type=processed_query.query_type,
            )
            
            # Create search tasks for each source
            search_tasks = []
            for source in target_sources:
                task = asyncio.create_task(
                    self._search_source(source, processed_query, session)
                )
                search_tasks.append(task)
            
            # Process results as they become available
            result_count = 0
            max_results = max_results or self.config.max_results_per_source * len(target_sources)
            
            # Use asyncio.as_completed to yield results as soon as they're ready
            async for result in self._process_search_results(search_tasks):
                session.add_result(result)
                
                # Apply quality filtering
                if result.quality_score >= self.config.quality_threshold:
                    yield result
                    result_count += 1
                    
                    if result_count >= max_results:
                        logger.info("Reached maximum results limit", count=result_count)
                        break
            
            session.status = "completed"
            logger.info("Research completed", stats=session.get_stats())
            
        except Exception as e:
            session.status = "failed"
            logger.error("Research failed", error=str(e), query=query)
            raise
        finally:
            # Clean up session after a delay
            asyncio.create_task(self._cleanup_session(session.session_id, delay=300))
    
    async def _search_source(
        self,
        source: ResearchSource,
        query: ResearchQuery,
        session: ResearchSession,
    ) -> List[ResearchResult]:
        """Search a specific source for relevant content."""
        results = []
        
        try:
            logger.debug("Searching source", source=source.name, query=query.text)
            
            # Use source-specific search implementation
            async for result in source.search(query, self.http_session):
                if len(results) >= self.config.max_results_per_source:
                    break
                    
                # Extract and process content if enabled
                if self.config.enable_content_extraction:
                    await self._extract_content(result)
                
                # Generate summary if enabled
                if self.config.enable_summarization and result.content:
                    result.summary = await self.content_summarizer.summarize(
                        result.content,
                        max_length=300,
                    )
                
                # Calculate quality score
                result.quality_score = self._calculate_quality_score(result)
                
                results.append(result)
                
                logger.debug(
                    "Found result",
                    url=result.url,
                    title=result.title,
                    quality=result.quality_score,
                )
        
        except Exception as e:
            logger.error(
                "Source search failed",
                source=source.name,
                error=str(e),
                query=query.text,
            )
        
        return results
    
    async def _process_search_results(
        self, 
        search_tasks: List[asyncio.Task],
    ) -> AsyncGenerator[ResearchResult, None]:
        """Process search results as they become available."""
        pending = set(search_tasks)
        
        while pending:
            done, pending = await asyncio.wait(
                pending, 
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            for task in done:
                try:
                    results = await task
                    for result in results:
                        yield result
                except Exception as e:
                    logger.error("Search task failed", error=str(e))
    
    async def _extract_content(self, result: ResearchResult) -> None:
        """Extract main content from a research result."""
        if not self.http_session:
            return
            
        try:
            async with self.http_session.get(result.url) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Use content extraction library
                    # This would integrate with the Rust content extractor
                    # For now, use a simple implementation
                    from bs4 import BeautifulSoup
                    
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text = soup.get_text()
                    
                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    if len(text) >= self.config.min_content_length:
                        result.content = text[:5000]  # Limit content length
                        
        except Exception as e:
            logger.warning("Content extraction failed", url=result.url, error=str(e))
    
    def _calculate_quality_score(self, result: ResearchResult) -> float:
        """Calculate a quality score for a research result."""
        score = 0.0
        
        # Title quality
        if result.title and len(result.title) > 10:
            score += 0.2
            
        # Content quality
        if result.content:
            content_length = len(result.content)
            if content_length > self.config.min_content_length:
                score += 0.3
            if content_length > 1000:
                score += 0.1
                
        # Summary quality
        if result.summary and len(result.summary) > 50:
            score += 0.2
            
        # Source credibility (basic implementation)
        domain = urlparse(result.url).netloc.lower()
        if any(trusted in domain for trusted in ['github.com', 'arxiv.org', 'docs.', 'wikipedia.org']):
            score += 0.2
            
        return min(score, 1.0)
    
    async def organize_results(
        self,
        results: List[ResearchResult],
        query: str,
    ) -> Dict[str, List[Note]]:
        """
        Organize research results into structured notes in the vault.
        
        Returns:
            Dictionary mapping organization categories to created notes
        """
        logger.info("Organizing research results", count=len(results), query=query)
        
        organized = {
            "by_topic": [],
            "by_source": [], 
            "by_date": [],
        }
        
        # Create research library structure if it doesn't exist
        library_path = self.vault.path / "Research Library"
        await self._ensure_library_structure(library_path)
        
        # Categorize results by topic
        topics = await self._categorize_by_topic(results, query)
        
        for topic, topic_results in topics.items():
            topic_note = await self._create_topic_note(
                library_path / "By Topic" / topic,
                topic,
                topic_results,
                query,
            )
            organized["by_topic"].append(topic_note)
        
        # Organize by source
        by_source = {}
        for result in results:
            source_name = self._get_source_name(result.url)
            if source_name not in by_source:
                by_source[source_name] = []
            by_source[source_name].append(result)
        
        for source, source_results in by_source.items():
            source_note = await self._create_source_note(
                library_path / "By Source" / source,
                source,
                source_results,
                query,
            )
            organized["by_source"].append(source_note)
        
        # Create session summary
        session_note = await self._create_session_note(
            library_path / "By Date" / datetime.now().strftime("%Y-%m-%d"),
            query,
            results,
        )
        organized["by_date"].append(session_note)
        
        logger.info("Research organization completed", organized_counts={
            k: len(v) for k, v in organized.items()
        })
        
        return organized
    
    async def _ensure_library_structure(self, library_path: Path) -> None:
        """Ensure research library directory structure exists."""
        subdirs = ["By Topic", "By Source", "By Date", "_metadata"]
        
        for subdir in subdirs:
            dir_path = library_path / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create index file if it doesn't exist
        index_file = library_path / "_index.md"
        if not index_file.exists():
            index_content = """# Research Library

This directory contains research results organized by the Obsidian Librarian.

## Organization

- **By Topic/**: Research organized by subject matter
- **By Source/**: Research organized by origin (GitHub, ArXiv, etc.)
- **By Date/**: Research sessions organized chronologically
- **_metadata/**: System metadata and configuration

## Usage

Use the search functionality to find specific research across all categories.
Each research session is automatically tagged and linked for easy navigation.
"""
            await self.vault.create_note(index_file, index_content)
    
    async def _categorize_by_topic(
        self, 
        results: List[ResearchResult], 
        query: str,
    ) -> Dict[str, List[ResearchResult]]:
        """Categorize results by topic using AI."""
        # This would use the AI categorization service
        # For now, use simple keyword-based categorization
        
        categories = {}
        
        # Extract key terms from query
        query_words = set(query.lower().split())
        
        for result in results:
            # Simple topic assignment based on title/content keywords
            topics = self._extract_topics(result, query_words)
            
            for topic in topics:
                if topic not in categories:
                    categories[topic] = []
                categories[topic].append(result)
        
        return categories
    
    def _extract_topics(
        self, 
        result: ResearchResult, 
        query_words: Set[str],
    ) -> List[str]:
        """Extract topics from a research result."""
        # Simple implementation - in practice would use NLP
        topics = []
        
        text = f"{result.title} {result.summary or ''}".lower()
        
        # Check for common technical topics
        tech_topics = {
            'machine learning': ['ml', 'machine learning', 'neural', 'ai'],
            'rust': ['rust', 'cargo', 'rustc'],
            'python': ['python', 'pip', 'django', 'flask'],
            'web development': ['web', 'frontend', 'backend', 'api'],
            'documentation': ['docs', 'documentation', 'guide', 'tutorial'],
        }
        
        for topic, keywords in tech_topics.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        
        # If no specific topics found, use query-based topic
        if not topics:
            topics.append("general")
        
        return topics
    
    async def _create_topic_note(
        self,
        path: Path,
        topic: str,
        results: List[ResearchResult],
        query: str,
    ) -> Note:
        """Create a topic-organized note."""
        content = f"""---
topic: {topic}
query: "{query}"
results_count: {len(results)}
created: {datetime.now().isoformat()}
tags: [research, {topic.replace(' ', '-')}]
---

# {topic.title()} - Research Results

*Generated from query: "{query}"*

## Overview

Found {len(results)} results related to {topic}.

## Results

"""
        
        for i, result in enumerate(results, 1):
            content += f"""
### {i}. [{result.title}]({result.url})

**Source:** {self._get_source_name(result.url)}  
**Quality Score:** {result.quality_score:.2f}

{result.summary or 'No summary available.'}

---
"""
        
        return await self.vault.create_note(path, content)
    
    async def _create_source_note(
        self,
        path: Path,
        source: str,
        results: List[ResearchResult],
        query: str,
    ) -> Note:
        """Create a source-organized note."""
        content = f"""---
source: {source}
query: "{query}"
results_count: {len(results)}
created: {datetime.now().isoformat()}
tags: [research, source-{source.lower().replace('.', '-')}]
---

# {source} - Research Results

*Generated from query: "{query}"*

## Results from {source}

"""
        
        for result in results:
            content += f"""
- [{result.title}]({result.url}) (Score: {result.quality_score:.2f})
  {result.summary or 'No summary available.'}

"""
        
        return await self.vault.create_note(path, content)
    
    async def _create_session_note(
        self,
        path: Path,
        query: str,
        results: List[ResearchResult],
    ) -> Note:
        """Create a research session summary note."""
        session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        content = f"""---
session_id: {session_id}
query: "{query}"
total_results: {len(results)}
created: {datetime.now().isoformat()}
tags: [research, session]
---

# Research Session: {query}

**Session ID:** {session_id}  
**Total Results:** {len(results)}  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary

This research session explored: "{query}"

## Results by Quality

"""
        
        # Sort by quality score
        sorted_results = sorted(results, key=lambda r: r.quality_score, reverse=True)
        
        for result in sorted_results:
            content += f"""
- **{result.quality_score:.2f}** - [{result.title}]({result.url})
  *{self._get_source_name(result.url)}*
"""
        
        return await self.vault.create_note(path / f"{session_id}.md", content)
    
    def _get_source_name(self, url: str) -> str:
        """Extract a readable source name from URL."""
        domain = urlparse(url).netloc.lower()
        
        # Map common domains to readable names
        source_mapping = {
            'github.com': 'GitHub',
            'arxiv.org': 'ArXiv',
            'docs.python.org': 'Python Docs',
            'doc.rust-lang.org': 'Rust Docs',
            'stackoverflow.com': 'Stack Overflow',
            'wikipedia.org': 'Wikipedia',
        }
        
        for pattern, name in source_mapping.items():
            if pattern in domain:
                return name
        
        return domain.replace('www.', '').title()
    
    async def _cleanup_session(self, session_id: str, delay: int = 300) -> None:
        """Clean up a research session after a delay."""
        await asyncio.sleep(delay)
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.debug("Cleaned up research session", session_id=session_id)
    
    def get_active_sessions(self) -> Dict[str, Dict]:
        """Get statistics for all active research sessions."""
        return {
            session_id: session.get_stats()
            for session_id, session in self.active_sessions.items()
        }