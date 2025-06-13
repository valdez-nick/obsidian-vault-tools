"""
Source management system for research operations.
"""

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse
import uuid

import structlog
import aiohttp
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import UUID

from .ai.query_processor import QueryResult, QueryIntent, QueryType, ProcessedQuery
from .models import ResearchResult

logger = structlog.get_logger(__name__)

Base = declarative_base()


class SourceStatus(Enum):
    """Status of a research source."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class SourcePriority(Enum):
    """Priority levels for research sources."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SourceConfig:
    """Configuration for a research source."""
    # Basic info
    name: str
    base_url: str
    source_type: str
    
    # Access configuration
    api_key: Optional[str] = None
    rate_limit_rps: float = 1.0
    timeout_seconds: int = 30
    
    # Quality settings
    quality_threshold: float = 0.7
    max_results_per_query: int = 20
    
    # Filtering
    languages: List[str] = field(default_factory=lambda: ["en"])
    domains: List[str] = field(default_factory=list)
    exclude_domains: List[str] = field(default_factory=list)
    
    # Behavior
    auto_enable: bool = True
    priority: SourcePriority = SourcePriority.MEDIUM
    
    # Custom headers
    headers: Dict[str, str] = field(default_factory=dict)


class SourceMetrics(Base):
    """Database model for source performance metrics."""
    __tablename__ = 'source_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_name = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Performance metrics
    response_time_ms = Column(Float)
    success_rate = Column(Float)
    results_count = Column(Integer)
    quality_score = Column(Float)
    
    # Error tracking
    error_count = Column(Integer, default=0)
    last_error = Column(Text)
    
    # Usage statistics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)


class SourceSearch(Base):
    """Database model for tracking source searches."""
    __tablename__ = 'source_searches'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_name = Column(String, nullable=False)
    query_text = Column(Text, nullable=False)
    query_hash = Column(String, nullable=False)
    
    # Search metadata
    search_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    results_count = Column(Integer)
    avg_quality = Column(Float)
    
    # Query classification
    query_type = Column(String)
    query_intent = Column(String)
    
    # Performance
    search_duration_ms = Column(Float)
    success = Column(Boolean, default=True)


@dataclass
class ResearchSource:
    """A source for research content."""
    config: SourceConfig
    status: SourceStatus = SourceStatus.ACTIVE
    last_accessed: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {
                'total_requests': 0,
                'successful_requests': 0,
                'average_quality': 0.0,
                'average_response_time': 0.0,
                'error_count': 0,
                'last_error': None,
            }
    
    async def search(
        self,
        query: ProcessedQuery,
        session: aiohttp.ClientSession,
    ) -> AsyncGenerator[ResearchResult, None]:
        """Search this source for relevant content."""
        raise NotImplementedError("Subclasses must implement search method")
    
    def can_handle_query(self, query: ProcessedQuery) -> float:
        """Return confidence score (0-1) for handling this query."""
        return 0.5  # Default medium confidence
    
    def update_metrics(self, success: bool, response_time: float, quality: float = 0.0):
        """Update source performance metrics."""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
            
            # Update averages
            current_avg_time = self.metrics['average_response_time']
            current_avg_quality = self.metrics['average_quality']
            successful_count = self.metrics['successful_requests']
            
            self.metrics['average_response_time'] = (
                (current_avg_time * (successful_count - 1) + response_time) / successful_count
            )
            
            if quality > 0:
                self.metrics['average_quality'] = (
                    (current_avg_quality * (successful_count - 1) + quality) / successful_count
                )
        else:
            self.metrics['error_count'] += 1
        
        self.last_accessed = datetime.utcnow()


class GitHubSource(ResearchSource):
    """GitHub repository and issues source."""
    
    def can_handle_query(self, query: ProcessedQuery) -> float:
        """GitHub is good for code and technical queries."""
        confidence = 0.3  # Base confidence
        
        if query.query_type in [QueryType.TECHNICAL, QueryType.CODE]:
            confidence += 0.4
        
        # Check for code-related keywords
        code_keywords = ['library', 'framework', 'api', 'implementation', 'example']
        matching_keywords = sum(1 for keyword in code_keywords if keyword in query.keywords)
        confidence += min(matching_keywords * 0.1, 0.3)
        
        return min(confidence, 1.0)
    
    async def search(
        self,
        query: ProcessedQuery,
        session: aiohttp.ClientSession,
    ) -> AsyncGenerator[ResearchResult, None]:
        """Search GitHub repositories and issues."""
        start_time = datetime.utcnow()
        
        try:
            # Build search query
            search_terms = ' '.join(query.search_terms[:5])  # Limit terms
            
            # Search repositories
            repo_url = f"https://api.github.com/search/repositories"
            params = {
                'q': search_terms,
                'sort': 'updated',
                'order': 'desc',
                'per_page': min(self.config.max_results_per_query, 30),
            }
            
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'Obsidian-Librarian/0.1.0',
            }
            
            if self.config.api_key:
                headers['Authorization'] = f'token {self.config.api_key}'
            
            async with session.get(repo_url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for item in data.get('items', []):
                        # Calculate quality score
                        quality = self._calculate_repo_quality(item)
                        
                        if quality >= self.config.quality_threshold:
                            yield ResearchResult(
                                url=item['html_url'],
                                title=item['full_name'],
                                summary=item.get('description', ''),
                                content=None,  # Would need separate API call
                                source='GitHub',
                                quality_score=quality,
                                published_date=self._parse_github_date(item.get('created_at')),
                                metadata={
                                    'stars': item.get('stargazers_count', 0),
                                    'forks': item.get('forks_count', 0),
                                    'language': item.get('language', ''),
                                    'updated_at': item.get('updated_at'),
                                    'topics': item.get('topics', []),
                                }
                            )
                
                # Update metrics
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.update_metrics(True, response_time, 0.8)
                
        except Exception as e:
            logger.error("GitHub search failed", error=str(e), query=query.original_text)
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.update_metrics(False, response_time)
            self.metrics['last_error'] = str(e)
    
    def _calculate_repo_quality(self, repo_data: Dict) -> float:
        """Calculate quality score for a GitHub repository."""
        score = 0.0
        
        # Star count (logarithmic scale)
        stars = repo_data.get('stargazers_count', 0)
        if stars > 0:
            score += min(0.3, 0.1 * (stars ** 0.3) / 10)
        
        # Recent activity
        updated = repo_data.get('updated_at')
        if updated:
            try:
                update_date = datetime.fromisoformat(updated.replace('Z', '+00:00'))
                days_since_update = (datetime.now(update_date.tzinfo) - update_date).days
                if days_since_update < 30:
                    score += 0.2
                elif days_since_update < 365:
                    score += 0.1
            except:
                pass
        
        # Has description
        if repo_data.get('description'):
            score += 0.1
        
        # Has topics/tags
        if repo_data.get('topics'):
            score += 0.1
        
        # Fork ratio (not too many forks relative to stars)
        forks = repo_data.get('forks_count', 0)
        if stars > 0 and forks / stars < 0.5:
            score += 0.1
        
        # Language specified
        if repo_data.get('language'):
            score += 0.1
        
        # Not archived
        if not repo_data.get('archived', False):
            score += 0.1
        
        return min(score, 1.0)
    
    def _parse_github_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse GitHub date string."""
        if not date_str:
            return None
        
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            return None


class ArxivSource(ResearchSource):
    """ArXiv research papers source."""
    
    def can_handle_query(self, query: ProcessedQuery) -> float:
        """ArXiv is excellent for academic and research queries."""
        confidence = 0.2  # Base confidence
        
        if query.query_type == QueryType.ACADEMIC:
            confidence += 0.6
        elif query.query_type == QueryType.TECHNICAL:
            confidence += 0.3
        
        # Check for academic keywords
        academic_keywords = ['research', 'paper', 'study', 'analysis', 'algorithm', 'model']
        matching_keywords = sum(1 for keyword in academic_keywords if keyword in query.keywords)
        confidence += min(matching_keywords * 0.1, 0.3)
        
        return min(confidence, 1.0)
    
    async def search(
        self,
        query: ProcessedQuery,
        session: aiohttp.ClientSession,
    ) -> AsyncGenerator[ResearchResult, None]:
        """Search ArXiv papers."""
        start_time = datetime.utcnow()
        
        try:
            # Build search query
            search_terms = ' '.join(query.search_terms[:8])
            
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{search_terms}',
                'sortBy': 'lastUpdatedDate',
                'sortOrder': 'descending',
                'max_results': min(self.config.max_results_per_query, 20),
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    xml_content = await response.text()
                    
                    # Parse XML (simplified)
                    async for result in self._parse_arxiv_xml(xml_content):
                        if result.quality_score >= self.config.quality_threshold:
                            yield result
                
                # Update metrics
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.update_metrics(True, response_time, 0.9)
                
        except Exception as e:
            logger.error("ArXiv search failed", error=str(e), query=query.original_text)
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.update_metrics(False, response_time)
            self.metrics['last_error'] = str(e)
    
    async def _parse_arxiv_xml(self, xml_content: str) -> AsyncGenerator[ResearchResult, None]:
        """Parse ArXiv XML response."""
        # This is a simplified XML parser
        # In practice, you'd use a proper XML library like lxml or xml.etree
        
        import re
        
        # Extract entries
        entry_pattern = re.compile(r'<entry>(.*?)</entry>', re.DOTALL)
        entries = entry_pattern.findall(xml_content)
        
        for entry in entries:
            try:
                # Extract fields
                title = self._extract_xml_field(entry, 'title')
                summary = self._extract_xml_field(entry, 'summary')
                published = self._extract_xml_field(entry, 'published')
                link = self._extract_arxiv_link(entry)
                authors = self._extract_arxiv_authors(entry)
                
                if title and link:
                    yield ResearchResult(
                        url=link,
                        title=title.strip(),
                        summary=summary.strip() if summary else '',
                        content=None,
                        source='ArXiv',
                        quality_score=0.85,  # ArXiv papers are generally high quality
                        published_date=self._parse_arxiv_date(published),
                        metadata={
                            'authors': authors,
                            'published': published,
                            'source_type': 'academic_paper',
                        }
                    )
            except Exception as e:
                logger.warning("Failed to parse ArXiv entry", error=str(e))
                continue
    
    def _extract_xml_field(self, xml: str, field: str) -> Optional[str]:
        """Extract a field from XML content."""
        pattern = f'<{field}[^>]*>(.*?)</{field}>'
        match = re.search(pattern, xml, re.DOTALL)
        return match.group(1) if match else None
    
    def _extract_arxiv_link(self, entry: str) -> Optional[str]:
        """Extract ArXiv paper link."""
        link_pattern = r'<link[^>]*href="([^"]*arxiv[^"]*)"'
        match = re.search(link_pattern, entry)
        return match.group(1) if match else None
    
    def _extract_arxiv_authors(self, entry: str) -> List[str]:
        """Extract author names."""
        author_pattern = r'<name>([^<]+)</name>'
        authors = re.findall(author_pattern, entry)
        return authors[:5]  # Limit to 5 authors
    
    def _parse_arxiv_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse ArXiv date string."""
        if not date_str:
            return None
        
        try:
            # ArXiv dates are in ISO format
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            return None


class SourceManager:
    """
    Manages multiple research sources and coordinates searches.
    """
    
    def __init__(self, db_url: str = "sqlite:///sources.db"):
        self.sources: Dict[str, ResearchSource] = {}
        self.db_url = db_url
        self.engine = None
        self.SessionLocal = None
        
        # Initialize default sources
        self._init_default_sources()
    
    async def initialize(self):
        """Initialize the source manager and database."""
        # Setup database
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        logger.info("Source manager initialized", source_count=len(self.sources))
    
    def _init_default_sources(self):
        """Initialize default research sources."""
        # GitHub source
        github_config = SourceConfig(
            name="github",
            base_url="https://api.github.com",
            source_type="code_repository",
            rate_limit_rps=5.0,
            max_results_per_query=20,
            priority=SourcePriority.HIGH,
        )
        self.sources["github"] = GitHubSource(config=github_config)
        
        # ArXiv source
        arxiv_config = SourceConfig(
            name="arxiv",
            base_url="http://export.arxiv.org",
            source_type="academic_papers",
            rate_limit_rps=3.0,
            max_results_per_query=15,
            priority=SourcePriority.HIGH,
        )
        self.sources["arxiv"] = ArxivSource(config=arxiv_config)
    
    def add_source(self, source: ResearchSource):
        """Add a new research source."""
        self.sources[source.config.name] = source
        logger.info("Added research source", name=source.config.name)
    
    def remove_source(self, name: str):
        """Remove a research source."""
        if name in self.sources:
            del self.sources[name]
            logger.info("Removed research source", name=name)
    
    def get_source(self, name: str) -> Optional[ResearchSource]:
        """Get a source by name."""
        return self.sources.get(name)
    
    async def select_sources(
        self,
        query: ProcessedQuery,
        preferred_sources: Optional[List[str]] = None,
    ) -> List[ResearchSource]:
        """Select the best sources for a given query."""
        if preferred_sources:
            # Use only specified sources
            selected = []
            for name in preferred_sources:
                if name in self.sources:
                    source = self.sources[name]
                    if source.status == SourceStatus.ACTIVE:
                        selected.append(source)
            return selected
        
        # Automatic source selection based on query
        source_scores = []
        
        for source in self.sources.values():
            if source.status != SourceStatus.ACTIVE:
                continue
            
            # Calculate confidence score
            confidence = source.can_handle_query(query)
            
            # Apply priority weighting
            priority_weight = {
                SourcePriority.HIGH: 1.2,
                SourcePriority.MEDIUM: 1.0,
                SourcePriority.LOW: 0.8,
            }.get(source.config.priority, 1.0)
            
            # Apply performance weighting
            performance_weight = 1.0
            if source.metrics:
                success_rate = source.metrics['successful_requests'] / max(source.metrics['total_requests'], 1)
                performance_weight = 0.5 + (success_rate * 0.5)
            
            final_score = confidence * priority_weight * performance_weight
            source_scores.append((source, final_score))
        
        # Sort by score and return top sources
        source_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 sources with score > 0.3
        selected_sources = []
        for source, score in source_scores:
            if score > 0.3 and len(selected_sources) < 3:
                selected_sources.append(source)
        
        if not selected_sources:
            # Fallback: return all active sources
            selected_sources = [s for s in self.sources.values() if s.status == SourceStatus.ACTIVE]
        
        return selected_sources
    
    async def search_all(
        self,
        query: ProcessedQuery,
        sources: Optional[List[ResearchSource]] = None,
    ) -> AsyncGenerator[ResearchResult, None]:
        """Search across multiple sources concurrently."""
        if sources is None:
            sources = await self.select_sources(query)
        
        if not sources:
            logger.warning("No active sources available for search")
            return
        
        logger.info("Searching sources", query=query.original_text, source_count=len(sources))
        
        # Create HTTP session
        async with aiohttp.ClientSession() as session:
            # Create search tasks for each source
            search_tasks = []
            for source in sources:
                task = asyncio.create_task(
                    self._search_source_with_logging(source, query, session)
                )
                search_tasks.append(task)
            
            # Collect results as they become available
            for task in asyncio.as_completed(search_tasks):
                try:
                    async for result in await task:
                        yield result
                except Exception as e:
                    logger.error("Source search task failed", error=str(e))
    
    async def _search_source_with_logging(
        self,
        source: ResearchSource,
        query: ProcessedQuery,
        session: aiohttp.ClientSession,
    ) -> AsyncGenerator[ResearchResult, None]:
        """Search a source with logging and metrics."""
        start_time = datetime.utcnow()
        
        try:
            logger.debug("Searching source", source=source.config.name, query=query.original_text)
            
            result_count = 0
            quality_sum = 0.0
            
            async for result in source.search(query, session):
                result_count += 1
                quality_sum += result.quality_score
                yield result
            
            # Log search to database
            search_duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._log_search(
                source.config.name,
                query,
                result_count,
                quality_sum / max(result_count, 1),
                search_duration,
                True
            )
            
        except Exception as e:
            logger.error("Source search failed", 
                        source=source.config.name, 
                        query=query.original_text, 
                        error=str(e))
            
            search_duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._log_search(
                source.config.name,
                query,
                0,
                0.0,
                search_duration,
                False
            )
    
    async def _log_search(
        self,
        source_name: str,
        query: ProcessedQuery,
        result_count: int,
        avg_quality: float,
        duration_ms: float,
        success: bool,
    ):
        """Log search results to database."""
        if not self.SessionLocal:
            return
        
        try:
            db = self.SessionLocal()
            
            # Create query hash for deduplication
            query_hash = hashlib.md5(query.original_text.encode()).hexdigest()
            
            search_record = SourceSearch(
                source_name=source_name,
                query_text=query.original_text,
                query_hash=query_hash,
                results_count=result_count,
                avg_quality=avg_quality,
                query_type=query.query_type.value,
                query_intent=query.intent.value,
                search_duration_ms=duration_ms,
                success=success,
            )
            
            db.add(search_record)
            db.commit()
            
        except Exception as e:
            logger.warning("Failed to log search", error=str(e))
        finally:
            if 'db' in locals():
                db.close()
    
    async def get_source_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all sources."""
        stats = {}
        
        for name, source in self.sources.items():
            stats[name] = {
                'status': source.status.value,
                'last_accessed': source.last_accessed.isoformat() if source.last_accessed else None,
                'metrics': source.metrics.copy() if source.metrics else {},
                'config': {
                    'priority': source.config.priority.value,
                    'rate_limit_rps': source.config.rate_limit_rps,
                    'max_results': source.config.max_results_per_query,
                }
            }
        
        return stats
    
    async def update_source_status(self, name: str, status: SourceStatus):
        """Update the status of a source."""
        if name in self.sources:
            self.sources[name].status = status
            logger.info("Updated source status", source=name, status=status.value)
    
    async def get_query_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent query history from database."""
        if not self.SessionLocal:
            return []
        
        try:
            db = self.SessionLocal()
            
            searches = db.query(SourceSearch)\
                        .order_by(SourceSearch.search_timestamp.desc())\
                        .limit(limit)\
                        .all()
            
            history = []
            for search in searches:
                history.append({
                    'query': search.query_text,
                    'source': search.source_name,
                    'timestamp': search.search_timestamp.isoformat(),
                    'results_count': search.results_count,
                    'success': search.success,
                    'duration_ms': search.search_duration_ms,
                })
            
            return history
            
        except Exception as e:
            logger.error("Failed to get query history", error=str(e))
            return []
        finally:
            if 'db' in locals():
                db.close()
    
    async def optimize_sources(self):
        """Optimize source selection based on performance data."""
        logger.info("Optimizing source selection")
        
        for source in self.sources.values():
            if not source.metrics or source.metrics['total_requests'] < 10:
                continue
            
            success_rate = source.metrics['successful_requests'] / source.metrics['total_requests']
            avg_response_time = source.metrics['average_response_time']
            
            # Disable sources with poor performance
            if success_rate < 0.5:
                source.status = SourceStatus.ERROR
                logger.warning("Disabled poor performing source", 
                              source=source.config.name, 
                              success_rate=success_rate)
            
            # Mark slow sources as low priority
            elif avg_response_time > 10000:  # 10 seconds
                source.config.priority = SourcePriority.LOW
                logger.info("Lowered priority for slow source",
                           source=source.config.name,
                           avg_response_time=avg_response_time)
    
    async def close(self):
        """Close the source manager and cleanup resources."""
        if self.engine:
            self.engine.dispose()
        
        logger.info("Source manager closed")