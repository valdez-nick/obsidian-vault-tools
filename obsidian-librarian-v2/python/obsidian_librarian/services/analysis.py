"""
Content Analysis Service for intelligent note analysis and duplicate detection.
"""

import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from collections import defaultdict
import re

import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ..models import Note, AnalysisResult, DuplicateCluster, ContentSimilarity
from ..vault import Vault
from ..ai.content_summarizer import ContentSummarizer

logger = structlog.get_logger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for content analysis operations."""
    # Similarity thresholds
    exact_duplicate_threshold: float = 0.98
    near_duplicate_threshold: float = 0.85
    similar_content_threshold: float = 0.7
    
    # Content analysis
    min_content_length: int = 50
    max_content_length: int = 50000
    ignore_frontmatter: bool = True
    
    # Performance settings
    batch_size: int = 100
    max_concurrent_analysis: int = 10
    enable_caching: bool = True
    
    # Feature extraction
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.8
    
    # Quality scoring
    enable_quality_scoring: bool = True
    quality_factors: Dict[str, float] = field(default_factory=lambda: {
        'length': 0.2,
        'structure': 0.3,
        'links': 0.2,
        'completeness': 0.3,
    })


@dataclass
class ContentMetrics:
    """Metrics for content quality and characteristics."""
    word_count: int
    paragraph_count: int
    heading_count: int
    link_count: int
    task_count: int
    code_block_count: int
    
    # Quality indicators
    has_frontmatter: bool
    has_headings: bool
    has_links: bool
    has_tasks: bool
    
    # Readability
    avg_sentence_length: float
    reading_time_minutes: float
    
    # Structure
    heading_hierarchy_score: float
    content_organization_score: float


class AnalysisService:
    """
    Intelligent content analysis service for Obsidian notes.
    
    Provides:
    - Duplicate detection and clustering
    - Content similarity analysis  
    - Quality scoring and metrics
    - Content recommendations
    - Batch analysis operations
    """
    
    def __init__(
        self,
        vault: Vault,
        config: Optional[AnalysisConfig] = None,
    ):
        self.vault = vault
        self.config = config or AnalysisConfig()
        self.content_summarizer = ContentSummarizer()
        
        # Analysis cache
        self._content_cache: Dict[str, str] = {}
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
        self._metrics_cache: Dict[str, ContentMetrics] = {}
        
        # Feature extraction
        self.vectorizer = None
        self._feature_matrix = None
        self._note_ids = []
        
        # Compiled regex patterns for performance
        self._patterns = self._compile_patterns()
        
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for content analysis."""
        return {
            'frontmatter': re.compile(r'^---\s*\n.*?\n---\s*\n', re.DOTALL | re.MULTILINE),
            'headings': re.compile(r'^#+\s+(.+)$', re.MULTILINE),
            'wiki_links': re.compile(r'\[\[([^\]]+)\]\]'),
            'markdown_links': re.compile(r'\[([^\]]+)\]\([^)]+\)'),
            'tasks': re.compile(r'^\s*[-*]\s+\[[x\s]\]\s+(.+)$', re.MULTILINE),
            'code_blocks': re.compile(r'```[\s\S]*?```'),
            'sentences': re.compile(r'[.!?]+\s+'),
            'words': re.compile(r'\b\w+\b'),
        }
    
    async def analyze_note(self, note_id: str) -> AnalysisResult:
        """
        Perform comprehensive analysis of a single note.
        
        Args:
            note_id: ID of the note to analyze
            
        Returns:
            Comprehensive analysis result
        """
        logger.info("Analyzing note", note_id=note_id)
        
        # Get note content
        note = await self.vault.get_note(note_id)
        if not note:
            raise ValueError(f"Note not found: {note_id}")
        
        # Extract content metrics
        metrics = await self._extract_content_metrics(note)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(metrics) if self.config.enable_quality_scoring else 0.0
        
        # Find similar notes
        similar_notes = await self.find_similar_notes(note_id, limit=10)
        
        # Generate content summary if note is long enough
        summary = None
        if len(note.content) > 200:
            try:
                summary = await self.content_summarizer.summarize(
                    note.content,
                    max_length=150,
                )
            except Exception as e:
                logger.warning("Failed to generate summary", note_id=note_id, error=str(e))
        
        # Extract key topics/tags
        topics = self._extract_topics(note.content)
        
        return AnalysisResult(
            note_id=note_id,
            metrics=metrics,
            quality_score=quality_score,
            similar_notes=similar_notes,
            summary=summary,
            topics=topics,
            recommendations=self._generate_recommendations(note, metrics, similar_notes),
            analysis_timestamp=datetime.utcnow(),
        )
    
    async def find_duplicates(
        self,
        note_ids: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> List[DuplicateCluster]:
        """
        Find duplicate and near-duplicate notes.
        
        Args:
            note_ids: Specific notes to check (default: all notes)
            threshold: Similarity threshold (default: config threshold)
            
        Returns:
            List of duplicate clusters
        """
        threshold = threshold or self.config.near_duplicate_threshold
        note_ids = note_ids or await self.vault.get_all_note_ids()
        
        logger.info("Finding duplicates", count=len(note_ids), threshold=threshold)
        
        # Build feature matrix if needed
        await self._build_feature_matrix(note_ids)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(self._feature_matrix)
        
        # Find clusters of similar notes
        clusters = []
        processed = set()
        
        for i, note_id in enumerate(note_ids):
            if note_id in processed:
                continue
                
            # Find all notes similar to this one
            similar_indices = np.where(similarity_matrix[i] >= threshold)[0]
            similar_notes = [note_ids[j] for j in similar_indices if j != i]
            
            if similar_notes:
                # Create cluster
                cluster_notes = [note_id] + similar_notes
                similarities = [
                    ContentSimilarity(
                        note_a=note_id,
                        note_b=similar_note,
                        similarity_score=similarity_matrix[i][note_ids.index(similar_note)],
                        similarity_type="content"
                    )
                    for similar_note in similar_notes
                ]
                
                cluster = DuplicateCluster(
                    cluster_id=f"cluster_{len(clusters)}",
                    note_ids=cluster_notes,
                    similarities=similarities,
                    cluster_type="near_duplicate" if threshold < self.config.exact_duplicate_threshold else "exact_duplicate",
                    confidence_score=float(np.mean([s.similarity_score for s in similarities])),
                )
                
                clusters.append(cluster)
                processed.update(cluster_notes)
        
        logger.info("Found duplicate clusters", count=len(clusters))
        return clusters
    
    async def find_similar_notes(
        self,
        note_id: str,
        limit: int = 10,
        threshold: Optional[float] = None,
    ) -> List[ContentSimilarity]:
        """
        Find notes similar to the given note.
        
        Args:
            note_id: Reference note ID
            limit: Maximum number of similar notes to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar notes with similarity scores
        """
        threshold = threshold or self.config.similar_content_threshold
        
        # Get all note IDs
        all_note_ids = await self.vault.get_all_note_ids()
        
        # Build feature matrix if needed
        await self._build_feature_matrix(all_note_ids)
        
        # Find the index of the target note
        try:
            note_index = self._note_ids.index(note_id)
        except ValueError:
            logger.warning("Note not found in feature matrix", note_id=note_id)
            return []
        
        # Calculate similarities
        note_vector = self._feature_matrix[note_index].reshape(1, -1)
        similarities = cosine_similarity(note_vector, self._feature_matrix)[0]
        
        # Get top similar notes (excluding self)
        similar_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in similar_indices:
            if len(results) >= limit:
                break
                
            if idx == note_index:  # Skip self
                continue
                
            similarity_score = similarities[idx]
            if similarity_score < threshold:
                break
                
            similar_note_id = self._note_ids[idx]
            results.append(ContentSimilarity(
                note_a=note_id,
                note_b=similar_note_id,
                similarity_score=similarity_score,
                similarity_type="content",
            ))
        
        return results
    
    async def batch_analyze(
        self,
        note_ids: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
    ) -> AsyncGenerator[AnalysisResult, None]:
        """
        Perform batch analysis of multiple notes.
        
        Args:
            note_ids: Notes to analyze (default: all notes)
            progress_callback: Optional callback for progress updates
            
        Yields:
            Analysis results as they complete
        """
        note_ids = note_ids or await self.vault.get_all_note_ids()
        
        logger.info("Starting batch analysis", count=len(note_ids))
        
        # Process notes in batches
        for i in range(0, len(note_ids), self.config.batch_size):
            batch = note_ids[i:i + self.config.batch_size]
            
            # Create analysis tasks
            tasks = [self.analyze_note(note_id) for note_id in batch]
            
            # Process batch with concurrency limit
            semaphore = asyncio.Semaphore(self.config.max_concurrent_analysis)
            
            async def analyze_with_limit(note_id):
                async with semaphore:
                    return await self.analyze_note(note_id)
            
            # Execute batch
            for task in asyncio.as_completed([analyze_with_limit(note_id) for note_id in batch]):
                try:
                    result = await task
                    if progress_callback:
                        progress_callback(result.note_id, i + len(batch), len(note_ids))
                    yield result
                except Exception as e:
                    logger.error("Analysis failed for note", error=str(e))
    
    async def get_content_statistics(self) -> Dict:
        """Get overall content statistics for the vault."""
        all_note_ids = await self.vault.get_all_note_ids()
        
        stats = {
            'total_notes': len(all_note_ids),
            'total_words': 0,
            'total_links': 0,
            'total_tasks': 0,
            'avg_quality_score': 0.0,
            'content_distribution': defaultdict(int),
        }
        
        quality_scores = []
        
        for note_id in all_note_ids[:100]:  # Sample for performance
            try:
                note = await self.vault.get_note(note_id)
                if note:
                    metrics = await self._extract_content_metrics(note)
                    
                    stats['total_words'] += metrics.word_count
                    stats['total_links'] += metrics.link_count
                    stats['total_tasks'] += metrics.task_count
                    
                    if self.config.enable_quality_scoring:
                        quality_score = self._calculate_quality_score(metrics)
                        quality_scores.append(quality_score)
                    
                    # Content distribution
                    if metrics.word_count < 100:
                        stats['content_distribution']['short'] += 1
                    elif metrics.word_count < 500:
                        stats['content_distribution']['medium'] += 1
                    else:
                        stats['content_distribution']['long'] += 1
                        
            except Exception as e:
                logger.warning("Failed to analyze note for stats", note_id=note_id, error=str(e))
        
        if quality_scores:
            stats['avg_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        return dict(stats)
    
    async def _extract_content_metrics(self, note: Note) -> ContentMetrics:
        """Extract comprehensive metrics from note content."""
        content = note.content
        
        # Remove frontmatter if configured
        if self.config.ignore_frontmatter:
            content = self._patterns['frontmatter'].sub('', content)
        
        # Basic counts
        words = self._patterns['words'].findall(content)
        word_count = len(words)
        
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        headings = self._patterns['headings'].findall(content)
        heading_count = len(headings)
        
        wiki_links = self._patterns['wiki_links'].findall(content)
        markdown_links = self._patterns['markdown_links'].findall(content)
        link_count = len(wiki_links) + len(markdown_links)
        
        tasks = self._patterns['tasks'].findall(content)
        task_count = len(tasks)
        
        code_blocks = self._patterns['code_blocks'].findall(content)
        code_block_count = len(code_blocks)
        
        # Quality indicators
        has_frontmatter = bool(self._patterns['frontmatter'].search(note.content))
        has_headings = heading_count > 0
        has_links = link_count > 0
        has_tasks = task_count > 0
        
        # Readability metrics
        sentences = self._patterns['sentences'].split(content)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Reading time (assuming 200 WPM)
        reading_time_minutes = word_count / 200.0
        
        # Structure scores
        heading_hierarchy_score = self._calculate_heading_hierarchy_score(content)
        content_organization_score = self._calculate_organization_score(content)
        
        return ContentMetrics(
            word_count=word_count,
            paragraph_count=paragraph_count,
            heading_count=heading_count,
            link_count=link_count,
            task_count=task_count,
            code_block_count=code_block_count,
            has_frontmatter=has_frontmatter,
            has_headings=has_headings,
            has_links=has_links,
            has_tasks=has_tasks,
            avg_sentence_length=avg_sentence_length,
            reading_time_minutes=reading_time_minutes,
            heading_hierarchy_score=heading_hierarchy_score,
            content_organization_score=content_organization_score,
        )
    
    def _calculate_quality_score(self, metrics: ContentMetrics) -> float:
        """Calculate overall quality score for content."""
        factors = self.config.quality_factors
        score = 0.0
        
        # Length factor (optimal range: 300-2000 words)
        length_score = min(metrics.word_count / 2000, 1.0) if metrics.word_count < 2000 else max(2000 / metrics.word_count, 0.5)
        score += factors['length'] * length_score
        
        # Structure factor
        structure_score = 0.0
        if metrics.has_headings:
            structure_score += 0.3
        if metrics.heading_hierarchy_score > 0.7:
            structure_score += 0.3
        if metrics.content_organization_score > 0.6:
            structure_score += 0.4
        score += factors['structure'] * structure_score
        
        # Links factor (indicates connections)
        link_score = min(metrics.link_count / 10, 1.0)
        score += factors['links'] * link_score
        
        # Completeness factor
        completeness_score = 0.0
        if metrics.has_frontmatter:
            completeness_score += 0.25
        if metrics.paragraph_count > 2:
            completeness_score += 0.25
        if metrics.word_count > 100:
            completeness_score += 0.25
        if metrics.avg_sentence_length > 5 and metrics.avg_sentence_length < 25:
            completeness_score += 0.25
        score += factors['completeness'] * completeness_score
        
        return min(score, 1.0)
    
    def _calculate_heading_hierarchy_score(self, content: str) -> float:
        """Calculate how well headings follow a logical hierarchy."""
        heading_matches = [(m.start(), len(m.group().split()[0])) for m in re.finditer(r'^(#+)\s+', content, re.MULTILINE)]
        
        if len(heading_matches) < 2:
            return 0.5  # Neutral score for single/no headings
        
        # Check hierarchy consistency
        violations = 0
        for i in range(1, len(heading_matches)):
            prev_level = heading_matches[i-1][1]
            curr_level = heading_matches[i][1]
            
            # Violation if jumping more than one level down
            if curr_level > prev_level + 1:
                violations += 1
        
        hierarchy_score = 1.0 - (violations / len(heading_matches))
        return max(hierarchy_score, 0.0)
    
    def _calculate_organization_score(self, content: str) -> float:
        """Calculate how well content is organized."""
        score = 0.0
        
        # Check for introduction (first paragraph)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if paragraphs and len(paragraphs[0]) > 50:
            score += 0.3
        
        # Check for conclusion (last paragraph mentions summary keywords)
        if paragraphs and len(paragraphs) > 1:
            conclusion_keywords = ['conclusion', 'summary', 'in summary', 'to conclude', 'overall']
            last_paragraph = paragraphs[-1].lower()
            if any(keyword in last_paragraph for keyword in conclusion_keywords):
                score += 0.3
        
        # Check for balanced paragraph lengths
        if paragraphs:
            avg_length = sum(len(p) for p in paragraphs) / len(paragraphs)
            length_variance = sum((len(p) - avg_length) ** 2 for p in paragraphs) / len(paragraphs)
            if length_variance < avg_length:  # Low variance = good balance
                score += 0.4
        
        return score
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract key topics from content using simple keyword extraction."""
        # Remove common words and extract meaningful terms
        words = self._patterns['words'].findall(content.lower())
        
        # Simple frequency-based topic extraction
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 3:  # Filter short words
                word_freq[word] += 1
        
        # Get top terms
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [topic[0] for topic in topics]
    
    def _generate_recommendations(
        self,
        note: Note,
        metrics: ContentMetrics,
        similar_notes: List[ContentSimilarity],
    ) -> List[str]:
        """Generate improvement recommendations for a note."""
        recommendations = []
        
        # Content length recommendations
        if metrics.word_count < 100:
            recommendations.append("Consider expanding this note with more detailed content")
        elif metrics.word_count > 3000:
            recommendations.append("Consider breaking this note into smaller, focused notes")
        
        # Structure recommendations
        if not metrics.has_headings and metrics.word_count > 200:
            recommendations.append("Add headings to improve content structure")
        
        if metrics.heading_hierarchy_score < 0.6:
            recommendations.append("Review heading hierarchy for better organization")
        
        # Linking recommendations
        if metrics.link_count == 0 and metrics.word_count > 100:
            recommendations.append("Add links to related notes to improve connectivity")
        
        # Duplicate recommendations
        high_similarity_notes = [s for s in similar_notes if s.similarity_score > 0.9]
        if high_similarity_notes:
            recommendations.append(f"Consider merging with similar note: {high_similarity_notes[0].note_b}")
        
        # Frontmatter recommendations
        if not metrics.has_frontmatter:
            recommendations.append("Add frontmatter with tags and metadata")
        
        return recommendations
    
    async def _build_feature_matrix(self, note_ids: List[str]) -> None:
        """Build TF-IDF feature matrix for similarity calculations."""
        if self.vectorizer is not None and set(note_ids) == set(self._note_ids):
            return  # Already built for these notes
        
        logger.info("Building feature matrix", note_count=len(note_ids))
        
        # Extract content for all notes
        contents = []
        valid_note_ids = []
        
        for note_id in note_ids:
            note = await self.vault.get_note(note_id)
            if note and len(note.content) >= self.config.min_content_length:
                # Preprocess content
                content = note.content
                if self.config.ignore_frontmatter:
                    content = self._patterns['frontmatter'].sub('', content)
                
                contents.append(content)
                valid_note_ids.append(note_id)
        
        if not contents:
            logger.warning("No valid content found for feature matrix")
            return
        
        # Build TF-IDF matrix
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            stop_words='english',
            lowercase=True,
        )
        
        self._feature_matrix = self.vectorizer.fit_transform(contents)
        self._note_ids = valid_note_ids
        
        logger.info("Feature matrix built", 
                   shape=self._feature_matrix.shape, 
                   features=len(self.vectorizer.get_feature_names_out()))
    
    async def clear_cache(self) -> None:
        """Clear all analysis caches."""
        self._content_cache.clear()
        self._similarity_cache.clear()
        self._metrics_cache.clear()
        self.vectorizer = None
        self._feature_matrix = None
        self._note_ids = []
        logger.info("Analysis caches cleared")


# Content analysis result models
@dataclass 
class AnalysisResult:
    """Result from content analysis."""
    note_id: str
    metrics: ContentMetrics
    quality_score: float
    similar_notes: List[ContentSimilarity]
    summary: Optional[str]
    topics: List[str]
    recommendations: List[str]
    analysis_timestamp: datetime