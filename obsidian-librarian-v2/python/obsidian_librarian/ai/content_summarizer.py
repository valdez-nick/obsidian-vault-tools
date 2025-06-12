"""
Content summarization for intelligent text processing and note creation.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import math

import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = structlog.get_logger(__name__)


class SummaryType(Enum):
    """Types of summaries that can be generated."""
    EXTRACTIVE = "extractive"  # Extract key sentences
    ABSTRACTIVE = "abstractive"  # Generate new summary text
    BULLET_POINTS = "bullet_points"  # Key points as bullets
    STRUCTURED = "structured"  # Structured with headings
    BRIEF = "brief"  # Very short summary
    DETAILED = "detailed"  # Comprehensive summary


class ContentType(Enum):
    """Types of content being summarized."""
    ARTICLE = "article"
    RESEARCH_PAPER = "research_paper"
    DOCUMENTATION = "documentation"
    NEWS = "news"
    BLOG_POST = "blog_post"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    CONVERSATION = "conversation"


@dataclass
class SummaryConfig:
    """Configuration for content summarization."""
    # Summary parameters
    max_length: int = 300
    min_length: int = 50
    summary_type: SummaryType = SummaryType.EXTRACTIVE
    
    # Content analysis
    preserve_structure: bool = True
    include_key_points: bool = True
    maintain_context: bool = True
    
    # Language processing
    remove_redundancy: bool = True
    normalize_text: bool = True
    preserve_technical_terms: bool = True
    
    # Quality control
    coherence_threshold: float = 0.7
    relevance_threshold: float = 0.6
    
    # Output formatting
    use_markdown: bool = True
    include_statistics: bool = False


@dataclass
class SummaryResult:
    """Result from content summarization."""
    # Summary content
    summary: str
    summary_type: SummaryType
    
    # Metadata
    original_length: int
    summary_length: int
    compression_ratio: float
    
    # Key information
    key_points: List[str]
    main_topics: List[str]
    key_entities: List[str]
    
    # Quality metrics
    coherence_score: float
    coverage_score: float
    relevance_score: float
    
    # Additional data
    word_count: int
    reading_time_minutes: float
    complexity_score: float


class ContentSummarizer:
    """
    Intelligent content summarizer using extractive and abstractive techniques.
    
    Provides:
    - Multi-type summarization (extractive, abstractive, structured)
    - Content-aware processing for different document types
    - Quality assessment and optimization
    - Configurable output formats
    """
    
    def __init__(self):
        # Text processing patterns
        self._patterns = self._compile_patterns()
        
        # Stop words for content filtering
        self._stop_words = self._load_stop_words()
        
        # Content type classifiers
        self._content_classifiers = self._build_content_classifiers()
        
        # Summary cache
        self._summary_cache: Dict[str, SummaryResult] = {}
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for text processing."""
        return {
            # Sentence splitting
            'sentence_split': re.compile(r'(?<=[.!?])\s+(?=[A-Z])'),
            
            # Paragraph splitting
            'paragraph_split': re.compile(r'\n\s*\n'),
            
            # Headings
            'markdown_headings': re.compile(r'^#+\s+(.+)$', re.MULTILINE),
            'underline_headings': re.compile(r'^(.+)\n[-=]+\s*$', re.MULTILINE),
            
            # Lists
            'bullet_lists': re.compile(r'^\s*[-*+]\s+(.+)$', re.MULTILINE),
            'numbered_lists': re.compile(r'^\s*\d+\.\s+(.+)$', re.MULTILINE),
            
            # Code blocks
            'code_blocks': re.compile(r'```[\s\S]*?```|`[^`]+`'),
            
            # Links and references
            'markdown_links': re.compile(r'\[([^\]]+)\]\([^)]+\)'),
            'urls': re.compile(r'https?://[^\s]+'),
            
            # Technical terms
            'technical_terms': re.compile(
                r'\b(?:API|JSON|XML|HTTP|REST|GraphQL|SQL|NoSQL|ML|AI|NLP|CNN|RNN|LSTM|GAN)\b'
            ),
            
            # Noise patterns
            'noise_chars': re.compile(r'[^\w\s\-\.\,\!\?\:\;\(\)]'),
            'excessive_whitespace': re.compile(r'\s{2,}'),
            
            # Emphasis
            'bold_text': re.compile(r'\*\*([^*]+)\*\*|__([^_]+)__'),
            'italic_text': re.compile(r'\*([^*]+)\*|_([^_]+)_'),
        }
    
    def _load_stop_words(self) -> set:
        """Load stop words for content filtering."""
        # Basic English stop words
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
            'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
            'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
    
    def _build_content_classifiers(self) -> Dict[ContentType, Dict[str, Any]]:
        """Build classifiers for different content types."""
        return {
            ContentType.RESEARCH_PAPER: {
                'keywords': ['abstract', 'methodology', 'results', 'conclusion', 'references'],
                'structure_indicators': ['introduction', 'methods', 'discussion'],
                'weight_multiplier': 1.2,
            },
            ContentType.DOCUMENTATION: {
                'keywords': ['api', 'function', 'parameter', 'example', 'usage'],
                'structure_indicators': ['overview', 'examples', 'reference'],
                'weight_multiplier': 1.1,
            },
            ContentType.TUTORIAL: {
                'keywords': ['step', 'tutorial', 'guide', 'how to', 'example'],
                'structure_indicators': ['getting started', 'next steps', 'summary'],
                'weight_multiplier': 1.0,
            },
            ContentType.NEWS: {
                'keywords': ['breaking', 'reported', 'according', 'sources'],
                'structure_indicators': ['who', 'what', 'when', 'where', 'why'],
                'weight_multiplier': 0.9,
            },
        }
    
    async def summarize(
        self,
        text: str,
        config: Optional[SummaryConfig] = None,
        content_type: Optional[ContentType] = None,
    ) -> str:
        """
        Generate a summary of the given text.
        
        Args:
            text: The text to summarize
            config: Summarization configuration
            content_type: Type of content being summarized
            
        Returns:
            Generated summary text
        """
        full_result = await self.summarize_detailed(text, config, content_type)
        return full_result.summary
    
    async def summarize_detailed(
        self,
        text: str,
        config: Optional[SummaryConfig] = None,
        content_type: Optional[ContentType] = None,
    ) -> SummaryResult:
        """
        Generate a detailed summary with metadata and quality metrics.
        
        Args:
            text: The text to summarize
            config: Summarization configuration
            content_type: Type of content being summarized
            
        Returns:
            Detailed summary result with metadata
        """
        config = config or SummaryConfig()
        
        logger.debug("Generating summary", 
                    length=len(text), 
                    type=config.summary_type.value,
                    content_type=content_type.value if content_type else None)
        
        # Check cache
        cache_key = self._generate_cache_key(text, config)
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]
        
        # Detect content type if not provided
        if content_type is None:
            content_type = self._detect_content_type(text)
        
        # Preprocess text
        processed_text = self._preprocess_text(text, config)
        
        # Extract structure
        structure = self._extract_structure(processed_text)
        
        # Generate summary based on type
        if config.summary_type == SummaryType.EXTRACTIVE:
            summary = await self._extractive_summarize(processed_text, config, structure)
        elif config.summary_type == SummaryType.BULLET_POINTS:
            summary = await self._bullet_point_summarize(processed_text, config, structure)
        elif config.summary_type == SummaryType.STRUCTURED:
            summary = await self._structured_summarize(processed_text, config, structure)
        else:
            # Default to extractive
            summary = await self._extractive_summarize(processed_text, config, structure)
        
        # Extract metadata
        key_points = self._extract_key_points(processed_text, structure)
        main_topics = self._extract_main_topics(processed_text)
        key_entities = self._extract_key_entities(processed_text)
        
        # Calculate quality metrics
        coherence_score = self._calculate_coherence(summary, processed_text)
        coverage_score = self._calculate_coverage(summary, processed_text, key_points)
        relevance_score = self._calculate_relevance(summary, main_topics)
        
        # Create result
        result = SummaryResult(
            summary=summary,
            summary_type=config.summary_type,
            original_length=len(text),
            summary_length=len(summary),
            compression_ratio=len(summary) / len(text),
            key_points=key_points,
            main_topics=main_topics,
            key_entities=key_entities,
            coherence_score=coherence_score,
            coverage_score=coverage_score,
            relevance_score=relevance_score,
            word_count=len(summary.split()),
            reading_time_minutes=len(summary.split()) / 200,  # 200 WPM average
            complexity_score=self._calculate_complexity(summary),
        )
        
        # Cache result
        self._summary_cache[cache_key] = result
        
        # Limit cache size
        if len(self._summary_cache) > 100:
            oldest_key = next(iter(self._summary_cache))
            del self._summary_cache[oldest_key]
        
        logger.debug("Summary generated", 
                    compression_ratio=result.compression_ratio,
                    coherence=result.coherence_score,
                    coverage=result.coverage_score)
        
        return result
    
    def _generate_cache_key(self, text: str, config: SummaryConfig) -> str:
        """Generate a cache key for the text and config."""
        import hashlib
        
        # Create a hash of the text and key config parameters
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        config_str = f"{config.max_length}_{config.summary_type.value}_{config.preserve_structure}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{text_hash}_{config_hash}"
    
    def _detect_content_type(self, text: str) -> ContentType:
        """Automatically detect the type of content."""
        text_lower = text.lower()
        
        # Check for research paper indicators
        research_indicators = ['abstract', 'methodology', 'references', 'citation']
        if sum(1 for indicator in research_indicators if indicator in text_lower) >= 2:
            return ContentType.RESEARCH_PAPER
        
        # Check for documentation indicators
        doc_indicators = ['api', 'function', 'parameter', 'usage', 'example']
        if sum(1 for indicator in doc_indicators if indicator in text_lower) >= 2:
            return ContentType.DOCUMENTATION
        
        # Check for tutorial indicators
        tutorial_indicators = ['step', 'tutorial', 'guide', 'how to']
        if any(indicator in text_lower for indicator in tutorial_indicators):
            return ContentType.TUTORIAL
        
        # Check for news indicators
        news_indicators = ['reported', 'breaking', 'according to', 'sources say']
        if any(indicator in text_lower for indicator in news_indicators):
            return ContentType.NEWS
        
        # Default to article
        return ContentType.ARTICLE
    
    def _preprocess_text(self, text: str, config: SummaryConfig) -> str:
        """Preprocess text for summarization."""
        processed = text
        
        if config.normalize_text:
            # Remove excessive whitespace
            processed = self._patterns['excessive_whitespace'].sub(' ', processed)
            
            # Clean noise characters but preserve technical terms
            if not config.preserve_technical_terms:
                processed = self._patterns['noise_chars'].sub(' ', processed)
        
        # Remove code blocks if not preserving structure
        if not config.preserve_structure:
            processed = self._patterns['code_blocks'].sub('[CODE]', processed)
        
        return processed.strip()
    
    def _extract_structure(self, text: str) -> Dict[str, Any]:
        """Extract structural elements from text."""
        structure = {
            'headings': [],
            'paragraphs': [],
            'lists': [],
            'emphasized_text': [],
        }
        
        # Extract headings
        for match in self._patterns['markdown_headings'].finditer(text):
            structure['headings'].append(match.group(1).strip())
        
        # Extract paragraphs
        paragraphs = self._patterns['paragraph_split'].split(text)
        structure['paragraphs'] = [p.strip() for p in paragraphs if p.strip()]
        
        # Extract list items
        for match in self._patterns['bullet_lists'].finditer(text):
            structure['lists'].append(match.group(1).strip())
        
        for match in self._patterns['numbered_lists'].finditer(text):
            structure['lists'].append(match.group(1).strip())
        
        # Extract emphasized text
        for match in self._patterns['bold_text'].finditer(text):
            emphasized = match.group(1) or match.group(2)
            if emphasized:
                structure['emphasized_text'].append(emphasized.strip())
        
        return structure
    
    async def _extractive_summarize(
        self,
        text: str,
        config: SummaryConfig,
        structure: Dict[str, Any],
    ) -> str:
        """Generate extractive summary by selecting key sentences."""
        # Split into sentences
        sentences = self._patterns['sentence_split'].split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            return text  # Too short to summarize
        
        # Score sentences
        sentence_scores = self._score_sentences(sentences, structure)
        
        # Select top sentences
        num_sentences = max(2, min(len(sentences) // 3, config.max_length // 50))
        
        # Get top-scored sentences maintaining order
        top_indices = sorted(
            range(len(sentence_scores)),
            key=lambda i: sentence_scores[i],
            reverse=True
        )[:num_sentences]
        
        top_indices.sort()  # Maintain original order
        
        summary_sentences = [sentences[i] for i in top_indices]
        summary = ' '.join(summary_sentences)
        
        # Trim to max length if needed
        if len(summary) > config.max_length:
            words = summary.split()
            summary = ' '.join(words[:config.max_length//5])  # Rough word count
            if not summary.endswith('.'):
                summary += '...'
        
        return summary
    
    async def _bullet_point_summarize(
        self,
        text: str,
        config: SummaryConfig,
        structure: Dict[str, Any],
    ) -> str:
        """Generate bullet point summary."""
        # Extract key points from structure
        key_points = []
        
        # Use existing list items
        key_points.extend(structure['lists'][:5])
        
        # Extract key sentences if not enough list items
        if len(key_points) < 3:
            sentences = self._patterns['sentence_split'].split(text)
            sentence_scores = self._score_sentences(sentences, structure)
            
            top_sentences = sorted(
                zip(sentences, sentence_scores),
                key=lambda x: x[1],
                reverse=True
            )[:5-len(key_points)]
            
            for sentence, _ in top_sentences:
                key_points.append(sentence.strip())
        
        # Format as bullet points
        formatted_points = []
        for point in key_points[:5]:  # Limit to 5 points
            # Clean and truncate
            clean_point = point.strip().rstrip('.')
            if len(clean_point) > 100:
                clean_point = clean_point[:97] + '...'
            formatted_points.append(f"• {clean_point}")
        
        return '\n'.join(formatted_points)
    
    async def _structured_summarize(
        self,
        text: str,
        config: SummaryConfig,
        structure: Dict[str, Any],
    ) -> str:
        """Generate structured summary with sections."""
        summary_parts = []
        
        # Overview section
        overview = await self._extractive_summarize(text, config, structure)
        summary_parts.append(f"## Overview\n\n{overview}")
        
        # Key points section
        if structure['lists'] or structure['emphasized_text']:
            summary_parts.append("## Key Points")
            
            points = []
            if structure['lists']:
                points.extend(structure['lists'][:3])
            if structure['emphasized_text']:
                points.extend(structure['emphasized_text'][:3])
            
            for point in points[:5]:
                summary_parts.append(f"- {point.strip()}")
        
        # Main topics (if we have headings)
        if structure['headings']:
            summary_parts.append("## Topics Covered")
            for heading in structure['headings'][:5]:
                summary_parts.append(f"- {heading}")
        
        return '\n'.join(summary_parts)
    
    def _score_sentences(self, sentences: List[str], structure: Dict[str, Any]) -> List[float]:
        """Score sentences for importance."""
        if not sentences:
            return []
        
        scores = []
        
        # TF-IDF based scoring
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        except:
            # Fallback to simple scoring
            sentence_scores = [1.0] * len(sentences)
        
        for i, sentence in enumerate(sentences):
            score = sentence_scores[i] if i < len(sentence_scores) else 0.0
            
            # Position bonus (first and last sentences often important)
            if i == 0 or i == len(sentences) - 1:
                score *= 1.2
            
            # Length penalty for very short or very long sentences
            word_count = len(sentence.split())
            if word_count < 5 or word_count > 40:
                score *= 0.8
            
            # Emphasis bonus
            if any(emp in sentence for emp in structure['emphasized_text']):
                score *= 1.3
            
            # Technical term bonus
            if self._patterns['technical_terms'].search(sentence):
                score *= 1.1
            
            # Numbers and statistics bonus
            if re.search(r'\d+%|\d+\.\d+|\$\d+', sentence):
                score *= 1.1
            
            scores.append(score)
        
        return scores
    
    def _extract_key_points(self, text: str, structure: Dict[str, Any]) -> List[str]:
        """Extract key points from the text."""
        key_points = []
        
        # Use list items from structure
        key_points.extend(structure['lists'][:5])
        
        # Use emphasized text
        key_points.extend(structure['emphasized_text'][:3])
        
        # Extract sentences with key indicators
        sentences = self._patterns['sentence_split'].split(text)
        key_indicators = ['important', 'key', 'main', 'primary', 'essential', 'crucial']
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in key_indicators):
                key_points.append(sentence.strip())
                if len(key_points) >= 8:
                    break
        
        # Remove duplicates and limit
        unique_points = []
        for point in key_points:
            if point not in unique_points and len(point) > 10:
                unique_points.append(point)
        
        return unique_points[:6]
    
    def _extract_main_topics(self, text: str) -> List[str]:
        """Extract main topics from the text."""
        # Use TF-IDF to find important terms
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [word for word in words if word not in self._stop_words]
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [word for word, freq in top_words if freq > 1]
    
    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities (simplified approach)."""
        entities = []
        
        # Technical terms
        tech_matches = self._patterns['technical_terms'].findall(text)
        entities.extend(tech_matches)
        
        # Capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(capitalized[:5])
        
        # Remove duplicates
        return list(set(entities))[:8]
    
    def _calculate_coherence(self, summary: str, original: str) -> float:
        """Calculate coherence score of the summary."""
        # Simple coherence based on sentence connectivity
        summary_sentences = self._patterns['sentence_split'].split(summary)
        
        if len(summary_sentences) < 2:
            return 1.0
        
        # Check for transition words and coherence indicators
        transition_words = ['however', 'therefore', 'moreover', 'furthermore', 'additionally']
        coherence_score = 0.5  # Base score
        
        for sentence in summary_sentences:
            if any(word in sentence.lower() for word in transition_words):
                coherence_score += 0.1
        
        # Penalize abrupt topic changes (simplified)
        words_per_sentence = [len(s.split()) for s in summary_sentences]
        if max(words_per_sentence) - min(words_per_sentence) < 15:
            coherence_score += 0.2
        
        return min(coherence_score, 1.0)
    
    def _calculate_coverage(self, summary: str, original: str, key_points: List[str]) -> float:
        """Calculate how well the summary covers the original content."""
        summary_lower = summary.lower()
        
        # Check coverage of key points
        covered_points = sum(1 for point in key_points if any(word in summary_lower for word in point.lower().split()))
        
        coverage = covered_points / max(len(key_points), 1) if key_points else 0.5
        
        return min(coverage, 1.0)
    
    def _calculate_relevance(self, summary: str, main_topics: List[str]) -> float:
        """Calculate relevance of summary to main topics."""
        if not main_topics:
            return 0.5
        
        summary_lower = summary.lower()
        covered_topics = sum(1 for topic in main_topics if topic in summary_lower)
        
        relevance = covered_topics / len(main_topics)
        return min(relevance, 1.0)
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        sentences = self._patterns['sentence_split'].split(text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple complexity based on average sentence length
        # 10-15 words = simple, 15-20 = medium, 20+ = complex
        if avg_sentence_length < 15:
            return 0.3
        elif avg_sentence_length < 20:
            return 0.6
        else:
            return 0.9
    
    async def get_summary_statistics(self) -> Dict[str, Any]:
        """Get statistics about summarization performance."""
        if not self._summary_cache:
            return {'cache_size': 0}
        
        cached_results = list(self._summary_cache.values())
        
        return {
            'cache_size': len(self._summary_cache),
            'average_compression_ratio': sum(r.compression_ratio for r in cached_results) / len(cached_results),
            'average_coherence': sum(r.coherence_score for r in cached_results) / len(cached_results),
            'average_coverage': sum(r.coverage_score for r in cached_results) / len(cached_results),
            'average_relevance': sum(r.relevance_score for r in cached_results) / len(cached_results),
        }
    
    def clear_cache(self) -> None:
        """Clear the summary cache."""
        self._summary_cache.clear()
        logger.info("Summary cache cleared")