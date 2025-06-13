"""
Advanced content analyzer using AI models.

Provides intelligent content analysis, quality scoring, and
automated categorization with multiple AI providers.
"""

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

import structlog

from .language_models import LanguageModelService, ChatRequest, ChatMessage
from .embeddings import EmbeddingService
from ..database.base import DatabaseManager
from ..models import Note

logger = structlog.get_logger(__name__)


class ContentType(str, Enum):
    """Types of content."""
    ARTICLE = "article"
    MEETING_NOTES = "meeting_notes"
    DAILY_JOURNAL = "daily_journal"
    PROJECT_PLAN = "project_plan"
    RESEARCH_NOTES = "research_notes"
    REFERENCE = "reference"
    TODO_LIST = "todo_list"
    BRAINSTORM = "brainstorm"
    CODE_NOTES = "code_notes"
    UNKNOWN = "unknown"


class QualityMetric(str, Enum):
    """Quality metrics for content."""
    STRUCTURE = "structure"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    ACTIONABILITY = "actionability"


@dataclass
class ContentAnalysis:
    """Results of content analysis."""
    note_id: str
    content_type: ContentType
    quality_score: float  # 0.0 to 1.0
    quality_metrics: Dict[QualityMetric, float]
    key_topics: List[str]
    sentiment: str  # positive, negative, neutral
    reading_time_minutes: float
    complexity_score: float  # 0.0 to 1.0 (readability)
    suggestions: List[str]
    tags_suggested: List[str]
    structure_issues: List[str]
    similar_notes: List[str]


@dataclass
class TopicCluster:
    """Cluster of related topics."""
    cluster_id: str
    main_topic: str
    related_topics: List[str]
    note_ids: List[str]
    coherence_score: float


class ContentAnalyzer:
    """AI-powered content analyzer."""
    
    def __init__(
        self,
        language_service: LanguageModelService,
        embedding_service: EmbeddingService,
        database_manager: Optional[DatabaseManager] = None
    ):
        self.language_service = language_service
        self.embedding_service = embedding_service
        self.db_manager = database_manager
        self._analysis_cache: Dict[str, ContentAnalysis] = {}
        self._lock = asyncio.Lock()
    
    async def analyze_note(
        self, 
        note: Note,
        include_suggestions: bool = True,
        find_similar: bool = True
    ) -> ContentAnalysis:
        """Perform comprehensive analysis of a note."""
        # Check cache first
        cache_key = f"analysis:{note.id}:{note.content_hash}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        logger.debug("Analyzing note", note_id=note.id)
        
        # Parallel analysis tasks
        tasks = [
            self._classify_content_type(note.content),
            self._analyze_quality(note.content),
            self._extract_topics(note.content),
            self._analyze_sentiment(note.content),
            self._calculate_reading_time(note.content),
            self._calculate_complexity(note.content),
        ]
        
        if include_suggestions:
            tasks.append(self._generate_suggestions(note.content))
            tasks.append(self._suggest_tags(note.content))
            tasks.append(self._analyze_structure(note.content))
        
        similar_notes = []
        if find_similar:
            similar_notes = await self._find_similar_notes(note)
        
        # Execute tasks
        results = await asyncio.gather(*tasks)
        
        content_type = results[0]
        quality_metrics = results[1]
        topics = results[2]
        sentiment = results[3]
        reading_time = results[4]
        complexity = results[5]
        
        # Calculate overall quality score
        quality_score = sum(quality_metrics.values()) / len(quality_metrics)
        
        suggestions = results[6] if include_suggestions else []
        suggested_tags = results[7] if include_suggestions else []
        structure_issues = results[8] if include_suggestions else []
        
        analysis = ContentAnalysis(
            note_id=note.id,
            content_type=content_type,
            quality_score=quality_score,
            quality_metrics=quality_metrics,
            key_topics=topics,
            sentiment=sentiment,
            reading_time_minutes=reading_time,
            complexity_score=complexity,
            suggestions=suggestions,
            tags_suggested=suggested_tags,
            structure_issues=structure_issues,
            similar_notes=[sn[0] for sn in similar_notes[:5]],  # Top 5 similar
        )
        
        # Cache result
        self._analysis_cache[cache_key] = analysis
        
        # Store in database
        if self.db_manager and self.db_manager.analytics:
            await self._store_analysis(analysis, note)
        
        return analysis
    
    async def _classify_content_type(self, content: str) -> ContentType:
        """Classify the type of content using AI."""
        prompt = f"""Analyze the following note content and classify it into one of these categories:
- article: Long-form informational content
- meeting_notes: Notes from meetings or discussions
- daily_journal: Personal daily reflections or logs
- project_plan: Project planning and management content
- research_notes: Research findings and analysis
- reference: Reference material or documentation
- todo_list: Task lists and action items
- brainstorm: Brainstorming sessions and idea lists
- code_notes: Programming and technical notes
- unknown: Cannot be clearly categorized

Content:
{content[:1000]}...

Respond with only the category name."""
        
        try:
            request = ChatRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=50,
            )
            
            response = await self.language_service.chat_completion(request)
            classification = response.text.strip().lower()
            
            # Map to enum
            for content_type in ContentType:
                if content_type.value in classification:
                    return content_type
            
            return ContentType.UNKNOWN
            
        except Exception as e:
            logger.warning("Content classification failed", error=str(e))
            return ContentType.UNKNOWN
    
    async def _analyze_quality(self, content: str) -> Dict[QualityMetric, float]:
        """Analyze content quality across multiple dimensions."""
        prompt = f"""Analyze the quality of this note content across these dimensions and provide scores from 0.0 to 1.0:

1. Structure: How well-organized and structured is the content?
2. Clarity: How clear and understandable is the writing?
3. Completeness: How complete and comprehensive is the information?
4. Relevance: How relevant and focused is the content?
5. Actionability: How actionable and useful is the content?

Content:
{content[:1500]}...

Respond in this exact format:
Structure: 0.X
Clarity: 0.X
Completeness: 0.X
Relevance: 0.X
Actionability: 0.X"""
        
        try:
            request = ChatRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.1,
                max_tokens=200,
            )
            
            response = await self.language_service.chat_completion(request)
            
            # Parse scores
            scores = {}
            for line in response.text.split('\n'):
                if ':' in line:
                    metric, score_str = line.split(':', 1)
                    metric = metric.strip().lower()
                    try:
                        score = float(score_str.strip())
                        if metric == "structure":
                            scores[QualityMetric.STRUCTURE] = score
                        elif metric == "clarity":
                            scores[QualityMetric.CLARITY] = score
                        elif metric == "completeness":
                            scores[QualityMetric.COMPLETENESS] = score
                        elif metric == "relevance":
                            scores[QualityMetric.RELEVANCE] = score
                        elif metric == "actionability":
                            scores[QualityMetric.ACTIONABILITY] = score
                    except ValueError:
                        continue
            
            # Ensure all metrics have scores
            for metric in QualityMetric:
                if metric not in scores:
                    scores[metric] = 0.5  # Default neutral score
            
            return scores
            
        except Exception as e:
            logger.warning("Quality analysis failed", error=str(e))
            return {metric: 0.5 for metric in QualityMetric}
    
    async def _extract_topics(self, content: str) -> List[str]:
        """Extract key topics from content."""
        prompt = f"""Extract the 3-5 most important topics or themes from this content. 
Provide them as a comma-separated list of 1-3 word phrases.

Content:
{content[:1500]}...

Topics:"""
        
        try:
            request = ChatRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=100,
            )
            
            response = await self.language_service.chat_completion(request)
            
            # Parse topics
            topics = [topic.strip() for topic in response.text.split(',')]
            topics = [topic for topic in topics if topic and len(topic) > 2]
            
            return topics[:5]  # Limit to 5 topics
            
        except Exception as e:
            logger.warning("Topic extraction failed", error=str(e))
            return []
    
    async def _analyze_sentiment(self, content: str) -> str:
        """Analyze sentiment of content."""
        prompt = f"""Analyze the overall sentiment/tone of this content. 
Respond with exactly one word: positive, negative, or neutral.

Content:
{content[:1000]}...

Sentiment:"""
        
        try:
            request = ChatRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.1,
                max_tokens=10,
            )
            
            response = await self.language_service.chat_completion(request)
            sentiment = response.text.strip().lower()
            
            if sentiment in ["positive", "negative", "neutral"]:
                return sentiment
            else:
                return "neutral"
                
        except Exception as e:
            logger.warning("Sentiment analysis failed", error=str(e))
            return "neutral"
    
    async def _calculate_reading_time(self, content: str) -> float:
        """Calculate estimated reading time in minutes."""
        # Remove markdown formatting
        text = re.sub(r'[#*_`\[\]()]', '', content)
        words = len(text.split())
        
        # Average reading speed: 200-250 words per minute
        reading_speed = 225
        return max(0.5, words / reading_speed)
    
    async def _calculate_complexity(self, content: str) -> float:
        """Calculate content complexity/readability score."""
        # Simple complexity metrics
        sentences = content.split('.')
        words = content.split()
        
        if not sentences or not words:
            return 0.5
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Syllable approximation (vowel groups)
        syllables = sum(max(1, len(re.findall(r'[aeiouAEIOU]+', word))) for word in words)
        avg_syllables_per_word = syllables / len(words) if words else 1
        
        # Flesch Reading Ease approximation
        # Higher score = easier to read
        if avg_sentence_length > 0:
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            # Convert to 0-1 scale (0 = complex, 1 = simple)
            complexity = max(0.0, min(1.0, flesch_score / 100))
        else:
            complexity = 0.5
        
        return 1.0 - complexity  # Invert so higher = more complex
    
    async def _generate_suggestions(self, content: str) -> List[str]:
        """Generate improvement suggestions for content."""
        prompt = f"""Analyze this note content and provide 2-3 specific, actionable suggestions for improvement.
Focus on structure, clarity, completeness, and usefulness.

Content:
{content[:1500]}...

Suggestions (one per line):"""
        
        try:
            request = ChatRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.7,
                max_tokens=300,
            )
            
            response = await self.language_service.chat_completion(request)
            
            suggestions = [
                s.strip().lstrip('- ').lstrip('* ') 
                for s in response.text.split('\n') 
                if s.strip()
            ]
            
            return suggestions[:3]  # Limit to 3 suggestions
            
        except Exception as e:
            logger.warning("Suggestion generation failed", error=str(e))
            return []
    
    async def _suggest_tags(self, content: str) -> List[str]:
        """Suggest tags for content."""
        prompt = f"""Suggest 3-5 relevant tags for this note content. 
Provide them as a comma-separated list of single words or short phrases.

Content:
{content[:1000]}...

Tags:"""
        
        try:
            request = ChatRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.5,
                max_tokens=100,
            )
            
            response = await self.language_service.chat_completion(request)
            
            tags = [tag.strip().lower() for tag in response.text.split(',')]
            tags = [tag for tag in tags if tag and len(tag) > 1]
            
            return tags[:5]
            
        except Exception as e:
            logger.warning("Tag suggestion failed", error=str(e))
            return []
    
    async def _analyze_structure(self, content: str) -> List[str]:
        """Analyze structural issues in content."""
        issues = []
        
        # Check for headers
        if not re.search(r'^#+\s', content, re.MULTILINE):
            issues.append("Consider adding headers to structure content")
        
        # Check for very long paragraphs
        paragraphs = content.split('\n\n')
        long_paragraphs = [p for p in paragraphs if len(p.split()) > 100]
        if long_paragraphs:
            issues.append("Some paragraphs are very long; consider breaking them up")
        
        # Check for no lists when appropriate
        if len(content.split()) > 50 and not re.search(r'^[-*+]\s|^\d+\.\s', content, re.MULTILINE):
            if any(word in content.lower() for word in ['step', 'first', 'second', 'then', 'next']):
                issues.append("Consider using lists for step-by-step content")
        
        # Check for missing links
        if 'http' not in content and len(content.split()) > 100:
            issues.append("Consider adding relevant links or references")
        
        return issues
    
    async def _find_similar_notes(self, note: Note) -> List[Tuple[str, float]]:
        """Find similar notes using embeddings."""
        try:
            # Generate embedding for current note
            embedding = await self.embedding_service.embed_text(note.content)
            
            # Search for similar notes in vector database
            if self.db_manager and self.db_manager.vector:
                similar = await self.db_manager.vector.search_similar(
                    embedding,
                    limit=10,
                    score_threshold=0.7
                )
                
                return [(item["note_id"], item["score"]) for item in similar]
            
            return []
            
        except Exception as e:
            logger.warning("Similar note search failed", error=str(e))
            return []
    
    async def _store_analysis(self, analysis: ContentAnalysis, note: Note) -> None:
        """Store analysis results in database."""
        try:
            metrics = {
                "note_id": analysis.note_id,
                "content_type": analysis.content_type.value,
                "quality_score": analysis.quality_score,
                "sentiment": analysis.sentiment,
                "reading_time_minutes": analysis.reading_time_minutes,
                "complexity_score": analysis.complexity_score,
                "key_topics": analysis.key_topics,
                "suggested_tags": analysis.tags_suggested,
                "similar_count": len(analysis.similar_notes),
                "file_path": str(note.path),
                "word_count": note.word_count,
                "link_count": len(note.links),
                "task_count": len(note.tasks),
            }
            
            await self.db_manager.analytics.insert_note_metrics(analysis.note_id, metrics)
            
        except Exception as e:
            logger.warning("Failed to store analysis", error=str(e))
    
    async def analyze_vault_content(
        self, 
        notes: List[Note],
        batch_size: int = 10
    ) -> List[ContentAnalysis]:
        """Analyze multiple notes efficiently."""
        logger.info("Starting vault content analysis", note_count=len(notes))
        
        analyses = []
        
        for i in range(0, len(notes), batch_size):
            batch = notes[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.analyze_note(note, include_suggestions=False, find_similar=False)
                for note in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.warning("Note analysis failed", error=str(result))
                else:
                    analyses.append(result)
            
            logger.debug("Processed batch", batch_num=i//batch_size + 1, total_batches=(len(notes) + batch_size - 1) // batch_size)
        
        logger.info("Vault analysis completed", successful_analyses=len(analyses))
        return analyses
    
    async def find_topic_clusters(
        self, 
        analyses: List[ContentAnalysis],
        min_cluster_size: int = 3
    ) -> List[TopicCluster]:
        """Find clusters of related topics across analyses."""
        # Collect all topics
        topic_notes = {}
        for analysis in analyses:
            for topic in analysis.key_topics:
                if topic not in topic_notes:
                    topic_notes[topic] = []
                topic_notes[topic].append(analysis.note_id)
        
        # Filter topics by frequency
        frequent_topics = {
            topic: notes for topic, notes in topic_notes.items()
            if len(notes) >= min_cluster_size
        }
        
        # Create clusters
        clusters = []
        for i, (main_topic, note_ids) in enumerate(frequent_topics.items()):
            # Find related topics (topics that appear with this one)
            related_topics = []
            for analysis in analyses:
                if analysis.note_id in note_ids and main_topic in analysis.key_topics:
                    for topic in analysis.key_topics:
                        if topic != main_topic and topic not in related_topics:
                            related_topics.append(topic)
            
            # Calculate coherence score
            coherence = len(note_ids) / len(analyses)  # Simple coherence metric
            
            cluster = TopicCluster(
                cluster_id=f"cluster_{i}",
                main_topic=main_topic,
                related_topics=related_topics[:5],  # Top 5 related
                note_ids=note_ids,
                coherence_score=coherence
            )
            
            clusters.append(cluster)
        
        # Sort by coherence
        clusters.sort(key=lambda c: c.coherence_score, reverse=True)
        
        return clusters
    
    async def get_content_quality_report(
        self, 
        analyses: List[ContentAnalysis]
    ) -> Dict[str, Any]:
        """Generate a comprehensive content quality report."""
        if not analyses:
            return {"error": "No analyses provided"}
        
        # Overall statistics
        total_notes = len(analyses)
        avg_quality = sum(a.quality_score for a in analyses) / total_notes
        avg_complexity = sum(a.complexity_score for a in analyses) / total_notes
        avg_reading_time = sum(a.reading_time_minutes for a in analyses) / total_notes
        
        # Quality distribution
        quality_ranges = {
            "excellent": len([a for a in analyses if a.quality_score >= 0.8]),
            "good": len([a for a in analyses if 0.6 <= a.quality_score < 0.8]),
            "fair": len([a for a in analyses if 0.4 <= a.quality_score < 0.6]),
            "poor": len([a for a in analyses if a.quality_score < 0.4]),
        }
        
        # Content type distribution
        content_types = {}
        for analysis in analyses:
            content_type = analysis.content_type.value
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        # Sentiment distribution
        sentiment_dist = {}
        for analysis in analyses:
            sentiment = analysis.sentiment
            sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1
        
        # Most common topics
        all_topics = []
        for analysis in analyses:
            all_topics.extend(analysis.key_topics)
        
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Notes needing attention
        low_quality_notes = [
            {"note_id": a.note_id, "quality_score": a.quality_score, "suggestions": a.suggestions}
            for a in analyses 
            if a.quality_score < 0.5
        ]
        
        return {
            "summary": {
                "total_notes": total_notes,
                "average_quality_score": round(avg_quality, 3),
                "average_complexity_score": round(avg_complexity, 3),
                "average_reading_time_minutes": round(avg_reading_time, 2),
            },
            "quality_distribution": quality_ranges,
            "content_types": content_types,
            "sentiment_distribution": sentiment_dist,
            "top_topics": top_topics,
            "notes_needing_attention": low_quality_notes[:10],  # Top 10
            "recommendations": self._generate_vault_recommendations(analyses),
        }
    
    def _generate_vault_recommendations(self, analyses: List[ContentAnalysis]) -> List[str]:
        """Generate recommendations for vault improvement."""
        recommendations = []
        
        # Quality recommendations
        low_quality_count = len([a for a in analyses if a.quality_score < 0.5])
        if low_quality_count > len(analyses) * 0.2:  # More than 20% low quality
            recommendations.append("Focus on improving note structure and clarity")
        
        # Content type recommendations
        content_types = [a.content_type for a in analyses]
        if content_types.count(ContentType.UNKNOWN) > len(analyses) * 0.3:
            recommendations.append("Consider using consistent templates for different note types")
        
        # Tagging recommendations
        untagged_count = len([a for a in analyses if not a.tags_suggested])
        if untagged_count > len(analyses) * 0.5:
            recommendations.append("Add more descriptive tags to improve discoverability")
        
        # Structure recommendations
        structure_issues = sum(len(a.structure_issues) for a in analyses)
        if structure_issues > len(analyses):  # More than 1 issue per note on average
            recommendations.append("Improve note structure with headers and lists")
        
        return recommendations
    
    async def clear_cache(self) -> None:
        """Clear analysis cache."""
        async with self._lock:
            self._analysis_cache.clear()