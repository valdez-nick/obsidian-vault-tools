"""
Directory Organization Core Service for Obsidian Librarian.

This module provides intelligent auto-organization capabilities including:
- Multi-modal content analysis (text, metadata, links)
- Smart routing algorithms with rule-based and AI-powered classification
- Real-time file monitoring and organization
- Pattern recognition and conflict resolution
- User feedback learning system
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

import aiofiles
import structlog
from pydantic import BaseModel, Field, ConfigDict
from sentence_transformers import SentenceTransformer

from ..ai import ContentAnalyzer, EmbeddingService, QueryProcessor
from ..models import Note, NoteMetadata, LibrarianConfig
from ..vault import Vault


logger = structlog.get_logger(__name__)


class ClassificationConfidence(Enum):
    """Classification confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CERTAIN = "certain"


class OrganizationAction(Enum):
    """Types of organization actions."""
    MOVE = "move"
    COPY = "copy"
    LINK = "link"
    TAG = "tag"
    IGNORE = "ignore"


@dataclass
class ContentFeatures:
    """Features extracted from content for classification."""
    # Text features
    word_count: int = 0
    unique_words: int = 0
    readability_score: float = 0.0
    
    # Structure features
    header_count: int = 0
    list_count: int = 0
    link_count: int = 0
    image_count: int = 0
    
    # Metadata features
    tags: Set[str] = field(default_factory=set)
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    
    # Semantic features
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Link analysis
    inbound_links: Set[str] = field(default_factory=set)
    outbound_links: Set[str] = field(default_factory=set)
    backlink_count: int = 0


@dataclass 
class ClassificationResult:
    """Result of content classification."""
    suggested_path: Path
    confidence: ClassificationConfidence
    reasoning: str
    action: OrganizationAction
    score: float
    alternatives: List[Tuple[Path, float]] = field(default_factory=list)
    features_used: List[str] = field(default_factory=list)


@dataclass
class OrganizationRule:
    """Rule for organizing content."""
    name: str
    conditions: Dict[str, Any]
    action: OrganizationAction
    target_pattern: str
    priority: int = 0
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0


@dataclass
class UserFeedback:
    """User feedback on organization decisions."""
    original_path: Path
    suggested_path: Path
    actual_path: Path
    accepted: bool
    timestamp: datetime
    feedback_type: str  # "correction", "approval", "rejection"
    user_notes: Optional[str] = None


class ContentClassifier:
    """Multi-modal content classifier for intelligent file organization."""
    
    def __init__(
        self,
        vault: Vault,
        content_analyzer: ContentAnalyzer,
        embedding_service: EmbeddingService,
        config: LibrarianConfig
    ):
        self.vault = vault
        self.content_analyzer = content_analyzer
        self.embedding_service = embedding_service
        self.config = config
        
        # Classification models and thresholds
        self.text_similarity_threshold = 0.75
        self.topic_confidence_threshold = 0.6
        self.structure_weight = 0.3
        self.content_weight = 0.5
        self.metadata_weight = 0.2
        
        # Cache for embeddings and features
        self._embedding_cache: Dict[str, Any] = {}
        self._feature_cache: Dict[str, ContentFeatures] = {}
        
        # Known patterns for classification
        self.content_patterns = {
            "meeting_notes": [
                r"meeting.*notes?",
                r"\d{4}-\d{2}-\d{2}.*meeting",
                r"agenda.*\d{4}",
                r"action.*items?"
            ],
            "daily_notes": [
                r"\d{4}-\d{2}-\d{2}(?:\.md)?$",
                r"daily.*note",
                r"journal.*\d{4}"
            ],
            "project_docs": [
                r"project.*\w+",
                r"prd.*\w+",
                r"requirements",
                r"specification"
            ],
            "research": [
                r"research.*\w+",
                r"literature.*review",
                r"survey.*\w+",
                r"analysis.*\w+"
            ],
            "templates": [
                r"template.*\w+",
                r"\w+.*template",
                r"boilerplate"
            ]
        }
        
    async def extract_features(self, note: Note) -> ContentFeatures:
        """Extract comprehensive features from a note."""
        cache_key = f"{note.path}:{note.metadata.modified_time}"
        
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
            
        features = ContentFeatures()
        
        # Basic text features
        content = note.content or ""
        words = content.split()
        features.word_count = len(words)
        features.unique_words = len(set(word.lower() for word in words))
        
        # Structure analysis
        features.header_count = len(re.findall(r'^#+\s', content, re.MULTILINE))
        features.list_count = len(re.findall(r'^[-*+]\s', content, re.MULTILINE))
        features.link_count = len(re.findall(r'\[\[.*?\]\]|\[.*?\]\(.*?\)', content))
        features.image_count = len(re.findall(r'!\[.*?\]\(.*?\)', content))
        
        # Metadata extraction
        if note.metadata.tags:
            features.tags = set(note.metadata.tags)
        if note.metadata.frontmatter:
            features.frontmatter = note.metadata.frontmatter
        features.creation_date = note.metadata.created_time
        features.modification_date = note.metadata.modified_time
        
        # Link analysis
        wiki_links = re.findall(r'\[\[(.*?)\]\]', content)
        md_links = re.findall(r'\[.*?\]\((.*?)\)', content)
        features.outbound_links = set(wiki_links + md_links)
        
        # AI-powered semantic analysis
        if content.strip():
            try:
                analysis = await self.content_analyzer.analyze_content(content)
                if hasattr(analysis, 'topics'):
                    features.topics = analysis.topics[:5]  # Top 5 topics
                if hasattr(analysis, 'entities'):
                    features.entities = analysis.entities[:10]  # Top 10 entities
                if hasattr(analysis, 'keywords'):
                    features.keywords = analysis.keywords[:10]  # Top 10 keywords
            except Exception as e:
                logger.warning("Failed to analyze content semantics", error=str(e))
        
        # Cache the features
        self._feature_cache[cache_key] = features
        return features
    
    async def classify_content(self, note: Note) -> ClassificationResult:
        """Classify content and suggest organization."""
        features = await self.extract_features(note)
        
        # Try pattern-based classification first
        pattern_result = await self._classify_by_patterns(note, features)
        if pattern_result.confidence in [ClassificationConfidence.HIGH, ClassificationConfidence.CERTAIN]:
            return pattern_result
            
        # Try semantic classification
        semantic_result = await self._classify_by_semantics(note, features)
        
        # Try structural classification
        structural_result = await self._classify_by_structure(note, features)
        
        # Combine results with weighted scoring
        combined_result = await self._combine_classifications(
            note, features, [pattern_result, semantic_result, structural_result]
        )
        
        return combined_result
    
    async def _classify_by_patterns(self, note: Note, features: ContentFeatures) -> ClassificationResult:
        """Classify based on filename and content patterns."""
        filename = note.path.name.lower()
        content = (note.content or "").lower()
        
        best_match = None
        best_score = 0.0
        best_category = None
        
        for category, patterns in self.content_patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, filename):
                    score += 0.8
                if re.search(pattern, content):
                    score += 0.4
            
            if score > best_score:
                best_score = score
                best_category = category
                best_match = pattern
        
        if best_category and best_score > 0.5:
            suggested_path = self._get_category_path(best_category, note)
            confidence = ClassificationConfidence.HIGH if best_score > 0.8 else ClassificationConfidence.MEDIUM
            
            return ClassificationResult(
                suggested_path=suggested_path,
                confidence=confidence,
                reasoning=f"Pattern match: {best_match} (score: {best_score:.2f})",
                action=OrganizationAction.MOVE,
                score=best_score,
                features_used=["filename_patterns", "content_patterns"]
            )
        
        return ClassificationResult(
            suggested_path=note.path,
            confidence=ClassificationConfidence.LOW,
            reasoning="No strong pattern matches found",
            action=OrganizationAction.IGNORE,
            score=0.0
        )
    
    async def _classify_by_semantics(self, note: Note, features: ContentFeatures) -> ClassificationResult:
        """Classify based on semantic content analysis."""
        if not features.topics:
            return ClassificationResult(
                suggested_path=note.path,
                confidence=ClassificationConfidence.LOW,
                reasoning="No semantic topics identified",
                action=OrganizationAction.IGNORE,
                score=0.0
            )
        
        # Get embeddings for the content
        content = note.content or ""
        if len(content.split()) < 10:  # Too short for meaningful semantic analysis
            return ClassificationResult(
                suggested_path=note.path,
                confidence=ClassificationConfidence.LOW,
                reasoning="Content too short for semantic analysis",
                action=OrganizationAction.IGNORE,
                score=0.0
            )
        
        try:
            # Find most similar existing notes
            similar_notes = await self._find_similar_notes(note)
            
            if similar_notes:
                # Use the directory of the most similar note
                most_similar = similar_notes[0]
                similarity_score = most_similar[1]
                similar_note_path = most_similar[0]
                
                if similarity_score > self.text_similarity_threshold:
                    suggested_dir = similar_note_path.parent
                    suggested_path = suggested_dir / note.path.name
                    
                    return ClassificationResult(
                        suggested_path=suggested_path,
                        confidence=ClassificationConfidence.HIGH if similarity_score > 0.85 else ClassificationConfidence.MEDIUM,
                        reasoning=f"Similar to {similar_note_path.name} (similarity: {similarity_score:.2f})",
                        action=OrganizationAction.MOVE,
                        score=similarity_score,
                        features_used=["content_embeddings", "semantic_similarity"]
                    )
            
            # Fallback to topic-based organization
            primary_topic = features.topics[0] if features.topics else "miscellaneous"
            suggested_path = Path("Knowledge Base") / primary_topic.title() / note.path.name
            
            return ClassificationResult(
                suggested_path=suggested_path,
                confidence=ClassificationConfidence.MEDIUM,
                reasoning=f"Organized by primary topic: {primary_topic}",
                action=OrganizationAction.MOVE,
                score=0.6,
                features_used=["topic_modeling"]
            )
            
        except Exception as e:
            logger.error("Semantic classification failed", error=str(e))
            return ClassificationResult(
                suggested_path=note.path,
                confidence=ClassificationConfidence.LOW,
                reasoning=f"Semantic analysis failed: {str(e)}",
                action=OrganizationAction.IGNORE,
                score=0.0
            )
    
    async def _classify_by_structure(self, note: Note, features: ContentFeatures) -> ClassificationResult:
        """Classify based on document structure and metadata."""
        score = 0.0
        reasoning_parts = []
        
        # Analyze document structure
        if features.header_count > 3:
            score += 0.2
            reasoning_parts.append("well-structured document")
        
        if features.list_count > 2:
            score += 0.1
            reasoning_parts.append("contains lists")
        
        if features.link_count > 5:
            score += 0.2
            reasoning_parts.append("highly linked")
        
        # Analyze metadata
        if features.tags:
            score += 0.3
            tag_based_path = self._get_tag_based_path(features.tags, note)
            reasoning_parts.append(f"tagged with {', '.join(list(features.tags)[:3])}")
        else:
            tag_based_path = note.path
        
        # Date-based organization for timestamped content
        if features.creation_date:
            date_path = self._get_date_based_path(features.creation_date, note)
            if self._is_daily_note_pattern(note.path.name):
                score += 0.4
                reasoning_parts.append("daily note pattern")
                tag_based_path = date_path
        
        if score > 0.4:
            return ClassificationResult(
                suggested_path=tag_based_path,
                confidence=ClassificationConfidence.MEDIUM,
                reasoning=f"Structural analysis: {', '.join(reasoning_parts)}",
                action=OrganizationAction.MOVE,
                score=score,
                features_used=["document_structure", "metadata", "tags"]
            )
        
        return ClassificationResult(
            suggested_path=note.path,
            confidence=ClassificationConfidence.LOW,
            reasoning="Insufficient structural indicators",
            action=OrganizationAction.IGNORE,
            score=score
        )
    
    async def _combine_classifications(
        self, 
        note: Note, 
        features: ContentFeatures, 
        results: List[ClassificationResult]
    ) -> ClassificationResult:
        """Combine multiple classification results using weighted scoring."""
        valid_results = [r for r in results if r.confidence != ClassificationConfidence.LOW]
        
        if not valid_results:
            return results[0]  # Return the first (lowest confidence) result
        
        # Weight the results
        weights = {
            "pattern": 0.4,
            "semantic": 0.4, 
            "structural": 0.2
        }
        
        best_result = None
        best_weighted_score = 0.0
        
        for i, result in enumerate(valid_results):
            weight_key = ["pattern", "semantic", "structural"][i] if i < 3 else "structural"
            weighted_score = result.score * weights.get(weight_key, 0.1)
            
            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                best_result = result
        
        if best_result:
            # Adjust confidence based on agreement between methods
            agreement_count = len([r for r in valid_results if r.suggested_path.parent == best_result.suggested_path.parent])
            if agreement_count >= 2:
                if best_result.confidence == ClassificationConfidence.MEDIUM:
                    best_result.confidence = ClassificationConfidence.HIGH
            
            best_result.score = best_weighted_score
            return best_result
        
        return results[0]
    
    async def _find_similar_notes(self, note: Note) -> List[Tuple[Path, float]]:
        """Find notes with similar content using embeddings."""
        try:
            content = note.content or ""
            if len(content.split()) < 10:
                return []
            
            # Get embedding for current note
            current_embedding = await self.embedding_service.embed_text(content)
            
            # Compare with existing notes (simplified - in real implementation, 
            # you'd use a vector database)
            similar_notes = []
            
            # This is a placeholder - in real implementation, you'd:
            # 1. Query vector database for similar embeddings
            # 2. Return top N most similar notes with scores
            
            return similar_notes
            
        except Exception as e:
            logger.error("Failed to find similar notes", error=str(e))
            return []
    
    def _get_category_path(self, category: str, note: Note) -> Path:
        """Get the suggested path for a content category."""
        category_mapping = {
            "meeting_notes": Path("Meetings"),
            "daily_notes": Path("Daily Notes"),
            "project_docs": Path("Projects"),
            "research": Path("Research"),
            "templates": Path("Templates")
        }
        
        base_path = category_mapping.get(category, Path("Miscellaneous"))
        return base_path / note.path.name
    
    def _get_tag_based_path(self, tags: Set[str], note: Note) -> Path:
        """Get suggested path based on tags."""
        if not tags:
            return note.path
        
        # Use the first tag as the primary category
        primary_tag = next(iter(tags))
        # Remove # if present
        clean_tag = primary_tag.lstrip('#')
        
        return Path("By Tag") / clean_tag.title() / note.path.name
    
    def _get_date_based_path(self, date: datetime, note: Note) -> Path:
        """Get date-based path organization."""
        year = date.strftime("%Y")
        month = date.strftime("%m-%B")
        
        return Path("Daily Notes") / year / month / note.path.name
    
    def _is_daily_note_pattern(self, filename: str) -> bool:
        """Check if filename matches daily note pattern."""
        return bool(re.match(r'\d{4}-\d{2}-\d{2}', filename))


class DirectoryRouter:
    """Smart routing system for determining optimal file organization."""
    
    def __init__(self, vault: Vault, config: LibrarianConfig):
        self.vault = vault
        self.config = config
        
        # Directory structure analysis
        self.directory_stats: Dict[Path, Dict[str, Any]] = {}
        self.routing_cache: Dict[str, Path] = {}
        
        # Smart routing parameters
        self.max_files_per_directory = 50
        self.min_similarity_for_grouping = 0.7
        self.balance_threshold = 0.8
    
    async def route_file(self, note: Note, classification: ClassificationResult) -> Path:
        """Determine the optimal directory for a file."""
        suggested_path = classification.suggested_path
        
        # Check directory capacity and balance
        optimal_path = await self._optimize_directory_placement(suggested_path)
        
        # Ensure no conflicts
        final_path = await self._resolve_naming_conflicts(optimal_path)
        
        return final_path
    
    async def _optimize_directory_placement(self, suggested_path: Path) -> Path:
        """Optimize directory placement based on capacity and organization."""
        directory = suggested_path.parent
        
        # Get directory statistics
        stats = await self._get_directory_stats(directory)
        
        # Check if directory is overcrowded
        if stats["file_count"] > self.max_files_per_directory:
            # Try to create a subdirectory or suggest alternative
            alternative_dir = await self._suggest_subdirectory(directory, suggested_path.name)
            if alternative_dir:
                return alternative_dir / suggested_path.name
        
        return suggested_path
    
    async def _get_directory_stats(self, directory: Path) -> Dict[str, Any]:
        """Get statistics for a directory."""
        cache_key = str(directory)
        
        if cache_key in self.directory_stats:
            stats = self.directory_stats[cache_key]
            # Check if cache is still valid (less than 5 minutes old)
            if datetime.now() - stats.get("last_updated", datetime.min) < timedelta(minutes=5):
                return stats
        
        # Calculate fresh statistics
        stats = {
            "file_count": 0,
            "total_size": 0,
            "avg_file_size": 0,
            "file_types": {},
            "last_updated": datetime.now()
        }
        
        try:
            vault_path = self.vault.path
            full_directory = vault_path / directory
            
            if full_directory.exists():
                files = list(full_directory.glob("*.md"))
                stats["file_count"] = len(files)
                
                if files:
                    total_size = sum(f.stat().st_size for f in files if f.is_file())
                    stats["total_size"] = total_size
                    stats["avg_file_size"] = total_size / len(files)
        
        except Exception as e:
            logger.warning("Failed to get directory stats", directory=str(directory), error=str(e))
        
        self.directory_stats[cache_key] = stats
        return stats
    
    async def _suggest_subdirectory(self, parent_dir: Path, filename: str) -> Optional[Path]:
        """Suggest a subdirectory to better organize files."""
        # Try date-based subdirectory for daily notes
        if re.match(r'\d{4}-\d{2}-\d{2}', filename):
            date_match = re.match(r'(\d{4})-(\d{2})-\d{2}', filename)
            if date_match:
                year, month = date_match.groups()
                return parent_dir / year / f"{month}-{datetime.strptime(month, '%m').strftime('%B')}"
        
        # Try topic-based subdirectory
        # This is a simplified version - in practice, you'd analyze content
        return parent_dir / "Recent"
    
    async def _resolve_naming_conflicts(self, target_path: Path) -> Path:
        """Resolve naming conflicts by suggesting alternative names."""
        vault_path = self.vault.path
        full_path = vault_path / target_path
        
        if not full_path.exists():
            return target_path
        
        # Generate alternative names
        base_name = target_path.stem
        extension = target_path.suffix
        parent = target_path.parent
        
        counter = 1
        while True:
            alternative_name = f"{base_name}_{counter:02d}{extension}"
            alternative_path = parent / alternative_name
            full_alternative = vault_path / alternative_path
            
            if not full_alternative.exists():
                return alternative_path
            
            counter += 1
            if counter > 99:  # Prevent infinite loop
                # Use timestamp as fallback
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                return parent / f"{base_name}_{timestamp}{extension}"


class OrganizationLearner:
    """Learning system that adapts organization based on user feedback."""
    
    def __init__(self, config: LibrarianConfig):
        self.config = config
        
        # Feedback storage
        self.feedback_history: List[UserFeedback] = []
        self.learned_patterns: Dict[str, OrganizationRule] = {}
        
        # Learning parameters
        self.min_feedback_for_pattern = 3
        self.feedback_weight = 0.7
        self.pattern_confidence_threshold = 0.8
    
    async def record_feedback(self, feedback: UserFeedback) -> None:
        """Record user feedback on organization decisions."""
        self.feedback_history.append(feedback)
        
        # Update learned patterns
        await self._update_learned_patterns(feedback)
        
        # Persist feedback
        await self._persist_feedback(feedback)
    
    async def get_learned_suggestions(self, note: Note, features: ContentFeatures) -> Optional[ClassificationResult]:
        """Get suggestions based on learned patterns."""
        for pattern_name, rule in self.learned_patterns.items():
            if await self._matches_rule(note, features, rule):
                suggested_path = self._apply_rule_pattern(rule, note)
                
                return ClassificationResult(
                    suggested_path=suggested_path,
                    confidence=ClassificationConfidence.HIGH if rule.success_rate > 0.8 else ClassificationConfidence.MEDIUM,
                    reasoning=f"Learned pattern: {pattern_name} (success rate: {rule.success_rate:.2f})",
                    action=rule.action,
                    score=rule.success_rate,
                    features_used=["learned_patterns", "user_feedback"]
                )
        
        return None
    
    async def _update_learned_patterns(self, feedback: UserFeedback) -> None:
        """Update learned patterns based on feedback."""
        if feedback.accepted:
            # Positive feedback - strengthen or create pattern
            pattern_key = self._generate_pattern_key(feedback)
            
            if pattern_key in self.learned_patterns:
                rule = self.learned_patterns[pattern_key]
                rule.usage_count += 1
                # Update success rate (weighted average)
                rule.success_rate = (rule.success_rate * (rule.usage_count - 1) + 1.0) / rule.usage_count
            else:
                # Create new pattern
                rule = OrganizationRule(
                    name=f"learned_pattern_{len(self.learned_patterns)}",
                    conditions=self._extract_conditions_from_feedback(feedback),
                    action=OrganizationAction.MOVE,
                    target_pattern=str(feedback.actual_path.parent),
                    usage_count=1,
                    success_rate=1.0
                )
                self.learned_patterns[pattern_key] = rule
        else:
            # Negative feedback - weaken pattern
            pattern_key = self._generate_pattern_key(feedback)
            
            if pattern_key in self.learned_patterns:
                rule = self.learned_patterns[pattern_key]
                rule.usage_count += 1
                # Update success rate (add failure)
                rule.success_rate = (rule.success_rate * (rule.usage_count - 1) + 0.0) / rule.usage_count
                
                # Remove pattern if success rate drops too low
                if rule.success_rate < 0.3:
                    del self.learned_patterns[pattern_key]
    
    def _generate_pattern_key(self, feedback: UserFeedback) -> str:
        """Generate a key for pattern matching based on feedback."""
        # This is simplified - in practice, you'd analyze the file characteristics
        # that led to the original suggestion
        return f"{feedback.original_path.parent}_{feedback.actual_path.parent}"
    
    def _extract_conditions_from_feedback(self, feedback: UserFeedback) -> Dict[str, Any]:
        """Extract conditions that can be used for pattern matching."""
        return {
            "source_directory": str(feedback.original_path.parent),
            "target_directory": str(feedback.actual_path.parent),
            "filename_pattern": feedback.original_path.name,
            "feedback_type": feedback.feedback_type
        }
    
    async def _matches_rule(self, note: Note, features: ContentFeatures, rule: OrganizationRule) -> bool:
        """Check if a note matches a learned rule."""
        conditions = rule.conditions
        
        # Check source directory
        if "source_directory" in conditions:
            if str(note.path.parent) != conditions["source_directory"]:
                return False
        
        # Check filename pattern
        if "filename_pattern" in conditions:
            pattern = conditions["filename_pattern"]
            if not re.search(pattern, note.path.name):
                return False
        
        return True
    
    def _apply_rule_pattern(self, rule: OrganizationRule, note: Note) -> Path:
        """Apply a learned rule to suggest a path."""
        target_pattern = rule.target_pattern
        
        # Simple pattern substitution
        if "{filename}" in target_pattern:
            target_pattern = target_pattern.replace("{filename}", note.path.name)
        
        return Path(target_pattern) / note.path.name
    
    async def _persist_feedback(self, feedback: UserFeedback) -> None:
        """Persist feedback to storage."""
        # In a real implementation, you'd save to database or file
        pass


class RuleEngine:
    """Rule-based organization engine with conflict resolution."""
    
    def __init__(self, config: LibrarianConfig):
        self.config = config
        
        # Built-in rules
        self.built_in_rules: List[OrganizationRule] = []
        self.custom_rules: List[OrganizationRule] = []
        
        # Rule evaluation cache
        self.rule_cache: Dict[str, List[OrganizationRule]] = {}
        
        self._initialize_built_in_rules()
    
    def _initialize_built_in_rules(self) -> None:
        """Initialize built-in organization rules."""
        self.built_in_rules = [
            OrganizationRule(
                name="daily_notes_by_date",
                conditions={"filename_pattern": r"\d{4}-\d{2}-\d{2}"},
                action=OrganizationAction.MOVE,
                target_pattern="Daily Notes/{year}/{month}",
                priority=10
            ),
            OrganizationRule(
                name="meeting_notes_by_meeting",
                conditions={"content_pattern": r"meeting|agenda|minutes"},
                action=OrganizationAction.MOVE,
                target_pattern="Meetings/{year}",
                priority=8
            ),
            OrganizationRule(
                name="templates_to_templates",
                conditions={"filename_pattern": r"template|boilerplate"},
                action=OrganizationAction.MOVE,
                target_pattern="Templates",
                priority=9
            ),
            OrganizationRule(
                name="tagged_content_by_tag",
                conditions={"has_tags": True},
                action=OrganizationAction.MOVE,
                target_pattern="By Tag/{primary_tag}",
                priority=5
            )
        ]
    
    async def evaluate_rules(self, note: Note, features: ContentFeatures) -> List[ClassificationResult]:
        """Evaluate all applicable rules for a note."""
        results = []
        
        # Combine built-in and custom rules
        all_rules = self.built_in_rules + self.custom_rules
        
        # Sort by priority (higher first)
        all_rules.sort(key=lambda r: r.priority, reverse=True)
        
        for rule in all_rules:
            if not rule.enabled:
                continue
                
            if await self._rule_matches(note, features, rule):
                result = await self._apply_rule(note, features, rule)
                if result:
                    results.append(result)
        
        return results
    
    async def _rule_matches(self, note: Note, features: ContentFeatures, rule: OrganizationRule) -> bool:
        """Check if a rule matches the given note."""
        conditions = rule.conditions
        
        # Check filename pattern
        if "filename_pattern" in conditions:
            pattern = conditions["filename_pattern"]
            if not re.search(pattern, note.path.name, re.IGNORECASE):
                return False
        
        # Check content pattern
        if "content_pattern" in conditions:
            pattern = conditions["content_pattern"]
            content = note.content or ""
            if not re.search(pattern, content, re.IGNORECASE):
                return False
        
        # Check tags
        if "has_tags" in conditions and conditions["has_tags"]:
            if not features.tags:
                return False
        
        # Check specific tags
        if "required_tags" in conditions:
            required_tags = set(conditions["required_tags"])
            if not required_tags.issubset(features.tags):
                return False
        
        return True
    
    async def _apply_rule(self, note: Note, features: ContentFeatures, rule: OrganizationRule) -> Optional[ClassificationResult]:
        """Apply a rule to generate a classification result."""
        try:
            target_pattern = rule.target_pattern
            
            # Replace placeholders
            replacements = {
                "{filename}": note.path.stem,
                "{year}": str(features.creation_date.year) if features.creation_date else "Unknown",
                "{month}": f"{features.creation_date.month:02d}-{features.creation_date.strftime('%B')}" if features.creation_date else "Unknown",
                "{primary_tag}": list(features.tags)[0].lstrip('#') if features.tags else "Untagged"
            }
            
            for placeholder, value in replacements.items():
                target_pattern = target_pattern.replace(placeholder, value)
            
            suggested_path = Path(target_pattern) / note.path.name
            
            return ClassificationResult(
                suggested_path=suggested_path,
                confidence=ClassificationConfidence.HIGH,
                reasoning=f"Rule: {rule.name}",
                action=rule.action,
                score=0.9,
                features_used=["rule_engine"]
            )
            
        except Exception as e:
            logger.error("Failed to apply rule", rule=rule.name, error=str(e))
            return None
    
    def add_custom_rule(self, rule: OrganizationRule) -> None:
        """Add a custom organization rule."""
        self.custom_rules.append(rule)
    
    def remove_custom_rule(self, rule_name: str) -> bool:
        """Remove a custom rule by name."""
        for i, rule in enumerate(self.custom_rules):
            if rule.name == rule_name:
                del self.custom_rules[i]
                return True
        return False


class FileWatcher:
    """Real-time file monitoring and organization system."""
    
    def __init__(
        self,
        vault: Vault,
        classifier: ContentClassifier,
        router: DirectoryRouter,
        learner: OrganizationLearner,
        rule_engine: RuleEngine,
        config: LibrarianConfig
    ):
        self.vault = vault
        self.classifier = classifier
        self.router = router
        self.learner = learner
        self.rule_engine = rule_engine
        self.config = config
        
        # Monitoring state
        self.is_watching = False
        self.watch_tasks: Set[asyncio.Task] = set()
        
        # Batch processing
        self.pending_files: Dict[Path, datetime] = {}
        self.batch_delay = 2.0  # seconds
        self.batch_task: Optional[asyncio.Task] = None
        
        # Safety mechanisms
        self.dry_run_mode = True
        self.auto_organize_enabled = False
        self.require_user_confirmation = True
    
    async def start_watching(self) -> None:
        """Start monitoring the vault for changes."""
        if self.is_watching:
            return
        
        self.is_watching = True
        logger.info("Starting file watcher", vault_path=str(self.vault.path))
        
        # Start the main watch task
        watch_task = asyncio.create_task(self._watch_vault())
        self.watch_tasks.add(watch_task)
        
        # Start batch processing task
        self.batch_task = asyncio.create_task(self._process_batch())
    
    async def stop_watching(self) -> None:
        """Stop monitoring the vault."""
        if not self.is_watching:
            return
        
        self.is_watching = False
        logger.info("Stopping file watcher")
        
        # Cancel all watch tasks
        for task in self.watch_tasks:
            task.cancel()
        
        if self.batch_task:
            self.batch_task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.watch_tasks, self.batch_task, return_exceptions=True)
        
        self.watch_tasks.clear()
        self.batch_task = None
    
    async def _watch_vault(self) -> None:
        """Main file watching loop."""
        # This is a simplified implementation
        # In practice, you'd use a file system watcher like watchdog
        
        vault_path = self.vault.path
        last_check = datetime.now()
        
        while self.is_watching:
            try:
                # Check for new or modified files
                current_time = datetime.now()
                
                for md_file in vault_path.rglob("*.md"):
                    try:
                        stat = md_file.stat()
                        modified_time = datetime.fromtimestamp(stat.st_mtime)
                        
                        if modified_time > last_check:
                            await self._queue_file_for_processing(md_file)
                    
                    except Exception as e:
                        logger.warning("Failed to check file", file=str(md_file), error=str(e))
                
                last_check = current_time
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error("Error in file watcher", error=str(e))
                await asyncio.sleep(10)  # Back off on error
    
    async def _queue_file_for_processing(self, file_path: Path) -> None:
        """Queue a file for batch processing."""
        relative_path = file_path.relative_to(self.vault.path)
        self.pending_files[relative_path] = datetime.now()
        
        logger.debug("Queued file for processing", file=str(relative_path))
    
    async def _process_batch(self) -> None:
        """Process queued files in batches."""
        while self.is_watching:
            try:
                await asyncio.sleep(self.batch_delay)
                
                if not self.pending_files:
                    continue
                
                # Get files that have been waiting long enough
                now = datetime.now()
                ready_files = [
                    path for path, queued_time in self.pending_files.items()
                    if (now - queued_time).total_seconds() >= self.batch_delay
                ]
                
                if ready_files:
                    await self._process_files_batch(ready_files)
                    
                    # Remove processed files from queue
                    for path in ready_files:
                        self.pending_files.pop(path, None)
                
            except Exception as e:
                logger.error("Error in batch processing", error=str(e))
                await asyncio.sleep(10)
    
    async def _process_files_batch(self, file_paths: List[Path]) -> None:
        """Process a batch of files for organization."""
        logger.info("Processing file batch", count=len(file_paths))
        
        for file_path in file_paths:
            try:
                await self._process_single_file(file_path)
            except Exception as e:
                logger.error("Failed to process file", file=str(file_path), error=str(e))
    
    async def _load_note_by_path(self, file_path: Path) -> Optional[Note]:
        """Load a note by its file path."""
        full_path = self.vault.path / file_path
        if not full_path.exists():
            return None
            
        try:
            # Read the file content
            async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Extract frontmatter and content
            import frontmatter
            post = frontmatter.loads(content)
            
            # Get file stats
            stat = full_path.stat()
            created_time = datetime.fromtimestamp(stat.st_ctime)
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            
            # Extract tags from frontmatter and content
            tags = []
            if 'tags' in post.metadata:
                if isinstance(post.metadata['tags'], list):
                    tags.extend(post.metadata['tags'])
                else:
                    tags.append(str(post.metadata['tags']))
            
            # Extract inline tags
            import re
            inline_tags = re.findall(r'#[\w-]+', post.content)
            tags.extend(inline_tags)
            
            # Create note metadata
            metadata = NoteMetadata(
                created_time=created_time,
                modified_time=modified_time,
                tags=list(set(tags)),
                frontmatter=post.metadata
            )
            
            # Create and return note
            return Note(
                path=file_path,
                content=post.content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error("Failed to load note by path", path=str(file_path), error=str(e))
            return None
    
    async def _process_single_file(self, file_path: Path) -> None:
        """Process a single file for organization."""
        # Load the note
        full_path = self.vault.path / file_path
        if not full_path.exists():
            return
        
        try:
            note = await self._load_note_by_path(file_path)
            if not note:
                return
            
            # Extract features
            features = await self.classifier.extract_features(note)
            
            # Check learned patterns first
            learned_result = await self.learner.get_learned_suggestions(note, features)
            if learned_result and learned_result.confidence in [ClassificationConfidence.HIGH, ClassificationConfidence.CERTAIN]:
                await self._apply_organization(note, learned_result)
                return
            
            # Try rule engine
            rule_results = await self.rule_engine.evaluate_rules(note, features)
            if rule_results:
                best_rule_result = max(rule_results, key=lambda r: r.score)
                if best_rule_result.confidence in [ClassificationConfidence.HIGH, ClassificationConfidence.CERTAIN]:
                    await self._apply_organization(note, best_rule_result)
                    return
            
            # Fall back to classifier
            classification = await self.classifier.classify_content(note)
            if classification.confidence != ClassificationConfidence.LOW:
                # Route through directory optimizer
                optimal_path = await self.router.route_file(note, classification)
                classification.suggested_path = optimal_path
                
                await self._apply_organization(note, classification)
        
        except Exception as e:
            logger.error("Failed to process file for organization", file=str(file_path), error=str(e))
    
    async def _apply_organization(self, note: Note, result: ClassificationResult) -> None:
        """Apply the organization decision."""
        if result.action == OrganizationAction.IGNORE:
            return
        
        if self.dry_run_mode:
            logger.info(
                "DRY RUN: Would organize file",
                source=str(note.path),
                destination=str(result.suggested_path),
                confidence=result.confidence.value,
                reasoning=result.reasoning
            )
            return
        
        if not self.auto_organize_enabled and self.require_user_confirmation:
            logger.info(
                "Organization suggestion requires confirmation",
                source=str(note.path),
                destination=str(result.suggested_path),
                confidence=result.confidence.value,
                reasoning=result.reasoning
            )
            return
        
        # Apply the organization
        try:
            if result.action == OrganizationAction.MOVE:
                await self._move_file(note.path, result.suggested_path)
            elif result.action == OrganizationAction.COPY:
                await self._copy_file(note.path, result.suggested_path)
            elif result.action == OrganizationAction.TAG:
                await self._add_tags(note, result)
            
            logger.info(
                "File organized successfully",
                source=str(note.path),
                destination=str(result.suggested_path),
                action=result.action.value,
                reasoning=result.reasoning
            )
            
        except Exception as e:
            logger.error(
                "Failed to organize file",
                source=str(note.path),
                destination=str(result.suggested_path),
                error=str(e)
            )
    
    async def _move_file(self, source_path: Path, target_path: Path) -> None:
        """Safely move a file to a new location."""
        vault_path = self.vault.path
        source_full = vault_path / source_path
        target_full = vault_path / target_path
        
        # Ensure target directory exists
        target_full.parent.mkdir(parents=True, exist_ok=True)
        
        # Move the file
        source_full.rename(target_full)
    
    async def _copy_file(self, source_path: Path, target_path: Path) -> None:
        """Copy a file to a new location."""
        vault_path = self.vault.path
        source_full = vault_path / source_path
        target_full = vault_path / target_path
        
        # Ensure target directory exists
        target_full.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        async with aiofiles.open(source_full, 'r', encoding='utf-8') as src:
            content = await src.read()
        
        async with aiofiles.open(target_full, 'w', encoding='utf-8') as dst:
            await dst.write(content)
    
    async def _add_tags(self, note: Note, result: ClassificationResult) -> None:
        """Add organizational tags to a note."""
        # This would modify the note's frontmatter or content to add tags
        # Implementation depends on the note format and tagging system
        pass
    
    def enable_auto_organization(self, enabled: bool = True) -> None:
        """Enable or disable automatic organization."""
        self.auto_organize_enabled = enabled
        logger.info("Auto-organization", status="enabled" if enabled else "disabled")
    
    def set_dry_run_mode(self, dry_run: bool = True) -> None:
        """Enable or disable dry run mode."""
        self.dry_run_mode = dry_run
        logger.info("Dry run mode", status="enabled" if dry_run else "disabled")


class AutoOrganizer:
    """Main auto-organization service that coordinates all components."""
    
    def __init__(
        self,
        vault: Vault,
        content_analyzer: ContentAnalyzer,
        embedding_service: EmbeddingService,
        query_processor: QueryProcessor,
        config: LibrarianConfig
    ):
        self.vault = vault
        self.config = config
        
        # Initialize all components
        self.classifier = ContentClassifier(
            vault, content_analyzer, embedding_service, config
        )
        self.router = DirectoryRouter(vault_session, config)
        self.learner = OrganizationLearner(config)
        self.rule_engine = RuleEngine(config)
        self.file_watcher = FileWatcher(
            vault, self.classifier, self.router, 
            self.learner, self.rule_engine, config
        )
        
        # Service state
        self.is_running = False
    
    async def start(self) -> None:
        """Start the auto-organization service."""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Starting auto-organization service")
        
        # Start file watcher
        await self.file_watcher.start_watching()
    
    async def stop(self) -> None:
        """Stop the auto-organization service."""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping auto-organization service")
        
        # Stop file watcher
        await self.file_watcher.stop_watching()
    
    async def _load_note_by_path(self, file_path: Path) -> Optional[Note]:
        """Load a note by its file path."""
        full_path = self.vault.path / file_path
        if not full_path.exists():
            return None
            
        try:
            # Read the file content
            async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Extract frontmatter and content
            import frontmatter
            post = frontmatter.loads(content)
            
            # Get file stats
            stat = full_path.stat()
            created_time = datetime.fromtimestamp(stat.st_ctime)
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            
            # Extract tags from frontmatter and content
            tags = []
            if 'tags' in post.metadata:
                if isinstance(post.metadata['tags'], list):
                    tags.extend(post.metadata['tags'])
                else:
                    tags.append(str(post.metadata['tags']))
            
            # Extract inline tags
            import re
            inline_tags = re.findall(r'#[\w-]+', post.content)
            tags.extend(inline_tags)
            
            # Create note metadata
            metadata = NoteMetadata(
                created_time=created_time,
                modified_time=modified_time,
                tags=list(set(tags)),
                frontmatter=post.metadata
            )
            
            # Create and return note
            return Note(
                path=file_path,
                content=post.content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error("Failed to load note by path", path=str(file_path), error=str(e))
            return None

    async def organize_file(self, file_path: Path) -> ClassificationResult:
        """Manually organize a single file."""
        note = await self._load_note_by_path(file_path)
        if not note:
            raise ValueError(f"Could not load note: {file_path}")
        
        # Get classification
        classification = await self.classifier.classify_content(note)
        
        # Route to optimal directory
        optimal_path = await self.router.route_file(note, classification)
        classification.suggested_path = optimal_path
        
        return classification
    
    async def organize_vault(self, dry_run: bool = True) -> Dict[str, Any]:
        """Organize the entire vault."""
        logger.info("Starting vault organization", dry_run=dry_run)
        
        results = {
            "processed": 0,
            "organized": 0,
            "errors": 0,
            "suggestions": []
        }
        
        # Get all markdown files
        vault_path = self.vault.path
        md_files = list(vault_path.rglob("*.md"))
        
        for md_file in md_files:
            try:
                relative_path = md_file.relative_to(vault_path)
                classification = await self.organize_file(relative_path)
                
                results["processed"] += 1
                
                if classification.action != OrganizationAction.IGNORE:
                    results["suggestions"].append({
                        "source": str(relative_path),
                        "destination": str(classification.suggested_path),
                        "confidence": classification.confidence.value,
                        "reasoning": classification.reasoning,
                        "action": classification.action.value
                    })
                    
                    if not dry_run and classification.confidence in [ClassificationConfidence.HIGH, ClassificationConfidence.CERTAIN]:
                        # Actually move the file
                        await self.file_watcher._apply_organization(
                            await self._load_note_by_path(relative_path),
                            classification
                        )
                        results["organized"] += 1
                
            except Exception as e:
                logger.error("Failed to organize file", file=str(md_file), error=str(e))
                results["errors"] += 1
        
        logger.info("Vault organization complete", **results)
        return results
    
    async def add_feedback(self, feedback: UserFeedback) -> None:
        """Add user feedback to improve organization."""
        await self.learner.record_feedback(feedback)
    
    def add_custom_rule(self, rule: OrganizationRule) -> None:
        """Add a custom organization rule."""
        self.rule_engine.add_custom_rule(rule)
    
    def configure_auto_organization(
        self,
        enabled: bool = True,
        dry_run: bool = False,
        require_confirmation: bool = True
    ) -> None:
        """Configure auto-organization settings."""
        self.file_watcher.enable_auto_organization(enabled)
        self.file_watcher.set_dry_run_mode(dry_run)
        self.file_watcher.require_user_confirmation = require_confirmation