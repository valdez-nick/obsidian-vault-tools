"""
Data models for Obsidian Librarian.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field


# Core Note Models

@dataclass
class WikiLink:
    """Represents a wiki-style link in Obsidian."""
    target: str
    alias: Optional[str] = None
    position: int = 0
    
    def __str__(self):
        if self.alias:
            return f"[[{self.target}|{self.alias}]]"
        return f"[[{self.target}]]"


@dataclass
class Task:
    """Represents a task/checkbox in markdown."""
    description: str
    completed: bool
    position: int
    tags: List[str] = field(default_factory=list)
    due_date: Optional[datetime] = None
    
    def __str__(self):
        checkbox = "[x]" if self.completed else "[ ]"
        return f"- {checkbox} {self.description}"


@dataclass
class NoteMetadata:
    """Metadata for a note, typically from frontmatter."""
    title: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Note:
    """Represents an Obsidian note."""
    id: str
    path: Path
    content: str
    metadata: NoteMetadata
    links: List[WikiLink] = field(default_factory=list)
    backlinks: List[str] = field(default_factory=list)
    tasks: List[Task] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)
    
    # File metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0
    
    # Content analysis
    word_count: int = 0
    reading_time_minutes: float = 0.0
    
    @property
    def title(self) -> str:
        """Get the note title from metadata or filename."""
        if self.metadata.title:
            return self.metadata.title
        return self.path.stem
    
    @property
    def tags(self) -> List[str]:
        """Get all tags from metadata and content."""
        return self.metadata.tags
    
    @property
    def frontmatter(self) -> Dict[str, Any]:
        """Get frontmatter as a dictionary."""
        result = {}
        if self.metadata.title:
            result['title'] = self.metadata.title
        if self.metadata.tags:
            result['tags'] = self.metadata.tags
        if self.metadata.aliases:
            result['aliases'] = self.metadata.aliases
        result.update(self.metadata.custom_fields)
        return result
    
    @classmethod
    def from_rust_note(cls, rust_note: Any) -> 'Note':
        """Create a Note from Rust binding object."""
        # This would convert from the Rust note type
        # For now, return a placeholder implementation
        metadata = NoteMetadata(
            title=getattr(rust_note, 'title', None),
            tags=getattr(rust_note, 'tags', []),
            created=getattr(rust_note, 'created', None),
            modified=getattr(rust_note, 'modified', None),
        )
        
        return cls(
            id=rust_note.id,
            path=Path(rust_note.path),
            content=rust_note.content,
            metadata=metadata,
            word_count=getattr(rust_note, 'word_count', 0),
            size_bytes=getattr(rust_note, 'file_size', 0),
        )


# Vault Models

@dataclass
class VaultStats:
    """Statistics about a vault."""
    note_count: int = 0
    total_words: int = 0
    total_links: int = 0
    total_backlinks: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    total_tags: int = 0
    unique_tags: int = 0
    orphaned_notes: int = 0
    
    # File stats
    total_size_bytes: int = 0
    largest_note_size: int = 0
    average_note_size: float = 0.0
    
    # Dates
    oldest_note: Optional[datetime] = None
    newest_note: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    
    @classmethod
    def from_rust_stats(cls, rust_stats: Any) -> 'VaultStats':
        """Create VaultStats from Rust binding object."""
        return cls(
            note_count=rust_stats.note_count,
            total_words=rust_stats.total_words,
            total_links=rust_stats.total_links,
            total_backlinks=rust_stats.total_backlinks,
            total_tasks=rust_stats.total_tasks,
            completed_tasks=rust_stats.completed_tasks,
            total_tags=rust_stats.total_tags,
            unique_tags=rust_stats.unique_tags,
            orphaned_notes=rust_stats.orphaned_notes,
            total_size_bytes=rust_stats.total_size_bytes,
        )


@dataclass
class VaultConfig:
    """Configuration for vault operations."""
    # File handling
    include_patterns: List[str] = field(default_factory=lambda: ["*.md"])
    exclude_patterns: List[str] = field(default_factory=lambda: [".obsidian/", ".git/"])
    follow_symlinks: bool = False
    
    # Performance
    cache_size: int = 1000
    batch_size: int = 100
    max_file_size_mb: int = 10
    
    # Features
    enable_file_watching: bool = True
    enable_auto_backup: bool = True
    enable_link_resolution: bool = True
    
    # Parsing
    parse_frontmatter: bool = True
    parse_tasks: bool = True
    parse_tags: bool = True
    
    def to_rust_config(self) -> Dict[str, Any]:
        """Convert to format expected by Rust bindings."""
        return {
            'include_patterns': self.include_patterns,
            'exclude_patterns': self.exclude_patterns,
            'follow_symlinks': self.follow_symlinks,
            'cache_size': self.cache_size,
            'batch_size': self.batch_size,
            'max_file_size': self.max_file_size_mb * 1024 * 1024,
            'enable_watcher': self.enable_file_watching,
            'parse_frontmatter': self.parse_frontmatter,
            'parse_tasks': self.parse_tasks,
            'parse_tags': self.parse_tags,
        }


# Research Models

@dataclass
class ResearchQuery:
    """A processed research query."""
    text: str
    session_id: str
    query_type: str = "general"
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResearchResult:
    """Result from research operations."""
    url: str
    title: str
    summary: Optional[str] = None
    content: Optional[str] = None
    source: str = "unknown"
    quality_score: float = 0.0
    published_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Analysis Models

@dataclass
class ContentSimilarity:
    """Similarity between two notes."""
    note_a: str
    note_b: str
    similarity_score: float
    similarity_type: str = "content"  # content, structure, semantic


@dataclass
class DuplicateCluster:
    """A cluster of duplicate or near-duplicate notes."""
    cluster_id: str
    note_ids: List[str]
    similarities: List[ContentSimilarity]
    cluster_type: str  # exact_duplicate, near_duplicate, similar
    confidence_score: float


class AnalysisResult:
    """Result from content analysis."""
    pass  # Defined in analysis.py


# Template Models

@dataclass
class TemplateMatch:
    """A template that matches a note."""
    template_name: str
    confidence_score: float
    match_reasons: List[str]
    template_type: str = "custom"


@dataclass
class TemplateRule:
    """Rule for automatic template application."""
    name: str
    template_name: str
    trigger_type: str
    conditions: List[Dict[str, Any]]
    priority: int = 0


@dataclass
class TemplateApplication:
    """Result of applying a template."""
    note_id: str
    template_name: str
    success: bool
    error: Optional[str] = None
    rendered_content: Optional[str] = None
    applied_at: datetime = field(default_factory=datetime.utcnow)


# Configuration Models

@dataclass
class LibrarianConfig:
    """Global configuration for Obsidian Librarian."""
    # Vault settings
    vault_cache_size: int = 1000
    enable_file_watching: bool = True
    
    # Research settings
    max_concurrent_requests: int = 10
    enable_content_extraction: bool = True
    research_timeout_seconds: int = 30
    
    # Analysis settings
    enable_quality_scoring: bool = True
    analysis_batch_size: int = 100
    duplicate_threshold: float = 0.85
    
    # Template settings
    auto_apply_templates: bool = True
    template_confidence_threshold: float = 0.7
    
    # AI settings
    enable_ai_features: bool = True
    ai_model: str = "gpt-3.5-turbo"
    ai_temperature: float = 0.7
    
    # Performance settings
    max_workers: int = 4
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600


@dataclass 
class LibrarianStats:
    """Global statistics for the librarian system."""
    total_sessions: int = 0
    total_research_queries: int = 0
    total_results_found: int = 0
    total_notes_analyzed: int = 0
    total_duplicates_found: int = 0
    total_templates_applied: int = 0
    total_errors: int = 0
    
    # Performance metrics
    average_query_time_ms: float = 0.0
    average_analysis_time_ms: float = 0.0
    
    # Timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)


# Tag Management Models

@dataclass
class TagInfo:
    """Information about a specific tag."""
    name: str
    normalized_name: str
    usage_count: int
    first_seen: datetime
    last_used: datetime
    notes: Set[str] = field(default_factory=set)
    
    # Hierarchy information
    parent_tags: List[str] = field(default_factory=list)
    child_tags: List[str] = field(default_factory=list)
    hierarchy_level: int = 0
    
    # Analysis results
    similar_tags: List[Tuple[str, float]] = field(default_factory=list)
    semantic_clusters: List[str] = field(default_factory=list)
    
    @property
    def is_hierarchical(self) -> bool:
        """Check if tag is part of a hierarchy."""
        return bool(self.parent_tags or self.child_tags)
    
    @property
    def hierarchy_path(self) -> str:
        """Get full hierarchy path for the tag."""
        if not self.parent_tags:
            return self.name
        return f"{'/'.join(self.parent_tags)}/{self.name}"


@dataclass
class TagSuggestion:
    """A suggested tag for a note."""
    tag: str
    confidence: float
    reason: str
    source: str  # "semantic", "content", "pattern", "ai"
    context: Optional[str] = None


@dataclass
class TagSimilarity:
    """Similarity between two tags."""
    tag_a: str
    tag_b: str
    similarity_score: float
    similarity_type: str  # "fuzzy", "semantic", "exact"
    confidence: float = 1.0


@dataclass
class TagCluster:
    """A cluster of related tags."""
    cluster_id: str
    tags: List[str]
    cluster_type: str  # "similar", "semantic", "hierarchical"
    confidence: float
    representative_tag: Optional[str] = None
    suggested_merge: bool = False


@dataclass
class TagHierarchy:
    """Represents a tag hierarchy structure."""
    root_tag: str
    children: Dict[str, 'TagHierarchy'] = field(default_factory=dict)
    level: int = 0
    usage_count: int = 0
    notes: Set[str] = field(default_factory=set)
    
    def add_child(self, tag: str, hierarchy: 'TagHierarchy'):
        """Add a child hierarchy."""
        hierarchy.level = self.level + 1
        self.children[tag] = hierarchy
    
    def get_all_tags(self) -> List[str]:
        """Get all tags in this hierarchy."""
        tags = [self.root_tag]
        for child in self.children.values():
            tags.extend(child.get_all_tags())
        return tags


@dataclass
class TagOperation:
    """Represents a tag operation to be performed."""
    operation_type: str  # "add", "remove", "rename", "merge"
    note_id: str
    old_tag: Optional[str] = None
    new_tag: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class TagAnalysisResult:
    """Result of comprehensive tag analysis."""
    total_tags: int
    unique_tags: int
    orphaned_tags: List[str]
    duplicate_candidates: List[TagCluster]
    hierarchy_suggestions: List[TagHierarchy]
    usage_statistics: Dict[str, Any]
    quality_issues: List[str]
    optimization_suggestions: List[str]
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TagManagerConfig:
    """Configuration for tag management operations."""
    # Similarity thresholds
    fuzzy_similarity_threshold: float = 0.8
    semantic_similarity_threshold: float = 0.7
    exact_match_threshold: float = 0.95
    
    # String similarity settings
    enable_fuzzy_matching: bool = True
    case_insensitive: bool = True
    normalize_special_chars: bool = True
    
    # Semantic analysis
    enable_semantic_analysis: bool = True
    embedding_model: Optional[str] = None
    semantic_cache_ttl: int = 86400  # 24 hours
    
    # Hierarchy analysis
    max_hierarchy_depth: int = 5
    min_usage_threshold: int = 2
    hierarchy_confidence_threshold: float = 0.6
    
    # Auto-tagging
    auto_tag_confidence_threshold: float = 0.7
    max_auto_tags_per_note: int = 10
    enable_ai_suggestions: bool = True
    
    # Performance settings
    batch_size: int = 100
    max_concurrent_operations: int = 10
    enable_caching: bool = True
    
    # Content analysis for tagging
    min_content_length: int = 50
    tag_extraction_methods: List[str] = field(default_factory=lambda: [
        "frontmatter", "inline", "content_based"
    ])


# Type aliases for clarity
NoteId = str
Embedding = np.ndarray
DocumentId = str