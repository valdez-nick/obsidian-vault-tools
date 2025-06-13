"""
Data models for Obsidian Librarian.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
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
    
    # Git settings
    enable_git_integration: bool = True
    git_auto_backup_threshold: int = 10
    git_auto_backup_interval: int = 3600
    git_backup_branch_prefix: str = "backup"
    git_experiment_branch_prefix: str = "experiment"
    git_commit_template: str = "Obsidian Librarian: {action} - {timestamp}"
    git_branch: str = "main"
    
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


# Directory Organization Models

@dataclass
class DirectoryRule:
    """Rule for pattern-based directory routing."""
    id: str
    name: str
    pattern: str  # File pattern to match (glob or regex)
    destination: str  # Target directory path
    conditions: List[Dict[str, Any]] = field(default_factory=list)  # Additional conditions
    priority: int = 0  # Higher priority rules are applied first
    enabled: bool = True
    rule_type: str = "pattern"  # pattern, content, metadata, ai
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    applied_count: int = 0
    success_rate: float = 1.0
    
    def matches_file(self, file_path: Path) -> bool:
        """Check if this rule matches the given file path."""
        if not self.enabled:
            return False
            
        if self.rule_type == "pattern":
            # Use pathlib's match for glob patterns
            return file_path.match(self.pattern)
        elif self.rule_type == "regex":
            import re
            return bool(re.search(self.pattern, str(file_path)))
        
        return False
    
    def validate(self) -> List[str]:
        """Validate the rule configuration."""
        errors = []
        
        if not self.name.strip():
            errors.append("Rule name cannot be empty")
        if not self.pattern.strip():
            errors.append("Rule pattern cannot be empty")
        if not self.destination.strip():
            errors.append("Rule destination cannot be empty")
        if self.priority < 0:
            errors.append("Rule priority must be non-negative")
            
        return errors


@dataclass 
class ClassificationResult:
    """Result from AI-powered content classification."""
    file_path: Path
    confidence: float  # 0.0 to 1.0
    predicted_category: str
    suggested_directory: str
    reasoning: str = ""
    alternatives: List[Dict[str, Any]] = field(default_factory=list)  # Alternative classifications
    
    # Classification metadata
    model_used: str = "unknown"
    features_used: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def is_confident(self, threshold: float = 0.7) -> bool:
        """Check if classification confidence meets threshold."""
        return self.confidence >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'file_path': str(self.file_path),
            'confidence': self.confidence,
            'predicted_category': self.predicted_category,
            'suggested_directory': self.suggested_directory,
            'reasoning': self.reasoning,
            'alternatives': self.alternatives,
            'model_used': self.model_used,
            'features_used': self.features_used,
            'processing_time_ms': self.processing_time_ms,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class MoveOperation:
    """Safe file move operation with validation and rollback."""
    id: str
    source_path: Path
    destination_path: Path
    operation_type: str = "move"  # move, copy, link
    dry_run: bool = True
    backup_created: bool = False
    backup_path: Optional[Path] = None
    
    # Operation state
    status: str = "pending"  # pending, in_progress, completed, failed, rolled_back
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Validation results
    conflicts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata preservation
    preserve_timestamps: bool = True
    preserve_permissions: bool = True
    update_links: bool = True  # Update internal links after move
    
    def validate(self) -> bool:
        """Validate the move operation before execution."""
        self.conflicts.clear()
        self.warnings.clear()
        
        if not self.source_path.exists():
            self.conflicts.append(f"Source file does not exist: {self.source_path}")
        
        if self.destination_path.exists():
            self.conflicts.append(f"Destination already exists: {self.destination_path}")
        
        if not self.destination_path.parent.exists():
            self.warnings.append(f"Destination directory will be created: {self.destination_path.parent}")
        
        # Check for permission issues
        try:
            if self.source_path.exists() and not self.source_path.parent.stat().st_mode & 0o200:
                self.conflicts.append(f"No write permission for source directory: {self.source_path.parent}")
        except OSError as e:
            self.conflicts.append(f"Cannot check source permissions: {e}")
        
        return len(self.conflicts) == 0
    
    def can_execute(self) -> bool:
        """Check if operation can be safely executed."""
        return self.validate() and self.status == "pending"
    
    def mark_started(self):
        """Mark operation as started."""
        self.status = "in_progress"
        self.started_at = datetime.utcnow()
    
    def mark_completed(self):
        """Mark operation as completed."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self, error: str):
        """Mark operation as failed."""
        self.status = "failed"
        self.error_message = error
        self.completed_at = datetime.utcnow()


@dataclass
class OrganizationPlan:
    """Comprehensive plan for vault organization."""
    id: str
    name: str
    description: str = ""
    
    # Operations to perform
    move_operations: List[MoveOperation] = field(default_factory=list)
    directory_rules: List[DirectoryRule] = field(default_factory=list)
    classification_results: List[ClassificationResult] = field(default_factory=list)
    
    # Plan metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    estimated_duration_minutes: float = 0.0
    total_files_affected: int = 0
    
    # Execution state
    status: str = "draft"  # draft, validated, executing, completed, failed
    execution_started_at: Optional[datetime] = None
    execution_completed_at: Optional[datetime] = None
    
    # Results
    successful_operations: int = 0
    failed_operations: int = 0
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_move_operation(self, source: Path, destination: Path, 
                          operation_type: str = "move", dry_run: bool = True) -> MoveOperation:
        """Add a move operation to the plan."""
        op_id = f"move_{len(self.move_operations):04d}"
        operation = MoveOperation(
            id=op_id,
            source_path=source,
            destination_path=destination,
            operation_type=operation_type,
            dry_run=dry_run
        )
        self.move_operations.append(operation)
        self.total_files_affected += 1
        return operation
    
    def validate_plan(self) -> Dict[str, Any]:
        """Validate the entire organization plan."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {
                'total_operations': len(self.move_operations),
                'valid_operations': 0,
                'invalid_operations': 0,
                'estimated_conflicts': 0
            }
        }
        
        for operation in self.move_operations:
            if operation.validate():
                validation_result['stats']['valid_operations'] += 1
            else:
                validation_result['stats']['invalid_operations'] += 1
                validation_result['errors'].extend(operation.conflicts)
                validation_result['valid'] = False
            
            validation_result['warnings'].extend(operation.warnings)
            validation_result['stats']['estimated_conflicts'] += len(operation.conflicts)
        
        # Check for duplicate destinations
        destinations = [op.destination_path for op in self.move_operations]
        duplicates = [dest for dest in set(destinations) if destinations.count(dest) > 1]
        if duplicates:
            validation_result['valid'] = False
            validation_result['errors'].extend([f"Duplicate destination: {dest}" for dest in duplicates])
        
        return validation_result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the organization plan."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'status': self.status,
            'total_operations': len(self.move_operations),
            'total_files_affected': self.total_files_affected,
            'estimated_duration_minutes': self.estimated_duration_minutes,
            'created_at': self.created_at.isoformat(),
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'move_operations': [
                {
                    'id': op.id,
                    'source_path': str(op.source_path),
                    'destination_path': str(op.destination_path),
                    'operation_type': op.operation_type,
                    'status': op.status,
                    'dry_run': op.dry_run
                }
                for op in self.move_operations
            ],
            'directory_rules': [
                {
                    'id': rule.id,
                    'name': rule.name,
                    'pattern': rule.pattern,
                    'destination': rule.destination,
                    'rule_type': rule.rule_type,
                    'enabled': rule.enabled,
                    'priority': rule.priority
                }
                for rule in self.directory_rules
            ],
            'status': self.status,
            'total_files_affected': self.total_files_affected,
            'created_at': self.created_at.isoformat(),
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations
        }


# Type aliases for clarity
NoteId = str
Embedding = np.ndarray
DocumentId = str