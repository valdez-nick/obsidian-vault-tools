"""Data models for Obsidian Librarian."""

from .models import (
    # Core models
    WikiLink,
    Task,
    NoteMetadata,
    Note,
    
    # Vault models
    VaultStats,
    VaultConfig,
    
    # Research models
    ResearchQuery,
    ResearchResult,
    
    # Analysis models
    ContentSimilarity,
    DuplicateCluster,
    AnalysisResult,
    
    # Template models
    TemplateMatch,
    TemplateRule,
    TemplateApplication,
    
    # Configuration
    LibrarianConfig,
    LibrarianStats,
    
    # Type aliases
    NoteId,
    Embedding,
    DocumentId,
)

__all__ = [
    # Core models
    "WikiLink",
    "Task", 
    "NoteMetadata",
    "Note",
    
    # Vault models
    "VaultStats",
    "VaultConfig",
    
    # Research models
    "ResearchQuery",
    "ResearchResult",
    
    # Analysis models  
    "ContentSimilarity",
    "DuplicateCluster",
    "AnalysisResult",
    
    # Template models
    "TemplateMatch",
    "TemplateRule",
    "TemplateApplication",
    
    # Configuration
    "LibrarianConfig",
    "LibrarianStats",
    
    # Type aliases
    "NoteId",
    "Embedding",
    "DocumentId",
]