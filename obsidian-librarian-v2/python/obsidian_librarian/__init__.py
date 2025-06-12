"""
Obsidian Librarian - An intelligent content management system for Obsidian vaults.

This package combines high-performance Rust core operations with Python's rich
AI/ML ecosystem to provide comprehensive vault management capabilities.
"""

__version__ = "0.1.0"
__author__ = "Obsidian Librarian Team"
__license__ = "MIT"

# Import core Rust bindings
try:
    from ._core import (
        Vault as RustVault,
        VaultConfig as RustVaultConfig,
        Note as RustNote,
        VaultStats as RustVaultStats,
        FileOps as RustFileOps,
        VaultEvent as RustVaultEvent,
        WatcherConfig as RustWatcherConfig,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import Rust core components. "
        "Please ensure the package was built correctly with maturin."
    ) from e

# Import Python components
from .models import Note, NoteMetadata, Task, WikiLink
from .services import ResearchService, AnalysisService, TemplateService
from .vault import Vault

__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__license__",
    
    # Core classes
    "Vault",
    "Note",
    "NoteMetadata",
    "Task",
    "WikiLink",
    
    # Services
    "ResearchService",
    "AnalysisService", 
    "TemplateService",
    
    # Rust bindings (prefixed for clarity)
    "RustVault",
    "RustVaultConfig", 
    "RustNote",
    "RustVaultStats",
    "RustFileOps",
    "RustVaultEvent",
    "RustWatcherConfig",
]