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
    RUST_BINDINGS_AVAILABLE = True
except ImportError as e:
    # For development/testing without Rust bindings
    import warnings
    warnings.warn(
        "Rust core components not available. "
        "Some functionality will be limited. "
        "Build with maturin for full functionality.",
        ImportWarning
    )
    RUST_BINDINGS_AVAILABLE = False
    
    # Create dummy classes for development
    class RustVault: pass
    class RustVaultConfig: pass
    class RustNote: pass
    class RustVaultStats: pass
    class RustFileOps: pass
    class RustVaultEvent: pass
    class RustWatcherConfig: pass

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