"""
Obsidian Vault Tools - Comprehensive toolkit for managing Obsidian vaults
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core modules
from .manager import VaultManager, EnhancedVaultManager
from .analysis import TagAnalyzer, VaultAnalyzer
from .organization import TagFixer, FileOrganizer
from .backup import BackupManager
from .utils import Config

# Optional AI modules
try:
    from .ai import QuerySystem, ModelManager
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# Optional creative modules  
try:
    from .creative import ASCIIArtConverter, FlowchartGenerator
    CREATIVE_AVAILABLE = True
except ImportError:
    CREATIVE_AVAILABLE = False

__all__ = [
    "VaultManager",
    "EnhancedVaultManager", 
    "TagAnalyzer",
    "VaultAnalyzer",
    "TagFixer",
    "FileOrganizer",
    "BackupManager",
    "Config",
    "AI_AVAILABLE",
    "CREATIVE_AVAILABLE",
]