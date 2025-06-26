"""
Obsidian Vault Tools - Comprehensive toolkit for managing Obsidian vaults
"""

__version__ = "2.2.1"
__author__ = "Nick Valdez"
__email__ = "nvaldez@siftscience.com"

# Core modules
from .manager import VaultManager, EnhancedVaultManager
from .analysis import TagAnalyzer, VaultAnalyzer
from .organization import TagFixer, FileOrganizer
from .backup import BackupManager
from .utils import Config

# Optional AI modules
try:
    # Import from parent directory's ai module
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    from ai.llm_query_system import LLMQuerySystem as QuerySystem
    from ai.model_manager import ModelManager
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    QuerySystem = None
    ModelManager = None

# Optional creative modules  
try:
    from creative.ascii_art_converter import ASCIIArtConverter
    from creative.flowchart_generator import FlowchartGenerator
    CREATIVE_AVAILABLE = True
except ImportError as e:
    CREATIVE_AVAILABLE = False
    ASCIIArtConverter = None
    FlowchartGenerator = None

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