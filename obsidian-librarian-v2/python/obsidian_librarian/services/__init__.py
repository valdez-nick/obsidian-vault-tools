"""Services for Obsidian Librarian."""

from .research import ResearchService
from .analysis import AnalysisService
from .template import TemplateService
from .git_service import GitService

# Import tag management services if available
try:
    from .tag_manager import (
        TagManagerService,
        TagAnalyzer,
        TagSimilarityDetector,
        TagHierarchyBuilder,
        AutoTagger,
        TagOperations,
    )
    TAG_SERVICES_AVAILABLE = True
except ImportError:
    TAG_SERVICES_AVAILABLE = False

# Import directory organization services if available
try:
    from .auto_organizer import (
        ContentClassifier,
        DirectoryRouter,
        OrganizationLearner,
        RuleEngine,
        FileWatcher,
    )
    DIRECTORY_SERVICES_AVAILABLE = True
except ImportError:
    DIRECTORY_SERVICES_AVAILABLE = False

__all__ = [
    "ResearchService",
    "AnalysisService", 
    "TemplateService",
    "GitService",
]

if TAG_SERVICES_AVAILABLE:
    __all__.extend([
        "TagManagerService",
        "TagAnalyzer",
        "TagSimilarityDetector", 
        "TagHierarchyBuilder",
        "AutoTagger",
        "TagOperations",
    ])

if DIRECTORY_SERVICES_AVAILABLE:
    __all__.extend([
        "ContentClassifier",
        "DirectoryRouter",
        "OrganizationLearner",
        "RuleEngine",
        "FileWatcher",
    ])