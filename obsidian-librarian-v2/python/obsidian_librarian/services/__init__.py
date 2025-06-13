"""Services for Obsidian Librarian."""

from .research import ResearchService
from .analysis import AnalysisService
from .template import TemplateService
from .auto_organizer import (
    AutoOrganizer,
    ContentClassifier,
    DirectoryRouter,
    OrganizationLearner,
    RuleEngine,
    FileWatcher,
    ClassificationResult,
    OrganizationRule,
    UserFeedback,
    ContentFeatures,
    ClassificationConfidence,
    OrganizationAction,
)

__all__ = [
    "ResearchService",
    "AnalysisService", 
    "TemplateService",
    # Auto-organization service
    "AutoOrganizer",
    "ContentClassifier",
    "DirectoryRouter", 
    "OrganizationLearner",
    "RuleEngine",
    "FileWatcher",
    # Auto-organization models
    "ClassificationResult",
    "OrganizationRule",
    "UserFeedback",
    "ContentFeatures",
    "ClassificationConfidence",
    "OrganizationAction",
]