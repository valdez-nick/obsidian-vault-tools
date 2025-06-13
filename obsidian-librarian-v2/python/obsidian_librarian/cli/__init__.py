"""CLI components for Obsidian Librarian."""

from .main import app

try:
    from .git_commands import git_app
    GIT_COMMANDS_AVAILABLE = True
except ImportError:
    GIT_COMMANDS_AVAILABLE = False

try:
    from .tag_commands import tag_commands
    TAG_COMMANDS_AVAILABLE = True
except ImportError:
    TAG_COMMANDS_AVAILABLE = False

__all__ = ["app"]

if GIT_COMMANDS_AVAILABLE:
    __all__.append("git_app")

if TAG_COMMANDS_AVAILABLE:
    __all__.append("tag_commands")
