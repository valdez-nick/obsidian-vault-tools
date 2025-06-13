"""
Main CLI entry point for Obsidian Librarian.

This module provides the main CLI application with all command groups
including the new tag management commands.
"""

import click
from pathlib import Path
from typing import Optional

from .utils import setup_logging, console
from .utils.console import create_header_panel
from .commands.tag_commands import tag_commands
from ..models import LibrarianConfig


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def app(ctx, verbose: bool, config: Optional[str]):
    """Obsidian Librarian - Intelligent vault management and research assistant.
    
    Comprehensive tag management, content analysis, and research capabilities
    for Obsidian vaults with AI-powered automation and organization.
    """
    setup_logging(verbose)
    
    # Load configuration
    if config:
        config_path = Path(config)
        # In practice, you'd load from JSON/YAML
        ctx.obj = LibrarianConfig()
    else:
        ctx.obj = LibrarianConfig()
    
    console.print(create_header_panel(
        "Obsidian Librarian v0.1.0",
        description="Intelligent vault management and research assistant"
    ))


# Add the new tag command group
app.add_command(tag_commands)

# For now, we'll defer importing existing commands to avoid circular imports
# They can be added separately or the old CLI can be updated to use this structure

if __name__ == '__main__':
    app()