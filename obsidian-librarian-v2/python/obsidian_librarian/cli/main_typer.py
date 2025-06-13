"""
Main CLI entry point for Obsidian Librarian using Typer.

This module provides the main CLI application with all command groups
including tag management, git operations, and core functionality.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from ..models import LibrarianConfig

# Simple config loader stub
def load_config_from_yaml(config_path: Path) -> LibrarianConfig:
    """Load configuration from YAML file."""
    # For now, just return default config
    # TODO: Implement proper YAML config loading
    return LibrarianConfig()

# Initialize console
console = Console()

# Create main app
app = typer.Typer(
    name="obsidian-librarian",
    help="Obsidian Librarian - Intelligent vault management and research assistant",
    add_completion=False,
    rich_markup_mode="rich",
)

# Global state for configuration
state = {"config": None, "verbose": False}


def version_callback(value: bool):
    """Show version information."""
    if value:
        console.print(Panel.fit(
            "[bold blue]Obsidian Librarian v0.1.0[/bold blue]\n"
            "Intelligent vault management and research assistant\n"
            "Built with 🦀 Rust + 🐍 Python",
            border_style="blue"
        ))
        raise typer.Exit()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    version: bool = typer.Option(None, "--version", callback=version_callback, help="Show version"),
):
    """
    Obsidian Librarian - Intelligent vault management and research assistant.
    
    Comprehensive tag management, content analysis, and research capabilities
    for Obsidian vaults with AI-powered automation and organization.
    """
    # Store global state
    state["verbose"] = verbose
    
    # Load configuration
    if config and config.exists():
        try:
            state["config"] = load_config_from_yaml(config)
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            raise typer.Exit(1)
    else:
        state["config"] = LibrarianConfig()
    
    # Show header only in verbose mode
    if verbose:
        console.print(Panel.fit(
            "[bold blue]Obsidian Librarian[/bold blue]\n"
            "Intelligent vault management and research assistant",
            border_style="blue"
        ))


# Import and add subcommands
def setup_commands():
    """Setup all command groups with error handling."""
    # Add tag commands
    try:
        from .commands.tag_commands_typer import app as tag_app
        app.add_typer(tag_app, name="tags", help="Comprehensive tag management")
    except ImportError as e:
        if state.get("verbose"):
            console.print(f"[yellow]Warning: Tag commands not available: {e}[/yellow]")

    # Add git commands
    try:
        from .git_commands import git_app
        app.add_typer(git_app, name="git", help="Git-based version control")
    except ImportError as e:
        if state.get("verbose"):
            console.print(f"[yellow]Warning: Git commands not available: {e}[/yellow]")

    # Add core commands
    try:
        from .commands.core_commands import (
            init_command,
            stats_command,
            organize_command,
            research_command,
            analyze_command,
            curate_command,
        )

        app.command(name="init", help="Initialize a new Obsidian vault")(init_command)
        app.command(name="stats", help="Show vault statistics")(stats_command)
        app.command(name="organize", help="Organize vault structure")(organize_command)
        app.command(name="research", help="Perform research queries")(research_command)
        app.command(name="analyze", help="Analyze vault for insights")(analyze_command)
        app.command(name="curate", help="Curate and improve content")(curate_command)
    except ImportError as e:
        console.print(f"[red]Error: Core commands not available: {e}[/red]")

# Setup commands after app definition
setup_commands()


if __name__ == "__main__":
    app()