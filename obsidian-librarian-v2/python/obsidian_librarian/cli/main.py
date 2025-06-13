"""
Comprehensive CLI implementation for Obsidian Librarian v2.

This module provides a complete command-line interface using Typer for all
Obsidian Librarian functionality including vault management, research, analysis,
and backup operations.
"""

import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import os

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.tree import Tree
from rich.markdown import Markdown
from rich import print as rprint
import structlog

from ..librarian import ObsidianLibrarian, LibrarianSession
from ..models import LibrarianConfig
from ..vault import Vault, scan_vault_async
from .git_commands import git_app
from ..config_loader import load_vault_config_from_yaml

# Initialize app and console
app = typer.Typer(
    name="obsidian-librarian",
    help="Obsidian Librarian - Intelligent vault management and research assistant",
    add_completion=False,
    rich_markup_mode="rich",
)

# Add Git subcommand
app.add_typer(git_app, name="git", help="Git-based version control commands")

console = Console()
logger = structlog.get_logger(__name__)

# Configuration file paths
CONFIG_DIR = Path.home() / ".obsidian-librarian"
CONFIG_FILE = CONFIG_DIR / "config.json"
VAULT_CONFIG_FILE = ".obsidian-librarian.json"


def setup_logging(debug: bool = False, quiet: bool = False) -> None:
    """Configure structured logging with appropriate level."""
    if quiet:
        level = "ERROR"
    elif debug:
        level = "DEBUG"
    else:
        level = "INFO"
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if debug else structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def load_config() -> Dict[str, Any]:
    """Load global configuration."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load config", error=str(e))
    return {}


def save_config(config: Dict[str, Any]) -> None:
    """Save global configuration."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def load_vault_config(vault_path: Path) -> Dict[str, Any]:
    """Load vault-specific configuration."""
    config_path = vault_path / VAULT_CONFIG_FILE
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load vault config", error=str(e))
    return {}


def save_vault_config(vault_path: Path, config: Dict[str, Any]) -> None:
    """Save vault-specific configuration."""
    config_path = vault_path / VAULT_CONFIG_FILE
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def ensure_dependencies() -> bool:
    """Check if required dependencies are available."""
    missing = []
    
    # Check for git
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("git")
    
    if missing:
        console.print(f"[red]Missing required dependencies: {', '.join(missing)}[/red]")
        console.print("Please install the missing dependencies and try again.")
        return False
    
    return True


@app.command()
def init(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-initialization"),
) -> None:
    """Initialize vault configuration for Obsidian Librarian."""
    console.print(Panel.fit(
        "[bold blue]Obsidian Librarian Initialization[/bold blue]",
        border_style="blue"
    ))
    
    # Check if vault exists
    if not vault_path.exists():
        console.print(f"[red]Vault path does not exist: {vault_path}[/red]")
        raise typer.Exit(1)
    
    # Check if it's an Obsidian vault
    obsidian_dir = vault_path / ".obsidian"
    if not obsidian_dir.exists():
        console.print("[yellow]Warning: No .obsidian directory found. This may not be an Obsidian vault.[/yellow]")
        if not Confirm.ask("Continue anyway?"):
            raise typer.Exit(0)
    
    # Check if already initialized
    config_path = vault_path / VAULT_CONFIG_FILE
    if config_path.exists() and not force:
        console.print("[yellow]Vault already initialized. Use --force to re-initialize.[/yellow]")
        raise typer.Exit(0)
    
    # Create configuration
    config = {
        "version": "0.1.0",
        "initialized_at": datetime.now().isoformat(),
        "vault_name": vault_path.name,
        "settings": {
            "auto_backup": True,
            "backup_frequency": "daily",
            "enable_ai_features": True,
            "default_template_dir": "Templates",
            "research_output_dir": "Research Library",
        }
    }
    
    # Create necessary directories
    dirs_to_create = [
        vault_path / ".obsidian-librarian",
        vault_path / config["settings"]["default_template_dir"],
        vault_path / config["settings"]["research_output_dir"],
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created directory: {dir_path.relative_to(vault_path)}")
    
    # Save configuration
    save_vault_config(vault_path, config)
    console.print(f"[green]✓[/green] Saved configuration to {VAULT_CONFIG_FILE}")
    
    # Update global config
    global_config = load_config()
    if "vaults" not in global_config:
        global_config["vaults"] = {}
    global_config["vaults"][str(vault_path.absolute())] = {
        "name": vault_path.name,
        "last_accessed": datetime.now().isoformat(),
    }
    save_config(global_config)
    
    console.print("\n[bold green]Vault initialized successfully![/bold green]")
    console.print(f"Path: {vault_path.absolute()}")


@app.command()
def stats(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed statistics"),
) -> None:
    """Show vault statistics and insights."""
    async def run_stats():
        console.print(Panel.fit(
            f"[bold blue]Vault Statistics: {vault_path.name}[/bold blue]",
            border_style="blue"
        ))
        
        analysis_results = {}
        vault_info = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing vault...", total=None)
            
            try:
                # Quick scan
                vault_info = await scan_vault_async(vault_path)
                
                # Create librarian session for detailed analysis
                async with ObsidianLibrarian() as librarian:
                    session_id = await librarian.create_session(vault_path)
                    
                    # Get detailed stats if requested
                    if detailed:
                        analysis_results = []
                        async for result in librarian.analyze_vault(session_id, find_duplicates=True):
                            if result.get('type') == 'complete':
                                analysis_results = result.get('data', {})
                                break
                    
                    session_status = await librarian.get_session_status(session_id)
                
                progress.update(task, description="Analysis complete!")
            except Exception as e:
                console.print(f"[red]Error during analysis: {e}[/red]")
                logger.error("Stats command failed", error=str(e))
                raise typer.Exit(1)
        
        # Display basic stats
        table = Table(title="Basic Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Notes", str(vault_info.get('note_count', 0)))
        table.add_row("Vault Size", f"{vault_info.get('total_size', 0) / 1024 / 1024:.2f} MB")
        
        if vault_info.get('last_modified'):
            table.add_row("Last Modified", vault_info['last_modified'].strftime('%Y-%m-%d %H:%M'))
        
        console.print(table)
        
        # Display detailed stats if available
        if detailed and analysis_results:
            vault_stats = analysis_results.get('vault_stats', {})
            
            detail_table = Table(title="Detailed Analysis")
            detail_table.add_column("Metric", style="cyan")
            detail_table.add_column("Value", style="magenta")
            
            detail_table.add_row("Total Words", f"{vault_stats.get('total_words', 0):,}")
            detail_table.add_row("Total Links", str(vault_stats.get('total_links', 0)))
            detail_table.add_row("Total Tasks", str(vault_stats.get('total_tasks', 0)))
            detail_table.add_row("Average Quality Score", f"{vault_stats.get('avg_quality_score', 0):.2f}")
            
            if 'duplicate_clusters' in analysis_results:
                detail_table.add_row("Duplicate Clusters", str(analysis_results['duplicate_clusters']))
            
            console.print("\n")
            console.print(detail_table)
    
    asyncio.run(run_stats())


@app.command()
def duplicates(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    threshold: float = typer.Option(0.85, "--threshold", "-t", help="Similarity threshold (0-1)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Preview without making changes"),
) -> None:
    """Find and manage duplicate content in the vault."""
    async def run_duplicates():
        console.print(Panel.fit(
            "[bold blue]Duplicate Detection[/bold blue]",
            border_style="blue"
        ))
        
        if not dry_run:
            console.print("[yellow]Warning: Running without --dry-run will modify your vault![/yellow]")
            if not Confirm.ask("Are you sure you want to continue?"):
                raise typer.Exit(0)
        
        async with ObsidianLibrarian() as librarian:
            session_id = await librarian.create_session(vault_path)
            
            duplicates_found = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("Finding duplicates...", total=100)
                
                async for result in librarian.analyze_vault(session_id, find_duplicates=True, quality_analysis=False):
                    if result.get('type') == 'progress':
                        current = result['progress']['current']
                        total = result['progress']['total']
                        progress.update(task, completed=(current/total) * 100)
                    elif result.get('type') == 'duplicates':
                        duplicates_found = result['data']['clusters']
                        progress.update(task, completed=100)
            
            # Display results
            if not duplicates_found:
                console.print("[green]No duplicate content found![/green]")
                return
            
            console.print(f"\n[yellow]Found {len(duplicates_found)} duplicate clusters[/yellow]")
            
            # Create detailed report
            report_data = []
            
            for i, cluster in enumerate(duplicates_found, 1):
                console.print(f"\n[bold]Cluster {i}:[/bold]")
                console.print(f"  Type: {cluster['type']}")
                console.print(f"  Confidence: {cluster['confidence']:.2%}")
                console.print(f"  Notes: {cluster['note_count']}")
                
                report_data.append({
                    "cluster_id": cluster['cluster_id'],
                    "type": cluster['type'],
                    "confidence": cluster['confidence'],
                    "note_count": cluster['note_count'],
                })
                
                if i >= 5 and not output:  # Limit console output
                    console.print(f"\n[dim]... and {len(duplicates_found) - 5} more clusters[/dim]")
                    break
            
            # Save results if requested
            if output:
                with open(output, 'w') as f:
                    json.dump({
                        "timestamp": datetime.now().isoformat(),
                        "vault": str(vault_path),
                        "threshold": threshold,
                        "clusters": report_data
                    }, f, indent=2)
                console.print(f"\n[blue]Full report saved to {output}[/blue]")
    
    asyncio.run(run_duplicates())


@app.command()
def organize(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    templates: bool = typer.Option(True, "--templates/--no-templates", help="Apply templates"),
    structure: bool = typer.Option(True, "--structure/--no-structure", help="Organize structure"),
    quality: bool = typer.Option(True, "--quality/--no-quality", help="Improve quality"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Preview without making changes"),
) -> None:
    """Organize and improve vault content intelligently."""
    async def run_organize():
        console.print(Panel.fit(
            "[bold blue]Vault Organization[/bold blue]",
            border_style="blue"
        ))
        
        if not dry_run:
            console.print("[yellow]Warning: This will modify your vault content![/yellow]")
            if not Confirm.ask("Are you sure you want to continue?"):
                raise typer.Exit(0)
        
        async with ObsidianLibrarian() as librarian:
            session_id = await librarian.create_session(vault_path)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Organizing vault...", total=None)
                
                # Run curation
                results = await librarian.curate_content(
                    session_id,
                    remove_duplicates=False,  # Handled separately
                    improve_quality=quality,
                    organize_structure=structure,
                )
                
                # Apply templates if requested
                if templates:
                    progress.update(task, description="Applying templates...")
                    template_results = await librarian.apply_templates(
                        session_id,
                        auto_detect=True
                    )
                    results['templates_applied'] = template_results['successful']
                
                progress.update(task, description="Organization complete!")
            
            # Display results
            table = Table(title="Organization Results")
            table.add_column("Operation", style="cyan")
            table.add_column("Count", style="magenta")
            
            if quality:
                table.add_row("Quality Improvements", str(results.get('quality_improvements', 0)))
            if structure:
                table.add_row("Structure Improvements", str(results.get('structure_improvements', 0)))
            if templates:
                table.add_row("Templates Applied", str(results.get('templates_applied', 0)))
            table.add_row("Errors", str(len(results.get('errors', []))))
            
            console.print(table)
            
            if results.get('errors'):
                console.print("\n[red]Errors encountered:[/red]")
                for error in results['errors'][:5]:
                    console.print(f"  • {error}")
    
    asyncio.run(run_organize())


@app.command()
def research(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    query: str = typer.Argument(..., help="Research query"),
    sources: Optional[List[str]] = typer.Option(None, "--source", "-s", help="Specific sources to search"),
    max_results: int = typer.Option(20, "--max-results", "-n", help="Maximum results"),
    organize: bool = typer.Option(True, "--organize/--no-organize", help="Organize results in vault"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to file"),
) -> None:
    """Research topics and add findings to vault."""
    async def run_research():
        console.print(Panel.fit(
            f"[bold blue]Research: {query}[/bold blue]",
            border_style="blue"
        ))
        
        if sources:
            console.print(f"[cyan]Sources: {', '.join(sources)}[/cyan]")
        
        results_collected = []
        
        async with ObsidianLibrarian() as librarian:
            session_id = await librarian.create_session(vault_path)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Researching...", total=None)
                
                async for result in librarian.research(
                    session_id, 
                    query, 
                    sources, 
                    max_results, 
                    organize
                ):
                    if result.get('type') == 'result':
                        results_collected.append(result['data'])
                        progress.update(
                            task, 
                            description=f"Found {len(results_collected)} results..."
                        )
                    elif result.get('type') == 'status':
                        progress.update(task, description=result['message'])
                    elif result.get('type') == 'complete':
                        progress.update(task, description="Research complete!")
            
            # Display results
            console.print(f"\n[bold green]Found {len(results_collected)} results[/bold green]")
            
            # Show top results
            for i, result in enumerate(results_collected[:5], 1):
                console.print(f"\n[bold cyan]{i}. {result['title']}[/bold cyan]")
                console.print(f"[blue]Source:[/blue] {result['source']}")
                console.print(f"[blue]Quality:[/blue] {result['quality_score']:.2f}")
                console.print(f"[blue]URL:[/blue] {result['url']}")
                
                if result.get('summary'):
                    summary = result['summary']
                    if len(summary) > 200:
                        summary = summary[:200] + "..."
                    console.print(f"[dim]{summary}[/dim]")
            
            if len(results_collected) > 5:
                console.print(f"\n[dim]... and {len(results_collected) - 5} more results[/dim]")
            
            # Save results if requested
            if output:
                output_data = {
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "sources": sources or ["all"],
                    "results": results_collected
                }
                with open(output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                console.print(f"\n[blue]Results saved to {output}[/blue]")
            
            if organize:
                console.print("\n[green]Results have been organized in your vault[/green]")
    
    asyncio.run(run_research())


@app.command()
def analyze(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    quality: bool = typer.Option(True, "--quality/--no-quality", help="Analyze content quality"),
    links: bool = typer.Option(True, "--links/--no-links", help="Analyze link structure"),
    tasks: bool = typer.Option(True, "--tasks/--no-tasks", help="Analyze tasks"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save analysis to file"),
) -> None:
    """Analyze vault health and provide insights."""
    async def run_analysis():
        console.print(Panel.fit(
            "[bold blue]Vault Health Analysis[/bold blue]",
            border_style="blue"
        ))
        
        analysis_data = {}
        
        async with ObsidianLibrarian() as librarian:
            session_id = await librarian.create_session(vault_path)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("Analyzing vault...", total=100)
                
                async for result in librarian.analyze_vault(
                    session_id,
                    find_duplicates=True,
                    quality_analysis=quality
                ):
                    if result.get('type') == 'progress':
                        current = result['progress']['current']
                        total = result['progress']['total']
                        progress.update(task, completed=(current/total) * 100)
                    elif result.get('type') == 'complete':
                        analysis_data = result.get('data', {})
                        progress.update(task, completed=100)
            
            # Display health report
            console.print("\n[bold green]Vault Health Report[/bold green]")
            
            if 'vault_stats' in analysis_data:
                stats = analysis_data['vault_stats']
                
                # Overall health score (simple calculation)
                health_score = min(100, stats.get('avg_quality_score', 0) * 100)
                
                # Health indicator
                if health_score >= 80:
                    health_status = "[green]Excellent[/green]"
                elif health_score >= 60:
                    health_status = "[yellow]Good[/yellow]"
                elif health_score >= 40:
                    health_status = "[orange1]Fair[/orange1]"
                else:
                    health_status = "[red]Needs Attention[/red]"
                
                console.print(f"\nOverall Health: {health_status} ({health_score:.0f}/100)")
                
                # Detailed metrics
                metrics_table = Table(title="Key Metrics")
                metrics_table.add_column("Metric", style="cyan")
                metrics_table.add_column("Value", style="magenta")
                metrics_table.add_column("Status", style="green")
                
                # Quality metrics
                if quality:
                    avg_quality = stats.get('avg_quality_score', 0)
                    quality_status = "✓" if avg_quality >= 0.7 else "⚠"
                    metrics_table.add_row(
                        "Average Quality",
                        f"{avg_quality:.2f}",
                        quality_status
                    )
                
                # Link health
                if links:
                    total_links = stats.get('total_links', 0)
                    broken_links = stats.get('broken_links', 0)
                    link_health = "✓" if broken_links == 0 else f"⚠ {broken_links} broken"
                    metrics_table.add_row(
                        "Link Health",
                        f"{total_links} total",
                        link_health
                    )
                
                # Task tracking
                if tasks:
                    total_tasks = stats.get('total_tasks', 0)
                    completed_tasks = stats.get('completed_tasks', 0)
                    task_completion = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 100
                    task_status = "✓" if task_completion >= 80 else "⚠"
                    metrics_table.add_row(
                        "Task Completion",
                        f"{task_completion:.0f}%",
                        task_status
                    )
                
                console.print(metrics_table)
                
                # Recommendations
                console.print("\n[bold]Recommendations:[/bold]")
                recommendations = []
                
                if avg_quality < 0.7:
                    recommendations.append("• Consider improving note quality with templates and structure")
                if 'duplicate_clusters' in analysis_data and analysis_data['duplicate_clusters'] > 0:
                    recommendations.append(f"• Found {analysis_data['duplicate_clusters']} duplicate clusters - run 'duplicates' command")
                if broken_links > 0:
                    recommendations.append("• Fix broken links to improve vault connectivity")
                if task_completion < 80:
                    recommendations.append("• Review and complete pending tasks")
                
                if recommendations:
                    for rec in recommendations:
                        console.print(rec)
                else:
                    console.print("[green]• Vault is in excellent condition![/green]")
                
                # Save detailed analysis if requested
                if output:
                    with open(output, 'w') as f:
                        json.dump({
                            "timestamp": datetime.now().isoformat(),
                            "vault": str(vault_path),
                            "health_score": health_score,
                            "analysis": analysis_data
                        }, f, indent=2)
                    console.print(f"\n[blue]Detailed analysis saved to {output}[/blue]")
    
    asyncio.run(run_analysis())


@app.command()
def templates(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    list_templates: bool = typer.Option(False, "--list", "-l", help="List available templates"),
    apply: Optional[str] = typer.Option(None, "--apply", "-a", help="Apply template by name"),
    create: Optional[str] = typer.Option(None, "--create", "-c", help="Create new template"),
    auto: bool = typer.Option(False, "--auto", help="Auto-apply templates to applicable notes"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Preview without making changes"),
) -> None:
    """Manage and apply templates to notes."""
    template_dir = vault_path / "Templates"
    
    # List templates
    if list_templates:
        if not template_dir.exists():
            console.print("[yellow]No template directory found[/yellow]")
            return
        
        templates = list(template_dir.glob("*.md"))
        if not templates:
            console.print("[yellow]No templates found[/yellow]")
            return
        
        console.print("[bold]Available Templates:[/bold]")
        for template in templates:
            console.print(f"  • {template.stem}")
        return
    
    # Create template
    if create:
        template_dir.mkdir(exist_ok=True)
        template_path = template_dir / f"{create}.md"
        
        if template_path.exists():
            console.print(f"[red]Template '{create}' already exists[/red]")
            return
        
        # Basic template structure
        template_content = f"""---
tags: template
created: {datetime.now().strftime('%Y-%m-%d')}
---

# {create} Template

## Overview
<!-- Brief description of this template -->

## Sections
<!-- Template sections go here -->

### Section 1
<!-- Content -->

### Section 2
<!-- Content -->

## Notes
<!-- Any additional notes about using this template -->
"""
        
        template_path.write_text(template_content)
        console.print(f"[green]Created template: {template_path.relative_to(vault_path)}[/green]")
        return
    
    # Apply templates
    if apply or auto:
        async def run_apply():
            async with ObsidianLibrarian() as librarian:
                session_id = await librarian.create_session(vault_path)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Applying templates...", total=None)
                    
                    if auto:
                        results = await librarian.apply_templates(
                            session_id,
                            auto_detect=True
                        )
                    else:
                        # Apply specific template
                        # This would need to be implemented in the librarian
                        console.print("[yellow]Manual template application not yet implemented[/yellow]")
                        return
                    
                    progress.update(task, description="Complete!")
                
                # Display results
                console.print(f"\n[bold]Template Application Results[/bold]")
                console.print(f"Applied: {results['successful']}")
                console.print(f"Failed: {results['failed']}")
                
                if results.get('applications'):
                    console.print("\n[bold]Details:[/bold]")
                    for app in results['applications'][:10]:
                        status = "[green]✓[/green]" if app['success'] else "[red]✗[/red]"
                        console.print(f"{status} {app['note_id']} → {app['template']}")
        
        asyncio.run(run_apply())


@app.command()
def backup(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Commit message"),
    push: bool = typer.Option(False, "--push", "-p", help="Push to remote after commit"),
) -> None:
    """Create Git backup with intelligent commit messages.
    
    This is a compatibility wrapper. For full Git functionality, use:
    obsidian-librarian git backup
    """
    console.print("[yellow]Note: This command is deprecated. Use 'obsidian-librarian git backup' instead.[/yellow]")
    
    # Use the Git service for backup
    async def run_backup():
        # Load configuration from YAML file
        from ..models import VaultConfig
        config = load_vault_config_from_yaml(vault_path)
        if not config:
            config = VaultConfig(enable_git_integration=False)
        
        vault = Vault(vault_path, config)
        
        try:
            await vault.initialize()
            
            if not vault.git_service:
                console.print("[red]Git integration not enabled[/red]")
                return
            
            # Check if repo is initialized
            status = await vault.git_service.get_status()
            if not status.get('initialized'):
                console.print("[yellow]Initializing Git repository...[/yellow]")
                await vault.git_service.initialize_repo()
            
            # Create backup
            commit_hash = await vault.git_service.backup(message)
            
            if commit_hash:
                console.print(f"[green]✓ Created backup: {commit_hash[:8]}[/green]")
                
                # Push if requested (would need implementation in GitService)
                if push:
                    console.print("[yellow]Push functionality not yet implemented in GitService[/yellow]")
                    console.print("Use 'git push' manually for now")
            else:
                console.print("[yellow]No changes to backup[/yellow]")
                
        finally:
            await vault.close()
    
    asyncio.run(run_backup())


@app.command()
def restore(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    commit: Optional[str] = typer.Option(None, "--commit", "-c", help="Restore to specific commit"),
    list_commits: bool = typer.Option(False, "--list", "-l", help="List recent commits"),
) -> None:
    """Restore vault from Git history.
    
    This is a compatibility wrapper. For full Git functionality, use:
    obsidian-librarian git restore
    """
    console.print("[yellow]Note: This command is deprecated. Use 'obsidian-librarian git restore' instead.[/yellow]")
    
    if list_commits:
        # Show history using Git service
        async def run_history():
            # Load configuration from YAML file
            from ..models import VaultConfig
            config = load_vault_config_from_yaml(vault_path)
            if not config:
                config = VaultConfig(enable_git_integration=False)
            
            vault = Vault(vault_path, config)
            
            try:
                await vault.initialize()
                
                if not vault.git_service:
                    console.print("[red]Git integration not enabled[/red]")
                    return
                
                commits = await vault.git_service.get_history(20)
                
                console.print("[bold]Recent commits:[/bold]")
                for commit in commits:
                    console.print(f"{commit.hash[:8]} - {commit.message}")
                    
            finally:
                await vault.close()
        
        asyncio.run(run_history())
        return
    
    if commit:
        # Restore using Git service
        async def run_restore():
            # Load configuration from YAML file
            from ..models import VaultConfig
            config = load_vault_config_from_yaml(vault_path)
            if not config:
                config = VaultConfig(enable_git_integration=False)
            
            vault = Vault(vault_path, config)
            
            try:
                await vault.initialize()
                
                if not vault.git_service:
                    console.print("[red]Git integration not enabled[/red]")
                    return
                
                console.print(f"[yellow]Warning: This will restore vault to commit {commit}[/yellow]")
                if not Confirm.ask("Are you sure?"):
                    return
                
                success = await vault.git_service.restore(commit)
                
                if success:
                    console.print(f"[green]✓ Restored to commit {commit}[/green]")
                else:
                    console.print("[red]Failed to restore vault[/red]")
                    
            finally:
                await vault.close()
        
        asyncio.run(run_restore())
    else:
        console.print("[yellow]Please specify a commit with --commit or use --list to see commits[/yellow]")


@app.command()
def interactive(
    vault_path: Optional[Path] = typer.Argument(None, help="Path to Obsidian vault"),
) -> None:
    """Start interactive mode for exploratory operations."""
    console.print(Panel.fit(
        "[bold blue]Obsidian Librarian Interactive Mode[/bold blue]\n"
        "Type 'help' for commands, 'quit' to exit",
        border_style="blue"
    ))
    
    # Simple REPL implementation
    current_vault = vault_path
    
    commands = {
        "help": "Show available commands",
        "vault": "Set or show current vault",
        "stats": "Show vault statistics",
        "search": "Search vault content",
        "recent": "Show recent notes",
        "quit": "Exit interactive mode",
    }
    
    while True:
        try:
            if current_vault:
                prompt_text = f"[{current_vault.name}]> "
            else:
                prompt_text = "> "
            
            user_input = Prompt.ask(prompt_text).strip()
            
            if not user_input:
                continue
            
            parts = user_input.split()
            cmd = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            if cmd == "quit" or cmd == "exit":
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            elif cmd == "help":
                console.print("[bold]Available commands:[/bold]")
                for cmd_name, desc in commands.items():
                    console.print(f"  {cmd_name:<10} - {desc}")
            
            elif cmd == "vault":
                if args:
                    new_vault = Path(args[0])
                    if new_vault.exists():
                        current_vault = new_vault
                        console.print(f"[green]Switched to vault: {current_vault}[/green]")
                    else:
                        console.print("[red]Vault not found[/red]")
                else:
                    if current_vault:
                        console.print(f"Current vault: {current_vault}")
                    else:
                        console.print("[yellow]No vault selected[/yellow]")
            
            elif cmd == "stats":
                if not current_vault:
                    console.print("[red]No vault selected[/red]")
                    continue
                
                # Run stats command
                asyncio.run(scan_vault_async(current_vault))
            
            elif cmd == "search":
                if not current_vault:
                    console.print("[red]No vault selected[/red]")
                    continue
                
                if not args:
                    console.print("[yellow]Usage: search <query>[/yellow]")
                    continue
                
                query = " ".join(args)
                console.print(f"[cyan]Searching for: {query}[/cyan]")
                # Implement search functionality
                console.print("[yellow]Search not yet implemented in interactive mode[/yellow]")
            
            elif cmd == "recent":
                if not current_vault:
                    console.print("[red]No vault selected[/red]")
                    continue
                
                # Show recent notes
                notes = list(current_vault.glob("**/*.md"))
                notes.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                
                console.print("[bold]Recent notes:[/bold]")
                for note in notes[:10]:
                    rel_path = note.relative_to(current_vault)
                    modified = datetime.fromtimestamp(note.stat().st_mtime)
                    console.print(f"  {rel_path} - {modified.strftime('%Y-%m-%d %H:%M')}")
            
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
                console.print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'quit' to exit[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@app.command()
def curate(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    duplicates: bool = typer.Option(False, "--duplicates", "-d", help="Find and handle duplicate content"),
    remove_duplicates: bool = typer.Option(False, "--remove-duplicates", help="Actually remove/merge duplicates (use with caution)"),
    quality: bool = typer.Option(False, "--quality", "-q", help="Perform quality analysis and improvements"),
    structure: bool = typer.Option(False, "--structure", "-s", help="Improve note structure and organization"),
    templates: bool = typer.Option(False, "--templates", "-t", help="Auto-apply appropriate templates"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive curation mode"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without applying them"),
    batch_size: int = typer.Option(50, "--batch-size", help="Number of notes to process at once"),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Create backup before curation"),
) -> None:
    """Intelligently curate vault content with advanced analysis and improvements."""
    
    if not vault_path.exists():
        console.print(f"[red]Vault path does not exist: {vault_path}[/red]")
        raise typer.Exit(1)
    
    # Default to all operations if none specified
    if not any([duplicates, quality, structure, templates]):
        duplicates = quality = structure = templates = True
    
    console.print(Panel.fit(
        "Content Curation",
        style="bold cyan"
    ))
    
    async def run_curation():
        try:
            # Load configuration
            vault_config = load_vault_config_from_yaml(vault_path)
            if not vault_config:
                vault_config = VaultConfig()
            
            # Initialize librarian
            config = LibrarianConfig()
            librarian = ObsidianLibrarian(config)
            await librarian.initialize()
            
            # Create session
            session_id = await librarian.create_session(vault_path)
            
            # Create backup if requested
            if backup and not dry_run:
                console.print("[blue]Creating backup before curation...[/blue]")
                session = librarian._get_session(session_id)
                if hasattr(session.vault, 'git_service') and session.vault.git_service:
                    try:
                        commit_hash = await session.vault.git_backup("Pre-curation backup")
                        console.print(f"[green]✓ Backup created: {commit_hash[:8]}[/green]")
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not create Git backup: {e}[/yellow]")
            
            curation_results = {
                'duplicates_found': 0,
                'duplicates_processed': 0,
                'quality_improvements': 0,
                'structure_improvements': 0,
                'templates_applied': 0,
                'errors': [],
            }
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console
            ) as progress:
                
                # Handle duplicates
                if duplicates:
                    task = progress.add_task("Finding duplicate content...", total=None)
                    
                    session = librarian._get_session(session_id)
                    duplicate_clusters = await session.analysis_service.find_duplicates()
                    curation_results['duplicates_found'] = len(duplicate_clusters)
                    
                    if duplicate_clusters:
                        progress.update(task, description=f"Found {len(duplicate_clusters)} duplicate clusters")
                        
                        # Display duplicates
                        table = Table(title="Duplicate Content Found")
                        table.add_column("Cluster", style="cyan")
                        table.add_column("Notes", style="green")
                        table.add_column("Confidence", style="yellow")
                        table.add_column("Type", style="magenta")
                        
                        for i, cluster in enumerate(duplicate_clusters[:10]):  # Show first 10
                            table.add_row(
                                f"#{i+1}",
                                str(len(cluster.note_ids)),
                                f"{cluster.confidence_score:.2f}",
                                cluster.cluster_type
                            )
                        
                        console.print(table)
                        
                        if interactive and not dry_run:
                            for cluster in duplicate_clusters[:5]:  # Limit for safety
                                console.print(f"\n[bold]Cluster #{cluster.cluster_id}[/bold]")
                                for note_id in cluster.note_ids:
                                    console.print(f"  - {note_id}")
                                
                                if remove_duplicates:
                                    action = Prompt.ask(
                                        "Action", 
                                        choices=["merge", "skip", "stop"], 
                                        default="skip"
                                    )
                                    if action == "merge":
                                        console.print("[yellow]Merge functionality not yet implemented[/yellow]")
                                        curation_results['duplicates_processed'] += 1
                                    elif action == "stop":
                                        break
                        
                        elif remove_duplicates and not dry_run:
                            console.print("[yellow]Auto-merge not implemented yet - use --interactive for manual review[/yellow]")
                    
                    else:
                        progress.update(task, description="No duplicates found")
                
                # Quality analysis and improvements
                if quality:
                    task = progress.add_task("Analyzing content quality...", total=None)
                    
                    session = librarian._get_session(session_id)
                    all_notes = await session.vault.get_all_note_ids()
                    
                    progress.update(task, total=len(all_notes))
                    
                    quality_issues = []
                    for i, note_id in enumerate(all_notes):
                        try:
                            analysis = await session.analysis_service.analyze_note(note_id)
                            if analysis and analysis.recommendations:
                                quality_issues.append({
                                    'note_id': note_id,
                                    'score': analysis.quality_score,
                                    'recommendations': analysis.recommendations
                                })
                                curation_results['quality_improvements'] += 1
                        except Exception as e:
                            curation_results['errors'].append(f"Quality analysis failed for {note_id}: {str(e)}")
                        
                        progress.update(task, advance=1, description=f"Analyzed {i+1}/{len(all_notes)} notes")
                    
                    if quality_issues:
                        # Show quality issues
                        table = Table(title="Quality Improvement Opportunities")
                        table.add_column("Note", style="cyan")
                        table.add_column("Score", style="green")
                        table.add_column("Issues", style="yellow")
                        
                        for issue in quality_issues[:15]:  # Show first 15
                            recommendations = ", ".join(issue['recommendations'][:3])  # First 3 recommendations
                            table.add_row(
                                issue['note_id'],
                                f"{issue['score']:.2f}",
                                recommendations
                            )
                        
                        console.print(table)
                        
                        if interactive and not dry_run:
                            for issue in quality_issues[:10]:
                                console.print(f"\n[bold]{issue['note_id']}[/bold] (Score: {issue['score']:.2f})")
                                for rec in issue['recommendations']:
                                    console.print(f"  • {rec}")
                                
                                if Confirm.ask("Apply improvements?", default=False):
                                    console.print("[yellow]Auto-improvement not yet implemented[/yellow]")
                
                # Structure and templates
                if structure or templates:
                    task = progress.add_task("Improving structure and applying templates...", total=None)
                    
                    if templates:
                        template_results = await librarian.apply_templates(session_id, auto_detect=True)
                        curation_results['templates_applied'] = template_results.get('successful', 0)
                    
                    if structure:
                        # Run structure curation
                        results = await librarian.curate_content(
                            session_id,
                            remove_duplicates=False,
                            improve_quality=False,
                            organize_structure=True,
                        )
                        curation_results['structure_improvements'] = results.get('structure_improvements', 0)
                    
                    progress.update(task, description="Structure improvements complete")
            
            # Display final results
            console.print("\n")
            results_table = Table(title="Curation Results")
            results_table.add_column("Operation", style="cyan")
            results_table.add_column("Count", style="green")
            results_table.add_column("Status", style="yellow")
            
            results_table.add_row("Duplicates Found", str(curation_results['duplicates_found']), "✓")
            results_table.add_row("Duplicates Processed", str(curation_results['duplicates_processed']), "✓" if not dry_run else "Preview")
            results_table.add_row("Quality Issues Found", str(curation_results['quality_improvements']), "✓")
            results_table.add_row("Templates Applied", str(curation_results['templates_applied']), "✓" if not dry_run else "Preview")
            results_table.add_row("Structure Improvements", str(curation_results['structure_improvements']), "✓" if not dry_run else "Preview")
            
            if curation_results['errors']:
                results_table.add_row("Errors", str(len(curation_results['errors'])), "⚠️")
            
            console.print(results_table)
            
            if dry_run:
                console.print("\n[blue]This was a dry run - no changes were made[/blue]")
            
            if curation_results['errors']:
                console.print("\n[yellow]Errors encountered:[/yellow]")
                for error in curation_results['errors'][:5]:  # Show first 5 errors
                    console.print(f"  • {error}")
            
            await librarian.close()
            
        except Exception as e:
            logger.error("Curation failed", error=str(e))
            console.print(f"[red]Curation failed: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_curation())


@app.command()
def config(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    set_key: Optional[str] = typer.Option(None, "--set", help="Set configuration key=value"),
    get_key: Optional[str] = typer.Option(None, "--get", help="Get configuration value"),
    edit: bool = typer.Option(False, "--edit", "-e", help="Edit configuration file"),
) -> None:
    """Manage Obsidian Librarian configuration."""
    config = load_config()
    
    if show:
        console.print("[bold]Current Configuration:[/bold]")
        console.print(json.dumps(config, indent=2))
        return
    
    if get_key:
        keys = get_key.split(".")
        value = config
        for key in keys:
            value = value.get(key, {})
        console.print(f"{get_key} = {value}")
        return
    
    if set_key:
        if "=" not in set_key:
            console.print("[red]Format: --set key=value[/red]")
            return
        
        key, value = set_key.split("=", 1)
        keys = key.split(".")
        
        # Navigate to the right level
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        try:
            # Try to parse as JSON first
            current[keys[-1]] = json.loads(value)
        except json.JSONDecodeError:
            # Otherwise treat as string
            current[keys[-1]] = value
        
        save_config(config)
        console.print(f"[green]Set {key} = {value}[/green]")
        return
    
    if edit:
        # Open in default editor
        import subprocess
        import os
        
        editor = os.environ.get("EDITOR", "vi")
        subprocess.call([editor, str(CONFIG_FILE)])
        return
    
    # Default: show help
    console.print("Use --show to display config, --set to modify, --get to retrieve values")


@app.callback()
def main(
    debug: bool = typer.Option(False, "--debug", "-D", help="Enable debug logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output except errors"),
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
) -> None:
    """
    Obsidian Librarian - Intelligent vault management and research assistant.
    
    A comprehensive tool for managing Obsidian vaults with AI-powered features:
    
    • Research and content discovery
    
    • Duplicate detection and removal
    
    • Template management and application
    
    • Vault analysis and health monitoring
    
    • Git-based backup and restore
    
    • Content organization and curation
    """
    if version:
        console.print("Obsidian Librarian v0.1.0")
        raise typer.Exit(0)
    
    # Setup logging
    setup_logging(debug=debug, quiet=quiet)


if __name__ == "__main__":
    app(prog_name="obsidian-librarian")