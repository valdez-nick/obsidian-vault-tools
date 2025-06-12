"""
Command-line interface for Obsidian Librarian.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List

import click
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from .librarian import ObsidianLibrarian, analyze_vault_quick, research_quick
from .models import LibrarianConfig

# Setup console and logging
console = Console()
logger = structlog.get_logger(__name__)


def setup_logging(verbose: bool = False):
    """Setup structured logging."""
    level = "DEBUG" if verbose else "INFO"
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
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose: bool, config: Optional[str]):
    """Obsidian Librarian - Intelligent vault management and research assistant."""
    setup_logging(verbose)
    
    # Load configuration
    if config:
        config_path = Path(config)
        # In practice, you'd load from JSON/YAML
        ctx.obj = LibrarianConfig()
    else:
        ctx.obj = LibrarianConfig()
    
    console.print(Panel.fit(
        "[bold blue]Obsidian Librarian v0.1.0[/bold blue]\n"
        "Intelligent vault management and research assistant",
        border_style="blue"
    ))


@cli.command()
@click.argument('vault_path', type=click.Path(exists=True, file_okay=False))
@click.option('--duplicates/--no-duplicates', default=True, help='Find duplicate notes')
@click.option('--quality/--no-quality', default=True, help='Perform quality analysis')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.pass_obj
def analyze(config: LibrarianConfig, vault_path: str, duplicates: bool, quality: bool, output: Optional[str]):
    """Analyze an Obsidian vault for insights and issues."""
    vault_path = Path(vault_path)
    
    console.print(f"[bold green]Analyzing vault:[/bold green] {vault_path}")
    
    async def run_analysis():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                task = progress.add_task("Analyzing vault...", total=None)
                
                # Run quick analysis
                results = await analyze_vault_quick(vault_path)
                
                progress.update(task, description="Analysis complete!")
            
            # Display results
            console.print("\n[bold green]Analysis Results[/bold green]")
            
            if 'vault_stats' in results:
                stats = results['vault_stats']
                
                table = Table(title="Vault Statistics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")
                
                table.add_row("Total Notes", str(stats.get('total_notes', 0)))
                table.add_row("Total Words", str(stats.get('total_words', 0)))
                table.add_row("Total Links", str(stats.get('total_links', 0)))
                table.add_row("Total Tasks", str(stats.get('total_tasks', 0)))
                table.add_row("Average Quality", f"{stats.get('avg_quality_score', 0):.2f}")
                
                console.print(table)
            
            if duplicates and 'duplicate_clusters' in results:
                cluster_count = results['duplicate_clusters']
                if cluster_count > 0:
                    console.print(f"\n[yellow]Found {cluster_count} duplicate clusters[/yellow]")
                else:
                    console.print("\n[green]No duplicates found[/green]")
            
            # Save results if requested
            if output:
                output_path = Path(output)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                console.print(f"\n[blue]Results saved to {output_path}[/blue]")
                
        except Exception as e:
            console.print(f"[red]Analysis failed: {e}[/red]")
            logger.error("Analysis failed", error=str(e))
            sys.exit(1)
    
    asyncio.run(run_analysis())


@cli.command()
@click.argument('vault_path', type=click.Path(exists=True, file_okay=False))
@click.argument('query')
@click.option('--sources', '-s', multiple=True, help='Specific sources to search')
@click.option('--max-results', '-n', type=int, default=20, help='Maximum results to return')
@click.option('--organize/--no-organize', default=True, help='Organize results in vault')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.pass_obj
def research(config: LibrarianConfig, vault_path: str, query: str, sources: tuple, 
             max_results: int, organize: bool, output: Optional[str]):
    """Perform intelligent research and save results to vault."""
    vault_path = Path(vault_path)
    source_list = list(sources) if sources else None
    
    console.print(f"[bold green]Researching:[/bold green] {query}")
    console.print(f"[bold blue]Vault:[/bold blue] {vault_path}")
    
    if sources:
        console.print(f"[bold yellow]Sources:[/bold yellow] {', '.join(sources)}")
    
    async def run_research():
        try:
            results = []
            
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(vault_path)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    
                    task = progress.add_task("Searching...", total=None)
                    
                    async for result in librarian.research(
                        session_id, 
                        query, 
                        source_list, 
                        max_results, 
                        organize
                    ):
                        if result.get('type') == 'result':
                            results.append(result['data'])
                            progress.update(task, description=f"Found {len(results)} results...")
                        elif result.get('type') == 'complete':
                            progress.update(task, description="Research complete!")
                            break
                        elif result.get('type') == 'error':
                            console.print(f"[red]Error: {result['error']}[/red]")
                            break
            
            # Display results
            console.print(f"\n[bold green]Found {len(results)} results[/bold green]")
            
            for i, result in enumerate(results[:10], 1):  # Show top 10
                console.print(f"\n[bold cyan]{i}. {result['title']}[/bold cyan]")
                console.print(f"[blue]Source:[/blue] {result['source']}")
                console.print(f"[blue]Quality:[/blue] {result['quality_score']:.2f}")
                console.print(f"[blue]URL:[/blue] {result['url']}")
                
                if result['summary']:
                    summary = result['summary'][:200] + "..." if len(result['summary']) > 200 else result['summary']
                    console.print(f"[dim]{summary}[/dim]")
            
            if len(results) > 10:
                console.print(f"\n[dim]... and {len(results) - 10} more results[/dim]")
            
            # Save results if requested
            if output:
                output_path = Path(output)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                console.print(f"\n[blue]Results saved to {output_path}[/blue]")
                
        except Exception as e:
            console.print(f"[red]Research failed: {e}[/red]")
            logger.error("Research failed", error=str(e))
            sys.exit(1)
    
    asyncio.run(run_research())


@cli.command()
@click.argument('vault_path', type=click.Path(exists=True, file_okay=False))
@click.option('--duplicates/--no-duplicates', default=False, help='Remove duplicate notes')
@click.option('--quality/--no-quality', default=True, help='Improve note quality')
@click.option('--structure/--no-structure', default=True, help='Organize note structure')
@click.option('--dry-run', is_flag=True, help='Show what would be done without making changes')
@click.pass_obj
def curate(config: LibrarianConfig, vault_path: str, duplicates: bool, quality: bool, 
           structure: bool, dry_run: bool):
    """Intelligently curate and organize vault content."""
    vault_path = Path(vault_path)
    
    console.print(f"[bold green]Curating vault:[/bold green] {vault_path}")
    
    if dry_run:
        console.print("[yellow]Running in dry-run mode - no changes will be made[/yellow]")
    
    async def run_curation():
        try:
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(vault_path)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    
                    task = progress.add_task("Curating content...", total=None)
                    
                    results = await librarian.curate_content(
                        session_id,
                        remove_duplicates=duplicates,
                        improve_quality=quality,
                        organize_structure=structure,
                    )
                    
                    progress.update(task, description="Curation complete!")
                
                # Display results
                console.print("\n[bold green]Curation Results[/bold green]")
                
                table = Table()
                table.add_column("Operation", style="cyan")
                table.add_column("Count", style="magenta")
                
                table.add_row("Duplicates Processed", str(results['duplicates_processed']))
                table.add_row("Quality Improvements", str(results['quality_improvements']))
                table.add_row("Structure Improvements", str(results['structure_improvements']))
                table.add_row("Errors", str(len(results['errors'])))
                
                console.print(table)
                
                if results['errors']:
                    console.print("\n[red]Errors encountered:[/red]")
                    for error in results['errors'][:5]:  # Show first 5 errors
                        console.print(f"  • {error}")
                
        except Exception as e:
            console.print(f"[red]Curation failed: {e}[/red]")
            logger.error("Curation failed", error=str(e))
            sys.exit(1)
    
    asyncio.run(run_curation())


@cli.command()
@click.argument('vault_path', type=click.Path(exists=True, file_okay=False))
@click.option('--auto/--manual', default=True, help='Auto-detect templates vs manual suggestions')
@click.option('--notes', '-n', multiple=True, help='Specific notes to process')
@click.pass_obj
def templates(config: LibrarianConfig, vault_path: str, auto: bool, notes: tuple):
    """Apply templates to notes intelligently."""
    vault_path = Path(vault_path)
    note_list = list(notes) if notes else None
    
    console.print(f"[bold green]Applying templates:[/bold green] {vault_path}")
    
    async def run_templates():
        try:
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(vault_path)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    
                    task = progress.add_task("Applying templates...", total=None)
                    
                    results = await librarian.apply_templates(
                        session_id,
                        note_ids=note_list,
                        auto_detect=auto,
                    )
                    
                    progress.update(task, description="Template application complete!")
                
                # Display results
                console.print("\n[bold green]Template Application Results[/bold green]")
                
                table = Table()
                table.add_column("Metric", style="cyan")
                table.add_column("Count", style="magenta")
                
                table.add_row("Total Applications", str(results['total_applications']))
                table.add_row("Successful", str(results['successful']))
                table.add_row("Failed", str(results['failed']))
                
                console.print(table)
                
                # Show detailed results
                if results.get('applications'):
                    console.print("\n[bold blue]Application Details[/bold blue]")
                    
                    for app in results['applications'][:10]:  # Show first 10
                        status = "[green]✓[/green]" if app['success'] else "[red]✗[/red]"
                        console.print(f"{status} {app['note_id']} → {app['template']}")
                        
                        if not app['success'] and app['error']:
                            console.print(f"    [red]Error: {app['error']}[/red]")
                
        except Exception as e:
            console.print(f"[red]Template application failed: {e}[/red]")
            logger.error("Template application failed", error=str(e))
            sys.exit(1)
    
    asyncio.run(run_templates())


@cli.command()
@click.argument('vault_path', type=click.Path(exists=True, file_okay=False))
@click.pass_obj
def status(config: LibrarianConfig, vault_path: str):
    """Show vault status and statistics."""
    vault_path = Path(vault_path)
    
    async def show_status():
        try:
            # Quick vault scan
            from .vault import scan_vault_async
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                task = progress.add_task("Scanning vault...", total=None)
                vault_info = await scan_vault_async(vault_path)
                progress.update(task, description="Scan complete!")
            
            # Display vault information
            console.print(f"\n[bold green]Vault Status: {vault_path}[/bold green]")
            
            table = Table(title="Vault Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Path", str(vault_info['path']))
            table.add_row("Exists", "✓" if vault_info['exists'] else "✗")
            table.add_row("Note Count", str(vault_info['note_count']))
            table.add_row("Total Size", f"{vault_info['total_size'] / 1024:.1f} KB")
            
            if vault_info['last_modified']:
                table.add_row("Last Modified", vault_info['last_modified'].strftime('%Y-%m-%d %H:%M:%S'))
            
            console.print(table)
            
            # Check for .obsidian directory
            obsidian_dir = vault_path / '.obsidian'
            if obsidian_dir.exists():
                console.print("\n[green]✓ Valid Obsidian vault[/green]")
            else:
                console.print("\n[yellow]⚠ Not an Obsidian vault (missing .obsidian directory)[/yellow]")
            
            # Check for templates
            template_dirs = [vault_path / 'Templates', vault_path / 'templates']
            for template_dir in template_dirs:
                if template_dir.exists():
                    template_count = len(list(template_dir.glob('*.md')))
                    console.print(f"[blue]Templates found: {template_count}[/blue]")
                    break
            else:
                console.print("[dim]No template directory found[/dim]")
                
        except Exception as e:
            console.print(f"[red]Status check failed: {e}[/red]")
            logger.error("Status check failed", error=str(e))
            sys.exit(1)
    
    asyncio.run(show_status())


@cli.command()
@click.pass_obj
def interactive(config: LibrarianConfig):
    """Start interactive mode for exploratory analysis."""
    console.print("[bold blue]Interactive Mode[/bold blue]")
    console.print("Type 'help' for available commands, 'quit' to exit")
    
    # This would implement an interactive REPL
    # For now, just show a placeholder
    console.print("\n[yellow]Interactive mode not yet implemented[/yellow]")
    console.print("Use the individual commands for now:")
    console.print("  • analyze - Analyze a vault")
    console.print("  • research - Perform research")
    console.print("  • curate - Curate content")
    console.print("  • templates - Apply templates")
    console.print("  • status - Show vault status")


if __name__ == '__main__':
    cli()