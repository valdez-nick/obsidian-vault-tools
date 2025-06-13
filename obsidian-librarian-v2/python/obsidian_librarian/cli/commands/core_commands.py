"""
Core CLI commands for Obsidian Librarian using Typer.

This module provides the basic commands for vault management,
analysis, research, and organization.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.prompt import Confirm
from rich import print as rprint

from ...librarian import ObsidianLibrarian
from ...vault import Vault, scan_vault_async
from ...models import LibrarianConfig

# Initialize console
console = Console()


def init_command(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    name: str = typer.Option(None, "--name", "-n", help="Vault name"),
    force: bool = typer.Option(False, "--force", "-f", help="Force initialization"),
) -> None:
    """Initialize a new Obsidian vault with Librarian support."""
    console.print(Panel.fit(
        f"[bold green]Initializing Obsidian Vault[/bold green]\n"
        f"Path: {vault_path}",
        border_style="green"
    ))
    
    # Check if path exists
    if vault_path.exists() and not force:
        if not Confirm.ask(f"[yellow]Path {vault_path} already exists. Continue?[/yellow]"):
            console.print("[red]Initialization cancelled[/red]")
            raise typer.Exit(0)
    
    try:
        # Create vault directory
        vault_path.mkdir(parents=True, exist_ok=True)
        
        # Create standard Obsidian structure
        (vault_path / ".obsidian").mkdir(exist_ok=True)
        (vault_path / "Templates").mkdir(exist_ok=True)
        (vault_path / "Daily Notes").mkdir(exist_ok=True)
        (vault_path / "Archive").mkdir(exist_ok=True)
        
        # Create README
        readme_content = f"""# {name or vault_path.name}

Welcome to your Obsidian vault managed by Obsidian Librarian!

## Getting Started

- Use `obsidian-librarian stats {vault_path}` to see vault statistics
- Use `obsidian-librarian tags analyze {vault_path}` to analyze tags
- Use `obsidian-librarian research {vault_path} "your query"` to research topics

## Vault Structure

- `/Templates` - Note templates
- `/Daily Notes` - Daily journal entries
- `/Archive` - Archived notes

Created with Obsidian Librarian v0.1.0
"""
        (vault_path / "README.md").write_text(readme_content)
        
        console.print(f"[green]✓ Vault initialized successfully at {vault_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error initializing vault: {e}[/red]")
        raise typer.Exit(1)


def stats_command(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed statistics"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save stats to file"),
    format: str = typer.Option("table", "--format", help="Output format: table, json, markdown"),
) -> None:
    """Show vault statistics and insights."""
    async def run_stats():
        console.print(Panel.fit(
            f"[bold blue]Vault Statistics[/bold blue]\n"
            f"Path: {vault_path}",
            border_style="blue"
        ))
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Analyzing vault...", total=None)
                
                # Scan vault
                vault_info = await scan_vault_async(vault_path)
                
                # Get detailed stats if requested
                if detailed:
                    from ...librarian import analyze_vault_quick
                    vault_stats = await analyze_vault_quick(vault_path)
                else:
                    vault_stats = vault_info
                
                progress.update(task, description="Analysis complete!")
            
            # Display results based on format
            if format == "json":
                if output:
                    output.write_text(json.dumps(vault_stats, indent=2, default=str))
                    console.print(f"[green]✓ Stats saved to {output}[/green]")
                else:
                    console.print_json(data=vault_stats)
            else:
                # Create stats table
                table = Table(title="Vault Statistics")
                table.add_column("Metric", style="cyan", no_wrap=True)
                table.add_column("Value", style="magenta")
                
                # Basic stats
                table.add_row("Location", str(vault_path))
                table.add_row("Total Notes", str(vault_info.get('note_count', 0)))
                table.add_row("Total Size", f"{vault_info.get('total_size', 0) / 1024 / 1024:.1f} MB")
                
                if detailed and 'vault_stats' in vault_stats:
                    stats = vault_stats['vault_stats']
                    table.add_row("Total Words", f"{stats.get('total_words', 0):,}")
                    table.add_row("Total Links", str(stats.get('total_links', 0)))
                    table.add_row("Total Tags", str(stats.get('total_tags', 0)))
                    table.add_row("Avg. Note Length", f"{stats.get('avg_note_length', 0):.0f} words")
                
                console.print(table)
                
                # Save to file if requested
                if output and format == "markdown":
                    md_content = f"# Vault Statistics\n\n"
                    md_content += f"**Location:** {vault_path}\n"
                    md_content += f"**Total Notes:** {vault_info.get('note_count', 0)}\n"
                    md_content += f"**Total Size:** {vault_info.get('total_size', 0) / 1024 / 1024:.1f} MB\n"
                    output.write_text(md_content)
                    console.print(f"[green]✓ Stats saved to {output}[/green]")
                    
        except Exception as e:
            console.print(f"[red]Error analyzing vault: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_stats())


def organize_command(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    strategy: str = typer.Option("auto", "--strategy", "-s", help="Organization strategy: auto, date, topic, type"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without applying"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
) -> None:
    """Organize vault structure intelligently."""
    async def run_organize():
        console.print(Panel.fit(
            f"[bold green]Organizing Vault[/bold green]\n"
            f"Path: {vault_path}\n"
            f"Strategy: {strategy}",
            border_style="green"
        ))
        
        if dry_run:
            console.print("[yellow]Running in dry-run mode - no changes will be made[/yellow]")
        
        try:
            # Get app state config
            from ..main_typer import state
            config = state.get("config", LibrarianConfig())
            
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(vault_path)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Analyzing vault structure...", total=100)
                    
                    # Analyze current structure
                    progress.update(task, advance=20, description="Analyzing current structure...")
                    current_structure = await librarian.analyze_structure(session_id)
                    
                    # Generate organization plan
                    progress.update(task, advance=30, description="Creating organization plan...")
                    org_plan = await librarian.create_organization_plan(
                        session_id, 
                        strategy=strategy
                    )
                    
                    # Show plan
                    progress.update(task, advance=20, description="Preparing changes...")
                    
                    if interactive or dry_run:
                        console.print("\n[bold cyan]Proposed Changes:[/bold cyan]")
                        for change in org_plan.get('changes', [])[:10]:
                            console.print(f"  • Move [yellow]{change['from']}[/yellow] → [green]{change['to']}[/green]")
                        
                        if len(org_plan.get('changes', [])) > 10:
                            console.print(f"  ... and {len(org_plan['changes']) - 10} more changes")
                        
                        if interactive and not dry_run:
                            if not Confirm.ask("\n[yellow]Apply these changes?[/yellow]"):
                                console.print("[red]Organization cancelled[/red]")
                                raise typer.Exit(0)
                    
                    # Apply changes
                    if not dry_run:
                        progress.update(task, advance=30, description="Applying changes...")
                        results = await librarian.apply_organization_plan(session_id, org_plan)
                        
                        # Show results
                        console.print(f"\n[green]✓ Organization complete![/green]")
                        console.print(f"  Files moved: {results.get('files_moved', 0)}")
                        console.print(f"  Directories created: {results.get('dirs_created', 0)}")
                        console.print(f"  Errors: {len(results.get('errors', []))}")
                    else:
                        progress.update(task, advance=30, description="Dry run complete!")
                        
        except Exception as e:
            console.print(f"[red]Error organizing vault: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_organize())


def research_command(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    query: str = typer.Argument(..., help="Research query"),
    sources: Optional[List[str]] = typer.Option(None, "--source", "-s", help="Specific sources to search"),
    max_results: int = typer.Option(20, "--max-results", "-n", help="Maximum results"),
    organize: bool = typer.Option(True, "--organize/--no-organize", help="Organize results in vault"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to file"),
) -> None:
    """Perform intelligent research and save results to vault."""
    async def run_research():
        console.print(Panel.fit(
            f"[bold cyan]Research Query[/bold cyan]\n"
            f"Query: {query}\n"
            f"Vault: {vault_path}",
            border_style="cyan"
        ))
        
        try:
            from ..main_typer import state
            config = state.get("config", LibrarianConfig())
            
            results = []
            
            async with ObsidianLibrarian(config) as librarian:
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
                            results.append(result['data'])
                            progress.update(task, description=f"Found {len(results)} results...")
                        elif result.get('type') == 'complete':
                            progress.update(task, description="Research complete!")
                            break
                        elif result.get('type') == 'error':
                            console.print(f"[red]Error: {result['error']}[/red]")
                            break
            
            # Display results
            console.print(f"\n[bold green]Found {len(results)} results[/bold green]\n")
            
            for i, result in enumerate(results[:5], 1):
                console.print(f"[bold cyan]{i}. {result['title']}[/bold cyan]")
                console.print(f"   [blue]Source:[/blue] {result['source']}")
                console.print(f"   [blue]Relevance:[/blue] {result.get('relevance_score', 0):.2f}")
                if result.get('summary'):
                    summary = result['summary'][:150] + "..." if len(result['summary']) > 150 else result['summary']
                    console.print(f"   [dim]{summary}[/dim]")
                console.print()
            
            if len(results) > 5:
                console.print(f"[dim]... and {len(results) - 5} more results[/dim]")
            
            # Save results if requested
            if output:
                output_data = {
                    'query': query,
                    'results': results,
                    'total': len(results)
                }
                output.write_text(json.dumps(output_data, indent=2, default=str))
                console.print(f"\n[green]✓ Results saved to {output}[/green]")
                
        except Exception as e:
            console.print(f"[red]Error during research: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_research())


def analyze_command(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    duplicates: bool = typer.Option(True, "--duplicates/--no-duplicates", help="Find duplicate notes"),
    quality: bool = typer.Option(True, "--quality/--no-quality", help="Analyze note quality"),
    links: bool = typer.Option(True, "--links/--no-links", help="Analyze link structure"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save analysis to file"),
) -> None:
    """Analyze vault for insights, issues, and improvements."""
    async def run_analyze():
        console.print(Panel.fit(
            f"[bold yellow]Vault Analysis[/bold yellow]\n"
            f"Path: {vault_path}",
            border_style="yellow"
        ))
        
        try:
            from ..main_typer import state
            config = state.get("config", LibrarianConfig())
            
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(vault_path)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Analyzing vault...", total=100)
                    
                    analysis_results = {}
                    
                    # Analyze duplicates
                    if duplicates:
                        progress.update(task, advance=33, description="Finding duplicates...")
                        dup_results = await librarian.find_duplicates(session_id)
                        analysis_results['duplicates'] = dup_results
                    
                    # Analyze quality
                    if quality:
                        progress.update(task, advance=33, description="Analyzing quality...")
                        quality_results = await librarian.analyze_quality(session_id)
                        analysis_results['quality'] = quality_results
                    
                    # Analyze links
                    if links:
                        progress.update(task, advance=34, description="Analyzing links...")
                        link_results = await librarian.analyze_links(session_id)
                        analysis_results['links'] = link_results
                    
                    progress.update(task, description="Analysis complete!")
            
            # Display results
            console.print("\n[bold green]Analysis Results[/bold green]\n")
            
            # Duplicates
            if duplicates and 'duplicates' in analysis_results:
                dup_data = analysis_results['duplicates']
                console.print(f"[bold cyan]Duplicate Analysis:[/bold cyan]")
                console.print(f"  Duplicate clusters: {dup_data.get('cluster_count', 0)}")
                console.print(f"  Total duplicates: {dup_data.get('total_duplicates', 0)}")
                console.print()
            
            # Quality
            if quality and 'quality' in analysis_results:
                quality_data = analysis_results['quality']
                console.print(f"[bold cyan]Quality Analysis:[/bold cyan]")
                console.print(f"  Average quality score: {quality_data.get('avg_score', 0):.2f}/10")
                console.print(f"  Notes needing improvement: {quality_data.get('low_quality_count', 0)}")
                console.print()
            
            # Links
            if links and 'links' in analysis_results:
                link_data = analysis_results['links']
                console.print(f"[bold cyan]Link Analysis:[/bold cyan]")
                console.print(f"  Total links: {link_data.get('total_links', 0)}")
                console.print(f"  Broken links: {link_data.get('broken_links', 0)}")
                console.print(f"  Orphaned notes: {link_data.get('orphaned_notes', 0)}")
            
            # Save results
            if output:
                output.write_text(json.dumps(analysis_results, indent=2, default=str))
                console.print(f"\n[green]✓ Analysis saved to {output}[/green]")
                
        except Exception as e:
            console.print(f"[red]Error analyzing vault: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_analyze())


def curate_command(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    auto_fix: bool = typer.Option(False, "--auto-fix", help="Automatically fix issues"),
    merge_duplicates: bool = typer.Option(False, "--merge-duplicates", help="Merge duplicate notes"),
    fix_links: bool = typer.Option(False, "--fix-links", help="Fix broken links"),
    improve_quality: bool = typer.Option(False, "--improve-quality", help="Improve low-quality notes"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without applying"),
) -> None:
    """Intelligently curate and improve vault content."""
    async def run_curate():
        console.print(Panel.fit(
            f"[bold magenta]Vault Curation[/bold magenta]\n"
            f"Path: {vault_path}",
            border_style="magenta"
        ))
        
        if dry_run:
            console.print("[yellow]Running in dry-run mode - no changes will be made[/yellow]")
        
        if not any([auto_fix, merge_duplicates, fix_links, improve_quality]):
            console.print("[yellow]No curation actions selected. Use --help for options.[/yellow]")
            raise typer.Exit(0)
        
        try:
            from ..main_typer import state
            config = state.get("config", LibrarianConfig())
            
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(vault_path)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Curating vault...", total=None)
                    
                    results = await librarian.curate_content(
                        session_id,
                        remove_duplicates=merge_duplicates,
                        fix_broken_links=fix_links,
                        improve_quality=improve_quality,
                        auto_fix=auto_fix,
                        dry_run=dry_run
                    )
                    
                    progress.update(task, description="Curation complete!")
            
            # Display results
            console.print("\n[bold green]Curation Results[/bold green]\n")
            
            table = Table()
            table.add_column("Operation", style="cyan")
            table.add_column("Count", style="magenta")
            
            if merge_duplicates:
                table.add_row("Duplicates Merged", str(results.get('duplicates_merged', 0)))
            if fix_links:
                table.add_row("Links Fixed", str(results.get('links_fixed', 0)))
            if improve_quality:
                table.add_row("Notes Improved", str(results.get('notes_improved', 0)))
            
            table.add_row("Total Changes", str(results.get('total_changes', 0)))
            table.add_row("Errors", str(len(results.get('errors', []))))
            
            console.print(table)
            
            if results.get('errors'):
                console.print("\n[red]Errors encountered:[/red]")
                for error in results['errors'][:5]:
                    console.print(f"  • {error}")
                    
        except Exception as e:
            console.print(f"[red]Error curating vault: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_curate())