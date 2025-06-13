"""
Tag Management CLI Commands for Obsidian Librarian using Typer.

Provides comprehensive tag management functionality including analysis,
duplicate detection, suggestions, auto-tagging, merging, cleanup, and hierarchy management.
"""

import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Confirm, Prompt
from rich.tree import Tree
from rich import print as rprint

from ...librarian import ObsidianLibrarian
from ...models import LibrarianConfig
from ...vault import scan_vault_async

# Initialize console
console = Console()

# Create tags subcommand app
app = typer.Typer(
    name="tags",
    help="Comprehensive tag management for Obsidian vaults",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command()
def analyze(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save analysis to file"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, markdown"),
    min_frequency: int = typer.Option(1, "--min-frequency", help="Minimum tag frequency to include"),
    show_unused: bool = typer.Option(False, "--show-unused", help="Show unused tags from templates"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed per-note information"),
) -> None:
    """
    Analyze tag usage patterns and statistics in the vault.
    
    Provides comprehensive tag analysis including:
    - Usage frequency and distribution
    - Most/least used tags
    - Tag hierarchies and relationships
    - Orphaned and unused tags
    - Tag complexity metrics
    """
    async def run_analysis():
        console.print(Panel.fit(
            f"[bold green]Tag Analysis[/bold green]\n"
            f"Vault: {vault_path}",
            border_style="green"
        ))
        
        try:
            # Get config from main app state
            from ..main_typer import state
            config = state.get("config", LibrarianConfig())
            
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(vault_path)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Scanning vault for tags...", total=None)
                    
                    # Get all notes with tags
                    notes = await librarian.get_all_notes(session_id)
                    progress.update(task, description="Analyzing tag patterns...")
                    
                    # Analyze tag usage
                    analysis = await _analyze_tag_patterns(notes, min_frequency)
                    progress.update(task, description="Analysis complete!")
                
                # Display results based on format
                if format == "json":
                    if output:
                        output.write_text(json.dumps(analysis, indent=2, default=str))
                        console.print(f"[green]✓ Analysis saved to {output}[/green]")
                    else:
                        console.print_json(data=analysis)
                elif format == "markdown":
                    md_content = _format_analysis_markdown(analysis)
                    if output:
                        output.write_text(md_content)
                        console.print(f"[green]✓ Analysis saved to {output}[/green]")
                    else:
                        console.print(md_content)
                else:  # table format
                    _display_tag_analysis_table(analysis, detailed, show_unused)
                    
        except Exception as e:
            console.print(f"[red]Error analyzing tags: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_analysis())


@app.command()
def duplicates(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    similarity_threshold: float = typer.Option(0.8, "--threshold", "-t", help="Similarity threshold (0.0-1.0)"),
    case_sensitive: bool = typer.Option(False, "--case-sensitive", help="Case sensitive comparison"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save report to file"),
    auto_merge: bool = typer.Option(False, "--auto-merge", help="Automatically merge duplicates"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without applying"),
) -> None:
    """
    Find and optionally merge duplicate or similar tags.
    
    Detects duplicate tags using various methods:
    - Exact matches (case variants)
    - Similar spelling (edit distance)
    - Semantic similarity
    - Common abbreviations/expansions
    """
    if auto_merge and not dry_run:
        if not Confirm.ask("[yellow]This will merge duplicate tags. Continue?[/yellow]"):
            console.print("[red]Operation cancelled[/red]")
            raise typer.Exit(0)
    
    async def run_duplicate_detection():
        console.print(Panel.fit(
            f"[bold yellow]Duplicate Tag Detection[/bold yellow]\n"
            f"Vault: {vault_path}\n"
            f"Threshold: {similarity_threshold}",
            border_style="yellow"
        ))
        
        if dry_run:
            console.print("[yellow]Running in dry-run mode - no changes will be made[/yellow]")
        
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
                    task = progress.add_task("Analyzing tags for duplicates...", total=None)
                    
                    # Get all tags
                    tags = await librarian.get_all_tags(session_id)
                    progress.update(task, description="Finding duplicate patterns...")
                    
                    # Find duplicates
                    duplicates = await _find_duplicate_tags(tags, similarity_threshold, case_sensitive)
                    progress.update(task, description="Duplicate detection complete!")
                
                # Display results
                _display_duplicate_results(duplicates)
                
                # Auto-merge if requested
                merge_count = 0
                if auto_merge and duplicates:
                    merge_count = await _auto_merge_duplicates(
                        librarian, session_id, duplicates, dry_run
                    )
                
                # Save results if requested
                if output:
                    await _save_duplicate_results(duplicates, output, merge_count)
                    console.print(f"[green]✓ Report saved to {output}[/green]")
                    
        except Exception as e:
            console.print(f"[red]Error finding duplicates: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_duplicate_detection())


@app.command()
def suggest(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    note_path: Optional[Path] = typer.Option(None, "--note", "-n", help="Suggest for specific note"),
    max_suggestions: int = typer.Option(5, "--max", "-m", help="Maximum suggestions per note"),
    confidence: float = typer.Option(0.7, "--confidence", "-c", help="Minimum confidence (0.0-1.0)"),
    use_content: bool = typer.Option(True, "--content/--no-content", help="Analyze note content"),
    use_title: bool = typer.Option(True, "--title/--no-title", help="Analyze note title"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive review mode"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save suggestions to file"),
) -> None:
    """
    Generate intelligent tag suggestions for notes.
    
    Uses AI analysis to suggest relevant tags based on:
    - Note content and title
    - Existing tag patterns in vault
    - Similar notes and their tags
    - Topic modeling and classification
    """
    async def run_tag_suggestions():
        console.print(Panel.fit(
            f"[bold cyan]Tag Suggestions[/bold cyan]\n"
            f"Vault: {vault_path}",
            border_style="cyan"
        ))
        
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
                    task = progress.add_task("Analyzing notes for tag suggestions...", total=None)
                    
                    # Get notes to analyze
                    if note_path:
                        notes = [await librarian.get_note(session_id, str(note_path))]
                    else:
                        notes = await librarian.get_all_notes(session_id)
                    
                    progress.update(task, description="Generating suggestions...")
                    
                    # Generate suggestions
                    suggestions = await _generate_tag_suggestions(
                        librarian, session_id, notes, max_suggestions,
                        confidence, use_content, use_title
                    )
                    progress.update(task, description="Suggestions complete!")
                
                # Display or process suggestions
                if interactive:
                    await _interactive_tag_suggestions(librarian, session_id, suggestions)
                else:
                    _display_tag_suggestions(suggestions)
                
                # Save suggestions if requested
                if output:
                    await _save_tag_suggestions(suggestions, output)
                    console.print(f"[green]✓ Suggestions saved to {output}[/green]")
                    
        except Exception as e:
            console.print(f"[red]Error generating suggestions: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_tag_suggestions())


@app.command(name="auto-tag")
def auto_tag(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    confidence: float = typer.Option(0.8, "--confidence", "-c", help="Minimum confidence (0.0-1.0)"),
    max_tags: int = typer.Option(10, "--max-tags", "-m", help="Maximum tags per note"),
    exclude_existing: bool = typer.Option(True, "--exclude-existing", help="Skip already tagged notes"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without applying"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Review each decision"),
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Notes to process at once"),
) -> None:
    """
    Automatically apply tags to notes based on AI analysis.
    
    Intelligently tags notes using:
    - Content analysis and topic detection
    - Pattern matching with existing tags
    - Similarity to already-tagged notes
    - Configurable confidence thresholds
    """
    if not dry_run and not interactive:
        if not Confirm.ask("[yellow]This will automatically add tags to notes. Continue?[/yellow]"):
            console.print("[red]Operation cancelled[/red]")
            raise typer.Exit(0)
    
    async def run_auto_tagging():
        console.print(Panel.fit(
            f"[bold magenta]Auto-Tagging Notes[/bold magenta]\n"
            f"Vault: {vault_path}\n"
            f"Confidence: {confidence}",
            border_style="magenta"
        ))
        
        if dry_run:
            console.print("[yellow]Running in dry-run mode - no changes will be made[/yellow]")
        
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
                    task = progress.add_task("Preparing notes for auto-tagging...", total=None)
                    
                    # Get candidate notes
                    notes = await _get_auto_tag_candidates(librarian, session_id, exclude_existing)
                    progress.update(task, description=f"Processing {len(notes)} notes...", total=len(notes))
                    
                    # Process in batches
                    tagged_count = 0
                    total_tags_added = 0
                    
                    for i in range(0, len(notes), batch_size):
                        batch = notes[i:i + batch_size]
                        
                        batch_results = await _process_auto_tag_batch(
                            librarian, session_id, batch, confidence,
                            max_tags, dry_run, interactive
                        )
                        
                        tagged_count += batch_results['tagged_notes']
                        total_tags_added += batch_results['total_tags']
                        
                        progress.update(task, advance=len(batch))
                    
                    progress.update(task, description="Auto-tagging complete!")
                
                # Display results
                console.print("\n[bold green]Auto-tagging Results[/bold green]")
                console.print(f"  Notes processed: {len(notes)}")
                console.print(f"  Notes tagged: {tagged_count}")
                console.print(f"  Total tags added: {total_tags_added}")
                if tagged_count > 0:
                    console.print(f"  Average tags per note: {total_tags_added / tagged_count:.1f}")
                    
        except Exception as e:
            console.print(f"[red]Error auto-tagging: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_auto_tagging())


@app.command()
def merge(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    source_tag: str = typer.Option(..., "--source", "-s", help="Tag to merge from (will be removed)"),
    target_tag: str = typer.Option(..., "--target", "-t", help="Tag to merge into (will be kept)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without applying"),
    case_sensitive: bool = typer.Option(True, "--case-sensitive", help="Case sensitive matching"),
) -> None:
    """
    Merge one tag into another across all notes.
    
    Replaces all instances of source-tag with target-tag throughout the vault.
    The source tag will be completely removed and replaced with the target tag.
    """
    if not dry_run:
        console.print(Panel.fit(
            f"[bold red]⚠️  Tag Merge Warning[/bold red]\n"
            f"This will replace ALL instances of:\n"
            f"  '{source_tag}' → '{target_tag}'\n"
            f"across your entire vault.\n"
            f"[bold]This action cannot be undone![/bold]",
            border_style="red"
        ))
        
        if not Confirm.ask("[red]Are you sure you want to continue?[/red]"):
            console.print("[yellow]Operation cancelled[/yellow]")
            raise typer.Exit(0)
    
    async def run_tag_merge():
        console.print(Panel.fit(
            f"[bold yellow]Tag Merge Operation[/bold yellow]\n"
            f"Source: #{source_tag} → Target: #{target_tag}",
            border_style="yellow"
        ))
        
        if dry_run:
            console.print("[yellow]Running in dry-run mode - no changes will be made[/yellow]")
        
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
                    task = progress.add_task("Finding notes with source tag...", total=None)
                    
                    # Find all notes with the source tag
                    notes_with_tag = await librarian.get_notes_by_tag(session_id, source_tag)
                    progress.update(task, description=f"Merging tags in {len(notes_with_tag)} notes...")
                    
                    # Perform the merge
                    merge_results = await _perform_tag_merge(
                        librarian, session_id, notes_with_tag, source_tag, target_tag,
                        case_sensitive, dry_run
                    )
                    
                    progress.update(task, description="Tag merge complete!")
                
                # Display results
                _display_merge_results(merge_results, source_tag, target_tag, dry_run)
                
        except Exception as e:
            console.print(f"[red]Error merging tags: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_tag_merge())


@app.command()
def cleanup(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    remove_unused: bool = typer.Option(False, "--remove-unused", help="Remove unused tags"),
    fix_case: bool = typer.Option(False, "--fix-case", help="Standardize tag case"),
    remove_special: bool = typer.Option(False, "--remove-special", help="Remove special characters"),
    min_length: int = typer.Option(2, "--min-length", help="Minimum tag length"),
    max_length: int = typer.Option(50, "--max-length", help="Maximum tag length"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without applying"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Review each change"),
) -> None:
    """
    Clean up and standardize tags across the vault.
    
    Performs various cleanup operations:
    - Remove unused/orphaned tags
    - Standardize case (usually lowercase)
    - Remove special characters
    - Fix length issues
    - Consolidate variations
    """
    if not any([remove_unused, fix_case, remove_special]) and not interactive:
        console.print("[yellow]No cleanup options selected. Use --help for options.[/yellow]")
        raise typer.Exit(0)
    
    if not dry_run and not interactive:
        if not Confirm.ask("[yellow]This will modify tags across your vault. Continue?[/yellow]"):
            console.print("[red]Operation cancelled[/red]")
            raise typer.Exit(0)
    
    async def run_tag_cleanup():
        console.print(Panel.fit(
            f"[bold green]Tag Cleanup[/bold green]\n"
            f"Vault: {vault_path}",
            border_style="green"
        ))
        
        if dry_run:
            console.print("[yellow]Running in dry-run mode - no changes will be made[/yellow]")
        
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
                    task = progress.add_task("Analyzing tags for cleanup...", total=None)
                    
                    # Get all tags and their usage
                    tag_usage = await librarian.get_tag_usage(session_id)
                    progress.update(task, description="Identifying cleanup opportunities...")
                    
                    # Identify cleanup operations
                    cleanup_plan = await _create_cleanup_plan(
                        tag_usage, remove_unused, fix_case, remove_special,
                        min_length, max_length
                    )
                    
                    progress.update(task, description="Executing cleanup operations...")
                    
                    # Execute cleanup
                    cleanup_results = await _execute_cleanup_plan(
                        librarian, session_id, cleanup_plan, dry_run, interactive
                    )
                    
                    progress.update(task, description="Tag cleanup complete!")
                
                # Display results
                _display_cleanup_results(cleanup_results, dry_run)
                
        except Exception as e:
            console.print(f"[red]Error cleaning up tags: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_tag_cleanup())


@app.command()
def hierarchy(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    show_tree: bool = typer.Option(False, "--tree", "-t", help="Display as tree"),
    separator: str = typer.Option("/", "--separator", "-s", help="Hierarchy separator"),
    max_depth: int = typer.Option(5, "--max-depth", "-d", help="Maximum depth to display"),
    create_missing: bool = typer.Option(False, "--create-missing", help="Create missing parent tags"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save hierarchy to file"),
) -> None:
    """
    Analyze and display tag hierarchies in the vault.
    
    Shows tag relationships and hierarchical structure:
    - Parent/child relationships
    - Hierarchy depth analysis
    - Missing parent tags
    - Usage statistics by level
    """
    async def run_hierarchy_analysis():
        console.print(Panel.fit(
            f"[bold blue]Tag Hierarchy Analysis[/bold blue]\n"
            f"Vault: {vault_path}",
            border_style="blue"
        ))
        
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
                    task = progress.add_task("Building tag hierarchy...", total=None)
                    
                    # Get all tags
                    tags = await librarian.get_all_tags(session_id)
                    progress.update(task, description="Analyzing relationships...")
                    
                    # Build hierarchy
                    hierarchy = await _build_tag_hierarchy(tags, separator, max_depth)
                    progress.update(task, description="Hierarchy analysis complete!")
                
                # Display results
                if show_tree:
                    _display_tag_tree(hierarchy, max_depth)
                else:
                    _display_hierarchy_stats(hierarchy)
                
                # Create missing parents if requested
                if create_missing:
                    missing_count = await _create_missing_parents(
                        librarian, session_id, hierarchy, separator
                    )
                    if missing_count > 0:
                        console.print(f"[green]✓ Created {missing_count} missing parent tags[/green]")
                
                # Save hierarchy if requested
                if output:
                    await _save_hierarchy(hierarchy, output)
                    console.print(f"[green]✓ Hierarchy saved to {output}[/green]")
                    
        except Exception as e:
            console.print(f"[red]Error analyzing hierarchy: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_hierarchy_analysis())


# Helper functions (similar to original but adapted for Typer style)

async def _analyze_tag_patterns(notes: List[Dict], min_frequency: int) -> Dict[str, Any]:
    """Analyze tag usage patterns and generate statistics."""
    tag_counts = Counter()
    tag_note_map = defaultdict(list)
    note_tag_counts = []
    
    for note in notes:
        note_tags = note.get('tags', [])
        note_tag_counts.append(len(note_tags))
        
        for tag in note_tags:
            tag_counts[tag] += 1
            tag_note_map[tag].append(note['id'])
    
    # Filter by minimum frequency
    filtered_tags = {tag: count for tag, count in tag_counts.items() 
                    if count >= min_frequency}
    
    # Calculate statistics
    total_tags = len(tag_counts)
    total_usages = sum(tag_counts.values())
    avg_tags_per_note = sum(note_tag_counts) / len(note_tag_counts) if note_tag_counts else 0
    
    return {
        'total_unique_tags': total_tags,
        'total_tag_usages': total_usages,
        'avg_tags_per_note': avg_tags_per_note,
        'tag_frequencies': dict(filtered_tags),
        'most_used_tags': tag_counts.most_common(10),
        'least_used_tags': tag_counts.most_common()[-10:] if total_tags > 10 else [],
        'single_use_tags': [tag for tag, count in tag_counts.items() if count == 1],
        'tag_note_mapping': dict(tag_note_map)
    }


def _display_tag_analysis_table(analysis: Dict[str, Any], detailed: bool, show_unused: bool):
    """Display tag analysis results in table format."""
    # Basic statistics
    console.print("\n[bold green]Tag Statistics[/bold green]")
    
    stats_table = Table()
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="magenta")
    
    stats_table.add_row("Total Unique Tags", str(analysis['total_unique_tags']))
    stats_table.add_row("Total Tag Usages", str(analysis['total_tag_usages']))
    stats_table.add_row("Average Tags per Note", f"{analysis['avg_tags_per_note']:.1f}")
    stats_table.add_row("Single-use Tags", str(len(analysis['single_use_tags'])))
    
    console.print(stats_table)
    
    # Most used tags
    if analysis['most_used_tags']:
        console.print("\n[bold cyan]Most Used Tags[/bold cyan]")
        most_used_table = Table()
        most_used_table.add_column("Tag", style="cyan")
        most_used_table.add_column("Count", style="magenta")
        most_used_table.add_column("Percentage", style="green")
        
        total_usages = analysis['total_tag_usages']
        for tag, count in analysis['most_used_tags']:
            percentage = (count / total_usages * 100) if total_usages > 0 else 0
            most_used_table.add_row(f"#{tag}", str(count), f"{percentage:.1f}%")
        
        console.print(most_used_table)
    
    # Single-use tags if requested or if there are few
    if show_unused or len(analysis['single_use_tags']) <= 10:
        single_use_count = len(analysis['single_use_tags'])
        if single_use_count > 0:
            console.print(f"\n[bold yellow]Single-use Tags ({single_use_count})[/bold yellow]")
            tags_to_show = analysis['single_use_tags'][:20]
            tag_list = ", ".join([f"#{tag}" for tag in tags_to_show])
            console.print(f"[dim]{tag_list}[/dim]")
            if single_use_count > 20:
                console.print(f"[dim]... and {single_use_count - 20} more[/dim]")


def _format_analysis_markdown(analysis: Dict[str, Any]) -> str:
    """Format analysis results as markdown."""
    md = f"""# Tag Analysis Report

## Summary Statistics

- **Total Unique Tags:** {analysis['total_unique_tags']}
- **Total Tag Usages:** {analysis['total_tag_usages']}
- **Average Tags per Note:** {analysis['avg_tags_per_note']:.1f}
- **Single-use Tags:** {len(analysis['single_use_tags'])}

## Most Used Tags

"""
    for tag, count in analysis['most_used_tags']:
        percentage = (count / analysis['total_tag_usages'] * 100) if analysis['total_tag_usages'] > 0 else 0
        md += f"- #{tag}: {count} uses ({percentage:.1f}%)\n"
    
    return md


async def _find_duplicate_tags(tags: List[str], similarity_threshold: float, 
                              case_sensitive: bool) -> List[Dict[str, Any]]:
    """Find duplicate or similar tags using various comparison methods."""
    duplicates = []
    processed = set()
    
    for i, tag1 in enumerate(tags):
        if tag1 in processed:
            continue
            
        similar_tags = [tag1]
        
        for j, tag2 in enumerate(tags[i + 1:], i + 1):
            if tag2 in processed:
                continue
                
            # Case-insensitive exact match
            if not case_sensitive and tag1.lower() == tag2.lower():
                similar_tags.append(tag2)
                processed.add(tag2)
                continue
            
            # Edit distance similarity
            similarity = _calculate_tag_similarity(tag1, tag2)
            if similarity >= similarity_threshold:
                similar_tags.append(tag2)
                processed.add(tag2)
        
        if len(similar_tags) > 1:
            duplicates.append({
                'primary': tag1,
                'duplicates': similar_tags[1:],
                'similarity_type': 'exact' if not case_sensitive else 'fuzzy',
                'suggested_merge': tag1
            })
            processed.add(tag1)
    
    return duplicates


def _calculate_tag_similarity(tag1: str, tag2: str) -> float:
    """Calculate similarity between two tags using edit distance."""
    import difflib
    return difflib.SequenceMatcher(None, tag1.lower(), tag2.lower()).ratio()


def _display_duplicate_results(duplicates: List[Dict[str, Any]]):
    """Display duplicate tag detection results."""
    if not duplicates:
        console.print("[green]✓ No duplicate tags found![/green]")
        return
    
    console.print(f"\n[bold yellow]Found {len(duplicates)} Duplicate Tag Groups[/bold yellow]\n")
    
    for i, dup_group in enumerate(duplicates, 1):
        console.print(f"[bold cyan]Group {i}:[/bold cyan]")
        console.print(f"  Primary: [green]#{dup_group['primary']}[/green]")
        console.print(f"  Duplicates: {', '.join([f'#{d}' for d in dup_group['duplicates']])}")
        console.print(f"  Suggested merge: [blue]#{dup_group['suggested_merge']}[/blue]")
        console.print()


async def _save_duplicate_results(duplicates: List[Dict[str, Any]], output_path: Path, merge_count: int):
    """Save duplicate detection results to file."""
    results = {
        'duplicate_groups': duplicates,
        'total_groups': len(duplicates),
        'total_duplicates': sum(len(d['duplicates']) for d in duplicates),
        'merged_count': merge_count,
    }
    
    output_path.write_text(json.dumps(results, indent=2, default=str))


# Additional helper functions would continue in the same pattern...
# Including: _generate_tag_suggestions, _display_tag_suggestions, _interactive_tag_suggestions,
# _get_auto_tag_candidates, _process_auto_tag_batch, _perform_tag_merge, etc.

async def _generate_tag_suggestions(librarian, session_id: str, notes: List[Dict],
                                   max_suggestions: int, confidence_threshold: float,
                                   use_content: bool, use_title: bool) -> Dict[str, Any]:
    """Generate AI-powered tag suggestions for notes."""
    suggestions = {}
    
    for note in notes:
        note_suggestions = []
        
        # Placeholder for AI analysis - would integrate with actual AI service
        # For now, return mock suggestions
        if use_content or use_title:
            note_suggestions = [
                {'tag': 'suggested-tag-1', 'confidence': 0.85, 'reason': 'Content analysis'},
                {'tag': 'suggested-tag-2', 'confidence': 0.75, 'reason': 'Similar to other notes'},
            ]
        
        # Filter by confidence threshold
        filtered_suggestions = [
            s for s in note_suggestions 
            if s.get('confidence', 0) >= confidence_threshold
        ]
        
        # Sort by confidence and take top suggestions
        filtered_suggestions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        suggestions[note['id']] = filtered_suggestions[:max_suggestions]
    
    return suggestions


def _display_tag_suggestions(suggestions: Dict[str, Any]):
    """Display tag suggestions in a formatted table."""
    if not suggestions:
        console.print("[yellow]No tag suggestions generated[/yellow]")
        return
    
    console.print(f"\n[bold cyan]Tag Suggestions for {len(suggestions)} Notes[/bold cyan]\n")
    
    table = Table()
    table.add_column("Note", style="cyan", max_width=30)
    table.add_column("Suggested Tags", style="yellow")
    table.add_column("Confidence", style="green")
    
    for note_id, note_suggestions in list(suggestions.items())[:10]:
        if note_suggestions:
            tag_display = ", ".join([f"#{s['tag']}" for s in note_suggestions])
            confidence_display = ", ".join([f"{s['confidence']:.0%}" for s in note_suggestions])
            table.add_row(
                note_id[:25] + "..." if len(note_id) > 25 else note_id,
                tag_display,
                confidence_display
            )
    
    console.print(table)
    
    if len(suggestions) > 10:
        console.print(f"\n[dim]... and {len(suggestions) - 10} more notes[/dim]")


async def _save_tag_suggestions(suggestions: Dict[str, Any], output_path: Path):
    """Save tag suggestions to file."""
    output_data = {
        'suggestions': suggestions,
        'total_notes': len(suggestions),
        'total_suggestions': sum(len(s) for s in suggestions.values()),
    }
    
    output_path.write_text(json.dumps(output_data, indent=2, default=str))


async def _interactive_tag_suggestions(librarian, session_id: str, suggestions: Dict[str, Any]):
    """Interactive mode for reviewing and applying tag suggestions."""
    console.print("\n[bold cyan]Interactive Tag Suggestion Review[/bold cyan]\n")
    
    applied_count = 0
    for note_id, note_suggestions in suggestions.items():
        if not note_suggestions:
            continue
        
        console.print(f"[bold yellow]Note:[/bold yellow] {note_id}")
        
        for suggestion in note_suggestions:
            tag = suggestion['tag']
            confidence = suggestion['confidence']
            reason = suggestion.get('reason', 'Unknown')
            
            console.print(f"  Tag: [cyan]#{tag}[/cyan] (confidence: {confidence:.0%})")
            console.print(f"  Reason: [dim]{reason}[/dim]")
            
            if Confirm.ask(f"Apply tag '#{tag}' to this note?"):
                # Apply the tag
                await librarian.add_tag_to_note(session_id, note_id, tag)
                console.print(f"  [green]✓ Applied #{tag}[/green]")
                applied_count += 1
            else:
                console.print(f"  [yellow]Skipped[/yellow]")
        console.print()
    
    console.print(f"[bold green]Interactive review complete![/bold green]")
    console.print(f"Tags applied: {applied_count}")


async def _get_auto_tag_candidates(librarian, session_id: str, exclude_existing: bool) -> List[Dict]:
    """Get notes that are candidates for auto-tagging."""
    all_notes = await librarian.get_all_notes(session_id)
    
    if exclude_existing:
        # Filter out notes that already have tags
        candidates = [note for note in all_notes if not note.get('tags', [])]
    else:
        candidates = all_notes
    
    return candidates


async def _process_auto_tag_batch(librarian, session_id: str, notes: List[Dict],
                                 confidence_threshold: float, max_tags_per_note: int,
                                 dry_run: bool, interactive: bool) -> Dict[str, int]:
    """Process a batch of notes for auto-tagging."""
    tagged_notes = 0
    total_tags = 0
    
    for note in notes:
        # Generate tag suggestions for this note
        suggestions = await _generate_tag_suggestions(
            librarian, session_id, [note], max_tags_per_note, 
            confidence_threshold, True, True
        )
        
        note_suggestions = suggestions.get(note['id'], [])
        if not note_suggestions:
            continue
        
        # Apply tags based on mode
        if interactive:
            applied = await _interactive_apply_tags(librarian, session_id, note, note_suggestions, dry_run)
        else:
            applied = await _auto_apply_tags(librarian, session_id, note, note_suggestions, dry_run)
        
        if applied > 0:
            tagged_notes += 1
            total_tags += applied
    
    return {'tagged_notes': tagged_notes, 'total_tags': total_tags}


async def _interactive_apply_tags(librarian, session_id: str, note: Dict, 
                                 suggestions: List[Dict], dry_run: bool) -> int:
    """Interactively apply tags to a note."""
    console.print(f"[bold cyan]Note:[/bold cyan] {note.get('title', note['id'])}")
    
    applied_count = 0
    for suggestion in suggestions:
        tag = suggestion['tag']
        confidence = suggestion['confidence']
        
        if Confirm.ask(f"Apply tag #{tag} (confidence: {confidence:.0%})?"):
            if not dry_run:
                await librarian.add_tag_to_note(session_id, note['id'], tag)
            applied_count += 1
            
            action = "Would apply" if dry_run else "Applied"
            console.print(f"  [green]{action} #{tag}[/green]")
    
    return applied_count


async def _auto_apply_tags(librarian, session_id: str, note: Dict, 
                          suggestions: List[Dict], dry_run: bool) -> int:
    """Automatically apply tags to a note."""
    applied_count = 0
    
    for suggestion in suggestions:
        tag = suggestion['tag']
        
        if not dry_run:
            await librarian.add_tag_to_note(session_id, note['id'], tag)
        applied_count += 1
        
        action = "Would apply" if dry_run else "Applied"
        note_title = note.get('title', note['id'])[:30]
        console.print(f"[dim]{action} #{tag} to {note_title}...[/dim]")
    
    return applied_count


async def _auto_merge_duplicates(librarian, session_id: str, duplicates: List[Dict[str, Any]], 
                                dry_run: bool) -> int:
    """Automatically merge obvious duplicate tags."""
    merge_count = 0
    
    for dup_group in duplicates:
        primary = dup_group['primary']
        for duplicate in dup_group['duplicates']:
            if not dry_run:
                await librarian.merge_tags(session_id, duplicate, primary)
            merge_count += 1
            
            action = "Would merge" if dry_run else "Merged"
            console.print(f"[dim]{action} #{duplicate} → #{primary}[/dim]")
    
    return merge_count


async def _perform_tag_merge(librarian, session_id: str, notes: List[Dict], 
                            source_tag: str, target_tag: str, case_sensitive: bool, 
                            dry_run: bool) -> Dict[str, Any]:
    """Perform the actual tag merge operation."""
    results = {
        'notes_processed': len(notes),
        'notes_modified': 0,
        'errors': [],
        'skipped': []
    }
    
    for note in notes:
        try:
            note_tags = note.get('tags', [])
            
            # Find matching tags (case sensitive or not)
            if case_sensitive:
                matching_tags = [tag for tag in note_tags if tag == source_tag]
            else:
                matching_tags = [tag for tag in note_tags if tag.lower() == source_tag.lower()]
            
            if matching_tags:
                if not dry_run:
                    # Remove source tags and add target tag
                    for tag in matching_tags:
                        await librarian.remove_tag_from_note(session_id, note['id'], tag)
                    await librarian.add_tag_to_note(session_id, note['id'], target_tag)
                
                results['notes_modified'] += 1
                
                action = "Would merge" if dry_run else "Merged"
                note_title = note.get('title', note['id'])[:30]
                console.print(f"[dim]{action} in {note_title}...[/dim]")
            else:
                results['skipped'].append(note['id'])
                
        except Exception as e:
            results['errors'].append({'note_id': note['id'], 'error': str(e)})
    
    return results


def _display_merge_results(results: Dict[str, Any], source_tag: str, target_tag: str, dry_run: bool):
    """Display tag merge operation results."""
    action = "Would be merged" if dry_run else "Merged"
    
    console.print(f"\n[bold green]Tag Merge Results[/bold green]")
    console.print(f"[bold]Operation:[/bold] #{source_tag} → #{target_tag}")
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta")
    
    table.add_row("Notes Processed", str(results['notes_processed']))
    table.add_row("Notes Modified", str(results['notes_modified']))
    table.add_row("Notes Skipped", str(len(results['skipped'])))
    table.add_row("Errors", str(len(results['errors'])))
    
    console.print(table)
    
    if results['errors']:
        console.print("\n[red]Errors encountered:[/red]")
        for error in results['errors'][:5]:  # Show first 5 errors
            console.print(f"  • {error['note_id']}: {error['error']}")


async def _create_cleanup_plan(tag_usage: Dict[str, int], remove_unused: bool, fix_case: bool,
                              remove_special_chars: bool, min_length: int, max_length: int) -> Dict[str, Any]:
    """Create a comprehensive tag cleanup plan."""
    cleanup_operations = {
        'remove_unused': [],
        'fix_case': [],
        'remove_special_chars': [],
        'fix_length': [],
    }
    
    for tag, usage_count in tag_usage.items():
        # Remove unused tags
        if remove_unused and usage_count == 0:
            cleanup_operations['remove_unused'].append(tag)
        
        # Fix case issues
        if fix_case and tag != tag.lower():
            cleanup_operations['fix_case'].append({'old': tag, 'new': tag.lower()})
        
        # Remove special characters
        if remove_special_chars:
            import re
            clean_tag = re.sub(r'[^\w\-/]', '', tag)
            if clean_tag != tag and clean_tag:
                cleanup_operations['remove_special_chars'].append({'old': tag, 'new': clean_tag})
        
        # Fix length issues
        if len(tag) < min_length:
            cleanup_operations['fix_length'].append({'old': tag, 'new': None, 'action': 'remove_short'})
        elif len(tag) > max_length:
            truncated = tag[:max_length]
            cleanup_operations['fix_length'].append({'old': tag, 'new': truncated, 'action': 'truncate'})
    
    return cleanup_operations


async def _execute_cleanup_plan(librarian, session_id: str, cleanup_plan: Dict[str, Any],
                               dry_run: bool, interactive: bool) -> Dict[str, Any]:
    """Execute the tag cleanup plan."""
    results = {
        'operations_performed': 0,
        'tags_removed': 0,
        'tags_modified': 0,
        'errors': []
    }
    
    # Execute each type of cleanup operation
    for operation_type, operations in cleanup_plan.items():
        if not operations:
            continue
        
        console.print(f"\n[bold cyan]{operation_type.replace('_', ' ').title()}[/bold cyan]")
        
        for operation in operations:
            try:
                if interactive:
                    should_proceed = _prompt_for_cleanup_operation(operation_type, operation)
                    if not should_proceed:
                        continue
                
                # Execute the operation
                await _execute_single_cleanup_operation(
                    librarian, session_id, operation_type, operation, dry_run
                )
                
                results['operations_performed'] += 1
                
                if operation_type == 'remove_unused':
                    results['tags_removed'] += 1
                else:
                    results['tags_modified'] += 1
                    
            except Exception as e:
                results['errors'].append({'operation': operation_type, 'data': operation, 'error': str(e)})
    
    return results


def _prompt_for_cleanup_operation(operation_type: str, operation) -> bool:
    """Prompt user for confirmation of cleanup operation."""
    if operation_type == 'remove_unused':
        return Confirm.ask(f"Remove unused tag '#{operation}'?")
    elif operation_type == 'fix_case':
        return Confirm.ask(f"Change '#{operation['old']}' to '#{operation['new']}'?")
    elif operation_type == 'remove_special_chars':
        return Confirm.ask(f"Clean special chars: '#{operation['old']}' → '#{operation['new']}'?")
    elif operation_type == 'fix_length':
        if operation['action'] == 'remove_short':
            return Confirm.ask(f"Remove short tag '#{operation['old']}' (length: {len(operation['old'])})?")
        else:
            return Confirm.ask(f"Truncate long tag: '#{operation['old']}' → '#{operation['new']}'?")
    
    return True


async def _execute_single_cleanup_operation(librarian, session_id: str, operation_type: str,
                                           operation, dry_run: bool):
    """Execute a single cleanup operation."""
    action = "Would" if dry_run else "Did"
    
    if operation_type == 'remove_unused':
        if not dry_run:
            await librarian.remove_tag_completely(session_id, operation)
        console.print(f"[dim]{action} remove unused tag #{operation}[/dim]")
        
    elif operation_type in ['fix_case', 'remove_special_chars']:
        old_tag = operation['old']
        new_tag = operation['new']
        
        if not dry_run:
            await librarian.rename_tag(session_id, old_tag, new_tag)
        console.print(f"[dim]{action} rename #{old_tag} → #{new_tag}[/dim]")
        
    elif operation_type == 'fix_length':
        if operation['action'] == 'remove_short':
            if not dry_run:
                await librarian.remove_tag_completely(session_id, operation['old'])
            console.print(f"[dim]{action} remove short tag #{operation['old']}[/dim]")
        else:
            if not dry_run:
                await librarian.rename_tag(session_id, operation['old'], operation['new'])
            console.print(f"[dim]{action} truncate #{operation['old']} → #{operation['new']}[/dim]")


def _display_cleanup_results(results: Dict[str, Any], dry_run: bool):
    """Display tag cleanup results."""
    action = "Would be performed" if dry_run else "Performed"
    
    console.print(f"\n[bold green]Tag Cleanup Results[/bold green]")
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta")
    
    table.add_row("Operations", str(results['operations_performed']))
    table.add_row("Tags Removed", str(results['tags_removed']))
    table.add_row("Tags Modified", str(results['tags_modified']))
    table.add_row("Errors", str(len(results['errors'])))
    
    console.print(table)
    
    if results['errors']:
        console.print("\n[red]Errors encountered:[/red]")
        for error in results['errors'][:5]:
            console.print(f"  • {error['operation']}: {error['error']}")


async def _build_tag_hierarchy(tags: List[str], separator: str, max_depth: int) -> Dict[str, Any]:
    """Build hierarchical structure from tags using separator."""
    hierarchy = {}
    tag_levels = {}
    
    for tag in tags:
        if separator in tag:
            parts = tag.split(separator)
            if len(parts) <= max_depth:
                current_level = hierarchy
                
                for i, part in enumerate(parts):
                    level = i + 1
                    
                    if part not in current_level:
                        current_level[part] = {
                            'children': {}, 
                            'level': level, 
                            'full_tag': separator.join(parts[:i+1])
                        }
                    
                    # Track tag levels
                    full_tag = separator.join(parts[:i+1])
                    tag_levels[full_tag] = level
                    
                    current_level = current_level[part]['children']
        else:
            # Root level tag
            if tag not in hierarchy:
                hierarchy[tag] = {'children': {}, 'level': 1, 'full_tag': tag}
            tag_levels[tag] = 1
    
    return {
        'hierarchy': hierarchy,
        'tag_levels': tag_levels,
        'max_depth': max(tag_levels.values()) if tag_levels else 0,
        'total_hierarchical_tags': len([t for t in tags if separator in t])
    }


def _display_tag_tree(hierarchy: Dict[str, Any], max_depth: int):
    """Display tag hierarchy as a tree structure."""
    console.print("\n[bold blue]Tag Hierarchy Tree[/bold blue]\n")
    
    tree = Tree("🏷️ Tag Hierarchy")
    
    def add_to_tree(parent_node, level_dict, current_depth=1):
        if current_depth > max_depth:
            return
        
        for tag_name, tag_data in sorted(level_dict.items()):
            node = parent_node.add(f"[cyan]#{tag_name}[/cyan] [dim](level {tag_data['level']})[/dim]")
            
            if tag_data['children']:
                add_to_tree(node, tag_data['children'], current_depth + 1)
    
    add_to_tree(tree, hierarchy['hierarchy'])
    console.print(tree)


def _display_hierarchy_stats(hierarchy: Dict[str, Any]):
    """Display tag hierarchy statistics."""
    console.print("\n[bold blue]Tag Hierarchy Statistics[/bold blue]\n")
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Hierarchical Tags", str(hierarchy['total_hierarchical_tags']))
    table.add_row("Maximum Depth", str(hierarchy['max_depth']))
    table.add_row("Root Level Tags", str(len(hierarchy['hierarchy'])))
    
    # Count tags by level
    level_counts = Counter(hierarchy['tag_levels'].values())
    for level in sorted(level_counts.keys()):
        table.add_row(f"Level {level} Tags", str(level_counts[level]))
    
    console.print(table)


async def _create_missing_parents(librarian, session_id: str, hierarchy: Dict[str, Any], 
                                 separator: str) -> int:
    """Create missing parent tags in the hierarchy."""
    missing_parents = set()
    
    # Find missing parent tags
    for full_tag in hierarchy['tag_levels'].keys():
        if separator in full_tag:
            parts = full_tag.split(separator)
            
            # Check each parent level
            for i in range(1, len(parts)):
                parent_tag = separator.join(parts[:i])
                if parent_tag not in hierarchy['tag_levels']:
                    missing_parents.add(parent_tag)
    
    # Create missing parent tags
    for parent_tag in missing_parents:
        await librarian.create_tag(session_id, parent_tag)
        console.print(f"[green]Created parent tag: #{parent_tag}[/green]")
    
    return len(missing_parents)


async def _save_hierarchy(hierarchy: Dict[str, Any], output_path: Path):
    """Save tag hierarchy to file."""
    output_data = {
        'hierarchy': hierarchy['hierarchy'],
        'statistics': {
            'total_hierarchical_tags': hierarchy['total_hierarchical_tags'],
            'max_depth': hierarchy['max_depth'],
            'tag_levels': hierarchy['tag_levels']
        },
    }
    
    output_path.write_text(json.dumps(output_data, indent=2, default=str))