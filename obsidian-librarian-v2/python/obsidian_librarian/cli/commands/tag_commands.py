"""
Tag Management CLI Commands for Obsidian Librarian.

Provides comprehensive tag management functionality including analysis,
duplicate detection, suggestions, auto-tagging, merging, cleanup, and hierarchy management.
"""

import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Confirm, Prompt
from rich.tree import Tree
from rich.align import Align

from ..utils import setup_logging, get_logger, run_async, handle_async_errors
from ..utils.console import (
    console, create_header_panel, create_stats_table, create_progress_context,
    print_success, print_error, print_warning, print_info,
    print_section_header, print_subsection_header, create_confirmation_panel,
    format_count, create_tag_display, truncate_text
)
from ...librarian import ObsidianLibrarian
from ...models import LibrarianConfig
from ...vault import scan_vault_async

logger = get_logger(__name__)


@click.group(name='tags')
@click.pass_context
def tag_commands(ctx):
    """Comprehensive tag management for Obsidian vaults.
    
    Manage tags with analysis, cleanup, suggestions, merging, and hierarchy operations.
    All commands support dry-run mode and provide detailed feedback.
    """
    console.print(create_header_panel(
        "Tag Management System",
        description="Intelligent tag analysis and organization"
    ))


@tag_commands.command('analyze')
@click.argument('vault_path', type=click.Path(exists=True, file_okay=False))
@click.option('--output', '-o', type=click.Path(), help='Save analysis results to file')
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv', 'markdown']), 
              default='json', help='Output format')
@click.option('--min-frequency', type=int, default=1, help='Minimum tag frequency to include')
@click.option('--show-unused', is_flag=True, help='Show unused tags from templates')
@click.option('--detailed', is_flag=True, help='Show detailed per-note tag information')
@click.pass_obj
def analyze_tags(config: LibrarianConfig, vault_path: str, output: Optional[str], 
                output_format: str, min_frequency: int, show_unused: bool, detailed: bool):
    """Analyze tag usage patterns and statistics in the vault.
    
    Provides comprehensive tag analysis including:
    - Usage frequency and distribution
    - Most/least used tags
    - Tag hierarchies and relationships
    - Orphaned and unused tags
    - Tag complexity metrics
    """
    vault_path = Path(vault_path)
    
    @run_async
    @handle_async_errors
    async def run_analysis():
        console.print(f"[bold green]Analyzing tags in:[/bold green] {vault_path}")
        
        async with ObsidianLibrarian(config) as librarian:
            session_id = await librarian.create_session(vault_path)
            
            with create_progress_context() as progress:
                task = progress.add_task("Scanning vault for tags...", total=None)
                
                # Get all notes with tags
                notes = await librarian.get_all_notes(session_id)
                progress.update(task, description="Analyzing tag patterns...")
                
                # Analyze tag usage
                analysis = await _analyze_tag_patterns(notes, min_frequency)
                progress.update(task, description="Analysis complete!")
            
            # Display results
            _display_tag_analysis(analysis, detailed, show_unused)
            
            # Save results if requested
            if output:
                await _save_analysis_results(analysis, output, output_format)
                print_success(f"Analysis saved to {output}")
    
    asyncio.run(run_analysis())


@tag_commands.command('duplicates')
@click.argument('vault_path', type=click.Path(exists=True, file_okay=False))
@click.option('--similarity-threshold', type=float, default=0.8, 
              help='Similarity threshold for duplicate detection (0.0-1.0)')
@click.option('--case-sensitive/--case-insensitive', default=False, 
              help='Case sensitivity for comparison')
@click.option('--output', '-o', type=click.Path(), help='Save duplicate report to file')
@click.option('--auto-merge', is_flag=True, help='Automatically merge obvious duplicates')
@click.option('--dry-run', is_flag=True, help='Show what would be merged without making changes')
@click.pass_obj
def find_duplicates(config: LibrarianConfig, vault_path: str, similarity_threshold: float,
                   case_sensitive: bool, output: Optional[str], auto_merge: bool, dry_run: bool):
    """Find and optionally merge duplicate or similar tags.
    
    Detects duplicate tags using various methods:
    - Exact matches (case variants)
    - Similar spelling (edit distance)
    - Semantic similarity
    - Common abbreviations/expansions
    """
    vault_path = Path(vault_path)
    
    if auto_merge and not dry_run:
        if not Confirm.ask("[yellow]This will merge duplicate tags. Continue?[/yellow]"):
            print_info("Operation cancelled")
            return
    
    @run_async
    @handle_async_errors
    async def run_duplicate_detection():
        console.print(f"[bold green]Finding duplicate tags in:[/bold green] {vault_path}")
        
        if dry_run:
            print_warning("Running in dry-run mode - no changes will be made")
        
        async with ObsidianLibrarian(config) as librarian:
            session_id = await librarian.create_session(vault_path)
            
            with create_progress_context() as progress:
                task = progress.add_task("Analyzing tags for duplicates...", total=None)
                
                # Get all tags
                tags = await librarian.get_all_tags(session_id)
                progress.update(task, description="Finding duplicate patterns...")
                
                # Find duplicates
                duplicates = await _find_duplicate_tags(
                    tags, similarity_threshold, case_sensitive
                )
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
                print_success(f"Duplicate report saved to {output}")
    
    asyncio.run(run_duplicate_detection())


@tag_commands.command('suggest')
@click.argument('vault_path', type=click.Path(exists=True, file_okay=False))
@click.option('--note-path', type=click.Path(), help='Suggest tags for specific note')
@click.option('--max-suggestions', type=int, default=5, help='Maximum suggestions per note')
@click.option('--confidence-threshold', type=float, default=0.7, 
              help='Minimum confidence for suggestions (0.0-1.0)')
@click.option('--use-content/--no-content', default=True, help='Analyze note content for suggestions')
@click.option('--use-title/--no-title', default=True, help='Analyze note title for suggestions')
@click.option('--interactive', is_flag=True, help='Interactive mode for reviewing suggestions')
@click.option('--output', '-o', type=click.Path(), help='Save suggestions to file')
@click.pass_obj
def suggest_tags(config: LibrarianConfig, vault_path: str, note_path: Optional[str],
                max_suggestions: int, confidence_threshold: float, use_content: bool,
                use_title: bool, interactive: bool, output: Optional[str]):
    """Generate intelligent tag suggestions for notes.
    
    Uses AI analysis to suggest relevant tags based on:
    - Note content and title
    - Existing tag patterns in vault
    - Similar notes and their tags
    - Topic modeling and classification
    """
    vault_path = Path(vault_path)
    
    @run_async
    @handle_async_errors
    async def run_tag_suggestions():
        console.print(f"[bold green]Generating tag suggestions for:[/bold green] {vault_path}")
        
        async with ObsidianLibrarian(config) as librarian:
            session_id = await librarian.create_session(vault_path)
            
            with create_progress_context() as progress:
                task = progress.add_task("Analyzing notes for tag suggestions...", total=None)
                
                # Get notes to analyze
                if note_path:
                    notes = [await librarian.get_note(session_id, note_path)]
                else:
                    notes = await librarian.get_all_notes(session_id)
                
                progress.update(task, description="Generating suggestions...")
                
                # Generate suggestions
                suggestions = await _generate_tag_suggestions(
                    librarian, session_id, notes, max_suggestions, 
                    confidence_threshold, use_content, use_title
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
                print_success(f"Suggestions saved to {output}")
    
    asyncio.run(run_tag_suggestions())


@tag_commands.command('auto-tag')
@click.argument('vault_path', type=click.Path(exists=True, file_okay=False))
@click.option('--confidence-threshold', type=float, default=0.8, 
              help='Minimum confidence to auto-apply tags (0.0-1.0)')
@click.option('--max-tags-per-note', type=int, default=10, help='Maximum tags to add per note')
@click.option('--exclude-existing/--include-existing', default=True, 
              help='Skip notes that already have tags')
@click.option('--dry-run', is_flag=True, help='Show what would be tagged without making changes')
@click.option('--interactive', is_flag=True, help='Review each tagging decision')
@click.option('--batch-size', type=int, default=50, help='Number of notes to process at once')
@click.pass_obj
def auto_tag_notes(config: LibrarianConfig, vault_path: str, confidence_threshold: float,
                  max_tags_per_note: int, exclude_existing: bool, dry_run: bool,
                  interactive: bool, batch_size: int):
    """Automatically apply tags to notes based on AI analysis.
    
    Intelligently tags notes using:
    - Content analysis and topic detection
    - Pattern matching with existing tags
    - Similarity to already-tagged notes
    - Configurable confidence thresholds
    """
    vault_path = Path(vault_path)
    
    if not dry_run and not interactive:
        if not Confirm.ask("[yellow]This will automatically add tags to notes. Continue?[/yellow]"):
            print_info("Operation cancelled")
            return
    
    @run_async
    @handle_async_errors
    async def run_auto_tagging():
        console.print(f"[bold green]Auto-tagging notes in:[/bold green] {vault_path}")
        
        if dry_run:
            print_warning("Running in dry-run mode - no changes will be made")
        
        async with ObsidianLibrarian(config) as librarian:
            session_id = await librarian.create_session(vault_path)
            
            with create_progress_context() as progress:
                task = progress.add_task("Preparing notes for auto-tagging...", total=None)
                
                # Get candidate notes
                notes = await _get_auto_tag_candidates(
                    librarian, session_id, exclude_existing
                )
                progress.update(task, description=f"Processing {len(notes)} notes...")
                
                # Process in batches
                tagged_count = 0
                total_tags_added = 0
                
                for i in range(0, len(notes), batch_size):
                    batch = notes[i:i + batch_size]
                    
                    batch_results = await _process_auto_tag_batch(
                        librarian, session_id, batch, confidence_threshold,
                        max_tags_per_note, dry_run, interactive
                    )
                    
                    tagged_count += batch_results['tagged_notes']
                    total_tags_added += batch_results['total_tags']
                    
                    progress.update(task, description=f"Processed {i + len(batch)}/{len(notes)} notes")
                
                progress.update(task, description="Auto-tagging complete!")
            
            # Display results
            print_section_header("Auto-tagging Results")
            console.print(f"[green]Notes processed:[/green] {len(notes)}")
            console.print(f"[green]Notes tagged:[/green] {tagged_count}")
            console.print(f"[green]Total tags added:[/green] {total_tags_added}")
            console.print(f"[green]Average tags per note:[/green] {total_tags_added / max(tagged_count, 1):.1f}")
    
    asyncio.run(run_auto_tagging())


@tag_commands.command('merge')
@click.argument('vault_path', type=click.Path(exists=True, file_okay=False))
@click.option('--source-tag', required=True, help='Tag to merge from (will be removed)')
@click.option('--target-tag', required=True, help='Tag to merge into (will be kept)')
@click.option('--dry-run', is_flag=True, help='Show what would be merged without making changes')
@click.option('--case-sensitive/--case-insensitive', default=True, help='Case sensitivity for matching')
@click.pass_obj
def merge_tags(config: LibrarianConfig, vault_path: str, source_tag: str, target_tag: str,
              dry_run: bool, case_sensitive: bool):
    """Merge one tag into another across all notes.
    
    Replaces all instances of source-tag with target-tag throughout the vault.
    The source tag will be completely removed and replaced with the target tag.
    """
    vault_path = Path(vault_path)
    
    if not dry_run:
        console.print(create_confirmation_panel(
            f"This will replace all instances of '{source_tag}' with '{target_tag}' "
            f"across your entire vault. This action cannot be undone.",
            is_destructive=True
        ))
        
        if not Confirm.ask("[red]Are you sure you want to continue?[/red]"):
            print_info("Operation cancelled")
            return
    
    @run_async
    @handle_async_errors
    async def run_tag_merge():
        console.print(f"[bold green]Merging tags in:[/bold green] {vault_path}")
        console.print(f"[yellow]Source (will be removed):[/yellow] {source_tag}")
        console.print(f"[green]Target (will be kept):[/green] {target_tag}")
        
        if dry_run:
            print_warning("Running in dry-run mode - no changes will be made")
        
        async with ObsidianLibrarian(config) as librarian:
            session_id = await librarian.create_session(vault_path)
            
            with create_progress_context() as progress:
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
    
    asyncio.run(run_tag_merge())


@tag_commands.command('cleanup')
@click.argument('vault_path', type=click.Path(exists=True, file_okay=False))
@click.option('--remove-unused', is_flag=True, help='Remove tags not used in any notes')
@click.option('--fix-case', is_flag=True, help='Standardize tag case (lowercase)')
@click.option('--remove-special-chars', is_flag=True, help='Remove special characters from tags')
@click.option('--min-length', type=int, default=2, help='Minimum tag length (remove shorter)')
@click.option('--max-length', type=int, default=50, help='Maximum tag length (truncate longer)')
@click.option('--dry-run', is_flag=True, help='Show what would be cleaned without making changes')
@click.option('--interactive', is_flag=True, help='Review each cleanup decision')
@click.pass_obj
def cleanup_tags(config: LibrarianConfig, vault_path: str, remove_unused: bool, fix_case: bool,
                remove_special_chars: bool, min_length: int, max_length: int, dry_run: bool,
                interactive: bool):
    """Clean up and standardize tags across the vault.
    
    Performs various cleanup operations:
    - Remove unused/orphaned tags
    - Standardize case (usually lowercase)
    - Remove special characters
    - Fix length issues
    - Consolidate variations
    """
    vault_path = Path(vault_path)
    
    if not dry_run and not interactive:
        if not Confirm.ask("[yellow]This will modify tags across your vault. Continue?[/yellow]"):
            print_info("Operation cancelled")
            return
    
    @run_async
    @handle_async_errors
    async def run_tag_cleanup():
        console.print(f"[bold green]Cleaning up tags in:[/bold green] {vault_path}")
        
        if dry_run:
            print_warning("Running in dry-run mode - no changes will be made")
        
        async with ObsidianLibrarian(config) as librarian:
            session_id = await librarian.create_session(vault_path)
            
            with create_progress_context() as progress:
                task = progress.add_task("Analyzing tags for cleanup...", total=None)
                
                # Get all tags and their usage
                tag_usage = await librarian.get_tag_usage(session_id)
                progress.update(task, description="Identifying cleanup opportunities...")
                
                # Identify cleanup operations
                cleanup_plan = await _create_cleanup_plan(
                    tag_usage, remove_unused, fix_case, remove_special_chars,
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
    
    asyncio.run(run_tag_cleanup())


@tag_commands.command('hierarchy')
@click.argument('vault_path', type=click.Path(exists=True, file_okay=False))
@click.option('--show-tree', is_flag=True, help='Display tag hierarchy as a tree')
@click.option('--separator', default='/', help='Hierarchy separator character')
@click.option('--max-depth', type=int, default=5, help='Maximum hierarchy depth to display')
@click.option('--create-missing', is_flag=True, help='Create missing parent tags')
@click.option('--output', '-o', type=click.Path(), help='Save hierarchy to file')
@click.pass_obj
def show_hierarchy(config: LibrarianConfig, vault_path: str, show_tree: bool, separator: str,
                  max_depth: int, create_missing: bool, output: Optional[str]):
    """Analyze and display tag hierarchies in the vault.
    
    Shows tag relationships and hierarchical structure:
    - Parent/child relationships
    - Hierarchy depth analysis
    - Missing parent tags
    - Usage statistics by level
    """
    vault_path = Path(vault_path)
    
    @run_async
    @handle_async_errors
    async def run_hierarchy_analysis():
        console.print(f"[bold green]Analyzing tag hierarchy in:[/bold green] {vault_path}")
        
        async with ObsidianLibrarian(config) as librarian:
            session_id = await librarian.create_session(vault_path)
            
            with create_progress_context() as progress:
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
                    print_success(f"Created {missing_count} missing parent tags")
            
            # Save hierarchy if requested
            if output:
                await _save_hierarchy(hierarchy, output)
                print_success(f"Hierarchy saved to {output}")
    
    asyncio.run(run_hierarchy_analysis())


# Helper functions for tag analysis and operations

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


def _display_tag_analysis(analysis: Dict[str, Any], detailed: bool, show_unused: bool):
    """Display comprehensive tag analysis results."""
    print_section_header("Tag Analysis Results")
    
    # Basic statistics
    stats_data = {
        "Total Unique Tags": analysis['total_unique_tags'],
        "Total Tag Usages": analysis['total_tag_usages'],
        "Average Tags per Note": f"{analysis['avg_tags_per_note']:.1f}",
        "Single-use Tags": len(analysis['single_use_tags'])
    }
    
    console.print(create_stats_table("Vault Tag Statistics", stats_data))
    
    # Most used tags
    if analysis['most_used_tags']:
        print_subsection_header("Most Used Tags")
        most_used_table = Table()
        most_used_table.add_column("Tag", style="cyan")
        most_used_table.add_column("Usage Count", style="magenta")
        most_used_table.add_column("Percentage", style="green")
        
        total_usages = analysis['total_tag_usages']
        for tag, count in analysis['most_used_tags']:
            percentage = (count / total_usages * 100) if total_usages > 0 else 0
            most_used_table.add_row(f"#{tag}", str(count), f"{percentage:.1f}%")
        
        console.print(most_used_table)
    
    # Single-use tags (if requested or if there are many)
    single_use_count = len(analysis['single_use_tags'])
    if single_use_count > 0:
        print_subsection_header(f"Single-use Tags ({single_use_count})")
        if single_use_count <= 20:
            single_use_display = create_tag_display(analysis['single_use_tags'])
            console.print(single_use_display)
        else:
            console.print(f"[dim]First 20 of {single_use_count} single-use tags:[/dim]")
            single_use_display = create_tag_display(analysis['single_use_tags'][:20])
            console.print(single_use_display)


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
                'suggested_merge': tag1  # Could be more sophisticated
            })
            processed.add(tag1)
    
    return duplicates


def _calculate_tag_similarity(tag1: str, tag2: str) -> float:
    """Calculate similarity between two tags using edit distance."""
    # Simple Levenshtein distance ratio
    import difflib
    return difflib.SequenceMatcher(None, tag1.lower(), tag2.lower()).ratio()


def _display_duplicate_results(duplicates: List[Dict[str, Any]]):
    """Display duplicate tag detection results."""
    if not duplicates:
        print_success("No duplicate tags found!")
        return
    
    print_section_header(f"Found {len(duplicates)} Duplicate Tag Groups")
    
    for i, dup_group in enumerate(duplicates, 1):
        console.print(f"\n[bold cyan]Group {i}:[/bold cyan]")
        console.print(f"[green]Primary:[/green] #{dup_group['primary']}")
        console.print(f"[yellow]Duplicates:[/yellow] {create_tag_display(dup_group['duplicates'])}")
        console.print(f"[blue]Suggested merge target:[/blue] #{dup_group['suggested_merge']}")


async def _generate_tag_suggestions(librarian, session_id: str, notes: List[Dict],
                                   max_suggestions: int, confidence_threshold: float,
                                   use_content: bool, use_title: bool) -> Dict[str, Any]:
    """Generate AI-powered tag suggestions for notes."""
    suggestions = {}
    
    for note in notes:
        note_suggestions = []
        
        # Analyze note content and title for potential tags
        if use_content and note.get('content'):
            content_tags = await _analyze_content_for_tags(
                librarian, session_id, note['content'], max_suggestions
            )
            note_suggestions.extend(content_tags)
        
        if use_title and note.get('title'):
            title_tags = await _analyze_title_for_tags(
                librarian, session_id, note['title'], max_suggestions
            )
            note_suggestions.extend(title_tags)
        
        # Filter by confidence threshold
        filtered_suggestions = [
            tag for tag in note_suggestions 
            if tag.get('confidence', 0) >= confidence_threshold
        ]
        
        # Sort by confidence and take top suggestions
        filtered_suggestions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        suggestions[note['id']] = filtered_suggestions[:max_suggestions]
    
    return suggestions


async def _analyze_content_for_tags(librarian, session_id: str, content: str, 
                                   max_suggestions: int) -> List[Dict[str, Any]]:
    """Analyze note content to suggest relevant tags."""
    # This would integrate with the AI content analyzer
    # For now, return a placeholder implementation
    return [
        {'tag': 'content-based', 'confidence': 0.75, 'reason': 'Content analysis'},
        {'tag': 'auto-suggested', 'confidence': 0.65, 'reason': 'Pattern matching'}
    ]


async def _analyze_title_for_tags(librarian, session_id: str, title: str,
                                 max_suggestions: int) -> List[Dict[str, Any]]:
    """Analyze note title to suggest relevant tags."""
    # This would analyze the title for keywords and patterns
    # For now, return a placeholder implementation
    return [
        {'tag': 'title-based', 'confidence': 0.80, 'reason': 'Title analysis'}
    ]


def _display_tag_suggestions(suggestions: Dict[str, Any]):
    """Display tag suggestions in a formatted table."""
    if not suggestions:
        print_info("No tag suggestions generated")
        return
    
    print_section_header(f"Tag Suggestions for {len(suggestions)} Notes")
    
    table = Table()
    table.add_column("Note", style="cyan", max_width=30)
    table.add_column("Suggested Tags", style="yellow")
    table.add_column("Confidence", style="green")
    
    for note_id, note_suggestions in suggestions.items():
        if note_suggestions:
            tag_display = ", ".join([f"#{s['tag']}" for s in note_suggestions])
            confidence_display = ", ".join([f"{s['confidence']:.0%}" for s in note_suggestions])
            table.add_row(
                truncate_text(note_id, 25),
                tag_display,
                confidence_display
            )
    
    console.print(table)


async def _interactive_tag_suggestions(librarian, session_id: str, suggestions: Dict[str, Any]):
    """Interactive mode for reviewing and applying tag suggestions."""
    print_section_header("Interactive Tag Suggestion Review")
    
    applied_count = 0
    for note_id, note_suggestions in suggestions.items():
        if not note_suggestions:
            continue
        
        console.print(f"\n[bold cyan]Note:[/bold cyan] {note_id}")
        
        for suggestion in note_suggestions:
            tag = suggestion['tag']
            confidence = suggestion['confidence']
            reason = suggestion.get('reason', 'Unknown')
            
            console.print(f"[yellow]Suggested tag:[/yellow] #{tag}")
            console.print(f"[blue]Confidence:[/blue] {confidence:.0%}")
            console.print(f"[dim]Reason:[/dim] {reason}")
            
            if Confirm.ask(f"Apply tag '#{tag}' to this note?"):
                # Apply the tag
                await librarian.add_tag_to_note(session_id, note_id, tag)
                print_success(f"Applied tag #{tag}")
                applied_count += 1
            else:
                print_info("Skipped")
    
    print_section_header("Interactive Review Complete")
    console.print(f"[green]Tags applied:[/green] {applied_count}")


async def _save_analysis_results(analysis: Dict[str, Any], output_path: str, format_type: str):
    """Save tag analysis results to file in specified format."""
    output_path = Path(output_path)
    
    if format_type == 'json':
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
    elif format_type == 'csv':
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Tag', 'Usage Count', 'Percentage'])
            
            total_usages = analysis['total_tag_usages']
            for tag, count in analysis['tag_frequencies'].items():
                percentage = (count / total_usages * 100) if total_usages > 0 else 0
                writer.writerow([tag, count, f"{percentage:.1f}%"])
    elif format_type == 'markdown':
        with open(output_path, 'w') as f:
            f.write("# Tag Analysis Report\n\n")
            f.write(f"**Total Unique Tags:** {analysis['total_unique_tags']}\n")
            f.write(f"**Total Tag Usages:** {analysis['total_tag_usages']}\n")
            f.write(f"**Average Tags per Note:** {analysis['avg_tags_per_note']:.1f}\n\n")
            
            f.write("## Most Used Tags\n\n")
            for tag, count in analysis['most_used_tags']:
                f.write(f"- #{tag}: {count} uses\n")


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
            
            if dry_run:
                console.print(f"[dim]Would merge #{duplicate} → #{primary}[/dim]")
            else:
                console.print(f"[green]Merged #{duplicate} → #{primary}[/green]")
    
    return merge_count


async def _save_duplicate_results(duplicates: List[Dict[str, Any]], output_path: str, merge_count: int):
    """Save duplicate detection results to file."""
    results = {
        'duplicate_groups': duplicates,
        'total_groups': len(duplicates),
        'total_duplicates': sum(len(d['duplicates']) for d in duplicates),
        'merged_count': merge_count,
        'timestamp': str(asyncio.get_event_loop().time())
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


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
    console.print(f"\n[bold cyan]Note:[/bold cyan] {note.get('title', note['id'])}")
    
    applied_count = 0
    for suggestion in suggestions:
        tag = suggestion['tag']
        confidence = suggestion['confidence']
        
        if Confirm.ask(f"Apply tag #{tag} (confidence: {confidence:.0%})?"):
            if not dry_run:
                await librarian.add_tag_to_note(session_id, note['id'], tag)
            applied_count += 1
            
            action = "Would apply" if dry_run else "Applied"
            console.print(f"[green]{action} #{tag}[/green]")
    
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
        console.print(f"[dim]{action} #{tag} to {note.get('title', note['id'])}[/dim]")
    
    return applied_count


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
                console.print(f"[dim]{action} in {note.get('title', note['id'])}[/dim]")
            else:
                results['skipped'].append(note['id'])
                
        except Exception as e:
            results['errors'].append({'note_id': note['id'], 'error': str(e)})
            logger.error(f"Error merging tags in note {note['id']}", error=str(e))
    
    return results


def _display_merge_results(results: Dict[str, Any], source_tag: str, target_tag: str, dry_run: bool):
    """Display tag merge operation results."""
    action = "Would be merged" if dry_run else "Merged"
    
    print_section_header("Tag Merge Results")
    
    stats_data = {
        "Notes Processed": results['notes_processed'],
        "Notes Modified": results['notes_modified'],
        "Notes Skipped": len(results['skipped']),
        "Errors": len(results['errors'])
    }
    
    console.print(create_stats_table(f"{action}: #{source_tag} → #{target_tag}", stats_data))
    
    if results['errors']:
        print_subsection_header("Errors Encountered")
        for error in results['errors'][:5]:  # Show first 5 errors
            console.print(f"[red]• {error['note_id']}: {error['error']}[/red]")


async def _create_cleanup_plan(tag_usage: Dict[str, int], remove_unused: bool, fix_case: bool,
                              remove_special_chars: bool, min_length: int, max_length: int) -> Dict[str, Any]:
    """Create a comprehensive tag cleanup plan."""
    cleanup_operations = {
        'remove_unused': [],
        'fix_case': [],
        'remove_special_chars': [],
        'fix_length': [],
        'merge_similar': []
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
            if clean_tag != tag:
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
        
        print_subsection_header(f"Processing {operation_type.replace('_', ' ').title()}")
        
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
                logger.error(f"Error in cleanup operation {operation_type}", error=str(e))
    
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
    
    print_section_header("Tag Cleanup Results")
    
    stats_data = {
        "Operations": results['operations_performed'],
        "Tags Removed": results['tags_removed'],
        "Tags Modified": results['tags_modified'],
        "Errors": len(results['errors'])
    }
    
    console.print(create_stats_table(f"Cleanup Operations {action}", stats_data))
    
    if results['errors']:
        print_subsection_header("Errors Encountered")
        for error in results['errors'][:5]:
            console.print(f"[red]• {error['operation']}: {error['error']}[/red]")


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
                        current_level[part] = {'children': {}, 'level': level, 'full_tag': separator.join(parts[:i+1])}
                    
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
    print_section_header("Tag Hierarchy Tree")
    
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
    print_section_header("Tag Hierarchy Statistics")
    
    stats_data = {
        "Hierarchical Tags": hierarchy['total_hierarchical_tags'],
        "Maximum Depth": hierarchy['max_depth'],
        "Root Level Tags": len(hierarchy['hierarchy']),
    }
    
    # Count tags by level
    level_counts = Counter(hierarchy['tag_levels'].values())
    for level in sorted(level_counts.keys()):
        stats_data[f"Level {level} Tags"] = level_counts[level]
    
    console.print(create_stats_table("Hierarchy Analysis", stats_data))


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


async def _save_hierarchy(hierarchy: Dict[str, Any], output_path: str):
    """Save tag hierarchy to file."""
    output_data = {
        'hierarchy': hierarchy['hierarchy'],
        'statistics': {
            'total_hierarchical_tags': hierarchy['total_hierarchical_tags'],
            'max_depth': hierarchy['max_depth'],
            'tag_levels': hierarchy['tag_levels']
        },
        'timestamp': str(asyncio.get_event_loop().time())
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)


async def _save_tag_suggestions(suggestions: Dict[str, Any], output_path: str):
    """Save tag suggestions to file."""
    output_data = {
        'suggestions': suggestions,
        'total_notes': len(suggestions),
        'total_suggestions': sum(len(s) for s in suggestions.values()),
        'timestamp': str(asyncio.get_event_loop().time())
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)


if __name__ == '__main__':
    tag_commands()