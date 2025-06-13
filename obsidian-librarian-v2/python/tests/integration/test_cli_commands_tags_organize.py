"""
Comprehensive integration tests for tag and organize CLI commands.

Tests the complete CLI workflow for:
- Tag management commands (analyze, duplicates, suggest, merge, etc.)
- Directory organization commands (analyze, auto, setup, watch, etc.)
- Integration with the curate command
"""

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch, Mock, AsyncMock, MagicMock

import pytest
from click.testing import CliRunner

from obsidian_librarian.cli import cli
from obsidian_librarian.cli.commands.tag_commands import tag_group
from obsidian_librarian.models import (
    LibrarianConfig, 
    Note, 
    TagAnalysisResult,
    TagSimilarity,
    TagSuggestion,
    TagHierarchy,
    TagOperation,
    ClassificationResult,
    OrganizationRule,
    ClassificationConfidence,
    OrganizationAction,
)
from obsidian_librarian.services.tag_manager import TagManagerService
from obsidian_librarian.services.auto_organizer import AutoOrganizer


@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def comprehensive_vault(tmp_path):
    """Create a comprehensive test vault with various note types and tags."""
    vault_path = tmp_path / "test_vault"
    vault_path.mkdir()
    
    # Create .obsidian directory
    (vault_path / ".obsidian").mkdir()
    
    # Create directory structure
    directories = [
        "Daily Notes/2024/01",
        "Projects/Active",
        "Projects/Archive", 
        "Meetings/2024/01",
        "Research/ML",
        "Research/API",
        "Templates",
        "Archive",
        "Unsorted",
    ]
    
    for dir_path in directories:
        (vault_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create notes with various tag patterns
    notes = {
        # Daily notes
        "Daily Notes/2024/01/2024-01-15.md": """---
tags: [daily, personal]
---

# Monday, January 15, 2024

- Worked on #project/alpha implementation
- Meeting with #team/engineering about #api design
- TODO: Review #ml papers

#productivity #work-log""",

        # Project notes with hierarchical tags
        "Projects/Active/project-alpha.md": """---
tags: [project, active, python]
---

# Project Alpha

Building ML pipeline with #python and #machine-learning.
Related: #ml, #data-science, #api/v2

## Status
- Phase 1: Complete
- Phase 2: #in-progress

#PROJECT #Python #MachineLearning""",

        # Meeting notes
        "Meetings/2024/01/team-standup.md": """---
tags: [meeting, standup, team]
---

# Team Standup - 2024-01-15

Attendees: @alice @bob @carol

## Updates
- #project/alpha on track
- #api/v2 deployment pending
- #bug/critical fixed

#meeting #daily-standup""",

        # Research notes
        "Research/ML/transformer-research.md": """---
tags: [research, ml, nlp, deep-learning]
---

# Transformer Architecture Research

Studying #transformers for #nlp tasks.
Related: #deep_learning, #neural-networks, #AI

#research #machine_learning #ML""",

        # Duplicate/similar tags
        "Unsorted/random-note-1.md": """# Random Note 1

Working on #ml and #machine-learning tasks.
Also #API and #api development.
#todo #TODO #To-Do""",

        "Unsorted/random-note-2.md": """# Random Note 2

More #Python and #python code.
#project-alpha and #projectalpha updates.""",

        # Template (should be ignored)
        "Templates/daily.md": """---
tags: [template, daily]
---

# {{date}}

## Tasks
- [ ] 

## Notes

#daily""",

        # Archived (should be handled specially)
        "Archive/old-project.md": """---
tags: [archive, project, deprecated]
---

# Old Project

Archived content with #old-tech and #deprecated tags.""",
    }
    
    for note_path, content in notes.items():
        note_file = vault_path / note_path
        note_file.parent.mkdir(parents=True, exist_ok=True)
        note_file.write_text(content)
    
    return vault_path


class TestTagCLICommands:
    """Test suite for tag management CLI commands."""
    
    async def test_tags_analyze_command(self, cli_runner, comprehensive_vault):
        """Test the 'tags analyze' command."""
        with patch('obsidian_librarian.cli.commands.tag_commands.TagManagerService') as MockService:
            # Mock the analysis result
            mock_service = MockService.return_value.__aenter__.return_value
            mock_service.analyze_tags.return_value = TagAnalysisResult(
                total_tags=25,
                unique_tags=18,
                tag_info={
                    "ml": {"usage_count": 5, "notes": ["note1", "note2"]},
                    "project": {"usage_count": 8, "notes": ["note1", "note3"]},
                    "api": {"usage_count": 4, "notes": ["note2", "note4"]},
                },
                tag_clusters=[
                    {"tags": ["ml", "machine-learning", "machine_learning"], "confidence": 0.9},
                    {"tags": ["api", "API"], "confidence": 0.95},
                ],
                suggested_hierarchies=[
                    TagHierarchy(
                        root_tag="project",
                        children=[
                            TagHierarchy(root_tag="project/alpha", children=[], level=1),
                            TagHierarchy(root_tag="project/beta", children=[], level=1),
                        ],
                        level=0,
                        confidence=0.85
                    )
                ],
                usage_statistics={
                    "most_used": [("project", 8), ("ml", 5), ("api", 4)],
                    "least_used": [("deprecated", 1), ("old-tech", 1)],
                    "orphaned_tags": ["random-tag"],
                    "tag_growth": {"last_week": 5, "last_month": 12},
                },
                optimization_suggestions=[
                    "Consider merging 'ml' and 'machine-learning'",
                    "Tag 'api' has inconsistent casing (api, API)",
                    "Hierarchy suggested: project -> project/alpha, project/beta",
                ],
                similarity_groups={
                    "ml_group": ["ml", "machine-learning", "ML"],
                    "api_group": ["api", "API"],
                },
                analysis_timestamp="2024-01-15T10:00:00"
            )
            
            result = cli_runner.invoke(cli, ['tags', 'analyze', str(comprehensive_vault)])
            
            assert result.exit_code == 0
            assert "Tag Analysis Results" in result.output
            assert "Total tags: 25" in result.output
            assert "Unique tags: 18" in result.output
            assert "Similar Tag Groups" in result.output
            assert "ml, machine-learning" in result.output
            assert "Suggested Hierarchies" in result.output
            assert "project" in result.output
            assert "Optimization Suggestions" in result.output
    
    async def test_tags_duplicates_command(self, cli_runner, comprehensive_vault):
        """Test the 'tags duplicates' command."""
        with patch('obsidian_librarian.cli.commands.tag_commands.TagManagerService') as MockService:
            mock_service = MockService.return_value.__aenter__.return_value
            
            # Mock finding duplicates
            mock_service.find_similar_tags.return_value = [
                TagSimilarity("ml", "machine-learning", 0.9, "fuzzy"),
                TagSimilarity("api", "API", 0.95, "case"),
                TagSimilarity("todo", "TODO", 1.0, "case"),
                TagSimilarity("python", "Python", 1.0, "case"),
            ]
            
            # Mock tag info for usage counts
            mock_service.get_tag_statistics.return_value = {
                "tag_usage": {
                    "ml": 5,
                    "machine-learning": 3,
                    "api": 4,
                    "API": 2,
                    "todo": 6,
                    "TODO": 1,
                    "python": 8,
                    "Python": 2,
                }
            }
            
            result = cli_runner.invoke(cli, ['tags', 'duplicates', str(comprehensive_vault)])
            
            assert result.exit_code == 0
            assert "Similar/Duplicate Tags Found" in result.output
            assert "ml" in result.output
            assert "machine-learning" in result.output
            assert "0.9" in result.output  # Similarity score
            assert "api" in result.output
            assert "API" in result.output
    
    async def test_tags_suggest_command(self, cli_runner, comprehensive_vault):
        """Test the 'tags suggest' command for a specific note."""
        with patch('obsidian_librarian.cli.commands.tag_commands.TagManagerService') as MockService:
            mock_service = MockService.return_value.__aenter__.return_value
            
            # Mock tag suggestions
            mock_service.suggest_tags.return_value = [
                TagSuggestion("deep-learning", 0.85, "Related to ML content", "ai"),
                TagSuggestion("research", 0.8, "Academic paper format detected", "pattern"),
                TagSuggestion("2024", 0.75, "Year mentioned in content", "content"),
            ]
            
            note_path = "Research/ML/transformer-research.md"
            result = cli_runner.invoke(cli, [
                'tags', 'suggest', 
                str(comprehensive_vault), 
                note_path
            ])
            
            assert result.exit_code == 0
            assert f"Tag Suggestions for {note_path}" in result.output
            assert "deep-learning" in result.output
            assert "0.85" in result.output
            assert "research" in result.output
            assert "Academic paper format" in result.output
    
    async def test_tags_auto_tag_command(self, cli_runner, comprehensive_vault):
        """Test the 'tags auto-tag' command."""
        with patch('obsidian_librarian.cli.commands.tag_commands.TagManagerService') as MockService:
            mock_service = MockService.return_value.__aenter__.return_value
            
            # Mock auto-tagging results
            async def mock_auto_tag():
                yield {
                    'type': 'progress',
                    'note': 'Unsorted/random-note-1.md',
                    'suggested_tags': ['organization', 'todo-list'],
                    'confidence': 0.8
                }
                yield {
                    'type': 'progress', 
                    'note': 'Unsorted/random-note-2.md',
                    'suggested_tags': ['programming', 'development'],
                    'confidence': 0.75
                }
                yield {
                    'type': 'complete',
                    'summary': {
                        'notes_processed': 2,
                        'tags_added': 4,
                        'notes_modified': 2
                    }
                }
            
            mock_service.auto_tag_untagged_notes = mock_auto_tag
            
            # Test dry run
            result = cli_runner.invoke(cli, [
                'tags', 'auto-tag',
                str(comprehensive_vault),
                '--dry-run'
            ])
            
            assert result.exit_code == 0
            assert "Auto-tagging" in result.output
            assert "DRY RUN" in result.output
            assert "random-note-1.md" in result.output
            assert "organization" in result.output
            assert "Notes processed: 2" in result.output
    
    async def test_tags_merge_command(self, cli_runner, comprehensive_vault):
        """Test the 'tags merge' command."""
        with patch('obsidian_librarian.cli.commands.tag_commands.TagManagerService') as MockService:
            mock_service = MockService.return_value.__aenter__.return_value
            
            # Mock merge operation
            mock_service.merge_tags.return_value = [
                TagOperation(
                    operation_type="merge",
                    source_tags=["ml", "ML"],
                    target_tag="machine-learning",
                    affected_notes=["note1.md", "note2.md"],
                    success=True,
                    timestamp="2024-01-15T10:00:00"
                ),
                TagOperation(
                    operation_type="merge",
                    source_tags=["api"],
                    target_tag="API",
                    affected_notes=["note3.md"],
                    success=True,
                    timestamp="2024-01-15T10:00:01"
                ),
            ]
            
            # Test interactive mode with pre-selected merges
            with patch('click.confirm', return_value=True):
                result = cli_runner.invoke(cli, [
                    'tags', 'merge',
                    str(comprehensive_vault),
                    '--source', 'ml',
                    '--source', 'ML',
                    '--target', 'machine-learning'
                ])
            
            assert result.exit_code == 0
            assert "Merging tags" in result.output
            assert "ml, ML" in result.output
            assert "machine-learning" in result.output
            assert "Affected notes: 2" in result.output
    
    async def test_tags_cleanup_command(self, cli_runner, comprehensive_vault):
        """Test the 'tags cleanup' interactive command."""
        with patch('obsidian_librarian.cli.commands.tag_commands.TagManagerService') as MockService:
            mock_service = MockService.return_value.__aenter__.return_value
            
            # Mock the cleanup workflow
            mock_service.analyze_tags.return_value = TagAnalysisResult(
                total_tags=20,
                unique_tags=15,
                tag_clusters=[
                    {"tags": ["todo", "TODO", "To-Do"], "confidence": 0.95}
                ],
                optimization_suggestions=[
                    "Merge similar tags: todo, TODO, To-Do",
                    "Remove unused tag: obsolete-tag",
                ],
                similarity_groups={
                    "todo_group": ["todo", "TODO", "To-Do"]
                }
            )
            
            # Mock user interactions
            with patch('click.confirm') as mock_confirm:
                with patch('click.prompt') as mock_prompt:
                    mock_confirm.side_effect = [True, True, False]  # Yes, Yes, No (exit)
                    mock_prompt.return_value = "todo"  # Choose canonical form
                    
                    result = cli_runner.invoke(cli, [
                        'tags', 'cleanup',
                        str(comprehensive_vault),
                        '--interactive'
                    ])
            
            assert result.exit_code == 0
            assert "Tag Cleanup Wizard" in result.output
            assert "Similar tags found" in result.output
            assert "todo, TODO, To-Do" in result.output
    
    async def test_tags_hierarchy_command(self, cli_runner, comprehensive_vault):
        """Test the 'tags hierarchy' command."""
        with patch('obsidian_librarian.cli.commands.tag_commands.TagManagerService') as MockService:
            mock_service = MockService.return_value.__aenter__.return_value
            
            # Mock hierarchy suggestions
            mock_service.analyze_tags.return_value = TagAnalysisResult(
                suggested_hierarchies=[
                    TagHierarchy(
                        root_tag="project",
                        children=[
                            TagHierarchy(
                                root_tag="project/alpha",
                                children=[
                                    TagHierarchy(root_tag="project/alpha/frontend", children=[], level=2),
                                    TagHierarchy(root_tag="project/alpha/backend", children=[], level=2),
                                ],
                                level=1
                            ),
                            TagHierarchy(root_tag="project/beta", children=[], level=1),
                        ],
                        level=0,
                        confidence=0.9
                    ),
                    TagHierarchy(
                        root_tag="api",
                        children=[
                            TagHierarchy(root_tag="api/v1", children=[], level=1),
                            TagHierarchy(root_tag="api/v2", children=[], level=1),
                        ],
                        level=0,
                        confidence=0.85
                    ),
                ]
            )
            
            result = cli_runner.invoke(cli, ['tags', 'hierarchy', str(comprehensive_vault)])
            
            assert result.exit_code == 0
            assert "Tag Hierarchy Suggestions" in result.output
            assert "project" in result.output
            assert "├── project/alpha" in result.output
            assert "│   ├── project/alpha/frontend" in result.output
            assert "api" in result.output
            assert "├── api/v1" in result.output
    
    async def test_tags_export_import(self, cli_runner, comprehensive_vault, tmp_path):
        """Test exporting and importing tag data."""
        export_file = tmp_path / "tags_export.json"
        
        with patch('obsidian_librarian.cli.commands.tag_commands.TagManagerService') as MockService:
            mock_service = MockService.return_value.__aenter__.return_value
            
            # Mock export data
            mock_service.export_tag_data.return_value = {
                "tags": {
                    "ml": {"usage": 5, "notes": ["note1", "note2"]},
                    "project": {"usage": 8, "notes": ["note1", "note3"]},
                },
                "hierarchies": [
                    {"root": "project", "children": ["project/alpha", "project/beta"]}
                ],
                "rules": [
                    {"pattern": "ML|ml|machine-learning", "canonical": "machine-learning"}
                ]
            }
            
            # Test export
            result = cli_runner.invoke(cli, [
                'tags', 'export',
                str(comprehensive_vault),
                '--output', str(export_file)
            ])
            
            assert result.exit_code == 0
            assert export_file.exists()
            
            # Test import
            mock_service.import_tag_data.return_value = {
                "tags_imported": 2,
                "hierarchies_created": 1,
                "rules_added": 1
            }
            
            result = cli_runner.invoke(cli, [
                'tags', 'import',
                str(comprehensive_vault),
                '--input', str(export_file)
            ])
            
            assert result.exit_code == 0
            assert "Tags imported: 2" in result.output


class TestOrganizeCLICommands:
    """Test suite for directory organization CLI commands."""
    
    async def test_organize_analyze_command(self, cli_runner, comprehensive_vault):
        """Test the 'organize analyze' command."""
        with patch('obsidian_librarian.cli.commands.organize_commands.AutoOrganizer') as MockOrganizer:
            mock_organizer = MockOrganizer.return_value.__aenter__.return_value
            
            # Mock analysis results
            mock_organizer.analyze_vault_structure.return_value = {
                "total_notes": 8,
                "organized_notes": 5,
                "misplaced_notes": 3,
                "suggested_moves": [
                    {
                        "current": "Unsorted/random-note-1.md",
                        "suggested": "Projects/random-note-1.md",
                        "confidence": 0.8,
                        "reason": "Project-related content detected"
                    },
                    {
                        "current": "Unsorted/random-note-2.md",
                        "suggested": "Archive/random-note-2.md",
                        "confidence": 0.7,
                        "reason": "Old content, suggest archiving"
                    },
                ],
                "directory_stats": {
                    "Daily Notes": 1,
                    "Projects": 1,
                    "Meetings": 1,
                    "Research": 1,
                    "Unsorted": 2,
                },
                "organization_score": 0.625,  # 5/8 organized
            }
            
            result = cli_runner.invoke(cli, ['organize', 'analyze', str(comprehensive_vault)])
            
            assert result.exit_code == 0
            assert "Vault Organization Analysis" in result.output
            assert "Total notes: 8" in result.output
            assert "Organization score: 62.5%" in result.output
            assert "Misplaced notes: 3" in result.output
            assert "random-note-1.md" in result.output
            assert "Projects/random-note-1.md" in result.output
    
    async def test_organize_auto_command(self, cli_runner, comprehensive_vault):
        """Test the 'organize auto' command."""
        with patch('obsidian_librarian.cli.commands.organize_commands.AutoOrganizer') as MockOrganizer:
            mock_organizer = MockOrganizer.return_value.__aenter__.return_value
            
            # Mock auto-organization results
            async def mock_organize_vault(dry_run=False):
                results = []
                results.append(ClassificationResult(
                    suggested_path=Path("Projects/Active/random-note-1.md"),
                    confidence=ClassificationConfidence.HIGH,
                    reasoning="Project documentation detected",
                    action=OrganizationAction.MOVE,
                    score=0.85,
                    metadata={"original_path": "Unsorted/random-note-1.md"}
                ))
                results.append(ClassificationResult(
                    suggested_path=Path("Research/random-note-2.md"),
                    confidence=ClassificationConfidence.MEDIUM,
                    reasoning="Research content patterns",
                    action=OrganizationAction.SUGGEST,
                    score=0.7,
                    metadata={"original_path": "Unsorted/random-note-2.md"}
                ))
                
                return {
                    "processed": 2,
                    "organized": 1,
                    "suggested": 1,
                    "ignored": 0,
                    "errors": [],
                    "results": results
                }
            
            mock_organizer.organize_vault = mock_organize_vault
            
            # Test dry run
            result = cli_runner.invoke(cli, [
                'organize', 'auto',
                str(comprehensive_vault),
                '--dry-run'
            ])
            
            assert result.exit_code == 0
            assert "DRY RUN" in result.output
            assert "Auto-organizing vault" in result.output
            assert "Processed: 2" in result.output
            assert "Would organize: 1" in result.output
            assert "random-note-1.md → Projects/Active/random-note-1.md" in result.output
    
    async def test_organize_setup_command(self, cli_runner, comprehensive_vault):
        """Test the 'organize setup' interactive command."""
        with patch('obsidian_librarian.cli.commands.organize_commands.AutoOrganizer') as MockOrganizer:
            mock_organizer = MockOrganizer.return_value.__aenter__.return_value
            
            # Mock user inputs for interactive setup
            with patch('click.prompt') as mock_prompt:
                with patch('click.confirm') as mock_confirm:
                    # Simulate user creating rules
                    mock_prompt.side_effect = [
                        "meeting_notes",  # Rule name
                        "meeting|standup|agenda",  # Pattern
                        "Meetings/{year}/{month}/{filename}",  # Target
                        "8",  # Priority
                        "project_docs",  # Another rule name
                        "project|specification|requirements",  # Pattern
                        "Projects/Active/{filename}",  # Target
                        "7",  # Priority
                    ]
                    mock_confirm.side_effect = [True, True, False]  # Add rule, add another, done
                    
                    result = cli_runner.invoke(cli, [
                        'organize', 'setup',
                        str(comprehensive_vault)
                    ])
            
            assert result.exit_code == 0
            assert "Organization Rules Setup" in result.output
            assert "Rule added: meeting_notes" in result.output
            assert "Rule added: project_docs" in result.output
    
    async def test_organize_watch_command(self, cli_runner, comprehensive_vault):
        """Test the 'organize watch' command."""
        with patch('obsidian_librarian.cli.commands.organize_commands.AutoOrganizer') as MockOrganizer:
            mock_organizer = MockOrganizer.return_value.__aenter__.return_value
            
            # Mock file watcher
            mock_organizer.start_file_watcher = AsyncMock()
            mock_organizer.stop_file_watcher = AsyncMock()
            mock_organizer.is_watching = False
            
            # Simulate watch mode with keyboard interrupt
            with patch('asyncio.sleep', side_effect=KeyboardInterrupt):
                result = cli_runner.invoke(cli, [
                    'organize', 'watch',
                    str(comprehensive_vault)
                ])
            
            # Should handle gracefully
            assert "Watching for new files" in result.output
            assert "Stopping file watcher" in result.output
    
    async def test_organize_rules_command(self, cli_runner, comprehensive_vault):
        """Test the 'organize rules' command."""
        with patch('obsidian_librarian.cli.commands.organize_commands.AutoOrganizer') as MockOrganizer:
            mock_organizer = MockOrganizer.return_value.__aenter__.return_value
            
            # Mock existing rules
            mock_organizer.get_organization_rules.return_value = {
                "built_in": [
                    OrganizationRule(
                        name="daily_notes",
                        conditions={"filename_pattern": r"\d{4}-\d{2}-\d{2}"},
                        action=OrganizationAction.MOVE,
                        target_pattern="Daily Notes/{year}/{month}/{filename}",
                        priority=9,
                        enabled=True
                    ),
                    OrganizationRule(
                        name="meeting_notes",
                        conditions={"contains_text": ["meeting", "agenda", "minutes"]},
                        action=OrganizationAction.MOVE,
                        target_pattern="Meetings/{filename}",
                        priority=7,
                        enabled=True
                    ),
                ],
                "custom": [
                    OrganizationRule(
                        name="project_alpha",
                        conditions={"has_tag": "project-alpha"},
                        action=OrganizationAction.MOVE,
                        target_pattern="Projects/Alpha/{filename}",
                        priority=8,
                        enabled=True
                    ),
                ]
            }
            
            result = cli_runner.invoke(cli, ['organize', 'rules', str(comprehensive_vault)])
            
            assert result.exit_code == 0
            assert "Organization Rules" in result.output
            assert "Built-in Rules" in result.output
            assert "daily_notes" in result.output
            assert "Custom Rules" in result.output
            assert "project_alpha" in result.output
    
    async def test_organize_restructure_command(self, cli_runner, comprehensive_vault):
        """Test the 'organize restructure' command."""
        with patch('obsidian_librarian.cli.commands.organize_commands.AutoOrganizer') as MockOrganizer:
            mock_organizer = MockOrganizer.return_value.__aenter__.return_value
            
            # Mock restructure operation
            async def mock_restructure_vault(strategy="balanced", dry_run=True):
                return {
                    "strategy": strategy,
                    "changes": [
                        {
                            "type": "create_directory",
                            "path": "Projects/2024/Q1",
                            "reason": "Organize by time period"
                        },
                        {
                            "type": "move_file",
                            "from": "Projects/Active/project-alpha.md",
                            "to": "Projects/2024/Q1/project-alpha.md",
                            "reason": "Time-based organization"
                        },
                        {
                            "type": "merge_directories",
                            "from": ["Research/ML", "Research/AI"],
                            "to": "Research/AI-ML",
                            "reason": "Consolidate related topics"
                        },
                    ],
                    "summary": {
                        "directories_created": 3,
                        "files_moved": 15,
                        "directories_merged": 2,
                    }
                }
            
            mock_organizer.restructure_vault = mock_restructure_vault
            
            # Test with confirmation
            with patch('click.confirm', return_value=True):
                result = cli_runner.invoke(cli, [
                    'organize', 'restructure',
                    str(comprehensive_vault),
                    '--strategy', 'time-based',
                    '--dry-run'
                ])
            
            assert result.exit_code == 0
            assert "Vault Restructuring Plan" in result.output
            assert "Strategy: time-based" in result.output
            assert "create_directory" in result.output
            assert "Projects/2024/Q1" in result.output


class TestCurateIntegration:
    """Test integration of tag and organization features with curate command."""
    
    async def test_curate_with_tags_flag(self, cli_runner, comprehensive_vault):
        """Test curate command with --tags flag."""
        with patch('obsidian_librarian.librarian.ObsidianLibrarian') as MockLibrarian:
            mock_librarian = MockLibrarian.return_value.__aenter__.return_value
            mock_librarian.create_session.return_value = "test-session"
            
            # Mock curate with tag operations
            mock_librarian.curate_content.return_value = {
                'duplicates_processed': 0,
                'quality_improvements': 0,
                'structure_improvements': 0,
                'tag_operations': {
                    'duplicates_merged': 3,
                    'tags_normalized': 5,
                    'hierarchies_created': 2,
                    'auto_tagged_notes': 4,
                },
                'errors': []
            }
            
            result = cli_runner.invoke(cli, [
                'curate',
                str(comprehensive_vault),
                '--tags'
            ])
            
            assert result.exit_code == 0
            assert "Tag Operations" in result.output
            assert "Duplicates merged: 3" in result.output
            assert "Tags normalized: 5" in result.output
            assert "Auto-tagged notes: 4" in result.output
    
    async def test_curate_with_organize_flag(self, cli_runner, comprehensive_vault):
        """Test curate command with --organize flag."""
        with patch('obsidian_librarian.librarian.ObsidianLibrarian') as MockLibrarian:
            mock_librarian = MockLibrarian.return_value.__aenter__.return_value
            mock_librarian.create_session.return_value = "test-session"
            
            # Mock curate with organization operations
            mock_librarian.curate_content.return_value = {
                'duplicates_processed': 0,
                'quality_improvements': 0,
                'structure_improvements': 0,
                'organization_operations': {
                    'files_organized': 8,
                    'directories_created': 3,
                    'confidence_average': 0.82,
                },
                'errors': []
            }
            
            result = cli_runner.invoke(cli, [
                'curate',
                str(comprehensive_vault),
                '--organize'
            ])
            
            assert result.exit_code == 0
            assert "Organization Operations" in result.output
            assert "Files organized: 8" in result.output
            assert "Average confidence: 82.0%" in result.output
    
    async def test_curate_with_all_flags(self, cli_runner, comprehensive_vault):
        """Test curate command with both --tags and --organize flags."""
        with patch('obsidian_librarian.librarian.ObsidianLibrarian') as MockLibrarian:
            mock_librarian = MockLibrarian.return_value.__aenter__.return_value
            mock_librarian.create_session.return_value = "test-session"
            
            # Mock comprehensive curation
            mock_librarian.curate_content.return_value = {
                'duplicates_processed': 2,
                'quality_improvements': 5,
                'structure_improvements': 3,
                'tag_operations': {
                    'duplicates_merged': 4,
                    'tags_normalized': 6,
                    'hierarchies_created': 2,
                    'auto_tagged_notes': 5,
                },
                'organization_operations': {
                    'files_organized': 10,
                    'directories_created': 4,
                    'confidence_average': 0.85,
                },
                'errors': []
            }
            
            result = cli_runner.invoke(cli, [
                'curate',
                str(comprehensive_vault),
                '--tags',
                '--organize',
                '--dry-run'
            ])
            
            assert result.exit_code == 0
            assert "Running in dry-run mode" in result.output
            assert "Tag Operations" in result.output
            assert "Organization Operations" in result.output
            assert "Duplicates Processed" in result.output
            assert "Quality Improvements" in result.output


class TestErrorHandling:
    """Test error handling in tag and organize commands."""
    
    async def test_tags_command_vault_not_found(self, cli_runner, tmp_path):
        """Test tags command with non-existent vault."""
        non_existent = tmp_path / "does_not_exist"
        
        result = cli_runner.invoke(cli, ['tags', 'analyze', str(non_existent)])
        
        assert result.exit_code != 0
        assert "Error" in result.output or "not found" in result.output.lower()
    
    async def test_organize_command_permission_error(self, cli_runner, comprehensive_vault):
        """Test organize command with permission errors."""
        with patch('obsidian_librarian.cli.commands.organize_commands.AutoOrganizer') as MockOrganizer:
            MockOrganizer.side_effect = PermissionError("Access denied")
            
            result = cli_runner.invoke(cli, ['organize', 'auto', str(comprehensive_vault)])
            
            assert result.exit_code != 0
            assert "Error" in result.output
            assert "Access denied" in result.output
    
    async def test_tags_merge_conflict_handling(self, cli_runner, comprehensive_vault):
        """Test handling of merge conflicts."""
        with patch('obsidian_librarian.cli.commands.tag_commands.TagManagerService') as MockService:
            mock_service = MockService.return_value.__aenter__.return_value
            
            # Mock merge with conflicts
            mock_service.merge_tags.return_value = [
                TagOperation(
                    operation_type="merge",
                    source_tags=["ml", "ML"],
                    target_tag="machine-learning",
                    affected_notes=["note1.md"],
                    success=False,
                    error="Conflict: Target tag already exists with different meaning",
                    timestamp="2024-01-15T10:00:00"
                ),
            ]
            
            result = cli_runner.invoke(cli, [
                'tags', 'merge',
                str(comprehensive_vault),
                '--source', 'ml',
                '--target', 'machine-learning',
                '--force'
            ])
            
            assert "Conflict" in result.output
            assert "Target tag already exists" in result.output


class TestPerformance:
    """Test performance with large vaults."""
    
    @pytest.mark.slow
    async def test_tags_analyze_large_vault(self, cli_runner, tmp_path):
        """Test tag analysis performance with many tags."""
        # Create large vault
        large_vault = tmp_path / "large_vault"
        large_vault.mkdir()
        (large_vault / ".obsidian").mkdir()
        
        # Create many notes with varied tags
        for i in range(100):
            note_path = large_vault / f"note_{i}.md"
            tags = [f"tag{j}" for j in range(i % 10)]
            content = f"""---
tags: {tags}
---

# Note {i}

Content with #inline-tag-{i % 20} and #common-tag.
"""
            note_path.write_text(content)
        
        with patch('obsidian_librarian.cli.commands.tag_commands.TagManagerService') as MockService:
            mock_service = MockService.return_value.__aenter__.return_value
            
            # Mock fast analysis
            mock_service.analyze_tags.return_value = TagAnalysisResult(
                total_tags=500,
                unique_tags=200,
                tag_clusters=[],
                suggested_hierarchies=[],
                usage_statistics={},
                optimization_suggestions=[],
                similarity_groups={},
                analysis_timestamp="2024-01-15T10:00:00"
            )
            
            import time
            start = time.time()
            result = cli_runner.invoke(cli, ['tags', 'analyze', str(large_vault)])
            end = time.time()
            
            assert result.exit_code == 0
            assert end - start < 5  # Should complete within 5 seconds
    
    @pytest.mark.slow
    async def test_organize_auto_large_vault(self, cli_runner, tmp_path):
        """Test auto-organization performance with many files."""
        # Create large vault
        large_vault = tmp_path / "large_vault"
        large_vault.mkdir()
        (large_vault / ".obsidian").mkdir()
        
        # Create many unorganized files
        for i in range(100):
            categories = ["meeting", "project", "daily", "research"]
            category = categories[i % 4]
            note_path = large_vault / f"{category}_note_{i}.md"
            content = f"# {category.title()} Note {i}\n\nContent related to {category}"
            note_path.write_text(content)
        
        with patch('obsidian_librarian.cli.commands.organize_commands.AutoOrganizer') as MockOrganizer:
            mock_organizer = MockOrganizer.return_value.__aenter__.return_value
            
            # Mock fast organization
            async def mock_organize_vault(dry_run=False):
                return {
                    "processed": 100,
                    "organized": 85,
                    "suggested": 10,
                    "ignored": 5,
                    "errors": [],
                    "results": []
                }
            
            mock_organizer.organize_vault = mock_organize_vault
            
            import time
            start = time.time()
            result = cli_runner.invoke(cli, [
                'organize', 'auto',
                str(large_vault),
                '--dry-run'
            ])
            end = time.time()
            
            assert result.exit_code == 0
            assert end - start < 10  # Should complete within 10 seconds
            assert "Processed: 100" in result.output


class TestConfigurationIntegration:
    """Test configuration handling for tag and organize commands."""
    
    async def test_tags_with_custom_config(self, cli_runner, comprehensive_vault, tmp_path):
        """Test tags command with custom configuration."""
        config_file = tmp_path / "custom_config.yaml"
        config_file.write_text("""
tag_management:
  fuzzy_similarity_threshold: 0.9
  semantic_similarity_threshold: 0.8
  case_insensitive: true
  excluded_tags:
    - temp
    - draft
    - todo
  auto_tag_confidence: 0.8
  max_auto_tags: 3
""")
        
        with patch('obsidian_librarian.cli.commands.tag_commands.TagManagerService') as MockService:
            # Verify config is passed correctly
            result = cli_runner.invoke(cli, [
                '--config', str(config_file),
                'tags', 'analyze',
                str(comprehensive_vault)
            ])
            
            # MockService should be called with custom config
            assert MockService.called
    
    async def test_organize_with_custom_rules(self, cli_runner, comprehensive_vault, tmp_path):
        """Test organize command with custom rules configuration."""
        config_file = tmp_path / "organize_config.yaml"
        config_file.write_text("""
auto_organization:
  enabled: true
  confidence_threshold: 0.75
  rules:
    - name: team_meetings
      pattern: "team.*meeting"
      target: "Meetings/Team/{year}/{filename}"
      priority: 9
    - name: project_specs
      pattern: ".*specification.*"
      target: "Projects/Specifications/{filename}"
      priority: 8
""")
        
        with patch('obsidian_librarian.cli.commands.organize_commands.AutoOrganizer') as MockOrganizer:
            result = cli_runner.invoke(cli, [
                '--config', str(config_file),
                'organize', 'analyze',
                str(comprehensive_vault)
            ])
            
            assert MockOrganizer.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])