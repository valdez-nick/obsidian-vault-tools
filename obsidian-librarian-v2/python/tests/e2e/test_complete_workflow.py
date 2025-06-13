"""
End-to-end tests for Obsidian Librarian complete workflow.

Tests the entire application workflow using the example vault:
- Initial vault analysis
- Tag management workflow
- Directory organization workflow  
- Research and curation
- Complete integration scenarios
"""

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import time

import pytest
import pytest_asyncio

from obsidian_librarian import ObsidianLibrarian, Vault
from obsidian_librarian.models import LibrarianConfig, VaultConfig
from obsidian_librarian.services.tag_manager import TagManagerService
from obsidian_librarian.services.auto_organizer import AutoOrganizer
from obsidian_librarian.cli import cli
from click.testing import CliRunner


@pytest.fixture
def example_vault_path():
    """Get path to the example vault."""
    # Assuming we're in tests/e2e/ directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # Go up to python/
    example_vault = project_root / "example-vault"
    
    if not example_vault.exists():
        pytest.skip(f"Example vault not found at {example_vault}")
    
    return example_vault


@pytest.fixture
async def test_vault(tmp_path, example_vault_path):
    """Create a test copy of the example vault."""
    test_vault_path = tmp_path / "test_vault"
    
    # Copy example vault to test location
    shutil.copytree(example_vault_path, test_vault_path)
    
    # Add some additional test content
    additional_notes = {
        "Unsorted/random-note-1.md": """# Random Note 1

Some content about #python and #MachineLearning.
Working on #project-alpha and #API development.

TODO: Organize this note properly.""",

        "Unsorted/meeting-2024-01-16.md": """# Team Meeting - Jan 16

Attendees: @alice @bob @carol

## Agenda
- Project status
- Sprint planning
- Technical decisions

## Action Items
- [ ] Update documentation
- [ ] Review PRs
- [ ] Deploy to staging

#meeting #team #sprint-planning""",

        "duplicate-content-1.md": """# Duplicate Research

This is research about transformers and NLP.
Using #ml and #deep-learning techniques.""",

        "duplicate-content-2.md": """# Duplicate Research

This is research about transformers and NLP.
Using #ml and #deep-learning techniques.""",

        "old-project-notes.md": """# Old Project Notes

From 2020 project. Should be archived.
#deprecated #old-project #archive-me""",
    }
    
    # Create additional test notes
    for note_path, content in additional_notes.items():
        note_file = test_vault_path / note_path
        note_file.parent.mkdir(parents=True, exist_ok=True)
        note_file.write_text(content)
    
    yield test_vault_path
    
    # Cleanup is handled by tmp_path fixture


@pytest.fixture
def test_config(test_vault):
    """Create test configuration."""
    return LibrarianConfig(
        vault_path=test_vault,
        enable_ai_features=True,
        openai_api_key="test-key",  # Will use mocks
        cache_dir=test_vault / ".librarian_cache",
        backup_dir=test_vault / ".librarian_backups",
        features={
            "tag_management": {
                "enabled": True,
                "auto_tag_confidence": 0.7,
                "similarity_threshold": 0.8,
            },
            "auto_organization": {
                "enabled": True,
                "confidence_threshold": 0.7,
                "watch_mode": False,
            },
            "ai_features": {
                "content_analysis": True,
                "embedding_search": True,
                "auto_suggestions": True,
            }
        }
    )


@pytest.fixture
def cli_runner():
    """Create CLI test runner."""
    return CliRunner()


class TestEndToEndWorkflow:
    """Complete end-to-end workflow tests."""
    
    @pytest.mark.asyncio
    async def test_initial_vault_setup_and_analysis(self, test_vault, test_config):
        """Test initial vault setup and analysis."""
        async with ObsidianLibrarian(test_config) as librarian:
            # Create session
            session_id = await librarian.create_session()
            assert session_id is not None
            
            # Analyze vault
            analysis = await librarian.analyze_vault()
            
            # Verify analysis results
            assert analysis is not None
            assert analysis.get("total_notes", 0) > 0
            assert "vault_stats" in analysis
            assert "tag_analysis" in analysis
            assert "organization_analysis" in analysis
            
            # Check vault stats
            stats = analysis["vault_stats"]
            assert stats["total_notes"] >= 10  # Example vault + test notes
            assert stats["total_words"] > 0
            assert stats["total_links"] > 0
            
            # Check tag analysis
            tag_analysis = analysis["tag_analysis"]
            assert tag_analysis["total_tags"] > 0
            assert tag_analysis["unique_tags"] > 0
            assert len(tag_analysis.get("duplicate_groups", [])) > 0  # Should find ML/ml etc
            
            # Check organization analysis  
            org_analysis = analysis["organization_analysis"]
            assert org_analysis["organized_notes"] >= 0
            assert org_analysis["misplaced_notes"] > 0  # Our test notes in wrong places
            assert len(org_analysis.get("suggestions", [])) > 0
    
    @pytest.mark.asyncio
    async def test_tag_management_workflow(self, test_vault, test_config):
        """Test complete tag management workflow."""
        async with ObsidianLibrarian(test_config) as librarian:
            session_id = await librarian.create_session()
            
            # Step 1: Analyze tags
            tag_service = librarian._tag_manager
            tag_analysis = await tag_service.analyze_tags()
            
            assert tag_analysis.total_tags > 0
            assert len(tag_analysis.tag_clusters) > 0  # Should find similar tags
            assert len(tag_analysis.optimization_suggestions) > 0
            
            # Step 2: Find duplicates
            all_tags = list(tag_analysis.tag_info.keys())
            duplicates = await tag_service.find_similar_tags(all_tags)
            
            assert len(duplicates) > 0
            # Should find ml/ML, python/Python, etc.
            ml_duplicates = [d for d in duplicates if "ml" in d.tag_a.lower() or "ml" in d.tag_b.lower()]
            assert len(ml_duplicates) > 0
            
            # Step 3: Merge similar tags
            merge_map = {}
            for dup in duplicates:
                if dup.similarity_score > 0.9:
                    # Prefer lowercase
                    if dup.tag_a.lower() == dup.tag_b.lower():
                        canonical = dup.tag_a.lower()
                        merge_map[dup.tag_a] = canonical
                        merge_map[dup.tag_b] = canonical
            
            if merge_map:
                merge_results = await tag_service.merge_tags(merge_map, dry_run=False)
                assert len(merge_results) > 0
                assert any(r.success for r in merge_results)
            
            # Step 4: Auto-tag untagged notes
            # First, find notes that need tags
            untagged_notes = []
            all_notes = await librarian.vault.get_all_notes()
            
            for note in all_notes:
                tags = await tag_service.analyzer.extract_tags_from_note(note)
                if len(tags) < 3:  # Notes with few tags
                    untagged_notes.append(note.id)
            
            # Suggest tags for these notes
            for note_id in untagged_notes[:5]:  # Test first 5
                suggestions = await tag_service.suggest_tags(note_id, max_suggestions=3)
                assert len(suggestions) > 0
                assert all(s.confidence >= 0.7 for s in suggestions)
            
            # Step 5: Build tag hierarchies
            final_analysis = await tag_service.analyze_tags()
            if final_analysis.suggested_hierarchies:
                # Verify hierarchies make sense
                for hierarchy in final_analysis.suggested_hierarchies:
                    assert hierarchy.confidence > 0.5
                    assert len(hierarchy.root_tag) > 0
    
    @pytest.mark.asyncio
    async def test_directory_organization_workflow(self, test_vault, test_config):
        """Test complete directory organization workflow."""
        async with ObsidianLibrarian(test_config) as librarian:
            session_id = await librarian.create_session()
            
            # Step 1: Analyze current organization
            organizer = librarian._auto_organizer
            structure_analysis = await organizer.analyze_vault_structure()
            
            assert structure_analysis["total_notes"] > 0
            assert structure_analysis["misplaced_notes"] > 0
            assert len(structure_analysis["suggested_moves"]) > 0
            
            # Step 2: Test classification of specific notes
            test_files = [
                "Unsorted/meeting-2024-01-16.md",  # Should go to Meetings
                "Unsorted/random-note-1.md",  # Should go to Projects or Research
                "old-project-notes.md",  # Should go to Archive
            ]
            
            for file_path in test_files:
                if (test_vault / file_path).exists():
                    result = await organizer.organize_file(Path(file_path))
                    assert result is not None
                    assert result.confidence.value >= 0.5  # At least medium confidence
                    assert result.suggested_path != Path(file_path)  # Should suggest move
            
            # Step 3: Organize vault (dry run first)
            dry_run_results = await organizer.organize_vault(dry_run=True)
            assert dry_run_results["processed"] > 0
            assert dry_run_results["organized"] > 0
            
            # Step 4: Apply organization (selective)
            # Only organize high-confidence suggestions
            actual_results = await organizer.organize_vault(
                dry_run=False,
                confidence_threshold=0.8
            )
            
            # Verify files were moved
            if actual_results["organized"] > 0:
                # Check that meeting note was moved
                meeting_path = test_vault / "Meetings" / "2024" / "01" / "meeting-2024-01-16.md"
                assert meeting_path.exists() or \
                       (test_vault / "Meetings" / "meeting-2024-01-16.md").exists()
            
            # Step 5: Learn from corrections
            # Simulate user moving a file to different location
            if actual_results["organized"] > 0:
                # Create feedback for learning
                from obsidian_librarian.services.auto_organizer import UserFeedback
                
                feedback = UserFeedback(
                    original_path=Path("Unsorted/some-note.md"),
                    suggested_path=Path("Projects/some-note.md"),
                    actual_path=Path("Research/Papers/some-note.md"),
                    accepted=True,
                    timestamp=datetime.utcnow(),
                    feedback_type="correction"
                )
                
                await organizer.add_feedback(feedback)
                
                # Future similar notes should consider this feedback
                stats = await organizer.get_statistics()
                assert stats["learning_metrics"]["total_feedback"] > 0
    
    @pytest.mark.asyncio
    async def test_research_workflow(self, test_vault, test_config):
        """Test research workflow with actual content."""
        async with ObsidianLibrarian(test_config) as librarian:
            session_id = await librarian.create_session()
            
            # Mock external research sources
            from unittest.mock import patch, AsyncMock
            
            with patch('obsidian_librarian.sources.ArxivSource.search') as mock_arxiv:
                mock_arxiv.return_value = AsyncMock(return_value=[
                    {
                        "title": "Attention Is All You Need",
                        "authors": ["Vaswani et al."],
                        "summary": "Transformer architecture for NLP",
                        "url": "https://arxiv.org/abs/1706.03762",
                        "source": "arxiv",
                        "quality_score": 0.95
                    }
                ])
                
                # Research on transformers (relates to existing note)
                results = []
                async for result in librarian.research(
                    "transformer architecture NLP",
                    sources=["arxiv"],
                    max_results=5
                ):
                    if result.get("type") == "result":
                        results.append(result.get("data"))
                
                assert len(results) > 0
                
                # Create research note from results
                if results:
                    best_result = max(results, key=lambda r: r.get("quality_score", 0))
                    
                    note_content = f"""# {best_result['title']}

Source: {best_result['source']}
URL: {best_result['url']}

## Summary
{best_result['summary']}

## Notes
Related to [[transformer-research]] in the vault.

#research #transformers #nlp #paper-review"""
                    
                    # Add to vault
                    note_path = Path("Research/Papers") / f"{best_result['title'].replace(' ', '-')}.md"
                    await librarian.vault.create_note(note_path, note_content)
                    
                    # Verify note was created and organized correctly
                    assert (test_vault / note_path).exists()
    
    @pytest.mark.asyncio
    async def test_curation_workflow(self, test_vault, test_config):
        """Test complete curation workflow."""
        async with ObsidianLibrarian(test_config) as librarian:
            session_id = await librarian.create_session()
            
            # Run complete curation
            curation_results = await librarian.curate_content(
                check_duplicates=True,
                improve_quality=True,
                organize_structure=True,
                manage_tags=True,
                organize_files=True,
                dry_run=False
            )
            
            # Verify all aspects were processed
            assert "duplicates_processed" in curation_results
            assert "quality_improvements" in curation_results
            assert "structure_improvements" in curation_results
            assert "tag_operations" in curation_results
            assert "organization_operations" in curation_results
            
            # Check duplicates were found and processed
            if curation_results["duplicates_processed"] > 0:
                # Our test vault has duplicate-content-1.md and duplicate-content-2.md
                remaining_duplicates = [
                    f for f in test_vault.glob("duplicate-content-*.md")
                    if f.exists()
                ]
                # At least one should be processed
                assert len(remaining_duplicates) < 2
            
            # Check tag operations
            tag_ops = curation_results.get("tag_operations", {})
            assert tag_ops.get("duplicates_merged", 0) >= 0
            assert tag_ops.get("hierarchies_created", 0) >= 0
            
            # Check organization operations
            org_ops = curation_results.get("organization_operations", {})
            assert org_ops.get("files_organized", 0) >= 0
            
            # Verify no critical errors
            assert len(curation_results.get("errors", [])) == 0
    
    @pytest.mark.asyncio
    async def test_incremental_workflow(self, test_vault, test_config):
        """Test incremental updates and learning."""
        async with ObsidianLibrarian(test_config) as librarian:
            session_id = await librarian.create_session()
            
            # Initial state
            initial_analysis = await librarian.analyze_vault()
            initial_tag_count = initial_analysis["tag_analysis"]["total_tags"]
            initial_note_count = initial_analysis["vault_stats"]["total_notes"]
            
            # Add new note
            new_note_content = """# New Machine Learning Project

Starting work on #new-ml-project using #tensorflow and #pytorch.
This relates to our #research on #deep-learning.

## Tasks
- [ ] Setup environment
- [ ] Implement baseline
- [ ] Run experiments

#project #active #ml"""
            
            new_note_path = Path("new-ml-project.md")
            await librarian.vault.create_note(new_note_path, new_note_content)
            
            # Auto-organize the new note
            organizer = librarian._auto_organizer
            classification = await organizer.organize_file(new_note_path)
            
            assert classification is not None
            assert "Projects" in str(classification.suggested_path) or \
                   "Research" in str(classification.suggested_path)
            
            # Apply organization
            if classification.action.value == "move":
                await librarian.vault.move_note(
                    new_note_path,
                    classification.suggested_path
                )
            
            # Re-analyze vault
            updated_analysis = await librarian.analyze_vault()
            
            # Verify changes
            assert updated_analysis["vault_stats"]["total_notes"] == initial_note_count + 1
            assert updated_analysis["tag_analysis"]["total_tags"] > initial_tag_count
            
            # Check new tags were detected
            new_tags = ["new-ml-project", "tensorflow", "pytorch", "active"]
            tag_info = updated_analysis["tag_analysis"].get("tag_info", {})
            for tag in new_tags:
                assert tag in tag_info or tag.lower() in tag_info
    
    @pytest.mark.asyncio 
    async def test_error_recovery_workflow(self, test_vault, test_config):
        """Test error handling and recovery."""
        async with ObsidianLibrarian(test_config) as librarian:
            session_id = await librarian.create_session()
            
            # Test 1: Handle corrupted note
            corrupted_path = test_vault / "corrupted.md"
            corrupted_path.write_bytes(b'\xff\xfe Invalid UTF-8 \xff\xff')
            
            # Should handle gracefully
            analysis = await librarian.analyze_vault()
            assert analysis is not None  # Should complete despite error
            
            # Test 2: Handle missing file during processing
            temp_note = test_vault / "temp-note.md"
            temp_note.write_text("# Temporary\n\nWill be deleted")
            
            # Start organization
            organizer = librarian._auto_organizer
            
            # Delete file during processing
            from unittest.mock import patch
            original_load = librarian.vault.load_note
            
            async def load_with_delete(path):
                if path == Path("temp-note.md"):
                    temp_note.unlink()
                    return None
                return await original_load(path)
            
            with patch.object(librarian.vault, 'load_note', side_effect=load_with_delete):
                result = await organizer.organize_file(Path("temp-note.md"))
                assert result is None  # Should handle missing file
            
            # Test 3: Handle permission errors
            if sys.platform != "win32":  # Unix-like systems
                restricted_dir = test_vault / "restricted"
                restricted_dir.mkdir()
                restricted_file = restricted_dir / "no-access.md"
                restricted_file.write_text("# Restricted")
                restricted_file.chmod(0o000)  # No permissions
                
                # Should handle permission error
                try:
                    await librarian.vault.load_note(Path("restricted/no-access.md"))
                except PermissionError:
                    pass  # Expected
                
                # Cleanup
                restricted_file.chmod(0o644)
                restricted_file.unlink()
            
            # Test 4: Recovery from backup
            # Create backup before changes
            from obsidian_librarian.services.backup import BackupService
            backup_service = BackupService(librarian.vault, test_config)
            
            backup_id = await backup_service.create_backup("Before test changes")
            
            # Make changes
            test_note = test_vault / "Welcome.md"
            original_content = test_note.read_text() if test_note.exists() else ""
            test_note.write_text("# Modified\n\nChanged content")
            
            # Restore from backup
            await backup_service.restore_backup(backup_id)
            
            # Verify restoration
            restored_content = test_note.read_text() if test_note.exists() else ""
            assert restored_content == original_content


class TestCLIEndToEnd:
    """End-to-end tests using CLI commands."""
    
    def test_cli_complete_workflow(self, cli_runner, test_vault):
        """Test complete workflow using CLI commands."""
        # Step 1: Initial analysis
        result = cli_runner.invoke(cli, ['analyze', str(test_vault)])
        assert result.exit_code == 0
        assert "Analysis complete" in result.output
        
        # Step 2: Tag analysis
        result = cli_runner.invoke(cli, ['tags', 'analyze', str(test_vault)])
        assert result.exit_code == 0
        assert "Total tags" in result.output
        
        # Step 3: Find duplicate tags
        result = cli_runner.invoke(cli, ['tags', 'duplicates', str(test_vault)])
        assert result.exit_code == 0
        # Should find ml/ML, python/Python etc
        
        # Step 4: Organization analysis
        result = cli_runner.invoke(cli, ['organize', 'analyze', str(test_vault)])
        assert result.exit_code == 0
        assert "Organization score" in result.output
        
        # Step 5: Dry run organization
        result = cli_runner.invoke(cli, [
            'organize', 'auto',
            str(test_vault),
            '--dry-run'
        ])
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        
        # Step 6: Full curation (dry run)
        result = cli_runner.invoke(cli, [
            'curate',
            str(test_vault),
            '--tags',
            '--organize',
            '--dry-run'
        ])
        assert result.exit_code == 0
        assert "dry-run mode" in result.output
    
    def test_cli_interactive_workflow(self, cli_runner, test_vault):
        """Test interactive CLI workflows."""
        from unittest.mock import patch
        
        # Test tag cleanup wizard
        with patch('click.confirm') as mock_confirm:
            with patch('click.prompt') as mock_prompt:
                # Simulate user interactions
                mock_confirm.side_effect = [True, False]  # Yes to first, no to exit
                mock_prompt.return_value = "ml"  # Choose 'ml' as canonical
                
                result = cli_runner.invoke(cli, [
                    'tags', 'cleanup',
                    str(test_vault),
                    '--interactive'
                ])
                
                assert result.exit_code == 0
                assert "Tag Cleanup Wizard" in result.output
        
        # Test organization setup
        with patch('click.prompt') as mock_prompt:
            with patch('click.confirm') as mock_confirm:
                # Create one rule then exit
                mock_prompt.side_effect = [
                    "meeting_notes",  # Rule name
                    "meeting|standup|agenda",  # Pattern
                    "Meetings/{year}/{month}/{filename}",  # Target
                    "8",  # Priority
                ]
                mock_confirm.side_effect = [True, False]  # Add rule, don't add another
                
                result = cli_runner.invoke(cli, [
                    'organize', 'setup',
                    str(test_vault)
                ])
                
                assert result.exit_code == 0
                assert "Rule added" in result.output


class TestPerformanceEndToEnd:
    """Performance tests for end-to-end scenarios."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_vault_performance(self, tmp_path):
        """Test performance with a large vault."""
        # Create large test vault
        large_vault = tmp_path / "large_vault"
        large_vault.mkdir()
        (large_vault / ".obsidian").mkdir()
        
        # Create directory structure
        directories = [
            "Daily Notes/2024",
            "Projects/Active",
            "Projects/Archive",
            "Meetings/2024",
            "Research/Papers",
            "Research/Notes",
        ]
        
        for dir_path in directories:
            (large_vault / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Generate many notes
        note_count = 1000
        for i in range(note_count):
            category = ["daily", "project", "meeting", "research"][i % 4]
            
            if category == "daily":
                date = datetime(2024, 1, 1) + timedelta(days=i % 365)
                filename = date.strftime("%Y-%m-%d.md")
                content = f"# {date.strftime('%A, %B %d, %Y')}\n\n- Task 1\n- Task 2\n\n#daily"
                path = large_vault / "Daily Notes/2024" / filename
                
            elif category == "project":
                filename = f"project-{i}.md"
                content = f"# Project {i}\n\nProject description.\n\n#project #active"
                path = large_vault / "Projects/Active" / filename
                
            elif category == "meeting":
                filename = f"meeting-{i}.md"
                content = f"# Meeting {i}\n\nAgenda:\n- Item 1\n\n#meeting #team"
                path = large_vault / "Meetings/2024" / filename
                
            else:  # research
                filename = f"research-{i}.md"
                content = f"# Research Topic {i}\n\nFindings...\n\n#research #ml"
                path = large_vault / "Research/Notes" / filename
            
            path.write_text(content)
        
        # Test configuration
        config = LibrarianConfig(
            vault_path=large_vault,
            enable_ai_features=False,  # Disable AI for performance test
            features={
                "tag_management": {"enabled": True},
                "auto_organization": {"enabled": True},
            }
        )
        
        # Measure performance
        start_time = time.time()
        
        async with ObsidianLibrarian(config) as librarian:
            # Initial analysis
            analysis_start = time.time()
            analysis = await librarian.analyze_vault()
            analysis_time = time.time() - analysis_start
            
            assert analysis["vault_stats"]["total_notes"] == note_count
            assert analysis_time < 30  # Should complete within 30 seconds
            
            # Tag analysis
            tag_start = time.time()
            tag_service = librarian._tag_manager
            tag_analysis = await tag_service.analyze_tags()
            tag_time = time.time() - tag_start
            
            assert tag_analysis.total_tags > 0
            assert tag_time < 20  # Should complete within 20 seconds
            
            # Organization analysis (sample)
            org_start = time.time()
            organizer = librarian._auto_organizer
            
            # Test organizing a subset
            sample_files = list(large_vault.glob("**/*.md"))[:100]
            org_results = []
            
            for file in sample_files:
                result = await organizer.organize_file(file.relative_to(large_vault))
                org_results.append(result)
            
            org_time = time.time() - org_start
            
            assert len(org_results) == 100
            assert org_time < 30  # 100 files in 30 seconds
        
        total_time = time.time() - start_time
        
        print(f"\nPerformance Results for {note_count} notes:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Analysis time: {analysis_time:.2f}s")
        print(f"  Tag analysis time: {tag_time:.2f}s")
        print(f"  Organization time (100 files): {org_time:.2f}s")
        
        # Overall should complete reasonably fast
        assert total_time < 120  # 2 minutes for everything


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    @pytest.mark.asyncio
    async def test_academic_research_vault(self, test_vault, test_config):
        """Test with academic research vault scenario."""
        # Add academic content
        academic_notes = {
            "Literature Review/transformers-survey.md": """---
title: Transformers in NLP - A Survey
tags: [literature-review, transformers, nlp, survey]
authors: [Smith et al.]
year: 2023
---

# Transformers in NLP - A Survey

## Abstract
Comprehensive survey of transformer architectures...

## Related Work
- [[attention-mechanism]]
- [[bert-paper]]
- [[gpt-models]]

#research #nlp #transformers #survey #2023""",

            "Reading Notes/attention-paper.md": """# Attention Is All You Need - Reading Notes

Paper: https://arxiv.org/abs/1706.03762

## Key Points
- Self-attention mechanism
- Positional encoding
- Multi-head attention

#reading-notes #transformers #attention #seminal-paper""",

            "Projects/thesis-outline.md": """# PhD Thesis Outline

## Chapter 1: Introduction
- Motivation
- Research questions

## Chapter 2: Literature Review
See [[Literature Review/transformers-survey]]

#thesis #phd #outline #writing""",
        }
        
        for path, content in academic_notes.items():
            note_path = test_vault / path
            note_path.parent.mkdir(parents=True, exist_ok=True)
            note_path.write_text(content)
        
        async with ObsidianLibrarian(test_config) as librarian:
            # Analyze the academic vault
            analysis = await librarian.analyze_vault()
            
            # Should detect academic structure
            tag_analysis = await librarian._tag_manager.analyze_tags()
            
            # Should suggest hierarchies like:
            # - research -> literature-review, reading-notes
            # - transformers -> attention, bert, gpt
            assert len(tag_analysis.suggested_hierarchies) > 0
            
            # Auto-organize should recognize patterns
            organizer = librarian._auto_organizer
            
            # Test thesis outline classification
            thesis_result = await organizer.organize_file(Path("Projects/thesis-outline.md"))
            assert thesis_result is not None
            # Should keep in Projects or suggest Academic/Thesis
            
            # Curation should maintain academic structure
            curation = await librarian.curate_content(
                manage_tags=True,
                organize_files=True,
                dry_run=True
            )
            
            # Should preserve literature review structure
            assert curation.get("errors", []) == []
    
    @pytest.mark.asyncio
    async def test_project_management_vault(self, test_vault, test_config):
        """Test with project management vault scenario."""
        # Add project management content
        pm_notes = {
            "Projects/ProjectAlpha/sprint-23.md": """# Sprint 23 - Project Alpha

Sprint Goal: Complete API v2

## Sprint Backlog
- [ ] API endpoint refactoring
- [x] Database migration
- [ ] Testing suite

## Daily Standups
- [[Daily Notes/2024-01-15]] - Day 1
- [[Daily Notes/2024-01-16]] - Day 2

#sprint #project-alpha #agile #in-progress""",

            "Meetings/2024/01/sprint-planning.md": """# Sprint Planning - Sprint 23

Date: 2024-01-15
Attendees: @team-alpha

## Decisions
- 2-week sprint
- Focus on API v2

## Action Items
- [ ] @alice: Create JIRA tickets
- [ ] @bob: Setup CI/CD
- [x] @carol: Update documentation

#meeting #sprint-planning #project-alpha""",

            "Tasks/active-tasks.md": """# Active Tasks

## High Priority
- [ ] Complete API documentation #project-alpha #docs
- [ ] Review security audit #security #urgent

## Medium Priority  
- [ ] Refactor authentication #tech-debt
- [ ] Update team wiki #documentation

## Low Priority
- [ ] Explore new frameworks #research

#tasks #todo #project-management""",
        }
        
        for path, content in pm_notes.items():
            note_path = test_vault / path
            note_path.parent.mkdir(parents=True, exist_ok=True)
            note_path.write_text(content)
        
        async with ObsidianLibrarian(test_config) as librarian:
            # Should recognize project management patterns
            organizer = librarian._auto_organizer
            
            # Test sprint document organization
            sprint_result = await organizer.organize_file(
                Path("Projects/ProjectAlpha/sprint-23.md")
            )
            # Should keep in project hierarchy
            assert "Projects" in str(sprint_result.suggested_path)
            
            # Tag analysis should find project hierarchies
            tag_analysis = await librarian._tag_manager.analyze_tags()
            
            # Should identify task-related tags
            task_tags = [tag for tag in tag_analysis.tag_info if "task" in tag or "todo" in tag]
            assert len(task_tags) > 0
            
            # Should suggest merging todo/tasks
            similar_tags = await librarian._tag_manager.find_similar_tags(
                ["todo", "tasks", "task", "to-do"]
            )
            assert len(similar_tags) > 0


class TestBackupAndRecovery:
    """Test backup and recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_backup_before_major_changes(self, test_vault, test_config):
        """Test creating backups before major operations."""
        async with ObsidianLibrarian(test_config) as librarian:
            from obsidian_librarian.services.backup import BackupService
            backup_service = BackupService(librarian.vault, test_config)
            
            # Create backup before curation
            backup_id = await backup_service.create_backup("Before curation")
            assert backup_id is not None
            
            # Get initial state
            initial_notes = list(test_vault.glob("**/*.md"))
            initial_count = len(initial_notes)
            
            # Run curation with all features
            curation_results = await librarian.curate_content(
                check_duplicates=True,
                manage_tags=True,
                organize_files=True,
                dry_run=False
            )
            
            # Check changes were made
            current_notes = list(test_vault.glob("**/*.md"))
            current_count = len(current_notes)
            
            # If duplicates were merged, count should decrease
            if curation_results["duplicates_processed"] > 0:
                assert current_count <= initial_count
            
            # List available backups
            backups = await backup_service.list_backups()
            assert len(backups) > 0
            assert any(b["id"] == backup_id for b in backups)
            
            # Restore if needed
            if current_count < initial_count:
                await backup_service.restore_backup(backup_id)
                
                # Verify restoration
                restored_notes = list(test_vault.glob("**/*.md"))
                assert len(restored_notes) == initial_count


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])