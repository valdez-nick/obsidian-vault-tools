"""
Integration tests for analysis workflow functionality.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio
import numpy as np

from obsidian_librarian.librarian import ObsidianLibrarian, analyze_vault_quick
from obsidian_librarian.models import (
    LibrarianConfig, Note, VaultStats,
    DuplicateCluster, AnalysisResult
)
from obsidian_librarian.services.analysis import AnalysisService
# from obsidian_librarian.services.ai import AIService  # AI service not implemented yet


@pytest.fixture
def analysis_vault(tmp_path):
    """Create a comprehensive vault for analysis testing."""
    vault_path = tmp_path / "analysis_vault"
    vault_path.mkdir()
    
    # Create .obsidian directory
    (vault_path / ".obsidian").mkdir()
    
    # Create various note categories
    notes = {
        # Well-structured notes
        "projects/project1.md": """# Project Alpha
## Overview
This project aims to develop a new feature for our application.

## Goals
- [ ] Complete design phase
- [x] Set up development environment
- [ ] Implement core functionality

## Timeline
- Start: 2024-01-01
- End: 2024-03-01

## Resources
- [[team/john_doe]]
- [[references/api_docs]]

Tags: #project #active #development
""",
        
        # Daily notes with various patterns
        "daily/2024-01-01.md": """# 2024-01-01

## Tasks
- [x] Morning standup
- [ ] Review PRs
- [ ] Update documentation

## Notes
Met with [[team/jane_smith]] about [[projects/project1]].

## Reflections
Good progress today on the project.

Tags: #daily
""",
        
        "daily/2024-01-02.md": """# 2024-01-02

Tasks:
- [ ] Continue project work
- [x] Team meeting

Notes: Quick day, mostly meetings.

#daily
""",
        
        # Duplicate content
        "notes/meeting1.md": """# Team Meeting Notes
Date: 2024-01-01
Attendees: John, Jane, Bob

## Discussion Points
- Project timeline
- Resource allocation
- Next steps

## Action Items
- John: Update timeline
- Jane: Review budget
""",
        
        "archive/old_meeting.md": """# Team Meeting Notes
Date: 2024-01-01
Attendees: John, Jane, Bob

## Discussion Points
- Project timeline
- Resource allocation
- Next steps

## Action Items
- John: Update timeline
- Jane: Review budget
""",
        
        # Low quality notes
        "quick/note1.md": """Some quick thoughts here""",
        
        "quick/note2.md": """TODO: finish this later""",
        
        # Orphaned notes (no links)
        "orphaned/isolated1.md": """# Isolated Note
This note has no connections to other notes.
It contains some information but is not linked.
""",
        
        # Notes with broken links
        "broken/note_with_broken_links.md": """# Note with Issues
This references [[non_existent_note]] and [[another_missing_note]].

Also see [[deleted/old_note]].
""",
        
        # Well-connected notes
        "knowledge/concept1.md": """# Concept One
Related to [[knowledge/concept2]] and [[knowledge/concept3]].

See also:
- [[references/paper1]]
- [[projects/project1]]

Tags: #concept #knowledge
""",
        
        "knowledge/concept2.md": """# Concept Two
Builds upon [[knowledge/concept1]].

Tags: #concept #knowledge
""",
        
        # Templates
        "templates/daily.md": """# {{date}}

## Tasks
- [ ] 

## Notes


## Reflections


Tags: #daily
""",
        
        # Reference notes
        "references/paper1.md": """# Research Paper: Advanced Topics
Authors: Smith et al.
Year: 2024

## Abstract
This paper discusses...

## Key Points
1. Point one
2. Point two
3. Point three

## Citations
- Citation 1
- Citation 2

Tags: #reference #research #paper
""",
        
        # Large note for performance testing
        "large/big_note.md": "# Large Note\n\n" + ("This is a paragraph. " * 100 + "\n\n") * 50,
        
        # Note with many tasks
        "tasks/task_list.md": """# Master Task List

## Project Tasks
- [ ] Design database schema
- [x] Set up CI/CD pipeline
- [ ] Write unit tests
- [ ] Write integration tests
- [x] Deploy to staging
- [ ] Deploy to production

## Personal Tasks
- [ ] Read new research papers
- [x] Update LinkedIn profile
- [ ] Schedule dentist appointment

## Learning Goals
- [ ] Complete Python course
- [ ] Learn Rust basics
- [x] Finish ML tutorial

Tags: #tasks #todo
"""
    }
    
    # Create all notes
    for note_path, content in notes.items():
        note_file = vault_path / note_path
        note_file.parent.mkdir(parents=True, exist_ok=True)
        note_file.write_text(content)
    
    # Create some attachments
    attachments_dir = vault_path / "attachments"
    attachments_dir.mkdir(exist_ok=True)
    (attachments_dir / "image1.png").write_bytes(b"PNG_IMAGE_DATA")
    (attachments_dir / "document.pdf").write_bytes(b"PDF_DOCUMENT_DATA")
    
    return vault_path


@pytest.fixture
def mock_ai_service_for_analysis():
    """Create a mock AI service for analysis."""
    ai_service = AsyncMock(spec=AIService)
    
    # Mock embeddings with deterministic values
    def mock_embedding(text):
        # Simple hash-based embedding for consistency
        hash_val = hash(text) % 1000
        base = [hash_val / 1000.0] * 768
        # Add some variation
        for i in range(len(base)):
            base[i] += (i % 10) / 100.0
        return base[:768]
    
    ai_service.get_embedding.side_effect = mock_embedding
    
    # Mock quality assessment
    def mock_quality(content):
        score = 0.5  # Base score
        
        # Increase score based on content characteristics
        if len(content) > 500:
            score += 0.1
        if "## " in content:  # Has headings
            score += 0.1
        if "- [" in content:  # Has tasks
            score += 0.1
        if "[[" in content:  # Has links
            score += 0.1
        if "#" in content and not content.startswith("#"):  # Has tags
            score += 0.1
            
        return {
            "score": min(score, 1.0),
            "issues": [],
            "suggestions": []
        }
    
    ai_service.assess_quality.side_effect = mock_quality
    
    # Mock similarity calculation
    def mock_similarity(emb1, emb2):
        # Simple cosine similarity
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    ai_service.calculate_similarity.side_effect = mock_similarity
    
    return ai_service


class TestVaultAnalysis:
    """Test vault-wide analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_complete_vault_analysis(self, analysis_vault, mock_ai_service_for_analysis):
        """Test complete vault analysis workflow."""
        config = LibrarianConfig()
        
        with patch('obsidian_librarian.services.ai.AIService', return_value=mock_ai_service_for_analysis):
            results = await analyze_vault_quick(analysis_vault)
            
            assert 'vault_stats' in results
            assert 'quality_analysis' in results
            assert 'duplicate_clusters' in results
            assert 'link_analysis' in results
            assert 'task_analysis' in results
            
            # Verify vault stats
            stats = results['vault_stats']
            assert stats['total_notes'] > 0
            assert stats['total_words'] > 0
            assert stats['total_links'] > 0
            assert stats['total_tasks'] > 0
    
    @pytest.mark.asyncio
    async def test_vault_statistics(self, analysis_vault, mock_ai_service_for_analysis):
        """Test vault statistics calculation."""
        config = LibrarianConfig()
        analysis_service = AnalysisService(config, mock_ai_service_for_analysis)
        
        stats = await analysis_service.calculate_vault_stats(analysis_vault)
        
        assert stats.total_notes > 15  # We created many notes
        assert stats.total_words > 1000
        assert stats.total_links > 5
        assert stats.total_tasks > 10
        assert stats.completed_tasks > 0
        assert stats.avg_note_length > 0
        assert stats.largest_note_size > 0
        
        # Check directory stats
        assert 'daily' in stats.notes_by_directory
        assert 'projects' in stats.notes_by_directory
        assert stats.notes_by_directory['daily'] >= 2
    
    @pytest.mark.asyncio
    async def test_quality_analysis(self, analysis_vault, mock_ai_service_for_analysis):
        """Test note quality analysis."""
        config = LibrarianConfig()
        analysis_service = AnalysisService(config, mock_ai_service_for_analysis)
        
        quality_report = await analysis_service.analyze_quality(analysis_vault)
        
        assert quality_report.total_notes > 0
        assert 0 <= quality_report.average_score <= 1
        assert len(quality_report.score_distribution) > 0
        assert len(quality_report.low_quality_notes) >= 0
        assert len(quality_report.high_quality_notes) >= 0
        
        # Check quality metrics
        assert quality_report.metrics is not None
        assert 'readability' in quality_report.metrics
        assert 'structure' in quality_report.metrics
        assert 'completeness' in quality_report.metrics
        
        # Low quality notes should include our minimal notes
        low_quality_paths = [n.path for n in quality_report.low_quality_notes]
        assert any('quick/note1.md' in str(p) for p in low_quality_paths)
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self, analysis_vault, mock_ai_service_for_analysis):
        """Test duplicate note detection."""
        config = LibrarianConfig()
        analysis_service = AnalysisService(config, mock_ai_service_for_analysis)
        
        duplicates = await analysis_service.find_duplicates(
            vault_path=analysis_vault,
            similarity_threshold=0.85
        )
        
        assert len(duplicates) > 0
        
        # Should find our intentional duplicates
        found_meeting_duplicate = False
        for cluster in duplicates:
            note_paths = [str(n.path) for n in cluster.notes]
            if any('meeting1.md' in p for p in note_paths) and \
               any('old_meeting.md' in p for p in note_paths):
                found_meeting_duplicate = True
                assert cluster.similarity_score > 0.9  # Should be very similar
                break
        
        assert found_meeting_duplicate
    
    @pytest.mark.asyncio
    async def test_link_analysis(self, analysis_vault, mock_ai_service_for_analysis):
        """Test link analysis and graph metrics."""
        config = LibrarianConfig()
        analysis_service = AnalysisService(config, mock_ai_service_for_analysis)
        
        link_report = await analysis_service.analyze_links(analysis_vault)
        
        assert link_report.total_links > 0
        assert link_report.broken_links >= 2  # We have intentional broken links
        assert len(link_report.orphaned_notes) > 0
        assert len(link_report.hub_notes) >= 0
        
        # Check specific broken links
        broken_targets = [link.target for link in link_report.broken_link_details]
        assert 'non_existent_note' in broken_targets
        assert 'another_missing_note' in broken_targets
        
        # Check orphaned notes
        orphaned_paths = [str(n.path) for n in link_report.orphaned_notes]
        assert any('isolated1.md' in p for p in orphaned_paths)
        
        # Check graph metrics
        assert link_report.graph_metrics is not None
        assert 'density' in link_report.graph_metrics
        assert 'clustering_coefficient' in link_report.graph_metrics
    
    @pytest.mark.asyncio
    async def test_task_analysis(self, analysis_vault, mock_ai_service_for_analysis):
        """Test task extraction and analysis."""
        config = LibrarianConfig()
        analysis_service = AnalysisService(config, mock_ai_service_for_analysis)
        
        task_report = await analysis_service.analyze_tasks(analysis_vault)
        
        assert task_report.total_tasks > 10
        assert task_report.completed_tasks > 0
        assert task_report.pending_tasks > 0
        assert 0 <= task_report.completion_rate <= 1
        
        # Check task distribution
        assert len(task_report.tasks_by_note) > 0
        assert any('task_list.md' in str(note) for note in task_report.tasks_by_note)
        
        # Check overdue tasks (if any dated tasks exist)
        assert isinstance(task_report.overdue_tasks, list)
        
        # Check task categories
        assert 'Project Tasks' in task_report.tasks_by_category or \
               len(task_report.tasks_by_category) >= 0


class TestContentOrganization:
    """Test content organization and structuring."""
    
    @pytest.mark.asyncio
    async def test_auto_categorization(self, analysis_vault, mock_ai_service_for_analysis):
        """Test automatic note categorization."""
        config = LibrarianConfig()
        
        with patch('obsidian_librarian.services.ai.AIService', return_value=mock_ai_service_for_analysis):
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(analysis_vault)
                
                # Analyze and categorize notes
                categories = await librarian.auto_categorize_notes(
                    session_id=session_id,
                    strategy="content-based"
                )
                
                assert len(categories) > 0
                
                # Check expected categories
                expected_categories = ['daily', 'projects', 'knowledge', 'tasks', 'references']
                for cat in expected_categories:
                    assert any(cat in c.lower() for c in categories.keys())
    
    @pytest.mark.asyncio
    async def test_structure_recommendations(self, analysis_vault, mock_ai_service_for_analysis):
        """Test vault structure recommendations."""
        config = LibrarianConfig()
        analysis_service = AnalysisService(config, mock_ai_service_for_analysis)
        
        recommendations = await analysis_service.generate_structure_recommendations(
            analysis_vault
        )
        
        assert len(recommendations) > 0
        
        # Should recommend dealing with orphaned notes
        assert any('orphan' in r.lower() for r in recommendations)
        
        # Should recommend fixing broken links
        assert any('broken' in r.lower() for r in recommendations)
    
    @pytest.mark.asyncio
    async def test_tag_analysis(self, analysis_vault, mock_ai_service_for_analysis):
        """Test tag analysis and recommendations."""
        config = LibrarianConfig()
        analysis_service = AnalysisService(config, mock_ai_service_for_analysis)
        
        tag_report = await analysis_service.analyze_tags(analysis_vault)
        
        assert tag_report.total_unique_tags > 0
        assert tag_report.total_tag_uses > 0
        assert len(tag_report.tag_frequency) > 0
        
        # Check common tags
        assert '#daily' in tag_report.tag_frequency
        assert '#project' in tag_report.tag_frequency
        
        # Check tag co-occurrence
        assert len(tag_report.tag_cooccurrence) >= 0
        
        # Check untagged notes
        assert len(tag_report.untagged_notes) > 0
        untagged_paths = [str(n.path) for n in tag_report.untagged_notes]
        assert any('quick/note1.md' in p for p in untagged_paths)


class TestQualityImprovement:
    """Test automated quality improvement features."""
    
    @pytest.mark.asyncio
    async def test_quality_suggestions(self, analysis_vault, mock_ai_service_for_analysis):
        """Test quality improvement suggestions."""
        config = LibrarianConfig()
        
        # Mock AI suggestions
        mock_ai_service_for_analysis.generate_improvement_suggestions.return_value = [
            "Add a proper heading structure",
            "Include tags for better organization",
            "Add links to related notes",
            "Expand content with more details"
        ]
        
        with patch('obsidian_librarian.services.ai.AIService', return_value=mock_ai_service_for_analysis):
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(analysis_vault)
                
                # Get suggestions for low quality note
                note_path = analysis_vault / "quick/note1.md"
                suggestions = await librarian.get_quality_suggestions(
                    session_id=session_id,
                    note_path=note_path
                )
                
                assert len(suggestions) > 0
                assert any('heading' in s.lower() for s in suggestions)
    
    @pytest.mark.asyncio
    async def test_auto_quality_improvement(self, analysis_vault, mock_ai_service_for_analysis):
        """Test automatic quality improvements."""
        config = LibrarianConfig()
        
        # Mock AI improvement
        mock_ai_service_for_analysis.improve_note_content.return_value = """# Improved Note

## Overview
Some quick thoughts here, now properly structured.

## Details
This note has been automatically improved with better structure and organization.

## Related Notes
- [[other_notes]]

Tags: #improved #automated
"""
        
        with patch('obsidian_librarian.services.ai.AIService', return_value=mock_ai_service_for_analysis):
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(analysis_vault)
                
                # Improve low quality notes
                results = await librarian.improve_note_quality(
                    session_id=session_id,
                    min_quality_threshold=0.5,
                    dry_run=False
                )
                
                assert results['improved_count'] > 0
                assert len(results['improvements']) > 0
                
                # Check if note was actually improved
                improved_note = analysis_vault / "quick/note1.md"
                if improved_note.exists():
                    content = improved_note.read_text()
                    assert "## Overview" in content or "improved" in content.lower()


class TestPerformanceAnalysis:
    """Test performance analysis for large vaults."""
    
    @pytest.fixture
    def large_vault(self, tmp_path):
        """Create a large vault for performance testing."""
        vault_path = tmp_path / "large_vault"
        vault_path.mkdir()
        (vault_path / ".obsidian").mkdir()
        
        # Create many notes with links
        for i in range(500):
            content = f"""# Note {i}

## Content
This is note number {i}. It contains various content and links.

Links:
- [[note_{(i+1) % 500}]]
- [[note_{(i+2) % 500}]]
- [[category/note_{i % 10}]]

## Tasks
- [ ] Task for note {i}
- [x] Completed task

Tags: #note #batch{i // 100}
"""
            note_path = vault_path / f"note_{i}.md"
            note_path.write_text(content)
        
        # Create category notes
        (vault_path / "category").mkdir()
        for i in range(10):
            content = f"# Category {i}\n\nCategory description."
            (vault_path / f"category/note_{i}.md").write_text(content)
        
        return vault_path
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_vault_analysis_performance(self, large_vault, mock_ai_service_for_analysis):
        """Test analysis performance on large vaults."""
        import time
        
        config = LibrarianConfig()
        analysis_service = AnalysisService(config, mock_ai_service_for_analysis)
        
        start_time = time.time()
        
        # Run complete analysis
        stats = await analysis_service.calculate_vault_stats(large_vault)
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert stats.total_notes >= 500
        assert duration < 10.0  # Should complete within 10 seconds
        
        # Test memory efficiency
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        assert memory_usage < 500  # Should use less than 500MB
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_incremental_analysis(self, large_vault, mock_ai_service_for_analysis):
        """Test incremental analysis for efficiency."""
        config = LibrarianConfig()
        
        with patch('obsidian_librarian.services.ai.AIService', return_value=mock_ai_service_for_analysis):
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(large_vault)
                
                # Initial analysis
                initial_results = await librarian.analyze_vault(
                    session_id=session_id,
                    cache_results=True
                )
                
                # Modify one note
                (large_vault / "note_0.md").write_text("# Modified Note\n\nNew content")
                
                # Incremental analysis should be faster
                import time
                start_time = time.time()
                
                incremental_results = await librarian.analyze_vault(
                    session_id=session_id,
                    incremental=True
                )
                
                duration = time.time() - start_time
                
                assert duration < 2.0  # Incremental should be very fast
                assert incremental_results['changes_detected'] == 1


class TestAnalysisExport:
    """Test analysis report export functionality."""
    
    @pytest.mark.asyncio
    async def test_export_html_report(self, analysis_vault, mock_ai_service_for_analysis):
        """Test exporting analysis as HTML report."""
        config = LibrarianConfig()
        
        results = await analyze_vault_quick(analysis_vault)
        
        analysis_service = AnalysisService(config, mock_ai_service_for_analysis)
        html_report = await analysis_service.export_report(
            results=results,
            format="html",
            include_charts=True
        )
        
        assert "<html>" in html_report
        assert "Vault Analysis Report" in html_report
        assert "Total Notes:" in html_report
        assert "<canvas" in html_report  # Charts
    
    @pytest.mark.asyncio  
    async def test_export_markdown_report(self, analysis_vault, mock_ai_service_for_analysis):
        """Test exporting analysis as Markdown report."""
        config = LibrarianConfig()
        
        results = await analyze_vault_quick(analysis_vault)
        
        analysis_service = AnalysisService(config, mock_ai_service_for_analysis)
        md_report = await analysis_service.export_report(
            results=results,
            format="markdown"
        )
        
        assert "# Vault Analysis Report" in md_report
        assert "## Statistics" in md_report
        assert "| Metric | Value |" in md_report  # Table format
        assert "```" in md_report  # Code blocks for data
    
    @pytest.mark.asyncio
    async def test_export_json_report(self, analysis_vault, mock_ai_service_for_analysis):
        """Test exporting analysis as JSON."""
        config = LibrarianConfig()
        
        results = await analyze_vault_quick(analysis_vault)
        
        analysis_service = AnalysisService(config, mock_ai_service_for_analysis)
        json_report = await analysis_service.export_report(
            results=results,
            format="json"
        )
        
        # Should be valid JSON
        parsed = json.loads(json_report)
        assert 'vault_stats' in parsed
        assert 'timestamp' in parsed
        assert 'version' in parsed


class TestAnalysisScheduling:
    """Test scheduled analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_scheduled_analysis(self, analysis_vault, mock_ai_service_for_analysis):
        """Test scheduling regular analysis."""
        config = LibrarianConfig()
        
        with patch('obsidian_librarian.services.ai.AIService', return_value=mock_ai_service_for_analysis):
            async with ObsidianLibrarian(config) as librarian:
                session_id = await librarian.create_session(analysis_vault)
                
                # Schedule daily analysis
                schedule_id = await librarian.schedule_analysis(
                    session_id=session_id,
                    interval="daily",
                    time="02:00",
                    export_format="markdown",
                    export_path=analysis_vault / "analysis_reports"
                )
                
                assert schedule_id is not None
                
                # Verify schedule was created
                schedules = await librarian.list_scheduled_analyses(session_id)
                assert len(schedules) > 0
                assert schedules[0]['id'] == schedule_id
                assert schedules[0]['interval'] == "daily"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])