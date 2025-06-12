"""
Integration tests for the main Librarian orchestrator.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, patch, MagicMock

from obsidian_librarian.librarian import Librarian, LibrarianConfig
from obsidian_librarian.models import CuratorTask, ResearchQuery


@pytest.mark.integration
class TestLibrarianIntegration:
    """Test the main Librarian orchestrator."""
    
    @pytest.fixture
    async def librarian(self):
        """Create a Librarian instance for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir) / "test_vault"
            vault_path.mkdir()
            
            # Create basic vault structure
            (vault_path / ".obsidian").mkdir()
            (vault_path / "Templates").mkdir()
            (vault_path / "Research Library").mkdir()
            
            # Add some test notes
            (vault_path / "note1.md").write_text("""# Note 1
This is a test note about machine learning.
Tags: #ml #ai
""")
            
            (vault_path / "note2.md").write_text("""# Note 2
This is similar to note 1, discussing ML concepts.
Tags: #ml #duplicate
""")
            
            (vault_path / "Templates" / "research.md").write_text("""---
title: {{title}}
date: {{date}}
source: {{source}}
tags: [research, {{topic}}]
---

# {{title}}

## Summary
{{summary}}

## Key Points
{{key_points}}

## References
{{references}}
""")
            
            config = LibrarianConfig(
                vault_path=vault_path,
                enable_auto_organize=True,
                enable_duplicate_detection=True,
                enable_template_application=True,
                research_library_path=vault_path / "Research Library",
            )
            
            librarian = Librarian(config)
            await librarian.initialize()
            
            yield librarian
            
            await librarian.shutdown()
    
    @pytest.mark.asyncio
    async def test_librarian_initialization(self, librarian):
        """Test librarian initializes all components."""
        assert librarian.vault is not None
        assert librarian.research_service is not None
        assert librarian.analysis_service is not None
        assert librarian.template_service is not None
        
        # Check vault is loaded
        stats = await librarian.vault.get_stats()
        assert stats.total_notes >= 2
    
    @pytest.mark.asyncio
    async def test_curator_task_execution(self, librarian):
        """Test executing curator tasks."""
        # Organize notes task
        task = CuratorTask(
            type="organize",
            description="Organize notes by topic",
        )
        
        result = await librarian.execute_curator_task(task)
        assert result["status"] == "completed"
        assert "organized" in result.get("message", "").lower()
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self, librarian):
        """Test duplicate detection functionality."""
        # Create similar notes
        await librarian.vault.create_note(
            Path("duplicate1.md"),
            "# Machine Learning\nThis is about neural networks and deep learning.",
            {"tags": ["ml"]}
        )
        
        await librarian.vault.create_note(
            Path("duplicate2.md"),
            "# ML Concepts\nThis covers neural networks and deep learning topics.",
            {"tags": ["ml"]}
        )
        
        # Detect duplicates
        task = CuratorTask(type="detect_duplicates")
        result = await librarian.execute_curator_task(task)
        
        assert result["status"] == "completed"
        assert "duplicates" in result
        assert len(result["duplicates"]) > 0
    
    @pytest.mark.asyncio
    @patch('obsidian_librarian.services.research.WebScraper')
    async def test_research_and_organize(self, mock_scraper, librarian):
        """Test research and organization flow."""
        # Mock web scraper
        mock_scraper_instance = AsyncMock()
        async def mock_scrape(url):
            return {
                "url": url,
                "title": "Transformer Architecture Explained",
                "content": "Detailed explanation of transformer models...",
                "metadata": {
                    "author": "AI Researcher",
                    "date": "2024-01-01",
                },
            }
        
        mock_scraper_instance.scrape.side_effect = mock_scrape
        mock_scraper.return_value = mock_scraper_instance
        
        # Execute research
        query = ResearchQuery(
            query="transformer architecture",
            sources=["https://arxiv.org"],
            max_results=3,
        )
        
        results = await librarian.research_and_organize(query)
        
        assert "research_results" in results
        assert "organized_notes" in results
        assert len(results["research_results"]) > 0
        
        # Check that notes were created in Research Library
        research_notes = list(librarian.config.research_library_path.glob("**/*.md"))
        assert len(research_notes) > 0
    
    @pytest.mark.asyncio
    async def test_template_application(self, librarian):
        """Test applying templates to notes."""
        # Create a note without proper structure
        note_id = await librarian.vault.create_note(
            Path("unstructured.md"),
            "Some random content about transformers",
            {}
        )
        
        # Apply template
        task = CuratorTask(
            type="apply_template",
            target_notes=[note_id],
            template_name="research",
        )
        
        result = await librarian.execute_curator_task(task)
        assert result["status"] == "completed"
        
        # Check note was updated
        note = await librarian.vault.get_note(note_id)
        assert "title:" in note.content
        assert "tags:" in note.content
    
    @pytest.mark.asyncio
    async def test_git_backup_integration(self, librarian):
        """Test Git backup during major changes."""
        with patch('obsidian_librarian.librarian.git') as mock_git:
            # Mock git operations
            mock_repo = MagicMock()
            mock_git.Repo.return_value = mock_repo
            mock_repo.is_dirty.return_value = True
            
            # Make major changes
            for i in range(10):
                await librarian.vault.create_note(
                    Path(f"major_change_{i}.md"),
                    f"# Major Change {i}\nContent",
                    {}
                )
            
            # Trigger backup
            await librarian.backup_if_needed()
            
            # Check git operations were called
            assert mock_repo.index.add.called
            assert mock_repo.index.commit.called
    
    @pytest.mark.asyncio
    async def test_concurrent_curator_tasks(self, librarian):
        """Test running multiple curator tasks concurrently."""
        tasks = [
            CuratorTask(type="organize"),
            CuratorTask(type="detect_duplicates"),
            CuratorTask(type="validate_links"),
        ]
        
        # Execute tasks concurrently
        results = await asyncio.gather(*[
            librarian.execute_curator_task(task) 
            for task in tasks
        ])
        
        assert len(results) == 3
        assert all(r["status"] == "completed" for r in results)
    
    @pytest.mark.asyncio
    async def test_event_handling(self, librarian):
        """Test vault event handling."""
        events_received = []
        
        async def event_handler(event):
            events_received.append(event)
        
        librarian.vault.add_event_callback("note_created", event_handler)
        
        # Create a note
        note_id = await librarian.vault.create_note(
            Path("event_test.md"),
            "# Event Test",
            {}
        )
        
        # Give time for event to propagate
        await asyncio.sleep(0.1)
        
        assert len(events_received) > 0
        assert note_id in str(events_received[0])
    
    @pytest.mark.asyncio
    async def test_ai_query_processing(self, librarian):
        """Test AI-powered query processing."""
        with patch('obsidian_librarian.ai.query_processor.QueryProcessor') as mock_processor:
            mock_instance = AsyncMock()
            mock_instance.process_query.return_value = {
                "intent": "find_notes",
                "entities": ["machine learning", "transformers"],
                "filters": {"tags": ["ml", "ai"]},
            }
            mock_processor.return_value = mock_instance
            
            response = await librarian.process_query(
                "Show me all my notes about machine learning and transformers"
            )
            
            assert "results" in response
            assert response["intent"] == "find_notes"
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, librarian):
        """Test performance monitoring during operations."""
        # Enable performance monitoring
        librarian.config.enable_performance_monitoring = True
        
        # Perform various operations
        tasks = []
        for i in range(20):
            task = librarian.vault.create_note(
                Path(f"perf_test_{i}.md"),
                f"# Performance Test {i}",
                {}
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Get performance metrics
        metrics = await librarian.get_performance_metrics()
        
        assert "operation_times" in metrics
        assert "memory_usage" in metrics
        assert metrics["total_operations"] >= 20