"""
Integration tests for the main Librarian orchestrator.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, patch, MagicMock

from obsidian_librarian.librarian import ObsidianLibrarian
from obsidian_librarian.models import LibrarianConfig, ResearchQuery


@pytest.mark.integration
class TestLibrarianIntegration:
    """Test the main Librarian orchestrator."""
    
    @pytest.fixture
    async def librarian_setup(self):
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
            
            config = LibrarianConfig()
            
            librarian = ObsidianLibrarian(config)
            await librarian.initialize()
            session_id = await librarian.create_session(vault_path)
            
            # Store both librarian and session_id for tests
            yield librarian, session_id
            
            await librarian.close()
    
    @pytest.mark.asyncio
    async def test_librarian_initialization(self, librarian_setup):
        """Test librarian initializes all components."""
        librarian, session_id = librarian_setup
        
        # Check that librarian is initialized
        assert librarian.is_initialized
        assert session_id in librarian.sessions
        
        # Check session has required services
        session = librarian.sessions[session_id]
        assert session.vault is not None
        assert session.research_service is not None
        assert session.analysis_service is not None
        assert session.template_service is not None
    
    @pytest.mark.asyncio
    async def test_vault_analysis(self, librarian_setup):
        """Test vault analysis functionality."""
        librarian, session_id = librarian_setup
        
        # Run analysis
        results = []
        async for result in librarian.analyze_vault(session_id, find_duplicates=True):
            results.append(result)
        
        # Check that analysis completed
        assert any(r.get('type') == 'complete' for r in results)
        complete_result = next(r for r in results if r.get('type') == 'complete')
        assert 'analysis_results' in complete_result['data']
    
    @pytest.mark.asyncio
    async def test_session_status(self, librarian_setup):
        """Test getting session status."""
        librarian, session_id = librarian_setup
        
        status = await librarian.get_session_status(session_id)
        
        assert status['session_id'] == session_id
        assert 'vault_path' in status
        assert 'vault_stats' in status
        assert status['vault_stats'] is not None
    
    @pytest.mark.asyncio
    async def test_global_stats(self, librarian_setup):
        """Test getting global statistics."""
        librarian, session_id = librarian_setup
        
        stats = await librarian.get_global_stats()
        
        assert 'total_sessions' in stats
        assert 'active_sessions' in stats
        assert 'global_stats' in stats
        assert session_id in stats['active_sessions']
    
    @pytest.mark.asyncio
    async def test_template_application(self, librarian_setup):
        """Test applying templates to notes."""
        librarian, session_id = librarian_setup
        
        # Get session for direct vault access
        session = librarian.sessions[session_id]
        
        # Create a note without proper structure
        note_id = await session.vault.create_note(
            Path("unstructured.md"),
            "Some random content about transformers",
            {}
        )
        
        # Apply templates (auto-detect mode)
        result = await librarian.apply_templates(session_id, note_ids=[note_id], auto_detect=True)
        
        assert 'total_applications' in result
        assert result['total_applications'] >= 0  # May be 0 if no templates match
    
    @pytest.mark.asyncio
    async def test_content_curation(self, librarian_setup):
        """Test content curation functionality."""
        librarian, session_id = librarian_setup
        
        result = await librarian.curate_content(
            session_id,
            remove_duplicates=False,  # Don't actually remove
            improve_quality=True,
            organize_structure=True
        )
        
        assert 'duplicates_processed' in result
        assert 'quality_improvements' in result
        assert 'structure_improvements' in result
        assert isinstance(result['errors'], list)