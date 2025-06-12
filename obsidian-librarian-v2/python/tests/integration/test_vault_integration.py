"""
Integration tests for Vault operations.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from obsidian_librarian import Vault, Note
from obsidian_librarian.models import VaultConfig


@pytest.mark.integration
class TestVaultIntegration:
    """Test Vault operations with real file system."""
    
    @pytest.fixture
    async def vault(self):
        """Create a temporary vault for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir) / "test_vault"
            vault_path.mkdir()
            
            # Create basic structure
            (vault_path / ".obsidian").mkdir()
            (vault_path / "Templates").mkdir()
            (vault_path / "Notes").mkdir()
            
            # Create test notes
            (vault_path / "note1.md").write_text("""---
title: Test Note 1
tags: [test, sample]
created: 2024-01-01
---

# Test Note 1

This is a test note with a [[note2|link to note 2]].

## Tasks
- [ ] Task 1
- [x] Completed task
""")
            
            (vault_path / "note2.md").write_text("""---
title: Test Note 2
tags: [test]
---

# Test Note 2

This links back to [[note1]].
""")
            
            (vault_path / "Notes" / "nested.md").write_text("""# Nested Note

This is in a subdirectory.
""")
            
            # Initialize vault
            config = VaultConfig(
                enable_file_watching=False,  # Disable for tests
                cache_size=100,
            )
            vault = Vault(vault_path, config)
            await vault.initialize()
            
            yield vault
            
            await vault.close()
    
    @pytest.mark.asyncio
    async def test_vault_initialization(self, vault):
        """Test vault initializes correctly."""
        stats = await vault.get_stats()
        assert stats.total_notes >= 3
        assert stats.total_words > 0
        assert stats.total_links >= 2
    
    @pytest.mark.asyncio
    async def test_get_all_notes(self, vault):
        """Test retrieving all notes."""
        notes = await vault.get_all_notes()
        assert len(notes) >= 3
        
        # Check note properties
        note_titles = {note.metadata.get("title") for note in notes if note.metadata}
        assert "Test Note 1" in note_titles
        assert "Test Note 2" in note_titles
    
    @pytest.mark.asyncio
    async def test_search_notes(self, vault):
        """Test searching notes."""
        # Search by content
        results = await vault.search_notes("test note")
        assert len(results) >= 2
        
        # Search by tag
        notes_with_tag = await vault.get_notes_by_tag("test")
        assert len(notes_with_tag) >= 2
    
    @pytest.mark.asyncio
    async def test_note_links(self, vault):
        """Test link detection and backlinks."""
        # Find note1
        notes = await vault.get_all_notes()
        note1 = next(n for n in notes if "Test Note 1" in n.content)
        
        # Check outgoing links
        linked_notes = await vault.get_linked_notes(note1.id)
        assert len(linked_notes) >= 1
        
        # Check backlinks
        note2 = next(n for n in notes if "Test Note 2" in n.content)
        backlinks = await vault.get_backlinks(note2.id)
        assert len(backlinks) >= 1
    
    @pytest.mark.asyncio
    async def test_create_update_delete(self, vault):
        """Test CRUD operations."""
        # Create
        note_id = await vault.create_note(
            Path("test_create.md"),
            "# Test Create\n\nNew note content",
            {"title": "Created Note", "tags": ["new"]}
        )
        assert note_id
        
        # Read
        note = await vault.get_note(note_id)
        assert note is not None
        assert "Test Create" in note.content
        
        # Update
        success = await vault.update_note(
            note_id,
            "# Test Create\n\nUpdated content",
            {"title": "Updated Note"}
        )
        assert success
        
        # Verify update
        updated = await vault.get_note(note_id)
        assert "Updated content" in updated.content
        
        # Delete
        success = await vault.delete_note(note_id)
        assert success
        
        # Verify deletion
        deleted = await vault.get_note(note_id)
        assert deleted is None
    
    @pytest.mark.asyncio
    async def test_orphaned_notes(self, vault):
        """Test finding orphaned notes."""
        # Create an orphaned note
        note_id = await vault.create_note(
            Path("orphan.md"),
            "# Orphaned Note\n\nNo links here.",
            {}
        )
        
        orphans = await vault.get_orphaned_notes()
        assert note_id in orphans
    
    @pytest.mark.asyncio
    async def test_vault_backup(self, vault):
        """Test vault backup functionality."""
        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir) / "backup"
            success = await vault.backup(backup_path)
            assert success
            
            # Check backup exists
            assert backup_path.exists()
            assert (backup_path / "note1.md").exists()
            assert (backup_path / "note2.md").exists()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, vault):
        """Test concurrent vault operations."""
        # Create multiple notes concurrently
        tasks = []
        for i in range(10):
            task = vault.create_note(
                Path(f"concurrent_{i}.md"),
                f"# Concurrent Note {i}\n\nContent {i}",
                {"index": i}
            )
            tasks.append(task)
        
        note_ids = await asyncio.gather(*tasks)
        assert len(note_ids) == 10
        
        # Read them concurrently
        read_tasks = [vault.get_note(nid) for nid in note_ids]
        notes = await asyncio.gather(*read_tasks)
        
        assert all(note is not None for note in notes)
        assert all(f"Concurrent Note {i}" in notes[i].content for i in range(10))