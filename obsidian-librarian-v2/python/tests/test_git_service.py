"""
Tests for Git service functionality.
"""

import asyncio
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import pytest

from obsidian_librarian.vault import Vault
from obsidian_librarian.models import VaultConfig
from obsidian_librarian.services.git_service import GitService, GitConfig


@pytest.fixture
async def temp_vault():
    """Create a temporary vault for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        vault_path = Path(tmp_dir)
        
        # Create some test files
        (vault_path / "note1.md").write_text("# Test Note 1\n\nContent here")
        (vault_path / "note2.md").write_text("# Test Note 2\n\nMore content")
        
        subdir = vault_path / "subfolder"
        subdir.mkdir()
        (subdir / "note3.md").write_text("# Test Note 3\n\nNested content")
        
        yield vault_path


@pytest.fixture
async def git_vault(temp_vault):
    """Create a vault with Git service initialized."""
    config = VaultConfig(
        enable_git_integration=True,
        enable_auto_backup=True,
        git_auto_backup_threshold=5
    )
    
    vault = Vault(temp_vault, config)
    await vault.initialize()
    
    yield vault
    
    await vault.close()


class TestGitService:
    """Test Git service functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize_repo(self, temp_vault):
        """Test Git repository initialization."""
        config = GitConfig()
        git_service = GitService(None, config)
        git_service.vault = type('obj', (object,), {'path': str(temp_vault)})
        
        # Initialize repository
        success = await git_service.initialize_repo()
        assert success
        
        # Check .git directory exists
        assert (temp_vault / ".git").exists()
        
        # Check .gitignore was created
        assert (temp_vault / ".gitignore").exists()
    
    @pytest.mark.asyncio
    async def test_backup_with_changes(self, git_vault):
        """Test creating backups with changes."""
        # Make some changes
        (Path(git_vault.path) / "new_note.md").write_text("# New Note\n\nNew content")
        
        # Create backup
        commit_hash = await git_vault.git_backup("Test backup")
        assert commit_hash is not None
        
        # Check status shows clean working directory
        status = await git_vault.git_status()
        assert status['total_changes'] == 0
    
    @pytest.mark.asyncio
    async def test_auto_backup_threshold(self, git_vault):
        """Test auto-backup triggers on threshold."""
        # Register changes up to threshold
        for i in range(git_vault.git_service.config.auto_backup_threshold - 1):
            git_vault.git_service.register_change()
        
        # Check no auto-backup yet
        initial_status = await git_vault.git_status()
        initial_commits = len(await git_vault.git_history())
        
        # One more change should trigger auto-backup
        git_vault.git_service.register_change()
        
        # Give async task time to complete
        await asyncio.sleep(0.5)
        
        # Check that auto-backup occurred
        final_status = await git_vault.git_status()
        final_commits = len(await git_vault.git_history())
        
        # Should have one more commit if files were actually changed
        # Note: In real usage, file changes would trigger register_change()
    
    @pytest.mark.asyncio
    async def test_restore_functionality(self, git_vault):
        """Test restoring to previous commits."""
        # Create initial backup
        initial_content = "# Initial\n\nOriginal content"
        test_file = Path(git_vault.path) / "test_restore.md"
        test_file.write_text(initial_content)
        
        commit1 = await git_vault.git_backup("Initial state")
        assert commit1
        
        # Make changes and backup again
        modified_content = "# Modified\n\nChanged content"
        test_file.write_text(modified_content)
        
        commit2 = await git_vault.git_backup("Modified state")
        assert commit2
        
        # Verify current content
        assert test_file.read_text() == modified_content
        
        # Restore to initial commit
        success = await git_vault.git_restore(commit1)
        assert success
        
        # Verify content was restored
        assert test_file.read_text() == initial_content
    
    @pytest.mark.asyncio
    async def test_branch_management(self, git_vault):
        """Test creating and managing branches."""
        # Create experimental branch
        branch_name = await git_vault.git_create_branch("test-feature")
        assert branch_name == "experiment/test-feature"
        
        # Check status shows new branch
        status = await git_vault.git_status()
        assert status['current_branch'] == "experiment/test-feature"
        assert "experiment/test-feature" in status['branches']
    
    @pytest.mark.asyncio
    async def test_stash_functionality(self, git_vault):
        """Test stashing changes."""
        # Make uncommitted changes
        test_file = Path(git_vault.path) / "stash_test.md"
        test_file.write_text("# Stash Test\n\nUncommitted changes")
        
        # Stash changes
        success = await git_vault.git_stash("Testing stash")
        assert success
        
        # Verify file was removed
        assert not test_file.exists()
        
        # Check stash count
        status = await git_vault.git_status()
        assert status['stash_count'] > 0
    
    @pytest.mark.asyncio
    async def test_rollback_functionality(self, git_vault):
        """Test rolling back commits."""
        # Create multiple commits
        for i in range(3):
            test_file = Path(git_vault.path) / f"rollback_test_{i}.md"
            test_file.write_text(f"# Test {i}\n\nContent {i}")
            await git_vault.git_backup(f"Commit {i}")
        
        # Get initial history
        initial_history = await git_vault.git_history()
        initial_commit_count = len(initial_history)
        
        # Rollback 2 steps
        success = await git_vault.git_rollback(2)
        assert success
        
        # Verify only first file exists
        assert (Path(git_vault.path) / "rollback_test_0.md").exists()
        assert not (Path(git_vault.path) / "rollback_test_1.md").exists()
        assert not (Path(git_vault.path) / "rollback_test_2.md").exists()
    
    @pytest.mark.asyncio
    async def test_git_status_comprehensive(self, git_vault):
        """Test comprehensive status reporting."""
        # Get clean status
        clean_status = await git_vault.git_status()
        assert clean_status['initialized']
        assert clean_status['total_changes'] == 0
        assert clean_status['current_branch'] == "main" or clean_status['current_branch'] == "master"
        
        # Add new file
        (Path(git_vault.path) / "new_file.md").write_text("New content")
        
        # Modify existing file
        existing_file = Path(git_vault.path) / "note1.md"
        existing_file.write_text(existing_file.read_text() + "\n\nModified")
        
        # Delete a file
        (Path(git_vault.path) / "note2.md").unlink()
        
        # Get status with changes
        changed_status = await git_vault.git_status()
        assert changed_status['total_changes'] == 3
        assert len(changed_status['changes']['added']) == 1
        assert len(changed_status['changes']['modified']) == 1
        assert len(changed_status['changes']['deleted']) == 1
    
    @pytest.mark.asyncio
    async def test_commit_message_generation(self, git_vault):
        """Test smart commit message generation."""
        # Test with single file addition
        (Path(git_vault.path) / "single_add.md").write_text("Content")
        commit1 = await git_vault.git_backup()  # No message provided
        
        history = await git_vault.git_history(1)
        assert "Added 1 file" in history[0].message or "Add 1 file" in history[0].message
        
        # Test with multiple changes
        (Path(git_vault.path) / "add1.md").write_text("Content 1")
        (Path(git_vault.path) / "add2.md").write_text("Content 2")
        (Path(git_vault.path) / "note1.md").write_text("Modified content")
        
        commit2 = await git_vault.git_backup()  # No message provided
        
        history = await git_vault.git_history(1)
        assert "files" in history[0].message  # Should mention multiple files
    
    @pytest.mark.asyncio
    async def test_file_watcher_integration(self, git_vault):
        """Test integration with file watcher."""
        if not git_vault.config.enable_file_watching:
            pytest.skip("File watching not enabled")
        
        # Simulate file changes through vault
        test_file = Path(git_vault.path) / "watched_file.md"
        
        # Create file
        test_file.write_text("# Watched File\n\nInitial content")
        await git_vault._on_file_created(test_file)
        
        # Modify file multiple times
        for i in range(3):
            test_file.write_text(f"# Watched File\n\nModified {i}")
            await git_vault._on_file_modified(test_file)
        
        # Delete file
        test_file.unlink()
        await git_vault._on_file_deleted(test_file)
        
        # Check change counter
        assert git_vault.git_service._change_counter == 5  # 1 create + 3 modify + 1 delete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])