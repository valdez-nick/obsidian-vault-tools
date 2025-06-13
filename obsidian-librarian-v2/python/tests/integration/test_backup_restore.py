"""
Integration tests for backup and restore functionality.
"""

import asyncio
import json
import shutil
import tarfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio

from obsidian_librarian.vault import Vault
from obsidian_librarian.models import VaultConfig, BackupMetadata
from obsidian_librarian.services.backup import BackupService, BackupStrategy


@pytest.fixture
def sample_vault(tmp_path):
    """Create a sample vault with various content."""
    vault_path = tmp_path / "sample_vault"
    vault_path.mkdir()
    
    # Create .obsidian directory with config
    obsidian_dir = vault_path / ".obsidian"
    obsidian_dir.mkdir()
    
    config = {
        "theme": "dark",
        "plugins": ["daily-notes", "templates"],
        "hotkeys": {}
    }
    (obsidian_dir / "app.json").write_text(json.dumps(config))
    
    # Create notes with various content
    notes = {
        "index.md": "# Index\n\nMain vault index [[projects/project1]]",
        "daily/2024-01-01.md": "# 2024-01-01\n\n- [ ] Morning routine\n- [x] Review notes",
        "daily/2024-01-02.md": "# 2024-01-02\n\n- [ ] Team meeting\n- [ ] Code review",
        "projects/project1.md": "# Project Alpha\n\n## Overview\nProject description...\n\n## Tasks\n- [ ] Design\n- [ ] Implementation",
        "projects/project2.md": "# Project Beta\n\n## Status\nIn progress...",
        "templates/daily.md": "# {{date}}\n\n## Tasks\n- [ ] \n\n## Notes\n",
        "templates/project.md": "# {{title}}\n\n## Overview\n\n## Tasks\n- [ ] ",
        "attachments/diagram.png": b"PNG_DATA_HERE",  # Binary file
        "research/paper1.md": "# Research Paper 1\n\nAbstract...\n\n## References\n1. Source 1\n2. Source 2"
    }
    
    for file_path, content in notes.items():
        file_full_path = vault_path / file_path
        file_full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(content, bytes):
            file_full_path.write_bytes(content)
        else:
            file_full_path.write_text(content)
    
    return vault_path


@pytest.fixture
def backup_service():
    """Create a backup service instance."""
    return BackupService()


@pytest.fixture
def backup_dir(tmp_path):
    """Create a directory for backups."""
    backup_path = tmp_path / "backups"
    backup_path.mkdir()
    return backup_path


class TestBackupFunctionality:
    """Test suite for backup functionality."""
    
    @pytest.mark.asyncio
    async def test_create_full_backup(self, sample_vault, backup_dir, backup_service):
        """Test creating a full vault backup."""
        backup_path = backup_dir / "full_backup.tar.gz"
        
        # Create backup
        metadata = await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=backup_path,
            strategy=BackupStrategy.FULL
        )
        
        assert backup_path.exists()
        assert metadata.backup_type == "full"
        assert metadata.vault_path == str(sample_vault)
        assert metadata.file_count > 0
        assert metadata.total_size > 0
        
        # Verify backup contents
        with tarfile.open(backup_path, 'r:gz') as tar:
            members = tar.getmembers()
            filenames = [m.name for m in members]
            
            # Check key files are included
            assert any("index.md" in f for f in filenames)
            assert any("daily/2024-01-01.md" in f for f in filenames)
            assert any(".obsidian/app.json" in f for f in filenames)
            assert any("attachments/diagram.png" in f for f in filenames)
    
    @pytest.mark.asyncio
    async def test_create_incremental_backup(self, sample_vault, backup_dir, backup_service):
        """Test creating incremental backups."""
        # First create a full backup
        full_backup_path = backup_dir / "full_backup.tar.gz"
        await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=full_backup_path,
            strategy=BackupStrategy.FULL
        )
        
        # Modify some files
        (sample_vault / "index.md").write_text("# Index\n\nUpdated content")
        (sample_vault / "new_note.md").write_text("# New Note\n\nNew content")
        
        # Create incremental backup
        incr_backup_path = backup_dir / "incremental_backup.tar.gz"
        metadata = await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=incr_backup_path,
            strategy=BackupStrategy.INCREMENTAL,
            base_backup=full_backup_path
        )
        
        assert incr_backup_path.exists()
        assert metadata.backup_type == "incremental"
        assert metadata.base_backup == str(full_backup_path)
        
        # Verify only changed files are in incremental backup
        with tarfile.open(incr_backup_path, 'r:gz') as tar:
            members = tar.getmembers()
            filenames = [m.name for m in members]
            
            assert any("index.md" in f for f in filenames)
            assert any("new_note.md" in f for f in filenames)
            # Unchanged files should not be included
            assert not any("projects/project1.md" in f for f in filenames)
    
    @pytest.mark.asyncio
    async def test_backup_with_exclusions(self, sample_vault, backup_dir, backup_service):
        """Test backup with file exclusions."""
        backup_path = backup_dir / "filtered_backup.tar.gz"
        
        # Create backup excluding certain patterns
        metadata = await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=backup_path,
            strategy=BackupStrategy.FULL,
            exclude_patterns=["*.png", "daily/*", ".obsidian/workspace"]
        )
        
        assert backup_path.exists()
        
        # Verify exclusions
        with tarfile.open(backup_path, 'r:gz') as tar:
            members = tar.getmembers()
            filenames = [m.name for m in members]
            
            # Excluded files should not be present
            assert not any("diagram.png" in f for f in filenames)
            assert not any("daily/" in f for f in filenames)
            # Included files should be present
            assert any("index.md" in f for f in filenames)
            assert any("projects/project1.md" in f for f in filenames)
    
    @pytest.mark.asyncio
    async def test_backup_compression_formats(self, sample_vault, backup_dir, backup_service):
        """Test different backup compression formats."""
        formats = [
            ("backup.tar.gz", "tar.gz"),
            ("backup.tar.bz2", "tar.bz2"),
            ("backup.zip", "zip")
        ]
        
        for filename, format_type in formats:
            backup_path = backup_dir / filename
            
            metadata = await backup_service.create_backup(
                vault_path=sample_vault,
                backup_path=backup_path,
                strategy=BackupStrategy.FULL,
                compression_format=format_type
            )
            
            assert backup_path.exists()
            assert metadata.compression_format == format_type
            
            # Verify we can open the backup
            if format_type == "zip":
                with zipfile.ZipFile(backup_path, 'r') as zf:
                    assert len(zf.namelist()) > 0
            else:
                with tarfile.open(backup_path, f'r:{format_type.split(".")[-1]}') as tf:
                    assert len(tf.getmembers()) > 0
    
    @pytest.mark.asyncio
    async def test_backup_metadata_storage(self, sample_vault, backup_dir, backup_service):
        """Test backup metadata is properly stored."""
        backup_path = backup_dir / "metadata_test.tar.gz"
        
        metadata = await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=backup_path,
            strategy=BackupStrategy.FULL,
            description="Test backup with metadata"
        )
        
        # Check metadata file exists
        metadata_path = backup_path.with_suffix('.meta.json')
        assert metadata_path.exists()
        
        # Verify metadata contents
        with open(metadata_path) as f:
            saved_metadata = json.load(f)
            
        assert saved_metadata['backup_id'] == metadata.backup_id
        assert saved_metadata['description'] == "Test backup with metadata"
        assert saved_metadata['vault_path'] == str(sample_vault)
        assert 'timestamp' in saved_metadata
        assert 'checksum' in saved_metadata


class TestRestoreFunctionality:
    """Test suite for restore functionality."""
    
    @pytest.mark.asyncio
    async def test_restore_full_backup(self, sample_vault, backup_dir, backup_service, tmp_path):
        """Test restoring from a full backup."""
        # Create a backup
        backup_path = backup_dir / "restore_test.tar.gz"
        await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=backup_path,
            strategy=BackupStrategy.FULL
        )
        
        # Create restore target
        restore_path = tmp_path / "restored_vault"
        
        # Restore backup
        result = await backup_service.restore_backup(
            backup_path=backup_path,
            restore_path=restore_path
        )
        
        assert result.success
        assert result.files_restored > 0
        assert restore_path.exists()
        
        # Verify restored content
        assert (restore_path / "index.md").exists()
        assert (restore_path / "daily/2024-01-01.md").exists()
        assert (restore_path / ".obsidian/app.json").exists()
        
        # Verify content matches
        original_content = (sample_vault / "index.md").read_text()
        restored_content = (restore_path / "index.md").read_text()
        assert original_content == restored_content
    
    @pytest.mark.asyncio
    async def test_restore_to_existing_vault(self, sample_vault, backup_dir, backup_service):
        """Test restoring to an existing vault with conflict handling."""
        # Create a backup
        backup_path = backup_dir / "restore_test.tar.gz"
        await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=backup_path,
            strategy=BackupStrategy.FULL
        )
        
        # Modify the original vault
        (sample_vault / "index.md").write_text("# Modified Index\n\nChanged content")
        (sample_vault / "new_file.md").write_text("# New File\n\nAdded after backup")
        
        # Restore with different strategies
        # 1. Overwrite strategy
        result = await backup_service.restore_backup(
            backup_path=backup_path,
            restore_path=sample_vault,
            conflict_strategy="overwrite"
        )
        
        assert result.success
        assert (sample_vault / "index.md").read_text().startswith("# Index")  # Original content
        assert (sample_vault / "new_file.md").exists()  # Preserved
        
        # 2. Merge strategy
        (sample_vault / "index.md").write_text("# Modified Again")
        
        result = await backup_service.restore_backup(
            backup_path=backup_path,
            restore_path=sample_vault,
            conflict_strategy="merge"
        )
        
        assert result.success
        # Should create conflict file
        conflict_files = list(sample_vault.glob("index.md.conflict.*"))
        assert len(conflict_files) > 0
    
    @pytest.mark.asyncio
    async def test_restore_incremental_chain(self, sample_vault, backup_dir, backup_service, tmp_path):
        """Test restoring from a chain of incremental backups."""
        # Create full backup
        full_backup = backup_dir / "full.tar.gz"
        await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=full_backup,
            strategy=BackupStrategy.FULL
        )
        
        # Make changes and create first incremental
        (sample_vault / "index.md").write_text("# Index v2")
        incr1_backup = backup_dir / "incr1.tar.gz"
        await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=incr1_backup,
            strategy=BackupStrategy.INCREMENTAL,
            base_backup=full_backup
        )
        
        # Make more changes and create second incremental
        (sample_vault / "new_note.md").write_text("# New Note")
        incr2_backup = backup_dir / "incr2.tar.gz"
        await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=incr2_backup,
            strategy=BackupStrategy.INCREMENTAL,
            base_backup=incr1_backup
        )
        
        # Restore from the chain
        restore_path = tmp_path / "restored_chain"
        result = await backup_service.restore_backup(
            backup_path=incr2_backup,
            restore_path=restore_path,
            restore_chain=True
        )
        
        assert result.success
        assert (restore_path / "index.md").read_text() == "# Index v2"
        assert (restore_path / "new_note.md").exists()
    
    @pytest.mark.asyncio
    async def test_restore_with_rollback(self, sample_vault, backup_dir, backup_service):
        """Test restore with rollback capability."""
        # Create a backup
        backup_path = backup_dir / "rollback_test.tar.gz"
        await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=backup_path,
            strategy=BackupStrategy.FULL
        )
        
        # Simulate a failed restore
        with patch.object(backup_service, '_extract_files', side_effect=Exception("Extraction failed")):
            result = await backup_service.restore_backup(
                backup_path=backup_path,
                restore_path=sample_vault,
                enable_rollback=True
            )
            
            assert not result.success
            assert result.error == "Extraction failed"
            assert result.rollback_performed
            
            # Vault should be in original state
            assert (sample_vault / "index.md").read_text().startswith("# Index")
    
    @pytest.mark.asyncio
    async def test_selective_restore(self, sample_vault, backup_dir, backup_service, tmp_path):
        """Test selective file restoration."""
        # Create a backup
        backup_path = backup_dir / "selective_test.tar.gz"
        await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=backup_path,
            strategy=BackupStrategy.FULL
        )
        
        # Restore only specific files
        restore_path = tmp_path / "selective_restore"
        restore_path.mkdir()
        
        result = await backup_service.restore_backup(
            backup_path=backup_path,
            restore_path=restore_path,
            include_patterns=["daily/*", "*.md"],
            exclude_patterns=["templates/*"]
        )
        
        assert result.success
        assert (restore_path / "daily/2024-01-01.md").exists()
        assert (restore_path / "index.md").exists()
        assert not (restore_path / "templates/daily.md").exists()
        assert not (restore_path / "attachments/diagram.png").exists()


class TestBackupScheduling:
    """Test suite for scheduled backup functionality."""
    
    @pytest.mark.asyncio
    async def test_scheduled_backup_creation(self, sample_vault, backup_dir, backup_service):
        """Test creating scheduled backups."""
        schedule_config = {
            "interval": "daily",
            "time": "02:00",
            "retention_days": 7,
            "strategy": "incremental",
            "full_backup_interval": 7  # Full backup every 7 days
        }
        
        scheduler = backup_service.create_scheduler(
            vault_path=sample_vault,
            backup_dir=backup_dir,
            config=schedule_config
        )
        
        # Simulate scheduled backup execution
        await scheduler.execute_scheduled_backup()
        
        # Verify backup was created
        backups = list(backup_dir.glob("scheduled_*.tar.gz"))
        assert len(backups) == 1
        
        # Check metadata
        metadata_files = list(backup_dir.glob("scheduled_*.meta.json"))
        assert len(metadata_files) == 1
        
        with open(metadata_files[0]) as f:
            metadata = json.load(f)
            assert metadata['scheduled'] == True
            assert metadata['schedule_config'] == schedule_config
    
    @pytest.mark.asyncio
    async def test_backup_retention_policy(self, sample_vault, backup_dir, backup_service):
        """Test backup retention and cleanup."""
        # Create multiple backups with different timestamps
        base_time = datetime.now()
        
        for i in range(10):
            backup_time = base_time - timedelta(days=i)
            backup_path = backup_dir / f"backup_{i}.tar.gz"
            
            # Create backup with mocked timestamp
            with patch('datetime.datetime') as mock_datetime:
                mock_datetime.now.return_value = backup_time
                await backup_service.create_backup(
                    vault_path=sample_vault,
                    backup_path=backup_path,
                    strategy=BackupStrategy.FULL
                )
        
        # Apply retention policy (keep last 5 days)
        await backup_service.apply_retention_policy(
            backup_dir=backup_dir,
            retention_days=5
        )
        
        # Verify old backups were removed
        remaining_backups = list(backup_dir.glob("backup_*.tar.gz"))
        assert len(remaining_backups) == 5
        
        # Verify newest backups were kept
        for i in range(5):
            assert (backup_dir / f"backup_{i}.tar.gz").exists()
        
        # Verify old backups were removed
        for i in range(5, 10):
            assert not (backup_dir / f"backup_{i}.tar.gz").exists()


class TestBackupVerification:
    """Test suite for backup verification and integrity."""
    
    @pytest.mark.asyncio
    async def test_backup_checksum_verification(self, sample_vault, backup_dir, backup_service):
        """Test backup integrity verification via checksums."""
        backup_path = backup_dir / "checksum_test.tar.gz"
        
        # Create backup
        metadata = await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=backup_path,
            strategy=BackupStrategy.FULL
        )
        
        # Verify backup integrity
        is_valid = await backup_service.verify_backup(
            backup_path=backup_path,
            expected_checksum=metadata.checksum
        )
        
        assert is_valid
        
        # Corrupt the backup
        with open(backup_path, 'ab') as f:
            f.write(b'corrupted data')
        
        # Verification should fail
        is_valid = await backup_service.verify_backup(
            backup_path=backup_path,
            expected_checksum=metadata.checksum
        )
        
        assert not is_valid
    
    @pytest.mark.asyncio
    async def test_backup_content_verification(self, sample_vault, backup_dir, backup_service):
        """Test deep content verification of backups."""
        backup_path = backup_dir / "content_test.tar.gz"
        
        # Create backup
        await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=backup_path,
            strategy=BackupStrategy.FULL
        )
        
        # Perform deep verification
        verification_result = await backup_service.verify_backup_contents(
            backup_path=backup_path,
            original_vault=sample_vault
        )
        
        assert verification_result.is_valid
        assert verification_result.file_count_match
        assert len(verification_result.missing_files) == 0
        assert len(verification_result.corrupted_files) == 0


class TestBackupEncryption:
    """Test suite for backup encryption functionality."""
    
    @pytest.mark.asyncio
    async def test_encrypted_backup_creation(self, sample_vault, backup_dir, backup_service):
        """Test creating encrypted backups."""
        backup_path = backup_dir / "encrypted.tar.gz.enc"
        password = "test_password_123"
        
        # Create encrypted backup
        metadata = await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=backup_path,
            strategy=BackupStrategy.FULL,
            encrypt=True,
            password=password
        )
        
        assert backup_path.exists()
        assert metadata.encrypted
        
        # Verify file is actually encrypted (not readable as tar)
        with pytest.raises(Exception):
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.getmembers()
    
    @pytest.mark.asyncio
    async def test_encrypted_backup_restore(self, sample_vault, backup_dir, backup_service, tmp_path):
        """Test restoring from encrypted backups."""
        backup_path = backup_dir / "encrypted.tar.gz.enc"
        password = "test_password_123"
        
        # Create encrypted backup
        await backup_service.create_backup(
            vault_path=sample_vault,
            backup_path=backup_path,
            strategy=BackupStrategy.FULL,
            encrypt=True,
            password=password
        )
        
        # Restore with correct password
        restore_path = tmp_path / "decrypted_restore"
        result = await backup_service.restore_backup(
            backup_path=backup_path,
            restore_path=restore_path,
            password=password
        )
        
        assert result.success
        assert (restore_path / "index.md").exists()
        
        # Restore with wrong password should fail
        wrong_restore_path = tmp_path / "wrong_password_restore"
        result = await backup_service.restore_backup(
            backup_path=backup_path,
            restore_path=wrong_restore_path,
            password="wrong_password"
        )
        
        assert not result.success
        assert "decryption failed" in result.error.lower()


class TestBackupCLIIntegration:
    """Test backup functionality through CLI."""
    
    def test_backup_command(self, cli_runner, sample_vault, backup_dir):
        """Test backup CLI command."""
        # Note: This assumes backup command exists in CLI
        result = cli_runner.invoke(cli, [
            'backup',
            'create',
            str(sample_vault),
            '--output', str(backup_dir / "cli_backup.tar.gz"),
            '--strategy', 'full'
        ])
        
        assert result.exit_code == 0
        assert (backup_dir / "cli_backup.tar.gz").exists()
    
    def test_restore_command(self, cli_runner, sample_vault, backup_dir, tmp_path):
        """Test restore CLI command."""
        # First create a backup
        backup_path = backup_dir / "cli_restore_test.tar.gz"
        
        # Create backup via CLI
        result = cli_runner.invoke(cli, [
            'backup',
            'create',
            str(sample_vault),
            '--output', str(backup_path)
        ])
        
        assert result.exit_code == 0
        
        # Restore via CLI
        restore_path = tmp_path / "cli_restored"
        result = cli_runner.invoke(cli, [
            'backup',
            'restore',
            str(backup_path),
            '--target', str(restore_path)
        ])
        
        assert result.exit_code == 0
        assert restore_path.exists()
    
    def test_backup_list_command(self, cli_runner, backup_dir):
        """Test listing backups via CLI."""
        # Create some test backup metadata files
        for i in range(3):
            metadata = {
                "backup_id": f"backup_{i}",
                "timestamp": datetime.now().isoformat(),
                "vault_path": "/test/vault",
                "file_count": 100 + i,
                "total_size": 1024 * (i + 1)
            }
            
            metadata_path = backup_dir / f"backup_{i}.meta.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        
        result = cli_runner.invoke(cli, [
            'backup',
            'list',
            '--backup-dir', str(backup_dir)
        ])
        
        assert result.exit_code == 0
        assert "backup_0" in result.output
        assert "backup_1" in result.output
        assert "backup_2" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])