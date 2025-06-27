#!/usr/bin/env python3
"""
Smart Backup Manager - Hybrid backup system for meeting notes organizer.

Combines in-memory backups for instant undo with hidden file backups
for crash protection, while avoiding file bloat through smart cleanup.
"""

import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SmartBackupManager:
    """
    Hybrid backup system that provides both instant undo and crash protection
    without creating visible backup file bloat.
    """
    
    def __init__(self, vault_path: str):
        """
        Initialize the backup manager.
        
        Args:
            vault_path: Path to the Obsidian vault
        """
        self.vault_path = Path(vault_path)
        self.backup_dir = self.vault_path / ".meeting_backups"
        self.session_backups: Dict[str, Dict[str, Any]] = {}
        self.active_session_id: Optional[str] = None
        
        # Ensure backup directory exists but is hidden
        self.backup_dir.mkdir(exist_ok=True)
        
        # Create .gitignore to prevent backup files from being committed
        gitignore_path = self.backup_dir / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, 'w') as f:
                f.write("# Auto-generated meeting note backups\n*\n")
    
    def create_session_backup(self, file_path: str, content: str) -> str:
        """
        Create both in-memory and hidden file backup for a session.
        
        Args:
            file_path: Path to the file being backed up
            content: Original content to backup
            
        Returns:
            Session ID for this backup
        """
        session_id = self._generate_session_id()
        timestamp = datetime.now()
        
        # Store in memory for instant access
        self.session_backups[session_id] = {
            'original_content': content,
            'file_path': file_path,
            'timestamp': timestamp,
            'restored': False,
            'backup_file': None
        }
        
        # Create hidden file backup as safety net
        try:
            backup_file = self._create_hidden_backup(file_path, content, session_id)
            self.session_backups[session_id]['backup_file'] = backup_file
            logger.debug(f"Created session backup: {session_id} for {file_path}")
        except Exception as e:
            logger.warning(f"Could not create file backup: {e}")
            # Continue with memory-only backup
        
        self.active_session_id = session_id
        
        # Clean up old backups in background
        self._cleanup_old_backups()
        
        return session_id
    
    def undo_from_memory(self, session_id: str) -> bool:
        """
        Instant undo from memory - no file I/O delay.
        
        Args:
            session_id: Session to restore from
            
        Returns:
            True if restore was successful
        """
        if session_id not in self.session_backups:
            logger.error(f"Session backup not found: {session_id}")
            return False
        
        backup = self.session_backups[session_id]
        
        try:
            # Write original content back to file
            with open(backup['file_path'], 'w', encoding='utf-8') as f:
                f.write(backup['original_content'])
            
            backup['restored'] = True
            logger.info(f"Successfully restored from memory backup: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from memory backup: {e}")
            return False
    
    def restore_from_file(self, session_id: str) -> bool:
        """
        Restore from hidden file backup (fallback for crashes).
        
        Args:
            session_id: Session to restore from
            
        Returns:
            True if restore was successful
        """
        if session_id not in self.session_backups:
            logger.error(f"Session backup not found: {session_id}")
            return False
        
        backup = self.session_backups[session_id]
        backup_file = backup.get('backup_file')
        
        if not backup_file or not Path(backup_file).exists():
            logger.error(f"File backup not found: {backup_file}")
            return False
        
        try:
            # Read backup content
            with open(backup_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Write back to original file
            with open(backup['file_path'], 'w', encoding='utf-8') as f:
                f.write(content)
            
            backup['restored'] = True
            logger.info(f"Successfully restored from file backup: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from file backup: {e}")
            return False
    
    def complete_session(self, session_id: str, success: bool = True):
        """
        Clean up after operation completion.
        
        Args:
            session_id: Session to complete
            success: Whether the operation was successful
        """
        if session_id not in self.session_backups:
            return
        
        backup = self.session_backups[session_id]
        
        # If operation was successful and not restored, cleanup file backup
        if success and not backup['restored']:
            self._cleanup_file_backup(backup.get('backup_file'))
        
        # Always remove from memory
        del self.session_backups[session_id]
        
        if self.active_session_id == session_id:
            self.active_session_id = None
        
        logger.debug(f"Completed session: {session_id}")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a backup session.
        
        Args:
            session_id: Session to get info for
            
        Returns:
            Session information or None if not found
        """
        backup = self.session_backups.get(session_id)
        if not backup:
            return None
        
        return {
            'session_id': session_id,
            'file_path': backup['file_path'],
            'timestamp': backup['timestamp'].isoformat(),
            'restored': backup['restored'],
            'has_file_backup': backup.get('backup_file') is not None,
            'is_active': session_id == self.active_session_id
        }
    
    def list_active_sessions(self) -> list:
        """
        List all active backup sessions.
        
        Returns:
            List of session information
        """
        return [
            self.get_session_info(session_id)
            for session_id in self.session_backups.keys()
        ]
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _create_hidden_backup(self, file_path: str, content: str, session_id: str) -> str:
        """
        Create hidden file backup.
        
        Args:
            file_path: Original file path
            content: Content to backup
            session_id: Session identifier
            
        Returns:
            Path to backup file
        """
        # Create backup filename with session ID
        original_path = Path(file_path)
        backup_name = f"{original_path.stem}_{session_id}.backup"
        backup_path = self.backup_dir / backup_name
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(backup_path)
    
    def _cleanup_file_backup(self, backup_file: Optional[str]):
        """
        Remove a specific backup file.
        
        Args:
            backup_file: Path to backup file to remove
        """
        if not backup_file:
            return
        
        try:
            backup_path = Path(backup_file)
            if backup_path.exists():
                backup_path.unlink()
                logger.debug(f"Cleaned up backup file: {backup_file}")
        except Exception as e:
            logger.warning(f"Could not remove backup file {backup_file}: {e}")
    
    def _cleanup_old_backups(self, max_age_hours: int = 24):
        """
        Auto-cleanup old backup files.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        if not self.backup_dir.exists():
            return
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        try:
            for backup_file in self.backup_dir.glob("*.backup"):
                file_mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_mtime < cutoff_time:
                    backup_file.unlink()
                    cleaned_count += 1
            
            if cleaned_count > 0:
                logger.debug(f"Cleaned up {cleaned_count} old backup files")
                
        except Exception as e:
            logger.warning(f"Error during backup cleanup: {e}")
    
    def emergency_restore_all(self) -> Dict[str, bool]:
        """
        Emergency function to restore all active sessions from file backups.
        Useful for crash recovery scenarios.
        
        Returns:
            Dictionary mapping session IDs to restore success status
        """
        results = {}
        
        for session_id in list(self.session_backups.keys()):
            try:
                success = self.restore_from_file(session_id)
                results[session_id] = success
                if success:
                    self.complete_session(session_id, success=True)
            except Exception as e:
                logger.error(f"Emergency restore failed for {session_id}: {e}")
                results[session_id] = False
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get backup manager statistics.
        
        Returns:
            Statistics about backup usage
        """
        active_sessions = len(self.session_backups)
        
        # Count backup files
        backup_files = 0
        if self.backup_dir.exists():
            backup_files = len(list(self.backup_dir.glob("*.backup")))
        
        return {
            'active_sessions': active_sessions,
            'backup_files_on_disk': backup_files,
            'backup_directory': str(self.backup_dir),
            'current_session': self.active_session_id,
            'memory_usage_kb': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """
        Estimate memory usage of stored backups.
        
        Returns:
            Estimated memory usage in KB
        """
        total_chars = sum(
            len(backup['original_content'])
            for backup in self.session_backups.values()
        )
        # Rough estimate: 1 character ≈ 1 byte
        return total_chars / 1024.0


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import logging
    
    logging.basicConfig(level=logging.DEBUG)
    
    # Test with temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = SmartBackupManager(temp_dir)
        
        # Create test file
        test_file = Path(temp_dir) / "test_note.md"
        original_content = """# Test Meeting Notes

Some unorganized notes here.
- Action item 1
- Random thought
- Another action item
"""
        
        with open(test_file, 'w') as f:
            f.write(original_content)
        
        print("Testing SmartBackupManager...")
        
        # Create backup
        session_id = manager.create_session_backup(str(test_file), original_content)
        print(f"Created session: {session_id}")
        
        # Modify file
        modified_content = """# Test Meeting Notes - MODIFIED

Organized notes here.
"""
        with open(test_file, 'w') as f:
            f.write(modified_content)
        
        print(f"Modified file. Current content length: {len(modified_content)}")
        
        # Test undo
        success = manager.undo_from_memory(session_id)
        print(f"Undo from memory: {success}")
        
        # Verify restoration
        with open(test_file, 'r') as f:
            restored_content = f.read()
        
        print(f"Restored correctly: {restored_content == original_content}")
        print(f"Stats: {manager.get_stats()}")
        
        # Complete session
        manager.complete_session(session_id, success=True)
        print("Session completed")