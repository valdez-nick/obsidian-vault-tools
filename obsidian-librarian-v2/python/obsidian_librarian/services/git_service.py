"""
Git Service for Obsidian Librarian - Comprehensive backup and restore functionality.

This service provides Git-based version control for Obsidian vaults, including:
- Automatic backups based on change thresholds
- Smart commit messages based on file changes
- Branch management for experiments
- Rollback and restore functionality
- Stash management
- Conflict resolution helpers
"""

import asyncio
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum

import structlog
from git import Repo, GitCommandError, InvalidGitRepositoryError
from git.objects import Commit
from git.diff import Diff

from ..models import Note
from ..vault import Vault

logger = structlog.get_logger(__name__)


class BackupTrigger(Enum):
    """Types of backup triggers."""
    MANUAL = "manual"
    AUTO_THRESHOLD = "auto_threshold"
    AUTO_TIME = "auto_time"
    SHUTDOWN = "shutdown"


@dataclass
class GitConfig:
    """Configuration for Git operations."""
    auto_backup_enabled: bool = True
    auto_backup_threshold: int = 10  # Number of changes before auto-backup
    auto_backup_interval: int = 3600  # Seconds between time-based backups
    backup_branch_prefix: str = "backup"
    experiment_branch_prefix: str = "experiment"
    stash_before_operations: bool = True
    create_backup_before_restore: bool = True
    default_commit_author: str = "Obsidian Librarian"
    default_commit_email: str = "librarian@obsidian.local"
    commit_message_template: str = "Obsidian Librarian: {action} - {timestamp}"
    include_stats_in_commit: bool = True
    default_branch: str = "main"
    ignore_patterns: List[str] = None
    
    def __post_init__(self):
        if self.ignore_patterns is None:
            self.ignore_patterns = [
                ".obsidian/workspace*",
                ".obsidian/cache",
                ".trash/",
                ".DS_Store",
                "*.tmp",
                "*.temp",
                "~*"
            ]


@dataclass
class BackupStats:
    """Statistics about backup operations."""
    total_commits: int = 0
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0
    last_backup: Optional[datetime] = None
    pending_changes: int = 0


@dataclass
class CommitInfo:
    """Information about a Git commit."""
    hash: str
    author: str
    date: datetime
    message: str
    branch: str
    files_changed: int
    insertions: int
    deletions: int
    
    @classmethod
    def from_commit(cls, commit: Commit, branch: str = "main") -> 'CommitInfo':
        """Create CommitInfo from a Git commit object."""
        stats = commit.stats.total
        return cls(
            hash=commit.hexsha,
            author=f"{commit.author.name} <{commit.author.email}>",
            date=datetime.fromtimestamp(commit.committed_date),
            message=commit.message.strip(),
            branch=branch,
            files_changed=stats.get('files', 0),
            insertions=stats.get('insertions', 0),
            deletions=stats.get('deletions', 0)
        )


class GitService:
    """
    Manages Git operations for Obsidian vaults.
    
    This service provides comprehensive Git functionality including automatic
    backups, branch management, rollback capabilities, and conflict resolution.
    """
    
    def __init__(self, vault: Vault, config: Optional[GitConfig] = None):
        self.vault = vault
        self.config = config or GitConfig()
        self.repo: Optional[Repo] = None
        self._change_counter = 0
        self._last_auto_backup = datetime.utcnow()
        self._initialize_gitignore_patterns()
        
    def _initialize_gitignore_patterns(self) -> None:
        """Initialize .gitignore with default patterns."""
        gitignore_path = Path(self.vault.path) / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text("\n".join(self.config.ignore_patterns))
            
    async def initialize_repo(self) -> bool:
        """
        Initialize a Git repository in the vault.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            vault_path = Path(self.vault.path)
            
            # Check if already a Git repo
            try:
                self.repo = Repo(vault_path)
                logger.info("Existing Git repository found", path=str(vault_path))
                return True
            except InvalidGitRepositoryError:
                pass
            
            # Initialize new repository
            self.repo = Repo.init(vault_path)
            logger.info("Initialized new Git repository", path=str(vault_path))
            
            # Configure Git user
            with self.repo.config_writer() as config:
                config.set_value("user", "name", self.config.default_commit_author)
                config.set_value("user", "email", self.config.default_commit_email)
            
            # Create initial commit
            self._initialize_gitignore_patterns()
            self.repo.index.add([".gitignore"])
            self.repo.index.commit("Initial commit - Obsidian Librarian vault initialized")
            
            return True
            
        except Exception as e:
            logger.error("Failed to initialize Git repository", error=str(e))
            return False
    
    def _ensure_repo(self) -> None:
        """Ensure Git repository is initialized."""
        if self.repo is None:
            vault_path = Path(self.vault.path)
            try:
                self.repo = Repo(vault_path)
            except InvalidGitRepositoryError:
                raise RuntimeError("Git repository not initialized. Call initialize_repo() first.")
    
    def _generate_commit_message(self, changes: Dict[str, List[str]], auto: bool = False) -> str:
        """
        Generate a smart commit message based on changes.
        
        Args:
            changes: Dictionary with 'added', 'modified', 'deleted' file lists
            auto: Whether this is an automatic backup
            
        Returns:
            str: Generated commit message
        """
        parts = []
        
        # Add trigger type
        if auto:
            parts.append("[Auto-backup]")
        
        # Summarize changes
        added = len(changes.get('added', []))
        modified = len(changes.get('modified', []))
        deleted = len(changes.get('deleted', []))
        
        change_parts = []
        if added:
            change_parts.append(f"Added {added} file{'s' if added > 1 else ''}")
        if modified:
            change_parts.append(f"Modified {modified} file{'s' if modified > 1 else ''}")
        if deleted:
            change_parts.append(f"Deleted {deleted} file{'s' if deleted > 1 else ''}")
        
        if change_parts:
            parts.append(", ".join(change_parts))
        
        # Add timestamp for auto-backups
        if auto:
            parts.append(f"at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Analyze content changes for more context
        if modified and len(modified) <= 3:
            # For small number of files, include names
            file_names = [Path(f).stem for f in modified[:3]]
            parts.append(f"({', '.join(file_names)})")
        
        return " ".join(parts) if parts else "Update vault"
    
    def _get_changes(self) -> Dict[str, List[str]]:
        """Get current uncommitted changes."""
        self._ensure_repo()
        
        changes = {
            'added': [],
            'modified': [],
            'deleted': []
        }
        
        # Get staged changes
        staged = self.repo.index.diff("HEAD")
        for item in staged:
            if item.change_type == 'A':
                changes['added'].append(item.a_path)
            elif item.change_type == 'M':
                changes['modified'].append(item.a_path)
            elif item.change_type == 'D':
                changes['deleted'].append(item.a_path)
        
        # Get unstaged changes
        unstaged = self.repo.index.diff(None)
        for item in unstaged:
            if item.change_type == 'A' and item.a_path not in changes['added']:
                changes['added'].append(item.a_path)
            elif item.change_type == 'M' and item.a_path not in changes['modified']:
                changes['modified'].append(item.a_path)
            elif item.change_type == 'D' and item.a_path not in changes['deleted']:
                changes['deleted'].append(item.a_path)
        
        # Get untracked files
        untracked = self.repo.untracked_files
        changes['added'].extend(untracked)
        
        return changes
    
    async def backup(self, message: Optional[str] = None, auto: bool = False) -> Optional[str]:
        """
        Create a backup commit.
        
        Args:
            message: Custom commit message (auto-generated if None)
            auto: Whether this is an automatic backup
            
        Returns:
            Optional[str]: Commit hash if successful, None otherwise
        """
        try:
            self._ensure_repo()
            
            # Get current changes
            changes = self._get_changes()
            total_changes = sum(len(v) for v in changes.values())
            
            if total_changes == 0:
                logger.debug("No changes to backup")
                return None
            
            # Stage all changes
            for file_path in changes['added'] + changes['modified']:
                self.repo.index.add([file_path])
            
            for file_path in changes['deleted']:
                self.repo.index.remove([file_path])
            
            # Generate commit message if not provided
            if message is None:
                message = self._generate_commit_message(changes, auto)
            
            # Create commit
            commit = self.repo.index.commit(message)
            
            # Update statistics
            self._change_counter = 0
            self._last_auto_backup = datetime.utcnow()
            
            logger.info("Backup created", 
                       commit_hash=commit.hexsha[:8],
                       files_changed=total_changes,
                       auto=auto)
            
            return commit.hexsha
            
        except Exception as e:
            logger.error("Backup failed", error=str(e))
            return None
    
    async def restore(self, commit_hash: str) -> bool:
        """
        Restore vault to a specific commit.
        
        Args:
            commit_hash: Git commit hash to restore to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._ensure_repo()
            
            # Create backup branch if configured
            if self.config.create_backup_before_restore:
                backup_branch_name = f"{self.config.backup_branch_prefix}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
                self.repo.create_head(backup_branch_name)
                logger.info("Created backup branch", branch=backup_branch_name)
            
            # Stash current changes if any
            if self.config.stash_before_operations:
                changes = self._get_changes()
                if sum(len(v) for v in changes.values()) > 0:
                    self.repo.git.stash('save', f"Before restore to {commit_hash[:8]}")
                    logger.info("Stashed current changes")
            
            # Perform the restore
            self.repo.git.reset('--hard', commit_hash)
            
            # Verify vault integrity
            if not await self._verify_vault_integrity():
                logger.error("Vault integrity check failed after restore")
                return False
            
            logger.info("Successfully restored vault", commit_hash=commit_hash[:8])
            return True
            
        except Exception as e:
            logger.error("Restore failed", error=str(e), commit_hash=commit_hash)
            return False
    
    async def _verify_vault_integrity(self) -> bool:
        """Verify vault integrity after restore."""
        try:
            # Basic check: ensure vault can be accessed
            stats = await self.vault.get_stats()
            return stats.total_notes > 0
        except Exception:
            return False
    
    async def create_branch(self, name: str, checkout: bool = True) -> Optional[str]:
        """
        Create a new branch for experiments.
        
        Args:
            name: Branch name (will be prefixed with experiment prefix)
            checkout: Whether to checkout the new branch
            
        Returns:
            Optional[str]: Full branch name if successful, None otherwise
        """
        try:
            self._ensure_repo()
            
            # Create full branch name
            full_name = f"{self.config.experiment_branch_prefix}/{name}"
            
            # Create the branch
            new_branch = self.repo.create_head(full_name)
            
            # Checkout if requested
            if checkout:
                new_branch.checkout()
            
            logger.info("Created branch", branch=full_name, checked_out=checkout)
            return full_name
            
        except Exception as e:
            logger.error("Failed to create branch", error=str(e), name=name)
            return None
    
    async def stash_changes(self, message: Optional[str] = None) -> bool:
        """
        Stash current changes.
        
        Args:
            message: Optional stash message
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._ensure_repo()
            
            # Check if there are changes to stash
            changes = self._get_changes()
            if sum(len(v) for v in changes.values()) == 0:
                logger.debug("No changes to stash")
                return True
            
            # Create stash
            stash_message = message or f"Stashed on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
            self.repo.git.stash('save', stash_message)
            
            logger.info("Changes stashed", message=stash_message)
            return True
            
        except Exception as e:
            logger.error("Failed to stash changes", error=str(e))
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get current repository status.
        
        Returns:
            Dict containing status information
        """
        try:
            self._ensure_repo()
            
            # Get current branch
            current_branch = self.repo.active_branch.name
            
            # Get changes
            changes = self._get_changes()
            
            # Get last commit info
            last_commit = None
            if self.repo.head.is_valid():
                commit = self.repo.head.commit
                last_commit = CommitInfo.from_commit(commit, current_branch)
            
            # Count stashes
            stash_count = len(self.repo.git.stash('list').splitlines()) if self.repo.git.stash('list') else 0
            
            # Get branches
            branches = [branch.name for branch in self.repo.branches]
            
            return {
                'initialized': True,
                'current_branch': current_branch,
                'branches': branches,
                'changes': changes,
                'total_changes': sum(len(v) for v in changes.values()),
                'last_commit': last_commit,
                'stash_count': stash_count,
                'auto_backup_enabled': self.config.auto_backup_enabled,
                'changes_since_backup': self._change_counter
            }
            
        except Exception as e:
            logger.error("Failed to get status", error=str(e))
            return {
                'initialized': False,
                'error': str(e)
            }
    
    async def get_history(self, limit: int = 10, branch: Optional[str] = None) -> List[CommitInfo]:
        """
        Get commit history.
        
        Args:
            limit: Maximum number of commits to return
            branch: Branch to get history from (current branch if None)
            
        Returns:
            List of CommitInfo objects
        """
        try:
            self._ensure_repo()
            
            # Use current branch if not specified
            if branch is None:
                branch = self.repo.active_branch.name
            
            # Get commits
            commits = []
            for commit in self.repo.iter_commits(branch, max_count=limit):
                commits.append(CommitInfo.from_commit(commit, branch))
            
            return commits
            
        except Exception as e:
            logger.error("Failed to get history", error=str(e))
            return []
    
    async def rollback(self, steps: int = 1) -> bool:
        """
        Rollback to a previous commit by number of steps.
        
        Args:
            steps: Number of commits to roll back
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._ensure_repo()
            
            # Get the target commit
            target_commit = f"HEAD~{steps}"
            
            # Perform rollback (which is essentially a restore)
            return await self.restore(target_commit)
            
        except Exception as e:
            logger.error("Rollback failed", error=str(e), steps=steps)
            return False
    
    async def get_diff(self, commit_hash: Optional[str] = None, staged: bool = False) -> List[Dict[str, Any]]:
        """
        Get diff for a commit or current changes.
        
        Args:
            commit_hash: Commit to get diff for (current changes if None)
            staged: Whether to get staged changes only
            
        Returns:
            List of diff information
        """
        try:
            self._ensure_repo()
            
            diffs = []
            
            if commit_hash:
                # Get diff for specific commit
                commit = self.repo.commit(commit_hash)
                diff_index = commit.diff(commit.parents[0] if commit.parents else None)
            elif staged:
                # Get staged changes
                diff_index = self.repo.index.diff("HEAD")
            else:
                # Get all changes
                diff_index = self.repo.index.diff(None)
            
            for diff_item in diff_index:
                diff_info = {
                    'file': diff_item.a_path or diff_item.b_path,
                    'change_type': diff_item.change_type,
                    'additions': 0,
                    'deletions': 0
                }
                
                # Try to get line statistics
                if hasattr(diff_item, 'diff') and diff_item.diff:
                    diff_text = diff_item.diff.decode('utf-8', errors='ignore')
                    diff_info['additions'] = diff_text.count('\n+')
                    diff_info['deletions'] = diff_text.count('\n-')
                
                diffs.append(diff_info)
            
            return diffs
            
        except Exception as e:
            logger.error("Failed to get diff", error=str(e))
            return []
    
    async def merge_branch(self, branch_name: str, delete_after: bool = True) -> bool:
        """
        Merge a branch into the current branch.
        
        Args:
            branch_name: Name of branch to merge
            delete_after: Whether to delete the branch after successful merge
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._ensure_repo()
            
            # Get current branch
            current_branch = self.repo.active_branch
            
            # Merge the branch
            self.repo.git.merge(branch_name)
            
            # Delete branch if requested
            if delete_after:
                self.repo.delete_head(branch_name)
            
            logger.info("Branch merged successfully", 
                       source=branch_name, 
                       target=current_branch.name,
                       deleted=delete_after)
            return True
            
        except GitCommandError as e:
            if "CONFLICT" in str(e):
                logger.error("Merge conflict detected", branch=branch_name)
            else:
                logger.error("Merge failed", error=str(e), branch=branch_name)
            return False
    
    async def resolve_conflicts(self, strategy: str = "ours") -> bool:
        """
        Attempt to resolve merge conflicts.
        
        Args:
            strategy: Resolution strategy ('ours', 'theirs', or 'manual')
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._ensure_repo()
            
            if strategy == "ours":
                self.repo.git.checkout('--ours', '.')
                self.repo.index.add('.')
            elif strategy == "theirs":
                self.repo.git.checkout('--theirs', '.')
                self.repo.index.add('.')
            else:
                # Manual resolution required
                logger.warning("Manual conflict resolution required")
                return False
            
            logger.info("Conflicts resolved", strategy=strategy)
            return True
            
        except Exception as e:
            logger.error("Failed to resolve conflicts", error=str(e))
            return False
    
    def register_change(self) -> None:
        """Register a change for auto-backup threshold tracking."""
        self._change_counter += 1
        
        # Check if auto-backup should trigger
        if self.config.auto_backup_enabled:
            if self._change_counter >= self.config.auto_backup_threshold:
                asyncio.create_task(self.backup(auto=True))
    
    async def check_time_based_backup(self) -> None:
        """Check if time-based auto-backup should trigger."""
        if not self.config.auto_backup_enabled:
            return
        
        time_since_backup = (datetime.utcnow() - self._last_auto_backup).total_seconds()
        if time_since_backup >= self.config.auto_backup_interval:
            await self.backup(auto=True)
    
    async def cleanup_old_backups(self, keep_last: int = 10) -> int:
        """
        Clean up old backup branches.
        
        Args:
            keep_last: Number of recent backup branches to keep
            
        Returns:
            int: Number of branches deleted
        """
        try:
            self._ensure_repo()
            
            # Get all backup branches
            backup_branches = [
                branch for branch in self.repo.branches
                if branch.name.startswith(self.config.backup_branch_prefix)
            ]
            
            # Sort by commit date
            backup_branches.sort(key=lambda b: b.commit.committed_date, reverse=True)
            
            # Delete old branches
            deleted = 0
            for branch in backup_branches[keep_last:]:
                self.repo.delete_head(branch, force=True)
                deleted += 1
            
            logger.info("Cleaned up old backup branches", deleted=deleted, kept=keep_last)
            return deleted
            
        except Exception as e:
            logger.error("Failed to cleanup backups", error=str(e))
            return 0