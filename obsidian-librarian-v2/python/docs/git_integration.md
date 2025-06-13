# Git Integration Guide

Obsidian Librarian provides comprehensive Git integration for version control of your Obsidian vaults. This guide covers all Git-related features and how to use them effectively.

## Features

### 1. Automatic Backups
- **Threshold-based**: Automatically commits after a configurable number of changes
- **Time-based**: Commits at regular intervals
- **Smart commit messages**: Automatically generates descriptive commit messages
- **Shutdown backups**: Creates a backup when closing the vault

### 2. Branch Management
- Create experimental branches for trying new ideas
- Switch between branches
- Merge branches back to main
- Automatic branch naming with prefixes

### 3. Restore & Rollback
- Restore vault to any previous commit
- Rollback by a specific number of commits
- Safety features prevent accidental data loss
- Automatic backup before restore operations

### 4. Stash Management
- Stash uncommitted changes
- Apply stashed changes later
- Multiple stashes supported

### 5. Conflict Resolution
- Helpers for resolving merge conflicts
- Support for different resolution strategies

## Installation

The Git integration requires GitPython, which is included in the dependencies:

```bash
pip install obsidian-librarian
```

## Configuration

### Vault Configuration

Enable Git integration when creating a vault:

```python
from obsidian_librarian import Vault, VaultConfig

config = VaultConfig(
    enable_git_integration=True,
    enable_auto_backup=True,
    git_auto_backup_threshold=10,  # Backup after 10 changes
    git_auto_backup_interval=3600,  # Backup every hour
    git_backup_branch_prefix="backup",
    git_experiment_branch_prefix="experiment"
)

vault = Vault(vault_path, config)
await vault.initialize()
```

### Git Service Configuration

For more fine-grained control:

```python
from obsidian_librarian.services import GitConfig

git_config = GitConfig(
    auto_backup_enabled=True,
    auto_backup_threshold=10,
    auto_backup_interval=3600,
    backup_branch_prefix="backup",
    experiment_branch_prefix="experiment",
    stash_before_operations=True,
    create_backup_before_restore=True,
    default_commit_author="Your Name",
    default_commit_email="your.email@example.com",
    ignore_patterns=[".obsidian/workspace*", ".trash/", "*.tmp"]
)
```

## Command Line Usage

### Git Subcommands

The CLI provides a comprehensive `git` subcommand group:

```bash
# Initialize Git repository
obsidian-librarian git init /path/to/vault

# Check status
obsidian-librarian git status /path/to/vault

# Create backup
obsidian-librarian git backup /path/to/vault --message "Your commit message"

# View history
obsidian-librarian git history /path/to/vault --limit 20

# Create branch
obsidian-librarian git branch /path/to/vault --create feature-name

# Restore to commit
obsidian-librarian git restore /path/to/vault abc123

# Rollback commits
obsidian-librarian git rollback /path/to/vault --steps 2

# Stash changes
obsidian-librarian git stash /path/to/vault --message "Work in progress"

# View differences
obsidian-librarian git diff /path/to/vault

# Configure Git settings
obsidian-librarian git config /path/to/vault --auto-backup true --threshold 5
```

### Legacy Commands

For compatibility, the original backup/restore commands are still available:

```bash
# Create backup (deprecated - use git backup)
obsidian-librarian backup /path/to/vault

# Restore vault (deprecated - use git restore)
obsidian-librarian restore /path/to/vault --commit abc123
```

## Python API Usage

### Basic Operations

```python
import asyncio
from obsidian_librarian import Vault, VaultConfig

async def git_operations():
    # Create vault with Git enabled
    config = VaultConfig(enable_git_integration=True)
    vault = Vault(vault_path, config)
    await vault.initialize()
    
    # Create backup
    commit_hash = await vault.git_backup("Manual backup")
    print(f"Created backup: {commit_hash}")
    
    # Check status
    status = await vault.git_status()
    print(f"Current branch: {status['current_branch']}")
    print(f"Changes: {status['total_changes']}")
    
    # View history
    commits = await vault.git_history(10)
    for commit in commits:
        print(f"{commit.hash[:8]} - {commit.message}")
    
    # Create experimental branch
    branch = await vault.git_create_branch("new-feature")
    print(f"Created branch: {branch}")
    
    # Stash changes
    await vault.git_stash("Work in progress")
    
    # Restore to previous commit
    await vault.git_restore(commit_hash)
    
    # Rollback one commit
    await vault.git_rollback(1)
    
    await vault.close()

asyncio.run(git_operations())
```

### Advanced Usage

```python
# Direct access to GitService
git_service = vault.git_service

# Get detailed diff
diffs = await git_service.get_diff()

# Merge branch
success = await git_service.merge_branch("experiment/new-feature")

# Resolve conflicts
await git_service.resolve_conflicts(strategy="ours")

# Clean up old backup branches
deleted = await git_service.cleanup_old_backups(keep_last=10)

# Manual change registration (for custom integrations)
git_service.register_change()

# Check time-based backup
await git_service.check_time_based_backup()
```

## Auto-Backup Behavior

### Change Threshold

The system tracks changes and automatically creates a backup when the threshold is reached:

1. File created → +1 change
2. File modified → +1 change
3. File deleted → +1 change
4. File moved → +1 change

When changes reach the configured threshold, an auto-backup is triggered.

### Time-Based Backups

Independent of change counting, the system can create backups at regular intervals. This ensures your work is saved even during long editing sessions with few file changes.

### Integration with File Watcher

When file watching is enabled, all file system events automatically register changes:

```python
# File watcher integration is automatic
config = VaultConfig(
    enable_file_watching=True,
    enable_git_integration=True,
    enable_auto_backup=True
)
```

## Best Practices

### 1. Commit Messages

While the system generates smart commit messages, you can provide custom ones:

```python
# Good: Descriptive message
await vault.git_backup("Add weekly review template and update project notes")

# The system generates messages like:
# "Added 3 files, Modified 2 files (note1, note2)"
# "[Auto-backup] Modified 5 files at 2024-01-15 10:30:00 UTC"
```

### 2. Branch Strategy

Use branches for experiments:

```python
# Create branch for trying new organization
branch = await vault.git_create_branch("reorganize-2024")

# Make changes...

# If happy, merge back
await git_service.merge_branch(branch)

# If not, just switch back to main
# (checkout functionality to be implemented)
```

### 3. Regular Maintenance

Clean up old backup branches periodically:

```python
# Keep only last 10 backup branches
await git_service.cleanup_old_backups(keep_last=10)
```

### 4. Safety First

Always use the safety features:

```python
# Configure safety options
config = GitConfig(
    stash_before_operations=True,      # Stash before restore
    create_backup_before_restore=True  # Create backup branch
)
```

## Troubleshooting

### Repository Not Initialized

```python
status = await vault.git_status()
if not status.get('initialized'):
    await vault.git_service.initialize_repo()
```

### Merge Conflicts

```python
# Try automatic resolution
success = await git_service.resolve_conflicts("ours")  # or "theirs"

if not success:
    # Manual resolution required
    print("Please resolve conflicts manually")
```

### Uncommitted Changes

```python
# Check for changes
status = await vault.git_status()
if status['total_changes'] > 0:
    # Option 1: Commit them
    await vault.git_backup("Save current work")
    
    # Option 2: Stash them
    await vault.git_stash("Work in progress")
```

## Examples

See the `examples/git_integration_demo.py` script for a comprehensive demonstration of all Git features.

## Future Enhancements

Planned features for future releases:

1. **Remote repository support**: Push/pull to GitHub, GitLab, etc.
2. **Advanced conflict resolution**: Three-way merge UI
3. **Git LFS support**: For large attachments
4. **Hooks integration**: Pre-commit, post-commit hooks
5. **Blame/annotation view**: See who changed what
6. **Cherry-pick support**: Select specific commits
7. **Interactive rebase**: Clean up commit history

## API Reference

For detailed API documentation, see the docstrings in:
- `obsidian_librarian/services/git_service.py`
- `obsidian_librarian/vault.py` (Git-related methods)