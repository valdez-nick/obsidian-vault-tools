#!/usr/bin/env python3
"""
Demonstration of Git integration features in Obsidian Librarian.

This script shows how to use the Git service for:
- Automatic backups
- Branch management for experiments
- Restore and rollback functionality
- Smart commit messages
"""

import asyncio
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from obsidian_librarian import Vault, VaultConfig, GitService
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def main():
    """Demonstrate Git integration features."""
    # Use a test vault path (change this to your vault path)
    vault_path = Path.home() / "Documents" / "TestVault"
    
    if not vault_path.exists():
        console.print(f"[yellow]Creating test vault at: {vault_path}[/yellow]")
        vault_path.mkdir(parents=True)
    
    # Configure vault with Git integration
    config = VaultConfig(
        enable_git_integration=True,
        enable_auto_backup=True,
        git_auto_backup_threshold=5,
        git_auto_backup_interval=300,  # 5 minutes
    )
    
    # Create vault instance
    vault = Vault(vault_path, config)
    
    try:
        # Initialize vault
        console.print(Panel.fit(
            "[bold blue]Obsidian Librarian Git Integration Demo[/bold blue]",
            border_style="blue"
        ))
        
        await vault.initialize()
        console.print("[green]✓ Vault initialized[/green]")
        
        # 1. Show Git status
        console.print("\n[bold]1. Git Repository Status[/bold]")
        status = await vault.git_status()
        
        if status.get('initialized'):
            console.print(f"  Branch: {status['current_branch']}")
            console.print(f"  Changes: {status['total_changes']}")
            console.print(f"  Auto-backup: {'Enabled' if status['auto_backup_enabled'] else 'Disabled'}")
        else:
            console.print("  [yellow]Repository not initialized[/yellow]")
        
        # 2. Create some test notes
        console.print("\n[bold]2. Creating Test Notes[/bold]")
        
        notes = [
            ("Daily Notes/2024-01-15.md", "# Daily Note\n\n## Tasks\n- [ ] Review Git integration\n- [ ] Test backup functionality"),
            ("Projects/Git Integration.md", "# Git Integration Project\n\n## Overview\nImplementing comprehensive Git support."),
            ("Ideas/Backup Strategy.md", "# Backup Strategy\n\nIdeas for automated backups:\n1. Time-based\n2. Change-based\n3. Event-based"),
        ]
        
        for path, content in notes:
            note_path = vault_path / path
            note_path.parent.mkdir(parents=True, exist_ok=True)
            note_path.write_text(content)
            console.print(f"  [green]✓[/green] Created: {path}")
        
        # 3. Create initial backup
        console.print("\n[bold]3. Creating Initial Backup[/bold]")
        commit_hash = await vault.git_backup("Initial vault setup with test notes")
        if commit_hash:
            console.print(f"  [green]✓[/green] Backup created: {commit_hash[:8]}")
        
        # 4. Show commit history
        console.print("\n[bold]4. Commit History[/bold]")
        history = await vault.git_history(5)
        
        history_table = Table(title="Recent Commits")
        history_table.add_column("Hash", style="yellow")
        history_table.add_column("Date", style="cyan")
        history_table.add_column("Message", style="white")
        
        for commit in history:
            history_table.add_row(
                commit.hash[:8],
                commit.date.strftime("%Y-%m-%d %H:%M"),
                commit.message[:50] + "..." if len(commit.message) > 50 else commit.message
            )
        
        console.print(history_table)
        
        # 5. Demonstrate branch creation for experiments
        console.print("\n[bold]5. Creating Experimental Branch[/bold]")
        branch_name = await vault.git_create_branch("ai-templates")
        if branch_name:
            console.print(f"  [green]✓[/green] Created branch: {branch_name}")
            console.print("  Now you can experiment without affecting main branch!")
        
        # 6. Make experimental changes
        console.print("\n[bold]6. Making Experimental Changes[/bold]")
        template_path = vault_path / "Templates" / "AI Research.md"
        template_path.parent.mkdir(exist_ok=True)
        template_path.write_text("""---
tags: [template, ai, research]
---

# {{title}}

## Research Question
{{question}}

## Methodology
{{methodology}}

## Key Findings
{{findings}}

## References
{{references}}
""")
        console.print("  [green]✓[/green] Created AI research template")
        
        # Commit experimental changes
        await vault.git_backup("Add AI research template")
        
        # 7. Show status with changes
        console.print("\n[bold]7. Current Status[/bold]")
        status = await vault.git_status()
        console.print(f"  Current branch: {status['current_branch']}")
        console.print(f"  Total branches: {len(status['branches'])}")
        
        # 8. Demonstrate auto-backup threshold
        console.print("\n[bold]8. Auto-backup Demonstration[/bold]")
        console.print(f"  Auto-backup threshold: {vault.git_service.config.auto_backup_threshold} changes")
        console.print("  Simulating file changes...")
        
        for i in range(3):
            test_file = vault_path / f"test_auto_{i}.md"
            test_file.write_text(f"Test content {i}")
            # Simulate file watcher detection
            await vault._on_file_created(test_file)
            console.print(f"  Change {i+1}: Created {test_file.name}")
        
        console.print(f"  Current change count: {vault.git_service._change_counter}")
        
        # 9. Stash demonstration
        console.print("\n[bold]9. Stash Functionality[/bold]")
        unstaged_file = vault_path / "unstaged_work.md"
        unstaged_file.write_text("# Work in Progress\n\nThis is not ready to commit yet.")
        
        success = await vault.git_stash("Work in progress - AI templates")
        if success:
            console.print("  [green]✓[/green] Changes stashed successfully")
            console.print("  You can now switch branches or restore without losing work")
        
        # 10. Show final summary
        console.print("\n[bold]10. Final Summary[/bold]")
        final_status = await vault.git_status()
        final_history = await vault.git_history(3)
        
        summary_table = Table(title="Git Integration Summary")
        summary_table.add_column("Feature", style="cyan")
        summary_table.add_column("Status", style="green")
        
        summary_table.add_row("Repository", "✓ Initialized")
        summary_table.add_row("Auto-backup", "✓ Enabled")
        summary_table.add_row("Branches", f"{len(final_status['branches'])} branches")
        summary_table.add_row("Commits", f"{len(final_history)} recent commits")
        summary_table.add_row("Stashes", f"{final_status.get('stash_count', 0)} stashes")
        
        console.print(summary_table)
        
        # Cleanup - switch back to main branch
        console.print("\n[yellow]Switching back to main branch...[/yellow]")
        # Would need to implement checkout in GitService
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
    
    finally:
        await vault.close()
        console.print("\n[green]Demo completed![/green]")


if __name__ == "__main__":
    asyncio.run(main())