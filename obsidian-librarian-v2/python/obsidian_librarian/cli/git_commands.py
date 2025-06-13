"""
Git-specific CLI commands for Obsidian Librarian.

This module provides comprehensive Git functionality through the CLI,
including backup, restore, branch management, and history operations.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import subprocess

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich import print as rprint

from ..vault import Vault
from ..models import VaultConfig
from ..services.git_service import GitService, GitConfig
from ..config_loader import load_vault_config_from_yaml

# Initialize console
console = Console()

# Create Git subcommand app
git_app = typer.Typer(
    name="git",
    help="Git-based version control for Obsidian vaults",
    add_completion=False,
    rich_markup_mode="rich",
)


@git_app.command()
def init(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    message: str = typer.Option("Initial commit", "--message", "-m", help="Initial commit message"),
) -> None:
    """Initialize Git repository in vault."""
    async def run_init():
        console.print(Panel.fit(
            "[bold blue]Git Repository Initialization[/bold blue]",
            border_style="blue"
        ))
        
        # Check if already a Git repo
        git_dir = vault_path / ".git"
        if git_dir.exists():
            console.print("[yellow]Vault is already a Git repository[/yellow]")
            return
        
        # Create vault instance
        config = VaultConfig(enable_git_integration=True)
        vault = Vault(vault_path, config)
        
        try:
            await vault.initialize()
            
            # Initialize Git
            if vault.git_service:
                success = await vault.git_service.initialize_repo()
                if success:
                    console.print("[green]✓ Git repository initialized[/green]")
                    
                    # Create initial backup
                    commit_hash = await vault.git_service.backup(message)
                    if commit_hash:
                        console.print(f"[green]✓ Created initial commit: {commit_hash[:8]}[/green]")
                else:
                    console.print("[red]Failed to initialize Git repository[/red]")
            
        finally:
            await vault.close()
    
    asyncio.run(run_init())


@git_app.command()
def status(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed status"),
) -> None:
    """Show Git repository status."""
    async def run_status():
        # Load configuration from YAML file
        config = load_vault_config_from_yaml(vault_path)
        if not config:
            # If no config file, create a default one with git disabled
            config = VaultConfig(enable_git_integration=False)
        
        vault = Vault(vault_path, config)
        
        try:
            await vault.initialize()
            
            if not vault.git_service:
                console.print("[red]Git integration not enabled[/red]")
                return
            
            status = await vault.git_service.get_status()
            
            if not status.get('initialized'):
                console.print("[red]Not a Git repository[/red]")
                return
            
            # Display status
            console.print(Panel.fit(
                f"[bold blue]Git Status: {vault_path.name}[/bold blue]",
                border_style="blue"
            ))
            
            # Basic info
            console.print(f"[cyan]Current branch:[/cyan] {status['current_branch']}")
            console.print(f"[cyan]Total changes:[/cyan] {status['total_changes']}")
            
            if status.get('stash_count', 0) > 0:
                console.print(f"[yellow]Stashes:[/yellow] {status['stash_count']}")
            
            # Changes summary
            changes = status['changes']
            if any(changes.values()):
                console.print("\n[bold]Changes:[/bold]")
                if changes['added']:
                    console.print(f"  [green]Added:[/green] {len(changes['added'])} files")
                if changes['modified']:
                    console.print(f"  [yellow]Modified:[/yellow] {len(changes['modified'])} files")
                if changes['deleted']:
                    console.print(f"  [red]Deleted:[/red] {len(changes['deleted'])} files")
                
                if detailed:
                    console.print("\n[bold]File details:[/bold]")
                    for file in changes['added'][:5]:
                        console.print(f"  [green]+[/green] {file}")
                    for file in changes['modified'][:5]:
                        console.print(f"  [yellow]M[/yellow] {file}")
                    for file in changes['deleted'][:5]:
                        console.print(f"  [red]-[/red] {file}")
                    
                    total_files = sum(len(v) for v in changes.values())
                    if total_files > 15:
                        console.print(f"  [dim]... and {total_files - 15} more files[/dim]")
            else:
                console.print("\n[green]Working directory clean[/green]")
            
            # Last commit info
            if status.get('last_commit'):
                commit = status['last_commit']
                console.print(f"\n[bold]Last commit:[/bold]")
                console.print(f"  [cyan]Hash:[/cyan] {commit.hash[:8]}")
                console.print(f"  [cyan]Author:[/cyan] {commit.author}")
                console.print(f"  [cyan]Date:[/cyan] {commit.date.strftime('%Y-%m-%d %H:%M')}")
                console.print(f"  [cyan]Message:[/cyan] {commit.message}")
            
        finally:
            await vault.close()
    
    asyncio.run(run_status())


@git_app.command()
def backup(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Commit message"),
    auto: bool = typer.Option(False, "--auto", "-a", help="Mark as auto-backup"),
) -> None:
    """Create a Git backup of vault changes."""
    async def run_backup():
        # Load configuration from YAML file
        config = load_vault_config_from_yaml(vault_path)
        if not config:
            config = VaultConfig(enable_git_integration=False)
        
        vault = Vault(vault_path, config)
        
        try:
            await vault.initialize()
            
            if not vault.git_service:
                console.print("[red]Git integration not enabled[/red]")
                return
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Creating backup...", total=None)
                
                commit_hash = await vault.git_service.backup(message, auto)
                
                if commit_hash:
                    progress.update(task, description="Backup complete!")
                    console.print(f"[green]✓ Created backup: {commit_hash[:8]}[/green]")
                else:
                    console.print("[yellow]No changes to backup[/yellow]")
            
        finally:
            await vault.close()
    
    asyncio.run(run_backup())


@git_app.command()
def history(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of commits to show"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Branch to show history for"),
) -> None:
    """Show Git commit history."""
    async def run_history():
        # Load configuration from YAML file
        config = load_vault_config_from_yaml(vault_path)
        if not config:
            config = VaultConfig(enable_git_integration=False)
        
        vault = Vault(vault_path, config)
        
        try:
            await vault.initialize()
            
            if not vault.git_service:
                console.print("[red]Git integration not enabled[/red]")
                return
            
            commits = await vault.git_service.get_history(limit, branch)
            
            if not commits:
                console.print("[yellow]No commits found[/yellow]")
                return
            
            # Display history
            console.print(Panel.fit(
                f"[bold blue]Commit History{f' ({branch})' if branch else ''}[/bold blue]",
                border_style="blue"
            ))
            
            for commit in commits:
                # Commit header
                console.print(f"\n[yellow]{commit.hash[:8]}[/yellow] - {commit.date.strftime('%Y-%m-%d %H:%M')}")
                console.print(f"Author: {commit.author}")
                console.print(f"Message: {commit.message}")
                
                # Stats
                if commit.files_changed > 0:
                    stats = f"{commit.files_changed} files changed"
                    if commit.insertions > 0:
                        stats += f", [green]+{commit.insertions}[/green]"
                    if commit.deletions > 0:
                        stats += f", [red]-{commit.deletions}[/red]"
                    console.print(f"[dim]{stats}[/dim]")
            
        finally:
            await vault.close()
    
    asyncio.run(run_history())


@git_app.command()
def restore(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    commit: str = typer.Argument(..., help="Commit hash or reference to restore to"),
    force: bool = typer.Option(False, "--force", "-f", help="Force restore without confirmation"),
) -> None:
    """Restore vault to a specific commit."""
    async def run_restore():
        if not force:
            console.print(f"[yellow]Warning: This will restore vault to commit {commit}[/yellow]")
            console.print("[yellow]All current changes will be lost![/yellow]")
            if not Confirm.ask("Are you sure you want to continue?"):
                return
        
        # Load configuration from YAML file
        config = load_vault_config_from_yaml(vault_path)
        if not config:
            config = VaultConfig(enable_git_integration=False)
        
        vault = Vault(vault_path, config)
        
        try:
            await vault.initialize()
            
            if not vault.git_service:
                console.print("[red]Git integration not enabled[/red]")
                return
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Restoring vault...", total=None)
                
                success = await vault.git_service.restore(commit)
                
                if success:
                    progress.update(task, description="Restore complete!")
                    console.print(f"[green]✓ Successfully restored to commit {commit[:8]}[/green]")
                else:
                    console.print("[red]Failed to restore vault[/red]")
            
        finally:
            await vault.close()
    
    asyncio.run(run_restore())


@git_app.command()
def rollback(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    steps: int = typer.Option(1, "--steps", "-s", help="Number of commits to roll back"),
    force: bool = typer.Option(False, "--force", "-f", help="Force rollback without confirmation"),
) -> None:
    """Rollback vault by a number of commits."""
    async def run_rollback():
        if not force:
            console.print(f"[yellow]Warning: This will rollback {steps} commit(s)[/yellow]")
            if not Confirm.ask("Are you sure?"):
                return
        
        # Load configuration from YAML file
        config = load_vault_config_from_yaml(vault_path)
        if not config:
            config = VaultConfig(enable_git_integration=False)
        
        vault = Vault(vault_path, config)
        
        try:
            await vault.initialize()
            
            if not vault.git_service:
                console.print("[red]Git integration not enabled[/red]")
                return
            
            success = await vault.git_service.rollback(steps)
            
            if success:
                console.print(f"[green]✓ Successfully rolled back {steps} commit(s)[/green]")
            else:
                console.print("[red]Failed to rollback[/red]")
            
        finally:
            await vault.close()
    
    asyncio.run(run_rollback())


@git_app.command()
def branch(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    create: Optional[str] = typer.Option(None, "--create", "-c", help="Create new branch"),
    list_branches: bool = typer.Option(False, "--list", "-l", help="List all branches"),
    checkout: Optional[str] = typer.Option(None, "--checkout", "-o", help="Checkout branch"),
    delete: Optional[str] = typer.Option(None, "--delete", "-d", help="Delete branch"),
) -> None:
    """Manage Git branches for experiments."""
    async def run_branch():
        # Load configuration from YAML file
        config = load_vault_config_from_yaml(vault_path)
        if not config:
            config = VaultConfig(enable_git_integration=False)
        
        vault = Vault(vault_path, config)
        
        try:
            await vault.initialize()
            
            if not vault.git_service:
                console.print("[red]Git integration not enabled[/red]")
                return
            
            status = await vault.git_service.get_status()
            
            # List branches
            if list_branches or (not create and not checkout and not delete):
                console.print("[bold]Branches:[/bold]")
                for branch in status['branches']:
                    if branch == status['current_branch']:
                        console.print(f"  [green]* {branch}[/green]")
                    else:
                        console.print(f"    {branch}")
                return
            
            # Create branch
            if create:
                branch_name = await vault.git_service.create_branch(create)
                if branch_name:
                    console.print(f"[green]✓ Created branch: {branch_name}[/green]")
                else:
                    console.print("[red]Failed to create branch[/red]")
            
            # Checkout branch
            if checkout:
                # This would need to be implemented in GitService
                console.print("[yellow]Branch checkout not yet implemented[/yellow]")
            
            # Delete branch
            if delete:
                if delete == status['current_branch']:
                    console.print("[red]Cannot delete current branch[/red]")
                    return
                
                if Confirm.ask(f"Delete branch '{delete}'?"):
                    # This would need to be implemented in GitService
                    console.print("[yellow]Branch deletion not yet implemented[/yellow]")
            
        finally:
            await vault.close()
    
    asyncio.run(run_branch())


@git_app.command()
def stash(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Stash message"),
    list_stashes: bool = typer.Option(False, "--list", "-l", help="List stashes"),
    pop: bool = typer.Option(False, "--pop", "-p", help="Pop latest stash"),
) -> None:
    """Stash uncommitted changes."""
    async def run_stash():
        # Load configuration from YAML file
        config = load_vault_config_from_yaml(vault_path)
        if not config:
            config = VaultConfig(enable_git_integration=False)
        
        vault = Vault(vault_path, config)
        
        try:
            await vault.initialize()
            
            if not vault.git_service:
                console.print("[red]Git integration not enabled[/red]")
                return
            
            if list_stashes:
                # This would need stash listing implementation
                status = await vault.git_service.get_status()
                console.print(f"[bold]Stashes:[/bold] {status.get('stash_count', 0)}")
                return
            
            if pop:
                # This would need stash pop implementation
                console.print("[yellow]Stash pop not yet implemented[/yellow]")
                return
            
            # Create stash
            success = await vault.git_service.stash_changes(message)
            
            if success:
                console.print("[green]✓ Changes stashed successfully[/green]")
            else:
                console.print("[yellow]No changes to stash[/yellow]")
            
        finally:
            await vault.close()
    
    asyncio.run(run_stash())


@git_app.command()
def diff(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    commit: Optional[str] = typer.Option(None, "--commit", "-c", help="Show diff for specific commit"),
    staged: bool = typer.Option(False, "--staged", "-s", help="Show staged changes only"),
) -> None:
    """Show differences in files."""
    async def run_diff():
        # Load configuration from YAML file
        config = load_vault_config_from_yaml(vault_path)
        if not config:
            config = VaultConfig(enable_git_integration=False)
        
        vault = Vault(vault_path, config)
        
        try:
            await vault.initialize()
            
            if not vault.git_service:
                console.print("[red]Git integration not enabled[/red]")
                return
            
            diffs = await vault.git_service.get_diff(commit, staged)
            
            if not diffs:
                console.print("[green]No differences found[/green]")
                return
            
            # Display diffs
            console.print(Panel.fit(
                f"[bold blue]File Differences{f' (commit {commit[:8]})' if commit else ''}[/bold blue]",
                border_style="blue"
            ))
            
            for diff in diffs:
                change_symbol = {
                    'A': '[green]+[/green]',
                    'M': '[yellow]±[/yellow]',
                    'D': '[red]-[/red]',
                }.get(diff['change_type'], '?')
                
                console.print(f"{change_symbol} {diff['file']}")
                
                if diff['additions'] > 0 or diff['deletions'] > 0:
                    console.print(f"   [green]+{diff['additions']}[/green] [red]-{diff['deletions']}[/red]")
            
        finally:
            await vault.close()
    
    asyncio.run(run_diff())


@git_app.command()
def config(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    auto_backup: Optional[bool] = typer.Option(None, "--auto-backup", help="Enable/disable auto-backup"),
    threshold: Optional[int] = typer.Option(None, "--threshold", "-t", help="Auto-backup threshold"),
    interval: Optional[int] = typer.Option(None, "--interval", "-i", help="Auto-backup interval (seconds)"),
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
) -> None:
    """Configure Git integration settings."""
    async def run_config():
        # Load configuration from YAML file
        config = load_vault_config_from_yaml(vault_path)
        if not config:
            config = VaultConfig(enable_git_integration=False)
        
        vault = Vault(vault_path, config)
        
        try:
            await vault.initialize()
            
            if not vault.git_service:
                console.print("[red]Git integration not enabled[/red]")
                return
            
            if show:
                config = vault.git_service.config
                console.print("[bold]Git Configuration:[/bold]")
                console.print(f"  Auto-backup: {'[green]enabled[/green]' if config.auto_backup_enabled else '[red]disabled[/red]'}")
                console.print(f"  Threshold: {config.auto_backup_threshold} changes")
                console.print(f"  Interval: {config.auto_backup_interval} seconds")
                console.print(f"  Backup branch prefix: {config.backup_branch_prefix}")
                console.print(f"  Experiment branch prefix: {config.experiment_branch_prefix}")
                return
            
            # Update configuration
            updated = False
            
            if auto_backup is not None:
                vault.git_service.config.auto_backup_enabled = auto_backup
                updated = True
                console.print(f"[green]✓ Auto-backup {'enabled' if auto_backup else 'disabled'}[/green]")
            
            if threshold is not None:
                vault.git_service.config.auto_backup_threshold = threshold
                updated = True
                console.print(f"[green]✓ Auto-backup threshold set to {threshold}[/green]")
            
            if interval is not None:
                vault.git_service.config.auto_backup_interval = interval
                updated = True
                console.print(f"[green]✓ Auto-backup interval set to {interval} seconds[/green]")
            
            if not updated:
                console.print("[yellow]No configuration changes specified[/yellow]")
                console.print("Use --show to view current configuration")
            
        finally:
            await vault.close()
    
    asyncio.run(run_config())


# Export the Git app for integration with main CLI
__all__ = ["git_app"]