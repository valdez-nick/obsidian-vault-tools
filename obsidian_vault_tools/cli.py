#!/usr/bin/env python3
"""
Main CLI entry point for Obsidian Vault Tools
"""
import click
import sys
import os
from pathlib import Path

console = None
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    # Fallback if rich is not available
    class SimpleConsole:
        def print(self, text):
            print(text)
    console = SimpleConsole()

@click.group(invoke_without_command=True)
@click.option('--vault', '-v', help='Path to Obsidian vault', type=click.Path(exists=True))
@click.option('--config', '-c', help='Path to config file', type=click.Path())
@click.pass_context
def cli(ctx, vault, config):
    """Obsidian Vault Tools - Comprehensive toolkit for managing Obsidian vaults"""
    
    # Store vault path in context
    ctx.ensure_object(dict)
    ctx.obj['vault'] = vault
    ctx.obj['config'] = config
    
    # If no command provided, show help or interactive mode
    if ctx.invoked_subcommand is None:
        if vault:
            # Basic vault info
            console.print(f"Vault: {vault}")
            # For now, just show help since interactive mode needs more setup
            ctx.get_help()
        else:
            # Show help
            console.print("Obsidian Vault Tools")
            console.print("\nUse --vault to specify vault path or run 'ovt config set-vault' to set default vault.")
            console.print("For commands: ovt [command] --help")
            ctx.get_help()

@cli.command()
@click.option('--vault', '-v', help='Path to Obsidian vault', type=click.Path(exists=True))
def tags(vault):
    """Analyze tags in the vault"""
    if not vault:
        from .utils import Config
        config = Config()
        vault = config.get_vault_path()
        if not vault:
            console.print("No vault specified. Use --vault or set default vault with 'ovt config set-vault'")
            return
    
    from .analysis import TagAnalyzer
    analyzer = TagAnalyzer(vault)
    result = analyzer.analyze_all_tags()
    
    console.print(f"Tag Analysis for: {vault}")
    console.print(f"Total files: {result['total_files']}")
    console.print(f"Total unique tags: {result['total_tags']}")
    console.print("\nMost used tags:")
    for tag, count in result['most_used_tags'][:5]:
        console.print(f"  #{tag}: {count} files")

@cli.command()
@click.option('--vault', '-v', help='Path to Obsidian vault', type=click.Path(exists=True))
def backup(vault):
    """Create a backup of the vault"""
    if not vault:
        from .utils import Config
        config = Config()
        vault = config.get_vault_path()
        if not vault:
            console.print("No vault specified. Use --vault or set default vault with 'ovt config set-vault'")
            return
    
    from .backup import BackupManager
    manager = BackupManager(vault)
    result = manager.create_backup()
    
    if result['success']:
        console.print(f"Backup created: {result['backup_name']}")
        console.print(f"Size: {result['backup_size']} bytes")
    else:
        console.print(f"Backup failed: {result['error']}")

@cli.command()
@click.option('--vault', '-v', help='Path to Obsidian vault', type=click.Path(exists=True))
def organize(vault):
    """Suggest file organization for the vault"""
    if not vault:
        from .utils import Config
        config = Config()
        vault = config.get_vault_path()
        if not vault:
            console.print("No vault specified. Use --vault or set default vault with 'ovt config set-vault'")
            return
    
    from .organization import FileOrganizer
    organizer = FileOrganizer(vault)
    suggestions = organizer.suggest_folder_structure()
    
    console.print(f"Organization suggestions for: {vault}")
    console.print(suggestions['summary'])
    
    if suggestions['suggestions']:
        console.print("\nTop suggestions:")
        for file_path, suggestion in list(suggestions['suggestions'].items())[:5]:
            console.print(f"  {file_path}")
            console.print(f"    Current: {suggestion['current']}")
            console.print(f"    Suggested: {suggestion['suggested']}")
            console.print(f"    Reason: {suggestion['reason']}")
            console.print()

@cli.group()
def config():
    """Configuration management"""
    pass

@config.command()
@click.argument('vault_path', type=click.Path(exists=True))
def set_vault(vault_path):
    """Set default vault path"""
    from .utils import Config
    config = Config()
    config.set_vault_path(vault_path)
    console.print(f"Default vault set to: {vault_path}")

@config.command()
def show():
    """Show current configuration"""
    from .utils import Config
    config = Config()
    
    console.print("Configuration:")
    console.print(f"  Vault path: {config.get_vault_path()}")
    console.print(f"  Config file: {config.config_file}")

@cli.command()
@click.option('--vault', '-v', help='Path to Obsidian vault', type=click.Path(exists=True))
def analyze_vault(vault):
    """Analyze vault structure and health"""
    if not vault:
        from .utils import Config
        config = Config()
        vault = config.get_vault_path()
        if not vault:
            console.print("No vault specified. Use --vault or set default vault with 'ovt config set-vault'")
            return
    
    from .analysis import VaultAnalyzer
    analyzer = VaultAnalyzer(vault)
    report = analyzer.generate_health_report()
    
    console.print(f"Vault: {vault}")
    console.print(f"Total files: {report['structure']['total_files']}")
    console.print(f"Total tags: {report['tags']['total_tags']}")
    console.print(f"Health score: {report['health_score']}")

@cli.command()
@click.option('--vault', '-v', help='Path to Obsidian vault', type=click.Path(exists=True))
def interactive(vault):
    """Start interactive menu system"""
    if not vault:
        from .utils import Config
        config = Config()
        vault = config.get_vault_path()
        if not vault:
            console.print("No vault specified. Use --vault or set default vault with 'ovt config set-vault'")
            return
    
    try:
        # Import the enhanced manager
        sys.path.append(str(Path(__file__).parent.parent))
        from vault_manager_enhanced import EnhancedVaultManager
        
        # Create and configure the manager
        manager = EnhancedVaultManager()
        if vault:
            manager.current_vault = vault
            manager.config['current_vault'] = vault
            manager.save_config()
        
        # Start the interactive system
        manager.run()
        
    except ImportError as e:
        console.print(f"[red]Error loading interactive system: {e}[/red]")
        console.print("Make sure all dependencies are installed: pip install pygame pillow")
    except Exception as e:
        console.print(f"[red]Error starting interactive system: {e}[/red]")

@cli.command()
def version():
    """Show version information"""
    from . import __version__
    console.print(f"Obsidian Vault Tools version {__version__}")

def main():
    """Main entry point"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

if __name__ == '__main__':
    main()