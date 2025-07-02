#!/usr/bin/env python3
"""
Main CLI entry point for Obsidian Vault Tools
"""

# Import configuration to suppress startup warnings
try:
    import ovt_config
except ImportError:
    pass

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
    
    # If no command provided, launch interactive mode
    if ctx.invoked_subcommand is None:
        # Launch the unified interactive manager
        try:
            import sys
            import os
            # Add parent directory to path to import unified_vault_manager
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            from unified_vault_manager import UnifiedVaultManager
            
            # Create and run the unified manager
            manager = UnifiedVaultManager(vault_path=vault)
            manager.run()
            
        except ImportError as e:
            # Fallback to showing help if unified manager not available
            console.print(f"[yellow]Interactive mode not available: {e}[/yellow]")
            console.print("\nAvailable commands:")
            console.print(ctx.get_help())
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            sys.exit(0)

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
    
    click.echo(f"Organization suggestions for: {vault}")
    click.echo(suggestions['summary'])
    
    if suggestions['suggestions']:
        click.echo("\nTop suggestions:")
        # Get first 5 suggestions 
        # Note: Avoiding list() builtin to prevent name collision with the 'list' MCP command
        suggestions_items = []
        for i, (file_path, suggestion) in enumerate(suggestions['suggestions'].items()):
            if i >= 5:
                break
            suggestions_items.append((file_path, suggestion))
        
        for item in suggestions_items:
            file_path, suggestion = item
            click.echo(f"  {file_path}")
            click.echo(f"    Current: {suggestion['current']}")
            click.echo(f"    Suggested: {suggestion['suggested']}")
            click.echo(f"    Reason: {suggestion['reason']}")
            click.echo()

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

@cli.group()
def mcp():
    """MCP (Model Context Protocol) server management"""
    pass

@mcp.command()
def list():
    """List configured MCP servers"""
    try:
        from .mcp_tools import MCPConfig
        from .mcp_tools.setup_wizard import run_setup_wizard_if_needed
        
        # Run setup wizard if this is first time
        if run_setup_wizard_if_needed():
            console.print("\n" + "="*50)
        
        config = MCPConfig()
        servers = config.get_server_status()
        
        if not servers:
            console.print("No MCP servers configured.")
            console.print("Use 'ovt mcp add' to add a server or run the setup wizard again.")
            return
        
        from rich.table import Table
        table = Table(title="MCP Servers")
        table.add_column("Name", style="cyan")
        table.add_column("Command", style="green") 
        table.add_column("Status", style="yellow")
        table.add_column("Ready", style="bold")
        
        for name, info in servers.items():
            command = info["config"].get("command", "N/A")
            validation = info["validation"]
            status = "✓ Valid" if validation["command_exists"] else "✗ Invalid"
            ready = "✓ Ready" if info["ready"] else "✗ Not Ready"
            
            table.add_row(name, command, status, ready)
        
        console.print(table)
        
    except ImportError:
        console.print("[red]MCP features require additional dependencies: pip install mcp cryptography[/red]")

@mcp.command()
@click.argument('name')
def status(name):
    """Show status of a specific MCP server"""
    try:
        from .mcp_tools import get_client_manager
        import asyncio
        
        manager = get_client_manager()
        server_status = manager.get_server_status(name)
        
        if not server_status:
            console.print(f"[red]Server '{name}' not found[/red]")
            return
        
        console.print(f"Server: {name}")
        console.print(f"Running: {'✓ Yes' if server_status['running'] else '✗ No'}")
        if server_status.get('pid'):
            console.print(f"PID: {server_status['pid']}")
        if server_status.get('uptime'):
            console.print(f"Uptime: {server_status['uptime']:.1f}s")
        if server_status.get('last_error'):
            console.print(f"[red]Last Error: {server_status['last_error']}[/red]")
            
    except ImportError:
        console.print("[red]MCP features require additional dependencies: pip install mcp cryptography[/red]")

@mcp.command()
@click.argument('name')
def start(name):
    """Start an MCP server"""
    try:
        from .mcp_tools import get_client_manager
        import asyncio
        
        async def start_server():
            manager = get_client_manager()
            success = await manager.start_server(name)
            if success:
                console.print(f"[green]✓ Started MCP server: {name}[/green]")
            else:
                console.print(f"[red]✗ Failed to start MCP server: {name}[/red]")
        
        asyncio.run(start_server())
        
    except ImportError:
        console.print("[red]MCP features require additional dependencies: pip install mcp cryptography[/red]")

@mcp.command()
@click.argument('name')
def stop(name):
    """Stop an MCP server"""
    try:
        from .mcp_tools import get_client_manager
        import asyncio
        
        async def stop_server():
            manager = get_client_manager()
            success = await manager.stop_server(name)
            if success:
                console.print(f"[green]✓ Stopped MCP server: {name}[/green]")
            else:
                console.print(f"[red]✗ Failed to stop MCP server: {name}[/red]")
        
        asyncio.run(stop_server())
        
    except ImportError:
        console.print("[red]MCP features require additional dependencies: pip install mcp cryptography[/red]")

@mcp.command()
@click.argument('name')
@click.argument('template', type=click.Choice(['github', 'memory', 'confluence', 'obsidian-pm']))
@click.option('--script-path', help='Path to script file (for obsidian-pm template)')
def add(name, template, script_path):
    """Add a new MCP server from template"""
    try:
        from .mcp_tools import MCPConfig
        config = MCPConfig()
        
        kwargs = {}
        if script_path:
            kwargs['script_path'] = script_path
        
        success = config.create_server_from_template(name, template, **kwargs)
        if success:
            console.print(f"[green]✓ Added MCP server '{name}' from template '{template}'[/green]")
            console.print("Configure credentials with 'ovt mcp credentials'")
        else:
            console.print(f"[red]✗ Failed to add server from template '{template}'[/red]")
            
    except ImportError:
        console.print("[red]MCP features require additional dependencies: pip install mcp cryptography[/red]")

@mcp.command()
def credentials():
    """Manage MCP credentials"""
    try:
        from .mcp_tools import get_credential_manager
        cred_manager = get_credential_manager()
        
        console.print("Stored credentials:")
        creds = cred_manager.list_credentials()
        
        if not creds:
            console.print("No credentials stored.")
        else:
            for cred in creds:
                console.print(f"  - {cred}")
        
        console.print("\nTo add credentials, use environment variables or you'll be prompted when starting servers.")
        
    except ImportError:
        console.print("[red]MCP features require additional dependencies: pip install mcp cryptography[/red]")

@mcp.command(name='check-requirements')
def check_requirements():
    """Check system requirements for MCP servers"""
    try:
        import shutil
        from rich.table import Table
        
        table = Table(title="MCP Server Requirements")
        table.add_column("Requirement", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details", style="dim")
        
        # Check Node.js
        node_available = shutil.which('node') is not None
        node_status = "✓ Installed" if node_available else "✗ Not Found"
        node_details = "Required for most MCP servers" if not node_available else ""
        table.add_row("Node.js", node_status, node_details)
        
        # Check NPX
        npx_available = shutil.which('npx') is not None
        npx_status = "✓ Installed" if npx_available else "✗ Not Found"
        npx_details = "Required for NPM-based servers" if not npx_available else ""
        table.add_row("NPX", npx_status, npx_details)
        
        # Check Docker
        docker_available = shutil.which('docker') is not None
        docker_status = "✓ Installed" if docker_available else "✗ Not Found"
        docker_details = "Required for Atlassian servers" if not docker_available else ""
        table.add_row("Docker", docker_status, docker_details)
        
        # Check Git
        git_available = shutil.which('git') is not None
        git_status = "✓ Installed" if git_available else "✗ Not Found"
        git_details = "Required for GitHub server" if not git_available else ""
        table.add_row("Git", git_status, git_details)
        
        console.print(table)
        
        # Installation instructions
        if not all([node_available, npx_available]):
            console.print("\n[yellow]To install Node.js and NPX:[/yellow]")
            console.print("  • macOS: brew install node")
            console.print("  • Ubuntu: sudo apt install nodejs npm")
            console.print("  • Windows: Download from https://nodejs.org")
        
        if not docker_available:
            console.print("\n[yellow]To install Docker:[/yellow]")
            console.print("  • Visit https://www.docker.com/get-started")
            console.print("  • Download Docker Desktop for your platform")
        
        # Check MCP config
        from .mcp_tools import MCPConfig
        config = MCPConfig()
        servers = config.list_servers()
        
        if servers:
            console.print(f"\n[green]✓ {len(servers)} MCP server(s) configured[/green]")
        else:
            console.print("\n[yellow]No MCP servers configured yet[/yellow]")
            console.print("Run 'ovt mcp add' to configure servers")
            
    except Exception as e:
        console.print(f"[red]Error checking requirements: {e}[/red]")

@mcp.command()
def audit():
    """Audit repository for credential exposure"""
    try:
        import subprocess
        import re
        
        console.print("🔍 Auditing repository for credential exposure...")
        
        # Patterns that might indicate credentials
        credential_patterns = [
            r'[a-zA-Z0-9]{20,}',  # Long alphanumeric strings
            r'ghp_[a-zA-Z0-9]{36}',  # GitHub tokens
            r'sk-[a-zA-Z0-9]{48}',  # OpenAI API keys  
            r'xoxb-[a-zA-Z0-9-]{72}',  # Slack tokens
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email addresses
        ]
        
        # Files to check
        result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True)
        if result.returncode != 0:
            console.print("[red]Error: Not a git repository or git not available[/red]")
            return
        
        files_to_check = [f for f in result.stdout.strip().split('\n') 
                         if f and not f.startswith('.git') and f.endswith(('.py', '.json', '.md', '.txt', '.env'))]
        
        issues_found = []
        
        for file_path in files_to_check:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for line_num, line in enumerate(content.split('\n'), 1):
                    # Skip lines that are clearly examples or templates
                    if any(marker in line.lower() for marker in ['[your_', '[example', 'placeholder', 'template', 'example.com']):
                        continue
                        
                    for pattern in credential_patterns:
                        matches = re.findall(pattern, line)
                        for match in matches:
                            if len(match) > 10:  # Only flag longer strings
                                issues_found.append({
                                    'file': file_path,
                                    'line': line_num,
                                    'match': match[:20] + '...' if len(match) > 20 else match,
                                    'type': 'potential_credential'
                                })
            except Exception as e:
                continue
        
        if issues_found:
            console.print(f"[yellow]⚠️  Found {len(issues_found)} potential credential exposure(s):[/yellow]")
            for issue in issues_found[:10]:  # Limit output
                console.print(f"  {issue['file']}:{issue['line']} - {issue['match']}")
            if len(issues_found) > 10:
                console.print(f"  ... and {len(issues_found) - 10} more")
        else:
            console.print("[green]✅ No credential exposure detected in tracked files[/green]")
        
        # Check .gitignore effectiveness
        gitignore_path = Path('.gitignore')
        if gitignore_path.exists():
            console.print("[green]✅ .gitignore file exists[/green]")
        else:
            console.print("[red]❌ No .gitignore file found[/red]")
        
        console.print("\n🔒 Security reminders:")
        console.print("• Never commit actual credentials to version control")
        console.print("• Use environment variables or encrypted credential storage")
        console.print("• Review files before committing with 'git diff --cached'")
        
    except ImportError:
        console.print("[red]MCP features require additional dependencies: pip install mcp cryptography[/red]")
    except Exception as e:
        console.print(f"[red]Audit failed: {e}[/red]")

@cli.group()
def config():
    """Configuration management commands"""
    pass

@config.command(name='set-vault')
@click.argument('vault_path', type=click.Path(exists=True))
def set_vault(vault_path):
    """Set the default vault path"""
    from .utils import Config
    config = Config()
    
    # Validate the vault path
    from .utils import validate_vault_path
    valid, message = validate_vault_path(vault_path)
    
    if not valid:
        console.print(f"[red]Invalid vault path: {message}[/red]")
        return
    
    # Save the vault path
    if config.set_vault_path(vault_path):
        console.print(f"[green]✓ Default vault path set to: {vault_path}[/green]")
    else:
        console.print(f"[red]Failed to set vault path[/red]")

@config.command(name='show')
def show_config():
    """Show current configuration"""
    from .utils import Config
    config = Config()
    
    # Display configuration
    console.print("\n[bold]Current Configuration:[/bold]")
    
    vault_path = config.get_vault_path()
    if vault_path:
        console.print(f"Default Vault: [green]{vault_path}[/green]")
    else:
        console.print("Default Vault: [yellow]Not set[/yellow]")
    
    # Show config file location
    console.print(f"\nConfig File: [dim]{config.config_file}[/dim]")
    
    # Show other settings if available
    if config.config.get("output_directory"):
        console.print(f"Output Directory: {config.config['output_directory']}")
    
    if config.config.get("backup_settings"):
        backup = config.config["backup_settings"]
        console.print(f"\nBackup Settings:")
        console.print(f"  Auto Backup: {backup.get('auto_backup', False)}")
        console.print(f"  Backup Count: {backup.get('backup_count', 5)}")

@config.command(name='reset')
@click.confirmation_option(prompt='Are you sure you want to reset all configuration?')
def reset_config():
    """Reset configuration to defaults"""
    from .utils import Config
    config = Config()
    
    # Reset to default configuration
    config.config = config._default_config()
    config.save_config()
    
    console.print("[green]✓ Configuration reset to defaults[/green]")

@cli.command()
def version():
    """Show version information"""
    from . import __version__
    console.print(f"Obsidian Vault Tools version {__version__}")

@cli.command(name='check-system')
def check_system():
    """Check system requirements and dependencies"""
    from .system_requirements import SystemRequirementsChecker
    
    console.print("[bold]System Requirements Check for Obsidian Vault Tools[/bold]\n")
    
    checker = SystemRequirementsChecker()
    results = checker.check_all()
    
    # If running on first launch or with missing deps, show detailed report
    checker.print_report()
    
    # Return appropriate exit code
    if results['missing_deps']:
        sys.exit(1)
    else:
        sys.exit(0)

def check_virtual_env():
    """Check if running in virtual environment and show warning if not"""
    import platform
    
    # Only show warning on Linux/Ubuntu
    if platform.system() != 'Linux':
        return
    
    # Check if in virtual environment
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
               os.environ.get('VIRTUAL_ENV'))
    
    if not in_venv and not os.environ.get('OVT_SKIP_VENV_CHECK'):
        console.print("[yellow]⚠️  Not running in a virtual environment[/yellow]")
        console.print("[dim]This is recommended on Ubuntu to avoid conflicts with system packages[/dim]")
        console.print("[dim]To create a virtual environment:[/dim]")
        console.print("[dim]  python3 -m venv ~/ovt-env[/dim]")
        console.print("[dim]  source ~/ovt-env/bin/activate[/dim]")
        console.print("[dim]  pip install obsidian-vault-tools[all][/dim]")
        console.print("[dim]Set OVT_SKIP_VENV_CHECK=1 to hide this message[/dim]\n")

def main():
    """Main entry point"""
    try:
        # Check virtual environment on startup
        check_virtual_env()
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

if __name__ == '__main__':
    main()