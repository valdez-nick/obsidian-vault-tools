"""
Interactive setup wizard for MCP configuration
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from .config import MCPConfig
from .credentials import get_credential_manager


class MCPSetupWizard:
    """Interactive MCP setup wizard for new users"""
    
    def __init__(self):
        self.config = MCPConfig()
        self.credential_manager = get_credential_manager()
    
    def run_first_time_setup(self) -> bool:
        """Run first-time setup wizard"""
        print("🔗 Welcome to MCP (Model Context Protocol) Setup!")
        print("This wizard will help you configure MCP servers for enhanced vault functionality.\n")
        
        # Check if user wants to proceed
        proceed = input("Would you like to set up MCP servers now? (y/N): ").lower().strip()
        if proceed not in ['y', 'yes']:
            print("Setup skipped. You can run 'ovt mcp' anytime to configure servers.")
            return False
        
        print("\n" + "="*60)
        print("MCP SERVER SETUP WIZARD")
        print("="*60)
        
        servers_configured = 0
        
        # Offer to set up common servers
        server_options = [
            ('memory', 'Memory Server', 'Persistent conversation memory across sessions'),
            ('github', 'GitHub Server', 'Access GitHub repositories, issues, and PRs'),
            ('web-fetch', 'Web Fetch Server', 'Fetch and analyze web content'),
            ('confluence', 'Confluence/Jira Server', 'Access Atlassian Confluence and Jira'),
            ('obsidian-pm', 'Obsidian PM Intelligence', 'Custom PM intelligence server (if available)')
        ]
        
        for server_key, server_name, description in server_options:
            print(f"\n📋 {server_name}")
            print(f"   {description}")
            
            setup = input(f"   Set up {server_name}? (y/N): ").lower().strip()
            if setup in ['y', 'yes']:
                if self._setup_server(server_key, server_name):
                    servers_configured += 1
        
        print(f"\n✅ Setup complete! Configured {servers_configured} MCP servers.")
        
        if servers_configured > 0:
            print("\nNext steps:")
            print("• Run 'ovt mcp list' to see your configured servers")
            print("• Run 'ovt mcp start <server-name>' to start a server")
            print("• Run 'ovt interactive' to use MCP features in the vault manager")
            print("• Set environment variables or you'll be prompted for credentials when starting servers")
        
        return servers_configured > 0
    
    def _setup_server(self, server_type: str, server_name: str) -> bool:
        """Set up a specific server type"""
        print(f"\n  🔧 Setting up {server_name}...")
        
        # Generate unique server name
        base_name = server_type.replace('-', '_')
        server_instance_name = input(f"  Server name [{base_name}]: ").strip() or base_name
        
        try:
            if server_type == 'memory':
                return self._setup_memory_server(server_instance_name)
            elif server_type == 'github':
                return self._setup_github_server(server_instance_name)
            elif server_type == 'web-fetch':
                return self._setup_web_fetch_server(server_instance_name)
            elif server_type == 'confluence':
                return self._setup_confluence_server(server_instance_name)
            elif server_type == 'obsidian-pm':
                return self._setup_obsidian_pm_server(server_instance_name)
            else:
                print(f"  ❌ Unknown server type: {server_type}")
                return False
        except Exception as e:
            print(f"  ❌ Setup failed: {e}")
            return False
    
    def _setup_memory_server(self, server_name: str) -> bool:
        """Set up memory server"""
        print("  Memory server stores conversation history across sessions.")
        
        # Get memory path
        default_memory_path = str(Path.home() / '.obsidian_vault_tools' / 'memory')
        memory_path = input(f"  Memory storage path [{default_memory_path}]: ").strip() or default_memory_path
        
        # Create directory if it doesn't exist
        Path(memory_path).mkdir(parents=True, exist_ok=True)
        
        # Store credential
        self.credential_manager.set_credential('YOUR_MEMORY_PATH', memory_path)
        
        # Create server config
        success = self.config.create_server_from_template(server_name, 'memory')
        
        if success:
            print(f"  ✅ Memory server '{server_name}' configured!")
            print(f"  📁 Memory path: {memory_path}")
        
        return success
    
    def _setup_github_server(self, server_name: str) -> bool:
        """Set up GitHub server"""
        print("  GitHub server requires a personal access token.")
        print("  Create one at: https://github.com/settings/tokens")
        print("  Required scopes: repo, read:org, read:user")
        
        # Check if token already exists
        existing_token = self.credential_manager.get_credential('YOUR_GITHUB_TOKEN', prompt=False)
        if existing_token:
            print("  ✅ GitHub token already configured.")
        else:
            print("  You'll be prompted for your GitHub token when starting the server.")
            print("  Or set the environment variable: export YOUR_GITHUB_TOKEN='your_token_here'")
        
        # Create server config
        success = self.config.create_server_from_template(server_name, 'github')
        
        if success:
            print(f"  ✅ GitHub server '{server_name}' configured!")
        
        return success
    
    def _setup_web_fetch_server(self, server_name: str) -> bool:
        """Set up web fetch server"""
        print("  Web fetch server can retrieve and analyze web content.")
        print("  No credentials required.")
        
        # Create basic config
        web_fetch_config = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-fetch"]
        }
        
        self.config.add_server(server_name, web_fetch_config)
        print(f"  ✅ Web fetch server '{server_name}' configured!")
        return True
    
    def _setup_confluence_server(self, server_name: str) -> bool:
        """Set up Confluence/Jira server"""
        print("  Confluence server requires Atlassian Cloud credentials.")
        print("  You'll need:")
        print("  • Cloud ID (from your Atlassian URL)")
        print("  • Email address")  
        print("  • API tokens for Confluence and Jira")
        
        print("  You'll be prompted for credentials when starting the server.")
        print("  Or set environment variables:")
        print("    export YOUR_CLOUD_ID='your_cloud_id'")
        print("    export YOUR_EMAIL='your@email.com'")
        print("    export YOUR_CONFLUENCE_TOKEN='your_token'")
        print("    export YOUR_JIRA_TOKEN='your_token'")
        
        # Create server config
        success = self.config.create_server_from_template(server_name, 'confluence')
        
        if success:
            print(f"  ✅ Confluence server '{server_name}' configured!")
        
        return success
    
    def _setup_obsidian_pm_server(self, server_name: str) -> bool:
        """Set up Obsidian PM intelligence server"""
        print("  Custom Obsidian PM intelligence server.")
        
        # Check if script exists at expected location
        default_script = "/Users/nvaldez/Documents/repos/assistant-mcp/src/obsidian-pm-intelligence.js"
        if os.path.exists(default_script):
            script_path = default_script
            print(f"  ✅ Found script at: {script_path}")
        else:
            script_path = input("  Path to obsidian-pm-intelligence.js: ").strip()
            if not script_path or not os.path.exists(script_path):
                print("  ❌ Script not found. Skipping this server.")
                return False
        
        # Get vault and memory paths
        vault_path = self.credential_manager.get_credential('YOUR_VAULT_PATH', prompt=False)
        if not vault_path:
            from ..utils import Config
            utils_config = Config()
            vault_path = utils_config.get_vault_path()
        
        if vault_path:
            self.credential_manager.set_credential('YOUR_VAULT_PATH', vault_path)
        
        memory_path = self.credential_manager.get_credential('YOUR_MEMORY_PATH', prompt=False)
        if not memory_path:
            memory_path = str(Path.home() / '.obsidian_vault_tools' / 'memory')
            self.credential_manager.set_credential('YOUR_MEMORY_PATH', memory_path)
            Path(memory_path).mkdir(parents=True, exist_ok=True)
        
        # Create server config
        success = self.config.create_server_from_template(
            server_name, 'obsidian-pm', script_path=script_path
        )
        
        if success:
            print(f"  ✅ Obsidian PM server '{server_name}' configured!")
            print(f"  📄 Script: {script_path}")
            if vault_path:
                print(f"  📁 Vault: {vault_path}")
            print(f"  💾 Memory: {memory_path}")
        
        return success
    
    def check_first_run(self) -> bool:
        """Check if this is the first run and should show setup wizard"""
        config_file = self.config.config_file
        template_file = self.config.template_file
        
        # If neither config nor template exists, it's first run
        if not config_file.exists() and not template_file.exists():
            return True
        
        # If only template exists (no actual config), offer setup
        if template_file.exists() and not config_file.exists():
            servers = self.config.list_servers()
            return len(servers) == 0
        
        return False


def run_setup_wizard_if_needed():
    """Run setup wizard if this appears to be first time use"""
    wizard = MCPSetupWizard()
    
    if wizard.check_first_run():
        print("\n🔗 First time using MCP features!")
        return wizard.run_first_time_setup()
    
    return False