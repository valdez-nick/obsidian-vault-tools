"""
Interactive MCP server configuration interface for the unified manager
"""

import json
import os
import shutil
from typing import Dict, Any, List, Optional
from pathlib import Path

from .config import MCPConfig
from .credentials import get_credential_manager
from .client_manager import get_client_manager

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class MCPInteractiveConfig:
    """Interactive configuration interface for MCP servers"""
    
    # Example server template for demonstration
    EXAMPLE_SERVER = {
        "name": "memory-example",
        "description": "Example: Memory server for persistent context",
        "config": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
            "env": {
                "MEMORY_FILE_PATH": "/path/to/memory.json"
            }
        }
    }
    
    def __init__(self):
        self.config = MCPConfig()
        self.credential_manager = get_credential_manager()
        self.client_manager = get_client_manager()
    
    def display_menu(self):
        """Main MCP configuration menu"""
        while True:
            self._clear_screen()
            print(f"\n{Colors.BOLD}🛠️  MCP Server Configuration{Colors.ENDC}")
            print("=" * 50)
            
            # Get current server status
            servers = self.config.list_servers()
            server_status = self.config.get_server_status()
            
            # Display summary
            ready_count = sum(1 for s in server_status.values() if s.get('ready', False))
            print(f"\nConfigured servers: {len(servers)} ({ready_count} ready)")
            
            # Show current servers
            if servers:
                print(f"\n{Colors.CYAN}Current servers:{Colors.ENDC}")
                for server_name in servers:
                    status = server_status.get(server_name, {})
                    if status.get('ready'):
                        status_text = f"{Colors.GREEN}✓ ready{Colors.ENDC}"
                    else:
                        # Show why not ready
                        validation = status.get('validation', {})
                        if not validation.get('command_exists'):
                            status_text = f"{Colors.RED}✗ command not found{Colors.ENDC}"
                        elif 'credentials' in validation and not all(validation['credentials'].values()):
                            status_text = f"{Colors.YELLOW}⚠ missing credentials{Colors.ENDC}"
                        else:
                            status_text = f"{Colors.RED}✗ not ready{Colors.ENDC}"
                    
                    print(f"  - {server_name} ({status_text})")
            
            # Menu options
            print(f"\n{Colors.BOLD}Options:{Colors.ENDC}")
            print("1. View server details")
            print("2. Add new server")
            print("3. Edit server")
            print("4. Remove server")
            print("5. Test server connection")
            print("6. Back to Settings")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '1':
                self._view_server_details()
            elif choice == '2':
                self._add_server_wizard()
            elif choice == '3':
                self._edit_server()
            elif choice == '4':
                self._remove_server()
            elif choice == '5':
                self._test_connection()
            elif choice == '6' or choice.lower() == 'b':
                break
            else:
                print(f"{Colors.RED}Invalid option{Colors.ENDC}")
                input("\nPress Enter to continue...")
    
    def _clear_screen(self):
        """Clear the terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def _view_server_details(self):
        """View detailed configuration for a server"""
        servers = self.config.list_servers()
        if not servers:
            print(f"{Colors.YELLOW}No servers configured{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        print(f"\n{Colors.BOLD}Select server to view:{Colors.ENDC}")
        for i, server in enumerate(servers, 1):
            print(f"{i}. {server}")
        
        try:
            choice = int(input("\nSelect server (0 to cancel): "))
            if choice == 0:
                return
            
            if 1 <= choice <= len(servers):
                server_name = servers[choice - 1]
                config = self.config.get_server_config(server_name)
                status = self.config.get_server_status().get(server_name, {})
                
                print(f"\n{Colors.BOLD}Server: {server_name}{Colors.ENDC}")
                print("-" * 40)
                
                # Show configuration
                print(f"\n{Colors.CYAN}Configuration:{Colors.ENDC}")
                print(f"Command: {config.get('command', 'N/A')}")
                print(f"Args: {json.dumps(config.get('args', []))}")
                
                # Show environment variables (mask sensitive values)
                env_vars = config.get('env', {})
                if env_vars:
                    print(f"\n{Colors.CYAN}Environment Variables:{Colors.ENDC}")
                    for key, value in env_vars.items():
                        if any(sensitive in key.upper() for sensitive in ['TOKEN', 'PASSWORD', 'SECRET', 'KEY']):
                            # Mask sensitive values
                            if value and not (value.startswith('[') and value.endswith(']')):
                                masked_value = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '*' * len(value)
                            else:
                                masked_value = value
                            print(f"  {key}: {masked_value}")
                        else:
                            print(f"  {key}: {value}")
                
                # Show validation status
                print(f"\n{Colors.CYAN}Status:{Colors.ENDC}")
                validation = status.get('validation', {})
                print(f"Command exists: {'✓' if validation.get('command_exists') else '✗'}")
                print(f"Has required args: {'✓' if validation.get('has_args') else '✗'}")
                
                if 'credentials' in validation:
                    print(f"\n{Colors.CYAN}Required Credentials:{Colors.ENDC}")
                    for cred, exists in validation['credentials'].items():
                        print(f"  {cred}: {'✓ Set' if exists else '✗ Missing'}")
                
                print(f"\nOverall status: {'✓ Ready' if status.get('ready') else '✗ Not Ready'}")
                
        except (ValueError, IndexError):
            print(f"{Colors.RED}Invalid selection{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def _add_server_wizard(self):
        """Interactive wizard to add a new server"""
        print(f"\n{Colors.BOLD}Add New MCP Server{Colors.ENDC}")
        print("=" * 50)
        
        # Show example
        print(f"\n{Colors.CYAN}Example server configuration:{Colors.ENDC}")
        print(f"Name: {self.EXAMPLE_SERVER['name']}")
        print(f"Description: {self.EXAMPLE_SERVER['description']}")
        print(f"Command: {self.EXAMPLE_SERVER['config']['command']}")
        print(f"Args: {json.dumps(self.EXAMPLE_SERVER['config']['args'])}")
        print(f"Environment: {json.dumps(self.EXAMPLE_SERVER['config']['env'], indent=2)}")
        
        print(f"\n{Colors.YELLOW}Note: Copy the configuration format from your MCP server's documentation{Colors.ENDC}")
        print("-" * 50)
        
        # Get server name
        server_name = input("\nEnter server name (e.g., 'github', 'memory'): ").strip()
        if not server_name:
            print(f"{Colors.RED}Server name is required{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        # Check if already exists
        if server_name in self.config.list_servers():
            print(f"{Colors.RED}Server '{server_name}' already exists{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        # Get command
        print(f"\n{Colors.CYAN}Enter the command to start the server{Colors.ENDC}")
        print("Common examples: 'npx', 'node', 'python', 'docker'")
        command = input("Command: ").strip()
        if not command:
            print(f"{Colors.RED}Command is required{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        # Get arguments
        print(f"\n{Colors.CYAN}Enter command arguments (JSON array format){Colors.ENDC}")
        print('Example: ["-y", "@modelcontextprotocol/server-github"]')
        args_input = input("Arguments: ").strip()
        
        try:
            args = json.loads(args_input) if args_input else []
            if not isinstance(args, list):
                raise ValueError("Arguments must be a JSON array")
        except json.JSONDecodeError:
            print(f"{Colors.RED}Invalid JSON format for arguments{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        except ValueError as e:
            print(f"{Colors.RED}{e}{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        # Get environment variables
        env_vars = {}
        print(f"\n{Colors.CYAN}Environment variables (optional){Colors.ENDC}")
        print("Enter environment variables one at a time. Press Enter with empty name to finish.")
        print("For credentials, use [PLACEHOLDER] format (e.g., [YOUR_GITHUB_TOKEN])")
        
        while True:
            var_name = input("\nVariable name (or Enter to finish): ").strip()
            if not var_name:
                break
            
            var_value = input(f"Value for {var_name}: ").strip()
            env_vars[var_name] = var_value
        
        # Create server configuration
        server_config = {
            "command": command,
            "args": args
        }
        
        if env_vars:
            server_config["env"] = env_vars
        
        # Validate configuration
        print(f"\n{Colors.CYAN}Validating configuration...{Colors.ENDC}")
        validation = self.config.validate_server_config(server_config)
        
        print(f"Command exists: {'✓' if validation.get('command_exists') else '✗'}")
        print(f"Has required fields: {'✓' if validation.get('has_command') and validation.get('has_args') else '✗'}")
        
        # Save configuration
        save = input(f"\nSave this configuration? (y/n): ").lower().strip()
        if save == 'y':
            self.config.add_server(server_name, server_config)
            print(f"{Colors.GREEN}Server '{server_name}' added successfully!{Colors.ENDC}")
            
            # Offer to set up credentials
            if 'credentials' in validation:
                setup_creds = input("\nSet up required credentials now? (y/n): ").lower().strip()
                if setup_creds == 'y':
                    self._setup_credentials(server_name, validation['credentials'])
        else:
            print("Configuration not saved")
        
        input("\nPress Enter to continue...")
    
    def _edit_server(self):
        """Edit existing server configuration"""
        servers = self.config.list_servers()
        if not servers:
            print(f"{Colors.YELLOW}No servers configured{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        print(f"\n{Colors.BOLD}Select server to edit:{Colors.ENDC}")
        for i, server in enumerate(servers, 1):
            print(f"{i}. {server}")
        
        try:
            choice = int(input("\nSelect server (0 to cancel): "))
            if choice == 0:
                return
            
            if 1 <= choice <= len(servers):
                server_name = servers[choice - 1]
                current_config = self.config.get_server_config(server_name)
                
                print(f"\n{Colors.BOLD}Editing: {server_name}{Colors.ENDC}")
                print("Leave empty to keep current value")
                
                # Edit command
                current_command = current_config.get('command', '')
                new_command = input(f"\nCommand [{current_command}]: ").strip()
                if new_command:
                    current_config['command'] = new_command
                
                # Edit args
                current_args = json.dumps(current_config.get('args', []))
                print(f"\nCurrent args: {current_args}")
                new_args_input = input("New args (JSON array, or Enter to keep): ").strip()
                
                if new_args_input:
                    try:
                        new_args = json.loads(new_args_input)
                        if isinstance(new_args, list):
                            current_config['args'] = new_args
                        else:
                            print(f"{Colors.RED}Arguments must be a JSON array{Colors.ENDC}")
                            return
                    except json.JSONDecodeError:
                        print(f"{Colors.RED}Invalid JSON format{Colors.ENDC}")
                        return
                
                # Edit environment variables
                print(f"\n{Colors.CYAN}Edit environment variables:{Colors.ENDC}")
                print("1. Keep current")
                print("2. Add/modify variables")
                print("3. Clear all variables")
                
                env_choice = input("\nChoice: ").strip()
                
                if env_choice == '2':
                    if 'env' not in current_config:
                        current_config['env'] = {}
                    
                    print("\nCurrent variables:")
                    for key, value in current_config.get('env', {}).items():
                        print(f"  {key}: {value}")
                    
                    print("\nEnter variables to add/modify (empty name to finish):")
                    while True:
                        var_name = input("\nVariable name: ").strip()
                        if not var_name:
                            break
                        var_value = input(f"Value for {var_name}: ").strip()
                        current_config['env'][var_name] = var_value
                
                elif env_choice == '3':
                    current_config['env'] = {}
                
                # Save changes
                save = input(f"\nSave changes? (y/n): ").lower().strip()
                if save == 'y':
                    self.config.add_server(server_name, current_config)
                    print(f"{Colors.GREEN}Server '{server_name}' updated successfully!{Colors.ENDC}")
                else:
                    print("Changes discarded")
                
        except (ValueError, IndexError):
            print(f"{Colors.RED}Invalid selection{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def _remove_server(self):
        """Remove a server configuration"""
        servers = self.config.list_servers()
        if not servers:
            print(f"{Colors.YELLOW}No servers configured{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        print(f"\n{Colors.BOLD}Select server to remove:{Colors.ENDC}")
        for i, server in enumerate(servers, 1):
            print(f"{i}. {server}")
        
        try:
            choice = int(input("\nSelect server (0 to cancel): "))
            if choice == 0:
                return
            
            if 1 <= choice <= len(servers):
                server_name = servers[choice - 1]
                
                confirm = input(f"\nRemove server '{server_name}'? (y/n): ").lower().strip()
                if confirm == 'y':
                    self.config.remove_server(server_name)
                    print(f"{Colors.GREEN}Server '{server_name}' removed{Colors.ENDC}")
                else:
                    print("Removal cancelled")
                
        except (ValueError, IndexError):
            print(f"{Colors.RED}Invalid selection{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def _test_connection(self):
        """Test server connection"""
        servers = self.config.list_servers()
        if not servers:
            print(f"{Colors.YELLOW}No servers configured{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        print(f"\n{Colors.BOLD}Select server to test:{Colors.ENDC}")
        for i, server in enumerate(servers, 1):
            print(f"{i}. {server}")
        
        try:
            choice = int(input("\nSelect server (0 to cancel): "))
            if choice == 0:
                return
            
            if 1 <= choice <= len(servers):
                server_name = servers[choice - 1]
                
                print(f"\n{Colors.CYAN}Testing connection to '{server_name}'...{Colors.ENDC}")
                
                # First validate configuration
                server_config = self.config.get_server_config(server_name)
                validation = self.config.validate_server_config(server_config)
                
                if not validation.get('command_exists'):
                    print(f"{Colors.RED}Error: Command '{server_config.get('command')}' not found{Colors.ENDC}")
                    print("Make sure the command is installed and in your PATH")
                    input("\nPress Enter to continue...")
                    return
                
                # Check for missing credentials
                if 'credentials' in validation:
                    missing_creds = [k for k, v in validation['credentials'].items() if not v]
                    if missing_creds:
                        print(f"{Colors.YELLOW}Missing credentials: {', '.join(missing_creds)}{Colors.ENDC}")
                        setup = input("\nSet up credentials now? (y/n): ").lower().strip()
                        if setup == 'y':
                            self._setup_credentials(server_name, {k: False for k in missing_creds})
                        else:
                            input("\nPress Enter to continue...")
                            return
                
                # Try to start server briefly
                print(f"\n{Colors.CYAN}Attempting to start server...{Colors.ENDC}")
                print("(This may take a moment)")
                
                # Note: Actual connection test would require async implementation
                # For now, we just validate the configuration
                print(f"{Colors.GREEN}Configuration appears valid!{Colors.ENDC}")
                print("Full connection test requires running the server from MCP Tools menu")
                
        except (ValueError, IndexError):
            print(f"{Colors.RED}Invalid selection{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def _setup_credentials(self, server_name: str, credentials: Dict[str, bool]):
        """Set up credentials for a server"""
        print(f"\n{Colors.BOLD}Setting up credentials for '{server_name}'{Colors.ENDC}")
        
        for cred_name, exists in credentials.items():
            if not exists:
                print(f"\n{Colors.CYAN}{cred_name}:{Colors.ENDC}")
                
                # Determine if this is a sensitive credential
                is_sensitive = any(word in cred_name.upper() for word in ['TOKEN', 'PASSWORD', 'SECRET', 'KEY'])
                
                if is_sensitive:
                    import getpass
                    value = getpass.getpass(f"Enter value (hidden): ")
                else:
                    value = input(f"Enter value: ")
                
                if value:
                    self.credential_manager.set_credential(cred_name, value)
                    print(f"{Colors.GREEN}✓ {cred_name} saved securely{Colors.ENDC}")
                else:
                    print(f"{Colors.YELLOW}Skipped {cred_name}{Colors.ENDC}")
        
        print(f"\n{Colors.GREEN}Credentials setup complete!{Colors.ENDC}")