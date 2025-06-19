"""
MCP configuration management
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from .credentials import get_credential_manager


class MCPConfig:
    """MCP server configuration management"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or (Path.home() / '.obsidian_vault_tools')
        self.config_dir.mkdir(exist_ok=True)
        
        self.config_file = self.config_dir / 'mcp_config.json'
        self.template_file = self.config_dir / 'mcp_config_template.json'
        
        self.credential_manager = get_credential_manager()
        
        # Initialize with default template if no config exists
        if not self.config_file.exists() and not self.template_file.exists():
            self._create_default_template()
    
    def _create_default_template(self):
        """Create default MCP configuration template"""
        default_config = {
            "mcpServers": {
                "memory": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-memory"],
                    "env": {
                        "MEMORY_FILE_PATH": "[MEMORY_PATH]/memory.json"
                    }
                },
                "web-fetch": {
                    "command": "npx", 
                    "args": ["-y", "@modelcontextprotocol/server-fetch"]
                },
                "github": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {
                        "GITHUB_PERSONAL_ACCESS_TOKEN": "[GITHUB_PERSONAL_ACCESS_TOKEN]"
                    }
                }
            }
        }
        
        with open(self.template_file, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    def load_config(self) -> Dict[str, Any]:
        """Load MCP configuration"""
        # Try local config first, then template
        config_file = self.config_file if self.config_file.exists() else self.template_file
        
        if not config_file.exists():
            return {"mcpServers": {}}
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Substitute credential placeholders
            return self.credential_manager.substitute_placeholders(config)
            
        except Exception as e:
            print(f"Error loading MCP config: {e}")
            return {"mcpServers": {}}
    
    def save_config(self, config: Dict[str, Any]):
        """Save MCP configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving MCP config: {e}")
    
    def add_server(self, name: str, server_config: Dict[str, Any]):
        """Add a new MCP server configuration"""
        config = self.load_config()
        if "mcpServers" not in config:
            config["mcpServers"] = {}
        
        config["mcpServers"][name] = server_config
        self.save_config(config)
    
    def remove_server(self, name: str):
        """Remove an MCP server configuration"""
        config = self.load_config()
        if "mcpServers" in config and name in config["mcpServers"]:
            del config["mcpServers"][name]
            self.save_config(config)
    
    def get_server_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific server"""
        config = self.load_config()
        return config.get("mcpServers", {}).get(name)
    
    def list_servers(self) -> List[str]:
        """List configured MCP servers"""
        config = self.load_config()
        return list(config.get("mcpServers", {}).keys())
    
    def validate_server_config(self, server_config: Dict[str, Any]) -> Dict[str, bool]:
        """Validate a server configuration"""
        results = {
            "has_command": "command" in server_config,
            "has_args": "args" in server_config,
            "command_exists": False
        }
        
        if results["has_command"]:
            command = server_config["command"]
            # Check if command exists in PATH
            try:
                import shutil
                results["command_exists"] = shutil.which(command) is not None
            except:
                results["command_exists"] = False
        
        # Check for required credentials
        env_vars = server_config.get("env", {})
        required_creds = []
        for key, value in env_vars.items():
            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                required_creds.append(value[1:-1])
        
        if required_creds:
            cred_validation = self.credential_manager.validate_credentials(required_creds)
            results["credentials"] = cred_validation
        
        return results
    
    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all configured servers"""
        config = self.load_config()
        status = {}
        
        for name, server_config in config.get("mcpServers", {}).items():
            validation = self.validate_server_config(server_config)
            status[name] = {
                "config": server_config,
                "validation": validation,
                "ready": all([
                    validation["has_command"],
                    validation["has_args"], 
                    validation["command_exists"],
                    validation.get("credentials", {}).values() if "credentials" in validation else [True]
                ])
            }
        
        return status
    
    def create_server_from_template(self, name: str, template_name: str, **kwargs) -> bool:
        """Create server configuration from template"""
        templates = {
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": "[GITHUB_PERSONAL_ACCESS_TOKEN]"
                }
            },
            "memory": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-memory"],
                "env": {
                    "MEMORY_FILE_PATH": f"[MEMORY_PATH]/{name}_memory.json"
                }
            },
            "confluence": {
                "command": "docker",
                "args": [
                    "run", "-i", "--rm",
                    "--platform", "linux/amd64",
                    "ghcr.io/sooperset/mcp-atlassian:latest",
                    "--confluence-url", "https://api.atlassian.com/ex/confluence/[CONFLUENCE_CLOUD_ID]",
                    "--confluence-username", "[CONFLUENCE_EMAIL]",
                    "--confluence-token", "[CONFLUENCE_TOKEN]"
                ]
            },
            "obsidian-pm": {
                "command": "/opt/homebrew/bin/node",
                "args": [kwargs.get("script_path", "/path/to/obsidian-pm-intelligence.js")],
                "env": {
                    "VAULT_PATH": "[VAULT_PATH]",
                    "ENABLE_LEARNING": "true",
                    "MEMORY_PATH": "[MEMORY_PATH]",
                    "ENABLE_AGENT_ORCHESTRATION": "false",
                    "LOG_LEVEL": "info"
                }
            }
        }
        
        if template_name not in templates:
            return False
        
        template = templates[template_name].copy()
        
        # Apply any custom parameters
        for key, value in kwargs.items():
            if key in template:
                template[key] = value
            elif "env" in template and key.upper() in template["env"]:
                template["env"][key.upper()] = value
        
        self.add_server(name, template)
        return True