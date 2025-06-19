"""
MCP (Model Context Protocol) integration for Obsidian Vault Tools
"""

from .client_manager import MCPClientManager
from .config import MCPConfig
from .credentials import CredentialManager

__all__ = [
    'MCPClientManager',
    'MCPConfig', 
    'CredentialManager'
]