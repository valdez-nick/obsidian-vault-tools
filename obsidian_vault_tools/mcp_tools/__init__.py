"""
MCP (Model Context Protocol) integration for Obsidian Vault Tools
"""

from .client_manager import MCPClientManager, get_client_manager
from .config import MCPConfig
from .credentials import CredentialManager, get_credential_manager
from .tools.discovery import MCPToolDiscovery, get_discovery_service
from .tools.executor import MCPToolExecutor, get_executor
from .tools.menu_builder import DynamicMenuBuilder, get_menu_builder
from .interactive_config import MCPInteractiveConfig
from .memory_client import MemoryMCPClient, get_memory_client, Entity, Relation, Observation

__all__ = [
    'MCPClientManager',
    'MCPConfig', 
    'CredentialManager',
    'MCPToolDiscovery',
    'MCPToolExecutor',
    'DynamicMenuBuilder',
    'MCPInteractiveConfig',
    'MemoryMCPClient',
    'Entity',
    'Relation', 
    'Observation',
    'get_client_manager',
    'get_credential_manager',
    'get_discovery_service',
    'get_executor',
    'get_menu_builder',
    'get_memory_client'
]