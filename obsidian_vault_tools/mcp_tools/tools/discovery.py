"""
MCP Tool Discovery Service

Discovers and catalogs available tools from running MCP servers.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from ..client_manager import get_client_manager

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Represents an MCP tool with its metadata"""
    name: str
    server: str
    description: str
    parameters: Dict[str, Any]
    input_schema: Optional[Dict[str, Any]] = None
    category: str = "general"
    
    def __post_init__(self):
        """Process tool data after initialization"""
        # Infer category from tool name or description
        if not self.category or self.category == "general":
            self.category = self._infer_category()
    
    def _infer_category(self) -> str:
        """Infer tool category from name and description"""
        name_lower = self.name.lower()
        desc_lower = self.description.lower()
        
        # Repository/Code related
        if any(word in name_lower for word in ['repo', 'code', 'commit', 'pull', 'issue', 'github']):
            return "repository"
        if any(word in desc_lower for word in ['repository', 'code', 'github', 'commit']):
            return "repository"
        
        # Memory/Context related
        if any(word in name_lower for word in ['memory', 'context', 'remember', 'store']):
            return "memory"
        if any(word in desc_lower for word in ['memory', 'remember', 'context', 'conversation']):
            return "memory"
        
        # Content/Analysis related
        if any(word in name_lower for word in ['analyze', 'content', 'text', 'document']):
            return "analysis"
        if any(word in desc_lower for word in ['analyze', 'content', 'document', 'text']):
            return "analysis"
        
        # Web/Network related
        if any(word in name_lower for word in ['web', 'fetch', 'url', 'http']):
            return "web"
        if any(word in desc_lower for word in ['web', 'fetch', 'url', 'website']):
            return "web"
        
        # Productivity/Organization
        if any(word in name_lower for word in ['organize', 'task', 'project', 'manage']):
            return "productivity"
        if any(word in desc_lower for word in ['organize', 'task', 'project', 'manage']):
            return "productivity"
        
        return "general"


@dataclass  
class MCPResource:
    """Represents an MCP resource with its metadata"""
    name: str
    server: str
    description: str
    uri: str
    mime_type: Optional[str] = None
    category: str = "general"


class MCPToolDiscovery:
    """Service for discovering and managing MCP tools from running servers"""
    
    def __init__(self):
        self.client_manager = get_client_manager()
        self._tool_cache: Dict[str, List[MCPTool]] = {}
        self._resource_cache: Dict[str, List[MCPResource]] = {}
        self._last_refresh: Dict[str, float] = {}
        self._cache_ttl = 30  # 30 seconds cache TTL
        
    async def discover_all_tools(self, force_refresh: bool = False) -> Dict[str, List[MCPTool]]:
        """Discover tools from all running MCP servers"""
        all_tools = {}
        server_status = self.client_manager.get_all_server_status()
        
        for server_name, status in server_status.items():
            if status.get('running', False):
                try:
                    tools = await self.discover_tools(server_name, force_refresh)
                    if tools:
                        all_tools[server_name] = tools
                except Exception as e:
                    logger.error(f"Failed to discover tools from server '{server_name}': {e}")
        
        return all_tools
    
    async def discover_tools(self, server_name: str, force_refresh: bool = False) -> List[MCPTool]:
        """Discover tools from a specific MCP server"""
        # Check cache first
        if not force_refresh and self._is_cache_valid(server_name):
            return self._tool_cache.get(server_name, [])
        
        try:
            # Get tools from server
            result = await self.client_manager.list_tools(server_name)
            
            if not result.get('success', False):
                logger.warning(f"Failed to list tools from server '{server_name}': {result.get('error', 'Unknown error')}")
                return []
            
            tools_data = result.get('tools', [])
            tools = []
            
            for tool_data in tools_data:
                try:
                    tool = self._parse_tool_data(tool_data, server_name)
                    if tool:
                        tools.append(tool)
                except Exception as e:
                    logger.warning(f"Failed to parse tool data from '{server_name}': {e}")
                    continue
            
            # Update cache
            self._tool_cache[server_name] = tools
            self._last_refresh[server_name] = time.time()
            
            logger.info(f"Discovered {len(tools)} tools from server '{server_name}'")
            return tools
            
        except Exception as e:
            logger.error(f"Error discovering tools from server '{server_name}': {e}")
            return []
    
    async def discover_resources(self, server_name: str, force_refresh: bool = False) -> List[MCPResource]:
        """Discover resources from a specific MCP server"""
        cache_key = f"{server_name}_resources"
        
        # Check cache first
        if not force_refresh and self._is_cache_valid(cache_key):
            return self._resource_cache.get(server_name, [])
        
        try:
            # Get resources from server
            result = await self.client_manager.list_resources(server_name)
            
            if not result.get('success', False):
                logger.warning(f"Failed to list resources from server '{server_name}': {result.get('error', 'Unknown error')}")
                return []
            
            resources_data = result.get('resources', [])
            resources = []
            
            for resource_data in resources_data:
                try:
                    resource = self._parse_resource_data(resource_data, server_name)
                    if resource:
                        resources.append(resource)
                except Exception as e:
                    logger.warning(f"Failed to parse resource data from '{server_name}': {e}")
                    continue
            
            # Update cache
            self._resource_cache[server_name] = resources
            self._last_refresh[cache_key] = time.time()
            
            logger.info(f"Discovered {len(resources)} resources from server '{server_name}'")
            return resources
            
        except Exception as e:
            logger.error(f"Error discovering resources from server '{server_name}': {e}")
            return []
    
    def _parse_tool_data(self, tool_data: Dict[str, Any], server_name: str) -> Optional[MCPTool]:
        """Parse tool data from MCP server response"""
        try:
            name = tool_data.get('name', '')
            description = tool_data.get('description', '')
            
            if not name:
                logger.warning(f"Tool from server '{server_name}' missing name")
                return None
            
            # Extract input schema
            input_schema = tool_data.get('inputSchema', {})
            parameters = {}
            
            if input_schema:
                # Parse JSON schema properties
                properties = input_schema.get('properties', {})
                required = input_schema.get('required', [])
                
                for param_name, param_data in properties.items():
                    parameters[param_name] = {
                        'type': param_data.get('type', 'string'),
                        'description': param_data.get('description', ''),
                        'required': param_name in required,
                        'default': param_data.get('default'),
                        'enum': param_data.get('enum')
                    }
            
            return MCPTool(
                name=name,
                server=server_name,
                description=description,
                parameters=parameters,
                input_schema=input_schema
            )
            
        except Exception as e:
            logger.error(f"Error parsing tool data: {e}")
            return None
    
    def _parse_resource_data(self, resource_data: Dict[str, Any], server_name: str) -> Optional[MCPResource]:
        """Parse resource data from MCP server response"""
        try:
            name = resource_data.get('name', '')
            description = resource_data.get('description', '')
            uri = resource_data.get('uri', '')
            
            if not name or not uri:
                logger.warning(f"Resource from server '{server_name}' missing name or URI")
                return None
            
            return MCPResource(
                name=name,
                server=server_name,
                description=description,
                uri=uri,
                mime_type=resource_data.get('mimeType')
            )
            
        except Exception as e:
            logger.error(f"Error parsing resource data: {e}")
            return None
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is still valid for given key"""
        last_refresh = self._last_refresh.get(cache_key, 0)
        return (time.time() - last_refresh) < self._cache_ttl
    
    def get_cached_tools(self, server_name: str) -> List[MCPTool]:
        """Get cached tools for a server (may be stale)"""
        return self._tool_cache.get(server_name, [])
    
    def get_cached_resources(self, server_name: str) -> List[MCPResource]:
        """Get cached resources for a server (may be stale)"""
        return self._resource_cache.get(server_name, [])
    
    def clear_cache(self, server_name: Optional[str] = None):
        """Clear cache for specific server or all servers"""
        if server_name:
            self._tool_cache.pop(server_name, None)
            self._resource_cache.pop(server_name, None)
            self._last_refresh.pop(server_name, None)
            self._last_refresh.pop(f"{server_name}_resources", None)
        else:
            self._tool_cache.clear()
            self._resource_cache.clear()
            self._last_refresh.clear()
    
    def get_tools_by_category(self, tools: List[MCPTool]) -> Dict[str, List[MCPTool]]:
        """Group tools by category"""
        categories = {}
        for tool in tools:
            category = tool.category
            if category not in categories:
                categories[category] = []
            categories[category].append(tool)
        return categories
    
    def search_tools(self, query: str, tools: List[MCPTool]) -> List[MCPTool]:
        """Search tools by name or description"""
        query_lower = query.lower()
        matching_tools = []
        
        for tool in tools:
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower() or
                query_lower in tool.category.lower()):
                matching_tools.append(tool)
        
        return matching_tools


# Global discovery service instance
_discovery_service = None

def get_discovery_service() -> MCPToolDiscovery:
    """Get global MCP tool discovery service instance"""
    global _discovery_service
    if _discovery_service is None:
        _discovery_service = MCPToolDiscovery()
    return _discovery_service