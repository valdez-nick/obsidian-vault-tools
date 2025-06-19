"""
Dynamic Menu Builder

Builds interactive menus from available MCP tools and resources.
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any
from ..tools.discovery import MCPTool, MCPResource, get_discovery_service
from ..tools.executor import get_executor

logger = logging.getLogger(__name__)


class DynamicMenuBuilder:
    """Builds dynamic menus from MCP server capabilities"""
    
    def __init__(self):
        self.discovery_service = get_discovery_service()
        self.executor = get_executor()
        
        # Category display names and icons
        self.category_info = {
            'repository': ('🔍 Repository & Code', 'Tools for working with repositories and code'),
            'memory': ('🧠 Memory & Context', 'Tools for managing conversation memory and context'),
            'analysis': ('📊 Analysis & Content', 'Tools for analyzing and processing content'),
            'web': ('🌐 Web & Network', 'Tools for web fetching and network operations'),
            'productivity': ('📋 Productivity', 'Tools for organization and task management'),
            'general': ('🛠️ General Tools', 'General purpose tools')
        }
    
    async def build_tools_menu(self, force_refresh: bool = False) -> List[Tuple[str, str]]:
        """Build main MCP tools menu with server categories"""
        try:
            all_tools = await self.discovery_service.discover_all_tools(force_refresh)
            
            if not all_tools:
                return [('0', 'No MCP tools available (no servers running)')]
            
            menu_options = []
            option_num = 1
            
            for server_name, tools in all_tools.items():
                if tools:  # Only show servers with tools
                    server_display = self._get_server_display_name(server_name)
                    tool_count = len(tools)
                    
                    menu_options.append((
                        str(option_num),
                        f"{server_display} ({tool_count} tools)"
                    ))
                    option_num += 1
            
            # Add utility options
            menu_options.extend([
                ('r', '🔄 Refresh tool discovery'),
                ('s', '📊 Show execution statistics'),
                ('h', '📜 Show execution history'),
                ('0', '← Back to advanced tools')
            ])
            
            return menu_options
            
        except Exception as e:
            logger.error(f"Error building tools menu: {e}")
            return [('0', f'Error loading tools: {str(e)[:50]}...')]
    
    async def build_server_tools_menu(self, server_name: str, force_refresh: bool = False) -> List[Tuple[str, str]]:
        """Build menu for tools from a specific server"""
        try:
            tools = await self.discovery_service.discover_tools(server_name, force_refresh)
            
            if not tools:
                return [('0', f'No tools available from {server_name}')]
            
            # Group tools by category
            categorized_tools = self.discovery_service.get_tools_by_category(tools)
            
            menu_options = []
            option_num = 1
            
            # Add category sections
            for category, category_tools in categorized_tools.items():
                category_display, category_desc = self.category_info.get(category, (f'📁 {category.title()}', ''))
                
                # Add category header (not selectable, just for display)
                if len(categorized_tools) > 1:  # Only show categories if there are multiple
                    menu_options.append((f"header_{category}", f"\n{category_display}"))
                
                # Add tools in this category
                for tool in category_tools:
                    display_name = self._format_tool_display(tool)
                    menu_options.append((str(option_num), display_name))
                    option_num += 1
            
            # Add utility options
            menu_options.extend([
                ('r', '🔄 Refresh tools'),
                ('s', '🔍 Search tools'),
                ('0', '← Back to tools menu')
            ])
            
            return menu_options
            
        except Exception as e:
            logger.error(f"Error building server tools menu for {server_name}: {e}")
            return [('0', f'Error loading tools: {str(e)[:50]}...')]
    
    def _get_server_display_name(self, server_name: str) -> str:
        """Get user-friendly display name for server"""
        server_icons = {
            'obsidian-pm-intelligence': '🧠 Obsidian PM Intelligence',
            'memory': '💾 Memory Server',
            'github': '🔍 GitHub Integration',
            'confluence': '📚 Confluence/Jira',
            'web-fetch': '🌐 Web Fetch',
            'sequential-thinking': '🤔 Sequential Thinking'
        }
        
        return server_icons.get(server_name, f'⚙️ {server_name.replace("-", " ").title()}')
    
    def _format_tool_display(self, tool: MCPTool) -> str:
        """Format tool for display in menu"""
        # Get appropriate icon for tool
        icon = self._get_tool_icon(tool)
        
        # Format: "Icon Tool Name - Brief description"
        description = tool.description[:50] + "..." if len(tool.description) > 50 else tool.description
        
        if description:
            return f"{icon} {tool.name} - {description}"
        else:
            return f"{icon} {tool.name}"
    
    def _get_tool_icon(self, tool: MCPTool) -> str:
        """Get appropriate icon for tool based on category and name"""
        category_icons = {
            'repository': '📁',
            'memory': '🧠',
            'analysis': '📊',
            'web': '🌐',
            'productivity': '📋',
            'general': '🔧'
        }
        
        # Specific tool name patterns
        name_lower = tool.name.lower()
        if any(word in name_lower for word in ['search', 'find', 'query']):
            return '🔍'
        elif any(word in name_lower for word in ['create', 'add', 'new']):
            return '➕'
        elif any(word in name_lower for word in ['update', 'edit', 'modify']):
            return '✏️'
        elif any(word in name_lower for word in ['delete', 'remove']):
            return '🗑️'
        elif any(word in name_lower for word in ['analyze', 'process']):
            return '⚙️'
        elif any(word in name_lower for word in ['export', 'download']):
            return '📥'
        elif any(word in name_lower for word in ['import', 'upload']):
            return '📤'
        
        return category_icons.get(tool.category, '🔧')
    
    async def get_tool_by_menu_selection(self, server_name: str, selection: str) -> Optional[MCPTool]:
        """Get tool object from menu selection"""
        try:
            if not selection.isdigit():
                return None
            
            selection_num = int(selection)
            tools = await self.discovery_service.discover_tools(server_name)
            
            if 1 <= selection_num <= len(tools):
                return tools[selection_num - 1]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting tool by selection: {e}")
            return None
    
    def build_parameter_prompts(self, tool: MCPTool) -> List[Dict[str, Any]]:
        """Build parameter input prompts for a tool"""
        prompts = []
        
        for param_name, param_info in tool.parameters.items():
            prompt = {
                'name': param_name,
                'description': param_info.get('description', ''),
                'type': param_info.get('type', 'string'),
                'required': param_info.get('required', False),
                'default': param_info.get('default'),
                'enum': param_info.get('enum'),
                'display_name': param_name.replace('_', ' ').title()
            }
            
            prompts.append(prompt)
        
        # Sort: required parameters first, then optional
        prompts.sort(key=lambda p: (not p['required'], p['name']))
        
        return prompts
    
    def format_tool_result(self, result: Any) -> str:
        """Format tool execution result for display"""
        if result is None:
            return "Tool executed successfully (no result returned)"
        
        if isinstance(result, str):
            return result
        
        if isinstance(result, dict):
            # Try to format common result structures
            if 'content' in result:
                return result['content']
            elif 'message' in result:
                return result['message']
            elif 'data' in result:
                return str(result['data'])
            else:
                # Format as JSON for complex objects
                import json
                try:
                    return json.dumps(result, indent=2)
                except:
                    return str(result)
        
        if isinstance(result, list):
            if len(result) == 0:
                return "Empty result"
            elif len(result) == 1:
                return self.format_tool_result(result[0])
            else:
                return f"List with {len(result)} items:\n" + "\n".join(
                    f"  {i+1}. {str(item)[:100]}" for i, item in enumerate(result[:5])
                )
        
        return str(result)
    
    async def search_tools(self, query: str) -> List[Tuple[str, MCPTool]]:
        """Search for tools across all servers"""
        try:
            all_tools = await self.discovery_service.discover_all_tools()
            found_tools = []
            
            for server_name, tools in all_tools.items():
                matching_tools = self.discovery_service.search_tools(query, tools)
                for tool in matching_tools:
                    found_tools.append((server_name, tool))
            
            return found_tools
            
        except Exception as e:
            logger.error(f"Error searching tools: {e}")
            return []


# Global menu builder instance
_menu_builder = None

def get_menu_builder() -> DynamicMenuBuilder:
    """Get global dynamic menu builder instance"""
    global _menu_builder
    if _menu_builder is None:
        _menu_builder = DynamicMenuBuilder()
    return _menu_builder