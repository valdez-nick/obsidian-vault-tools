"""
MCP Tool Executor

Executes MCP tools with parameter validation and error handling.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

from .discovery import MCPTool, get_discovery_service
from ..client_manager import get_client_manager

logger = logging.getLogger(__name__)


@dataclass
class MCPResult:
    """Result of an MCP tool execution"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    tool_name: str = ""
    server_name: str = ""
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MCPToolExecutor:
    """Service for executing MCP tools with validation and error handling"""
    
    def __init__(self):
        self.client_manager = get_client_manager()
        self.discovery_service = get_discovery_service()
        self._execution_history: List[MCPResult] = []
        self._max_history = 100  # Keep last 100 executions
    
    async def execute_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """Execute an MCP tool with the given arguments"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Validate server is running
            server_status = self.client_manager.get_server_status(server_name)
            if not server_status or not server_status.get('running', False):
                return MCPResult(
                    success=False,
                    error=f"Server '{server_name}' is not running",
                    tool_name=tool_name,
                    server_name=server_name
                )
            
            # Get tool metadata for validation
            tool_metadata = await self._get_tool_metadata(server_name, tool_name)
            if not tool_metadata:
                return MCPResult(
                    success=False,
                    error=f"Tool '{tool_name}' not found on server '{server_name}'",
                    tool_name=tool_name,
                    server_name=server_name
                )
            
            # Validate parameters
            validation_result = self.validate_parameters(tool_metadata, arguments)
            if not validation_result.success:
                return MCPResult(
                    success=False,
                    error=f"Parameter validation failed: {validation_result.error}",
                    tool_name=tool_name,
                    server_name=server_name
                )
            
            # Execute the tool
            result = await self.client_manager.call_tool(server_name, tool_name, arguments)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            if result.get('success', False):
                mcp_result = MCPResult(
                    success=True,
                    result=result.get('result'),
                    execution_time=execution_time,
                    tool_name=tool_name,
                    server_name=server_name
                )
            else:
                mcp_result = MCPResult(
                    success=False,
                    error=result.get('error', 'Unknown error'),
                    execution_time=execution_time,
                    tool_name=tool_name,
                    server_name=server_name
                )
            
            # Add to history
            self._add_to_history(mcp_result)
            
            logger.info(f"Executed tool '{tool_name}' on server '{server_name}' in {execution_time:.2f}s")
            return mcp_result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.error(error_msg)
            
            mcp_result = MCPResult(
                success=False,
                error=error_msg,
                execution_time=execution_time,
                tool_name=tool_name,
                server_name=server_name
            )
            
            self._add_to_history(mcp_result)
            return mcp_result
    
    async def _get_tool_metadata(self, server_name: str, tool_name: str) -> Optional[MCPTool]:
        """Get tool metadata from discovery service"""
        try:
            tools = await self.discovery_service.discover_tools(server_name)
            for tool in tools:
                if tool.name == tool_name:
                    return tool
            return None
        except Exception as e:
            logger.error(f"Error getting tool metadata: {e}")
            return None
    
    def validate_parameters(self, tool: MCPTool, arguments: Dict[str, Any]) -> MCPResult:
        """Validate tool parameters against schema"""
        try:
            # Check required parameters
            for param_name, param_info in tool.parameters.items():
                if param_info.get('required', False) and param_name not in arguments:
                    return MCPResult(
                        success=False,
                        error=f"Required parameter '{param_name}' is missing"
                    )
            
            # Validate parameter types and values
            for param_name, value in arguments.items():
                if param_name in tool.parameters:
                    param_info = tool.parameters[param_name]
                    validation_error = self._validate_parameter_value(param_name, value, param_info)
                    if validation_error:
                        return MCPResult(
                            success=False,
                            error=validation_error
                        )
            
            return MCPResult(success=True)
            
        except Exception as e:
            return MCPResult(
                success=False,
                error=f"Parameter validation error: {str(e)}"
            )
    
    def _validate_parameter_value(self, param_name: str, value: Any, param_info: Dict[str, Any]) -> Optional[str]:
        """Validate a single parameter value"""
        param_type = param_info.get('type', 'string')
        
        # Type validation
        if param_type == 'string' and not isinstance(value, str):
            return f"Parameter '{param_name}' must be a string"
        elif param_type == 'number' and not isinstance(value, (int, float)):
            return f"Parameter '{param_name}' must be a number"
        elif param_type == 'integer' and not isinstance(value, int):
            return f"Parameter '{param_name}' must be an integer"
        elif param_type == 'boolean' and not isinstance(value, bool):
            return f"Parameter '{param_name}' must be a boolean"
        elif param_type == 'array' and not isinstance(value, list):
            return f"Parameter '{param_name}' must be an array"
        elif param_type == 'object' and not isinstance(value, dict):
            return f"Parameter '{param_name}' must be an object"
        
        # Enum validation
        enum_values = param_info.get('enum')
        if enum_values and value not in enum_values:
            return f"Parameter '{param_name}' must be one of: {', '.join(map(str, enum_values))}"
        
        return None
    
    def _add_to_history(self, result: MCPResult):
        """Add execution result to history"""
        self._execution_history.append(result)
        
        # Limit history size
        if len(self._execution_history) > self._max_history:
            self._execution_history = self._execution_history[-self._max_history:]
    
    def get_execution_history(self, server_name: Optional[str] = None, tool_name: Optional[str] = None) -> List[MCPResult]:
        """Get execution history, optionally filtered by server or tool"""
        history = self._execution_history
        
        if server_name:
            history = [r for r in history if r.server_name == server_name]
        
        if tool_name:
            history = [r for r in history if r.tool_name == tool_name]
        
        return history
    
    def get_recent_tools(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recently used tools"""
        recent = []
        seen = set()
        
        for result in reversed(self._execution_history):
            tool_key = f"{result.server_name}:{result.tool_name}"
            if tool_key not in seen and result.success:
                recent.append({
                    'server': result.server_name,
                    'tool': result.tool_name,
                    'last_used': result.timestamp.isoformat() if result.timestamp else ''
                })
                seen.add(tool_key)
                
                if len(recent) >= limit:
                    break
        
        return recent
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self._execution_history:
            return {
                'total_executions': 0,
                'success_rate': 0.0,
                'average_execution_time': 0.0,
                'most_used_tools': []
            }
        
        total = len(self._execution_history)
        successful = sum(1 for r in self._execution_history if r.success)
        
        execution_times = [r.execution_time for r in self._execution_history if r.execution_time]
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
        
        # Count tool usage
        tool_counts = {}
        for result in self._execution_history:
            if result.success:
                tool_key = f"{result.server_name}:{result.tool_name}"
                tool_counts[tool_key] = tool_counts.get(tool_key, 0) + 1
        
        most_used = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_executions': total,
            'success_rate': (successful / total) * 100 if total > 0 else 0.0,
            'average_execution_time': avg_time,
            'most_used_tools': [{'tool': tool, 'count': count} for tool, count in most_used]
        }
    
    def clear_history(self):
        """Clear execution history"""
        self._execution_history.clear()


# Global executor instance
_executor = None

def get_executor() -> MCPToolExecutor:
    """Get global MCP tool executor instance"""
    global _executor
    if _executor is None:
        _executor = MCPToolExecutor()
    return _executor