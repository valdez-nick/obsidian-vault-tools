"""
MCP client manager for connecting to and managing MCP servers
"""

import asyncio
import subprocess
import json
import time
import signal
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from contextlib import AsyncExitStack
import logging

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from .config import MCPConfig
from .credentials import get_credential_manager


logger = logging.getLogger(__name__)


class MCPServerProcess:
    """Represents a running MCP server process"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.session: Optional[Any] = None  # ClientSession when MCP is available
        self.started_at: Optional[float] = None
        self.last_error: Optional[str] = None
    
    def is_running(self) -> bool:
        """Check if server process is running"""
        return self.process is not None and self.process.poll() is None
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status information"""
        return {
            "name": self.name,
            "running": self.is_running(),
            "pid": self.process.pid if self.process else None,
            "started_at": self.started_at,
            "uptime": time.time() - self.started_at if self.started_at else None,
            "last_error": self.last_error
        }


class MCPClientManager:
    """Manages MCP server connections and lifecycle"""
    
    def __init__(self):
        self.config = MCPConfig()
        self.credential_manager = get_credential_manager()
        self.servers: Dict[str, MCPServerProcess] = {}
        self.exit_stack: Optional[AsyncExitStack] = None
        
        # Check MCP availability
        if not MCP_AVAILABLE:
            logger.warning("MCP library not available. Install with: pip install mcp")
    
    async def start_server(self, name: str) -> bool:
        """Start an MCP server"""
        if not MCP_AVAILABLE:
            logger.error("MCP library not available")
            return False
        
        server_config = self.config.get_server_config(name)
        if not server_config:
            logger.error(f"Server '{name}' not found in configuration")
            return False
        
        # Check if already running
        if name in self.servers and self.servers[name].is_running():
            logger.info(f"Server '{name}' is already running")
            return True
        
        try:
            # Substitute credentials in config
            resolved_config = self.credential_manager.substitute_placeholders(server_config)
            
            # Validate configuration
            validation = self.config.validate_server_config(resolved_config)
            if not validation.get("command_exists", False):
                logger.error(f"Command not found: {resolved_config['command']}")
                return False
            
            # Create server process
            server_process = MCPServerProcess(name, resolved_config)
            
            # Handle different transport types
            if self._is_stdio_server(resolved_config):
                success = await self._start_stdio_server(server_process)
            else:
                # For HTTP/SSE servers, just validate the config for now
                logger.info(f"HTTP/SSE server '{name}' configuration validated")
                success = True
            
            if success:
                self.servers[name] = server_process
                logger.info(f"Started MCP server: {name}")
                return True
            else:
                logger.error(f"Failed to start MCP server: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting server '{name}': {e}")
            return False
    
    def _is_stdio_server(self, config: Dict[str, Any]) -> bool:
        """Check if server uses stdio transport"""
        command = config.get("command", "")
        # Stdio servers typically use local commands like npx, node, python, etc.
        stdio_commands = ["npx", "node", "python", "python3", "/usr/bin/node", "/opt/homebrew/bin/node"]
        return any(command.endswith(cmd) or command == cmd for cmd in stdio_commands)
    
    async def _start_stdio_server(self, server_process: MCPServerProcess) -> bool:
        """Start a stdio-based MCP server"""
        try:
            config = server_process.config
            command = config["command"]
            args = config.get("args", [])
            env = config.get("env", {})
            
            # Prepare environment
            server_env = os.environ.copy()
            server_env.update(env)
            
            # Create server parameters
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=server_env
            )
            
            # Initialize exit stack if needed
            if self.exit_stack is None:
                self.exit_stack = AsyncExitStack()
            
            # Start the server
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            
            # Create session
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio_transport[0], stdio_transport[1])
            )
            
            # Initialize session
            await session.initialize()
            
            server_process.session = session
            server_process.started_at = time.time()
            
            logger.info(f"Stdio server '{server_process.name}' started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stdio server '{server_process.name}': {e}")
            server_process.last_error = str(e)
            return False
    
    async def stop_server(self, name: str) -> bool:
        """Stop an MCP server"""
        if name not in self.servers:
            logger.warning(f"Server '{name}' not found")
            return False
        
        server_process = self.servers[name]
        
        try:
            # Close session if it exists
            if server_process.session:
                # MCP sessions are managed by the exit stack
                pass
            
            # Terminate process if it exists
            if server_process.process and server_process.is_running():
                server_process.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    server_process.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    server_process.process.kill()
                    server_process.process.wait()
            
            # Remove from servers dict
            del self.servers[name]
            logger.info(f"Stopped MCP server: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping server '{name}': {e}")
            return False
    
    async def restart_server(self, name: str) -> bool:
        """Restart an MCP server"""
        if name in self.servers:
            await self.stop_server(name)
        return await self.start_server(name)
    
    async def stop_all_servers(self):
        """Stop all running servers"""
        for name in list(self.servers.keys()):
            await self.stop_server(name)
        
        # Close exit stack
        if self.exit_stack:
            await self.exit_stack.aclose()
            self.exit_stack = None
    
    def get_server_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific server"""
        if name not in self.servers:
            return None
        return self.servers[name].get_status()
    
    def get_all_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all servers"""
        status = {}
        
        # Get configured servers
        configured_servers = self.config.list_servers()
        
        for name in configured_servers:
            if name in self.servers:
                status[name] = self.servers[name].get_status()
            else:
                status[name] = {
                    "name": name,
                    "running": False,
                    "configured": True
                }
        
        return status
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on an MCP server"""
        if not MCP_AVAILABLE:
            return {"error": "MCP library not available"}
        
        if server_name not in self.servers:
            return {"error": f"Server '{server_name}' not found"}
        
        server_process = self.servers[server_name]
        if not server_process.session:
            return {"error": f"Server '{server_name}' not connected"}
        
        try:
            # Call the tool
            result = await server_process.session.call_tool(tool_name, arguments)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}' on server '{server_name}': {e}")
            return {"error": str(e)}
    
    async def list_tools(self, server_name: str) -> Dict[str, Any]:
        """List available tools from an MCP server"""
        if not MCP_AVAILABLE:
            return {"error": "MCP library not available"}
        
        if server_name not in self.servers:
            return {"error": f"Server '{server_name}' not found"}
        
        server_process = self.servers[server_name]
        if not server_process.session:
            return {"error": f"Server '{server_name}' not connected"}
        
        try:
            # List tools
            tools = await server_process.session.list_tools()
            return {"success": True, "tools": tools}
        except Exception as e:
            logger.error(f"Error listing tools from server '{server_name}': {e}")
            return {"error": str(e)}
    
    async def list_resources(self, server_name: str) -> Dict[str, Any]:
        """List available resources from an MCP server"""
        if not MCP_AVAILABLE:
            return {"error": "MCP library not available"}
        
        if server_name not in self.servers:
            return {"error": f"Server '{server_name}' not found"}
        
        server_process = self.servers[server_name]
        if not server_process.session:
            return {"error": f"Server '{server_name}' not connected"}
        
        try:
            # List resources
            resources = await server_process.session.list_resources()
            return {"success": True, "resources": resources}
        except Exception as e:
            logger.error(f"Error listing resources from server '{server_name}': {e}")
            return {"error": str(e)}


# Global client manager instance
_client_manager = None

def get_client_manager() -> MCPClientManager:
    """Get global MCP client manager instance"""
    global _client_manager
    if _client_manager is None:
        _client_manager = MCPClientManager()
    return _client_manager