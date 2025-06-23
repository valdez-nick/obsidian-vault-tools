"""
Memory MCP Client

High-level wrapper around the MCP memory server for knowledge graph operations.
Provides typed interfaces for entities, relations, and observations.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .client_manager import get_client_manager
from .config import MCPConfig

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    name: str
    entity_type: str
    observations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MCP calls"""
        return {
            "name": self.name,
            "entityType": self.entity_type,
            "observations": self.observations
        }


@dataclass 
class Relation:
    """Represents a relation between entities"""
    from_entity: str
    to_entity: str
    relation_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MCP calls"""
        return {
            "from": self.from_entity,
            "to": self.to_entity,
            "relationType": self.relation_type
        }


@dataclass
class Observation:
    """Represents an observation about an entity"""
    entity_name: str
    observations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MCP calls"""
        return {
            "entityName": self.entity_name,
            "observations": self.observations
        }


class MemoryMCPClient:
    """High-level client for MCP memory server operations"""
    
    def __init__(self, server_name: str = "memory"):
        self.server_name = server_name
        self.client_manager = get_client_manager()
        self.config = MCPConfig()
        self._is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the memory server connection"""
        try:
            # Check if server is configured
            server_config = self.config.get_server_config(self.server_name)
            if not server_config:
                logger.error(f"Memory server '{self.server_name}' not configured")
                return False
            
            # Validate configuration
            validation = self.config.validate_memory_server_config(server_config)
            if not validation.get("command_exists", False):
                logger.error(f"Memory server command not found: {server_config.get('command')}")
                return False
            
            if not validation.get("memory_path_accessible", False):
                logger.error("Memory storage path not accessible")
                return False
            
            # Start the server
            success = await self.client_manager.start_server(self.server_name)
            if success:
                self._is_initialized = True
                logger.info(f"Memory server '{self.server_name}' initialized successfully")
            else:
                logger.error(f"Failed to start memory server '{self.server_name}'")
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing memory server: {e}")
            return False
    
    async def ensure_initialized(self) -> bool:
        """Ensure the client is initialized"""
        if not self._is_initialized:
            return await self.initialize()
        return True
    
    async def create_entities(self, entities: List[Entity]) -> Dict[str, Any]:
        """Create entities in the knowledge graph"""
        if not await self.ensure_initialized():
            return {"error": "Memory server not initialized"}
        
        try:
            entities_data = [entity.to_dict() for entity in entities]
            result = await self.client_manager.call_tool(
                self.server_name,
                "create_entities",
                {"entities": entities_data}
            )
            
            if result.get("success"):
                logger.debug(f"Created {len(entities)} entities")
            else:
                logger.error(f"Failed to create entities: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating entities: {e}")
            return {"error": str(e)}
    
    async def create_entity(self, name: str, entity_type: str, observations: List[str] = None) -> Dict[str, Any]:
        """Create a single entity"""
        if observations is None:
            observations = []
        
        entity = Entity(name=name, entity_type=entity_type, observations=observations)
        return await self.create_entities([entity])
    
    async def create_relations(self, relations: List[Relation]) -> Dict[str, Any]:
        """Create relations between entities"""
        if not await self.ensure_initialized():
            return {"error": "Memory server not initialized"}
        
        try:
            relations_data = [relation.to_dict() for relation in relations]
            result = await self.client_manager.call_tool(
                self.server_name,
                "create_relations", 
                {"relations": relations_data}
            )
            
            if result.get("success"):
                logger.debug(f"Created {len(relations)} relations")
            else:
                logger.error(f"Failed to create relations: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating relations: {e}")
            return {"error": str(e)}
    
    async def create_relation(self, from_entity: str, to_entity: str, relation_type: str) -> Dict[str, Any]:
        """Create a single relation"""
        relation = Relation(from_entity=from_entity, to_entity=to_entity, relation_type=relation_type)
        return await self.create_relations([relation])
    
    async def add_observations(self, entity_name: str, observations: List[str]) -> Dict[str, Any]:
        """Add observations to an entity"""
        if not await self.ensure_initialized():
            return {"error": "Memory server not initialized"}
        
        try:
            result = await self.client_manager.call_tool(
                self.server_name,
                "add_observations",
                {
                    "entityName": entity_name,
                    "observations": observations
                }
            )
            
            if result.get("success"):
                logger.debug(f"Added {len(observations)} observations to {entity_name}")
            else:
                logger.error(f"Failed to add observations: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error adding observations: {e}")
            return {"error": str(e)}
    
    async def read_graph(self) -> Dict[str, Any]:
        """Read the entire knowledge graph"""
        if not await self.ensure_initialized():
            return {"error": "Memory server not initialized"}
        
        try:
            result = await self.client_manager.call_tool(
                self.server_name,
                "read_graph",
                {}
            )
            
            if result.get("success"):
                logger.debug("Successfully read knowledge graph")
            else:
                logger.error(f"Failed to read graph: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error reading graph: {e}")
            return {"error": str(e)}
    
    async def search_nodes(self, query: str) -> Dict[str, Any]:
        """Search for nodes in the knowledge graph"""
        if not await self.ensure_initialized():
            return {"error": "Memory server not initialized"}
        
        try:
            result = await self.client_manager.call_tool(
                self.server_name,
                "search_nodes",
                {"query": query}
            )
            
            if result.get("success"):
                logger.debug(f"Search completed for query: {query}")
            else:
                logger.error(f"Failed to search nodes: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching nodes: {e}")
            return {"error": str(e)}
    
    async def open_nodes(self, node_names: List[str]) -> Dict[str, Any]:
        """Open and retrieve specific nodes"""
        if not await self.ensure_initialized():
            return {"error": "Memory server not initialized"}
        
        try:
            result = await self.client_manager.call_tool(
                self.server_name,
                "open_nodes",
                {"nodeNames": node_names}
            )
            
            if result.get("success"):
                logger.debug(f"Opened {len(node_names)} nodes")
            else:
                logger.error(f"Failed to open nodes: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error opening nodes: {e}")
            return {"error": str(e)}
    
    async def delete_entities(self, entity_names: List[str]) -> Dict[str, Any]:
        """Delete entities from the knowledge graph"""
        if not await self.ensure_initialized():
            return {"error": "Memory server not initialized"}
        
        try:
            result = await self.client_manager.call_tool(
                self.server_name,
                "delete_entities",
                {"entityNames": entity_names}
            )
            
            if result.get("success"):
                logger.debug(f"Deleted {len(entity_names)} entities")
            else:
                logger.error(f"Failed to delete entities: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error deleting entities: {e}")
            return {"error": str(e)}
    
    async def delete_observations(self, entity_name: str, observations: List[str]) -> Dict[str, Any]:
        """Delete specific observations from an entity"""
        if not await self.ensure_initialized():
            return {"error": "Memory server not initialized"}
        
        try:
            result = await self.client_manager.call_tool(
                self.server_name,
                "delete_observations",
                {
                    "entityName": entity_name,
                    "observations": observations
                }
            )
            
            if result.get("success"):
                logger.debug(f"Deleted {len(observations)} observations from {entity_name}")
            else:
                logger.error(f"Failed to delete observations: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error deleting observations: {e}")
            return {"error": str(e)}
    
    async def delete_relations(self, relations: List[Relation]) -> Dict[str, Any]:
        """Delete relations from the knowledge graph"""
        if not await self.ensure_initialized():
            return {"error": "Memory server not initialized"}
        
        try:
            relations_data = [relation.to_dict() for relation in relations]
            result = await self.client_manager.call_tool(
                self.server_name,
                "delete_relations",
                {"relations": relations_data}
            )
            
            if result.get("success"):
                logger.debug(f"Deleted {len(relations)} relations")
            else:
                logger.error(f"Failed to delete relations: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error deleting relations: {e}")
            return {"error": str(e)}
    
    # High-level convenience methods
    
    async def get_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific entity by name"""
        try:
            result = await self.open_nodes([entity_name])
            if result.get("success") and result.get("result"):
                nodes = result["result"].get("nodes", [])
                if nodes:
                    return nodes[0]
            return None
        except Exception as e:
            logger.error(f"Error getting entity {entity_name}: {e}")
            return None
    
    async def entity_exists(self, entity_name: str) -> bool:
        """Check if an entity exists"""
        entity = await self.get_entity(entity_name)
        return entity is not None
    
    async def get_related_entities(self, entity_name: str) -> List[Dict[str, Any]]:
        """Get entities related to the specified entity"""
        try:
            graph = await self.read_graph()
            if not graph.get("success"):
                return []
            
            graph_data = graph.get("result", {})
            relations = graph_data.get("relations", [])
            
            # Find related entities
            related = []
            for relation in relations:
                if relation.get("from") == entity_name:
                    related.append({
                        "entity": relation.get("to"),
                        "relation": relation.get("relationType"),
                        "direction": "outgoing"
                    })
                elif relation.get("to") == entity_name:
                    related.append({
                        "entity": relation.get("from"), 
                        "relation": relation.get("relationType"),
                        "direction": "incoming"
                    })
            
            return related
            
        except Exception as e:
            logger.error(f"Error getting related entities for {entity_name}: {e}")
            return []
    
    async def close(self):
        """Close the memory client connection"""
        try:
            await self.client_manager.stop_server(self.server_name)
            self._is_initialized = False
            logger.debug("Memory client connection closed")
        except Exception as e:
            logger.error(f"Error closing memory client: {e}")


# Global memory client instance
_memory_client = None

def get_memory_client(server_name: str = "memory") -> MemoryMCPClient:
    """Get global memory client instance"""
    global _memory_client
    if _memory_client is None or _memory_client.server_name != server_name:
        _memory_client = MemoryMCPClient(server_name)
    return _memory_client