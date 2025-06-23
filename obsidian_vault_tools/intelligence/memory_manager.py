"""
Memory Manager

High-level memory operations for the Obsidian Vault Tools suite.
Orchestrates memory functionality across different tool categories.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
import json

from ..mcp_tools.memory_client import get_memory_client, Entity, Relation
from .memory_models import (
    EntityType, RelationType, MemoryEntityFactory, MemoryRelationFactory,
    ObservationTemplates, MemoryQueryBuilder
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """High-level memory manager for the Obsidian Vault Tools suite"""
    
    def __init__(self, user_id: str = "default_user", vault_path: Optional[str] = None):
        self.user_id = user_id
        self.vault_path = vault_path
        self.memory_client = get_memory_client()
        self.is_initialized = False
        
        # Cache for frequently accessed data
        self._user_preferences_cache = {}
        self._vault_entities_cache = {}
        self._last_cache_update = None
        
    async def initialize(self) -> bool:
        """Initialize the memory manager"""
        try:
            # Initialize memory client
            self.is_initialized = await self.memory_client.initialize()
            
            if self.is_initialized:
                # Ensure basic entities exist
                await self._ensure_core_entities()
                
                # Load cache
                await self._update_cache()
                
                logger.info("Memory manager initialized successfully")
            else:
                logger.warning("Memory manager initialization failed")
            
            return self.is_initialized
            
        except Exception as e:
            logger.error(f"Error initializing memory manager: {e}")
            return False
    
    async def _ensure_core_entities(self):
        """Ensure core entities exist in memory"""
        try:
            # User entity
            if not await self.memory_client.entity_exists(self.user_id):
                user_entity = MemoryEntityFactory.create_user_entity(self.user_id)
                await self.memory_client.create_entities([Entity(**user_entity)])
            
            # Vault entity (if vault path provided)
            if self.vault_path:
                vault_name = Path(self.vault_path).name
                vault_entity_name = f"vault_{vault_name.replace(' ', '_')}"
                
                if not await self.memory_client.entity_exists(vault_entity_name):
                    vault_entity = MemoryEntityFactory.create_vault_entity(self.vault_path, vault_name)
                    await self.memory_client.create_entities([Entity(**vault_entity)])
                    
                    # Create user owns vault relation
                    relation = MemoryRelationFactory.create_user_owns_vault(self.user_id, vault_name)
                    await self.memory_client.create_relations([Relation(**relation)])
            
        except Exception as e:
            logger.error(f"Error ensuring core entities: {e}")
    
    async def _update_cache(self):
        """Update internal cache from memory"""
        try:
            # Update user preferences cache
            user_patterns = await self.get_user_patterns()
            self._user_preferences_cache = user_patterns
            
            # Update vault entities cache
            if self.vault_path:
                vault_entities = await self.get_vault_entities()
                self._vault_entities_cache = vault_entities
            
            self._last_cache_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating cache: {e}")
    
    # User-related memory operations
    
    async def record_user_preference(self, preference_name: str, preference_value: str, strength: str = "moderate") -> bool:
        """Record a user preference with strength indicator"""
        if not self.is_initialized:
            return False
        
        try:
            # Create preference entity
            pref_entity = MemoryEntityFactory.create_preference_entity(preference_name, preference_value)
            await self.memory_client.create_entities([Entity(**pref_entity)])
            
            # Add strength observation
            strength_obs = ObservationTemplates.user_preference_strength(preference_name, strength)
            pref_entity_name = f"pref_{preference_name.replace(' ', '_')}"
            await self.memory_client.add_observations(pref_entity_name, [strength_obs])
            
            # Create user prefers relation
            relation = MemoryRelationFactory.create_user_prefers(self.user_id, preference_name)
            await self.memory_client.create_relations([Relation(**relation)])
            
            # Update cache
            self._user_preferences_cache[preference_name] = {
                "value": preference_value,
                "strength": strength,
                "updated": datetime.now().isoformat()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording user preference: {e}")
            return False
    
    async def get_user_preferences(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Get user preferences, optionally filtered by category"""
        if not self.is_initialized:
            return {}
        
        try:
            # Use cache if recent
            if (self._last_cache_update and 
                datetime.now() - self._last_cache_update < timedelta(minutes=5)):
                preferences = self._user_preferences_cache
            else:
                # Refresh from memory
                user_query = MemoryQueryBuilder.build_user_profile_query(self.user_id)
                search_result = await self.memory_client.search_nodes(user_query)
                
                preferences = {}
                if search_result.get("success"):
                    # Parse preferences from search results
                    # This would need to be implemented based on actual memory server response format
                    preferences = search_result.get("result", {})
                
                self._user_preferences_cache = preferences
                self._last_cache_update = datetime.now()
            
            # Filter by category if specified
            if category:
                return {k: v for k, v in preferences.items() if category.lower() in k.lower()}
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}
    
    async def get_user_patterns(self) -> Dict[str, Any]:
        """Get comprehensive user behavior patterns"""
        if not self.is_initialized:
            return {}
        
        try:
            # Get related entities for user
            related_entities = await self.memory_client.get_related_entities(self.user_id)
            
            patterns = {
                "frequent_actions": [],
                "preferred_tools": [],
                "research_interests": [],
                "time_patterns": {},
                "productivity_metrics": {}
            }
            
            # Process related entities to extract patterns
            for entity_info in related_entities:
                entity_name = entity_info.get("entity", "")
                relation_type = entity_info.get("relation", "")
                
                if entity_name.startswith("action_") and relation_type == RelationType.PERFORMS.value:
                    action_name = entity_name.replace("action_", "").replace("_", " ")
                    patterns["frequent_actions"].append(action_name)
                
                elif entity_name.startswith("tool_") and relation_type == RelationType.USES.value:
                    tool_name = entity_name.replace("tool_", "").replace("_", " ")
                    patterns["preferred_tools"].append(tool_name)
                
                elif entity_name.startswith("topic_") and relation_type == RelationType.RESEARCHES.value:
                    topic_name = entity_name.replace("topic_", "").replace("_", " ")
                    patterns["research_interests"].append(topic_name)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting user patterns: {e}")
            return {}
    
    # Vault-related memory operations
    
    async def track_vault_structure(self, structure_data: Dict[str, Any]) -> bool:
        """Track vault structure changes in memory"""
        if not self.is_initialized or not self.vault_path:
            return False
        
        try:
            vault_name = Path(self.vault_path).name
            vault_entity_name = f"vault_{vault_name.replace(' ', '_')}"
            
            # Create observations for vault structure
            structure_observations = [
                f"Total folders: {structure_data.get('folder_count', 0)}",
                f"Total notes: {structure_data.get('note_count', 0)}",
                f"Total tags: {structure_data.get('tag_count', 0)}",
                f"Average note length: {structure_data.get('avg_note_length', 0)} words",
                f"Structure updated: {datetime.now().isoformat()}"
            ]
            
            await self.memory_client.add_observations(vault_entity_name, structure_observations)
            
            # Track folder entities
            for folder_path in structure_data.get('folders', []):
                folder_name = Path(folder_path).name
                folder_entity = {
                    "name": f"folder_{folder_name.replace(' ', '_')}",
                    "entityType": EntityType.FOLDER.value,
                    "observations": [f"Folder path: {folder_path}"]
                }
                
                await self.memory_client.create_entities([Entity(**folder_entity)])
                
                # Create vault contains folder relation
                relation = {
                    "from": vault_entity_name,
                    "to": f"folder_{folder_name.replace(' ', '_')}",
                    "relationType": RelationType.CONTAINS.value
                }
                await self.memory_client.create_relations([Relation(**relation)])
            
            return True
            
        except Exception as e:
            logger.error(f"Error tracking vault structure: {e}")
            return False
    
    async def track_note_interaction(self, note_path: str, interaction_type: str, metadata: Dict[str, Any] = None) -> bool:
        """Track user interaction with a specific note"""
        if not self.is_initialized:
            return False
        
        try:
            note_name = Path(note_path).stem
            note_entity_name = f"note_{note_name.replace(' ', '_')}"
            
            # Create note entity if it doesn't exist
            if not await self.memory_client.entity_exists(note_entity_name):
                note_entity = MemoryEntityFactory.create_note_entity(note_path, note_name)
                await self.memory_client.create_entities([Entity(**note_entity)])
                
                # Create vault contains note relation if vault is tracked
                if self.vault_path:
                    vault_name = Path(self.vault_path).name
                    vault_entity_name = f"vault_{vault_name.replace(' ', '_')}"
                    
                    relation = MemoryRelationFactory.create_vault_contains_note(vault_name, note_name)
                    await self.memory_client.create_relations([Relation(**relation)])
            
            # Add interaction observation
            interaction_obs = f"User {interaction_type} note on {datetime.now().isoformat()}"
            await self.memory_client.add_observations(note_entity_name, [interaction_obs])
            
            # Add metadata observations if provided
            if metadata:
                metadata_obs = []
                for key, value in metadata.items():
                    metadata_obs.append(f"{key}: {value}")
                
                await self.memory_client.add_observations(note_entity_name, metadata_obs)
            
            return True
            
        except Exception as e:
            logger.error(f"Error tracking note interaction: {e}")
            return False
    
    async def get_vault_entities(self) -> Dict[str, List[str]]:
        """Get entities related to the vault"""
        if not self.is_initialized or not self.vault_path:
            return {}
        
        try:
            vault_name = Path(self.vault_path).name
            vault_entity_name = f"vault_{vault_name.replace(' ', '_')}"
            
            related_entities = await self.memory_client.get_related_entities(vault_entity_name)
            
            entities = {
                "notes": [],
                "folders": [],
                "tags": []
            }
            
            for entity_info in related_entities:
                entity_name = entity_info.get("entity", "")
                
                if entity_name.startswith("note_"):
                    entities["notes"].append(entity_name.replace("note_", "").replace("_", " "))
                elif entity_name.startswith("folder_"):
                    entities["folders"].append(entity_name.replace("folder_", "").replace("_", " "))
                elif entity_name.startswith("tag_"):
                    entities["tags"].append(entity_name.replace("tag_", "").replace("_", " "))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error getting vault entities: {e}")
            return {}
    
    # Tool usage tracking
    
    async def track_tool_usage(self, tool_name: str, category: str, success: bool, duration: float = None, metadata: Dict[str, Any] = None) -> bool:
        """Track tool usage with success metrics"""
        if not self.is_initialized:
            return False
        
        try:
            # Create tool entity if it doesn't exist
            tool_entity_name = f"tool_{tool_name.replace(' ', '_')}"
            if not await self.memory_client.entity_exists(tool_entity_name):
                tool_entity = MemoryEntityFactory.create_tool_entity(tool_name, category)
                await self.memory_client.create_entities([Entity(**tool_entity)])
            
            # Create usage observations
            usage_observations = [
                f"Used on {datetime.now().isoformat()}",
                f"Success: {success}",
                f"Category: {category}"
            ]
            
            if duration is not None:
                usage_observations.append(f"Duration: {duration:.2f} seconds")
            
            if metadata:
                for key, value in metadata.items():
                    usage_observations.append(f"{key}: {value}")
            
            await self.memory_client.add_observations(tool_entity_name, usage_observations)
            
            # Create user uses tool relation
            relation = {
                "from": self.user_id,
                "to": tool_entity_name,
                "relationType": RelationType.USES.value
            }
            await self.memory_client.create_relations([Relation(**relation)])
            
            return True
            
        except Exception as e:
            logger.error(f"Error tracking tool usage: {e}")
            return False
    
    async def get_tool_recommendations(self, context: str = None) -> List[Dict[str, Any]]:
        """Get tool recommendations based on user patterns and context"""
        if not self.is_initialized:
            return []
        
        try:
            # Get user's tool usage patterns
            related_entities = await self.memory_client.get_related_entities(self.user_id)
            
            tool_scores = {}
            
            for entity_info in related_entities:
                entity_name = entity_info.get("entity", "")
                
                if entity_name.startswith("tool_"):
                    tool_name = entity_name.replace("tool_", "").replace("_", " ")
                    
                    # Get tool details
                    tool_entity = await self.memory_client.get_entity(entity_name)
                    if tool_entity:
                        observations = tool_entity.get("observations", [])
                        
                        # Calculate score based on usage frequency and success rate
                        usage_count = len([obs for obs in observations if "Used on" in obs])
                        success_count = len([obs for obs in observations if "Success: True" in obs])
                        
                        success_rate = success_count / max(usage_count, 1)
                        score = usage_count * success_rate
                        
                        tool_scores[tool_name] = {
                            "score": score,
                            "usage_count": usage_count,
                            "success_rate": success_rate,
                            "entity_name": entity_name
                        }
            
            # Sort by score and return recommendations
            recommendations = []
            for tool_name, info in sorted(tool_scores.items(), key=lambda x: x[1]["score"], reverse=True):
                recommendations.append({
                    "tool": tool_name,
                    "confidence": min(info["score"] / 10, 1.0),  # Normalize to 0-1
                    "usage_count": info["usage_count"],
                    "success_rate": info["success_rate"],
                    "reason": f"Used {info['usage_count']} times with {info['success_rate']:.1%} success rate"
                })
            
            return recommendations[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error getting tool recommendations: {e}")
            return []
    
    # Research tracking
    
    async def track_research_topic(self, topic: str, domain: str = None, confidence: float = 1.0) -> bool:
        """Track user research interests"""
        if not self.is_initialized:
            return False
        
        try:
            # Create topic entity
            topic_entity = MemoryEntityFactory.create_topic_entity(topic, domain)
            await self.memory_client.create_entities([Entity(**topic_entity)])
            
            # Add confidence observation
            topic_entity_name = f"topic_{topic.replace(' ', '_')}"
            confidence_obs = f"Interest confidence: {confidence:.2f}"
            await self.memory_client.add_observations(topic_entity_name, [confidence_obs])
            
            # Create user researches topic relation
            relation = MemoryRelationFactory.create_user_researches_topic(self.user_id, topic)
            await self.memory_client.create_relations([Relation(**relation)])
            
            return True
            
        except Exception as e:
            logger.error(f"Error tracking research topic: {e}")
            return False
    
    async def get_research_interests(self) -> List[Dict[str, Any]]:
        """Get user's research interests with confidence scores"""
        if not self.is_initialized:
            return []
        
        try:
            query = MemoryQueryBuilder.build_research_interests_query(self.user_id)
            search_result = await self.memory_client.search_nodes(query)
            
            interests = []
            if search_result.get("success"):
                # Parse research interests from search results
                # This would need to be implemented based on actual memory server response format
                result_data = search_result.get("result", {})
                
                # For now, get related entities
                related_entities = await self.memory_client.get_related_entities(self.user_id)
                
                for entity_info in related_entities:
                    entity_name = entity_info.get("entity", "")
                    relation_type = entity_info.get("relation", "")
                    
                    if entity_name.startswith("topic_") and relation_type == RelationType.RESEARCHES.value:
                        topic_name = entity_name.replace("topic_", "").replace("_", " ")
                        
                        # Get topic entity details
                        topic_entity = await self.memory_client.get_entity(entity_name)
                        if topic_entity:
                            observations = topic_entity.get("observations", [])
                            
                            # Extract confidence if available
                            confidence = 1.0
                            domain = None
                            
                            for obs in observations:
                                if "Interest confidence:" in obs:
                                    try:
                                        confidence = float(obs.split(":")[1].strip())
                                    except:
                                        pass
                                elif "Domain:" in obs:
                                    domain = obs.split(":")[1].strip()
                            
                            interests.append({
                                "topic": topic_name,
                                "confidence": confidence,
                                "domain": domain,
                                "entity_name": entity_name
                            })
            
            # Sort by confidence
            return sorted(interests, key=lambda x: x["confidence"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting research interests: {e}")
            return []
    
    # Utility methods
    
    async def export_memory_data(self, output_path: str = None) -> Dict[str, Any]:
        """Export all memory data for backup or analysis"""
        if not self.is_initialized:
            return {}
        
        try:
            # Read entire knowledge graph
            graph_result = await self.memory_client.read_graph()
            
            if graph_result.get("success"):
                graph_data = graph_result.get("result", {})
                
                # Add metadata
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "user_id": self.user_id,
                    "vault_path": self.vault_path,
                    "graph_data": graph_data
                }
                
                # Save to file if path provided
                if output_path:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2)
                    
                    logger.info(f"Memory data exported to {output_path}")
                
                return export_data
            
            return {}
            
        except Exception as e:
            logger.error(f"Error exporting memory data: {e}")
            return {}
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        if not self.is_initialized:
            return {}
        
        try:
            # Read graph to get statistics
            graph_result = await self.memory_client.read_graph()
            
            if graph_result.get("success"):
                graph_data = graph_result.get("result", {})
                
                entities = graph_data.get("entities", [])
                relations = graph_data.get("relations", [])
                
                # Count entities by type
                entity_counts = {}
                for entity in entities:
                    entity_type = entity.get("entityType", "unknown")
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                
                # Count relations by type
                relation_counts = {}
                for relation in relations:
                    relation_type = relation.get("relationType", "unknown")
                    relation_counts[relation_type] = relation_counts.get(relation_type, 0) + 1
                
                return {
                    "total_entities": len(entities),
                    "total_relations": len(relations),
                    "entity_types": entity_counts,
                    "relation_types": relation_counts,
                    "last_updated": self._last_cache_update.isoformat() if self._last_cache_update else None
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}
    
    async def close(self):
        """Close memory manager and cleanup resources"""
        try:
            await self.memory_client.close()
            self.is_initialized = False
            logger.debug("Memory manager closed")
        except Exception as e:
            logger.error(f"Error closing memory manager: {e}")


# Global memory manager instance
_memory_manager = None

def get_memory_manager(user_id: str = "default_user", vault_path: Optional[str] = None) -> MemoryManager:
    """Get global memory manager instance"""
    global _memory_manager
    if _memory_manager is None or _memory_manager.user_id != user_id:
        _memory_manager = MemoryManager(user_id, vault_path)
    return _memory_manager