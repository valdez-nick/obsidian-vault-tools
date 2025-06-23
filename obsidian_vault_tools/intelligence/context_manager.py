"""
Context Manager

Tracks user context, vault state, and interaction history.
Enhanced with memory capabilities for persistent learning.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import deque, defaultdict
import json
import os
import logging
import asyncio

from ..mcp_tools.memory_client import get_memory_client, Entity, Relation
from .memory_models import (
    EntityType, RelationType, MemoryEntityFactory, MemoryRelationFactory,
    ObservationTemplates, MemoryQueryBuilder
)

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages context for intelligent decision making"""
    
    def __init__(self, vault_path: Optional[str] = None, user_id: str = "default_user"):
        self.vault_path = vault_path
        self.user_id = user_id
        self.current_context = {
            'current_menu': None,
            'previous_menu': None,
            'last_action': None,
            'last_action_time': None,
            'session_start': datetime.now().isoformat(),
        }
        
        # History tracking
        self.action_history = deque(maxlen=50)  # Last 50 actions
        self.menu_history = deque(maxlen=20)    # Last 20 menu navigations
        self.query_history = deque(maxlen=100)  # Last 100 queries
        
        # Usage patterns
        self.action_frequency = defaultdict(int)
        self.time_patterns = defaultdict(list)  # Action -> [timestamps]
        
        # Vault state cache
        self.vault_state = {}
        self._last_vault_scan = None
        
        # Memory integration
        self.memory_client = get_memory_client()
        self.memory_enabled = False
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load persisted context if available
        self._load_context()
        
        # Initialize memory asynchronously
        try:
            asyncio.create_task(self._initialize_memory())
        except RuntimeError:
            # No event loop running, will initialize later
            pass
    
    def update_menu_context(self, menu_name: str):
        """Update current menu context"""
        self.current_context['previous_menu'] = self.current_context.get('current_menu')
        self.current_context['current_menu'] = menu_name
        self.current_context['menu_entered_time'] = datetime.now().isoformat()
        
        self.menu_history.append({
            'menu': menu_name,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.debug(f"Context updated: Entered {menu_name} menu")
    
    def record_action(self, action: str, result: Any = None):
        """Record an action taken by the user"""
        timestamp = datetime.now()
        
        action_record = {
            'action': action,
            'timestamp': timestamp.isoformat(),
            'menu': self.current_context.get('current_menu'),
            'success': result.success if hasattr(result, 'success') else True,
        }
        
        self.action_history.append(action_record)
        self.action_frequency[action] += 1
        self.time_patterns[action].append(timestamp)
        
        self.current_context['last_action'] = action
        self.current_context['last_action_time'] = timestamp.isoformat()
        
        # Record to memory system asynchronously
        if self.memory_enabled:
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self.record_action_to_memory(action, result))
            except RuntimeError:
                # No event loop available
                pass
        
        # Persist context periodically
        if len(self.action_history) % 10 == 0:
            self._save_context()
    
    def record_query(self, query: str, intent: Optional[str] = None):
        """Record a user query"""
        self.query_history.append({
            'query': query,
            'intent': intent,
            'timestamp': datetime.now().isoformat(),
            'menu': self.current_context.get('current_menu')
        })
    
    def get_context_hints(self) -> Dict[str, Any]:
        """Get context hints for intent detection"""
        hints = {
            'current_menu': self.current_context.get('current_menu'),
            'recent_actions': list(self.action_history)[-5:] if self.action_history else [],
            'frequent_actions': self.get_frequent_actions(5),
            'time_of_day': self._get_time_context(),
            'vault_state': self.get_vault_state(),
        }
        
        return hints
    
    def get_frequent_actions(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most frequent actions"""
        sorted_actions = sorted(
            self.action_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_actions[:limit]
    
    def get_suggested_actions(self) -> List[str]:
        """Get suggested actions based on context and memory"""
        suggestions = []
        
        # Get memory-based suggestions first (if available)
        if self.memory_enabled:
            try:
                loop = asyncio.get_event_loop()
                memory_suggestions = loop.run_until_complete(self.get_suggested_actions_from_memory())
                suggestions.extend(memory_suggestions)
            except RuntimeError:
                # No event loop available, continue with local suggestions
                pass
        
        # Time-based suggestions
        hour = datetime.now().hour
        if 6 <= hour < 9:  # Morning
            suggestions.append("create_daily_note")
            suggestions.append("review_tasks")
        elif 17 <= hour < 20:  # Evening
            suggestions.append("daily_review")
            suggestions.append("backup_vault")
        
        # Menu-based suggestions
        current_menu = self.current_context.get('current_menu')
        if current_menu == 'tag_menu':
            suggestions.extend(["analyze_tags", "cleanup_tags"])
        elif current_menu == 'research_menu':
            suggestions.extend(["research_topic", "create_note"])
        
        # History-based suggestions
        if self.action_history:
            last_action = self.action_history[-1]['action']
            if last_action == 'analyze_tags':
                suggestions.append("cleanup_tags")
            elif last_action == 'create_research_note':
                suggestions.append("web_search")
        
        # Remove duplicates while preserving order
        seen = set()
        return [s for s in suggestions if not (s in seen or seen.add(s))]
    
    def get_vault_state(self) -> Dict[str, Any]:
        """Get current vault state (cached)"""
        # Update vault state if stale (> 5 minutes)
        if (not self._last_vault_scan or 
            (datetime.now() - self._last_vault_scan).seconds > 300):
            self._update_vault_state()
        
        return self.vault_state
    
    def _update_vault_state(self):
        """Update vault state cache"""
        if not self.vault_path or not os.path.exists(self.vault_path):
            return
        
        try:
            state = {
                'total_files': 0,
                'markdown_files': 0,
                'recent_files': [],
                'last_modified': None,
            }
            
            # Scan vault
            for root, dirs, files in os.walk(self.vault_path):
                for file in files:
                    state['total_files'] += 1
                    if file.endswith('.md'):
                        state['markdown_files'] += 1
                        
                        # Track recent files
                        file_path = os.path.join(root, file)
                        mtime = os.path.getmtime(file_path)
                        state['recent_files'].append({
                            'path': file_path,
                            'name': file,
                            'modified': mtime
                        })
            
            # Sort by modification time
            state['recent_files'].sort(key=lambda x: x['modified'], reverse=True)
            state['recent_files'] = state['recent_files'][:10]  # Keep top 10
            
            if state['recent_files']:
                state['last_modified'] = datetime.fromtimestamp(
                    state['recent_files'][0]['modified']
                ).isoformat()
            
            self.vault_state = state
            self._last_vault_scan = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating vault state: {e}")
    
    def _get_time_context(self) -> str:
        """Get time-based context"""
        hour = datetime.now().hour
        
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def _save_context(self):
        """Save context to disk"""
        if not self.vault_path:
            return
        
        try:
            context_dir = os.path.join(self.vault_path, '.obsidian-tools')
            os.makedirs(context_dir, exist_ok=True)
            
            context_file = os.path.join(context_dir, 'context.json')
            
            context_data = {
                'current_context': self.current_context,
                'action_frequency': dict(self.action_frequency),
                'recent_actions': list(self.action_history)[-20:],
                'recent_queries': list(self.query_history)[-20:],
                'vault_state': self.vault_state,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump(context_data, f, indent=2)
                
            logger.debug("Context saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving context: {e}")
    
    def _load_context(self):
        """Load context from disk"""
        if not self.vault_path:
            return
        
        try:
            context_file = os.path.join(self.vault_path, '.obsidian-tools', 'context.json')
            
            if os.path.exists(context_file):
                with open(context_file, 'r', encoding='utf-8') as f:
                    context_data = json.load(f)
                
                # Restore action frequency
                self.action_frequency = defaultdict(int, context_data.get('action_frequency', {}))
                
                # Restore recent actions
                for action in context_data.get('recent_actions', []):
                    self.action_history.append(action)
                
                # Restore recent queries
                for query in context_data.get('recent_queries', []):
                    self.query_history.append(query)
                
                # Restore vault state
                self.vault_state = context_data.get('vault_state', {})
                
                logger.debug("Context loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading context: {e}")
    
    def clear_session(self):
        """Clear current session data"""
        self.current_context = {
            'current_menu': None,
            'previous_menu': None,
            'last_action': None,
            'last_action_time': None,
            'session_start': datetime.now().isoformat(),
        }
        
        self.menu_history.clear()
        logger.info("Session context cleared")
    
    # Memory integration methods
    
    async def _initialize_memory(self):
        """Initialize memory system and create basic entities"""
        try:
            # Initialize memory client
            self.memory_enabled = await self.memory_client.initialize()
            
            if not self.memory_enabled:
                logger.warning("Memory system not available")
                return
            
            # Create or ensure basic entities exist
            await self._ensure_basic_entities()
            
            # Start session tracking
            await self._start_session_tracking()
            
            logger.info("Memory system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            self.memory_enabled = False
    
    async def _ensure_basic_entities(self):
        """Ensure basic entities (user, vault, session) exist in memory"""
        try:
            # Create user entity if it doesn't exist
            if not await self.memory_client.entity_exists(self.user_id):
                user_entity = MemoryEntityFactory.create_user_entity(self.user_id)
                await self.memory_client.create_entities([Entity(**user_entity)])
            
            # Create vault entity if vault path is provided
            if self.vault_path:
                vault_name = os.path.basename(self.vault_path)
                vault_entity_name = f"vault_{vault_name.replace(' ', '_')}"
                
                if not await self.memory_client.entity_exists(vault_entity_name):
                    vault_entity = MemoryEntityFactory.create_vault_entity(self.vault_path, vault_name)
                    await self.memory_client.create_entities([Entity(**vault_entity)])
                    
                    # Create user owns vault relation
                    relation = MemoryRelationFactory.create_user_owns_vault(self.user_id, vault_name)
                    await self.memory_client.create_relations([Relation(**relation)])
            
            # Create session entity
            session_entity = MemoryEntityFactory.create_session_entity(self.session_id)
            await self.memory_client.create_entities([Entity(**session_entity)])
            
        except Exception as e:
            logger.error(f"Error ensuring basic entities: {e}")
    
    async def _start_session_tracking(self):
        """Start tracking the current session"""
        try:
            # Create relation between user and session
            relation = {
                "from": self.user_id,
                "to": f"session_{self.session_id}",
                "relationType": RelationType.USES.value
            }
            await self.memory_client.create_relations([Relation(**relation)])
            
        except Exception as e:
            logger.error(f"Error starting session tracking: {e}")
    
    async def record_action_to_memory(self, action: str, result: Any = None):
        """Record action to memory system"""
        if not self.memory_enabled:
            return
        
        try:
            # Create action entity if it doesn't exist
            action_entity_name = f"action_{action.replace(' ', '_')}"
            if not await self.memory_client.entity_exists(action_entity_name):
                action_entity = MemoryEntityFactory.create_action_entity(action)
                await self.memory_client.create_entities([Entity(**action_entity)])
            
            # Update action observations with frequency and timing
            frequency = self.action_frequency[action]
            frequency_obs = ObservationTemplates.user_action_frequency(action, frequency)
            
            # Add time pattern observation
            hour = datetime.now().hour
            time_period = self._get_time_context()
            time_obs = ObservationTemplates.user_time_pattern(action, f"during {time_period}")
            
            await self.memory_client.add_observations(action_entity_name, [frequency_obs, time_obs])
            
            # Create/update relations
            user_performs_relation = MemoryRelationFactory.create_user_performs_action(self.user_id, action)
            await self.memory_client.create_relations([Relation(**user_performs_relation)])
            
            action_session_relation = MemoryRelationFactory.create_action_used_during_session(action, self.session_id)
            await self.memory_client.create_relations([Relation(**action_session_relation)])
            
        except Exception as e:
            logger.error(f"Error recording action to memory: {e}")
    
    async def record_preference_to_memory(self, preference_name: str, preference_value: str):
        """Record user preference to memory"""
        if not self.memory_enabled:
            return
        
        try:
            # Create preference entity
            pref_entity = MemoryEntityFactory.create_preference_entity(preference_name, preference_value)
            await self.memory_client.create_entities([Entity(**pref_entity)])
            
            # Create user prefers relation
            relation = MemoryRelationFactory.create_user_prefers(self.user_id, preference_name)
            await self.memory_client.create_relations([Relation(**relation)])
            
        except Exception as e:
            logger.error(f"Error recording preference to memory: {e}")
    
    async def get_user_patterns_from_memory(self) -> Dict[str, Any]:
        """Get user patterns and preferences from memory"""
        if not self.memory_enabled:
            return {}
        
        try:
            # Search for user-related information
            user_query = MemoryQueryBuilder.build_user_profile_query(self.user_id)
            search_result = await self.memory_client.search_nodes(user_query)
            
            if search_result.get("success"):
                return search_result.get("result", {})
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting user patterns from memory: {e}")
            return {}
    
    async def get_suggested_actions_from_memory(self) -> List[str]:
        """Get suggested actions based on memory patterns"""
        if not self.memory_enabled:
            return []
        
        try:
            # Get user's most frequent actions
            related_entities = await self.memory_client.get_related_entities(self.user_id)
            
            # Filter for actions and get suggestions
            action_suggestions = []
            for entity in related_entities:
                if entity.get("entity", "").startswith("action_"):
                    action_name = entity["entity"].replace("action_", "").replace("_", " ")
                    action_suggestions.append(action_name)
            
            return action_suggestions[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error getting suggestions from memory: {e}")
            return []
    
    async def finalize_session_memory(self):
        """Finalize session memory with summary observations"""
        if not self.memory_enabled:
            return
        
        try:
            # Calculate session metrics
            session_duration = (datetime.now() - datetime.fromisoformat(self.current_context['session_start'])).total_seconds()
            actions_count = len(self.action_history)
            
            # Create session summary observations
            session_observations = [
                f"Session duration: {session_duration:.0f} seconds",
                f"Actions performed: {actions_count}",
                f"Menus visited: {len(self.menu_history)}",
                f"Queries made: {len(self.query_history)}"
            ]
            
            # Add productivity score observation
            productivity_score = min(actions_count / max(session_duration / 60, 1), 10)  # Actions per minute, capped at 10
            productivity_obs = ObservationTemplates.session_productivity(productivity_score, actions_count)
            session_observations.append(productivity_obs)
            
            # Update session entity with observations
            session_entity_name = f"session_{self.session_id}"
            await self.memory_client.add_observations(session_entity_name, session_observations)
            
        except Exception as e:
            logger.error(f"Error finalizing session memory: {e}")
    
    def enable_memory_integration(self, enabled: bool = True):
        """Enable or disable memory integration"""
        self.memory_enabled = enabled
        if enabled and not self.memory_client._is_initialized:
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self._initialize_memory())
            except RuntimeError:
                logger.warning("No event loop available for memory initialization")