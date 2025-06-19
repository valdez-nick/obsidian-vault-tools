"""
Context Manager

Tracks user context, vault state, and interaction history.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import deque, defaultdict
import json
import os
import logging

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages context for intelligent decision making"""
    
    def __init__(self, vault_path: Optional[str] = None):
        self.vault_path = vault_path
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
        
        # Load persisted context if available
        self._load_context()
    
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
        """Get suggested actions based on context"""
        suggestions = []
        
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