"""
Memory Models

Defines entity types, relation schemas, and observation structures for the
Obsidian Vault Tools memory system using the MCP memory server.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import json


class EntityType(Enum):
    """Supported entity types in the knowledge graph"""
    USER = "user"
    VAULT = "vault" 
    NOTE = "note"
    TAG = "tag"
    FOLDER = "folder"
    ACTION = "action"
    SESSION = "session"
    TOOL = "tool"
    QUERY = "query"
    PREFERENCE = "preference"
    PATTERN = "pattern"
    TOPIC = "topic"
    PROJECT = "project"


class RelationType(Enum):
    """Supported relation types between entities"""
    # User relations
    OWNS = "owns"
    USES = "uses"
    PREFERS = "prefers"
    AVOIDS = "avoids"
    
    # Vault relations
    CONTAINS = "contains"
    LOCATED_IN = "located_in"
    LINKED_TO = "linked_to"
    TAGGED_WITH = "tagged_with"
    
    # Action relations
    PERFORMS = "performs"
    TRIGGERS = "triggers"
    FOLLOWS = "follows"
    PRECEDES = "precedes"
    
    # Content relations
    RELATES_TO = "relates_to"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    REFERENCES = "references"
    
    # Temporal relations
    CREATED_DURING = "created_during"
    USED_DURING = "used_during"
    MODIFIED_DURING = "modified_during"
    
    # Research relations
    RESEARCHES = "researches"
    INTERESTED_IN = "interested_in"
    EXPERT_IN = "expert_in"


@dataclass
class ObservationSchema:
    """Schema for different types of observations"""
    
    # User observations
    USER_PREFERENCES = [
        "preferred_file_format",
        "favorite_tags", 
        "common_folder_structure",
        "writing_time_preference",
        "backup_frequency_preference",
        "interface_theme_preference"
    ]
    
    USER_BEHAVIORS = [
        "average_session_duration",
        "most_active_hours",
        "common_action_sequences",
        "note_creation_patterns",
        "tag_usage_patterns",
        "folder_organization_style"
    ]
    
    USER_SKILLS = [
        "markdown_proficiency_level",
        "regex_usage_comfort",
        "automation_preference_level",
        "technical_knowledge_level"
    ]
    
    # Vault observations
    VAULT_CHARACTERISTICS = [
        "total_note_count",
        "primary_content_types",
        "organizational_structure",
        "link_density",
        "tag_complexity",
        "vault_size",
        "creation_date",
        "last_major_reorganization"
    ]
    
    # Note observations
    NOTE_PROPERTIES = [
        "word_count",
        "creation_date",
        "last_modified",
        "link_count",
        "tag_count",
        "content_type",
        "importance_level",
        "completion_status"
    ]
    
    # Action observations
    ACTION_METRICS = [
        "frequency_of_use",
        "success_rate",
        "average_duration",
        "user_satisfaction",
        "error_rate",
        "context_of_use"
    ]
    
    # Session observations
    SESSION_DATA = [
        "duration",
        "actions_performed",
        "tools_used",
        "goals_achieved",
        "frustration_indicators",
        "productivity_score"
    ]


class MemoryEntityFactory:
    """Factory for creating standardized memory entities"""
    
    @staticmethod
    def create_user_entity(user_id: str = "default_user") -> Dict[str, Any]:
        """Create a user entity with initial observations"""
        return {
            "name": user_id,
            "entityType": EntityType.USER.value,
            "observations": [
                f"User created on {datetime.now().isoformat()}",
                "New user with default preferences"
            ]
        }
    
    @staticmethod
    def create_vault_entity(vault_path: str, vault_name: str = None) -> Dict[str, Any]:
        """Create a vault entity"""
        if not vault_name:
            from pathlib import Path
            vault_name = Path(vault_path).name
        
        return {
            "name": f"vault_{vault_name.replace(' ', '_')}",
            "entityType": EntityType.VAULT.value,
            "observations": [
                f"Vault path: {vault_path}",
                f"Vault initialized on {datetime.now().isoformat()}"
            ]
        }
    
    @staticmethod
    def create_note_entity(note_path: str, note_name: str = None) -> Dict[str, Any]:
        """Create a note entity"""
        if not note_name:
            from pathlib import Path
            note_name = Path(note_path).stem
        
        return {
            "name": f"note_{note_name.replace(' ', '_')}",
            "entityType": EntityType.NOTE.value,
            "observations": [
                f"Note path: {note_path}",
                f"Note tracked on {datetime.now().isoformat()}"
            ]
        }
    
    @staticmethod
    def create_tag_entity(tag_name: str) -> Dict[str, Any]:
        """Create a tag entity"""
        return {
            "name": f"tag_{tag_name.replace('#', '').replace(' ', '_')}",
            "entityType": EntityType.TAG.value,
            "observations": [
                f"Tag name: {tag_name}",
                f"Tag first seen on {datetime.now().isoformat()}"
            ]
        }
    
    @staticmethod
    def create_action_entity(action_name: str, tool_name: str = None) -> Dict[str, Any]:
        """Create an action entity"""
        observations = [
            f"Action first performed on {datetime.now().isoformat()}"
        ]
        
        if tool_name:
            observations.append(f"Associated with tool: {tool_name}")
        
        return {
            "name": f"action_{action_name.replace(' ', '_')}",
            "entityType": EntityType.ACTION.value,
            "observations": observations
        }
    
    @staticmethod
    def create_session_entity(session_id: str = None) -> Dict[str, Any]:
        """Create a session entity"""
        if not session_id:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return {
            "name": f"session_{session_id}",
            "entityType": EntityType.SESSION.value,
            "observations": [
                f"Session started on {datetime.now().isoformat()}"
            ]
        }
    
    @staticmethod
    def create_tool_entity(tool_name: str, tool_category: str = None) -> Dict[str, Any]:
        """Create a tool entity"""
        observations = [
            f"Tool first used on {datetime.now().isoformat()}"
        ]
        
        if tool_category:
            observations.append(f"Category: {tool_category}")
        
        return {
            "name": f"tool_{tool_name.replace(' ', '_')}",
            "entityType": EntityType.TOOL.value,
            "observations": observations
        }
    
    @staticmethod
    def create_preference_entity(preference_name: str, preference_value: str) -> Dict[str, Any]:
        """Create a preference entity"""
        return {
            "name": f"pref_{preference_name.replace(' ', '_')}",
            "entityType": EntityType.PREFERENCE.value,
            "observations": [
                f"Preference value: {preference_value}",
                f"Preference set on {datetime.now().isoformat()}"
            ]
        }
    
    @staticmethod
    def create_topic_entity(topic_name: str, domain: str = None) -> Dict[str, Any]:
        """Create a topic/research interest entity"""
        observations = [
            f"Topic first identified on {datetime.now().isoformat()}"
        ]
        
        if domain:
            observations.append(f"Domain: {domain}")
        
        return {
            "name": f"topic_{topic_name.replace(' ', '_')}",
            "entityType": EntityType.TOPIC.value,
            "observations": observations
        }


class MemoryRelationFactory:
    """Factory for creating standardized memory relations"""
    
    @staticmethod
    def create_user_owns_vault(user_id: str, vault_name: str) -> Dict[str, Any]:
        """Create user owns vault relation"""
        return {
            "from": user_id,
            "to": f"vault_{vault_name.replace(' ', '_')}",
            "relationType": RelationType.OWNS.value
        }
    
    @staticmethod
    def create_user_prefers(user_id: str, preference_name: str) -> Dict[str, Any]:
        """Create user prefers relation"""
        return {
            "from": user_id,
            "to": f"pref_{preference_name.replace(' ', '_')}",
            "relationType": RelationType.PREFERS.value
        }
    
    @staticmethod
    def create_user_performs_action(user_id: str, action_name: str) -> Dict[str, Any]:
        """Create user performs action relation"""
        return {
            "from": user_id,
            "to": f"action_{action_name.replace(' ', '_')}",
            "relationType": RelationType.PERFORMS.value
        }
    
    @staticmethod
    def create_vault_contains_note(vault_name: str, note_name: str) -> Dict[str, Any]:
        """Create vault contains note relation"""
        return {
            "from": f"vault_{vault_name.replace(' ', '_')}",
            "to": f"note_{note_name.replace(' ', '_')}",
            "relationType": RelationType.CONTAINS.value
        }
    
    @staticmethod
    def create_note_tagged_with(note_name: str, tag_name: str) -> Dict[str, Any]:
        """Create note tagged with relation"""
        return {
            "from": f"note_{note_name.replace(' ', '_')}",
            "to": f"tag_{tag_name.replace('#', '').replace(' ', '_')}",
            "relationType": RelationType.TAGGED_WITH.value
        }
    
    @staticmethod
    def create_note_links_to(from_note: str, to_note: str) -> Dict[str, Any]:
        """Create note links to note relation"""
        return {
            "from": f"note_{from_note.replace(' ', '_')}",
            "to": f"note_{to_note.replace(' ', '_')}",
            "relationType": RelationType.LINKED_TO.value
        }
    
    @staticmethod
    def create_user_researches_topic(user_id: str, topic_name: str) -> Dict[str, Any]:
        """Create user researches topic relation"""
        return {
            "from": user_id,
            "to": f"topic_{topic_name.replace(' ', '_')}",
            "relationType": RelationType.RESEARCHES.value
        }
    
    @staticmethod
    def create_action_used_during_session(action_name: str, session_id: str) -> Dict[str, Any]:
        """Create action used during session relation"""
        return {
            "from": f"action_{action_name.replace(' ', '_')}",
            "to": f"session_{session_id}",
            "relationType": RelationType.USED_DURING.value
        }


class ObservationTemplates:
    """Templates for common observation types"""
    
    @staticmethod
    def user_action_frequency(action_name: str, frequency: int, time_period: str = "week") -> str:
        """Generate user action frequency observation"""
        return f"Performs {action_name} {frequency} times per {time_period}"
    
    @staticmethod
    def user_time_pattern(action_name: str, time_info: str) -> str:
        """Generate user time pattern observation"""
        return f"Usually performs {action_name} {time_info}"
    
    @staticmethod
    def user_preference_strength(preference: str, strength: str) -> str:
        """Generate user preference strength observation"""
        return f"Shows {strength} preference for {preference}"
    
    @staticmethod
    def tool_success_rate(tool_name: str, success_rate: float) -> str:
        """Generate tool success rate observation"""
        return f"{tool_name} has {success_rate:.1%} success rate for this user"
    
    @staticmethod
    def vault_growth_pattern(growth_info: str) -> str:
        """Generate vault growth pattern observation"""
        return f"Vault growth pattern: {growth_info}"
    
    @staticmethod
    def note_importance_indicator(indicator: str, confidence: float) -> str:
        """Generate note importance indicator observation"""
        return f"Importance indicator: {indicator} (confidence: {confidence:.2f})"
    
    @staticmethod
    def session_productivity(productivity_score: float, actions_count: int) -> str:
        """Generate session productivity observation"""
        return f"Productivity score: {productivity_score:.2f} with {actions_count} actions"


class MemoryQueryBuilder:
    """Builder for common memory queries and operations"""
    
    @staticmethod
    def build_user_profile_query(user_id: str) -> str:
        """Build query to get comprehensive user profile"""
        return f"Find all information about user {user_id} including preferences, behaviors, and patterns"
    
    @staticmethod
    def build_vault_analysis_query(vault_name: str) -> str:
        """Build query to analyze vault characteristics"""
        return f"Find all information about vault {vault_name} including structure, content, and usage patterns"
    
    @staticmethod
    def build_tool_usage_query(tool_name: str) -> str:
        """Build query to analyze tool usage patterns"""
        return f"Find usage patterns and effectiveness data for tool {tool_name}"
    
    @staticmethod
    def build_research_interests_query(user_id: str) -> str:
        """Build query to find user's research interests"""
        return f"Find topics and subjects that {user_id} frequently researches or shows interest in"
    
    @staticmethod
    def build_collaboration_pattern_query(user_id: str) -> str:
        """Build query to find collaboration patterns"""
        return f"Find patterns in how {user_id} collaborates, shares, or works with others"


# Export main classes and enums
__all__ = [
    'EntityType',
    'RelationType', 
    'ObservationSchema',
    'MemoryEntityFactory',
    'MemoryRelationFactory',
    'ObservationTemplates',
    'MemoryQueryBuilder'
]