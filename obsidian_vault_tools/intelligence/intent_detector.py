"""
Intent Detection Engine

Analyzes user input and context to determine intent and suggest actions.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Enumeration of possible user intents"""
    # Analysis intents
    ANALYZE_TAGS = "analyze_tags"
    ANALYZE_VAULT = "analyze_vault"
    ANALYZE_LINKS = "analyze_links"
    ANALYZE_CONTENT = "analyze_content"
    
    # Research intents
    RESEARCH_TOPIC = "research_topic"
    CREATE_NOTE = "create_note"
    WEB_SEARCH = "web_search"
    
    # Organization intents
    ORGANIZE_FILES = "organize_files"
    CLEANUP_TAGS = "cleanup_tags"
    MERGE_DUPLICATES = "merge_duplicates"
    
    # Search intents
    FIND_FILES = "find_files"
    FIND_TAGS = "find_tags"
    FIND_CONTENT = "find_content"
    
    # Management intents
    BACKUP_VAULT = "backup_vault"
    EXPORT_DATA = "export_data"
    CONFIGURE_SETTINGS = "configure_settings"
    
    # MCP intents
    EXECUTE_TOOL = "execute_tool"
    LIST_TOOLS = "list_tools"
    
    # General
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass
class DetectedIntent:
    """Represents a detected user intent with metadata"""
    intent_type: IntentType
    confidence: float
    entities: Dict[str, Any]
    suggested_actions: List[str]
    context_used: Dict[str, Any]
    raw_input: str


class IntentDetector:
    """Detects user intent from natural language input"""
    
    def __init__(self):
        self.intent_patterns = self._build_intent_patterns()
        self.entity_patterns = self._build_entity_patterns()
        self.context_hints = self._build_context_hints()
        
    def _build_intent_patterns(self) -> Dict[IntentType, List[re.Pattern]]:
        """Build regex patterns for intent detection"""
        return {
            IntentType.ANALYZE_TAGS: [
                re.compile(r'\b(tag|tags|most used tags|popular tags|tag analysis|analyze tags)\b', re.I),
                re.compile(r'\b(what|which|show|list).*(tags?)\b', re.I),
                re.compile(r'\b(tag).*(statistics?|stats?|analysis|usage)\b', re.I),
            ],
            
            IntentType.ANALYZE_VAULT: [
                re.compile(r'\b(vault|overview|summary|health|status)\b', re.I),
                re.compile(r'\b(analyze|analysis|check|scan).*(vault|notes?|files?)\b', re.I),
                re.compile(r'\b(how many|count).*(files?|notes?)\b', re.I),
            ],
            
            IntentType.RESEARCH_TOPIC: [
                re.compile(r'\b(research|learn|study|investigate|explore)\s+(.+)', re.I),
                re.compile(r'\b(tell me about|what is|explain|define)\s+(.+)', re.I),
                re.compile(r'^([A-Z][a-zA-Z\s]+)$'),  # Single capitalized topic
            ],
            
            IntentType.ORGANIZE_FILES: [
                re.compile(r'\b(organize|reorganize|sort|arrange|structure)\b', re.I),
                re.compile(r'\b(clean|cleanup|tidy).*(files?|notes?|vault)\b', re.I),
            ],
            
            IntentType.FIND_FILES: [
                re.compile(r'\b(find|search|locate|where|show).*(files?|notes?)\b', re.I),
                re.compile(r'\b(files?|notes?).*(about|contain|with)\s+(.+)', re.I),
            ],
            
            IntentType.CLEANUP_TAGS: [
                re.compile(r'\b(clean|cleanup|fix|merge|organize).*(tags?)\b', re.I),
                re.compile(r'\b(duplicate|similar|redundant).*(tags?)\b', re.I),
            ],
            
            IntentType.BACKUP_VAULT: [
                re.compile(r'\b(backup|save|archive|export).*(vault|notes?|files?)\b', re.I),
                re.compile(r'\b(create|make).*(backup|archive)\b', re.I),
            ],
        }
    
    def _build_entity_patterns(self) -> Dict[str, re.Pattern]:
        """Build patterns for entity extraction"""
        return {
            'tag_name': re.compile(r'#([a-zA-Z0-9_-]+)'),
            'file_name': re.compile(r'([a-zA-Z0-9_-]+\.md)'),
            'topic': re.compile(r'"([^"]+)"|\'([^\']+)\''),
            'number': re.compile(r'\b(\d+)\b'),
            'date': re.compile(r'\b(\d{4}-\d{2}-\d{2})\b'),
        }
    
    def _build_context_hints(self) -> Dict[str, IntentType]:
        """Build context-based intent hints"""
        return {
            'tag_menu': IntentType.ANALYZE_TAGS,
            'research_menu': IntentType.RESEARCH_TOPIC,
            'analysis_menu': IntentType.ANALYZE_VAULT,
            'organization_menu': IntentType.ORGANIZE_FILES,
        }
    
    def detect_intent(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> DetectedIntent:
        """Detect intent from user input and context"""
        if not user_input:
            return self._unknown_intent(user_input, context)
        
        # Clean input
        user_input = user_input.strip()
        
        # Extract entities first
        entities = self._extract_entities(user_input)
        
        # Check context hints
        if context and 'current_menu' in context:
            hint_intent = self.context_hints.get(context['current_menu'])
            if hint_intent and self._is_simple_query(user_input):
                return self._create_intent(
                    hint_intent, 0.8, entities, user_input, context,
                    reason="Context-based detection from menu"
                )
        
        # Pattern matching
        best_match = None
        best_confidence = 0.0
        
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = pattern.search(user_input)
                if match:
                    confidence = self._calculate_confidence(match, user_input, pattern)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = intent_type
                        
                        # Extract topic for research
                        if intent_type == IntentType.RESEARCH_TOPIC and match.groups():
                            entities['topic'] = match.group(1).strip()
        
        if best_match:
            return self._create_intent(best_match, best_confidence, entities, user_input, context)
        
        # Fallback: Check if it's a simple topic (for research)
        if self._looks_like_topic(user_input):
            entities['topic'] = user_input
            return self._create_intent(
                IntentType.RESEARCH_TOPIC, 0.6, entities, user_input, context,
                reason="Detected as research topic"
            )
        
        return self._unknown_intent(user_input, context)
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = pattern.findall(text)
            if matches:
                if entity_type == 'topic' and matches:
                    # Handle quoted topics
                    entities[entity_type] = matches[0][0] or matches[0][1]
                else:
                    entities[entity_type] = matches[0] if len(matches) == 1 else matches
        
        return entities
    
    def _calculate_confidence(self, match: re.Match, text: str, pattern: re.Pattern) -> float:
        """Calculate confidence score for a match"""
        # Base confidence
        confidence = 0.7
        
        # Boost if match is at start
        if match.start() == 0:
            confidence += 0.1
            
        # Boost for exact matches
        if match.group(0).lower() == text.lower():
            confidence += 0.1
            
        # Boost for specific patterns
        if 'most used tags' in text.lower():
            confidence = 0.95
            
        return min(confidence, 1.0)
    
    def _is_simple_query(self, text: str) -> bool:
        """Check if input is a simple query without explicit intent"""
        # Simple if: short, no command words, possibly just a topic
        return len(text.split()) <= 3 and not any(
            word in text.lower() 
            for word in ['find', 'search', 'analyze', 'show', 'list']
        )
    
    def _looks_like_topic(self, text: str) -> bool:
        """Check if text looks like a research topic"""
        # Topics are typically: capitalized, 1-4 words, no special chars
        words = text.split()
        return (
            1 <= len(words) <= 4 and
            text[0].isupper() and
            not any(char in text for char in ['#', '@', '/', '\\', '.'])
        )
    
    def _create_intent(self, intent_type: IntentType, confidence: float, 
                      entities: Dict[str, Any], raw_input: str, 
                      context: Optional[Dict[str, Any]], reason: str = "") -> DetectedIntent:
        """Create a DetectedIntent object with suggested actions"""
        
        # Generate suggested actions based on intent
        suggested_actions = self._get_suggested_actions(intent_type, entities)
        
        return DetectedIntent(
            intent_type=intent_type,
            confidence=confidence,
            entities=entities,
            suggested_actions=suggested_actions,
            context_used=context or {},
            raw_input=raw_input
        )
    
    def _get_suggested_actions(self, intent_type: IntentType, entities: Dict[str, Any]) -> List[str]:
        """Get suggested actions for an intent"""
        actions = {
            IntentType.ANALYZE_TAGS: [
                "analyze_all_tags",
                "show_tag_statistics",
                "find_most_used_tags"
            ],
            IntentType.RESEARCH_TOPIC: [
                "research_topic",
                "create_research_note",
                "search_web_sources"
            ],
            IntentType.ORGANIZE_FILES: [
                "analyze_organization",
                "suggest_file_structure",
                "auto_organize_files"
            ],
            IntentType.ANALYZE_VAULT: [
                "show_vault_statistics",
                "analyze_vault_health",
                "generate_vault_report"
            ],
            IntentType.CLEANUP_TAGS: [
                "find_duplicate_tags",
                "merge_similar_tags",
                "fix_tag_formatting"
            ],
        }
        
        return actions.get(intent_type, ["show_help"])
    
    def _unknown_intent(self, raw_input: str, context: Optional[Dict[str, Any]]) -> DetectedIntent:
        """Create an unknown intent response"""
        return DetectedIntent(
            intent_type=IntentType.UNKNOWN,
            confidence=0.0,
            entities={},
            suggested_actions=["show_help", "list_available_commands"],
            context_used=context or {},
            raw_input=raw_input
        )