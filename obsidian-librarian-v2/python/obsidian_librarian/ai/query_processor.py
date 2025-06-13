"""
Query processing for natural language understanding.

Uses AI models to understand user queries and extract actionable intents,
entities, and filters for intelligent vault operations.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog

from .language_models import LanguageModelService, ChatRequest, ChatMessage
from .embeddings import EmbeddingService

logger = structlog.get_logger(__name__)


class QueryIntent(str, Enum):
    """Possible query intents."""
    SEARCH = "search"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ANALYZE = "analyze"
    ORGANIZE = "organize"
    RESEARCH = "research"
    SUMMARIZE = "summarize"
    FIND_SIMILAR = "find_similar"
    GET_STATS = "get_stats"
    UNKNOWN = "unknown"


@dataclass
class QueryEntity:
    """Extracted entity from query."""
    text: str
    type: str  # note_name, tag, date, author, etc.
    confidence: float


@dataclass
class QueryResult:
    """Result of query processing."""
    intent: QueryIntent
    entities: List[QueryEntity]
    filters: Dict[str, Any]
    search_terms: List[str]
    confidence: float
    semantic_query: Optional[str] = None


class QueryProcessor:
    """Processes natural language queries for the librarian."""
    
    def __init__(
        self, 
        language_service: Optional[LanguageModelService] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.language_service = language_service
        self.embedding_service = embedding_service
        self.cache: Dict[str, QueryResult] = {}
        
        # Rule-based patterns for quick intent detection
        self.intent_patterns = {
            QueryIntent.SEARCH: [
                r'\b(find|search|look for|show me)\b',
                r'\b(where is|what is)\b',
                r'\b(list|get)\b.*\b(notes|files)\b',
            ],
            QueryIntent.CREATE: [
                r'\b(create|make|new|add)\b.*\b(note|file)\b',
                r'\b(write|draft)\b',
            ],
            QueryIntent.RESEARCH: [
                r'\b(research|investigate|explore)\b',
                r'\b(find information about|learn about)\b',
                r'\b(look up|search for)\b.*\b(online|web|internet)\b',
            ],
            QueryIntent.ANALYZE: [
                r'\b(analyze|examine|review)\b',
                r'\b(quality|duplicates|similar)\b',
            ],
            QueryIntent.ORGANIZE: [
                r'\b(organize|sort|arrange|structure)\b',
                r'\b(clean up|tidy)\b',
            ],
            QueryIntent.SUMMARIZE: [
                r'\b(summarize|summary|overview)\b',
                r'\b(tldr|brief)\b',
            ],
            QueryIntent.GET_STATS: [
                r'\b(stats|statistics|count|how many)\b',
                r'\b(overview|dashboard)\b',
            ],
        }
    
    async def process_query(self, query: str, use_cache: bool = True) -> QueryResult:
        """Process a natural language query and extract intent."""
        # Check cache first
        if use_cache and query in self.cache:
            return self.cache[query]
        
        logger.debug("Processing query", query=query)
        
        # Try rule-based processing first (fast)
        result = await self._rule_based_processing(query)
        
        # Enhance with AI if available
        if self.language_service and result.confidence < 0.8:
            ai_result = await self._ai_processing(query)
            if ai_result.confidence > result.confidence:
                result = ai_result
        
        # Add semantic query if embedding service available
        if self.embedding_service and result.intent == QueryIntent.SEARCH:
            result.semantic_query = await self._generate_semantic_query(query)
        
        # Cache result
        if use_cache:
            self.cache[query] = result
        
        return result
    
    async def _rule_based_processing(self, query: str) -> QueryResult:
        """Process query using rule-based patterns."""
        query_lower = query.lower()
        
        # Detect intent
        intent = QueryIntent.UNKNOWN
        intent_confidence = 0.0
        
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    intent = intent_type
                    intent_confidence = 0.7  # Medium confidence for rule-based
                    break
            if intent != QueryIntent.UNKNOWN:
                break
        
        # Extract entities
        entities = self._extract_entities_rule_based(query)
        
        # Extract filters
        filters = self._extract_filters_rule_based(query)
        
        # Extract search terms
        search_terms = self._extract_search_terms(query)
        
        return QueryResult(
            intent=intent,
            entities=entities,
            filters=filters,
            search_terms=search_terms,
            confidence=intent_confidence,
        )
    
    async def _ai_processing(self, query: str) -> QueryResult:
        """Process query using AI models."""
        try:
            prompt = f"""Analyze this user query for a note-taking system and extract:

1. Intent: What does the user want to do?
   - search: Find existing notes
   - create: Create new content
   - research: Find information online
   - analyze: Analyze existing content
   - organize: Organize or restructure notes
   - summarize: Create summaries
   - get_stats: Get statistics/overview
   - unknown: Unclear intent

2. Entities: Important pieces of information (note names, tags, dates, etc.)

3. Filters: Specific constraints (date ranges, tags, file types, etc.)

4. Search terms: Key words for searching

Query: "{query}"

Respond in this exact JSON format:
{{
  "intent": "intent_name",
  "entities": [
    {{"text": "entity_text", "type": "entity_type", "confidence": 0.9}}
  ],
  "filters": {{
    "filter_name": "filter_value"
  }},
  "search_terms": ["term1", "term2"],
  "confidence": 0.9
}}"""
            
            request = ChatRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.1,  # Low temperature for consistent parsing
                max_tokens=500,
            )
            
            response = await self.language_service.chat_completion(request)
            
            # Parse JSON response
            import json
            try:
                data = json.loads(response.text)
                
                # Convert to our format
                intent = QueryIntent(data.get("intent", "unknown"))
                
                entities = [
                    QueryEntity(
                        text=e.get("text", ""),
                        type=e.get("type", "unknown"),
                        confidence=e.get("confidence", 0.5)
                    )
                    for e in data.get("entities", [])
                ]
                
                return QueryResult(
                    intent=intent,
                    entities=entities,
                    filters=data.get("filters", {}),
                    search_terms=data.get("search_terms", []),
                    confidence=data.get("confidence", 0.5),
                )
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse AI query response", response=response.text)
                return await self._rule_based_processing(query)
                
        except Exception as e:
            logger.warning("AI query processing failed", error=str(e))
            return await self._rule_based_processing(query)
    
    def _extract_entities_rule_based(self, query: str) -> List[QueryEntity]:
        """Extract entities using rule-based patterns."""
        entities = []
        
        # Extract quoted strings (likely note names or exact terms)
        quoted_pattern = r'"([^"]+)"'
        for match in re.finditer(quoted_pattern, query):
            entities.append(QueryEntity(
                text=match.group(1),
                type="exact_term",
                confidence=0.8
            ))
        
        # Extract hashtags (tags)
        tag_pattern = r'#(\w+)'
        for match in re.finditer(tag_pattern, query):
            entities.append(QueryEntity(
                text=match.group(1),
                type="tag",
                confidence=0.9
            ))
        
        # Extract date patterns
        date_patterns = [
            r'\b(today|yesterday|tomorrow)\b',
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b(last|this|next)\s+(week|month|year)\b',
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                entities.append(QueryEntity(
                    text=match.group(0),
                    type="date",
                    confidence=0.7
                ))
        
        # Extract note references (wiki links)
        wiki_pattern = r'\[\[([^\]]+)\]\]'
        for match in re.finditer(wiki_pattern, query):
            entities.append(QueryEntity(
                text=match.group(1),
                type="note_name",
                confidence=0.9
            ))
        
        return entities
    
    def _extract_filters_rule_based(self, query: str) -> Dict[str, Any]:
        """Extract filters using rule-based patterns."""
        filters = {}
        query_lower = query.lower()
        
        # File type filters
        if any(ext in query_lower for ext in ['.md', 'markdown']):
            filters['file_type'] = 'markdown'
        
        # Date filters
        if 'today' in query_lower:
            filters['created_date'] = 'today'
        elif 'yesterday' in query_lower:
            filters['created_date'] = 'yesterday'
        elif 'this week' in query_lower:
            filters['created_date'] = 'this_week'
        elif 'this month' in query_lower:
            filters['created_date'] = 'this_month'
        
        # Size filters
        if 'long' in query_lower or 'large' in query_lower:
            filters['min_word_count'] = 500
        elif 'short' in query_lower or 'small' in query_lower:
            filters['max_word_count'] = 100
        
        # Content filters
        if 'empty' in query_lower:
            filters['max_word_count'] = 10
        elif 'incomplete' in query_lower:
            filters['has_todos'] = True
        
        return filters
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract key search terms from query."""
        # Remove common stop words and query words
        stop_words = {
            'find', 'search', 'look', 'show', 'get', 'list', 'all', 'my', 'the',
            'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'about', 'notes', 'note', 'file', 'files'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        search_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return search_terms[:10]  # Limit to 10 terms
    
    async def _generate_semantic_query(self, query: str) -> Optional[str]:
        """Generate a semantic query for embedding-based search."""
        if not self.embedding_service:
            return None
        
        # For semantic search, we want to extract the core meaning
        # Remove query structure words and focus on content
        semantic_patterns = [
            r'\b(?:find|search|look for|show me)\s+(.+)',
            r'\b(?:notes about|files about)\s+(.+)',
            r'\b(?:anything on|content about)\s+(.+)',
        ]
        
        for pattern in semantic_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, clean up the query
        cleaned = re.sub(r'\b(?:find|search|look|show|get|list|all|my|notes?|files?)\b', '', query, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned if cleaned else query
    
    async def expand_query_with_synonyms(
        self, 
        query_result: QueryResult,
        context_notes: Optional[List[str]] = None
    ) -> QueryResult:
        """Expand query with synonyms and related terms."""
        if not self.language_service:
            return query_result
        
        try:
            search_terms_str = ', '.join(query_result.search_terms)
            context_str = '\n'.join(context_notes[:3]) if context_notes else ''
            
            prompt = f"""Given these search terms: "{search_terms_str}"
            
{f"And this context from recent notes: {context_str}" if context_str else ""}

Suggest 3-5 related terms or synonyms that might help find relevant notes.
Respond with just a comma-separated list of terms."""
            
            request = ChatRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.7,
                max_tokens=100,
            )
            
            response = await self.language_service.chat_completion(request)
            
            # Parse synonyms
            synonyms = [term.strip() for term in response.text.split(',') if term.strip()]
            
            # Add to search terms
            expanded_terms = query_result.search_terms + synonyms[:3]  # Limit additions
            
            # Create new result with expanded terms
            return QueryResult(
                intent=query_result.intent,
                entities=query_result.entities,
                filters=query_result.filters,
                search_terms=expanded_terms,
                confidence=query_result.confidence,
                semantic_query=query_result.semantic_query,
            )
            
        except Exception as e:
            logger.warning("Query expansion failed", error=str(e))
            return query_result
    
    async def suggest_queries(self, context: str) -> List[str]:
        """Suggest helpful queries based on context."""
        if not self.language_service:
            return []
        
        try:
            prompt = f"""Based on this context about a user's notes: "{context[:500]}"

Suggest 3-5 helpful queries the user might want to make to explore their vault.
Examples: "Find all project notes from this month", "Show me notes about machine learning"

Respond with one query per line."""
            
            request = ChatRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.8,
                max_tokens=200,
            )
            
            response = await self.language_service.chat_completion(request)
            
            suggestions = [
                line.strip().strip('"').strip("'")
                for line in response.text.split('\n')
                if line.strip()
            ]
            
            return suggestions[:5]
            
        except Exception as e:
            logger.warning("Query suggestion failed", error=str(e))
            return []
    
    def clear_cache(self):
        """Clear the query cache."""
        self.cache.clear()
    
    async def get_processor_stats(self) -> Dict[str, Any]:
        """Get query processor statistics."""
        return {
            "cache_size": len(self.cache),
            "has_language_service": self.language_service is not None,
            "has_embedding_service": self.embedding_service is not None,
            "supported_intents": [intent.value for intent in QueryIntent],
        }