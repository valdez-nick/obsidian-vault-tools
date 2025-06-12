"""
Query processing for intelligent research queries and natural language understanding.
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = structlog.get_logger(__name__)


class QueryType(Enum):
    """Types of research queries."""
    GENERAL = "general"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    NEWS = "news"
    DOCUMENTATION = "documentation"
    CODE = "code"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"


class QueryIntent(Enum):
    """Intent behind a query."""
    LEARN = "learn"
    RESEARCH = "research"
    FIND_SOLUTION = "find_solution"
    COMPARE = "compare"
    GET_UPDATES = "get_updates"
    FIND_TOOLS = "find_tools"
    UNDERSTAND = "understand"


@dataclass
class QueryContext:
    """Context for query processing."""
    # User context
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    recent_queries: List[str] = field(default_factory=list)
    
    # Temporal context
    time_sensitive: bool = False
    date_range: Optional[Tuple[datetime, datetime]] = None
    
    # Domain context
    preferred_domains: List[str] = field(default_factory=list)
    excluded_domains: List[str] = field(default_factory=list)
    
    # Language preferences
    languages: List[str] = field(default_factory=lambda: ["en"])
    
    # Quality preferences
    quality_threshold: float = 0.7
    max_results: int = 50


@dataclass
class ProcessedQuery:
    """A processed and enriched research query."""
    # Original query
    original_text: str
    
    # Processed components
    cleaned_text: str
    keywords: List[str]
    key_phrases: List[str]
    entities: List[str]
    
    # Classification
    query_type: QueryType
    intent: QueryIntent
    confidence: float
    
    # Enrichment
    synonyms: Dict[str, List[str]]
    expanded_terms: List[str]
    related_concepts: List[str]
    
    # Search parameters
    search_terms: List[str]
    boost_terms: List[str]
    filter_terms: List[str]
    
    # Metadata
    session_id: str
    processed_at: datetime = field(default_factory=datetime.utcnow)
    context: Optional[QueryContext] = None


class QueryProcessor:
    """
    Intelligent query processor for research queries.
    
    Analyzes natural language queries to:
    - Extract key terms and concepts
    - Classify query type and intent
    - Expand queries with synonyms and related terms
    - Generate optimized search parameters
    - Provide context-aware processing
    """
    
    def __init__(self):
        # Pattern matching for query analysis
        self._patterns = self._compile_patterns()
        
        # Domain-specific vocabularies
        self._vocabularies = self._load_vocabularies()
        
        # Query history for learning
        self._query_history: List[ProcessedQuery] = []
        
        # Session tracking
        self._current_session = None
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for query analysis."""
        return {
            # Technical patterns
            'programming_languages': re.compile(
                r'\b(python|rust|javascript|typescript|java|c\+\+|golang|swift|kotlin|ruby|php)\b',
                re.IGNORECASE
            ),
            'frameworks': re.compile(
                r'\b(react|vue|angular|django|flask|fastapi|spring|rails|express|tokio|actix)\b',
                re.IGNORECASE
            ),
            'tools': re.compile(
                r'\b(docker|kubernetes|git|github|gitlab|aws|azure|gcp|terraform|ansible)\b',
                re.IGNORECASE
            ),
            
            # Academic patterns
            'research_terms': re.compile(
                r'\b(research|study|analysis|paper|thesis|methodology|algorithm|model|theory)\b',
                re.IGNORECASE
            ),
            'academic_fields': re.compile(
                r'\b(machine learning|ai|nlp|computer vision|data science|statistics|mathematics)\b',
                re.IGNORECASE
            ),
            
            # Intent patterns
            'learning_intent': re.compile(
                r'\b(how to|learn|tutorial|guide|beginner|introduction|getting started)\b',
                re.IGNORECASE
            ),
            'problem_solving': re.compile(
                r'\b(error|issue|problem|fix|solve|debug|troubleshoot|not working)\b',
                re.IGNORECASE
            ),
            'comparison': re.compile(
                r'\b(vs|versus|compare|comparison|difference|better|best|alternative)\b',
                re.IGNORECASE
            ),
            'recent_updates': re.compile(
                r'\b(latest|recent|new|update|2024|2023|current|modern)\b',
                re.IGNORECASE
            ),
            
            # Time patterns
            'date_references': re.compile(r'\b(\d{4}|\d{1,2}/\d{1,2}/\d{4}|last year|this year|recent)\b'),
            'time_sensitive': re.compile(r'\b(latest|recent|current|new|updated|breaking|live)\b', re.IGNORECASE),
            
            # Quality indicators
            'quality_terms': re.compile(
                r'\b(official|documentation|docs|reference|specification|standard|best practices)\b',
                re.IGNORECASE
            ),
            
            # Common stopwords and noise
            'noise_words': re.compile(
                r'\b(the|a|an|and|or|but|in|on|at|to|for|of|with|by|from|up|about|into|through|during)\b',
                re.IGNORECASE
            ),
        }
    
    def _load_vocabularies(self) -> Dict[str, Dict[str, List[str]]]:
        """Load domain-specific vocabularies and synonyms."""
        return {
            'programming': {
                'python': ['py', 'python3', 'cpython', 'pypy'],
                'javascript': ['js', 'ecmascript', 'node.js', 'nodejs'],
                'rust': ['rustlang', 'rust-lang'],
                'machine learning': ['ml', 'artificial intelligence', 'ai', 'deep learning'],
                'database': ['db', 'rdbms', 'nosql', 'sql'],
                'api': ['rest', 'graphql', 'endpoint', 'web service'],
            },
            'academic': {
                'research': ['study', 'investigation', 'analysis', 'examination'],
                'methodology': ['method', 'approach', 'technique', 'procedure'],
                'algorithm': ['algo', 'procedure', 'method', 'technique'],
                'paper': ['article', 'publication', 'document', 'study'],
            },
            'tools': {
                'version control': ['git', 'svn', 'mercurial', 'bazaar'],
                'containerization': ['docker', 'podman', 'containerd'],
                'orchestration': ['kubernetes', 'k8s', 'docker swarm', 'nomad'],
                'cloud': ['aws', 'azure', 'gcp', 'cloud computing'],
            }
        }
    
    async def process(self, query: str, context: Optional[QueryContext] = None) -> ProcessedQuery:
        """
        Process a natural language query into structured search parameters.
        
        Args:
            query: The natural language query
            context: Optional context for query processing
            
        Returns:
            Processed query with enriched metadata and search parameters
        """
        logger.debug("Processing query", query=query)
        
        # Generate session ID if not in context
        session_id = context.user_preferences.get('session_id', f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}") if context else f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Clean and normalize the query
        cleaned_text = self._clean_query(query)
        
        # Extract components
        keywords = self._extract_keywords(cleaned_text)
        key_phrases = self._extract_key_phrases(cleaned_text)
        entities = self._extract_entities(cleaned_text)
        
        # Classify query
        query_type, type_confidence = self._classify_query_type(cleaned_text)
        intent, intent_confidence = self._classify_intent(cleaned_text)
        
        # Overall confidence (average of type and intent confidence)
        confidence = (type_confidence + intent_confidence) / 2
        
        # Expand query terms
        synonyms = self._find_synonyms(keywords)
        expanded_terms = self._expand_terms(keywords, synonyms)
        related_concepts = self._find_related_concepts(keywords, query_type)
        
        # Generate search parameters
        search_terms = self._generate_search_terms(keywords, key_phrases, expanded_terms)
        boost_terms = self._identify_boost_terms(keywords, query_type, intent)
        filter_terms = self._identify_filter_terms(cleaned_text)
        
        processed_query = ProcessedQuery(
            original_text=query,
            cleaned_text=cleaned_text,
            keywords=keywords,
            key_phrases=key_phrases,
            entities=entities,
            query_type=query_type,
            intent=intent,
            confidence=confidence,
            synonyms=synonyms,
            expanded_terms=expanded_terms,
            related_concepts=related_concepts,
            search_terms=search_terms,
            boost_terms=boost_terms,
            filter_terms=filter_terms,
            session_id=session_id,
            context=context,
        )
        
        # Add to history for learning
        self._query_history.append(processed_query)
        
        # Limit history size
        if len(self._query_history) > 1000:
            self._query_history = self._query_history[-1000:]
        
        logger.debug("Query processed", 
                    type=query_type.value, 
                    intent=intent.value, 
                    confidence=confidence,
                    keywords=keywords)
        
        return processed_query
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query text."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters that don't add meaning
        cleaned = re.sub(r'[^\w\s\-\+\.]', ' ', cleaned)
        
        # Normalize common abbreviations
        cleaned = re.sub(r'\bjs\b', 'javascript', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bpy\b', 'python', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bml\b', 'machine learning', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bai\b', 'artificial intelligence', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from the query."""
        # Split into words
        words = text.lower().split()
        
        # Remove noise words
        filtered_words = []
        for word in words:
            if (len(word) > 2 and 
                not self._patterns['noise_words'].match(word) and
                word.isalpha()):
                filtered_words.append(word)
        
        # Remove duplicates while preserving order
        keywords = []
        seen = set()
        for word in filtered_words:
            if word not in seen:
                keywords.append(word)
                seen.add(word)
        
        return keywords
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases (2-3 word combinations)."""
        words = text.lower().split()
        phrases = []
        
        # Extract 2-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if not self._is_noise_phrase(phrase):
                phrases.append(phrase)
        
        # Extract 3-word phrases
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            if not self._is_noise_phrase(phrase):
                phrases.append(phrase)
        
        # Filter and deduplicate
        key_phrases = []
        for phrase in phrases:
            if self._is_meaningful_phrase(phrase) and phrase not in key_phrases:
                key_phrases.append(phrase)
        
        return key_phrases[:10]  # Limit to top 10
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simplified approach)."""
        entities = []
        
        # Programming languages
        for match in self._patterns['programming_languages'].finditer(text):
            entities.append(match.group().lower())
        
        # Frameworks
        for match in self._patterns['frameworks'].finditer(text):
            entities.append(match.group().lower())
        
        # Tools
        for match in self._patterns['tools'].finditer(text):
            entities.append(match.group().lower())
        
        # Remove duplicates
        return list(set(entities))
    
    def _classify_query_type(self, text: str) -> Tuple[QueryType, float]:
        """Classify the type of query."""
        scores = {query_type: 0.0 for query_type in QueryType}
        
        # Technical indicators
        if (self._patterns['programming_languages'].search(text) or
            self._patterns['frameworks'].search(text) or
            self._patterns['tools'].search(text)):
            scores[QueryType.TECHNICAL] += 0.4
            scores[QueryType.CODE] += 0.3
        
        # Academic indicators
        if (self._patterns['research_terms'].search(text) or
            self._patterns['academic_fields'].search(text)):
            scores[QueryType.ACADEMIC] += 0.4
        
        # Documentation indicators
        if self._patterns['quality_terms'].search(text):
            scores[QueryType.DOCUMENTATION] += 0.3
            scores[QueryType.REFERENCE] += 0.2
        
        # Tutorial indicators
        if self._patterns['learning_intent'].search(text):
            scores[QueryType.TUTORIAL] += 0.4
        
        # News indicators
        if self._patterns['recent_updates'].search(text):
            scores[QueryType.NEWS] += 0.3
        
        # Default to general if no strong indicators
        if max(scores.values()) < 0.3:
            scores[QueryType.GENERAL] = 0.6
        
        # Find the highest scoring type
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return best_type, confidence
    
    def _classify_intent(self, text: str) -> Tuple[QueryIntent, float]:
        """Classify the intent behind the query."""
        scores = {intent: 0.0 for intent in QueryIntent}
        
        # Learning intent
        if self._patterns['learning_intent'].search(text):
            scores[QueryIntent.LEARN] += 0.5
        
        # Problem solving intent
        if self._patterns['problem_solving'].search(text):
            scores[QueryIntent.FIND_SOLUTION] += 0.5
        
        # Comparison intent
        if self._patterns['comparison'].search(text):
            scores[QueryIntent.COMPARE] += 0.5
        
        # Research intent
        if self._patterns['research_terms'].search(text):
            scores[QueryIntent.RESEARCH] += 0.4
        
        # Updates intent
        if self._patterns['recent_updates'].search(text):
            scores[QueryIntent.GET_UPDATES] += 0.4
        
        # Tool finding intent
        if 'tool' in text.lower() or 'library' in text.lower():
            scores[QueryIntent.FIND_TOOLS] += 0.4
        
        # Understanding intent
        if any(word in text.lower() for word in ['what', 'why', 'how', 'explain', 'understand']):
            scores[QueryIntent.UNDERSTAND] += 0.3
        
        # Default to research if no clear intent
        if max(scores.values()) < 0.3:
            scores[QueryIntent.RESEARCH] = 0.5
        
        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]
        
        return best_intent, confidence
    
    def _find_synonyms(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Find synonyms for keywords using vocabularies."""
        synonyms = {}
        
        for keyword in keywords:
            keyword_synonyms = []
            
            # Check all vocabularies
            for domain, vocab in self._vocabularies.items():
                for term, term_synonyms in vocab.items():
                    if keyword in term.lower() or keyword in [s.lower() for s in term_synonyms]:
                        keyword_synonyms.extend(term_synonyms)
                        keyword_synonyms.append(term)
            
            # Remove duplicates and the original keyword
            unique_synonyms = list(set(keyword_synonyms))
            if keyword in unique_synonyms:
                unique_synonyms.remove(keyword)
            
            if unique_synonyms:
                synonyms[keyword] = unique_synonyms[:5]  # Limit to 5 synonyms
        
        return synonyms
    
    def _expand_terms(self, keywords: List[str], synonyms: Dict[str, List[str]]) -> List[str]:
        """Expand search terms with synonyms and related terms."""
        expanded = keywords.copy()
        
        # Add synonyms
        for keyword, keyword_synonyms in synonyms.items():
            expanded.extend(keyword_synonyms)
        
        # Remove duplicates
        return list(set(expanded))
    
    def _find_related_concepts(self, keywords: List[str], query_type: QueryType) -> List[str]:
        """Find concepts related to the keywords and query type."""
        related = []
        
        # Add type-specific related terms
        if query_type == QueryType.TECHNICAL:
            tech_related = ['implementation', 'best practices', 'architecture', 'performance']
            related.extend(tech_related)
        elif query_type == QueryType.ACADEMIC:
            academic_related = ['methodology', 'literature review', 'case study', 'evaluation']
            related.extend(academic_related)
        elif query_type == QueryType.TUTORIAL:
            tutorial_related = ['examples', 'step by step', 'beginner', 'guide']
            related.extend(tutorial_related)
        
        # Add keyword-specific related terms
        for keyword in keywords:
            if keyword in ['python', 'programming']:
                related.extend(['syntax', 'libraries', 'frameworks', 'development'])
            elif keyword in ['machine learning', 'ai']:
                related.extend(['algorithms', 'models', 'training', 'datasets'])
            elif keyword in ['web', 'frontend']:
                related.extend(['html', 'css', 'responsive', 'browser'])
        
        return list(set(related))[:10]  # Limit to 10 related concepts
    
    def _generate_search_terms(
        self, 
        keywords: List[str], 
        key_phrases: List[str], 
        expanded_terms: List[str]
    ) -> List[str]:
        """Generate optimized search terms."""
        search_terms = []
        
        # Add original keywords (highest priority)
        search_terms.extend(keywords)
        
        # Add key phrases
        search_terms.extend(key_phrases)
        
        # Add most relevant expanded terms
        for term in expanded_terms:
            if term not in search_terms and len(search_terms) < 20:
                search_terms.append(term)
        
        return search_terms
    
    def _identify_boost_terms(
        self, 
        keywords: List[str], 
        query_type: QueryType, 
        intent: QueryIntent
    ) -> List[str]:
        """Identify terms that should be boosted in search results."""
        boost_terms = []
        
        # Quality indicators should be boosted
        quality_terms = ['official', 'documentation', 'reference', 'specification']
        boost_terms.extend(quality_terms)
        
        # Intent-specific boosts
        if intent == QueryIntent.LEARN:
            boost_terms.extend(['tutorial', 'guide', 'beginner', 'introduction'])
        elif intent == QueryIntent.FIND_SOLUTION:
            boost_terms.extend(['solution', 'fix', 'resolve', 'answer'])
        elif intent == QueryIntent.GET_UPDATES:
            boost_terms.extend(['latest', 'recent', 'new', '2024', '2023'])
        
        # Type-specific boosts
        if query_type == QueryType.ACADEMIC:
            boost_terms.extend(['paper', 'research', 'study', 'analysis'])
        elif query_type == QueryType.CODE:
            boost_terms.extend(['example', 'implementation', 'code', 'github'])
        
        return list(set(boost_terms))
    
    def _identify_filter_terms(self, text: str) -> List[str]:
        """Identify terms that should be used as filters."""
        filter_terms = []
        
        # Time-based filters
        if self._patterns['time_sensitive'].search(text):
            filter_terms.append('recent')
        
        # Quality filters
        if self._patterns['quality_terms'].search(text):
            filter_terms.append('quality')
        
        # Language filters (if mentioned)
        for lang in ['python', 'rust', 'javascript', 'java']:
            if lang in text.lower():
                filter_terms.append(lang)
        
        return filter_terms
    
    def _is_noise_phrase(self, phrase: str) -> bool:
        """Check if a phrase is mostly noise."""
        words = phrase.split()
        noise_count = sum(1 for word in words if self._patterns['noise_words'].match(word))
        return noise_count / len(words) > 0.5
    
    def _is_meaningful_phrase(self, phrase: str) -> bool:
        """Check if a phrase is meaningful."""
        # Phrases with technical terms are meaningful
        if (self._patterns['programming_languages'].search(phrase) or
            self._patterns['frameworks'].search(phrase) or
            self._patterns['tools'].search(phrase)):
            return True
        
        # Phrases with academic terms are meaningful
        if self._patterns['research_terms'].search(phrase):
            return True
        
        # Phrases that are too short or too common are not meaningful
        if len(phrase) < 6 or phrase in ['the best', 'how to', 'what is']:
            return False
        
        return True
    
    async def get_query_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get query suggestions based on partial input and history."""
        suggestions = []
        
        # Find similar queries from history
        if self._query_history:
            partial_lower = partial_query.lower()
            
            for past_query in self._query_history[-100:]:  # Check last 100 queries
                if (past_query.original_text.lower().startswith(partial_lower) and
                    past_query.original_text not in suggestions):
                    suggestions.append(past_query.original_text)
                    if len(suggestions) >= limit:
                        break
        
        # Add common query patterns if not enough suggestions
        common_patterns = [
            f"{partial_query} tutorial",
            f"{partial_query} documentation",
            f"{partial_query} examples",
            f"how to {partial_query}",
            f"{partial_query} best practices",
        ]
        
        for pattern in common_patterns:
            if len(suggestions) < limit and pattern not in suggestions:
                suggestions.append(pattern)
        
        return suggestions[:limit]
    
    async def analyze_query_performance(self, query: ProcessedQuery) -> Dict[str, Any]:
        """Analyze how well a query was processed."""
        analysis = {
            'query_id': query.session_id,
            'complexity_score': len(query.keywords) + len(query.key_phrases),
            'confidence_score': query.confidence,
            'expansion_ratio': len(query.expanded_terms) / max(len(query.keywords), 1),
            'type_classification': query.query_type.value,
            'intent_classification': query.intent.value,
            'processing_quality': self._assess_processing_quality(query),
        }
        
        return analysis
    
    def _assess_processing_quality(self, query: ProcessedQuery) -> str:
        """Assess the quality of query processing."""
        if query.confidence > 0.8:
            return "high"
        elif query.confidence > 0.6:
            return "medium"
        else:
            return "low"