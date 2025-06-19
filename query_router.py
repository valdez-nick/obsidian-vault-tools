#!/usr/bin/env python3
"""
Query Router
Routes queries to appropriate models based on intent and content
"""

import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

from llm_model_manager import LLMModelManager, ModelResponse

logger = logging.getLogger(__name__)

@dataclass
class RoutedQuery:
    """Represents a query with routing information"""
    original_query: str
    intent: str
    entities: List[str]
    recommended_models: List[str]
    context_needed: bool
    complexity_score: float

class QueryRouter:
    """Routes queries to appropriate models and strategies"""
    
    def __init__(self, model_manager: LLMModelManager):
        self.model_manager = model_manager
        self.routing_rules = self._load_routing_rules()
        self.model_capabilities = self._define_model_capabilities()
        
    def _load_routing_rules(self) -> Dict[str, Any]:
        """Load routing rules from config"""
        config = self.model_manager.config
        return config.get("routing", {})
        
    def _define_model_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Define what each model is good at"""
        return {
            "general_qa": {
                "strengths": ["general_knowledge", "explanations", "reasoning"],
                "max_context": 4096,
                "speed": "medium",
                "cost": "medium"
            },
            "code_analyzer": {
                "strengths": ["code_analysis", "debugging", "technical_content"],
                "max_context": 8192,
                "speed": "slow",
                "cost": "high"
            },
            "summarizer": {
                "strengths": ["summarization", "key_points", "condensing"],
                "max_context": 8192,
                "speed": "fast",
                "cost": "low"
            },
            "entity_extractor": {
                "strengths": ["ner", "dates", "people", "projects"],
                "max_context": 2048,
                "speed": "very_fast",
                "cost": "very_low"
            },
            "intent_classifier": {
                "strengths": ["classification", "routing"],
                "max_context": 512,
                "speed": "very_fast",
                "cost": "very_low"
            }
        }
        
    async def route_query(self, query: str, vault_context: Optional[Dict] = None) -> RoutedQuery:
        """Analyze query and determine best routing strategy"""
        # Get intent
        intent = await self.model_manager.classify_intent(query)
        
        # Extract entities
        entities = await self._extract_entities(query)
        
        # Calculate complexity
        complexity = self._calculate_complexity(query, intent)
        
        # Determine which models to use
        recommended_models = self._recommend_models(intent, complexity, entities)
        
        # Check if we need vault context
        context_needed = self._needs_context(query, intent)
        
        return RoutedQuery(
            original_query=query,
            intent=intent,
            entities=entities,
            recommended_models=recommended_models,
            context_needed=context_needed,
            complexity_score=complexity
        )
        
    async def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query"""
        # Use entity extractor if available
        if "entity_extractor" in self.model_manager.config.get("models", {}):
            prompt = f"""Extract all named entities from this query. Include:
            - People names
            - Project names
            - Dates
            - File names
            - Technical terms
            
            Query: {query}
            
            Entities (comma-separated):"""
            
            try:
                response = await self.model_manager.query_model("entity_extractor", prompt)
                entities = [e.strip() for e in response.content.split(",") if e.strip()]
                return entities
            except:
                pass
                
        # Fallback to regex patterns
        entities = []
        
        # Extract quoted strings
        entities.extend(re.findall(r'"([^"]+)"', query))
        entities.extend(re.findall(r"'([^']+)'", query))
        
        # Extract potential file names
        entities.extend(re.findall(r'\b[\w-]+\.md\b', query))
        
        # Extract dates
        entities.extend(re.findall(r'\b\d{4}-\d{2}-\d{2}\b', query))
        
        # Extract hashtags
        entities.extend(re.findall(r'#\w+', query))
        
        return list(set(entities))
        
    def _calculate_complexity(self, query: str, intent: str) -> float:
        """Calculate query complexity score (0-1)"""
        complexity = 0.0
        
        # Length factor
        word_count = len(query.split())
        if word_count > 20:
            complexity += 0.2
        if word_count > 50:
            complexity += 0.2
            
        # Intent complexity
        complex_intents = ["analyze", "summarize", "extract"]
        if intent in complex_intents:
            complexity += 0.3
            
        # Multiple questions
        if query.count("?") > 1:
            complexity += 0.2
            
        # Technical terms
        tech_terms = ["algorithm", "architecture", "implementation", "analysis", "correlation"]
        if any(term in query.lower() for term in tech_terms):
            complexity += 0.1
            
        return min(complexity, 1.0)
        
    def _recommend_models(self, intent: str, complexity: float, entities: List[str]) -> List[str]:
        """Recommend models based on query characteristics"""
        models = []
        
        # Intent-based recommendations
        intent_models = {
            "search": ["general_qa"],
            "count": ["entity_extractor", "general_qa"],
            "summarize": ["summarizer"],
            "analyze": ["code_analyzer", "general_qa"],
            "extract": ["entity_extractor", "general_qa"]
        }
        
        models.extend(intent_models.get(intent, ["general_qa"]))
        
        # Add more models for complex queries
        if complexity > 0.7:
            models.append("general_qa")
            
        # Add code analyzer if code-related entities
        code_indicators = [".py", ".js", "function", "class", "method"]
        if any(indicator in str(entities).lower() for indicator in code_indicators):
            models.append("code_analyzer")
            
        # Remove duplicates while preserving order
        seen = set()
        unique_models = []
        for model in models:
            if model not in seen and model in self.model_manager.config.get("models", {}):
                seen.add(model)
                unique_models.append(model)
                
        return unique_models if unique_models else ["general_qa"]
        
    def _needs_context(self, query: str, intent: str) -> bool:
        """Determine if vault context is needed"""
        # Always need context for certain intents
        if intent in ["search", "summarize", "analyze", "extract"]:
            return True
            
        # Check for context indicators
        context_indicators = [
            "in my vault", "my notes", "my files", "my documents",
            "show me", "find", "search", "look for"
        ]
        
        return any(indicator in query.lower() for indicator in context_indicators)
        
    async def execute_routed_query(self, routed_query: RoutedQuery, 
                                 vault_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a query using the routing information"""
        results = {
            "query": routed_query.original_query,
            "intent": routed_query.intent,
            "entities": routed_query.entities,
            "responses": []
        }
        
        # Prepare context if needed
        context_prompt = ""
        if routed_query.context_needed and vault_context:
            context_prompt = self._prepare_context_prompt(vault_context, routed_query)
            
        # Build the full prompt
        full_prompt = self._build_prompt(routed_query, context_prompt)
        
        # Determine ensemble strategy based on complexity
        if routed_query.complexity_score > 0.7:
            strategy = "weighted_confidence"
        elif len(routed_query.recommended_models) > 2:
            strategy = "voting"
        else:
            strategy = "cascade"
            
        # Execute query with recommended models
        try:
            ensemble_result = await self.model_manager.ensemble_query(
                full_prompt,
                models=routed_query.recommended_models,
                strategy=strategy
            )
            
            results["response"] = ensemble_result["response"]
            results["models_used"] = ensemble_result["models_used"]
            results["strategy"] = ensemble_result["strategy"]
            results["confidence"] = ensemble_result.get("confidence", 0.0)
            
        except Exception as e:
            logger.error(f"Error executing routed query: {e}")
            results["error"] = str(e)
            results["response"] = "I encountered an error processing your query."
            
        return results
        
    def _prepare_context_prompt(self, vault_context: Dict, routed_query: RoutedQuery) -> str:
        """Prepare context from vault for the prompt"""
        context_parts = []
        
        # Add relevant files if searching
        if routed_query.intent in ["search", "extract"]:
            if "matching_files" in vault_context:
                context_parts.append("Relevant files from vault:")
                for file in vault_context["matching_files"][:5]:  # Limit to top 5
                    context_parts.append(f"- {file['name']}: {file.get('summary', '')}")
                    
        # Add statistics for count queries
        if routed_query.intent == "count":
            if "statistics" in vault_context:
                stats = vault_context["statistics"]
                context_parts.append(f"Vault statistics:")
                context_parts.append(f"- Total files: {stats.get('total_files', 0)}")
                context_parts.append(f"- Total words: {stats.get('total_words', 0)}")
                context_parts.append(f"- Total tags: {stats.get('total_tags', 0)}")
                
        # Add content snippets for analysis
        if routed_query.intent in ["analyze", "summarize"]:
            if "content_snippets" in vault_context:
                context_parts.append("Content excerpts:")
                for snippet in vault_context["content_snippets"][:3]:
                    context_parts.append(f"---\n{snippet}\n---")
                    
        return "\n".join(context_parts)
        
    def _build_prompt(self, routed_query: RoutedQuery, context: str) -> str:
        """Build the full prompt for the model"""
        prompt_parts = []
        
        # Add role/instruction based on intent
        if routed_query.intent == "search":
            prompt_parts.append("You are a helpful search assistant. Find and explain relevant information.")
        elif routed_query.intent == "summarize":
            prompt_parts.append("You are a summarization expert. Provide concise, accurate summaries.")
        elif routed_query.intent == "analyze":
            prompt_parts.append("You are an analysis expert. Provide deep insights and connections.")
        elif routed_query.intent == "extract":
            prompt_parts.append("You are an information extraction specialist. Extract and organize key data.")
        elif routed_query.intent == "count":
            prompt_parts.append("You are a data analyst. Provide accurate counts and statistics.")
            
        # Add context if available
        if context:
            prompt_parts.append(f"\nContext from vault:\n{context}")
            
        # Add the query
        prompt_parts.append(f"\nUser query: {routed_query.original_query}")
        
        # Add specific instructions based on entities
        if routed_query.entities:
            prompt_parts.append(f"\nPay special attention to: {', '.join(routed_query.entities)}")
            
        prompt_parts.append("\nProvide a helpful, accurate response:")
        
        return "\n".join(prompt_parts)

# Integration test
async def test_query_router():
    """Test the query router"""
    print("🚀 Testing Query Router")
    
    # Initialize components
    model_manager = LLMModelManager()
    router = QueryRouter(model_manager)
    
    # Test queries
    test_queries = [
        "How many meeting notes do I have from last month?",
        "Summarize the key points from the project Alpha documentation",
        "Find all mentions of machine learning in my notes",
        "What are the main themes in my daily notes?",
        "Extract all action items from meeting notes"
    ]
    
    # Mock vault context
    mock_context = {
        "statistics": {
            "total_files": 150,
            "total_words": 50000,
            "total_tags": 45
        },
        "matching_files": [
            {"name": "Project Alpha.md", "summary": "Main project documentation"},
            {"name": "ML Research.md", "summary": "Machine learning research notes"}
        ]
    }
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        
        # Route query
        routed = await router.route_query(query, mock_context)
        
        print(f"Intent: {routed.intent}")
        print(f"Entities: {routed.entities}")
        print(f"Recommended models: {routed.recommended_models}")
        print(f"Complexity: {routed.complexity_score:.2f}")
        print(f"Context needed: {routed.context_needed}")
        
        # Execute if models available
        provider_status = await model_manager.check_providers()
        if provider_status.get("ollama", False):
            print("\n🔄 Executing query...")
            result = await router.execute_routed_query(routed, mock_context)
            print(f"Response preview: {result.get('response', '')[:100]}...")
            print(f"Models used: {result.get('models_used', [])}")
            print(f"Strategy: {result.get('strategy', 'none')}")

if __name__ == "__main__":
    asyncio.run(test_query_router())