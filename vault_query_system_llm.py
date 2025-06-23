#!/usr/bin/env python3
"""
Enhanced Vault Query System with LLM Integration
Replaces pattern matching with actual LLM models via Ollama
"""

import os
import re
import json
import asyncio
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging

# Import our LLM components
from llm_model_manager import LLMModelManager
from query_router import QueryRouter
from feedback_collector import FeedbackCollector, QueryFeedback
from models.embedding_adapter import EmbeddingAdapter, Document

# Import original vault query system for backward compatibility
from vault_query_system import VaultQuerySystem as BaseVaultQuerySystem

logger = logging.getLogger(__name__)

class VaultQuerySystemLLM(BaseVaultQuerySystem):
    """
    Enhanced vault query system using LLM models
    Inherits from base system for compatibility
    """
    
    def __init__(self, vault_path: str, config_path: str = "model_config.yaml"):
        super().__init__(vault_path)
        
        # Initialize LLM components
        self.model_manager = LLMModelManager(config_path)
        self.query_router = QueryRouter(self.model_manager)
        self.feedback_collector = FeedbackCollector()
        self.embedding_adapter = EmbeddingAdapter()
        
        # Enhanced caching
        self.embedding_cache = {}
        self.llm_cache = {}
        
        # Initialize asyncio event loop for sync methods
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None
            
    def _run_async(self, coro):
        """Helper to run async code in sync context"""
        if self.loop and self.loop.is_running():
            # We're already in an async context
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(coro)
        else:
            # Create new event loop
            return asyncio.run(coro)
            
    async def initialize(self) -> bool:
        """Initialize LLM components"""
        try:
            # Initialize model manager async components
            await self.model_manager.initialize_async()
            
            # Check if Ollama is available
            provider_status = await self.model_manager.check_providers()
            
            if not provider_status.get("ollama", False):
                logger.error("Ollama not available - AI models are required")
                raise RuntimeError(
                    "Ollama is not available. Please ensure:\n"
                    "1. Ollama is installed: https://ollama.ai\n"
                    "2. Ollama service is running: 'ollama serve'\n"
                    "3. Required models are pulled: 'ollama pull llama2'"
                )
                
            # Load embedding model
            await self.embedding_adapter.load_model()
            
            # Create vector index for vault
            await self._create_vault_index()
            
            logger.info("LLM components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM components: {e}")
            return False
            
    async def _create_vault_index(self):
        """Create vector index of vault content"""
        vault_data = self.scan_vault()
        
        # Create FAISS index if available
        try:
            dimension = 384  # Default for all-MiniLM-L6-v2
            await self.embedding_adapter.create_index("vault_content", dimension, "hnsw")
            
            # Add documents to index
            documents = []
            for file_info in vault_data['files'][:100]:  # Limit for demo
                file_path = file_info['path']
                content = vault_data['content_index'].get(file_path, '')
                
                if content:
                    # Generate embedding
                    embedding = await self.embedding_adapter.generate_embedding(content)
                    
                    doc = Document(
                        id=file_path,
                        text=content,
                        embedding=embedding,
                        metadata=file_info
                    )
                    documents.append(doc)
                    
            if documents:
                await self.embedding_adapter.add_documents("vault_content", documents)
                logger.info(f"Added {len(documents)} documents to vector index")
                
        except Exception as e:
            logger.warning(f"Could not create vector index: {e}")
            
    async def query_llm(self, query_text: str, use_feedback: bool = True) -> Dict[str, Any]:
        """
        Process query using LLM models
        
        Args:
            query_text: Natural language query
            use_feedback: Whether to collect feedback for improvement
            
        Returns:
            Query results with LLM-generated responses
        """
        start_time = datetime.now()
        query_id = str(uuid.uuid4())
        
        # Check cache first
        cache_key = f"{query_text}:{self.last_scan}"
        if cache_key in self.llm_cache:
            logger.info("Returning cached LLM response")
            return self.llm_cache[cache_key]
            
        try:
            # First, check if this is a vault-specific query that should be handled locally
            from obsidian_vault_tools.intelligence import IntelligenceOrchestrator
            
            # Create intelligence orchestrator with vault context
            orchestrator = IntelligenceOrchestrator(self)
            
            # Process through intelligence system first
            intel_result = await orchestrator.process_input(query_text)
            
            # If intelligence system handled it successfully, return that result
            if intel_result.success and intel_result.action_taken != "suggest_only":
                logger.info(f"Query handled by Intelligence System: {intel_result.action_taken}")
                
                # Format result for query system
                formatted_result = {
                    'query': query_text,
                    'response': intel_result.message,
                    'data': intel_result.data,
                    'action_taken': intel_result.action_taken,
                    'intent': intel_result.intent.intent_type.value if hasattr(intel_result, 'intent') else 'unknown',
                    'confidence': intel_result.confidence if hasattr(intel_result, 'confidence') else 1.0,
                    'llm_powered': False,  # Handled by intelligence system
                    'intelligence_handled': True
                }
                
                # Add metadata
                elapsed_time = (datetime.now() - start_time).total_seconds()
                formatted_result['query_id'] = query_id
                formatted_result['processing_time'] = elapsed_time
                
                # Cache result
                self.llm_cache[cache_key] = formatted_result
                
                return formatted_result
            
            # If intelligence system couldn't handle it, fall back to LLM
            logger.info("Falling back to LLM for general query processing")
            
            # Get vault context
            vault_context = await self._prepare_vault_context(query_text)
            
            # Route query to appropriate models
            routed_query = await self.query_router.route_query(query_text, vault_context)
            
            logger.info(f"Query routed - Intent: {routed_query.intent}, "
                       f"Models: {routed_query.recommended_models}")
            
            # Execute query with LLM
            result = await self.query_router.execute_routed_query(routed_query, vault_context)
            
            # Add metadata
            elapsed_time = (datetime.now() - start_time).total_seconds()
            result['query_id'] = query_id
            result['processing_time'] = elapsed_time
            result['llm_powered'] = True
            
            # Cache result
            self.llm_cache[cache_key] = result
            
            # Collect feedback if enabled
            if use_feedback:
                feedback = QueryFeedback(
                    query_id=query_id,
                    query=query_text,
                    intent=routed_query.intent,
                    response=result.get('response', ''),
                    models_used=result.get('models_used', []),
                    strategy=result.get('strategy', ''),
                    confidence=result.get('confidence', 0.0),
                    vault_context=vault_context,
                    response_time=elapsed_time
                )
                await self.feedback_collector.add_feedback(feedback)
                
            return result
            
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            # Do NOT automatically fall back - require user confirmation
            error_response = {
                'error': True,
                'error_type': 'llm_query_failed',
                'message': (
                    f"LLM query failed: {str(e)}\n\n"
                    "The AI-powered query system encountered an error. Possible causes:\n"
                    "- Ollama service is not running\n"
                    "- Required models are not installed\n"
                    "- Network connectivity issues\n\n"
                    "To use pattern matching as a fallback, you must explicitly request it.\n"
                    "Pattern matching provides basic search functionality but lacks the\n"
                    "intelligence and context understanding of the AI models."
                ),
                'fallback_available': True,
                'fallback_instructions': (
                    "To use pattern matching fallback, call:\n"
                    "system.use_pattern_matching_fallback(query_text)"
                ),
                'query': query_text,
                'query_id': query_id,
                'timestamp': datetime.now().isoformat()
            }
            return error_response
            
    async def _prepare_vault_context(self, query_text: str) -> Dict[str, Any]:
        """Prepare relevant vault context for LLM"""
        vault_data = self.scan_vault()
        context = {
            'statistics': vault_data['statistics'],
            'file_types': dict(vault_data['file_types'])
        }
        
        # Use vector search if available
        if hasattr(self.embedding_adapter, 'indices') and 'vault_content' in self.embedding_adapter.indices:
            try:
                # Search for relevant documents
                results = await self.embedding_adapter.search_by_text(
                    "vault_content", query_text, k=5
                )
                
                context['relevant_files'] = []
                context['content_snippets'] = []
                
                for doc, distance in results:
                    file_info = doc.metadata
                    context['relevant_files'].append({
                        'path': doc.id,
                        'name': file_info['name'],
                        'similarity_score': 1 / (1 + distance)  # Convert distance to similarity
                    })
                    
                    # Add content snippet
                    content_preview = doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
                    context['content_snippets'].append(content_preview)
                    
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                
        # Add top tags
        if vault_data['tags']:
            sorted_tags = sorted(vault_data['tags'].items(), 
                               key=lambda x: len(x[1]), reverse=True)
            context['top_tags'] = [tag for tag, _ in sorted_tags[:10]]
            
        # Add recent files
        recent_files = sorted(vault_data['files'], 
                            key=lambda x: x['modified'], reverse=True)[:5]
        context['recent_files'] = [
            {'name': f['name'], 'path': f['path']} 
            for f in recent_files
        ]
        
        return context
        
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for LLM query
        Requires AI models to be available - no automatic fallback
        """
        try:
            # Try LLM-powered query
            return self._run_async(self.query_llm(query_text))
        except Exception as e:
            # Do NOT automatically fall back to pattern matching
            logger.error(f"LLM query failed: {e}")
            error_response = {
                'error': True,
                'error_type': 'llm_unavailable',
                'message': (
                    "AI models are required for this query system but are not available.\n\n"
                    "Please ensure Ollama is running:\n"
                    "1. Check if Ollama is installed: 'ollama --version'\n"
                    "2. Start Ollama service: 'ollama serve'\n"
                    "3. Pull required models: 'ollama pull llama2'\n\n"
                    "The system is designed to be AI-first and requires LLM models to function properly.\n"
                    "Pattern matching fallback is available but must be explicitly requested."
                ),
                'original_error': str(e),
                'query': query_text,
                'timestamp': datetime.now().isoformat()
            }
            return error_response
            
    def use_pattern_matching_fallback(self, query_text: str) -> Dict[str, Any]:
        """
        Explicitly use pattern matching fallback when AI models are unavailable
        This method must be called directly by the user - no automatic fallback
        """
        logger.info("User explicitly requested pattern matching fallback")
        # Add a warning to the result
        result = super().query(query_text)
        result['fallback_mode'] = True
        result['fallback_warning'] = (
            "This result was generated using pattern matching, not AI models. "
            "Results may be less accurate and lack contextual understanding."
        )
        return result
        
    async def rate_response(self, query_id: str, rating: int, feedback: Optional[str] = None):
        """Rate a query response for improvement"""
        await self.feedback_collector.update_rating(query_id, rating, feedback)
        logger.info(f"Rated query {query_id}: {rating}/5")
        
    async def get_improvement_report(self) -> str:
        """Get model improvement recommendations"""
        return await self.feedback_collector.generate_improvement_report()
        
    async def train_custom_classifier(self, classifier_name: str, examples: List[Dict]):
        """Train a custom classifier for vault-specific tasks"""
        try:
            from models.classifier_adapter import ClassifierAdapter, ClassificationExample, ClassifierConfig
            
            adapter = ClassifierAdapter()
            
            # Create classifier config
            config = ClassifierConfig(
                name=classifier_name,
                classifier_type="nb",  # Naive Bayes for text
                vectorizer_type="tfidf"
            )
            
            # Create classifier
            await adapter.create_classifier(config)
            
            # Convert examples
            training_examples = [
                ClassificationExample(
                    text=ex['text'],
                    label=ex['label'],
                    metadata=ex.get('metadata')
                )
                for ex in examples
            ]
            
            # Add examples
            await adapter.add_training_examples(classifier_name, training_examples)
            
            # Train
            result = await adapter.train_classifier(classifier_name)
            
            # Save
            if 'error' not in result:
                await adapter.save_classifier(classifier_name)
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to train classifier: {e}")
            return {"error": str(e)}
            
    async def analyze_with_llm(self, file_path: str, analysis_type: str = "general") -> str:
        """
        Analyze a specific file using LLM
        
        Args:
            file_path: Path to file to analyze
            analysis_type: Type of analysis (general, summary, technical, etc.)
        """
        # Read file content
        full_path = self.vault_path / file_path
        if not full_path.exists():
            return f"File {file_path} not found"
            
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Build prompt based on analysis type
        prompts = {
            "general": "Analyze this document and provide key insights:",
            "summary": "Provide a concise summary of this document:",
            "technical": "Analyze the technical aspects of this document:",
            "action_items": "Extract all action items and tasks from this document:",
            "sentiment": "Analyze the overall sentiment and tone of this document:"
        }
        
        prompt = prompts.get(analysis_type, prompts["general"])
        full_prompt = f"{prompt}\n\n{content[:3000]}"  # Limit content length
        
        try:
            # Use the general_qa model for analysis
            response = await self.model_manager.query_model("general_qa", full_prompt)
            return response.content
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return f"Analysis failed: {str(e)}"
            
    async def generate_vault_insights(self) -> Dict[str, Any]:
        """Generate AI-powered insights about the vault"""
        vault_data = self.scan_vault()
        
        insights = {
            "generated_at": datetime.now().isoformat(),
            "vault_overview": {},
            "content_themes": [],
            "recommendations": [],
            "anomalies": []
        }
        
        # Generate overview with LLM
        overview_prompt = f"""
        Analyze this vault statistics and provide insights:
        - Total files: {vault_data['statistics']['total_files']}
        - Total words: {vault_data['statistics']['total_words']}
        - File types: {json.dumps(vault_data['statistics']['file_type_distribution'])}
        - Top tags: {', '.join([f"#{tag}" for tag in list(vault_data['tags'].keys())[:10]])}
        
        Provide a brief analysis of the vault's content organization and usage patterns.
        """
        
        try:
            response = await self.model_manager.query_model("general_qa", overview_prompt)
            insights['vault_overview'] = response.content
            
            # Identify content themes using embeddings
            if hasattr(self.embedding_adapter, 'indices') and 'vault_content' in self.embedding_adapter.indices:
                # Find clusters of similar content
                duplicates = await self.embedding_adapter.find_duplicates("vault_content", threshold=0.85)
                
                if duplicates:
                    insights['anomalies'].append({
                        'type': 'potential_duplicates',
                        'count': len(duplicates),
                        'examples': [
                            {
                                'file1': dup[0].id,
                                'file2': dup[1].id,
                                'similarity': dup[2]
                            }
                            for dup in duplicates[:3]
                        ]
                    })
                    
            # Generate recommendations
            if vault_data['statistics']['avg_words_per_file'] > 2000:
                insights['recommendations'].append(
                    "Consider breaking down large documents into smaller, focused notes"
                )
                
            if len(vault_data['tags']) < vault_data['statistics']['total_files'] * 0.5:
                insights['recommendations'].append(
                    "Increase tag usage for better organization and discoverability"
                )
                
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            insights['error'] = str(e)
            return insights

# Convenience functions for backward compatibility
def create_llm_vault_query_system(vault_path: str) -> VaultQuerySystemLLM:
    """Create an LLM-enhanced vault query system"""
    system = VaultQuerySystemLLM(vault_path)
    
    # Try to initialize LLM components
    try:
        import asyncio
        initialized = asyncio.run(system.initialize())
        if initialized:
            logger.info("LLM-enhanced vault query system ready")
    except Exception as e:
        logger.error(f"LLM initialization failed: {e}")
        print("\n❌ AI Model Initialization Failed")
        print("=" * 50)
        print("This system requires AI models to function properly.")
        print("\nPlease ensure Ollama is running:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Start service: ollama serve")
        print("3. Pull models: ollama pull llama2")
        print("\nPattern matching is available as a fallback but must")
        print("be explicitly requested using:")
        print("  system.use_pattern_matching_fallback(query)")
        print("=" * 50)
        
    return system

# Example usage
async def demo_llm_query_system():
    """Demonstrate LLM-enhanced query system"""
    print("🤖 LLM-Enhanced Vault Query System Demo")
    
    # Create system
    vault_path = "/Users/nvaldez/Documents/repos/Obsidian"
    system = VaultQuerySystemLLM(vault_path)
    
    # Initialize
    print("\n📚 Initializing LLM components...")
    initialized = await system.initialize()
    
    if not initialized:
        print("\n❌ LLM components not available")
        print("This system is AI-first and requires LLM models to function.")
        print("\nTo test pattern matching fallback (limited functionality):")
        print("  result = system.use_pattern_matching_fallback('your query')")
        return
        
    print("✅ LLM components initialized")
    
    # Test queries
    test_queries = [
        "What are the main themes in my vault?",
        "Summarize my recent meeting notes",
        "Find all documents related to machine learning",
        "How is my vault organized?",
        "What projects am I currently working on?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        result = await system.query_llm(query)
        
        print(f"Intent: {result.get('intent', 'unknown')}")
        print(f"Models used: {', '.join(result.get('models_used', []))}")
        print(f"Response: {result.get('response', 'No response')[:200]}...")
        
        # Simulate user rating
        rating = 4  # Good response
        await system.rate_response(result['query_id'], rating)
        
    # Generate insights
    print("\n🔍 Generating vault insights...")
    insights = await system.generate_vault_insights()
    print(f"Overview: {insights.get('vault_overview', 'No overview')[:200]}...")
    
    # Get improvement report
    print("\n📊 Model Performance Report:")
    report = await system.get_improvement_report()
    print(report[:500] + "..." if len(report) > 500 else report)

if __name__ == "__main__":
    asyncio.run(demo_llm_query_system())