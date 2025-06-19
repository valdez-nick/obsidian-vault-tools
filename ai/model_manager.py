#!/usr/bin/env python3
"""
LLM Model Manager
Manages Ollama and custom model integration for the vault query system
"""

import os
import json
import re
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Handle optional dependencies with graceful fallback
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not available - using JSON fallback for config files")
    # Fallback for basic YAML operations - use JSON instead
    class yaml:
        @staticmethod
        def safe_load(content):
            try:
                if content and isinstance(content, str):
                    # Try to parse as JSON first
                    return json.loads(content)
                else:
                    return content if content else {}
            except json.JSONDecodeError:
                logger.error("Cannot parse config without PyYAML - using empty config")
                return {}
        @staticmethod
        def safe_dump(data, *args, **kwargs):
            return json.dumps(data, indent=2)

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    # This will cause graceful degradation - models won't work without aiohttp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """Standard response from any model"""
    content: str
    model_name: str
    confidence: float
    metadata: Dict[str, Any]
    elapsed_time: float

class ModelProvider(ABC):
    """Abstract base class for model providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from model"""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if provider is available"""
        pass

class OllamaProvider(ModelProvider):
    """Ollama model provider implementation"""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        self.base_url = base_url
        self.timeout = timeout
        self.session = None
        
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available - Ollama provider will not work")
            raise ImportError("aiohttp is required for Ollama provider")
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def generate(self, prompt: str, model: str = "llama2", 
                      temperature: float = 0.7, max_tokens: int = 500, **kwargs) -> ModelResponse:
        """Generate response using Ollama API"""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
            
        start_time = datetime.now()
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "options": {
                        "num_predict": max_tokens,
                        **kwargs
                    }
                }
            ) as response:
                if response.status != 200:
                    raise Exception(f"Ollama API error: {response.status}")
                    
                # Collect streaming response
                full_response = ""
                async for line in response.content:
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            full_response += data["response"]
                            
                elapsed = (datetime.now() - start_time).total_seconds()
                
                return ModelResponse(
                    content=full_response.strip(),
                    model_name=model,
                    confidence=0.8,  # Default confidence
                    metadata={"temperature": temperature, "max_tokens": max_tokens},
                    elapsed_time=elapsed
                )
                
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise
            
    async def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
        except:
            return False
            
    async def list_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model["name"] for model in data.get("models", [])]
        except:
            return []

class LLMModelManager:
    """Manages multiple LLM models and ensemble strategies"""
    
    def __init__(self, config_path: str = "model_config.yaml"):
        self.config_path = Path(config_path)
        self.user_config_path = Path.home() / ".obsidian_ai_config"
        self.defaults_path = Path("model_config_defaults.yaml")
        self.config = self._load_config()
        self.providers = {}
        self.models = {}
        self.cache = {}
        self.available_models = []
        self.detected_model = None
        self._initialize_providers()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML with auto-detection"""
        # Load defaults first
        config = self._load_defaults()
        
        # Load user preferences
        user_prefs = self._load_user_preferences()
        
        # Load custom config if exists
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                self._merge_config(config, custom_config)
        
        # Apply environment variables
        config = self._apply_env_vars(config)
        
        # Auto-detect models if needed (will be done during initialization)
        self._auto_detect_on_init = user_prefs.get("auto_detect", True)
        
        return config
        
    def _load_defaults(self) -> Dict[str, Any]:
        """Load default configuration"""
        if self.defaults_path.exists():
            with open(self.defaults_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Defaults file {self.defaults_path} not found, using fallback")
            return self._default_config()
            
    def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences from home directory"""
        if self.user_config_path.exists():
            try:
                with open(self.user_config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load user preferences: {e}")
        return {"auto_detect": True, "first_run": True}
        
    def _save_user_preferences(self, prefs: Dict[str, Any]):
        """Save user preferences to home directory"""
        try:
            with open(self.user_config_path, 'w') as f:
                json.dump(prefs, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save user preferences: {e}")
            
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Merge configuration dictionaries"""
        for key, value in override.items():
            if isinstance(value, dict) and key in base:
                self._merge_config(base[key], value)
            else:
                base[key] = value
                
    def _apply_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable substitutions"""
        def substitute_env(obj):
            if isinstance(obj, str):
                # Handle ${VAR:-default} syntax
                pattern = r'\$\{([^}]+)\}'
                def replace_env(match):
                    env_expr = match.group(1)
                    if ':-' in env_expr:
                        var_name, default_value = env_expr.split(':-', 1)
                        return os.getenv(var_name, default_value)
                    else:
                        return os.getenv(env_expr, match.group(0))
                return re.sub(pattern, replace_env, obj)
            elif isinstance(obj, dict):
                return {k: substitute_env(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_env(item) for item in obj]
            return obj
        
        return substitute_env(config)
            
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if file not found"""
        return {
            "providers": {
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "timeout": 300
                }
            },
            "models": {
                "general_qa": {
                    "provider": "ollama",
                    "model": "llama2",
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            },
            "ensemble": {
                "strategies": {
                    "voting": {"enabled": True}
                }
            }
        }
        
    def _initialize_providers(self):
        """Initialize model providers"""
        provider_config = self.config.get("providers", {})
        
        # Initialize Ollama provider
        if "ollama" in provider_config:
            ollama_config = provider_config["ollama"]
            self.providers["ollama"] = OllamaProvider(
                base_url=ollama_config.get("base_url", "http://localhost:11434"),
                timeout=ollama_config.get("timeout", 300)
            )
            
    async def initialize_async(self):
        """Perform async initialization tasks"""
        if hasattr(self, '_auto_detect_on_init') and self._auto_detect_on_init:
            await self._auto_configure_models(self.config)
            
    async def auto_detect_models(self) -> List[str]:
        """Auto-detect available Ollama models"""
        if "ollama" not in self.providers:
            return []
            
        try:
            provider = self.providers["ollama"]
            models = await provider.list_models()
            self.available_models = models
            logger.info(f"Auto-detected {len(models)} Ollama models: {models}")
            return models
        except Exception as e:
            logger.warning(f"Could not auto-detect models: {e}")
            return []
            
    async def _auto_configure_models(self, config: Dict[str, Any]):
        """Auto-configure models based on what's available"""
        # Detect available models
        available = await self.auto_detect_models()
        
        if not available:
            logger.warning("No models detected - cannot auto-configure")
            return
            
        # Get model preferences
        preferences = config.get("model_preferences", {})
        priority = preferences.get("priority", ["dolphin3:latest", "llama3.2:latest", "mistral:latest"])
        
        # Find the best available model
        best_model = None
        for preferred in priority:
            # Check exact match or partial match
            for available_model in available:
                if (available_model == preferred or 
                    available_model.startswith(preferred.split(":")[0]) or
                    preferred.startswith(available_model.split(":")[0])):
                    best_model = available_model
                    break
            if best_model:
                break
                
        if not best_model:
            best_model = available[0]  # Use first available as fallback
            
        self.detected_model = best_model
        logger.info(f"Auto-selected model: {best_model}")
        
        # Update config to use the detected model
        for model_key, model_config in config.get("models", {}).items():
            if model_config.get("model") == "auto":
                model_config["model"] = best_model
                logger.info(f"Configured {model_key} to use {best_model}")
                
        # Configure ensemble for single model if only one is available
        if len(available) == 1 and config.get("ensemble", {}).get("auto_single_model", True):
            config["ensemble"]["strategies"] = {"single": {"enabled": True}}
            logger.info("Configured for single model mode")
            
        # Update fallback chain
        routing = config.get("routing", {})
        routing["fallback_chain"] = available[:3]  # Use top 3 available models
        
    def get_detected_model(self) -> Optional[str]:
        """Get the auto-detected primary model"""
        return self.detected_model
        
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.available_models
        
    async def select_model_interactive(self) -> Optional[str]:
        """Interactive model selection for first-time setup"""
        available = await self.auto_detect_models()
        
        if not available:
            print("❌ No Ollama models found. Please install and pull a model first.")
            print("Example: ollama pull dolphin3")
            return None
            
        print("\n🤖 Available AI Models:")
        for i, model in enumerate(available, 1):
            print(f"  {i}. {model}")
            
        print("\nRecommended model for your setup:")
        # Check for dolphin3 specifically
        dolphin_models = [m for m in available if "dolphin" in m.lower()]
        if dolphin_models:
            recommended = dolphin_models[0]
            print(f"  🐬 {recommended} (Dolphin - Excellent all-around performance)")
        else:
            llama_models = [m for m in available if "llama" in m.lower()]
            if llama_models:
                recommended = llama_models[0]
                print(f"  🦙 {recommended} (LLaMA - Strong general capabilities)")
            else:
                recommended = available[0]
                print(f"  🤖 {recommended} (Available model)")
                
        choice = input(f"\nSelect model (1-{len(available)}) or press Enter for recommended: ").strip()
        
        if not choice:
            selected_model = recommended
        else:
            try:
                index = int(choice) - 1
                if 0 <= index < len(available):
                    selected_model = available[index]
                else:
                    print("Invalid selection, using recommended model")
                    selected_model = recommended
            except ValueError:
                print("Invalid input, using recommended model")
                selected_model = recommended
                
        # Save user preference
        prefs = self._load_user_preferences()
        prefs.update({
            "preferred_model": selected_model,
            "auto_detect": False,
            "first_run": False
        })
        self._save_user_preferences(prefs)
        
        print(f"✅ Selected model: {selected_model}")
        return selected_model
            
    async def query_model(self, model_key: str, prompt: str, **kwargs) -> ModelResponse:
        """Query a specific model"""
        if model_key not in self.config.get("models", {}):
            raise ValueError(f"Model {model_key} not found in configuration")
            
        model_config = self.config["models"][model_key]
        provider_name = model_config["provider"]
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not initialized")
            
        provider = self.providers[provider_name]
        
        # Merge config with kwargs
        generate_kwargs = {
            "model": model_config.get("model", "llama2"),
            "temperature": model_config.get("temperature", 0.7),
            "max_tokens": model_config.get("max_tokens", 500),
            **kwargs
        }
        
        return await provider.generate(prompt, **generate_kwargs)
        
    async def ensemble_query(self, prompt: str, models: List[str] = None, 
                           strategy: str = "voting") -> Dict[str, Any]:
        """Query multiple models and aggregate results"""
        if models is None:
            # Use all configured models
            models = list(self.config.get("models", {}).keys())
            
        # Query all models concurrently
        tasks = [self.query_model(model_key, prompt) for model_key in models]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        valid_responses = [r for r in responses if isinstance(r, ModelResponse)]
        
        # Apply ensemble strategy
        if strategy == "voting":
            return self._voting_ensemble(valid_responses)
        elif strategy == "weighted_confidence":
            return self._weighted_ensemble(valid_responses)
        elif strategy == "cascade":
            return self._cascade_ensemble(valid_responses)
        else:
            # Default to first valid response
            return {
                "response": valid_responses[0].content if valid_responses else "",
                "models_used": [r.model_name for r in valid_responses],
                "strategy": "first_valid"
            }
            
    def _voting_ensemble(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """Simple voting ensemble - most common response wins"""
        if not responses:
            return {"response": "", "models_used": [], "strategy": "voting"}
            
        # For now, return the response with highest confidence
        best_response = max(responses, key=lambda r: r.confidence)
        
        return {
            "response": best_response.content,
            "models_used": [r.model_name for r in responses],
            "confidence": best_response.confidence,
            "strategy": "voting"
        }
        
    def _weighted_ensemble(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """Weighted ensemble based on confidence scores"""
        if not responses:
            return {"response": "", "models_used": [], "strategy": "weighted_confidence"}
            
        # For text generation, we'll return the highest confidence response
        # In future, could implement actual weighted averaging for embeddings
        best_response = max(responses, key=lambda r: r.confidence)
        
        return {
            "response": best_response.content,
            "models_used": [r.model_name for r in responses],
            "confidence": best_response.confidence,
            "strategy": "weighted_confidence"
        }
        
    def _cascade_ensemble(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """Cascade ensemble - use simpler models first, escalate if needed"""
        if not responses:
            return {"response": "", "models_used": [], "strategy": "cascade"}
            
        # Sort by model complexity (approximated by response time)
        sorted_responses = sorted(responses, key=lambda r: r.elapsed_time)
        
        # Use confidence threshold from config
        threshold = self.config.get("ensemble", {}).get("strategies", {}).get(
            "cascade", {}
        ).get("escalation_threshold", 0.5)
        
        for response in sorted_responses:
            if response.confidence >= threshold:
                return {
                    "response": response.content,
                    "models_used": [response.model_name],
                    "confidence": response.confidence,
                    "strategy": "cascade"
                }
                
        # If no response meets threshold, return best one
        best_response = max(responses, key=lambda r: r.confidence)
        return {
            "response": best_response.content,
            "models_used": [r.model_name for r in responses],
            "confidence": best_response.confidence,
            "strategy": "cascade"
        }
        
    async def classify_intent(self, query: str) -> str:
        """Classify the intent of a query"""
        if "intent_classifier" not in self.config.get("models", {}):
            # Fallback to pattern matching
            return self._pattern_based_classification(query)
            
        prompt = f"""Classify the following query into one of these categories:
        - search: Looking for specific information
        - count: Counting or statistics
        - summarize: Summarization request
        - analyze: Deep analysis request
        - extract: Entity or information extraction
        
        Query: {query}
        
        Category:"""
        
        response = await self.query_model("intent_classifier", prompt)
        intent = response.content.strip().lower()
        
        # Validate intent
        valid_intents = ["search", "count", "summarize", "analyze", "extract"]
        if any(i in intent for i in valid_intents):
            for i in valid_intents:
                if i in intent:
                    return i
                    
        return "search"  # Default
        
    def _pattern_based_classification(self, query: str) -> str:
        """Fallback pattern-based classification"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how many', 'count', 'number of']):
            return "count"
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview']):
            return "summarize"
        elif any(word in query_lower for word in ['analyze', 'analysis', 'deep dive']):
            return "analyze"
        elif any(word in query_lower for word in ['who', 'what', 'extract', 'find all']):
            return "extract"
        else:
            return "search"
            
    async def check_providers(self) -> Dict[str, bool]:
        """Check availability of all providers"""
        status = {}
        for name, provider in self.providers.items():
            status[name] = await provider.is_available()
        return status

# Example usage and testing
async def test_model_manager():
    """Test the model manager"""
    manager = LLMModelManager()
    
    # Check provider availability
    print("🔍 Checking providers...")
    status = await manager.check_providers()
    print(f"Provider status: {status}")
    
    if status.get("ollama", False):
        # Test single model query
        print("\n📝 Testing single model query...")
        response = await manager.query_model(
            "general_qa",
            "What is the capital of France?"
        )
        print(f"Response: {response.content[:100]}...")
        print(f"Model: {response.model_name}")
        print(f"Time: {response.elapsed_time:.2f}s")
        
        # Test intent classification
        print("\n🏷️ Testing intent classification...")
        queries = [
            "How many files are in my vault?",
            "Summarize the meeting notes from last week",
            "Find all mentions of project Alpha"
        ]
        
        for query in queries:
            intent = await manager.classify_intent(query)
            print(f"Query: {query}")
            print(f"Intent: {intent}")
            
        # Test ensemble query
        print("\n🎭 Testing ensemble query...")
        result = await manager.ensemble_query(
            "What are the key features of Python?",
            strategy="voting"
        )
        print(f"Ensemble response: {result['response'][:100]}...")
        print(f"Models used: {result['models_used']}")
        print(f"Strategy: {result['strategy']}")
    else:
        print("⚠️ Ollama not available. Please ensure Ollama is running.")
        print("Run: ollama serve")

if __name__ == "__main__":
    asyncio.run(test_model_manager())