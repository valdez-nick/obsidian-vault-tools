"""
AI model management and configuration.

Handles different AI providers (OpenAI, Anthropic, local models) with
unified interfaces and automatic fallbacks.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class AIProvider(str, Enum):
    """Available AI providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class ModelCapability(str, Enum):
    """Model capabilities."""
    TEXT_GENERATION = "text_generation"
    EMBEDDING = "embedding"
    CHAT = "chat"
    COMPLETION = "completion"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"


@dataclass
class AIConfig:
    """Configuration for AI services."""
    
    # Provider settings
    primary_provider: AIProvider = AIProvider.OPENAI
    fallback_providers: List[AIProvider] = field(default_factory=lambda: [AIProvider.OLLAMA])
    
    # Model configurations
    text_model: str = "gpt-4"
    embedding_model: str = "text-embedding-ada-002"
    fast_model: str = "gpt-3.5-turbo"  # For quick operations
    
    # API settings
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    anthropic_base_url: Optional[str] = None
    
    # Local model settings
    ollama_base_url: str = "http://localhost:11434"
    local_model_path: Optional[Path] = None
    
    # Generation settings
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Performance settings
    request_timeout: int = 60
    max_retries: int = 3
    rate_limit_per_minute: int = 60
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    # Cost management
    max_tokens_per_day: Optional[int] = None
    enable_cost_tracking: bool = True
    cost_alert_threshold: float = 10.0  # USD
    
    def __post_init__(self):
        """Set API keys from environment if not provided."""
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


class ModelInfo(BaseModel):
    """Information about an AI model."""
    name: str
    provider: AIProvider
    capabilities: List[ModelCapability]
    context_length: int
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    embedding_dimensions: Optional[int] = None
    is_available: bool = True
    last_checked: Optional[str] = None


class AIModelManager:
    """Manages AI models across different providers."""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self._models: Dict[str, ModelInfo] = {}
        self._clients: Dict[AIProvider, Any] = {}
        self._rate_limiters: Dict[AIProvider, asyncio.Semaphore] = {}
        self._cost_tracker = CostTracker(config)
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize AI providers and discover available models."""
        logger.info("Initializing AI model manager")
        
        # Initialize providers
        await self._init_openai()
        await self._init_anthropic()
        await self._init_ollama()
        
        # Discover models
        await self._discover_models()
        
        logger.info("AI model manager initialized", models=len(self._models))
    
    async def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        if not self.config.openai_api_key:
            logger.warning("OpenAI API key not provided")
            return
        
        try:
            import openai
            
            client = openai.AsyncOpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
                timeout=self.config.request_timeout,
                max_retries=self.config.max_retries,
            )
            
            # Test connection
            await client.models.list()
            
            self._clients[AIProvider.OPENAI] = client
            self._rate_limiters[AIProvider.OPENAI] = asyncio.Semaphore(
                self.config.rate_limit_per_minute
            )
            
            logger.info("OpenAI client initialized")
            
        except Exception as e:
            logger.warning("Failed to initialize OpenAI", error=str(e))
    
    async def _init_anthropic(self) -> None:
        """Initialize Anthropic client."""
        if not self.config.anthropic_api_key:
            logger.warning("Anthropic API key not provided")
            return
        
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(
                api_key=self.config.anthropic_api_key,
                base_url=self.config.anthropic_base_url,
                timeout=self.config.request_timeout,
                max_retries=self.config.max_retries,
            )
            
            # Test connection with a simple request
            await client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}]
            )
            
            self._clients[AIProvider.ANTHROPIC] = client
            self._rate_limiters[AIProvider.ANTHROPIC] = asyncio.Semaphore(
                self.config.rate_limit_per_minute
            )
            
            logger.info("Anthropic client initialized")
            
        except Exception as e:
            logger.warning("Failed to initialize Anthropic", error=str(e))
    
    async def _init_ollama(self) -> None:
        """Initialize Ollama client."""
        try:
            import aiohttp
            
            # Test Ollama connection
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config.ollama_base_url}/api/tags") as response:
                    if response.status == 200:
                        self._clients[AIProvider.OLLAMA] = session
                        self._rate_limiters[AIProvider.OLLAMA] = asyncio.Semaphore(10)
                        logger.info("Ollama client initialized")
                    else:
                        logger.warning("Ollama not available", status=response.status)
                        
        except Exception as e:
            logger.warning("Failed to initialize Ollama", error=str(e))
    
    async def _discover_models(self) -> None:
        """Discover available models from all providers."""
        # OpenAI models
        if AIProvider.OPENAI in self._clients:
            await self._discover_openai_models()
        
        # Anthropic models
        if AIProvider.ANTHROPIC in self._clients:
            await self._discover_anthropic_models()
        
        # Ollama models
        if AIProvider.OLLAMA in self._clients:
            await self._discover_ollama_models()
    
    async def _discover_openai_models(self) -> None:
        """Discover OpenAI models."""
        try:
            client = self._clients[AIProvider.OPENAI]
            models = await client.models.list()
            
            for model in models.data:
                # Categorize models by capabilities
                capabilities = []
                if "gpt" in model.id:
                    capabilities.extend([ModelCapability.TEXT_GENERATION, ModelCapability.CHAT])
                if "embedding" in model.id:
                    capabilities.append(ModelCapability.EMBEDDING)
                if "davinci" in model.id or "curie" in model.id:
                    capabilities.append(ModelCapability.COMPLETION)
                
                # Set embedding dimensions
                embedding_dims = None
                if "ada-002" in model.id:
                    embedding_dims = 1536
                elif "embedding-3-small" in model.id:
                    embedding_dims = 1536
                elif "embedding-3-large" in model.id:
                    embedding_dims = 3072
                
                # Estimate costs (approximate)
                input_cost = 0.0
                output_cost = 0.0
                if "gpt-4" in model.id:
                    input_cost = 0.03
                    output_cost = 0.06
                elif "gpt-3.5" in model.id:
                    input_cost = 0.001
                    output_cost = 0.002
                elif "embedding" in model.id:
                    input_cost = 0.0001
                
                self._models[model.id] = ModelInfo(
                    name=model.id,
                    provider=AIProvider.OPENAI,
                    capabilities=capabilities,
                    context_length=self._get_context_length(model.id),
                    input_cost_per_1k=input_cost,
                    output_cost_per_1k=output_cost,
                    embedding_dimensions=embedding_dims,
                )
                
        except Exception as e:
            logger.error("Failed to discover OpenAI models", error=str(e))
    
    async def _discover_anthropic_models(self) -> None:
        """Discover Anthropic models."""
        # Anthropic doesn't have a model list API, so we'll hardcode known models
        anthropic_models = [
            {
                "name": "claude-3-opus-20240229",
                "context_length": 200000,
                "input_cost": 0.015,
                "output_cost": 0.075,
            },
            {
                "name": "claude-3-sonnet-20240229", 
                "context_length": 200000,
                "input_cost": 0.003,
                "output_cost": 0.015,
            },
            {
                "name": "claude-3-haiku-20240307",
                "context_length": 200000,
                "input_cost": 0.00025,
                "output_cost": 0.00125,
            },
        ]
        
        for model_info in anthropic_models:
            self._models[model_info["name"]] = ModelInfo(
                name=model_info["name"],
                provider=AIProvider.ANTHROPIC,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT],
                context_length=model_info["context_length"],
                input_cost_per_1k=model_info["input_cost"],
                output_cost_per_1k=model_info["output_cost"],
            )
    
    async def _discover_ollama_models(self) -> None:
        """Discover Ollama models."""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config.ollama_base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for model in data.get("models", []):
                            model_name = model["name"]
                            
                            # Determine capabilities based on model name
                            capabilities = [ModelCapability.TEXT_GENERATION, ModelCapability.CHAT]
                            if "embed" in model_name.lower():
                                capabilities = [ModelCapability.EMBEDDING]
                            
                            self._models[model_name] = ModelInfo(
                                name=model_name,
                                provider=AIProvider.OLLAMA,
                                capabilities=capabilities,
                                context_length=4096,  # Default, varies by model
                                input_cost_per_1k=0.0,  # Local models are free
                                output_cost_per_1k=0.0,
                                embedding_dimensions=384 if "embed" in model_name.lower() else None,
                            )
                            
        except Exception as e:
            logger.error("Failed to discover Ollama models", error=str(e))
    
    def _get_context_length(self, model_name: str) -> int:
        """Get context length for a model."""
        context_lengths = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "text-davinci-003": 4096,
            "text-embedding-ada-002": 8191,
            "text-embedding-3-small": 8191,
            "text-embedding-3-large": 8191,
        }
        
        for key, length in context_lengths.items():
            if key in model_name:
                return length
        
        return 4096  # Default
    
    async def get_available_models(
        self, 
        capability: Optional[ModelCapability] = None,
        provider: Optional[AIProvider] = None
    ) -> List[ModelInfo]:
        """Get available models with optional filtering."""
        models = list(self._models.values())
        
        if capability:
            models = [m for m in models if capability in m.capabilities]
        
        if provider:
            models = [m for m in models if m.provider == provider]
        
        # Filter by availability
        models = [m for m in models if m.is_available]
        
        return models
    
    async def get_best_model(
        self, 
        capability: ModelCapability,
        prefer_fast: bool = False,
        max_cost: Optional[float] = None
    ) -> Optional[ModelInfo]:
        """Get the best model for a capability with cost/speed preferences."""
        models = await self.get_available_models(capability)
        
        if not models:
            return None
        
        # Apply cost filter
        if max_cost is not None:
            models = [m for m in models if m.input_cost_per_1k <= max_cost]
        
        # Sort by preference
        if prefer_fast:
            # Prefer faster/cheaper models
            models.sort(key=lambda m: (m.input_cost_per_1k, m.context_length))
        else:
            # Prefer quality models (reverse cost order for capabilities)
            if capability == ModelCapability.TEXT_GENERATION:
                models.sort(key=lambda m: -m.input_cost_per_1k)
            else:
                models.sort(key=lambda m: m.input_cost_per_1k)
        
        return models[0] if models else None
    
    async def check_model_availability(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        if model_name not in self._models:
            return False
        
        model = self._models[model_name]
        
        # Quick availability check based on provider
        if model.provider not in self._clients:
            return False
        
        # Could add more sophisticated health checks here
        return True
    
    async def get_cost_estimate(
        self, 
        model_name: str, 
        input_tokens: int, 
        output_tokens: int = 0
    ) -> float:
        """Estimate cost for a model operation."""
        if model_name not in self._models:
            return 0.0
        
        model = self._models[model_name]
        
        input_cost = (input_tokens / 1000) * model.input_cost_per_1k
        output_cost = (output_tokens / 1000) * model.output_cost_per_1k
        
        return input_cost + output_cost
    
    async def close(self) -> None:
        """Close all AI clients."""
        logger.info("Closing AI model manager")
        
        for provider, client in self._clients.items():
            try:
                if hasattr(client, 'close'):
                    await client.close()
            except Exception as e:
                logger.warning("Error closing AI client", provider=provider, error=str(e))
        
        self._clients.clear()
    
    def get_client(self, provider: AIProvider) -> Optional[Any]:
        """Get client for a specific provider."""
        return self._clients.get(provider)
    
    def get_rate_limiter(self, provider: AIProvider) -> Optional[asyncio.Semaphore]:
        """Get rate limiter for a provider."""
        return self._rate_limiters.get(provider)


class CostTracker:
    """Tracks AI usage costs."""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.daily_costs: Dict[str, float] = {}
        self.total_cost = 0.0
        self._lock = asyncio.Lock()
    
    async def record_usage(
        self, 
        model_name: str, 
        input_tokens: int, 
        output_tokens: int = 0
    ) -> float:
        """Record usage and return cost."""
        # Implementation would track costs per day/model
        # For now, return 0
        return 0.0
    
    async def get_daily_cost(self, date: str = None) -> float:
        """Get cost for a specific day."""
        date = date or self._get_today()
        return self.daily_costs.get(date, 0.0)
    
    async def check_limits(self) -> bool:
        """Check if usage is within limits."""
        if not self.config.max_tokens_per_day:
            return True
        
        # Implementation would check actual token usage
        return True
    
    def _get_today(self) -> str:
        """Get today's date string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")