"""
Language model service for text generation and chat.

Provides unified interface for text generation across multiple providers
with automatic fallbacks and cost optimization.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
from dataclasses import dataclass

import structlog

from .models import AIModelManager, AIProvider, ModelCapability, AIConfig
from ..database.base import DatabaseManager

logger = structlog.get_logger(__name__)


@dataclass
class GenerationRequest:
    """Request for text generation."""
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    stream: bool = False
    system_prompt: Optional[str] = None


@dataclass
class GenerationResponse:
    """Response from text generation."""
    text: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    cost: float = 0.0


@dataclass
class ChatMessage:
    """Chat message."""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None


@dataclass
class ChatRequest:
    """Request for chat completion."""
    messages: List[ChatMessage]
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    stream: bool = False


class LanguageModelService:
    """Service for language model operations."""
    
    def __init__(
        self,
        ai_manager: AIModelManager,
        database_manager: Optional[DatabaseManager] = None
    ):
        self.ai_manager = ai_manager
        self.db_manager = database_manager
        self._generation_cache: Dict[str, GenerationResponse] = {}
        self._lock = asyncio.Lock()
    
    async def generate_text(
        self, 
        request: GenerationRequest,
        use_cache: bool = True
    ) -> GenerationResponse:
        """Generate text from a prompt."""
        # Create cache key
        if use_cache:
            cache_key = self._create_cache_key(request)
            cached = await self._get_cached_response(cache_key)
            if cached:
                return cached
        
        # Select model
        model = await self._select_model(request.model, ModelCapability.TEXT_GENERATION)
        
        # Generate text
        response = await self._generate_with_model(request, model)
        
        # Cache response
        if use_cache and response:
            await self._cache_response(cache_key, response)
        
        return response
    
    async def generate_stream(
        self, 
        request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming response."""
        model = await self._select_model(request.model, ModelCapability.TEXT_GENERATION)
        
        async for chunk in self._generate_stream_with_model(request, model):
            yield chunk
    
    async def chat_completion(
        self,
        request: ChatRequest,
        use_cache: bool = True
    ) -> GenerationResponse:
        """Generate chat completion."""
        # Create cache key
        if use_cache:
            cache_key = self._create_chat_cache_key(request)
            cached = await self._get_cached_response(cache_key)
            if cached:
                return cached
        
        # Select model
        model = await self._select_model(request.model, ModelCapability.CHAT)
        
        # Generate response
        response = await self._chat_with_model(request, model)
        
        # Cache response
        if use_cache and response:
            await self._cache_response(cache_key, response)
        
        return response
    
    async def chat_stream(
        self,
        request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion."""
        model = await self._select_model(request.model, ModelCapability.CHAT)
        
        async for chunk in self._chat_stream_with_model(request, model):
            yield chunk
    
    async def _select_model(
        self, 
        requested_model: Optional[str], 
        capability: ModelCapability
    ) -> str:
        """Select the best available model."""
        if requested_model:
            # Check if requested model is available
            if await self.ai_manager.check_model_availability(requested_model):
                return requested_model
            else:
                logger.warning("Requested model not available", model=requested_model)
        
        # Get best available model
        model_info = await self.ai_manager.get_best_model(capability)
        if not model_info:
            raise RuntimeError(f"No models available for {capability}")
        
        return model_info.name
    
    async def _generate_with_model(
        self, 
        request: GenerationRequest, 
        model: str
    ) -> GenerationResponse:
        """Generate text with a specific model."""
        model_info = self.ai_manager._models[model]
        provider = model_info.provider
        
        if provider == AIProvider.OPENAI:
            return await self._generate_openai(request, model)
        elif provider == AIProvider.ANTHROPIC:
            return await self._generate_anthropic(request, model)
        elif provider == AIProvider.OLLAMA:
            return await self._generate_ollama(request, model)
        else:
            raise ValueError(f"Unsupported provider for generation: {provider}")
    
    async def _generate_stream_with_model(
        self,
        request: GenerationRequest,
        model: str
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text with a specific model."""
        model_info = self.ai_manager._models[model]
        provider = model_info.provider
        
        if provider == AIProvider.OPENAI:
            async for chunk in self._generate_openai_stream(request, model):
                yield chunk
        elif provider == AIProvider.ANTHROPIC:
            async for chunk in self._generate_anthropic_stream(request, model):
                yield chunk
        elif provider == AIProvider.OLLAMA:
            async for chunk in self._generate_ollama_stream(request, model):
                yield chunk
        else:
            raise ValueError(f"Unsupported provider for streaming: {provider}")
    
    async def _chat_with_model(
        self,
        request: ChatRequest,
        model: str
    ) -> GenerationResponse:
        """Generate chat completion with a specific model."""
        model_info = self.ai_manager._models[model]
        provider = model_info.provider
        
        if provider == AIProvider.OPENAI:
            return await self._chat_openai(request, model)
        elif provider == AIProvider.ANTHROPIC:
            return await self._chat_anthropic(request, model)
        elif provider == AIProvider.OLLAMA:
            return await self._chat_ollama(request, model)
        else:
            raise ValueError(f"Unsupported provider for chat: {provider}")
    
    async def _chat_stream_with_model(
        self,
        request: ChatRequest,
        model: str
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion with a specific model."""
        model_info = self.ai_manager._models[model]
        provider = model_info.provider
        
        if provider == AIProvider.OPENAI:
            async for chunk in self._chat_openai_stream(request, model):
                yield chunk
        elif provider == AIProvider.ANTHROPIC:
            async for chunk in self._chat_anthropic_stream(request, model):
                yield chunk
        elif provider == AIProvider.OLLAMA:
            async for chunk in self._chat_ollama_stream(request, model):
                yield chunk
        else:
            raise ValueError(f"Unsupported provider for streaming chat: {provider}")
    
    # OpenAI implementations
    
    async def _generate_openai(
        self, 
        request: GenerationRequest, 
        model: str
    ) -> GenerationResponse:
        """Generate text using OpenAI."""
        client = self.ai_manager.get_client(AIProvider.OPENAI)
        if not client:
            raise RuntimeError("OpenAI client not available")
        
        try:
            rate_limiter = self.ai_manager.get_rate_limiter(AIProvider.OPENAI)
            async with rate_limiter:
                response = await client.completions.create(
                    model=model,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens or self.ai_manager.config.max_tokens,
                    temperature=request.temperature or self.ai_manager.config.temperature,
                    top_p=request.top_p or self.ai_manager.config.top_p,
                    stop=request.stop,
                    stream=False,
                )
                
                usage = response.usage
                cost = await self.ai_manager.get_cost_estimate(
                    model, usage.prompt_tokens, usage.completion_tokens
                )
                
                return GenerationResponse(
                    text=response.choices[0].text,
                    model=model,
                    usage={
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                    finish_reason=response.choices[0].finish_reason,
                    cost=cost,
                )
                
        except Exception as e:
            logger.error("OpenAI generation failed", model=model, error=str(e))
            raise
    
    async def _chat_openai(
        self,
        request: ChatRequest,
        model: str
    ) -> GenerationResponse:
        """Generate chat completion using OpenAI."""
        client = self.ai_manager.get_client(AIProvider.OPENAI)
        if not client:
            raise RuntimeError("OpenAI client not available")
        
        try:
            # Convert messages
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            rate_limiter = self.ai_manager.get_rate_limiter(AIProvider.OPENAI)
            async with rate_limiter:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=request.max_tokens or self.ai_manager.config.max_tokens,
                    temperature=request.temperature or self.ai_manager.config.temperature,
                    top_p=request.top_p or self.ai_manager.config.top_p,
                    stop=request.stop,
                    stream=False,
                )
                
                usage = response.usage
                cost = await self.ai_manager.get_cost_estimate(
                    model, usage.prompt_tokens, usage.completion_tokens
                )
                
                return GenerationResponse(
                    text=response.choices[0].message.content,
                    model=model,
                    usage={
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                    finish_reason=response.choices[0].finish_reason,
                    cost=cost,
                )
                
        except Exception as e:
            logger.error("OpenAI chat failed", model=model, error=str(e))
            raise
    
    async def _generate_openai_stream(
        self,
        request: GenerationRequest,
        model: str
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using OpenAI."""
        client = self.ai_manager.get_client(AIProvider.OPENAI)
        if not client:
            raise RuntimeError("OpenAI client not available")
        
        try:
            rate_limiter = self.ai_manager.get_rate_limiter(AIProvider.OPENAI)
            async with rate_limiter:
                stream = await client.completions.create(
                    model=model,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens or self.ai_manager.config.max_tokens,
                    temperature=request.temperature or self.ai_manager.config.temperature,
                    top_p=request.top_p or self.ai_manager.config.top_p,
                    stop=request.stop,
                    stream=True,
                )
                
                async for chunk in stream:
                    if chunk.choices[0].text:
                        yield chunk.choices[0].text
                        
        except Exception as e:
            logger.error("OpenAI streaming failed", model=model, error=str(e))
            raise
    
    async def _chat_openai_stream(
        self,
        request: ChatRequest,
        model: str
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat using OpenAI."""
        client = self.ai_manager.get_client(AIProvider.OPENAI)
        if not client:
            raise RuntimeError("OpenAI client not available")
        
        try:
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            rate_limiter = self.ai_manager.get_rate_limiter(AIProvider.OPENAI)
            async with rate_limiter:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=request.max_tokens or self.ai_manager.config.max_tokens,
                    temperature=request.temperature or self.ai_manager.config.temperature,
                    top_p=request.top_p or self.ai_manager.config.top_p,
                    stop=request.stop,
                    stream=True,
                )
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                        
        except Exception as e:
            logger.error("OpenAI chat streaming failed", model=model, error=str(e))
            raise
    
    # Anthropic implementations
    
    async def _chat_anthropic(
        self,
        request: ChatRequest,
        model: str
    ) -> GenerationResponse:
        """Generate chat completion using Anthropic."""
        client = self.ai_manager.get_client(AIProvider.ANTHROPIC)
        if not client:
            raise RuntimeError("Anthropic client not available")
        
        try:
            # Convert messages (Anthropic format is slightly different)
            messages = []
            system_message = None
            
            for msg in request.messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            rate_limiter = self.ai_manager.get_rate_limiter(AIProvider.ANTHROPIC)
            async with rate_limiter:
                response = await client.messages.create(
                    model=model,
                    max_tokens=request.max_tokens or self.ai_manager.config.max_tokens,
                    messages=messages,
                    system=system_message,
                    temperature=request.temperature or self.ai_manager.config.temperature,
                    top_p=request.top_p or self.ai_manager.config.top_p,
                    stop_sequences=request.stop,
                    stream=False,
                )
                
                usage = response.usage
                cost = await self.ai_manager.get_cost_estimate(
                    model, usage.input_tokens, usage.output_tokens
                )
                
                return GenerationResponse(
                    text=response.content[0].text,
                    model=model,
                    usage={
                        "prompt_tokens": usage.input_tokens,
                        "completion_tokens": usage.output_tokens,
                        "total_tokens": usage.input_tokens + usage.output_tokens,
                    },
                    finish_reason=response.stop_reason,
                    cost=cost,
                )
                
        except Exception as e:
            logger.error("Anthropic chat failed", model=model, error=str(e))
            raise
    
    async def _chat_anthropic_stream(
        self,
        request: ChatRequest,
        model: str
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat using Anthropic."""
        client = self.ai_manager.get_client(AIProvider.ANTHROPIC)
        if not client:
            raise RuntimeError("Anthropic client not available")
        
        try:
            messages = []
            system_message = None
            
            for msg in request.messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            rate_limiter = self.ai_manager.get_rate_limiter(AIProvider.ANTHROPIC)
            async with rate_limiter:
                stream = await client.messages.create(
                    model=model,
                    max_tokens=request.max_tokens or self.ai_manager.config.max_tokens,
                    messages=messages,
                    system=system_message,
                    temperature=request.temperature or self.ai_manager.config.temperature,
                    top_p=request.top_p or self.ai_manager.config.top_p,
                    stop_sequences=request.stop,
                    stream=True,
                )
                
                async for chunk in stream:
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                        yield chunk.delta.text
                        
        except Exception as e:
            logger.error("Anthropic streaming failed", model=model, error=str(e))
            raise
    
    # Ollama implementations
    
    async def _generate_ollama(
        self,
        request: GenerationRequest,
        model: str
    ) -> GenerationResponse:
        """Generate text using Ollama."""
        import aiohttp
        
        try:
            url = f"{self.ai_manager.config.ollama_base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": request.prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature or self.ai_manager.config.temperature,
                    "top_p": request.top_p or self.ai_manager.config.top_p,
                    "num_predict": request.max_tokens or self.ai_manager.config.max_tokens,
                }
            }
            
            if request.stop:
                payload["options"]["stop"] = request.stop
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return GenerationResponse(
                            text=data["response"],
                            model=model,
                            usage={
                                "prompt_tokens": data.get("prompt_eval_count", 0),
                                "completion_tokens": data.get("eval_count", 0),
                                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                            },
                            finish_reason=data.get("done_reason", "completed"),
                            cost=0.0,  # Local models are free
                        )
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Ollama generation failed: {error_text}")
                        
        except Exception as e:
            logger.error("Ollama generation failed", model=model, error=str(e))
            raise
    
    async def _chat_ollama(
        self,
        request: ChatRequest,
        model: str
    ) -> GenerationResponse:
        """Generate chat completion using Ollama."""
        import aiohttp
        
        try:
            url = f"{self.ai_manager.config.ollama_base_url}/api/chat"
            
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature or self.ai_manager.config.temperature,
                    "top_p": request.top_p or self.ai_manager.config.top_p,
                    "num_predict": request.max_tokens or self.ai_manager.config.max_tokens,
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return GenerationResponse(
                            text=data["message"]["content"],
                            model=model,
                            usage={
                                "prompt_tokens": data.get("prompt_eval_count", 0),
                                "completion_tokens": data.get("eval_count", 0),
                                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                            },
                            finish_reason=data.get("done_reason", "completed"),
                            cost=0.0,
                        )
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Ollama chat failed: {error_text}")
                        
        except Exception as e:
            logger.error("Ollama chat failed", model=model, error=str(e))
            raise
    
    async def _generate_ollama_stream(
        self,
        request: GenerationRequest,
        model: str
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using Ollama."""
        import aiohttp
        
        try:
            url = f"{self.ai_manager.config.ollama_base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": request.prompt,
                "stream": True,
                "options": {
                    "temperature": request.temperature or self.ai_manager.config.temperature,
                    "top_p": request.top_p or self.ai_manager.config.top_p,
                    "num_predict": request.max_tokens or self.ai_manager.config.max_tokens,
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line)
                                    if "response" in data:
                                        yield data["response"]
                                    if data.get("done", False):
                                        break
                                except json.JSONDecodeError:
                                    continue
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Ollama streaming failed: {error_text}")
                        
        except Exception as e:
            logger.error("Ollama streaming failed", model=model, error=str(e))
            raise
    
    async def _chat_ollama_stream(
        self,
        request: ChatRequest,
        model: str
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat using Ollama."""
        import aiohttp
        
        try:
            url = f"{self.ai_manager.config.ollama_base_url}/api/chat"
            
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": request.temperature or self.ai_manager.config.temperature,
                    "top_p": request.top_p or self.ai_manager.config.top_p,
                    "num_predict": request.max_tokens or self.ai_manager.config.max_tokens,
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line)
                                    if "message" in data and "content" in data["message"]:
                                        yield data["message"]["content"]
                                    if data.get("done", False):
                                        break
                                except json.JSONDecodeError:
                                    continue
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Ollama chat streaming failed: {error_text}")
                        
        except Exception as e:
            logger.error("Ollama chat streaming failed", model=model, error=str(e))
            raise
    
    # Utility methods
    
    def _create_cache_key(self, request: GenerationRequest) -> str:
        """Create cache key for generation request."""
        import hashlib
        
        key_data = {
            "prompt": request.prompt,
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return f"gen:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    def _create_chat_cache_key(self, request: ChatRequest) -> str:
        """Create cache key for chat request."""
        import hashlib
        
        messages_data = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        key_data = {
            "messages": messages_data,
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return f"chat:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[GenerationResponse]:
        """Get cached response."""
        # Check in-memory cache
        if cache_key in self._generation_cache:
            return self._generation_cache[cache_key]
        
        # Check database cache
        if self.db_manager and self.db_manager.cache:
            try:
                cached = await self.db_manager.cache.get(cache_key)
                if cached:
                    response = GenerationResponse(**cached)
                    self._generation_cache[cache_key] = response
                    return response
            except Exception as e:
                logger.warning("Failed to get cached response", error=str(e))
        
        return None
    
    async def _cache_response(self, cache_key: str, response: GenerationResponse) -> None:
        """Cache response."""
        # Store in memory
        self._generation_cache[cache_key] = response
        
        # Store in database
        if self.db_manager and self.db_manager.cache:
            try:
                await self.db_manager.cache.set(
                    cache_key,
                    {
                        "text": response.text,
                        "model": response.model,
                        "usage": response.usage,
                        "finish_reason": response.finish_reason,
                        "cost": response.cost,
                    },
                    ttl=3600  # 1 hour
                )
            except Exception as e:
                logger.warning("Failed to cache response", error=str(e))
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get language model service statistics."""
        stats = {
            "cache_size": len(self._generation_cache),
            "providers_available": len(self.ai_manager._clients),
        }
        
        # Add model-specific stats
        text_models = await self.ai_manager.get_available_models(ModelCapability.TEXT_GENERATION)
        chat_models = await self.ai_manager.get_available_models(ModelCapability.CHAT)
        
        stats["text_models"] = [model.name for model in text_models]
        stats["chat_models"] = [model.name for model in chat_models]
        stats["total_models"] = len(text_models) + len(chat_models)
        
        return stats
    
    async def clear_cache(self) -> None:
        """Clear generation cache."""
        async with self._lock:
            self._generation_cache.clear()
            
            if self.db_manager and self.db_manager.cache:
                try:
                    await self.db_manager.cache.clear_pattern("gen:*")
                    await self.db_manager.cache.clear_pattern("chat:*")
                except Exception as e:
                    logger.warning("Failed to clear database cache", error=str(e))