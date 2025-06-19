#!/usr/bin/env python3
"""
Ollama Adapter
Enhanced adapter for Ollama model integration with advanced features
"""

import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class OllamaAdapter:
    """Enhanced Ollama adapter with streaming, embedding, and model management"""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        self.base_url = base_url
        self.timeout = timeout
        self.available_models = []
        self._session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        await self.refresh_models()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
            
    async def refresh_models(self) -> List[str]:
        """Refresh list of available models"""
        try:
            async with self._session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    self.available_models = [
                        model["name"] for model in data.get("models", [])
                    ]
                    logger.info(f"Found {len(self.available_models)} Ollama models")
                    return self.available_models
        except Exception as e:
            logger.error(f"Failed to refresh models: {e}")
            return []
            
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model if not available"""
        if model_name in self.available_models:
            return True
            
        try:
            logger.info(f"Pulling model {model_name}...")
            async with self._session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name}
            ) as response:
                if response.status == 200:
                    # Stream the pull progress
                    async for line in response.content:
                        if line:
                            data = json.loads(line)
                            if "status" in data:
                                logger.info(f"Pull progress: {data['status']}")
                    await self.refresh_models()
                    return model_name in self.available_models
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
            
    async def generate(self, model: str, prompt: str, 
                      temperature: float = 0.7,
                      max_tokens: int = 500,
                      stream: bool = False,
                      **kwargs) -> Any:
        """Generate text using Ollama model"""
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "options": {
                "num_predict": max_tokens,
                **kwargs
            },
            "stream": stream
        }
        
        try:
            async with self._session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama error {response.status}: {error_text}")
                    
                if stream:
                    return self._stream_response(response)
                else:
                    full_response = ""
                    async for line in response.content:
                        if line:
                            data = json.loads(line)
                            if "response" in data:
                                full_response += data["response"]
                    return full_response.strip()
                    
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
            
    async def _stream_response(self, response) -> AsyncGenerator[str, None]:
        """Stream response tokens as they arrive"""
        async for line in response.content:
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
                    
    async def embeddings(self, model: str, text: str) -> List[float]:
        """Generate embeddings for text"""
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
        try:
            async with self._session.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("embedding", [])
                else:
                    raise Exception(f"Embedding error: {response.status}")
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise
            
    async def chat(self, model: str, messages: List[Dict[str, str]], 
                  temperature: float = 0.7, **kwargs) -> str:
        """Chat completion API"""
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "options": kwargs
        }
        
        try:
            async with self._session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                if response.status == 200:
                    full_response = ""
                    async for line in response.content:
                        if line:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                full_response += data["message"]["content"]
                    return full_response.strip()
                else:
                    raise Exception(f"Chat error: {response.status}")
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise
            
    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
        try:
            async with self._session.post(
                f"{self.base_url}/api/show",
                json={"name": model}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}
            
    async def model_exists(self, model: str) -> bool:
        """Check if a model exists locally"""
        await self.refresh_models()
        return model in self.available_models
        
    async def delete_model(self, model: str) -> bool:
        """Delete a model"""
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
        try:
            async with self._session.delete(
                f"{self.base_url}/api/delete",
                json={"name": model}
            ) as response:
                if response.status == 200:
                    await self.refresh_models()
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False

# Example usage
async def test_ollama_adapter():
    """Test Ollama adapter functionality"""
    async with OllamaAdapter() as adapter:
        print("🔍 Testing Ollama Adapter")
        
        # List available models
        models = adapter.available_models
        print(f"\nAvailable models: {models}")
        
        if models:
            model = models[0]
            print(f"\nUsing model: {model}")
            
            # Test generation
            print("\n📝 Testing text generation...")
            response = await adapter.generate(
                model=model,
                prompt="What is the meaning of life?",
                temperature=0.7,
                max_tokens=50
            )
            print(f"Response: {response}")
            
            # Test streaming
            print("\n🌊 Testing streaming generation...")
            stream = await adapter.generate(
                model=model,
                prompt="Count from 1 to 5",
                stream=True
            )
            print("Stream: ", end="")
            async for token in stream:
                print(token, end="", flush=True)
            print()
            
            # Test embeddings (if model supports it)
            try:
                print("\n🔢 Testing embeddings...")
                embedding = await adapter.embeddings(
                    model=model,
                    text="Hello world"
                )
                print(f"Embedding length: {len(embedding)}")
            except:
                print("Model doesn't support embeddings")
                
            # Test chat
            print("\n💬 Testing chat...")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
            chat_response = await adapter.chat(model=model, messages=messages)
            print(f"Chat response: {chat_response}")
            
            # Get model info
            print("\n📊 Getting model info...")
            info = await adapter.get_model_info(model)
            print(f"Model info keys: {list(info.keys())}")
        else:
            print("\n⚠️ No models available. Please run 'ollama pull llama2' first.")

if __name__ == "__main__":
    asyncio.run(test_ollama_adapter())