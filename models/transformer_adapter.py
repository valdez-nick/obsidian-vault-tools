#!/usr/bin/env python3
"""
Transformer Adapter
Adapter for loading and using custom transformer models locally
"""

import os
import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Check if transformers is available
try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        pipeline,
        BitsAndBytesConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Create dummy torch module
    class DummyTorch:
        cuda = type('cuda', (), {'is_available': lambda: False})()
        backends = type('backends', (), {'mps': type('mps', (), {'is_available': lambda: False})()})()
        def no_grad(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    torch = DummyTorch()
    
# Global flag to control warning display
_warning_shown = False

def _show_import_warning():
    """Show import warning only once when feature is actually used"""
    global _warning_shown
    if not _warning_shown and not TRANSFORMERS_AVAILABLE:
        logger.warning("transformers/torch library not available. Install with: pip install transformers torch")
        _warning_shown = True

@dataclass
class ModelConfig:
    """Configuration for a transformer model"""
    name: str
    model_type: str  # "generation", "classification", "ner"
    local_path: Optional[str] = None
    device: str = "cpu"
    quantization: Optional[str] = None  # "8bit", "4bit"
    max_length: int = 512
    trust_remote_code: bool = False

class TransformerAdapter:
    """Adapter for using custom transformer models"""
    
    def __init__(self, cache_dir: str = "./model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """Determine best device to use"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
            
    async def load_model(self, config: ModelConfig) -> bool:
        """Load a transformer model"""
        if not TRANSFORMERS_AVAILABLE:
            _show_import_warning()
            logger.error("Transformers library not available")
            return False
            
        try:
            model_key = config.name
            
            # Check if already loaded
            if model_key in self.loaded_models:
                return True
                
            logger.info(f"Loading model {config.name}...")
            
            # Prepare quantization config if needed
            quantization_config = None
            if config.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif config.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.local_path or config.name,
                cache_dir=self.cache_dir,
                trust_remote_code=config.trust_remote_code
            )
            
            # Load model based on type
            if config.model_type == "generation":
                model = AutoModelForCausalLM.from_pretrained(
                    config.local_path or config.name,
                    cache_dir=self.cache_dir,
                    quantization_config=quantization_config,
                    device_map="auto" if config.device != "cpu" else None,
                    trust_remote_code=config.trust_remote_code
                )
            elif config.model_type == "classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    config.local_path or config.name,
                    cache_dir=self.cache_dir,
                    device_map="auto" if config.device != "cpu" else None,
                    trust_remote_code=config.trust_remote_code
                )
            elif config.model_type == "ner":
                model = AutoModelForTokenClassification.from_pretrained(
                    config.local_path or config.name,
                    cache_dir=self.cache_dir,
                    device_map="auto" if config.device != "cpu" else None,
                    trust_remote_code=config.trust_remote_code
                )
            else:
                raise ValueError(f"Unknown model type: {config.model_type}")
                
            # Move to device if not using device_map
            if config.device == "cpu":
                model = model.to(self.device)
                
            # Store loaded model
            self.loaded_models[model_key] = {
                "model": model,
                "tokenizer": tokenizer,
                "config": config,
                "pipeline": None
            }
            
            logger.info(f"Model {config.name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {config.name}: {e}")
            return False
            
    async def generate(self, model_name: str, prompt: str, 
                      max_tokens: int = 100,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      **kwargs) -> str:
        """Generate text using a loaded model"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
            
        model_data = self.loaded_models[model_name]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        config = model_data["config"]
        
        if config.model_type != "generation":
            raise ValueError(f"Model {model_name} is not a generation model")
            
        try:
            # Tokenize input
            inputs = tokenizer(
                prompt, 
                return_tensors="pt",
                max_length=config.max_length,
                truncation=True
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    **kwargs
                )
                
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
                
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
            
    async def classify(self, model_name: str, text: str, 
                      return_all_scores: bool = False) -> Union[str, List[Dict]]:
        """Classify text using a loaded model"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
            
        model_data = self.loaded_models[model_name]
        config = model_data["config"]
        
        if config.model_type != "classification":
            raise ValueError(f"Model {model_name} is not a classification model")
            
        # Create pipeline if not exists
        if model_data["pipeline"] is None:
            model_data["pipeline"] = pipeline(
                "text-classification",
                model=model_data["model"],
                tokenizer=model_data["tokenizer"],
                device=0 if self.device != "cpu" else -1
            )
            
        try:
            results = model_data["pipeline"](text)
            
            if return_all_scores:
                return results
            else:
                return results[0]["label"]
                
        except Exception as e:
            logger.error(f"Classification error: {e}")
            raise
            
    async def extract_entities(self, model_name: str, text: str) -> List[Dict]:
        """Extract named entities using a loaded model"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
            
        model_data = self.loaded_models[model_name]
        config = model_data["config"]
        
        if config.model_type != "ner":
            raise ValueError(f"Model {model_name} is not a NER model")
            
        # Create pipeline if not exists
        if model_data["pipeline"] is None:
            model_data["pipeline"] = pipeline(
                "ner",
                model=model_data["model"],
                tokenizer=model_data["tokenizer"],
                aggregation_strategy="simple",
                device=0 if self.device != "cpu" else -1
            )
            
        try:
            entities = model_data["pipeline"](text)
            return entities
            
        except Exception as e:
            logger.error(f"NER error: {e}")
            raise
            
    async def get_embeddings(self, model_name: str, text: str) -> np.ndarray:
        """Get embeddings from a model"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
            
        model_data = self.loaded_models[model_name]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        try:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(model.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                # Use last hidden state mean as embedding
                if hasattr(outputs, "hidden_states"):
                    embeddings = outputs.hidden_states[-1].mean(dim=1)
                else:
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    
            return embeddings.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise
            
    def unload_model(self, model_name: str):
        """Unload a model to free memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info(f"Model {model_name} unloaded")
            
    def list_loaded_models(self) -> List[str]:
        """List all loaded models"""
        return list(self.loaded_models.keys())
        
    async def download_model(self, model_name: str, model_type: str) -> bool:
        """Download a model from HuggingFace"""
        if not TRANSFORMERS_AVAILABLE:
            return False
            
        try:
            logger.info(f"Downloading {model_name}...")
            
            # Download tokenizer
            AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            # Download model based on type
            if model_type == "generation":
                AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir
                )
            elif model_type == "classification":
                AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir
                )
            elif model_type == "ner":
                AutoModelForTokenClassification.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir
                )
                
            logger.info(f"Model {model_name} downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False

# Example usage
async def test_transformer_adapter():
    """Test transformer adapter"""
    if not TRANSFORMERS_AVAILABLE:
        print("⚠️ Transformers not available. Install with: pip install transformers torch")
        return
        
    adapter = TransformerAdapter()
    
    print("🤖 Testing Transformer Adapter")
    print(f"Device: {adapter.device}")
    
    # Example: Load a small model for testing
    # You can replace with any HuggingFace model
    test_config = ModelConfig(
        name="gpt2",  # Small model for testing
        model_type="generation",
        device=adapter.device
    )
    
    # Load model
    print(f"\n📥 Loading model {test_config.name}...")
    success = await adapter.load_model(test_config)
    
    if success:
        print("✅ Model loaded successfully")
        
        # Test generation
        print("\n📝 Testing text generation...")
        prompt = "The future of AI is"
        response = await adapter.generate(
            test_config.name,
            prompt,
            max_tokens=50,
            temperature=0.8
        )
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        
        # Test embeddings
        print("\n🔢 Testing embeddings...")
        embeddings = await adapter.get_embeddings(
            test_config.name,
            "Hello world"
        )
        print(f"Embedding shape: {embeddings.shape}")
        
        # List loaded models
        print(f"\n📋 Loaded models: {adapter.list_loaded_models()}")
        
        # Unload model
        adapter.unload_model(test_config.name)
        print("♻️ Model unloaded")
    else:
        print("❌ Failed to load model")

if __name__ == "__main__":
    asyncio.run(test_transformer_adapter())