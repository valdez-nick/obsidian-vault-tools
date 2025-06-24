"""
Transformer Model Manager for Obsidian Vault Tools Memory Service.

Specialized manager for transformer models with support for 2025 state-of-the-art
models like Gemma 2-2B and SmolLM2 variants, including quantization and Flash Attention.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Union, Any, Tuple, Generator
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass

try:
    from .model_config import TransformerModelConfig, QuantizationType, DeviceType
    from .model_manager import ModelManager, ModelInfo, ModelLoadError
except ImportError:
    from model_config import TransformerModelConfig, QuantizationType, DeviceType
    from model_manager import ModelManager, ModelInfo, ModelLoadError

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from text generation."""
    generated_text: str
    input_text: str
    model_id: str
    generation_time: float
    token_count: Optional[int] = None
    tokens_per_second: Optional[float] = None
    metadata: Optional[Dict] = None


class TransformerModelManager:
    """
    Specialized manager for transformer models with advanced features.
    
    Features:
    - 2025 state-of-the-art model support (Gemma 2-2B, SmolLM2)
    - Automatic model selection based on system resources
    - Flash Attention optimization
    - Multiple quantization strategies
    - Streaming generation support
    - Model ensembling capabilities
    - Performance benchmarking
    """
    
    def __init__(self, config: Optional[TransformerModelConfig] = None):
        """Initialize transformer model manager."""
        self.config = config or TransformerModelConfig()
        self.config.optimize_for_device()
        
        # Base model manager for lifecycle operations
        self.model_manager = ModelManager(self.config)
        
        # Transformer-specific state
        self._generation_lock = threading.Lock()
        self._performance_stats: Dict[str, Dict] = {}
        
        # Model registry with 2025 state-of-the-art models
        self.model_registry = self.config.get_model_registry()
        
        logger.info(f"Initialized TransformerModelManager with device: {self.config.get_device_str()}")
        logger.info(f"Primary model: {self.config.primary_model}")
    
    def _ensure_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import transformers
            import torch
            
            # Check for optional optimization dependencies
            try:
                import flash_attn
                logger.info("Flash Attention available")
            except ImportError:
                logger.warning("Flash Attention not available - install flash-attn for better performance")
            
            return True
        except ImportError as e:
            logger.error(f"Required dependencies not available: {e}")
            return False
    
    def load_model(
        self,
        model_id: Optional[str] = None,
        quantization: Optional[QuantizationType] = None,
        force_reload: bool = False,
        **kwargs
    ) -> ModelInfo:
        """
        Load a transformer model with optimization.
        
        Args:
            model_id: Model to load (defaults to primary model)
            quantization: Quantization strategy
            force_reload: Force reload even if cached
            **kwargs: Additional model parameters
            
        Returns:
            ModelInfo with loaded transformer model
        """
        if not self._ensure_dependencies():
            raise ModelLoadError("Required dependencies not available")
        
        # Use primary model if none specified
        if model_id is None:
            model_id = self.config.primary_model
        
        # Use configured quantization if none specified
        if quantization is None:
            quantization = self.config.quantization
        
        # Validate model availability
        if model_id not in self.model_registry:
            logger.warning(f"Model {model_id} not in registry, attempting to load anyway")
        
        try:
            # Load using base model manager
            model_info = self.model_manager.load_model(
                model_id=model_id,
                model_type=self.config.model_type,
                quantization=quantization,
                force_reload=force_reload,
                trust_remote_code=kwargs.get('trust_remote_code', False),
                device_map=kwargs.get('device_map', self.config.device_map),
                use_flash_attention=self.config.use_flash_attention,
                **kwargs
            )
            
            # Transformer-specific post-processing
            self._configure_model_for_generation(model_info)
            
            # Initialize performance tracking
            self._init_performance_tracking(model_id)
            
            return model_info
            
        except Exception as e:
            # Try fallback models if primary fails
            if model_id == self.config.primary_model and self.config.fallback_models:
                logger.warning(f"Primary model {model_id} failed, trying fallbacks")
                
                for fallback_model in self.config.fallback_models:
                    try:
                        logger.info(f"Attempting fallback model: {fallback_model}")
                        return self.load_model(
                            model_id=fallback_model,
                            quantization=quantization,
                            force_reload=force_reload,
                            **kwargs
                        )
                    except Exception as fallback_error:
                        logger.warning(f"Fallback model {fallback_model} also failed: {fallback_error}")
                        continue
            
            raise ModelLoadError(f"Failed to load any transformer model: {e}")
    
    def _configure_model_for_generation(self, model_info: ModelInfo):
        """Configure model for optimal text generation."""
        try:
            model_data = model_info.model_object
            if isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
                tokenizer = model_data['tokenizer']
                
                # Set generation parameters
                if hasattr(model, 'generation_config'):
                    model.generation_config.max_new_tokens = self.config.max_new_tokens
                    model.generation_config.temperature = self.config.temperature
                    model.generation_config.top_p = self.config.top_p
                    model.generation_config.do_sample = self.config.do_sample
                    model.generation_config.repetition_penalty = self.config.repetition_penalty
                    
                    # Set special tokens
                    if self.config.pad_token_id is not None:
                        model.generation_config.pad_token_id = self.config.pad_token_id
                    elif tokenizer.pad_token_id is not None:
                        model.generation_config.pad_token_id = tokenizer.pad_token_id
                    
                    if self.config.eos_token_id is not None:
                        model.generation_config.eos_token_id = self.config.eos_token_id
                    elif tokenizer.eos_token_id is not None:
                        model.generation_config.eos_token_id = tokenizer.eos_token_id
                
                logger.debug(f"Configured generation parameters for {model_info.model_id}")
                
        except Exception as e:
            logger.warning(f"Failed to configure generation parameters: {e}")
    
    def _init_performance_tracking(self, model_id: str):
        """Initialize performance tracking for a model."""
        if model_id not in self._performance_stats:
            self._performance_stats[model_id] = {
                'total_generations': 0,
                'total_generation_time': 0,
                'total_tokens_generated': 0,
                'average_tokens_per_second': 0,
                'first_used': datetime.now().isoformat(),
                'last_used': datetime.now().isoformat()
            }
    
    def generate_text(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text using a transformer model.
        
        Args:
            prompt: Input text prompt
            model_id: Model to use (loads if not already loaded)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with generated text and metadata
        """
        with self._generation_lock:
            # Load model if needed
            if model_id is None:
                model_id = self.config.primary_model
            
            model_info = self.model_manager.get_model(model_id)
            if model_info is None:
                model_info = self.load_model(model_id)
            
            try:
                model_data = model_info.model_object
                model = model_data['model']
                tokenizer = model_data['tokenizer']
                
                # Prepare generation parameters
                generation_kwargs = {
                    'max_new_tokens': max_new_tokens or self.config.max_new_tokens,
                    'temperature': temperature or self.config.temperature,
                    'top_p': top_p or self.config.top_p,
                    'do_sample': self.config.do_sample,
                    'repetition_penalty': self.config.repetition_penalty,
                    'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
                    **kwargs
                }
                
                # Tokenize input
                start_time = time.time()
                inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                
                # Move to device
                device = model_info.device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(**inputs, **generation_kwargs)
                
                # Decode output
                generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                generation_time = time.time() - start_time
                token_count = len(generated_ids)
                tokens_per_second = token_count / generation_time if generation_time > 0 else 0
                
                # Update performance stats
                self._update_performance_stats(model_id, generation_time, token_count)
                
                return GenerationResult(
                    generated_text=generated_text,
                    input_text=prompt,
                    model_id=model_id,
                    generation_time=generation_time,
                    token_count=token_count,
                    tokens_per_second=tokens_per_second,
                    metadata={
                        'generation_params': generation_kwargs,
                        'device': device,
                        'quantization': model_info.quantization_type.value if model_info.quantization_type else None
                    }
                )
                
            except Exception as e:
                logger.error(f"Text generation failed for model {model_id}: {e}")
                raise
    
    def generate_stream(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, GenerationResult]:
        """
        Generate text with streaming output.
        
        Args:
            prompt: Input text prompt
            model_id: Model to use
            **kwargs: Generation parameters
            
        Yields:
            Incremental text tokens
            
        Returns:
            Final GenerationResult
        """
        # Note: This is a simplified streaming implementation
        # Real streaming would require more sophisticated token-by-token generation
        
        result = self.generate_text(prompt, model_id, **kwargs)
        
        # Simulate streaming by yielding chunks
        text = result.generated_text
        chunk_size = max(1, len(text) // 10)  # 10 chunks
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            yield chunk
            time.sleep(0.05)  # Small delay to simulate streaming
        
        return result
    
    def _update_performance_stats(self, model_id: str, generation_time: float, token_count: int):
        """Update performance statistics for a model."""
        stats = self._performance_stats.get(model_id, {})
        
        stats['total_generations'] = stats.get('total_generations', 0) + 1
        stats['total_generation_time'] = stats.get('total_generation_time', 0) + generation_time
        stats['total_tokens_generated'] = stats.get('total_tokens_generated', 0) + token_count
        stats['last_used'] = datetime.now().isoformat()
        
        # Calculate average tokens per second
        if stats['total_generation_time'] > 0:
            stats['average_tokens_per_second'] = stats['total_tokens_generated'] / stats['total_generation_time']
        
        self._performance_stats[model_id] = stats
    
    def benchmark_model(self, model_id: Optional[str] = None, num_samples: int = 5) -> Dict[str, Any]:
        """
        Benchmark a model's performance.
        
        Args:
            model_id: Model to benchmark
            num_samples: Number of test samples
            
        Returns:
            Benchmark results
        """
        if model_id is None:
            model_id = self.config.primary_model
        
        test_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "The most important aspect of machine learning is",
            "When considering the impact of AI on society,",
            "The key to successful software development is"
        ]
        
        # Select prompts for testing
        selected_prompts = (test_prompts * ((num_samples // len(test_prompts)) + 1))[:num_samples]
        
        results = []
        total_time = 0
        total_tokens = 0
        
        logger.info(f"Benchmarking model {model_id} with {num_samples} samples")
        
        for i, prompt in enumerate(selected_prompts):
            try:
                result = self.generate_text(
                    prompt=prompt,
                    model_id=model_id,
                    max_new_tokens=50  # Shorter for benchmarking
                )
                
                results.append({
                    'prompt': prompt,
                    'generation_time': result.generation_time,
                    'token_count': result.token_count,
                    'tokens_per_second': result.tokens_per_second
                })
                
                total_time += result.generation_time
                total_tokens += result.token_count or 0
                
                logger.debug(f"Benchmark {i+1}/{num_samples}: {result.tokens_per_second:.1f} tokens/sec")
                
            except Exception as e:
                logger.error(f"Benchmark sample {i+1} failed: {e}")
                results.append({
                    'prompt': prompt,
                    'error': str(e)
                })
        
        # Calculate statistics
        valid_results = [r for r in results if 'error' not in r]
        avg_tokens_per_second = sum(r['tokens_per_second'] for r in valid_results) / len(valid_results) if valid_results else 0
        avg_generation_time = sum(r['generation_time'] for r in valid_results) / len(valid_results) if valid_results else 0
        
        return {
            'model_id': model_id,
            'num_samples': num_samples,
            'successful_samples': len(valid_results),
            'failed_samples': num_samples - len(valid_results),
            'average_tokens_per_second': avg_tokens_per_second,
            'average_generation_time': avg_generation_time,
            'total_time': total_time,
            'total_tokens': total_tokens,
            'device': self.config.get_device_str(),
            'quantization': self.config.quantization.value,
            'detailed_results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def compare_models(self, model_ids: List[str], num_samples: int = 3) -> Dict[str, Any]:
        """
        Compare performance of multiple models.
        
        Args:
            model_ids: List of models to compare
            num_samples: Number of test samples per model
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(model_ids)} models with {num_samples} samples each")
        
        comparisons = {}
        for model_id in model_ids:
            try:
                benchmark = self.benchmark_model(model_id, num_samples)
                comparisons[model_id] = benchmark
            except Exception as e:
                logger.error(f"Failed to benchmark model {model_id}: {e}")
                comparisons[model_id] = {'error': str(e)}
        
        # Determine best model
        valid_comparisons = {k: v for k, v in comparisons.items() if 'error' not in v}
        
        best_model = None
        best_tokens_per_second = 0
        
        if valid_comparisons:
            for model_id, results in valid_comparisons.items():
                tokens_per_second = results.get('average_tokens_per_second', 0)
                if tokens_per_second > best_tokens_per_second:
                    best_tokens_per_second = tokens_per_second
                    best_model = model_id
        
        return {
            'models_compared': model_ids,
            'comparison_results': comparisons,
            'best_model': best_model,
            'best_tokens_per_second': best_tokens_per_second,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_model_recommendations(self, use_case: str = "general") -> List[Dict[str, Any]]:
        """
        Get model recommendations based on use case and system resources.
        
        Args:
            use_case: Type of use case ("general", "creative", "technical", "fast")
            
        Returns:
            List of recommended models with rationale
        """
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            available_memory_gb = 8.0  # Conservative default
        
        recommendations = []
        
        for model_id, info in self.model_registry.items():
            memory_needed = info["memory_gb"]
            if self.config.quantization != QuantizationType.NONE:
                memory_needed *= 0.5  # Rough estimate for quantization savings
            
            # Check if model fits in memory
            if memory_needed <= available_memory_gb * 0.8:  # Leave 20% buffer
                score = 0
                rationale = []
                
                # Score based on use case
                if use_case == "fast" and info["recommended_batch_size"] > 2:
                    score += 10
                    rationale.append("Optimized for speed")
                
                if use_case == "creative" and "creative" in model_id.lower():
                    score += 15
                    rationale.append("Creative text generation")
                
                if use_case == "technical" and info["parameters"].endswith("B"):
                    score += 10
                    rationale.append("Large parameter count for technical tasks")
                
                # Score based on efficiency
                if memory_needed < 2.0:
                    score += 5
                    rationale.append("Memory efficient")
                
                # Score based on features
                if info.get("flash_attention", False):
                    score += 5
                    rationale.append("Flash Attention support")
                
                if "quantization_support" in info and len(info["quantization_support"]) > 2:
                    score += 3
                    rationale.append("Multiple quantization options")
                
                recommendations.append({
                    'model_id': model_id,
                    'name': info["name"],
                    'description': info["description"],
                    'score': score,
                    'memory_gb': memory_needed,
                    'rationale': rationale,
                    'fits_in_memory': True
                })
            else:
                recommendations.append({
                    'model_id': model_id,
                    'name': info["name"],
                    'description': info["description"],
                    'score': 0,
                    'memory_gb': memory_needed,
                    'rationale': ["Insufficient memory"],
                    'fits_in_memory': False
                })
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'model_performance': self._performance_stats,
            'device': self.config.get_device_str(),
            'quantization': self.config.quantization.value,
            'flash_attention_enabled': self.config.use_flash_attention,
            'loaded_models': self.model_manager.get_model_stats(),
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup(self):
        """Clean up transformer model manager resources."""
        if self.model_manager:
            self.model_manager.cleanup()
        logger.info("TransformerModelManager cleanup completed")