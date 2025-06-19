# LLM Setup Guide for Obsidian Librarian

This guide will help you set up local Large Language Models (LLMs) to work with Obsidian Librarian v2, enabling AI-powered features while keeping all your data private and processed locally.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installing Ollama](#installing-ollama)
3. [Starting Ollama Service](#starting-ollama-service)
4. [Recommended Models](#recommended-models)
5. [Configuration Options](#configuration-options)
6. [Testing Your Setup](#testing-your-setup)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tuning](#performance-tuning)
9. [Adding Custom Models](#adding-custom-models)
10. [Privacy and Security](#privacy-and-security)

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores (8 threads recommended)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space for models
- **OS**: macOS 11+, Linux (Ubuntu 20.04+), Windows 10+

### Recommended Requirements for Optimal Performance
- **CPU**: 8+ cores with AVX2 support
- **RAM**: 16GB+ (32GB for larger models)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (optional but significantly faster)
- **Storage**: 50GB+ free space for multiple models

### GPU Support (Optional)
- **NVIDIA**: CUDA 11.7+ with cuDNN 8+
- **Apple Silicon**: Metal support built-in
- **AMD**: ROCm support (experimental)

## Installing Ollama

Ollama is the backend service that runs LLMs locally. Choose your platform below:

### macOS
```bash
# Using Homebrew (recommended)
brew install ollama

# Or download directly
curl -fsSL https://ollama.com/install.sh | sh
```

### Linux
```bash
# One-line installer
curl -fsSL https://ollama.com/install.sh | sh

# Or using package manager (Ubuntu/Debian)
sudo apt update
sudo apt install ollama
```

### Windows
1. Download the installer from [https://ollama.com/download/windows](https://ollama.com/download/windows)
2. Run the installer (OllamaSetup.exe)
3. Follow the installation wizard
4. Ollama will be added to your PATH automatically

### Docker (All Platforms)
```bash
docker pull ollama/ollama
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

## Starting Ollama Service

### macOS/Linux
```bash
# Start the service
ollama serve

# Or run as a background service
ollama serve &

# Check if running
curl http://localhost:11434/api/version
```

### Windows
Ollama runs as a Windows service automatically after installation. To manage it:
```powershell
# Check status
Get-Service Ollama

# Start service
Start-Service Ollama

# Stop service
Stop-Service Ollama
```

### Verify Installation
```bash
# Check Ollama version
ollama --version

# List installed models
ollama list

# Test API endpoint
curl http://localhost:11434/api/tags
```

## Recommended Models

Pull these models based on your needs and hardware capabilities:

### For Research and Analysis (Recommended)
```bash
# Mistral 7B - Excellent balance of performance and quality
ollama pull mistral:7b

# Mixtral 8x7B - High quality, requires 48GB RAM
ollama pull mixtral:8x7b
```

### For Summarization
```bash
# Llama 3 8B - Great for content summarization
ollama pull llama3:8b

# Phi-3 Mini - Lightweight, good for quick summaries
ollama pull phi3:mini
```

### For Code Understanding
```bash
# CodeLlama 7B - Specialized for code analysis
ollama pull codellama:7b

# DeepSeek Coder - Excellent for technical content
ollama pull deepseek-coder:6.7b
```

### For Embeddings
```bash
# Nomic Embed Text - High-quality embeddings
ollama pull nomic-embed-text

# All-MiniLM - Lightweight embeddings
ollama pull all-minilm:latest
```

### Model Descriptions

| Model | Size | RAM Required | Best For | Speed |
|-------|------|--------------|----------|-------|
| mistral:7b | 4.1GB | 8GB | General purpose, research | Fast |
| mixtral:8x7b | 26GB | 48GB | High-quality analysis | Moderate |
| llama3:8b | 4.7GB | 8GB | Summarization, Q&A | Fast |
| phi3:mini | 2.3GB | 4GB | Quick tasks, low resources | Very Fast |
| codellama:7b | 3.8GB | 8GB | Code analysis | Fast |
| nomic-embed-text | 274MB | 1GB | Text embeddings | Very Fast |

## Configuration Options

Edit the `model_config.yaml` file in your Obsidian Librarian installation:

```yaml
# obsidian-librarian-v2/config/model_config.yaml

llm:
  # Primary model for content analysis
  primary_model: "mistral:7b"
  
  # Model for summarization tasks
  summarization_model: "llama3:8b"
  
  # Model for code understanding
  code_model: "codellama:7b"
  
  # Embedding model for semantic search
  embedding_model: "nomic-embed-text"
  
  # Ollama API settings
  api:
    base_url: "http://localhost:11434"
    timeout: 120  # seconds
    max_retries: 3
  
  # Generation parameters
  generation:
    temperature: 0.7
    top_p: 0.9
    max_tokens: 2048
    repeat_penalty: 1.1
    
  # Context window settings
  context:
    max_length: 4096
    overlap: 200
    
  # Performance settings
  performance:
    batch_size: 4
    concurrent_requests: 2
    cache_responses: true
    cache_ttl: 3600  # seconds
```

### Model-Specific Configurations

```yaml
# Advanced per-model settings
models:
  mistral:7b:
    temperature: 0.7
    context_length: 8192
    
  llama3:8b:
    temperature: 0.5
    system_prompt: "You are a helpful assistant that summarizes content concisely."
    
  codellama:7b:
    temperature: 0.3
    format: "llama"
```

## Testing Your Setup

### 1. Test Ollama Connection
```bash
# Check if Ollama is responding
curl http://localhost:11434/api/version

# Expected output:
# {"version":"0.1.x"}
```

### 2. Test Model Loading
```bash
# Generate a test response
curl http://localhost:11434/api/generate -d '{
  "model": "mistral:7b",
  "prompt": "What is the capital of France?",
  "stream": false
}'
```

### 3. Test with Obsidian Librarian
```bash
# Navigate to obsidian-librarian-v2
cd obsidian-librarian-v2

# Run the test command
obsidian-librarian test-llm --model mistral:7b

# Test specific functionality
obsidian-librarian analyze /path/to/vault --test-mode
```

### 4. Python Test Script
Create a test script `test_llm.py`:

```python
import requests
import json

def test_ollama():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral:7b",
        "prompt": "Explain quantum computing in one sentence.",
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ LLM Response: {result['response']}")
            return True
        else:
            print(f"✗ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_ollama()
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Ollama Service Not Starting
```bash
# Check if port is already in use
lsof -i :11434

# Kill existing process
kill -9 $(lsof -t -i:11434)

# Start with verbose logging
OLLAMA_DEBUG=1 ollama serve
```

#### 2. Model Download Failures
```bash
# Clear partial downloads
rm -rf ~/.ollama/models/.partial

# Retry with specific version
ollama pull mistral:7b-instruct-v0.2

# Use different registry
OLLAMA_REGISTRY=https://registry.ollama.ai ollama pull mistral:7b
```

#### 3. Out of Memory Errors
```bash
# Set memory limit
OLLAMA_MAX_LOADED_MODELS=1 ollama serve

# Use quantized models
ollama pull mistral:7b-q4_0  # 4-bit quantization

# Adjust context length in config
# Reduce context.max_length in model_config.yaml
```

#### 4. Slow Performance
```bash
# Enable GPU acceleration (NVIDIA)
OLLAMA_CUDA_VISIBLE_DEVICES=0 ollama serve

# Check GPU usage
nvidia-smi

# For Apple Silicon
# GPU acceleration is automatic
```

#### 5. Connection Refused
```bash
# Check firewall settings
sudo ufw allow 11434/tcp  # Linux

# Check if service is binding to localhost only
OLLAMA_HOST=0.0.0.0:11434 ollama serve  # Allow external connections
```

### Debug Commands
```bash
# Check Ollama logs
journalctl -u ollama -f  # Linux systemd

# macOS logs
tail -f ~/Library/Logs/ollama.log

# Windows logs
Get-Content "$env:LOCALAPPDATA\Ollama\logs\server.log" -Tail 50 -Wait

# Test model loading
time ollama run mistral:7b "Hello"

# Check model info
ollama show mistral:7b --modelfile
```

## Performance Tuning

### 1. GPU Acceleration

#### NVIDIA GPUs
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Set GPU memory fraction
export OLLAMA_GPU_MEMORY_FRACTION=0.8

# Use multiple GPUs
export OLLAMA_CUDA_VISIBLE_DEVICES=0,1
```

#### Apple Silicon
```bash
# Metal acceleration is automatic
# Monitor GPU usage in Activity Monitor
```

### 2. CPU Optimization
```bash
# Set thread count
export OLLAMA_NUM_THREADS=8

# Enable AVX2 (if supported)
export OLLAMA_USE_AVX2=1

# Set CPU affinity (Linux)
taskset -c 0-7 ollama serve
```

### 3. Memory Management
```yaml
# In model_config.yaml
performance:
  # Limit concurrent model loads
  max_loaded_models: 1
  
  # Set memory per model
  memory_per_model: 8192  # MB
  
  # Enable memory mapping
  use_mmap: true
  
  # Garbage collection interval
  gc_interval: 300  # seconds
```

### 4. Caching Strategies
```yaml
cache:
  # Response caching
  enable_response_cache: true
  response_cache_size: 1000  # entries
  
  # Embedding cache
  enable_embedding_cache: true
  embedding_cache_path: "~/.ollama/cache/embeddings"
  
  # Model cache
  keep_models_loaded: ["mistral:7b", "nomic-embed-text"]
```

### 5. Batch Processing
```python
# Example batch configuration
batch_config = {
    "batch_size": 8,
    "max_concurrent": 2,
    "timeout_per_item": 30,
    "retry_failed": True
}
```

## Adding Custom Models

### 1. Create a Modelfile
```dockerfile
# Modelfile for custom research model
FROM mistral:7b

# Set custom parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.2

# Set system prompt
SYSTEM """You are an expert research assistant specialized in analyzing and organizing information from academic papers and technical documentation. Focus on extracting key insights and creating structured summaries."""

# Set template
TEMPLATE """{{ .System }}
User: {{ .Prompt }}
Assistant: I'll analyze this information and provide a structured response.

{{ .Response }}"""
```

### 2. Build Custom Model
```bash
# Create the model
ollama create research-assistant -f ./Modelfile

# Test it
ollama run research-assistant "Summarize the key points about transformer architecture"
```

### 3. Fine-tuned Models
```bash
# Import GGUF format models
ollama create my-finetuned-model -f ./Modelfile --quantize q4_0

# Import from HuggingFace (with conversion)
# First convert to GGUF format, then import
```

### 4. Model Variants
```yaml
# Configure variants in model_config.yaml
model_variants:
  research:
    base_model: "mistral:7b"
    system_prompt: "Research assistant prompt..."
    temperature: 0.6
    
  creative:
    base_model: "mistral:7b"
    system_prompt: "Creative writing assistant..."
    temperature: 0.9
    
  technical:
    base_model: "codellama:7b"
    system_prompt: "Technical documentation expert..."
    temperature: 0.3
```

## Privacy and Security

### Local Processing Guarantee
- **All LLM processing happens on your machine** - no data is sent to external servers
- Models are downloaded once and stored locally in `~/.ollama/models/`
- Your vault content never leaves your computer

### Security Best Practices

#### 1. Network Security
```bash
# Bind to localhost only (default)
OLLAMA_HOST=127.0.0.1:11434 ollama serve

# If you need network access, use firewall rules
sudo ufw allow from 192.168.1.0/24 to any port 11434
```

#### 2. Model Verification
```bash
# Check model hash
ollama show mistral:7b --modelfile | grep FROM

# Verify model source
ollama list --format json | jq '.models[].digest'
```

#### 3. Data Isolation
```yaml
# In model_config.yaml
security:
  # Disable telemetry
  disable_telemetry: true
  
  # Local-only mode
  offline_mode: true
  
  # Sanitize prompts
  sanitize_inputs: true
  remove_pii: true
```

#### 4. Access Control
```bash
# Set up authentication (optional)
export OLLAMA_API_KEY="your-secret-key"

# Restrict model access
chmod 700 ~/.ollama/models
```

### Privacy Features
- **No API keys required** - everything runs locally
- **No usage tracking** - Ollama doesn't collect usage data
- **Offline capable** - works without internet after model download
- **Data sovereignty** - you control where models and data are stored

### Compliance Considerations
- Models are stored in `~/.ollama/models/`
- Temporary files are created in system temp directory
- No logs are sent externally
- All processing is ephemeral unless explicitly cached

---

## Quick Start Checklist

- [ ] Install Ollama for your platform
- [ ] Start Ollama service
- [ ] Pull recommended model: `ollama pull mistral:7b`
- [ ] Test with: `curl http://localhost:11434/api/version`
- [ ] Configure `model_config.yaml`
- [ ] Run Obsidian Librarian test command
- [ ] Adjust performance settings as needed

## Getting Help

- Ollama Documentation: https://github.com/ollama/ollama
- Obsidian Librarian Issues: https://github.com/yourusername/obsidian-librarian-v2/issues
- Model Library: https://ollama.com/library

Remember: All processing happens locally on your machine, ensuring complete privacy and control over your data!