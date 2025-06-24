# Embeddings Module

Vector embeddings and similarity search capabilities for the Obsidian Vault Tools memory service.

## Overview

This module provides a complete embedding service with local model support and persistent vector storage using ChromaDB. It's designed to work offline and handle graceful fallbacks when dependencies are not available.

## Components

### 1. Configuration (`config.py`)
- **EmbeddingConfig**: Configuration management for models, performance tuning, and storage
- **Features**:
  - Multiple model support with fallbacks
  - Device optimization (CPU/CUDA/MPS)
  - Performance tuning parameters
  - ChromaDB integration settings
  - Validation and serialization

### 2. Embedding Service (`embedding_service.py`)
- **EmbeddingService**: Local text embedding generation using sentence-transformers
- **Features**:
  - Model caching and lazy loading
  - Batch processing for efficiency
  - Multiple similarity metrics
  - Performance tracking
  - Graceful fallbacks

### 3. Vector Store (`chroma_store.py`)
- **ChromaStore**: Persistent vector storage and similarity search
- **Features**:
  - CRUD operations for embeddings
  - Metadata filtering and search
  - Collection management
  - Backup and restore capabilities

## Installation

### Required Dependencies
```bash
# For embedding generation
pip install sentence-transformers>=2.0
pip install torch>=1.9
pip install numpy>=1.21

# For vector storage
pip install chromadb>=0.4.0
```

### Using pip extras
```bash
# Install with AI features (includes all embedding dependencies)
pip install -e ".[ai]"
```

## Usage

### Basic Usage

```python
from obsidian_vault_tools.memory.embeddings import (
    EmbeddingService, 
    ChromaStore, 
    EmbeddingConfig
)

# Create configuration
config = EmbeddingConfig()

# Initialize services
with EmbeddingService(config) as embedder:
    with ChromaStore(config) as store:
        # Generate embeddings
        texts = ["Hello world", "Machine learning", "Vector search"]
        embeddings = embedder.encode_text(texts)
        
        # Store embeddings
        ids = store.add_embeddings(
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": "example"} for _ in texts]
        )
        
        # Search for similar content
        query_embedding = embedder.encode_text("AI and ML")
        results = store.query_similar(query_embedding, n_results=2)
        
        print(f"Found {len(results['documents'][0])} similar documents")
```

### Advanced Configuration

```python
# Custom configuration
config = EmbeddingConfig(
    primary_model="all-mpnet-base-v2",  # Higher quality model
    batch_size=64,                      # Larger batch size
    device="cuda",                      # Use GPU if available
    similarity_threshold=0.8            # Higher similarity threshold
)

# Optimize for current hardware
config.optimize_for_device()
```

### Performance Monitoring

```python
with EmbeddingService(config) as service:
    # Generate embeddings
    embeddings = service.encode_text(large_text_list)
    
    # Get performance statistics
    stats = service.get_performance_stats()
    print(f"Processed {stats['total_embeddings_generated']} embeddings")
    print(f"Average speed: {stats['embeddings_per_second']:.2f} emb/sec")
```

### Metadata Filtering

```python
# Add embeddings with metadata
store.add_embeddings(
    embeddings=embeddings,
    documents=documents,
    metadatas=[
        {"source": "notes", "topic": "AI", "date": "2024-01-01"},
        {"source": "articles", "topic": "ML", "date": "2024-01-02"}
    ]
)

# Search with metadata filters
results = store.query_similar(
    query_embedding,
    where={"source": "notes", "topic": "AI"},
    n_results=5
)
```

## Available Models

### Primary Models (Optimized)
- **multi-qa-MiniLM-L6-cos-v1**: Optimized for semantic search and Q&A
- **all-MiniLM-L6-v2**: General purpose, good balance of speed/quality

### Alternative Models
- **all-mpnet-base-v2**: Higher quality, larger size
- **paraphrase-MiniLM-L6-v2**: Specialized for paraphrase detection
- **all-distilroberta-v1**: DistilRoBERTa based model

## Error Handling

The module includes comprehensive error handling:

1. **Graceful Fallbacks**: Automatically falls back to alternative models if primary fails
2. **Dependency Checks**: Validates required packages are available
3. **Device Detection**: Automatically selects best available device
4. **Connection Recovery**: Handles ChromaDB connection issues

## Testing

### Run Basic Tests
```bash
cd obsidian_vault_tools/memory/embeddings
python test_basic.py
```

### Run with Dependencies
```bash
# Install dependencies first
pip install sentence-transformers chromadb

# Then run comprehensive tests
python test_basic.py
```

## Performance Considerations

### Memory Usage
- Models are cached after first load
- Use `lazy_loading=True` to defer model loading
- Clean up with `service.cleanup()` when done

### Speed Optimization
- Increase `batch_size` for bulk operations
- Use GPU if available (`device="cuda"`)
- Consider smaller models for real-time use

### Storage
- ChromaDB data stored in `~/.local/share/obsidian_vault_tools/embeddings/`
- Model cache in `~/.cache/obsidian_vault_tools/models/`
- Use `backup_collection()` for data protection

## Architecture

```
EmbeddingConfig
    ├── Model Selection & Caching
    ├── Performance Parameters  
    ├── Device Optimization
    └── Storage Configuration

EmbeddingService
    ├── Model Loading & Management
    ├── Text → Vector Conversion
    ├── Similarity Computation
    └── Performance Tracking

ChromaStore
    ├── Vector Storage (CRUD)
    ├── Similarity Search
    ├── Metadata Filtering
    └── Collection Management
```

## Integration

This module integrates with:
- **Memory Service**: Provides vector search for user behavior patterns
- **Research Assistant**: Semantic search across notes and documents  
- **Tag System**: Similarity-based tag suggestions
- **Vault Analysis**: Content clustering and organization

## Future Enhancements

- [ ] Support for multilingual models
- [ ] Hierarchical clustering capabilities
- [ ] Real-time embedding updates
- [ ] Custom model fine-tuning
- [ ] Distributed storage options