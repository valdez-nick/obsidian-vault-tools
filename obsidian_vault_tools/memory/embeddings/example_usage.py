#!/usr/bin/env python3
"""
Example usage of the embedding service and ChromaDB integration.

This demonstrates how to use the embedding components with and without
the optional dependencies.
"""

import numpy as np
from config import EmbeddingConfig
from embedding_service import EmbeddingService  
from chroma_store import ChromaStore


def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===")
    
    # Create configuration
    config = EmbeddingConfig(
        batch_size=16,
        device="cpu"
    )
    
    print(f"Configuration created:")
    print(f"  Primary model: {config.primary_model}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Device: {config.device}")
    
    # Initialize embedding service
    with EmbeddingService(config) as embedder:
        print(f"\nEmbedding service initialized")
        
        # Check if dependencies are available
        has_models = embedder._ensure_dependencies()
        print(f"Models available: {has_models}")
        
        if has_models:
            # Generate embeddings for sample texts
            texts = [
                "How to use Obsidian for note-taking",
                "Machine learning with Python",
                "Vector databases and similarity search",
                "Building knowledge graphs"
            ]
            
            print(f"\nGenerating embeddings for {len(texts)} texts...")
            embeddings = embedder.encode_text(texts)
            print(f"Generated embeddings shape: {embeddings.shape}")
            
            # Compute similarities
            query_text = "Note-taking and knowledge management"
            query_embedding = embedder.encode_text(query_text)
            
            similarities = embedder.compute_similarity(
                query_embedding.reshape(1, -1),
                embeddings
            )
            
            print(f"\nSimilarities to '{query_text}':")
            for i, sim in enumerate(similarities[0]):
                print(f"  {texts[i]}: {sim:.3f}")
        else:
            print("Embedding models not available - install sentence-transformers")


def example_vector_storage():
    """Vector storage example."""
    print("\n=== Vector Storage Example ===")
    
    config = EmbeddingConfig()
    
    with ChromaStore(config) as store:
        print("ChromaDB store initialized")
        
        # Check if ChromaDB is available
        has_chroma = store._ensure_dependencies()
        print(f"ChromaDB available: {has_chroma}")
        
        if has_chroma:
            # Create sample embeddings and documents
            documents = [
                "Understanding machine learning fundamentals",
                "Deep learning neural networks",
                "Natural language processing techniques"
            ]
            
            # Generate random embeddings for demo
            embeddings = np.random.rand(len(documents), 384)
            
            # Add to store
            print(f"\nAdding {len(documents)} documents to store...")
            ids = store.add_embeddings(
                embeddings=embeddings,
                documents=documents,
                metadatas=[
                    {"topic": "ml", "difficulty": "beginner"},
                    {"topic": "dl", "difficulty": "intermediate"}, 
                    {"topic": "nlp", "difficulty": "intermediate"}
                ]
            )
            
            print(f"Added documents with IDs: {ids[:2]}...")
            
            # Query similar documents
            query_embedding = np.random.rand(384)
            results = store.query_similar(
                query_embedding,
                n_results=2,
                where={"difficulty": "intermediate"}
            )
            
            print(f"Found {len(results.get('documents', [[]])[0])} similar documents")
            
        else:
            print("ChromaDB not available - install chromadb")


def example_similarity_search():
    """Similarity search without external dependencies."""
    print("\n=== Similarity Search Example ===")
    
    config = EmbeddingConfig()
    service = EmbeddingService(config)
    
    # Create sample embeddings
    np.random.seed(42)  # For reproducible results
    
    # Simulate document embeddings
    documents = [
        "Python programming tutorial",
        "Machine learning basics", 
        "Web development guide",
        "Data science fundamentals",
        "Software engineering principles"
    ]
    
    # Generate random embeddings (384 dimensions like MiniLM)
    doc_embeddings = np.random.rand(len(documents), 384)
    
    # Normalize for better cosine similarity
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    # Query embedding
    query = "programming and software development"
    query_embedding = np.random.rand(384)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    print(f"Searching for documents similar to: '{query}'")
    
    # Find most similar
    results = service.find_most_similar(
        query_embedding,
        doc_embeddings,
        top_k=3,
        threshold=0.1
    )
    
    print(f"\nTop {len(results)} similar documents:")
    for idx, score in results:
        print(f"  {documents[idx]}: {score:.3f}")


def example_configuration():
    """Configuration examples."""
    print("\n=== Configuration Examples ===")
    
    # Default configuration
    default_config = EmbeddingConfig()
    print(f"Default configuration:")
    print(f"  Model: {default_config.primary_model}")
    print(f"  Batch size: {default_config.batch_size}")
    print(f"  Device: {default_config.device}")
    
    # Custom configuration
    custom_config = EmbeddingConfig(
        primary_model="all-mpnet-base-v2",
        batch_size=64,
        similarity_threshold=0.8,
        collection_name="custom_vault_embeddings"
    )
    
    print(f"\nCustom configuration:")
    print(f"  Model: {custom_config.primary_model}")
    print(f"  Batch size: {custom_config.batch_size}")
    print(f"  Threshold: {custom_config.similarity_threshold}")
    print(f"  Collection: {custom_config.collection_name}")
    
    # Validate configuration
    is_valid = custom_config.validate()
    print(f"  Valid: {is_valid}")
    
    # Device optimization
    custom_config.optimize_for_device()
    print(f"  Optimized device: {custom_config.device}")
    
    # Available models
    models = custom_config.get_available_models()
    print(f"\nAvailable models: {len(models)}")
    for model in models[:3]:
        print(f"  - {model}")


if __name__ == "__main__":
    print("Obsidian Vault Tools - Embedding Service Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_vector_storage()
    example_similarity_search()
    example_configuration()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo enable full functionality, install:")
    print("  pip install sentence-transformers chromadb")