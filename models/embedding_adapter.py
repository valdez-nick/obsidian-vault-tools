#!/usr/bin/env python3
"""
Embedding Adapter
Adapter for generating and managing embeddings with vector similarity search
"""

import os
import json
import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
import pickle

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    # Create dummy SentenceTransformer
    class DummySentenceTransformer:
        def __init__(self, *args, **kwargs): pass
        def encode(self, *args, **kwargs): return None
    SentenceTransformer = DummySentenceTransformer
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

@dataclass
class Document:
    """Represents a document with its embedding"""
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]

class EmbeddingAdapter:
    """Adapter for embedding generation and similarity search"""
    
    def __init__(self, cache_dir: str = "./embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.models = {}
        self.indices = {}
        self.documents = {}
        
    async def load_model(self, model_name: str = "all-MiniLM-L6-v2") -> bool:
        """Load an embedding model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available")
            return False
            
        try:
            if model_name not in self.models:
                logger.info(f"Loading embedding model {model_name}...")
                self.models[model_name] = SentenceTransformer(
                    model_name,
                    cache_folder=str(self.cache_dir)
                )
                logger.info(f"Model {model_name} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
            
    async def generate_embedding(self, text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
        """Generate embedding for a single text"""
        if model_name not in self.models:
            success = await self.load_model(model_name)
            if not success:
                raise ValueError(f"Failed to load model {model_name}")
                
        model = self.models[model_name]
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: model.encode(text, convert_to_numpy=True)
        )
        
        return embedding
        
    async def generate_embeddings(self, texts: List[str], 
                                model_name: str = "all-MiniLM-L6-v2",
                                batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if model_name not in self.models:
            success = await self.load_model(model_name)
            if not success:
                raise ValueError(f"Failed to load model {model_name}")
                
        model = self.models[model_name]
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100
            )
        )
        
        return embeddings
        
    async def create_index(self, index_name: str, dimension: int, 
                         index_type: str = "flat") -> bool:
        """Create a FAISS index for similarity search"""
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available")
            return False
            
        try:
            if index_type == "flat":
                # Exact search
                index = faiss.IndexFlatL2(dimension)
            elif index_type == "hnsw":
                # Approximate search with HNSW
                index = faiss.IndexHNSWFlat(dimension, 32)
            elif index_type == "ivf":
                # Inverted file index
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            else:
                raise ValueError(f"Unknown index type: {index_type}")
                
            self.indices[index_name] = {
                "index": index,
                "type": index_type,
                "dimension": dimension,
                "trained": index_type == "flat"  # Flat indices don't need training
            }
            
            logger.info(f"Created {index_type} index '{index_name}' with dimension {dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
            
    async def add_documents(self, index_name: str, documents: List[Document]):
        """Add documents to an index"""
        if index_name not in self.indices:
            raise ValueError(f"Index {index_name} not found")
            
        index_data = self.indices[index_name]
        index = index_data["index"]
        
        # Train index if needed
        if not index_data["trained"] and len(documents) > 0:
            embeddings = np.array([doc.embedding for doc in documents])
            index.train(embeddings)
            index_data["trained"] = True
            
        # Add to index
        embeddings = np.array([doc.embedding for doc in documents])
        start_id = len(self.documents.get(index_name, []))
        index.add(embeddings)
        
        # Store documents
        if index_name not in self.documents:
            self.documents[index_name] = []
        self.documents[index_name].extend(documents)
        
        logger.info(f"Added {len(documents)} documents to index '{index_name}'")
        
    async def search(self, index_name: str, query_embedding: np.ndarray, 
                   k: int = 10) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if index_name not in self.indices:
            raise ValueError(f"Index {index_name} not found")
            
        index = self.indices[index_name]["index"]
        documents = self.documents.get(index_name, [])
        
        if len(documents) == 0:
            return []
            
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search
        distances, indices = index.search(query_embedding, min(k, len(documents)))
        
        # Return documents with distances
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(documents):
                results.append((documents[idx], float(distances[0][i])))
                
        return results
        
    async def search_by_text(self, index_name: str, query_text: str, 
                           model_name: str = "all-MiniLM-L6-v2",
                           k: int = 10) -> List[Tuple[Document, float]]:
        """Search using text query"""
        # Generate embedding for query
        query_embedding = await self.generate_embedding(query_text, model_name)
        
        # Search
        return await self.search(index_name, query_embedding, k)
        
    async def save_index(self, index_name: str, file_path: str):
        """Save index to disk"""
        if index_name not in self.indices:
            raise ValueError(f"Index {index_name} not found")
            
        index_data = self.indices[index_name]
        documents = self.documents.get(index_name, [])
        
        # Save FAISS index
        faiss.write_index(index_data["index"], f"{file_path}.faiss")
        
        # Save metadata and documents
        metadata = {
            "type": index_data["type"],
            "dimension": index_data["dimension"],
            "documents": [
                {
                    "id": doc.id,
                    "text": doc.text,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
        }
        
        with open(f"{file_path}.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Save embeddings separately
        if documents:
            embeddings = np.array([doc.embedding for doc in documents])
            np.save(f"{file_path}.npy", embeddings)
            
        logger.info(f"Saved index '{index_name}' to {file_path}")
        
    async def load_index(self, index_name: str, file_path: str):
        """Load index from disk"""
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available")
            return False
            
        try:
            # Load FAISS index
            index = faiss.read_index(f"{file_path}.faiss")
            
            # Load metadata
            with open(f"{file_path}.json", "r") as f:
                metadata = json.load(f)
                
            # Load embeddings
            embeddings = np.load(f"{file_path}.npy")
            
            # Reconstruct documents
            documents = []
            for i, doc_data in enumerate(metadata["documents"]):
                doc = Document(
                    id=doc_data["id"],
                    text=doc_data["text"],
                    embedding=embeddings[i],
                    metadata=doc_data["metadata"]
                )
                documents.append(doc)
                
            # Store in memory
            self.indices[index_name] = {
                "index": index,
                "type": metadata["type"],
                "dimension": metadata["dimension"],
                "trained": True
            }
            self.documents[index_name] = documents
            
            logger.info(f"Loaded index '{index_name}' from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
            
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        # Normalize
        norm1 = embedding1 / np.linalg.norm(embedding1)
        norm2 = embedding2 / np.linalg.norm(embedding2)
        
        # Compute dot product
        return float(np.dot(norm1, norm2))
        
    async def find_duplicates(self, index_name: str, threshold: float = 0.95) -> List[Tuple[Document, Document, float]]:
        """Find duplicate documents based on embedding similarity"""
        if index_name not in self.documents:
            return []
            
        documents = self.documents[index_name]
        duplicates = []
        
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = self.compute_similarity(
                    documents[i].embedding,
                    documents[j].embedding
                )
                
                if similarity >= threshold:
                    duplicates.append((documents[i], documents[j], similarity))
                    
        return duplicates

# Example usage
async def test_embedding_adapter():
    """Test embedding adapter functionality"""
    adapter = EmbeddingAdapter()
    
    print("🧮 Testing Embedding Adapter")
    
    # Load model
    print("\n📥 Loading embedding model...")
    success = await adapter.load_model("all-MiniLM-L6-v2")
    
    if success:
        print("✅ Model loaded successfully")
        
        # Test single embedding
        print("\n🔢 Testing single embedding...")
        text = "This is a test document about machine learning."
        embedding = await adapter.generate_embedding(text)
        print(f"Embedding shape: {embedding.shape}")
        
        # Test batch embeddings
        print("\n📊 Testing batch embeddings...")
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand human language.",
            "Computer vision enables machines to interpret visual information."
        ]
        embeddings = await adapter.generate_embeddings(texts)
        print(f"Batch embeddings shape: {embeddings.shape}")
        
        if FAISS_AVAILABLE:
            # Create index
            print("\n🗂️ Creating FAISS index...")
            await adapter.create_index("test_index", embedding.shape[0], "flat")
            
            # Add documents
            print("\n📄 Adding documents to index...")
            documents = []
            for i, (text, emb) in enumerate(zip(texts, embeddings)):
                doc = Document(
                    id=f"doc_{i}",
                    text=text,
                    embedding=emb,
                    metadata={"category": "AI"}
                )
                documents.append(doc)
                
            await adapter.add_documents("test_index", documents)
            
            # Search
            print("\n🔍 Testing similarity search...")
            query = "What is deep learning?"
            results = await adapter.search_by_text("test_index", query, k=3)
            
            print(f"Query: {query}")
            print("Results:")
            for doc, distance in results:
                print(f"  - {doc.text[:50]}... (distance: {distance:.4f})")
                
            # Find duplicates
            print("\n🔁 Testing duplicate detection...")
            # Add a near-duplicate
            dup_doc = Document(
                id="doc_dup",
                text="Deep learning utilizes neural networks with many layers.",
                embedding=await adapter.generate_embedding(
                    "Deep learning utilizes neural networks with many layers."
                ),
                metadata={"category": "AI"}
            )
            await adapter.add_documents("test_index", [dup_doc])
            
            duplicates = await adapter.find_duplicates("test_index", threshold=0.9)
            print(f"Found {len(duplicates)} potential duplicates")
            
            # Test similarity computation
            print("\n📏 Testing similarity computation...")
            sim = adapter.compute_similarity(embeddings[0], embeddings[1])
            print(f"Similarity between first two documents: {sim:.4f}")
            
            # Save and load index
            print("\n💾 Testing save/load...")
            await adapter.save_index("test_index", "./test_embedding_index")
            
            # Clear and reload
            adapter.indices.clear()
            adapter.documents.clear()
            
            await adapter.load_index("test_index_reloaded", "./test_embedding_index")
            print("✅ Index saved and reloaded successfully")
            
            # Cleanup
            import os
            for ext in [".faiss", ".json", ".npy"]:
                if os.path.exists(f"./test_embedding_index{ext}"):
                    os.remove(f"./test_embedding_index{ext}")
        else:
            print("\n⚠️ FAISS not available. Install with: pip install faiss-cpu")
    else:
        print("❌ Failed to load embedding model")
        print("Install with: pip install sentence-transformers")

if __name__ == "__main__":
    asyncio.run(test_embedding_adapter())