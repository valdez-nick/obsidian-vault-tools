"""
ChromaDB integration for vector storage and similarity search.

Provides persistent storage, CRUD operations, and metadata support
for embedding vectors with efficient similarity search.
"""

import logging
import time
import uuid
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
import numpy as np

try:
    from .config import EmbeddingConfig, DEFAULT_CONFIG
except ImportError:
    from config import EmbeddingConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    ChromaDB integration for vector storage and retrieval.
    
    Features:
    - Persistent vector storage
    - Metadata filtering and search
    - Collection management
    - Efficient similarity search
    - Batch operations
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize ChromaDB store with configuration."""
        self.config = config or DEFAULT_CONFIG
        self._client = None
        self._collection = None
        
        logger.info(f"Initialized ChromaStore with collection: {self.config.collection_name}")
    
    def _ensure_dependencies(self) -> bool:
        """Check if ChromaDB is available."""
        try:
            import chromadb
            return True
        except ImportError as e:
            logger.error(f"ChromaDB not available: {e}")
            return False
    
    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            if not self._ensure_dependencies():
                raise ImportError("chromadb not available. Install with: pip install chromadb")
            
            try:
                import chromadb
                from chromadb.config import Settings
                
                # Create persistent client
                self._client = chromadb.PersistentClient(
                    path=self.config.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                
                logger.info(f"Connected to ChromaDB at: {self.config.persist_directory}")
                
            except Exception as e:
                logger.error(f"Failed to create ChromaDB client: {e}")
                raise
        
        return self._client
    
    def _get_collection(self):
        """Get or create collection."""
        if self._collection is None:
            client = self._get_client()
            
            try:
                # Try to get existing collection
                self._collection = client.get_collection(
                    name=self.config.collection_name
                )
                logger.info(f"Connected to existing collection: {self.config.collection_name}")
                
            except Exception:
                # Create new collection
                import chromadb
                
                distance_function = self.config.distance_metric
                if distance_function == "cosine":
                    distance_function = "cosine"
                elif distance_function == "l2":
                    distance_function = "l2"
                elif distance_function == "ip":
                    distance_function = "ip"
                
                self._collection = client.create_collection(
                    name=self.config.collection_name,
                    metadata={
                        "hnsw:space": distance_function,
                        "description": "Obsidian Vault Tools embeddings"
                    }
                )
                
                logger.info(f"Created new collection: {self.config.collection_name}")
        
        return self._collection
    
    def add_embeddings(
        self,
        embeddings: Union[np.ndarray, List[List[float]]],
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add embeddings to the collection.
        
        Args:
            embeddings: Vector embeddings
            documents: Original documents
            metadatas: Optional metadata for each document
            ids: Optional custom IDs, will generate if not provided
            
        Returns:
            List of document IDs
        """
        collection = self._get_collection()
        
        # Convert numpy arrays to lists if needed
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Ensure metadatas exist
        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]
        
        # Add timestamp to metadata
        current_time = time.time()
        for metadata in metadatas:
            if 'timestamp' not in metadata:
                metadata['timestamp'] = current_time
        
        try:
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} embeddings to collection")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise
    
    def query_similar(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        n_results: Optional[int] = None,
        where: Optional[Dict] = None,
        include: Optional[List[str]] = None
    ) -> Dict:
        """
        Query for similar embeddings.
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Metadata filter conditions
            include: What to include in results
            
        Returns:
            Query results with documents, distances, metadatas
        """
        collection = self._get_collection()
        
        # Convert numpy array to list if needed
        if isinstance(query_embedding, np.ndarray):
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.tolist()
            else:
                query_embedding = query_embedding[0].tolist()
        
        # Set defaults
        n_results = n_results or self.config.default_search_limit
        include = include or ["documents", "distances", "metadatas"]
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=include
            )
            
            logger.debug(f"Query returned {len(results.get('ids', [[]])[0])} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to query similar embeddings: {e}")
            raise
    
    def get_by_ids(self, ids: List[str], include: Optional[List[str]] = None) -> Dict:
        """
        Get documents by their IDs.
        
        Args:
            ids: List of document IDs
            include: What to include in results
            
        Returns:
            Documents and metadata
        """
        collection = self._get_collection()
        include = include or ["documents", "metadatas", "embeddings"]
        
        try:
            results = collection.get(ids=ids, include=include)
            logger.debug(f"Retrieved {len(results.get('ids', []))} documents by ID")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            raise
    
    def update_embeddings(
        self,
        ids: List[str],
        embeddings: Optional[Union[np.ndarray, List[List[float]]]] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None
    ) -> bool:
        """
        Update existing embeddings.
        
        Args:
            ids: Document IDs to update
            embeddings: New embeddings (optional)
            documents: New documents (optional)
            metadatas: New metadata (optional)
            
        Returns:
            Success status
        """
        collection = self._get_collection()
        
        # Convert numpy arrays if needed
        if embeddings is not None and isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        try:
            collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Updated {len(ids)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update embeddings: {e}")
            raise
    
    def delete_embeddings(self, ids: List[str]) -> bool:
        """
        Delete embeddings by ID.
        
        Args:
            ids: Document IDs to delete
            
        Returns:
            Success status
        """
        collection = self._get_collection()
        
        try:
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            raise
    
    def delete_by_metadata(self, where: Dict) -> bool:
        """
        Delete embeddings by metadata filter.
        
        Args:
            where: Metadata filter conditions
            
        Returns:
            Success status
        """
        collection = self._get_collection()
        
        try:
            collection.delete(where=where)
            logger.info(f"Deleted embeddings matching filter: {where}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete by metadata: {e}")
            raise
    
    def search_by_text(
        self,
        query_text: str,
        n_results: Optional[int] = None,
        where: Optional[Dict] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search using text query (requires embedding service).
        
        Args:
            query_text: Text to search for
            n_results: Number of results
            where: Metadata filters
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of matching documents with scores
        """
        # This method would typically require the embedding service
        # For now, it's a placeholder that could be implemented by
        # the calling code that has access to both services
        raise NotImplementedError(
            "Text search requires embedding service integration. "
            "Use query_similar() with pre-computed embeddings."
        )
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            collection = self._get_collection()
            
            # Get collection count
            count = collection.count()
            
            # Get a sample to understand the data
            sample = collection.peek(limit=1)
            
            stats = {
                'total_documents': count,
                'collection_name': self.config.collection_name,
                'persist_directory': self.config.persist_directory,
                'distance_metric': self.config.distance_metric
            }
            
            if sample.get('embeddings') and len(sample['embeddings']) > 0:
                stats['embedding_dimension'] = len(sample['embeddings'][0])
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {'error': str(e)}
    
    def list_collections(self) -> List[str]:
        """List all available collections."""
        try:
            client = self._get_client()
            collections = client.list_collections()
            return [c.name for c in collections]
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def reset_collection(self) -> bool:
        """Reset the current collection (delete all data)."""
        try:
            client = self._get_client()
            client.delete_collection(name=self.config.collection_name)
            self._collection = None  # Will be recreated on next access
            
            logger.warning(f"Reset collection: {self.config.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False
    
    def backup_collection(self, backup_path: str) -> bool:
        """
        Create a backup of the collection.
        Note: This is a simplified backup - full backup would require
        ChromaDB's native backup capabilities.
        """
        try:
            collection = self._get_collection()
            
            # Get all data
            all_data = collection.get(include=["documents", "metadatas", "embeddings"])
            
            # Save to file (simplified - could use more robust serialization)
            import json
            backup_data = {
                'collection_name': self.config.collection_name,
                'config': self.config.to_dict(),
                'data': {
                    'ids': all_data.get('ids', []),
                    'documents': all_data.get('documents', []),
                    'metadatas': all_data.get('metadatas', []),
                    'embeddings': all_data.get('embeddings', [])
                },
                'timestamp': time.time()
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"Collection backed up to: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup collection: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources."""
        if self._collection is not None:
            self._collection = None
        
        if self._client is not None:
            # ChromaDB client doesn't need explicit cleanup
            self._client = None
        
        logger.info("ChromaStore cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()