"""
Vector database implementation using Qdrant.

Provides semantic search, similarity detection, and content clustering
using vector embeddings.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
import structlog

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    models = None
    ResponseHandlingException = Exception
    UnexpectedResponse = Exception

from .base import VectorDatabase, DatabaseConfig, ConnectionError, QueryError

logger = structlog.get_logger(__name__)


class VectorDB(VectorDatabase):
    """Qdrant-based vector database for semantic search and similarity."""
    
    def __init__(self, config: DatabaseConfig):
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant is required for vector database. Install with: pip install qdrant-client>=1.6.0"
            )
        
        self.config = config
        self.client: Optional[QdrantClient] = None
        self.collection_name = config.vector_collection
        self._lock = asyncio.Lock()
        self._local_mode = False
    
    async def initialize(self) -> None:
        """Initialize Qdrant connection."""
        async with self._lock:
            try:
                # Try to connect to remote Qdrant instance first
                try:
                    self.client = QdrantClient(url=self.config.vector_url)
                    # Test connection
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.client.get_collections
                    )
                    logger.info("Connected to remote Qdrant instance", url=self.config.vector_url)
                    
                except Exception as e:
                    logger.warning(
                        "Failed to connect to remote Qdrant, falling back to local mode", 
                        error=str(e)
                    )
                    
                    # Fall back to local embedded instance
                    if self.config.vector_local_path:
                        self.config.vector_local_path.mkdir(parents=True, exist_ok=True)
                        self.client = QdrantClient(path=str(self.config.vector_local_path))
                        self._local_mode = True
                        logger.info(
                            "Using local Qdrant instance", 
                            path=self.config.vector_local_path
                        )
                    else:
                        raise ConnectionError("No valid Qdrant configuration available")
                
                # Create collection if it doesn't exist
                await self.create_collection()
                
                logger.info("Vector database initialized", collection=self.collection_name)
                
            except Exception as e:
                logger.error("Failed to initialize vector database", error=str(e))
                raise ConnectionError(f"Vector database initialization failed: {e}")
    
    async def close(self) -> None:
        """Close Qdrant connection."""
        async with self._lock:
            if self.client:
                try:
                    # Qdrant client doesn't need explicit closing in most cases
                    self.client = None
                    logger.debug("Vector database connection closed")
                except Exception as e:
                    logger.error("Error closing vector database", error=str(e))
    
    async def health_check(self) -> bool:
        """Check Qdrant health."""
        try:
            if not self.client:
                return False
            
            # Try to get collection info
            info = await asyncio.get_event_loop().run_in_executor(
                None, self.client.get_collection, self.collection_name
            )
            return info is not None
            
        except Exception:
            return False
    
    async def create_collection(self) -> None:
        """Create vector collection if it doesn't exist."""
        if not self.client:
            raise ConnectionError("Vector database not initialized")
        
        try:
            # Check if collection exists
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.client.get_collection, self.collection_name
                )
                logger.debug("Vector collection already exists", collection=self.collection_name)
                return
            except (ResponseHandlingException, UnexpectedResponse):
                # Collection doesn't exist, create it
                pass
            
            # Create collection with vector configuration
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.create_collection,
                self.collection_name,
                models.VectorParams(
                    size=self.config.vector_dimension,
                    distance=getattr(models.Distance, self.config.vector_distance.upper())
                )
            )
            
            # Create payload indexes for better filtering performance
            payload_indexes = [
                ("note_id", models.PayloadSchemaType.KEYWORD),
                ("file_path", models.PayloadSchemaType.KEYWORD),
                ("tags", models.PayloadSchemaType.KEYWORD),
                ("created_date", models.PayloadSchemaType.DATETIME),
                ("word_count", models.PayloadSchemaType.INTEGER),
            ]
            
            for field_name, field_type in payload_indexes:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.client.create_payload_index,
                    self.collection_name,
                    field_name,
                    field_type
                )
            
            logger.info("Vector collection created", collection=self.collection_name)
            
        except Exception as e:
            logger.error("Failed to create vector collection", error=str(e))
            raise QueryError(f"Collection creation failed: {e}")
    
    async def upsert_embedding(
        self, 
        note_id: str, 
        embedding: List[float], 
        metadata: Dict[str, Any]
    ) -> None:
        """Insert or update note embedding."""
        if not self.client:
            raise ConnectionError("Vector database not initialized")
        
        if len(embedding) != self.config.vector_dimension:
            raise ValueError(
                f"Embedding dimension {len(embedding)} doesn't match "
                f"configured dimension {self.config.vector_dimension}"
            )
        
        try:
            point = models.PointStruct(
                id=note_id,
                vector=embedding,
                payload={
                    "note_id": note_id,
                    "file_path": metadata.get("file_path", ""),
                    "title": metadata.get("title", ""),
                    "content_preview": metadata.get("content", "")[:500],  # First 500 chars
                    "tags": metadata.get("tags", []),
                    "word_count": metadata.get("word_count", 0),
                    "created_date": metadata.get("created_date"),
                    "last_modified": metadata.get("last_modified"),
                    "link_count": metadata.get("link_count", 0),
                    "task_count": metadata.get("task_count", 0),
                    "updated_at": metadata.get("updated_at"),
                }
            )
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.upsert,
                self.collection_name,
                [point]
            )
            
            logger.debug("Embedding upserted", note_id=note_id)
            
        except Exception as e:
            logger.error("Failed to upsert embedding", note_id=note_id, error=str(e))
            raise QueryError(f"Embedding upsert failed: {e}")
    
    async def search_similar(
        self, 
        embedding: List[float], 
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar notes using vector similarity."""
        if not self.client:
            raise ConnectionError("Vector database not initialized")
        
        if len(embedding) != self.config.vector_dimension:
            raise ValueError(
                f"Query embedding dimension {len(embedding)} doesn't match "
                f"configured dimension {self.config.vector_dimension}"
            )
        
        try:
            # Build filter conditions
            filter_dict = None
            if filter_conditions:
                filter_dict = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                )
            
            # Perform search
            search_result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.search,
                self.collection_name,
                embedding,
                filter_dict,
                limit,
                score_threshold
            )
            
            # Format results
            results = []
            for hit in search_result:
                result = {
                    "note_id": hit.id,
                    "score": hit.score,
                    "metadata": hit.payload
                }
                results.append(result)
            
            logger.debug(
                "Vector search completed", 
                results_count=len(results), 
                threshold=score_threshold
            )
            
            return results
            
        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            raise QueryError(f"Vector search failed: {e}")
    
    async def search_similar_by_note(
        self, 
        note_id: str, 
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find notes similar to a given note."""
        if not self.client:
            raise ConnectionError("Vector database not initialized")
        
        try:
            # Get the note's vector
            points = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.retrieve,
                self.collection_name,
                [note_id],
                with_vectors=True
            )
            
            if not points:
                raise ValueError(f"Note {note_id} not found in vector database")
            
            note_vector = points[0].vector
            
            # Search for similar notes, excluding the source note
            filter_conditions = {"note_id": {"$ne": note_id}}
            
            return await self.search_similar(
                note_vector, 
                limit=limit, 
                score_threshold=score_threshold,
                filter_conditions=filter_conditions
            )
            
        except Exception as e:
            logger.error("Similar note search failed", note_id=note_id, error=str(e))
            raise QueryError(f"Similar note search failed: {e}")
    
    async def delete_embedding(self, note_id: str) -> None:
        """Delete note embedding."""
        if not self.client:
            raise ConnectionError("Vector database not initialized")
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.delete,
                self.collection_name,
                [note_id]
            )
            
            logger.debug("Embedding deleted", note_id=note_id)
            
        except Exception as e:
            logger.error("Failed to delete embedding", note_id=note_id, error=str(e))
            raise QueryError(f"Embedding deletion failed: {e}")
    
    async def batch_upsert_embeddings(
        self, 
        embeddings: List[Dict[str, Any]], 
        batch_size: int = 100
    ) -> None:
        """Batch upsert multiple embeddings for better performance."""
        if not self.client:
            raise ConnectionError("Vector database not initialized")
        
        total_embeddings = len(embeddings)
        logger.info("Starting batch embedding upsert", total_count=total_embeddings)
        
        for i in range(0, total_embeddings, batch_size):
            batch = embeddings[i:i + batch_size]
            
            try:
                points = []
                for item in batch:
                    point = models.PointStruct(
                        id=item["note_id"],
                        vector=item["embedding"],
                        payload=item["metadata"]
                    )
                    points.append(point)
                
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.client.upsert,
                    self.collection_name,
                    points
                )
                
                logger.debug(
                    "Batch upserted", 
                    batch_start=i, 
                    batch_size=len(batch),
                    progress=f"{i + len(batch)}/{total_embeddings}"
                )
                
            except Exception as e:
                logger.error("Batch upsert failed", batch_start=i, error=str(e))
                # Continue with next batch rather than failing completely
                continue
        
        logger.info("Batch embedding upsert completed", total_count=total_embeddings)
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.client:
            raise ConnectionError("Vector database not initialized")
        
        try:
            info = await asyncio.get_event_loop().run_in_executor(
                None, self.client.get_collection, self.collection_name
            )
            
            stats = {
                "total_points": info.points_count,
                "vector_dimension": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance.name,
                "status": info.status,
                "indexed_vectors": info.vectors_count if hasattr(info, 'vectors_count') else None,
            }
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get collection stats", error=str(e))
            raise QueryError(f"Collection stats failed: {e}")
    
    async def find_duplicates(
        self, 
        similarity_threshold: float = 0.95,
        batch_size: int = 50
    ) -> List[Dict[str, Any]]:
        """Find potential duplicate notes based on vector similarity."""
        if not self.client:
            raise ConnectionError("Vector database not initialized")
        
        try:
            # Get all points with vectors
            scroll_result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.scroll,
                self.collection_name,
                limit=10000,  # Adjust based on vault size
                with_vectors=True
            )
            
            points = scroll_result[0]
            duplicates = []
            processed_ids = set()
            
            logger.info("Scanning for duplicates", total_points=len(points))
            
            for i, point in enumerate(points):
                if point.id in processed_ids:
                    continue
                
                # Search for similar notes
                similar = await self.search_similar(
                    point.vector,
                    limit=batch_size,
                    score_threshold=similarity_threshold
                )
                
                # Filter out the point itself and already processed points
                similar_notes = [
                    s for s in similar 
                    if s["note_id"] != point.id and s["note_id"] not in processed_ids
                ]
                
                if similar_notes:
                    duplicate_group = {
                        "primary_note": {
                            "note_id": point.id,
                            "metadata": point.payload
                        },
                        "similar_notes": similar_notes,
                        "max_similarity": max(s["score"] for s in similar_notes)
                    }
                    
                    duplicates.append(duplicate_group)
                    
                    # Mark all notes in this group as processed
                    processed_ids.add(point.id)
                    for similar_note in similar_notes:
                        processed_ids.add(similar_note["note_id"])
                
                if i % 100 == 0:
                    logger.debug("Duplicate scan progress", processed=i, total=len(points))
            
            logger.info("Duplicate scan completed", duplicate_groups=len(duplicates))
            return duplicates
            
        except Exception as e:
            logger.error("Duplicate detection failed", error=str(e))
            raise QueryError(f"Duplicate detection failed: {e}")
    
    async def cluster_notes(
        self, 
        num_clusters: int = 10,
        sample_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Cluster notes based on semantic similarity."""
        # This is a simplified clustering approach
        # For production, consider using more sophisticated clustering algorithms
        
        if not self.client:
            raise ConnectionError("Vector database not initialized")
        
        try:
            # Get all points with vectors
            scroll_result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.scroll,
                self.collection_name,
                limit=sample_size or 10000,
                with_vectors=True
            )
            
            points = scroll_result[0]
            
            if len(points) < num_clusters:
                logger.warning(
                    "Not enough points for clustering", 
                    points=len(points), 
                    clusters=num_clusters
                )
                return []
            
            # Extract vectors and metadata
            vectors = np.array([point.vector for point in points])
            metadata = [{"note_id": point.id, "payload": point.payload} for point in points]
            
            # Simple k-means clustering (you might want to use scikit-learn for production)
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = await asyncio.get_event_loop().run_in_executor(
                None, kmeans.fit_predict, vectors
            )
            
            # Group notes by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(metadata[i])
            
            # Format results
            cluster_results = []
            for cluster_id, notes in clusters.items():
                cluster_results.append({
                    "cluster_id": cluster_id,
                    "notes": notes,
                    "size": len(notes)
                })
            
            # Sort by cluster size (largest first)
            cluster_results.sort(key=lambda x: x["size"], reverse=True)
            
            logger.info("Note clustering completed", clusters=len(cluster_results))
            return cluster_results
            
        except Exception as e:
            logger.error("Note clustering failed", error=str(e))
            raise QueryError(f"Note clustering failed: {e}")
    
    async def backup_collection(self, backup_path: Path) -> None:
        """Backup the vector collection."""
        if not self.client:
            raise ConnectionError("Vector database not initialized")
        
        backup_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Export all points
            scroll_result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.scroll,
                self.collection_name,
                limit=100000,  # Adjust based on expected vault size
                with_vectors=True,
                with_payload=True
            )
            
            points = scroll_result[0]
            
            # Save to JSON
            backup_data = {
                "collection_name": self.collection_name,
                "vector_dimension": self.config.vector_dimension,
                "distance_metric": self.config.vector_distance,
                "points": [
                    {
                        "id": point.id,
                        "vector": point.vector,
                        "payload": point.payload
                    }
                    for point in points
                ]
            }
            
            backup_file = backup_path / f"{self.collection_name}_backup.json"
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, default=str, indent=2)
            
            logger.info("Vector collection backed up", file=backup_file, points=len(points))
            
        except Exception as e:
            logger.error("Vector backup failed", error=str(e))
            raise QueryError(f"Vector backup failed: {e}")