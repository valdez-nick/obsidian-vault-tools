/*!
Vector index for high-performance similarity search.
*/

use crate::{SearchError, Result, Embedding, DocumentId};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

#[cfg(feature = "hnsw")]
use hnswlib::{Index as HnswIndex, SearchResult as HnswSearchResult};

/// Configuration for vector index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Embedding dimension
    pub dimension: usize,
    
    /// Maximum number of elements
    pub max_elements: usize,
    
    /// HNSW M parameter (connectivity)
    pub hnsw_m: usize,
    
    /// HNSW ef_construction parameter
    pub ef_construction: usize,
    
    /// HNSW ef parameter for search
    pub ef_search: usize,
    
    /// Index persistence path
    pub index_path: Option<PathBuf>,
    
    /// Whether to use multiple threads
    pub parallel: bool,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dimension: 384,
            max_elements: 1_000_000,
            hnsw_m: 16,
            ef_construction: 200,
            ef_search: 50,
            index_path: None,
            parallel: true,
        }
    }
}

/// Search result from vector index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Document ID
    pub document_id: DocumentId,
    
    /// Similarity score (0.0 to 1.0)
    pub score: f32,
    
    /// Distance (lower is more similar)
    pub distance: f32,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Document metadata stored with embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Document ID
    pub id: DocumentId,
    
    /// Document content preview
    pub preview: String,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    
    /// Timestamp when added
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// High-performance vector index
pub struct VectorIndex {
    config: IndexConfig,
    
    #[cfg(feature = "hnsw")]
    index: Arc<RwLock<Option<HnswIndex>>>,
    
    /// Document metadata lookup
    metadata: Arc<RwLock<HashMap<DocumentId, DocumentMetadata>>>,
    
    /// Internal ID to document ID mapping
    id_mapping: Arc<RwLock<HashMap<u32, DocumentId>>>,
    
    /// Next internal ID
    next_id: Arc<RwLock<u32>>,
}

impl VectorIndex {
    /// Create a new vector index
    pub fn new(config: IndexConfig) -> Result<Self> {
        info!("Creating vector index with dimension {}", config.dimension);
        
        Ok(Self {
            config,
            #[cfg(feature = "hnsw")]
            index: Arc::new(RwLock::new(None)),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            id_mapping: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(RwLock::new(0)),
        })
    }

    /// Initialize the index
    pub async fn initialize(&self) -> Result<()> {
        #[cfg(feature = "hnsw")]
        {
            let mut index_guard = self.index.write().await;
            
            let index = HnswIndex::new(
                &hnswlib::Space::Cosine,
                self.config.dimension,
            ).map_err(|e| SearchError::index(format!("Failed to create HNSW index: {}", e)))?;
            
            index.set_ef_construction(self.config.ef_construction)
                .map_err(|e| SearchError::index(format!("Failed to set ef_construction: {}", e)))?;
            
            index.set_ef(self.config.ef_search)
                .map_err(|e| SearchError::index(format!("Failed to set ef: {}", e)))?;
            
            index.resize(self.config.max_elements)
                .map_err(|e| SearchError::index(format!("Failed to resize index: {}", e)))?;
            
            *index_guard = Some(index);
            info!("HNSW index initialized");
        }
        
        #[cfg(not(feature = "hnsw"))]
        {
            warn!("HNSW feature not enabled, using fallback implementation");
        }
        
        Ok(())
    }

    /// Add a document embedding to the index
    pub async fn add_document(
        &self,
        document_id: DocumentId,
        embedding: Embedding,
        metadata: DocumentMetadata,
    ) -> Result<()> {
        if embedding.len() != self.config.dimension {
            return Err(SearchError::dimension_mismatch(
                self.config.dimension,
                embedding.len(),
            ));
        }

        #[cfg(feature = "hnsw")]
        {
            let mut next_id_guard = self.next_id.write().await;
            let internal_id = *next_id_guard;
            *next_id_guard += 1;
            drop(next_id_guard);

            // Add to HNSW index
            {
                let index_guard = self.index.read().await;
                if let Some(ref index) = *index_guard {
                    let embedding_vec: Vec<f32> = embedding.to_vec();
                    index.add_point(&embedding_vec, internal_id)
                        .map_err(|e| SearchError::index(format!("Failed to add point: {}", e)))?;
                } else {
                    return Err(SearchError::index("Index not initialized".to_string()));
                }
            }

            // Store metadata and mapping
            {
                let mut metadata_guard = self.metadata.write().await;
                metadata_guard.insert(document_id.clone(), metadata);
            }
            
            {
                let mut mapping_guard = self.id_mapping.write().await;
                mapping_guard.insert(internal_id, document_id.clone());
            }

            debug!("Added document {} to index with internal ID {}", document_id, internal_id);
        }

        Ok(())
    }

    /// Search for similar documents
    pub async fn search(
        &self,
        query_embedding: Embedding,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        if query_embedding.len() != self.config.dimension {
            return Err(SearchError::dimension_mismatch(
                self.config.dimension,
                query_embedding.len(),
            ));
        }

        #[cfg(feature = "hnsw")]
        {
            let results = {
                let index_guard = self.index.read().await;
                if let Some(ref index) = *index_guard {
                    let query_vec: Vec<f32> = query_embedding.to_vec();
                    index.search(&query_vec, k)
                        .map_err(|e| SearchError::search(format!("Search failed: {}", e)))?
                } else {
                    return Err(SearchError::search("Index not initialized".to_string()));
                }
            };

            // Convert results to our format
            let mut search_results = Vec::new();
            
            let metadata_guard = self.metadata.read().await;
            let mapping_guard = self.id_mapping.read().await;
            
            for result in results {
                if let Some(document_id) = mapping_guard.get(&result.0) {
                    let metadata = metadata_guard.get(document_id)
                        .map(|m| m.metadata.clone())
                        .unwrap_or_default();
                    
                    search_results.push(SearchResult {
                        document_id: document_id.clone(),
                        score: 1.0 - result.1, // Convert distance to similarity
                        distance: result.1,
                        metadata,
                    });
                }
            }

            Ok(search_results)
        }

        #[cfg(not(feature = "hnsw"))]
        {
            // Fallback implementation using brute force
            self.brute_force_search(query_embedding, k).await
        }
    }

    /// Brute force search (fallback when HNSW is not available)
    #[cfg(not(feature = "hnsw"))]
    async fn brute_force_search(
        &self,
        query_embedding: Embedding,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        warn!("Using brute force search - consider enabling HNSW feature for better performance");
        
        // This is a placeholder implementation
        // In practice, you'd iterate through all stored embeddings and compute similarities
        Ok(Vec::new())
    }

    /// Remove a document from the index
    pub async fn remove_document(&self, document_id: &DocumentId) -> Result<bool> {
        // Find the internal ID for this document
        let internal_id = {
            let mapping_guard = self.id_mapping.read().await;
            mapping_guard.iter()
                .find(|(_, doc_id)| *doc_id == document_id)
                .map(|(id, _)| *id)
        };

        if let Some(internal_id) = internal_id {
            #[cfg(feature = "hnsw")]
            {
                // Note: HNSW doesn't support removal, so we'd need to rebuild
                // For now, just remove from metadata
                warn!("HNSW doesn't support removal - document will remain in index but metadata removed");
            }

            // Remove metadata and mapping
            {
                let mut metadata_guard = self.metadata.write().await;
                metadata_guard.remove(document_id);
            }
            
            {
                let mut mapping_guard = self.id_mapping.write().await;
                mapping_guard.remove(&internal_id);
            }

            debug!("Removed document {} from index", document_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get document metadata
    pub async fn get_metadata(&self, document_id: &DocumentId) -> Option<DocumentMetadata> {
        let metadata_guard = self.metadata.read().await;
        metadata_guard.get(document_id).cloned()
    }

    /// Get index statistics
    pub async fn stats(&self) -> IndexStats {
        let metadata_guard = self.metadata.read().await;
        let document_count = metadata_guard.len();
        
        #[cfg(feature = "hnsw")]
        let index_size = {
            let index_guard = self.index.read().await;
            if let Some(ref index) = *index_guard {
                index.get_current_count() as usize
            } else {
                0
            }
        };
        
        #[cfg(not(feature = "hnsw"))]
        let index_size = document_count;

        IndexStats {
            document_count,
            index_size,
            dimension: self.config.dimension,
            max_elements: self.config.max_elements,
        }
    }

    /// Save index to disk
    pub async fn save(&self, path: &Path) -> Result<()> {
        #[cfg(feature = "hnsw")]
        {
            let index_guard = self.index.read().await;
            if let Some(ref index) = *index_guard {
                index.save_index(path.to_str().unwrap())
                    .map_err(|e| SearchError::storage(path.to_path_buf(), format!("Failed to save index: {}", e)))?;
            }
        }

        // Save metadata separately
        let metadata_path = path.with_extension("metadata");
        let metadata_guard = self.metadata.read().await;
        let mapping_guard = self.id_mapping.read().await;
        
        let index_data = IndexData {
            metadata: metadata_guard.clone(),
            id_mapping: mapping_guard.clone(),
            next_id: *self.next_id.read().await,
        };

        let serialized = bincode::serialize(&index_data)?;
        tokio::fs::write(&metadata_path, serialized).await
            .map_err(|e| SearchError::storage(metadata_path, e.to_string()))?;

        info!("Saved index to {:?}", path);
        Ok(())
    }

    /// Load index from disk
    pub async fn load(&self, path: &Path) -> Result<()> {
        #[cfg(feature = "hnsw")]
        {
            let mut index_guard = self.index.write().await;
            
            let index = HnswIndex::load_index(
                path.to_str().unwrap(),
                &hnswlib::Space::Cosine,
                self.config.dimension,
            ).map_err(|e| SearchError::storage(path.to_path_buf(), format!("Failed to load index: {}", e)))?;
            
            index.set_ef(self.config.ef_search)
                .map_err(|e| SearchError::index(format!("Failed to set ef: {}", e)))?;
            
            *index_guard = Some(index);
        }

        // Load metadata
        let metadata_path = path.with_extension("metadata");
        if metadata_path.exists() {
            let data = tokio::fs::read(&metadata_path).await
                .map_err(|e| SearchError::storage(metadata_path.clone(), e.to_string()))?;
            
            let index_data: IndexData = bincode::deserialize(&data)?;
            
            {
                let mut metadata_guard = self.metadata.write().await;
                *metadata_guard = index_data.metadata;
            }
            
            {
                let mut mapping_guard = self.id_mapping.write().await;
                *mapping_guard = index_data.id_mapping;
            }
            
            {
                let mut next_id_guard = self.next_id.write().await;
                *next_id_guard = index_data.next_id;
            }
        }

        info!("Loaded index from {:?}", path);
        Ok(())
    }

    /// Clear the entire index
    pub async fn clear(&self) -> Result<()> {
        {
            let mut metadata_guard = self.metadata.write().await;
            metadata_guard.clear();
        }
        
        {
            let mut mapping_guard = self.id_mapping.write().await;
            mapping_guard.clear();
        }
        
        {
            let mut next_id_guard = self.next_id.write().await;
            *next_id_guard = 0;
        }

        #[cfg(feature = "hnsw")]
        {
            let mut index_guard = self.index.write().await;
            *index_guard = None;
        }

        // Reinitialize
        self.initialize().await?;

        info!("Cleared vector index");
        Ok(())
    }
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub document_count: usize,
    pub index_size: usize,
    pub dimension: usize,
    pub max_elements: usize,
}

/// Serializable index data
#[derive(Debug, Serialize, Deserialize)]
struct IndexData {
    metadata: HashMap<DocumentId, DocumentMetadata>,
    id_mapping: HashMap<u32, DocumentId>,
    next_id: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[tokio::test]
    async fn test_index_creation() {
        let config = IndexConfig::default();
        let index = VectorIndex::new(config).unwrap();
        index.initialize().await.unwrap();

        let stats = index.stats().await;
        assert_eq!(stats.dimension, 384);
        assert_eq!(stats.document_count, 0);
    }

    #[tokio::test]
    async fn test_add_and_search() {
        let config = IndexConfig {
            dimension: 3,
            ..Default::default()
        };
        let index = VectorIndex::new(config).unwrap();
        index.initialize().await.unwrap();

        // Add a document
        let embedding = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let metadata = DocumentMetadata {
            id: "doc1".to_string(),
            preview: "Test document".to_string(),
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        };

        index.add_document("doc1".to_string(), embedding.clone(), metadata).await.unwrap();

        // Search for similar documents
        let query = Array1::from_vec(vec![0.9, 0.1, 0.0]); // Similar to the added document
        let results = index.search(query, 1).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document_id, "doc1");
        assert!(results[0].score > 0.8); // Should be quite similar
    }

    #[tokio::test]
    async fn test_metadata_operations() {
        let config = IndexConfig::default();
        let index = VectorIndex::new(config).unwrap();
        index.initialize().await.unwrap();

        let embedding = Array1::from_vec(vec![1.0; 384]);
        let metadata = DocumentMetadata {
            id: "doc1".to_string(),
            preview: "Test document".to_string(),
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        };

        index.add_document("doc1".to_string(), embedding, metadata.clone()).await.unwrap();

        // Get metadata
        let retrieved = index.get_metadata("doc1").await.unwrap();
        assert_eq!(retrieved.id, metadata.id);
        assert_eq!(retrieved.preview, metadata.preview);

        // Remove document
        let removed = index.remove_document("doc1").await.unwrap();
        assert!(removed);

        // Metadata should be gone
        let retrieved = index.get_metadata("doc1").await;
        assert!(retrieved.is_none());
    }
}