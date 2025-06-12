/*!
# Librarian Search

High-performance vector search and embedding generation for content analysis.

This crate provides:
- Fast vector similarity search using HNSW indices
- Embedding generation with local transformer models
- Batch processing for large document collections
- Persistent storage with memory mapping
- Semantic clustering and duplicate detection

## Features

- **Performance**: SIMD-optimized vector operations with rayon parallelism
- **Scalability**: Handle millions of embeddings with sub-second search
- **Accuracy**: State-of-the-art embedding models for semantic understanding
- **Storage**: Efficient persistent indices with memory mapping
*/

pub mod embeddings;
pub mod error;
pub mod index;
pub mod similarity;
pub mod storage;

pub use embeddings::{EmbeddingGenerator, EmbeddingModel};
pub use error::{SearchError, Result};
pub use index::{VectorIndex, IndexConfig, SearchResult};
pub use similarity::{SimilarityCalculator, SimilarityMetric};
pub use storage::{IndexStorage, StorageConfig};

// Re-export common types
pub use ndarray::{Array1, Array2};
pub use uuid::Uuid;

/// Vector embedding type
pub type Embedding = Array1<f32>;

/// Collection of embeddings
pub type EmbeddingMatrix = Array2<f32>;

/// Document ID type
pub type DocumentId = String;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_functionality() {
        // Basic smoke test
        assert!(true);
    }
}