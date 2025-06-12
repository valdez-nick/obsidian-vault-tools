/*!
Error types for the librarian-search crate.
*/

use thiserror::Error;
use std::path::PathBuf;

/// Main error type for search operations
#[derive(Error, Debug)]
pub enum SearchError {
    /// IO errors from file operations
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Model loading errors
    #[error("Failed to load model: {reason}")]
    ModelLoad { reason: String },

    /// Embedding generation errors
    #[error("Failed to generate embedding: {reason}")]
    EmbeddingGeneration { reason: String },

    /// Index errors
    #[error("Index error: {reason}")]
    Index { reason: String },

    /// Search errors
    #[error("Search failed: {reason}")]
    Search { reason: String },

    /// Storage errors
    #[error("Storage error at {path}: {reason}")]
    Storage { path: PathBuf, reason: String },

    /// Dimension mismatch errors
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    /// Generic errors with context
    #[error("Operation failed: {context}")]
    Generic { context: String },
}

/// Convenience result type
pub type Result<T> = std::result::Result<T, SearchError>;

impl SearchError {
    /// Create a new model loading error
    pub fn model_load<S: Into<String>>(reason: S) -> Self {
        Self::ModelLoad {
            reason: reason.into(),
        }
    }

    /// Create a new embedding generation error
    pub fn embedding_generation<S: Into<String>>(reason: S) -> Self {
        Self::EmbeddingGeneration {
            reason: reason.into(),
        }
    }

    /// Create a new index error
    pub fn index<S: Into<String>>(reason: S) -> Self {
        Self::Index {
            reason: reason.into(),
        }
    }

    /// Create a new search error
    pub fn search<S: Into<String>>(reason: S) -> Self {
        Self::Search {
            reason: reason.into(),
        }
    }

    /// Create a new storage error
    pub fn storage<P: Into<PathBuf>, S: Into<String>>(path: P, reason: S) -> Self {
        Self::Storage {
            path: path.into(),
            reason: reason.into(),
        }
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create a configuration error
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::Config(msg.into())
    }

    /// Create a generic error with context
    pub fn generic<S: Into<String>>(context: S) -> Self {
        Self::Generic {
            context: context.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_error_creation() {
        let err = SearchError::model_load("test model loading error");
        assert!(matches!(err, SearchError::ModelLoad { .. }));
    }

    #[test]
    fn test_error_display() {
        let err = SearchError::config("test config error");
        assert_eq!(err.to_string(), "Configuration error: test config error");
    }

    #[test]
    fn test_dimension_mismatch() {
        let err = SearchError::dimension_mismatch(384, 512);
        assert!(err.to_string().contains("384"));
        assert!(err.to_string().contains("512"));
    }
}