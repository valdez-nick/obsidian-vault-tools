/*!
Error types for the librarian-core crate.
*/

use std::path::PathBuf;
use thiserror::Error;

/// Main error type for librarian-core operations
#[derive(Error, Debug)]
pub enum LibrarianError {
    /// IO errors from file operations
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// YAML frontmatter parsing errors
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// UTF-8 conversion errors
    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    /// File path errors
    #[error("Invalid path: {path}")]
    InvalidPath { path: PathBuf },

    /// Vault not found or inaccessible
    #[error("Vault not found or inaccessible: {path}")]
    VaultNotFound { path: PathBuf },

    /// Invalid vault structure
    #[error("Invalid vault structure: {reason}")]
    InvalidVault { reason: String },

    /// Note parsing errors
    #[error("Failed to parse note {path}: {reason}")]
    NoteParsing { path: PathBuf, reason: String },

    /// File watching errors
    #[error("File watcher error: {0}")]
    Watcher(#[from] notify::Error),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Generic errors with context
    #[error("Operation failed: {context}")]
    Generic { context: String },
}

/// Convenience result type
pub type Result<T> = std::result::Result<T, LibrarianError>;

impl LibrarianError {
    /// Create a new configuration error
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::Config(msg.into())
    }

    /// Create a new generic error with context
    pub fn generic<S: Into<String>>(context: S) -> Self {
        Self::Generic {
            context: context.into(),
        }
    }

    /// Create an invalid path error
    pub fn invalid_path<P: Into<PathBuf>>(path: P) -> Self {
        Self::InvalidPath { path: path.into() }
    }

    /// Create a vault not found error
    pub fn vault_not_found<P: Into<PathBuf>>(path: P) -> Self {
        Self::VaultNotFound { path: path.into() }
    }

    /// Create an invalid vault error
    pub fn invalid_vault<S: Into<String>>(reason: S) -> Self {
        Self::InvalidVault {
            reason: reason.into(),
        }
    }

    /// Create a note parsing error
    pub fn note_parsing<P: Into<PathBuf>, S: Into<String>>(path: P, reason: S) -> Self {
        Self::NoteParsing {
            path: path.into(),
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_error_creation() {
        let path = Path::new("/test/path");
        let err = LibrarianError::invalid_path(path);
        assert!(matches!(err, LibrarianError::InvalidPath { .. }));
    }

    #[test]
    fn test_error_display() {
        let err = LibrarianError::config("test config error");
        assert_eq!(err.to_string(), "Configuration error: test config error");
    }
}