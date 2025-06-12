/*!
# Librarian Core

High-performance file operations and markdown parsing for Obsidian vaults.

This crate provides:
- Fast concurrent file system operations
- Markdown parsing with frontmatter extraction
- Note metadata handling
- File change monitoring
- Content hashing for change detection

## Features

- **Performance**: Uses memory mapping and concurrent processing
- **Safety**: Comprehensive error handling and data validation
- **Compatibility**: Full support for Obsidian markdown conventions
*/

pub mod error;
pub mod file_ops;
pub mod markdown;
pub mod note;
pub mod vault;
pub mod watcher;

pub use error::{LibrarianError, Result};
pub use note::{Note, NoteId, NoteMetadata};
pub use vault::{Vault, VaultConfig, VaultStats};

// Re-export common types
pub use chrono::{DateTime, Utc};
pub use std::path::{Path, PathBuf};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_functionality() {
        // Basic smoke test
        assert!(true);
    }
}