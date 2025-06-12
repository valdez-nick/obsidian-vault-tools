/*!
High-performance file operations for vault management.
*/

use crate::{
    markdown::{MarkdownParser, ParsedContent},
    note::{Note, NoteId, NoteMetadata},
    LibrarianError, Result,
};
use chrono::{DateTime, Utc};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tokio::fs as async_fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, info, warn};
use walkdir::WalkDir;

/// High-performance file operations manager
pub struct FileOps {
    parser: MarkdownParser,
}

impl Default for FileOps {
    fn default() -> Self {
        Self::new()
    }
}

impl FileOps {
    /// Create a new file operations manager
    pub fn new() -> Self {
        Self {
            parser: MarkdownParser::new(),
        }
    }

    /// Read a note from disk asynchronously
    pub async fn read_note_async<P: AsRef<Path>>(&self, path: P) -> Result<Note> {
        let path = path.as_ref();
        let path_buf = path.to_path_buf();
        
        debug!("Reading note: {:?}", path);

        // Read file content
        let content = async_fs::read_to_string(path).await
            .map_err(|e| LibrarianError::note_parsing(
                path_buf.clone(),
                format!("Failed to read file: {}", e)
            ))?;

        // Get file metadata
        let metadata = async_fs::metadata(path).await?;
        let modified_at = metadata.modified()
            .map(|time| DateTime::from(time))
            .unwrap_or_else(|_| Utc::now());
        let created_at = metadata.created()
            .map(|time| DateTime::from(time))
            .unwrap_or_else(|_| Utc::now());

        // Create note ID from path
        let note_id = NoteId::new(path_buf.to_string_lossy().to_string());

        // Parse content
        let parsed = self.parser.parse_note(&content)
            .map_err(|e| LibrarianError::note_parsing(
                path_buf.clone(),
                format!("Failed to parse markdown: {}", e)
            ))?;

        // Create note
        let mut note = Note::new(note_id, path_buf, content)?;
        note.metadata = parsed.metadata;
        note.metadata.created_at = Some(created_at);
        note.metadata.modified_at = Some(modified_at);
        note.links = parsed.links;
        note.tasks = parsed.tasks;
        note.file_size = metadata.len();

        Ok(note)
    }

    /// Read a note from disk synchronously using memory mapping for large files
    pub fn read_note_sync<P: AsRef<Path>>(&self, path: P) -> Result<Note> {
        let path = path.as_ref();
        let path_buf = path.to_path_buf();
        
        debug!("Reading note (sync): {:?}", path);

        // Get file metadata
        let metadata = fs::metadata(path)?;
        let file_size = metadata.len();

        // Read content efficiently based on file size
        let content = if file_size > 1024 * 1024 {
            // Use memory mapping for large files (>1MB)
            self.read_large_file(path)?
        } else {
            // Regular read for small files
            fs::read_to_string(path)
                .map_err(|e| LibrarianError::note_parsing(
                    path_buf.clone(),
                    format!("Failed to read file: {}", e)
                ))?
        };

        let modified_at = metadata.modified()
            .map(|time| DateTime::from(time))
            .unwrap_or_else(|_| Utc::now());
        let created_at = metadata.created()
            .map(|time| DateTime::from(time))
            .unwrap_or_else(|_| Utc::now());

        // Create note ID from path
        let note_id = NoteId::new(path_buf.to_string_lossy().to_string());

        // Parse content
        let parsed = self.parser.parse_note(&content)
            .map_err(|e| LibrarianError::note_parsing(
                path_buf.clone(),
                format!("Failed to parse markdown: {}", e)
            ))?;

        // Create note
        let mut note = Note::new(note_id, path_buf, content)?;
        note.metadata = parsed.metadata;
        note.metadata.created_at = Some(created_at);
        note.metadata.modified_at = Some(modified_at);
        note.links = parsed.links;
        note.tasks = parsed.tasks;
        note.file_size = file_size;

        Ok(note)
    }

    /// Read large file using memory mapping
    fn read_large_file<P: AsRef<Path>>(&self, path: P) -> Result<String> {
        let file = fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        std::str::from_utf8(&mmap)
            .map(|s| s.to_string())
            .map_err(LibrarianError::from)
    }

    /// Write a note to disk asynchronously
    pub async fn write_note_async(&self, note: &Note) -> Result<()> {
        debug!("Writing note: {:?}", note.path);

        // Create parent directory if it doesn't exist
        if let Some(parent) = note.path.parent() {
            async_fs::create_dir_all(parent).await?;
        }

        // Create parsed content structure
        let parsed = ParsedContent {
            frontmatter: self.create_frontmatter(&note.metadata)?,
            body: note.content.clone(),
            metadata: note.metadata.clone(),
            links: note.links.clone(),
            tasks: note.tasks.clone(),
        };

        // Convert to markdown
        let markdown = self.parser.to_markdown(&parsed)?;

        // Write to file
        let mut file = async_fs::File::create(&note.path).await?;
        file.write_all(markdown.as_bytes()).await?;
        file.flush().await?;

        info!("Wrote note: {:?}", note.path);
        Ok(())
    }

    /// Create frontmatter from note metadata
    fn create_frontmatter(&self, metadata: &NoteMetadata) -> Result<HashMap<String, serde_json::Value>> {
        let mut frontmatter = metadata.frontmatter.clone();

        // Add standard fields
        if let Some(ref title) = metadata.title {
            frontmatter.insert("title".to_string(), serde_json::Value::String(title.clone()));
        }

        if !metadata.tags.is_empty() {
            let tags: Vec<_> = metadata.tags.iter().cloned().collect();
            frontmatter.insert("tags".to_string(), serde_json::Value::Array(
                tags.into_iter().map(serde_json::Value::String).collect()
            ));
        }

        if !metadata.aliases.is_empty() {
            frontmatter.insert("aliases".to_string(), serde_json::Value::Array(
                metadata.aliases.iter().map(|s| serde_json::Value::String(s.clone())).collect()
            ));
        }

        if let Some(ref initiative) = metadata.initiative {
            frontmatter.insert("initiative".to_string(), serde_json::Value::String(initiative.clone()));
        }

        if let Some(ref product) = metadata.product {
            frontmatter.insert("product".to_string(), serde_json::Value::String(product.clone()));
        }

        if let Some(ref priority) = metadata.priority {
            frontmatter.insert("priority".to_string(), serde_json::Value::String(priority.clone()));
        }

        Ok(frontmatter)
    }

    /// Scan directory for markdown files
    pub fn scan_vault<P: AsRef<Path>>(&self, vault_path: P) -> Result<Vec<PathBuf>> {
        let vault_path = vault_path.as_ref();
        let mut files = Vec::new();

        debug!("Scanning vault: {:?}", vault_path);

        for entry in WalkDir::new(vault_path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            
            // Skip directories
            if !path.is_file() {
                continue;
            }

            // Check if it's a markdown file
            if let Some(extension) = path.extension() {
                if extension == "md" || extension == "markdown" {
                    // Skip excluded patterns
                    if self.should_exclude(path, vault_path) {
                        continue;
                    }
                    
                    files.push(path.to_path_buf());
                }
            }
        }

        info!("Found {} markdown files", files.len());
        Ok(files)
    }

    /// Check if a file should be excluded
    fn should_exclude(&self, file_path: &Path, vault_path: &Path) -> bool {
        // Get relative path
        let relative_path = match file_path.strip_prefix(vault_path) {
            Ok(path) => path,
            Err(_) => return true, // If we can't get relative path, exclude it
        };

        let path_str = relative_path.to_string_lossy();

        // Exclude Obsidian system files
        if path_str.starts_with(".obsidian/") {
            return true;
        }

        // Exclude trash
        if path_str.starts_with(".trash/") {
            return true;
        }

        // Exclude temporary files
        if path_str.ends_with(".tmp") || path_str.ends_with(".bak") {
            return true;
        }

        // Exclude workspace files
        if path_str.contains("workspace") {
            return true;
        }

        false
    }

    /// Create backup of a file
    pub async fn create_backup<P: AsRef<Path>>(&self, file_path: P) -> Result<PathBuf> {
        let file_path = file_path.as_ref();
        
        if !file_path.exists() {
            return Err(LibrarianError::invalid_path(file_path));
        }

        let timestamp = Utc::now().format("%Y%m%d-%H%M%S");
        let backup_path = file_path.with_extension(
            format!("bak-{}.md", timestamp)
        );

        async_fs::copy(file_path, &backup_path).await?;
        
        info!("Created backup: {:?} -> {:?}", file_path, backup_path);
        Ok(backup_path)
    }

    /// Move a file to trash (create .trash directory)
    pub async fn move_to_trash<P: AsRef<Path>>(&self, file_path: P, vault_path: P) -> Result<PathBuf> {
        let file_path = file_path.as_ref();
        let vault_path = vault_path.as_ref();
        
        let trash_dir = vault_path.join(".trash");
        async_fs::create_dir_all(&trash_dir).await?;

        let timestamp = Utc::now().format("%Y%m%d-%H%M%S");
        let file_name = file_path.file_name()
            .ok_or_else(|| LibrarianError::invalid_path(file_path))?;
        
        let trash_path = trash_dir.join(format!("{}-{}", timestamp, file_name.to_string_lossy()));
        
        async_fs::rename(file_path, &trash_path).await?;
        
        info!("Moved to trash: {:?} -> {:?}", file_path, trash_path);
        Ok(trash_path)
    }

    /// Calculate directory statistics
    pub fn calculate_stats<P: AsRef<Path>>(&self, vault_path: P) -> Result<VaultStats> {
        let vault_path = vault_path.as_ref();
        let mut stats = VaultStats::default();

        for entry in WalkDir::new(vault_path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            
            if path.is_dir() {
                stats.folder_count += 1;
            } else if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension == "md" || extension == "markdown" {
                        if !self.should_exclude(path, vault_path) {
                            stats.note_count += 1;
                            
                            if let Ok(metadata) = fs::metadata(path) {
                                stats.total_size += metadata.len();
                            }
                        }
                    }
                }
            }
        }

        Ok(stats)
    }
}

/// Vault statistics
#[derive(Debug, Default, Clone)]
pub struct VaultStats {
    pub note_count: usize,
    pub folder_count: usize,
    pub total_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio;

    #[tokio::test]
    async fn test_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.md");
        
        // Create test content
        let content = r#"---
title: Test Note
tags: [test]
---

# Test Note

This is a test note with [[links]] and #tags."#;

        tokio::fs::write(&file_path, content).await.unwrap();

        let file_ops = FileOps::new();
        let note = file_ops.read_note_async(&file_path).await.unwrap();

        assert_eq!(note.metadata.title, Some("Test Note".to_string()));
        assert!(note.metadata.tags.contains("test"));
        assert!(note.metadata.tags.contains("tags"));
        assert_eq!(note.links.len(), 1);
        assert_eq!(note.links[0].target, "links");
    }

    #[test]
    fn test_vault_scanning() {
        let temp_dir = TempDir::new().unwrap();
        let vault_path = temp_dir.path();
        
        // Create test files
        std::fs::create_dir_all(vault_path.join("notes")).unwrap();
        std::fs::write(vault_path.join("notes/test1.md"), "# Test 1").unwrap();
        std::fs::write(vault_path.join("notes/test2.md"), "# Test 2").unwrap();
        std::fs::write(vault_path.join("notes/test.txt"), "Not markdown").unwrap();

        let file_ops = FileOps::new();
        let files = file_ops.scan_vault(vault_path).unwrap();

        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|p| p.file_name().unwrap() == "test1.md"));
        assert!(files.iter().any(|p| p.file_name().unwrap() == "test2.md"));
    }

    #[test]
    fn test_exclusion_patterns() {
        let file_ops = FileOps::new();
        let vault_path = Path::new("/vault");
        
        assert!(file_ops.should_exclude(
            Path::new("/vault/.obsidian/workspace.json"),
            vault_path
        ));
        
        assert!(file_ops.should_exclude(
            Path::new("/vault/.trash/deleted.md"),
            vault_path
        ));
        
        assert!(!file_ops.should_exclude(
            Path::new("/vault/notes/test.md"),
            vault_path
        ));
    }
}