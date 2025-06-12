/*!
Vault management and coordination.
*/

use crate::{
    file_ops::{FileOps, VaultStats},
    note::{Note, NoteId},
    LibrarianError, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Configuration for vault operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultConfig {
    /// Path to the vault
    pub path: PathBuf,
    
    /// Patterns to exclude from processing
    pub exclude_patterns: Vec<String>,
    
    /// Patterns to include (file extensions)
    pub include_patterns: Vec<String>,
    
    /// Maximum file size to process (in bytes)
    pub max_file_size: u64,
    
    /// Enable caching
    pub enable_cache: bool,
    
    /// Cache size limit (number of notes)
    pub cache_size_limit: usize,
}

impl Default for VaultConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            exclude_patterns: vec![
                ".obsidian/workspace*".to_string(),
                ".obsidian/cache".to_string(),
                ".trash/".to_string(),
                "*.tmp".to_string(),
                "*.bak".to_string(),
            ],
            include_patterns: vec![
                "*.md".to_string(),
                "*.markdown".to_string(),
            ],
            max_file_size: 10 * 1024 * 1024, // 10MB
            enable_cache: true,
            cache_size_limit: 10_000,
        }
    }
}

/// High-performance vault manager
pub struct Vault {
    config: VaultConfig,
    file_ops: FileOps,
    cache: Arc<RwLock<HashMap<NoteId, Note>>>,
    stats: Arc<RwLock<VaultStats>>,
}

impl Vault {
    /// Create a new vault manager
    pub fn new(config: VaultConfig) -> Result<Self> {
        // Validate vault path
        if !config.path.exists() {
            return Err(LibrarianError::vault_not_found(&config.path));
        }

        if !config.path.is_dir() {
            return Err(LibrarianError::invalid_vault(
                "Vault path must be a directory"
            ));
        }

        // Check for .obsidian directory (optional but recommended)
        let obsidian_dir = config.path.join(".obsidian");
        if !obsidian_dir.exists() {
            warn!("No .obsidian directory found - this may not be an Obsidian vault");
        }

        Ok(Self {
            config,
            file_ops: FileOps::new(),
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(VaultStats::default())),
        })
    }

    /// Get vault configuration
    pub fn config(&self) -> &VaultConfig {
        &self.config
    }

    /// Get vault path
    pub fn path(&self) -> &Path {
        &self.config.path
    }

    /// Initialize vault by scanning all files
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing vault: {:?}", self.config.path);

        // Scan for all markdown files
        let files = self.file_ops.scan_vault(&self.config.path)?;
        info!("Found {} files to process", files.len());

        // Update stats
        let vault_stats = self.file_ops.calculate_stats(&self.config.path)?;
        {
            let mut stats = self.stats.write().await;
            *stats = vault_stats;
        }

        info!("Vault initialization complete: {} notes, {} folders", 
               self.get_stats().await.note_count,
               self.get_stats().await.folder_count);

        Ok(())
    }

    /// Read a note by path
    pub async fn read_note<P: AsRef<Path>>(&self, path: P) -> Result<Note> {
        let path = path.as_ref();
        
        // Create note ID from relative path
        let relative_path = path.strip_prefix(&self.config.path)
            .map_err(|_| LibrarianError::invalid_path(path))?;
        let note_id = NoteId::new(relative_path.to_string_lossy().to_string());

        // Check cache first
        if self.config.enable_cache {
            let cache = self.cache.read().await;
            if let Some(note) = cache.get(&note_id) {
                // Check if file has been modified
                if let Ok(metadata) = tokio::fs::metadata(path).await {
                    if let Ok(modified) = metadata.modified() {
                        if let Some(cached_modified) = note.metadata.modified_at {
                            if chrono::DateTime::from(modified) <= cached_modified {
                                debug!("Cache hit for note: {:?}", path);
                                return Ok(note.clone());
                            }
                        }
                    }
                }
            }
        }

        // Read from disk
        debug!("Reading note from disk: {:?}", path);
        let note = self.file_ops.read_note_async(path).await?;

        // Update cache
        if self.config.enable_cache {
            self.update_cache(note_id, note.clone()).await;
        }

        Ok(note)
    }

    /// Write a note to disk
    pub async fn write_note(&self, note: &Note) -> Result<()> {
        // Validate note path is within vault
        let absolute_path = if note.path.is_absolute() {
            note.path.clone()
        } else {
            self.config.path.join(&note.path)
        };

        if !absolute_path.starts_with(&self.config.path) {
            return Err(LibrarianError::invalid_path(&absolute_path));
        }

        // Write to disk
        self.file_ops.write_note_async(note).await?;

        // Update cache
        if self.config.enable_cache {
            self.update_cache(note.id.clone(), note.clone()).await;
        }

        Ok(())
    }

    /// Create a new note
    pub async fn create_note<P: AsRef<Path>>(
        &self,
        relative_path: P,
        content: String,
    ) -> Result<Note> {
        let relative_path = relative_path.as_ref();
        let absolute_path = self.config.path.join(relative_path);

        // Check if file already exists
        if absolute_path.exists() {
            return Err(LibrarianError::generic(
                format!("Note already exists: {:?}", absolute_path)
            ));
        }

        // Create note ID
        let note_id = NoteId::new(relative_path.to_string_lossy().to_string());

        // Create note
        let mut note = Note::new(note_id, absolute_path, content)?;

        // Parse content to extract metadata
        let parsed = self.file_ops.parser.parse_note(&note.content)?;
        note.metadata = parsed.metadata;
        note.links = parsed.links;
        note.tasks = parsed.tasks;

        // Write to disk
        self.write_note(&note).await?;

        Ok(note)
    }

    /// Delete a note (move to trash)
    pub async fn delete_note<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let path = path.as_ref();
        
        // Get note ID for cache removal
        if let Ok(relative_path) = path.strip_prefix(&self.config.path) {
            let note_id = NoteId::new(relative_path.to_string_lossy().to_string());
            
            // Remove from cache
            if self.config.enable_cache {
                let mut cache = self.cache.write().await;
                cache.remove(&note_id);
            }
        }

        // Move to trash
        self.file_ops.move_to_trash(path, &self.config.path).await
    }

    /// Move a note to a new location
    pub async fn move_note<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        old_path: P,
        new_path: Q,
    ) -> Result<()> {
        let old_path = old_path.as_ref();
        let new_path = new_path.as_ref();

        // Read the note first
        let mut note = self.read_note(old_path).await?;

        // Update note path
        let new_absolute_path = if new_path.is_absolute() {
            new_path.to_path_buf()
        } else {
            self.config.path.join(new_path)
        };

        note.path = new_absolute_path;

        // Create new note ID
        if let Ok(relative_path) = note.path.strip_prefix(&self.config.path) {
            let old_note_id = note.id.clone();
            note.id = NoteId::new(relative_path.to_string_lossy().to_string());

            // Remove old cache entry
            if self.config.enable_cache {
                let mut cache = self.cache.write().await;
                cache.remove(&old_note_id);
            }
        }

        // Write to new location
        self.write_note(&note).await?;

        // Delete old file
        tokio::fs::remove_file(old_path).await?;

        Ok(())
    }

    /// Get all notes in the vault
    pub async fn get_all_notes(&self) -> Result<Vec<Note>> {
        let files = self.file_ops.scan_vault(&self.config.path)?;
        let mut notes = Vec::new();

        for file_path in files {
            match self.read_note(&file_path).await {
                Ok(note) => notes.push(note),
                Err(e) => {
                    warn!("Failed to read note {:?}: {}", file_path, e);
                }
            }
        }

        Ok(notes)
    }

    /// Search notes by tag
    pub async fn find_notes_by_tag(&self, tag: &str) -> Result<Vec<Note>> {
        let all_notes = self.get_all_notes().await?;
        Ok(all_notes
            .into_iter()
            .filter(|note| note.metadata.tags.contains(tag))
            .collect())
    }

    /// Search notes by title pattern
    pub async fn find_notes_by_title(&self, pattern: &str) -> Result<Vec<Note>> {
        let pattern_lower = pattern.to_lowercase();
        let all_notes = self.get_all_notes().await?;
        
        Ok(all_notes
            .into_iter()
            .filter(|note| {
                if let Some(ref title) = note.metadata.title {
                    title.to_lowercase().contains(&pattern_lower)
                } else if let Some(name) = note.name() {
                    name.to_lowercase().contains(&pattern_lower)
                } else {
                    false
                }
            })
            .collect())
    }

    /// Get vault statistics
    pub async fn get_stats(&self) -> VaultStats {
        self.stats.read().await.clone()
    }

    /// Update cache with size limit enforcement
    async fn update_cache(&self, note_id: NoteId, note: Note) {
        let mut cache = self.cache.write().await;
        
        // Enforce cache size limit
        if cache.len() >= self.config.cache_size_limit {
            // Remove oldest entries (simple LRU would be better)
            let keys_to_remove: Vec<_> = cache.keys().take(cache.len() / 2).cloned().collect();
            for key in keys_to_remove {
                cache.remove(&key);
            }
        }

        cache.insert(note_id, note);
    }

    /// Clear cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        info!("Cache cleared");
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read().await;
        (cache.len(), self.config.cache_size_limit)
    }

    /// Refresh vault statistics
    pub async fn refresh_stats(&self) -> Result<()> {
        let vault_stats = self.file_ops.calculate_stats(&self.config.path)?;
        {
            let mut stats = self.stats.write().await;
            *stats = vault_stats;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_vault() -> (TempDir, Vault) {
        let temp_dir = TempDir::new().unwrap();
        let vault_path = temp_dir.path().to_path_buf();

        // Create .obsidian directory
        tokio::fs::create_dir_all(vault_path.join(".obsidian"))
            .await
            .unwrap();

        // Create test note
        tokio::fs::write(
            vault_path.join("test.md"),
            "# Test Note\n\nThis is a test.",
        )
        .await
        .unwrap();

        let config = VaultConfig {
            path: vault_path,
            ..Default::default()
        };

        let vault = Vault::new(config).unwrap();
        vault.initialize().await.unwrap();

        (temp_dir, vault)
    }

    #[tokio::test]
    async fn test_vault_creation() {
        let (_temp_dir, vault) = create_test_vault().await;
        
        let stats = vault.get_stats().await;
        assert_eq!(stats.note_count, 1);
    }

    #[tokio::test]
    async fn test_note_operations() {
        let (_temp_dir, vault) = create_test_vault().await;

        // Read existing note
        let note = vault.read_note("test.md").await.unwrap();
        assert!(note.content.contains("Test Note"));

        // Create new note
        let new_note = vault
            .create_note("new_note.md", "# New Note\n\nContent".to_string())
            .await
            .unwrap();
        assert_eq!(new_note.metadata.title, Some("New Note".to_string()));

        // Verify it exists
        let read_note = vault.read_note("new_note.md").await.unwrap();
        assert_eq!(read_note.id, new_note.id);
    }

    #[tokio::test]
    async fn test_note_search() {
        let (_temp_dir, vault) = create_test_vault().await;

        // Create note with tags
        vault
            .create_note(
                "tagged_note.md",
                "---\ntags: [rust, testing]\n---\n# Tagged Note".to_string(),
            )
            .await
            .unwrap();

        // Search by tag
        let notes = vault.find_notes_by_tag("rust").await.unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].metadata.title, Some("Tagged Note".to_string()));

        // Search by title
        let notes = vault.find_notes_by_title("Tagged").await.unwrap();
        assert_eq!(notes.len(), 1);
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let (_temp_dir, vault) = create_test_vault().await;

        // Read note (should cache it)
        let note1 = vault.read_note("test.md").await.unwrap();
        let (cache_size, _) = vault.get_cache_stats().await;
        assert_eq!(cache_size, 1);

        // Read again (should hit cache)
        let note2 = vault.read_note("test.md").await.unwrap();
        assert_eq!(note1.id, note2.id);

        // Clear cache
        vault.clear_cache().await;
        let (cache_size, _) = vault.get_cache_stats().await;
        assert_eq!(cache_size, 0);
    }
}