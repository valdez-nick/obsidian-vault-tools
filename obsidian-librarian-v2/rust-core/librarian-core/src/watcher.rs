/*!
High-performance file system watcher for vault changes.
*/

use crate::{LibrarianError, Result};
use notify::{
    Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

/// Types of file system events we care about
#[derive(Debug, Clone, PartialEq)]
pub enum VaultEvent {
    /// File was created
    Created { path: PathBuf },
    /// File was modified
    Modified { path: PathBuf },
    /// File was deleted
    Deleted { path: PathBuf },
    /// File was moved/renamed
    Moved { from: PathBuf, to: PathBuf },
}

/// Configuration for the file watcher
#[derive(Debug, Clone)]
pub struct WatcherConfig {
    /// Debounce duration to avoid duplicate events
    pub debounce_duration: Duration,
    /// Maximum number of events to buffer
    pub buffer_size: usize,
    /// Patterns to exclude from watching
    pub exclude_patterns: Vec<String>,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            debounce_duration: Duration::from_millis(500),
            buffer_size: 1000,
            exclude_patterns: vec![
                ".obsidian/workspace*".to_string(),
                ".obsidian/cache".to_string(),
                ".trash/".to_string(),
                "*.tmp".to_string(),
                "*.bak".to_string(),
            ],
        }
    }
}

/// High-performance file system watcher
pub struct VaultWatcher {
    config: WatcherConfig,
    vault_path: PathBuf,
    watcher: Option<RecommendedWatcher>,
    event_sender: Option<mpsc::UnboundedSender<VaultEvent>>,
    debounce_map: Arc<RwLock<HashMap<PathBuf, Instant>>>,
}

impl VaultWatcher {
    /// Create a new vault watcher
    pub fn new<P: AsRef<Path>>(vault_path: P, config: WatcherConfig) -> Self {
        Self {
            config,
            vault_path: vault_path.as_ref().to_path_buf(),
            watcher: None,
            event_sender: None,
            debounce_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start watching the vault
    pub async fn start(&mut self) -> Result<mpsc::UnboundedReceiver<VaultEvent>> {
        info!("Starting vault watcher for: {:?}", self.vault_path);

        let (tx, rx) = mpsc::unbounded_channel();
        self.event_sender = Some(tx.clone());

        // Clone necessary data for the watcher callback
        let vault_path = self.vault_path.clone();
        let debounce_duration = self.config.debounce_duration;
        let debounce_map = Arc::clone(&self.debounce_map);
        let exclude_patterns = self.config.exclude_patterns.clone();

        // Create the file system watcher
        let mut watcher = RecommendedWatcher::new(
            move |result: notify::Result<Event>| {
                let tx = tx.clone();
                let vault_path = vault_path.clone();
                let debounce_map = Arc::clone(&debounce_map);
                let exclude_patterns = exclude_patterns.clone();

                tokio::spawn(async move {
                    if let Err(e) = Self::handle_fs_event(
                        result,
                        tx,
                        vault_path,
                        debounce_duration,
                        debounce_map,
                        exclude_patterns,
                    )
                    .await
                    {
                        warn!("Error handling file system event: {}", e);
                    }
                });
            },
            Config::default(),
        )
        .map_err(LibrarianError::from)?;

        // Start watching the vault directory
        watcher
            .watch(&self.vault_path, RecursiveMode::Recursive)
            .map_err(LibrarianError::from)?;

        self.watcher = Some(watcher);

        info!("Vault watcher started successfully");
        Ok(rx)
    }

    /// Stop watching the vault
    pub fn stop(&mut self) -> Result<()> {
        if let Some(mut watcher) = self.watcher.take() {
            watcher
                .unwatch(&self.vault_path)
                .map_err(LibrarianError::from)?;
            info!("Vault watcher stopped");
        }

        self.event_sender = None;
        Ok(())
    }

    /// Handle a file system event
    async fn handle_fs_event(
        result: notify::Result<Event>,
        tx: mpsc::UnboundedSender<VaultEvent>,
        vault_path: PathBuf,
        debounce_duration: Duration,
        debounce_map: Arc<RwLock<HashMap<PathBuf, Instant>>>,
        exclude_patterns: Vec<String>,
    ) -> Result<()> {
        let event = result.map_err(LibrarianError::from)?;

        debug!("File system event: {:?}", event);

        // Process each path in the event
        for path in &event.paths {
            // Skip if not a markdown file
            if !Self::is_markdown_file(path) {
                continue;
            }

            // Skip if excluded
            if Self::is_excluded(path, &vault_path, &exclude_patterns) {
                continue;
            }

            // Check debouncing
            if !Self::should_process_event(path, debounce_duration, &debounce_map).await {
                continue;
            }

            // Convert to our event type
            let vault_event = match event.kind {
                EventKind::Create(_) => VaultEvent::Created {
                    path: path.clone(),
                },
                EventKind::Modify(_) => VaultEvent::Modified {
                    path: path.clone(),
                },
                EventKind::Remove(_) => VaultEvent::Deleted {
                    path: path.clone(),
                },
                _ => continue, // Ignore other event types
            };

            // Send the event
            if let Err(e) = tx.send(vault_event) {
                warn!("Failed to send vault event: {}", e);
            }
        }

        Ok(())
    }

    /// Check if we should process this event (debouncing)
    async fn should_process_event(
        path: &Path,
        debounce_duration: Duration,
        debounce_map: &Arc<RwLock<HashMap<PathBuf, Instant>>>,
    ) -> bool {
        let now = Instant::now();
        let path_buf = path.to_path_buf();

        // Check if we've seen this path recently
        {
            let debounce_map_read = debounce_map.read().await;
            if let Some(&last_seen) = debounce_map_read.get(&path_buf) {
                if now.duration_since(last_seen) < debounce_duration {
                    return false; // Skip this event
                }
            }
        }

        // Update the last seen time
        {
            let mut debounce_map_write = debounce_map.write().await;
            debounce_map_write.insert(path_buf, now);

            // Clean up old entries to prevent memory leaks
            let cutoff = now - debounce_duration * 2;
            debounce_map_write.retain(|_, &mut last_seen| last_seen > cutoff);
        }

        true
    }

    /// Check if a file is a markdown file
    fn is_markdown_file(path: &Path) -> bool {
        if let Some(extension) = path.extension() {
            let ext = extension.to_string_lossy().to_lowercase();
            ext == "md" || ext == "markdown"
        } else {
            false
        }
    }

    /// Check if a path should be excluded
    fn is_excluded(path: &Path, vault_path: &Path, exclude_patterns: &[String]) -> bool {
        // Get relative path
        let relative_path = match path.strip_prefix(vault_path) {
            Ok(rel_path) => rel_path,
            Err(_) => return true, // If we can't get relative path, exclude it
        };

        let path_str = relative_path.to_string_lossy();

        // Check against exclude patterns
        for pattern in exclude_patterns {
            if Self::matches_pattern(&path_str, pattern) {
                return true;
            }
        }

        false
    }

    /// Simple pattern matching (supports * wildcards)
    fn matches_pattern(path: &str, pattern: &str) -> bool {
        if pattern.contains('*') {
            // Simple wildcard matching
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                let prefix = parts[0];
                let suffix = parts[1];
                return path.starts_with(prefix) && path.ends_with(suffix);
            }
        }

        // Exact match or starts with for directory patterns
        path == pattern || (pattern.ends_with('/') && path.starts_with(pattern))
    }

    /// Get current debounce map size (for monitoring)
    pub async fn get_debounce_map_size(&self) -> usize {
        self.debounce_map.read().await.len()
    }

    /// Clear old entries from debounce map
    pub async fn cleanup_debounce_map(&self) {
        let cutoff = Instant::now() - self.config.debounce_duration * 2;
        let mut map = self.debounce_map.write().await;
        map.retain(|_, &mut last_seen| last_seen > cutoff);
    }
}

impl Drop for VaultWatcher {
    fn drop(&mut self) {
        if let Err(e) = self.stop() {
            warn!("Error stopping vault watcher: {}", e);
        }
    }
}

/// Batch processor for vault events
pub struct EventBatcher {
    batch_size: usize,
    batch_timeout: Duration,
    current_batch: Vec<VaultEvent>,
    last_batch_time: Instant,
}

impl EventBatcher {
    /// Create a new event batcher
    pub fn new(batch_size: usize, batch_timeout: Duration) -> Self {
        Self {
            batch_size,
            batch_timeout,
            current_batch: Vec::new(),
            last_batch_time: Instant::now(),
        }
    }

    /// Add an event to the current batch
    pub fn add_event(&mut self, event: VaultEvent) -> Option<Vec<VaultEvent>> {
        self.current_batch.push(event);

        // Check if we should flush the batch
        if self.should_flush() {
            self.flush_batch()
        } else {
            None
        }
    }

    /// Check if the batch should be flushed
    fn should_flush(&self) -> bool {
        self.current_batch.len() >= self.batch_size
            || self.last_batch_time.elapsed() >= self.batch_timeout
    }

    /// Flush the current batch
    pub fn flush_batch(&mut self) -> Option<Vec<VaultEvent>> {
        if self.current_batch.is_empty() {
            return None;
        }

        let batch = std::mem::take(&mut self.current_batch);
        self.last_batch_time = Instant::now();
        Some(batch)
    }

    /// Force flush any remaining events
    pub fn force_flush(&mut self) -> Option<Vec<VaultEvent>> {
        self.flush_batch()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_watcher_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = WatcherConfig::default();
        let watcher = VaultWatcher::new(temp_dir.path(), config);

        assert_eq!(watcher.vault_path, temp_dir.path());
    }

    #[test]
    fn test_pattern_matching() {
        assert!(VaultWatcher::matches_pattern(
            ".obsidian/workspace.json",
            ".obsidian/workspace*"
        ));
        assert!(VaultWatcher::matches_pattern("test.tmp", "*.tmp"));
        assert!(VaultWatcher::matches_pattern(
            ".trash/deleted.md",
            ".trash/"
        ));
        assert!(!VaultWatcher::matches_pattern("normal.md", "*.tmp"));
    }

    #[test]
    fn test_markdown_file_detection() {
        assert!(VaultWatcher::is_markdown_file(Path::new("test.md")));
        assert!(VaultWatcher::is_markdown_file(Path::new("test.markdown")));
        assert!(!VaultWatcher::is_markdown_file(Path::new("test.txt")));
        assert!(!VaultWatcher::is_markdown_file(Path::new("test")));
    }

    #[tokio::test]
    async fn test_event_batching() {
        let mut batcher = EventBatcher::new(3, Duration::from_millis(100));

        // Add events
        let event1 = VaultEvent::Created {
            path: PathBuf::from("test1.md"),
        };
        let event2 = VaultEvent::Modified {
            path: PathBuf::from("test2.md"),
        };

        assert!(batcher.add_event(event1).is_none());
        assert!(batcher.add_event(event2).is_none());

        // Third event should trigger batch
        let event3 = VaultEvent::Deleted {
            path: PathBuf::from("test3.md"),
        };
        let batch = batcher.add_event(event3);
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn test_debouncing() {
        let debounce_map = Arc::new(RwLock::new(HashMap::new()));
        let path = Path::new("test.md");
        let duration = Duration::from_millis(100);

        // First event should be processed
        assert!(
            VaultWatcher::should_process_event(path, duration, &debounce_map).await
        );

        // Second event immediately after should be skipped
        assert!(
            !VaultWatcher::should_process_event(path, duration, &debounce_map).await
        );

        // After waiting, should be processed again
        sleep(Duration::from_millis(150)).await;
        assert!(
            VaultWatcher::should_process_event(path, duration, &debounce_map).await
        );
    }
}