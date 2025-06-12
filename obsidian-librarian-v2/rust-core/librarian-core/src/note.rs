/*!
Note data structures and operations.
*/

use crate::{LibrarianError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// Unique identifier for a note
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NoteId(String);

impl NoteId {
    /// Create a new note ID from a path relative to vault root
    pub fn new<S: Into<String>>(id: S) -> Self {
        Self(id.into())
    }

    /// Get the string representation
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for NoteId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for NoteId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Metadata extracted from a note
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteMetadata {
    /// Title of the note (from frontmatter or first heading)
    pub title: Option<String>,
    
    /// Tags associated with the note
    pub tags: HashSet<String>,
    
    /// Aliases for the note
    pub aliases: Vec<String>,
    
    /// Creation date
    pub created_at: Option<DateTime<Utc>>,
    
    /// Last modification date
    pub modified_at: Option<DateTime<Utc>>,
    
    /// Custom frontmatter fields
    pub frontmatter: HashMap<String, serde_json::Value>,
    
    /// Initiative/project metadata
    pub initiative: Option<String>,
    
    /// Product area
    pub product: Option<String>,
    
    /// Meeting date (for meeting notes)
    pub meeting_date: Option<DateTime<Utc>>,
    
    /// Priority level
    pub priority: Option<String>,
}

impl Default for NoteMetadata {
    fn default() -> Self {
        Self {
            title: None,
            tags: HashSet::new(),
            aliases: Vec::new(),
            created_at: None,
            modified_at: None,
            frontmatter: HashMap::new(),
            initiative: None,
            product: None,
            meeting_date: None,
            priority: None,
        }
    }
}

/// A wiki-style link found in markdown content
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WikiLink {
    /// The target of the link
    pub target: String,
    
    /// Optional alias for display
    pub alias: Option<String>,
    
    /// Full text of the link as it appears in the document
    pub text: String,
    
    /// Character position in the document
    pub position: usize,
}

impl WikiLink {
    /// Create a new wiki link
    pub fn new(target: String, alias: Option<String>, text: String, position: usize) -> Self {
        Self {
            target,
            alias,
            text,
            position,
        }
    }

    /// Get the display text for this link
    pub fn display_text(&self) -> &str {
        self.alias.as_deref().unwrap_or(&self.target)
    }
}

/// A task found in markdown content
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    /// Task text content
    pub text: String,
    
    /// Whether the task is completed
    pub completed: bool,
    
    /// Line number in the document
    pub line: usize,
    
    /// Character position in the document
    pub position: usize,
    
    /// Tags associated with this task
    pub tags: HashSet<String>,
    
    /// Due date if specified
    pub due_date: Option<DateTime<Utc>>,
}

impl Task {
    /// Create a new task
    pub fn new(text: String, completed: bool, line: usize, position: usize) -> Self {
        Self {
            text,
            completed,
            line,
            position,
            tags: HashSet::new(),
            due_date: None,
        }
    }

    /// Extract tags from task text
    pub fn extract_tags(&mut self) {
        let tag_regex = regex::Regex::new(r"#([a-zA-Z0-9_\-/]+)").unwrap();
        for cap in tag_regex.captures_iter(&self.text) {
            if let Some(tag) = cap.get(1) {
                self.tags.insert(tag.as_str().to_string());
            }
        }
    }
}

/// Complete note structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    /// Unique identifier
    pub id: NoteId,
    
    /// File path relative to vault root
    pub path: PathBuf,
    
    /// Raw markdown content
    pub content: String,
    
    /// Content hash for change detection
    pub content_hash: String,
    
    /// Extracted metadata
    pub metadata: NoteMetadata,
    
    /// Wiki links found in the content
    pub links: Vec<WikiLink>,
    
    /// Tasks found in the content
    pub tasks: Vec<Task>,
    
    /// Word count
    pub word_count: usize,
    
    /// File size in bytes
    pub file_size: u64,
}

impl Note {
    /// Create a new note
    pub fn new(id: NoteId, path: PathBuf, content: String) -> Result<Self> {
        let content_hash = Self::calculate_hash(&content);
        let word_count = Self::count_words(&content);
        let file_size = content.len() as u64;

        Ok(Self {
            id,
            path,
            content,
            content_hash,
            metadata: NoteMetadata::default(),
            links: Vec::new(),
            tasks: Vec::new(),
            word_count,
            file_size,
        })
    }

    /// Calculate content hash using BLAKE3
    pub fn calculate_hash(content: &str) -> String {
        hex::encode(blake3::hash(content.as_bytes()).as_bytes())
    }

    /// Count words in content
    pub fn count_words(content: &str) -> usize {
        content
            .split_whitespace()
            .filter(|word| !word.is_empty())
            .count()
    }

    /// Check if content has changed
    pub fn has_changed(&self, new_content: &str) -> bool {
        self.content_hash != Self::calculate_hash(new_content)
    }

    /// Get the note name without extension
    pub fn name(&self) -> Option<&str> {
        self.path.file_stem()?.to_str()
    }

    /// Get the note's directory
    pub fn directory(&self) -> Option<&std::path::Path> {
        self.path.parent()
    }

    /// Update content and recalculate derived fields
    pub fn update_content(&mut self, new_content: String) -> Result<()> {
        self.content = new_content;
        self.content_hash = Self::calculate_hash(&self.content);
        self.word_count = Self::count_words(&self.content);
        self.file_size = self.content.len() as u64;
        
        // Clear extracted data that needs to be recalculated
        self.links.clear();
        self.tasks.clear();
        
        Ok(())
    }

    /// Convert to a JSON representation
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(LibrarianError::from)
    }

    /// Create from JSON representation
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(LibrarianError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_note_creation() {
        let id = NoteId::new("test-note");
        let path = PathBuf::from("test-note.md");
        let content = "# Test Note\n\nThis is a test.".to_string();
        
        let note = Note::new(id.clone(), path.clone(), content).unwrap();
        
        assert_eq!(note.id, id);
        assert_eq!(note.path, path);
        assert_eq!(note.word_count, 5);
    }

    #[test]
    fn test_content_hash() {
        let content1 = "Hello, world!";
        let content2 = "Hello, world!";
        let content3 = "Hello, universe!";
        
        assert_eq!(Note::calculate_hash(content1), Note::calculate_hash(content2));
        assert_ne!(Note::calculate_hash(content1), Note::calculate_hash(content3));
    }

    #[test]
    fn test_word_count() {
        assert_eq!(Note::count_words(""), 0);
        assert_eq!(Note::count_words("hello"), 1);
        assert_eq!(Note::count_words("hello world"), 2);
        assert_eq!(Note::count_words("  hello   world  "), 2);
    }

    #[test]
    fn test_wiki_link() {
        let link = WikiLink::new(
            "target".to_string(),
            Some("alias".to_string()),
            "[[target|alias]]".to_string(),
            0,
        );
        
        assert_eq!(link.display_text(), "alias");
    }

    #[test]
    fn test_task_creation() {
        let mut task = Task::new("Complete #project review".to_string(), false, 1, 0);
        task.extract_tags();
        
        assert!(task.tags.contains("project"));
        assert!(!task.completed);
    }
}