/*!
Markdown parsing and content extraction.
*/

use crate::{note::*, LibrarianError, Result};
use chrono::{DateTime, Utc};
use pulldown_cmark::{Event, Parser, Tag};
use regex::Regex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// Markdown parser with Obsidian-specific extensions
pub struct MarkdownParser {
    /// Regex for wiki links
    wiki_link_regex: Regex,
    /// Regex for tags
    tag_regex: Regex,
    /// Regex for tasks
    task_regex: Regex,
}

impl Default for MarkdownParser {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkdownParser {
    /// Create a new markdown parser
    pub fn new() -> Self {
        Self {
            wiki_link_regex: Regex::new(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]").unwrap(),
            tag_regex: Regex::new(r"#([a-zA-Z0-9_\-/]+)").unwrap(),
            task_regex: Regex::new(r"^(\s*)- \[([ x])\] (.+)$").unwrap(),
        }
    }

    /// Parse a complete note from markdown content
    pub fn parse_note(&self, content: &str) -> Result<ParsedContent> {
        let (frontmatter, body) = self.extract_frontmatter(content)?;
        
        let mut parsed = ParsedContent {
            frontmatter,
            body: body.to_string(),
            metadata: NoteMetadata::default(),
            links: Vec::new(),
            tasks: Vec::new(),
        };

        // Parse metadata from frontmatter
        self.parse_frontmatter_metadata(&mut parsed)?;
        
        // Extract content elements
        self.extract_wiki_links(&mut parsed)?;
        self.extract_tags(&mut parsed)?;
        self.extract_tasks(&mut parsed)?;
        self.extract_title(&mut parsed)?;

        Ok(parsed)
    }

    /// Extract frontmatter and body content
    fn extract_frontmatter(&self, content: &str) -> Result<(HashMap<String, Value>, &str)> {
        let content = content.trim();
        
        if !content.starts_with("---") {
            return Ok((HashMap::new(), content));
        }

        let mut lines = content.lines();
        lines.next(); // Skip opening ---

        let mut frontmatter_lines = Vec::new();
        let mut body_start = 0;

        for (i, line) in lines.enumerate() {
            if line.trim() == "---" {
                body_start = content
                    .lines()
                    .take(i + 2) // +2 for opening and closing ---
                    .map(|l| l.len() + 1) // +1 for newline
                    .sum::<usize>();
                break;
            }
            frontmatter_lines.push(line);
        }

        if body_start == 0 {
            // No closing ---, treat as regular content
            return Ok((HashMap::new(), content));
        }

        let frontmatter_text = frontmatter_lines.join("\n");
        let frontmatter: HashMap<String, Value> = if frontmatter_text.trim().is_empty() {
            HashMap::new()
        } else {
            serde_yaml::from_str(&frontmatter_text)
                .map_err(|e| LibrarianError::generic(format!("Invalid YAML frontmatter: {}", e)))?
        };

        let body = &content[body_start..];
        Ok((frontmatter, body))
    }

    /// Parse metadata from frontmatter
    fn parse_frontmatter_metadata(&self, parsed: &mut ParsedContent) -> Result<()> {
        let fm = &parsed.frontmatter;
        
        // Title
        if let Some(Value::String(title)) = fm.get("title") {
            parsed.metadata.title = Some(title.clone());
        }

        // Tags
        if let Some(tags) = fm.get("tags") {
            match tags {
                Value::String(tag) => {
                    parsed.metadata.tags.insert(tag.clone());
                }
                Value::Array(tags) => {
                    for tag in tags {
                        if let Value::String(tag_str) = tag {
                            parsed.metadata.tags.insert(tag_str.clone());
                        }
                    }
                }
                _ => {}
            }
        }

        // Aliases
        if let Some(aliases) = fm.get("aliases") {
            match aliases {
                Value::String(alias) => {
                    parsed.metadata.aliases.push(alias.clone());
                }
                Value::Array(aliases) => {
                    for alias in aliases {
                        if let Value::String(alias_str) = alias {
                            parsed.metadata.aliases.push(alias_str.clone());
                        }
                    }
                }
                _ => {}
            }
        }

        // Dates
        if let Some(Value::String(date_str)) = fm.get("created") {
            if let Ok(date) = DateTime::parse_from_rfc3339(date_str) {
                parsed.metadata.created_at = Some(date.with_timezone(&Utc));
            }
        }

        if let Some(Value::String(date_str)) = fm.get("modified") {
            if let Ok(date) = DateTime::parse_from_rfc3339(date_str) {
                parsed.metadata.modified_at = Some(date.with_timezone(&Utc));
            }
        }

        // Custom fields
        if let Some(Value::String(initiative)) = fm.get("initiative") {
            parsed.metadata.initiative = Some(initiative.clone());
        }

        if let Some(Value::String(product)) = fm.get("product") {
            parsed.metadata.product = Some(product.clone());
        }

        if let Some(Value::String(priority)) = fm.get("priority") {
            parsed.metadata.priority = Some(priority.clone());
        }

        // Meeting date
        if let Some(Value::String(date_str)) = fm.get("meeting-date") {
            if let Ok(date) = DateTime::parse_from_rfc3339(date_str) {
                parsed.metadata.meeting_date = Some(date.with_timezone(&Utc));
            }
        }

        Ok(())
    }

    /// Extract wiki links from content
    fn extract_wiki_links(&self, parsed: &mut ParsedContent) -> Result<()> {
        for (pos, capture) in self.wiki_link_regex.captures_iter(&parsed.body).enumerate() {
            let full_match = capture.get(0).unwrap();
            let target = capture.get(1).unwrap().as_str().to_string();
            let alias = capture.get(2).map(|m| m.as_str().to_string());
            let text = full_match.as_str().to_string();
            let position = full_match.start();

            parsed.links.push(WikiLink::new(target, alias, text, position));
        }

        Ok(())
    }

    /// Extract hashtags from content
    fn extract_tags(&self, parsed: &mut ParsedContent) -> Result<()> {
        for capture in self.tag_regex.captures_iter(&parsed.body) {
            if let Some(tag) = capture.get(1) {
                parsed.metadata.tags.insert(tag.as_str().to_string());
            }
        }

        Ok(())
    }

    /// Extract tasks from content
    fn extract_tasks(&self, parsed: &mut ParsedContent) -> Result<()> {
        for (line_num, line) in parsed.body.lines().enumerate() {
            if let Some(capture) = self.task_regex.captures(line) {
                let completed = capture.get(2).unwrap().as_str() == "x";
                let text = capture.get(3).unwrap().as_str().to_string();
                
                // Calculate position in document
                let position = parsed.body
                    .lines()
                    .take(line_num)
                    .map(|l| l.len() + 1)
                    .sum();

                let mut task = Task::new(text, completed, line_num + 1, position);
                task.extract_tags();
                
                parsed.tasks.push(task);
            }
        }

        Ok(())
    }

    /// Extract title from first heading if not in frontmatter
    fn extract_title(&self, parsed: &mut ParsedContent) -> Result<()> {
        if parsed.metadata.title.is_some() {
            return Ok(());
        }

        let parser = Parser::new(&parsed.body);
        for event in parser {
            match event {
                Event::Start(Tag::Heading(level, _, _)) if level == 1 => {
                    // This is an h1 heading, extract the text
                    // Note: This is simplified - a full implementation would
                    // need to collect text events until the heading ends
                }
                _ => {}
            }
        }

        // Alternative simple approach using regex
        let heading_regex = Regex::new(r"^# (.+)$").unwrap();
        for line in parsed.body.lines() {
            if let Some(capture) = heading_regex.captures(line) {
                if let Some(title) = capture.get(1) {
                    parsed.metadata.title = Some(title.as_str().trim().to_string());
                    break;
                }
            }
        }

        Ok(())
    }

    /// Convert content back to markdown with frontmatter
    pub fn to_markdown(&self, content: &ParsedContent) -> Result<String> {
        let mut result = String::new();

        // Add frontmatter if not empty
        if !content.frontmatter.is_empty() {
            result.push_str("---\n");
            let yaml = serde_yaml::to_string(&content.frontmatter)
                .map_err(|e| LibrarianError::generic(format!("Failed to serialize frontmatter: {}", e)))?;
            result.push_str(&yaml);
            result.push_str("---\n\n");
        }

        // Add body content
        result.push_str(&content.body);

        Ok(result)
    }
}

/// Parsed markdown content
#[derive(Debug, Clone)]
pub struct ParsedContent {
    /// Frontmatter as key-value pairs
    pub frontmatter: HashMap<String, Value>,
    /// Body content without frontmatter
    pub body: String,
    /// Extracted metadata
    pub metadata: NoteMetadata,
    /// Wiki links found in content
    pub links: Vec<WikiLink>,
    /// Tasks found in content
    pub tasks: Vec<Task>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frontmatter_extraction() {
        let content = r#"---
title: Test Note
tags: [test, sample]
---

# Test Content

This is a test note."#;

        let parser = MarkdownParser::new();
        let (frontmatter, body) = parser.extract_frontmatter(content).unwrap();
        
        assert_eq!(frontmatter.get("title").unwrap(), "Test Note");
        assert!(body.contains("# Test Content"));
    }

    #[test]
    fn test_wiki_link_extraction() {
        let content = "This links to [[Other Note]] and [[Target|Alias]].";
        
        let parser = MarkdownParser::new();
        let mut parsed = ParsedContent {
            frontmatter: HashMap::new(),
            body: content.to_string(),
            metadata: NoteMetadata::default(),
            links: Vec::new(),
            tasks: Vec::new(),
        };

        parser.extract_wiki_links(&mut parsed).unwrap();
        
        assert_eq!(parsed.links.len(), 2);
        assert_eq!(parsed.links[0].target, "Other Note");
        assert_eq!(parsed.links[1].target, "Target");
        assert_eq!(parsed.links[1].alias, Some("Alias".to_string()));
    }

    #[test]
    fn test_task_extraction() {
        let content = r#"- [x] Completed task
- [ ] Incomplete task #urgent
- [ ] Another task #project"#;

        let parser = MarkdownParser::new();
        let mut parsed = ParsedContent {
            frontmatter: HashMap::new(),
            body: content.to_string(),
            metadata: NoteMetadata::default(),
            links: Vec::new(),
            tasks: Vec::new(),
        };

        parser.extract_tasks(&mut parsed).unwrap();
        
        assert_eq!(parsed.tasks.len(), 3);
        assert!(parsed.tasks[0].completed);
        assert!(!parsed.tasks[1].completed);
        assert!(parsed.tasks[1].tags.contains("urgent"));
    }

    #[test]
    fn test_tag_extraction() {
        let content = "This note has #tags and #multiple/nested tags.";
        
        let parser = MarkdownParser::new();
        let mut parsed = ParsedContent {
            frontmatter: HashMap::new(),
            body: content.to_string(),
            metadata: NoteMetadata::default(),
            links: Vec::new(),
            tasks: Vec::new(),
        };

        parser.extract_tags(&mut parsed).unwrap();
        
        assert!(parsed.metadata.tags.contains("tags"));
        assert!(parsed.metadata.tags.contains("multiple/nested"));
    }
}