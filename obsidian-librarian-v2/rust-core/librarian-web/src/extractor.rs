/*!
Content extraction from web pages and documents.
*/

use crate::{WebError, Result};
use ammonia::clean;
use html2text::from_read;
use readability::extractor;
use regex::Regex;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, warn};
use url::Url;

/// Configuration for content extraction
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Maximum content length in characters
    pub max_content_length: usize,
    
    /// Minimum content length in characters
    pub min_content_length: usize,
    
    /// Whether to preserve HTML formatting
    pub preserve_html: bool,
    
    /// Whether to extract metadata
    pub extract_metadata: bool,
    
    /// Whether to extract links
    pub extract_links: bool,
    
    /// Custom selectors for content extraction
    pub content_selectors: Vec<String>,
    
    /// Selectors to remove from content
    pub remove_selectors: Vec<String>,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            max_content_length: 50_000,
            min_content_length: 100,
            preserve_html: false,
            extract_metadata: true,
            extract_links: true,
            content_selectors: vec![
                "article".to_string(),
                "main".to_string(),
                ".content".to_string(),
                "#content".to_string(),
                ".post-content".to_string(),
                ".entry-content".to_string(),
            ],
            remove_selectors: vec![
                "nav".to_string(),
                "header".to_string(),
                "footer".to_string(),
                ".navigation".to_string(),
                ".sidebar".to_string(),
                ".advertisement".to_string(),
                ".ads".to_string(),
                "script".to_string(),
                "style".to_string(),
                "noscript".to_string(),
            ],
        }
    }
}

/// Extracted content from a web page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedContent {
    /// Main text content
    pub text: String,
    
    /// Page title
    pub title: Option<String>,
    
    /// Meta description
    pub description: Option<String>,
    
    /// Author information
    pub author: Option<String>,
    
    /// Publication date
    pub published_date: Option<chrono::DateTime<chrono::Utc>>,
    
    /// Language detected
    pub language: Option<String>,
    
    /// Content readability score (0.0 to 1.0)
    pub readability_score: f32,
    
    /// Extracted links
    pub links: Vec<ExtractedLink>,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    
    /// Original URL
    pub url: String,
    
    /// Content type (article, documentation, etc.)
    pub content_type: ContentType,
}

/// Information about an extracted link
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedLink {
    /// Link URL
    pub url: String,
    
    /// Link text
    pub text: String,
    
    /// Link type (internal, external, etc.)
    pub link_type: LinkType,
}

/// Type of extracted link
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LinkType {
    Internal,
    External,
    Anchor,
    Email,
    Phone,
}

/// Type of content detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentType {
    Article,
    Documentation,
    BlogPost,
    NewsArticle,
    TechnicalDoc,
    Tutorial,
    Reference,
    Forum,
    Unknown,
}

/// High-performance content extractor
pub struct ContentExtractor {
    config: ExtractionConfig,
    
    // Compiled regexes for performance
    whitespace_regex: Regex,
    url_regex: Regex,
    email_regex: Regex,
    phone_regex: Regex,
    
    // Compiled selectors
    content_selectors: Vec<Selector>,
    remove_selectors: Vec<Selector>,
}

impl ContentExtractor {
    /// Create a new content extractor
    pub fn new(config: ExtractionConfig) -> Result<Self> {
        // Compile regexes
        let whitespace_regex = Regex::new(r"\s+").map_err(|e| {
            WebError::config(format!("Failed to compile whitespace regex: {}", e))
        })?;
        
        let url_regex = Regex::new(r"https?://[^\s]+").map_err(|e| {
            WebError::config(format!("Failed to compile URL regex: {}", e))
        })?;
        
        let email_regex = Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").map_err(|e| {
            WebError::config(format!("Failed to compile email regex: {}", e))
        })?;
        
        let phone_regex = Regex::new(r"\+?[\d\s\-\(\)]{10,}").map_err(|e| {
            WebError::config(format!("Failed to compile phone regex: {}", e))
        })?;
        
        // Compile selectors
        let content_selectors: Result<Vec<Selector>> = config.content_selectors.iter()
            .map(|s| Selector::parse(s).map_err(|e| {
                WebError::config(format!("Failed to parse content selector '{}': {:?}", s, e))
            }))
            .collect();
        
        let remove_selectors: Result<Vec<Selector>> = config.remove_selectors.iter()
            .map(|s| Selector::parse(s).map_err(|e| {
                WebError::config(format!("Failed to parse remove selector '{}': {:?}", s, e))
            }))
            .collect();
        
        Ok(Self {
            config,
            whitespace_regex,
            url_regex,
            email_regex,
            phone_regex,
            content_selectors: content_selectors?,
            remove_selectors: remove_selectors?,
        })
    }
    
    /// Extract content from HTML
    pub async fn extract(&self, html: &str, url: &Url) -> Result<ExtractedContent> {
        debug!("Extracting content from {}", url);
        
        let document = Html::parse_document(html);
        
        // Remove unwanted elements
        let cleaned_html = self.clean_html(html)?;
        
        // Try readability extraction first
        let content = match self.extract_with_readability(&cleaned_html) {
            Ok(content) => content,
            Err(_) => {
                warn!("Readability extraction failed, falling back to manual extraction");
                self.extract_manually(&document)?
            }
        };
        
        // Extract metadata
        let metadata = if self.config.extract_metadata {
            self.extract_metadata(&document)
        } else {
            HashMap::new()
        };
        
        // Extract links
        let links = if self.config.extract_links {
            self.extract_links(&document, url)?
        } else {
            Vec::new()
        };
        
        // Detect content type
        let content_type = self.detect_content_type(&content, &metadata, url);
        
        // Calculate readability score
        let readability_score = self.calculate_readability(&content);
        
        // Clean and truncate content
        let cleaned_content = self.clean_text(&content)?;
        
        Ok(ExtractedContent {
            text: cleaned_content,
            title: self.extract_title(&document),
            description: self.extract_description(&document),
            author: self.extract_author(&document),
            published_date: self.extract_published_date(&document),
            language: self.detect_language(&content),
            readability_score,
            links,
            metadata,
            url: url.to_string(),
            content_type,
        })
    }
    
    fn clean_html(&self, html: &str) -> Result<String> {
        // Use ammonia to clean HTML and remove dangerous elements
        let cleaned = clean(html);
        
        // Parse with scraper to remove specified elements
        let document = Html::parse_document(&cleaned);
        let mut cleaned_html = cleaned;
        
        // Remove elements matching remove selectors
        for selector in &self.remove_selectors {
            // This is a simplified approach - in practice you'd need to
            // actually modify the DOM tree and serialize it back
            // For now, we'll just clean the text later
        }
        
        Ok(cleaned_html)
    }
    
    fn extract_with_readability(&self, html: &str) -> Result<String> {
        let extracted = extractor::extract(html, "")
            .map_err(|e| WebError::content_extraction(format!("Readability extraction failed: {}", e)))?;
        
        if extracted.text.len() < self.config.min_content_length {
            return Err(WebError::content_validation("Content too short"));
        }
        
        Ok(extracted.text)
    }
    
    fn extract_manually(&self, document: &Html) -> Result<String> {
        // Try content selectors in order
        for selector in &self.content_selectors {
            if let Some(element) = document.select(selector).next() {
                let text = self.element_to_text(element);
                if text.len() >= self.config.min_content_length {
                    return Ok(text);
                }
            }
        }
        
        // Fallback to body text
        if let Ok(body_selector) = Selector::parse("body") {
            if let Some(body) = document.select(&body_selector).next() {
                let text = self.element_to_text(body);
                if text.len() >= self.config.min_content_length {
                    return Ok(text);
                }
            }
        }
        
        Err(WebError::content_extraction("No suitable content found"))
    }
    
    fn element_to_text(&self, element: scraper::ElementRef) -> String {
        if self.config.preserve_html {
            element.html()
        } else {
            // Convert HTML to text
            let html = element.html();
            from_read(html.as_bytes(), 80)
        }
    }
    
    fn extract_title(&self, document: &Html) -> Option<String> {
        // Try various title selectors
        let title_selectors = [
            "h1",
            "title", 
            ".title",
            "#title",
            ".post-title",
            ".entry-title",
            "[property='og:title']",
        ];
        
        for selector_str in &title_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    let title = element.text().collect::<String>().trim().to_string();
                    if !title.is_empty() {
                        return Some(title);
                    }
                }
            }
        }
        
        None
    }
    
    fn extract_description(&self, document: &Html) -> Option<String> {
        // Try meta description
        if let Ok(selector) = Selector::parse("meta[name='description']") {
            if let Some(element) = document.select(&selector).next() {
                if let Some(content) = element.value().attr("content") {
                    return Some(content.to_string());
                }
            }
        }
        
        // Try og:description
        if let Ok(selector) = Selector::parse("meta[property='og:description']") {
            if let Some(element) = document.select(&selector).next() {
                if let Some(content) = element.value().attr("content") {
                    return Some(content.to_string());
                }
            }
        }
        
        None
    }
    
    fn extract_author(&self, document: &Html) -> Option<String> {
        let author_selectors = [
            "meta[name='author']",
            ".author",
            ".byline",
            "[rel='author']",
            "[property='article:author']",
        ];
        
        for selector_str in &author_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    let author = if let Some(content) = element.value().attr("content") {
                        content.to_string()
                    } else {
                        element.text().collect::<String>().trim().to_string()
                    };
                    
                    if !author.is_empty() {
                        return Some(author);
                    }
                }
            }
        }
        
        None
    }
    
    fn extract_published_date(&self, document: &Html) -> Option<chrono::DateTime<chrono::Utc>> {
        let date_selectors = [
            "meta[property='article:published_time']",
            "meta[name='date']",
            "time[datetime]",
            ".published",
            ".date",
        ];
        
        for selector_str in &date_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    let date_str = if let Some(datetime) = element.value().attr("datetime") {
                        datetime
                    } else if let Some(content) = element.value().attr("content") {
                        content
                    } else {
                        &element.text().collect::<String>()
                    };
                    
                    // Try to parse the date
                    if let Ok(date) = chrono::DateTime::parse_from_rfc3339(date_str) {
                        return Some(date.with_timezone(&chrono::Utc));
                    }
                    
                    // Try other common formats
                    if let Ok(date) = chrono::DateTime::parse_from_str(date_str, "%Y-%m-%d %H:%M:%S %z") {
                        return Some(date.with_timezone(&chrono::Utc));
                    }
                }
            }
        }
        
        None
    }
    
    fn extract_metadata(&self, document: &Html) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        
        // Extract meta tags
        if let Ok(selector) = Selector::parse("meta") {
            for element in document.select(&selector) {
                let attrs = element.value();
                if let Some(name) = attrs.attr("name") {
                    if let Some(content) = attrs.attr("content") {
                        metadata.insert(format!("meta:{}", name), content.to_string());
                    }
                }
                if let Some(property) = attrs.attr("property") {
                    if let Some(content) = attrs.attr("content") {
                        metadata.insert(format!("og:{}", property), content.to_string());
                    }
                }
            }
        }
        
        metadata
    }
    
    fn extract_links(&self, document: &Html, base_url: &Url) -> Result<Vec<ExtractedLink>> {
        let mut links = Vec::new();
        
        if let Ok(selector) = Selector::parse("a[href]") {
            for element in document.select(&selector) {
                if let Some(href) = element.value().attr("href") {
                    let text = element.text().collect::<String>().trim().to_string();
                    
                    let link_type = if href.starts_with('#') {
                        LinkType::Anchor
                    } else if href.starts_with("mailto:") {
                        LinkType::Email
                    } else if href.starts_with("tel:") {
                        LinkType::Phone
                    } else if let Ok(url) = base_url.join(href) {
                        if url.host() == base_url.host() {
                            LinkType::Internal
                        } else {
                            LinkType::External
                        }
                    } else {
                        continue; // Skip invalid URLs
                    };
                    
                    // Resolve relative URLs
                    let full_url = if href.starts_with("http") {
                        href.to_string()
                    } else {
                        base_url.join(href).map(|u| u.to_string()).unwrap_or_else(|_| href.to_string())
                    };
                    
                    links.push(ExtractedLink {
                        url: full_url,
                        text,
                        link_type,
                    });
                }
            }
        }
        
        Ok(links)
    }
    
    fn detect_content_type(&self, content: &str, metadata: &HashMap<String, String>, url: &Url) -> ContentType {
        let content_lower = content.to_lowercase();
        let url_str = url.to_string().to_lowercase();
        
        // Check metadata first
        if let Some(og_type) = metadata.get("og:type") {
            match og_type.as_str() {
                "article" => return ContentType::Article,
                _ => {}
            }
        }
        
        // Check URL patterns
        if url_str.contains("/docs/") || url_str.contains("/documentation/") {
            return ContentType::Documentation;
        }
        
        if url_str.contains("/blog/") || url_str.contains("/post/") {
            return ContentType::BlogPost;
        }
        
        if url_str.contains("/tutorial/") || url_str.contains("/guide/") {
            return ContentType::Tutorial;
        }
        
        if url_str.contains("/api/") || url_str.contains("/reference/") {
            return ContentType::Reference;
        }
        
        if url_str.contains("/forum/") || url_str.contains("/discussion/") {
            return ContentType::Forum;
        }
        
        // Check content patterns
        if content_lower.contains("tutorial") || content_lower.contains("step by step") {
            return ContentType::Tutorial;
        }
        
        if content_lower.contains("documentation") || content_lower.contains("api reference") {
            return ContentType::TechnicalDoc;
        }
        
        // Default classification
        if content.len() > 1000 {
            ContentType::Article
        } else {
            ContentType::Unknown
        }
    }
    
    fn calculate_readability(&self, content: &str) -> f32 {
        // Simple readability score based on sentence and word complexity
        let sentences = content.split(['.', '!', '?']).count() as f32;
        let words = content.split_whitespace().count() as f32;
        
        if sentences == 0.0 || words == 0.0 {
            return 0.0;
        }
        
        let avg_sentence_length = words / sentences;
        
        // Simple scoring: shorter sentences = higher readability
        let score = if avg_sentence_length < 15.0 {
            1.0 - (avg_sentence_length / 30.0)
        } else {
            0.5 - ((avg_sentence_length - 15.0) / 100.0).min(0.5)
        };
        
        score.max(0.0).min(1.0)
    }
    
    fn detect_language(&self, content: &str) -> Option<String> {
        // This is a placeholder - in practice you'd use a language detection library
        // For now, assume English for simplicity
        if content.len() > 100 {
            Some("en".to_string())
        } else {
            None
        }
    }
    
    fn clean_text(&self, content: &str) -> Result<String> {
        // Normalize whitespace
        let cleaned = self.whitespace_regex.replace_all(content, " ");
        
        // Trim to max length
        let truncated = if cleaned.len() > self.config.max_content_length {
            let mut truncated = cleaned.chars().take(self.config.max_content_length).collect::<String>();
            
            // Try to end at a word boundary
            if let Some(last_space) = truncated.rfind(' ') {
                truncated.truncate(last_space);
                truncated.push_str("...");
            }
            
            truncated
        } else {
            cleaned.to_string()
        };
        
        // Validate minimum length
        if truncated.len() < self.config.min_content_length {
            return Err(WebError::content_validation("Content too short after cleaning"));
        }
        
        Ok(truncated.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_content_extraction() {
        let config = ExtractionConfig::default();
        let extractor = ContentExtractor::new(config).unwrap();
        
        let html = r#"
            <!DOCTYPE html>
            <html>
            <head>
                <title>Test Article</title>
                <meta name="description" content="This is a test article">
                <meta name="author" content="Test Author">
            </head>
            <body>
                <nav>Navigation</nav>
                <article>
                    <h1>Test Article Title</h1>
                    <p>This is the main content of the article. It contains multiple sentences to test extraction.</p>
                    <p>This is another paragraph with <a href="https://example.com">external link</a> and <a href="/internal">internal link</a>.</p>
                </article>
                <footer>Footer content</footer>
            </body>
            </html>
        "#;
        
        let url = Url::parse("https://test.com/article").unwrap();
        let result = extractor.extract(html, &url).await.unwrap();
        
        assert!(result.text.len() > 100);
        assert_eq!(result.title, Some("Test Article Title".to_string()));
        assert_eq!(result.description, Some("This is a test article".to_string()));
        assert_eq!(result.author, Some("Test Author".to_string()));
        assert!(result.links.len() >= 2);
        assert!(matches!(result.content_type, ContentType::Article));
    }
    
    #[test]
    fn test_content_type_detection() {
        let config = ExtractionConfig::default();
        let extractor = ContentExtractor::new(config).unwrap();
        
        let doc_url = Url::parse("https://docs.example.com/api/reference").unwrap();
        let content_type = extractor.detect_content_type("API documentation", &HashMap::new(), &doc_url);
        assert!(matches!(content_type, ContentType::Documentation));
        
        let blog_url = Url::parse("https://example.com/blog/my-post").unwrap();
        let content_type = extractor.detect_content_type("This is a blog post", &HashMap::new(), &blog_url);
        assert!(matches!(content_type, ContentType::BlogPost));
    }
    
    #[test]
    fn test_readability_calculation() {
        let config = ExtractionConfig::default();
        let extractor = ContentExtractor::new(config).unwrap();
        
        let simple_text = "This is simple. It has short sentences. Easy to read.";
        let complex_text = "This is a very complex sentence with many clauses and subclauses that makes it difficult to understand and follow the main point being conveyed.";
        
        let simple_score = extractor.calculate_readability(simple_text);
        let complex_score = extractor.calculate_readability(complex_text);
        
        assert!(simple_score > complex_score);
        assert!(simple_score <= 1.0);
        assert!(complex_score >= 0.0);
    }
}