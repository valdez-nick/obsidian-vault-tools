/*!
Content source implementations for different websites and APIs.
*/

use crate::{WebError, Result, extractor::ExtractedContent};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, warn};
use url::Url;

/// Type of content source
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SourceType {
    /// General web search
    Web,
    /// GitHub repositories and issues
    GitHub,
    /// ArXiv research papers
    ArXiv,
    /// Documentation sites
    Documentation,
    /// Stack Overflow and similar Q&A sites
    QA,
    /// Academic papers and journals
    Academic,
    /// News and blog articles
    News,
    /// Technical forums
    Forum,
}

/// Search result from a content source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceResult {
    /// Result URL
    pub url: String,
    
    /// Title or heading
    pub title: String,
    
    /// Brief description or snippet
    pub snippet: String,
    
    /// Source-specific score (0.0 to 1.0)
    pub score: f32,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    
    /// Source type
    pub source_type: SourceType,
}

/// Search query for content sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceQuery {
    /// Query text
    pub text: String,
    
    /// Language preference
    pub language: Option<String>,
    
    /// Date range filter
    pub date_range: Option<DateRange>,
    
    /// File type filter
    pub file_type: Option<String>,
    
    /// Domain restrictions
    pub domains: Vec<String>,
    
    /// Maximum results
    pub max_results: Option<usize>,
}

/// Date range for filtering results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    pub start: chrono::DateTime<chrono::Utc>,
    pub end: chrono::DateTime<chrono::Utc>,
}

/// Trait for content sources
#[async_trait]
pub trait ContentSource: Send + Sync {
    /// Get the source type
    fn source_type(&self) -> SourceType;
    
    /// Get the source name
    fn name(&self) -> &str;
    
    /// Check if this source can handle the given query
    fn can_handle(&self, query: &SourceQuery) -> bool;
    
    /// Search for content
    async fn search(&self, query: &SourceQuery) -> Result<Vec<SourceResult>>;
    
    /// Extract content from a URL (if supported)
    async fn extract(&self, url: &Url) -> Result<Option<ExtractedContent>> {
        // Default implementation returns None
        Ok(None)
    }
    
    /// Get rate limit for this source (requests per second)
    fn rate_limit(&self) -> u32 {
        2 // Default 2 RPS
    }
}

/// GitHub content source
pub struct GitHubSource {
    api_token: Option<String>,
}

impl GitHubSource {
    pub fn new(api_token: Option<String>) -> Self {
        Self { api_token }
    }
}

#[async_trait]
impl ContentSource for GitHubSource {
    fn source_type(&self) -> SourceType {
        SourceType::GitHub
    }
    
    fn name(&self) -> &str {
        "GitHub"
    }
    
    fn can_handle(&self, query: &SourceQuery) -> bool {
        // Can handle any query, but prioritize code-related terms
        let code_terms = ["rust", "python", "javascript", "library", "framework", "api"];
        code_terms.iter().any(|term| query.text.to_lowercase().contains(term))
    }
    
    async fn search(&self, query: &SourceQuery) -> Result<Vec<SourceResult>> {
        debug!("Searching GitHub for: {}", query.text);
        
        // GitHub search API endpoint
        let search_url = format!(
            "https://api.github.com/search/repositories?q={}&sort=updated&order=desc",
            urlencoding::encode(&query.text)
        );
        
        let client = reqwest::Client::new();
        let mut request = client.get(&search_url)
            .header("User-Agent", "Obsidian-Librarian/0.1.0")
            .header("Accept", "application/vnd.github.v3+json");
        
        // Add auth token if available
        if let Some(token) = &self.api_token {
            request = request.header("Authorization", format!("token {}", token));
        }
        
        let response = request.send().await.map_err(WebError::from)?;
        
        if !response.status().is_success() {
            return Err(WebError::generic(format!(
                "GitHub API request failed: {}",
                response.status()
            )));
        }
        
        let search_response: GitHubSearchResponse = response.json().await.map_err(WebError::from)?;
        
        let mut results = Vec::new();
        let max_results = query.max_results.unwrap_or(20).min(search_response.items.len());
        
        for item in search_response.items.into_iter().take(max_results) {
            let mut metadata = HashMap::new();
            metadata.insert("stars".to_string(), item.stargazers_count.to_string());
            metadata.insert("language".to_string(), item.language.unwrap_or_default());
            metadata.insert("updated".to_string(), item.updated_at);
            
            results.push(SourceResult {
                url: item.html_url,
                title: item.full_name,
                snippet: item.description.unwrap_or_default(),
                score: self.calculate_github_score(&item),
                metadata,
                source_type: SourceType::GitHub,
            });
        }
        
        Ok(results)
    }
    
    fn rate_limit(&self) -> u32 {
        if self.api_token.is_some() {
            10 // Higher rate limit with auth
        } else {
            5 // Lower rate limit without auth
        }
    }
}

impl GitHubSource {
    fn calculate_github_score(&self, repo: &GitHubRepository) -> f32 {
        let stars = repo.stargazers_count as f32;
        let forks = repo.forks_count as f32;
        
        // Simple scoring based on popularity
        let popularity_score = (stars.ln() + forks.ln()) / 20.0;
        
        // Boost score for recently updated repos
        let days_since_update = chrono::Utc::now()
            .signed_duration_since(
                chrono::DateTime::parse_from_rfc3339(&repo.updated_at)
                    .unwrap_or_else(|_| chrono::Utc::now().into())
                    .with_timezone(&chrono::Utc)
            )
            .num_days();
        
        let recency_boost = if days_since_update < 30 {
            1.2
        } else if days_since_update < 365 {
            1.0
        } else {
            0.8
        };
        
        (popularity_score * recency_boost).min(1.0).max(0.0)
    }
}

/// ArXiv content source
pub struct ArXivSource;

#[async_trait]
impl ContentSource for ArXivSource {
    fn source_type(&self) -> SourceType {
        SourceType::ArXiv
    }
    
    fn name(&self) -> &str {
        "ArXiv"
    }
    
    fn can_handle(&self, query: &SourceQuery) -> bool {
        // Prioritize academic and research terms
        let academic_terms = ["research", "paper", "study", "analysis", "algorithm", "model"];
        academic_terms.iter().any(|term| query.text.to_lowercase().contains(term))
    }
    
    async fn search(&self, query: &SourceQuery) -> Result<Vec<SourceResult>> {
        debug!("Searching ArXiv for: {}", query.text);
        
        // ArXiv API endpoint
        let search_url = format!(
            "http://export.arxiv.org/api/query?search_query=all:{}&sortBy=lastUpdatedDate&sortOrder=descending&max_results={}",
            urlencoding::encode(&query.text),
            query.max_results.unwrap_or(20)
        );
        
        let client = reqwest::Client::new();
        let response = client.get(&search_url).send().await.map_err(WebError::from)?;
        
        if !response.status().is_success() {
            return Err(WebError::generic(format!(
                "ArXiv API request failed: {}",
                response.status()
            )));
        }
        
        let xml_content = response.text().await.map_err(WebError::from)?;
        
        // Parse ArXiv XML response
        self.parse_arxiv_response(&xml_content)
    }
    
    fn rate_limit(&self) -> u32 {
        3 // ArXiv recommends 3 seconds between requests
    }
}

impl ArXivSource {
    fn parse_arxiv_response(&self, xml: &str) -> Result<Vec<SourceResult>> {
        // This is a simplified XML parser - in practice you'd use a proper XML library
        let mut results = Vec::new();
        
        // Very basic XML parsing for demo purposes
        // In practice, use a proper XML parser like quick-xml
        for entry in xml.split("<entry>").skip(1) {
            if let Some(end) = entry.find("</entry>") {
                let entry_xml = &entry[..end];
                
                // Extract basic fields
                let title = self.extract_xml_field(entry_xml, "title").unwrap_or_default();
                let summary = self.extract_xml_field(entry_xml, "summary").unwrap_or_default();
                let link = self.extract_xml_link(entry_xml).unwrap_or_default();
                let published = self.extract_xml_field(entry_xml, "published").unwrap_or_default();
                
                if !title.is_empty() && !link.is_empty() {
                    let mut metadata = HashMap::new();
                    metadata.insert("published".to_string(), published);
                    metadata.insert("source".to_string(), "ArXiv".to_string());
                    
                    results.push(SourceResult {
                        url: link,
                        title: title.trim().to_string(),
                        snippet: summary.trim().chars().take(200).collect::<String>(),
                        score: 0.8, // ArXiv papers generally high quality
                        metadata,
                        source_type: SourceType::ArXiv,
                    });
                }
            }
        }
        
        Ok(results)
    }
    
    fn extract_xml_field(&self, xml: &str, field: &str) -> Option<String> {
        let start_tag = format!("<{}>", field);
        let end_tag = format!("</{}>", field);
        
        if let Some(start) = xml.find(&start_tag) {
            let content_start = start + start_tag.len();
            if let Some(end) = xml[content_start..].find(&end_tag) {
                return Some(xml[content_start..content_start + end].to_string());
            }
        }
        
        None
    }
    
    fn extract_xml_link(&self, xml: &str) -> Option<String> {
        // Look for <link href="..." />
        if let Some(link_start) = xml.find("<link") {
            if let Some(href_start) = xml[link_start..].find("href=\"") {
                let href_content_start = link_start + href_start + 6;
                if let Some(href_end) = xml[href_content_start..].find('"') {
                    return Some(xml[href_content_start..href_content_start + href_end].to_string());
                }
            }
        }
        
        None
    }
}

/// Documentation source for various doc sites
pub struct DocumentationSource {
    /// Prioritized documentation domains
    domains: Vec<String>,
}

impl DocumentationSource {
    pub fn new() -> Self {
        Self {
            domains: vec![
                "docs.python.org".to_string(),
                "doc.rust-lang.org".to_string(),
                "docs.microsoft.com".to_string(),
                "developer.mozilla.org".to_string(),
                "kubernetes.io".to_string(),
                "docker.com".to_string(),
            ],
        }
    }
}

#[async_trait]
impl ContentSource for DocumentationSource {
    fn source_type(&self) -> SourceType {
        SourceType::Documentation
    }
    
    fn name(&self) -> &str {
        "Documentation"
    }
    
    fn can_handle(&self, query: &SourceQuery) -> bool {
        // Handle documentation-related queries
        let doc_terms = ["documentation", "docs", "api", "tutorial", "guide", "reference"];
        doc_terms.iter().any(|term| query.text.to_lowercase().contains(term))
    }
    
    async fn search(&self, query: &SourceQuery) -> Result<Vec<SourceResult>> {
        debug!("Searching documentation sites for: {}", query.text);
        
        let mut results = Vec::new();
        
        // For now, use a simple Google site-specific search approach
        // In practice, you'd implement source-specific APIs
        for domain in &self.domains {
            let search_query = format!("site:{} {}", domain, query.text);
            
            // This is a placeholder - you'd implement actual search logic here
            // For demonstration, create a mock result
            let mut metadata = HashMap::new();
            metadata.insert("domain".to_string(), domain.clone());
            
            results.push(SourceResult {
                url: format!("https://{}/search?q={}", domain, urlencoding::encode(&query.text)),
                title: format!("{} Documentation", domain),
                snippet: format!("Documentation results for '{}' on {}", query.text, domain),
                score: 0.7,
                metadata,
                source_type: SourceType::Documentation,
            });
        }
        
        Ok(results)
    }
    
    fn rate_limit(&self) -> u32 {
        5 // Conservative rate limit for doc sites
    }
}

/// Source manager that coordinates multiple content sources
pub struct SourceManager {
    sources: Vec<Box<dyn ContentSource>>,
}

impl SourceManager {
    /// Create a new source manager with default sources
    pub fn new() -> Self {
        let sources: Vec<Box<dyn ContentSource>> = vec![
            Box::new(GitHubSource::new(None)),
            Box::new(ArXivSource),
            Box::new(DocumentationSource::new()),
        ];
        
        Self { sources }
    }
    
    /// Add a custom content source
    pub fn add_source(&mut self, source: Box<dyn ContentSource>) {
        self.sources.push(source);
    }
    
    /// Search across all applicable sources
    pub async fn search_all(&self, query: &SourceQuery) -> Result<Vec<SourceResult>> {
        let mut all_results = Vec::new();
        
        for source in &self.sources {
            if source.can_handle(query) {
                match source.search(query).await {
                    Ok(mut results) => {
                        all_results.append(&mut results);
                    }
                    Err(e) => {
                        warn!("Search failed for source {}: {}", source.name(), e);
                    }
                }
            }
        }
        
        // Sort by score descending
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(all_results)
    }
    
    /// Get all available sources
    pub fn get_sources(&self) -> Vec<&dyn ContentSource> {
        self.sources.iter().map(|s| s.as_ref()).collect()
    }
}

// GitHub API response types
#[derive(Debug, Deserialize)]
struct GitHubSearchResponse {
    items: Vec<GitHubRepository>,
}

#[derive(Debug, Deserialize)]
struct GitHubRepository {
    full_name: String,
    html_url: String,
    description: Option<String>,
    stargazers_count: u32,
    forks_count: u32,
    language: Option<String>,
    updated_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_source_types() {
        let github = GitHubSource::new(None);
        assert_eq!(github.source_type(), SourceType::GitHub);
        assert_eq!(github.name(), "GitHub");
        
        let arxiv = ArXivSource;
        assert_eq!(arxiv.source_type(), SourceType::ArXiv);
        assert_eq!(arxiv.name(), "ArXiv");
    }
    
    #[test]
    fn test_query_handling() {
        let github = GitHubSource::new(None);
        let arxiv = ArXivSource;
        
        let code_query = SourceQuery {
            text: "rust web framework".to_string(),
            language: None,
            date_range: None,
            file_type: None,
            domains: vec![],
            max_results: None,
        };
        
        let research_query = SourceQuery {
            text: "machine learning research paper".to_string(),
            language: None,
            date_range: None,
            file_type: None,
            domains: vec![],
            max_results: None,
        };
        
        assert!(github.can_handle(&code_query));
        assert!(arxiv.can_handle(&research_query));
    }
    
    #[tokio::test]
    async fn test_source_manager() {
        let manager = SourceManager::new();
        assert!(manager.get_sources().len() > 0);
        
        let query = SourceQuery {
            text: "test query".to_string(),
            language: None,
            date_range: None,
            file_type: None,
            domains: vec![],
            max_results: Some(5),
        };
        
        // This will fail without network, but tests the structure
        let _results = manager.search_all(&query).await;
    }
}