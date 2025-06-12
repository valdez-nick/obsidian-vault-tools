/*!
High-level web scraping orchestrator.
*/

use crate::{
    client::{WebClient, ClientConfig, WebResponse},
    sources::{SourceManager, SourceQuery, SourceResult, ContentSource},
    extractor::ExtractedContent,
    WebError, Result,
};
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};
use url::Url;

/// Configuration for web scraping operations
#[derive(Debug, Clone)]
pub struct ScrapingConfig {
    /// Maximum concurrent requests
    pub max_concurrent: usize,
    
    /// Request timeout per URL
    pub request_timeout: std::time::Duration,
    
    /// Whether to extract content automatically
    pub auto_extract: bool,
    
    /// Whether to follow redirects
    pub follow_redirects: bool,
    
    /// Maximum redirect depth
    pub max_redirects: usize,
    
    /// Client configuration
    pub client_config: ClientConfig,
    
    /// Retry configuration
    pub retry_config: RetryConfig,
}

impl Default for ScrapingConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 10,
            request_timeout: std::time::Duration::from_secs(30),
            auto_extract: true,
            follow_redirects: true,
            max_redirects: 5,
            client_config: ClientConfig::default(),
            retry_config: RetryConfig::default(),
        }
    }
}

/// Configuration for retry logic
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: usize,
    
    /// Base delay between retries
    pub base_delay: std::time::Duration,
    
    /// Maximum delay between retries
    pub max_delay: std::time::Duration,
    
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    
    /// Whether to retry on rate limit errors
    pub retry_on_rate_limit: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: std::time::Duration::from_millis(500),
            max_delay: std::time::Duration::from_secs(30),
            backoff_multiplier: 2.0,
            retry_on_rate_limit: true,
        }
    }
}

/// Result from a scraping operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapingResult {
    /// URL that was scraped
    pub url: String,
    
    /// Whether the scraping was successful
    pub success: bool,
    
    /// HTTP response (if successful)
    pub response: Option<WebResponse>,
    
    /// Extracted content (if auto_extract is enabled)
    pub content: Option<ExtractedContent>,
    
    /// Error message (if failed)
    pub error: Option<String>,
    
    /// Number of retry attempts made
    pub retry_attempts: usize,
    
    /// Total time taken
    pub duration: std::time::Duration,
    
    /// Metadata about the scraping process
    pub metadata: std::collections::HashMap<String, String>,
}

/// High-level web scraper
pub struct WebScraper {
    client: Arc<WebClient>,
    source_manager: Arc<SourceManager>,
    config: ScrapingConfig,
    semaphore: Arc<Semaphore>,
}

impl WebScraper {
    /// Create a new web scraper
    pub fn new(config: ScrapingConfig) -> Result<Self> {
        let client = Arc::new(WebClient::new(config.client_config.clone())?);
        let source_manager = Arc::new(SourceManager::new());
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
        
        Ok(Self {
            client,
            source_manager,
            config,
            semaphore,
        })
    }
    
    /// Scrape a single URL
    pub async fn scrape_url(&self, url: &Url) -> ScrapingResult {
        let start_time = std::time::Instant::now();
        let mut retry_attempts = 0;
        let mut last_error = None;
        
        for attempt in 0..self.config.retry_config.max_attempts {
            retry_attempts = attempt;
            
            match self.try_scrape_url(url).await {
                Ok((response, content)) => {
                    return ScrapingResult {
                        url: url.to_string(),
                        success: true,
                        response: Some(response),
                        content,
                        error: None,
                        retry_attempts,
                        duration: start_time.elapsed(),
                        metadata: std::collections::HashMap::new(),
                    };
                }
                Err(e) => {
                    last_error = Some(e.to_string());
                    
                    // Check if we should retry
                    if !self.should_retry(&e, attempt) {
                        break;
                    }
                    
                    // Calculate delay for next attempt
                    let delay = self.calculate_retry_delay(attempt);
                    debug!("Retrying {} after {:?} (attempt {})", url, delay, attempt + 1);
                    tokio::time::sleep(delay).await;
                }
            }
        }
        
        ScrapingResult {
            url: url.to_string(),
            success: false,
            response: None,
            content: None,
            error: last_error,
            retry_attempts,
            duration: start_time.elapsed(),
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// Scrape multiple URLs concurrently
    pub async fn scrape_urls(&self, urls: Vec<Url>) -> Vec<ScrapingResult> {
        info!("Starting batch scraping of {} URLs", urls.len());
        
        let results = stream::iter(urls)
            .map(|url| async move {
                let _permit = self.semaphore.acquire().await.unwrap();
                self.scrape_url(&url).await
            })
            .buffer_unordered(self.config.max_concurrent)
            .collect::<Vec<_>>()
            .await;
        
        let successful = results.iter().filter(|r| r.success).count();
        info!("Batch scraping completed: {}/{} successful", successful, results.len());
        
        results
    }
    
    /// Search and scrape content from multiple sources
    pub async fn search_and_scrape(&self, query: &SourceQuery) -> Result<Vec<ScrapingResult>> {
        info!("Searching and scraping for query: {}", query.text);
        
        // Search across all sources
        let search_results = self.source_manager.search_all(query).await?;
        
        // Convert search results to URLs
        let urls: Vec<Url> = search_results
            .into_iter()
            .filter_map(|result| Url::parse(&result.url).ok())
            .take(query.max_results.unwrap_or(50))
            .collect();
        
        debug!("Found {} URLs to scrape", urls.len());
        
        // Scrape all URLs
        let scraping_results = self.scrape_urls(urls).await;
        
        Ok(scraping_results)
    }
    
    /// Check if a URL is accessible without scraping
    pub async fn check_url(&self, url: &Url) -> bool {
        match self.client.check_accessibility(url).await {
            Ok(accessible) => accessible,
            Err(_) => false,
        }
    }
    
    /// Get scraper statistics
    pub async fn get_stats(&self) -> ScraperStats {
        let client_stats = self.client.get_stats().await;
        
        ScraperStats {
            client_stats,
            available_permits: self.semaphore.available_permits(),
            max_concurrent: self.config.max_concurrent,
        }
    }
    
    async fn try_scrape_url(&self, url: &Url) -> Result<(WebResponse, Option<ExtractedContent>)> {
        let response = self.client.fetch(url).await?;
        
        let content = if self.config.auto_extract {
            match self.client.fetch_and_extract(url).await {
                Ok(extracted) => Some(extracted),
                Err(e) => {
                    warn!("Content extraction failed for {}: {}", url, e);
                    None
                }
            }
        } else {
            None
        };
        
        Ok((response, content))
    }
    
    fn should_retry(&self, error: &WebError, attempt: usize) -> bool {
        if attempt >= self.config.retry_config.max_attempts - 1 {
            return false;
        }
        
        match error {
            WebError::Http(e) => {
                // Retry on temporary HTTP errors
                if let Some(status) = e.status() {
                    match status.as_u16() {
                        // Don't retry on client errors (4xx)
                        400..=499 => false,
                        // Retry on server errors (5xx)
                        500..=599 => true,
                        // Don't retry on other status codes
                        _ => false,
                    }
                } else {
                    // Retry on network errors
                    true
                }
            }
            WebError::Timeout(_) => true,
            WebError::RateLimit(_) => self.config.retry_config.retry_on_rate_limit,
            _ => false,
        }
    }
    
    fn calculate_retry_delay(&self, attempt: usize) -> std::time::Duration {
        let base_delay = self.config.retry_config.base_delay.as_millis() as f64;
        let multiplier = self.config.retry_config.backoff_multiplier;
        let delay_ms = base_delay * multiplier.powi(attempt as i32);
        
        let delay = std::time::Duration::from_millis(delay_ms as u64);
        delay.min(self.config.retry_config.max_delay)
    }
}

/// Statistics for the web scraper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScraperStats {
    /// Client statistics
    pub client_stats: crate::client::ClientStats,
    
    /// Available concurrency permits
    pub available_permits: usize,
    
    /// Maximum concurrent requests
    pub max_concurrent: usize,
}

/// Builder for creating a web scraper with custom configuration
pub struct WebScraperBuilder {
    config: ScrapingConfig,
}

impl WebScraperBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: ScrapingConfig::default(),
        }
    }
    
    /// Set maximum concurrent requests
    pub fn max_concurrent(mut self, max: usize) -> Self {
        self.config.max_concurrent = max;
        self
    }
    
    /// Set request timeout
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.config.request_timeout = timeout;
        self.config.client_config.timeout = timeout;
        self
    }
    
    /// Enable/disable automatic content extraction
    pub fn auto_extract(mut self, extract: bool) -> Self {
        self.config.auto_extract = extract;
        self
    }
    
    /// Set retry configuration
    pub fn retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.config.retry_config = retry_config;
        self
    }
    
    /// Set user agent
    pub fn user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        self.config.client_config.user_agent = user_agent.into();
        self
    }
    
    /// Build the web scraper
    pub fn build(self) -> Result<WebScraper> {
        WebScraper::new(self.config)
    }
}

impl Default for WebScraperBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_scraper_builder() {
        let scraper = WebScraperBuilder::new()
            .max_concurrent(5)
            .timeout(Duration::from_secs(10))
            .auto_extract(false)
            .user_agent("Test Agent")
            .build();
        
        assert!(scraper.is_ok());
        let scraper = scraper.unwrap();
        assert_eq!(scraper.config.max_concurrent, 5);
        assert_eq!(scraper.config.request_timeout, Duration::from_secs(10));
        assert!(!scraper.config.auto_extract);
    }
    
    #[test]
    fn test_retry_config() {
        let config = RetryConfig {
            max_attempts: 5,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            retry_on_rate_limit: true,
        };
        
        let scraping_config = ScrapingConfig {
            retry_config: config,
            ..Default::default()
        };
        
        let scraper = WebScraper::new(scraping_config).unwrap();
        
        // Test retry delay calculation
        let delay1 = scraper.calculate_retry_delay(0);
        let delay2 = scraper.calculate_retry_delay(1);
        let delay3 = scraper.calculate_retry_delay(2);
        
        assert!(delay2 > delay1);
        assert!(delay3 > delay2);
        assert!(delay3 <= Duration::from_secs(10)); // Max delay
    }
    
    #[tokio::test]
    async fn test_scraper_creation() {
        let config = ScrapingConfig::default();
        let scraper = WebScraper::new(config).unwrap();
        
        let stats = scraper.get_stats().await;
        assert_eq!(stats.available_permits, stats.max_concurrent);
    }
    
    #[test]
    fn test_should_retry_logic() {
        let config = ScrapingConfig::default();
        let scraper = WebScraper::new(config).unwrap();
        
        // Test HTTP errors
        let http_500 = WebError::generic("500 Internal Server Error");
        assert!(scraper.should_retry(&http_500, 0));
        assert!(!scraper.should_retry(&http_500, 3)); // Max attempts reached
        
        // Test timeout errors
        let timeout_error = WebError::timeout("Request timeout");
        assert!(scraper.should_retry(&timeout_error, 0));
        assert!(scraper.should_retry(&timeout_error, 1));
    }
}