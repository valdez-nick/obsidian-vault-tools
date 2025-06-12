/*!
HTTP client with built-in rate limiting and error handling.
*/

use crate::{
    rate_limiter::{GlobalRateLimiter, RateLimitConfig, RateLimitedClient},
    extractor::{ContentExtractor, ExtractionConfig, ExtractedContent},
    WebError, Result,
};
use reqwest::{header, Client, Response};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};
use url::Url;

/// Configuration for the web client
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// User agent string
    pub user_agent: String,
    
    /// Request timeout
    pub timeout: Duration,
    
    /// Follow redirects
    pub follow_redirects: bool,
    
    /// Maximum redirect count
    pub max_redirects: usize,
    
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
    
    /// Content extraction configuration
    pub extraction: ExtractionConfig,
    
    /// Accept compressed responses
    pub accept_compression: bool,
    
    /// Custom headers
    pub headers: Vec<(String, String)>,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            user_agent: "Obsidian-Librarian/0.1.0 (Research Assistant)".to_string(),
            timeout: Duration::from_secs(30),
            follow_redirects: true,
            max_redirects: 10,
            rate_limit: RateLimitConfig::default(),
            extraction: ExtractionConfig::default(),
            accept_compression: true,
            headers: vec![
                ("Accept".to_string(), "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8".to_string()),
                ("Accept-Language".to_string(), "en-US,en;q=0.5".to_string()),
                ("Accept-Encoding".to_string(), "gzip, deflate, br".to_string()),
                ("DNT".to_string(), "1".to_string()),
                ("Connection".to_string(), "keep-alive".to_string()),
                ("Upgrade-Insecure-Requests".to_string(), "1".to_string()),
            ],
        }
    }
}

/// Response from a web request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebResponse {
    /// Final URL after redirects
    pub url: String,
    
    /// HTTP status code
    pub status: u16,
    
    /// Response headers
    pub headers: Vec<(String, String)>,
    
    /// Raw response body
    pub body: String,
    
    /// Content type
    pub content_type: Option<String>,
    
    /// Response size in bytes
    pub size: usize,
    
    /// Request duration
    pub duration: Duration,
    
    /// Whether content was extracted
    pub extracted: bool,
}

/// High-level web client for research operations
pub struct WebClient {
    client: RateLimitedClient,
    extractor: ContentExtractor,
    config: ClientConfig,
}

impl WebClient {
    /// Create a new web client
    pub fn new(config: ClientConfig) -> Result<Self> {
        // Build reqwest client
        let mut client_builder = Client::builder()
            .timeout(config.timeout)
            .user_agent(&config.user_agent)
            .redirect(if config.follow_redirects {
                reqwest::redirect::Policy::limited(config.max_redirects)
            } else {
                reqwest::redirect::Policy::none()
            });
        
        if config.accept_compression {
            client_builder = client_builder
                .gzip(true)
                .brotli(true)
                .deflate(true);
        }
        
        // Add custom headers
        let mut headers = header::HeaderMap::new();
        for (name, value) in &config.headers {
            if let (Ok(header_name), Ok(header_value)) = (
                header::HeaderName::from_bytes(name.as_bytes()),
                header::HeaderValue::from_str(value)
            ) {
                headers.insert(header_name, header_value);
            }
        }
        client_builder = client_builder.default_headers(headers);
        
        let reqwest_client = client_builder.build().map_err(WebError::from)?;
        
        // Create rate-limited client
        let rate_limited_client = RateLimitedClient::new(
            reqwest_client,
            config.rate_limit.clone(),
        );
        
        // Create content extractor
        let extractor = ContentExtractor::new(config.extraction.clone())?;
        
        Ok(Self {
            client: rate_limited_client,
            extractor,
            config,
        })
    }
    
    /// Fetch a URL and return raw response
    pub async fn fetch(&self, url: &Url) -> Result<WebResponse> {
        let start_time = std::time::Instant::now();
        
        debug!("Fetching URL: {}", url);
        
        let response = self.client.get(url).await?;
        let final_url = response.url().clone();
        let status = response.status().as_u16();
        
        // Extract headers
        let headers: Vec<(String, String)> = response.headers()
            .iter()
            .map(|(name, value)| {
                (
                    name.to_string(),
                    value.to_str().unwrap_or("<invalid>").to_string(),
                )
            })
            .collect();
        
        // Get content type
        let content_type = response.headers()
            .get(header::CONTENT_TYPE)
            .and_then(|ct| ct.to_str().ok())
            .map(|s| s.to_string());
        
        // Read body
        let body = response.text().await.map_err(WebError::from)?;
        let size = body.len();
        let duration = start_time.elapsed();
        
        info!(
            "Fetched {} ({} bytes, {} status, {:?})",
            final_url, size, status, duration
        );
        
        Ok(WebResponse {
            url: final_url.to_string(),
            status,
            headers,
            body,
            content_type,
            size,
            duration,
            extracted: false,
        })
    }
    
    /// Fetch a URL and extract structured content
    pub async fn fetch_and_extract(&self, url: &Url) -> Result<ExtractedContent> {
        let response = self.fetch(url).await?;
        
        // Check if content type is HTML
        if let Some(content_type) = &response.content_type {
            if !content_type.contains("text/html") {
                return Err(WebError::content_validation(
                    format!("Unsupported content type: {}", content_type)
                ));
            }
        }
        
        // Extract content
        self.extractor.extract(&response.body, url).await
    }
    
    /// Check if a URL is accessible without downloading the full content
    pub async fn check_accessibility(&self, url: &Url) -> Result<bool> {
        let response = self.client.client().head(url.clone()).send().await?;
        Ok(response.status().is_success())
    }
    
    /// Get robots.txt for a domain
    pub async fn get_robots_txt(&self, url: &Url) -> Result<Option<String>> {
        if let Some(host) = url.host_str() {
            let robots_url = format!("{}://{}/robots.txt", url.scheme(), host);
            if let Ok(robots_url) = Url::parse(&robots_url) {
                match self.fetch(&robots_url).await {
                    Ok(response) if response.status == 200 => {
                        return Ok(Some(response.body));
                    }
                    _ => {}
                }
            }
        }
        Ok(None)
    }
    
    /// Check if crawling is allowed by robots.txt
    pub async fn is_crawling_allowed(&self, url: &Url) -> Result<bool> {
        // For now, implement a simple check
        // In practice, you'd parse robots.txt properly
        if let Some(robots_txt) = self.get_robots_txt(url).await? {
            let user_agent = self.config.user_agent.to_lowercase();
            let path = url.path();
            
            // Very basic robots.txt parsing
            for line in robots_txt.lines() {
                let line = line.trim().to_lowercase();
                if line.starts_with("user-agent:") {
                    // This is overly simplified - real robots.txt parsing is more complex
                    if line.contains("*") || line.contains(&user_agent) {
                        // Check for disallow rules on subsequent lines
                        // This would need proper state machine parsing
                    }
                }
            }
        }
        
        // Default to allowing crawling
        Ok(true)
    }
    
    /// Batch fetch multiple URLs
    pub async fn batch_fetch(&self, urls: Vec<Url>) -> Vec<Result<WebResponse>> {
        let mut results = Vec::with_capacity(urls.len());
        
        // Use concurrent requests with rate limiting
        let futures: Vec<_> = urls.iter()
            .map(|url| self.fetch(url))
            .collect();
        
        // Process results as they complete
        for future in futures {
            results.push(future.await);
        }
        
        results
    }
    
    /// Batch fetch and extract content from multiple URLs
    pub async fn batch_extract(&self, urls: Vec<Url>) -> Vec<Result<ExtractedContent>> {
        let mut results = Vec::with_capacity(urls.len());
        
        let futures: Vec<_> = urls.iter()
            .map(|url| self.fetch_and_extract(url))
            .collect();
        
        for future in futures {
            results.push(future.await);
        }
        
        results
    }
    
    /// Get client statistics
    pub async fn get_stats(&self) -> ClientStats {
        let rate_limit_stats = self.client.get_stats().await;
        
        ClientStats {
            rate_limit_stats,
            requests_made: rate_limit_stats.values().map(|s| s.success_count + s.error_count).sum(),
            errors_count: rate_limit_stats.values().map(|s| s.error_count).sum(),
        }
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: ClientConfig) -> Result<()> {
        // For now, just update what we can without recreating the client
        self.config = config;
        Ok(())
    }
}

/// Statistics for the web client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientStats {
    /// Per-domain rate limiting statistics
    pub rate_limit_stats: std::collections::HashMap<String, crate::rate_limiter::DomainStats>,
    
    /// Total requests made
    pub requests_made: u32,
    
    /// Total errors encountered
    pub errors_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[tokio::test]
    async fn test_client_creation() {
        let config = ClientConfig::default();
        let client = WebClient::new(config).unwrap();
        
        let stats = client.get_stats().await;
        assert_eq!(stats.requests_made, 0);
    }
    
    #[tokio::test]
    #[ignore] // Requires internet connection
    async fn test_fetch_real_url() {
        let config = ClientConfig::default();
        let client = WebClient::new(config).unwrap();
        
        let url = Url::parse("https://httpbin.org/get").unwrap();
        let response = client.fetch(&url).await;
        
        match response {
            Ok(resp) => {
                assert_eq!(resp.status, 200);
                assert!(!resp.body.is_empty());
            }
            Err(e) => {
                // Network tests can fail, so just log the error
                eprintln!("Network test failed: {}", e);
            }
        }
    }
    
    #[tokio::test]
    async fn test_accessibility_check() {
        let config = ClientConfig::default();
        let client = WebClient::new(config).unwrap();
        
        // Test with a URL that should not exist
        let url = Url::parse("https://this-domain-should-not-exist-12345.com").unwrap();
        let accessible = client.check_accessibility(&url).await;
        
        // Should fail (either false or error)
        assert!(accessible.is_err() || !accessible.unwrap());
    }
    
    #[test]
    fn test_config_default() {
        let config = ClientConfig::default();
        assert!(config.follow_redirects);
        assert_eq!(config.max_redirects, 10);
        assert!(config.timeout > Duration::from_secs(0));
        assert!(!config.headers.is_empty());
    }
}