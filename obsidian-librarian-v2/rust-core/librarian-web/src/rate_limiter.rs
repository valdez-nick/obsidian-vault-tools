/*!
Rate limiting for web scraping operations.
*/

use crate::{WebError, Result};
use governor::{Quota, RateLimiter, DefaultDirectRateLimiter};
use leaky_bucket::RateLimiter as LeakyBucketLimiter;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, warn};
use url::Url;

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Default requests per second
    pub default_rps: u32,
    
    /// Per-domain rate limits (domain -> requests per second)
    pub domain_limits: HashMap<String, u32>,
    
    /// Burst capacity multiplier
    pub burst_multiplier: u32,
    
    /// Enable adaptive rate limiting
    pub adaptive: bool,
    
    /// Respect robots.txt rate limits
    pub respect_robots_txt: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        let mut domain_limits = HashMap::new();
        domain_limits.insert("github.com".to_string(), 10);
        domain_limits.insert("api.github.com".to_string(), 5);
        domain_limits.insert("arxiv.org".to_string(), 3);
        domain_limits.insert("docs.python.org".to_string(), 5);
        domain_limits.insert("doc.rust-lang.org".to_string(), 5);
        
        Self {
            default_rps: 2,
            domain_limits,
            burst_multiplier: 3,
            adaptive: true,
            respect_robots_txt: true,
        }
    }
}

/// Rate limiter for a specific domain
#[derive(Debug)]
struct DomainLimiter {
    /// Token bucket rate limiter
    limiter: DefaultDirectRateLimiter,
    
    /// Leaky bucket for smoothing
    leaky_bucket: LeakyBucketLimiter,
    
    /// Current rate (requests per second)
    current_rps: u32,
    
    /// Last request timestamp
    last_request: std::time::Instant,
    
    /// Error count for adaptive limiting
    error_count: u32,
    
    /// Success count for adaptive limiting
    success_count: u32,
}

impl DomainLimiter {
    fn new(rps: u32) -> Self {
        let quota = Quota::per_second(NonZeroU32::new(rps).unwrap());
        let limiter = RateLimiter::direct(quota);
        
        let leaky_bucket = LeakyBucketLimiter::builder()
            .max(rps as usize * 2) // Burst capacity
            .tokens(rps as usize)
            .interval(Duration::from_millis(1000 / rps as u64))
            .build();
        
        Self {
            limiter,
            leaky_bucket,
            current_rps: rps,
            last_request: std::time::Instant::now(),
            error_count: 0,
            success_count: 0,
        }
    }
    
    async fn acquire(&mut self) -> Result<()> {
        // Use governor rate limiter first
        self.limiter.until_ready().await;
        
        // Then use leaky bucket for smoothing
        self.leaky_bucket.acquire_one().await;
        
        self.last_request = std::time::Instant::now();
        Ok(())
    }
    
    fn record_success(&mut self) {
        self.success_count += 1;
        self.error_count = self.error_count.saturating_sub(1);
    }
    
    fn record_error(&mut self) {
        self.error_count += 1;
        // Adaptive rate limiting: slow down on errors
        if self.error_count > 3 && self.current_rps > 1 {
            self.adjust_rate(self.current_rps / 2);
        }
    }
    
    fn adjust_rate(&mut self, new_rps: u32) {
        if new_rps != self.current_rps && new_rps > 0 {
            debug!("Adjusting rate limit to {} rps", new_rps);
            self.current_rps = new_rps;
            
            // Create new limiters with updated rate
            let quota = Quota::per_second(NonZeroU32::new(new_rps).unwrap());
            self.limiter = RateLimiter::direct(quota);
            
            self.leaky_bucket = LeakyBucketLimiter::builder()
                .max(new_rps as usize * 2)
                .tokens(new_rps as usize)
                .interval(Duration::from_millis(1000 / new_rps as u64))
                .build();
        }
    }
}

/// Global rate limiter managing per-domain limits
pub struct GlobalRateLimiter {
    config: RateLimitConfig,
    domain_limiters: Arc<RwLock<HashMap<String, DomainLimiter>>>,
}

impl GlobalRateLimiter {
    /// Create a new global rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            domain_limiters: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Wait for permission to make a request to the given URL
    pub async fn acquire(&self, url: &Url) -> Result<()> {
        let domain = self.extract_domain(url);
        
        // Get or create domain limiter
        let mut limiters = self.domain_limiters.write().await;
        let limiter = limiters.entry(domain.clone()).or_insert_with(|| {
            let rps = self.config.domain_limits.get(&domain)
                .copied()
                .unwrap_or(self.config.default_rps);
            DomainLimiter::new(rps)
        });
        
        limiter.acquire().await?;
        
        debug!("Rate limit acquired for domain: {}", domain);
        Ok(())
    }
    
    /// Record a successful request
    pub async fn record_success(&self, url: &Url) {
        let domain = self.extract_domain(url);
        let mut limiters = self.domain_limiters.write().await;
        if let Some(limiter) = limiters.get_mut(&domain) {
            limiter.record_success();
        }
    }
    
    /// Record a failed request (for adaptive rate limiting)
    pub async fn record_error(&self, url: &Url) {
        let domain = self.extract_domain(url);
        let mut limiters = self.domain_limiters.write().await;
        if let Some(limiter) = limiters.get_mut(&domain) {
            limiter.record_error();
        }
    }
    
    /// Get current rate limit for a domain
    pub async fn get_rate_limit(&self, url: &Url) -> u32 {
        let domain = self.extract_domain(url);
        let limiters = self.domain_limiters.read().await;
        limiters.get(&domain)
            .map(|l| l.current_rps)
            .unwrap_or(self.config.default_rps)
    }
    
    /// Manually adjust rate limit for a domain
    pub async fn adjust_rate_limit(&self, url: &Url, new_rps: u32) {
        let domain = self.extract_domain(url);
        let mut limiters = self.domain_limiters.write().await;
        if let Some(limiter) = limiters.get_mut(&domain) {
            limiter.adjust_rate(new_rps);
        }
    }
    
    /// Clear all rate limiters (useful for testing)
    pub async fn clear(&self) {
        let mut limiters = self.domain_limiters.write().await;
        limiters.clear();
    }
    
    /// Get statistics for all domains
    pub async fn get_stats(&self) -> HashMap<String, DomainStats> {
        let limiters = self.domain_limiters.read().await;
        limiters.iter().map(|(domain, limiter)| {
            let stats = DomainStats {
                current_rps: limiter.current_rps,
                error_count: limiter.error_count,
                success_count: limiter.success_count,
                last_request: limiter.last_request,
            };
            (domain.clone(), stats)
        }).collect()
    }
    
    fn extract_domain(&self, url: &Url) -> String {
        url.host_str()
            .unwrap_or("unknown")
            .to_lowercase()
    }
}

/// Statistics for a domain rate limiter
#[derive(Debug, Clone)]
pub struct DomainStats {
    pub current_rps: u32,
    pub error_count: u32,
    pub success_count: u32,
    pub last_request: std::time::Instant,
}

/// Rate limiter middleware for HTTP requests
pub struct RateLimitedClient {
    client: reqwest::Client,
    rate_limiter: Arc<GlobalRateLimiter>,
}

impl RateLimitedClient {
    /// Create a new rate-limited HTTP client
    pub fn new(client: reqwest::Client, config: RateLimitConfig) -> Self {
        let rate_limiter = Arc::new(GlobalRateLimiter::new(config));
        Self {
            client,
            rate_limiter,
        }
    }
    
    /// Make a rate-limited GET request
    pub async fn get(&self, url: &Url) -> Result<reqwest::Response> {
        self.rate_limiter.acquire(url).await?;
        
        let result = self.client.get(url.clone()).send().await;
        
        match &result {
            Ok(response) => {
                if response.status().is_success() {
                    self.rate_limiter.record_success(url).await;
                } else {
                    self.rate_limiter.record_error(url).await;
                }
            }
            Err(_) => {
                self.rate_limiter.record_error(url).await;
            }
        }
        
        result.map_err(WebError::from)
    }
    
    /// Make a rate-limited POST request
    pub async fn post(&self, url: &Url, body: reqwest::Body) -> Result<reqwest::Response> {
        self.rate_limiter.acquire(url).await?;
        
        let result = self.client.post(url.clone()).body(body).send().await;
        
        match &result {
            Ok(response) => {
                if response.status().is_success() {
                    self.rate_limiter.record_success(url).await;
                } else {
                    self.rate_limiter.record_error(url).await;
                }
            }
            Err(_) => {
                self.rate_limiter.record_error(url).await;
            }
        }
        
        result.map_err(WebError::from)
    }
    
    /// Get rate limiter statistics
    pub async fn get_stats(&self) -> HashMap<String, DomainStats> {
        self.rate_limiter.get_stats().await
    }
    
    /// Get underlying client
    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    
    #[tokio::test]
    async fn test_rate_limiter_basic() {
        let config = RateLimitConfig {
            default_rps: 5,
            ..Default::default()
        };
        
        let limiter = GlobalRateLimiter::new(config);
        let url = Url::parse("https://example.com").unwrap();
        
        let start = Instant::now();
        
        // Make several requests
        for _ in 0..3 {
            limiter.acquire(&url).await.unwrap();
        }
        
        let elapsed = start.elapsed();
        // Should take at least 400ms for 3 requests at 5 rps
        assert!(elapsed >= Duration::from_millis(300));
    }
    
    #[tokio::test]
    async fn test_domain_specific_limits() {
        let mut domain_limits = HashMap::new();
        domain_limits.insert("slow.example.com".to_string(), 1); // 1 rps
        domain_limits.insert("fast.example.com".to_string(), 10); // 10 rps
        
        let config = RateLimitConfig {
            default_rps: 5,
            domain_limits,
            ..Default::default()
        };
        
        let limiter = GlobalRateLimiter::new(config);
        
        let slow_url = Url::parse("https://slow.example.com").unwrap();
        let fast_url = Url::parse("https://fast.example.com").unwrap();
        
        // Fast domain should allow quick requests
        let start = Instant::now();
        limiter.acquire(&fast_url).await.unwrap();
        limiter.acquire(&fast_url).await.unwrap();
        let fast_elapsed = start.elapsed();
        
        // Slow domain should enforce stricter limits
        let start = Instant::now();
        limiter.acquire(&slow_url).await.unwrap();
        limiter.acquire(&slow_url).await.unwrap();
        let slow_elapsed = start.elapsed();
        
        assert!(slow_elapsed > fast_elapsed);
    }
    
    #[tokio::test]
    async fn test_adaptive_rate_limiting() {
        let config = RateLimitConfig {
            default_rps: 4,
            adaptive: true,
            ..Default::default()
        };
        
        let limiter = GlobalRateLimiter::new(config);
        let url = Url::parse("https://example.com").unwrap();
        
        // Record several errors to trigger adaptive limiting
        for _ in 0..5 {
            limiter.record_error(&url).await;
        }
        
        let rate = limiter.get_rate_limit(&url).await;
        assert!(rate < 4); // Should be reduced due to errors
    }
}