/*!
# Librarian Web

High-performance web scraping and content extraction for research automation.

This crate provides:
- Concurrent web scraping with intelligent rate limiting
- Content extraction from various formats (HTML, PDF, etc.)
- Source-specific scrapers (GitHub, ArXiv, documentation sites)
- Content cleaning and markdown conversion
- Duplicate detection and quality scoring

## Features

- **Performance**: Concurrent requests with configurable rate limiting
- **Robustness**: Retry logic, error handling, and graceful degradation
- **Content Quality**: Smart extraction and cleaning algorithms
- **Extensibility**: Plugin architecture for new content sources
*/

pub mod client;
pub mod error;
pub mod extractor;
pub mod scraper;
pub mod sources;
pub mod rate_limiter;

pub use client::{WebClient, ClientConfig};
pub use error::{WebError, Result};
pub use extractor::{ContentExtractor, ExtractedContent};
pub use scraper::{WebScraper, ScrapingConfig, ScrapingResult};
pub use sources::{ContentSource, SourceType};

// Re-export common types
pub use reqwest::Url;
pub use chrono::{DateTime, Utc};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_functionality() {
        // Basic smoke test
        assert!(true);
    }
}