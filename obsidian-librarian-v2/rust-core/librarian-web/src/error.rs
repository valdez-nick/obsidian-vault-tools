/*! 
Error types for web scraping operations.
*/

use std::fmt;
use thiserror::Error;

/// Result type for web operations
pub type Result<T> = std::result::Result<T, WebError>;

/// Errors that can occur during web operations
#[derive(Error, Debug)]
pub enum WebError {
    /// HTTP request errors
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    
    /// URL parsing errors  
    #[error("Invalid URL: {0}")]
    InvalidUrl(#[from] url::ParseError),
    
    /// Content extraction errors
    #[error("Content extraction failed: {0}")]
    ContentExtraction(String),
    
    /// Rate limiting errors
    #[error("Rate limited: {0}")]
    RateLimit(String),
    
    /// Timeout errors
    #[error("Request timeout: {0}")]
    Timeout(String),
    
    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// JSON parsing errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    /// HTML parsing errors  
    #[error("HTML parsing failed: {0}")]
    HtmlParsing(String),
    
    /// Content validation errors
    #[error("Content validation failed: {0}")]
    ContentValidation(String),
    
    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// Generic errors
    #[error("Web scraping error: {0}")]
    Generic(String),
}

impl WebError {
    /// Create a content extraction error
    pub fn content_extraction<S: Into<String>>(msg: S) -> Self {
        Self::ContentExtraction(msg.into())
    }
    
    /// Create a rate limit error
    pub fn rate_limit<S: Into<String>>(msg: S) -> Self {
        Self::RateLimit(msg.into())
    }
    
    /// Create a timeout error
    pub fn timeout<S: Into<String>>(msg: S) -> Self {
        Self::Timeout(msg.into())
    }
    
    /// Create an HTML parsing error
    pub fn html_parsing<S: Into<String>>(msg: S) -> Self {
        Self::HtmlParsing(msg.into())
    }
    
    /// Create a content validation error
    pub fn content_validation<S: Into<String>>(msg: S) -> Self {
        Self::ContentValidation(msg.into())
    }
    
    /// Create a configuration error
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::Config(msg.into())
    }
    
    /// Create a generic error
    pub fn generic<S: Into<String>>(msg: S) -> Self {
        Self::Generic(msg.into())
    }
}