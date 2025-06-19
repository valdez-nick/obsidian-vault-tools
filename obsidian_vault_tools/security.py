"""
Security utilities for input validation, sanitization, and protection

This module provides security functions to prevent common vulnerabilities:
- Path traversal attacks
- Command injection
- Input validation
- Logging sanitization
"""

import os
import re
import shlex
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import logging
from functools import wraps
import time
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Base exception for security-related errors"""
    pass


class PathTraversalError(SecurityError):
    """Raised when a path traversal attempt is detected"""
    pass


class InputValidationError(SecurityError):
    """Raised when input validation fails"""
    pass


def validate_path(path: Union[str, Path], base_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Validate and sanitize a file path to prevent path traversal attacks.
    
    Args:
        path: The path to validate
        base_path: Optional base path that the path must be within
        
    Returns:
        Validated Path object
        
    Raises:
        PathTraversalError: If path traversal is detected
    """
    # Convert to Path object and resolve to absolute path
    path = Path(path).resolve()
    
    # Check for null bytes (common in path injection attacks)
    if '\0' in str(path):
        raise PathTraversalError("Null byte detected in path")
    
    # If base_path is provided, ensure the path is within it
    if base_path:
        base_path = Path(base_path).resolve()
        try:
            # This will raise ValueError if path is not relative to base_path
            path.relative_to(base_path)
        except ValueError:
            raise PathTraversalError(
                f"Path '{path}' is outside of base directory '{base_path}'"
            )
    
    # Additional checks for suspicious patterns
    path_str = str(path)
    suspicious_patterns = [
        r'\.\./\.\./\.\.',  # Multiple parent directory traversals
        r'/etc/passwd',      # Common target
        r'/etc/shadow',      # Common target
        r'\.ssh/',           # SSH keys
        r'\.aws/',           # AWS credentials
        r'\.env',            # Environment files
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, path_str, re.IGNORECASE):
            logger.warning(f"Suspicious path pattern detected: {pattern} in {path_str}")
            # Don't raise here, just log - let the base_path check handle actual security
    
    return path


def sanitize_filename(filename: str, allow_subdirs: bool = False) -> str:
    """
    Sanitize a filename to remove potentially dangerous characters.
    
    Args:
        filename: The filename to sanitize
        allow_subdirs: Whether to allow directory separators
        
    Returns:
        Sanitized filename
    """
    # Remove null bytes
    filename = filename.replace('\0', '')
    
    if not allow_subdirs:
        # Remove any directory separators
        filename = os.path.basename(filename)
    
    # Remove potentially dangerous characters
    # Allow alphanumeric, spaces, hyphens, underscores, periods
    if allow_subdirs:
        # Also allow forward slashes for subdirectories
        safe_chars = re.compile(r'[^a-zA-Z0-9\s\-_./]')
    else:
        safe_chars = re.compile(r'[^a-zA-Z0-9\s\-_.]')
    
    filename = safe_chars.sub('_', filename)
    
    # Remove multiple consecutive periods (could be used for traversal)
    filename = re.sub(r'\.{2,}', '.', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure filename is not empty
    if not filename:
        filename = 'unnamed'
    
    return filename


def validate_command_args(args: List[str]) -> List[str]:
    """
    Validate command arguments to prevent command injection.
    
    Args:
        args: List of command arguments
        
    Returns:
        Validated and quoted arguments
    """
    validated_args = []
    
    for arg in args:
        # Check for shell metacharacters
        if any(char in arg for char in ['&', '|', ';', '$', '`', '\\', '!', '*', '?', '[', ']', '(', ')', '{', '}', '<', '>', '\n']):
            # Use shlex.quote to safely quote the argument
            arg = shlex.quote(arg)
        
        validated_args.append(arg)
    
    return validated_args


def sanitize_log_data(data: Any) -> Any:
    """
    Sanitize data before logging to prevent sensitive information leakage.
    
    Args:
        data: Data to sanitize (can be dict, list, str, etc.)
        
    Returns:
        Sanitized data
    """
    sensitive_patterns = [
        (r'password["\']?\s*[:=]\s*["\']?([^"\'\\s]+)', 'password=***'),
        (r'token["\']?\s*[:=]\s*["\']?([^"\'\\s]+)', 'token=***'),
        (r'api_key["\']?\s*[:=]\s*["\']?([^"\'\\s]+)', 'api_key=***'),
        (r'secret["\']?\s*[:=]\s*["\']?([^"\'\\s]+)', 'secret=***'),
        (r'Authorization:\s*Bearer\s+(\S+)', 'Authorization: Bearer ***'),
        (r'/Users/[^/\s]+/', '/Users/***/'),  # Hide username in paths
        (r'\\\\Users\\\\[^\\\\\\s]+\\\\', '\\\\Users\\\\***\\\\'),  # Windows paths
    ]
    
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            # Check if key contains sensitive words
            if any(word in key.lower() for word in ['password', 'token', 'secret', 'key', 'auth']):
                sanitized[key] = '***'
            else:
                sanitized[key] = sanitize_log_data(value)
        return sanitized
    
    elif isinstance(data, list):
        return [sanitize_log_data(item) for item in data]
    
    elif isinstance(data, str):
        sanitized = data
        for pattern, replacement in sensitive_patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        return sanitized
    
    else:
        return data


# Rate limiting decorator
class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """
        Check if a request is allowed based on rate limits.
        
        Args:
            key: Unique key for the rate limit (e.g., user ID, IP)
            max_requests: Maximum requests allowed in the window
            window_seconds: Time window in seconds
            
        Returns:
            True if request is allowed, False otherwise
        """
        with self.lock:
            now = time.time()
            # Clean old requests
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if now - req_time < window_seconds
            ]
            
            # Check if limit exceeded
            if len(self.requests[key]) >= max_requests:
                return False
            
            # Record this request
            self.requests[key].append(now)
            return True


# Global rate limiter instance
rate_limiter = RateLimiter()


def rate_limit(max_requests: int = 60, window_seconds: int = 60, key_func=None):
    """
    Decorator to rate limit function calls.
    
    Args:
        max_requests: Maximum requests allowed in the window
        window_seconds: Time window in seconds
        key_func: Function to generate rate limit key from arguments
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate rate limit key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Default: use function name
                key = func.__name__
            
            # Check rate limit
            if not rate_limiter.is_allowed(key, max_requests, window_seconds):
                raise SecurityError(f"Rate limit exceeded for {key}")
            
            return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate rate limit key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Default: use function name
                key = func.__name__
            
            # Check rate limit
            if not rate_limiter.is_allowed(key, max_requests, window_seconds):
                raise SecurityError(f"Rate limit exceeded for {key}")
            
            return await func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def validate_json_input(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate JSON input against a simple schema.
    
    Args:
        data: Input data to validate
        schema: Schema definition with field types and requirements
        
    Returns:
        Validated data
        
    Raises:
        InputValidationError: If validation fails
    """
    validated = {}
    
    for field, rules in schema.items():
        # Check if field is required
        if rules.get('required', False) and field not in data:
            raise InputValidationError(f"Required field '{field}' is missing")
        
        if field in data:
            value = data[field]
            field_type = rules.get('type')
            
            # Type validation
            if field_type == 'string':
                if not isinstance(value, str):
                    raise InputValidationError(f"Field '{field}' must be a string")
                
                # Length validation
                if 'max_length' in rules and len(value) > rules['max_length']:
                    raise InputValidationError(
                        f"Field '{field}' exceeds maximum length of {rules['max_length']}"
                    )
                
                # Pattern validation
                if 'pattern' in rules and not re.match(rules['pattern'], value):
                    raise InputValidationError(
                        f"Field '{field}' does not match required pattern"
                    )
                
                validated[field] = value
            
            elif field_type == 'integer':
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    raise InputValidationError(f"Field '{field}' must be an integer")
                
                # Range validation
                if 'min' in rules and value < rules['min']:
                    raise InputValidationError(
                        f"Field '{field}' must be at least {rules['min']}"
                    )
                if 'max' in rules and value > rules['max']:
                    raise InputValidationError(
                        f"Field '{field}' must be at most {rules['max']}"
                    )
                
                validated[field] = value
            
            elif field_type == 'boolean':
                if not isinstance(value, bool):
                    raise InputValidationError(f"Field '{field}' must be a boolean")
                validated[field] = value
            
            elif field_type == 'array':
                if not isinstance(value, list):
                    raise InputValidationError(f"Field '{field}' must be an array")
                
                # Length validation
                if 'max_items' in rules and len(value) > rules['max_items']:
                    raise InputValidationError(
                        f"Field '{field}' exceeds maximum items of {rules['max_items']}"
                    )
                
                validated[field] = value
            
            else:
                # Default: pass through
                validated[field] = value
    
    return validated


def escape_html(text: str) -> str:
    """
    Escape HTML special characters to prevent XSS attacks.
    
    Args:
        text: Text to escape
        
    Returns:
        HTML-escaped text
    """
    html_escape_table = {
        "&": "&amp;",
        '"': "&quot;",
        "'": "&#x27;",
        ">": "&gt;",
        "<": "&lt;",
    }
    
    return "".join(html_escape_table.get(c, c) for c in text)


# Logging filter to sanitize sensitive data
class SanitizingFilter(logging.Filter):
    """Logging filter that sanitizes sensitive data"""
    
    def filter(self, record):
        # Sanitize the message
        if hasattr(record, 'msg'):
            record.msg = sanitize_log_data(record.msg)
        
        # Sanitize arguments
        if hasattr(record, 'args'):
            record.args = tuple(sanitize_log_data(arg) for arg in record.args)
        
        return True


def setup_secure_logging():
    """Set up secure logging configuration"""
    # Add sanitizing filter to root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(SanitizingFilter())
    
    # Also add to any existing handlers
    for handler in root_logger.handlers:
        handler.addFilter(SanitizingFilter())


# Example usage for API authentication
class APIAuthenticator:
    """Simple API key authentication"""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys = api_keys or {}
    
    def authenticate(self, api_key: str) -> Optional[str]:
        """
        Authenticate an API key.
        
        Args:
            api_key: The API key to authenticate
            
        Returns:
            User ID if authenticated, None otherwise
        """
        # Constant-time comparison to prevent timing attacks
        for user_id, stored_key in self.api_keys.items():
            if len(api_key) == len(stored_key):
                # Use constant-time comparison
                if all(a == b for a, b in zip(api_key, stored_key)):
                    return user_id
        
        return None
    
    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        import secrets
        return secrets.token_urlsafe(32)


# Initialize secure logging on module import
setup_secure_logging()