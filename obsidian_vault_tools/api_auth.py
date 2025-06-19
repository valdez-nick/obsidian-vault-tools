"""
API Authentication Module

Provides secure authentication mechanisms for web APIs and external integrations.
"""

import os
import hashlib
import hmac
import secrets
import time
from typing import Optional, Dict, Any, Callable
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .security import APIAuthenticator, rate_limit, SecurityError

logger = logging.getLogger(__name__)


@dataclass
class AuthConfig:
    """Authentication configuration"""
    api_key_length: int = 32
    session_timeout_minutes: int = 60
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    require_https: bool = True
    jwt_secret: Optional[str] = None


class TokenManager:
    """Manages API tokens and sessions"""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.sessions = {}  # In production, use Redis or database
        self.failed_attempts = {}
        
    def generate_api_key(self, user_id: str) -> str:
        """Generate a secure API key for a user"""
        # Generate a secure random token
        token = secrets.token_urlsafe(self.config.api_key_length)
        
        # Add timestamp and user ID for tracking
        timestamp = int(time.time())
        data = f"{user_id}:{timestamp}:{token}"
        
        # Create HMAC signature if JWT secret is available
        if self.config.jwt_secret:
            signature = hmac.new(
                self.config.jwt_secret.encode(),
                data.encode(),
                hashlib.sha256
            ).hexdigest()
            return f"{data}:{signature}"
        
        return token
    
    def validate_api_key(self, api_key: str, user_id: str = None) -> bool:
        """Validate an API key"""
        if not api_key:
            return False
        
        # Check for lockout
        if self._is_locked_out(api_key):
            logger.warning(f"API key validation attempted during lockout: {api_key[:8]}...")
            return False
        
        try:
            # If JWT secret is configured, validate signature
            if self.config.jwt_secret and ':' in api_key:
                parts = api_key.rsplit(':', 1)
                if len(parts) == 2:
                    data, signature = parts
                    expected_signature = hmac.new(
                        self.config.jwt_secret.encode(),
                        data.encode(),
                        hashlib.sha256
                    ).hexdigest()
                    
                    if not hmac.compare_digest(signature, expected_signature):
                        self._record_failed_attempt(api_key)
                        return False
                    
                    # Extract user_id and timestamp for additional validation
                    data_parts = data.split(':')
                    if len(data_parts) >= 3:
                        token_user_id, timestamp_str = data_parts[0], data_parts[1]
                        timestamp = int(timestamp_str)
                        
                        # Check if user_id matches (if provided)
                        if user_id and token_user_id != user_id:
                            self._record_failed_attempt(api_key)
                            return False
                        
                        # Check token age (optional)
                        # For now, we don't expire API keys based on generation time
                        
                        return True
            
            # Simple validation for basic tokens
            # In production, store API keys in database with user associations
            return len(api_key) >= 16  # Minimum length check
            
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            self._record_failed_attempt(api_key)
            return False
    
    def create_session(self, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new session"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(minutes=self.config.session_timeout_minutes)
        
        self.sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'expires_at': expires_at,
            'metadata': metadata or {}
        }
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate a session and return session data"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session has expired
        if datetime.now() > session['expires_at']:
            del self.sessions[session_id]
            return None
        
        # Extend session on activity
        session['expires_at'] = datetime.now() + timedelta(minutes=self.config.session_timeout_minutes)
        
        return session
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def _record_failed_attempt(self, identifier: str):
        """Record a failed authentication attempt"""
        now = datetime.now()
        key = hashlib.sha256(identifier.encode()).hexdigest()[:16]  # Anonymize
        
        if key not in self.failed_attempts:
            self.failed_attempts[key] = []
        
        self.failed_attempts[key].append(now)
        
        # Clean old attempts
        cutoff = now - timedelta(minutes=self.config.lockout_duration_minutes)
        self.failed_attempts[key] = [
            attempt for attempt in self.failed_attempts[key]
            if attempt > cutoff
        ]
    
    def _is_locked_out(self, identifier: str) -> bool:
        """Check if an identifier is locked out due to failed attempts"""
        key = hashlib.sha256(identifier.encode()).hexdigest()[:16]
        
        if key not in self.failed_attempts:
            return False
        
        now = datetime.now()
        cutoff = now - timedelta(minutes=self.config.lockout_duration_minutes)
        
        # Count recent failed attempts
        recent_attempts = [
            attempt for attempt in self.failed_attempts[key]
            if attempt > cutoff
        ]
        
        return len(recent_attempts) >= self.config.max_failed_attempts


# Global authentication manager
_auth_config = AuthConfig()
_token_manager = TokenManager(_auth_config)
_api_authenticator = APIAuthenticator()


def get_auth_config() -> AuthConfig:
    """Get the current authentication configuration"""
    return _auth_config


def get_token_manager() -> TokenManager:
    """Get the token manager instance"""
    return _token_manager


def configure_auth(
    api_key_length: int = 32,
    session_timeout_minutes: int = 60,
    max_failed_attempts: int = 5,
    lockout_duration_minutes: int = 15,
    require_https: bool = True,
    jwt_secret: Optional[str] = None
):
    """Configure authentication settings"""
    global _auth_config, _token_manager
    
    _auth_config = AuthConfig(
        api_key_length=api_key_length,
        session_timeout_minutes=session_timeout_minutes,
        max_failed_attempts=max_failed_attempts,
        lockout_duration_minutes=lockout_duration_minutes,
        require_https=require_https,
        jwt_secret=jwt_secret or os.environ.get('JWT_SECRET')
    )
    
    _token_manager = TokenManager(_auth_config)


def require_api_key(user_id: str = None):
    """Decorator to require API key authentication"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract API key from various sources
            api_key = None
            
            # Try to get from kwargs (for function calls)
            if 'api_key' in kwargs:
                api_key = kwargs.pop('api_key')
            
            # Try to get from environment
            if not api_key:
                api_key = os.environ.get('API_KEY')
            
            # Validate API key
            if not api_key or not _token_manager.validate_api_key(api_key, user_id):
                raise SecurityError("Invalid or missing API key")
            
            return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract API key from various sources
            api_key = None
            
            # Try to get from kwargs
            if 'api_key' in kwargs:
                api_key = kwargs.pop('api_key')
            
            # Try to get from environment
            if not api_key:
                api_key = os.environ.get('API_KEY')
            
            # Validate API key
            if not api_key or not _token_manager.validate_api_key(api_key, user_id):
                raise SecurityError("Invalid or missing API key")
            
            return await func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def require_session():
    """Decorator to require session authentication"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract session ID
            session_id = kwargs.pop('session_id', None)
            
            if not session_id:
                raise SecurityError("Missing session ID")
            
            # Validate session
            session = _token_manager.validate_session(session_id)
            if not session:
                raise SecurityError("Invalid or expired session")
            
            # Add session data to kwargs
            kwargs['session'] = session
            
            return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract session ID
            session_id = kwargs.pop('session_id', None)
            
            if not session_id:
                raise SecurityError("Missing session ID")
            
            # Validate session
            session = _token_manager.validate_session(session_id)
            if not session:
                raise SecurityError("Invalid or expired session")
            
            # Add session data to kwargs
            kwargs['session'] = session
            
            return await func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


# Rate-limited authentication functions
@rate_limit(max_requests=10, window_seconds=60)
def generate_api_key_for_user(user_id: str) -> str:
    """Generate an API key for a user (rate limited)"""
    return _token_manager.generate_api_key(user_id)


@rate_limit(max_requests=100, window_seconds=60)
def validate_api_key(api_key: str, user_id: str = None) -> bool:
    """Validate an API key (rate limited)"""
    return _token_manager.validate_api_key(api_key, user_id)


@rate_limit(max_requests=20, window_seconds=60)
def create_session(user_id: str, metadata: Dict[str, Any] = None) -> str:
    """Create a session (rate limited)"""
    return _token_manager.create_session(user_id, metadata)


# Example usage and testing functions
def demo_authentication():
    """Demonstrate authentication functionality"""
    print("=== API Authentication Demo ===")
    
    # Configure authentication
    configure_auth(jwt_secret="demo-secret-key")
    
    # Generate API key
    user_id = "demo_user"
    api_key = generate_api_key_for_user(user_id)
    print(f"Generated API key: {api_key[:16]}...")
    
    # Validate API key
    is_valid = validate_api_key(api_key, user_id)
    print(f"API key valid: {is_valid}")
    
    # Create session
    session_id = create_session(user_id, {"role": "admin"})
    print(f"Created session: {session_id[:16]}...")
    
    # Validate session
    session = _token_manager.validate_session(session_id)
    print(f"Session valid: {session is not None}")
    if session:
        print(f"Session user: {session['user_id']}")
    
    # Test decorators
    @require_api_key(user_id="demo_user")
    def protected_function(data: str):
        return f"Protected data accessed: {data}"
    
    try:
        # This should work with valid API key
        result = protected_function("test data", api_key=api_key)
        print(f"Protected function result: {result}")
    except SecurityError as e:
        print(f"Security error: {e}")
    
    try:
        # This should fail with invalid API key
        result = protected_function("test data", api_key="invalid-key")
        print(f"This shouldn't print: {result}")
    except SecurityError as e:
        print(f"Expected security error: {e}")


if __name__ == "__main__":
    demo_authentication()