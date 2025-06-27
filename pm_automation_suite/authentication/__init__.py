"""
Authentication module for PM Automation Suite.

Provides OAuth 2.0 authentication flows and secure credential management
for multiple providers (Google, Atlassian, etc.).
"""

from .auth_manager import AuthenticationManager
from .credentials_helper import CredentialsHelper, OAuthCredential

__all__ = [
    'AuthenticationManager',
    'CredentialsHelper',
    'OAuthCredential'
]