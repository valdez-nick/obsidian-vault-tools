"""
Authentication Manager for PM Automation Suite.

Provides OAuth 2.0 authentication flows for multiple providers
with automatic token refresh, multi-tenant support, and secure
credential storage.
"""

import asyncio
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Callable
from urllib.parse import urlencode, urlparse, parse_qs
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from aiohttp import web
import keyring

from .credentials_helper import CredentialsHelper, OAuthCredential

logger = logging.getLogger(__name__)


@dataclass
class OAuthConfig:
    """OAuth provider configuration."""
    provider: str
    auth_url: str
    token_url: str
    scopes: List[str]
    redirect_uri: str = "http://localhost:8888/callback"
    additional_params: Dict[str, str] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class AuthenticationManager:
    """
    Manages OAuth 2.0 authentication for multiple providers.
    
    Supports:
    - Google OAuth 2.0
    - Atlassian OAuth 2.0
    - Microsoft OAuth 2.0
    - Generic OAuth 2.0 providers
    
    Features:
    - Automatic token refresh
    - Multi-tenant support
    - Secure credential storage
    - State validation for security
    """
    
    # OAuth configurations for supported providers
    OAUTH_CONFIGS = {
        "google": OAuthConfig(
            provider="google",
            auth_url="https://accounts.google.com/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
            scopes=[
                "https://www.googleapis.com/auth/drive",
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/presentations"
            ],
            additional_params={
                "access_type": "offline",
                "prompt": "consent"
            }
        ),
        "atlassian": OAuthConfig(
            provider="atlassian",
            auth_url="https://auth.atlassian.com/authorize",
            token_url="https://auth.atlassian.com/oauth/token",
            scopes=[
                "read:jira-work",
                "write:jira-work",
                "read:confluence-content.all",
                "write:confluence-content"
            ],
            additional_params={
                "audience": "api.atlassian.com"
            }
        ),
        "microsoft": OAuthConfig(
            provider="microsoft",
            auth_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
            scopes=[
                "https://graph.microsoft.com/Files.ReadWrite.All",
                "https://graph.microsoft.com/Sites.ReadWrite.All"
            ]
        )
    }
    
    def __init__(self, credentials_helper: Optional[CredentialsHelper] = None,
                 callback_port: int = 8888):
        """
        Initialize authentication manager.
        
        Args:
            credentials_helper: Optional CredentialsHelper instance
            callback_port: Port for OAuth callback server
        """
        self.credentials_helper = credentials_helper or CredentialsHelper()
        self.callback_port = callback_port
        self._active_states: Dict[str, Dict[str, Any]] = {}  # state -> context
        self._callback_server: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        
    async def authenticate(self, provider: str, tenant_id: str,
                         client_id: str, client_secret: str,
                         scopes: Optional[List[str]] = None,
                         force_reauth: bool = False) -> OAuthCredential:
        """
        Authenticate with an OAuth provider.
        
        Args:
            provider: OAuth provider name (google, atlassian, etc.)
            tenant_id: Tenant/workspace identifier
            client_id: OAuth client ID
            client_secret: OAuth client secret
            scopes: Optional custom scopes (uses defaults if not provided)
            force_reauth: Force re-authentication even if valid token exists
            
        Returns:
            OAuthCredential with valid access token
            
        Raises:
            ValueError: If provider is not supported
            RuntimeError: If authentication fails
        """
        # Check for existing valid credential
        if not force_reauth:
            existing = self.credentials_helper.get_credential(provider, tenant_id)
            if existing and not existing.is_expired():
                logger.info(f"Using existing valid credential for {provider}/{tenant_id}")
                return existing
            elif existing and existing.refresh_token:
                # Try to refresh the token
                try:
                    refreshed = await self.refresh_token(existing)
                    if refreshed:
                        return refreshed
                except Exception as e:
                    logger.warning(f"Failed to refresh token: {e}")
        
        # Get OAuth config
        if provider not in self.OAUTH_CONFIGS:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
        
        config = self.OAUTH_CONFIGS[provider]
        if scopes:
            config.scopes = scopes
        
        # Start OAuth flow
        logger.info(f"Starting OAuth flow for {provider}/{tenant_id}")
        
        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)
        
        # Store state with context
        self._active_states[state] = {
            "provider": provider,
            "tenant_id": tenant_id,
            "client_id": client_id,
            "client_secret": client_secret,
            "scopes": config.scopes,
            "timestamp": time.time()
        }
        
        # Start callback server
        await self._start_callback_server()
        
        # Build authorization URL
        auth_params = {
            "client_id": client_id,
            "redirect_uri": config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(config.scopes),
            "state": state,
            **config.additional_params
        }
        
        auth_url = f"{config.auth_url}?{urlencode(auth_params)}"
        
        # Open browser for authorization
        logger.info(f"Opening browser for authorization: {auth_url}")
        webbrowser.open(auth_url)
        
        # Wait for callback (with timeout)
        try:
            credential = await self._wait_for_callback(state, config)
            
            # Store the credential
            self.credentials_helper.store_credential(credential)
            
            return credential
            
        except asyncio.TimeoutError:
            raise RuntimeError("OAuth callback timeout - no response received")
        finally:
            # Clean up state
            self._active_states.pop(state, None)
            await self._stop_callback_server()
    
    async def refresh_token(self, credential: OAuthCredential) -> Optional[OAuthCredential]:
        """
        Refresh an OAuth access token.
        
        Args:
            credential: OAuthCredential with refresh token
            
        Returns:
            Updated OAuthCredential or None if refresh fails
        """
        if not credential.refresh_token:
            logger.error("No refresh token available")
            return None
        
        config = self.OAUTH_CONFIGS.get(credential.provider)
        if not config:
            logger.error(f"Unknown provider: {credential.provider}")
            return None
        
        logger.info(f"Refreshing token for {credential.provider}/{credential.tenant_id}")
        
        token_data = {
            "grant_type": "refresh_token",
            "refresh_token": credential.refresh_token,
            "client_id": credential.client_id,
            "client_secret": credential.client_secret
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(config.token_url, data=token_data) as response:
                    if response.status == 200:
                        token_response = await response.json()
                        
                        # Update credential
                        credential.access_token = token_response["access_token"]
                        if "refresh_token" in token_response:
                            credential.refresh_token = token_response["refresh_token"]
                        
                        expires_in = token_response.get("expires_in", 3600)
                        credential.expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
                        
                        # Store updated credential
                        self.credentials_helper.store_credential(credential)
                        
                        logger.info(f"Successfully refreshed token for {credential.provider}/{credential.tenant_id}")
                        return credential
                    else:
                        error_text = await response.text()
                        logger.error(f"Token refresh failed: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return None
    
    async def revoke_token(self, credential: OAuthCredential) -> bool:
        """
        Revoke an OAuth token.
        
        Args:
            credential: OAuthCredential to revoke
            
        Returns:
            True if revoked successfully
        """
        # Provider-specific revocation endpoints
        revoke_urls = {
            "google": "https://oauth2.googleapis.com/revoke",
            "atlassian": "https://auth.atlassian.com/oauth/revoke"
        }
        
        revoke_url = revoke_urls.get(credential.provider)
        if not revoke_url:
            logger.warning(f"No revoke endpoint for provider: {credential.provider}")
            return False
        
        logger.info(f"Revoking token for {credential.provider}/{credential.tenant_id}")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Try to revoke refresh token first, then access token
                for token in [credential.refresh_token, credential.access_token]:
                    if not token:
                        continue
                    
                    data = {"token": token}
                    if credential.provider == "google":
                        # Google uses query parameters
                        async with session.post(f"{revoke_url}?token={token}") as response:
                            if response.status == 200:
                                logger.info(f"Revoked token for {credential.provider}")
                    else:
                        # Others use POST data
                        async with session.post(revoke_url, data=data) as response:
                            if response.status == 200:
                                logger.info(f"Revoked token for {credential.provider}")
                
                # Delete from storage
                self.credentials_helper.delete_credential(credential.provider, credential.tenant_id)
                return True
                
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
            return False
    
    def get_valid_credential(self, provider: str, tenant_id: str) -> Optional[OAuthCredential]:
        """
        Get a valid credential, refreshing if necessary.
        
        Args:
            provider: OAuth provider name
            tenant_id: Tenant identifier
            
        Returns:
            Valid OAuthCredential or None
        """
        credential = self.credentials_helper.get_credential(provider, tenant_id)
        if not credential:
            return None
        
        if credential.is_expired() and credential.refresh_token:
            # Synchronously refresh token
            loop = asyncio.get_event_loop()
            credential = loop.run_until_complete(self.refresh_token(credential))
        
        return credential if credential and not credential.is_expired() else None
    
    def list_credentials(self) -> Dict[str, List[Tuple[str, bool]]]:
        """
        List all stored credentials with validity status.
        
        Returns:
            Dictionary of provider -> list of (tenant_id, is_valid) tuples
        """
        result = {}
        all_creds = self.credentials_helper.list_credentials()
        
        for provider, tenant_ids in all_creds.items():
            result[provider] = []
            for tenant_id in tenant_ids:
                credential = self.credentials_helper.get_credential(provider, tenant_id)
                is_valid = credential and not credential.is_expired() if credential else False
                result[provider].append((tenant_id, is_valid))
        
        return result
    
    async def _start_callback_server(self):
        """Start the OAuth callback server."""
        if self._callback_server:
            return
        
        app = web.Application()
        app.router.add_get('/callback', self._handle_callback)
        
        self._callback_server = app
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        
        site = web.TCPSite(self._runner, 'localhost', self.callback_port)
        await site.start()
        
        logger.info(f"OAuth callback server started on port {self.callback_port}")
    
    async def _stop_callback_server(self):
        """Stop the OAuth callback server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._callback_server = None
            logger.info("OAuth callback server stopped")
    
    async def _handle_callback(self, request: web.Request) -> web.Response:
        """Handle OAuth callback request."""
        # Extract parameters
        code = request.query.get('code')
        state = request.query.get('state')
        error = request.query.get('error')
        
        if error:
            logger.error(f"OAuth error: {error}")
            return web.Response(
                text=f"Authentication failed: {error}",
                status=400
            )
        
        if not code or not state:
            return web.Response(
                text="Missing code or state parameter",
                status=400
            )
        
        # Validate state
        if state not in self._active_states:
            logger.error("Invalid state parameter")
            return web.Response(
                text="Invalid state - possible CSRF attack",
                status=400
            )
        
        # Process the authorization code
        context = self._active_states[state]
        context['code'] = code
        
        # Return success page
        success_html = """
        <html>
        <head><title>Authentication Successful</title></head>
        <body>
            <h1>Authentication Successful!</h1>
            <p>You can now close this window.</p>
            <script>window.close();</script>
        </body>
        </html>
        """
        
        return web.Response(text=success_html, content_type='text/html')
    
    async def _wait_for_callback(self, state: str, config: OAuthConfig,
                                timeout: int = 300) -> OAuthCredential:
        """
        Wait for OAuth callback and exchange code for tokens.
        
        Args:
            state: OAuth state parameter
            config: OAuth configuration
            timeout: Timeout in seconds
            
        Returns:
            OAuthCredential with tokens
        """
        start_time = time.time()
        
        # Wait for authorization code
        while time.time() - start_time < timeout:
            if state in self._active_states and 'code' in self._active_states[state]:
                context = self._active_states[state]
                code = context['code']
                break
            await asyncio.sleep(0.5)
        else:
            raise asyncio.TimeoutError("Timeout waiting for OAuth callback")
        
        # Exchange code for tokens
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": config.redirect_uri,
            "client_id": context['client_id'],
            "client_secret": context['client_secret']
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.token_url, data=token_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Token exchange failed: {error_text}")
                
                token_response = await response.json()
        
        # Create credential
        expires_in = token_response.get("expires_in", 3600)
        credential = OAuthCredential(
            provider=context['provider'],
            tenant_id=context['tenant_id'],
            client_id=context['client_id'],
            client_secret=context['client_secret'],
            access_token=token_response["access_token"],
            refresh_token=token_response.get("refresh_token"),
            token_type=token_response.get("token_type", "Bearer"),
            expires_at=datetime.utcnow() + timedelta(seconds=expires_in),
            scopes=context['scopes']
        )
        
        return credential
    
    def validate_credentials(self, provider: str, tenant_id: str) -> Dict[str, Any]:
        """
        Validate OAuth credentials by testing API access.
        
        Args:
            provider: OAuth provider name
            tenant_id: Tenant identifier
            
        Returns:
            Validation result with status and details
        """
        result = {
            "valid": False,
            "error": None,
            "token_status": None,
            "scopes": []
        }
        
        try:
            credential = self.get_valid_credential(provider, tenant_id)
            if not credential:
                result["error"] = "No valid credential found"
                return result
            
            # Provider-specific validation
            validation_endpoints = {
                "google": "https://www.googleapis.com/oauth2/v1/tokeninfo",
                "atlassian": "https://api.atlassian.com/oauth/token/accessible-resources",
                "microsoft": "https://graph.microsoft.com/v1.0/me"
            }
            
            endpoint = validation_endpoints.get(provider)
            if not endpoint:
                result["error"] = f"No validation endpoint for provider: {provider}"
                return result
            
            # Make validation request
            import requests
            headers = {"Authorization": f"{credential.token_type} {credential.access_token}"}
            
            if provider == "google":
                response = requests.get(f"{endpoint}?access_token={credential.access_token}")
            else:
                response = requests.get(endpoint, headers=headers)
            
            if response.status_code == 200:
                result["valid"] = True
                result["token_status"] = "active"
                result["scopes"] = credential.scopes
                
                # Add provider-specific info
                if provider == "google":
                    token_info = response.json()
                    result["scopes"] = token_info.get("scope", "").split()
                    result["expires_in"] = token_info.get("expires_in")
            else:
                result["error"] = f"Validation failed: {response.status_code}"
                result["token_status"] = "invalid"
                
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    async def cleanup_expired_states(self, max_age: int = 3600):
        """
        Clean up expired OAuth states.
        
        Args:
            max_age: Maximum age of states in seconds
        """
        current_time = time.time()
        expired_states = []
        
        for state, context in self._active_states.items():
            if current_time - context['timestamp'] > max_age:
                expired_states.append(state)
        
        for state in expired_states:
            del self._active_states[state]
        
        if expired_states:
            logger.info(f"Cleaned up {len(expired_states)} expired OAuth states")