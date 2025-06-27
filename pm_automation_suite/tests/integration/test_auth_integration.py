"""
Integration tests for authentication with real services.

These tests require actual OAuth credentials and will perform real authentication.
They are skipped by default unless credentials are provided via environment variables.
"""

import os
import pytest
import asyncio
from datetime import datetime
import pandas as pd

from authentication import AuthenticationManager, CredentialsHelper
from connectors.google_connector_oauth import GoogleConnectorOAuth

# Check for test credentials in environment
GOOGLE_CLIENT_ID = os.getenv("TEST_GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("TEST_GOOGLE_CLIENT_SECRET")
ATLASSIAN_CLIENT_ID = os.getenv("TEST_ATLASSIAN_CLIENT_ID")
ATLASSIAN_CLIENT_SECRET = os.getenv("TEST_ATLASSIAN_CLIENT_SECRET")

# Skip tests if credentials not available
skip_google = pytest.mark.skipif(
    not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET),
    reason="Google OAuth credentials not provided"
)
skip_atlassian = pytest.mark.skipif(
    not (ATLASSIAN_CLIENT_ID and ATLASSIAN_CLIENT_SECRET),
    reason="Atlassian OAuth credentials not provided"
)


class TestGoogleOAuthIntegration:
    """Test Google OAuth integration."""
    
    @skip_google
    @pytest.mark.asyncio
    async def test_google_oauth_flow(self):
        """Test complete Google OAuth flow."""
        auth_manager = AuthenticationManager()
        
        # Authenticate
        credential = await auth_manager.authenticate(
            provider="google",
            tenant_id="test-integration",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            force_reauth=True
        )
        
        assert credential is not None
        assert credential.access_token is not None
        assert credential.provider == "google"
        assert not credential.is_expired()
        
        # Validate the credential
        validation = auth_manager.validate_credentials("google", "test-integration")
        assert validation["valid"] is True
        assert len(validation["scopes"]) > 0
    
    @skip_google
    def test_google_connector_oauth(self):
        """Test Google connector with OAuth."""
        config = {
            "tenant_id": "test-connector",
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "scopes": [
                "https://www.googleapis.com/auth/drive.readonly",
                "https://www.googleapis.com/auth/spreadsheets.readonly"
            ]
        }
        
        connector = GoogleConnectorOAuth(config)
        
        # Connect
        assert connector.connect() is True
        
        # Validate connection
        assert connector.validate_connection() is True
        
        # Get metadata
        metadata = connector.get_metadata()
        assert metadata["auth_type"] == "OAuth 2.0"
        assert metadata["token_valid"] is True
        assert "drive" in metadata["available_services"]
        
        # Search for files
        query = {
            "service": "drive",
            "query": "mimeType='application/vnd.google-apps.spreadsheet'",
            "page_size": 5
        }
        
        df = connector.extract_data(query)
        assert isinstance(df, pd.DataFrame)
        print(f"Found {len(df)} spreadsheets")
    
    @skip_google
    @pytest.mark.asyncio
    async def test_token_refresh(self):
        """Test OAuth token refresh."""
        auth_manager = AuthenticationManager()
        
        # Get existing credential
        credential = auth_manager.get_valid_credential("google", "test-integration")
        
        if credential and credential.refresh_token:
            # Force refresh
            refreshed = await auth_manager.refresh_token(credential)
            
            assert refreshed is not None
            assert refreshed.access_token != credential.access_token
            assert refreshed.expires_at > datetime.utcnow()


class TestAtlassianOAuthIntegration:
    """Test Atlassian OAuth integration."""
    
    @skip_atlassian
    @pytest.mark.asyncio
    async def test_atlassian_oauth_flow(self):
        """Test complete Atlassian OAuth flow."""
        auth_manager = AuthenticationManager()
        
        # Authenticate
        credential = await auth_manager.authenticate(
            provider="atlassian",
            tenant_id="test-workspace",
            client_id=ATLASSIAN_CLIENT_ID,
            client_secret=ATLASSIAN_CLIENT_SECRET,
            force_reauth=True
        )
        
        assert credential is not None
        assert credential.access_token is not None
        assert credential.provider == "atlassian"
        assert not credential.is_expired()
        
        # Validate the credential
        validation = auth_manager.validate_credentials("atlassian", "test-workspace")
        assert validation["valid"] is True


class TestMultiTenantAuth:
    """Test multi-tenant authentication scenarios."""
    
    @skip_google
    @pytest.mark.asyncio
    async def test_multiple_tenants(self):
        """Test managing credentials for multiple tenants."""
        auth_manager = AuthenticationManager()
        
        # Authenticate first tenant
        cred1 = await auth_manager.authenticate(
            provider="google",
            tenant_id="tenant-prod",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET
        )
        
        # Authenticate second tenant (same provider, different tenant)
        cred2 = await auth_manager.authenticate(
            provider="google",
            tenant_id="tenant-dev",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET
        )
        
        assert cred1.tenant_id != cred2.tenant_id
        
        # List all credentials
        all_creds = auth_manager.list_credentials()
        assert "google" in all_creds
        assert any(t[0] == "tenant-prod" for t in all_creds["google"])
        assert any(t[0] == "tenant-dev" for t in all_creds["google"])


class TestCredentialPersistence:
    """Test credential persistence and security."""
    
    def test_credential_encryption(self):
        """Test that credentials are encrypted at rest."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "test_creds.enc")
            helper = CredentialsHelper(storage_path=storage_path, use_keyring=False)
            
            # Store a test credential
            from authentication.credentials_helper import OAuthCredential
            
            credential = OAuthCredential(
                provider="test",
                tenant_id="test-tenant",
                client_id="test-client",
                client_secret="super-secret",
                access_token="test-token",
                refresh_token="test-refresh"
            )
            
            helper.store_credential(credential)
            
            # Read the raw file
            with open(storage_path, 'r') as f:
                raw_content = f.read()
            
            # Verify sensitive data is not in plaintext
            assert "super-secret" not in raw_content
            assert "test-token" not in raw_content
            assert "test-refresh" not in raw_content
            
            # Verify we can still decrypt and retrieve
            loaded = helper.get_credential("test", "test-tenant")
            assert loaded.client_secret == "super-secret"
            assert loaded.access_token == "test-token"


class TestErrorHandling:
    """Test error handling in authentication flows."""
    
    @pytest.mark.asyncio
    async def test_invalid_provider(self):
        """Test handling of invalid provider."""
        auth_manager = AuthenticationManager()
        
        with pytest.raises(ValueError, match="Unsupported OAuth provider"):
            await auth_manager.authenticate(
                provider="invalid-provider",
                tenant_id="test",
                client_id="test",
                client_secret="test"
            )
    
    @pytest.mark.asyncio
    async def test_authentication_timeout(self):
        """Test OAuth callback timeout."""
        auth_manager = AuthenticationManager()
        
        # Mock browser open to do nothing
        with pytest.patch('webbrowser.open'):
            # This should timeout since no callback will arrive
            with pytest.raises(RuntimeError, match="OAuth callback timeout"):
                await auth_manager.authenticate(
                    provider="google",
                    tenant_id="test-timeout",
                    client_id="test-client",
                    client_secret="test-secret",
                    force_reauth=True
                )


if __name__ == "__main__":
    # Run only non-skipped tests
    pytest.main([__file__, "-v", "-m", "not skip"])