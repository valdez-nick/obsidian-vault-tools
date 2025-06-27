"""
Example usage of AuthenticationManager for OAuth authentication.

This script demonstrates:
- Authenticating with Google OAuth
- Authenticating with Atlassian OAuth
- Refreshing expired tokens
- Managing multiple tenants
- Validating credentials
"""

import asyncio
import logging
from typing import Optional

from authentication import AuthenticationManager, CredentialsHelper, OAuthCredential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def authenticate_google(auth_manager: AuthenticationManager, 
                            client_id: str, 
                            client_secret: str,
                            tenant_id: str = "default") -> Optional[OAuthCredential]:
    """
    Authenticate with Google OAuth.
    
    Args:
        auth_manager: AuthenticationManager instance
        client_id: Google OAuth client ID
        client_secret: Google OAuth client secret
        tenant_id: Tenant identifier (for multi-tenant support)
        
    Returns:
        OAuthCredential if successful, None otherwise
    """
    try:
        logger.info(f"Authenticating with Google for tenant: {tenant_id}")
        
        # Define custom scopes if needed
        scopes = [
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/presentations",
            "https://www.googleapis.com/auth/calendar"
        ]
        
        credential = await auth_manager.authenticate(
            provider="google",
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            scopes=scopes
        )
        
        logger.info(f"Successfully authenticated with Google!")
        logger.info(f"Access token expires at: {credential.expires_at}")
        logger.info(f"Authorized scopes: {', '.join(credential.scopes)}")
        
        return credential
        
    except Exception as e:
        logger.error(f"Failed to authenticate with Google: {e}")
        return None


async def authenticate_atlassian(auth_manager: AuthenticationManager,
                               client_id: str,
                               client_secret: str,
                               tenant_id: str) -> Optional[OAuthCredential]:
    """
    Authenticate with Atlassian OAuth.
    
    Args:
        auth_manager: AuthenticationManager instance
        client_id: Atlassian OAuth client ID
        client_secret: Atlassian OAuth client secret
        tenant_id: Atlassian workspace/site identifier
        
    Returns:
        OAuthCredential if successful, None otherwise
    """
    try:
        logger.info(f"Authenticating with Atlassian for workspace: {tenant_id}")
        
        credential = await auth_manager.authenticate(
            provider="atlassian",
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
        
        logger.info(f"Successfully authenticated with Atlassian!")
        logger.info(f"Access token expires at: {credential.expires_at}")
        
        return credential
        
    except Exception as e:
        logger.error(f"Failed to authenticate with Atlassian: {e}")
        return None


async def demonstrate_token_refresh(auth_manager: AuthenticationManager):
    """Demonstrate automatic token refresh."""
    logger.info("\n=== Demonstrating Token Refresh ===")
    
    # Check for existing Google credential
    credential = auth_manager.get_valid_credential("google", "default")
    
    if credential:
        logger.info(f"Found existing credential for Google/default")
        logger.info(f"Token expires at: {credential.expires_at}")
        
        if credential.is_expired():
            logger.info("Token is expired, refreshing...")
            refreshed = await auth_manager.refresh_token(credential)
            if refreshed:
                logger.info("Token refreshed successfully!")
                logger.info(f"New expiration: {refreshed.expires_at}")
            else:
                logger.error("Failed to refresh token")
        else:
            logger.info("Token is still valid")
    else:
        logger.info("No existing credential found")


async def demonstrate_multi_tenant():
    """Demonstrate multi-tenant credential management."""
    logger.info("\n=== Demonstrating Multi-Tenant Support ===")
    
    # Create auth manager
    auth_manager = AuthenticationManager()
    
    # List all stored credentials
    all_creds = auth_manager.list_credentials()
    logger.info("Stored credentials by provider:")
    
    for provider, tenants in all_creds.items():
        logger.info(f"\n{provider}:")
        for tenant_id, is_valid in tenants:
            status = "valid" if is_valid else "expired/invalid"
            logger.info(f"  - {tenant_id}: {status}")
    
    # Validate specific credentials
    logger.info("\n=== Validating Credentials ===")
    
    for provider in ["google", "atlassian"]:
        for tenant_id, _ in all_creds.get(provider, []):
            result = auth_manager.validate_credentials(provider, tenant_id)
            logger.info(f"\n{provider}/{tenant_id}:")
            logger.info(f"  Valid: {result['valid']}")
            logger.info(f"  Status: {result.get('token_status', 'unknown')}")
            if result.get('error'):
                logger.info(f"  Error: {result['error']}")


async def main():
    """Main example execution."""
    # Create authentication manager
    auth_manager = AuthenticationManager()
    
    # Example 1: Check existing credentials
    logger.info("=== Checking Existing Credentials ===")
    creds = auth_manager.list_credentials()
    if creds:
        logger.info("Found existing credentials:")
        for provider, tenants in creds.items():
            logger.info(f"  {provider}: {len(tenants)} tenant(s)")
    else:
        logger.info("No existing credentials found")
    
    # Example 2: Authenticate with Google (uncomment to run)
    # Note: You need to set up OAuth client ID and secret first
    # google_cred = await authenticate_google(
    #     auth_manager,
    #     client_id="YOUR_GOOGLE_CLIENT_ID",
    #     client_secret="YOUR_GOOGLE_CLIENT_SECRET",
    #     tenant_id="my-workspace"
    # )
    
    # Example 3: Authenticate with Atlassian (uncomment to run)
    # Note: You need to set up OAuth client ID and secret first
    # atlassian_cred = await authenticate_atlassian(
    #     auth_manager,
    #     client_id="YOUR_ATLASSIAN_CLIENT_ID",
    #     client_secret="YOUR_ATLASSIAN_CLIENT_SECRET",
    #     tenant_id="my-site.atlassian.net"
    # )
    
    # Example 4: Demonstrate token refresh
    await demonstrate_token_refresh(auth_manager)
    
    # Example 5: Demonstrate multi-tenant support
    await demonstrate_multi_tenant()
    
    # Example 6: Clean up expired OAuth states
    logger.info("\n=== Cleaning Up Expired States ===")
    await auth_manager.cleanup_expired_states()
    logger.info("Cleanup completed")


def run_interactive_auth():
    """Run interactive authentication flow."""
    import sys
    
    print("\n=== Interactive OAuth Authentication ===")
    print("1. Google")
    print("2. Atlassian")
    print("3. Exit")
    
    choice = input("\nSelect provider (1-3): ").strip()
    
    if choice == "3":
        sys.exit(0)
    
    provider = "google" if choice == "1" else "atlassian"
    
    client_id = input(f"\nEnter {provider} client ID: ").strip()
    client_secret = input(f"Enter {provider} client secret: ").strip()
    tenant_id = input(f"Enter tenant ID (press Enter for 'default'): ").strip() or "default"
    
    async def do_auth():
        auth_manager = AuthenticationManager()
        try:
            credential = await auth_manager.authenticate(
                provider=provider,
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
                force_reauth=True
            )
            print(f"\nAuthentication successful!")
            print(f"Access token expires at: {credential.expires_at}")
            print(f"Authorized scopes: {', '.join(credential.scopes)}")
        except Exception as e:
            print(f"\nAuthentication failed: {e}")
    
    asyncio.run(do_auth())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive_auth()
    else:
        asyncio.run(main())