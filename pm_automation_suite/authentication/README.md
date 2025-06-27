# Authentication Module

The authentication module provides OAuth 2.0 authentication support for multiple providers with automatic token refresh, multi-tenant support, and secure credential storage.

## Features

- **OAuth 2.0 Support**: Full OAuth 2.0 authorization code flow implementation
- **Multiple Providers**: Built-in support for Google, Atlassian, and Microsoft
- **Automatic Token Refresh**: Tokens are automatically refreshed when expired
- **Multi-Tenant Support**: Manage credentials for multiple workspaces/tenants
- **Secure Storage**: Credentials encrypted at rest with keyring integration
- **State Validation**: CSRF protection with state parameter validation

## Components

### AuthenticationManager

The main class for handling OAuth authentication flows.

```python
from authentication import AuthenticationManager

# Initialize
auth_manager = AuthenticationManager()

# Authenticate with a provider
credential = await auth_manager.authenticate(
    provider="google",
    tenant_id="my-workspace",
    client_id="your-client-id",
    client_secret="your-client-secret"
)
```

### CredentialsHelper

Helper class for secure credential storage and retrieval.

```python
from authentication import CredentialsHelper, OAuthCredential

# Initialize
helper = CredentialsHelper()

# Store a credential
credential = OAuthCredential(
    provider="google",
    tenant_id="workspace1",
    client_id="client123",
    client_secret="secret123",
    access_token="token123",
    refresh_token="refresh123"
)
helper.store_credential(credential)

# Retrieve a credential
stored = helper.get_credential("google", "workspace1")
```

### OAuthCredential

Data class representing OAuth credentials.

```python
from authentication import OAuthCredential

credential = OAuthCredential(
    provider="atlassian",
    tenant_id="site.atlassian.net",
    client_id="client456",
    client_secret="secret456",
    access_token="access456",
    refresh_token="refresh456",
    scopes=["read:jira-work", "write:jira-work"]
)

# Check if expired
if credential.is_expired():
    print("Token needs refresh")
```

## Supported Providers

### Google
- Auth URL: `https://accounts.google.com/o/oauth2/v2/auth`
- Token URL: `https://oauth2.googleapis.com/token`
- Default Scopes:
  - `https://www.googleapis.com/auth/drive`
  - `https://www.googleapis.com/auth/spreadsheets`
  - `https://www.googleapis.com/auth/presentations`

### Atlassian
- Auth URL: `https://auth.atlassian.com/authorize`
- Token URL: `https://auth.atlassian.com/oauth/token`
- Default Scopes:
  - `read:jira-work`
  - `write:jira-work`
  - `read:confluence-content.all`
  - `write:confluence-content`

### Microsoft
- Auth URL: `https://login.microsoftonline.com/common/oauth2/v2.0/authorize`
- Token URL: `https://login.microsoftonline.com/common/oauth2/v2.0/token`
- Default Scopes:
  - `https://graph.microsoft.com/Files.ReadWrite.All`
  - `https://graph.microsoft.com/Sites.ReadWrite.All`

## Usage Examples

### Basic Authentication

```python
import asyncio
from authentication import AuthenticationManager

async def main():
    auth_manager = AuthenticationManager()
    
    # Authenticate with Google
    credential = await auth_manager.authenticate(
        provider="google",
        tenant_id="default",
        client_id="your-client-id",
        client_secret="your-client-secret"
    )
    
    print(f"Access token: {credential.access_token}")
    print(f"Expires at: {credential.expires_at}")

asyncio.run(main())
```

### Multi-Tenant Setup

```python
# Authenticate multiple workspaces
auth_manager = AuthenticationManager()

# Production workspace
prod_cred = await auth_manager.authenticate(
    provider="atlassian",
    tenant_id="prod.atlassian.net",
    client_id="prod-client-id",
    client_secret="prod-secret"
)

# Development workspace
dev_cred = await auth_manager.authenticate(
    provider="atlassian",
    tenant_id="dev.atlassian.net",
    client_id="dev-client-id",
    client_secret="dev-secret"
)
```

### Automatic Token Refresh

```python
# Get valid credential (automatically refreshes if needed)
credential = auth_manager.get_valid_credential("google", "default")

if credential:
    # Use the access token
    headers = {"Authorization": f"Bearer {credential.access_token}"}
    # Make API calls...
```

### Credential Validation

```python
# Validate credentials
result = auth_manager.validate_credentials("google", "default")

if result["valid"]:
    print(f"Token is valid with scopes: {result['scopes']}")
else:
    print(f"Token validation failed: {result['error']}")
```

## Security Considerations

1. **Credential Storage**: All credentials are encrypted at rest using Fernet encryption
2. **Keyring Integration**: Sensitive tokens stored in system keyring when available
3. **State Validation**: OAuth state parameter used to prevent CSRF attacks
4. **Token Expiry**: Automatic expiry checking with 5-minute refresh buffer
5. **Secure Defaults**: Restrictive file permissions (0600) for credential files

## Testing

Run the unit tests:

```bash
pytest tests/unit/test_auth_manager.py -v
```

Run the example script:

```bash
# Non-interactive mode
python examples/auth_example.py

# Interactive authentication
python examples/auth_example.py --interactive
```

## Environment Variables

The module respects the following environment variables:

- `PM_SUITE_GOOGLE_CLIENT_ID`: Default Google OAuth client ID
- `PM_SUITE_GOOGLE_CLIENT_SECRET`: Default Google OAuth client secret
- `PM_SUITE_ATLASSIAN_CLIENT_ID`: Default Atlassian OAuth client ID
- `PM_SUITE_ATLASSIAN_CLIENT_SECRET`: Default Atlassian OAuth client secret

## Troubleshooting

### "No keyring backend found"
The module will fall back to file-based storage if keyring is not available. Install keyring backend:
```bash
# macOS
brew install python-keyring

# Linux
sudo apt-get install python3-keyring

# Windows (usually works out of the box)
pip install keyring
```

### "Token refresh failed"
Ensure the refresh token is still valid. Some providers expire refresh tokens after a period of inactivity.

### "Invalid state - possible CSRF attack"
This occurs when the OAuth callback state doesn't match. Ensure you're not using multiple auth manager instances simultaneously.

## License

See the main project LICENSE file.