"""
Credentials Helper for OAuth and API Key Management.

Provides utilities for managing OAuth credentials, API keys,
and other authentication credentials with secure storage.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
import keyring
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


@dataclass
class OAuthCredential:
    """
    Represents an OAuth 2.0 credential set.
    
    Attributes:
        provider: OAuth provider (e.g., 'google', 'atlassian')
        tenant_id: Tenant/workspace identifier for multi-tenant support
        client_id: OAuth client ID
        client_secret: OAuth client secret (encrypted)
        access_token: Current access token (encrypted)
        refresh_token: Refresh token for obtaining new access tokens (encrypted)
        token_type: Type of token (usually 'Bearer')
        expires_at: When the access token expires
        scopes: List of authorized scopes
        metadata: Additional provider-specific metadata
    """
    provider: str
    tenant_id: str
    client_id: str
    client_secret: str  # Encrypted
    access_token: Optional[str] = None  # Encrypted
    refresh_token: Optional[str] = None  # Encrypted
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None
    scopes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if the access token is expired."""
        if not self.expires_at:
            return True
        # Add 5 minute buffer for token refresh
        return datetime.utcnow() >= (self.expires_at - timedelta(minutes=5))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OAuthCredential':
        """Create from dictionary."""
        if 'expires_at' in data and data['expires_at']:
            if isinstance(data['expires_at'], str):
                data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)


class CredentialsHelper:
    """
    Helper class for managing OAuth and API credentials.
    
    Provides secure storage, retrieval, and validation of credentials
    with support for multiple tenants and providers.
    """
    
    def __init__(self, storage_path: Optional[str] = None, use_keyring: bool = True):
        """
        Initialize credentials helper.
        
        Args:
            storage_path: Path for encrypted credential storage
            use_keyring: Whether to use system keyring for sensitive data
        """
        self.storage_path = storage_path or self._get_default_storage_path()
        self.use_keyring = use_keyring and self._keyring_available()
        self._cipher_suite: Optional[Fernet] = None
        self._credentials: Dict[str, Dict[str, OAuthCredential]] = {}  # provider -> tenant_id -> credential
        
        # Initialize encryption
        self._init_encryption()
        
        # Load existing credentials
        self.load_credentials()
        
    def _get_default_storage_path(self) -> str:
        """Get default OAuth credential storage path."""
        home = Path.home()
        cred_dir = home / '.pm_automation_suite' / 'oauth'
        cred_dir.mkdir(parents=True, exist_ok=True)
        return str(cred_dir / 'oauth_credentials.enc')
    
    def _keyring_available(self) -> bool:
        """Check if system keyring is available."""
        try:
            keyring.get_keyring()
            return True
        except Exception:
            logger.warning("System keyring not available, using file-based storage only")
            return False
    
    def _init_encryption(self):
        """Initialize encryption for credential storage."""
        key = self._get_or_create_key()
        self._cipher_suite = Fernet(key)
    
    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key."""
        key_path = Path(self.storage_path).parent / '.oauth_key'
        
        if key_path.exists():
            with open(key_path, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            import os
            os.chmod(key_path, 0o600)
            return key
    
    def store_credential(self, credential: OAuthCredential) -> bool:
        """
        Store an OAuth credential securely.
        
        Args:
            credential: OAuthCredential object to store
            
        Returns:
            True if stored successfully
        """
        try:
            # Store sensitive tokens in keyring if available
            if self.use_keyring:
                keyring_base = f"pm_oauth_{credential.provider}_{credential.tenant_id}"
                
                if credential.access_token:
                    keyring.set_password(
                        "pm_automation_suite",
                        f"{keyring_base}_access",
                        credential.access_token
                    )
                
                if credential.refresh_token:
                    keyring.set_password(
                        "pm_automation_suite",
                        f"{keyring_base}_refresh",
                        credential.refresh_token
                    )
                
                if credential.client_secret:
                    keyring.set_password(
                        "pm_automation_suite",
                        f"{keyring_base}_secret",
                        credential.client_secret
                    )
            
            # Store credential in memory
            if credential.provider not in self._credentials:
                self._credentials[credential.provider] = {}
            
            self._credentials[credential.provider][credential.tenant_id] = credential
            
            # Persist to disk
            self.save_credentials()
            
            logger.info(f"Stored OAuth credential for {credential.provider}/{credential.tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store credential: {e}")
            return False
    
    def get_credential(self, provider: str, tenant_id: str) -> Optional[OAuthCredential]:
        """
        Retrieve an OAuth credential.
        
        Args:
            provider: OAuth provider name
            tenant_id: Tenant identifier
            
        Returns:
            OAuthCredential if found, None otherwise
        """
        if provider not in self._credentials or tenant_id not in self._credentials[provider]:
            return None
        
        credential = self._credentials[provider][tenant_id]
        
        # Retrieve sensitive data from keyring if available
        if self.use_keyring:
            keyring_base = f"pm_oauth_{provider}_{tenant_id}"
            
            try:
                access_token = keyring.get_password(
                    "pm_automation_suite",
                    f"{keyring_base}_access"
                )
                if access_token:
                    credential.access_token = access_token
                
                refresh_token = keyring.get_password(
                    "pm_automation_suite",
                    f"{keyring_base}_refresh"
                )
                if refresh_token:
                    credential.refresh_token = refresh_token
                
                client_secret = keyring.get_password(
                    "pm_automation_suite",
                    f"{keyring_base}_secret"
                )
                if client_secret:
                    credential.client_secret = client_secret
                    
            except Exception as e:
                logger.warning(f"Failed to retrieve from keyring: {e}")
        
        return credential
    
    def delete_credential(self, provider: str, tenant_id: str) -> bool:
        """
        Delete an OAuth credential.
        
        Args:
            provider: OAuth provider name
            tenant_id: Tenant identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            # Remove from keyring
            if self.use_keyring:
                keyring_base = f"pm_oauth_{provider}_{tenant_id}"
                for suffix in ['_access', '_refresh', '_secret']:
                    try:
                        keyring.delete_password(
                            "pm_automation_suite",
                            f"{keyring_base}{suffix}"
                        )
                    except Exception:
                        pass
            
            # Remove from storage
            if provider in self._credentials and tenant_id in self._credentials[provider]:
                del self._credentials[provider][tenant_id]
                if not self._credentials[provider]:
                    del self._credentials[provider]
                
                self.save_credentials()
                logger.info(f"Deleted OAuth credential for {provider}/{tenant_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete credential: {e}")
            return False
    
    def list_credentials(self) -> Dict[str, List[str]]:
        """
        List all stored credentials (without sensitive data).
        
        Returns:
            Dictionary of provider -> list of tenant IDs
        """
        result = {}
        for provider, tenants in self._credentials.items():
            result[provider] = list(tenants.keys())
        return result
    
    def update_tokens(self, provider: str, tenant_id: str, 
                     access_token: str, refresh_token: Optional[str] = None,
                     expires_in: Optional[int] = None) -> bool:
        """
        Update OAuth tokens for a credential.
        
        Args:
            provider: OAuth provider name
            tenant_id: Tenant identifier
            access_token: New access token
            refresh_token: New refresh token (if provided)
            expires_in: Token expiry in seconds
            
        Returns:
            True if updated successfully
        """
        credential = self.get_credential(provider, tenant_id)
        if not credential:
            logger.error(f"Credential not found: {provider}/{tenant_id}")
            return False
        
        credential.access_token = access_token
        if refresh_token:
            credential.refresh_token = refresh_token
        
        if expires_in:
            credential.expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        
        return self.store_credential(credential)
    
    def _encrypt(self, value: str) -> str:
        """Encrypt a string value."""
        if not self._cipher_suite:
            raise ValueError("Encryption not initialized")
        encrypted = self._cipher_suite.encrypt(value.encode())
        return encrypted.decode('utf-8')
    
    def _decrypt(self, encrypted_value: str) -> str:
        """Decrypt a string value."""
        if not self._cipher_suite:
            raise ValueError("Encryption not initialized")
        decrypted = self._cipher_suite.decrypt(encrypted_value.encode())
        return decrypted.decode('utf-8')
    
    def save_credentials(self):
        """Save credentials to encrypted file."""
        try:
            data = {}
            
            for provider, tenants in self._credentials.items():
                data[provider] = {}
                for tenant_id, credential in tenants.items():
                    # Create a copy without sensitive data for file storage
                    cred_data = credential.to_dict()
                    
                    # Encrypt sensitive fields if not using keyring
                    if not self.use_keyring:
                        if credential.access_token:
                            cred_data['access_token'] = self._encrypt(credential.access_token)
                        if credential.refresh_token:
                            cred_data['refresh_token'] = self._encrypt(credential.refresh_token)
                        if credential.client_secret:
                            cred_data['client_secret'] = self._encrypt(credential.client_secret)
                    else:
                        # Remove sensitive data when using keyring
                        cred_data.pop('access_token', None)
                        cred_data.pop('refresh_token', None)
                        cred_data.pop('client_secret', None)
                    
                    data[provider][tenant_id] = cred_data
            
            # Encrypt entire file
            json_data = json.dumps(data, indent=2)
            encrypted_data = self._encrypt(json_data)
            
            with open(self.storage_path, 'w') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            import os
            os.chmod(self.storage_path, 0o600)
            
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
    
    def load_credentials(self):
        """Load credentials from encrypted file."""
        if not Path(self.storage_path).exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                encrypted_data = f.read()
            
            json_data = self._decrypt(encrypted_data)
            data = json.loads(json_data)
            
            for provider, tenants in data.items():
                self._credentials[provider] = {}
                for tenant_id, cred_data in tenants.items():
                    # Decrypt sensitive fields if not using keyring
                    if not self.use_keyring:
                        if 'access_token' in cred_data and cred_data['access_token']:
                            cred_data['access_token'] = self._decrypt(cred_data['access_token'])
                        if 'refresh_token' in cred_data and cred_data['refresh_token']:
                            cred_data['refresh_token'] = self._decrypt(cred_data['refresh_token'])
                        if 'client_secret' in cred_data and cred_data['client_secret']:
                            cred_data['client_secret'] = self._decrypt(cred_data['client_secret'])
                    
                    credential = OAuthCredential.from_dict(cred_data)
                    self._credentials[provider][tenant_id] = credential
            
            logger.info(f"Loaded {sum(len(t) for t in self._credentials.values())} OAuth credentials")
            
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")