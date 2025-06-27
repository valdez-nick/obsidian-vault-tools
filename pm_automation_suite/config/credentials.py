"""
Credentials Management

Secure credential storage and management:
- Encrypted credential storage
- Environment variable integration
- Keyring/keychain support
- Credential validation
- Rotation reminders
"""

import os
import json
import base64
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import keyring
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


@dataclass
class Credential:
    """
    Represents a stored credential.
    
    Attributes:
        service: Service name (jira, snowflake, etc.)
        name: Credential name (api_key, password, etc.)
        value: Encrypted credential value
        created_at: When credential was created
        updated_at: When credential was last updated
        expires_at: Optional expiration date
        metadata: Additional metadata
    """
    service: str
    name: str
    value: str  # Encrypted
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if credential is expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
        
    def days_until_expiry(self) -> Optional[int]:
        """Get days until expiry."""
        if self.expires_at:
            delta = self.expires_at - datetime.utcnow()
            return delta.days
        return None


class CredentialManager:
    """
    Manages secure credential storage and retrieval.
    """
    
    def __init__(self, storage_path: Optional[str] = None, 
                 use_keyring: bool = True):
        """
        Initialize credential manager.
        
        Args:
            storage_path: Path for encrypted credential storage
            use_keyring: Whether to use system keyring
        """
        self.storage_path = storage_path or self._get_default_storage_path()
        self.use_keyring = use_keyring and self._keyring_available()
        self.credentials: Dict[str, Dict[str, Credential]] = {}
        self._cipher_suite = None
        
        # Initialize encryption
        self._init_encryption()
        
        # Load existing credentials
        self.load_credentials()
        
    def _get_default_storage_path(self) -> str:
        """Get default credential storage path."""
        home = Path.home()
        cred_dir = home / '.pm_automation_suite' / 'credentials'
        cred_dir.mkdir(parents=True, exist_ok=True)
        return str(cred_dir / 'credentials.enc')
        
    def _keyring_available(self) -> bool:
        """Check if system keyring is available."""
        try:
            keyring.get_keyring()
            return True
        except Exception:
            return False
            
    def _init_encryption(self):
        """Initialize encryption for credential storage."""
        # Get or create encryption key
        key = self._get_or_create_key()
        self._cipher_suite = Fernet(key)
        
    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key."""
        key_path = Path(self.storage_path).parent / '.key'
        
        if key_path.exists():
            with open(key_path, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(key_path, 0o600)
            return key
            
    def set_credential(self, service: str, name: str, value: str,
                      expires_in_days: Optional[int] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        """
        Store a credential securely.
        
        Args:
            service: Service name (e.g., 'jira', 'snowflake')
            name: Credential name (e.g., 'api_key', 'password')
            value: Credential value (will be encrypted)
            expires_in_days: Optional expiration in days
            metadata: Optional metadata
        """
        # Use keyring if available for certain credentials
        if self.use_keyring and name in ['password', 'api_key', 'token']:
            keyring_key = f"pm_suite_{service}_{name}"
            try:
                keyring.set_password("pm_automation_suite", keyring_key, value)
                logger.info(f"Stored {service}.{name} in system keyring")
            except Exception as e:
                logger.warning(f"Failed to use keyring: {e}")
                
        # Always store encrypted copy
        encrypted_value = self._encrypt(value)
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
        credential = Credential(
            service=service,
            name=name,
            value=encrypted_value,
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        # Store in memory
        if service not in self.credentials:
            self.credentials[service] = {}
        self.credentials[service][name] = credential
        
        # Persist to disk
        self.save_credentials()
        
        logger.info(f"Stored credential {service}.{name}")
        
    def get_credential(self, service: str, name: str) -> Optional[str]:
        """
        Retrieve a credential.
        
        Args:
            service: Service name
            name: Credential name
            
        Returns:
            Decrypted credential value or None
        """
        # Check environment variable first
        env_var = f"PM_SUITE_{service.upper()}_{name.upper()}"
        env_value = os.getenv(env_var)
        if env_value:
            logger.debug(f"Using credential from environment: {env_var}")
            return env_value
            
        # Try keyring
        if self.use_keyring and name in ['password', 'api_key', 'token']:
            keyring_key = f"pm_suite_{service}_{name}"
            try:
                value = keyring.get_password("pm_automation_suite", keyring_key)
                if value:
                    logger.debug(f"Retrieved {service}.{name} from keyring")
                    return value
            except Exception as e:
                logger.warning(f"Failed to access keyring: {e}")
                
        # Get from encrypted storage
        if service in self.credentials and name in self.credentials[service]:
            credential = self.credentials[service][name]
            
            # Check expiration
            if credential.is_expired():
                logger.warning(f"Credential {service}.{name} has expired")
                return None
                
            # Warn if expiring soon
            days_left = credential.days_until_expiry()
            if days_left is not None and days_left < 7:
                logger.warning(f"Credential {service}.{name} expires in {days_left} days")
                
            return self._decrypt(credential.value)
            
        return None
        
    def delete_credential(self, service: str, name: str) -> bool:
        """
        Delete a credential.
        
        Args:
            service: Service name
            name: Credential name
            
        Returns:
            True if deleted successfully
        """
        # Remove from keyring
        if self.use_keyring:
            keyring_key = f"pm_suite_{service}_{name}"
            try:
                keyring.delete_password("pm_automation_suite", keyring_key)
            except Exception:
                pass
                
        # Remove from storage
        if service in self.credentials and name in self.credentials[service]:
            del self.credentials[service][name]
            if not self.credentials[service]:
                del self.credentials[service]
            self.save_credentials()
            logger.info(f"Deleted credential {service}.{name}")
            return True
            
        return False
        
    def list_credentials(self) -> Dict[str, List[str]]:
        """
        List all stored credentials (without values).
        
        Returns:
            Dictionary of service -> list of credential names
        """
        result = {}
        for service, creds in self.credentials.items():
            result[service] = list(creds.keys())
        return result
        
    def validate_credentials(self, service: str) -> Dict[str, bool]:
        """
        Validate credentials for a service.
        
        Args:
            service: Service to validate
            
        Returns:
            Dictionary of credential name -> is_valid
        """
        results = {}
        
        if service not in self.credentials:
            return results
            
        for name, credential in self.credentials[service].items():
            # Check if credential exists and is not expired
            value = self.get_credential(service, name)
            results[name] = value is not None and not credential.is_expired()
            
        return results
        
    def get_expiring_credentials(self, days: int = 7) -> List[Tuple[str, str, int]]:
        """
        Get credentials expiring within specified days.
        
        Args:
            days: Number of days to check
            
        Returns:
            List of (service, name, days_left) tuples
        """
        expiring = []
        
        for service, creds in self.credentials.items():
            for name, credential in creds.items():
                days_left = credential.days_until_expiry()
                if days_left is not None and days_left <= days:
                    expiring.append((service, name, days_left))
                    
        return sorted(expiring, key=lambda x: x[2])
        
    def _encrypt(self, value: str) -> str:
        """Encrypt a value."""
        encrypted = self._cipher_suite.encrypt(value.encode())
        return base64.b64encode(encrypted).decode()
        
    def _decrypt(self, encrypted_value: str) -> str:
        """Decrypt a value."""
        encrypted = base64.b64decode(encrypted_value.encode())
        decrypted = self._cipher_suite.decrypt(encrypted)
        return decrypted.decode()
        
    def save_credentials(self):
        """Save credentials to encrypted file."""
        data = {}
        
        for service, creds in self.credentials.items():
            data[service] = {}
            for name, credential in creds.items():
                data[service][name] = {
                    "value": credential.value,
                    "created_at": credential.created_at.isoformat(),
                    "updated_at": credential.updated_at.isoformat(),
                    "expires_at": credential.expires_at.isoformat() if credential.expires_at else None,
                    "metadata": credential.metadata
                }
                
        # Encrypt entire file
        json_data = json.dumps(data)
        encrypted_data = self._encrypt(json_data)
        
        with open(self.storage_path, 'w') as f:
            f.write(encrypted_data)
            
        # Set restrictive permissions
        os.chmod(self.storage_path, 0o600)
        
    def load_credentials(self):
        """Load credentials from encrypted file."""
        if not os.path.exists(self.storage_path):
            return
            
        try:
            with open(self.storage_path, 'r') as f:
                encrypted_data = f.read()
                
            json_data = self._decrypt(encrypted_data)
            data = json.loads(json_data)
            
            for service, creds in data.items():
                self.credentials[service] = {}
                for name, cred_data in creds.items():
                    expires_at = None
                    if cred_data.get('expires_at'):
                        expires_at = datetime.fromisoformat(cred_data['expires_at'])
                        
                    credential = Credential(
                        service=service,
                        name=name,
                        value=cred_data['value'],
                        created_at=datetime.fromisoformat(cred_data['created_at']),
                        updated_at=datetime.fromisoformat(cred_data['updated_at']),
                        expires_at=expires_at,
                        metadata=cred_data.get('metadata', {})
                    )
                    self.credentials[service][name] = credential
                    
            logger.info(f"Loaded {sum(len(c) for c in self.credentials.values())} credentials")
            
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")


# Global credential manager instance
credential_manager = CredentialManager()