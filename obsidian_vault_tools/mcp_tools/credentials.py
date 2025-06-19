"""
Secure credential management for MCP servers
"""

import json
import os
import getpass
from pathlib import Path
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
import base64
import hashlib


class CredentialManager:
    """Secure credential storage and management"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or (Path.home() / '.obsidian_vault_tools')
        self.config_dir.mkdir(exist_ok=True)
        
        self.credentials_file = self.config_dir / 'credentials.json'
        self.key_file = self.config_dir / '.cred_key'
        
        # Ensure key file is created securely
        self._ensure_key_file()
        
    def _ensure_key_file(self):
        """Ensure encryption key exists"""
        if not self.key_file.exists():
            # Generate new key
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(self.key_file, 0o600)
    
    def _get_cipher(self) -> Fernet:
        """Get encryption cipher"""
        with open(self.key_file, 'rb') as f:
            key = f.read()
        return Fernet(key)
    
    def _load_credentials(self) -> Dict[str, Any]:
        """Load encrypted credentials"""
        if not self.credentials_file.exists():
            return {}
        
        try:
            cipher = self._get_cipher()
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()
            
            if not encrypted_data:
                return {}
                
            decrypted_data = cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            print(f"Error loading credentials: {e}")
            return {}
    
    def _save_credentials(self, credentials: Dict[str, Any]):
        """Save encrypted credentials"""
        try:
            cipher = self._get_cipher()
            data = json.dumps(credentials, indent=2).encode()
            encrypted_data = cipher.encrypt(data)
            
            with open(self.credentials_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(self.credentials_file, 0o600)
        except Exception as e:
            print(f"Error saving credentials: {e}")
    
    def set_credential(self, key: str, value: str):
        """Set a credential value"""
        credentials = self._load_credentials()
        credentials[key] = value
        self._save_credentials(credentials)
    
    def get_credential(self, key: str, prompt: bool = True) -> Optional[str]:
        """Get a credential value, prompting if necessary"""
        credentials = self._load_credentials()
        
        if key in credentials:
            return credentials[key]
        
        # Check environment variables
        env_value = os.getenv(key)
        if env_value:
            return env_value
        
        # Check environment variable variations
        env_variations = [
            key.upper(),
            key.upper().replace('-', '_'),
            key.upper().replace(' ', '_')
        ]
        
        for env_key in env_variations:
            env_value = os.getenv(env_key)
            if env_value:
                return env_value
        
        if prompt:
            # Prompt user for credential
            friendly_name = key.replace('_', ' ').replace('-', ' ').title()
            value = getpass.getpass(f"Enter {friendly_name}: ")
            if value:
                self.set_credential(key, value)
                return value
        
        return None
    
    def delete_credential(self, key: str):
        """Delete a credential"""
        credentials = self._load_credentials()
        if key in credentials:
            del credentials[key]
            self._save_credentials(credentials)
    
    def list_credentials(self) -> list[str]:
        """List stored credential keys (not values)"""
        credentials = self._load_credentials()
        return list(credentials.keys())
    
    def clear_all_credentials(self):
        """Clear all stored credentials"""
        self._save_credentials({})
    
    def substitute_placeholders(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Replace credential placeholders in configuration"""
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Replace [PLACEHOLDER] with actual credentials
                if obj.startswith('[') and obj.endswith(']'):
                    placeholder = obj[1:-1]
                    credential = self.get_credential(placeholder, prompt=False)
                    return credential if credential else obj
                return obj
            else:
                return obj
        
        return substitute_recursive(config)
    
    def validate_credentials(self, required_keys: list[str]) -> Dict[str, bool]:
        """Validate that required credentials are available"""
        results = {}
        for key in required_keys:
            credential = self.get_credential(key, prompt=False)
            results[key] = credential is not None
        return results


# Global credential manager instance
_credential_manager = None

def get_credential_manager() -> CredentialManager:
    """Get global credential manager instance"""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager()
    return _credential_manager