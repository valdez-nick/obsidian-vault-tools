"""
Obsidian Vault Manager - Core management functionality
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

class VaultManager:
    """Core vault management functionality"""
    
    def __init__(self):
        self.config_file = Path.home() / '.obsidian_vault_manager.json'
        self.config = self.load_config()
        self.current_vault = self.config.get('last_vault', '')
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                return {}
        return {}
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def set_vault_path(self, vault_path: str):
        """Set the vault path"""
        vault_path = os.path.expanduser(vault_path)
        if os.path.exists(vault_path):
            self.current_vault = vault_path
            self.config['last_vault'] = vault_path
            self.save_config()
            return True
        return False
    
    def get_vault_path(self) -> str:
        """Get current vault path"""
        return self.current_vault
    
    def list_vault_files(self, extension: str = '.md') -> list:
        """List files in vault with given extension"""
        if not self.current_vault or not os.path.exists(self.current_vault):
            return []
        
        files = []
        for root, dirs, filenames in os.walk(self.current_vault):
            for filename in filenames:
                if filename.endswith(extension):
                    files.append(os.path.join(root, filename))
        return files


class EnhancedVaultManager(VaultManager):
    """Enhanced vault manager with additional features"""
    
    def __init__(self):
        super().__init__()
        self.output_dirs = {}
    
    def get_output_directory(self, feature_name: str) -> str:
        """Get organized output directory for a specific feature"""
        if feature_name not in self.output_dirs:
            base_dir = os.path.dirname(self.current_vault) if self.current_vault else os.getcwd()
            output_dir = os.path.join(base_dir, f"{feature_name}-output")
            os.makedirs(output_dir, exist_ok=True)
            self.output_dirs[feature_name] = output_dir
        
        return self.output_dirs[feature_name]
    
    def analyze_vault_health(self) -> Dict[str, Any]:
        """Analyze vault health and return report"""
        if not self.current_vault:
            return {"error": "No vault configured"}
        
        md_files = self.list_vault_files('.md')
        
        return {
            "total_files": len(md_files),
            "vault_path": self.current_vault,
            "last_analyzed": datetime.now().isoformat(),
            "health_score": "Good" if len(md_files) > 0 else "Empty"
        }