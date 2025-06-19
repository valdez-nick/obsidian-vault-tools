"""
Utility functions and configuration management
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Configuration management for vault tools"""
    
    def __init__(self):
        self.config_file = Path.home() / '.obsidian_vault_tools.json'
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                return self._default_config()
        return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "vault_path": "",
            "output_directory": "",
            "backup_settings": {
                "auto_backup": False,
                "backup_count": 5
            },
            "ui_settings": {
                "use_colors": True,
                "show_progress": True
            }
        }
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config()
    
    def set_vault_path(self, vault_path: str) -> bool:
        """Set and validate vault path"""
        vault_path = os.path.expanduser(vault_path)
        
        if os.path.exists(vault_path) and os.path.isdir(vault_path):
            self.set('vault_path', vault_path)
            return True
        
        return False
    
    def get_vault_path(self) -> str:
        """Get configured vault path"""
        return self.get('vault_path', '')
    
    def is_vault_configured(self) -> bool:
        """Check if vault is properly configured"""
        vault_path = self.get_vault_path()
        return vault_path and os.path.exists(vault_path)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def validate_vault_path(path: str) -> tuple[bool, str]:
    """Validate if path is a valid Obsidian vault"""
    path = os.path.expanduser(path)
    
    if not os.path.exists(path):
        return False, "Path does not exist"
    
    if not os.path.isdir(path):
        return False, "Path is not a directory"
    
    # Check for .obsidian directory (indicates Obsidian vault)
    obsidian_dir = os.path.join(path, '.obsidian')
    if not os.path.exists(obsidian_dir):
        return False, "No .obsidian directory found (not an Obsidian vault)"
    
    # Check for markdown files
    has_md_files = False
    for root, dirs, files in os.walk(path):
        if any(f.endswith('.md') for f in files):
            has_md_files = True
            break
    
    if not has_md_files:
        return False, "No markdown files found in vault"
    
    return True, "Valid Obsidian vault"


def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing problematic characters"""
    # Replace problematic characters
    replacements = {
        '/': '_',
        '\\': '_',
        ':': '_',
        '*': '_',
        '?': '_',
        '"': '_',
        '<': '_',
        '>': '_',
        '|': '_'
    }
    
    safe_name = filename
    for old, new in replacements.items():
        safe_name = safe_name.replace(old, new)
    
    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip('. ')
    
    # Ensure it's not empty
    if not safe_name:
        safe_name = "untitled"
    
    return safe_name


def count_files_by_extension(directory: str) -> Dict[str, int]:
    """Count files by extension in directory"""
    counts = {}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if not ext:
                ext = 'no_extension'
            counts[ext] = counts.get(ext, 0) + 1
    
    return counts