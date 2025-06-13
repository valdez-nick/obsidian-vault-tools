"""
Configuration loader for Obsidian Librarian.

This module provides utilities to load configuration from YAML files
in the .obsidian-librarian directory.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import structlog

from .models import VaultConfig
from .services.git_service import GitConfig

logger = structlog.get_logger(__name__)


def load_vault_config_from_yaml(vault_path: Path) -> Optional[VaultConfig]:
    """
    Load vault configuration from .obsidian-librarian/config.yaml file.
    
    Args:
        vault_path: Path to the Obsidian vault
        
    Returns:
        VaultConfig object if config file exists and is valid, None otherwise
    """
    config_file = vault_path / ".obsidian-librarian" / "config.yaml"
    
    if not config_file.exists():
        logger.debug("No config.yaml found", path=config_file)
        return None
    
    try:
        with open(config_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        if not yaml_data:
            logger.warning("Empty config.yaml file", path=config_file)
            return None
        
        # Create VaultConfig with defaults
        vault_config = VaultConfig()
        
        # Update with YAML values if present
        if 'vault' in yaml_data:
            vault_settings = yaml_data['vault']
            if 'file_watching' in vault_settings:
                vault_config.enable_file_watching = vault_settings['file_watching']
            if 'cache_size' in vault_settings:
                vault_config.cache_size = vault_settings['cache_size']
        
        # Handle git configuration
        if 'git' in yaml_data:
            git_settings = yaml_data['git']
            vault_config.enable_git_integration = True
            
            if 'auto_backup' in git_settings:
                vault_config.enable_auto_backup = git_settings['auto_backup']
            if 'change_threshold' in git_settings:
                vault_config.git_auto_backup_threshold = git_settings['change_threshold']
            if 'commit_message_template' in git_settings:
                # Store for later use when creating GitService
                vault_config.git_commit_template = git_settings['commit_message_template']
            if 'branch' in git_settings:
                vault_config.git_branch = git_settings.get('branch', 'main')
        else:
            # No git configuration means git is disabled
            vault_config.enable_git_integration = False
        
        logger.info("Loaded vault configuration", 
                   git_enabled=vault_config.enable_git_integration,
                   auto_backup=vault_config.enable_auto_backup)
        
        return vault_config
        
    except yaml.YAMLError as e:
        logger.error("Failed to parse config.yaml", error=str(e))
        return None
    except Exception as e:
        logger.error("Failed to load config.yaml", error=str(e))
        return None


def create_git_config_from_yaml(yaml_data: Dict[str, Any]) -> GitConfig:
    """
    Create GitConfig from YAML git settings.
    
    Args:
        yaml_data: The 'git' section of the YAML config
        
    Returns:
        GitConfig object
    """
    config = GitConfig()
    
    if 'auto_backup' in yaml_data:
        config.auto_backup_enabled = yaml_data['auto_backup']
    if 'change_threshold' in yaml_data:
        config.auto_backup_threshold = yaml_data['change_threshold']
    if 'commit_message_template' in yaml_data:
        config.commit_message_template = yaml_data['commit_message_template']
    if 'include_stats' in yaml_data:
        config.include_stats_in_commit = yaml_data['include_stats']
    if 'branch' in yaml_data:
        config.default_branch = yaml_data['branch']
    
    return config