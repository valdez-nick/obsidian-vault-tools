"""
Settings Management

Handles application settings and configuration:
- Environment-based configuration
- Feature flags
- Default values
- Configuration validation
- Hot-reloading support
"""

import os
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
import logging
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)


@dataclass
class FeatureFlags:
    """
    Feature flags for controlling functionality.
    """
    enable_ai_insights: bool = True
    enable_auto_scheduling: bool = True
    enable_slack_notifications: bool = False
    enable_email_reports: bool = True
    enable_advanced_analytics: bool = False
    enable_beta_features: bool = False
    max_concurrent_workflows: int = 5
    cache_ttl_minutes: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_ai_insights": self.enable_ai_insights,
            "enable_auto_scheduling": self.enable_auto_scheduling,
            "enable_slack_notifications": self.enable_slack_notifications,
            "enable_email_reports": self.enable_email_reports,
            "enable_advanced_analytics": self.enable_advanced_analytics,
            "enable_beta_features": self.enable_beta_features,
            "max_concurrent_workflows": self.max_concurrent_workflows,
            "cache_ttl_minutes": self.cache_ttl_minutes
        }


@dataclass
class DataSourceSettings:
    """
    Settings for data source connections.
    """
    jira: Dict[str, Any] = field(default_factory=dict)
    confluence: Dict[str, Any] = field(default_factory=dict)
    snowflake: Dict[str, Any] = field(default_factory=dict)
    google: Dict[str, Any] = field(default_factory=dict)
    
    # Connection settings
    connection_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5
    rate_limit_per_minute: int = 60


@dataclass
class WorkflowSettings:
    """
    Settings for workflow execution.
    """
    default_timeout_minutes: int = 60
    max_parallel_steps: int = 10
    enable_dry_run: bool = False
    save_execution_history: bool = True
    history_retention_days: int = 90
    notification_channels: List[str] = field(default_factory=list)


class Settings:
    """
    Main settings manager for PM Automation Suite.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.environment = os.getenv('PM_SUITE_ENV', 'development')
        
        # Initialize settings
        self.feature_flags = FeatureFlags()
        self.data_sources = DataSourceSettings()
        self.workflows = WorkflowSettings()
        self.custom_settings: Dict[str, Any] = {}
        
        # Load configuration
        self.load()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        home = Path.home()
        config_dir = home / '.pm_automation_suite'
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / 'config.json')
        
    def load(self):
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
                        
                self._apply_config(config)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
        else:
            logger.info("No configuration file found, using defaults")
            self.save()  # Save default configuration
            
    def save(self):
        """Save current configuration to file."""
        config = self.to_dict()
        
        try:
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    json.dump(config, f, indent=2)
                    
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            
    def _apply_config(self, config: Dict[str, Any]):
        """Apply configuration dictionary to settings."""
        # Apply feature flags
        if 'feature_flags' in config:
            for key, value in config['feature_flags'].items():
                if hasattr(self.feature_flags, key):
                    setattr(self.feature_flags, key, value)
                    
        # Apply data source settings
        if 'data_sources' in config:
            for key, value in config['data_sources'].items():
                if hasattr(self.data_sources, key):
                    setattr(self.data_sources, key, value)
                    
        # Apply workflow settings
        if 'workflows' in config:
            for key, value in config['workflows'].items():
                if hasattr(self.workflows, key):
                    setattr(self.workflows, key, value)
                    
        # Apply custom settings
        if 'custom' in config:
            self.custom_settings = config['custom']
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value by key.
        
        Args:
            key: Setting key (dot notation supported)
            default: Default value if key not found
            
        Returns:
            Setting value
        """
        # Support dot notation
        parts = key.split('.')
        value = self
        
        try:
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = getattr(value, part)
            return value if value is not None else default
        except (AttributeError, KeyError):
            return default
            
    def set(self, key: str, value: Any):
        """
        Set a setting value.
        
        Args:
            key: Setting key (dot notation supported)
            value: Value to set
        """
        # Support dot notation
        parts = key.split('.')
        target = self
        
        # Navigate to parent
        for part in parts[:-1]:
            if isinstance(target, dict):
                if part not in target:
                    target[part] = {}
                target = target[part]
            else:
                if not hasattr(target, part):
                    setattr(target, part, {})
                target = getattr(target, part)
                
        # Set the value
        final_key = parts[-1]
        if isinstance(target, dict):
            target[final_key] = value
        else:
            setattr(target, final_key, value)
            
    def update_feature_flag(self, flag_name: str, enabled: bool):
        """
        Update a feature flag.
        
        Args:
            flag_name: Name of the feature flag
            enabled: Whether to enable or disable
        """
        if hasattr(self.feature_flags, flag_name):
            setattr(self.feature_flags, flag_name, enabled)
            logger.info(f"Updated feature flag {flag_name} to {enabled}")
            self.save()
        else:
            logger.warning(f"Unknown feature flag: {flag_name}")
            
    def validate(self) -> List[str]:
        """
        Validate current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate data source settings
        if self.data_sources.connection_timeout <= 0:
            errors.append("Connection timeout must be positive")
            
        if self.data_sources.max_retries < 0:
            errors.append("Max retries cannot be negative")
            
        # Validate workflow settings
        if self.workflows.default_timeout_minutes <= 0:
            errors.append("Default timeout must be positive")
            
        if self.workflows.history_retention_days < 0:
            errors.append("History retention days cannot be negative")
            
        return errors
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "environment": self.environment,
            "feature_flags": self.feature_flags.to_dict(),
            "data_sources": {
                "jira": self.data_sources.jira,
                "confluence": self.data_sources.confluence,
                "snowflake": self.data_sources.snowflake,
                "google": self.data_sources.google,
                "connection_timeout": self.data_sources.connection_timeout,
                "max_retries": self.data_sources.max_retries,
                "retry_delay": self.data_sources.retry_delay,
                "rate_limit_per_minute": self.data_sources.rate_limit_per_minute
            },
            "workflows": {
                "default_timeout_minutes": self.workflows.default_timeout_minutes,
                "max_parallel_steps": self.workflows.max_parallel_steps,
                "enable_dry_run": self.workflows.enable_dry_run,
                "save_execution_history": self.workflows.save_execution_history,
                "history_retention_days": self.workflows.history_retention_days,
                "notification_channels": self.workflows.notification_channels
            },
            "custom": self.custom_settings
        }
        
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        env_configs = {
            "development": {
                "debug": True,
                "log_level": "DEBUG",
                "cache_enabled": False
            },
            "staging": {
                "debug": False,
                "log_level": "INFO",
                "cache_enabled": True
            },
            "production": {
                "debug": False,
                "log_level": "WARNING",
                "cache_enabled": True
            }
        }
        
        return env_configs.get(self.environment, env_configs["development"])


# Global settings instance
settings = Settings()