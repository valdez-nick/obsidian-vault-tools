"""
Data Source Connectors Module

This module provides connectors for various PM data sources:
- Jira: Issue tracking and sprint management
- Confluence: Documentation and knowledge base
- Snowflake: Analytics and metrics database
- Google Suite: Slides, Sheets, and Drive integration

Each connector implements the base DataSourceConnector interface
for consistent API across all data sources.
"""

from typing import List, Type

# Import connectors when implemented
# from .base_connector import DataSourceConnector
# from .jira_connector import JiraConnector
# from .confluence_connector import ConfluenceConnector
# from .snowflake_connector import SnowflakeConnector
# from .google_connector import GoogleConnector

__all__ = [
    "DataSourceConnector",
    "JiraConnector", 
    "ConfluenceConnector",
    "SnowflakeConnector",
    "GoogleConnector"
]