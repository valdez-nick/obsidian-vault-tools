"""
Jira Connector Implementation

Provides integration with Atlassian Jira for:
- Issue tracking and management
- Sprint data extraction
- JQL query execution
- Bulk operations
- Webhook handling
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from urllib.parse import quote, urljoin
import re

import pandas as pd
import aiohttp
from aiohttp import ClientSession, ClientTimeout
from atlassian import Jira
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base_connector import DataSourceConnector
from authentication.auth_manager import AuthenticationManager
from authentication.credentials_helper import OAuthCredential

logger = logging.getLogger(__name__)


class JiraRateLimiter:
    """Rate limiter for Jira API calls."""
    
    def __init__(self, requests_per_second: float = 10):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
        """
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request."""
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                await asyncio.sleep(sleep_time)
            
            self.last_request_time = time.time()


class JQLBuilder:
    """Helper class for building JQL queries."""
    
    @staticmethod
    def build(
        project: Optional[Union[str, List[str]]] = None,
        issue_type: Optional[Union[str, List[str]]] = None,
        status: Optional[Union[str, List[str]]] = None,
        assignee: Optional[str] = None,
        reporter: Optional[str] = None,
        sprint: Optional[Union[int, str]] = None,
        labels: Optional[List[str]] = None,
        components: Optional[List[str]] = None,
        created_after: Optional[datetime] = None,
        updated_after: Optional[datetime] = None,
        custom_fields: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Build a JQL query string from parameters.
        
        Args:
            project: Project key(s)
            issue_type: Issue type(s)
            status: Status(es)
            assignee: Assignee username or 'currentUser()'
            reporter: Reporter username or 'currentUser()'
            sprint: Sprint ID or name
            labels: List of labels
            components: List of components
            created_after: Issues created after this date
            updated_after: Issues updated after this date
            custom_fields: Dictionary of custom field filters
            order_by: Field to order by (e.g., 'created DESC')
            **kwargs: Additional JQL parameters
            
        Returns:
            JQL query string
        """
        conditions = []
        
        # Handle project
        if project:
            if isinstance(project, list):
                projects = ','.join(f'"{p}"' for p in project)
                conditions.append(f"project in ({projects})")
            else:
                conditions.append(f'project = "{project}"')
        
        # Handle issue type
        if issue_type:
            if isinstance(issue_type, list):
                types = ','.join(f'"{t}"' for t in issue_type)
                conditions.append(f"issuetype in ({types})")
            else:
                conditions.append(f'issuetype = "{issue_type}"')
        
        # Handle status
        if status:
            if isinstance(status, list):
                statuses = ','.join(f'"{s}"' for s in status)
                conditions.append(f"status in ({statuses})")
            else:
                conditions.append(f'status = "{status}"')
        
        # Handle assignee
        if assignee:
            conditions.append(f"assignee = {assignee}")
        
        # Handle reporter
        if reporter:
            conditions.append(f"reporter = {reporter}")
        
        # Handle sprint
        if sprint:
            if isinstance(sprint, int):
                conditions.append(f"sprint = {sprint}")
            else:
                conditions.append(f'sprint = "{sprint}"')
        
        # Handle labels
        if labels:
            for label in labels:
                conditions.append(f'labels = "{label}"')
        
        # Handle components
        if components:
            for component in components:
                conditions.append(f'component = "{component}"')
        
        # Handle date filters
        if created_after:
            date_str = created_after.strftime("%Y-%m-%d")
            conditions.append(f'created >= "{date_str}"')
        
        if updated_after:
            date_str = updated_after.strftime("%Y-%m-%d")
            conditions.append(f'updated >= "{date_str}"')
        
        # Handle custom fields
        if custom_fields:
            for field, value in custom_fields.items():
                if isinstance(value, list):
                    values = ','.join(f'"{v}"' for v in value)
                    conditions.append(f'"{field}" in ({values})')
                else:
                    conditions.append(f'"{field}" = "{value}"')
        
        # Handle additional kwargs
        for key, value in kwargs.items():
            conditions.append(f'{key} = "{value}"')
        
        # Build query
        jql = " AND ".join(conditions)
        
        # Add ordering
        if order_by:
            jql += f" ORDER BY {order_by}"
        
        return jql


class JiraConnector(DataSourceConnector):
    """
    Connector for Atlassian Jira REST API v3.
    
    Supports OAuth 2.0 authentication and provides methods for
    extracting project, sprint, and issue data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Jira connector with configuration.
        
        Config should include:
        - jira_url: Base URL of Jira instance
        - auth_method: 'oauth2' or 'api_token'
        - For OAuth2:
          - client_id: OAuth client ID
          - client_secret: OAuth client secret
          - tenant_id: Tenant/workspace ID
        - For API token:
          - email: User email for authentication
          - api_token: API token for authentication
        - default_project: Optional default project key
        - rate_limit: Requests per second (default: 10)
        """
        super().__init__("Jira", config)
        self.base_url = config.get('jira_url', '').rstrip('/')
        self.auth_method = config.get('auth_method', 'api_token')
        self.default_project = config.get('default_project')
        self.rate_limit = config.get('rate_limit', 10)
        
        # Initialize components
        self.session: Optional[ClientSession] = None
        self.jira_client: Optional[Jira] = None
        self.auth_manager: Optional[AuthenticationManager] = None
        self.rate_limiter = JiraRateLimiter(self.rate_limit)
        self.jql_builder = JQLBuilder()
        
        # Cache for metadata
        self._metadata_cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=30)
        
        # Webhook configuration
        self._webhook_callbacks: Dict[str, List[callable]] = {}
        self._webhook_server = None
        
    async def _get_oauth_headers(self) -> Dict[str, str]:
        """Get OAuth headers for API requests."""
        if self.auth_method != 'oauth2':
            return {}
        
        credential = self.auth_manager.get_valid_credential(
            "atlassian",
            self.config.get('tenant_id')
        )
        
        if not credential:
            raise ValueError("No valid OAuth credential found")
        
        return {
            "Authorization": f"{credential.token_type} {credential.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    def connect(self) -> bool:
        """
        Establish connection to Jira instance.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.auth_method == 'oauth2':
                # Initialize OAuth authentication
                self.auth_manager = AuthenticationManager()
                
                # Run async authentication in sync context
                loop = asyncio.get_event_loop()
                credential = loop.run_until_complete(
                    self.auth_manager.authenticate(
                        provider="atlassian",
                        tenant_id=self.config.get('tenant_id'),
                        client_id=self.config.get('client_id'),
                        client_secret=self.config.get('client_secret')
                    )
                )
                
                # Create session with OAuth headers
                self.session = ClientSession(
                    timeout=ClientTimeout(total=30),
                    headers={
                        "Authorization": f"{credential.token_type} {credential.access_token}",
                        "Accept": "application/json"
                    }
                )
                
            else:
                # Use API token authentication with atlassian-python-api
                self.jira_client = Jira(
                    url=self.base_url,
                    username=self.config.get('email'),
                    password=self.config.get('api_token'),
                    cloud=True  # Assuming Jira Cloud
                )
            
            # Validate connection
            self.is_connected = self.validate_connection()
            
            if self.is_connected:
                logger.info(f"Successfully connected to Jira at {self.base_url}")
            else:
                logger.error(f"Failed to connect to Jira at {self.base_url}")
            
            return self.is_connected
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.last_error = str(e)
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Jira instance.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            if self.session:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.session.close())
                self.session = None
            
            self.jira_client = None
            self.is_connected = False
            
            logger.info("Disconnected from Jira")
            return True
            
        except Exception as e:
            logger.error(f"Disconnection error: {e}")
            self.last_error = str(e)
            return False
    
    def validate_connection(self) -> bool:
        """
        Validate Jira connection by testing API access.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            if self.auth_method == 'oauth2' and self.session:
                # Test OAuth connection
                loop = asyncio.get_event_loop()
                response = loop.run_until_complete(
                    self._make_request("GET", "/rest/api/3/myself")
                )
                return response is not None
            
            elif self.jira_client:
                # Test API token connection
                self.jira_client.myself()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make an API request with retry logic and rate limiting.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            
        Returns:
            Response data or None if failed
        """
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        url = urljoin(self.base_url, endpoint)
        
        try:
            if self.auth_method == 'oauth2' and self.session:
                headers = await self._get_oauth_headers()
                
                async with self.session.request(
                    method,
                    url,
                    json=data,
                    params=params,
                    headers=headers
                ) as response:
                    self.log_response(endpoint, response.status, 0)  # TODO: Add proper timing
                    
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        # Handle rate limiting
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited. Retrying after {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        raise aiohttp.ClientError("Rate limited")
                    else:
                        error_text = await response.text()
                        logger.error(f"Request failed: {response.status} - {error_text}")
                        return None
            else:
                # Fallback to sync client
                # This is a simplified version - in production, you'd want to handle this better
                logger.warning("Using sync client in async context")
                return None
                
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise
    
    def extract_data(self, query: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract data from Jira based on query parameters.
        
        Query parameters:
        - jql: JQL query string (or use other params to build JQL)
        - fields: List of fields to retrieve
        - expand: List of fields to expand
        - max_results: Maximum number of results
        - include_changelog: Include issue changelog
        - include_comments: Include issue comments
        
        Returns:
            DataFrame containing issue data
        """
        try:
            # Build JQL if not provided
            if 'jql' not in query:
                query['jql'] = self.jql_builder.build(**query)
            
            jql = query.get('jql', '')
            fields = query.get('fields', ['summary', 'status', 'assignee', 'created', 'updated'])
            expand = query.get('expand', [])
            max_results = query.get('max_results', 1000)
            include_changelog = query.get('include_changelog', False)
            include_comments = query.get('include_comments', False)
            
            logger.info(f"Extracting Jira data with JQL: {jql}")
            
            all_issues = []
            start_at = 0
            
            if self.jira_client:
                # Use atlassian-python-api
                while True:
                    results = self.jira_client.jql(
                        jql,
                        start=start_at,
                        limit=min(100, max_results - len(all_issues)),
                        fields=fields,
                        expand=','.join(expand) if expand else None
                    )
                    
                    issues = results.get('issues', [])
                    if not issues:
                        break
                    
                    all_issues.extend(issues)
                    
                    if len(all_issues) >= max_results or len(issues) < 100:
                        break
                    
                    start_at += len(issues)
            
            else:
                # Use OAuth session
                loop = asyncio.get_event_loop()
                
                while True:
                    params = {
                        'jql': jql,
                        'startAt': start_at,
                        'maxResults': min(100, max_results - len(all_issues)),
                        'fields': ','.join(fields) if fields else None,
                        'expand': ','.join(expand) if expand else None
                    }
                    
                    response = loop.run_until_complete(
                        self._make_request("GET", "/rest/api/3/search", params=params)
                    )
                    
                    if not response:
                        break
                    
                    issues = response.get('issues', [])
                    if not issues:
                        break
                    
                    all_issues.extend(issues)
                    
                    if len(all_issues) >= max_results or len(issues) < 100:
                        break
                    
                    start_at += len(issues)
            
            # Process additional data if requested
            if include_changelog or include_comments:
                loop = asyncio.get_event_loop()
                for issue in all_issues:
                    if include_changelog:
                        issue['changelog'] = loop.run_until_complete(
                            self._get_issue_changelog(issue['key'])
                        )
                    if include_comments:
                        issue['comments'] = loop.run_until_complete(
                            self._get_issue_comments(issue['key'])
                        )
            
            # Convert to DataFrame
            df = self._issues_to_dataframe(all_issues)
            
            logger.info(f"Extracted {len(df)} issues from Jira")
            return df
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            self.last_error = str(e)
            return pd.DataFrame()
    
    def _issues_to_dataframe(self, issues: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert Jira issues to DataFrame with flattened structure."""
        if not issues:
            return pd.DataFrame()
        
        flattened_issues = []
        
        for issue in issues:
            flat_issue = {
                'key': issue.get('key'),
                'id': issue.get('id'),
                'self': issue.get('self'),
                'created': issue.get('fields', {}).get('created'),
                'updated': issue.get('fields', {}).get('updated'),
            }
            
            # Flatten fields
            fields = issue.get('fields', {})
            for field_name, field_value in fields.items():
                if isinstance(field_value, dict):
                    # Handle nested objects like assignee, reporter
                    if 'displayName' in field_value:
                        flat_issue[f'{field_name}_name'] = field_value.get('displayName')
                        flat_issue[f'{field_name}_email'] = field_value.get('emailAddress')
                    elif 'name' in field_value:
                        flat_issue[f'{field_name}_name'] = field_value.get('name')
                    else:
                        flat_issue[field_name] = json.dumps(field_value)
                elif isinstance(field_value, list):
                    # Handle lists like labels, components
                    if field_value and isinstance(field_value[0], dict):
                        flat_issue[field_name] = json.dumps(field_value)
                    else:
                        flat_issue[field_name] = field_value
                else:
                    flat_issue[field_name] = field_value
            
            flattened_issues.append(flat_issue)
        
        df = pd.DataFrame(flattened_issues)
        
        # Convert date columns
        date_columns = ['created', 'updated', 'resolutiondate', 'duedate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get Jira instance metadata.
        
        Returns:
            Dictionary containing:
            - version: Jira version
            - projects: Available projects
            - issue_types: Available issue types
            - statuses: Available statuses
            - custom_fields: Custom field definitions
            - priorities: Available priorities
            - users: List of users (if permitted)
        """
        try:
            # Check cache
            if 'metadata' in self._metadata_cache:
                if datetime.now() < self._cache_expiry.get('metadata', datetime.min):
                    return self._metadata_cache['metadata']
            
            metadata = {
                "version": "Unknown",
                "projects": [],
                "issue_types": [],
                "statuses": [],
                "custom_fields": [],
                "priorities": [],
                "users": []
            }
            
            if self.jira_client:
                # Get server info
                try:
                    server_info = self.jira_client.get_server_info()
                    metadata['version'] = server_info.get('version', 'Unknown')
                except:
                    pass
                
                # Get projects
                projects = self.jira_client.projects()
                metadata['projects'] = [
                    {
                        'key': p.get('key'),
                        'name': p.get('name'),
                        'id': p.get('id')
                    }
                    for p in projects
                ]
                
                # Get issue types
                issue_types = self.jira_client.get_all_issue_types()
                metadata['issue_types'] = [
                    {
                        'id': it.get('id'),
                        'name': it.get('name'),
                        'description': it.get('description')
                    }
                    for it in issue_types
                ]
                
                # Get statuses
                statuses = self.jira_client.get_all_statuses()
                metadata['statuses'] = [
                    {
                        'id': s.get('id'),
                        'name': s.get('name'),
                        'category': s.get('statusCategory', {}).get('name')
                    }
                    for s in statuses
                ]
                
                # Get custom fields
                fields = self.jira_client.get_all_fields()
                metadata['custom_fields'] = [
                    {
                        'id': f.get('id'),
                        'name': f.get('name'),
                        'type': f.get('schema', {}).get('type') if f.get('schema') else None,
                        'custom': f.get('custom', False)
                    }
                    for f in fields
                    if f.get('custom', False)
                ]
                
                # Get priorities
                priorities = self.jira_client.get_all_priorities()
                metadata['priorities'] = [
                    {
                        'id': p.get('id'),
                        'name': p.get('name')
                    }
                    for p in priorities
                ]
            
            # Cache metadata
            self._metadata_cache['metadata'] = metadata
            self._cache_expiry['metadata'] = datetime.now() + self._cache_ttl
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            return {
                "version": "Unknown",
                "projects": [],
                "issue_types": [],
                "custom_fields": []
            }
    
    def get_sprint_data(self, board_id: int, sprint_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get sprint data for a specific board.
        
        Args:
            board_id: Jira board ID
            sprint_id: Optional specific sprint ID (gets active sprint if not provided)
            
        Returns:
            DataFrame with sprint information and metrics
        """
        try:
            sprints_data = []
            
            if self.jira_client:
                # Get board info
                board = self.jira_client.get_board(board_id)
                
                if sprint_id:
                    # Get specific sprint
                    sprint = self.jira_client.get_sprint(sprint_id)
                    sprints = [sprint]
                else:
                    # Get active sprints
                    sprints = self.jira_client.get_all_sprints_from_board(
                        board_id,
                        state='active'
                    )
                
                for sprint in sprints:
                    sprint_data = {
                        'id': sprint.get('id'),
                        'name': sprint.get('name'),
                        'state': sprint.get('state'),
                        'startDate': sprint.get('startDate'),
                        'endDate': sprint.get('endDate'),
                        'goal': sprint.get('goal'),
                        'board_id': board_id,
                        'board_name': board.get('name')
                    }
                    
                    # Get sprint issues
                    issues = self.jira_client.get_all_issues_for_sprint(sprint['id'])
                    
                    # Calculate sprint metrics
                    total_issues = len(issues)
                    completed_issues = sum(
                        1 for issue in issues
                        if issue.get('fields', {}).get('status', {}).get('statusCategory', {}).get('key') == 'done'
                    )
                    
                    sprint_data['total_issues'] = total_issues
                    sprint_data['completed_issues'] = completed_issues
                    sprint_data['completion_rate'] = (
                        completed_issues / total_issues if total_issues > 0 else 0
                    )
                    
                    # Calculate story points if available
                    story_points_field = self._get_story_points_field()
                    if story_points_field:
                        total_points = sum(
                            float(issue.get('fields', {}).get(story_points_field, 0) or 0)
                            for issue in issues
                        )
                        completed_points = sum(
                            float(issue.get('fields', {}).get(story_points_field, 0) or 0)
                            for issue in issues
                            if issue.get('fields', {}).get('status', {}).get('statusCategory', {}).get('key') == 'done'
                        )
                        
                        sprint_data['total_story_points'] = total_points
                        sprint_data['completed_story_points'] = completed_points
                        sprint_data['velocity'] = completed_points
                    
                    sprints_data.append(sprint_data)
            
            df = pd.DataFrame(sprints_data)
            
            # Convert date columns
            date_columns = ['startDate', 'endDate']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get sprint data: {e}")
            return pd.DataFrame()
    
    def _get_story_points_field(self) -> Optional[str]:
        """Get the custom field ID for story points."""
        metadata = self.get_metadata()
        custom_fields = metadata.get('custom_fields', [])
        
        # Common names for story points field
        story_point_names = ['Story Points', 'Story Point', 'StoryPoints', 'SP']
        
        for field in custom_fields:
            if field.get('name') in story_point_names:
                return field.get('id')
        
        return None
    
    def create_issue(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new issue in Jira.
        
        Args:
            issue_data: Issue fields and values
            Required fields:
            - project: Project key
            - issuetype: Issue type name
            - summary: Issue summary
            Optional fields:
            - description: Issue description
            - assignee: Assignee account ID
            - priority: Priority name
            - labels: List of labels
            - Any custom fields
            
        Returns:
            Created issue data including key and ID
        """
        try:
            if self.jira_client:
                # Format issue data for atlassian-python-api
                fields = {
                    'project': {'key': issue_data.get('project')},
                    'issuetype': {'name': issue_data.get('issuetype')},
                    'summary': issue_data.get('summary')
                }
                
                # Add optional fields
                if 'description' in issue_data:
                    fields['description'] = issue_data['description']
                
                if 'assignee' in issue_data:
                    fields['assignee'] = {'accountId': issue_data['assignee']}
                
                if 'priority' in issue_data:
                    fields['priority'] = {'name': issue_data['priority']}
                
                if 'labels' in issue_data:
                    fields['labels'] = issue_data['labels']
                
                # Add custom fields
                for key, value in issue_data.items():
                    if key.startswith('customfield_'):
                        fields[key] = value
                
                # Create issue
                result = self.jira_client.issue_create(fields)
                
                logger.info(f"Created issue: {result.get('key')}")
                return result
            
            else:
                # Use OAuth session
                loop = asyncio.get_event_loop()
                response = loop.run_until_complete(
                    self._make_request(
                        "POST",
                        "/rest/api/3/issue",
                        data={'fields': issue_data}
                    )
                )
                
                if response:
                    logger.info(f"Created issue: {response.get('key')}")
                    return response
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"Failed to create issue: {e}")
            self.last_error = str(e)
            return {}
    
    def update_issue(self, issue_key: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing issue.
        
        Args:
            issue_key: Issue key (e.g., 'PROJ-123')
            update_data: Fields to update
            
        Returns:
            True if update successful
        """
        try:
            if self.jira_client:
                # Update issue
                self.jira_client.update_issue_field(
                    issue_key,
                    update_data
                )
                logger.info(f"Updated issue: {issue_key}")
                return True
            
            else:
                # Use OAuth session
                loop = asyncio.get_event_loop()
                response = loop.run_until_complete(
                    self._make_request(
                        "PUT",
                        f"/rest/api/3/issue/{issue_key}",
                        data={'fields': update_data}
                    )
                )
                
                if response is not None:
                    logger.info(f"Updated issue: {issue_key}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to update issue {issue_key}: {e}")
            return False
    
    def bulk_create_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create multiple issues in a single request.
        
        Args:
            issues: List of issue data dictionaries
            
        Returns:
            List of created issues with keys and IDs
        """
        try:
            if self.jira_client:
                # atlassian-python-api doesn't have bulk create, so we'll do it one by one
                # In production, you might want to implement proper bulk API calls
                created_issues = []
                
                for issue_data in issues:
                    result = self.create_issue(issue_data)
                    if result:
                        created_issues.append(result)
                
                logger.info(f"Bulk created {len(created_issues)} issues")
                return created_issues
            
            else:
                # Use OAuth session for bulk create
                loop = asyncio.get_event_loop()
                
                # Format for bulk create API
                bulk_data = {
                    'issueUpdates': [
                        {'fields': issue_data}
                        for issue_data in issues
                    ]
                }
                
                response = loop.run_until_complete(
                    self._make_request(
                        "POST",
                        "/rest/api/3/issue/bulk",
                        data=bulk_data
                    )
                )
                
                if response:
                    created_issues = response.get('issues', [])
                    logger.info(f"Bulk created {len(created_issues)} issues")
                    return created_issues
                return []
                
        except Exception as e:
            logger.error(f"Bulk create failed: {e}")
            return []
    
    def get_issue_dependencies(self, issue_key: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get issue dependencies (blocks/is blocked by).
        
        Args:
            issue_key: Issue key
            
        Returns:
            Dictionary with 'blocks' and 'blocked_by' lists
        """
        try:
            dependencies = {
                'blocks': [],
                'blocked_by': []
            }
            
            if self.jira_client:
                # Get issue with links expanded
                issue = self.jira_client.issue(issue_key, expand='issuelinks')
                
                for link in issue.get('fields', {}).get('issuelinks', []):
                    link_type = link.get('type', {})
                    
                    if link_type.get('name') == 'Blocks':
                        if 'outwardIssue' in link:
                            # This issue blocks the linked issue
                            linked_issue = link['outwardIssue']
                            dependencies['blocks'].append({
                                'key': linked_issue.get('key'),
                                'summary': linked_issue.get('fields', {}).get('summary'),
                                'status': linked_issue.get('fields', {}).get('status', {}).get('name')
                            })
                        elif 'inwardIssue' in link:
                            # This issue is blocked by the linked issue
                            linked_issue = link['inwardIssue']
                            dependencies['blocked_by'].append({
                                'key': linked_issue.get('key'),
                                'summary': linked_issue.get('fields', {}).get('summary'),
                                'status': linked_issue.get('fields', {}).get('status', {}).get('name')
                            })
            
            return dependencies
            
        except Exception as e:
            logger.error(f"Failed to get dependencies for {issue_key}: {e}")
            return {'blocks': [], 'blocked_by': []}
    
    async def _get_issue_changelog(self, issue_key: str) -> List[Dict[str, Any]]:
        """Get issue changelog."""
        try:
            response = await self._make_request(
                "GET",
                f"/rest/api/3/issue/{issue_key}/changelog"
            )
            
            if response:
                return response.get('values', [])
            return []
            
        except Exception as e:
            logger.error(f"Failed to get changelog for {issue_key}: {e}")
            return []
    
    async def _get_issue_comments(self, issue_key: str) -> List[Dict[str, Any]]:
        """Get issue comments."""
        try:
            response = await self._make_request(
                "GET",
                f"/rest/api/3/issue/{issue_key}/comment"
            )
            
            if response:
                return response.get('comments', [])
            return []
            
        except Exception as e:
            logger.error(f"Failed to get comments for {issue_key}: {e}")
            return []
    
    def register_webhook(self, name: str, url: str, events: List[str],
                        filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Register a webhook for real-time updates.
        
        Args:
            name: Webhook name
            url: Callback URL
            events: List of Jira events to listen for
            filters: Optional JQL filter
            
        Returns:
            Webhook registration details
        """
        try:
            webhook_data = {
                'name': name,
                'url': url,
                'events': events,
                'filters': filters or {},
                'excludeBody': False
            }
            
            if self.jira_client:
                # Note: atlassian-python-api doesn't have webhook support
                # You would need to implement this using the REST API directly
                logger.warning("Webhook registration not implemented for API token auth")
                return {}
            
            else:
                loop = asyncio.get_event_loop()
                response = loop.run_until_complete(
                    self._make_request(
                        "POST",
                        "/rest/webhooks/1.0/webhook",
                        data=webhook_data
                    )
                )
                
                if response:
                    logger.info(f"Registered webhook: {name}")
                    return response
                return {}
                
        except Exception as e:
            logger.error(f"Failed to register webhook: {e}")
            return {}
    
    def handle_webhook_callback(self, event_type: str, callback: callable):
        """
        Register a callback for webhook events.
        
        Args:
            event_type: Jira event type
            callback: Function to call when event occurs
        """
        if event_type not in self._webhook_callbacks:
            self._webhook_callbacks[event_type] = []
        
        self._webhook_callbacks[event_type].append(callback)
        logger.info(f"Registered callback for event: {event_type}")
    
    # Helper methods for common PM workflows
    
    def get_sprint_progress(self, board_id: int, sprint_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get detailed sprint progress metrics.
        
        Args:
            board_id: Board ID
            sprint_id: Sprint ID (uses active sprint if not provided)
            
        Returns:
            Dictionary with sprint progress metrics
        """
        try:
            sprint_df = self.get_sprint_data(board_id, sprint_id)
            
            if sprint_df.empty:
                return {}
            
            sprint = sprint_df.iloc[0]
            
            # Get sprint issues with more detail
            jql = f'sprint = {sprint["id"]}'
            issues_df = self.extract_data({
                'jql': jql,
                'fields': ['status', 'issuetype', 'priority', 'created', 'updated', 'resolutiondate']
            })
            
            # Calculate metrics
            total_issues = len(issues_df)
            
            # Status breakdown
            status_counts = issues_df['status_name'].value_counts().to_dict() if 'status_name' in issues_df else {}
            
            # Issue type breakdown
            type_counts = issues_df['issuetype_name'].value_counts().to_dict() if 'issuetype_name' in issues_df else {}
            
            # Calculate burn rate
            if pd.notna(sprint['startDate']) and pd.notna(sprint['endDate']):
                sprint_duration = (sprint['endDate'] - sprint['startDate']).days
                days_elapsed = (datetime.now() - pd.to_datetime(sprint['startDate'])).days
                
                if days_elapsed > 0 and sprint_duration > 0:
                    expected_completion = (days_elapsed / sprint_duration) * 100
                    actual_completion = sprint['completion_rate'] * 100
                    burn_rate = actual_completion / expected_completion if expected_completion > 0 else 0
                else:
                    burn_rate = 0
            else:
                burn_rate = 0
            
            return {
                'sprint_id': sprint['id'],
                'sprint_name': sprint['name'],
                'state': sprint['state'],
                'start_date': sprint['startDate'].isoformat() if pd.notna(sprint['startDate']) else None,
                'end_date': sprint['endDate'].isoformat() if pd.notna(sprint['endDate']) else None,
                'total_issues': total_issues,
                'completed_issues': sprint['completed_issues'],
                'completion_rate': sprint['completion_rate'],
                'burn_rate': burn_rate,
                'status_breakdown': status_counts,
                'issue_type_breakdown': type_counts,
                'velocity': sprint.get('velocity', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get sprint progress: {e}")
            return {}
    
    def get_team_velocity(self, board_id: int, num_sprints: int = 5) -> pd.DataFrame:
        """
        Calculate team velocity over recent sprints.
        
        Args:
            board_id: Board ID
            num_sprints: Number of recent sprints to analyze
            
        Returns:
            DataFrame with velocity metrics per sprint
        """
        try:
            if self.jira_client:
                # Get completed sprints
                sprints = self.jira_client.get_all_sprints_from_board(
                    board_id,
                    state='closed'
                )
                
                # Sort by end date and take most recent
                sprints = sorted(
                    sprints,
                    key=lambda s: s.get('endDate', ''),
                    reverse=True
                )[:num_sprints]
                
                velocity_data = []
                
                for sprint in sprints:
                    sprint_df = self.get_sprint_data(board_id, sprint['id'])
                    
                    if not sprint_df.empty:
                        sprint_info = sprint_df.iloc[0]
                        
                        velocity_data.append({
                            'sprint_id': sprint['id'],
                            'sprint_name': sprint['name'],
                            'start_date': sprint.get('startDate'),
                            'end_date': sprint.get('endDate'),
                            'completed_story_points': sprint_info.get('completed_story_points', 0),
                            'total_story_points': sprint_info.get('total_story_points', 0),
                            'completed_issues': sprint_info.get('completed_issues', 0),
                            'total_issues': sprint_info.get('total_issues', 0)
                        })
                
                df = pd.DataFrame(velocity_data)
                
                # Calculate rolling averages
                if not df.empty:
                    df['avg_velocity_3'] = df['completed_story_points'].rolling(window=3, min_periods=1).mean()
                    df['avg_velocity_5'] = df['completed_story_points'].rolling(window=5, min_periods=1).mean()
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to calculate team velocity: {e}")
            return pd.DataFrame()
    
    def get_burndown_data(self, sprint_id: int) -> pd.DataFrame:
        """
        Get burndown chart data for a sprint.
        
        Args:
            sprint_id: Sprint ID
            
        Returns:
            DataFrame with daily burndown data
        """
        try:
            # This would require accessing Jira's burndown report API
            # which is not directly exposed in the standard REST API
            # You would typically need to use the Agile REST API
            
            logger.warning("Burndown data retrieval not fully implemented")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get burndown data: {e}")
            return pd.DataFrame()
    
    def build_jql_query(self, **kwargs) -> str:
        """
        Build a JQL query string from parameters.
        
        This is a convenience method that delegates to the JQLBuilder.
        
        Args:
            **kwargs: Query parameters
            
        Returns:
            JQL query string
        """
        return self.jql_builder.build(**kwargs)