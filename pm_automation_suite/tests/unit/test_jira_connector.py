"""
Unit tests for JiraConnector
"""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pytest
import pandas as pd
import aiohttp
from aiohttp import ClientTimeout

from connectors.jira_connector import JiraConnector, JQLBuilder, JiraRateLimiter
from authentication.credentials_helper import OAuthCredential


class TestJQLBuilder:
    """Test JQL query builder."""
    
    def test_simple_project_query(self):
        """Test building query with single project."""
        jql = JQLBuilder.build(project="PROJ")
        assert jql == 'project = "PROJ"'
    
    def test_multiple_projects_query(self):
        """Test building query with multiple projects."""
        jql = JQLBuilder.build(project=["PROJ1", "PROJ2"])
        assert jql == 'project in ("PROJ1","PROJ2")'
    
    def test_complex_query(self):
        """Test building complex query with multiple parameters."""
        jql = JQLBuilder.build(
            project="PROJ",
            status=["Open", "In Progress"],
            assignee="currentUser()",
            labels=["backend", "bug"],
            created_after=datetime(2024, 1, 1),
            order_by="created DESC"
        )
        
        assert 'project = "PROJ"' in jql
        assert 'status in ("Open","In Progress")' in jql
        assert 'assignee = currentUser()' in jql
        assert 'labels = "backend"' in jql
        assert 'labels = "bug"' in jql
        assert 'created >= "2024-01-01"' in jql
        assert 'ORDER BY created DESC' in jql
    
    def test_sprint_query(self):
        """Test building query with sprint."""
        jql = JQLBuilder.build(sprint=123)
        assert jql == "sprint = 123"
        
        jql = JQLBuilder.build(sprint="Sprint 1")
        assert jql == 'sprint = "Sprint 1"'
    
    def test_custom_fields(self):
        """Test building query with custom fields."""
        jql = JQLBuilder.build(
            custom_fields={
                "Story Points": 5,
                "Epic Link": ["EPIC-1", "EPIC-2"]
            }
        )
        
        assert '"Story Points" = "5"' in jql
        assert '"Epic Link" in ("EPIC-1","EPIC-2")' in jql
    
    def test_empty_query(self):
        """Test building empty query."""
        jql = JQLBuilder.build()
        assert jql == ""


class TestJiraRateLimiter:
    """Test rate limiter functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test that rate limiter enforces request intervals."""
        limiter = JiraRateLimiter(requests_per_second=10)
        
        start_time = asyncio.get_event_loop().time()
        
        # Make rapid requests
        for _ in range(3):
            await limiter.acquire()
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Should take at least 0.2 seconds (3 requests at 10/sec)
        assert elapsed >= 0.2
    
    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting(self):
        """Test rate limiter with concurrent requests."""
        limiter = JiraRateLimiter(requests_per_second=5)
        
        async def make_request():
            await limiter.acquire()
            return asyncio.get_event_loop().time()
        
        # Make 5 concurrent requests
        times = await asyncio.gather(*[make_request() for _ in range(5)])
        
        # Check that requests are properly spaced
        for i in range(1, len(times)):
            interval = times[i] - times[i-1]
            assert interval >= 0.19  # Allow small margin for 0.2 sec interval


class TestJiraConnector:
    """Test JiraConnector functionality."""
    
    @pytest.fixture
    def api_token_config(self):
        """Configuration for API token authentication."""
        return {
            'jira_url': 'https://example.atlassian.net',
            'auth_method': 'api_token',
            'email': 'test@example.com',
            'api_token': 'test_token',
            'default_project': 'PROJ',
            'rate_limit': 10
        }
    
    @pytest.fixture
    def oauth2_config(self):
        """Configuration for OAuth2 authentication."""
        return {
            'jira_url': 'https://example.atlassian.net',
            'auth_method': 'oauth2',
            'client_id': 'test_client_id',
            'client_secret': 'test_client_secret',
            'tenant_id': 'test_tenant',
            'default_project': 'PROJ',
            'rate_limit': 10
        }
    
    @pytest.fixture
    def mock_jira_client(self):
        """Mock Jira client."""
        client = Mock()
        client.myself.return_value = {'accountId': '123', 'displayName': 'Test User'}
        client.projects.return_value = [
            {'key': 'PROJ', 'name': 'Project', 'id': '10000'}
        ]
        return client
    
    def test_init_api_token(self, api_token_config):
        """Test initialization with API token config."""
        connector = JiraConnector(api_token_config)
        
        assert connector.base_url == 'https://example.atlassian.net'
        assert connector.auth_method == 'api_token'
        assert connector.default_project == 'PROJ'
        assert connector.rate_limit == 10
    
    def test_init_oauth2(self, oauth2_config):
        """Test initialization with OAuth2 config."""
        connector = JiraConnector(oauth2_config)
        
        assert connector.base_url == 'https://example.atlassian.net'
        assert connector.auth_method == 'oauth2'
        assert connector.default_project == 'PROJ'
    
    @patch('connectors.jira_connector.Jira')
    def test_connect_api_token(self, mock_jira_class, api_token_config, mock_jira_client):
        """Test connection with API token."""
        mock_jira_class.return_value = mock_jira_client
        
        connector = JiraConnector(api_token_config)
        result = connector.connect()
        
        assert result is True
        assert connector.is_connected is True
        mock_jira_class.assert_called_once_with(
            url='https://example.atlassian.net',
            username='test@example.com',
            password='test_token',
            cloud=True
        )
    
    @patch('connectors.jira_connector.AuthenticationManager')
    @patch('connectors.jira_connector.ClientSession')
    def test_connect_oauth2(self, mock_session_class, mock_auth_manager_class, oauth2_config):
        """Test connection with OAuth2."""
        # Mock OAuth credential
        mock_credential = OAuthCredential(
            provider='atlassian',
            tenant_id='test_tenant',
            client_id='test_client_id',
            client_secret='test_client_secret',
            access_token='test_access_token',
            token_type='Bearer',
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        # Mock auth manager
        mock_auth_manager = Mock()
        mock_auth_manager.authenticate = AsyncMock(return_value=mock_credential)
        mock_auth_manager_class.return_value = mock_auth_manager
        
        # Mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        connector = JiraConnector(oauth2_config)
        
        # Mock validate_connection to return True
        with patch.object(connector, 'validate_connection', return_value=True):
            result = connector.connect()
        
        assert result is True
        assert connector.is_connected is True
        assert connector.auth_manager is not None
    
    def test_disconnect(self, api_token_config):
        """Test disconnection."""
        connector = JiraConnector(api_token_config)
        connector.jira_client = Mock()
        connector.is_connected = True
        
        result = connector.disconnect()
        
        assert result is True
        assert connector.is_connected is False
        assert connector.jira_client is None
    
    @patch('connectors.jira_connector.Jira')
    def test_validate_connection_api_token(self, mock_jira_class, api_token_config, mock_jira_client):
        """Test connection validation with API token."""
        connector = JiraConnector(api_token_config)
        connector.jira_client = mock_jira_client
        
        result = connector.validate_connection()
        
        assert result is True
        mock_jira_client.myself.assert_called_once()
    
    def test_validate_connection_failed(self, api_token_config):
        """Test connection validation failure."""
        connector = JiraConnector(api_token_config)
        connector.jira_client = Mock()
        connector.jira_client.myself.side_effect = Exception("Connection failed")
        
        result = connector.validate_connection()
        
        assert result is False
    
    @patch('connectors.jira_connector.Jira')
    def test_extract_data_with_jql(self, mock_jira_class, api_token_config, mock_jira_client):
        """Test data extraction with JQL query."""
        # Mock issue data
        mock_issues = {
            'issues': [
                {
                    'key': 'PROJ-1',
                    'id': '10001',
                    'fields': {
                        'summary': 'Test Issue 1',
                        'status': {'name': 'Open'},
                        'created': '2024-01-01T00:00:00.000+0000'
                    }
                },
                {
                    'key': 'PROJ-2',
                    'id': '10002',
                    'fields': {
                        'summary': 'Test Issue 2',
                        'status': {'name': 'Done'},
                        'created': '2024-01-02T00:00:00.000+0000'
                    }
                }
            ]
        }
        
        mock_jira_client.jql.return_value = mock_issues
        
        connector = JiraConnector(api_token_config)
        connector.jira_client = mock_jira_client
        
        query = {
            'jql': 'project = PROJ',
            'fields': ['summary', 'status', 'created'],
            'max_results': 100
        }
        
        df = connector.extract_data(query)
        
        assert len(df) == 2
        assert 'key' in df.columns
        assert df.iloc[0]['key'] == 'PROJ-1'
        assert df.iloc[1]['key'] == 'PROJ-2'
    
    @patch('connectors.jira_connector.Jira')
    def test_extract_data_build_jql(self, mock_jira_class, api_token_config, mock_jira_client):
        """Test data extraction with automatic JQL building."""
        mock_issues = {'issues': []}
        mock_jira_client.jql.return_value = mock_issues
        
        connector = JiraConnector(api_token_config)
        connector.jira_client = mock_jira_client
        
        query = {
            'project': 'PROJ',
            'status': 'Open',
            'max_results': 50
        }
        
        df = connector.extract_data(query)
        
        # Verify JQL was built and used
        mock_jira_client.jql.assert_called()
        call_args = mock_jira_client.jql.call_args[1]
        assert 'project = "PROJ"' in call_args['jql']
        assert 'status = "Open"' in call_args['jql']
    
    @patch('connectors.jira_connector.Jira')
    def test_get_metadata(self, mock_jira_class, api_token_config, mock_jira_client):
        """Test metadata retrieval."""
        # Mock metadata responses
        mock_jira_client.get_server_info.return_value = {'version': '8.20.0'}
        mock_jira_client.projects.return_value = [
            {'key': 'PROJ', 'name': 'Project', 'id': '10000'}
        ]
        mock_jira_client.get_all_issue_types.return_value = [
            {'id': '10001', 'name': 'Story', 'description': 'User story'}
        ]
        mock_jira_client.get_all_statuses.return_value = [
            {'id': '1', 'name': 'Open', 'statusCategory': {'name': 'To Do'}}
        ]
        mock_jira_client.get_all_fields.return_value = [
            {'id': 'customfield_10001', 'name': 'Story Points', 'custom': True, 'schema': {'type': 'number'}}
        ]
        mock_jira_client.get_all_priorities.return_value = [
            {'id': '1', 'name': 'High'}
        ]
        
        connector = JiraConnector(api_token_config)
        connector.jira_client = mock_jira_client
        
        metadata = connector.get_metadata()
        
        assert metadata['version'] == '8.20.0'
        assert len(metadata['projects']) == 1
        assert metadata['projects'][0]['key'] == 'PROJ'
        assert len(metadata['issue_types']) == 1
        assert len(metadata['custom_fields']) == 1
        assert metadata['custom_fields'][0]['name'] == 'Story Points'
    
    @patch('connectors.jira_connector.Jira')
    def test_metadata_caching(self, mock_jira_class, api_token_config, mock_jira_client):
        """Test that metadata is cached."""
        mock_jira_client.get_server_info.return_value = {'version': '8.20.0'}
        mock_jira_client.projects.return_value = []
        
        connector = JiraConnector(api_token_config)
        connector.jira_client = mock_jira_client
        
        # First call
        metadata1 = connector.get_metadata()
        assert mock_jira_client.get_server_info.call_count == 1
        
        # Second call should use cache
        metadata2 = connector.get_metadata()
        assert mock_jira_client.get_server_info.call_count == 1
        assert metadata1 == metadata2
    
    @patch('connectors.jira_connector.Jira')
    def test_create_issue(self, mock_jira_class, api_token_config, mock_jira_client):
        """Test issue creation."""
        mock_jira_client.issue_create.return_value = {
            'key': 'PROJ-3',
            'id': '10003',
            'self': 'https://example.atlassian.net/rest/api/3/issue/10003'
        }
        
        connector = JiraConnector(api_token_config)
        connector.jira_client = mock_jira_client
        
        issue_data = {
            'project': 'PROJ',
            'issuetype': 'Story',
            'summary': 'New test issue',
            'description': 'Test description',
            'labels': ['test', 'automated']
        }
        
        result = connector.create_issue(issue_data)
        
        assert result['key'] == 'PROJ-3'
        mock_jira_client.issue_create.assert_called_once()
        
        # Check field formatting
        call_args = mock_jira_client.issue_create.call_args[0][0]
        assert call_args['project']['key'] == 'PROJ'
        assert call_args['issuetype']['name'] == 'Story'
        assert call_args['summary'] == 'New test issue'
        assert call_args['labels'] == ['test', 'automated']
    
    @patch('connectors.jira_connector.Jira')
    def test_update_issue(self, mock_jira_class, api_token_config, mock_jira_client):
        """Test issue update."""
        connector = JiraConnector(api_token_config)
        connector.jira_client = mock_jira_client
        
        update_data = {
            'summary': 'Updated summary',
            'status': 'In Progress'
        }
        
        result = connector.update_issue('PROJ-1', update_data)
        
        assert result is True
        mock_jira_client.update_issue_field.assert_called_once_with(
            'PROJ-1',
            update_data
        )
    
    @patch('connectors.jira_connector.Jira')
    def test_bulk_create_issues(self, mock_jira_class, api_token_config, mock_jira_client):
        """Test bulk issue creation."""
        # Mock individual create responses
        mock_jira_client.issue_create.side_effect = [
            {'key': 'PROJ-4', 'id': '10004'},
            {'key': 'PROJ-5', 'id': '10005'}
        ]
        
        connector = JiraConnector(api_token_config)
        connector.jira_client = mock_jira_client
        
        issues = [
            {
                'project': 'PROJ',
                'issuetype': 'Story',
                'summary': 'Bulk issue 1'
            },
            {
                'project': 'PROJ',
                'issuetype': 'Bug',
                'summary': 'Bulk issue 2'
            }
        ]
        
        results = connector.bulk_create_issues(issues)
        
        assert len(results) == 2
        assert results[0]['key'] == 'PROJ-4'
        assert results[1]['key'] == 'PROJ-5'
        assert mock_jira_client.issue_create.call_count == 2
    
    @patch('connectors.jira_connector.Jira')
    def test_get_sprint_data(self, mock_jira_class, api_token_config, mock_jira_client):
        """Test sprint data retrieval."""
        # Mock board and sprint data
        mock_jira_client.get_board.return_value = {
            'id': 1,
            'name': 'Test Board'
        }
        
        mock_jira_client.get_all_sprints_from_board.return_value = [
            {
                'id': 10,
                'name': 'Sprint 1',
                'state': 'active',
                'startDate': '2024-01-01T00:00:00.000Z',
                'endDate': '2024-01-14T00:00:00.000Z',
                'goal': 'Sprint goal'
            }
        ]
        
        mock_jira_client.get_all_issues_for_sprint.return_value = [
            {
                'key': 'PROJ-1',
                'fields': {
                    'status': {'statusCategory': {'key': 'done'}},
                    'customfield_10001': 5  # Story points
                }
            },
            {
                'key': 'PROJ-2',
                'fields': {
                    'status': {'statusCategory': {'key': 'indeterminate'}},
                    'customfield_10001': 3
                }
            }
        ]
        
        connector = JiraConnector(api_token_config)
        connector.jira_client = mock_jira_client
        
        # Mock story points field
        with patch.object(connector, '_get_story_points_field', return_value='customfield_10001'):
            df = connector.get_sprint_data(board_id=1)
        
        assert len(df) == 1
        assert df.iloc[0]['name'] == 'Sprint 1'
        assert df.iloc[0]['total_issues'] == 2
        assert df.iloc[0]['completed_issues'] == 1
        assert df.iloc[0]['completion_rate'] == 0.5
        assert df.iloc[0]['total_story_points'] == 8
        assert df.iloc[0]['completed_story_points'] == 5
    
    @patch('connectors.jira_connector.Jira')
    def test_get_issue_dependencies(self, mock_jira_class, api_token_config, mock_jira_client):
        """Test getting issue dependencies."""
        mock_issue = {
            'fields': {
                'issuelinks': [
                    {
                        'type': {'name': 'Blocks'},
                        'outwardIssue': {
                            'key': 'PROJ-2',
                            'fields': {
                                'summary': 'Blocked issue',
                                'status': {'name': 'Open'}
                            }
                        }
                    },
                    {
                        'type': {'name': 'Blocks'},
                        'inwardIssue': {
                            'key': 'PROJ-3',
                            'fields': {
                                'summary': 'Blocking issue',
                                'status': {'name': 'Done'}
                            }
                        }
                    }
                ]
            }
        }
        
        mock_jira_client.issue.return_value = mock_issue
        
        connector = JiraConnector(api_token_config)
        connector.jira_client = mock_jira_client
        
        dependencies = connector.get_issue_dependencies('PROJ-1')
        
        assert len(dependencies['blocks']) == 1
        assert dependencies['blocks'][0]['key'] == 'PROJ-2'
        assert len(dependencies['blocked_by']) == 1
        assert dependencies['blocked_by'][0]['key'] == 'PROJ-3'
    
    @patch('connectors.jira_connector.Jira')
    def test_get_sprint_progress(self, mock_jira_class, api_token_config, mock_jira_client):
        """Test sprint progress calculation."""
        # Mock sprint data
        mock_sprint_df = pd.DataFrame([{
            'id': 10,
            'name': 'Sprint 1',
            'state': 'active',
            'startDate': pd.Timestamp('2024-01-01'),
            'endDate': pd.Timestamp('2024-01-14'),
            'completed_issues': 5,
            'completion_rate': 0.5,
            'velocity': 20
        }])
        
        # Mock issues data
        mock_issues_df = pd.DataFrame([
            {'status_name': 'Done', 'issuetype_name': 'Story'},
            {'status_name': 'Done', 'issuetype_name': 'Bug'},
            {'status_name': 'In Progress', 'issuetype_name': 'Story'},
            {'status_name': 'Open', 'issuetype_name': 'Story'}
        ])
        
        connector = JiraConnector(api_token_config)
        
        with patch.object(connector, 'get_sprint_data', return_value=mock_sprint_df):
            with patch.object(connector, 'extract_data', return_value=mock_issues_df):
                progress = connector.get_sprint_progress(board_id=1)
        
        assert progress['sprint_id'] == 10
        assert progress['sprint_name'] == 'Sprint 1'
        assert progress['total_issues'] == 4
        assert progress['completed_issues'] == 5  # From sprint data
        assert progress['status_breakdown']['Done'] == 2
        assert progress['status_breakdown']['In Progress'] == 1
        assert progress['issue_type_breakdown']['Story'] == 3
    
    @patch('connectors.jira_connector.Jira')
    def test_get_team_velocity(self, mock_jira_class, api_token_config, mock_jira_client):
        """Test team velocity calculation."""
        # Mock completed sprints
        mock_jira_client.get_all_sprints_from_board.return_value = [
            {
                'id': 10,
                'name': 'Sprint 1',
                'startDate': '2024-01-01',
                'endDate': '2024-01-14'
            },
            {
                'id': 11,
                'name': 'Sprint 2',
                'startDate': '2024-01-15',
                'endDate': '2024-01-28'
            }
        ]
        
        connector = JiraConnector(api_token_config)
        connector.jira_client = mock_jira_client
        
        # Mock sprint data for each sprint
        sprint_data = [
            pd.DataFrame([{
                'completed_story_points': 20,
                'total_story_points': 25,
                'completed_issues': 8,
                'total_issues': 10
            }]),
            pd.DataFrame([{
                'completed_story_points': 25,
                'total_story_points': 30,
                'completed_issues': 10,
                'total_issues': 12
            }])
        ]
        
        with patch.object(connector, 'get_sprint_data', side_effect=sprint_data):
            df = connector.get_team_velocity(board_id=1, num_sprints=2)
        
        assert len(df) == 2
        assert df.iloc[0]['completed_story_points'] == 20
        assert df.iloc[1]['completed_story_points'] == 25
        assert 'avg_velocity_3' in df.columns
        assert 'avg_velocity_5' in df.columns
    
    def test_build_jql_query_method(self, api_token_config):
        """Test the build_jql_query convenience method."""
        connector = JiraConnector(api_token_config)
        
        jql = connector.build_jql_query(
            project="PROJ",
            status="Open",
            assignee="currentUser()"
        )
        
        assert 'project = "PROJ"' in jql
        assert 'status = "Open"' in jql
        assert 'assignee = currentUser()' in jql
    
    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, oauth2_config):
        """Test rate limiting in request handling."""
        connector = JiraConnector(oauth2_config)
        connector.session = AsyncMock()
        
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'test': 'data'})
        
        connector.session.request = AsyncMock(return_value=mock_response)
        
        # Mock OAuth headers
        with patch.object(connector, '_get_oauth_headers', return_value={}):
            # Make multiple requests
            start_time = asyncio.get_event_loop().time()
            
            results = await asyncio.gather(*[
                connector._make_request("GET", "/test")
                for _ in range(3)
            ])
            
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # With rate limit of 10/sec, 3 requests should take at least 0.2 sec
            assert elapsed >= 0.2
            assert all(r == {'test': 'data'} for r in results)
    
    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, oauth2_config):
        """Test retry logic on rate limiting."""
        connector = JiraConnector(oauth2_config)
        connector.session = AsyncMock()
        
        # Mock rate limit response followed by success
        mock_response_429 = AsyncMock()
        mock_response_429.status = 429
        mock_response_429.headers = {'Retry-After': '1'}
        mock_response_429.text = AsyncMock(return_value='Rate limited')
        
        mock_response_200 = AsyncMock()
        mock_response_200.status = 200
        mock_response_200.json = AsyncMock(return_value={'success': True})
        
        connector.session.request = AsyncMock(
            side_effect=[mock_response_429, mock_response_200]
        )
        
        with patch.object(connector, '_get_oauth_headers', return_value={}):
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                result = await connector._make_request("GET", "/test")
        
        assert result == {'success': True}
        mock_sleep.assert_called_with(1)
        assert connector.session.request.call_count == 2
    
    def test_webhook_registration(self, api_token_config):
        """Test webhook registration."""
        connector = JiraConnector(api_token_config)
        
        # Should warn for API token auth
        result = connector.register_webhook(
            name="Test Webhook",
            url="https://example.com/webhook",
            events=["jira:issue_created", "jira:issue_updated"]
        )
        
        assert result == {}
    
    def test_webhook_callback_registration(self, api_token_config):
        """Test webhook callback registration."""
        connector = JiraConnector(api_token_config)
        
        def callback(event):
            pass
        
        connector.handle_webhook_callback("jira:issue_created", callback)
        
        assert "jira:issue_created" in connector._webhook_callbacks
        assert callback in connector._webhook_callbacks["jira:issue_created"]
    
    def test_error_handling_connection(self, api_token_config):
        """Test error handling during connection."""
        connector = JiraConnector(api_token_config)
        
        with patch('connectors.jira_connector.Jira', side_effect=Exception("Connection failed")):
            result = connector.connect()
        
        assert result is False
        assert connector.last_error == "Connection failed"
    
    def test_error_handling_data_extraction(self, api_token_config):
        """Test error handling during data extraction."""
        connector = JiraConnector(api_token_config)
        connector.jira_client = Mock()
        connector.jira_client.jql.side_effect = Exception("API error")
        
        df = connector.extract_data({'jql': 'project = PROJ'})
        
        assert df.empty
        assert connector.last_error == "API error"