"""
Integration tests for JiraConnector with real Jira instance.

These tests require a real Jira instance and valid credentials.
They can be skipped if the environment variables are not set.
"""

import os
import asyncio
from datetime import datetime, timedelta
import pytest
import pandas as pd
from dotenv import load_dotenv

from connectors.jira_connector import JiraConnector

# Load environment variables
load_dotenv()

# Check if integration tests should run
JIRA_URL = os.getenv('JIRA_URL')
JIRA_EMAIL = os.getenv('JIRA_EMAIL')
JIRA_API_TOKEN = os.getenv('JIRA_API_TOKEN')
JIRA_TEST_PROJECT = os.getenv('JIRA_TEST_PROJECT', 'TEST')
JIRA_TEST_BOARD_ID = os.getenv('JIRA_TEST_BOARD_ID', '1')

# Skip all tests if credentials not available
pytestmark = pytest.mark.skipif(
    not all([JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN]),
    reason="Jira credentials not configured in environment"
)


class TestJiraIntegration:
    """Integration tests for JiraConnector."""
    
    @pytest.fixture
    def jira_config(self):
        """Configuration for API token authentication."""
        return {
            'jira_url': JIRA_URL,
            'auth_method': 'api_token',
            'email': JIRA_EMAIL,
            'api_token': JIRA_API_TOKEN,
            'default_project': JIRA_TEST_PROJECT,
            'rate_limit': 5  # Lower rate limit for testing
        }
    
    @pytest.fixture
    def jira_connector(self, jira_config):
        """Create and connect JiraConnector."""
        connector = JiraConnector(jira_config)
        if not connector.connect():
            pytest.skip("Failed to connect to Jira")
        yield connector
        connector.disconnect()
    
    def test_connection(self, jira_config):
        """Test basic connection to Jira."""
        connector = JiraConnector(jira_config)
        
        assert connector.connect() is True
        assert connector.is_connected is True
        assert connector.validate_connection() is True
        
        assert connector.disconnect() is True
        assert connector.is_connected is False
    
    def test_get_metadata(self, jira_connector):
        """Test retrieving Jira metadata."""
        metadata = jira_connector.get_metadata()
        
        assert 'version' in metadata
        assert 'projects' in metadata
        assert 'issue_types' in metadata
        assert 'statuses' in metadata
        assert 'custom_fields' in metadata
        assert 'priorities' in metadata
        
        # Verify we have at least some data
        assert len(metadata['projects']) > 0
        assert len(metadata['issue_types']) > 0
        assert len(metadata['statuses']) > 0
        
        # Verify project structure
        if metadata['projects']:
            project = metadata['projects'][0]
            assert 'key' in project
            assert 'name' in project
            assert 'id' in project
    
    def test_extract_data_simple(self, jira_connector):
        """Test simple data extraction."""
        query = {
            'project': JIRA_TEST_PROJECT,
            'max_results': 10
        }
        
        df = jira_connector.extract_data(query)
        
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert 'key' in df.columns
            assert 'summary' in df.columns
            assert 'created' in df.columns
            
            # Verify date parsing
            assert pd.api.types.is_datetime64_any_dtype(df['created'])
    
    def test_extract_data_with_jql(self, jira_connector):
        """Test data extraction with JQL query."""
        # Get issues created in the last 30 days
        query = {
            'jql': f'project = {JIRA_TEST_PROJECT} AND created >= -30d',
            'fields': ['summary', 'status', 'assignee', 'created', 'updated', 'priority'],
            'max_results': 20
        }
        
        df = jira_connector.extract_data(query)
        
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            # Check requested fields are present
            assert 'summary' in df.columns
            assert 'priority' in df.columns or 'priority_name' in df.columns
    
    def test_create_and_update_issue(self, jira_connector):
        """Test creating and updating an issue."""
        # Create issue
        issue_data = {
            'project': JIRA_TEST_PROJECT,
            'issuetype': 'Task',  # Use a common issue type
            'summary': f'Integration test issue - {datetime.now().isoformat()}',
            'description': 'This is an automated test issue created by integration tests.',
            'labels': ['integration-test', 'automated']
        }
        
        created_issue = jira_connector.create_issue(issue_data)
        
        assert created_issue is not None
        assert 'key' in created_issue
        assert 'id' in created_issue
        
        issue_key = created_issue['key']
        
        # Update issue
        update_data = {
            'summary': f'Updated integration test issue - {datetime.now().isoformat()}',
            'labels': ['integration-test', 'automated', 'updated']
        }
        
        update_result = jira_connector.update_issue(issue_key, update_data)
        assert update_result is True
        
        # Verify update by fetching the issue
        verify_query = {
            'jql': f'key = {issue_key}',
            'fields': ['summary', 'labels']
        }
        
        df = jira_connector.extract_data(verify_query)
        assert len(df) == 1
        assert 'Updated integration test issue' in df.iloc[0]['summary']
    
    def test_bulk_create_issues(self, jira_connector):
        """Test bulk issue creation."""
        issues = [
            {
                'project': JIRA_TEST_PROJECT,
                'issuetype': 'Task',
                'summary': f'Bulk test issue 1 - {datetime.now().isoformat()}'
            },
            {
                'project': JIRA_TEST_PROJECT,
                'issuetype': 'Task',
                'summary': f'Bulk test issue 2 - {datetime.now().isoformat()}'
            },
            {
                'project': JIRA_TEST_PROJECT,
                'issuetype': 'Task',
                'summary': f'Bulk test issue 3 - {datetime.now().isoformat()}'
            }
        ]
        
        created_issues = jira_connector.bulk_create_issues(issues)
        
        assert len(created_issues) == 3
        for issue in created_issues:
            assert 'key' in issue
            assert issue['key'].startswith(JIRA_TEST_PROJECT)
    
    @pytest.mark.skipif(
        not JIRA_TEST_BOARD_ID,
        reason="No test board ID configured"
    )
    def test_get_sprint_data(self, jira_connector):
        """Test retrieving sprint data."""
        board_id = int(JIRA_TEST_BOARD_ID)
        
        df = jira_connector.get_sprint_data(board_id)
        
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            # Check expected columns
            expected_columns = [
                'id', 'name', 'state', 'startDate', 'endDate',
                'total_issues', 'completed_issues', 'completion_rate'
            ]
            for col in expected_columns:
                assert col in df.columns
            
            # Verify date columns are parsed
            if 'startDate' in df.columns:
                assert pd.api.types.is_datetime64_any_dtype(df['startDate'])
    
    def test_jql_builder_integration(self, jira_connector):
        """Test JQL builder with real queries."""
        # Build complex query
        jql = jira_connector.build_jql_query(
            project=JIRA_TEST_PROJECT,
            created_after=datetime.now() - timedelta(days=30),
            status=['Open', 'In Progress', 'To Do'],
            order_by='created DESC'
        )
        
        # Use the JQL to fetch data
        df = jira_connector.extract_data({
            'jql': jql,
            'max_results': 10
        })
        
        assert isinstance(df, pd.DataFrame)
        # If we have results, they should all be from our test project
        if not df.empty:
            assert all(key.startswith(JIRA_TEST_PROJECT) for key in df['key'])
    
    def test_get_issue_dependencies(self, jira_connector):
        """Test retrieving issue dependencies."""
        # First, get an issue that might have dependencies
        query = {
            'project': JIRA_TEST_PROJECT,
            'max_results': 10
        }
        
        df = jira_connector.extract_data(query)
        
        if not df.empty:
            # Test with the first issue
            issue_key = df.iloc[0]['key']
            dependencies = jira_connector.get_issue_dependencies(issue_key)
            
            assert isinstance(dependencies, dict)
            assert 'blocks' in dependencies
            assert 'blocked_by' in dependencies
            assert isinstance(dependencies['blocks'], list)
            assert isinstance(dependencies['blocked_by'], list)
    
    @pytest.mark.skipif(
        not JIRA_TEST_BOARD_ID,
        reason="No test board ID configured"
    )
    def test_get_sprint_progress(self, jira_connector):
        """Test sprint progress calculation."""
        board_id = int(JIRA_TEST_BOARD_ID)
        
        progress = jira_connector.get_sprint_progress(board_id)
        
        if progress:  # Only test if we have an active sprint
            assert 'sprint_id' in progress
            assert 'sprint_name' in progress
            assert 'total_issues' in progress
            assert 'completed_issues' in progress
            assert 'completion_rate' in progress
            assert 'status_breakdown' in progress
            assert 'issue_type_breakdown' in progress
            
            # Verify data types
            assert isinstance(progress['total_issues'], int)
            assert isinstance(progress['completion_rate'], (int, float))
            assert isinstance(progress['status_breakdown'], dict)
    
    @pytest.mark.skipif(
        not JIRA_TEST_BOARD_ID,
        reason="No test board ID configured"
    )
    def test_get_team_velocity(self, jira_connector):
        """Test team velocity calculation."""
        board_id = int(JIRA_TEST_BOARD_ID)
        
        df = jira_connector.get_team_velocity(board_id, num_sprints=3)
        
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            # Check expected columns
            expected_columns = [
                'sprint_id', 'sprint_name', 'completed_story_points',
                'total_story_points', 'completed_issues', 'total_issues'
            ]
            for col in expected_columns:
                assert col in df.columns
            
            # Check rolling averages were calculated
            if len(df) > 1:
                assert 'avg_velocity_3' in df.columns
                assert 'avg_velocity_5' in df.columns
    
    def test_rate_limiting(self, jira_connector):
        """Test that rate limiting is working."""
        import time
        
        start_time = time.time()
        
        # Make multiple rapid requests
        for i in range(5):
            query = {
                'jql': f'key = {JIRA_TEST_PROJECT}-99999',  # Non-existent issue
                'max_results': 1
            }
            jira_connector.extract_data(query)
        
        elapsed = time.time() - start_time
        
        # With rate limit of 5/sec, 5 requests should take at least 0.8 seconds
        assert elapsed >= 0.8
    
    def test_error_handling_invalid_project(self, jira_connector):
        """Test error handling with invalid project."""
        query = {
            'project': 'INVALID_PROJECT_KEY_12345',
            'max_results': 10
        }
        
        df = jira_connector.extract_data(query)
        
        # Should return empty DataFrame on error
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert jira_connector.last_error is not None
    
    def test_custom_field_handling(self, jira_connector):
        """Test handling of custom fields."""
        # Get metadata to find custom fields
        metadata = jira_connector.get_metadata()
        custom_fields = metadata.get('custom_fields', [])
        
        if custom_fields:
            # Get all field IDs
            field_ids = [f['id'] for f in custom_fields[:5]]  # Limit to 5 fields
            
            query = {
                'project': JIRA_TEST_PROJECT,
                'fields': ['summary'] + field_ids,
                'max_results': 5
            }
            
            df = jira_connector.extract_data(query)
            
            assert isinstance(df, pd.DataFrame)
            # Custom fields should be included if they have values
    
    @pytest.mark.slow
    def test_pagination(self, jira_connector):
        """Test pagination for large result sets."""
        # This test might be slow depending on the number of issues
        query = {
            'project': JIRA_TEST_PROJECT,
            'max_results': 150  # Force pagination (Jira limits to 100 per request)
        }
        
        df = jira_connector.extract_data(query)
        
        assert isinstance(df, pd.DataFrame)
        # If project has more than 100 issues, verify pagination worked
        if len(df) > 100:
            assert len(df) <= 150
    
    def test_concurrent_requests(self, jira_connector):
        """Test handling concurrent requests."""
        import concurrent.futures
        
        def fetch_issue(issue_num):
            query = {
                'jql': f'key = {JIRA_TEST_PROJECT}-{issue_num}',
                'max_results': 1
            }
            return jira_connector.extract_data(query)
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_issue, i) for i in range(1, 11)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should return DataFrames
        assert all(isinstance(r, pd.DataFrame) for r in results)
    
    def test_connection_resilience(self, jira_config):
        """Test connection resilience and reconnection."""
        connector = JiraConnector(jira_config)
        
        # Connect
        assert connector.connect() is True
        
        # Disconnect
        assert connector.disconnect() is True
        
        # Try to use after disconnect (should fail gracefully)
        df = connector.extract_data({'project': JIRA_TEST_PROJECT})
        assert df.empty
        
        # Reconnect
        assert connector.connect() is True
        
        # Should work again
        df = connector.extract_data({'project': JIRA_TEST_PROJECT, 'max_results': 1})
        assert isinstance(df, pd.DataFrame)