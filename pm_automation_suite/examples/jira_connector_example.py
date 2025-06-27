"""
Example usage of JiraConnector

This example demonstrates various features of the JiraConnector including:
- Basic connection and authentication
- Data extraction with JQL
- Issue creation and updates
- Sprint management
- Bulk operations
- Common PM workflows
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectors.jira_connector import JiraConnector
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def example_api_token_auth():
    """Example using API token authentication."""
    print("\n=== API Token Authentication Example ===")
    
    config = {
        'jira_url': os.getenv('JIRA_URL', 'https://example.atlassian.net'),
        'auth_method': 'api_token',
        'email': os.getenv('JIRA_EMAIL', 'your-email@example.com'),
        'api_token': os.getenv('JIRA_API_TOKEN', 'your-api-token'),
        'default_project': 'PROJ',
        'rate_limit': 10
    }
    
    # Create and connect
    jira = JiraConnector(config)
    
    if jira.connect():
        print("✓ Connected to Jira successfully")
        
        # Get metadata
        metadata = jira.get_metadata()
        print(f"\nJira Version: {metadata['version']}")
        print(f"Number of projects: {len(metadata['projects'])}")
        print(f"Number of custom fields: {len(metadata['custom_fields'])}")
        
        # Disconnect
        jira.disconnect()
    else:
        print("✗ Failed to connect to Jira")


def example_oauth2_auth():
    """Example using OAuth 2.0 authentication."""
    print("\n=== OAuth 2.0 Authentication Example ===")
    
    config = {
        'jira_url': 'https://example.atlassian.net',
        'auth_method': 'oauth2',
        'client_id': os.getenv('JIRA_CLIENT_ID', 'your-client-id'),
        'client_secret': os.getenv('JIRA_CLIENT_SECRET', 'your-client-secret'),
        'tenant_id': os.getenv('JIRA_TENANT_ID', 'your-tenant-id'),
        'default_project': 'PROJ',
        'rate_limit': 10
    }
    
    jira = JiraConnector(config)
    
    if jira.connect():
        print("✓ Connected to Jira with OAuth 2.0")
        jira.disconnect()
    else:
        print("✗ Failed to connect with OAuth 2.0")


def example_data_extraction():
    """Example of extracting data from Jira."""
    print("\n=== Data Extraction Example ===")
    
    config = {
        'jira_url': os.getenv('JIRA_URL', 'https://example.atlassian.net'),
        'auth_method': 'api_token',
        'email': os.getenv('JIRA_EMAIL'),
        'api_token': os.getenv('JIRA_API_TOKEN'),
    }
    
    jira = JiraConnector(config)
    
    if not jira.connect():
        print("Failed to connect")
        return
    
    # Example 1: Simple project query
    print("\n1. Getting all issues from a project:")
    df = jira.extract_data({
        'project': 'PROJ',
        'max_results': 10
    })
    print(f"Found {len(df)} issues")
    if not df.empty:
        print(df[['key', 'summary']].head())
    
    # Example 2: Using JQL builder
    print("\n2. Using JQL builder for complex query:")
    jql = jira.build_jql_query(
        project=['PROJ', 'PROJ2'],
        status=['Open', 'In Progress'],
        created_after=datetime.now() - timedelta(days=30),
        assignee='currentUser()',
        order_by='priority DESC, created DESC'
    )
    print(f"Generated JQL: {jql}")
    
    df = jira.extract_data({'jql': jql, 'max_results': 20})
    print(f"Found {len(df)} issues matching criteria")
    
    # Example 3: Include additional data
    print("\n3. Getting issues with changelog and comments:")
    df = jira.extract_data({
        'project': 'PROJ',
        'max_results': 5,
        'include_changelog': True,
        'include_comments': True
    })
    
    jira.disconnect()


def example_issue_management():
    """Example of creating and managing issues."""
    print("\n=== Issue Management Example ===")
    
    config = {
        'jira_url': os.getenv('JIRA_URL'),
        'auth_method': 'api_token',
        'email': os.getenv('JIRA_EMAIL'),
        'api_token': os.getenv('JIRA_API_TOKEN'),
    }
    
    jira = JiraConnector(config)
    
    if not jira.connect():
        print("Failed to connect")
        return
    
    # Create a single issue
    print("\n1. Creating a single issue:")
    issue_data = {
        'project': 'PROJ',
        'issuetype': 'Task',
        'summary': 'Example task from JiraConnector',
        'description': 'This is a test issue created via the API',
        'priority': 'Medium',
        'labels': ['api-test', 'example']
    }
    
    created = jira.create_issue(issue_data)
    if created:
        print(f"✓ Created issue: {created.get('key')}")
        issue_key = created.get('key')
        
        # Update the issue
        print("\n2. Updating the issue:")
        update_data = {
            'summary': 'Updated example task',
            'labels': ['api-test', 'example', 'updated']
        }
        
        if jira.update_issue(issue_key, update_data):
            print(f"✓ Updated issue {issue_key}")
    
    # Bulk create issues
    print("\n3. Bulk creating issues:")
    bulk_issues = [
        {
            'project': 'PROJ',
            'issuetype': 'Story',
            'summary': f'Bulk story {i}',
            'description': f'Story description {i}'
        }
        for i in range(1, 4)
    ]
    
    created_issues = jira.bulk_create_issues(bulk_issues)
    print(f"✓ Created {len(created_issues)} issues in bulk")
    
    jira.disconnect()


def example_sprint_management():
    """Example of sprint management features."""
    print("\n=== Sprint Management Example ===")
    
    config = {
        'jira_url': os.getenv('JIRA_URL'),
        'auth_method': 'api_token',
        'email': os.getenv('JIRA_EMAIL'),
        'api_token': os.getenv('JIRA_API_TOKEN'),
    }
    
    jira = JiraConnector(config)
    
    if not jira.connect():
        print("Failed to connect")
        return
    
    board_id = int(os.getenv('JIRA_BOARD_ID', '1'))
    
    # Get sprint data
    print(f"\n1. Getting sprint data for board {board_id}:")
    sprint_df = jira.get_sprint_data(board_id)
    
    if not sprint_df.empty:
        print(f"Found {len(sprint_df)} sprint(s)")
        for _, sprint in sprint_df.iterrows():
            print(f"\nSprint: {sprint['name']}")
            print(f"  State: {sprint['state']}")
            print(f"  Issues: {sprint['completed_issues']}/{sprint['total_issues']}")
            print(f"  Completion: {sprint['completion_rate']:.1%}")
    
    # Get sprint progress
    print(f"\n2. Getting detailed sprint progress:")
    progress = jira.get_sprint_progress(board_id)
    
    if progress:
        print(f"Sprint: {progress['sprint_name']}")
        print(f"Burn rate: {progress.get('burn_rate', 0):.2f}")
        print(f"Status breakdown: {progress.get('status_breakdown', {})}")
    
    # Get team velocity
    print(f"\n3. Calculating team velocity:")
    velocity_df = jira.get_team_velocity(board_id, num_sprints=5)
    
    if not velocity_df.empty:
        print(f"Analyzed {len(velocity_df)} sprints")
        avg_velocity = velocity_df['completed_story_points'].mean()
        print(f"Average velocity: {avg_velocity:.1f} story points")
    
    jira.disconnect()


def example_dependencies():
    """Example of tracking issue dependencies."""
    print("\n=== Issue Dependencies Example ===")
    
    config = {
        'jira_url': os.getenv('JIRA_URL'),
        'auth_method': 'api_token',
        'email': os.getenv('JIRA_EMAIL'),
        'api_token': os.getenv('JIRA_API_TOKEN'),
    }
    
    jira = JiraConnector(config)
    
    if not jira.connect():
        print("Failed to connect")
        return
    
    # Get a sample issue
    df = jira.extract_data({
        'project': 'PROJ',
        'max_results': 1
    })
    
    if not df.empty:
        issue_key = df.iloc[0]['key']
        print(f"\nChecking dependencies for {issue_key}:")
        
        deps = jira.get_issue_dependencies(issue_key)
        
        print(f"\nBlocks {len(deps['blocks'])} issue(s):")
        for blocked in deps['blocks']:
            print(f"  - {blocked['key']}: {blocked['summary']} ({blocked['status']})")
        
        print(f"\nBlocked by {len(deps['blocked_by'])} issue(s):")
        for blocker in deps['blocked_by']:
            print(f"  - {blocker['key']}: {blocker['summary']} ({blocker['status']})")
    
    jira.disconnect()


def example_error_handling():
    """Example of error handling and resilience."""
    print("\n=== Error Handling Example ===")
    
    config = {
        'jira_url': os.getenv('JIRA_URL'),
        'auth_method': 'api_token',
        'email': os.getenv('JIRA_EMAIL'),
        'api_token': os.getenv('JIRA_API_TOKEN'),
    }
    
    jira = JiraConnector(config)
    
    if not jira.connect():
        print("Failed to connect")
        return
    
    # Try invalid project
    print("\n1. Handling invalid project:")
    df = jira.extract_data({
        'project': 'INVALID_PROJECT_12345',
        'max_results': 10
    })
    
    if df.empty:
        print(f"✓ Gracefully handled error: {jira.last_error}")
    
    # Try invalid JQL
    print("\n2. Handling invalid JQL:")
    df = jira.extract_data({
        'jql': 'invalid jql syntax @@@@',
        'max_results': 10
    })
    
    if df.empty:
        print(f"✓ Gracefully handled error: {jira.last_error}")
    
    jira.disconnect()


def main():
    """Run all examples."""
    print("JiraConnector Examples")
    print("=" * 50)
    
    # Check if we have credentials
    if not os.getenv('JIRA_URL') or not os.getenv('JIRA_EMAIL') or not os.getenv('JIRA_API_TOKEN'):
        print("\nPlease set the following environment variables:")
        print("  JIRA_URL - Your Jira instance URL")
        print("  JIRA_EMAIL - Your Jira email")
        print("  JIRA_API_TOKEN - Your Jira API token")
        print("  JIRA_BOARD_ID - A board ID for sprint examples (optional)")
        print("\nYou can create a .env file with these values.")
        return
    
    # Run examples
    try:
        example_api_token_auth()
        # example_oauth2_auth()  # Uncomment if you have OAuth credentials
        example_data_extraction()
        example_issue_management()
        example_sprint_management()
        example_dependencies()
        example_error_handling()
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()