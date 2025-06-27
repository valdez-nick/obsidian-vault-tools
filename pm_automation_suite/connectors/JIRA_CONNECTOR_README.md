# JiraConnector Documentation

The JiraConnector provides comprehensive integration with Atlassian Jira, supporting both API token and OAuth 2.0 authentication methods. It includes features for issue management, sprint tracking, bulk operations, and common PM workflows.

## Features

### Core Functionality
- **Dual Authentication**: Support for both API token and OAuth 2.0
- **Issue Management**: Create, update, and retrieve issues
- **Bulk Operations**: Create multiple issues in a single request
- **Sprint Management**: Access sprint data, velocity, and burndown metrics
- **JQL Query Builder**: Programmatically build complex JQL queries
- **Dependency Tracking**: Track issue dependencies (blocks/blocked by)
- **Custom Field Support**: Handle custom fields with automatic mapping
- **Webhook Support**: Register webhooks for real-time updates (OAuth only)

### Advanced Features
- **Rate Limiting**: Built-in rate limiter to respect API limits
- **Retry Logic**: Automatic retry with exponential backoff
- **Caching**: Metadata caching to reduce API calls
- **Error Handling**: Graceful error handling with detailed logging
- **Pagination**: Automatic handling of large result sets
- **Data Flattening**: Convert nested Jira responses to flat DataFrames

## Installation

```bash
pip install atlassian-python-api pandas aiohttp tenacity
```

## Configuration

### API Token Authentication

```python
config = {
    'jira_url': 'https://your-domain.atlassian.net',
    'auth_method': 'api_token',
    'email': 'your-email@example.com',
    'api_token': 'your-api-token',
    'default_project': 'PROJ',  # Optional
    'rate_limit': 10  # Requests per second
}
```

### OAuth 2.0 Authentication

```python
config = {
    'jira_url': 'https://your-domain.atlassian.net',
    'auth_method': 'oauth2',
    'client_id': 'your-client-id',
    'client_secret': 'your-client-secret',
    'tenant_id': 'your-tenant-id',
    'default_project': 'PROJ',  # Optional
    'rate_limit': 10  # Requests per second
}
```

## Quick Start

```python
from connectors.jira_connector import JiraConnector

# Create connector
jira = JiraConnector(config)

# Connect to Jira
if jira.connect():
    # Extract data
    df = jira.extract_data({
        'project': 'PROJ',
        'status': ['Open', 'In Progress'],
        'max_results': 100
    })
    
    print(f"Found {len(df)} issues")
    
    # Disconnect when done
    jira.disconnect()
```

## Usage Examples

### 1. JQL Query Building

```python
# Build complex queries programmatically
jql = jira.build_jql_query(
    project=['PROJ1', 'PROJ2'],
    status=['Open', 'In Progress'],
    assignee='currentUser()',
    created_after=datetime.now() - timedelta(days=30),
    labels=['backend', 'bug'],
    custom_fields={'Story Points': [1, 2, 3]},
    order_by='priority DESC, created DESC'
)

# Use the generated JQL
df = jira.extract_data({'jql': jql})
```

### 2. Issue Creation and Updates

```python
# Create a single issue
issue_data = {
    'project': 'PROJ',
    'issuetype': 'Story',
    'summary': 'Implement new feature',
    'description': 'Detailed description here',
    'priority': 'High',
    'labels': ['feature', 'backend'],
    'customfield_10001': 5  # Story points
}

created = jira.create_issue(issue_data)
print(f"Created issue: {created['key']}")

# Update the issue
update_data = {
    'status': 'In Progress',
    'assignee': 'user-account-id'
}

jira.update_issue(created['key'], update_data)
```

### 3. Bulk Operations

```python
# Create multiple issues at once
issues = [
    {
        'project': 'PROJ',
        'issuetype': 'Task',
        'summary': f'Task {i}: Implement component {i}'
    }
    for i in range(1, 11)
]

created_issues = jira.bulk_create_issues(issues)
print(f"Created {len(created_issues)} issues")
```

### 4. Sprint Management

```python
# Get sprint data
board_id = 1
sprint_df = jira.get_sprint_data(board_id)

# Get detailed sprint progress
progress = jira.get_sprint_progress(board_id)
print(f"Sprint: {progress['sprint_name']}")
print(f"Completion: {progress['completion_rate']:.1%}")
print(f"Burn rate: {progress['burn_rate']:.2f}")

# Calculate team velocity
velocity_df = jira.get_team_velocity(board_id, num_sprints=5)
avg_velocity = velocity_df['completed_story_points'].mean()
print(f"Average velocity: {avg_velocity:.1f} story points")
```

### 5. Dependency Tracking

```python
# Get issue dependencies
deps = jira.get_issue_dependencies('PROJ-123')

print(f"This issue blocks: {len(deps['blocks'])} issues")
for blocked in deps['blocks']:
    print(f"  - {blocked['key']}: {blocked['summary']}")

print(f"This issue is blocked by: {len(deps['blocked_by'])} issues")
for blocker in deps['blocked_by']:
    print(f"  - {blocker['key']}: {blocker['summary']}")
```

### 6. Advanced Data Extraction

```python
# Extract data with additional information
df = jira.extract_data({
    'jql': 'project = PROJ AND updated >= -7d',
    'fields': ['summary', 'status', 'assignee', 'created', 'updated', 
               'priority', 'labels', 'customfield_10001'],
    'expand': ['changelog', 'renderedFields'],
    'include_changelog': True,
    'include_comments': True,
    'max_results': 500
})

# The DataFrame will include flattened data with columns like:
# - key, id, summary, created, updated
# - status_name, assignee_name, assignee_email
# - priority_name, labels (as list)
# - changelog, comments (if requested)
```

### 7. Metadata and Discovery

```python
# Get Jira instance metadata
metadata = jira.get_metadata()

print(f"Jira Version: {metadata['version']}")

# List all projects
for project in metadata['projects']:
    print(f"  {project['key']}: {project['name']}")

# List custom fields
for field in metadata['custom_fields']:
    print(f"  {field['id']}: {field['name']} ({field['type']})")
```

### 8. Webhook Support (OAuth only)

```python
# Register a webhook
webhook = jira.register_webhook(
    name="Issue Update Webhook",
    url="https://your-server.com/webhook/jira",
    events=["jira:issue_created", "jira:issue_updated"],
    filters={'project': 'PROJ'}
)

# Register callback handlers
def handle_issue_created(event_data):
    print(f"New issue created: {event_data['issue']['key']}")

jira.handle_webhook_callback("jira:issue_created", handle_issue_created)
```

## Data Extraction Query Parameters

The `extract_data` method accepts the following parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `jql` | str | Direct JQL query string |
| `project` | str/list | Project key(s) |
| `issue_type` | str/list | Issue type(s) |
| `status` | str/list | Status(es) |
| `assignee` | str | Assignee (use 'currentUser()' for current user) |
| `reporter` | str | Reporter |
| `sprint` | int/str | Sprint ID or name |
| `labels` | list | List of labels |
| `components` | list | List of components |
| `created_after` | datetime | Issues created after this date |
| `updated_after` | datetime | Issues updated after this date |
| `custom_fields` | dict | Custom field filters |
| `fields` | list | Fields to retrieve |
| `expand` | list | Fields to expand |
| `include_changelog` | bool | Include issue changelog |
| `include_comments` | bool | Include issue comments |
| `max_results` | int | Maximum number of results |
| `order_by` | str | Order by clause |

## Error Handling

The connector provides comprehensive error handling:

```python
# Connection errors
if not jira.connect():
    print(f"Connection failed: {jira.last_error}")

# Data extraction errors
df = jira.extract_data({'project': 'INVALID'})
if df.empty:
    print(f"Query failed: {jira.last_error}")

# The connector will retry failed requests automatically
# with exponential backoff (configurable)
```

## Performance Considerations

### Rate Limiting
- Default: 10 requests per second
- Configurable via `rate_limit` parameter
- Automatic handling of 429 (rate limit) responses

### Caching
- Metadata cached for 30 minutes
- Reduces repeated API calls for field definitions

### Pagination
- Automatic handling of large result sets
- Fetches 100 issues per request
- Respects `max_results` parameter

### Best Practices
1. Use specific JQL queries to reduce data transfer
2. Request only needed fields via `fields` parameter
3. Use bulk operations for multiple creates/updates
4. Cache sprint and velocity data when possible
5. Implement webhook handlers for real-time updates

## Testing

### Unit Tests

```bash
pytest tests/unit/test_jira_connector.py -v
```

### Integration Tests

Set environment variables:
```bash
export JIRA_URL="https://your-domain.atlassian.net"
export JIRA_EMAIL="your-email@example.com"
export JIRA_API_TOKEN="your-api-token"
export JIRA_TEST_PROJECT="TEST"
export JIRA_TEST_BOARD_ID="1"
```

Run integration tests:
```bash
pytest tests/integration/test_jira_integration.py -v
```

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify API token is valid and not expired
   - Check email/token combination
   - For OAuth, ensure client credentials are correct

2. **Rate Limiting**
   - Reduce `rate_limit` parameter
   - Implement caching for frequently accessed data
   - Use webhooks for real-time updates

3. **Custom Fields**
   - Use `get_metadata()` to discover field IDs
   - Reference custom fields as `customfield_XXXXX`
   - Check field permissions

4. **Large Result Sets**
   - Use more specific JQL queries
   - Implement date-based filtering
   - Process data in batches

### Debug Logging

Enable debug logging:
```python
import logging
logging.getLogger('connectors.jira_connector').setLevel(logging.DEBUG)
```

## Security Considerations

1. **Credential Storage**
   - Never hardcode credentials
   - Use environment variables or secure vaults
   - OAuth tokens are managed by AuthenticationManager

2. **API Token Permissions**
   - Use tokens with minimal required permissions
   - Regularly rotate tokens
   - Monitor token usage

3. **Data Privacy**
   - Be mindful of sensitive data in issues
   - Implement field-level filtering if needed
   - Follow your organization's data policies

## Limitations

1. **Webhook Support**: Only available with OAuth authentication
2. **Burndown Data**: Limited by Jira's API (not fully exposed)
3. **Bulk Updates**: Not implemented (only bulk create)
4. **Attachment Handling**: Not currently implemented
5. **Agile Boards**: Limited to basic sprint operations

## Future Enhancements

- [ ] Bulk update operations
- [ ] Attachment upload/download
- [ ] Advanced Agile board operations
- [ ] Real-time collaboration features
- [ ] Enhanced caching strategies
- [ ] GraphQL API support
- [ ] Async/await throughout
- [ ] Streaming for large datasets