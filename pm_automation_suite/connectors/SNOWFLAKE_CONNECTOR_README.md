# Snowflake Connector

A comprehensive Snowflake data warehouse connector with advanced features for product management analytics, query optimization, and MCP server integration.

## Features

### Core Functionality
- **Multiple Authentication Methods**
  - Password authentication
  - OAuth 2.0 authentication
  - Key-pair authentication with RSA keys

- **Connection Management**
  - Thread-safe connection pooling
  - Automatic connection validation and recovery
  - Configurable pool sizes for optimal performance

- **Query Execution**
  - Safe parameterized queries to prevent SQL injection
  - Batch query execution with parallel processing
  - Query result caching with intelligent TTL
  - Performance monitoring and metrics

### PM-Specific Analytics
Built-in query templates for common product management metrics:

- **DAU/MAU Analysis**: Daily and Monthly Active Users with ratios
- **Retention Cohorts**: User retention analysis by cohort period
- **Feature Adoption**: Feature usage and adoption rates
- **Revenue Analysis**: Revenue trends, ARPU, and growth metrics
- **Conversion Funnels**: Multi-step user journey analysis

### Advanced Features
- **Query Optimization**: Automated suggestions for query performance
- **Warehouse Management**: Programmatic warehouse control (resume, suspend, resize)
- **Auto-scaling**: Intelligent warehouse sizing recommendations
- **Incremental Loading**: Efficient data synchronization
- **MCP Server Integration**: Claude AI integration via Model Context Protocol

## Installation

```bash
# Install with Snowflake support
pip install snowflake-connector-python[pandas]
pip install cryptography  # For key-pair authentication
```

## Configuration

### Basic Configuration
```python
from connectors.snowflake_connector import SnowflakeConnector

config = {
    'account': 'your_account',
    'user': 'your_user',
    'password': 'your_password',
    'warehouse': 'COMPUTE_WH',
    'database': 'ANALYTICS',
    'schema': 'PUBLIC',
    'role': 'ANALYST_ROLE',  # Optional
    'pool_size': 5,          # Connection pool size
    'cache_ttl': 3600        # Cache TTL in seconds
}

connector = SnowflakeConnector(config)
```

### OAuth Authentication
```python
config = {
    'account': 'your_account',
    'user': 'your_user',
    'auth_type': 'oauth',
    'token': 'your_oauth_token',
    'warehouse': 'COMPUTE_WH',
    'database': 'ANALYTICS',
    'schema': 'PUBLIC'
}
```

### Key-Pair Authentication
```python
config = {
    'account': 'your_account',
    'user': 'your_user',
    'auth_type': 'key_pair',
    'private_key_path': '/path/to/rsa_key.p8',
    'private_key_passphrase': 'optional_passphrase',
    'warehouse': 'COMPUTE_WH',
    'database': 'ANALYTICS',
    'schema': 'PUBLIC'
}
```

## Usage Examples

### Basic Query Execution
```python
# Connect to Snowflake
if connector.connect():
    # Execute a simple query
    df = connector.execute_query("""
        SELECT COUNT(*) as user_count, 
               DATE_TRUNC('day', created_at) as date
        FROM users
        WHERE created_at >= CURRENT_DATE - 7
        GROUP BY date
        ORDER BY date DESC
    """)
    
    print(f"Found {len(df)} days of data")
    
    # Parameterized query for safety
    result = connector.execute_query(
        "SELECT * FROM orders WHERE user_id = %(user_id)s",
        params={'user_id': 12345}
    )
    
    connector.disconnect()
```

### Batch Query Execution
```python
queries = [
    "SELECT COUNT(*) FROM users",
    "SELECT COUNT(*) FROM orders",
    "SELECT SUM(amount) FROM transactions"
]

# Execute queries in parallel
results = connector.execute_batch(queries, parallel=True, max_workers=3)

for i, df in enumerate(results):
    print(f"Query {i+1}: {df.iloc[0, 0]}")
```

### PM Analytics Examples

#### DAU/MAU Analysis
```python
dau_mau = connector.get_dau_mau(
    table='user_events',
    user_id_col='user_id',
    timestamp_col='event_time',
    days_back=30
)

# Latest metrics
latest = dau_mau.iloc[0]
print(f"DAU: {latest['dau']:,}")
print(f"MAU: {latest['mau']:,}")
print(f"DAU/MAU: {latest['dau_mau_ratio']:.1f}%")
```

#### User Retention
```python
retention = connector.get_retention_cohort(
    table='user_activity',
    user_id_col='user_id',
    timestamp_col='activity_date',
    cohort_period='week',
    max_periods=12
)

# Create retention heatmap
pivot = retention.pivot(
    index='cohort_date',
    columns='periods_later',
    values='retention_rate'
)
```

#### Feature Adoption
```python
adoption = connector.get_feature_adoption(
    table='product_events',
    user_id_col='user_id',
    feature_col='feature_name',
    timestamp_col='event_time',
    days_back=30
)

# Top adopted features
for _, row in adoption.head(10).iterrows():
    print(f"{row['feature_name']}: {row['adoption_rate']:.1f}% "
          f"({row['unique_users']:,} users)")
```

#### Revenue Analysis
```python
revenue = connector.get_revenue_analysis(
    table='payments',
    user_id_col='user_id',
    transaction_id_col='payment_id',
    amount_col='amount',
    timestamp_col='payment_date',
    segment_col='user_segment',
    period='month',
    periods_back=12
)

# Revenue trends
for _, row in revenue.iterrows():
    growth = row.get('revenue_growth_pct', 0)
    print(f"{row['period']}: ${row['total_revenue']:,.0f} "
          f"({growth:+.1f}% growth)")
```

#### Conversion Funnel
```python
funnel = connector.get_user_funnel(
    table='user_events',
    user_id_col='user_id',
    step_col='event_name',
    timestamp_col='event_time',
    steps=['landing', 'signup', 'activation', 'purchase', 'retention'],
    days_back=30
)

# Funnel visualization
for _, step in funnel.iterrows():
    bars = '█' * int(step['pct_of_total'] / 5)
    print(f"{step['funnel_step']:<20} {bars} {step['pct_of_total']:.1f}%")
```

### Query Optimization
```python
# Get optimization suggestions
bad_query = """
SELECT * FROM large_table 
ORDER BY created_at DESC
"""

optimization = connector.optimize_query(bad_query)

print(f"Optimization Score: {optimization['optimization_score']}/100")
print("Suggestions:")
for suggestion in optimization['suggestions']:
    print(f"- {suggestion['suggestion']}")

print(f"\nOptimized Query:\n{optimization['optimized_query']}")
```

### Warehouse Management
```python
# Check warehouse status
metadata = connector.get_metadata()
warehouses = metadata['warehouses']

# Suspend unused warehouse
connector.manage_warehouse('ETL_WH', 'suspend')

# Resume and resize based on workload
queries_per_minute = 150
recommended_size = connector.auto_scale_warehouse(
    queries_per_minute, 
    current_size='SMALL'
)

if recommended_size:
    connector.manage_warehouse('COMPUTE_WH', 'resize', size=recommended_size)
```

### Performance Monitoring
```python
# Get performance statistics
perf_stats = connector.get_performance_stats()

print(f"Total Queries: {perf_stats['total_queries']}")
print(f"Avg Execution: {perf_stats['avg_execution_time']:.2f}s")
print(f"Success Rate: {perf_stats['success_rate']:.1f}%")

# Check for slow queries
alerts = connector._performance_monitor.get_alerts(threshold_seconds=30)
for alert in alerts:
    print(f"[{alert['severity']}] {alert['sql_preview']} "
          f"({alert['execution_time']:.1f}s)")

# Cache statistics
cache_stats = connector.get_cache_stats()
print(f"Cache Hit Rate: {cache_stats.get('hit_rate', 0):.1f}%")
```

### Incremental Data Loading
```python
# Load new data incrementally
rows_loaded = connector.incremental_load(
    source_query="""
        SELECT user_id, event_name, event_time, properties
        FROM raw_events
        WHERE event_time >= DATEADD('hour', -1, CURRENT_TIMESTAMP())
    """,
    target_table='processed_events',
    timestamp_col='event_time',
    unique_key='event_id'
)

print(f"Loaded {rows_loaded:,} new rows")
```

## MCP Server Integration

The Snowflake connector includes an MCP server wrapper for Claude integration:

### Running the MCP Server
```bash
# Set environment variables
export SNOWFLAKE_ACCOUNT=your_account
export SNOWFLAKE_USER=your_user
export SNOWFLAKE_PASSWORD=your_password
export SNOWFLAKE_WAREHOUSE=COMPUTE_WH
export SNOWFLAKE_DATABASE=ANALYTICS

# Run the server
python -m connectors.snowflake_mcp_server
```

### Available MCP Tools
- `snowflake_query`: Execute SQL queries
- `snowflake_dau_mau`: Get DAU/MAU metrics
- `snowflake_retention`: Analyze user retention
- `snowflake_feature_adoption`: Track feature usage
- `snowflake_revenue`: Analyze revenue metrics
- `snowflake_funnel`: User conversion analysis
- `snowflake_optimize`: Query optimization
- `snowflake_warehouse`: Warehouse management
- `snowflake_metadata`: Account metadata
- `snowflake_performance`: Performance statistics

## Performance Best Practices

### 1. Connection Pooling
```python
# Configure appropriate pool size
config['pool_size'] = 10  # For high-concurrency applications
```

### 2. Query Caching
```python
# Use caching for repeated queries
df = connector.execute_query(sql, use_cache=True)

# Clear cache when data changes
connector.clear_cache(pattern='user_metrics')
```

### 3. Batch Processing
```python
# Process multiple queries efficiently
queries = generate_queries()
results = connector.execute_batch(queries, parallel=True, max_workers=5)
```

### 4. Warehouse Sizing
```python
# Monitor and adjust warehouse size
current_qpm = get_current_queries_per_minute()
recommended = connector.auto_scale_warehouse(current_qpm, 'MEDIUM')
```

### 5. Query Optimization
```python
# Always optimize before production
optimized = connector.optimize_query(your_query)
if optimized['optimization_score'] < 70:
    # Review and improve query
    pass
```

## Error Handling

```python
try:
    result = connector.execute_query(sql)
except ConnectionError as e:
    # Handle connection issues
    logger.error(f"Connection error: {e}")
    connector.connect()  # Reconnect
except ProgrammingError as e:
    # Handle SQL errors
    logger.error(f"SQL error: {e}")
except Exception as e:
    # Handle other errors
    logger.error(f"Unexpected error: {e}")
```

## Security Considerations

1. **Use parameterized queries** to prevent SQL injection
2. **Store credentials securely** using environment variables or secret managers
3. **Use key-pair authentication** for production environments
4. **Implement role-based access** with appropriate Snowflake roles
5. **Monitor query patterns** for unusual activity

## Troubleshooting

### Connection Issues
```python
# Enable debug logging
import logging
logging.getLogger('snowflake.connector').setLevel(logging.DEBUG)

# Test connection
if not connector.validate_connection():
    print(f"Connection invalid: {connector.last_error}")
```

### Performance Issues
```python
# Check cache effectiveness
cache_stats = connector.get_cache_stats()
if cache_stats['hit_rate'] < 50:
    # Review caching strategy
    pass

# Monitor slow queries
perf_stats = connector.get_performance_stats()
for query in perf_stats['slowest_queries']:
    # Optimize slow queries
    pass
```

### Memory Issues
```python
# Clear cache periodically
connector.clear_cache()

# Use iterators for large results
# (Future enhancement)
```

## Contributing

When adding new features:
1. Follow the existing patterns for PM query templates
2. Add comprehensive unit tests
3. Update documentation with examples
4. Ensure thread safety for concurrent usage
5. Add performance monitoring for new query types

## License

This connector is part of the PM Automation Suite and follows the same license terms.