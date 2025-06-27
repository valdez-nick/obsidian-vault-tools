"""
Snowflake Connector Usage Examples

Demonstrates comprehensive usage of the SnowflakeConnector,
including authentication methods, query execution, caching,
PM analytics, and performance optimization.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.snowflake_connector import SnowflakeConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_basic_connection():
    """Example: Basic connection with password authentication."""
    print("\n=== Example: Basic Connection ===")
    
    config = {
        'account': 'your_account',
        'user': 'your_user',
        'password': 'your_password',
        'warehouse': 'COMPUTE_WH',
        'database': 'ANALYTICS',
        'schema': 'PUBLIC',
        'pool_size': 5,
        'cache_ttl': 3600
    }
    
    connector = SnowflakeConnector(config)
    
    # Connect to Snowflake
    if connector.connect():
        print("✓ Successfully connected to Snowflake")
        
        # Validate connection
        if connector.validate_connection():
            print("✓ Connection validated")
            
        # Get metadata
        metadata = connector.get_metadata()
        print(f"\nCurrent context:")
        print(f"- Account: {metadata['current_context']['account']}")
        print(f"- Warehouse: {metadata['current_context']['warehouse']}")
        print(f"- Database: {metadata['current_context']['database']}")
        
        # Disconnect
        connector.disconnect()
        print("\n✓ Disconnected from Snowflake")
    else:
        print(f"✗ Failed to connect: {connector.last_error}")


def example_oauth_authentication():
    """Example: OAuth 2.0 authentication."""
    print("\n=== Example: OAuth Authentication ===")
    
    config = {
        'account': 'your_account',
        'user': 'your_user',
        'auth_type': 'oauth',
        'token': 'your_oauth_token',
        'warehouse': 'COMPUTE_WH',
        'database': 'ANALYTICS',
        'schema': 'PUBLIC'
    }
    
    try:
        connector = SnowflakeConnector(config)
        print("✓ OAuth connector configured successfully")
    except Exception as e:
        print(f"✗ OAuth configuration failed: {e}")


def example_key_pair_authentication():
    """Example: Key-pair authentication."""
    print("\n=== Example: Key-Pair Authentication ===")
    
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
    
    try:
        connector = SnowflakeConnector(config)
        print("✓ Key-pair connector configured successfully")
    except Exception as e:
        print(f"✗ Key-pair configuration failed: {e}")


def example_query_execution(connector: SnowflakeConnector):
    """Example: Query execution with caching."""
    print("\n=== Example: Query Execution ===")
    
    # Simple query
    sql = """
    SELECT 
        DATE_TRUNC('day', created_at) as date,
        COUNT(*) as user_count
    FROM users
    WHERE created_at >= DATEADD('day', -7, CURRENT_DATE())
    GROUP BY 1
    ORDER BY 1 DESC
    """
    
    # First execution - hits database
    start = datetime.now()
    result1 = connector.execute_query(sql)
    time1 = (datetime.now() - start).total_seconds()
    print(f"First execution: {len(result1)} rows in {time1:.2f}s")
    
    # Second execution - hits cache
    start = datetime.now()
    result2 = connector.execute_query(sql)
    time2 = (datetime.now() - start).total_seconds()
    print(f"Cached execution: {len(result2)} rows in {time2:.2f}s")
    print(f"Cache speedup: {time1/time2:.1f}x faster")
    
    # Parameterized query
    sql_param = """
    SELECT * FROM orders
    WHERE order_date = %(date)s
    AND status = %(status)s
    LIMIT 10
    """
    
    params = {
        'date': '2024-01-15',
        'status': 'completed'
    }
    
    result3 = connector.execute_query(sql_param, params=params)
    print(f"\nParameterized query: {len(result3)} rows")


def example_batch_queries(connector: SnowflakeConnector):
    """Example: Batch query execution."""
    print("\n=== Example: Batch Queries ===")
    
    queries = [
        "SELECT COUNT(*) as total_users FROM users",
        "SELECT COUNT(*) as total_orders FROM orders",
        "SELECT SUM(amount) as total_revenue FROM transactions",
        "SELECT COUNT(DISTINCT product_id) as total_products FROM products"
    ]
    
    # Execute in parallel
    start = datetime.now()
    results = connector.execute_batch(queries, parallel=True)
    time_parallel = (datetime.now() - start).total_seconds()
    
    print(f"Parallel execution of {len(queries)} queries in {time_parallel:.2f}s")
    for i, result in enumerate(results):
        print(f"Query {i+1}: {result.iloc[0].to_dict()}")


def example_query_optimization(connector: SnowflakeConnector):
    """Example: Query optimization suggestions."""
    print("\n=== Example: Query Optimization ===")
    
    # Poorly optimized query
    bad_query = """
    SELECT * 
    FROM large_table lt
    JOIN users u ON lt.user_id = u.id OR lt.guest_id = u.id
    ORDER BY lt.created_at DESC
    """
    
    optimization = connector.optimize_query(bad_query)
    
    print(f"Original Query Optimization Score: {optimization['optimization_score']}/100")
    print(f"Estimated Improvement: {optimization['estimated_improvement']}%")
    print("\nSuggestions:")
    for suggestion in optimization['suggestions']:
        print(f"- [{suggestion['impact'].upper()}] {suggestion['suggestion']}")
    
    print(f"\nOptimized Query:")
    print(optimization['optimized_query'])


def example_warehouse_management(connector: SnowflakeConnector):
    """Example: Warehouse management."""
    print("\n=== Example: Warehouse Management ===")
    
    warehouse = "COMPUTE_WH"
    
    # Get current warehouse info
    metadata = connector.get_metadata()
    warehouses = metadata.get('warehouses', [])
    
    current_wh = next((w for w in warehouses if w.get('name') == warehouse), None)
    if current_wh:
        print(f"Current warehouse: {current_wh.get('name')}")
        print(f"- State: {current_wh.get('state')}")
        print(f"- Size: {current_wh.get('size')}")
    
    # Auto-scale recommendation
    queries_per_minute = 75
    current_size = current_wh.get('size', 'SMALL') if current_wh else 'SMALL'
    
    recommended_size = connector.auto_scale_warehouse(queries_per_minute, current_size)
    if recommended_size:
        print(f"\nRecommended resize: {current_size} -> {recommended_size}")
        print(f"Based on {queries_per_minute} queries/minute")
        
        # Resize warehouse (commented out for safety)
        # success = connector.manage_warehouse(warehouse, 'resize', size=recommended_size)
        # print(f"Resize {'succeeded' if success else 'failed'}")


def example_pm_analytics(connector: SnowflakeConnector):
    """Example: PM-specific analytics queries."""
    print("\n=== Example: PM Analytics ===")
    
    # Example table/column names - adjust for your schema
    activity_table = "user_events"
    user_id_col = "user_id"
    timestamp_col = "event_timestamp"
    
    # 1. DAU/MAU Analysis
    print("\n1. DAU/MAU Metrics:")
    try:
        dau_mau = connector.get_dau_mau(
            table=activity_table,
            user_id_col=user_id_col,
            timestamp_col=timestamp_col,
            days_back=30
        )
        if not dau_mau.empty:
            latest = dau_mau.iloc[0]
            print(f"   Latest DAU: {latest['dau']:,}")
            print(f"   Latest MAU: {latest['mau']:,}")
            print(f"   DAU/MAU Ratio: {latest['dau_mau_ratio']:.1f}%")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. Retention Analysis
    print("\n2. Weekly Retention Cohorts:")
    try:
        retention = connector.get_retention_cohort(
            table=activity_table,
            user_id_col=user_id_col,
            timestamp_col=timestamp_col,
            cohort_period='week',
            max_periods=8
        )
        if not retention.empty:
            # Show retention for most recent cohort
            recent_cohort = retention[retention['cohort_date'] == retention['cohort_date'].max()]
            print(f"   Cohort: {recent_cohort.iloc[0]['cohort_date']}")
            for _, row in recent_cohort.iterrows():
                print(f"   Week {row['periods_later']}: {row['retention_rate']:.1f}%")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. Feature Adoption
    print("\n3. Feature Adoption:")
    try:
        adoption = connector.get_feature_adoption(
            table=activity_table,
            user_id_col=user_id_col,
            feature_col="event_name",
            timestamp_col=timestamp_col,
            days_back=30
        )
        if not adoption.empty:
            print("   Top 5 adopted features:")
            for _, row in adoption.head(5).iterrows():
                print(f"   - {row['feature_name']}: {row['adoption_rate']:.1f}% adoption")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 4. Revenue Analysis
    print("\n4. Monthly Revenue Trends:")
    try:
        revenue = connector.get_revenue_analysis(
            table="transactions",
            user_id_col="user_id",
            transaction_id_col="transaction_id",
            amount_col="amount",
            timestamp_col="created_at",
            period='month',
            periods_back=6
        )
        if not revenue.empty:
            for _, row in revenue.head(3).iterrows():
                print(f"   {row['period']}: ${row['total_revenue']:,.2f} "
                      f"({row['paying_users']:,} users, "
                      f"ARPU: ${row['arpu']:.2f})")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 5. Conversion Funnel
    print("\n5. User Conversion Funnel:")
    try:
        funnel = connector.get_user_funnel(
            table=activity_table,
            user_id_col=user_id_col,
            step_col="event_name",
            timestamp_col=timestamp_col,
            steps=['signup', 'onboarding', 'first_action', 'purchase'],
            days_back=30
        )
        if not funnel.empty:
            for _, row in funnel.iterrows():
                print(f"   {row['funnel_step']}: {row['users']:,} users "
                      f"({row['pct_of_previous']:.1f}% of previous)")
    except Exception as e:
        print(f"   Error: {e}")


def example_performance_monitoring(connector: SnowflakeConnector):
    """Example: Performance monitoring and alerts."""
    print("\n=== Example: Performance Monitoring ===")
    
    # Get performance statistics
    perf_stats = connector.get_performance_stats()
    
    print("Query Performance Statistics:")
    print(f"- Total queries: {perf_stats['total_queries']}")
    print(f"- Average execution time: {perf_stats['avg_execution_time']:.2f}s")
    print(f"- 95th percentile: {perf_stats['p95_execution_time']:.2f}s")
    print(f"- Success rate: {perf_stats['success_rate']:.1f}%")
    
    if perf_stats.get('slowest_queries'):
        print("\nSlowest queries:")
        for q in perf_stats['slowest_queries'][:3]:
            print(f"- {q['sql_preview'][:50]}... ({q['execution_time']:.2f}s)")
    
    # Get cache statistics
    cache_stats = connector.get_cache_stats()
    
    print(f"\nCache Statistics:")
    print(f"- Entries: {cache_stats['entries']}")
    print(f"- Total size: {cache_stats['total_size_mb']:.2f} MB")
    if cache_stats.get('most_accessed'):
        print(f"- Most accessed query: {cache_stats['most_accessed'][:50]}...")


def example_incremental_load(connector: SnowflakeConnector):
    """Example: Incremental data loading."""
    print("\n=== Example: Incremental Load ===")
    
    source_query = """
    SELECT 
        user_id,
        event_name,
        event_timestamp,
        properties
    FROM raw_events
    WHERE event_timestamp >= DATEADD('hour', -24, CURRENT_TIMESTAMP())
    """
    
    try:
        rows_loaded = connector.incremental_load(
            source_query=source_query,
            target_table="processed_events",
            timestamp_col="event_timestamp",
            unique_key="event_id"
        )
        
        print(f"Incremental load completed: {rows_loaded:,} rows processed")
        
    except Exception as e:
        print(f"Incremental load failed: {e}")


def main():
    """Run all examples."""
    print("Snowflake Connector Examples")
    print("=" * 50)
    
    # Note: Update these examples with your actual Snowflake credentials
    # For demonstration, we'll show the structure without connecting
    
    # Basic examples
    example_basic_connection()
    example_oauth_authentication()
    example_key_pair_authentication()
    
    # If you have valid credentials, uncomment below:
    """
    # Create connector with your credentials
    config = {
        'account': os.getenv('SNOWFLAKE_ACCOUNT'),
        'user': os.getenv('SNOWFLAKE_USER'),
        'password': os.getenv('SNOWFLAKE_PASSWORD'),
        'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
        'database': os.getenv('SNOWFLAKE_DATABASE', 'ANALYTICS'),
        'schema': os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC')
    }
    
    connector = SnowflakeConnector(config)
    
    if connector.connect():
        try:
            # Run examples
            example_query_execution(connector)
            example_batch_queries(connector)
            example_query_optimization(connector)
            example_warehouse_management(connector)
            example_pm_analytics(connector)
            example_performance_monitoring(connector)
            example_incremental_load(connector)
            
        finally:
            connector.disconnect()
    """
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()