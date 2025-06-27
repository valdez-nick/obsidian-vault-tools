"""
Unit tests for SnowflakeConnector

Tests authentication methods, query execution, caching,
performance monitoring, and PM-specific queries.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
from datetime import datetime, timedelta
import json
import time
from queue import Queue, Empty

from connectors.snowflake_connector import (
    SnowflakeConnector, ConnectionPool, QueryCache, 
    PerformanceMonitor, QueryOptimizer
)


class TestSnowflakeConnector(unittest.TestCase):
    """Test cases for SnowflakeConnector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'account': 'test_account',
            'user': 'test_user',
            'password': 'test_password',
            'warehouse': 'TEST_WH',
            'database': 'TEST_DB',
            'schema': 'TEST_SCHEMA',
            'role': 'TEST_ROLE',
            'pool_size': 3,
            'cache_ttl': 1800
        }
        
    def test_init_with_password_auth(self):
        """Test initialization with password authentication."""
        connector = SnowflakeConnector(self.config)
        
        self.assertEqual(connector.account, 'test_account')
        self.assertEqual(connector.user, 'test_user')
        self.assertEqual(connector.auth_type, 'password')
        self.assertEqual(connector.warehouse, 'TEST_WH')
        self.assertEqual(connector.pool_size, 3)
        self.assertEqual(connector.cache_ttl, 1800)
        
    def test_init_with_oauth_auth(self):
        """Test initialization with OAuth authentication."""
        config = self.config.copy()
        config['auth_type'] = 'oauth'
        config['token'] = 'oauth_token'
        del config['password']
        
        connector = SnowflakeConnector(config)
        self.assertEqual(connector.auth_type, 'oauth')
        self.assertEqual(connector.token, 'oauth_token')
        
    def test_init_with_key_pair_auth(self):
        """Test initialization with key-pair authentication."""
        config = self.config.copy()
        config['auth_type'] = 'key_pair'
        config['private_key_path'] = '/path/to/key.pem'
        config['private_key_passphrase'] = 'passphrase'
        del config['password']
        
        connector = SnowflakeConnector(config)
        self.assertEqual(connector.auth_type, 'key_pair')
        self.assertEqual(connector.private_key_path, '/path/to/key.pem')
        
    def test_validate_config_missing_account(self):
        """Test config validation with missing account."""
        config = self.config.copy()
        del config['account']
        
        with self.assertRaises(ValueError) as context:
            SnowflakeConnector(config)
        self.assertIn("account is required", str(context.exception))
        
    def test_validate_config_invalid_auth_type(self):
        """Test config validation with invalid auth type."""
        config = self.config.copy()
        config['auth_type'] = 'invalid'
        
        with self.assertRaises(ValueError) as context:
            SnowflakeConnector(config)
        self.assertIn("Invalid auth_type", str(context.exception))
        
    @patch('connectors.snowflake_connector.snowflake.connector.connect')
    def test_connect_success(self, mock_connect):
        """Test successful connection."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ['3.5.0']
        mock_connect.return_value = mock_conn
        
        connector = SnowflakeConnector(self.config)
        result = connector.connect()
        
        self.assertTrue(result)
        self.assertTrue(connector.is_connected)
        self.assertIsNotNone(connector._connection_pool)
        
    @patch('connectors.snowflake_connector.snowflake.connector.connect')
    def test_connect_failure(self, mock_connect):
        """Test connection failure."""
        mock_connect.side_effect = Exception("Connection failed")
        
        connector = SnowflakeConnector(self.config)
        result = connector.connect()
        
        self.assertFalse(result)
        self.assertFalse(connector.is_connected)
        self.assertEqual(connector.last_error, "Connection failed")
        
    @patch('connectors.snowflake_connector.snowflake.connector.connect')
    def test_disconnect(self, mock_connect):
        """Test disconnection."""
        mock_conn = Mock()
        mock_conn.is_closed.return_value = False
        mock_connect.return_value = mock_conn
        
        connector = SnowflakeConnector(self.config)
        connector.connect()
        result = connector.disconnect()
        
        self.assertTrue(result)
        self.assertFalse(connector.is_connected)
        self.assertIsNone(connector._connection_pool)
        
    @patch('connectors.snowflake_connector.snowflake.connector.connect')
    def test_execute_query_with_cache(self, mock_connect):
        """Test query execution with caching."""
        # Setup mock connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.description = [('col1',), ('col2',)]
        mock_cursor.fetchall.return_value = [
            {'col1': 1, 'col2': 'a'},
            {'col1': 2, 'col2': 'b'}
        ]
        mock_connect.return_value = mock_conn
        
        connector = SnowflakeConnector(self.config)
        connector.connect()
        
        # First query - should hit database
        sql = "SELECT * FROM test_table"
        result1 = connector.execute_query(sql)
        
        self.assertEqual(len(result1), 2)
        self.assertEqual(list(result1.columns), ['col1', 'col2'])
        mock_cursor.execute.assert_called_with(sql)
        
        # Second query - should hit cache
        mock_cursor.execute.reset_mock()
        result2 = connector.execute_query(sql)
        
        self.assertEqual(len(result2), 2)
        mock_cursor.execute.assert_not_called()
        
    @patch('connectors.snowflake_connector.snowflake.connector.connect')
    def test_execute_query_with_params(self, mock_connect):
        """Test parameterized query execution."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.description = [('count',)]
        mock_cursor.fetchall.return_value = [{'count': 42}]
        mock_connect.return_value = mock_conn
        
        connector = SnowflakeConnector(self.config)
        connector.connect()
        
        sql = "SELECT COUNT(*) FROM users WHERE created_date > %(date)s"
        params = {'date': '2024-01-01'}
        
        result = connector.execute_query(sql, params=params)
        
        self.assertEqual(result.iloc[0]['count'], 42)
        mock_cursor.execute.assert_called_with(sql, params)
        
    @patch('connectors.snowflake_connector.snowflake.connector.connect')
    def test_execute_batch_parallel(self, mock_connect):
        """Test batch query execution in parallel."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.description = [('result',)]
        mock_cursor.fetchall.side_effect = [
            [{'result': 1}],
            [{'result': 2}],
            [{'result': 3}]
        ]
        mock_connect.return_value = mock_conn
        
        connector = SnowflakeConnector(self.config)
        connector.connect()
        
        queries = [
            "SELECT 1 as result",
            "SELECT 2 as result",
            "SELECT 3 as result"
        ]
        
        results = connector.execute_batch(queries, parallel=True)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].iloc[0]['result'], 1)
        self.assertEqual(results[1].iloc[0]['result'], 2)
        self.assertEqual(results[2].iloc[0]['result'], 3)
        
    def test_query_optimization(self):
        """Test query optimization suggestions."""
        connector = SnowflakeConnector(self.config)
        
        # Test SELECT * optimization
        sql = "SELECT * FROM large_table"
        result = connector.optimize_query(sql)
        
        self.assertIn("Avoid SELECT *", result['suggestions'][0]['suggestion'])
        self.assertIn("specify columns here", result['optimized_query'])
        
        # Test ORDER BY without LIMIT
        sql = "SELECT col1 FROM table ORDER BY col2"
        result = connector.optimize_query(sql)
        
        self.assertIn("LIMIT", result['optimized_query'])
        self.assertIn("Consider adding LIMIT", result['suggestions'][0]['suggestion'])
        
    @patch('connectors.snowflake_connector.snowflake.connector.connect')
    def test_warehouse_management(self, mock_connect):
        """Test warehouse management operations."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        connector = SnowflakeConnector(self.config)
        connector.connect()
        
        # Test resume
        result = connector.manage_warehouse('TEST_WH', 'resume')
        self.assertTrue(result)
        mock_cursor.execute.assert_called_with("ALTER WAREHOUSE TEST_WH RESUME")
        
        # Test suspend
        result = connector.manage_warehouse('TEST_WH', 'suspend')
        self.assertTrue(result)
        mock_cursor.execute.assert_called_with("ALTER WAREHOUSE TEST_WH SUSPEND")
        
        # Test resize
        result = connector.manage_warehouse('TEST_WH', 'resize', size='MEDIUM')
        self.assertTrue(result)
        mock_cursor.execute.assert_called_with(
            "ALTER WAREHOUSE TEST_WH SET WAREHOUSE_SIZE = 'MEDIUM'"
        )
        
    def test_auto_scale_warehouse(self):
        """Test warehouse auto-scaling recommendations."""
        connector = SnowflakeConnector(self.config)
        
        # Test scale up recommendation
        result = connector.auto_scale_warehouse(75, 'X-SMALL')
        self.assertEqual(result, 'SMALL')
        
        # Test no change needed
        result = connector.auto_scale_warehouse(30, 'SMALL')
        self.assertIsNone(result)
        
        # Test scale down recommendation
        result = connector.auto_scale_warehouse(5, 'LARGE')
        self.assertEqual(result, 'X-SMALL')
        
    @patch('connectors.snowflake_connector.snowflake.connector.connect')
    def test_pm_query_dau_mau(self, mock_connect):
        """Test DAU/MAU PM query template."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.description = [('activity_date',), ('dau',), ('mau',), ('dau_mau_ratio',)]
        mock_cursor.fetchall.return_value = [
            {'activity_date': '2024-01-01', 'dau': 1000, 'mau': 5000, 'dau_mau_ratio': 20.0}
        ]
        mock_connect.return_value = mock_conn
        
        connector = SnowflakeConnector(self.config)
        connector.connect()
        
        result = connector.get_dau_mau(
            table='user_events',
            user_id_col='user_id',
            timestamp_col='event_time',
            days_back=30
        )
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['dau'], 1000)
        self.assertEqual(result.iloc[0]['dau_mau_ratio'], 20.0)
        
    @patch('connectors.snowflake_connector.snowflake.connector.connect')
    def test_incremental_load(self, mock_connect):
        """Test incremental data loading."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock responses for different queries
        mock_cursor.fetchall.side_effect = [
            [{'max_ts': '2024-01-01 00:00:00'}],  # MAX timestamp query
            [],  # CREATE TEMP TABLE (no result)
            [{'count': 100}],  # COUNT query
            [],  # MERGE query
            []   # DROP TABLE query
        ]
        mock_cursor.description = [('max_ts',)]
        
        mock_connect.return_value = mock_conn
        
        connector = SnowflakeConnector(self.config)
        connector.connect()
        
        rows_loaded = connector.incremental_load(
            source_query="SELECT * FROM source_table",
            target_table="target_table",
            timestamp_col="updated_at",
            unique_key="id"
        )
        
        self.assertEqual(rows_loaded, 100)
        
        # Verify queries were executed
        calls = mock_cursor.execute.call_args_list
        self.assertTrue(any("MAX(updated_at)" in str(call) for call in calls))
        self.assertTrue(any("CREATE TEMPORARY TABLE" in str(call) for call in calls))
        self.assertTrue(any("MERGE INTO" in str(call) for call in calls))


class TestConnectionPool(unittest.TestCase):
    """Test cases for ConnectionPool class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_connector = Mock()
        self.mock_connector._create_connection = Mock()
        
    def test_pool_initialization(self):
        """Test connection pool initialization."""
        mock_conn = Mock()
        mock_conn.is_closed.return_value = False
        self.mock_connector._create_connection.return_value = mock_conn
        
        pool = ConnectionPool(self.mock_connector, min_size=2, max_size=5)
        
        # Should create min_size connections
        self.assertEqual(self.mock_connector._create_connection.call_count, 2)
        self.assertEqual(pool._created_connections, 2)
        
    def test_get_connection_from_pool(self):
        """Test getting connection from pool."""
        mock_conn = Mock()
        mock_conn.is_closed.return_value = False
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = [1]
        
        self.mock_connector._create_connection.return_value = mock_conn
        
        pool = ConnectionPool(self.mock_connector, min_size=1, max_size=3)
        
        with pool.get_connection() as conn:
            self.assertEqual(conn, mock_conn)
            
        # Connection should be returned to pool
        self.assertFalse(pool._pool.empty())
        
    def test_pool_exhaustion(self):
        """Test behavior when pool is exhausted."""
        mock_conn = Mock()
        mock_conn.is_closed.return_value = False
        self.mock_connector._create_connection.return_value = mock_conn
        
        pool = ConnectionPool(self.mock_connector, min_size=1, max_size=1)
        
        # Get the only connection
        with pool.get_connection() as conn1:
            # Try to get another connection - should timeout
            with self.assertRaises(Exception):
                with pool.get_connection(timeout=0.1) as conn2:
                    pass
                    
    def test_connection_validation(self):
        """Test connection validation in pool."""
        mock_conn = Mock()
        mock_conn.is_closed.return_value = False
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        # First call succeeds (validation), second fails
        mock_cursor.execute.side_effect = [
            mock_cursor,  # Validation succeeds
            Exception("Connection lost")  # Validation fails
        ]
        mock_cursor.fetchone.side_effect = [
            [1],  # First validation
            Exception("Connection lost")  # Second validation
        ]
        
        self.mock_connector._create_connection.return_value = mock_conn
        
        pool = ConnectionPool(self.mock_connector, min_size=1, max_size=2)
        
        # First get should succeed
        with pool.get_connection() as conn:
            self.assertEqual(conn, mock_conn)
            
        # Reset for failed validation
        mock_cursor.execute.side_effect = Exception("Connection lost")
        
        # Second get should create new connection
        self.mock_connector._create_connection.reset_mock()
        with pool.get_connection() as conn:
            self.mock_connector._create_connection.assert_called_once()


class TestQueryCache(unittest.TestCase):
    """Test cases for QueryCache class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = QueryCache(default_ttl=3600)
        
    def test_cache_hit(self):
        """Test cache hit scenario."""
        sql = "SELECT * FROM users"
        df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        
        # Store in cache
        self.cache.set(sql, df)
        
        # Retrieve from cache
        result = self.cache.get(sql)
        
        self.assertIsNotNone(result)
        pd.testing.assert_frame_equal(result, df)
        
    def test_cache_miss(self):
        """Test cache miss scenario."""
        sql = "SELECT * FROM products"
        result = self.cache.get(sql)
        
        self.assertIsNone(result)
        
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        sql = "SELECT * FROM orders"
        df = pd.DataFrame({'id': [1]})
        
        # Store with very short TTL
        self.cache.set(sql, df, ttl=0.1)
        
        # Should be available immediately
        self.assertIsNotNone(self.cache.get(sql))
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        self.assertIsNone(self.cache.get(sql))
        
    def test_cache_with_params(self):
        """Test caching with query parameters."""
        sql = "SELECT * FROM users WHERE id = %(id)s"
        params1 = {'id': 1}
        params2 = {'id': 2}
        
        df1 = pd.DataFrame({'id': [1], 'name': ['Alice']})
        df2 = pd.DataFrame({'id': [2], 'name': ['Bob']})
        
        # Store different results for different params
        self.cache.set(sql, df1, params=params1)
        self.cache.set(sql, df2, params=params2)
        
        # Retrieve with different params
        result1 = self.cache.get(sql, params=params1)
        result2 = self.cache.get(sql, params=params2)
        
        pd.testing.assert_frame_equal(result1, df1)
        pd.testing.assert_frame_equal(result2, df2)
        
    def test_ttl_for_query_patterns(self):
        """Test TTL determination based on query patterns."""
        # Real-time query - short TTL
        sql = "SELECT * FROM sessions WHERE last_active > NOW() - INTERVAL '5 minutes'"
        ttl = self.cache.get_ttl_for_query(sql)
        self.assertEqual(ttl, 300)  # 5 minutes
        
        # Historical query - long TTL
        sql = "SELECT * FROM historical_data WHERE year = 2023"
        ttl = self.cache.get_ttl_for_query(sql)
        self.assertEqual(ttl, 86400)  # 24 hours
        
        # Aggregated query - medium TTL
        sql = "SELECT COUNT(*), AVG(price) FROM orders GROUP BY category"
        ttl = self.cache.get_ttl_for_query(sql)
        self.assertEqual(ttl, 3600)  # 1 hour
        
    def test_cache_clear_pattern(self):
        """Test clearing cache entries by pattern."""
        # Add multiple entries
        self.cache.set("SELECT * FROM users", pd.DataFrame({'a': [1]}))
        self.cache.set("SELECT * FROM products", pd.DataFrame({'b': [2]}))
        self.cache.set("SELECT * FROM users WHERE active = 1", pd.DataFrame({'c': [3]}))
        
        # Clear entries matching pattern
        self.cache.clear(pattern="users")
        
        # Check what remains
        self.assertIsNone(self.cache.get("SELECT * FROM users"))
        self.assertIsNone(self.cache.get("SELECT * FROM users WHERE active = 1"))
        self.assertIsNotNone(self.cache.get("SELECT * FROM products"))
        
    def test_cache_stats(self):
        """Test cache statistics."""
        # Add some entries
        self.cache.set("query1", pd.DataFrame({'a': [1, 2, 3]}))
        self.cache.set("query2", pd.DataFrame({'b': [4, 5]}))
        
        # Access one entry multiple times
        self.cache.get("query1")
        self.cache.get("query1")
        self.cache.get("query2")
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['entries'], 2)
        self.assertGreater(stats['total_size_mb'], 0)
        self.assertIn('hit_counts', stats)


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for PerformanceMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()
        
    def test_record_successful_query(self):
        """Test recording successful query metrics."""
        self.monitor.record_query(
            sql="SELECT * FROM users",
            execution_time=1.5,
            warehouse="SMALL",
            rows_returned=100
        )
        
        stats = self.monitor.get_stats()
        
        self.assertEqual(stats['total_queries'], 1)
        self.assertEqual(stats['avg_execution_time'], 1.5)
        self.assertEqual(stats['success_rate'], 100.0)
        self.assertEqual(stats['queries_by_warehouse']['SMALL'], 1)
        
    def test_record_failed_query(self):
        """Test recording failed query metrics."""
        self.monitor.record_query(
            sql="SELECT * FROM invalid_table",
            execution_time=0.1,
            warehouse="SMALL",
            error="Table not found"
        )
        
        stats = self.monitor.get_stats()
        
        self.assertEqual(stats['success_rate'], 0.0)
        
    def test_performance_alerts(self):
        """Test performance alert generation."""
        # Record some slow queries
        self.monitor.record_query("SELECT 1", 35, "SMALL", 1)
        self.monitor.record_query("SELECT 2", 65, "LARGE", 1)
        self.monitor.record_query("SELECT 3", 5, "SMALL", 1)
        
        alerts = self.monitor.get_alerts(threshold_seconds=30)
        
        self.assertEqual(len(alerts), 2)
        self.assertEqual(alerts[0]['severity'], 'medium')
        self.assertEqual(alerts[1]['severity'], 'high')
        
    def test_metrics_limit(self):
        """Test that metrics are limited to prevent memory issues."""
        # Record more than limit
        for i in range(1100):
            self.monitor.record_query(f"SELECT {i}", 0.1, "SMALL", 1)
            
        # Should keep only last 1000
        self.assertEqual(len(self.monitor._metrics), 1000)


class TestQueryOptimizer(unittest.TestCase):
    """Test cases for QueryOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = QueryOptimizer()
        
    def test_select_star_optimization(self):
        """Test SELECT * optimization detection."""
        sql = "SELECT * FROM large_table WHERE status = 'active'"
        result = self.optimizer.optimize(sql)
        
        suggestions = [s['rule'] for s in result['suggestions']]
        self.assertIn('select_star', suggestions)
        self.assertIn('specify columns here', result['optimized_query'])
        
    def test_missing_where_clause(self):
        """Test detection of missing WHERE clause."""
        sql = "SELECT user_id, name FROM users"
        result = self.optimizer.optimize(sql)
        
        suggestions = [s['rule'] for s in result['suggestions']]
        self.assertIn('missing_where', suggestions)
        
    def test_order_without_limit(self):
        """Test ORDER BY without LIMIT detection."""
        sql = "SELECT * FROM events ORDER BY created_at DESC"
        result = self.optimizer.optimize(sql)
        
        suggestions = [s['rule'] for s in result['suggestions']]
        self.assertIn('order_without_limit', suggestions)
        self.assertIn('LIMIT 1000', result['optimized_query'])
        
    def test_inefficient_join(self):
        """Test inefficient JOIN detection."""
        sql = """
        SELECT * FROM orders o
        JOIN users u ON o.user_id = u.id OR o.guest_id = u.id
        """
        result = self.optimizer.optimize(sql)
        
        suggestions = [s['rule'] for s in result['suggestions']]
        self.assertIn('inefficient_join', suggestions)
        
    def test_optimization_score(self):
        """Test optimization score calculation."""
        # Well-optimized query
        sql = "SELECT id, name FROM users WHERE active = 1 LIMIT 100"
        result = self.optimizer.optimize(sql)
        self.assertGreater(result['optimization_score'], 80)
        
        # Poorly optimized query
        sql = "SELECT * FROM users"
        result = self.optimizer.optimize(sql)
        self.assertLess(result['optimization_score'], 80)


if __name__ == '__main__':
    unittest.main()