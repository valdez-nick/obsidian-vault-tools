"""
Snowflake Connector Implementation

Provides integration with Snowflake data warehouse for:
- Query execution and optimization
- Data extraction for analytics
- Result caching with TTL
- MCP server wrapper functionality
- Warehouse management
- OAuth 2.0 and key-pair authentication
- Connection pooling
- Query performance monitoring
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
from datetime import datetime, timedelta
from .base_connector import DataSourceConnector
import logging
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import re
from contextlib import contextmanager
from threading import Lock
from queue import Queue, Empty
import os
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.backends import default_backend

# Initialize logger first
logger = logging.getLogger(__name__)

try:
    import snowflake.connector
    from snowflake.connector import DictCursor, ProgrammingError, DatabaseError
    from snowflake.connector.errors import OperationalError
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    logger.warning("snowflake-connector-python not installed. Snowflake features will be limited.")


class ConnectionPool:
    """Thread-safe connection pool for Snowflake connections."""
    
    def __init__(self, connector, min_size: int = 2, max_size: int = 10):
        self.connector = connector
        self.min_size = min_size
        self.max_size = max_size
        self._pool = Queue(maxsize=max_size)
        self._lock = Lock()
        self._created_connections = 0
        
        # Pre-create minimum connections
        for _ in range(min_size):
            self._create_connection()
    
    def _create_connection(self):
        """Create a new connection and add to pool."""
        if self._created_connections < self.max_size:
            conn = self.connector._create_connection()
            if conn:
                self._pool.put(conn)
                self._created_connections += 1
                return True
        return False
    
    @contextmanager
    def get_connection(self, timeout: float = 5.0):
        """Get connection from pool with context manager."""
        conn = None
        try:
            # Try to get existing connection
            try:
                conn = self._pool.get(timeout=timeout)
            except Empty:
                # Create new connection if pool is empty and under max
                with self._lock:
                    if self._create_connection():
                        conn = self._pool.get_nowait()
                    else:
                        raise OperationalError("Connection pool exhausted")
            
            # Validate connection is still alive
            if not self._validate_connection(conn):
                conn.close()
                conn = self.connector._create_connection()
            
            yield conn
            
        finally:
            # Return connection to pool
            if conn and not conn.is_closed():
                self._pool.put(conn)
    
    def _validate_connection(self, conn) -> bool:
        """Check if connection is still valid."""
        try:
            conn.cursor().execute("SELECT 1").fetchone()
            return True
        except:
            return False
    
    def close_all(self):
        """Close all connections in pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break


class QueryCache:
    """Query result cache with TTL support."""
    
    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self._cache = {}
        self._lock = Lock()
        self._access_counts = {}
        self._last_cleanup = time.time()
    
    def _generate_key(self, sql: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from query and params."""
        cache_data = {
            'sql': sql.strip().lower(),
            'params': params or {}
        }
        return hashlib.sha256(
            json.dumps(cache_data, sort_keys=True).encode()
        ).hexdigest()
    
    def get(self, sql: str, params: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """Get cached result if available and not expired."""
        key = self._generate_key(sql, params)
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() < entry['expires_at']:
                    self._access_counts[key] = self._access_counts.get(key, 0) + 1
                    logger.debug(f"Cache hit for query: {sql[:50]}...")
                    return entry['data'].copy()
                else:
                    # Expired entry
                    del self._cache[key]
                    if key in self._access_counts:
                        del self._access_counts[key]
        
        return None
    
    def set(self, sql: str, data: pd.DataFrame, params: Optional[Dict] = None, 
            ttl: Optional[int] = None):
        """Store query result in cache."""
        key = self._generate_key(sql, params)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            self._cache[key] = {
                'data': data.copy(),
                'expires_at': time.time() + ttl,
                'created_at': time.time(),
                'sql': sql,
                'size_bytes': data.memory_usage(deep=True).sum()
            }
            
            # Cleanup old entries periodically
            if time.time() - self._last_cleanup > 3600:  # Every hour
                self._cleanup()
    
    def _cleanup(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time >= entry['expires_at']
        ]
        
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_counts:
                del self._access_counts[key]
        
        self._last_cleanup = current_time
        logger.debug(f"Cache cleanup: removed {len(expired_keys)} expired entries")
    
    def get_ttl_for_query(self, sql: str) -> int:
        """Determine optimal TTL based on query pattern."""
        sql_lower = sql.lower()
        
        # Real-time or frequently changing data - short TTL
        if any(keyword in sql_lower for keyword in ['real-time', 'current', 'now()', 'sysdate']):
            return 300  # 5 minutes
        
        # Historical data - longer TTL
        elif any(keyword in sql_lower for keyword in ['historical', 'archive', 'year', 'month']):
            return 86400  # 24 hours
        
        # Aggregated metrics - medium TTL
        elif any(keyword in sql_lower for keyword in ['sum', 'avg', 'count', 'group by']):
            return 3600  # 1 hour
        
        # Default TTL
        return self.default_ttl
    
    def clear(self, pattern: Optional[str] = None):
        """Clear cache entries matching pattern."""
        with self._lock:
            if pattern:
                keys_to_remove = [
                    key for key, entry in self._cache.items()
                    if pattern.lower() in entry['sql'].lower()
                ]
                for key in keys_to_remove:
                    del self._cache[key]
                    if key in self._access_counts:
                        del self._access_counts[key]
                logger.info(f"Cleared {len(keys_to_remove)} cache entries matching pattern: {pattern}")
            else:
                self._cache.clear()
                self._access_counts.clear()
                logger.info("Cleared entire query cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(entry['size_bytes'] for entry in self._cache.values())
            return {
                'entries': len(self._cache),
                'total_size_mb': total_size / (1024 * 1024),
                'hit_counts': dict(self._access_counts),
                'most_accessed': max(self._access_counts.items(), key=lambda x: x[1])[0] 
                                if self._access_counts else None
            }


class SnowflakeConnector(DataSourceConnector):
    """
    Comprehensive Snowflake data warehouse connector.
    
    Features:
    - OAuth 2.0 and key-pair authentication
    - Connection pooling for performance
    - Query result caching with intelligent TTL
    - Query optimization suggestions
    - Warehouse auto-scaling
    - Performance monitoring
    - PM-specific query templates
    """
    
    # PM Query Templates
    PM_QUERY_TEMPLATES = {
        'dau_mau': """
            WITH daily_active AS (
                SELECT 
                    DATE_TRUNC('day', {timestamp_col}) as activity_date,
                    COUNT(DISTINCT {user_id_col}) as dau
                FROM {table}
                WHERE {timestamp_col} >= DATEADD(day, -{days_back}, CURRENT_DATE())
                  AND {timestamp_col} < CURRENT_DATE()
                GROUP BY 1
            ),
            monthly_active AS (
                SELECT 
                    DATE_TRUNC('month', {timestamp_col}) as activity_month,
                    COUNT(DISTINCT {user_id_col}) as mau
                FROM {table}
                WHERE {timestamp_col} >= DATEADD(month, -3, CURRENT_DATE())
                  AND {timestamp_col} < CURRENT_DATE()
                GROUP BY 1
            )
            SELECT 
                d.activity_date,
                d.dau,
                m.mau,
                ROUND(d.dau::FLOAT / m.mau * 100, 2) as dau_mau_ratio
            FROM daily_active d
            JOIN monthly_active m 
                ON DATE_TRUNC('month', d.activity_date) = m.activity_month
            ORDER BY d.activity_date DESC
        """,
        
        'retention_cohort': """
            WITH cohorts AS (
                SELECT 
                    {user_id_col},
                    DATE_TRUNC('{cohort_period}', MIN({timestamp_col})) as cohort_date
                FROM {table}
                GROUP BY 1
            ),
            user_activities AS (
                SELECT 
                    {user_id_col},
                    DATE_TRUNC('{cohort_period}', {timestamp_col}) as activity_date
                FROM {table}
                GROUP BY 1, 2
            ),
            cohort_size AS (
                SELECT 
                    cohort_date,
                    COUNT(DISTINCT {user_id_col}) as cohort_users
                FROM cohorts
                GROUP BY 1
            ),
            retention_raw AS (
                SELECT 
                    c.cohort_date,
                    DATEDIFF('{cohort_period}', c.cohort_date, a.activity_date) as periods_later,
                    COUNT(DISTINCT c.{user_id_col}) as retained_users
                FROM cohorts c
                JOIN user_activities a ON c.{user_id_col} = a.{user_id_col}
                WHERE a.activity_date >= c.cohort_date
                GROUP BY 1, 2
            )
            SELECT 
                r.cohort_date,
                r.periods_later,
                r.retained_users,
                cs.cohort_users,
                ROUND(r.retained_users::FLOAT / cs.cohort_users * 100, 2) as retention_rate
            FROM retention_raw r
            JOIN cohort_size cs ON r.cohort_date = cs.cohort_date
            WHERE r.periods_later <= {max_periods}
            ORDER BY r.cohort_date DESC, r.periods_later
        """,
        
        'feature_adoption': """
            WITH feature_users AS (
                SELECT 
                    {feature_col} as feature_name,
                    COUNT(DISTINCT {user_id_col}) as unique_users,
                    COUNT(*) as total_events,
                    MIN({timestamp_col}) as first_used,
                    MAX({timestamp_col}) as last_used
                FROM {table}
                WHERE {timestamp_col} >= DATEADD(day, -{days_back}, CURRENT_DATE())
                  AND {feature_col} IS NOT NULL
                GROUP BY 1
            ),
            total_users AS (
                SELECT COUNT(DISTINCT {user_id_col}) as total_active_users
                FROM {table}
                WHERE {timestamp_col} >= DATEADD(day, -{days_back}, CURRENT_DATE())
            )
            SELECT 
                f.feature_name,
                f.unique_users,
                f.total_events,
                t.total_active_users,
                ROUND(f.unique_users::FLOAT / t.total_active_users * 100, 2) as adoption_rate,
                ROUND(f.total_events::FLOAT / f.unique_users, 2) as avg_events_per_user,
                f.first_used,
                f.last_used,
                DATEDIFF('day', f.first_used, f.last_used) as days_in_use
            FROM feature_users f
            CROSS JOIN total_users t
            ORDER BY f.unique_users DESC
        """,
        
        'revenue_analysis': """
            WITH revenue_metrics AS (
                SELECT 
                    DATE_TRUNC('{period}', {timestamp_col}) as period,
                    {segment_col} as segment,
                    COUNT(DISTINCT {user_id_col}) as paying_users,
                    COUNT(DISTINCT {transaction_id_col}) as transactions,
                    SUM({amount_col}) as total_revenue,
                    AVG({amount_col}) as avg_transaction_value,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {amount_col}) as median_transaction_value
                FROM {table}
                WHERE {timestamp_col} >= DATEADD({period}, -{periods_back}, CURRENT_DATE())
                  AND {amount_col} > 0
                GROUP BY 1, 2
            ),
            period_over_period AS (
                SELECT 
                    period,
                    segment,
                    total_revenue,
                    LAG(total_revenue) OVER (PARTITION BY segment ORDER BY period) as prev_period_revenue,
                    paying_users,
                    LAG(paying_users) OVER (PARTITION BY segment ORDER BY period) as prev_period_users
                FROM revenue_metrics
            )
            SELECT 
                period,
                segment,
                paying_users,
                transactions,
                total_revenue,
                avg_transaction_value,
                median_transaction_value,
                ROUND(total_revenue::FLOAT / paying_users, 2) as arpu,
                ROUND((total_revenue - prev_period_revenue)::FLOAT / prev_period_revenue * 100, 2) as revenue_growth_pct,
                ROUND((paying_users - prev_period_users)::FLOAT / prev_period_users * 100, 2) as user_growth_pct
            FROM period_over_period
            ORDER BY period DESC, total_revenue DESC
        """,
        
        'user_funnel': """
            WITH funnel_steps AS (
                SELECT 
                    {user_id_col},
                    MAX(CASE WHEN {step_col} = '{step1}' THEN 1 ELSE 0 END) as step1,
                    MAX(CASE WHEN {step_col} = '{step2}' THEN 1 ELSE 0 END) as step2,
                    MAX(CASE WHEN {step_col} = '{step3}' THEN 1 ELSE 0 END) as step3,
                    MAX(CASE WHEN {step_col} = '{step4}' THEN 1 ELSE 0 END) as step4,
                    MAX(CASE WHEN {step_col} = '{step5}' THEN 1 ELSE 0 END) as step5
                FROM {table}
                WHERE {timestamp_col} >= DATEADD(day, -{days_back}, CURRENT_DATE())
                GROUP BY 1
            ),
            funnel_summary AS (
                SELECT 
                    COUNT(*) as total_users,
                    SUM(step1) as reached_step1,
                    SUM(CASE WHEN step1 = 1 THEN step2 ELSE 0 END) as reached_step2,
                    SUM(CASE WHEN step1 = 1 AND step2 = 1 THEN step3 ELSE 0 END) as reached_step3,
                    SUM(CASE WHEN step1 = 1 AND step2 = 1 AND step3 = 1 THEN step4 ELSE 0 END) as reached_step4,
                    SUM(CASE WHEN step1 = 1 AND step2 = 1 AND step3 = 1 AND step4 = 1 THEN step5 ELSE 0 END) as reached_step5
                FROM funnel_steps
            )
            SELECT 
                'Step 1: {step1}' as funnel_step,
                reached_step1 as users,
                ROUND(reached_step1::FLOAT / total_users * 100, 2) as pct_of_total,
                100.00 as pct_of_previous
            FROM funnel_summary
            UNION ALL
            SELECT 
                'Step 2: {step2}' as funnel_step,
                reached_step2 as users,
                ROUND(reached_step2::FLOAT / total_users * 100, 2) as pct_of_total,
                ROUND(reached_step2::FLOAT / reached_step1 * 100, 2) as pct_of_previous
            FROM funnel_summary
            UNION ALL
            SELECT 
                'Step 3: {step3}' as funnel_step,
                reached_step3 as users,
                ROUND(reached_step3::FLOAT / total_users * 100, 2) as pct_of_total,
                ROUND(reached_step3::FLOAT / reached_step2 * 100, 2) as pct_of_previous
            FROM funnel_summary
            WHERE reached_step2 > 0
            UNION ALL
            SELECT 
                'Step 4: {step4}' as funnel_step,
                reached_step4 as users,
                ROUND(reached_step4::FLOAT / total_users * 100, 2) as pct_of_total,
                ROUND(reached_step4::FLOAT / reached_step3 * 100, 2) as pct_of_previous
            FROM funnel_summary
            WHERE reached_step3 > 0
            UNION ALL
            SELECT 
                'Step 5: {step5}' as funnel_step,
                reached_step5 as users,
                ROUND(reached_step5::FLOAT / total_users * 100, 2) as pct_of_total,
                ROUND(reached_step5::FLOAT / reached_step4 * 100, 2) as pct_of_previous
            FROM funnel_summary
            WHERE reached_step4 > 0
        """
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Snowflake connector with configuration.
        
        Config should include:
        - account: Snowflake account identifier
        - auth_type: 'password', 'oauth', or 'key_pair'
        - user: Username for authentication
        - password: Password (for password auth)
        - token: OAuth token (for OAuth auth)
        - private_key_path: Path to private key file (for key-pair auth)
        - private_key_passphrase: Passphrase for private key (optional)
        - warehouse: Default warehouse to use
        - database: Default database
        - schema: Default schema
        - role: Optional role to use
        - pool_size: Connection pool size (default: 5)
        - cache_ttl: Default cache TTL in seconds (default: 3600)
        """
        super().__init__("Snowflake", config)
        
        # Basic configuration
        self.account = config.get('account')
        self.auth_type = config.get('auth_type', 'password')
        self.user = config.get('user')
        self.warehouse = config.get('warehouse')
        self.database = config.get('database')
        self.schema = config.get('schema')
        self.role = config.get('role')
        
        # Authentication configuration
        self.password = config.get('password')
        self.token = config.get('token')
        self.private_key_path = config.get('private_key_path')
        self.private_key_passphrase = config.get('private_key_passphrase')
        
        # Advanced configuration
        self.pool_size = config.get('pool_size', 5)
        self.cache_ttl = config.get('cache_ttl', 3600)
        
        # Initialize components
        self._connection_pool = None
        self._query_cache = QueryCache(default_ttl=self.cache_ttl)
        self._performance_monitor = PerformanceMonitor()
        self._query_optimizer = QueryOptimizer()
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate connector configuration."""
        if not self.account:
            raise ValueError("Snowflake account is required")
        if not self.user:
            raise ValueError("Username is required")
        
        if self.auth_type == 'password' and not self.password:
            raise ValueError("Password is required for password authentication")
        elif self.auth_type == 'oauth' and not self.token:
            raise ValueError("OAuth token is required for OAuth authentication")
        elif self.auth_type == 'key_pair' and not self.private_key_path:
            raise ValueError("Private key path is required for key-pair authentication")
        elif self.auth_type not in ['password', 'oauth', 'key_pair']:
            raise ValueError(f"Invalid auth_type: {self.auth_type}")
        
    def _get_private_key(self) -> bytes:
        """Load and return private key for key-pair authentication."""
        with open(self.private_key_path, 'rb') as key_file:
            private_key = load_pem_private_key(
                key_file.read(),
                password=self.private_key_passphrase.encode() if self.private_key_passphrase else None,
                backend=default_backend()
            )
            
        return private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
    
    def _create_connection(self):
        """Create a new Snowflake connection."""
        if not SNOWFLAKE_AVAILABLE:
            raise ImportError("snowflake-connector-python is not installed")
        
        try:
            conn_params = {
                'account': self.account,
                'user': self.user,
                'warehouse': self.warehouse,
                'database': self.database,
                'schema': self.schema,
                'autocommit': True,
                'client_session_keep_alive': True
            }
            
            if self.role:
                conn_params['role'] = self.role
            
            # Set authentication parameters
            if self.auth_type == 'password':
                conn_params['password'] = self.password
            elif self.auth_type == 'oauth':
                conn_params['token'] = self.token
                conn_params['authenticator'] = 'oauth'
            elif self.auth_type == 'key_pair':
                conn_params['private_key'] = self._get_private_key()
            
            # Create connection
            conn = snowflake.connector.connect(**conn_params)
            logger.info(f"Successfully created Snowflake connection for user {self.user}")
            return conn
            
        except Exception as e:
            logger.error(f"Failed to create Snowflake connection: {str(e)}")
            self.last_error = str(e)
            raise
    
    def connect(self) -> bool:
        """
        Establish connection pool to Snowflake.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if not self._connection_pool:
                self._connection_pool = ConnectionPool(
                    self,
                    min_size=max(1, self.pool_size // 2),
                    max_size=self.pool_size
                )
            
            # Test connection
            with self._connection_pool.get_connection() as conn:
                conn.cursor().execute("SELECT CURRENT_VERSION()").fetchone()
            
            self.is_connected = True
            logger.info(f"Successfully connected to Snowflake account {self.account}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
            self.last_error = str(e)
            self.is_connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Snowflake and close connection pool.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            if self._connection_pool:
                self._connection_pool.close_all()
                self._connection_pool = None
            
            self.is_connected = False
            logger.info("Successfully disconnected from Snowflake")
            return True
            
        except Exception as e:
            logger.error(f"Error during disconnection: {str(e)}")
            self.last_error = str(e)
            return False
    
    def validate_connection(self) -> bool:
        """
        Validate Snowflake connection by running a test query.
        
        Returns:
            True if connection is valid, False otherwise
        """
        if not self.is_connected or not self._connection_pool:
            return False
        
        try:
            with self._connection_pool.get_connection() as conn:
                result = conn.cursor().execute("SELECT CURRENT_TIMESTAMP()").fetchone()
                logger.debug(f"Connection validated at {result[0]}")
                return True
        except Exception as e:
            logger.error(f"Connection validation failed: {str(e)}")
            return False
    
    def execute_query(self, sql: str, params: Optional[Dict[str, Any]] = None,
                     use_cache: bool = True, warehouse: Optional[str] = None) -> pd.DataFrame:
        """
        Execute a SQL query with caching and performance monitoring.
        
        Args:
            sql: SQL query string
            params: Optional query parameters for safe parameterization
            use_cache: Whether to use cached results
            warehouse: Optional warehouse override
            
        Returns:
            DataFrame with query results
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Snowflake")
        
        # Check cache first
        if use_cache:
            cached_result = self._query_cache.get(sql, params)
            if cached_result is not None:
                return cached_result
        
        # Start performance monitoring
        start_time = time.time()
        
        try:
            with self._connection_pool.get_connection() as conn:
                cursor = conn.cursor(DictCursor)
                
                # Switch warehouse if specified
                if warehouse and warehouse != self.warehouse:
                    cursor.execute(f"USE WAREHOUSE {warehouse}")
                
                # Execute query with parameters
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                
                # Fetch results
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=columns)
                
                # Record performance metrics
                execution_time = time.time() - start_time
                self._performance_monitor.record_query(
                    sql=sql,
                    execution_time=execution_time,
                    rows_returned=len(df),
                    warehouse=warehouse or self.warehouse
                )
                
                # Cache results with intelligent TTL
                if use_cache:
                    ttl = self._query_cache.get_ttl_for_query(sql)
                    self._query_cache.set(sql, df, params, ttl)
                
                logger.info(f"Query executed in {execution_time:.2f}s, returned {len(df)} rows")
                return df
                
        except Exception as e:
            execution_time = time.time() - start_time
            self._performance_monitor.record_query(
                sql=sql,
                execution_time=execution_time,
                error=str(e),
                warehouse=warehouse or self.warehouse
            )
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def execute_batch(self, queries: List[str], parallel: bool = True,
                     max_workers: int = 5) -> List[pd.DataFrame]:
        """
        Execute multiple queries in batch with optional parallelization.
        
        Args:
            queries: List of SQL queries
            parallel: Whether to execute queries in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of DataFrames with results
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Snowflake")
        
        if parallel and len(queries) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(queries))) as executor:
                futures = [executor.submit(self.execute_query, query, use_cache=True) 
                          for query in queries]
                results = [future.result() for future in futures]
        else:
            results = [self.execute_query(query, use_cache=True) for query in queries]
        
        logger.info(f"Executed batch of {len(queries)} queries")
        return results
    
    def extract_data(self, query: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract data from Snowflake based on query parameters.
        
        Query parameters:
        - sql: SQL query string
        - params: Optional query parameters
        - use_cache: Whether to use cached results
        - warehouse: Optional warehouse override
        
        Returns:
            DataFrame containing query results
        """
        sql = query.get('sql')
        if not sql:
            raise ValueError("SQL query is required")
        
        return self.execute_query(
            sql=sql,
            params=query.get('params'),
            use_cache=query.get('use_cache', True),
            warehouse=query.get('warehouse')
        )
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get comprehensive Snowflake account metadata.
        
        Returns:
            Dictionary containing:
            - warehouses: Available warehouses with status
            - databases: Available databases
            - schemas: Available schemas
            - tables: Table information
            - current_role: Current role
            - account_info: Account details
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Snowflake")
        
        metadata = {}
        
        try:
            # Get warehouses
            warehouses_df = self.execute_query(
                "SHOW WAREHOUSES",
                use_cache=False
            )
            metadata['warehouses'] = warehouses_df.to_dict('records')
            
            # Get databases
            databases_df = self.execute_query(
                "SHOW DATABASES",
                use_cache=False
            )
            metadata['databases'] = databases_df.to_dict('records')
            
            # Get schemas in current database
            if self.database:
                schemas_df = self.execute_query(
                    f"SHOW SCHEMAS IN DATABASE {self.database}",
                    use_cache=False
                )
                metadata['schemas'] = schemas_df.to_dict('records')
            
            # Get tables in current schema
            if self.database and self.schema:
                tables_df = self.execute_query(
                    f"SHOW TABLES IN {self.database}.{self.schema}",
                    use_cache=False
                )
                metadata['tables'] = tables_df.to_dict('records')
            
            # Get current role and account info
            info_df = self.execute_query(
                "SELECT CURRENT_ROLE() as role, CURRENT_ACCOUNT() as account, "
                "CURRENT_WAREHOUSE() as warehouse, CURRENT_DATABASE() as database, "
                "CURRENT_SCHEMA() as schema",
                use_cache=False
            )
            metadata['current_context'] = info_df.iloc[0].to_dict()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get metadata: {str(e)}")
            raise
    
    def optimize_query(self, sql: str) -> Dict[str, Any]:
        """
        Analyze and provide optimization suggestions for SQL query.
        
        Args:
            sql: Original SQL query
            
        Returns:
            Dictionary with:
            - original_query: The input query
            - optimized_query: Suggested optimized version
            - suggestions: List of optimization suggestions
            - estimated_improvement: Estimated performance improvement
        """
        return self._query_optimizer.optimize(sql)
    
    def get_query_history(self, limit: int = 100, 
                         filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Get recent query history with optional filters.
        
        Args:
            limit: Maximum number of queries to return
            filters: Optional filters (user, warehouse, date_from, date_to)
            
        Returns:
            DataFrame with query history
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Snowflake")
        
        # Build query with filters
        base_query = """
            SELECT 
                QUERY_ID,
                QUERY_TEXT,
                DATABASE_NAME,
                SCHEMA_NAME,
                QUERY_TYPE,
                USER_NAME,
                ROLE_NAME,
                WAREHOUSE_NAME,
                WAREHOUSE_SIZE,
                EXECUTION_STATUS,
                ERROR_CODE,
                ERROR_MESSAGE,
                START_TIME,
                END_TIME,
                TOTAL_ELAPSED_TIME,
                BYTES_SCANNED,
                ROWS_PRODUCED,
                COMPILATION_TIME,
                EXECUTION_TIME,
                QUEUED_PROVISIONING_TIME,
                QUEUED_REPAIR_TIME,
                QUEUED_OVERLOAD_TIME,
                TRANSACTION_BLOCKED_TIME,
                CREDITS_USED_CLOUD_SERVICES
            FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())
            WHERE 1=1
        """
        
        conditions = []
        params = {}
        
        if filters:
            if 'user' in filters:
                conditions.append("AND USER_NAME = %(user)s")
                params['user'] = filters['user']
            
            if 'warehouse' in filters:
                conditions.append("AND WAREHOUSE_NAME = %(warehouse)s")
                params['warehouse'] = filters['warehouse']
            
            if 'date_from' in filters:
                conditions.append("AND START_TIME >= %(date_from)s")
                params['date_from'] = filters['date_from']
            
            if 'date_to' in filters:
                conditions.append("AND START_TIME <= %(date_to)s")
                params['date_to'] = filters['date_to']
        
        query = base_query + " ".join(conditions) + f" ORDER BY START_TIME DESC LIMIT {limit}"
        
        return self.execute_query(query, params=params, use_cache=False)
    
    def manage_warehouse(self, warehouse: str, action: str, 
                        size: Optional[str] = None) -> bool:
        """
        Manage warehouse operations (resume, suspend, resize).
        
        Args:
            warehouse: Warehouse name
            action: Action to perform (resume, suspend, resize, create, drop)
            size: New size for resize action (e.g., 'X-SMALL', 'SMALL', 'MEDIUM', 'LARGE')
            
        Returns:
            True if operation successful
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Snowflake")
        
        try:
            if action == 'resume':
                self.execute_query(f"ALTER WAREHOUSE {warehouse} RESUME", use_cache=False)
            elif action == 'suspend':
                self.execute_query(f"ALTER WAREHOUSE {warehouse} SUSPEND", use_cache=False)
            elif action == 'resize':
                if not size:
                    raise ValueError("Size parameter required for resize action")
                self.execute_query(f"ALTER WAREHOUSE {warehouse} SET WAREHOUSE_SIZE = '{size}'", 
                                 use_cache=False)
            elif action == 'create':
                if not size:
                    size = 'X-SMALL'
                self.execute_query(
                    f"CREATE WAREHOUSE IF NOT EXISTS {warehouse} WITH WAREHOUSE_SIZE = '{size}'",
                    use_cache=False
                )
            elif action == 'drop':
                self.execute_query(f"DROP WAREHOUSE IF EXISTS {warehouse}", use_cache=False)
            else:
                raise ValueError(f"Invalid action: {action}")
            
            logger.info(f"Successfully performed {action} on warehouse {warehouse}")
            return True
            
        except Exception as e:
            logger.error(f"Warehouse management failed: {str(e)}")
            self.last_error = str(e)
            return False
    
    def auto_scale_warehouse(self, queries_per_minute: int, 
                           current_size: str = 'SMALL') -> Optional[str]:
        """
        Recommend warehouse size based on workload.
        
        Args:
            queries_per_minute: Current query rate
            current_size: Current warehouse size
            
        Returns:
            Recommended warehouse size or None if no change needed
        """
        size_thresholds = {
            'X-SMALL': 10,
            'SMALL': 50,
            'MEDIUM': 100,
            'LARGE': 200,
            'X-LARGE': 500,
            '2X-LARGE': 1000,
            '3X-LARGE': 2000,
            '4X-LARGE': 5000
        }
        
        sizes = list(size_thresholds.keys())
        current_index = sizes.index(current_size)
        
        # Find appropriate size
        recommended_size = current_size
        for size, threshold in size_thresholds.items():
            if queries_per_minute <= threshold:
                recommended_size = size
                break
        
        # Only recommend change if different from current
        if recommended_size != current_size:
            logger.info(f"Recommending warehouse resize from {current_size} to {recommended_size}")
            return recommended_size
        
        return None
    
    def clear_cache(self, query_pattern: Optional[str] = None):
        """
        Clear query result cache.
        
        Args:
            query_pattern: Optional pattern to match queries to clear
        """
        self._query_cache.clear(query_pattern)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._query_cache.get_stats()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get query performance statistics."""
        return self._performance_monitor.get_stats()
    
    # PM-specific query methods
    def get_dau_mau(self, table: str, user_id_col: str, timestamp_col: str,
                    days_back: int = 30) -> pd.DataFrame:
        """
        Get Daily Active Users (DAU) and Monthly Active Users (MAU) metrics.
        
        Args:
            table: Table containing user activity
            user_id_col: Column name for user ID
            timestamp_col: Column name for timestamp
            days_back: Number of days to look back
            
        Returns:
            DataFrame with DAU, MAU, and DAU/MAU ratio
        """
        query = self.PM_QUERY_TEMPLATES['dau_mau'].format(
            table=table,
            user_id_col=user_id_col,
            timestamp_col=timestamp_col,
            days_back=days_back
        )
        return self.execute_query(query)
    
    def get_retention_cohort(self, table: str, user_id_col: str, timestamp_col: str,
                           cohort_period: str = 'week', max_periods: int = 12) -> pd.DataFrame:
        """
        Get user retention cohort analysis.
        
        Args:
            table: Table containing user activity
            user_id_col: Column name for user ID
            timestamp_col: Column name for timestamp
            cohort_period: Cohort period ('day', 'week', 'month')
            max_periods: Maximum number of periods to track
            
        Returns:
            DataFrame with cohort retention data
        """
        query = self.PM_QUERY_TEMPLATES['retention_cohort'].format(
            table=table,
            user_id_col=user_id_col,
            timestamp_col=timestamp_col,
            cohort_period=cohort_period,
            max_periods=max_periods
        )
        return self.execute_query(query)
    
    def get_feature_adoption(self, table: str, user_id_col: str, 
                           feature_col: str, timestamp_col: str,
                           days_back: int = 30) -> pd.DataFrame:
        """
        Get feature adoption metrics.
        
        Args:
            table: Table containing feature usage
            user_id_col: Column name for user ID
            feature_col: Column name for feature/event name
            timestamp_col: Column name for timestamp
            days_back: Number of days to analyze
            
        Returns:
            DataFrame with feature adoption metrics
        """
        query = self.PM_QUERY_TEMPLATES['feature_adoption'].format(
            table=table,
            user_id_col=user_id_col,
            feature_col=feature_col,
            timestamp_col=timestamp_col,
            days_back=days_back
        )
        return self.execute_query(query)
    
    def get_revenue_analysis(self, table: str, user_id_col: str, 
                           transaction_id_col: str, amount_col: str,
                           timestamp_col: str, segment_col: str = "'All'",
                           period: str = 'month', periods_back: int = 12) -> pd.DataFrame:
        """
        Get revenue analysis by period and segment.
        
        Args:
            table: Table containing revenue data
            user_id_col: Column name for user ID
            transaction_id_col: Column name for transaction ID
            amount_col: Column name for transaction amount
            timestamp_col: Column name for timestamp
            segment_col: Column name for segmentation (or static value)
            period: Analysis period ('day', 'week', 'month', 'quarter')
            periods_back: Number of periods to analyze
            
        Returns:
            DataFrame with revenue metrics
        """
        query = self.PM_QUERY_TEMPLATES['revenue_analysis'].format(
            table=table,
            user_id_col=user_id_col,
            transaction_id_col=transaction_id_col,
            amount_col=amount_col,
            timestamp_col=timestamp_col,
            segment_col=segment_col,
            period=period,
            periods_back=periods_back
        )
        return self.execute_query(query)
    
    def get_user_funnel(self, table: str, user_id_col: str, step_col: str,
                       timestamp_col: str, steps: List[str], 
                       days_back: int = 30) -> pd.DataFrame:
        """
        Get user funnel analysis for up to 5 steps.
        
        Args:
            table: Table containing funnel events
            user_id_col: Column name for user ID
            step_col: Column name for funnel step/event
            timestamp_col: Column name for timestamp
            steps: List of funnel steps (up to 5)
            days_back: Number of days to analyze
            
        Returns:
            DataFrame with funnel conversion metrics
        """
        if len(steps) > 5:
            raise ValueError("Maximum 5 funnel steps supported")
        
        # Pad steps list to 5 elements
        steps_padded = steps + [''] * (5 - len(steps))
        
        query = self.PM_QUERY_TEMPLATES['user_funnel'].format(
            table=table,
            user_id_col=user_id_col,
            step_col=step_col,
            timestamp_col=timestamp_col,
            days_back=days_back,
            step1=steps_padded[0],
            step2=steps_padded[1],
            step3=steps_padded[2],
            step4=steps_padded[3],
            step5=steps_padded[4]
        )
        
        # Remove empty steps from result
        result = self.execute_query(query)
        return result[result['users'] > 0]
    
    def incremental_load(self, source_query: str, target_table: str,
                        timestamp_col: str, unique_key: str,
                        last_loaded_timestamp: Optional[datetime] = None) -> int:
        """
        Perform incremental data load from query to target table.
        
        Args:
            source_query: Source data query
            target_table: Target table name
            timestamp_col: Column to track incremental loads
            unique_key: Unique key column for merge
            last_loaded_timestamp: Last loaded timestamp (auto-detected if None)
            
        Returns:
            Number of rows loaded
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Snowflake")
        
        try:
            # Get last loaded timestamp if not provided
            if not last_loaded_timestamp:
                max_ts_query = f"SELECT MAX({timestamp_col}) FROM {target_table}"
                result = self.execute_query(max_ts_query, use_cache=False)
                last_loaded_timestamp = result.iloc[0, 0] if not result.empty else None
            
            # Build incremental query
            if last_loaded_timestamp:
                incremental_query = f"""
                    WITH source_data AS ({source_query})
                    SELECT * FROM source_data
                    WHERE {timestamp_col} > '{last_loaded_timestamp}'
                """
            else:
                incremental_query = f"WITH source_data AS ({source_query}) SELECT * FROM source_data"
            
            # Create temporary table with new data
            temp_table = f"{target_table}_temp_{int(time.time())}"
            self.execute_query(
                f"CREATE TEMPORARY TABLE {temp_table} AS {incremental_query}",
                use_cache=False
            )
            
            # Get row count
            count_result = self.execute_query(
                f"SELECT COUNT(*) FROM {temp_table}",
                use_cache=False
            )
            rows_to_load = count_result.iloc[0, 0]
            
            if rows_to_load > 0:
                # Merge into target table
                merge_query = f"""
                    MERGE INTO {target_table} t
                    USING {temp_table} s
                    ON t.{unique_key} = s.{unique_key}
                    WHEN MATCHED THEN
                        UPDATE SET t.* = s.*
                    WHEN NOT MATCHED THEN
                        INSERT VALUES (s.*)
                """
                self.execute_query(merge_query, use_cache=False)
                logger.info(f"Incremental load completed: {rows_to_load} rows processed")
            else:
                logger.info("No new data to load")
            
            # Cleanup
            self.execute_query(f"DROP TABLE IF EXISTS {temp_table}", use_cache=False)
            
            return rows_to_load
            
        except Exception as e:
            logger.error(f"Incremental load failed: {str(e)}")
            raise


class PerformanceMonitor:
    """Monitor and track query performance metrics."""
    
    def __init__(self):
        self._metrics = []
        self._lock = Lock()
    
    def record_query(self, sql: str, execution_time: float, 
                    warehouse: str, rows_returned: int = 0,
                    error: Optional[str] = None):
        """Record query performance metrics."""
        with self._lock:
            self._metrics.append({
                'timestamp': datetime.now(),
                'sql_hash': hashlib.md5(sql.encode()).hexdigest(),
                'sql_preview': sql[:100],
                'execution_time': execution_time,
                'warehouse': warehouse,
                'rows_returned': rows_returned,
                'success': error is None,
                'error': error
            })
            
            # Keep only last 1000 queries
            if len(self._metrics) > 1000:
                self._metrics = self._metrics[-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if not self._metrics:
                return {
                    'total_queries': 0,
                    'avg_execution_time': 0,
                    'success_rate': 0,
                    'queries_by_warehouse': {}
                }
            
            df = pd.DataFrame(self._metrics)
            
            return {
                'total_queries': len(df),
                'avg_execution_time': df['execution_time'].mean(),
                'median_execution_time': df['execution_time'].median(),
                'p95_execution_time': df['execution_time'].quantile(0.95),
                'success_rate': (df['success'].sum() / len(df)) * 100,
                'queries_by_warehouse': df.groupby('warehouse').size().to_dict(),
                'slowest_queries': df.nlargest(5, 'execution_time')[
                    ['sql_preview', 'execution_time', 'warehouse']
                ].to_dict('records'),
                'error_rate_by_hour': df.set_index('timestamp').resample('H')['success'].apply(
                    lambda x: (1 - x.mean()) * 100
                ).to_dict()
            }
    
    def get_alerts(self, threshold_seconds: float = 30) -> List[Dict[str, Any]]:
        """Get performance alerts for slow queries."""
        with self._lock:
            slow_queries = [
                m for m in self._metrics 
                if m['execution_time'] > threshold_seconds and m['success']
            ]
            
            return [{
                'timestamp': q['timestamp'],
                'sql_preview': q['sql_preview'],
                'execution_time': q['execution_time'],
                'warehouse': q['warehouse'],
                'severity': 'high' if q['execution_time'] > threshold_seconds * 2 else 'medium'
            } for q in slow_queries[-10:]]  # Last 10 slow queries


class QueryOptimizer:
    """Provide query optimization suggestions."""
    
    def __init__(self):
        self.optimization_rules = {
            'select_star': {
                'pattern': r'SELECT\s+\*',
                'suggestion': 'Avoid SELECT * - specify only required columns',
                'impact': 'medium'
            },
            'missing_where': {
                'pattern': r'^SELECT[^W]*FROM[^W]*$',
                'suggestion': 'Consider adding WHERE clause to filter data',
                'impact': 'high'
            },
            'order_without_limit': {
                'pattern': r'ORDER\s+BY(?!.*LIMIT)',
                'suggestion': 'Consider adding LIMIT when using ORDER BY',
                'impact': 'medium'
            },
            'missing_partition': {
                'pattern': r'WHERE(?!.*DATE_TRUNC)',
                'suggestion': 'Consider using partition columns in WHERE clause',
                'impact': 'high'
            },
            'inefficient_join': {
                'pattern': r'JOIN.*ON.*OR',
                'suggestion': 'Avoid OR conditions in JOIN - consider UNION instead',
                'impact': 'high'
            },
            'subquery_in_select': {
                'pattern': r'SELECT.*\(SELECT',
                'suggestion': 'Consider moving subquery to JOIN or CTE',
                'impact': 'medium'
            }
        }
    
    def optimize(self, sql: str) -> Dict[str, Any]:
        """Analyze query and provide optimization suggestions."""
        sql_upper = sql.upper()
        suggestions = []
        estimated_improvement = 0
        
        # Check optimization rules
        for rule_name, rule in self.optimization_rules.items():
            if re.search(rule['pattern'], sql_upper, re.IGNORECASE):
                suggestions.append({
                    'rule': rule_name,
                    'suggestion': rule['suggestion'],
                    'impact': rule['impact']
                })
                
                # Estimate improvement
                if rule['impact'] == 'high':
                    estimated_improvement += 30
                elif rule['impact'] == 'medium':
                    estimated_improvement += 15
                else:
                    estimated_improvement += 5
        
        # Generate optimized query (basic transformations)
        optimized_sql = sql
        
        # Replace SELECT * with column list (if possible)
        if re.search(r'SELECT\s+\*', sql_upper):
            optimized_sql = re.sub(
                r'SELECT\s+\*',
                'SELECT /* specify columns here */',
                optimized_sql,
                flags=re.IGNORECASE
            )
        
        # Add LIMIT if ORDER BY without LIMIT
        if re.search(r'ORDER\s+BY(?!.*LIMIT)', sql_upper):
            optimized_sql += '\nLIMIT 1000'
        
        return {
            'original_query': sql,
            'optimized_query': optimized_sql,
            'suggestions': suggestions,
            'estimated_improvement': min(estimated_improvement, 80),  # Cap at 80%
            'optimization_score': max(0, 100 - len(suggestions) * 15)
        }