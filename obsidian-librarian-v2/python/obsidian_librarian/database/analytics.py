"""
Analytics database implementation using DuckDB.

Provides fast analytics and reporting on vault metrics, usage patterns,
and performance data.
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import structlog

from .base import AnalyticsDatabase, DatabaseConfig, ConnectionError, QueryError

logger = structlog.get_logger(__name__)


class AnalyticsDB(AnalyticsDatabase):
    """DuckDB-based analytics database for vault metrics."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.db_path = config.analytics_path
        self.connection: Optional[duckdb.DuckDBPyConnection] = None
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize DuckDB connection and create database."""
        async with self._lock:
            try:
                # Ensure directory exists
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create connection
                self.connection = duckdb.connect(str(self.db_path))
                
                # Configure DuckDB settings
                self.connection.execute(f"SET memory_limit='{self.config.analytics_memory_limit}'")
                self.connection.execute(f"SET threads={self.config.analytics_threads}")
                
                # Create tables
                await self.create_tables()
                
                logger.info("Analytics database initialized", path=self.db_path)
                
            except Exception as e:
                logger.error("Failed to initialize analytics database", error=str(e))
                raise ConnectionError(f"Analytics database initialization failed: {e}")
    
    async def close(self) -> None:
        """Close database connection."""
        async with self._lock:
            if self.connection:
                try:
                    self.connection.close()
                    self.connection = None
                    logger.debug("Analytics database connection closed")
                except Exception as e:
                    logger.error("Error closing analytics database", error=str(e))
    
    async def health_check(self) -> bool:
        """Check database health."""
        try:
            if not self.connection:
                return False
            
            # Simple query to test connection
            result = await self._execute_query("SELECT 1")
            return len(result) > 0
            
        except Exception:
            return False
    
    async def create_tables(self) -> None:
        """Create analytics tables."""
        tables = [
            # Note metrics table
            """
            CREATE TABLE IF NOT EXISTS note_metrics (
                note_id VARCHAR PRIMARY KEY,
                file_path VARCHAR NOT NULL,
                title VARCHAR,
                word_count INTEGER,
                character_count INTEGER,
                line_count INTEGER,
                link_count INTEGER,
                backlink_count INTEGER,
                tag_count INTEGER,
                task_count INTEGER,
                completed_task_count INTEGER,
                last_modified TIMESTAMP,
                created_date TIMESTAMP,
                file_size INTEGER,
                reading_time_minutes REAL,
                complexity_score REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Vault statistics over time
            """
            CREATE TABLE IF NOT EXISTS vault_stats (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_notes INTEGER,
                total_words INTEGER,
                total_characters INTEGER,
                total_links INTEGER,
                total_tasks INTEGER,
                total_tags INTEGER,
                average_note_length REAL,
                vault_size_bytes INTEGER,
                orphaned_notes INTEGER,
                most_linked_note VARCHAR,
                newest_note VARCHAR,
                oldest_note VARCHAR
            )
            """,
            
            # Usage patterns
            """
            CREATE TABLE IF NOT EXISTS usage_events (
                id INTEGER PRIMARY KEY,
                event_type VARCHAR NOT NULL,
                event_data JSON,
                note_id VARCHAR,
                user_action VARCHAR,
                duration_ms INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Search queries and results
            """
            CREATE TABLE IF NOT EXISTS search_metrics (
                id INTEGER PRIMARY KEY,
                query VARCHAR NOT NULL,
                result_count INTEGER,
                execution_time_ms REAL,
                search_type VARCHAR,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Performance metrics
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY,
                operation VARCHAR NOT NULL,
                duration_ms REAL,
                memory_used_mb REAL,
                cpu_percent REAL,
                notes_processed INTEGER,
                success BOOLEAN,
                error_message VARCHAR,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Research activities
            """
            CREATE TABLE IF NOT EXISTS research_metrics (
                id INTEGER PRIMARY KEY,
                query VARCHAR NOT NULL,
                sources_scraped INTEGER,
                results_found INTEGER,
                notes_created INTEGER,
                processing_time_ms REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for table_sql in tables:
            await self._execute_query(table_sql)
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_note_metrics_modified ON note_metrics(last_modified)",
            "CREATE INDEX IF NOT EXISTS idx_vault_stats_timestamp ON vault_stats(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_usage_events_timestamp ON usage_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_usage_events_type ON usage_events(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_search_metrics_timestamp ON search_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_research_timestamp ON research_metrics(timestamp)",
        ]
        
        for index_sql in indexes:
            await self._execute_query(index_sql)
        
        logger.debug("Analytics tables and indexes created")
    
    async def insert_note_metrics(self, note_id: str, metrics: Dict[str, Any]) -> None:
        """Insert or update note metrics."""
        query = """
        INSERT OR REPLACE INTO note_metrics (
            note_id, file_path, title, word_count, character_count, line_count,
            link_count, backlink_count, tag_count, task_count, completed_task_count,
            last_modified, created_date, file_size, reading_time_minutes,
            complexity_score, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
        
        params = (
            note_id,
            metrics.get("file_path"),
            metrics.get("title"),
            metrics.get("word_count", 0),
            metrics.get("character_count", 0),
            metrics.get("line_count", 0),
            metrics.get("link_count", 0),
            metrics.get("backlink_count", 0),
            metrics.get("tag_count", 0),
            metrics.get("task_count", 0),
            metrics.get("completed_task_count", 0),
            metrics.get("last_modified"),
            metrics.get("created_date"),
            metrics.get("file_size", 0),
            metrics.get("reading_time_minutes", 0.0),
            metrics.get("complexity_score", 0.0),
        )
        
        await self._execute_query(query, params)
    
    async def get_vault_stats(self) -> Dict[str, Any]:
        """Get comprehensive vault statistics."""
        stats_query = """
        SELECT 
            COUNT(*) as total_notes,
            SUM(word_count) as total_words,
            SUM(character_count) as total_characters,
            SUM(link_count) as total_links,
            SUM(task_count) as total_tasks,
            SUM(tag_count) as total_tags,
            AVG(word_count) as average_note_length,
            SUM(file_size) as vault_size_bytes,
            COUNT(CASE WHEN link_count = 0 AND backlink_count = 0 THEN 1 END) as orphaned_notes
        FROM note_metrics
        """
        
        result = await self._execute_query(stats_query)
        stats = dict(zip([
            "total_notes", "total_words", "total_characters", "total_links",
            "total_tasks", "total_tags", "average_note_length", "vault_size_bytes",
            "orphaned_notes"
        ], result[0] if result else [0] * 9))
        
        # Get most linked note
        most_linked_query = """
        SELECT note_id, title, backlink_count 
        FROM note_metrics 
        ORDER BY backlink_count DESC 
        LIMIT 1
        """
        most_linked = await self._execute_query(most_linked_query)
        if most_linked:
            stats["most_linked_note"] = {
                "id": most_linked[0][0],
                "title": most_linked[0][1],
                "backlinks": most_linked[0][2]
            }
        
        # Get newest and oldest notes
        newest_query = "SELECT note_id, title, created_date FROM note_metrics ORDER BY created_date DESC LIMIT 1"
        oldest_query = "SELECT note_id, title, created_date FROM note_metrics ORDER BY created_date ASC LIMIT 1"
        
        newest = await self._execute_query(newest_query)
        oldest = await self._execute_query(oldest_query)
        
        if newest:
            stats["newest_note"] = {"id": newest[0][0], "title": newest[0][1], "created": newest[0][2]}
        if oldest:
            stats["oldest_note"] = {"id": oldest[0][0], "title": oldest[0][1], "created": oldest[0][2]}
        
        return stats
    
    async def get_usage_patterns(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get usage patterns over the specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = """
        SELECT 
            DATE_TRUNC('day', timestamp) as day,
            event_type,
            COUNT(*) as count,
            AVG(duration_ms) as avg_duration
        FROM usage_events 
        WHERE timestamp >= ?
        GROUP BY DATE_TRUNC('day', timestamp), event_type
        ORDER BY day DESC
        """
        
        result = await self._execute_query(query, (cutoff_date,))
        
        patterns = []
        for row in result:
            patterns.append({
                "day": row[0],
                "event_type": row[1],
                "count": row[2],
                "avg_duration_ms": row[3]
            })
        
        return patterns
    
    async def record_usage_event(
        self, 
        event_type: str, 
        event_data: Optional[Dict[str, Any]] = None,
        note_id: Optional[str] = None,
        user_action: Optional[str] = None,
        duration_ms: Optional[int] = None
    ) -> None:
        """Record a usage event."""
        query = """
        INSERT INTO usage_events (event_type, event_data, note_id, user_action, duration_ms)
        VALUES (?, ?, ?, ?, ?)
        """
        
        params = (
            event_type,
            json.dumps(event_data) if event_data else None,
            note_id,
            user_action,
            duration_ms
        )
        
        await self._execute_query(query, params)
    
    async def record_search_metrics(
        self,
        query: str,
        result_count: int,
        execution_time_ms: float,
        search_type: str = "full_text"
    ) -> None:
        """Record search performance metrics."""
        sql = """
        INSERT INTO search_metrics (query, result_count, execution_time_ms, search_type)
        VALUES (?, ?, ?, ?)
        """
        
        await self._execute_query(sql, (query, result_count, execution_time_ms, search_type))
    
    async def record_performance_metrics(
        self,
        operation: str,
        duration_ms: float,
        memory_used_mb: Optional[float] = None,
        cpu_percent: Optional[float] = None,
        notes_processed: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Record performance metrics for operations."""
        query = """
        INSERT INTO performance_metrics 
        (operation, duration_ms, memory_used_mb, cpu_percent, notes_processed, success, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (operation, duration_ms, memory_used_mb, cpu_percent, notes_processed, success, error_message)
        await self._execute_query(query, params)
    
    async def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Get performance report for the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = """
        SELECT 
            operation,
            COUNT(*) as total_operations,
            AVG(duration_ms) as avg_duration,
            MIN(duration_ms) as min_duration,
            MAX(duration_ms) as max_duration,
            AVG(memory_used_mb) as avg_memory,
            AVG(cpu_percent) as avg_cpu,
            SUM(notes_processed) as total_notes_processed,
            COUNT(CASE WHEN success THEN 1 END) as successful_operations,
            COUNT(CASE WHEN NOT success THEN 1 END) as failed_operations
        FROM performance_metrics
        WHERE timestamp >= ?
        GROUP BY operation
        ORDER BY total_operations DESC
        """
        
        result = await self._execute_query(query, (cutoff_date,))
        
        report = {"operations": []}
        for row in result:
            report["operations"].append({
                "operation": row[0],
                "total_operations": row[1],
                "avg_duration_ms": row[2],
                "min_duration_ms": row[3],
                "max_duration_ms": row[4],
                "avg_memory_mb": row[5],
                "avg_cpu_percent": row[6],
                "total_notes_processed": row[7],
                "successful_operations": row[8],
                "failed_operations": row[9],
                "success_rate": row[8] / row[1] if row[1] > 0 else 0
            })
        
        return report
    
    async def cleanup_old_data(self, days: int = 90) -> int:
        """Clean up old analytics data beyond the specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        tables = ["usage_events", "search_metrics", "performance_metrics", "research_metrics"]
        total_deleted = 0
        
        for table in tables:
            query = f"DELETE FROM {table} WHERE timestamp < ?"
            result = await self._execute_query(query, (cutoff_date,))
            # DuckDB doesn't return affected rows directly, so we'll log the operation
            logger.debug(f"Cleaned up old data from {table}")
        
        logger.info(f"Cleaned up analytics data older than {days} days")
        return total_deleted
    
    async def _execute_query(self, query: str, params: Optional[tuple] = None) -> List[tuple]:
        """Execute a query with error handling."""
        if not self.connection:
            raise ConnectionError("Analytics database not initialized")
        
        try:
            if params:
                result = self.connection.execute(query, params).fetchall()
            else:
                result = self.connection.execute(query).fetchall()
            
            return result
            
        except Exception as e:
            logger.error("Analytics query failed", query=query[:100], error=str(e))
            raise QueryError(f"Query execution failed: {e}")
    
    async def export_data(self, export_path: Path, format: str = "parquet") -> None:
        """Export analytics data to file."""
        export_path.mkdir(parents=True, exist_ok=True)
        
        tables = [
            "note_metrics", "vault_stats", "usage_events", 
            "search_metrics", "performance_metrics", "research_metrics"
        ]
        
        for table in tables:
            if format == "parquet":
                query = f"COPY {table} TO '{export_path / f'{table}.parquet'}' (FORMAT PARQUET)"
            elif format == "csv":
                query = f"COPY {table} TO '{export_path / f'{table}.csv'}' (FORMAT CSV, HEADER)"
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            await self._execute_query(query)
        
        logger.info(f"Analytics data exported to {export_path}")