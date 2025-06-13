"""
Database migration utilities for Obsidian Librarian.

Handles database schema updates, data migrations, and compatibility checks.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from .base import DatabaseManager, DatabaseConfig, MigrationError

logger = structlog.get_logger(__name__)


class Migration:
    """Represents a database migration."""
    
    def __init__(
        self, 
        version: str, 
        description: str, 
        up_func: callable, 
        down_func: Optional[callable] = None
    ):
        self.version = version
        self.description = description
        self.up_func = up_func
        self.down_func = down_func
        self.timestamp = datetime.now()
    
    async def apply(self, db_manager: DatabaseManager) -> None:
        """Apply the migration."""
        logger.info("Applying migration", version=self.version, description=self.description)
        try:
            await self.up_func(db_manager)
            logger.info("Migration applied successfully", version=self.version)
        except Exception as e:
            logger.error("Migration failed", version=self.version, error=str(e))
            raise MigrationError(f"Migration {self.version} failed: {e}")
    
    async def rollback(self, db_manager: DatabaseManager) -> None:
        """Rollback the migration."""
        if not self.down_func:
            raise MigrationError(f"Migration {self.version} has no rollback function")
        
        logger.info("Rolling back migration", version=self.version)
        try:
            await self.down_func(db_manager)
            logger.info("Migration rolled back successfully", version=self.version)
        except Exception as e:
            logger.error("Migration rollback failed", version=self.version, error=str(e))
            raise MigrationError(f"Migration {self.version} rollback failed: {e}")


class MigrationManager:
    """Manages database migrations."""
    
    def __init__(self, db_manager: DatabaseManager, migrations_dir: Optional[Path] = None):
        self.db_manager = db_manager
        self.migrations_dir = migrations_dir or Path(".obsidian-librarian/migrations")
        self.migrations: List[Migration] = []
        self._init_builtin_migrations()
    
    def _init_builtin_migrations(self) -> None:
        """Initialize built-in migrations."""
        # Migration 001: Initial database setup
        self.migrations.append(Migration(
            version="001",
            description="Initial database setup",
            up_func=self._migration_001_up,
            down_func=self._migration_001_down
        ))
        
        # Migration 002: Add performance indexes
        self.migrations.append(Migration(
            version="002",
            description="Add performance indexes",
            up_func=self._migration_002_up,
            down_func=self._migration_002_down
        ))
        
        # Migration 003: Vector database optimization
        self.migrations.append(Migration(
            version="003",
            description="Vector database optimization",
            up_func=self._migration_003_up,
            down_func=self._migration_003_down
        ))
    
    async def get_current_version(self) -> Optional[str]:
        """Get the current migration version."""
        try:
            if not self.db_manager.analytics:
                return None
            
            # Check if migration table exists
            result = await self.db_manager.analytics._execute_query("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='migrations'
            """)
            
            if not result:
                return None
            
            # Get latest migration
            result = await self.db_manager.analytics._execute_query("""
                SELECT version FROM migrations 
                ORDER BY applied_at DESC 
                LIMIT 1
            """)
            
            return result[0][0] if result else None
            
        except Exception as e:
            logger.warning("Could not determine current migration version", error=str(e))
            return None
    
    async def create_migration_table(self) -> None:
        """Create the migrations tracking table."""
        if not self.db_manager.analytics:
            logger.warning("Analytics database not available, skipping migration table creation")
            return
        
        await self.db_manager.analytics._execute_query("""
            CREATE TABLE IF NOT EXISTS migrations (
                version VARCHAR PRIMARY KEY,
                description VARCHAR,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    async def record_migration(self, migration: Migration) -> None:
        """Record a completed migration."""
        if not self.db_manager.analytics:
            logger.warning("Analytics database not available, skipping migration recording")
            return
        
        await self.db_manager.analytics._execute_query("""
            INSERT OR REPLACE INTO migrations (version, description, applied_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (migration.version, migration.description))
    
    async def remove_migration_record(self, version: str) -> None:
        """Remove a migration record."""
        if not self.db_manager.analytics:
            raise MigrationError("Analytics database not available")
        
        await self.db_manager.analytics._execute_query("""
            DELETE FROM migrations WHERE version = ?
        """, (version,))
    
    async def migrate(self, target_version: Optional[str] = None) -> None:
        """Run migrations up to target version."""
        await self.create_migration_table()
        current_version = await self.get_current_version()
        
        logger.info("Starting migration", current_version=current_version, target_version=target_version)
        
        # Determine which migrations to run
        migrations_to_run = []
        for migration in self.migrations:
            if current_version is None or migration.version > current_version:
                if target_version is None or migration.version <= target_version:
                    migrations_to_run.append(migration)
        
        if not migrations_to_run:
            logger.info("No migrations to run")
            return
        
        # Run migrations
        for migration in migrations_to_run:
            await migration.apply(self.db_manager)
            await self.record_migration(migration)
        
        logger.info("Migration completed", migrations_applied=len(migrations_to_run))
    
    async def rollback(self, target_version: str) -> None:
        """Rollback migrations to target version."""
        current_version = await self.get_current_version()
        
        if not current_version:
            logger.info("No migrations to rollback")
            return
        
        logger.info("Starting rollback", current_version=current_version, target_version=target_version)
        
        # Determine which migrations to rollback
        migrations_to_rollback = []
        for migration in reversed(self.migrations):
            if migration.version > target_version and migration.version <= current_version:
                migrations_to_rollback.append(migration)
        
        if not migrations_to_rollback:
            logger.info("No migrations to rollback")
            return
        
        # Rollback migrations
        for migration in migrations_to_rollback:
            await migration.rollback(self.db_manager)
            await self.remove_migration_record(migration.version)
        
        logger.info("Rollback completed", migrations_rolled_back=len(migrations_to_rollback))
    
    # Built-in migration functions
    
    async def _migration_001_up(self, db_manager: DatabaseManager) -> None:
        """Migration 001: Initial database setup."""
        # Analytics tables are created in AnalyticsDB.create_tables()
        if db_manager.analytics:
            await db_manager.analytics.create_tables()
        
        # Vector collection is created in VectorDB.create_collection()
        if db_manager.vector:
            await db_manager.vector.create_collection()
        
        # Cache tables are created in CacheDB initialization
        # No additional setup needed
    
    async def _migration_001_down(self, db_manager: DatabaseManager) -> None:
        """Migration 001 rollback: Drop all tables."""
        logger.warning("Rollback will destroy all data!")
        
        if db_manager.analytics:
            tables = [
                "note_metrics", "vault_stats", "usage_events",
                "search_metrics", "performance_metrics", "research_metrics"
            ]
            for table in tables:
                try:
                    await db_manager.analytics._execute_query(f"DROP TABLE IF EXISTS {table}")
                except Exception as e:
                    logger.warning(f"Failed to drop table {table}", error=str(e))
    
    async def _migration_002_up(self, db_manager: DatabaseManager) -> None:
        """Migration 002: Add performance indexes."""
        if not db_manager.analytics:
            return
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_note_metrics_word_count ON note_metrics(word_count)",
            "CREATE INDEX IF NOT EXISTS idx_note_metrics_link_count ON note_metrics(link_count)",
            "CREATE INDEX IF NOT EXISTS idx_usage_events_note_id ON usage_events(note_id)",
            "CREATE INDEX IF NOT EXISTS idx_search_metrics_query ON search_metrics(query)",
            "CREATE INDEX IF NOT EXISTS idx_performance_operation ON performance_metrics(operation)",
        ]
        
        for index_sql in indexes:
            await db_manager.analytics._execute_query(index_sql)
    
    async def _migration_002_down(self, db_manager: DatabaseManager) -> None:
        """Migration 002 rollback: Drop performance indexes."""
        if not db_manager.analytics:
            return
        
        indexes = [
            "DROP INDEX IF EXISTS idx_note_metrics_word_count",
            "DROP INDEX IF EXISTS idx_note_metrics_link_count", 
            "DROP INDEX IF EXISTS idx_usage_events_note_id",
            "DROP INDEX IF EXISTS idx_search_metrics_query",
            "DROP INDEX IF EXISTS idx_performance_operation",
        ]
        
        for index_sql in indexes:
            try:
                await db_manager.analytics._execute_query(index_sql)
            except Exception as e:
                logger.warning("Failed to drop index", error=str(e))
    
    async def _migration_003_up(self, db_manager: DatabaseManager) -> None:
        """Migration 003: Vector database optimization."""
        if not db_manager.vector:
            return
        
        # Add additional payload indexes for better filtering
        try:
            await db_manager.vector.client.create_payload_index(
                db_manager.vector.collection_name,
                "word_count",
                "integer"
            )
            await db_manager.vector.client.create_payload_index(
                db_manager.vector.collection_name,
                "last_modified",
                "datetime"
            )
        except Exception as e:
            logger.warning("Failed to create vector payload indexes", error=str(e))
    
    async def _migration_003_down(self, db_manager: DatabaseManager) -> None:
        """Migration 003 rollback: Remove vector optimization."""
        if not db_manager.vector:
            return
        
        # Qdrant doesn't provide easy index removal, so we'll skip this
        logger.info("Vector optimization rollback skipped (indexes will remain)")


class DatabaseBackup:
    """Database backup and restore utilities."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def backup(self, backup_path: Path) -> None:
        """Create a full database backup."""
        backup_path.mkdir(parents=True, exist_ok=True)
        logger.info("Starting database backup", path=backup_path)
        
        # Backup analytics database
        if self.db_manager.analytics:
            analytics_backup = backup_path / "analytics"
            await self._backup_analytics(analytics_backup)
        
        # Backup vector database
        if self.db_manager.vector:
            vector_backup = backup_path / "vector"
            await self.db_manager.vector.backup_collection(vector_backup)
        
        # Create backup manifest
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "version": await self._get_backup_version(),
            "components": {
                "analytics": self.db_manager.analytics is not None,
                "vector": self.db_manager.vector is not None,
                "cache": self.db_manager.cache is not None,
            }
        }
        
        with open(backup_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info("Database backup completed", path=backup_path)
    
    async def _backup_analytics(self, backup_path: Path) -> None:
        """Backup analytics database."""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Export all tables to Parquet format for better compression
        await self.db_manager.analytics.export_data(backup_path, format="parquet")
    
    async def _get_backup_version(self) -> Optional[str]:
        """Get current migration version for backup manifest."""
        try:
            migration_manager = MigrationManager(self.db_manager)
            return await migration_manager.get_current_version()
        except Exception:
            return None
    
    async def restore(self, backup_path: Path) -> None:
        """Restore database from backup."""
        logger.info("Starting database restore", path=backup_path)
        
        # Check backup manifest
        manifest_path = backup_path / "manifest.json"
        if not manifest_path.exists():
            raise MigrationError("Backup manifest not found")
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        logger.info("Restoring from backup", 
                   timestamp=manifest.get("timestamp"),
                   version=manifest.get("version"))
        
        # TODO: Implement restore logic
        # This would involve:
        # 1. Stopping all database connections
        # 2. Clearing existing data
        # 3. Importing backup data
        # 4. Running any necessary migrations
        
        logger.warning("Database restore not yet implemented")
        raise NotImplementedError("Database restore is not yet implemented")


async def setup_databases(config: DatabaseConfig) -> DatabaseManager:
    """Set up and migrate databases."""
    db_manager = DatabaseManager(config)
    
    try:
        await db_manager.initialize()
        
        # Run migrations
        migration_manager = MigrationManager(db_manager)
        await migration_manager.migrate()
        
        logger.info("Database setup completed successfully")
        return db_manager
        
    except Exception as e:
        logger.error("Database setup failed", error=str(e))
        await db_manager.close()
        raise