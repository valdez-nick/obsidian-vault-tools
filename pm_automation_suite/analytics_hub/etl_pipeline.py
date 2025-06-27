"""
ETL Pipeline Implementation

Comprehensive Extract, Transform, Load pipeline for PM data analytics.
Integrates with multiple data sources, applies business logic transformations,
and loads data into analytical data warehouses.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid
from pathlib import Path

try:
    import sqlalchemy as sa
    from sqlalchemy import create_engine, MetaData, Table
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Import our data connectors
from connectors.jira_connector import JiraConnector
from connectors.snowflake_connector import SnowflakeConnector

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of data sources."""
    JIRA = "jira"
    SNOWFLAKE = "snowflake"
    DATABASE = "database"
    FILE = "file"
    API = "api"
    KAFKA = "kafka"
    S3 = "s3"


class TransformationType(Enum):
    """Types of data transformations."""
    FILTER = "filter"
    AGGREGATE = "aggregate"
    JOIN = "join"
    PIVOT = "pivot"
    NORMALIZE = "normalize"
    ENRICH = "enrich"
    VALIDATE = "validate"
    CLEAN = "clean"


class LoadStrategy(Enum):
    """Data loading strategies."""
    FULL_REFRESH = "full_refresh"
    INCREMENTAL = "incremental"
    UPSERT = "upsert"
    APPEND = "append"
    DELETE_INSERT = "delete_insert"


@dataclass
class DataSource:
    """Configuration for a data source."""
    name: str
    source_type: DataSourceType
    connection_config: Dict[str, Any]
    query: Optional[str] = None
    file_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    extraction_query: Optional[str] = None
    incremental_column: Optional[str] = None
    last_extract_value: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'source_type': self.source_type.value,
            'connection_config': self.connection_config,
            'query': self.query,
            'file_path': self.file_path,
            'api_endpoint': self.api_endpoint,
            'extraction_query': self.extraction_query,
            'incremental_column': self.incremental_column,
            'last_extract_value': self.last_extract_value
        }


@dataclass
class DataTransformation:
    """Configuration for a data transformation."""
    name: str
    transformation_type: TransformationType
    function: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    sql_query: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'transformation_type': self.transformation_type.value,
            'parameters': self.parameters,
            'sql_query': self.sql_query,
            'dependencies': self.dependencies
        }


@dataclass
class DataTarget:
    """Configuration for a data target."""
    name: str
    target_type: DataSourceType
    connection_config: Dict[str, Any]
    table_name: str
    load_strategy: LoadStrategy = LoadStrategy.FULL_REFRESH
    create_table_sql: Optional[str] = None
    pre_load_sql: Optional[str] = None
    post_load_sql: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'target_type': self.target_type.value,
            'connection_config': self.connection_config,
            'table_name': self.table_name,
            'load_strategy': self.load_strategy.value,
            'create_table_sql': self.create_table_sql,
            'pre_load_sql': self.pre_load_sql,
            'post_load_sql': self.post_load_sql
        }


@dataclass
class ETLMetrics:
    """Metrics for ETL pipeline execution."""
    pipeline_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    records_extracted: int = 0
    records_transformed: int = 0
    records_loaded: int = 0
    errors_count: int = 0
    warnings_count: int = 0
    stage_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def calculate_duration(self):
        """Calculate pipeline duration."""
        if self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pipeline_id': self.pipeline_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'records_extracted': self.records_extracted,
            'records_transformed': self.records_transformed,
            'records_loaded': self.records_loaded,
            'errors_count': self.errors_count,
            'warnings_count': self.warnings_count,
            'stage_metrics': self.stage_metrics
        }


class ETLPipeline:
    """
    Comprehensive ETL Pipeline for PM data analytics.
    
    Supports multiple data sources, complex transformations,
    and various loading strategies with monitoring and error handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ETL Pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.pipeline_id = config.get('pipeline_id', f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Initialize data sources, transformations, and targets
        self.data_sources = self._load_data_sources(config.get('data_sources', []))
        self.transformations = self._load_transformations(config.get('transformations', []))
        self.data_targets = self._load_data_targets(config.get('data_targets', []))
        
        # Pipeline settings
        self.parallel_execution = config.get('parallel_execution', True)
        self.error_handling_strategy = config.get('error_handling_strategy', 'continue')
        self.max_retries = config.get('max_retries', 3)
        self.batch_size = config.get('batch_size', 10000)
        
        # Initialize connectors
        self.connectors = {}
        self._initialize_connectors()
        
        # Caching and state management
        self.cache_enabled = config.get('cache_enabled', True)
        self.state_store = None
        if REDIS_AVAILABLE and config.get('redis_config'):
            self.state_store = redis.Redis(**config['redis_config'])
        
        # Execution state
        self.current_execution_id = None
        self.execution_metrics = None
        
    def _load_data_sources(self, sources_config: List[Dict[str, Any]]) -> List[DataSource]:
        """Load data source configurations."""
        sources = []
        for source_config in sources_config:
            source = DataSource(
                name=source_config['name'],
                source_type=DataSourceType(source_config['source_type']),
                connection_config=source_config['connection_config'],
                query=source_config.get('query'),
                file_path=source_config.get('file_path'),
                api_endpoint=source_config.get('api_endpoint'),
                extraction_query=source_config.get('extraction_query'),
                incremental_column=source_config.get('incremental_column'),
                last_extract_value=source_config.get('last_extract_value')
            )
            sources.append(source)
        return sources
    
    def _load_transformations(self, transformations_config: List[Dict[str, Any]]) -> List[DataTransformation]:
        """Load transformation configurations."""
        transformations = []
        for transform_config in transformations_config:
            transformation = DataTransformation(
                name=transform_config['name'],
                transformation_type=TransformationType(transform_config['transformation_type']),
                parameters=transform_config.get('parameters', {}),
                sql_query=transform_config.get('sql_query'),
                dependencies=transform_config.get('dependencies', [])
            )
            
            # Load custom transformation function if specified
            if 'function_module' in transform_config:
                try:
                    module = __import__(transform_config['function_module'], fromlist=[transform_config['function_name']])
                    transformation.function = getattr(module, transform_config['function_name'])
                except ImportError as e:
                    logger.warning(f"Could not load custom transformation function: {e}")
            
            transformations.append(transformation)
        return transformations
    
    def _load_data_targets(self, targets_config: List[Dict[str, Any]]) -> List[DataTarget]:
        """Load data target configurations."""
        targets = []
        for target_config in targets_config:
            target = DataTarget(
                name=target_config['name'],
                target_type=DataSourceType(target_config['target_type']),
                connection_config=target_config['connection_config'],
                table_name=target_config['table_name'],
                load_strategy=LoadStrategy(target_config.get('load_strategy', 'full_refresh')),
                create_table_sql=target_config.get('create_table_sql'),
                pre_load_sql=target_config.get('pre_load_sql'),
                post_load_sql=target_config.get('post_load_sql')
            )
            targets.append(target)
        return targets
    
    def _initialize_connectors(self):
        """Initialize data source connectors."""
        for source in self.data_sources:
            if source.source_type == DataSourceType.JIRA:
                try:
                    self.connectors[source.name] = JiraConnector(source.connection_config)
                except Exception as e:
                    logger.warning(f"Could not initialize Jira connector for {source.name}: {e}")
            
            elif source.source_type == DataSourceType.SNOWFLAKE:
                try:
                    self.connectors[source.name] = SnowflakeConnector(source.connection_config)
                except Exception as e:
                    logger.warning(f"Could not initialize Snowflake connector for {source.name}: {e}")
            
            elif source.source_type == DataSourceType.DATABASE and SQLALCHEMY_AVAILABLE:
                try:
                    engine = create_engine(source.connection_config['connection_string'])
                    self.connectors[source.name] = engine
                except Exception as e:
                    logger.warning(f"Could not initialize database connector for {source.name}: {e}")
    
    async def execute_pipeline(self) -> ETLMetrics:
        """
        Execute the complete ETL pipeline.
        
        Returns:
            Execution metrics
        """
        self.current_execution_id = str(uuid.uuid4())
        self.execution_metrics = ETLMetrics(
            pipeline_id=self.pipeline_id,
            start_time=datetime.now()
        )
        
        logger.info(f"Starting ETL pipeline execution: {self.current_execution_id}")
        
        try:
            # Extract data from all sources
            extracted_data = await self._extract_phase()
            
            # Transform data
            transformed_data = await self._transform_phase(extracted_data)
            
            # Load data to targets
            await self._load_phase(transformed_data)
            
            # Finalize metrics
            self.execution_metrics.end_time = datetime.now()
            self.execution_metrics.calculate_duration()
            
            logger.info(f"ETL pipeline completed successfully in {self.execution_metrics.duration_seconds:.2f}s")
            
            # Save execution state
            if self.state_store:
                await self._save_execution_state()
            
            return self.execution_metrics
            
        except Exception as e:
            self.execution_metrics.end_time = datetime.now()
            self.execution_metrics.calculate_duration()
            self.execution_metrics.errors_count += 1
            
            logger.error(f"ETL pipeline failed: {e}")
            raise
    
    async def _extract_phase(self) -> Dict[str, pd.DataFrame]:
        """Extract data from all configured sources."""
        logger.info("Starting extract phase")
        extracted_data = {}
        
        # Extract from sources in parallel if enabled
        if self.parallel_execution:
            extraction_tasks = []
            for source in self.data_sources:
                task = asyncio.create_task(self._extract_from_source(source))
                extraction_tasks.append((source.name, task))
            
            for source_name, task in extraction_tasks:
                try:
                    data = await task
                    extracted_data[source_name] = data
                    self.execution_metrics.records_extracted += len(data)
                except Exception as e:
                    logger.error(f"Failed to extract from {source_name}: {e}")
                    self.execution_metrics.errors_count += 1
                    if self.error_handling_strategy == 'stop':
                        raise
        else:
            # Sequential extraction
            for source in self.data_sources:
                try:
                    data = await self._extract_from_source(source)
                    extracted_data[source.name] = data
                    self.execution_metrics.records_extracted += len(data)
                except Exception as e:
                    logger.error(f"Failed to extract from {source.name}: {e}")
                    self.execution_metrics.errors_count += 1
                    if self.error_handling_strategy == 'stop':
                        raise
        
        logger.info(f"Extract phase completed: {self.execution_metrics.records_extracted} records extracted")
        return extracted_data
    
    async def _extract_from_source(self, source: DataSource) -> pd.DataFrame:
        """Extract data from a single source."""
        logger.info(f"Extracting from source: {source.name}")
        
        start_time = datetime.now()
        
        try:
            if source.source_type == DataSourceType.JIRA:
                data = await self._extract_from_jira(source)
            elif source.source_type == DataSourceType.SNOWFLAKE:
                data = await self._extract_from_snowflake(source)
            elif source.source_type == DataSourceType.DATABASE:
                data = await self._extract_from_database(source)
            elif source.source_type == DataSourceType.FILE:
                data = await self._extract_from_file(source)
            else:
                raise ValueError(f"Unsupported source type: {source.source_type}")
            
            # Record extraction metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.execution_metrics.stage_metrics[f"extract_{source.name}"] = {
                'duration_seconds': duration,
                'records_count': len(data),
                'status': 'success'
            }
            
            logger.info(f"Extracted {len(data)} records from {source.name} in {duration:.2f}s")
            return data
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.execution_metrics.stage_metrics[f"extract_{source.name}"] = {
                'duration_seconds': duration,
                'records_count': 0,
                'status': 'error',
                'error': str(e)
            }
            raise
    
    async def _extract_from_jira(self, source: DataSource) -> pd.DataFrame:
        """Extract data from Jira."""
        if source.name not in self.connectors:
            raise ValueError(f"No Jira connector available for {source.name}")
        
        connector = self.connectors[source.name]
        
        if source.query:
            # Use JQL query
            issues = await connector.search_issues(source.query)
            return pd.DataFrame([issue.to_dict() for issue in issues])
        else:
            # Extract default project data
            projects = await connector.get_projects()
            return pd.DataFrame([project.to_dict() for project in projects])
    
    async def _extract_from_snowflake(self, source: DataSource) -> pd.DataFrame:
        """Extract data from Snowflake."""
        if source.name not in self.connectors:
            raise ValueError(f"No Snowflake connector available for {source.name}")
        
        connector = self.connectors[source.name]
        
        if source.extraction_query:
            query = source.extraction_query
            
            # Handle incremental extraction
            if source.incremental_column and source.last_extract_value:
                query += f" WHERE {source.incremental_column} > '{source.last_extract_value}'"
            
            result = await connector.execute_query(query)
            return pd.DataFrame(result)
        else:
            raise ValueError(f"No extraction query specified for Snowflake source: {source.name}")
    
    async def _extract_from_database(self, source: DataSource) -> pd.DataFrame:
        """Extract data from SQL database."""
        if source.name not in self.connectors:
            raise ValueError(f"No database connector available for {source.name}")
        
        engine = self.connectors[source.name]
        
        query = source.extraction_query or source.query
        if not query:
            raise ValueError(f"No query specified for database source: {source.name}")
        
        # Handle incremental extraction
        if source.incremental_column and source.last_extract_value:
            query += f" WHERE {source.incremental_column} > '{source.last_extract_value}'"
        
        return pd.read_sql(query, engine)
    
    async def _extract_from_file(self, source: DataSource) -> pd.DataFrame:
        """Extract data from file."""
        if not source.file_path:
            raise ValueError(f"No file path specified for file source: {source.name}")
        
        file_path = Path(source.file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {source.file_path}")
        
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.json':
            return pd.read_json(file_path)
        elif file_path.suffix.lower() == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    async def _transform_phase(self, extracted_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply all transformations to extracted data."""
        logger.info("Starting transform phase")
        
        transformed_data = extracted_data.copy()
        
        # Sort transformations by dependencies
        sorted_transformations = self._sort_transformations_by_dependencies()
        
        for transformation in sorted_transformations:
            start_time = datetime.now()
            
            try:
                transformed_data = await self._apply_transformation(transformation, transformed_data)
                
                # Record transformation metrics
                duration = (datetime.now() - start_time).total_seconds()
                total_records = sum(len(df) for df in transformed_data.values())
                
                self.execution_metrics.stage_metrics[f"transform_{transformation.name}"] = {
                    'duration_seconds': duration,
                    'records_count': total_records,
                    'status': 'success'
                }
                
                logger.info(f"Applied transformation {transformation.name} in {duration:.2f}s")
                
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                self.execution_metrics.stage_metrics[f"transform_{transformation.name}"] = {
                    'duration_seconds': duration,
                    'status': 'error',
                    'error': str(e)
                }
                
                logger.error(f"Transformation {transformation.name} failed: {e}")
                self.execution_metrics.errors_count += 1
                
                if self.error_handling_strategy == 'stop':
                    raise
        
        # Update total transformed records
        self.execution_metrics.records_transformed = sum(len(df) for df in transformed_data.values())
        
        logger.info(f"Transform phase completed: {self.execution_metrics.records_transformed} records")
        return transformed_data
    
    def _sort_transformations_by_dependencies(self) -> List[DataTransformation]:
        """Sort transformations based on their dependencies."""
        sorted_transformations = []
        remaining = self.transformations.copy()
        
        while remaining:
            # Find transformations with no unmet dependencies
            ready = []
            for transformation in remaining:
                dependencies_met = all(
                    dep in [t.name for t in sorted_transformations]
                    for dep in transformation.dependencies
                )
                if dependencies_met:
                    ready.append(transformation)
            
            if not ready:
                # Circular dependency or missing dependency
                logger.warning("Circular or missing dependencies detected in transformations")
                ready = remaining  # Process remaining transformations anyway
            
            sorted_transformations.extend(ready)
            for transformation in ready:
                remaining.remove(transformation)
        
        return sorted_transformations
    
    async def _apply_transformation(
        self, 
        transformation: DataTransformation, 
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Apply a single transformation to the data."""
        
        if transformation.transformation_type == TransformationType.FILTER:
            return self._apply_filter_transformation(transformation, data)
        elif transformation.transformation_type == TransformationType.AGGREGATE:
            return self._apply_aggregate_transformation(transformation, data)
        elif transformation.transformation_type == TransformationType.JOIN:
            return self._apply_join_transformation(transformation, data)
        elif transformation.transformation_type == TransformationType.NORMALIZE:
            return self._apply_normalize_transformation(transformation, data)
        elif transformation.transformation_type == TransformationType.CLEAN:
            return self._apply_clean_transformation(transformation, data)
        elif transformation.function:
            # Custom transformation function
            return transformation.function(data, transformation.parameters)
        else:
            logger.warning(f"Unsupported transformation type: {transformation.transformation_type}")
            return data
    
    def _apply_filter_transformation(self, transformation: DataTransformation, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply filter transformation."""
        params = transformation.parameters
        source_table = params.get('source_table')
        filter_condition = params.get('filter_condition')
        
        if source_table in data and filter_condition:
            data[source_table] = data[source_table].query(filter_condition)
        
        return data
    
    def _apply_aggregate_transformation(self, transformation: DataTransformation, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply aggregation transformation."""
        params = transformation.parameters
        source_table = params.get('source_table')
        group_by = params.get('group_by', [])
        aggregations = params.get('aggregations', {})
        target_table = params.get('target_table', f"{source_table}_agg")
        
        if source_table in data:
            df = data[source_table]
            if group_by:
                aggregated = df.groupby(group_by).agg(aggregations).reset_index()
            else:
                aggregated = df.agg(aggregations).to_frame().T
            
            data[target_table] = aggregated
        
        return data
    
    def _apply_join_transformation(self, transformation: DataTransformation, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply join transformation."""
        params = transformation.parameters
        left_table = params.get('left_table')
        right_table = params.get('right_table')
        join_keys = params.get('join_keys', [])
        join_type = params.get('join_type', 'inner')
        target_table = params.get('target_table', f"{left_table}_{right_table}_joined")
        
        if left_table in data and right_table in data:
            left_df = data[left_table]
            right_df = data[right_table]
            
            joined = left_df.merge(right_df, on=join_keys, how=join_type)
            data[target_table] = joined
        
        return data
    
    def _apply_normalize_transformation(self, transformation: DataTransformation, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply normalization transformation."""
        params = transformation.parameters
        source_table = params.get('source_table')
        columns = params.get('columns', [])
        method = params.get('method', 'min_max')
        
        if source_table in data:
            df = data[source_table].copy()
            
            for column in columns:
                if column in df.columns:
                    if method == 'min_max':
                        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
                    elif method == 'z_score':
                        df[column] = (df[column] - df[column].mean()) / df[column].std()
            
            data[source_table] = df
        
        return data
    
    def _apply_clean_transformation(self, transformation: DataTransformation, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply data cleaning transformation."""
        params = transformation.parameters
        source_table = params.get('source_table')
        
        if source_table in data:
            df = data[source_table].copy()
            
            # Remove duplicates
            if params.get('remove_duplicates', False):
                df = df.drop_duplicates()
            
            # Handle missing values
            if 'fill_na' in params:
                df = df.fillna(params['fill_na'])
            
            # Remove outliers using IQR method
            if params.get('remove_outliers', False):
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                for column in numeric_columns:
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            
            data[source_table] = df
        
        return data
    
    async def _load_phase(self, transformed_data: Dict[str, pd.DataFrame]):
        """Load data to all configured targets."""
        logger.info("Starting load phase")
        
        # Load to targets in parallel if enabled
        if self.parallel_execution:
            load_tasks = []
            for target in self.data_targets:
                task = asyncio.create_task(self._load_to_target(target, transformed_data))
                load_tasks.append((target.name, task))
            
            for target_name, task in load_tasks:
                try:
                    records_loaded = await task
                    self.execution_metrics.records_loaded += records_loaded
                except Exception as e:
                    logger.error(f"Failed to load to {target_name}: {e}")
                    self.execution_metrics.errors_count += 1
                    if self.error_handling_strategy == 'stop':
                        raise
        else:
            # Sequential loading
            for target in self.data_targets:
                try:
                    records_loaded = await self._load_to_target(target, transformed_data)
                    self.execution_metrics.records_loaded += records_loaded
                except Exception as e:
                    logger.error(f"Failed to load to {target.name}: {e}")
                    self.execution_metrics.errors_count += 1
                    if self.error_handling_strategy == 'stop':
                        raise
        
        logger.info(f"Load phase completed: {self.execution_metrics.records_loaded} records loaded")
    
    async def _load_to_target(self, target: DataTarget, data: Dict[str, pd.DataFrame]) -> int:
        """Load data to a single target."""
        logger.info(f"Loading to target: {target.name}")
        
        start_time = datetime.now()
        records_loaded = 0
        
        try:
            # Get data to load (assume table name matches data key)
            if target.table_name not in data:
                logger.warning(f"No data found for target table: {target.table_name}")
                return 0
            
            df = data[target.table_name]
            
            if target.target_type == DataSourceType.SNOWFLAKE:
                records_loaded = await self._load_to_snowflake(target, df)
            elif target.target_type == DataSourceType.DATABASE:
                records_loaded = await self._load_to_database(target, df)
            elif target.target_type == DataSourceType.FILE:
                records_loaded = await self._load_to_file(target, df)
            else:
                raise ValueError(f"Unsupported target type: {target.target_type}")
            
            # Record load metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.execution_metrics.stage_metrics[f"load_{target.name}"] = {
                'duration_seconds': duration,
                'records_count': records_loaded,
                'status': 'success'
            }
            
            logger.info(f"Loaded {records_loaded} records to {target.name} in {duration:.2f}s")
            return records_loaded
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.execution_metrics.stage_metrics[f"load_{target.name}"] = {
                'duration_seconds': duration,
                'records_count': 0,
                'status': 'error',
                'error': str(e)
            }
            raise
    
    async def _load_to_snowflake(self, target: DataTarget, df: pd.DataFrame) -> int:
        """Load data to Snowflake."""
        # Implementation would use Snowflake connector
        logger.info(f"Loading {len(df)} records to Snowflake table: {target.table_name}")
        
        # Simulate loading (replace with actual Snowflake connector call)
        await asyncio.sleep(0.1)  # Simulate load time
        return len(df)
    
    async def _load_to_database(self, target: DataTarget, df: pd.DataFrame) -> int:
        """Load data to SQL database."""
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy required for database loading")
        
        logger.info(f"Loading {len(df)} records to database table: {target.table_name}")
        
        # Use SQLAlchemy to load data
        engine = create_engine(target.connection_config['connection_string'])
        
        if target.load_strategy == LoadStrategy.FULL_REFRESH:
            df.to_sql(target.table_name, engine, if_exists='replace', index=False)
        elif target.load_strategy == LoadStrategy.APPEND:
            df.to_sql(target.table_name, engine, if_exists='append', index=False)
        else:
            # For more complex strategies, implement custom logic
            df.to_sql(target.table_name, engine, if_exists='replace', index=False)
        
        return len(df)
    
    async def _load_to_file(self, target: DataTarget, df: pd.DataFrame) -> int:
        """Load data to file."""
        file_path = Path(target.connection_config.get('file_path', f"{target.table_name}.csv"))
        
        logger.info(f"Loading {len(df)} records to file: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            df.to_csv(file_path, index=False)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df.to_excel(file_path, index=False)
        elif file_path.suffix.lower() == '.json':
            df.to_json(file_path, orient='records', indent=2)
        elif file_path.suffix.lower() == '.parquet':
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return len(df)
    
    async def _save_execution_state(self):
        """Save execution state to state store."""
        if not self.state_store:
            return
        
        state_key = f"etl_execution:{self.current_execution_id}"
        state_data = {
            'pipeline_id': self.pipeline_id,
            'execution_id': self.current_execution_id,
            'metrics': self.execution_metrics.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            self.state_store.setex(
                state_key,
                timedelta(days=7),  # Keep state for 7 days
                json.dumps(state_data)
            )
            logger.info(f"Saved execution state: {state_key}")
        except Exception as e:
            logger.warning(f"Failed to save execution state: {e}")
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get complete pipeline configuration."""
        return {
            'pipeline_id': self.pipeline_id,
            'data_sources': [source.to_dict() for source in self.data_sources],
            'transformations': [transform.to_dict() for transform in self.transformations],
            'data_targets': [target.to_dict() for target in self.data_targets],
            'parallel_execution': self.parallel_execution,
            'error_handling_strategy': self.error_handling_strategy,
            'max_retries': self.max_retries,
            'batch_size': self.batch_size
        }
    
    def validate_pipeline_config(self) -> List[str]:
        """Validate pipeline configuration."""
        errors = []
        
        if not self.data_sources:
            errors.append("No data sources configured")
        
        if not self.data_targets:
            errors.append("No data targets configured")
        
        # Validate source connections
        for source in self.data_sources:
            if source.name not in self.connectors:
                errors.append(f"No connector available for source: {source.name}")
        
        # Validate transformation dependencies
        transform_names = [t.name for t in self.transformations]
        for transformation in self.transformations:
            for dependency in transformation.dependencies:
                if dependency not in transform_names:
                    errors.append(f"Transformation {transformation.name} has missing dependency: {dependency}")
        
        return errors