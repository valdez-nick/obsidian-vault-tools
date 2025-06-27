"""
MCP Server wrapper for SnowflakeConnector

Provides Model Context Protocol (MCP) server functionality
for Claude integration with Snowflake data warehouse.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("mcp package not installed. MCP server features will be limited.")

from .snowflake_connector import SnowflakeConnector

logger = logging.getLogger(__name__)


class SnowflakeMCPServer:
    """
    MCP Server wrapper for Snowflake connector.
    
    Exposes Snowflake functionality as tools for Claude to use,
    including query execution, PM analytics, and warehouse management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MCP server with Snowflake configuration.
        
        Args:
            config: Snowflake configuration dict or None to load from env
        """
        if not MCP_AVAILABLE:
            raise ImportError("mcp package is required for MCP server functionality")
            
        self.server = Server("snowflake-connector")
        self.connector = None
        self.config = config or self._load_config_from_env()
        
        # Register handlers
        self._register_handlers()
        
    def _load_config_from_env(self) -> Dict[str, Any]:
        """Load Snowflake configuration from environment variables."""
        return {
            'account': os.getenv('SNOWFLAKE_ACCOUNT'),
            'user': os.getenv('SNOWFLAKE_USER'),
            'password': os.getenv('SNOWFLAKE_PASSWORD'),
            'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
            'database': os.getenv('SNOWFLAKE_DATABASE'),
            'schema': os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC'),
            'role': os.getenv('SNOWFLAKE_ROLE'),
            'auth_type': os.getenv('SNOWFLAKE_AUTH_TYPE', 'password'),
            'pool_size': int(os.getenv('SNOWFLAKE_POOL_SIZE', '5')),
            'cache_ttl': int(os.getenv('SNOWFLAKE_CACHE_TTL', '3600'))
        }
        
    def _register_handlers(self):
        """Register MCP server handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available Snowflake tools."""
            return [
                Tool(
                    name="snowflake_query",
                    description="Execute a SQL query on Snowflake",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "SQL query to execute"
                            },
                            "use_cache": {
                                "type": "boolean",
                                "description": "Whether to use cached results",
                                "default": True
                            },
                            "warehouse": {
                                "type": "string",
                                "description": "Warehouse to use (optional)"
                            }
                        },
                        "required": ["sql"]
                    }
                ),
                Tool(
                    name="snowflake_dau_mau",
                    description="Get DAU/MAU metrics for product analytics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "Table containing user activity"
                            },
                            "user_id_col": {
                                "type": "string",
                                "description": "Column name for user ID"
                            },
                            "timestamp_col": {
                                "type": "string",
                                "description": "Column name for timestamp"
                            },
                            "days_back": {
                                "type": "integer",
                                "description": "Number of days to analyze",
                                "default": 30
                            }
                        },
                        "required": ["table", "user_id_col", "timestamp_col"]
                    }
                ),
                Tool(
                    name="snowflake_retention",
                    description="Get user retention cohort analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "Table containing user activity"
                            },
                            "user_id_col": {
                                "type": "string",
                                "description": "Column name for user ID"
                            },
                            "timestamp_col": {
                                "type": "string",
                                "description": "Column name for timestamp"
                            },
                            "cohort_period": {
                                "type": "string",
                                "description": "Cohort period: day, week, or month",
                                "default": "week"
                            },
                            "max_periods": {
                                "type": "integer",
                                "description": "Maximum periods to track",
                                "default": 12
                            }
                        },
                        "required": ["table", "user_id_col", "timestamp_col"]
                    }
                ),
                Tool(
                    name="snowflake_feature_adoption",
                    description="Analyze feature adoption rates",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "Table containing feature usage"
                            },
                            "user_id_col": {
                                "type": "string",
                                "description": "Column name for user ID"
                            },
                            "feature_col": {
                                "type": "string",
                                "description": "Column name for feature/event"
                            },
                            "timestamp_col": {
                                "type": "string",
                                "description": "Column name for timestamp"
                            },
                            "days_back": {
                                "type": "integer",
                                "description": "Number of days to analyze",
                                "default": 30
                            }
                        },
                        "required": ["table", "user_id_col", "feature_col", "timestamp_col"]
                    }
                ),
                Tool(
                    name="snowflake_revenue",
                    description="Analyze revenue metrics and trends",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "Table containing revenue data"
                            },
                            "user_id_col": {
                                "type": "string",
                                "description": "Column name for user ID"
                            },
                            "transaction_id_col": {
                                "type": "string",
                                "description": "Column name for transaction ID"
                            },
                            "amount_col": {
                                "type": "string",
                                "description": "Column name for amount"
                            },
                            "timestamp_col": {
                                "type": "string",
                                "description": "Column name for timestamp"
                            },
                            "segment_col": {
                                "type": "string",
                                "description": "Column for segmentation",
                                "default": "'All'"
                            },
                            "period": {
                                "type": "string",
                                "description": "Analysis period",
                                "default": "month"
                            }
                        },
                        "required": ["table", "user_id_col", "transaction_id_col", 
                                   "amount_col", "timestamp_col"]
                    }
                ),
                Tool(
                    name="snowflake_funnel",
                    description="Analyze user conversion funnel",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "Table containing funnel events"
                            },
                            "user_id_col": {
                                "type": "string",
                                "description": "Column name for user ID"
                            },
                            "step_col": {
                                "type": "string",
                                "description": "Column name for funnel step"
                            },
                            "timestamp_col": {
                                "type": "string",
                                "description": "Column name for timestamp"
                            },
                            "steps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of funnel steps (max 5)"
                            },
                            "days_back": {
                                "type": "integer",
                                "description": "Number of days to analyze",
                                "default": 30
                            }
                        },
                        "required": ["table", "user_id_col", "step_col", 
                                   "timestamp_col", "steps"]
                    }
                ),
                Tool(
                    name="snowflake_optimize",
                    description="Get query optimization suggestions",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "SQL query to optimize"
                            }
                        },
                        "required": ["sql"]
                    }
                ),
                Tool(
                    name="snowflake_warehouse",
                    description="Manage Snowflake warehouse",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "warehouse": {
                                "type": "string",
                                "description": "Warehouse name"
                            },
                            "action": {
                                "type": "string",
                                "description": "Action: resume, suspend, resize",
                                "enum": ["resume", "suspend", "resize"]
                            },
                            "size": {
                                "type": "string",
                                "description": "Size for resize action",
                                "enum": ["X-SMALL", "SMALL", "MEDIUM", "LARGE", 
                                        "X-LARGE", "2X-LARGE", "3X-LARGE", "4X-LARGE"]
                            }
                        },
                        "required": ["warehouse", "action"]
                    }
                ),
                Tool(
                    name="snowflake_metadata",
                    description="Get Snowflake account metadata",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="snowflake_performance",
                    description="Get query performance statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
            
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> List[TextContent]:
            """Handle tool execution requests."""
            try:
                # Ensure connector is initialized
                if not self.connector:
                    self.connector = SnowflakeConnector(self.config)
                    if not self.connector.connect():
                        return [TextContent(
                            type="text",
                            text=f"Failed to connect to Snowflake: {self.connector.last_error}"
                        )]
                
                # Route to appropriate handler
                if name == "snowflake_query":
                    return await self._handle_query(arguments)
                elif name == "snowflake_dau_mau":
                    return await self._handle_dau_mau(arguments)
                elif name == "snowflake_retention":
                    return await self._handle_retention(arguments)
                elif name == "snowflake_feature_adoption":
                    return await self._handle_feature_adoption(arguments)
                elif name == "snowflake_revenue":
                    return await self._handle_revenue(arguments)
                elif name == "snowflake_funnel":
                    return await self._handle_funnel(arguments)
                elif name == "snowflake_optimize":
                    return await self._handle_optimize(arguments)
                elif name == "snowflake_warehouse":
                    return await self._handle_warehouse(arguments)
                elif name == "snowflake_metadata":
                    return await self._handle_metadata()
                elif name == "snowflake_performance":
                    return await self._handle_performance()
                else:
                    return [TextContent(
                        type="text",
                        text=f"Unknown tool: {name}"
                    )]
                    
            except Exception as e:
                logger.error(f"Error executing tool {name}: {str(e)}")
                return [TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]
                
    async def _handle_query(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle SQL query execution."""
        sql = arguments.get('sql')
        use_cache = arguments.get('use_cache', True)
        warehouse = arguments.get('warehouse')
        
        # Execute query in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.connector.execute_query,
            sql,
            None,
            use_cache,
            warehouse
        )
        
        # Format result
        if len(result) > 100:
            preview = result.head(100)
            text = f"Query returned {len(result)} rows (showing first 100):\n\n"
            text += preview.to_string()
        else:
            text = f"Query returned {len(result)} rows:\n\n"
            text += result.to_string()
            
        return [TextContent(type="text", text=text)]
        
    async def _handle_dau_mau(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle DAU/MAU metrics request."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.connector.get_dau_mau,
            arguments['table'],
            arguments['user_id_col'],
            arguments['timestamp_col'],
            arguments.get('days_back', 30)
        )
        
        # Format metrics
        text = "DAU/MAU Metrics:\n\n"
        text += result.to_string()
        
        # Add summary
        if not result.empty:
            latest = result.iloc[0]
            text += f"\n\nLatest metrics ({latest['activity_date']}):\n"
            text += f"- Daily Active Users: {latest['dau']:,}\n"
            text += f"- Monthly Active Users: {latest['mau']:,}\n"
            text += f"- DAU/MAU Ratio: {latest['dau_mau_ratio']:.1f}%"
            
        return [TextContent(type="text", text=text)]
        
    async def _handle_retention(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle retention analysis request."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.connector.get_retention_cohort,
            arguments['table'],
            arguments['user_id_col'],
            arguments['timestamp_col'],
            arguments.get('cohort_period', 'week'),
            arguments.get('max_periods', 12)
        )
        
        # Format retention data
        text = f"User Retention Analysis ({arguments.get('cohort_period', 'week')}ly cohorts):\n\n"
        
        # Create pivot table for better visualization
        if not result.empty:
            pivot = result.pivot(
                index='cohort_date',
                columns='periods_later',
                values='retention_rate'
            )
            text += pivot.to_string()
            
        return [TextContent(type="text", text=text)]
        
    async def _handle_feature_adoption(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle feature adoption analysis request."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.connector.get_feature_adoption,
            arguments['table'],
            arguments['user_id_col'],
            arguments['feature_col'],
            arguments['timestamp_col'],
            arguments.get('days_back', 30)
        )
        
        # Format adoption metrics
        text = f"Feature Adoption Analysis (last {arguments.get('days_back', 30)} days):\n\n"
        text += result.to_string()
        
        # Add summary of top features
        if not result.empty:
            text += f"\n\nTop 5 adopted features:\n"
            for idx, row in result.head(5).iterrows():
                text += f"- {row['feature_name']}: {row['adoption_rate']:.1f}% "
                text += f"({row['unique_users']:,} users, "
                text += f"{row['avg_events_per_user']:.1f} events/user)\n"
                
        return [TextContent(type="text", text=text)]
        
    async def _handle_revenue(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle revenue analysis request."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.connector.get_revenue_analysis,
            arguments['table'],
            arguments['user_id_col'],
            arguments['transaction_id_col'],
            arguments['amount_col'],
            arguments['timestamp_col'],
            arguments.get('segment_col', "'All'"),
            arguments.get('period', 'month'),
            arguments.get('periods_back', 12)
        )
        
        # Format revenue metrics
        text = f"Revenue Analysis by {arguments.get('period', 'month')}:\n\n"
        text += result.to_string()
        
        # Add growth summary
        if len(result) > 1:
            latest = result.iloc[0]
            text += f"\n\nLatest period metrics:\n"
            text += f"- Total Revenue: ${latest['total_revenue']:,.2f}\n"
            text += f"- Paying Users: {latest['paying_users']:,}\n"
            text += f"- ARPU: ${latest['arpu']:,.2f}\n"
            if pd.notna(latest.get('revenue_growth_pct')):
                text += f"- Revenue Growth: {latest['revenue_growth_pct']:+.1f}%\n"
                
        return [TextContent(type="text", text=text)]
        
    async def _handle_funnel(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle funnel analysis request."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.connector.get_user_funnel,
            arguments['table'],
            arguments['user_id_col'],
            arguments['step_col'],
            arguments['timestamp_col'],
            arguments['steps'],
            arguments.get('days_back', 30)
        )
        
        # Format funnel metrics
        text = f"User Funnel Analysis (last {arguments.get('days_back', 30)} days):\n\n"
        text += result.to_string()
        
        # Add conversion summary
        if len(result) > 1:
            text += f"\n\nOverall funnel conversion: "
            text += f"{result.iloc[-1]['pct_of_total']:.1f}%\n"
            text += f"Total users entered: {result.iloc[0]['users']:,}"
            
        return [TextContent(type="text", text=text)]
        
    async def _handle_optimize(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle query optimization request."""
        sql = arguments['sql']
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.connector.optimize_query,
            sql
        )
        
        # Format optimization suggestions
        text = "Query Optimization Analysis:\n\n"
        text += f"Optimization Score: {result['optimization_score']}/100\n"
        text += f"Estimated Improvement: {result['estimated_improvement']}%\n\n"
        
        if result['suggestions']:
            text += "Suggestions:\n"
            for suggestion in result['suggestions']:
                text += f"- [{suggestion['impact'].upper()}] {suggestion['suggestion']}\n"
                
        text += f"\n\nOptimized Query:\n```sql\n{result['optimized_query']}\n```"
        
        return [TextContent(type="text", text=text)]
        
    async def _handle_warehouse(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle warehouse management request."""
        warehouse = arguments['warehouse']
        action = arguments['action']
        size = arguments.get('size')
        
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            self.connector.manage_warehouse,
            warehouse,
            action,
            size
        )
        
        if success:
            text = f"Successfully performed {action} on warehouse {warehouse}"
            if action == 'resize' and size:
                text += f" to size {size}"
        else:
            text = f"Failed to {action} warehouse {warehouse}: {self.connector.last_error}"
            
        return [TextContent(type="text", text=text)]
        
    async def _handle_metadata(self) -> List[TextContent]:
        """Handle metadata request."""
        loop = asyncio.get_event_loop()
        metadata = await loop.run_in_executor(
            None,
            self.connector.get_metadata
        )
        
        # Format metadata
        text = "Snowflake Account Metadata:\n\n"
        
        # Current context
        ctx = metadata.get('current_context', {})
        text += "Current Context:\n"
        text += f"- Account: {ctx.get('account')}\n"
        text += f"- Role: {ctx.get('role')}\n"
        text += f"- Warehouse: {ctx.get('warehouse')}\n"
        text += f"- Database: {ctx.get('database')}\n"
        text += f"- Schema: {ctx.get('schema')}\n\n"
        
        # Warehouses
        if metadata.get('warehouses'):
            text += f"Warehouses ({len(metadata['warehouses'])}):\n"
            for wh in metadata['warehouses'][:5]:
                text += f"- {wh.get('name')} ({wh.get('state')}, {wh.get('size')})\n"
                
        # Databases
        if metadata.get('databases'):
            text += f"\nDatabases ({len(metadata['databases'])}):\n"
            for db in metadata['databases'][:5]:
                text += f"- {db.get('name')}\n"
                
        return [TextContent(type="text", text=text)]
        
    async def _handle_performance(self) -> List[TextContent]:
        """Handle performance stats request."""
        loop = asyncio.get_event_loop()
        
        # Get performance stats
        perf_stats = await loop.run_in_executor(
            None,
            self.connector.get_performance_stats
        )
        
        # Get cache stats
        cache_stats = await loop.run_in_executor(
            None,
            self.connector.get_cache_stats
        )
        
        # Format stats
        text = "Query Performance Statistics:\n\n"
        
        # Performance metrics
        text += f"Total Queries: {perf_stats['total_queries']}\n"
        text += f"Avg Execution Time: {perf_stats['avg_execution_time']:.2f}s\n"
        text += f"Median Execution Time: {perf_stats['median_execution_time']:.2f}s\n"
        text += f"95th Percentile: {perf_stats['p95_execution_time']:.2f}s\n"
        text += f"Success Rate: {perf_stats['success_rate']:.1f}%\n\n"
        
        # Slowest queries
        if perf_stats.get('slowest_queries'):
            text += "Slowest Queries:\n"
            for q in perf_stats['slowest_queries']:
                text += f"- {q['sql_preview'][:50]}... ({q['execution_time']:.2f}s)\n"
                
        # Cache stats
        text += f"\n\nCache Statistics:\n"
        text += f"Cached Entries: {cache_stats['entries']}\n"
        text += f"Cache Size: {cache_stats['total_size_mb']:.2f} MB\n"
        
        return [TextContent(type="text", text=text)]
        
    async def run(self):
        """Run the MCP server."""
        async with self.server.run() as running_server:
            logger.info("Snowflake MCP server started")
            try:
                # Initialize connector
                self.connector = SnowflakeConnector(self.config)
                if not self.connector.connect():
                    logger.error(f"Failed to connect to Snowflake: {self.connector.last_error}")
                else:
                    logger.info("Successfully connected to Snowflake")
                    
                # Keep server running
                await running_server
                
            finally:
                # Cleanup
                if self.connector and self.connector.is_connected:
                    self.connector.disconnect()
                    logger.info("Disconnected from Snowflake")


async def main():
    """Main entry point for MCP server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run server
    server = SnowflakeMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())