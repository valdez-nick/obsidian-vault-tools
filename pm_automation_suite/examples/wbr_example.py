"""
WBR Automation Example

Demonstrates complete WBR workflow automation including data extraction,
insight generation, and slide creation.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from wbr.wbr_workflow import WBRWorkflow
from wbr.wbr_data_extractor import WBRDataExtractor
from wbr.insight_generator import InsightGenerator
from wbr.slide_generator import SlideGenerator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_example_config():
    """Get example configuration for WBR automation."""
    return {
        'workflow_id': 'example_wbr',
        'state_dir': './wbr_state',
        
        # Data extraction configuration
        'data_extraction': {
            'snowflake': {
                'account': os.getenv('SNOWFLAKE_ACCOUNT', 'your_account'),
                'user': os.getenv('SNOWFLAKE_USER', 'your_user'),
                'password': os.getenv('SNOWFLAKE_PASSWORD', 'your_password'),
                'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
                'database': os.getenv('SNOWFLAKE_DATABASE', 'ANALYTICS'),
                'schema': os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC'),
                'pool_size': 5,
                'cache_ttl': 3600
            },
            'jira': {
                'server': os.getenv('JIRA_SERVER', 'https://your-domain.atlassian.net'),
                'username': os.getenv('JIRA_USERNAME', 'your@email.com'),
                'api_token': os.getenv('JIRA_API_TOKEN', 'your_token'),
                'timeout': 30
            },
            'jira_board_id': int(os.getenv('JIRA_BOARD_ID', '123'))
        },
        
        # Insight generation configuration
        'insight_generation': {
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
            'anomaly_threshold': 2.0,
            'trend_min_periods': 4,
            'correlation_threshold': 0.7,
            'confidence_threshold': 0.6
        },
        
        # Slide generation configuration
        'slide_generation': {
            'template_path': './templates/wbr_template.pptx',
            'output_path': './wbr_presentations',
            'brand': {
                'primary_color': '#1f4e79',
                'secondary_color': '#70ad47',
                'accent_color': '#ffc000',
                'warning_color': '#ff0000',
                'font_family': 'Calibri',
                'font_size': 12
            },
            'google': {
                'service_account_key': os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY'),
                'drive_folder_id': os.getenv('GOOGLE_DRIVE_FOLDER_ID')
            }
        },
        
        # Scheduler configuration
        'scheduler': {
            'timezone': 'UTC',
            'max_jobs': 10
        },
        
        # Notification configuration
        'notifications': {
            'enabled': True,
            'email_recipients': ['pm-team@company.com'],
            'slack_webhook': os.getenv('SLACK_WEBHOOK_URL'),
            'on_success': True,
            'on_failure': True,
            'include_preview': True
        },
        
        # Retry configuration
        'max_retries': 3,
        'retry_delay': 300,
        'exponential_backoff': True,
        'max_execution_history': 50
    }


async def example_manual_execution():
    """Example: Manual WBR execution."""
    print("\\n=== Example: Manual WBR Execution ===")
    
    config = get_example_config()
    workflow = WBRWorkflow(config)
    
    try:
        # Execute workflow manually
        execution = await workflow.execute_workflow(trigger_type='manual')
        
        print(f"✓ Workflow completed: {execution.execution_id}")
        print(f"  Duration: {execution.metrics.get('total_duration', 0):.1f} seconds")
        print(f"  Data Quality: {execution.metrics.get('data_quality_score', 0):.2f}")
        print(f"  Insights Generated: {execution.metrics.get('insights_count', 0)}")
        print(f"  Presentation: {execution.results.get('presentation_path', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Workflow failed: {e}")


async def example_scheduled_execution():
    """Example: Schedule weekly WBR execution."""
    print("\\n=== Example: Schedule Weekly Execution ===")
    
    config = get_example_config()
    workflow = WBRWorkflow(config)
    
    try:
        # Schedule for Monday at 9 AM UTC
        await workflow.schedule_weekly_execution(
            day_of_week=0,  # Monday
            hour=9,         # 9 AM
            timezone='UTC'
        )
        
        print("✓ Weekly WBR execution scheduled for Mondays at 9 AM UTC")
        
        # Get scheduler status
        print("\\nScheduled jobs:")
        # This would show actual scheduled jobs in a real implementation
        print("- wbr_weekly: Every Monday at 09:00 UTC")
        
    except Exception as e:
        print(f"✗ Scheduling failed: {e}")


async def example_component_testing():
    """Example: Test individual components."""
    print("\\n=== Example: Component Testing ===")
    
    config = get_example_config()
    
    # Test Data Extractor (with mock data)
    print("\\n1. Testing Data Extractor...")
    try:
        data_extractor = WBRDataExtractor(config['data_extraction'])
        
        # This would fail without real connections, so we'll show the structure
        print("   ✓ Data extractor initialized")
        print("   ✓ Metric configurations loaded:")
        for metric_type, config_data in data_extractor.metric_configs.items():
            print(f"     - {metric_type.value}: {config_data['source']}")
            
    except Exception as e:
        print(f"   ⚠ Data extractor test: {e}")
    
    # Test Insight Generator
    print("\\n2. Testing Insight Generator...")
    try:
        insight_generator = InsightGenerator(config['insight_generation'])
        
        # Test with sample data
        sample_metrics = [
            {
                'name': 'DAU/MAU Ratio',
                'value': 18.5,
                'change_percent': -15.0,
                'trend': 'down',
                'source': 'Snowflake',
                'target': 20.0,
                'alert_threshold': 15.0
            },
            {
                'name': 'Feature Adoption',
                'value': 25.0,
                'change_percent': 12.0,
                'trend': 'up',
                'source': 'Snowflake',
                'target': 30.0,
                'alert_threshold': 20.0
            }
        ]
        
        insights = await insight_generator.generate_comprehensive_insights(sample_metrics)
        
        print(f"   ✓ Generated {len(insights)} insights")
        for insight in insights[:3]:  # Show first 3
            print(f"     - [{insight.priority.value.upper()}] {insight.title}")
            
    except Exception as e:
        print(f"   ⚠ Insight generator test: {e}")
    
    # Test Slide Generator
    print("\\n3. Testing Slide Generator...")
    try:
        slide_generator = SlideGenerator(config['slide_generation'])
        
        print("   ✓ Slide generator initialized")
        print("   ✓ Brand colors configured:")
        for color_name, color_value in slide_generator.brand_colors.items():
            print(f"     - {color_name}: {color_value}")
            
        print("   ✓ Slide templates loaded:")
        for slide_type in slide_generator.slide_templates:
            preview = slide_generator.get_slide_preview(slide_type)
            print(f"     - {slide_type.value}: {preview}")
            
    except Exception as e:
        print(f"   ⚠ Slide generator test: {e}")


async def example_workflow_monitoring():
    """Example: Monitor workflow execution."""
    print("\\n=== Example: Workflow Monitoring ===")
    
    config = get_example_config()
    workflow = WBRWorkflow(config)
    
    try:
        # Get workflow metrics
        metrics = await workflow.get_workflow_metrics()
        
        if metrics:
            print("Workflow Performance Metrics:")
            print(f"  Total Executions: {metrics['total_executions']}")
            print(f"  Success Rate: {metrics['success_rate']:.1f}%")
            print(f"  Average Duration: {metrics['avg_duration']:.1f}s")
            print(f"  Average Data Quality: {metrics['avg_data_quality']:.2f}")
        else:
            print("No execution history available")
        
        # Get execution history
        history = workflow.get_execution_history(limit=5)
        if history:
            print("\\nRecent Executions:")
            for execution in history:
                status = execution.state.value
                duration = execution.metrics.get('total_duration', 0)
                print(f"  - {execution.execution_id}: {status} ({duration:.1f}s)")
        else:
            print("\\nNo execution history available")
            
    except Exception as e:
        print(f"✗ Monitoring failed: {e}")


async def example_configuration_validation():
    """Example: Validate configuration."""
    print("\\n=== Example: Configuration Validation ===")
    
    config = get_example_config()
    
    # Check required environment variables
    required_vars = [
        'SNOWFLAKE_ACCOUNT', 'SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD',
        'JIRA_SERVER', 'JIRA_USERNAME', 'JIRA_API_TOKEN'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var) or os.getenv(var).startswith('your_'):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠ Missing or placeholder environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\\n💡 Set these variables for full functionality:")
        print("   export SNOWFLAKE_ACCOUNT=your_account")
        print("   export SNOWFLAKE_USER=your_user")
        print("   # ... etc")
    else:
        print("✓ All required environment variables are set")
    
    # Validate configuration structure
    required_sections = ['data_extraction', 'insight_generation', 'slide_generation']
    for section in required_sections:
        if section in config:
            print(f"✓ {section} configuration present")
        else:
            print(f"✗ {section} configuration missing")


async def main():
    """Run all examples."""
    print("WBR Automation Suite Examples")
    print("=" * 50)
    
    # Configuration validation
    await example_configuration_validation()
    
    # Component testing
    await example_component_testing()
    
    # Workflow monitoring
    await example_workflow_monitoring()
    
    # Manual execution (commented out to avoid actual execution)
    # await example_manual_execution()
    
    # Scheduled execution example
    await example_scheduled_execution()
    
    print("\\n" + "=" * 50)
    print("Examples completed!")
    print("\\n💡 To run actual WBR generation:")
    print("   1. Set up your data source credentials")
    print("   2. Configure AI API keys")  
    print("   3. Run: await workflow.execute_workflow()")


if __name__ == "__main__":
    asyncio.run(main())