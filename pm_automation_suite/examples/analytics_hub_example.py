"""
Analytics Hub Example

Demonstrates comprehensive analytics capabilities including ETL pipelines,
ML-powered predictions, dashboard generation, and real-time monitoring.
"""

import asyncio
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from typing import Dict

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Analytics Hub components
from analytics_hub.etl_pipeline import (
    ETLPipeline, DataSource, DataSourceType, DataTransformation,
    TransformationType, DataTarget, LoadStrategy
)
from analytics_hub.ml_models import (
    PMPerformancePredictor, BurnoutPredictor, ProductivityAnalyzer,
    PerformanceMetric
)
from analytics_hub.dashboard_generator import (
    DashboardGenerator, MetricCard, MetricVisualization, ChartType,
    DashboardSection, DashboardLayout, AlertEngine
)
from analytics_hub.monitoring_system import (
    MonitoringSystem, MetricDefinition, MetricType, AlertRule,
    AlertSeverity, PerformanceThreshold, AnomalyDetectionMethod
)


def generate_sample_pm_data(days: int = 90) -> pd.DataFrame:
    """Generate sample PM performance data for demonstration."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Simulate realistic PM metrics with some patterns
    data = pd.DataFrame({
        'date': dates,
        'timestamp': dates,
        
        # Workload metrics
        'tickets_assigned': np.random.poisson(8, days) + np.sin(np.arange(days) * 0.1) * 2,
        'tickets_completed': np.random.poisson(7, days),
        'story_points': np.random.choice([1, 2, 3, 5, 8, 13], days),
        'story_points_completed': np.random.poisson(20, days),
        
        # Time metrics
        'work_hours': np.random.normal(8, 1.5, days).clip(4, 12),
        'meetings_count': np.random.poisson(4, days),
        'meetings_hours': np.random.gamma(2, 1, days),
        'break_time': np.random.normal(45, 15, days).clip(0, 90),
        
        # Communication metrics
        'emails_sent': np.random.poisson(15, days),
        'emails_received': np.random.poisson(25, days),
        'response_time': np.random.exponential(120, days),  # minutes
        'pr_reviews': np.random.poisson(3, days),
        
        # Quality metrics
        'bugs_reported': np.random.poisson(2, days),
        'bugs_fixed': np.random.poisson(2, days),
        'code_quality_score': np.random.beta(8, 2, days) * 100,
        
        # Team metrics
        'team_size': np.full(days, 5),
        'team_velocity': np.random.normal(35, 5, days),
        'team_satisfaction': np.random.beta(7, 3, days) * 10,
        
        # Deadline and delivery
        'on_time_delivery': np.random.choice([0, 1], days, p=[0.15, 0.85]),
        'overdue_tasks': np.random.poisson(1, days),
        'urgent_tasks': np.random.poisson(2, days),
        'total_tasks': np.random.poisson(10, days),
        
        # Weekend and overtime indicators
        'weekend_work_hours': np.where(
            pd.to_datetime(dates).dayofweek >= 5,
            np.random.exponential(2, days),
            0
        ),
        'days_since_vacation': np.minimum(np.arange(days), 120),
        
        # Interruptions and context switching
        'task_switches': np.random.poisson(8, days),
        'interruptions': np.random.poisson(5, days)
    })
    
    # Add some trends and seasonality
    data['velocity'] = data['story_points_completed'] + np.sin(np.arange(days) * 0.05) * 5
    data['productivity_score'] = (
        (data['tickets_completed'] / (data['tickets_assigned'] + 1)) * 50 +
        (1 - data['bugs_reported'] / (data['tickets_completed'] + 1)) * 30 +
        (data['pr_reviews'] / 5) * 20
    ).clip(0, 100)
    
    return data


async def run_etl_pipeline_example():
    """Demonstrate ETL pipeline for PM analytics."""
    print("\n" + "=" * 60)
    print("ETL Pipeline Example")
    print("=" * 60)
    
    # Generate sample data
    pm_data = generate_sample_pm_data(90)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        pm_data.to_csv(f.name, index=False)
        input_file = f.name
    
    try:
        # Configure ETL pipeline
        etl_config = {
            'pipeline_id': 'pm_analytics_etl',
            'data_sources': [
                {
                    'name': 'pm_metrics',
                    'source_type': 'file',
                    'connection_config': {},
                    'file_path': input_file
                }
            ],
            'transformations': [
                {
                    'name': 'calculate_burnout_indicators',
                    'transformation_type': 'clean',
                    'parameters': {}
                },
                {
                    'name': 'aggregate_weekly_metrics',
                    'transformation_type': 'aggregate',
                    'parameters': {
                        'source_table': 'pm_metrics',
                        'group_by': ['week'],
                        'aggregations': {
                            'velocity': 'mean',
                            'tickets_completed': 'sum',
                            'work_hours': 'mean',
                            'productivity_score': 'mean'
                        },
                        'target_table': 'weekly_metrics'
                    }
                },
                {
                    'name': 'clean_outliers',
                    'transformation_type': 'clean',
                    'parameters': {
                        'source_table': 'pm_metrics',
                        'remove_outliers': True
                    }
                }
            ],
            'data_targets': [
                {
                    'name': 'processed_metrics',
                    'target_type': 'file',
                    'connection_config': {'file_path': 'processed_pm_metrics.csv'},
                    'table_name': 'pm_metrics',
                    'load_strategy': 'full_refresh'
                },
                {
                    'name': 'weekly_aggregates',
                    'target_type': 'file',
                    'connection_config': {'file_path': 'weekly_pm_metrics.csv'},
                    'table_name': 'weekly_metrics',
                    'load_strategy': 'full_refresh'
                }
            ],
            'parallel_execution': True,
            'error_handling_strategy': 'continue'
        }
        
        # Create pipeline
        pipeline = ETLPipeline(etl_config)
        
        # Add custom transformation function
        def calculate_burnout_indicators(data: Dict[str, pd.DataFrame], params: Dict) -> Dict[str, pd.DataFrame]:
            """Calculate burnout risk indicators."""
            df = data['pm_metrics'].copy()
            
            # Burnout risk factors
            df['excessive_work_hours'] = (df['work_hours'] > 10).astype(int)
            df['high_meeting_load'] = (df['meetings_hours'] > 5).astype(int)
            df['poor_work_life_balance'] = (df['weekend_work_hours'] > 2).astype(int)
            df['high_stress_indicators'] = (
                df['excessive_work_hours'] + 
                df['high_meeting_load'] + 
                df['poor_work_life_balance']
            ) / 3
            
            # Add week column for aggregation
            df['week'] = pd.to_datetime(df['date']).dt.isocalendar().week
            
            data['pm_metrics'] = df
            return data
        
        pipeline.transformations[0].function = calculate_burnout_indicators
        
        print("\n📊 Executing ETL Pipeline...")
        
        # Execute pipeline
        metrics = await pipeline.execute_pipeline()
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"   - Records extracted: {metrics.records_extracted}")
        print(f"   - Records transformed: {metrics.records_transformed}")
        print(f"   - Records loaded: {metrics.records_loaded}")
        print(f"   - Duration: {metrics.duration_seconds:.2f} seconds")
        
        # Show sample of processed data
        if os.path.exists('processed_pm_metrics.csv'):
            processed_df = pd.read_csv('processed_pm_metrics.csv')
            print(f"\n📈 Sample processed data:")
            print(f"Columns: {list(processed_df.columns)[:5]}...")
            cols_to_show = [col for col in ['date', 'velocity', 'productivity_score', 'high_stress_indicators'] if col in processed_df.columns]
            if cols_to_show:
                print(processed_df[cols_to_show].head())
        
        return processed_df
        
    finally:
        # Cleanup
        os.unlink(input_file)


def run_ml_predictions_example(data: pd.DataFrame):
    """Demonstrate ML model predictions for PM performance."""
    print("\n" + "=" * 60)
    print("Machine Learning Predictions Example")
    print("=" * 60)
    
    # 1. PM Performance Prediction
    print("\n1️⃣ PM Performance Predictor")
    print("-" * 30)
    
    performance_predictor = PMPerformancePredictor({
        'metric': 'velocity',
        'use_ensemble': True
    })
    
    # Note: In real usage, you would train the model first
    print("   Model configuration:")
    print(f"   - Metric: {performance_predictor.metric_to_predict.value}")
    print(f"   - Ensemble: {performance_predictor.use_ensemble}")
    
    # Engineer features
    features = performance_predictor.engineer_features(data)
    print(f"\n   Engineered {len(features.columns)} features")
    print(f"   Key features: {list(features.columns[:5])}")
    
    # 2. Burnout Risk Prediction
    print("\n2️⃣ Burnout Risk Predictor")
    print("-" * 30)
    
    burnout_predictor = BurnoutPredictor({
        'sequence_length': 30,
        'use_lstm': False,  # Disable for demo without TensorFlow
        'burnout_threshold': 0.7
    })
    
    # Extract burnout features
    burnout_features = burnout_predictor.extract_burnout_features(data)
    
    # Simulate prediction (without training for demo)
    current_stress = burnout_features['stress_score'].iloc[-1]
    burnout_risk = "HIGH" if current_stress > 0.7 else "MEDIUM" if current_stress > 0.4 else "LOW"
    
    print(f"   Current burnout risk: {burnout_risk}")
    print(f"   Stress score: {current_stress:.2f}")
    
    # Risk factors
    risk_factors = []
    if burnout_features['excessive_hours'].iloc[-1] > 0:
        risk_factors.append("Excessive work hours")
    if burnout_features['meeting_overload'].iloc[-1] > 0:
        risk_factors.append("Too many meetings")
    if burnout_features.get('vacation_deficit', pd.Series([0])).iloc[-1] > 0:
        risk_factors.append("No recent vacation")
    
    if risk_factors:
        print(f"   Risk factors: {', '.join(risk_factors)}")
    
    # 3. Productivity Analysis
    print("\n3️⃣ Productivity Analyzer")
    print("-" * 30)
    
    productivity_analyzer = ProductivityAnalyzer({})
    
    # Calculate productivity scores
    productivity_scores = productivity_analyzer.calculate_productivity_score(data)
    current_productivity = productivity_scores.iloc[-1]
    avg_productivity = productivity_scores.mean()
    
    print(f"   Current productivity: {current_productivity:.1f}%")
    print(f"   Average productivity: {avg_productivity:.1f}%")
    
    # Identify patterns
    patterns = productivity_analyzer.identify_productivity_patterns(data)
    
    if 'daily_patterns' in patterns and patterns['daily_patterns'].get('peak_hours'):
        print(f"   Peak productivity hours: {patterns['daily_patterns']['peak_hours']}")
    
    # Generate recommendations
    current_metrics = data.iloc[-1].to_dict()
    recommendations = productivity_analyzer.generate_recommendations(current_metrics, patterns)
    
    if recommendations:
        print("\n   📋 Productivity Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"      {i}. {rec}")


def run_dashboard_generation_example(data: pd.DataFrame):
    """Demonstrate dashboard generation for PM analytics."""
    print("\n" + "=" * 60)
    print("Dashboard Generation Example")
    print("=" * 60)
    
    # Configure dashboard generator
    dashboard_gen = DashboardGenerator({
        'theme': 'light',
        'output_path': './analytics_dashboards'
    })
    
    # Prepare data for dashboard
    dashboard_data = {
        'summary': data[['date', 'velocity', 'productivity_score']].copy(),
        'performance': data[['date', 'velocity', 'tickets_completed', 'bugs_reported']].copy(),
        'team': pd.DataFrame({
            'team_member': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'assigned_points': [21, 34, 28, 31, 26],
            'completed_points': [18, 32, 25, 29, 24],
            'productivity': [85, 94, 89, 93, 92]
        }),
        'risks': pd.DataFrame({
            'risk_factor': ['Work Hours', 'Meeting Load', 'Task Switching', 'Deadline Pressure', 'Break Time'],
            'current_score': [75, 60, 85, 70, 40],
            'threshold': [80, 70, 80, 80, 50],
            'burnout_risk_score': [72, 72, 72, 72, 72]  # Same value for all rows
        })
    }
    
    # Add required columns for executive dashboard
    dashboard_data['summary']['on_time_rate'] = data['on_time_delivery'].rolling(7).mean()
    dashboard_data['summary']['quality_score'] = 100 - (data['bugs_reported'] / (data['tickets_completed'] + 1) * 100)
    dashboard_data['summary']['team_health'] = data['team_satisfaction']
    
    dashboard_data['performance']['velocity_target'] = 35
    dashboard_data['performance']['remaining_work'] = 100 - (data.index / len(data) * 100)
    dashboard_data['performance']['ideal_burndown'] = dashboard_data['performance']['remaining_work']
    
    print("\n🎨 Generating Executive Dashboard...")
    
    # Generate dashboard
    dashboard_path = dashboard_gen.create_executive_dashboard(dashboard_data)
    
    print(f"\n✅ Dashboard generated: {dashboard_path}")
    
    # Create custom dashboard section
    print("\n🎯 Creating Custom Analytics Section...")
    
    # Metric cards
    recent_data = data.tail(7)
    metric_cards = [
        MetricCard(
            title="Weekly Velocity",
            value=recent_data['velocity'].mean().round(1),
            previous_value=data.tail(14).head(7)['velocity'].mean().round(1),
            unit=" pts",
            sparkline_data=data.tail(30)['velocity'].tolist()
        ),
        MetricCard(
            title="Bug Rate",
            value=(recent_data['bugs_reported'].sum() / recent_data['tickets_completed'].sum() * 100).round(1),
            unit="%",
            change_type="negative"
        ),
        MetricCard(
            title="Team Satisfaction",
            value=recent_data['team_satisfaction'].mean().round(1),
            unit="/10",
            sparkline_data=data.tail(30)['team_satisfaction'].tolist()
        )
    ]
    
    # Visualizations
    visualizations = [
        MetricVisualization(
            name="productivity_trend",
            chart_type=ChartType.LINE,
            data=data.tail(30),
            x_column='date',
            y_columns=['productivity_score'],
            title="30-Day Productivity Trend"
        ),
        MetricVisualization(
            name="workload_heatmap",
            chart_type=ChartType.HEATMAP,
            data=data[['work_hours', 'meetings_hours', 'tickets_completed', 'bugs_reported']].tail(30),
            title="Workload Correlation Matrix"
        )
    ]
    
    custom_section = DashboardSection(
        name="analytics",
        title="Advanced Analytics",
        metric_cards=metric_cards,
        visualizations=visualizations,
        description="Deep dive into PM performance metrics"
    )
    
    # Generate dashboard with custom section
    custom_dashboard_path = dashboard_gen.generate_dashboard(
        [custom_section],
        "PM Analytics Dashboard"
    )
    
    print(f"✅ Custom dashboard generated: {custom_dashboard_path}")
    
    # Set up alerts
    print("\n🚨 Configuring Alert Engine...")
    
    alert_engine = AlertEngine({
        'notification_channels': [
            {'type': 'email', 'recipients': ['pm-team@example.com']},
            {'type': 'slack', 'webhook_url': 'https://hooks.slack.com/example'}
        ]
    })
    
    # Add alert rules
    alert_engine.add_alert_rule(
        metric='velocity',
        condition='<',
        threshold=25,
        severity='warning',
        message_template='Team velocity dropped below {threshold} (current: {value})'
    )
    
    alert_engine.add_alert_rule(
        metric='burnout_risk_score',
        condition='>',
        threshold=70,
        severity='critical',
        message_template='High burnout risk detected: {value}%'
    )
    
    # Check current metrics
    current_metrics = {
        'velocity': data['velocity'].iloc[-1],
        'burnout_risk_score': 72,  # Simulated
        'bug_rate': (data['bugs_reported'].iloc[-1] / (data['tickets_completed'].iloc[-1] + 1)) * 100
    }
    
    alerts = alert_engine.check_alerts(current_metrics)
    
    if alerts:
        print(f"\n⚠️  {len(alerts)} alerts triggered:")
        for alert in alerts:
            print(f"   - [{alert['severity'].upper()}] {alert['message']}")
    else:
        print("\n✅ No alerts triggered")


async def run_monitoring_system_example(data: pd.DataFrame):
    """Demonstrate real-time monitoring system."""
    print("\n" + "=" * 60)
    print("Real-time Monitoring System Example")
    print("=" * 60)
    
    # Configure monitoring system
    monitoring = MonitoringSystem({
        'retention_period': 3600,  # 1 hour
        'check_interval': 5,  # 5 seconds
        'enable_prometheus': False,  # For demo
        'collect_system_metrics': False  # For demo
    })
    
    print("\n📊 Setting up PM performance monitoring...")
    
    # Create default PM monitoring configuration
    monitoring.create_default_pm_monitoring()
    
    # Add custom metrics
    monitoring.define_metric(MetricDefinition(
        name="pm_focus_time",
        metric_type=MetricType.GAUGE,
        description="Uninterrupted focus time in hours",
        unit="hours"
    ))
    
    monitoring.define_metric(MetricDefinition(
        name="pm_context_switches",
        metric_type=MetricType.COUNTER,
        description="Number of context switches",
        labels=["task_type"]
    ))
    
    # Add custom alert rules
    monitoring.add_alert_rule(AlertRule(
        name="low_focus_time",
        metric_name="pm_focus_time",
        condition="value < 2",
        severity=AlertSeverity.WARNING,
        detection_method=AnomalyDetectionMethod.THRESHOLD,
        actions=["notify_pm", "suggest_focus_blocks"]
    ))
    
    monitoring.add_alert_rule(AlertRule(
        name="velocity_anomaly",
        metric_name="pm_velocity",
        condition="value < 20",
        severity=AlertSeverity.ERROR,
        detection_method=AnomalyDetectionMethod.STATISTICAL,
        window_size=1800  # 30 minutes
    ))
    
    # Set up event handler
    events_log = []
    
    def event_handler(event):
        events_log.append(event)
        if event.severity and event.severity.value >= AlertSeverity.WARNING.value:
            print(f"\n🔔 Alert: [{event.severity.name}] {event.metric_name} = {event.value}")
    
    monitoring.add_event_handler(event_handler)
    
    print("\n🚀 Starting monitoring system...")
    
    # Start monitoring
    await monitoring.start_monitoring()
    
    # Simulate recording metrics over time
    print("\n📈 Recording PM metrics...")
    
    for i in range(5):
        # Record current metrics
        current_idx = -5 + i if i < len(data) else -1
        current_data = data.iloc[current_idx]
        
        monitoring.record_metric('pm_velocity', float(current_data['velocity']))
        monitoring.record_metric('pm_burnout_risk', float(current_data.get('high_stress_indicators', 0.5) * 100))
        monitoring.record_metric('pm_quality_score', float(100 - current_data['bugs_reported'] * 10))
        monitoring.record_metric('pm_on_time_delivery', float(current_data['on_time_delivery'] * 100))
        monitoring.record_metric('pm_meeting_hours', float(current_data['meetings_hours']))
        monitoring.record_metric('pm_focus_time', float(8 - current_data['meetings_hours'] - current_data['interruptions'] * 0.25))
        
        print(f"   Recording metrics batch {i+1}/5...")
        await asyncio.sleep(2)
    
    # Get metrics summary
    print("\n📊 Metrics Summary (last 60 minutes):")
    summary = monitoring.get_metrics_summary(window_minutes=60)
    
    for metric, stats in summary.items():
        if stats['current'] is not None:
            print(f"   {metric}:")
            print(f"      Current: {stats['current']:.2f}")
            print(f"      Range: {stats['min']:.2f} - {stats['max']:.2f}")
            print(f"      Average: {stats['avg']:.2f}")
    
    # Check alert history
    print("\n🚨 Alert History:")
    alerts = monitoring.get_alert_history(hours=1)
    
    if alerts:
        for alert in alerts[-5:]:  # Show last 5 alerts
            print(f"   [{alert.timestamp.strftime('%H:%M:%S')}] {alert.metric_name}: {alert.metadata}")
    else:
        print("   No alerts in the last hour")
    
    # Export metrics
    print("\n💾 Exporting metrics...")
    metrics_export = monitoring.export_metrics(format="json")
    
    with open('pm_metrics_export.json', 'w') as f:
        f.write(metrics_export)
    
    print("   Metrics exported to pm_metrics_export.json")
    
    # Stop monitoring
    await monitoring.stop_monitoring()
    
    print("\n✅ Monitoring system stopped")
    
    return events_log


async def main():
    """Main execution function."""
    print("PM Analytics Hub - Comprehensive Example")
    print("=" * 70)
    
    try:
        # 1. Run ETL Pipeline
        processed_data = await run_etl_pipeline_example()
        
        # 2. Run ML Predictions
        run_ml_predictions_example(processed_data)
        
        # 3. Generate Dashboards
        run_dashboard_generation_example(processed_data)
        
        # 4. Run Monitoring System
        events = await run_monitoring_system_example(processed_data)
        
        # Summary
        print("\n" + "=" * 70)
        print("Analytics Hub Example Complete!")
        print("=" * 70)
        
        print("\n📊 Generated Artifacts:")
        artifacts = [
            "processed_pm_metrics.csv",
            "weekly_pm_metrics.csv",
            "./analytics_dashboards/dashboard_*.html",
            "pm_metrics_export.json"
        ]
        
        for artifact in artifacts:
            if '*' in artifact:
                print(f"   - {artifact}")
            elif os.path.exists(artifact):
                print(f"   - {artifact} ({os.path.getsize(artifact):,} bytes)")
        
        print(f"\n📈 Monitoring Events Captured: {len(events)}")
        
        print("\n🎯 Key Insights:")
        print("   - ETL pipeline processed 90 days of PM metrics")
        print("   - ML models analyzed performance and burnout risk")
        print("   - Interactive dashboards generated for executive review")
        print("   - Real-time monitoring system configured with alerts")
        
        print("\n✨ Next Steps:")
        print("   1. Deploy monitoring system for continuous tracking")
        print("   2. Train ML models with historical data")
        print("   3. Set up automated dashboard refresh")
        print("   4. Configure alert notifications")
        print("   5. Integrate with existing PM tools")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup temporary files
        temp_files = ['processed_pm_metrics.csv', 'weekly_pm_metrics.csv', 'pm_metrics_export.json']
        for file in temp_files:
            if os.path.exists(file):
                os.unlink(file)
                

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())