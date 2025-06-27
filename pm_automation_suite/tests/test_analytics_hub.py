"""
Test suite for Analytics Hub components.

Tests ETL pipelines, ML models, dashboard generation, and monitoring system.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import tempfile
import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Import modules under test
from analytics_hub.etl_pipeline import (
    ETLPipeline, DataSource, DataSourceType, DataTransformation,
    TransformationType, DataTarget, LoadStrategy, ETLMetrics
)
from analytics_hub.ml_models import (
    PMPerformancePredictor, BurnoutPredictor, ProductivityAnalyzer,
    PerformanceMetric, PredictionResult
)
from analytics_hub.dashboard_generator import (
    DashboardGenerator, MetricCard, MetricVisualization, ChartType,
    DashboardSection, AlertEngine
)
from analytics_hub.monitoring_system import (
    MonitoringSystem, MetricDefinition, MetricType, AlertRule,
    AlertSeverity, PerformanceThreshold, MonitoringEvent
)


class TestETLPipeline:
    """Test ETL Pipeline functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {
            'jira_data': pd.DataFrame({
                'issue_key': ['PROJ-1', 'PROJ-2', 'PROJ-3'],
                'status': ['Done', 'In Progress', 'Done'],
                'story_points': [5, 8, 3],
                'assignee': ['Alice', 'Bob', 'Alice'],
                'created_date': pd.date_range('2024-01-01', periods=3)
            }),
            'performance_data': pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=30),
                'velocity': np.random.randint(20, 40, 30),
                'bugs_reported': np.random.randint(0, 5, 30),
                'tasks_completed': np.random.randint(5, 15, 30)
            })
        }
    
    @pytest.fixture
    def etl_config(self):
        """ETL pipeline configuration for testing."""
        return {
            'pipeline_id': 'test_pipeline',
            'data_sources': [
                {
                    'name': 'test_file_source',
                    'source_type': 'file',
                    'connection_config': {},
                    'file_path': 'test_data.csv'
                }
            ],
            'transformations': [
                {
                    'name': 'filter_completed',
                    'transformation_type': 'filter',
                    'parameters': {
                        'source_table': 'jira_data',
                        'filter_condition': "status == 'Done'"
                    }
                }
            ],
            'data_targets': [
                {
                    'name': 'test_file_target',
                    'target_type': 'file',
                    'connection_config': {'file_path': 'output.csv'},
                    'table_name': 'jira_data',
                    'load_strategy': 'full_refresh'
                }
            ]
        }
    
    def test_etl_pipeline_initialization(self, etl_config):
        """Test ETL pipeline initialization."""
        pipeline = ETLPipeline(etl_config)
        
        assert pipeline is not None
        assert pipeline.pipeline_id == 'test_pipeline'
        assert len(pipeline.data_sources) == 1
        assert len(pipeline.transformations) == 1
        assert len(pipeline.data_targets) == 1
    
    @pytest.mark.asyncio
    async def test_extract_phase(self, etl_config, sample_data):
        """Test data extraction phase."""
        pipeline = ETLPipeline(etl_config)
        
        # Create test CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data['jira_data'].to_csv(f.name, index=False)
            pipeline.data_sources[0].file_path = f.name
        
        try:
            # Test extraction
            extracted_data = await pipeline._extract_phase()
            
            assert 'test_file_source' in extracted_data
            assert len(extracted_data['test_file_source']) == 3
            assert 'issue_key' in extracted_data['test_file_source'].columns
            
        finally:
            os.unlink(f.name)
    
    @pytest.mark.asyncio
    async def test_transform_phase(self, etl_config, sample_data):
        """Test data transformation phase."""
        pipeline = ETLPipeline(etl_config)
        
        # Mock extracted data
        extracted_data = {'jira_data': sample_data['jira_data']}
        
        # Test transformation
        transformed_data = await pipeline._transform_phase(extracted_data)
        
        assert 'jira_data' in transformed_data
        # Filter should keep only 'Done' status
        assert len(transformed_data['jira_data']) == 2
        assert all(transformed_data['jira_data']['status'] == 'Done')
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_execution(self, etl_config, sample_data):
        """Test complete ETL pipeline execution."""
        pipeline = ETLPipeline(etl_config)
        
        # Create test input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data['jira_data'].to_csv(f.name, index=False)
            pipeline.data_sources[0].file_path = f.name
        
        # Set output file
        output_file = tempfile.mktemp(suffix='.csv')
        pipeline.data_targets[0].connection_config['file_path'] = output_file
        
        try:
            # Execute pipeline
            metrics = await pipeline.execute_pipeline()
            
            assert isinstance(metrics, ETLMetrics)
            assert metrics.records_extracted > 0
            assert metrics.records_transformed > 0
            assert metrics.records_loaded > 0
            assert metrics.duration_seconds is not None
            
            # Check output file exists
            assert os.path.exists(output_file)
            
            # Verify output data
            output_df = pd.read_csv(output_file)
            assert len(output_df) == 2  # Filtered to only 'Done' status
            
        finally:
            # Cleanup
            os.unlink(f.name)
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_pipeline_validation(self, etl_config):
        """Test pipeline configuration validation."""
        pipeline = ETLPipeline(etl_config)
        
        errors = pipeline.validate_pipeline_config()
        
        # Should have some validation errors due to missing connectors
        assert isinstance(errors, list)


class TestMLModels:
    """Test Machine Learning models."""
    
    @pytest.fixture
    def sample_pm_data(self):
        """Create sample PM data for testing."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        return pd.DataFrame({
            'timestamp': dates,
            'tickets_assigned': np.random.randint(5, 15, 100),
            'tickets_completed': np.random.randint(3, 12, 100),
            'meetings_count': np.random.randint(2, 8, 100),
            'meetings_hours': np.random.randint(2, 6, 100),
            'work_hours': np.random.normal(8, 2, 100).clip(4, 12),
            'bugs_reported': np.random.randint(0, 3, 100),
            'story_points': np.random.randint(3, 13, 100),
            'on_time_delivery': np.random.choice([0, 1], 100, p=[0.2, 0.8]),
            'team_size': np.full(100, 5),
            'emails_sent': np.random.randint(10, 30, 100)
        })
    
    def test_pm_performance_predictor_initialization(self):
        """Test PM Performance Predictor initialization."""
        config = {
            'metric': 'velocity',
            'use_ensemble': True
        }
        
        predictor = PMPerformancePredictor(config)
        
        assert predictor is not None
        assert predictor.metric_to_predict == PerformanceMetric.VELOCITY
        assert predictor.use_ensemble is True
    
    def test_feature_engineering(self, sample_pm_data):
        """Test feature engineering for PM performance."""
        predictor = PMPerformancePredictor({'metric': 'velocity'})
        
        features = predictor.engineer_features(sample_pm_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_pm_data)
        
        # Check engineered features exist
        expected_features = [
            'tickets_assigned', 'completion_rate', 'meetings_count',
            'hour_of_day', 'day_of_week', 'is_weekend'
        ]
        
        for feature in expected_features:
            assert feature in features.columns
    
    @pytest.mark.skipif(not pytest.importorskip("sklearn"), reason="scikit-learn not available")
    def test_pm_performance_training(self, sample_pm_data):
        """Test training PM performance model."""
        predictor = PMPerformancePredictor({'metric': 'velocity'})
        
        # Create target variable
        y = sample_pm_data['tickets_completed'] * 3  # Simple velocity proxy
        
        # Split data
        train_size = 80
        X_train = sample_pm_data[:train_size]
        y_train = y[:train_size]
        X_val = sample_pm_data[train_size:]
        y_val = y[train_size:]
        
        # Train model
        predictor.train(X_train, y_train, X_val, y_val)
        
        assert predictor.is_trained is True
        assert predictor.model is not None
    
    def test_burnout_predictor_initialization(self):
        """Test Burnout Predictor initialization."""
        config = {
            'sequence_length': 30,
            'use_lstm': False,  # Disable for testing without TensorFlow
            'burnout_threshold': 0.7
        }
        
        predictor = BurnoutPredictor(config)
        
        assert predictor is not None
        assert predictor.sequence_length == 30
        assert predictor.burnout_threshold == 0.7
    
    def test_burnout_feature_extraction(self, sample_pm_data):
        """Test burnout-specific feature extraction."""
        predictor = BurnoutPredictor({'use_lstm': False})
        
        features = predictor.extract_burnout_features(sample_pm_data)
        
        assert isinstance(features, pd.DataFrame)
        assert 'daily_work_hours' in features.columns
        assert 'stress_score' in features.columns
        assert 'excessive_hours' in features.columns
    
    def test_productivity_analyzer(self, sample_pm_data):
        """Test Productivity Analyzer."""
        analyzer = ProductivityAnalyzer({})
        
        # Calculate productivity score
        scores = analyzer.calculate_productivity_score(sample_pm_data)
        
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_pm_data)
        assert all(0 <= score <= 100 for score in scores)
        
        # Identify patterns
        patterns = analyzer.identify_productivity_patterns(sample_pm_data)
        
        assert isinstance(patterns, dict)
        assert 'daily_patterns' in patterns
        assert 'correlation_insights' in patterns
        
        # Test predictions
        result = analyzer.predict(sample_pm_data)
        
        assert isinstance(result, PredictionResult)
        assert 'recommendations' in result.metadata


class TestDashboardGenerator:
    """Test Dashboard Generator functionality."""
    
    @pytest.fixture
    def dashboard_config(self):
        """Dashboard configuration for testing."""
        return {
            'theme': 'light',
            'output_path': tempfile.mkdtemp(),
            'colors': ['#1f77b4', '#ff7f0e', '#2ca02c']
        }
    
    @pytest.fixture
    def sample_metrics_data(self):
        """Sample metrics data for dashboard."""
        return {
            'summary': pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=30),
                'velocity': np.random.randint(20, 40, 30),
                'on_time_rate': np.random.uniform(0.7, 0.95, 30),
                'quality_score': np.random.randint(70, 95, 30),
                'team_health': np.random.uniform(6, 9, 30)
            }),
            'performance': pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=30),
                'velocity': np.random.randint(20, 40, 30),
                'velocity_target': np.full(30, 35),
                'remaining_work': np.linspace(100, 0, 30),
                'ideal_burndown': np.linspace(100, 0, 30)
            })
        }
    
    def test_dashboard_generator_initialization(self, dashboard_config):
        """Test Dashboard Generator initialization."""
        generator = DashboardGenerator(dashboard_config)
        
        assert generator is not None
        assert generator.theme == 'light'
        assert len(generator.default_colors) > 0
    
    def test_metric_card_creation(self, dashboard_config):
        """Test metric card HTML generation."""
        generator = DashboardGenerator(dashboard_config)
        
        card = MetricCard(
            title="Test Metric",
            value=42,
            previous_value=40,
            unit="%",
            change_type="positive"
        )
        
        html = generator.create_metric_card_html(card)
        
        assert isinstance(html, str)
        assert "Test Metric" in html
        assert "42%" in html
        assert "+5.0%" in html  # Change percentage
    
    @pytest.mark.skipif(not pytest.importorskip("plotly"), reason="Plotly not available")
    def test_visualization_creation(self, dashboard_config, sample_metrics_data):
        """Test visualization creation."""
        generator = DashboardGenerator(dashboard_config)
        
        viz = MetricVisualization(
            name="test_chart",
            chart_type=ChartType.LINE,
            data=sample_metrics_data['summary'],
            x_column='date',
            y_columns=['velocity'],
            title="Velocity Trend"
        )
        
        figure = generator.create_visualization(viz)
        
        assert figure is not None
        # Would need to check plotly figure properties
    
    def test_dashboard_generation(self, dashboard_config, sample_metrics_data):
        """Test complete dashboard generation."""
        generator = DashboardGenerator(dashboard_config)
        
        # Create dashboard sections
        sections = [
            DashboardSection(
                name="overview",
                title="Overview",
                metric_cards=[
                    MetricCard(title="Velocity", value=35, unit=" pts"),
                    MetricCard(title="Quality", value=85, unit="%")
                ]
            )
        ]
        
        # Generate dashboard
        dashboard_path = generator.generate_dashboard(sections, "Test Dashboard")
        
        assert os.path.exists(dashboard_path)
        
        # Check HTML content
        with open(dashboard_path, 'r') as f:
            html_content = f.read()
        
        assert "Test Dashboard" in html_content
        assert "Overview" in html_content
        assert "Velocity" in html_content
    
    def test_alert_engine(self):
        """Test Alert Engine functionality."""
        engine = AlertEngine({
            'notification_channels': [
                {'type': 'email', 'recipients': ['test@example.com']}
            ]
        })
        
        # Add alert rule
        engine.add_alert_rule(
            metric='velocity',
            condition='>',
            threshold=40,
            severity='warning'
        )
        
        assert len(engine.alert_rules) == 1
        
        # Check alerts
        metrics = {'velocity': 45}
        alerts = engine.check_alerts(metrics)
        
        assert len(alerts) == 1
        assert alerts[0]['metric'] == 'velocity'
        assert alerts[0]['severity'] == 'warning'


class TestMonitoringSystem:
    """Test Monitoring System functionality."""
    
    @pytest.fixture
    def monitoring_config(self):
        """Monitoring system configuration for testing."""
        return {
            'retention_period': 3600,  # 1 hour
            'check_interval': 1,  # 1 second for testing
            'enable_prometheus': False,  # Disable for testing
            'collect_system_metrics': False
        }
    
    def test_monitoring_system_initialization(self, monitoring_config):
        """Test Monitoring System initialization."""
        system = MonitoringSystem(monitoring_config)
        
        assert system is not None
        assert system.config == monitoring_config
        assert system.metric_store is not None
    
    def test_metric_definition(self, monitoring_config):
        """Test defining metrics."""
        system = MonitoringSystem(monitoring_config)
        
        metric_def = MetricDefinition(
            name="test_metric",
            metric_type=MetricType.GAUGE,
            description="Test metric",
            unit="units"
        )
        
        system.define_metric(metric_def)
        
        assert "test_metric" in system.metric_definitions
    
    def test_metric_recording(self, monitoring_config):
        """Test recording metric values."""
        system = MonitoringSystem(monitoring_config)
        
        # Define metric
        system.define_metric(MetricDefinition(
            name="test_gauge",
            metric_type=MetricType.GAUGE,
            description="Test gauge"
        ))
        
        # Record values
        system.record_metric("test_gauge", 42.0)
        system.record_metric("test_gauge", 45.0)
        
        # Check stored values
        values = system.metric_store.get_metric_values("test_gauge")
        
        assert len(values) == 2
        assert values[-1] == 45.0
    
    def test_alert_rules(self, monitoring_config):
        """Test alert rule functionality."""
        system = MonitoringSystem(monitoring_config)
        
        # Add alert rule
        rule = AlertRule(
            name="high_value_alert",
            metric_name="test_metric",
            condition="value > 100",
            severity=AlertSeverity.WARNING
        )
        
        system.add_alert_rule(rule)
        
        assert "high_value_alert" in system.alert_rules
        
        # Test rule evaluation
        assert rule.evaluate(150) is True
        assert rule.evaluate(50) is False
    
    def test_performance_thresholds(self, monitoring_config):
        """Test performance threshold functionality."""
        system = MonitoringSystem(monitoring_config)
        
        threshold = PerformanceThreshold(
            metric_name="cpu_usage",
            warning_threshold=70,
            critical_threshold=90,
            direction="above"
        )
        
        system.set_performance_threshold(threshold)
        
        # Test threshold checking
        assert threshold.check_threshold(60) is None
        assert threshold.check_threshold(75) == AlertSeverity.WARNING
        assert threshold.check_threshold(95) == AlertSeverity.CRITICAL
    
    def test_event_handling(self, monitoring_config):
        """Test event handling."""
        system = MonitoringSystem(monitoring_config)
        
        events_received = []
        
        def event_handler(event: MonitoringEvent):
            events_received.append(event)
        
        system.add_event_handler(event_handler)
        
        # Record metric to trigger event
        system.record_metric("test_metric", 100)
        
        assert len(events_received) > 0
        assert events_received[0].event_type == "metric_update"
        assert events_received[0].metric_name == "test_metric"
    
    @pytest.mark.asyncio
    async def test_monitoring_loop(self, monitoring_config):
        """Test monitoring loop execution."""
        system = MonitoringSystem(monitoring_config)
        
        # Define metric and alert rule
        system.define_metric(MetricDefinition(
            name="test_metric",
            metric_type=MetricType.GAUGE,
            description="Test metric"
        ))
        
        system.add_alert_rule(AlertRule(
            name="test_alert",
            metric_name="test_metric",
            condition="value > 50",
            severity=AlertSeverity.WARNING,
            cooldown_period=1
        ))
        
        # Start monitoring
        await system.start_monitoring()
        
        # Record metric that should trigger alert
        system.record_metric("test_metric", 60)
        
        # Wait for monitoring loop to process
        await asyncio.sleep(2)
        
        # Stop monitoring
        await system.stop_monitoring()
        
        # Check alert was triggered
        alerts = system.get_alert_history(hours=1)
        assert len(alerts) > 0
    
    def test_metrics_summary(self, monitoring_config):
        """Test metrics summary generation."""
        system = MonitoringSystem(monitoring_config)
        
        # Record some metrics
        system.record_metric("metric1", 10)
        system.record_metric("metric1", 20)
        system.record_metric("metric1", 30)
        
        summary = system.get_metrics_summary(window_minutes=60)
        
        assert "metric1" in summary
        assert summary["metric1"]["current"] == 30
        assert summary["metric1"]["min"] == 10
        assert summary["metric1"]["max"] == 30
        assert summary["metric1"]["avg"] == 20
    
    def test_default_pm_monitoring(self, monitoring_config):
        """Test default PM monitoring configuration."""
        system = MonitoringSystem(monitoring_config)
        
        system.create_default_pm_monitoring()
        
        # Check metrics were created
        assert "pm_velocity" in system.metric_definitions
        assert "pm_burnout_risk" in system.metric_definitions
        assert "pm_quality_score" in system.metric_definitions
        
        # Check thresholds were set
        assert "pm_velocity" in system.performance_thresholds
        assert "pm_burnout_risk" in system.performance_thresholds
        
        # Check alert rules were added
        assert "velocity_drop" in system.alert_rules
        assert "high_burnout_risk" in system.alert_rules


class TestIntegration:
    """Integration tests for Analytics Hub components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analytics_flow(self):
        """Test complete analytics flow from ETL to dashboard."""
        # Create test data
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'tickets_completed': np.random.randint(5, 15, 30),
            'bugs_reported': np.random.randint(0, 3, 30),
            'velocity': np.random.randint(20, 40, 30),
            'team_health': np.random.uniform(6, 9, 30)
        })
        
        # 1. ETL Pipeline
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            
            etl_config = {
                'data_sources': [{
                    'name': 'metrics',
                    'source_type': 'file',
                    'connection_config': {},
                    'file_path': f.name
                }],
                'transformations': [{
                    'name': 'calculate_quality',
                    'transformation_type': 'custom',
                    'parameters': {}
                }],
                'data_targets': [{
                    'name': 'processed',
                    'target_type': 'file',
                    'connection_config': {'file_path': tempfile.mktemp(suffix='.csv')},
                    'table_name': 'metrics',
                    'load_strategy': 'full_refresh'
                }]
            }
            
            pipeline = ETLPipeline(etl_config)
            
            # Custom transformation to add quality score
            def add_quality_score(data, params):
                data['metrics']['quality_score'] = 100 - (data['metrics']['bugs_reported'] * 10)
                return data
            
            pipeline.transformations[0].function = add_quality_score
            
            metrics = await pipeline.execute_pipeline()
            
            assert metrics.records_extracted > 0
            assert metrics.records_loaded > 0
        
        # 2. ML Models
        if pytest.importorskip("sklearn", reason="scikit-learn not available"):
            # Train performance predictor
            predictor = PMPerformancePredictor({'metric': 'velocity'})
            
            y = test_data['velocity']
            X = test_data.drop('velocity', axis=1)
            
            predictor.train(X[:25], y[:25], X[25:], y[25:])
            
            # Make prediction
            prediction = predictor.predict(X.tail(1))
            assert isinstance(prediction, PredictionResult)
        
        # 3. Dashboard Generation
        generator = DashboardGenerator({'output_path': tempfile.mkdtemp()})
        
        dashboard_data = {
            'summary': test_data[['date', 'velocity', 'team_health']].copy()
        }
        dashboard_data['summary']['on_time_rate'] = 0.85
        dashboard_data['summary']['quality_score'] = 90
        
        dashboard_path = generator.create_executive_dashboard(dashboard_data)
        
        assert os.path.exists(dashboard_path)
        
        # 4. Monitoring System
        monitoring = MonitoringSystem({
            'retention_period': 3600,
            'enable_prometheus': False
        })
        
        monitoring.create_default_pm_monitoring()
        
        # Record some metrics
        monitoring.record_metric('pm_velocity', 35)
        monitoring.record_metric('pm_quality_score', 85)
        monitoring.record_metric('pm_burnout_risk', 45)
        
        # Check summary
        summary = monitoring.get_metrics_summary()
        
        assert 'pm_velocity' in summary
        assert summary['pm_velocity']['current'] == 35
        
        # Cleanup
        os.unlink(f.name)
        if os.path.exists(etl_config['data_targets'][0]['connection_config']['file_path']):
            os.unlink(etl_config['data_targets'][0]['connection_config']['file_path'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])