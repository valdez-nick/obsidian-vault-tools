#!/usr/bin/env python3
"""Simple test runner to debug import issues."""

import sys
import traceback

try:
    print("Starting imports...")
    from analytics_hub.etl_pipeline import (
        ETLPipeline, DataSource, DataSourceType, DataTransformation,
        TransformationType, DataTarget, LoadStrategy, ETLMetrics
    )
    print("✓ ETL Pipeline imports successful")
    
    from analytics_hub.ml_models import (
        PMPerformancePredictor, BurnoutPredictor, ProductivityAnalyzer,
        PerformanceMetric, PredictionResult
    )
    print("✓ ML Models imports successful")
    
    from analytics_hub.dashboard_generator import (
        DashboardGenerator, MetricCard, MetricVisualization, ChartType,
        DashboardSection, AlertEngine
    )
    print("✓ Dashboard Generator imports successful")
    
    from analytics_hub.monitoring_system import (
        MonitoringSystem, MetricDefinition, MetricType, AlertRule,
        AlertSeverity, PerformanceThreshold, MonitoringEvent
    )
    print("✓ Monitoring System imports successful")
    
    print("\nAll imports successful! Running tests...")
    
    # Now try to run pytest
    import pytest
    sys.exit(pytest.main(['-v', 'tests/test_analytics_hub.py', '-o', 'addopts=']))
    
except Exception as e:
    print(f"\n❌ Import error: {e}")
    traceback.print_exc()
    sys.exit(1)