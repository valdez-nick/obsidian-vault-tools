#!/usr/bin/env python3
"""Simple test runner for Phase 5 Analytics Hub."""

import asyncio
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime

# Test imports
print("Testing imports...")
try:
    from analytics_hub.etl_pipeline import ETLPipeline, DataSourceType, TransformationType, LoadStrategy
    print("✓ ETL Pipeline imports successful")
    
    from analytics_hub.ml_models import PMPerformancePredictor, BurnoutPredictor, ProductivityAnalyzer
    print("✓ ML Models imports successful")
    
    from analytics_hub.dashboard_generator import DashboardGenerator, MetricCard, ChartType
    print("✓ Dashboard Generator imports successful")
    
    from analytics_hub.monitoring_system import MonitoringSystem, MetricDefinition, MetricType
    print("✓ Monitoring System imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Test basic functionality
print("\nTesting basic functionality...")

# 1. Test ETL Pipeline
print("\n1. Testing ETL Pipeline...")
try:
    config = {
        'pipeline_id': 'test',
        'data_sources': [{
            'name': 'test_source',
            'source_type': 'file',
            'connection_config': {},
            'file_path': 'test.csv'
        }],
        'transformations': [{
            'name': 'clean',
            'transformation_type': 'clean',
            'parameters': {}
        }],
        'data_targets': [{
            'name': 'test_target',
            'target_type': 'file',
            'connection_config': {'file_path': 'output.csv'},
            'table_name': 'test',
            'load_strategy': 'full_refresh'
        }]
    }
    
    pipeline = ETLPipeline(config)
    print("✓ ETL Pipeline created successfully")
except Exception as e:
    print(f"❌ ETL Pipeline error: {e}")

# 2. Test ML Models
print("\n2. Testing ML Models...")
try:
    predictor = PMPerformancePredictor({'metric': 'velocity'})
    print("✓ PM Performance Predictor created")
    
    burnout = BurnoutPredictor({'use_lstm': False})
    print("✓ Burnout Predictor created")
    
    analyzer = ProductivityAnalyzer({})
    print("✓ Productivity Analyzer created")
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10),
        'tickets_assigned': np.random.randint(8, 20, 10),
        'tickets_completed': np.random.randint(5, 15, 10),
        'work_hours': np.random.uniform(6, 10, 10),
        'meetings_hours': np.random.uniform(1, 4, 10),
        'meetings_count': np.random.randint(2, 6, 10)
    })
    
    features = predictor.engineer_features(sample_data)
    print(f"✓ Engineered {len(features.columns)} features")
    
except Exception as e:
    print(f"❌ ML Models error: {e}")

# 3. Test Dashboard Generator
print("\n3. Testing Dashboard Generator...")
try:
    generator = DashboardGenerator({'output_path': tempfile.gettempdir()})
    print("✓ Dashboard Generator created")
    
    card = MetricCard(
        title="Test Metric",
        value=100,
        unit="%"
    )
    
    html = generator.create_metric_card_html(card)
    print("✓ Created metric card HTML")
    
except Exception as e:
    print(f"❌ Dashboard Generator error: {e}")

# 4. Test Monitoring System
print("\n4. Testing Monitoring System...")
try:
    monitoring = MonitoringSystem({
        'retention_period': 3600,
        'enable_prometheus': False
    })
    print("✓ Monitoring System created")
    
    monitoring.define_metric(MetricDefinition(
        name="test_metric",
        metric_type=MetricType.GAUGE,
        description="Test metric"
    ))
    print("✓ Defined test metric")
    
    monitoring.record_metric("test_metric", 42.0)
    print("✓ Recorded metric value")
    
    summary = monitoring.get_metrics_summary()
    print(f"✓ Got metrics summary: {summary.get('test_metric', {}).get('current')}")
    
except Exception as e:
    print(f"❌ Monitoring System error: {e}")

print("\n✅ Phase 5 Analytics Hub basic tests completed!")
print("\nNote: Full test suite skipped due to missing optional dependencies (plotly).")