"""
Unit tests for WBR Data Extractor

Tests data extraction, validation, and metric aggregation functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pandas as pd
from datetime import datetime, timedelta
import asyncio

from wbr.wbr_data_extractor import (
    WBRDataExtractor, WBRMetric, WBRDataPackage, MetricType, DataValidationError
)


class TestWBRDataExtractor(unittest.TestCase):
    """Test cases for WBRDataExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'snowflake': {
                'account': 'test_account',
                'user': 'test_user',
                'password': 'test_password',
                'warehouse': 'TEST_WH',
                'database': 'TEST_DB',
                'schema': 'TEST_SCHEMA'
            },
            'jira': {
                'server': 'https://test.atlassian.net',
                'username': 'test@example.com',
                'api_token': 'test_token'
            },
            'jira_board_id': 123
        }
        
    def test_wbr_metric_creation(self):
        """Test WBR metric data class."""
        metric = WBRMetric(
            name="Test Metric",
            value=100.0,
            previous_value=80.0,
            change_percent=None,  # Should be calculated
            trend="",  # Should be calculated
            timestamp=datetime.now(),
            source="Test",
            unit="count"
        )
        
        # Check calculated fields
        self.assertEqual(metric.change_percent, 25.0)  # (100-80)/80 * 100
        self.assertEqual(metric.trend, "up")
    
    def test_metric_trend_calculation(self):
        """Test metric trend calculation."""
        # Test upward trend
        metric_up = WBRMetric(
            name="Test", value=110, previous_value=100, change_percent=None,
            trend="", timestamp=datetime.now(), source="Test", unit="count"
        )
        self.assertEqual(metric_up.trend, "up")
        
        # Test downward trend  
        metric_down = WBRMetric(
            name="Test", value=90, previous_value=100, change_percent=None,
            trend="", timestamp=datetime.now(), source="Test", unit="count"
        )
        self.assertEqual(metric_down.trend, "down")
        
        # Test stable trend (small change)
        metric_stable = WBRMetric(
            name="Test", value=101, previous_value=100, change_percent=None,
            trend="", timestamp=datetime.now(), source="Test", unit="count"
        )
        self.assertEqual(metric_stable.trend, "stable")
    
    @patch('wbr.wbr_data_extractor.SnowflakeConnector')
    @patch('wbr.wbr_data_extractor.JiraConnector')
    def test_extractor_initialization(self, mock_jira, mock_snowflake):
        """Test WBR data extractor initialization."""
        extractor = WBRDataExtractor(self.config)
        
        self.assertIsNotNone(extractor.snowflake)
        self.assertIsNotNone(extractor.jira)
        self.assertEqual(len(extractor.metric_configs), 3)  # DAU_MAU, FEATURE_ADOPTION, SPRINT_VELOCITY
    
    @patch('wbr.wbr_data_extractor.SnowflakeConnector')
    def test_product_metrics_extraction(self, mock_snowflake_class):
        """Test product metrics extraction."""
        # Setup mock Snowflake connector
        mock_snowflake = Mock()
        mock_snowflake_class.return_value = mock_snowflake
        mock_snowflake.connect.return_value = True
        
        # Mock DAU/MAU data
        dau_mau_data = pd.DataFrame([
            {'activity_date': '2024-01-15', 'dau': 1000, 'mau': 5000, 'dau_mau_ratio': 20.0},
            {'activity_date': '2024-01-14', 'dau': 950, 'mau': 4900, 'dau_mau_ratio': 19.4}
        ])
        mock_snowflake.get_dau_mau.return_value = dau_mau_data
        
        # Mock feature adoption data
        adoption_data = pd.DataFrame([
            {'feature_name': 'Feature A', 'adoption_rate': 25.0, 'unique_users': 500},
            {'feature_name': 'Feature B', 'adoption_rate': 15.0, 'unique_users': 300}
        ])
        mock_snowflake.get_feature_adoption.return_value = adoption_data
        
        extractor = WBRDataExtractor(self.config)
        
        # Run the async method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        result = loop.run_until_complete(
            extractor._extract_product_metrics(start_date, end_date)
        )
        
        # Verify results
        self.assertGreater(len(result), 0)
        
        # Check DAU/MAU metric
        dau_mau_metric = next((m for m in result if "DAU/MAU" in m.name), None)
        self.assertIsNotNone(dau_mau_metric)
        self.assertEqual(dau_mau_metric.value, 20.0)
        self.assertEqual(dau_mau_metric.previous_value, 19.4)
        self.assertEqual(dau_mau_metric.source, "Snowflake")
        
        # Check feature adoption metrics
        feature_metrics = [m for m in result if "Adoption" in m.name]
        self.assertEqual(len(feature_metrics), 2)
    
    @patch('wbr.wbr_data_extractor.JiraConnector')
    def test_engineering_metrics_extraction(self, mock_jira_class):
        """Test engineering metrics extraction."""
        # Setup mock Jira connector
        mock_jira = Mock()
        mock_jira_class.return_value = mock_jira
        mock_jira.connect.return_value = True
        
        # Mock sprint data
        sprint_data = pd.DataFrame([
            {
                'sprint_id': 100, 'velocity': 45, 'end_date': '2024-01-15',
                'bugs_opened': 3, 'stories_completed': 15
            },
            {
                'sprint_id': 99, 'velocity': 40, 'end_date': '2024-01-08', 
                'bugs_opened': 2, 'stories_completed': 12
            }
        ])
        mock_jira.get_sprint_progress.return_value = sprint_data
        
        extractor = WBRDataExtractor(self.config)
        
        # Run the async method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        result = loop.run_until_complete(
            extractor._extract_engineering_metrics(start_date, end_date)
        )
        
        # Verify results
        self.assertGreater(len(result), 0)
        
        # Check velocity metric
        velocity_metric = next((m for m in result if "Velocity" in m.name), None)
        self.assertIsNotNone(velocity_metric)
        self.assertEqual(velocity_metric.value, 45)
        self.assertEqual(velocity_metric.previous_value, 40)
        self.assertEqual(velocity_metric.source, "Jira")
        
        # Check bug rate metric
        bug_rate_metric = next((m for m in result if "Bug Rate" in m.name), None)
        self.assertIsNotNone(bug_rate_metric)
        self.assertEqual(bug_rate_metric.value, 20.0)  # 3/15 * 100
    
    def test_data_quality_validation(self):
        """Test data quality validation."""
        extractor = WBRDataExtractor(self.config)
        
        # Create test metrics
        metrics = [
            WBRMetric(
                name="Metric 1", value=100, previous_value=90, change_percent=None,
                trend="", timestamp=datetime.now() - timedelta(hours=1),
                source="Test", unit="count"
            ),
            WBRMetric(
                name="Metric 2", value=50, previous_value=45, change_percent=None,
                trend="", timestamp=datetime.now() - timedelta(hours=2),
                source="Test", unit="count"
            )
        ]
        
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        
        quality_score = extractor._validate_data_quality(metrics, start_date, end_date)
        
        # Should be a good quality score (fresh data, no nulls)
        self.assertGreater(quality_score, 0.8)
        self.assertLessEqual(quality_score, 1.0)
    
    def test_alert_identification(self):
        """Test alert identification."""
        extractor = WBRDataExtractor(self.config)
        
        # Create metrics with issues
        metrics = [
            WBRMetric(
                name="Below Threshold", value=5.0, previous_value=10.0, 
                change_percent=-50.0, trend="down", timestamp=datetime.now(),
                source="Test", unit="count", alert_threshold=8.0
            ),
            WBRMetric(
                name="Declining Fast", value=80.0, previous_value=100.0,
                change_percent=-20.0, trend="down", timestamp=datetime.now(),
                source="Test", unit="count"
            ),
            WBRMetric(
                name="Off Target", value=60.0, previous_value=65.0,
                change_percent=-7.7, trend="down", timestamp=datetime.now(),
                source="Test", unit="count", target=100.0
            )
        ]
        
        alerts = extractor._identify_alerts(metrics)
        
        # Should generate alerts for all three issues
        self.assertGreaterEqual(len(alerts), 2)
        self.assertTrue(any("Below Threshold" in alert for alert in alerts))
        self.assertTrue(any("declined significantly" in alert for alert in alerts))
    
    def test_metric_summary(self):
        """Test metric summary generation."""
        extractor = WBRDataExtractor(self.config)
        
        metrics = [
            WBRMetric(
                name="Metric 1", value=100, previous_value=90, change_percent=11.1,
                trend="up", timestamp=datetime.now(), source="Source A", 
                unit="count", target=95.0
            ),
            WBRMetric(
                name="Metric 2", value=80, previous_value=85, change_percent=-5.9,
                trend="down", timestamp=datetime.now(), source="Source B",
                unit="count", target=90.0
            ),
            WBRMetric(
                name="Metric 3", value=120, previous_value=110, change_percent=9.1,
                trend="up", timestamp=datetime.now(), source="Source A",
                unit="count", target=100.0
            )
        ]
        
        summary = extractor.get_metric_summary(metrics)
        
        self.assertEqual(summary['total_metrics'], 3)
        self.assertEqual(summary['metrics_with_targets'], 3)
        self.assertEqual(summary['metrics_above_target'], 2)  # Metric 1 and 3
        self.assertEqual(summary['metrics_trending_up'], 2)
        self.assertEqual(summary['metrics_trending_down'], 1)
        self.assertIn('Source A', summary['data_sources'])
        self.assertIn('Source B', summary['data_sources'])


class TestWBRDataPackage(unittest.TestCase):
    """Test cases for WBR data package."""
    
    def test_data_package_creation(self):
        """Test WBR data package creation."""
        metrics = [
            WBRMetric(
                name="Test Metric", value=100, previous_value=90,
                change_percent=None, trend="", timestamp=datetime.now(),
                source="Test", unit="count"
            )
        ]
        
        package = WBRDataPackage(
            metrics=metrics,
            insights=["Test insight"],
            alerts=["Test alert"],
            raw_data={'test': pd.DataFrame({'a': [1, 2, 3]})},
            generation_time=datetime.now(),
            data_freshness={'test': datetime.now()},
            quality_score=0.95
        )
        
        self.assertEqual(len(package.metrics), 1)
        self.assertEqual(len(package.insights), 1)
        self.assertEqual(len(package.alerts), 1)
        self.assertEqual(package.quality_score, 0.95)
        self.assertIn('test', package.raw_data)


if __name__ == '__main__':
    unittest.main()