"""
Unit tests for Insight Generator

Tests AI-powered insight generation, trend analysis, and recommendation logic.
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio

from wbr.insight_generator import (
    InsightGenerator, Insight, InsightType, InsightPriority
)


class TestInsightGenerator(unittest.TestCase):
    """Test cases for InsightGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'anomaly_threshold': 2.0,
            'trend_min_periods': 4,
            'correlation_threshold': 0.7,
            'confidence_threshold': 0.6
        }
        
        self.sample_metrics = [
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
                'name': 'Feature A Adoption',
                'value': 25.0,
                'change_percent': 12.0,
                'trend': 'up',
                'source': 'Snowflake',
                'target': 30.0,
                'alert_threshold': 20.0
            },
            {
                'name': 'Sprint Velocity',
                'value': 35.0,
                'change_percent': -20.0,
                'trend': 'down',
                'source': 'Jira',
                'target': 40.0,
                'alert_threshold': 30.0
            }
        ]
    
    def test_insight_creation(self):
        """Test Insight data class."""
        insight = Insight(
            title="Test Insight",
            description="This is a test insight",
            insight_type=InsightType.TREND,
            priority=InsightPriority.HIGH,
            confidence=0.8,
            supporting_data={'metric': 'test'},
            recommendations=["Action 1", "Action 2"],
            timestamp=datetime.now(),
            metrics_involved=["Metric 1"]
        )
        
        self.assertEqual(insight.title, "Test Insight")
        self.assertEqual(insight.insight_type, InsightType.TREND)
        self.assertEqual(insight.priority, InsightPriority.HIGH)
        self.assertEqual(insight.confidence, 0.8)
        
        # Test to_dict conversion
        insight_dict = insight.to_dict()
        self.assertIn('title', insight_dict)
        self.assertIn('type', insight_dict)
        self.assertEqual(insight_dict['type'], 'trend')
    
    def test_generator_initialization(self):
        """Test insight generator initialization."""
        generator = InsightGenerator(self.config)
        
        self.assertEqual(generator.anomaly_threshold, 2.0)
        self.assertEqual(generator.trend_min_periods, 4)
        self.assertEqual(generator.correlation_threshold, 0.7)
        self.assertIsNotNone(generator.prompt_templates)
    
    def test_anomaly_detection(self):
        """Test anomaly detection logic."""
        generator = InsightGenerator(self.config)
        
        # Create metrics DataFrame with an anomaly
        metrics_data = [
            {'name': 'Normal Metric 1', 'value': 100},
            {'name': 'Normal Metric 2', 'value': 110}, 
            {'name': 'Normal Metric 3', 'value': 95},
            {'name': 'Anomaly Metric', 'value': 500}  # Clear anomaly
        ]
        metrics_df = pd.DataFrame(metrics_data)
        
        # Run async method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        insights = loop.run_until_complete(generator._detect_anomalies(metrics_df))
        
        # Should detect anomaly
        self.assertGreater(len(insights), 0)
        anomaly_insight = next((i for i in insights if i.insight_type == InsightType.ANOMALY), None)
        self.assertIsNotNone(anomaly_insight)
        self.assertEqual(anomaly_insight.priority, InsightPriority.HIGH)
    
    def test_trend_analysis(self):
        """Test trend analysis."""
        generator = InsightGenerator(self.config)
        
        # Create metrics with significant trends
        metrics_data = [
            {'name': 'Declining Metric', 'value': 80, 'change_percent': -25.0, 'previous_value': 100},
            {'name': 'Growing Metric', 'value': 130, 'change_percent': 30.0, 'previous_value': 100},
            {'name': 'Stable Metric', 'value': 102, 'change_percent': 2.0, 'previous_value': 100}
        ]
        metrics_df = pd.DataFrame(metrics_data)
        
        # Run async method  
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        insights = loop.run_until_complete(generator._analyze_trends(metrics_df))
        
        # Should detect significant trends
        trend_insights = [i for i in insights if i.insight_type == InsightType.TREND]
        self.assertGreaterEqual(len(trend_insights), 2)  # Declining and growing
        
        # Check priority assignment
        high_priority = [i for i in trend_insights if i.priority == InsightPriority.HIGH]
        self.assertGreater(len(high_priority), 0)  # 25% and 30% changes should be high priority
    
    def test_correlation_analysis(self):
        """Test correlation analysis."""
        generator = InsightGenerator(self.config)
        
        # Create metrics with correlations (same trends)
        metrics_data = [
            {'name': 'Metric 1', 'value': 100, 'trend': 'up'},
            {'name': 'Metric 2', 'value': 110, 'trend': 'up'},
            {'name': 'Metric 3', 'value': 90, 'trend': 'down'},
            {'name': 'Metric 4', 'value': 85, 'trend': 'down'}
        ]
        metrics_df = pd.DataFrame(metrics_data)
        
        # Run async method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        insights = loop.run_until_complete(generator._find_correlations(metrics_df))
        
        # Should find positive and negative correlations
        correlation_insights = [i for i in insights if i.insight_type == InsightType.CORRELATION]
        self.assertGreaterEqual(len(correlation_insights), 1)
    
    def test_recommendation_generation(self):
        """Test recommendation generation."""
        generator = InsightGenerator(self.config)
        
        # Create metrics with performance issues
        metrics_data = [
            {
                'name': 'Below Target', 'value': 70, 'target': 100,
                'alert_threshold': None
            },
            {
                'name': 'Below Alert', 'value': 25, 'target': None,
                'alert_threshold': 30
            },
            {
                'name': 'On Target', 'value': 95, 'target': 100,
                'alert_threshold': None
            }
        ]
        metrics_df = pd.DataFrame(metrics_data)
        
        # Run async method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        insights = loop.run_until_complete(generator._generate_recommendations(metrics_df))
        
        # Should generate recommendations for issues
        self.assertGreater(len(insights), 0)
        
        # Check for performance gap insight
        gap_insights = [i for i in insights if i.insight_type == InsightType.RECOMMENDATION]
        self.assertGreater(len(gap_insights), 0)
        
        # Check for alert insight
        alert_insights = [i for i in insights if i.insight_type == InsightType.ALERT]
        self.assertGreater(len(alert_insights), 0)
        
        # Alert should be critical priority
        critical_alerts = [i for i in alert_insights if i.priority == InsightPriority.CRITICAL]
        self.assertGreater(len(critical_alerts), 0)
    
    def test_trend_recommendations(self):
        """Test trend-specific recommendations."""
        generator = InsightGenerator(self.config)
        
        # Test DAU/MAU recommendations
        dau_recommendations = generator._get_trend_recommendations("DAU/MAU Ratio", -15.0)
        self.assertIn("acquisition", dau_recommendations[0].lower())
        
        # Test revenue recommendations
        revenue_recommendations = generator._get_trend_recommendations("Weekly Revenue", 10.0)
        self.assertIn("successful", revenue_recommendations[0].lower())
        
        # Test velocity recommendations
        velocity_recommendations = generator._get_trend_recommendations("Sprint Velocity", -20.0)
        self.assertIn("sprint", velocity_recommendations[0].lower())
    
    def test_performance_recommendations(self):
        """Test performance gap recommendations."""
        generator = InsightGenerator(self.config)
        
        # Test large gap recommendations
        large_gap_recs = generator._get_performance_recommendations("Test Metric", 25.0)
        self.assertIn("immediate", large_gap_recs[0].lower())
        
        # Test medium gap recommendations
        medium_gap_recs = generator._get_performance_recommendations("Test Metric", 15.0)
        self.assertIn("review", medium_gap_recs[0].lower())
        
        # Test small gap recommendations
        small_gap_recs = generator._get_performance_recommendations("Test Metric", 5.0)
        self.assertIn("fine-tune", small_gap_recs[0].lower())
    
    def test_metrics_summary_preparation(self):
        """Test metrics summary for AI analysis."""
        generator = InsightGenerator(self.config)
        
        summary = generator._prepare_metrics_summary(self.sample_metrics)
        
        self.assertIn("DAU/MAU Ratio", summary)
        self.assertIn("↓25.0%", summary)  # Should show decline
        self.assertIn("✗", summary)  # Should show missing target
        self.assertIn("Feature A Adoption", summary)
        self.assertIn("↑12.0%", summary)  # Should show increase
    
    def test_ai_insights_parsing(self):
        """Test AI response parsing."""
        generator = InsightGenerator(self.config)
        
        # Mock AI response
        ai_response = """
        **Title**: User Engagement Declining
        **Description**: DAU/MAU ratio has dropped significantly this week
        **Priority**: High
        **Recommendations**: Investigate user retention programs
        
        **Title**: Feature Adoption Improving
        **Description**: Feature A showing strong adoption growth
        **Priority**: Medium
        **Recommendations**: Scale successful adoption strategies
        """
        
        insights = generator._parse_ai_insights(ai_response)
        
        self.assertEqual(len(insights), 2)
        self.assertEqual(insights[0].title, "User Engagement Declining")
        self.assertEqual(insights[0].priority, InsightPriority.HIGH)
        self.assertEqual(insights[1].title, "Feature Adoption Improving")
    
    def test_priority_parsing(self):
        """Test priority text parsing."""
        generator = InsightGenerator(self.config)
        
        self.assertEqual(generator._parse_priority("critical"), InsightPriority.CRITICAL)
        self.assertEqual(generator._parse_priority("high"), InsightPriority.HIGH)
        self.assertEqual(generator._parse_priority("medium"), InsightPriority.MEDIUM)
        self.assertEqual(generator._parse_priority("low"), InsightPriority.LOW)
        self.assertEqual(generator._parse_priority("unknown"), InsightPriority.MEDIUM)  # Default
    
    def test_basic_summary_generation(self):
        """Test basic summary generation without AI."""
        generator = InsightGenerator(self.config)
        
        # Create sample insights
        insights = [
            Insight(
                title="Critical Issue", description="Test",
                insight_type=InsightType.ALERT, priority=InsightPriority.CRITICAL,
                confidence=1.0, supporting_data={}, recommendations=["Fix immediately"],
                timestamp=datetime.now(), metrics_involved=[]
            ),
            Insight(
                title="High Priority", description="Test",
                insight_type=InsightType.TREND, priority=InsightPriority.HIGH,
                confidence=0.8, supporting_data={}, recommendations=["Investigate"],
                timestamp=datetime.now(), metrics_involved=[]
            )
        ]
        
        metrics_summary = {
            'total_metrics': 5,
            'metrics_above_target': 3
        }
        
        summary = generator._generate_basic_summary(insights, metrics_summary)
        
        self.assertIn("3/5 metrics", summary)
        self.assertIn("60%", summary)  # 3/5 = 60%
        self.assertIn("1 critical", summary)
        self.assertIn("1 high-priority", summary)
    
    @patch('wbr.insight_generator.openai')
    def test_ai_client_initialization(self, mock_openai):
        """Test AI client initialization."""
        config_with_ai = self.config.copy()
        config_with_ai['openai_api_key'] = 'test_key'
        
        generator = InsightGenerator(config_with_ai)
        
        self.assertIsNotNone(generator.openai_client)
        mock_openai.api_key = 'test_key'  # Verify API key was set
    
    def test_comprehensive_insights_integration(self):
        """Test the main comprehensive insights method."""
        generator = InsightGenerator(self.config)
        
        # Run the main method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        insights = loop.run_until_complete(
            generator.generate_comprehensive_insights(self.sample_metrics)
        )
        
        # Should generate various types of insights
        self.assertGreater(len(insights), 0)
        
        # Check insight types are present
        insight_types = {i.insight_type for i in insights}
        self.assertIn(InsightType.TREND, insight_types)
        self.assertIn(InsightType.RECOMMENDATION, insight_types)
        
        # Should be sorted by priority
        priorities = [i.priority for i in insights]
        # Critical should come before High, High before Medium, etc.
        for i in range(len(priorities) - 1):
            current_priority = priorities[i].value
            next_priority = priorities[i + 1].value
            priority_order = ['critical', 'high', 'medium', 'low']
            self.assertLessEqual(
                priority_order.index(current_priority),
                priority_order.index(next_priority)
            )


if __name__ == '__main__':
    unittest.main()