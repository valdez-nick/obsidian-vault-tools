"""
WBR Data Extractor Implementation

Extracts and aggregates data from multiple sources for Weekly Business Review generation.
Handles Snowflake metrics, Jira sprint data, Mixpanel integration, and data validation.
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from connectors.snowflake_connector import SnowflakeConnector
from connectors.jira_connector import JiraConnector
from intelligence.ai_analyzer import AIAnalyzer


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of WBR metrics."""
    DAU_MAU = "dau_mau"
    FEATURE_ADOPTION = "feature_adoption"
    REVENUE = "revenue"
    CHURN = "churn"
    SPRINT_VELOCITY = "sprint_velocity"
    BUG_RATE = "bug_rate"
    CUSTOMER_SATISFACTION = "customer_satisfaction"


@dataclass
class WBRMetric:
    """Data class for WBR metrics."""
    name: str
    value: float
    previous_value: Optional[float]
    change_percent: Optional[float]
    trend: str  # "up", "down", "stable"
    timestamp: datetime
    source: str
    unit: str
    target: Optional[float] = None
    alert_threshold: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.previous_value is not None and self.previous_value != 0:
            self.change_percent = ((self.value - self.previous_value) / self.previous_value) * 100
            
            if abs(self.change_percent) < 2:  # Less than 2% change
                self.trend = "stable"
            elif self.change_percent > 0:
                self.trend = "up"
            else:
                self.trend = "down"


@dataclass
class WBRDataPackage:
    """Complete data package for WBR generation."""
    metrics: List[WBRMetric]
    insights: List[str]
    alerts: List[str]
    raw_data: Dict[str, pd.DataFrame]
    generation_time: datetime
    data_freshness: Dict[str, datetime]
    quality_score: float


class DataValidationError(Exception):
    """Custom exception for data validation failures."""
    pass


class WBRDataExtractor:
    """
    WBR Data Extractor for automated business review generation.
    
    Orchestrates data extraction from multiple sources, validates data quality,
    and prepares comprehensive metrics packages for WBR slides.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize WBR Data Extractor.
        
        Args:
            config: Configuration dictionary with connector settings
        """
        self.config = config
        self.snowflake = None
        self.jira = None
        self.ai_analyzer = None
        
        # Data quality thresholds
        self.quality_thresholds = {
            'completeness': 0.95,  # 95% of expected data points
            'freshness': 24,       # Data should be < 24 hours old
            'accuracy': 0.98       # 98% accuracy for validation checks
        }
        
        # Metric configurations
        self.metric_configs = {
            MetricType.DAU_MAU: {
                'source': 'snowflake',
                'table': 'user_events',
                'target': 0.20,  # 20% DAU/MAU ratio target
                'alert_threshold': 0.15
            },
            MetricType.FEATURE_ADOPTION: {
                'source': 'snowflake',
                'table': 'product_events',
                'target': 0.30,  # 30% adoption rate target
                'alert_threshold': 0.20
            },
            MetricType.SPRINT_VELOCITY: {
                'source': 'jira',
                'board_id': config.get('jira_board_id'),
                'target': 40,  # 40 story points per sprint
                'alert_threshold': 30
            }
        }
        
        self._initialize_connectors()
        
    def _initialize_connectors(self):
        """Initialize all data source connectors."""
        try:
            # Initialize Snowflake connector
            if 'snowflake' in self.config:
                self.snowflake = SnowflakeConnector(self.config['snowflake'])
                if not self.snowflake.connect():
                    logger.warning("Failed to connect to Snowflake")
            
            # Initialize Jira connector
            if 'jira' in self.config:
                self.jira = JiraConnector(self.config['jira'])
                if not self.jira.connect():
                    logger.warning("Failed to connect to Jira")
            
            # Initialize AI analyzer
            if 'ai' in self.config:
                self.ai_analyzer = AIAnalyzer(self.config['ai'])
                
        except Exception as e:
            logger.error(f"Failed to initialize connectors: {e}")
            raise
    
    async def extract_wbr_data(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> WBRDataPackage:
        """
        Extract complete WBR data package.
        
        Args:
            start_date: Start date for data extraction (defaults to 7 days ago)
            end_date: End date for data extraction (defaults to today)
            
        Returns:
            Complete WBR data package with metrics and insights
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
            
        logger.info(f"Starting WBR data extraction for {start_date} to {end_date}")
        
        try:
            # Extract data from all sources in parallel
            tasks = [
                self._extract_product_metrics(start_date, end_date),
                self._extract_engineering_metrics(start_date, end_date),
                self._extract_business_metrics(start_date, end_date)
            ]
            
            product_data, engineering_data, business_data = await asyncio.gather(*tasks)
            
            # Combine all metrics
            all_metrics = product_data + engineering_data + business_data
            
            # Validate data quality
            quality_score = self._validate_data_quality(all_metrics, start_date, end_date)
            
            # Generate insights using AI
            insights = await self._generate_insights(all_metrics)
            
            # Identify alerts
            alerts = self._identify_alerts(all_metrics)
            
            # Prepare raw data dictionary
            raw_data = await self._collect_raw_data(start_date, end_date)
            
            # Create data freshness map
            data_freshness = self._calculate_data_freshness(raw_data)
            
            # Package everything
            wbr_package = WBRDataPackage(
                metrics=all_metrics,
                insights=insights,
                alerts=alerts,
                raw_data=raw_data,
                generation_time=datetime.now(),
                data_freshness=data_freshness,
                quality_score=quality_score
            )
            
            logger.info(f"WBR data extraction completed. Quality score: {quality_score:.2f}")
            return wbr_package
            
        except Exception as e:
            logger.error(f"Failed to extract WBR data: {e}")
            raise
    
    async def _extract_product_metrics(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[WBRMetric]:
        """Extract product-related metrics from Snowflake."""
        if not self.snowflake:
            logger.warning("Snowflake not configured, skipping product metrics")
            return []
            
        metrics = []
        
        try:
            # DAU/MAU metrics
            dau_mau_data = self.snowflake.get_dau_mau(
                table=self.metric_configs[MetricType.DAU_MAU]['table'],
                user_id_col='user_id',
                timestamp_col='event_timestamp',
                days_back=7
            )
            
            if not dau_mau_data.empty:
                latest = dau_mau_data.iloc[0]
                previous = dau_mau_data.iloc[1] if len(dau_mau_data) > 1 else None
                
                metrics.append(WBRMetric(
                    name="DAU/MAU Ratio",
                    value=latest['dau_mau_ratio'],
                    previous_value=previous['dau_mau_ratio'] if previous is not None else None,
                    change_percent=None,  # Will be calculated in __post_init__
                    trend="",  # Will be calculated in __post_init__
                    timestamp=datetime.fromisoformat(str(latest['activity_date'])),
                    source="Snowflake",
                    unit="percentage",
                    target=self.metric_configs[MetricType.DAU_MAU]['target'],
                    alert_threshold=self.metric_configs[MetricType.DAU_MAU]['alert_threshold']
                ))
            
            # Feature adoption metrics
            adoption_data = self.snowflake.get_feature_adoption(
                table=self.metric_configs[MetricType.FEATURE_ADOPTION]['table'],
                user_id_col='user_id',
                feature_col='feature_name',
                timestamp_col='event_timestamp',
                days_back=7
            )
            
            if not adoption_data.empty:
                # Get top 5 features
                top_features = adoption_data.head(5)
                for _, feature in top_features.iterrows():
                    metrics.append(WBRMetric(
                        name=f"{feature['feature_name']} Adoption",
                        value=feature['adoption_rate'],
                        previous_value=None,  # Would need historical data
                        change_percent=None,
                        trend="stable",
                        timestamp=end_date,
                        source="Snowflake",
                        unit="percentage",
                        target=self.metric_configs[MetricType.FEATURE_ADOPTION]['target'],
                        alert_threshold=self.metric_configs[MetricType.FEATURE_ADOPTION]['alert_threshold']
                    ))
            
        except Exception as e:
            logger.error(f"Failed to extract product metrics: {e}")
            
        return metrics
    
    async def _extract_engineering_metrics(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[WBRMetric]:
        """Extract engineering-related metrics from Jira."""
        if not self.jira:
            logger.warning("Jira not configured, skipping engineering metrics")
            return []
            
        metrics = []
        
        try:
            # Sprint velocity
            board_id = self.metric_configs[MetricType.SPRINT_VELOCITY]['board_id']
            if board_id:
                sprint_data = self.jira.get_sprint_progress(board_id)
                
                if not sprint_data.empty:
                    current_sprint = sprint_data.iloc[0]
                    previous_sprint = sprint_data.iloc[1] if len(sprint_data) > 1 else None
                    
                    metrics.append(WBRMetric(
                        name="Sprint Velocity",
                        value=current_sprint['velocity'],
                        previous_value=previous_sprint['velocity'] if previous_sprint is not None else None,
                        change_percent=None,
                        trend="",
                        timestamp=datetime.fromisoformat(str(current_sprint['end_date'])),
                        source="Jira",
                        unit="story points",
                        target=self.metric_configs[MetricType.SPRINT_VELOCITY]['target'],
                        alert_threshold=self.metric_configs[MetricType.SPRINT_VELOCITY]['alert_threshold']
                    ))
                    
                    # Bug rate calculation
                    bug_rate = (current_sprint['bugs_opened'] / current_sprint['stories_completed']) * 100
                    prev_bug_rate = None
                    if previous_sprint is not None:
                        prev_bug_rate = (previous_sprint['bugs_opened'] / previous_sprint['stories_completed']) * 100
                    
                    metrics.append(WBRMetric(
                        name="Bug Rate",
                        value=bug_rate,
                        previous_value=prev_bug_rate,
                        change_percent=None,
                        trend="",
                        timestamp=datetime.fromisoformat(str(current_sprint['end_date'])),
                        source="Jira",
                        unit="percentage",
                        target=5.0,  # 5% bug rate target
                        alert_threshold=10.0  # Alert if > 10%
                    ))
            
        except Exception as e:
            logger.error(f"Failed to extract engineering metrics: {e}")
            
        return metrics
    
    async def _extract_business_metrics(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[WBRMetric]:
        """Extract business-related metrics from Snowflake."""
        if not self.snowflake:
            logger.warning("Snowflake not configured, skipping business metrics")
            return []
            
        metrics = []
        
        try:
            # Revenue analysis
            revenue_data = self.snowflake.get_revenue_analysis(
                table='transactions',
                user_id_col='user_id',
                transaction_id_col='transaction_id',
                amount_col='amount',
                timestamp_col='created_at',
                period='week',
                periods_back=4
            )
            
            if not revenue_data.empty:
                current_week = revenue_data.iloc[0]
                previous_week = revenue_data.iloc[1] if len(revenue_data) > 1 else None
                
                metrics.append(WBRMetric(
                    name="Weekly Revenue",
                    value=current_week['total_revenue'],
                    previous_value=previous_week['total_revenue'] if previous_week is not None else None,
                    change_percent=None,
                    trend="",
                    timestamp=datetime.fromisoformat(str(current_week['period'])),
                    source="Snowflake",
                    unit="USD",
                    target=None,  # Set based on business goals
                    alert_threshold=None
                ))
                
                metrics.append(WBRMetric(
                    name="ARPU",
                    value=current_week['arpu'],
                    previous_value=previous_week['arpu'] if previous_week is not None else None,
                    change_percent=None,
                    trend="",
                    timestamp=datetime.fromisoformat(str(current_week['period'])),
                    source="Snowflake",
                    unit="USD",
                    target=None,
                    alert_threshold=None
                ))
            
        except Exception as e:
            logger.error(f"Failed to extract business metrics: {e}")
            
        return metrics
    
    def _validate_data_quality(
        self, 
        metrics: List[WBRMetric], 
        start_date: datetime, 
        end_date: datetime
    ) -> float:
        """
        Validate data quality and return quality score.
        
        Args:
            metrics: List of extracted metrics
            start_date: Start date for validation period
            end_date: End date for validation period
            
        Returns:
            Quality score between 0 and 1
        """
        total_score = 0
        checks_performed = 0
        
        # Check completeness - do we have expected metrics?
        expected_metrics = len(self.metric_configs)
        actual_metrics = len(metrics)
        completeness_score = min(actual_metrics / expected_metrics, 1.0)
        total_score += completeness_score
        checks_performed += 1
        
        # Check data freshness
        freshness_scores = []
        for metric in metrics:
            age_hours = (datetime.now() - metric.timestamp).total_seconds() / 3600
            freshness_score = max(0, 1 - (age_hours / self.quality_thresholds['freshness']))
            freshness_scores.append(freshness_score)
        
        if freshness_scores:
            avg_freshness = sum(freshness_scores) / len(freshness_scores)
            total_score += avg_freshness
            checks_performed += 1
        
        # Check for null/invalid values
        valid_values = sum(1 for m in metrics if m.value is not None and not pd.isna(m.value))
        accuracy_score = valid_values / len(metrics) if metrics else 0
        total_score += accuracy_score
        checks_performed += 1
        
        # Calculate overall score
        overall_score = total_score / checks_performed if checks_performed > 0 else 0
        
        # Log quality assessment
        logger.info(f"Data Quality Assessment:")
        logger.info(f"  Completeness: {completeness_score:.2f}")
        logger.info(f"  Freshness: {avg_freshness:.2f}" if freshness_scores else "  Freshness: N/A")
        logger.info(f"  Accuracy: {accuracy_score:.2f}")
        logger.info(f"  Overall Score: {overall_score:.2f}")
        
        if overall_score < 0.8:
            logger.warning(f"Data quality below threshold: {overall_score:.2f}")
        
        return overall_score
    
    async def _generate_insights(self, metrics: List[WBRMetric]) -> List[str]:
        """Generate AI-powered insights from metrics."""
        if not self.ai_analyzer:
            logger.warning("AI analyzer not configured, skipping insights generation")
            return []
        
        try:
            # Prepare metrics data for AI analysis
            metrics_data = []
            for metric in metrics:
                metrics_data.append({
                    'name': metric.name,
                    'value': metric.value,
                    'change_percent': metric.change_percent,
                    'trend': metric.trend,
                    'source': metric.source,
                    'target': metric.target,
                    'alert_threshold': metric.alert_threshold
                })
            
            # Generate insights using AI
            insights = await self.ai_analyzer.generate_insights(metrics_data)
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return []
    
    def _identify_alerts(self, metrics: List[WBRMetric]) -> List[str]:
        """Identify metrics that require attention."""
        alerts = []
        
        for metric in metrics:
            # Check if metric is below alert threshold
            if (metric.alert_threshold is not None and 
                metric.value < metric.alert_threshold):
                alerts.append(
                    f"⚠️ {metric.name} is below threshold: "
                    f"{metric.value:.2f} < {metric.alert_threshold:.2f}"
                )
            
            # Check for significant negative changes
            if (metric.change_percent is not None and 
                metric.change_percent < -10):  # More than 10% decline
                alerts.append(
                    f"📉 {metric.name} declined significantly: "
                    f"{metric.change_percent:.1f}%"
                )
            
            # Check missing target metrics
            if (metric.target is not None and 
                metric.value < metric.target * 0.8):  # 20% below target
                alerts.append(
                    f"🎯 {metric.name} is off target: "
                    f"{metric.value:.2f} vs target {metric.target:.2f}"
                )
        
        return alerts
    
    async def _collect_raw_data(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Collect raw data for detailed analysis."""
        raw_data = {}
        
        try:
            if self.snowflake:
                # Collect key raw datasets
                raw_data['user_activity'] = self.snowflake.execute_query(
                    """
                    SELECT DATE_TRUNC('day', event_timestamp) as date,
                           COUNT(DISTINCT user_id) as unique_users,
                           COUNT(*) as total_events
                    FROM user_events
                    WHERE event_timestamp >= %s AND event_timestamp <= %s
                    GROUP BY 1
                    ORDER BY 1 DESC
                    """,
                    params={'start_date': start_date, 'end_date': end_date}
                )
                
                raw_data['revenue_trends'] = self.snowflake.execute_query(
                    """
                    SELECT DATE_TRUNC('day', created_at) as date,
                           SUM(amount) as daily_revenue,
                           COUNT(DISTINCT user_id) as paying_users
                    FROM transactions
                    WHERE created_at >= %s AND created_at <= %s
                    GROUP BY 1
                    ORDER BY 1 DESC
                    """,
                    params={'start_date': start_date, 'end_date': end_date}
                )
            
            if self.jira:
                # Collect Jira raw data
                raw_data['sprint_details'] = pd.DataFrame()  # Would implement based on Jira connector
                
        except Exception as e:
            logger.error(f"Failed to collect raw data: {e}")
        
        return raw_data
    
    def _calculate_data_freshness(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, datetime]:
        """Calculate freshness timestamp for each data source."""
        freshness = {}
        
        for source, df in raw_data.items():
            if not df.empty and 'date' in df.columns:
                freshness[source] = df['date'].max()
            else:
                freshness[source] = datetime.now()
        
        return freshness
    
    def get_metric_summary(self, metrics: List[WBRMetric]) -> Dict[str, Any]:
        """Get summary statistics for metrics."""
        if not metrics:
            return {}
        
        summary = {
            'total_metrics': len(metrics),
            'metrics_with_targets': sum(1 for m in metrics if m.target is not None),
            'metrics_above_target': sum(
                1 for m in metrics 
                if m.target is not None and m.value >= m.target
            ),
            'metrics_trending_up': sum(1 for m in metrics if m.trend == "up"),
            'metrics_trending_down': sum(1 for m in metrics if m.trend == "down"),
            'avg_change_percent': sum(
                m.change_percent for m in metrics 
                if m.change_percent is not None
            ) / sum(1 for m in metrics if m.change_percent is not None),
            'data_sources': list(set(m.source for m in metrics))
        }
        
        return summary