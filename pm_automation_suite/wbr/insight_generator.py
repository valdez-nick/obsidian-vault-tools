"""
Insight Generator Implementation

AI-powered analysis engine for generating business insights from WBR metrics.
Provides trend analysis, anomaly detection, natural language insights, and executive summaries.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Types of insights."""
    TREND = "trend"
    ANOMALY = "anomaly" 
    CORRELATION = "correlation"
    FORECAST = "forecast"
    RECOMMENDATION = "recommendation"
    ALERT = "alert"


class InsightPriority(Enum):
    """Priority levels for insights."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Insight:
    """Data class for business insights."""
    title: str
    description: str
    insight_type: InsightType
    priority: InsightPriority
    confidence: float  # 0.0 to 1.0
    supporting_data: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime
    metrics_involved: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert insight to dictionary."""
        return {
            'title': self.title,
            'description': self.description,
            'type': self.insight_type.value,
            'priority': self.priority.value,
            'confidence': self.confidence,
            'supporting_data': self.supporting_data,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat(),
            'metrics_involved': self.metrics_involved
        }


class InsightGenerator:
    """
    AI-powered insight generator for WBR metrics.
    
    Combines statistical analysis with Large Language Models to generate
    actionable business insights from metric data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Insight Generator.
        
        Args:
            config: Configuration dictionary with AI settings
        """
        self.config = config
        self.openai_client = None
        self.anthropic_client = None
        
        # Analysis parameters
        self.anomaly_threshold = config.get('anomaly_threshold', 2.0)  # Standard deviations
        self.trend_min_periods = config.get('trend_min_periods', 4)
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        
        # Initialize AI clients
        self._initialize_ai_clients()
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
    def _initialize_ai_clients(self):
        """Initialize AI service clients."""
        try:
            if OPENAI_AVAILABLE and 'openai_api_key' in self.config:
                openai.api_key = self.config['openai_api_key']
                self.openai_client = openai
                logger.info("OpenAI client initialized")
                
            if ANTHROPIC_AVAILABLE and 'anthropic_api_key' in self.config:
                self.anthropic_client = anthropic.Client(
                    api_key=self.config['anthropic_api_key']
                )
                logger.info("Anthropic client initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize AI clients: {e}")
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load AI prompt templates."""
        return {
            'insight_generation': """
You are a senior data analyst generating insights for a Weekly Business Review. 
Analyze the following metrics and provide actionable insights.

Metrics Data:
{metrics_data}

Focus on:
1. Significant trends (>10% change)
2. Metrics below targets
3. Correlations between metrics
4. Potential causes and recommendations

Provide insights in this format:
- **Title**: Brief insight title
- **Description**: 2-3 sentence explanation
- **Priority**: Critical/High/Medium/Low
- **Recommendations**: Specific action items

Generate 3-5 key insights prioritized by business impact.
""",
            
            'executive_summary': """
Create an executive summary for the Weekly Business Review based on these insights:

{insights_data}

Key Metrics Performance:
{metrics_summary}

Write a concise 3-paragraph executive summary covering:
1. Overall performance vs targets
2. Key wins and areas of concern
3. Priority actions for next week

Keep it under 200 words, focusing on actionable insights for leadership.
""",
            
            'anomaly_explanation': """
Explain this anomaly in business terms:

Metric: {metric_name}
Current Value: {current_value}
Expected Range: {expected_range}
Historical Context: {historical_data}

Provide a 2-sentence explanation of what might be causing this anomaly and suggest 2 specific actions to investigate.
""",
            
            'trend_analysis': """
Analyze this trend and provide business context:

Metric: {metric_name}
Trend: {trend_direction}
Change: {change_percent}%
Time Period: {time_period}

Historical Pattern: {historical_pattern}

Explain in 2-3 sentences what this trend means for the business and provide 2 specific recommendations.
"""
        }
    
    async def generate_comprehensive_insights(
        self, 
        metrics_data: List[Dict[str, Any]],
        historical_data: Optional[pd.DataFrame] = None
    ) -> List[Insight]:
        """
        Generate comprehensive insights from metrics data.
        
        Args:
            metrics_data: List of metric dictionaries
            historical_data: Historical metrics for trend analysis
            
        Returns:
            List of generated insights
        """
        logger.info("Starting comprehensive insight generation")
        
        all_insights = []
        
        try:
            # Convert to DataFrame for analysis
            metrics_df = pd.DataFrame(metrics_data)
            
            # Generate different types of insights in parallel
            tasks = [
                self._detect_anomalies(metrics_df),
                self._analyze_trends(metrics_df, historical_data),
                self._find_correlations(metrics_df),
                self._generate_recommendations(metrics_df),
                self._create_ai_insights(metrics_data)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine all insights
            for result in results:
                if isinstance(result, list):
                    all_insights.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Insight generation error: {result}")
            
            # Sort by priority and confidence
            all_insights.sort(
                key=lambda x: (x.priority.value, -x.confidence)
            )
            
            # Limit to top insights
            top_insights = all_insights[:10]
            
            logger.info(f"Generated {len(top_insights)} insights")
            return top_insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return []
    
    async def _detect_anomalies(self, metrics_df: pd.DataFrame) -> List[Insight]:
        """Detect anomalies in metrics using statistical methods."""
        insights = []
        
        try:
            # Only analyze numeric metrics with values
            numeric_metrics = metrics_df.select_dtypes(include=[np.number])
            if numeric_metrics.empty:
                return insights
            
            # Use Isolation Forest for anomaly detection
            if len(numeric_metrics) > 1:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_metrics.fillna(0))
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_scores = iso_forest.fit_predict(scaled_data)
                
                # Find anomalous metrics
                for idx, score in enumerate(anomaly_scores):
                    if score == -1:  # Anomaly detected
                        metric_name = metrics_df.iloc[idx]['name']
                        metric_value = metrics_df.iloc[idx]['value']
                        
                        insights.append(Insight(
                            title=f"Anomaly Detected: {metric_name}",
                            description=f"{metric_name} shows unusual behavior with value {metric_value:.2f}",
                            insight_type=InsightType.ANOMALY,
                            priority=InsightPriority.HIGH,
                            confidence=0.8,
                            supporting_data={'metric': metric_name, 'value': metric_value},
                            recommendations=[
                                "Investigate data quality issues",
                                "Check for external factors affecting this metric"
                            ],
                            timestamp=datetime.now(),
                            metrics_involved=[metric_name]
                        ))
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
        
        return insights
    
    async def _analyze_trends(
        self, 
        metrics_df: pd.DataFrame,
        historical_data: Optional[pd.DataFrame] = None
    ) -> List[Insight]:
        """Analyze trends in metrics."""
        insights = []
        
        try:
            for _, metric in metrics_df.iterrows():
                if metric.get('change_percent') is not None:
                    change = metric['change_percent']
                    metric_name = metric['name']
                    
                    # Identify significant trends
                    if abs(change) > 15:  # More than 15% change
                        priority = InsightPriority.HIGH if abs(change) > 25 else InsightPriority.MEDIUM
                        trend_direction = "increased" if change > 0 else "decreased"
                        
                        insights.append(Insight(
                            title=f"Significant Trend: {metric_name}",
                            description=f"{metric_name} has {trend_direction} by {abs(change):.1f}% this period",
                            insight_type=InsightType.TREND,
                            priority=priority,
                            confidence=0.9,
                            supporting_data={
                                'change_percent': change,
                                'current_value': metric['value'],
                                'previous_value': metric.get('previous_value')
                            },
                            recommendations=self._get_trend_recommendations(metric_name, change),
                            timestamp=datetime.now(),
                            metrics_involved=[metric_name]
                        ))
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
        
        return insights
    
    async def _find_correlations(self, metrics_df: pd.DataFrame) -> List[Insight]:
        """Find correlations between metrics."""
        insights = []
        
        try:
            # Extract numeric values for correlation analysis
            numeric_data = []
            metric_names = []
            
            for _, metric in metrics_df.iterrows():
                if pd.notna(metric.get('value')):
                    numeric_data.append(metric['value'])
                    metric_names.append(metric['name'])
            
            if len(numeric_data) < 2:
                return insights
            
            # Calculate correlations (simplified - would need historical data for proper correlation)
            # For now, identify metrics that are both trending in same direction
            trending_up = []
            trending_down = []
            
            for _, metric in metrics_df.iterrows():
                if metric.get('trend') == 'up':
                    trending_up.append(metric['name'])
                elif metric.get('trend') == 'down':
                    trending_down.append(metric['name'])
            
            # Identify potential correlations
            if len(trending_up) > 1:
                insights.append(Insight(
                    title="Positive Trend Correlation",
                    description=f"Multiple metrics trending positively: {', '.join(trending_up)}",
                    insight_type=InsightType.CORRELATION,
                    priority=InsightPriority.MEDIUM,
                    confidence=0.7,
                    supporting_data={'correlated_metrics': trending_up},
                    recommendations=[
                        "Identify common success factors driving these improvements",
                        "Scale successful initiatives across other areas"
                    ],
                    timestamp=datetime.now(),
                    metrics_involved=trending_up
                ))
            
            if len(trending_down) > 1:
                insights.append(Insight(
                    title="Negative Trend Correlation",
                    description=f"Multiple metrics trending negatively: {', '.join(trending_down)}",
                    insight_type=InsightType.CORRELATION,
                    priority=InsightPriority.HIGH,
                    confidence=0.7,
                    supporting_data={'correlated_metrics': trending_down},
                    recommendations=[
                        "Investigate systemic issues affecting multiple metrics",
                        "Develop coordinated action plan to address root causes"
                    ],
                    timestamp=datetime.now(),
                    metrics_involved=trending_down
                ))
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
        
        return insights
    
    async def _generate_recommendations(self, metrics_df: pd.DataFrame) -> List[Insight]:
        """Generate specific recommendations based on metric performance."""
        insights = []
        
        try:
            for _, metric in metrics_df.iterrows():
                metric_name = metric['name']
                value = metric['value']
                target = metric.get('target')
                alert_threshold = metric.get('alert_threshold')
                
                # Check if metric is below target
                if target is not None and value < target * 0.9:  # 10% below target
                    gap_percent = ((target - value) / target) * 100
                    
                    insights.append(Insight(
                        title=f"Performance Gap: {metric_name}",
                        description=f"{metric_name} is {gap_percent:.1f}% below target ({value:.2f} vs {target:.2f})",
                        insight_type=InsightType.RECOMMENDATION,
                        priority=InsightPriority.HIGH,
                        confidence=0.9,
                        supporting_data={
                            'current_value': value,
                            'target': target,
                            'gap_percent': gap_percent
                        },
                        recommendations=self._get_performance_recommendations(metric_name, gap_percent),
                        timestamp=datetime.now(),
                        metrics_involved=[metric_name]
                    ))
                
                # Check if metric is below alert threshold
                if alert_threshold is not None and value < alert_threshold:
                    insights.append(Insight(
                        title=f"Alert: {metric_name} Below Threshold",
                        description=f"{metric_name} ({value:.2f}) is below alert threshold ({alert_threshold:.2f})",
                        insight_type=InsightType.ALERT,
                        priority=InsightPriority.CRITICAL,
                        confidence=1.0,
                        supporting_data={
                            'current_value': value,
                            'alert_threshold': alert_threshold
                        },
                        recommendations=[
                            "Investigate immediate causes",
                            "Implement emergency action plan",
                            "Increase monitoring frequency"
                        ],
                        timestamp=datetime.now(),
                        metrics_involved=[metric_name]
                    ))
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
        
        return insights
    
    async def _create_ai_insights(self, metrics_data: List[Dict[str, Any]]) -> List[Insight]:
        """Generate insights using AI language models."""
        insights = []
        
        if not (self.openai_client or self.anthropic_client):
            logger.warning("No AI clients available for insight generation")
            return insights
        
        try:
            # Prepare metrics data for AI analysis
            metrics_summary = self._prepare_metrics_summary(metrics_data)
            
            # Generate insights using AI
            ai_response = await self._call_ai_for_insights(metrics_summary)
            
            if ai_response:
                # Parse AI response into insights
                parsed_insights = self._parse_ai_insights(ai_response)
                insights.extend(parsed_insights)
            
        except Exception as e:
            logger.error(f"AI insight generation failed: {e}")
        
        return insights
    
    async def _call_ai_for_insights(self, metrics_summary: str) -> Optional[str]:
        """Call AI service to generate insights."""
        try:
            if self.openai_client:
                response = await self.openai_client.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a senior business analyst expert at identifying key insights from metrics."
                        },
                        {
                            "role": "user", 
                            "content": self.prompt_templates['insight_generation'].format(
                                metrics_data=metrics_summary
                            )
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                return response.choices[0].message.content
            
            elif self.anthropic_client:
                response = await self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": self.prompt_templates['insight_generation'].format(
                                metrics_data=metrics_summary
                            )
                        }
                    ]
                )
                return response.content[0].text
                
        except Exception as e:
            logger.error(f"AI API call failed: {e}")
            return None
    
    def _prepare_metrics_summary(self, metrics_data: List[Dict[str, Any]]) -> str:
        """Prepare metrics data for AI analysis."""
        summary_lines = []
        
        for metric in metrics_data:
            line = f"- {metric['name']}: {metric['value']:.2f}"
            
            if metric.get('change_percent'):
                change = metric['change_percent']
                direction = "↑" if change > 0 else "↓"
                line += f" ({direction}{abs(change):.1f}%)"
            
            if metric.get('target'):
                target_status = "✓" if metric['value'] >= metric['target'] else "✗"
                line += f" [Target: {metric['target']:.2f} {target_status}]"
            
            summary_lines.append(line)
        
        return "\n".join(summary_lines)
    
    def _parse_ai_insights(self, ai_response: str) -> List[Insight]:
        """Parse AI response into structured insights."""
        insights = []
        
        try:
            # Simple parsing - in practice would use more sophisticated NLP
            lines = ai_response.strip().split('\n')
            current_insight = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('**Title**:'):
                    if current_insight:
                        insights.append(self._create_insight_from_dict(current_insight))
                        current_insight = {}
                    current_insight['title'] = line.replace('**Title**:', '').strip()
                elif line.startswith('**Description**:'):
                    current_insight['description'] = line.replace('**Description**:', '').strip()
                elif line.startswith('**Priority**:'):
                    priority_text = line.replace('**Priority**:', '').strip().lower()
                    current_insight['priority'] = self._parse_priority(priority_text)
                elif line.startswith('**Recommendations**:'):
                    current_insight['recommendations'] = [line.replace('**Recommendations**:', '').strip()]
                elif line.startswith('- ') and 'recommendations' in current_insight:
                    current_insight['recommendations'].append(line[2:].strip())
            
            # Add the last insight
            if current_insight:
                insights.append(self._create_insight_from_dict(current_insight))
            
        except Exception as e:
            logger.error(f"Failed to parse AI insights: {e}")
        
        return insights
    
    def _create_insight_from_dict(self, insight_dict: Dict[str, Any]) -> Insight:
        """Create Insight object from dictionary."""
        return Insight(
            title=insight_dict.get('title', 'AI Generated Insight'),
            description=insight_dict.get('description', ''),
            insight_type=InsightType.RECOMMENDATION,
            priority=insight_dict.get('priority', InsightPriority.MEDIUM),
            confidence=0.7,  # AI insights get medium confidence
            supporting_data={'source': 'ai_analysis'},
            recommendations=insight_dict.get('recommendations', []),
            timestamp=datetime.now(),
            metrics_involved=[]  # Would need more sophisticated parsing to extract
        )
    
    def _parse_priority(self, priority_text: str) -> InsightPriority:
        """Parse priority text into enum."""
        priority_map = {
            'critical': InsightPriority.CRITICAL,
            'high': InsightPriority.HIGH,
            'medium': InsightPriority.MEDIUM,
            'low': InsightPriority.LOW
        }
        return priority_map.get(priority_text, InsightPriority.MEDIUM)
    
    def _get_trend_recommendations(self, metric_name: str, change_percent: float) -> List[str]:
        """Get specific recommendations based on trend analysis."""
        recommendations = []
        
        if "dau" in metric_name.lower() or "mau" in metric_name.lower():
            if change_percent > 0:
                recommendations.extend([
                    "Identify successful acquisition channels and scale them",
                    "Analyze user cohorts to understand retention drivers"
                ])
            else:
                recommendations.extend([
                    "Review user acquisition funnel for drop-off points",
                    "Implement re-engagement campaigns for dormant users"
                ])
        
        elif "revenue" in metric_name.lower():
            if change_percent > 0:
                recommendations.extend([
                    "Analyze high-value customer segments for expansion",
                    "Document successful sales strategies for replication"
                ])
            else:
                recommendations.extend([
                    "Review pricing strategy and competitor positioning",
                    "Implement customer retention programs"
                ])
        
        elif "velocity" in metric_name.lower():
            if change_percent > 0:
                recommendations.extend([
                    "Identify process improvements that increased velocity",
                    "Share best practices across all teams"
                ])
            else:
                recommendations.extend([
                    "Review sprint planning and estimation accuracy",
                    "Identify blockers affecting team productivity"
                ])
        
        else:
            # Generic recommendations
            if change_percent > 0:
                recommendations.append("Continue current strategies and scale successful initiatives")
            else:
                recommendations.append("Investigate root causes and develop improvement plan")
        
        return recommendations
    
    def _get_performance_recommendations(self, metric_name: str, gap_percent: float) -> List[str]:
        """Get specific recommendations based on performance gaps."""
        recommendations = []
        
        if gap_percent > 20:  # Large gap
            recommendations.append("Conduct immediate root cause analysis")
            recommendations.append("Develop emergency action plan with weekly check-ins")
        elif gap_percent > 10:  # Medium gap
            recommendations.append("Review current strategies and optimize execution")
            recommendations.append("Set milestone targets to close the gap incrementally")
        else:  # Small gap
            recommendations.append("Fine-tune current approaches to reach target")
            recommendations.append("Monitor closely to prevent further decline")
        
        return recommendations
    
    async def generate_executive_summary(
        self, 
        insights: List[Insight],
        metrics_summary: Dict[str, Any]
    ) -> str:
        """Generate executive summary from insights."""
        if not (self.openai_client or self.anthropic_client):
            return self._generate_basic_summary(insights, metrics_summary)
        
        try:
            # Prepare insights data
            insights_text = "\n".join([
                f"- {insight.title}: {insight.description}"
                for insight in insights[:5]  # Top 5 insights
            ])
            
            # Generate summary using AI
            summary = await self._call_ai_for_summary(insights_text, metrics_summary)
            return summary or self._generate_basic_summary(insights, metrics_summary)
            
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            return self._generate_basic_summary(insights, metrics_summary)
    
    async def _call_ai_for_summary(
        self, 
        insights_text: str, 
        metrics_summary: Dict[str, Any]
    ) -> Optional[str]:
        """Call AI to generate executive summary."""
        try:
            if self.openai_client:
                response = await self.openai_client.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "user",
                            "content": self.prompt_templates['executive_summary'].format(
                                insights_data=insights_text,
                                metrics_summary=str(metrics_summary)
                            )
                        }
                    ],
                    max_tokens=300,
                    temperature=0.3
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"AI summary generation failed: {e}")
            return None
    
    def _generate_basic_summary(
        self, 
        insights: List[Insight], 
        metrics_summary: Dict[str, Any]
    ) -> str:
        """Generate basic summary without AI."""
        summary_parts = []
        
        # Performance overview
        total_metrics = metrics_summary.get('total_metrics', 0)
        above_target = metrics_summary.get('metrics_above_target', 0)
        
        if total_metrics > 0:
            target_rate = (above_target / total_metrics) * 100
            summary_parts.append(
                f"Performance Overview: {above_target}/{total_metrics} metrics ({target_rate:.0f}%) "
                f"are meeting targets this week."
            )
        
        # Key insights
        critical_insights = [i for i in insights if i.priority == InsightPriority.CRITICAL]
        high_insights = [i for i in insights if i.priority == InsightPriority.HIGH]
        
        if critical_insights:
            summary_parts.append(
                f"Critical Issues: {len(critical_insights)} critical items require immediate attention."
            )
        
        if high_insights:
            summary_parts.append(
                f"Key Focus Areas: {len(high_insights)} high-priority improvements identified."
            )
        
        # Action items
        all_recommendations = []
        for insight in insights[:3]:  # Top 3 insights
            all_recommendations.extend(insight.recommendations[:1])  # First recommendation from each
        
        if all_recommendations:
            summary_parts.append(f"Priority Actions: {'; '.join(all_recommendations[:3])}.")
        
        return " ".join(summary_parts)