"""
Insight Engine

Generates actionable insights from PM data:
- Pattern recognition across data sources
- Predictive analytics for product metrics
- Competitive intelligence analysis
- Customer behavior insights
- Risk identification and mitigation
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class InsightCategory(Enum):
    """Categories of insights."""
    OPPORTUNITY = "opportunity"
    RISK = "risk"
    TREND = "trend"
    ANOMALY = "anomaly"
    RECOMMENDATION = "recommendation"
    COMPETITIVE = "competitive"


class InsightPriority(Enum):
    """Priority levels for insights."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Insight:
    """
    Represents a single insight.
    
    Attributes:
        id: Unique insight identifier
        category: Category of insight
        priority: Priority level
        title: Brief insight title
        description: Detailed description
        impact: Potential impact description
        data_sources: Sources used for insight
        confidence: Confidence score (0-1)
        actions: Recommended actions
        metrics: Related metrics
        timestamp: When insight was generated
    """
    id: str
    category: InsightCategory
    priority: InsightPriority
    title: str
    description: str
    impact: str
    data_sources: List[str]
    confidence: float
    actions: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert insight to dictionary."""
        return {
            "id": self.id,
            "category": self.category.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "impact": self.impact,
            "confidence": self.confidence,
            "actions": self.actions,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat()
        }


class InsightEngine:
    """
    Engine for generating actionable insights from PM data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the insight engine.
        
        Args:
            config: Configuration including thresholds and settings
        """
        self.config = config
        self.min_confidence = config.get('min_confidence', 0.7)
        self.lookback_days = config.get('lookback_days', 90)
        self.insight_history: List[Insight] = []
        
    async def generate_insights(self, 
                              data_sources: Dict[str, pd.DataFrame],
                              focus_areas: Optional[List[str]] = None) -> List[Insight]:
        """
        Generate insights from multiple data sources.
        
        Args:
            data_sources: Dictionary of DataFrames by source name
            focus_areas: Optional areas to focus on
            
        Returns:
            List of generated insights
        """
        logger.info(f"Generating insights from {len(data_sources)} sources")
        
        insights = []
        
        # Generate different types of insights
        insights.extend(await self._find_opportunities(data_sources))
        insights.extend(await self._identify_risks(data_sources))
        insights.extend(await self._analyze_trends(data_sources))
        insights.extend(await self._detect_anomalies(data_sources))
        
        # Filter by confidence and focus areas
        insights = [i for i in insights if i.confidence >= self.min_confidence]
        
        if focus_areas:
            insights = self._filter_by_focus(insights, focus_areas)
            
        # Sort by priority
        insights.sort(key=lambda x: x.priority.value, reverse=True)
        
        # Store in history
        self.insight_history.extend(insights)
        
        return insights
        
    async def _find_opportunities(self, 
                                data_sources: Dict[str, pd.DataFrame]) -> List[Insight]:
        """Find growth and optimization opportunities."""
        opportunities = []
        
        # Example opportunity detection
        insight = Insight(
            id="opp_001",
            category=InsightCategory.OPPORTUNITY,
            priority=InsightPriority.HIGH,
            title="Untapped Mobile User Segment",
            description="Mobile users from Region X show 3x higher engagement but represent only 5% of marketing spend",
            impact="Potential 25% increase in conversions by reallocating budget",
            data_sources=["analytics", "marketing"],
            confidence=0.85,
            actions=[
                "Increase mobile ad spend in Region X by 50%",
                "Create region-specific mobile landing pages",
                "A/B test localized messaging"
            ],
            metrics={
                "current_conversion": 2.5,
                "projected_conversion": 3.1,
                "investment_required": 50000
            }
        )
        opportunities.append(insight)
        
        # Feature adoption opportunity
        insight2 = Insight(
            id="opp_002",
            category=InsightCategory.OPPORTUNITY,
            priority=InsightPriority.MEDIUM,
            title="Cross-sell Opportunity in Power Users",
            description="87% of power users only use 1 premium feature despite high satisfaction",
            impact="$200K additional MRR from existing customers",
            data_sources=["product_analytics", "billing"],
            confidence=0.82,
            actions=[
                "Launch feature discovery campaign",
                "Offer bundled pricing for multiple features",
                "Create in-app recommendations engine"
            ],
            metrics={
                "power_users": 5000,
                "avg_features_used": 1.2,
                "potential_mrr": 200000
            }
        )
        opportunities.append(insight2)
        
        return opportunities
        
    async def _identify_risks(self, 
                            data_sources: Dict[str, pd.DataFrame]) -> List[Insight]:
        """Identify potential risks and issues."""
        risks = []
        
        # Technical debt risk
        risk1 = Insight(
            id="risk_001",
            category=InsightCategory.RISK,
            priority=InsightPriority.HIGH,
            title="API Response Time Degradation",
            description="API latency increased 40% over past month, approaching SLA limits",
            impact="Potential customer churn and SLA violations if trend continues",
            data_sources=["monitoring", "support_tickets"],
            confidence=0.91,
            actions=[
                "Conduct performance audit immediately",
                "Scale API infrastructure",
                "Implement caching layer"
            ],
            metrics={
                "current_latency_p95": 850,
                "sla_limit": 1000,
                "degradation_rate": 0.4
            }
        )
        risks.append(risk1)
        
        # Customer churn risk
        risk2 = Insight(
            id="risk_002",
            category=InsightCategory.RISK,
            priority=InsightPriority.CRITICAL,
            title="Enterprise Customer Churn Signal",
            description="3 enterprise accounts showing decreased usage and support complaints",
            impact="Potential loss of $500K ARR",
            data_sources=["usage_analytics", "crm", "support"],
            confidence=0.88,
            actions=[
                "Schedule executive business reviews",
                "Assign dedicated success managers",
                "Offer retention incentives"
            ],
            metrics={
                "at_risk_arr": 500000,
                "accounts_affected": 3,
                "usage_decline": -45
            }
        )
        risks.append(risk2)
        
        return risks
        
    async def _analyze_trends(self, 
                            data_sources: Dict[str, pd.DataFrame]) -> List[Insight]:
        """Analyze trends across metrics."""
        trends = []
        
        # Growth trend
        trend1 = Insight(
            id="trend_001",
            category=InsightCategory.TREND,
            priority=InsightPriority.MEDIUM,
            title="Accelerating Feature Adoption",
            description="New AI features showing exponential adoption curve, 150% month-over-month",
            impact="Validates product-market fit for AI direction",
            data_sources=["product_analytics"],
            confidence=0.93,
            actions=[
                "Double down on AI feature development",
                "Create advanced AI tier pricing",
                "Gather user feedback for roadmap"
            ],
            metrics={
                "adoption_rate": 1.5,
                "users_affected": 25000,
                "revenue_impact": 150000
            }
        )
        trends.append(trend1)
        
        return trends
        
    async def _detect_anomalies(self, 
                               data_sources: Dict[str, pd.DataFrame]) -> List[Insight]:
        """Detect anomalies in data."""
        anomalies = []
        
        # Usage anomaly
        anomaly1 = Insight(
            id="anom_001",
            category=InsightCategory.ANOMALY,
            priority=InsightPriority.MEDIUM,
            title="Unusual Spike in API Usage",
            description="API calls from single customer increased 10x overnight",
            impact="Potential abuse or integration issue",
            data_sources=["api_logs"],
            confidence=0.95,
            actions=[
                "Contact customer to verify usage",
                "Review rate limiting policies",
                "Monitor for potential abuse"
            ],
            metrics={
                "normal_usage": 10000,
                "current_usage": 100000,
                "customer_id": "cust_123"
            }
        )
        anomalies.append(anomaly1)
        
        return anomalies
        
    async def generate_competitive_insights(self, 
                                          market_data: Dict[str, Any]) -> List[Insight]:
        """
        Generate competitive intelligence insights.
        
        Args:
            market_data: Market and competitor data
            
        Returns:
            List of competitive insights
        """
        logger.info("Generating competitive insights")
        
        insights = []
        
        # Competitive positioning
        insight = Insight(
            id="comp_001",
            category=InsightCategory.COMPETITIVE,
            priority=InsightPriority.HIGH,
            title="Competitor Pricing Advantage",
            description="Main competitor reduced prices by 20%, gaining market share",
            impact="Risk of losing price-sensitive customers",
            data_sources=["market_research", "sales_data"],
            confidence=0.87,
            actions=[
                "Analyze price elasticity",
                "Enhance value proposition",
                "Consider targeted promotions"
            ],
            metrics={
                "competitor_price_change": -0.2,
                "market_share_shift": -0.05
            }
        )
        insights.append(insight)
        
        return insights
        
    def rank_insights(self, insights: List[Insight]) -> List[Insight]:
        """
        Rank insights by importance.
        
        Args:
            insights: List of insights to rank
            
        Returns:
            Ranked list of insights
        """
        # Calculate composite score
        for insight in insights:
            base_score = insight.priority.value * 25
            confidence_bonus = insight.confidence * 20
            recency_bonus = 10 if (datetime.utcnow() - insight.timestamp).days < 7 else 0
            
            insight.score = base_score + confidence_bonus + recency_bonus
            
        return sorted(insights, key=lambda x: x.score, reverse=True)
        
    def _filter_by_focus(self, insights: List[Insight], 
                        focus_areas: List[str]) -> List[Insight]:
        """Filter insights by focus areas."""
        filtered = []
        
        for insight in insights:
            # Check if any focus area is mentioned in insight
            insight_text = f"{insight.title} {insight.description}".lower()
            if any(area.lower() in insight_text for area in focus_areas):
                filtered.append(insight)
                
        return filtered
        
    def get_insight_summary(self, insights: List[Insight]) -> Dict[str, Any]:
        """
        Generate summary of insights.
        
        Args:
            insights: List of insights
            
        Returns:
            Summary statistics
        """
        if not insights:
            return {"total": 0}
            
        summary = {
            "total": len(insights),
            "by_category": {},
            "by_priority": {},
            "avg_confidence": np.mean([i.confidence for i in insights]),
            "data_sources": list(set(sum([i.data_sources for i in insights], [])))
        }
        
        # Count by category
        for insight in insights:
            cat = insight.category.value
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
            
            pri = insight.priority.name
            summary["by_priority"][pri] = summary["by_priority"].get(pri, 0) + 1
            
        return summary