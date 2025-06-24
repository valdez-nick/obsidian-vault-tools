"""
WSJF (Weighted Shortest Job First) Analyzer for PM tasks
Prioritizes tasks based on business value, time criticality, risk reduction, and effort
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from .task_extractor import Task

logger = logging.getLogger(__name__)


class BusinessValueScore(Enum):
    """Business value scoring levels"""
    CRITICAL = 5  # Customer impact, revenue impact, strategic alignment
    HIGH = 4      # Important but not critical
    MEDIUM = 3    # Moderate impact  
    LOW = 2       # Nice to have
    MINIMAL = 1   # Low impact


class TimeCriticalityScore(Enum):
    """Time criticality scoring levels"""
    URGENT = 5    # Must be done immediately
    HIGH = 4      # Should be done soon
    MEDIUM = 3    # Moderate time pressure
    LOW = 2       # Can wait
    FLEXIBLE = 1  # No time pressure


class RiskReductionScore(Enum):
    """Risk reduction scoring levels"""
    CRITICAL = 5  # Prevents major issues
    HIGH = 4      # Reduces significant risk
    MEDIUM = 3    # Moderate risk reduction
    LOW = 2       # Minor risk reduction
    MINIMAL = 1   # Little risk impact


class JobSizeScore(Enum):
    """Job size scoring levels (effort estimation)"""
    EXTRA_SMALL = 1  # < 30 min
    SMALL = 2        # 30 min - 2 hours
    MEDIUM = 3       # 2-8 hours
    LARGE = 5        # 1-2 days
    EXTRA_LARGE = 8  # > 2 days


@dataclass
class WSJFScore:
    """WSJF scoring breakdown"""
    business_value: int
    time_criticality: int
    risk_reduction: int
    job_size: int
    total_score: float
    
    def __post_init__(self):
        """Calculate total WSJF score"""
        if self.job_size == 0:
            self.total_score = 0
        else:
            self.total_score = (
                self.business_value + self.time_criticality + self.risk_reduction
            ) / self.job_size


class WSJFAnalyzer:
    """Analyzes tasks and provides WSJF prioritization scores"""
    
    def __init__(self):
        self.business_value_keywords = {
            BusinessValueScore.CRITICAL: [
                'customer', 'revenue', 'security', 'compliance', 'critical',
                'strategic', 'okr', 'goal', 'kpi', 'metric', 'performance',
                'user experience', 'customer experience', 'bug fix', 'outage'
            ],
            BusinessValueScore.HIGH: [
                'feature', 'enhancement', 'improvement', 'optimization',
                'efficiency', 'automation', 'integration', 'launch',
                'release', 'milestone', 'roadmap'
            ],
            BusinessValueScore.MEDIUM: [
                'documentation', 'process', 'workflow', 'standardization',
                'best practice', 'training', 'onboarding', 'analysis'
            ],
            BusinessValueScore.LOW: [
                'research', 'investigation', 'exploration', 'prototype',
                'proof of concept', 'experiment', 'spike'
            ],
            BusinessValueScore.MINIMAL: [
                'cleanup', 'refactor', 'tech debt', 'nice to have',
                'future', 'someday', 'maybe', 'idea'
            ]
        }
        
        self.time_criticality_keywords = {
            TimeCriticalityScore.URGENT: [
                'urgent', 'asap', 'immediately', 'today', 'now', 'emergency',
                'critical', 'blocking', 'blocker', 'deadline today'
            ],
            TimeCriticalityScore.HIGH: [
                'soon', 'this week', 'end of week', 'eow', 'deadline',
                'due', 'time sensitive', 'priority', 'important'
            ],
            TimeCriticalityScore.MEDIUM: [
                'next week', 'this month', 'end of month', 'eom',
                'quarterly', 'q1', 'q2', 'q3', 'q4'
            ],
            TimeCriticalityScore.LOW: [
                'next month', 'next quarter', 'future', 'later',
                'when time permits', 'low priority'
            ],
            TimeCriticalityScore.FLEXIBLE: [
                'someday', 'maybe', 'nice to have', 'if time',
                'backlog', 'future consideration'
            ]
        }
        
        self.risk_reduction_keywords = {
            RiskReductionScore.CRITICAL: [
                'security', 'vulnerability', 'breach', 'compliance',
                'audit', 'legal', 'regulation', 'privacy', 'gdpr'
            ],
            RiskReductionScore.HIGH: [
                'stability', 'reliability', 'performance', 'scalability',
                'monitoring', 'alerting', 'backup', 'disaster recovery'
            ],
            RiskReductionScore.MEDIUM: [
                'testing', 'validation', 'verification', 'quality',
                'documentation', 'knowledge transfer', 'training'
            ],
            RiskReductionScore.LOW: [
                'improvement', 'optimization', 'efficiency',
                'best practice', 'standardization'
            ],
            RiskReductionScore.MINIMAL: [
                'cleanup', 'refactor', 'tech debt', 'maintenance',
                'organization', 'nice to have'
            ]
        }
    
    def analyze_task(self, task: Task) -> WSJFScore:
        """Analyze a single task and return WSJF score"""
        # Combine task content, context, and file path for analysis
        text_to_analyze = f"{task.content} {task.context} {task.file_path}".lower()
        
        # Score each dimension
        business_value = self._score_business_value(text_to_analyze, task)
        time_criticality = self._score_time_criticality(text_to_analyze, task)
        risk_reduction = self._score_risk_reduction(text_to_analyze, task)
        job_size = self._score_job_size(task)
        
        return WSJFScore(
            business_value=business_value,
            time_criticality=time_criticality,
            risk_reduction=risk_reduction,
            job_size=job_size,
            total_score=0  # Will be calculated in __post_init__
        )
    
    def _score_business_value(self, text: str, task: Task) -> int:
        """Score business value based on keywords and context"""
        scores = []
        
        # Check keywords
        for score_level, keywords in self.business_value_keywords.items():
            if any(keyword in text for keyword in keywords):
                scores.append(score_level.value)
        
        # Product area specific scoring
        if task.product_area == "DFP 2.0":
            scores.append(BusinessValueScore.HIGH.value)  # Strategic priority
        elif task.product_area == "Payment Protection":
            scores.append(BusinessValueScore.CRITICAL.value)  # Revenue critical
        elif task.product_area == "Global Identity Intelligence":
            scores.append(BusinessValueScore.HIGH.value)  # Growth area
        
        # Task type specific scoring
        if task.task_type == "Planning/Strategy":
            scores.append(BusinessValueScore.HIGH.value)
        elif task.task_type == "Meeting/Communication":
            scores.append(BusinessValueScore.MEDIUM.value)
        elif task.task_type == "Follow-up/Communication":
            scores.append(BusinessValueScore.LOW.value)
        
        return max(scores) if scores else BusinessValueScore.MEDIUM.value
    
    def _score_time_criticality(self, text: str, task: Task) -> int:
        """Score time criticality based on keywords and context"""
        scores = []
        
        # Check keywords
        for score_level, keywords in self.time_criticality_keywords.items():
            if any(keyword in text for keyword in keywords):
                scores.append(score_level.value)
        
        # Date-based criticality (simple heuristic)
        if any(word in text for word in ['today', 'asap', 'urgent', 'emergency']):
            scores.append(TimeCriticalityScore.URGENT.value)
        elif any(word in text for word in ['week', 'soon', 'deadline']):
            scores.append(TimeCriticalityScore.HIGH.value)
        
        return max(scores) if scores else TimeCriticalityScore.MEDIUM.value
    
    def _score_risk_reduction(self, text: str, task: Task) -> int:
        """Score risk reduction based on keywords and context"""
        scores = []
        
        # Check keywords
        for score_level, keywords in self.risk_reduction_keywords.items():
            if any(keyword in text for keyword in keywords):
                scores.append(score_level.value)
        
        # Task type specific risk scoring
        if task.task_type == "Analysis/Research":
            scores.append(RiskReductionScore.MEDIUM.value)  # Reduces uncertainty
        elif task.task_type == "Documentation":
            scores.append(RiskReductionScore.MEDIUM.value)  # Reduces knowledge risk
        elif task.task_type == "Implementation":
            scores.append(RiskReductionScore.LOW.value)  # Depends on what's being implemented
        
        return max(scores) if scores else RiskReductionScore.LOW.value
    
    def _score_job_size(self, task: Task) -> int:
        """Score job size based on effort estimation"""
        effort_mapping = {
            "Quick (<30min)": JobSizeScore.EXTRA_SMALL.value,
            "Medium (1-2hrs)": JobSizeScore.SMALL.value,
            "Deep (>2hrs)": JobSizeScore.MEDIUM.value
        }
        
        # Default scoring if effort estimation isn't available
        if task.estimated_effort in effort_mapping:
            return effort_mapping[task.estimated_effort]
        
        # Fallback based on content length and complexity
        content_length = len(task.content)
        if content_length < 50:
            return JobSizeScore.EXTRA_SMALL.value
        elif content_length < 100:
            return JobSizeScore.SMALL.value
        elif content_length < 200:
            return JobSizeScore.MEDIUM.value
        else:
            return JobSizeScore.LARGE.value
    
    def prioritize_tasks(self, tasks: List[Task]) -> List[Dict[str, Any]]:
        """Prioritize tasks using WSJF and return sorted list"""
        scored_tasks = []
        
        for task in tasks:
            wsjf_score = self.analyze_task(task)
            scored_tasks.append({
                'task': task,
                'wsjf_score': wsjf_score,
                'priority_rank': None  # Will be set after sorting
            })
        
        # Sort by WSJF score (descending)
        scored_tasks.sort(key=lambda x: x['wsjf_score'].total_score, reverse=True)
        
        # Add priority rank
        for i, item in enumerate(scored_tasks):
            item['priority_rank'] = i + 1
        
        return scored_tasks
    
    def generate_wsjf_report(self, tasks: List[Task]) -> Dict[str, Any]:
        """Generate comprehensive WSJF analysis report"""
        if not tasks:
            return {'error': 'No tasks provided'}
        
        prioritized_tasks = self.prioritize_tasks(tasks)
        
        # Calculate statistics
        total_tasks = len(prioritized_tasks)
        avg_score = sum(item['wsjf_score'].total_score for item in prioritized_tasks) / total_tasks
        
        # Group by score ranges
        high_priority = [item for item in prioritized_tasks if item['wsjf_score'].total_score >= 3.0]
        medium_priority = [item for item in prioritized_tasks if 1.5 <= item['wsjf_score'].total_score < 3.0]
        low_priority = [item for item in prioritized_tasks if item['wsjf_score'].total_score < 1.5]
        
        # Top recommendations
        top_10 = prioritized_tasks[:10]
        
        return {
            'summary': {
                'total_tasks': total_tasks,
                'average_wsjf_score': avg_score,
                'high_priority_count': len(high_priority),
                'medium_priority_count': len(medium_priority),
                'low_priority_count': len(low_priority)
            },
            'top_10_recommendations': [
                {
                    'rank': item['priority_rank'],
                    'task_content': item['task'].content,
                    'file_path': item['task'].file_path,
                    'wsjf_score': item['wsjf_score'].total_score,
                    'business_value': item['wsjf_score'].business_value,
                    'time_criticality': item['wsjf_score'].time_criticality,
                    'risk_reduction': item['wsjf_score'].risk_reduction,
                    'job_size': item['wsjf_score'].job_size,
                    'product_area': item['task'].product_area,
                    'task_type': item['task'].task_type
                }
                for item in top_10
            ],
            'priority_distribution': {
                'high_priority': len(high_priority),
                'medium_priority': len(medium_priority),
                'low_priority': len(low_priority)
            },
            'all_tasks': prioritized_tasks
        }