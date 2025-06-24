"""
Eisenhower Matrix Classifier for PM tasks
Categorizes tasks into four quadrants: Important/Urgent, Important/Not Urgent, 
Not Important/Urgent, Not Important/Not Urgent
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .task_extractor import Task

logger = logging.getLogger(__name__)


class ImportanceLevel(Enum):
    """Importance levels for Eisenhower Matrix"""
    HIGH = 2    # Aligns with goals, strategic value, long-term impact
    LOW = 1     # Minor impact, little strategic value


class UrgencyLevel(Enum):
    """Urgency levels for Eisenhower Matrix"""
    HIGH = 2    # Time-sensitive, deadlines, immediate action needed
    LOW = 1     # Can be scheduled, no immediate deadline


class EisenhowerQuadrant(Enum):
    """Four quadrants of the Eisenhower Matrix"""
    DO_FIRST = "Do First"           # Important & Urgent (Quadrant 1)
    SCHEDULE = "Schedule"           # Important & Not Urgent (Quadrant 2)
    DELEGATE = "Delegate"           # Not Important & Urgent (Quadrant 3)
    ELIMINATE = "Eliminate"         # Not Important & Not Urgent (Quadrant 4)


@dataclass
class EisenhowerClassification:
    """Classification result for a task"""
    importance: int
    urgency: int
    quadrant: EisenhowerQuadrant
    confidence_score: float  # 0.0 to 1.0
    reasoning: str


class EisenhowerMatrixClassifier:
    """Classifies tasks using the Eisenhower Matrix framework"""
    
    def __init__(self):
        # Keywords that indicate high importance (strategic, goal-aligned, high impact)
        self.importance_keywords = {
            ImportanceLevel.HIGH: [
                # Strategic and goal-oriented
                'strategic', 'strategy', 'goal', 'objective', 'okr', 'kpi',
                'milestone', 'roadmap', 'vision', 'mission',
                # High impact
                'customer', 'revenue', 'growth', 'market', 'competitive',
                'business value', 'roi', 'impact', 'critical',
                # Leadership and decision making
                'decision', 'leadership', 'direction', 'priority',
                'architecture', 'design', 'planning',
                # Long-term value
                'improvement', 'optimization', 'efficiency', 'automation',
                'scalability', 'performance', 'quality',
                # Career and skill development
                'learning', 'development', 'skill', 'training', 'expertise',
                'career', 'growth', 'mentoring', 'coaching'
            ],
            ImportanceLevel.LOW: [
                # Administrative and routine
                'administrative', 'routine', 'housekeeping', 'cleanup',
                'maintenance', 'filing', 'organizing', 'sorting',
                # Low-impact activities
                'nice to have', 'minor', 'small', 'trivial', 'cosmetic',
                # Interruptions and distractions
                'interruption', 'distraction', 'social', 'break',
                'coffee', 'chat', 'gossip',
                # Busy work
                'busy work', 'filler', 'time killer', 'procrastination'
            ]
        }
        
        # Keywords that indicate high urgency (time-sensitive, deadlines)
        self.urgency_keywords = {
            UrgencyLevel.HIGH: [
                # Immediate action required
                'urgent', 'asap', 'immediately', 'now', 'today',
                'emergency', 'critical', 'crisis',
                # Deadlines and time pressure
                'deadline', 'due', 'overdue', 'late', 'behind schedule',
                'time sensitive', 'end of day', 'eod', 'end of week', 'eow',
                # Blocking others
                'blocking', 'blocker', 'waiting', 'dependency',
                # External pressure
                'client', 'customer', 'stakeholder', 'executive',
                'board', 'investors', 'audit', 'compliance',
                # Reactive work
                'fire', 'firefighting', 'incident', 'outage',
                'bug fix', 'hotfix', 'patch'
            ],
            UrgencyLevel.LOW: [
                # Flexible timing
                'when time permits', 'someday', 'maybe', 'future',
                'next week', 'next month', 'next quarter',
                'backlog', 'nice to have', 'low priority',
                # Proactive work
                'proactive', 'preventive', 'improvement',
                'optimization', 'research', 'exploration',
                # Planning and preparation
                'planning', 'preparation', 'analysis', 'research',
                'documentation', 'process improvement'
            ]
        }
        
        # Context clues for importance and urgency
        self.context_patterns = {
            'deadline_today': r'\b(today|asap|immediately|now|urgent)\b',
            'deadline_thisweek': r'\b(this week|end of week|eow|by friday)\b',
            'strategic_value': r'\b(strategic|goal|objective|okr|roadmap)\b',
            'customer_impact': r'\b(customer|client|user experience|revenue)\b',
            'blocking_others': r'\b(blocking|blocker|waiting|dependency)\b',
            'routine_work': r'\b(routine|administrative|housekeeping|cleanup)\b'
        }
    
    def classify_task(self, task: Task) -> EisenhowerClassification:
        """Classify a single task into Eisenhower Matrix quadrant"""
        # Combine task content, context, and metadata for analysis
        text_to_analyze = f"{task.content} {task.context} {task.file_path}".lower()
        
        # Score importance and urgency
        importance_score, importance_reasoning = self._score_importance(text_to_analyze, task)
        urgency_score, urgency_reasoning = self._score_urgency(text_to_analyze, task)
        
        # Determine quadrant
        quadrant = self._determine_quadrant(importance_score, urgency_score)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(text_to_analyze, task, quadrant)
        
        # Combine reasoning
        reasoning = f"Importance: {importance_reasoning}. Urgency: {urgency_reasoning}"
        
        return EisenhowerClassification(
            importance=importance_score,
            urgency=urgency_score,
            quadrant=quadrant,
            confidence_score=confidence,
            reasoning=reasoning
        )
    
    def _score_importance(self, text: str, task: Task) -> Tuple[int, str]:
        """Score task importance and provide reasoning"""
        importance_indicators = []
        reasoning_parts = []
        
        # Check keyword matches
        high_matches = sum(1 for keyword in self.importance_keywords[ImportanceLevel.HIGH] 
                          if keyword in text)
        low_matches = sum(1 for keyword in self.importance_keywords[ImportanceLevel.LOW] 
                         if keyword in text)
        
        if high_matches > 0:
            importance_indicators.append(ImportanceLevel.HIGH.value)
            reasoning_parts.append(f"High importance keywords found ({high_matches})")
        
        if low_matches > 0:
            importance_indicators.append(ImportanceLevel.LOW.value)
            reasoning_parts.append(f"Low importance keywords found ({low_matches})")
        
        # Context pattern analysis
        if re.search(self.context_patterns['strategic_value'], text):
            importance_indicators.append(ImportanceLevel.HIGH.value)
            reasoning_parts.append("Strategic value indicated")
        
        if re.search(self.context_patterns['customer_impact'], text):
            importance_indicators.append(ImportanceLevel.HIGH.value)  
            reasoning_parts.append("Customer impact indicated")
        
        if re.search(self.context_patterns['routine_work'], text):
            importance_indicators.append(ImportanceLevel.LOW.value)
            reasoning_parts.append("Routine work indicated")
        
        # Product area context
        if task.product_area in ["DFP 2.0", "Payment Protection", "Global Identity Intelligence"]:
            importance_indicators.append(ImportanceLevel.HIGH.value)
            reasoning_parts.append(f"Strategic product area: {task.product_area}")
        
        # Task type context
        if task.task_type == "Planning/Strategy":
            importance_indicators.append(ImportanceLevel.HIGH.value)
            reasoning_parts.append("Strategic task type")
        elif task.task_type in ["Follow-up/Communication", "Administrative"]:
            importance_indicators.append(ImportanceLevel.LOW.value)
            reasoning_parts.append("Lower importance task type")
        
        # Determine final score
        if not importance_indicators:
            final_score = ImportanceLevel.HIGH.value  # Default to high importance
            reasoning = "No clear importance indicators, defaulting to high"
        else:
            # Use the most frequent or highest score
            final_score = max(importance_indicators)
            reasoning = "; ".join(reasoning_parts)
        
        return final_score, reasoning
    
    def _score_urgency(self, text: str, task: Task) -> Tuple[int, str]:
        """Score task urgency and provide reasoning"""
        urgency_indicators = []
        reasoning_parts = []
        
        # Check keyword matches
        high_matches = sum(1 for keyword in self.urgency_keywords[UrgencyLevel.HIGH] 
                          if keyword in text)
        low_matches = sum(1 for keyword in self.urgency_keywords[UrgencyLevel.LOW] 
                         if keyword in text)
        
        if high_matches > 0:
            urgency_indicators.append(UrgencyLevel.HIGH.value)
            reasoning_parts.append(f"High urgency keywords found ({high_matches})")
        
        if low_matches > 0:
            urgency_indicators.append(UrgencyLevel.LOW.value)
            reasoning_parts.append(f"Low urgency keywords found ({low_matches})")
        
        # Context pattern analysis
        if re.search(self.context_patterns['deadline_today'], text):
            urgency_indicators.append(UrgencyLevel.HIGH.value)
            reasoning_parts.append("Immediate deadline indicated")
        
        if re.search(self.context_patterns['deadline_thisweek'], text):
            urgency_indicators.append(UrgencyLevel.HIGH.value)
            reasoning_parts.append("This week deadline indicated")
        
        if re.search(self.context_patterns['blocking_others'], text):
            urgency_indicators.append(UrgencyLevel.HIGH.value)
            reasoning_parts.append("Blocking others indicated")
        
        # Determine final score
        if not urgency_indicators:
            final_score = UrgencyLevel.LOW.value  # Default to low urgency
            reasoning = "No clear urgency indicators, defaulting to low"
        else:
            # Use the highest urgency score if any high urgency indicators
            final_score = max(urgency_indicators)
            reasoning = "; ".join(reasoning_parts)
        
        return final_score, reasoning
    
    def _determine_quadrant(self, importance: int, urgency: int) -> EisenhowerQuadrant:
        """Determine Eisenhower Matrix quadrant based on importance and urgency scores"""
        if importance == ImportanceLevel.HIGH.value and urgency == UrgencyLevel.HIGH.value:
            return EisenhowerQuadrant.DO_FIRST
        elif importance == ImportanceLevel.HIGH.value and urgency == UrgencyLevel.LOW.value:
            return EisenhowerQuadrant.SCHEDULE
        elif importance == ImportanceLevel.LOW.value and urgency == UrgencyLevel.HIGH.value:
            return EisenhowerQuadrant.DELEGATE
        else:  # Low importance, Low urgency
            return EisenhowerQuadrant.ELIMINATE
    
    def _calculate_confidence(self, text: str, task: Task, quadrant: EisenhowerQuadrant) -> float:
        """Calculate confidence score for the classification"""
        confidence_factors = 0
        total_factors = 0
        
        # Factor 1: Keyword strength
        total_factors += 1
        high_imp_matches = sum(1 for keyword in self.importance_keywords[ImportanceLevel.HIGH] 
                              if keyword in text)
        high_urg_matches = sum(1 for keyword in self.urgency_keywords[UrgencyLevel.HIGH] 
                              if keyword in text)
        
        if high_imp_matches > 0 or high_urg_matches > 0:
            confidence_factors += 0.5 + min(0.5, (high_imp_matches + high_urg_matches) * 0.1)
        
        # Factor 2: Context pattern matches
        total_factors += 1
        pattern_matches = sum(1 for pattern in self.context_patterns.values() 
                             if re.search(pattern, text))
        if pattern_matches > 0:
            confidence_factors += min(1.0, pattern_matches * 0.3)
        
        # Factor 3: Task metadata quality
        total_factors += 1
        if task.product_area and task.task_type:
            confidence_factors += 0.8
        elif task.product_area or task.task_type:
            confidence_factors += 0.5
        else:
            confidence_factors += 0.2
        
        # Factor 4: Content length (more content = better analysis)
        total_factors += 1
        content_length = len(task.content)
        if content_length > 100:
            confidence_factors += 0.9
        elif content_length > 50:
            confidence_factors += 0.7
        elif content_length > 20:
            confidence_factors += 0.5
        else:
            confidence_factors += 0.3
        
        return confidence_factors / total_factors
    
    def classify_tasks(self, tasks: List[Task]) -> Dict[str, List[Dict[str, Any]]]:
        """Classify multiple tasks and group by quadrant"""
        quadrants = {
            EisenhowerQuadrant.DO_FIRST.value: [],
            EisenhowerQuadrant.SCHEDULE.value: [],
            EisenhowerQuadrant.DELEGATE.value: [],
            EisenhowerQuadrant.ELIMINATE.value: []
        }
        
        for task in tasks:
            classification = self.classify_task(task)
            
            task_info = {
                'task': task,
                'classification': classification,
                'content': task.content,
                'file_path': task.file_path,
                'product_area': task.product_area,
                'task_type': task.task_type,
                'confidence': classification.confidence_score,
                'reasoning': classification.reasoning
            }
            
            quadrants[classification.quadrant.value].append(task_info)
        
        # Sort each quadrant by confidence score (descending)
        for quadrant_tasks in quadrants.values():
            quadrant_tasks.sort(key=lambda x: x['confidence'], reverse=True)
        
        return quadrants
    
    def generate_matrix_report(self, tasks: List[Task]) -> Dict[str, Any]:
        """Generate comprehensive Eisenhower Matrix report"""
        if not tasks:
            return {'error': 'No tasks provided'}
        
        classified_tasks = self.classify_tasks(tasks)
        
        # Calculate statistics
        total_tasks = len(tasks)
        quadrant_counts = {quadrant: len(task_list) 
                          for quadrant, task_list in classified_tasks.items()}
        
        # Calculate average confidence by quadrant
        quadrant_confidence = {}
        for quadrant, task_list in classified_tasks.items():
            if task_list:
                avg_confidence = sum(item['confidence'] for item in task_list) / len(task_list)
                quadrant_confidence[quadrant] = avg_confidence
            else:
                quadrant_confidence[quadrant] = 0.0
        
        # Action recommendations
        recommendations = self._generate_action_recommendations(classified_tasks)
        
        return {
            'summary': {
                'total_tasks': total_tasks,
                'quadrant_distribution': quadrant_counts,
                'average_confidence_by_quadrant': quadrant_confidence
            },
            'quadrants': classified_tasks,
            'recommendations': recommendations,
            'action_priorities': {
                'immediate_action': len(classified_tasks[EisenhowerQuadrant.DO_FIRST.value]),
                'schedule_soon': len(classified_tasks[EisenhowerQuadrant.SCHEDULE.value]),
                'consider_delegation': len(classified_tasks[EisenhowerQuadrant.DELEGATE.value]),
                'candidates_for_elimination': len(classified_tasks[EisenhowerQuadrant.ELIMINATE.value])
            }
        }
    
    def _generate_action_recommendations(self, classified_tasks: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """Generate actionable recommendations based on task distribution"""
        recommendations = {}
        
        do_first_count = len(classified_tasks[EisenhowerQuadrant.DO_FIRST.value])
        schedule_count = len(classified_tasks[EisenhowerQuadrant.SCHEDULE.value])
        delegate_count = len(classified_tasks[EisenhowerQuadrant.DELEGATE.value])
        eliminate_count = len(classified_tasks[EisenhowerQuadrant.ELIMINATE.value])
        
        # Do First (Q1) recommendations
        if do_first_count > 5:
            recommendations['do_first'] = "High number of urgent/important tasks. Focus on completing these immediately and consider if any can be delegated."
        elif do_first_count > 0:
            recommendations['do_first'] = "Address these urgent and important tasks first today."
        else:
            recommendations['do_first'] = "Great! No urgent/important tasks in backlog."
        
        # Schedule (Q2) recommendations
        if schedule_count > 10:
            recommendations['schedule'] = "Many important but not urgent tasks. Schedule dedicated time blocks for these to prevent them from becoming urgent."
        elif schedule_count > 0:
            recommendations['schedule'] = "Schedule time for these important tasks to maximize long-term value."
        else:
            recommendations['schedule'] = "Consider adding more strategic/important tasks to your backlog."
        
        # Delegate (Q3) recommendations
        if delegate_count > 0:
            recommendations['delegate'] = f"Consider delegating these {delegate_count} urgent but less important tasks to free up time for high-value work."
        else:
            recommendations['delegate'] = "No obvious delegation candidates identified."
        
        # Eliminate (Q4) recommendations
        if eliminate_count > 0:
            recommendations['eliminate'] = f"Review these {eliminate_count} low-priority tasks. Consider eliminating or significantly delaying them."
        else:
            recommendations['eliminate'] = "No low-priority tasks identified for elimination."
        
        return recommendations