#!/usr/bin/env python3
"""
Burnout Pattern Detection System
Early warning system to prevent PM overwhelm
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json

@dataclass
class BurnoutIndicators:
    """Indicators of potential burnout"""
    task_accumulation_rate: float  # New tasks per day
    completion_rate: float  # Percentage of tasks completed
    context_switch_frequency: int  # Product area changes per day
    overdue_task_count: int  # Tasks past their due date
    urgent_task_ratio: float  # Percentage of tasks marked urgent
    working_hours_pattern: List[int]  # Hours when tasks are created/modified
    task_age_distribution: Dict[str, int]  # Age buckets of incomplete tasks
    energy_indicators: List[str]  # Words/phrases indicating low energy

@dataclass
class BurnoutRiskAssessment:
    """Burnout risk assessment results"""
    risk_score: float  # 0-10 scale
    risk_level: str  # Low, Medium, High, Critical
    primary_factors: List[str]  # Main contributing factors
    recommendations: List[str]  # Specific actions to take
    trend: str  # Improving, Stable, Worsening
    detailed_metrics: Dict[str, Any]

class BurnoutDetector:
    """Detect burnout patterns in PM work"""
    
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        
        # Burnout indicator thresholds
        self.thresholds = {
            'task_accumulation_high': 15,  # >15 new tasks/day is concerning
            'completion_rate_low': 0.3,  # <30% completion is problematic
            'context_switches_high': 10,  # >10 switches/day causes fatigue
            'overdue_ratio_high': 0.2,  # >20% overdue is stressful
            'urgent_ratio_high': 0.4,  # >40% urgent indicates firefighting
            'old_task_days': 30,  # Tasks >30 days old are stale
        }
        
        # Energy indicator patterns
        self.low_energy_patterns = [
            r'\btired\b', r'\bexhausted\b', r'\boverwhelmed\b', r'\bburnt?\s?out\b',
            r'\bstressed\b', r'\banxious\b', r'\bcan\'t\s+sleep\b', r'\bbehind\b',
            r'\btoo\s+much\b', r'\btoo\s+many\b', r'\bswamped\b', r'\bdrowning\b',
            r'\bcrazy\b', r'\binsane\b', r'\bimpossible\b', r'\bgave\s+up\b'
        ]
        
        # Product area patterns
        self.product_patterns = {
            'dfp': [r'dfp', r'device\s+fingerprint', r'fingerprint'],
            'payments': [r'payment', r'fraud', r'chargeback', r'dispute'],
            'identity': [r'identity', r'global\s+identity', r'authentication'],
            'api': [r'api', r'platform', r'integration'],
        }
        
    def analyze_vault(self) -> BurnoutRiskAssessment:
        """Analyze entire vault for burnout patterns"""
        # Extract all notes with dates
        daily_notes = self._get_daily_notes()
        
        # Calculate key metrics
        indicators = self._calculate_burnout_indicators(daily_notes)
        
        # Assess risk level
        risk_assessment = self._assess_burnout_risk(indicators)
        
        return risk_assessment
    
    def _get_daily_notes(self) -> List[Tuple[datetime, Path]]:
        """Get all daily notes with their dates"""
        daily_notes = []
        
        # Common daily note patterns
        date_patterns = [
            (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),
            (r'(\d{4}-\d{1,2}-\d{1,2})', '%Y-%m-%d'),
            (r'(\d{2}-\d{2}-\d{4})', '%m-%d-%Y'),
        ]
        
        for md_file in self.vault_path.rglob("*.md"):
            filename = md_file.stem
            
            # Try to extract date from filename
            for pattern, date_format in date_patterns:
                match = re.search(pattern, filename)
                if match:
                    try:
                        date = datetime.strptime(match.group(1), date_format)
                        daily_notes.append((date, md_file))
                        break
                    except ValueError:
                        continue
        
        return sorted(daily_notes, key=lambda x: x[0])
    
    def _calculate_burnout_indicators(self, daily_notes: List[Tuple[datetime, Path]]) -> BurnoutIndicators:
        """Calculate burnout indicators from daily notes"""
        if not daily_notes:
            return BurnoutIndicators(
                task_accumulation_rate=0,
                completion_rate=0,
                context_switch_frequency=0,
                overdue_task_count=0,
                urgent_task_ratio=0,
                working_hours_pattern=[],
                task_age_distribution={},
                energy_indicators=[]
            )
        
        # Analyze recent period (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        recent_notes = [(date, path) for date, path in daily_notes if date >= cutoff_date]
        
        # Task tracking
        total_tasks = 0
        completed_tasks = 0
        overdue_tasks = 0
        urgent_tasks = 0
        task_ages = []
        
        # Context switching
        daily_contexts = defaultdict(set)
        
        # Energy indicators
        energy_phrases = []
        
        # Working hours (based on task timestamps)
        task_hours = []
        
        for date, note_path in recent_notes:
            try:
                with open(note_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count tasks
                incomplete_tasks = re.findall(r'- \[ \].*', content)
                complete_tasks = re.findall(r'- \[x\].*', content, re.IGNORECASE)
                
                total_tasks += len(incomplete_tasks) + len(complete_tasks)
                completed_tasks += len(complete_tasks)
                
                # Check for overdue and urgent
                for task in incomplete_tasks:
                    if re.search(r'overdue|late|past\s+due', task, re.IGNORECASE):
                        overdue_tasks += 1
                    if re.search(r'urgent|asap|emergency|🔺|⏫|🚨', task, re.IGNORECASE):
                        urgent_tasks += 1
                    
                    # Calculate task age if due date present
                    due_match = re.search(r'📅\s*(\d{4}-\d{2}-\d{2})', task)
                    if due_match:
                        due_date = datetime.strptime(due_match.group(1), '%Y-%m-%d')
                        age_days = (datetime.now() - due_date).days
                        if age_days > 0:
                            task_ages.append(age_days)
                
                # Track product context switches
                for product, patterns in self.product_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            daily_contexts[date].add(product)
                            break
                
                # Check for energy indicators
                for pattern in self.low_energy_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    energy_phrases.extend(matches)
                
                # Extract task creation times if available
                time_matches = re.findall(r'(\d{1,2}):\d{2}', content)
                for time_str in time_matches:
                    try:
                        hour = int(time_str)
                        if 0 <= hour <= 23:
                            task_hours.append(hour)
                    except ValueError:
                        continue
                        
            except Exception as e:
                print(f"Error processing {note_path}: {e}")
                continue
        
        # Calculate metrics
        days_analyzed = len(recent_notes)
        task_accumulation_rate = total_tasks / max(days_analyzed, 1)
        completion_rate = completed_tasks / max(total_tasks, 1)
        
        # Context switching frequency
        total_switches = 0
        for date_contexts in daily_contexts.values():
            if len(date_contexts) > 1:
                total_switches += len(date_contexts) - 1
        context_switch_frequency = total_switches / max(days_analyzed, 1)
        
        # Task age distribution
        age_distribution = {
            '0-7 days': sum(1 for age in task_ages if age <= 7),
            '8-14 days': sum(1 for age in task_ages if 8 <= age <= 14),
            '15-30 days': sum(1 for age in task_ages if 15 <= age <= 30),
            '30+ days': sum(1 for age in task_ages if age > 30),
        }
        
        return BurnoutIndicators(
            task_accumulation_rate=task_accumulation_rate,
            completion_rate=completion_rate,
            context_switch_frequency=int(context_switch_frequency),
            overdue_task_count=overdue_tasks,
            urgent_task_ratio=urgent_tasks / max(total_tasks, 1),
            working_hours_pattern=sorted(set(task_hours)),
            task_age_distribution=age_distribution,
            energy_indicators=list(set(energy_phrases))
        )
    
    def _assess_burnout_risk(self, indicators: BurnoutIndicators) -> BurnoutRiskAssessment:
        """Assess burnout risk based on indicators"""
        risk_score = 0.0
        risk_factors = []
        recommendations = []
        
        # Task accumulation (weight: 2.0)
        if indicators.task_accumulation_rate > self.thresholds['task_accumulation_high']:
            risk_score += 2.0
            risk_factors.append(f"High task accumulation ({indicators.task_accumulation_rate:.1f} tasks/day)")
            recommendations.append("Implement strict task intake limits (max 5 new tasks/day)")
        elif indicators.task_accumulation_rate > 10:
            risk_score += 1.0
            risk_factors.append(f"Moderate task accumulation ({indicators.task_accumulation_rate:.1f} tasks/day)")
        
        # Completion rate (weight: 2.5)
        if indicators.completion_rate < self.thresholds['completion_rate_low']:
            risk_score += 2.5
            risk_factors.append(f"Very low completion rate ({indicators.completion_rate:.1%})")
            recommendations.append("Focus on completing existing tasks before taking new ones")
        elif indicators.completion_rate < 0.5:
            risk_score += 1.5
            risk_factors.append(f"Low completion rate ({indicators.completion_rate:.1%})")
        
        # Context switching (weight: 1.5)
        if indicators.context_switch_frequency > self.thresholds['context_switches_high']:
            risk_score += 1.5
            risk_factors.append(f"Excessive context switching ({indicators.context_switch_frequency} switches/day)")
            recommendations.append("Block time for single product focus (min 2-hour blocks)")
        
        # Overdue tasks (weight: 1.5)
        overdue_ratio = indicators.overdue_task_count / max(indicators.task_accumulation_rate * 30, 1)
        if overdue_ratio > self.thresholds['overdue_ratio_high']:
            risk_score += 1.5
            risk_factors.append(f"Many overdue tasks ({indicators.overdue_task_count} overdue)")
            recommendations.append("Schedule overdue task triage session immediately")
        
        # Urgent task ratio (weight: 1.5)
        if indicators.urgent_task_ratio > self.thresholds['urgent_ratio_high']:
            risk_score += 1.5
            risk_factors.append(f"Too many urgent tasks ({indicators.urgent_task_ratio:.1%} marked urgent)")
            recommendations.append("Re-evaluate urgency criteria - not everything is urgent")
        
        # Old tasks (weight: 1.0)
        old_tasks = indicators.task_age_distribution.get('30+ days', 0)
        if old_tasks > 20:
            risk_score += 1.0
            risk_factors.append(f"Many stale tasks ({old_tasks} tasks >30 days old)")
            recommendations.append("Archive or delegate old tasks that haven't been touched")
        
        # Energy indicators (weight: 1.0)
        if len(indicators.energy_indicators) >= 5:
            risk_score += 1.0
            risk_factors.append(f"Multiple low energy indicators found")
            recommendations.append("Schedule recovery time and consider workload reduction")
        
        # Working hours pattern (weight: 1.0)
        if any(hour < 7 or hour > 20 for hour in indicators.working_hours_pattern):
            risk_score += 1.0
            risk_factors.append("Working outside normal hours")
            recommendations.append("Enforce work hour boundaries (7am-8pm max)")
        
        # Determine risk level
        if risk_score >= 8:
            risk_level = "Critical"
            recommendations.insert(0, "IMMEDIATE ACTION REQUIRED: Reduce workload by 50%")
        elif risk_score >= 6:
            risk_level = "High"
            recommendations.insert(0, "Take action this week to prevent burnout")
        elif risk_score >= 4:
            risk_level = "Medium"
            recommendations.insert(0, "Monitor closely and implement preventive measures")
        else:
            risk_level = "Low"
            recommendations.insert(0, "Maintain current practices with minor adjustments")
        
        # Determine trend (would need historical data for real trend)
        trend = "Unknown"  # Would calculate based on historical assessments
        
        return BurnoutRiskAssessment(
            risk_score=min(risk_score, 10.0),
            risk_level=risk_level,
            primary_factors=risk_factors[:3],  # Top 3 factors
            recommendations=recommendations[:5],  # Top 5 recommendations
            trend=trend,
            detailed_metrics={
                'task_accumulation_rate': indicators.task_accumulation_rate,
                'completion_rate': indicators.completion_rate,
                'context_switch_frequency': indicators.context_switch_frequency,
                'overdue_task_count': indicators.overdue_task_count,
                'urgent_task_ratio': indicators.urgent_task_ratio,
                'task_age_distribution': indicators.task_age_distribution,
                'energy_indicators_count': len(indicators.energy_indicators),
                'working_hours_span': len(indicators.working_hours_pattern)
            }
        )
    
    def generate_burnout_report(self) -> str:
        """Generate comprehensive burnout assessment report"""
        assessment = self.analyze_vault()
        
        report = f"""# Burnout Risk Assessment Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 🚨 Overall Risk Assessment

**Risk Score:** {assessment.risk_score:.1f}/10
**Risk Level:** {assessment.risk_level}
**Trend:** {assessment.trend}

## 📊 Key Metrics

- **Task Accumulation:** {assessment.detailed_metrics['task_accumulation_rate']:.1f} new tasks/day
- **Completion Rate:** {assessment.detailed_metrics['completion_rate']:.1%}
- **Context Switching:** {assessment.detailed_metrics['context_switch_frequency']} switches/day
- **Overdue Tasks:** {assessment.detailed_metrics['overdue_task_count']}
- **Urgent Task Ratio:** {assessment.detailed_metrics['urgent_task_ratio']:.1%}

## 🔍 Primary Risk Factors

"""
        for factor in assessment.primary_factors:
            report += f"- {factor}\n"
        
        report += "\n## 💡 Recommendations\n\n"
        for i, rec in enumerate(assessment.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
## 📈 Task Age Distribution

- **0-7 days:** {assessment.detailed_metrics['task_age_distribution'].get('0-7 days', 0)} tasks
- **8-14 days:** {assessment.detailed_metrics['task_age_distribution'].get('8-14 days', 0)} tasks
- **15-30 days:** {assessment.detailed_metrics['task_age_distribution'].get('15-30 days', 0)} tasks
- **30+ days:** {assessment.detailed_metrics['task_age_distribution'].get('30+ days', 0)} tasks

## 🔄 Next Steps

1. Review and implement top recommendations
2. Schedule follow-up assessment in 1 week
3. Track completion rate improvement
4. Monitor energy levels daily
"""
        
        return report

def main():
    """Test burnout detection"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python burnout_detector.py <vault_path>")
        sys.exit(1)
    
    vault_path = sys.argv[1]
    detector = BurnoutDetector(vault_path)
    
    print("Analyzing vault for burnout patterns...")
    report = detector.generate_burnout_report()
    
    print(report)
    
    # Save report
    report_path = Path(vault_path) / f"Burnout_Assessment_{datetime.now().strftime('%Y%m%d')}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()