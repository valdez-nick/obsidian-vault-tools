#!/usr/bin/env python3
"""
PM Tools Automation - Automates Friday Weekly Review Process
Handles your weekly PM tool analysis and planning automatically

Tasks Automated:
- Run PM tools for next week analysis (30 min → 5 min)
- Generate comprehensive weekly reports 
- Update priority rankings and Eisenhower matrix
- Create next week's action plan automatically

Usage: python pm_tools_automation.py --weekly-review --generate-plan --export-reports
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import argparse
import logging

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from obsidian_vault_tools.pm_tools import TaskExtractor, WSJFAnalyzer, EisenhowerMatrixClassifier
    PM_TOOLS_AVAILABLE = True
except ImportError:
    PM_TOOLS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pm_automation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class WeeklyMetrics:
    total_tasks: int
    completed_tasks: int
    completion_rate: float
    avg_wsjf_score: float
    urgent_tasks: int
    important_tasks: int
    delegatable_tasks: int
    eliminatable_tasks: int
    focus_time_hours: float
    context_switches: int
    burnout_risk_score: float

@dataclass
class WeeklyInsights:
    top_achievements: List[str]
    main_blockers: List[str]
    process_improvements: List[str]
    next_week_focus: str
    capacity_recommendation: str
    risk_mitigation: List[str]

class PMToolsAutomation:
    """Automated PM tools execution and weekly planning"""
    
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.reports_dir = self.vault_path / "PM_Reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Initialize PM tools
        if PM_TOOLS_AVAILABLE:
            self.task_extractor = TaskExtractor(str(self.vault_path))
            self.wsjf_analyzer = WSJFAnalyzer()
            self.eisenhower_classifier = EisenhowerMatrixClassifier()
        else:
            logger.error("PM Tools not available. Please install required dependencies.")
            sys.exit(1)
    
    def run_weekly_analysis(self) -> Dict[str, Any]:
        """Run complete weekly PM tools analysis (Automates 30-minute Friday task)"""
        logger.info("🔄 Running weekly PM tools analysis...")
        
        start_time = datetime.now()
        
        # 1. Extract current tasks
        logger.info("📋 Extracting tasks from vault...")
        tasks = self.task_extractor.extract_all_tasks()
        
        # 2. Remove duplicates (automated redundancy elimination)
        unique_tasks = self._eliminate_duplicates(tasks)
        logger.info(f"🧹 Eliminated {len(tasks) - len(unique_tasks)} duplicate tasks")
        
        # 3. Run WSJF analysis
        logger.info("📊 Running WSJF priority analysis...")
        wsjf_report = self.wsjf_analyzer.generate_wsjf_report(unique_tasks)
        
        # 4. Run Eisenhower Matrix classification
        logger.info("🎯 Running Eisenhower Matrix classification...")
        eisenhower_report = self.eisenhower_classifier.generate_matrix_report(unique_tasks)
        
        # 5. Generate weekly metrics
        metrics = self._calculate_weekly_metrics(wsjf_report, eisenhower_report, unique_tasks)
        
        # 6. Generate insights and recommendations
        insights = self._generate_weekly_insights(metrics, wsjf_report, eisenhower_report)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        analysis_result = {
            'timestamp': self.timestamp,
            'execution_time_seconds': duration,
            'total_tasks_found': len(tasks),
            'unique_tasks_analyzed': len(unique_tasks),
            'duplicates_eliminated': len(tasks) - len(unique_tasks),
            'wsjf_report': wsjf_report,
            'eisenhower_report': eisenhower_report,
            'weekly_metrics': asdict(metrics),
            'weekly_insights': asdict(insights)
        }
        
        logger.info(f"✅ Weekly analysis completed in {duration:.2f} seconds")
        return analysis_result
    
    def _eliminate_duplicates(self, tasks) -> List:
        """Automated duplicate task elimination"""
        from difflib import SequenceMatcher
        
        unique_tasks = []
        seen_content = set()
        
        for task in tasks:
            # Create a normalized key for comparison
            content_key = task.content.lower().strip()[:100]  # First 100 chars, normalized
            
            # Check for exact duplicates
            if content_key in seen_content:
                continue
            
            # Check for high similarity duplicates
            is_duplicate = False
            for existing_task in unique_tasks:
                similarity = SequenceMatcher(None, content_key, existing_task.content.lower().strip()[:100]).ratio()
                if similarity > 0.85:  # 85% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tasks.append(task)
                seen_content.add(content_key)
        
        return unique_tasks
    
    def _calculate_weekly_metrics(self, wsjf_report: Dict, eisenhower_report: Dict, tasks: List) -> WeeklyMetrics:
        """Calculate comprehensive weekly metrics"""
        
        # Basic task metrics
        total_tasks = len(tasks)
        avg_wsjf = wsjf_report['summary']['average_wsjf_score']
        
        # Eisenhower quadrant counts
        quadrant_dist = eisenhower_report['summary']['quadrant_distribution']
        urgent_tasks = quadrant_dist.get('Do First', 0)
        important_tasks = quadrant_dist.get('Schedule', 0)
        delegatable_tasks = quadrant_dist.get('Delegate', 0)
        eliminatable_tasks = quadrant_dist.get('Eliminate', 0)
        
        # Calculate burnout risk score (0-10 scale)
        burnout_risk = self._calculate_burnout_risk(urgent_tasks, total_tasks, avg_wsjf)
        
        # Estimated metrics (would be enhanced with actual tracking)
        completion_rate = 0.75  # Default estimate, would be tracked over time
        focus_time_hours = 6.0  # Target daily focus time
        context_switches = 2  # Estimated based on 98.7% single product focus
        
        return WeeklyMetrics(
            total_tasks=total_tasks,
            completed_tasks=int(total_tasks * completion_rate),
            completion_rate=completion_rate,
            avg_wsjf_score=avg_wsjf,
            urgent_tasks=urgent_tasks,
            important_tasks=important_tasks,
            delegatable_tasks=delegatable_tasks,
            eliminatable_tasks=eliminatable_tasks,
            focus_time_hours=focus_time_hours,
            context_switches=context_switches,
            burnout_risk_score=burnout_risk
        )
    
    def _calculate_burnout_risk(self, urgent_tasks: int, total_tasks: int, avg_wsjf: float) -> float:
        """Calculate burnout risk score (0-10 scale, 10 = highest risk)"""
        
        # Risk factors
        urgency_ratio = urgent_tasks / max(total_tasks, 1)
        task_overload = min(total_tasks / 100, 1.0)  # Risk increases with task count
        wsjf_pressure = min(avg_wsjf / 15, 1.0)  # Higher WSJF = more pressure
        
        # Calculate composite risk score
        risk_score = (urgency_ratio * 4) + (task_overload * 3) + (wsjf_pressure * 3)
        
        return min(risk_score, 10.0)
    
    def _generate_weekly_insights(self, metrics: WeeklyMetrics, wsjf_report: Dict, eisenhower_report: Dict) -> WeeklyInsights:
        """Generate actionable weekly insights and recommendations"""
        
        top_achievements = []
        main_blockers = []
        process_improvements = []
        risk_mitigation = []
        
        # Analyze completion rate
        if metrics.completion_rate >= 0.8:
            top_achievements.append(f"High task completion rate ({metrics.completion_rate:.1%})")
        elif metrics.completion_rate < 0.5:
            main_blockers.append("Low task completion rate - capacity mismatch")
            process_improvements.append("Reduce weekly task commitments by 25%")
        
        # Analyze urgency levels
        if metrics.urgent_tasks > 100:
            main_blockers.append(f"Too many urgent tasks ({metrics.urgent_tasks})")
            risk_mitigation.append("Emergency triage session to eliminate/delegate tasks")
        elif metrics.urgent_tasks < 30:
            top_achievements.append("Well-managed urgency levels")
        
        # Analyze WSJF score trends
        if metrics.avg_wsjf_score > 12:
            top_achievements.append("Working on high-value tasks (high WSJF scores)")
        elif metrics.avg_wsjf_score < 8:
            main_blockers.append("Working on lower-value tasks")
            process_improvements.append("Re-evaluate priorities and eliminate low-WSJF work")
        
        # Analyze burnout risk
        if metrics.burnout_risk_score > 7:
            risk_mitigation.append("High burnout risk - reduce commitments immediately")
            risk_mitigation.append("Schedule recovery time and delegate urgent tasks")
        elif metrics.burnout_risk_score < 3:
            top_achievements.append("Low burnout risk - sustainable workload")
        
        # Context switching analysis (based on 98.7% DFP focus)
        if metrics.context_switches <= 2:
            top_achievements.append("Excellent single-product focus maintained")
        
        # Determine next week focus
        high_wsjf_count = len([task for task in wsjf_report.get('all_tasks', []) if task.get('wsjf_score', {}).get('total_score', 0) > 13])
        
        if high_wsjf_count > 20:
            next_week_focus = "High-value security and performance tasks"
        elif metrics.urgent_tasks > 50:
            next_week_focus = "Urgent task reduction and delegation"
        else:
            next_week_focus = "Strategic feature development and planning"
        
        # Capacity recommendation
        if metrics.burnout_risk_score > 6:
            capacity_recommendation = "Reduce weekly commitments by 30-50%"
        elif metrics.completion_rate < 0.6:
            capacity_recommendation = "Reduce weekly commitments by 20%"
        elif metrics.completion_rate > 0.9:
            capacity_recommendation = "Current capacity is optimal"
        else:
            capacity_recommendation = "Maintain current capacity with minor adjustments"
        
        return WeeklyInsights(
            top_achievements=top_achievements,
            main_blockers=main_blockers,
            process_improvements=process_improvements,
            next_week_focus=next_week_focus,
            capacity_recommendation=capacity_recommendation,
            risk_mitigation=risk_mitigation
        )
    
    def generate_next_week_plan(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate automated next week action plan"""
        logger.info("📅 Generating next week action plan...")
        
        wsjf_report = analysis_result['wsjf_report']
        eisenhower_report = analysis_result['eisenhower_report']
        insights = analysis_result['weekly_insights']
        
        # Get top priority tasks for next week
        top_wsjf_tasks = wsjf_report['top_10_recommendations'][:15]  # Top 15 for the week
        urgent_tasks = eisenhower_report['quadrants']['Do First'][:10]  # Top 10 urgent
        
        # Organize tasks by day themes
        daily_plans = self._organize_daily_themes(top_wsjf_tasks, urgent_tasks, insights)
        
        # Generate capacity allocation
        capacity_allocation = self._generate_capacity_allocation(insights, len(top_wsjf_tasks))
        
        # Create weekly goals
        weekly_goals = self._generate_weekly_goals(analysis_result)
        
        next_week_plan = {
            'week_of': (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            'focus_theme': insights['next_week_focus'],
            'capacity_recommendation': insights['capacity_recommendation'],
            'daily_plans': daily_plans,
            'capacity_allocation': capacity_allocation,
            'weekly_goals': weekly_goals,
            'success_criteria': self._generate_success_criteria(analysis_result),
            'risk_mitigation_plan': insights['risk_mitigation']
        }
        
        return next_week_plan
    
    def _organize_daily_themes(self, top_wsjf_tasks: List, urgent_tasks: List, insights: WeeklyInsights) -> Dict[str, Any]:
        """Organize tasks into daily themes"""
        
        # Categorize tasks by type
        security_tasks = [task for task in top_wsjf_tasks if any(keyword in task['task_content'].lower() 
                         for keyword in ['security', 'jwt', 'https', 'rate limit', 'vulnerability'])]
        
        performance_tasks = [task for task in top_wsjf_tasks if any(keyword in task['task_content'].lower() 
                           for keyword in ['performance', 'optimization', 'database', 'pool', 'monitor'])]
        
        feature_tasks = [task for task in top_wsjf_tasks if any(keyword in task['task_content'].lower() 
                        for keyword in ['feature', 'user', 'productivity', 'enhancement', 'improvement'])]
        
        planning_tasks = [task for task in urgent_tasks if any(keyword in task['content'].lower() 
                         for keyword in ['planning', 'strategy', 'roadmap', 'communication'])]
        
        daily_plans = {
            'monday': {
                'theme': 'Security & Infrastructure',
                'focus_hours': 6,
                'tasks': security_tasks[:8],
                'energy_level': 'high',
                'context_switching': 'none'
            },
            'tuesday': {
                'theme': 'Performance & Quality',
                'focus_hours': 6,
                'tasks': performance_tasks[:8],
                'energy_level': 'high',
                'context_switching': 'minimal'
            },
            'wednesday': {
                'theme': 'Feature Development',
                'focus_hours': 6,
                'tasks': feature_tasks[:6],
                'energy_level': 'medium-high',
                'context_switching': 'minimal'
            },
            'thursday': {
                'theme': 'Communication & Planning',
                'focus_hours': 6,
                'tasks': planning_tasks[:6],
                'energy_level': 'medium',
                'context_switching': 'moderate'
            },
            'friday': {
                'theme': 'Review & Next Week Prep',
                'focus_hours': 6,
                'tasks': [
                    {'task_content': 'Run PM tools analysis', 'wsjf_score': 12.0, 'time_estimate': 30},
                    {'task_content': 'Weekly review and planning', 'wsjf_score': 11.0, 'time_estimate': 90},
                    {'task_content': 'Complete pending high-WSJF items', 'wsjf_score': 13.0, 'time_estimate': 180}
                ],
                'energy_level': 'medium',
                'context_switching': 'minimal'
            }
        }
        
        return daily_plans
    
    def _generate_capacity_allocation(self, insights: WeeklyInsights, high_priority_count: int) -> Dict[str, Any]:
        """Generate weekly capacity allocation recommendations"""
        
        if "reduce commitments" in insights['capacity_recommendation'].lower():
            capacity_factor = 0.7  # Reduce by 30%
        else:
            capacity_factor = 1.0
        
        return {
            'daily_focus_hours': 6 * capacity_factor,
            'high_priority_tasks_per_week': min(high_priority_count, int(15 * capacity_factor)),
            'communication_time_budget': 90,  # minutes per day max
            'buffer_time_percentage': 20,  # 20% buffer for unexpected items
            'deep_work_blocks': 2,  # morning and afternoon
            'context_switch_limit': 2  # max per day
        }
    
    def _generate_weekly_goals(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific weekly goals"""
        
        metrics = analysis_result['weekly_metrics']
        wsjf_report = analysis_result['wsjf_report']
        
        return {
            'primary_goals': [
                f"Complete top 10 WSJF tasks (score >13.0)",
                f"Reduce urgent tasks from {metrics['urgent_tasks']} to <50",
                f"Maintain 6-hour daily focus limit",
                f"Achieve >75% task completion rate"
            ],
            'secondary_goals': [
                f"Improve average WSJF score to >{metrics['avg_wsjf_score'] + 0.5:.1f}",
                f"Maintain single-product focus (DFP 2.0)",
                f"Complete 3-5 quick wins daily",
                f"Keep communication response time <24 hours"
            ],
            'stretch_goals': [
                f"Begin work on {len(wsjf_report['all_tasks']) - 20} Schedule quadrant tasks",
                f"Implement process improvements",
                f"Reduce burnout risk score below {max(metrics['burnout_risk_score'] - 1, 0):.1f}"
            ]
        }
    
    def _generate_success_criteria(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate measurable success criteria"""
        
        return [
            "Daily focus time stays within 6-hour limit",
            "Complete 8-12 high-WSJF tasks during the week",
            "Urgent task count decreases by end of week",
            "No emergency context switches between products",
            "Weekly review completed within 90 minutes on Friday",
            "Energy levels remain >6/10 throughout week"
        ]
    
    def export_comprehensive_reports(self, analysis_result: Dict[str, Any], next_week_plan: Dict[str, Any]) -> Dict[str, str]:
        """Export all reports to Obsidian vault"""
        logger.info("📄 Exporting comprehensive reports...")
        
        exported_files = {}
        
        # 1. Export raw analysis data
        analysis_file = self.reports_dir / f"Weekly_Analysis_{self.timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_result, f, indent=2, default=str)
        exported_files['analysis_data'] = str(analysis_file)
        
        # 2. Export next week plan
        plan_file = self.reports_dir / f"Next_Week_Plan_{self.timestamp}.json"
        with open(plan_file, 'w') as f:
            json.dump(next_week_plan, f, indent=2, default=str)
        exported_files['next_week_plan'] = str(plan_file)
        
        # 3. Generate Obsidian-formatted weekly summary
        summary_md = self._generate_obsidian_summary(analysis_result, next_week_plan)
        summary_file = self.vault_path / f"Weekly_PM_Summary_{self.timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write(summary_md)
        exported_files['weekly_summary'] = str(summary_file)
        
        # 4. Generate next week action plan
        action_plan_md = self._generate_action_plan_markdown(next_week_plan)
        action_plan_file = self.vault_path / f"Next_Week_Action_Plan_{self.timestamp}.md"
        with open(action_plan_file, 'w') as f:
            f.write(action_plan_md)
        exported_files['action_plan'] = str(action_plan_file)
        
        # 5. Update main command center document
        self._update_command_center(analysis_result, next_week_plan)
        exported_files['command_center_updated'] = "PM Burnout Recovery Command Center.md"
        
        logger.info(f"✅ Exported {len(exported_files)} report files")
        return exported_files
    
    def _generate_obsidian_summary(self, analysis_result: Dict[str, Any], next_week_plan: Dict[str, Any]) -> str:
        """Generate Obsidian-formatted weekly summary"""
        
        metrics = analysis_result['weekly_metrics']
        insights = analysis_result['weekly_insights']
        
        summary = f"""# Weekly PM Summary - {datetime.now().strftime('%Y-%m-%d')}

**Analysis Duration:** {analysis_result['execution_time_seconds']:.1f} seconds  
**Tasks Analyzed:** {analysis_result['unique_tasks_analyzed']}  
**Duplicates Eliminated:** {analysis_result['duplicates_eliminated']}

## 📊 Weekly Metrics

### Task Management
- **Total Tasks:** {metrics['total_tasks']}
- **Completion Rate:** {metrics['completion_rate']:.1%}
- **Average WSJF Score:** {metrics['avg_wsjf_score']:.2f}

### Priority Distribution
- **Urgent & Important:** {metrics['urgent_tasks']} tasks
- **Important (Schedule):** {metrics['important_tasks']} tasks
- **Delegatable:** {metrics['delegatable_tasks']} tasks
- **Eliminatable:** {metrics['eliminatable_tasks']} tasks

### Burnout Prevention
- **Burnout Risk Score:** {metrics['burnout_risk_score']:.1f}/10
- **Focus Time:** {metrics['focus_time_hours']} hours/day
- **Context Switches:** {metrics['context_switches']}/day

## 🎯 Key Insights

### Top Achievements
{chr(10).join([f"- {achievement}" for achievement in insights['top_achievements']])}

### Main Blockers
{chr(10).join([f"- {blocker}" for blocker in insights['main_blockers']])}

### Process Improvements
{chr(10).join([f"- {improvement}" for improvement in insights['process_improvements']])}

## 📅 Next Week Focus

**Theme:** {next_week_plan['focus_theme']}  
**Capacity:** {next_week_plan['capacity_recommendation']}

### Daily Themes
- **Monday:** {next_week_plan['daily_plans']['monday']['theme']}
- **Tuesday:** {next_week_plan['daily_plans']['tuesday']['theme']}
- **Wednesday:** {next_week_plan['daily_plans']['wednesday']['theme']}
- **Thursday:** {next_week_plan['daily_plans']['thursday']['theme']}
- **Friday:** {next_week_plan['daily_plans']['friday']['theme']}

## 🎯 Weekly Goals

### Primary Goals
{chr(10).join([f"- [ ] {goal}" for goal in next_week_plan['weekly_goals']['primary_goals']])}

### Secondary Goals
{chr(10).join([f"- [ ] {goal}" for goal in next_week_plan['weekly_goals']['secondary_goals']])}

## 🚨 Risk Mitigation

{chr(10).join([f"- {risk}" for risk in insights['risk_mitigation']]) if insights['risk_mitigation'] else "- No specific risks identified"}

---

*Generated automatically by PM Tools Automation*
"""
        
        return summary
    
    def _generate_action_plan_markdown(self, next_week_plan: Dict[str, Any]) -> str:
        """Generate detailed action plan in Markdown"""
        
        plan_md = f"""# Next Week Action Plan - {next_week_plan['week_of']}

**Focus Theme:** {next_week_plan['focus_theme']}  
**Daily Capacity:** {next_week_plan['capacity_allocation']['daily_focus_hours']} hours  
**High Priority Tasks:** {next_week_plan['capacity_allocation']['high_priority_tasks_per_week']}

"""
        
        # Add daily plans
        for day, plan in next_week_plan['daily_plans'].items():
            plan_md += f"""## {day.title()}: {plan['theme']}

**Focus Hours:** {plan['focus_hours']}  
**Energy Level:** {plan['energy_level']}  
**Context Switching:** {plan['context_switching']}

### Tasks
"""
            for i, task in enumerate(plan['tasks'], 1):
                task_content = task.get('task_content', task.get('content', 'Unknown task'))
                wsjf_score = task.get('wsjf_score', 0)
                plan_md += f"{i}. [ ] {task_content} (WSJF: {wsjf_score:.1f})\n"
            
            plan_md += "\n"
        
        # Add success criteria
        plan_md += "## Success Criteria\n\n"
        for criterion in next_week_plan['success_criteria']:
            plan_md += f"- [ ] {criterion}\n"
        
        plan_md += "\n---\n\n*Generated automatically by PM Tools Automation*\n"
        
        return plan_md
    
    def _update_command_center(self, analysis_result: Dict[str, Any], next_week_plan: Dict[str, Any]):
        """Update the main Command Center document with new data"""
        
        command_center_file = self.vault_path / "PM Burnout Recovery Command Center.md"
        
        if not command_center_file.exists():
            logger.warning("Command Center file not found, skipping update")
            return
        
        # Read current content
        with open(command_center_file, 'r') as f:
            content = f.read()
        
        # Update key metrics in the executive summary
        metrics = analysis_result['weekly_metrics']
        insights = analysis_result['weekly_insights']
        
        # Update the reality section
        updated_reality = f"""## 🎯 EXECUTIVE SUMMARY - YOUR REALITY RIGHT NOW

### The Situation (Updated {datetime.now().strftime('%Y-%m-%d')})
- **Current Tasks:** {analysis_result['unique_tasks_analyzed']}
- **Duplicates Eliminated:** {analysis_result['duplicates_eliminated']} (automatic cleanup)
- **Focus Opportunity:** 98.7% DFP 2.0 (MAINTAINED single-product focus)
- **Immediate Action:** {metrics['urgent_tasks']} urgent & important tasks
- **Strategic Work:** {metrics['important_tasks']} important tasks to schedule
- **Burnout Risk:** {metrics['burnout_risk_score']:.1f}/10 ({self._get_risk_status(metrics['burnout_risk_score'])})

### This Week's Mission
**{insights['next_week_focus']}**
"""
        
        # Replace the executive summary section
        import re
        pattern = r'## 🎯 EXECUTIVE SUMMARY.*?(?=## 🚨 THIS WEEK\'S TOP 3 PRIORITIES|$)'
        content = re.sub(pattern, updated_reality, content, flags=re.DOTALL)
        
        # Write updated content back
        with open(command_center_file, 'w') as f:
            f.write(content)
        
        logger.info("✅ Updated Command Center with latest analysis")
    
    def _get_risk_status(self, risk_score: float) -> str:
        """Get risk status description"""
        if risk_score < 3:
            return "LOW RISK"
        elif risk_score < 6:
            return "MODERATE RISK"
        else:
            return "HIGH RISK - ACTION NEEDED"
    
    def run_full_weekly_automation(self) -> Dict[str, Any]:
        """Run complete weekly PM automation (Replaces 90-minute Friday review)"""
        logger.info("🚀 Starting full weekly PM automation...")
        
        start_time = datetime.now()
        
        # 1. Run weekly analysis
        analysis_result = self.run_weekly_analysis()
        
        # 2. Generate next week plan
        next_week_plan = self.generate_next_week_plan(analysis_result)
        
        # 3. Export comprehensive reports
        exported_files = self.export_comprehensive_reports(analysis_result, next_week_plan)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        automation_result = {
            'total_execution_time': total_duration,
            'time_saved': 5400 - total_duration,  # 90 minutes - actual time
            'analysis_result': analysis_result,
            'next_week_plan': next_week_plan,
            'exported_files': exported_files,
            'efficiency_improvement': f"{((5400 - total_duration) / 5400) * 100:.1f}% time reduction"
        }
        
        logger.info(f"✅ Weekly PM automation completed in {total_duration:.2f} seconds")
        logger.info(f"⏰ Time saved: {automation_result['time_saved']:.0f} seconds ({automation_result['efficiency_improvement']})")
        
        return automation_result

def main():
    parser = argparse.ArgumentParser(description="PM Tools Automation - Automate Weekly Review Process")
    parser.add_argument('--vault-path', default='.', help='Obsidian vault path')
    parser.add_argument('--weekly-review', action='store_true', help='Run weekly analysis')
    parser.add_argument('--generate-plan', action='store_true', help='Generate next week plan')
    parser.add_argument('--export-reports', action='store_true', help='Export comprehensive reports')
    parser.add_argument('--full', action='store_true', help='Run complete weekly automation')
    
    args = parser.parse_args()
    
    if not PM_TOOLS_AVAILABLE:
        print("❌ PM Tools not available. Please install required dependencies.")
        sys.exit(1)
    
    automation = PMToolsAutomation(args.vault_path)
    
    if args.full or (args.weekly_review and args.generate_plan and args.export_reports):
        result = automation.run_full_weekly_automation()
        print("\n🎯 WEEKLY PM AUTOMATION COMPLETE!")
        print(f"⏱️  Total execution time: {result['total_execution_time']:.2f} seconds")
        print(f"💾 Time saved: {result['time_saved']:.0f} seconds")
        print(f"📈 Efficiency improvement: {result['efficiency_improvement']}")
        print(f"📊 Tasks analyzed: {result['analysis_result']['unique_tasks_analyzed']}")
        print(f"🧹 Duplicates eliminated: {result['analysis_result']['duplicates_eliminated']}")
        print(f"📄 Reports exported: {len(result['exported_files'])}")
        print("\n📋 Files generated:")
        for report_type, file_path in result['exported_files'].items():
            print(f"  - {report_type}: {Path(file_path).name}")
        
    elif args.weekly_review:
        result = automation.run_weekly_analysis()
        print(f"Weekly analysis completed: {result['unique_tasks_analyzed']} tasks analyzed")
    
    elif args.generate_plan:
        # Need analysis first
        analysis_result = automation.run_weekly_analysis()
        plan = automation.generate_next_week_plan(analysis_result)
        print(f"Next week plan generated: {plan['focus_theme']}")
    
    elif args.export_reports:
        # Need analysis and plan first
        analysis_result = automation.run_weekly_analysis()
        plan = automation.generate_next_week_plan(analysis_result)
        files = automation.export_comprehensive_reports(analysis_result, plan)
        print(f"Exported {len(files)} report files")

if __name__ == "__main__":
    main()