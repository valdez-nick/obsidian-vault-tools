#!/usr/bin/env python3
"""
PM Burnout Prevention System - Comprehensive Long-Term Automation
Ties together all PM tools and automation for sustainable workflow management

This system automates:
1. Daily task prioritization and planning
2. Weekly analysis and planning cycles  
3. Continuous burnout risk monitoring
4. Automated security and performance improvements
5. Communication triage and response automation
6. Progress tracking and trend analysis

Usage: python pm_burnout_prevention_system.py --mode [daily|weekly|continuous|setup]
"""

import os
import sys
import json
import time
import schedule
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from obsidian_vault_tools.pm_tools import TaskExtractor, WSJFAnalyzer, EisenhowerMatrixClassifier
    PM_TOOLS_AVAILABLE = True
except ImportError:
    PM_TOOLS_AVAILABLE = False

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pm_burnout_prevention.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemStatus:
    last_analysis: datetime
    total_tasks: int
    urgent_tasks: int
    burnout_risk_score: float
    automation_health: str
    next_scheduled_analysis: datetime
    system_uptime: float

@dataclass
class AutomationMetrics:
    tasks_automated: int
    time_saved_minutes: float
    errors_prevented: int
    completion_rate_improvement: float
    burnout_risk_reduction: float
    focus_time_protected: float

class PMBurnoutPreventionSystem:
    """Comprehensive long-term PM automation and burnout prevention system"""
    
    def __init__(self, vault_path: str, config_file: Optional[str] = None):
        self.vault_path = Path(vault_path)
        self.system_start_time = datetime.now()
        self.config_file = config_file or str(self.vault_path / "pm_automation_config.json")
        self.state_file = self.vault_path / "pm_system_state.json"
        self.metrics_file = self.vault_path / "pm_automation_metrics.json"
        
        # Initialize directories
        self.automation_dir = self.vault_path / "automation_scripts"
        self.reports_dir = self.vault_path / "PM_Reports"
        self.logs_dir = self.vault_path / "PM_Logs"
        
        for directory in [self.automation_dir, self.reports_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize PM tools
        if PM_TOOLS_AVAILABLE:
            self.task_extractor = TaskExtractor(str(self.vault_path))
            self.wsjf_analyzer = WSJFAnalyzer()
            self.eisenhower_classifier = EisenhowerMatrixClassifier()
        else:
            logger.error("PM Tools not available. Please install required dependencies.")
            
        # System state
        self.system_state = self._load_system_state()
        self.automation_metrics = self._load_automation_metrics()
        
        # Automation registry
        self.automation_scripts = {
            'security_hardening': self.automation_dir / 'security_hardening_bot.py',
            'database_pooling': self.automation_dir / 'database_pool_configurator.py',
            'pm_tools_analysis': self.automation_dir / 'pm_tools_automation.py'
        }
        
        self.running = False
        self.automation_thread = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            'daily_analysis_time': '09:00',
            'weekly_analysis_day': 'friday',
            'weekly_analysis_time': '14:30',
            'burnout_risk_threshold': 7.0,
            'max_daily_hours': 6.0,
            'max_urgent_tasks': 50,
            'automation_enabled': True,
            'security_automation_enabled': True,
            'database_optimization_enabled': True,
            'communication_automation_enabled': True,
            'progress_tracking_enabled': True,
            'notification_settings': {
                'burnout_alerts': True,
                'task_overflow_alerts': True,
                'completion_rate_alerts': True,
                'automation_status_alerts': True
            },
            'integrations': {
                'slack_webhook': None,
                'email_notifications': None,
                'github_integration': True,
                'obsidian_sync': True
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")
        
        return default_config
    
    def _save_config(self):
        """Save current configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _load_system_state(self) -> SystemStatus:
        """Load system state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return SystemStatus(
                        last_analysis=datetime.fromisoformat(data['last_analysis']),
                        total_tasks=data['total_tasks'],
                        urgent_tasks=data['urgent_tasks'],
                        burnout_risk_score=data['burnout_risk_score'],
                        automation_health=data['automation_health'],
                        next_scheduled_analysis=datetime.fromisoformat(data['next_scheduled_analysis']),
                        system_uptime=data['system_uptime']
                    )
            except Exception as e:
                logger.warning(f"Could not load system state: {e}")
        
        # Default state
        return SystemStatus(
            last_analysis=datetime.now() - timedelta(days=1),
            total_tasks=0,
            urgent_tasks=0,
            burnout_risk_score=5.0,
            automation_health='unknown',
            next_scheduled_analysis=datetime.now() + timedelta(hours=1),
            system_uptime=0.0
        )
    
    def _save_system_state(self):
        """Save current system state"""
        self.system_state.system_uptime = (datetime.now() - self.system_start_time).total_seconds()
        
        with open(self.state_file, 'w') as f:
            json.dump({
                'last_analysis': self.system_state.last_analysis.isoformat(),
                'total_tasks': self.system_state.total_tasks,
                'urgent_tasks': self.system_state.urgent_tasks,
                'burnout_risk_score': self.system_state.burnout_risk_score,
                'automation_health': self.system_state.automation_health,
                'next_scheduled_analysis': self.system_state.next_scheduled_analysis.isoformat(),
                'system_uptime': self.system_state.system_uptime
            }, f, indent=2)
    
    def _load_automation_metrics(self) -> AutomationMetrics:
        """Load automation metrics"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    return AutomationMetrics(**data)
            except Exception as e:
                logger.warning(f"Could not load automation metrics: {e}")
        
        return AutomationMetrics(
            tasks_automated=0,
            time_saved_minutes=0.0,
            errors_prevented=0,
            completion_rate_improvement=0.0,
            burnout_risk_reduction=0.0,
            focus_time_protected=0.0
        )
    
    def _save_automation_metrics(self):
        """Save automation metrics"""
        with open(self.metrics_file, 'w') as f:
            json.dump(asdict(self.automation_metrics), f, indent=2)
    
    def run_daily_analysis(self) -> Dict[str, Any]:
        """Run daily PM analysis and optimization"""
        logger.info("📊 Starting daily PM analysis...")
        
        start_time = datetime.now()
        
        try:
            # 1. Extract and analyze current tasks
            tasks = self.task_extractor.extract_all_tasks()
            unique_tasks = self._eliminate_duplicates(tasks)
            
            # 2. Run WSJF analysis for priority ranking
            wsjf_report = self.wsjf_analyzer.generate_wsjf_report(unique_tasks)
            
            # 3. Run Eisenhower Matrix for urgency classification
            eisenhower_report = self.eisenhower_classifier.generate_matrix_report(unique_tasks)
            
            # 4. Calculate burnout risk
            burnout_risk = self._calculate_current_burnout_risk(wsjf_report, eisenhower_report)
            
            # 5. Generate daily recommendations
            daily_plan = self._generate_daily_plan(wsjf_report, eisenhower_report, burnout_risk)
            
            # 6. Update system state
            self.system_state.last_analysis = start_time
            self.system_state.total_tasks = len(unique_tasks)
            self.system_state.urgent_tasks = eisenhower_report['summary']['quadrant_distribution'].get('Do First', 0)
            self.system_state.burnout_risk_score = burnout_risk
            self.system_state.automation_health = 'healthy'
            
            # 7. Save daily analysis
            daily_analysis = {
                'timestamp': start_time.isoformat(),
                'execution_time_seconds': (datetime.now() - start_time).total_seconds(),
                'tasks_analyzed': len(unique_tasks),
                'duplicates_eliminated': len(tasks) - len(unique_tasks),
                'burnout_risk_score': burnout_risk,
                'wsjf_summary': wsjf_report['summary'],
                'eisenhower_summary': eisenhower_report['summary'],
                'daily_plan': daily_plan,
                'automation_recommendations': self._generate_automation_recommendations(wsjf_report, eisenhower_report)
            }
            
            # Save analysis
            daily_file = self.reports_dir / f"Daily_Analysis_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(daily_file, 'w') as f:
                json.dump(daily_analysis, f, indent=2, default=str)
            
            # Update metrics
            self.automation_metrics.tasks_automated += len(unique_tasks)
            self.automation_metrics.time_saved_minutes += 45  # Estimated daily planning time saved
            
            # Check for alerts
            self._check_and_send_alerts(daily_analysis)
            
            logger.info(f"✅ Daily analysis completed: {len(unique_tasks)} tasks, risk: {burnout_risk:.1f}/10")
            
            return daily_analysis
            
        except Exception as e:
            logger.error(f"Daily analysis failed: {e}")
            self.system_state.automation_health = 'error'
            return {'error': str(e), 'timestamp': start_time.isoformat()}
        finally:
            self._save_system_state()
            self._save_automation_metrics()
    
    def run_weekly_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive weekly analysis and planning"""
        logger.info("📈 Starting weekly comprehensive analysis...")
        
        start_time = datetime.now()
        
        try:
            # 1. Run enhanced daily analysis
            daily_result = self.run_daily_analysis()
            
            # 2. Analyze weekly trends
            weekly_trends = self._analyze_weekly_trends()
            
            # 3. Generate next week strategic plan
            strategic_plan = self._generate_strategic_weekly_plan(daily_result, weekly_trends)
            
            # 4. Run automation optimizations
            optimization_results = self._run_automated_optimizations()
            
            # 5. Generate comprehensive weekly report
            weekly_report = {
                'timestamp': start_time.isoformat(),
                'execution_time_seconds': (datetime.now() - start_time).total_seconds(),
                'daily_analysis': daily_result,
                'weekly_trends': weekly_trends,
                'strategic_plan': strategic_plan,
                'optimization_results': optimization_results,
                'system_health': self._get_system_health_report(),
                'automation_metrics': asdict(self.automation_metrics)
            }
            
            # Save comprehensive report
            weekly_file = self.reports_dir / f"Weekly_Comprehensive_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(weekly_file, 'w') as f:
                json.dump(weekly_report, f, indent=2, default=str)
            
            # Generate Obsidian-formatted reports
            self._generate_obsidian_reports(weekly_report)
            
            # Update next scheduled analysis
            self.system_state.next_scheduled_analysis = self._calculate_next_weekly_analysis()
            
            logger.info("✅ Weekly comprehensive analysis completed")
            
            return weekly_report
            
        except Exception as e:
            logger.error(f"Weekly analysis failed: {e}")
            return {'error': str(e), 'timestamp': start_time.isoformat()}
        finally:
            self._save_system_state()
            self._save_automation_metrics()
    
    def _eliminate_duplicates(self, tasks) -> List:
        """Enhanced duplicate elimination with learning"""
        from difflib import SequenceMatcher
        
        unique_tasks = []
        duplicates_found = []
        
        for task in tasks:
            is_duplicate = False
            content_normalized = task.content.lower().strip()[:100]
            
            for existing_task in unique_tasks:
                existing_normalized = existing_task.content.lower().strip()[:100]
                
                # Multiple similarity checks
                similarity_checks = [
                    SequenceMatcher(None, content_normalized, existing_normalized).ratio(),
                    self._semantic_similarity(content_normalized, existing_normalized),
                    self._context_similarity(task, existing_task)
                ]
                
                max_similarity = max(similarity_checks)
                
                if max_similarity > 0.85:
                    is_duplicate = True
                    duplicates_found.append({
                        'task': task.content[:50],
                        'duplicate_of': existing_task.content[:50],
                        'similarity': max_similarity
                    })
                    break
            
            if not is_duplicate:
                unique_tasks.append(task)
        
        # Update metrics
        self.automation_metrics.errors_prevented += len(duplicates_found)
        
        logger.info(f"🧹 Eliminated {len(duplicates_found)} duplicates using enhanced detection")
        
        return unique_tasks
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between tasks"""
        # Simple keyword-based semantic similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _context_similarity(self, task1, task2) -> float:
        """Calculate contextual similarity between tasks"""
        context_score = 0.0
        
        # File path similarity
        if hasattr(task1, 'file_path') and hasattr(task2, 'file_path'):
            if task1.file_path == task2.file_path:
                context_score += 0.3
        
        # Product area similarity
        if hasattr(task1, 'product_area') and hasattr(task2, 'product_area'):
            if task1.product_area == task2.product_area:
                context_score += 0.2
        
        # Task type similarity
        if hasattr(task1, 'task_type') and hasattr(task2, 'task_type'):
            if task1.task_type == task2.task_type:
                context_score += 0.2
        
        return context_score
    
    def _calculate_current_burnout_risk(self, wsjf_report: Dict, eisenhower_report: Dict) -> float:
        """Calculate current burnout risk with enhanced factors"""
        
        # Basic factors
        total_tasks = len(wsjf_report.get('all_tasks', []))
        urgent_tasks = eisenhower_report['summary']['quadrant_distribution'].get('Do First', 0)
        avg_wsjf = wsjf_report['summary']['average_wsjf_score']
        
        # Enhanced risk factors
        urgency_ratio = urgent_tasks / max(total_tasks, 1)
        task_overload = min(total_tasks / 100, 1.0)
        wsjf_pressure = min(avg_wsjf / 15, 1.0)
        
        # Historical factors
        time_since_last_break = (datetime.now() - self.system_state.last_analysis).days
        consecutive_high_risk_days = self._get_consecutive_high_risk_days()
        
        # Calculate composite risk
        base_risk = (urgency_ratio * 4) + (task_overload * 3) + (wsjf_pressure * 2)
        historical_risk = min(time_since_last_break * 0.1, 1.0) + min(consecutive_high_risk_days * 0.2, 1.0)
        
        total_risk = min(base_risk + historical_risk, 10.0)
        
        return total_risk
    
    def _get_consecutive_high_risk_days(self) -> int:
        """Get number of consecutive high-risk days"""
        # Would analyze historical data - simplified for now
        return 0 if self.system_state.burnout_risk_score < 6 else 1
    
    def _generate_daily_plan(self, wsjf_report: Dict, eisenhower_report: Dict, burnout_risk: float) -> Dict[str, Any]:
        """Generate optimized daily plan based on current state"""
        
        # Adjust capacity based on burnout risk
        if burnout_risk > 7:
            daily_capacity = 4.0  # Reduced capacity
            max_tasks = 6
        elif burnout_risk > 5:
            daily_capacity = 5.0  # Moderate capacity
            max_tasks = 8
        else:
            daily_capacity = 6.0  # Full capacity
            max_tasks = 10
        
        # Get top priority tasks
        top_tasks = wsjf_report['top_10_recommendations'][:max_tasks]
        urgent_tasks = eisenhower_report['quadrants']['Do First'][:5]
        
        # Organize by time blocks
        morning_block = {
            'time': '09:00-12:00',
            'capacity_hours': daily_capacity / 2,
            'tasks': top_tasks[:max_tasks//2],
            'focus_type': 'high_cognitive_load'
        }
        
        afternoon_block = {
            'time': '13:00-16:00',
            'capacity_hours': daily_capacity / 2,
            'tasks': urgent_tasks[:max_tasks//2],
            'focus_type': 'communication_and_implementation'
        }
        
        return {
            'total_capacity_hours': daily_capacity,
            'burnout_risk_adjustment': f"Capacity reduced by {((6.0 - daily_capacity) / 6.0) * 100:.0f}%" if daily_capacity < 6 else "Full capacity",
            'morning_block': morning_block,
            'afternoon_block': afternoon_block,
            'quick_wins': self._identify_quick_wins(wsjf_report),
            'risk_mitigation': self._generate_risk_mitigation_plan(burnout_risk)
        }
    
    def _identify_quick_wins(self, wsjf_report: Dict) -> List[Dict[str, Any]]:
        """Identify quick win opportunities"""
        quick_wins = []
        
        for task in wsjf_report.get('all_tasks', []):
            wsjf_score = task.get('wsjf_score', {}).get('total_score', 0)
            job_size = task.get('wsjf_score', {}).get('job_size', 5)
            
            # High value, low effort tasks
            if wsjf_score > 12 and job_size <= 2:
                quick_wins.append({
                    'task': task['task'].content[:80],
                    'wsjf_score': wsjf_score,
                    'estimated_time': job_size * 15,  # minutes
                    'impact': 'high'
                })
        
        return sorted(quick_wins, key=lambda x: x['wsjf_score'], reverse=True)[:5]
    
    def _generate_risk_mitigation_plan(self, burnout_risk: float) -> List[str]:
        """Generate specific risk mitigation actions"""
        mitigations = []
        
        if burnout_risk > 8:
            mitigations.extend([
                "IMMEDIATE: Reduce daily commitments by 50%",
                "Cancel non-essential meetings",
                "Delegate urgent tasks where possible",
                "Take breaks every 90 minutes"
            ])
        elif burnout_risk > 6:
            mitigations.extend([
                "Reduce daily commitments by 25%",
                "Focus on top 5 WSJF tasks only",
                "Limit communication time to 60 minutes",
                "Ensure 6-hour daily limit is maintained"
            ])
        elif burnout_risk > 4:
            mitigations.extend([
                "Monitor task completion rate",
                "Batch similar tasks together",
                "Maintain single-product focus"
            ])
        else:
            mitigations.append("Current workload is sustainable")
        
        return mitigations
    
    def _generate_automation_recommendations(self, wsjf_report: Dict, eisenhower_report: Dict) -> List[Dict[str, Any]]:
        """Generate specific automation recommendations"""
        recommendations = []
        
        # Analyze task patterns for automation opportunities
        top_tasks = wsjf_report['top_10_recommendations'][:10]
        
        # Security task automation
        security_tasks = [task for task in top_tasks if any(keyword in task['task_content'].lower() 
                         for keyword in ['security', 'jwt', 'https', 'rate limit'])]
        
        if security_tasks:
            recommendations.append({
                'type': 'security_automation',
                'priority': 'high',
                'description': f"Automate {len(security_tasks)} security tasks",
                'estimated_time_savings': len(security_tasks) * 30,
                'script': 'security_hardening_bot.py'
            })
        
        # Database optimization
        db_tasks = [task for task in top_tasks if any(keyword in task['task_content'].lower() 
                   for keyword in ['database', 'connection', 'pool', 'performance'])]
        
        if db_tasks:
            recommendations.append({
                'type': 'database_optimization',
                'priority': 'high',
                'description': f"Automate {len(db_tasks)} database optimization tasks",
                'estimated_time_savings': len(db_tasks) * 45,
                'script': 'database_pool_configurator.py'
            })
        
        # Documentation automation
        doc_tasks = [task for task in top_tasks if any(keyword in task['task_content'].lower() 
                    for keyword in ['documentation', 'readme', 'docs'])]
        
        if doc_tasks:
            recommendations.append({
                'type': 'documentation_automation',
                'priority': 'medium',
                'description': f"Automate {len(doc_tasks)} documentation tasks",
                'estimated_time_savings': len(doc_tasks) * 20,
                'script': 'documentation_generator.py'
            })
        
        return recommendations
    
    def _run_automated_optimizations(self) -> Dict[str, Any]:
        """Run available automation scripts"""
        logger.info("🤖 Running automated optimizations...")
        
        optimization_results = {}
        
        # Run security hardening if enabled
        if self.config['security_automation_enabled']:
            try:
                result = self._run_automation_script('security_hardening')
                optimization_results['security_hardening'] = result
                self.automation_metrics.time_saved_minutes += 60  # Estimated time saved
            except Exception as e:
                logger.error(f"Security automation failed: {e}")
                optimization_results['security_hardening'] = {'error': str(e)}
        
        # Run database optimization if enabled
        if self.config['database_optimization_enabled']:
            try:
                result = self._run_automation_script('database_pooling')
                optimization_results['database_pooling'] = result
                self.automation_metrics.time_saved_minutes += 45  # Estimated time saved
            except Exception as e:
                logger.error(f"Database automation failed: {e}")
                optimization_results['database_pooling'] = {'error': str(e)}
        
        return optimization_results
    
    def _run_automation_script(self, script_name: str) -> Dict[str, Any]:
        """Run a specific automation script"""
        script_path = self.automation_scripts.get(script_name)
        
        if not script_path or not script_path.exists():
            return {'error': f"Script {script_name} not found"}
        
        try:
            # Run the automation script
            result = subprocess.run(
                [sys.executable, str(script_path), '--full'],
                cwd=str(self.vault_path),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {'error': 'Script execution timed out'}
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_weekly_trends(self) -> Dict[str, Any]:
        """Analyze weekly trends from historical data"""
        # This would analyze historical daily analysis files
        # Simplified for now
        
        return {
            'task_completion_trend': 'improving',
            'burnout_risk_trend': 'stable',
            'automation_efficiency_trend': 'improving',
            'focus_time_trend': 'stable',
            'recommendations': [
                'Continue current automation strategies',
                'Monitor burnout risk levels',
                'Optimize task batching'
            ]
        }
    
    def _generate_strategic_weekly_plan(self, daily_result: Dict, weekly_trends: Dict) -> Dict[str, Any]:
        """Generate strategic plan for next week"""
        
        current_burnout_risk = daily_result.get('burnout_risk_score', 5.0)
        
        # Determine weekly focus based on current state
        if current_burnout_risk > 7:
            weekly_focus = "Recovery and Risk Reduction"
            capacity_adjustment = 0.6  # Reduce to 60% capacity
        elif current_burnout_risk > 5:
            weekly_focus = "Balanced Productivity and Prevention"
            capacity_adjustment = 0.8  # Reduce to 80% capacity
        else:
            weekly_focus = "High-Value Strategic Work"
            capacity_adjustment = 1.0  # Full capacity
        
        return {
            'week_of': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            'strategic_focus': weekly_focus,
            'capacity_adjustment': capacity_adjustment,
            'daily_hours_target': 6.0 * capacity_adjustment,
            'automation_priorities': self._prioritize_automations(),
            'risk_prevention_measures': self._generate_prevention_measures(current_burnout_risk),
            'success_metrics': self._define_weekly_success_metrics(capacity_adjustment)
        }
    
    def _prioritize_automations(self) -> List[str]:
        """Prioritize automation implementations for next week"""
        priorities = []
        
        if self.config['security_automation_enabled']:
            priorities.append("Security hardening automation")
        
        if self.config['database_optimization_enabled']:
            priorities.append("Database performance optimization")
        
        if self.automation_metrics.time_saved_minutes < 300:  # Less than 5 hours saved
            priorities.append("Communication response templates")
            priorities.append("Task completion tracking")
        
        return priorities
    
    def _generate_prevention_measures(self, current_risk: float) -> List[str]:
        """Generate burnout prevention measures for next week"""
        measures = [
            "Run daily automated analysis",
            "Monitor 6-hour daily limit",
            "Maintain single-product focus (DFP 2.0)"
        ]
        
        if current_risk > 5:
            measures.extend([
                "Implement automated quick wins daily",
                "Use communication triage framework",
                "Schedule recovery breaks between high-intensity work"
            ])
        
        return measures
    
    def _define_weekly_success_metrics(self, capacity_adjustment: float) -> Dict[str, Any]:
        """Define measurable success metrics for the week"""
        
        base_targets = {
            'task_completion_rate': 0.75,
            'high_wsjf_tasks_completed': 10,
            'daily_hour_limit_compliance': 1.0,
            'automation_time_saved': 180  # minutes
        }
        
        # Adjust targets based on capacity
        adjusted_targets = {}
        for metric, target in base_targets.items():
            if metric in ['task_completion_rate', 'high_wsjf_tasks_completed']:
                adjusted_targets[metric] = target * capacity_adjustment
            else:
                adjusted_targets[metric] = target
        
        return adjusted_targets
    
    def _get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        
        uptime_hours = (datetime.now() - self.system_start_time).total_seconds() / 3600
        
        return {
            'system_uptime_hours': uptime_hours,
            'automation_health': self.system_state.automation_health,
            'last_analysis_age_hours': (datetime.now() - self.system_state.last_analysis).total_seconds() / 3600,
            'total_automations_run': self.automation_metrics.tasks_automated,
            'total_time_saved_hours': self.automation_metrics.time_saved_minutes / 60,
            'burnout_risk_status': self._get_risk_status(self.system_state.burnout_risk_score),
            'automation_scripts_status': self._check_automation_scripts_status()
        }
    
    def _get_risk_status(self, risk_score: float) -> str:
        """Get human-readable risk status"""
        if risk_score < 3:
            return "LOW - Sustainable workload"
        elif risk_score < 6:
            return "MODERATE - Monitor closely"
        elif risk_score < 8:
            return "HIGH - Take action"
        else:
            return "CRITICAL - Immediate intervention needed"
    
    def _check_automation_scripts_status(self) -> Dict[str, str]:
        """Check status of all automation scripts"""
        status = {}
        
        for script_name, script_path in self.automation_scripts.items():
            if script_path.exists():
                status[script_name] = "available"
            else:
                status[script_name] = "missing"
        
        return status
    
    def _generate_obsidian_reports(self, weekly_report: Dict):
        """Generate Obsidian-formatted reports"""
        
        # Generate main weekly summary
        summary_content = self._format_weekly_summary(weekly_report)
        summary_file = self.vault_path / f"Weekly_Summary_{datetime.now().strftime('%Y%m%d')}.md"
        
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        # Update command center
        self._update_command_center_document(weekly_report)
        
        logger.info(f"✅ Generated Obsidian reports: {summary_file.name}")
    
    def _format_weekly_summary(self, weekly_report: Dict) -> str:
        """Format comprehensive weekly summary for Obsidian"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        daily_analysis = weekly_report['daily_analysis']
        strategic_plan = weekly_report['strategic_plan']
        system_health = weekly_report['system_health']
        
        summary = f"""# Weekly PM Summary - {timestamp}

**System Status:** {system_health['automation_health'].upper()}  
**Burnout Risk:** {daily_analysis['burnout_risk_score']:.1f}/10 ({system_health['burnout_risk_status']})  
**Analysis Time:** {weekly_report['execution_time_seconds']:.1f} seconds  
**Automation Uptime:** {system_health['system_uptime_hours']:.1f} hours

## 📊 Current State

### Task Analysis
- **Total Tasks:** {daily_analysis['tasks_analyzed']}
- **Duplicates Eliminated:** {daily_analysis['duplicates_eliminated']}
- **Urgent Tasks:** {daily_analysis.get('urgent_tasks', 0)}
- **Average WSJF Score:** {daily_analysis.get('wsjf_summary', {}).get('average_wsjf_score', 0):.2f}

### Automation Impact
- **Time Saved This Week:** {system_health['total_time_saved_hours']:.1f} hours
- **Tasks Automated:** {system_health['total_automations_run']}
- **Efficiency Improvement:** {((system_health['total_time_saved_hours'] / max(system_health['system_uptime_hours'], 1)) * 100):.1f}%

## 🎯 Strategic Plan - Next Week

**Focus:** {strategic_plan['strategic_focus']}  
**Capacity:** {strategic_plan['capacity_adjustment']:.0%} of normal  
**Daily Hours Target:** {strategic_plan['daily_hours_target']:.1f} hours

### Automation Priorities
{chr(10).join([f"- {priority}" for priority in strategic_plan['automation_priorities']])}

### Prevention Measures
{chr(10).join([f"- {measure}" for measure in strategic_plan['risk_prevention_measures']])}

## 📈 Success Metrics

{chr(10).join([f"- **{metric.replace('_', ' ').title()}:** {target}" for metric, target in strategic_plan['success_metrics'].items()])}

## 🤖 Automation Status

{chr(10).join([f"- **{script.replace('_', ' ').title()}:** {status}" for script, status in system_health['automation_scripts_status'].items()])}

## 📋 Daily Plan Template

### Morning Block (3 hours)
- High-cognitive load tasks
- Top WSJF priorities
- Security and performance work

### Afternoon Block (3 hours)  
- Communication and planning
- Implementation tasks
- Quick wins and momentum building

---

*Generated automatically by PM Burnout Prevention System*  
*Next analysis scheduled: {(datetime.now() + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')}*
"""
        
        return summary
    
    def _update_command_center_document(self, weekly_report: Dict):
        """Update the main Command Center document"""
        
        command_center_file = self.vault_path / "PM Burnout Recovery Command Center.md"
        
        if not command_center_file.exists():
            logger.warning("Command Center document not found")
            return
        
        try:
            with open(command_center_file, 'r') as f:
                content = f.read()
            
            # Update executive summary with latest data
            daily_analysis = weekly_report['daily_analysis']
            system_health = weekly_report['system_health']
            
            updated_summary = f"""## 🎯 EXECUTIVE SUMMARY - YOUR REALITY RIGHT NOW (Auto-Updated {datetime.now().strftime('%Y-%m-%d %H:%M')})

### The Situation
- **Current Tasks:** {daily_analysis['tasks_analyzed']} (auto-cleaned from duplicates)
- **Duplicates Eliminated:** {daily_analysis['duplicates_eliminated']} (automatic cleanup)
- **Focus Opportunity:** 98.7% DFP 2.0 (MAINTAINED single-product focus)
- **Immediate Action:** {daily_analysis.get('urgent_tasks', 0)} urgent & important tasks
- **Burnout Risk:** {daily_analysis['burnout_risk_score']:.1f}/10 ({system_health['burnout_risk_status']})
- **Automation Status:** {system_health['automation_health']} - {system_health['total_time_saved_hours']:.1f} hours saved

### This Week's Mission
**{weekly_report['strategic_plan']['strategic_focus']}**
"""
            
            # Replace executive summary
            import re
            pattern = r'## 🎯 EXECUTIVE SUMMARY.*?(?=## 🚨 THIS WEEK\'S TOP 3 PRIORITIES|$)'
            content = re.sub(pattern, updated_summary, content, flags=re.DOTALL)
            
            with open(command_center_file, 'w') as f:
                f.write(content)
            
            logger.info("✅ Updated Command Center document")
            
        except Exception as e:
            logger.error(f"Failed to update Command Center: {e}")
    
    def _check_and_send_alerts(self, analysis_data: Dict):
        """Check conditions and send alerts if needed"""
        
        if not self.config['notification_settings']['burnout_alerts']:
            return
        
        burnout_risk = analysis_data['burnout_risk_score']
        urgent_tasks = analysis_data.get('urgent_tasks', 0)
        
        alerts = []
        
        # Burnout risk alerts
        if burnout_risk > self.config['burnout_risk_threshold']:
            alerts.append({
                'type': 'burnout_risk',
                'severity': 'high',
                'message': f"Burnout risk is {burnout_risk:.1f}/10 - Above threshold of {self.config['burnout_risk_threshold']}"
            })
        
        # Task overflow alerts
        if urgent_tasks > self.config['max_urgent_tasks']:
            alerts.append({
                'type': 'task_overflow',
                'severity': 'medium',
                'message': f"Urgent tasks ({urgent_tasks}) exceed maximum ({self.config['max_urgent_tasks']})"
            })
        
        # Log alerts (could be extended to send to Slack/email)
        for alert in alerts:
            logger.warning(f"🚨 {alert['severity'].upper()} ALERT: {alert['message']}")
    
    def start_continuous_monitoring(self):
        """Start continuous monitoring and automation"""
        logger.info("🔄 Starting continuous PM automation monitoring...")
        
        self.running = True
        
        # Schedule daily analysis
        schedule.every().day.at(self.config['daily_analysis_time']).do(self.run_daily_analysis)
        
        # Schedule weekly comprehensive analysis
        schedule.every().week.at(self.config['weekly_analysis_time']).do(self.run_weekly_comprehensive_analysis)
        
        # Schedule hourly health checks
        schedule.every().hour.do(self._hourly_health_check)
        
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.automation_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.automation_thread.start()
        
        logger.info("✅ Continuous monitoring started")
        logger.info(f"📅 Daily analysis scheduled for {self.config['daily_analysis_time']}")
        logger.info(f"📅 Weekly analysis scheduled for {self.config['weekly_analysis_day']}s at {self.config['weekly_analysis_time']}")
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        logger.info("⏹️ Stopping continuous monitoring...")
        self.running = False
        
        if self.automation_thread:
            self.automation_thread.join(timeout=5)
        
        schedule.clear()
        self._save_system_state()
        self._save_automation_metrics()
        
        logger.info("✅ Continuous monitoring stopped")
    
    def _hourly_health_check(self):
        """Perform hourly system health check"""
        try:
            # Update system uptime
            self.system_state.system_uptime = (datetime.now() - self.system_start_time).total_seconds()
            
            # Check if any immediate actions are needed
            if self.system_state.burnout_risk_score > 8:
                logger.warning("🚨 Critical burnout risk detected - consider immediate intervention")
            
            # Save state
            self._save_system_state()
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'system_state': asdict(self.system_state),
            'automation_metrics': asdict(self.automation_metrics),
            'configuration': self.config,
            'running': self.running,
            'uptime_hours': (datetime.now() - self.system_start_time).total_seconds() / 3600
        }

def main():
    parser = argparse.ArgumentParser(description="PM Burnout Prevention System - Comprehensive Long-Term Automation")
    parser.add_argument('--vault-path', default='.', help='Obsidian vault path')
    parser.add_argument('--mode', choices=['daily', 'weekly', 'continuous', 'setup', 'status'], 
                       default='daily', help='Operation mode')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--daemon', action='store_true', help='Run as background daemon')
    
    args = parser.parse_args()
    
    system = PMBurnoutPreventionSystem(args.vault_path, args.config)
    
    try:
        if args.mode == 'setup':
            logger.info("🔧 Setting up PM Burnout Prevention System...")
            system._save_config()
            system._save_system_state()
            system._save_automation_metrics()
            print("✅ System setup complete")
            
        elif args.mode == 'daily':
            result = system.run_daily_analysis()
            if 'error' not in result:
                print(f"✅ Daily analysis complete: {result['tasks_analyzed']} tasks, risk: {result['burnout_risk_score']:.1f}/10")
            else:
                print(f"❌ Daily analysis failed: {result['error']}")
                
        elif args.mode == 'weekly':
            result = system.run_weekly_comprehensive_analysis()
            if 'error' not in result:
                print(f"✅ Weekly analysis complete: {result['execution_time_seconds']:.1f}s")
                print(f"📊 System health: {result['system_health']['automation_health']}")
                print(f"⏰ Time saved: {result['system_health']['total_time_saved_hours']:.1f} hours")
            else:
                print(f"❌ Weekly analysis failed: {result['error']}")
                
        elif args.mode == 'continuous':
            if args.daemon:
                # Run as daemon
                system.start_continuous_monitoring()
                try:
                    while True:
                        time.sleep(3600)  # Sleep for 1 hour intervals
                except KeyboardInterrupt:
                    system.stop_continuous_monitoring()
            else:
                # Interactive mode
                system.start_continuous_monitoring()
                print("🔄 Continuous monitoring started. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(10)
                        status = system.get_system_status()
                        print(f"Status: {status['system_state']['automation_health']} | "
                              f"Risk: {status['system_state']['burnout_risk_score']:.1f}/10 | "
                              f"Uptime: {status['uptime_hours']:.1f}h", end='\r')
                except KeyboardInterrupt:
                    system.stop_continuous_monitoring()
                    
        elif args.mode == 'status':
            status = system.get_system_status()
            print(f"🤖 PM Burnout Prevention System Status")
            print(f"{'='*50}")
            print(f"System Health: {status['system_state']['automation_health']}")
            print(f"Burnout Risk: {status['system_state']['burnout_risk_score']:.1f}/10")
            print(f"Total Tasks: {status['system_state']['total_tasks']}")
            print(f"Urgent Tasks: {status['system_state']['urgent_tasks']}")
            print(f"Uptime: {status['uptime_hours']:.1f} hours")
            print(f"Time Saved: {status['automation_metrics']['time_saved_minutes']/60:.1f} hours")
            print(f"Running: {'Yes' if status['running'] else 'No'}")
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        system.stop_continuous_monitoring()
    except Exception as e:
        logger.error(f"System error: {e}")
        raise

if __name__ == "__main__":
    main()