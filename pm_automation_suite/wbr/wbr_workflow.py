"""
WBR Workflow Implementation

Orchestrates the complete Weekly Business Review automation workflow,
including scheduling, state management, error handling, and notifications.
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
from pathlib import Path

from .wbr_data_extractor import WBRDataExtractor, WBRDataPackage
from .insight_generator import InsightGenerator, Insight
from .slide_generator import SlideGenerator
from orchestration.scheduler import Scheduler
from orchestration.event_bus import EventBus


logger = logging.getLogger(__name__)


class WorkflowState(Enum):
    """Workflow execution states."""
    PENDING = "pending"
    RUNNING = "running"
    DATA_EXTRACTION = "data_extraction"
    INSIGHT_GENERATION = "insight_generation"
    SLIDE_GENERATION = "slide_generation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NotificationType(Enum):
    """Types of notifications."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"


@dataclass
class WorkflowExecution:
    """Workflow execution tracking."""
    execution_id: str
    state: WorkflowState
    start_time: datetime
    end_time: Optional[datetime]
    trigger_type: str  # "scheduled", "manual", "api"
    config: Dict[str, Any]
    results: Dict[str, Any]
    errors: List[str]
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['state'] = self.state.value
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        return data


@dataclass
class NotificationConfig:
    """Configuration for notifications."""
    enabled: bool = True
    email_recipients: List[str] = None
    slack_webhook: Optional[str] = None
    teams_webhook: Optional[str] = None
    on_success: bool = True
    on_failure: bool = True
    on_warning: bool = True
    include_preview: bool = True


class WBRWorkflow:
    """
    WBR Workflow orchestrator for automated business review generation.
    
    Manages the complete workflow from data extraction to slide generation,
    with scheduling, error handling, and notification capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize WBR Workflow.
        
        Args:
            config: Configuration dictionary with workflow settings
        """
        self.config = config
        self.workflow_id = config.get('workflow_id', 'default_wbr')
        self.state_dir = Path(config.get('state_dir', './wbr_state'))
        self.state_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_extractor = WBRDataExtractor(config.get('data_extraction', {}))
        self.insight_generator = InsightGenerator(config.get('insight_generation', {}))
        self.slide_generator = SlideGenerator(config.get('slide_generation', {}))
        
        # Scheduler and event bus
        self.scheduler = Scheduler(config.get('scheduler', {}))
        self.event_bus = EventBus()
        
        # Workflow state
        self.current_execution: Optional[WorkflowExecution] = None
        self.execution_history: List[WorkflowExecution] = []
        
        # Notification configuration
        self.notification_config = NotificationConfig(**config.get('notifications', {}))
        
        # Retry configuration
        self.retry_config = {
            'max_retries': config.get('max_retries', 3),
            'retry_delay': config.get('retry_delay', 300),  # 5 minutes
            'exponential_backoff': config.get('exponential_backoff', True)
        }
        
        # Register event handlers
        self._register_event_handlers()
        
        # Load execution history
        self._load_execution_history()
        
    def _register_event_handlers(self):
        """Register workflow event handlers."""
        self.event_bus.subscribe('workflow.started', self._on_workflow_started)
        self.event_bus.subscribe('workflow.completed', self._on_workflow_completed)
        self.event_bus.subscribe('workflow.failed', self._on_workflow_failed)
        self.event_bus.subscribe('data_extraction.completed', self._on_data_extraction_completed)
        self.event_bus.subscribe('insight_generation.completed', self._on_insight_generation_completed)
        self.event_bus.subscribe('slide_generation.completed', self._on_slide_generation_completed)
    
    async def schedule_weekly_execution(
        self, 
        day_of_week: int = 0,  # Monday
        hour: int = 9,         # 9 AM
        timezone: str = 'UTC'
    ):
        """
        Schedule weekly WBR execution.
        
        Args:
            day_of_week: Day of week (0=Monday, 6=Sunday)
            hour: Hour of day (24-hour format)
            timezone: Timezone for scheduling
        """
        try:
            # Create cron expression for weekly execution
            cron_expression = f"0 {hour} * * {day_of_week}"
            
            # Schedule the workflow
            await self.scheduler.schedule_job(
                job_id=f"{self.workflow_id}_weekly",
                cron_expression=cron_expression,
                func=self.execute_workflow,
                kwargs={'trigger_type': 'scheduled'},
                timezone=timezone
            )
            
            logger.info(f"Scheduled weekly WBR execution: {cron_expression} ({timezone})")
            
        except Exception as e:
            logger.error(f"Failed to schedule weekly execution: {e}")
            raise
    
    async def execute_workflow(
        self, 
        trigger_type: str = 'manual',
        custom_config: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """
        Execute complete WBR workflow.
        
        Args:
            trigger_type: How the workflow was triggered
            custom_config: Override configuration for this execution
            
        Returns:
            Workflow execution result
        """
        execution_id = f"wbr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create workflow execution
        execution = WorkflowExecution(
            execution_id=execution_id,
            state=WorkflowState.PENDING,
            start_time=datetime.now(),
            end_time=None,
            trigger_type=trigger_type,
            config=custom_config or self.config,
            results={},
            errors=[],
            metrics={}
        )
        
        self.current_execution = execution
        logger.info(f"Starting WBR workflow execution: {execution_id}")
        
        try:
            # Update state and emit event
            execution.state = WorkflowState.RUNNING
            await self.event_bus.emit('workflow.started', {'execution': execution})
            
            # Execute workflow steps
            await self._execute_with_retries(self._run_workflow_steps, execution)
            
            # Mark as completed
            execution.state = WorkflowState.COMPLETED
            execution.end_time = datetime.now()
            execution.metrics['total_duration'] = (execution.end_time - execution.start_time).total_seconds()
            
            await self.event_bus.emit('workflow.completed', {'execution': execution})
            logger.info(f"WBR workflow completed successfully: {execution_id}")
            
        except Exception as e:
            execution.state = WorkflowState.FAILED
            execution.end_time = datetime.now()
            execution.errors.append(str(e))
            
            await self.event_bus.emit('workflow.failed', {'execution': execution, 'error': str(e)})
            logger.error(f"WBR workflow failed: {execution_id} - {e}")
            
        finally:
            # Save execution state
            self._save_execution_state(execution)
            self.execution_history.append(execution)
            self.current_execution = None
        
        return execution
    
    async def _run_workflow_steps(self, execution: WorkflowExecution):
        """Run the complete workflow steps."""
        # Step 1: Data Extraction
        await self._execute_data_extraction(execution)
        
        # Step 2: Insight Generation
        await self._execute_insight_generation(execution)
        
        # Step 3: Slide Generation
        await self._execute_slide_generation(execution)
        
        # Step 4: Post-processing
        await self._execute_post_processing(execution)
    
    async def _execute_data_extraction(self, execution: WorkflowExecution):
        """Execute data extraction step."""
        execution.state = WorkflowState.DATA_EXTRACTION
        step_start = datetime.now()
        
        try:
            logger.info("Starting data extraction")
            
            # Extract WBR data
            wbr_data = await self.data_extractor.extract_wbr_data()
            
            # Store results
            execution.results['wbr_data'] = wbr_data
            execution.metrics['data_extraction_duration'] = (datetime.now() - step_start).total_seconds()
            execution.metrics['data_quality_score'] = wbr_data.quality_score
            execution.metrics['metrics_count'] = len(wbr_data.metrics)
            
            await self.event_bus.emit('data_extraction.completed', {
                'execution': execution,
                'data_quality': wbr_data.quality_score,
                'metrics_count': len(wbr_data.metrics)
            })
            
            logger.info(f"Data extraction completed. Quality score: {wbr_data.quality_score:.2f}")
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            raise
    
    async def _execute_insight_generation(self, execution: WorkflowExecution):
        """Execute insight generation step."""
        execution.state = WorkflowState.INSIGHT_GENERATION
        step_start = datetime.now()
        
        try:
            logger.info("Starting insight generation")
            
            wbr_data = execution.results['wbr_data']
            
            # Prepare metrics data for insight generation
            metrics_data = [
                {
                    'name': m.name,
                    'value': m.value,
                    'change_percent': m.change_percent,
                    'trend': m.trend,
                    'source': m.source,
                    'target': m.target,
                    'alert_threshold': m.alert_threshold
                }
                for m in wbr_data.metrics
            ]
            
            # Generate insights
            insights = await self.insight_generator.generate_comprehensive_insights(metrics_data)
            
            # Generate executive summary
            metrics_summary = self.data_extractor.get_metric_summary(wbr_data.metrics)
            executive_summary = await self.insight_generator.generate_executive_summary(insights, metrics_summary)
            
            # Store results
            execution.results['insights'] = insights
            execution.results['executive_summary'] = executive_summary
            execution.metrics['insight_generation_duration'] = (datetime.now() - step_start).total_seconds()
            execution.metrics['insights_count'] = len(insights)
            execution.metrics['critical_insights'] = len([i for i in insights if i.priority.value == 'critical'])
            
            await self.event_bus.emit('insight_generation.completed', {
                'execution': execution,
                'insights_count': len(insights),
                'critical_count': execution.metrics['critical_insights']
            })
            
            logger.info(f"Insight generation completed. Generated {len(insights)} insights")
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            raise
    
    async def _execute_slide_generation(self, execution: WorkflowExecution):
        """Execute slide generation step."""
        execution.state = WorkflowState.SLIDE_GENERATION
        step_start = datetime.now()
        
        try:
            logger.info("Starting slide generation")
            
            wbr_data = execution.results['wbr_data']
            insights = execution.results['insights']
            executive_summary = execution.results['executive_summary']
            
            # Generate presentation
            presentation_path = await self.slide_generator.generate_wbr_presentation(
                wbr_data=wbr_data,
                insights=insights,
                executive_summary=executive_summary
            )
            
            # Store results
            execution.results['presentation_path'] = presentation_path
            execution.metrics['slide_generation_duration'] = (datetime.now() - step_start).total_seconds()
            
            await self.event_bus.emit('slide_generation.completed', {
                'execution': execution,
                'presentation_path': presentation_path
            })
            
            logger.info(f"Slide generation completed: {presentation_path}")
            
        except Exception as e:
            logger.error(f"Slide generation failed: {e}")
            raise
    
    async def _execute_post_processing(self, execution: WorkflowExecution):
        """Execute post-processing step."""
        try:
            logger.info("Starting post-processing")
            
            # Send notifications
            await self._send_notifications(execution)
            
            # Archive old executions if needed
            await self._cleanup_old_executions()
            
            logger.info("Post-processing completed")
            
        except Exception as e:
            logger.warning(f"Post-processing failed (non-critical): {e}")
            # Don't raise - post-processing failures shouldn't fail the workflow
    
    async def _execute_with_retries(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic."""
        max_retries = self.retry_config['max_retries']
        base_delay = self.retry_config['retry_delay']
        exponential_backoff = self.retry_config['exponential_backoff']
        
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    raise  # Final attempt, re-raise the exception
                
                # Calculate delay
                delay = base_delay
                if exponential_backoff:
                    delay = base_delay * (2 ** attempt)
                
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
    
    async def _send_notifications(self, execution: WorkflowExecution):
        """Send workflow notifications."""
        if not self.notification_config.enabled:
            return
        
        try:
            # Determine notification type
            if execution.state == WorkflowState.COMPLETED:
                if not self.notification_config.on_success:
                    return
                notification_type = NotificationType.SUCCESS
                title = "WBR Generation Successful"
                message = f"Weekly Business Review generated successfully"
            elif execution.state == WorkflowState.FAILED:
                if not self.notification_config.on_failure:
                    return
                notification_type = NotificationType.ERROR
                title = "WBR Generation Failed"
                message = f"Weekly Business Review generation failed: {', '.join(execution.errors)}"
            else:
                return
            
            # Prepare notification content
            content = self._prepare_notification_content(execution, title, message)
            
            # Send notifications
            await self._send_email_notification(content, notification_type)
            await self._send_slack_notification(content, notification_type)
            await self._send_teams_notification(content, notification_type)
            
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")
    
    def _prepare_notification_content(
        self, 
        execution: WorkflowExecution,
        title: str,
        message: str
    ) -> Dict[str, Any]:
        """Prepare notification content."""
        return {
            'title': title,
            'message': message,
            'execution_id': execution.execution_id,
            'duration': execution.metrics.get('total_duration', 0),
            'data_quality': execution.metrics.get('data_quality_score', 0),
            'insights_count': execution.metrics.get('insights_count', 0),
            'critical_insights': execution.metrics.get('critical_insights', 0),
            'presentation_path': execution.results.get('presentation_path'),
            'timestamp': execution.start_time.isoformat()
        }
    
    async def _send_email_notification(self, content: Dict[str, Any], notification_type: NotificationType):
        """Send email notification."""
        if not self.notification_config.email_recipients:
            return
        
        try:
            # Email implementation would go here
            # For now, just log
            logger.info(f"Email notification would be sent to: {self.notification_config.email_recipients}")
            logger.info(f"Content: {content}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    async def _send_slack_notification(self, content: Dict[str, Any], notification_type: NotificationType):
        """Send Slack notification."""
        if not self.notification_config.slack_webhook:
            return
        
        try:
            # Slack webhook implementation would go here
            logger.info(f"Slack notification would be sent to: {self.notification_config.slack_webhook}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    async def _send_teams_notification(self, content: Dict[str, Any], notification_type: NotificationType):
        """Send Microsoft Teams notification."""
        if not self.notification_config.teams_webhook:
            return
        
        try:
            # Teams webhook implementation would go here
            logger.info(f"Teams notification would be sent to: {self.notification_config.teams_webhook}")
            
        except Exception as e:
            logger.error(f"Failed to send Teams notification: {e}")
    
    async def _cleanup_old_executions(self):
        """Clean up old execution files."""
        try:
            max_history = self.config.get('max_execution_history', 50)
            
            if len(self.execution_history) > max_history:
                # Remove oldest executions
                to_remove = len(self.execution_history) - max_history
                removed_executions = self.execution_history[:to_remove]
                self.execution_history = self.execution_history[to_remove:]
                
                # Remove state files
                for execution in removed_executions:
                    state_file = self.state_dir / f"{execution.execution_id}.pkl"
                    if state_file.exists():
                        state_file.unlink()
                
                logger.info(f"Cleaned up {to_remove} old execution records")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old executions: {e}")
    
    def _save_execution_state(self, execution: WorkflowExecution):
        """Save execution state to disk."""
        try:
            state_file = self.state_dir / f"{execution.execution_id}.pkl"
            with open(state_file, 'wb') as f:
                pickle.dump(execution, f)
                
        except Exception as e:
            logger.error(f"Failed to save execution state: {e}")
    
    def _load_execution_history(self):
        """Load execution history from disk."""
        try:
            for state_file in self.state_dir.glob("*.pkl"):
                try:
                    with open(state_file, 'rb') as f:
                        execution = pickle.load(f)
                        self.execution_history.append(execution)
                except Exception as e:
                    logger.warning(f"Failed to load execution state from {state_file}: {e}")
            
            # Sort by start time
            self.execution_history.sort(key=lambda x: x.start_time)
            logger.info(f"Loaded {len(self.execution_history)} execution records")
            
        except Exception as e:
            logger.error(f"Failed to load execution history: {e}")
    
    async def cancel_workflow(self, execution_id: Optional[str] = None):
        """Cancel running workflow."""
        if self.current_execution and (not execution_id or self.current_execution.execution_id == execution_id):
            self.current_execution.state = WorkflowState.CANCELLED
            self.current_execution.end_time = datetime.now()
            await self.event_bus.emit('workflow.cancelled', {'execution': self.current_execution})
            logger.info(f"Workflow cancelled: {self.current_execution.execution_id}")
    
    def get_execution_status(self, execution_id: Optional[str] = None) -> Optional[WorkflowExecution]:
        """Get execution status."""
        if execution_id:
            for execution in self.execution_history:
                if execution.execution_id == execution_id:
                    return execution
            return None
        else:
            return self.current_execution
    
    def get_execution_history(self, limit: int = 10) -> List[WorkflowExecution]:
        """Get execution history."""
        return self.execution_history[-limit:]
    
    async def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow performance metrics."""
        if not self.execution_history:
            return {}
        
        completed_executions = [e for e in self.execution_history if e.state == WorkflowState.COMPLETED]
        failed_executions = [e for e in self.execution_history if e.state == WorkflowState.FAILED]
        
        metrics = {
            'total_executions': len(self.execution_history),
            'successful_executions': len(completed_executions),
            'failed_executions': len(failed_executions),
            'success_rate': len(completed_executions) / len(self.execution_history) * 100,
            'avg_duration': sum(e.metrics.get('total_duration', 0) for e in completed_executions) / len(completed_executions) if completed_executions else 0,
            'avg_data_quality': sum(e.metrics.get('data_quality_score', 0) for e in completed_executions) / len(completed_executions) if completed_executions else 0,
            'last_execution': self.execution_history[-1].to_dict() if self.execution_history else None
        }
        
        return metrics
    
    # Event handlers
    async def _on_workflow_started(self, event_data: Dict[str, Any]):
        """Handle workflow started event."""
        logger.info(f"Workflow started: {event_data['execution'].execution_id}")
    
    async def _on_workflow_completed(self, event_data: Dict[str, Any]):
        """Handle workflow completed event."""
        execution = event_data['execution']
        logger.info(f"Workflow completed: {execution.execution_id} in {execution.metrics.get('total_duration', 0):.1f}s")
    
    async def _on_workflow_failed(self, event_data: Dict[str, Any]):
        """Handle workflow failed event."""
        execution = event_data['execution']
        error = event_data['error']
        logger.error(f"Workflow failed: {execution.execution_id} - {error}")
    
    async def _on_data_extraction_completed(self, event_data: Dict[str, Any]):
        """Handle data extraction completed event."""
        logger.info(f"Data extraction completed with quality score: {event_data['data_quality']:.2f}")
    
    async def _on_insight_generation_completed(self, event_data: Dict[str, Any]):
        """Handle insight generation completed event."""
        logger.info(f"Generated {event_data['insights_count']} insights ({event_data['critical_count']} critical)")
    
    async def _on_slide_generation_completed(self, event_data: Dict[str, Any]):
        """Handle slide generation completed event."""
        logger.info(f"Presentation generated: {event_data['presentation_path']}")