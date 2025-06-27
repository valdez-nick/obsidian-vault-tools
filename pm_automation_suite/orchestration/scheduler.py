"""
Scheduler

Handles time-based and event-driven scheduling for workflows:
- Cron-based scheduling
- Event-driven triggers
- One-time scheduled executions
- Recurring schedules with timezone support
- Schedule conflict resolution
"""

from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
from dataclasses import dataclass, field
import croniter
import pytz

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of schedules supported."""
    CRON = "cron"
    INTERVAL = "interval"
    ONE_TIME = "one_time"
    EVENT = "event"


class ScheduleStatus(Enum):
    """Schedule status."""
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    COMPLETED = "completed"


@dataclass
class Schedule:
    """
    Represents a workflow schedule.
    
    Attributes:
        id: Unique schedule identifier
        workflow_id: ID of workflow to execute
        schedule_type: Type of schedule
        schedule_config: Configuration for the schedule
        timezone: Timezone for schedule execution
        start_date: When the schedule becomes active
        end_date: When the schedule expires
        status: Current schedule status
        context: Context to pass to workflow execution
    """
    id: str
    workflow_id: str
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]
    timezone: str = "UTC"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    context: Dict[str, Any] = field(default_factory=dict)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0


class Scheduler:
    """
    Manages workflow scheduling and execution triggers.
    """
    
    def __init__(self, workflow_engine):
        """
        Initialize the scheduler.
        
        Args:
            workflow_engine: WorkflowEngine instance for executing workflows
        """
        self.workflow_engine = workflow_engine
        self.schedules: Dict[str, Schedule] = {}
        self.event_handlers: Dict[str, List[str]] = {}
        self._scheduler_task = None
        self._running = False
        
    def add_schedule(self, schedule: Schedule) -> bool:
        """
        Add a new schedule.
        
        Args:
            schedule: Schedule to add
            
        Returns:
            True if schedule added successfully
        """
        logger.info(f"Adding schedule: {schedule.id} for workflow {schedule.workflow_id}")
        self.schedules[schedule.id] = schedule
        self._calculate_next_run(schedule)
        return True
        
    def remove_schedule(self, schedule_id: str) -> bool:
        """
        Remove a schedule.
        
        Args:
            schedule_id: ID of schedule to remove
            
        Returns:
            True if schedule removed successfully
        """
        if schedule_id in self.schedules:
            logger.info(f"Removing schedule: {schedule_id}")
            del self.schedules[schedule_id]
            return True
        return False
        
    def pause_schedule(self, schedule_id: str) -> bool:
        """
        Pause a schedule.
        
        Args:
            schedule_id: ID of schedule to pause
            
        Returns:
            True if schedule paused successfully
        """
        if schedule_id in self.schedules:
            logger.info(f"Pausing schedule: {schedule_id}")
            self.schedules[schedule_id].status = ScheduleStatus.PAUSED
            return True
        return False
        
    def resume_schedule(self, schedule_id: str) -> bool:
        """
        Resume a paused schedule.
        
        Args:
            schedule_id: ID of schedule to resume
            
        Returns:
            True if schedule resumed successfully
        """
        if schedule_id in self.schedules:
            logger.info(f"Resuming schedule: {schedule_id}")
            self.schedules[schedule_id].status = ScheduleStatus.ACTIVE
            self._calculate_next_run(self.schedules[schedule_id])
            return True
        return False
        
    def add_cron_schedule(self, workflow_id: str, cron_expression: str,
                         timezone: str = "UTC", context: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a cron-based schedule.
        
        Args:
            workflow_id: Workflow to schedule
            cron_expression: Cron expression (e.g., "0 9 * * MON")
            timezone: Timezone for cron execution
            context: Optional context for workflow
            
        Returns:
            Created schedule ID
        """
        schedule_id = f"cron_{workflow_id}_{datetime.now().timestamp()}"
        schedule = Schedule(
            id=schedule_id,
            workflow_id=workflow_id,
            schedule_type=ScheduleType.CRON,
            schedule_config={"expression": cron_expression},
            timezone=timezone,
            context=context or {}
        )
        self.add_schedule(schedule)
        return schedule_id
        
    def add_interval_schedule(self, workflow_id: str, interval_minutes: int,
                            context: Optional[Dict[str, Any]] = None) -> str:
        """
        Add an interval-based schedule.
        
        Args:
            workflow_id: Workflow to schedule
            interval_minutes: Minutes between executions
            context: Optional context for workflow
            
        Returns:
            Created schedule ID
        """
        schedule_id = f"interval_{workflow_id}_{datetime.now().timestamp()}"
        schedule = Schedule(
            id=schedule_id,
            workflow_id=workflow_id,
            schedule_type=ScheduleType.INTERVAL,
            schedule_config={"interval_minutes": interval_minutes},
            context=context or {}
        )
        self.add_schedule(schedule)
        return schedule_id
        
    def register_event_trigger(self, event_name: str, workflow_id: str) -> bool:
        """
        Register a workflow to trigger on an event.
        
        Args:
            event_name: Name of the event
            workflow_id: Workflow to trigger
            
        Returns:
            True if registration successful
        """
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(workflow_id)
        logger.info(f"Registered {workflow_id} for event {event_name}")
        return True
        
    async def trigger_event(self, event_name: str, event_data: Dict[str, Any]):
        """
        Trigger workflows registered for an event.
        
        Args:
            event_name: Name of the event
            event_data: Data to pass to workflows
        """
        if event_name in self.event_handlers:
            logger.info(f"Triggering event: {event_name}")
            for workflow_id in self.event_handlers[event_name]:
                await self.workflow_engine.execute_workflow(workflow_id, event_data)
                
    async def start(self):
        """Start the scheduler."""
        logger.info("Starting scheduler")
        self._running = True
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
        
    async def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping scheduler")
        self._running = False
        if self._scheduler_task:
            await self._scheduler_task
            
    async def _run_scheduler(self):
        """Main scheduler loop."""
        while self._running:
            await self._check_schedules()
            await asyncio.sleep(60)  # Check every minute
            
    async def _check_schedules(self):
        """Check and execute due schedules."""
        now = datetime.now(pytz.UTC)
        for schedule in self.schedules.values():
            if (schedule.status == ScheduleStatus.ACTIVE and
                schedule.next_run and schedule.next_run <= now):
                await self._execute_schedule(schedule)
                
    async def _execute_schedule(self, schedule: Schedule):
        """Execute a scheduled workflow."""
        logger.info(f"Executing scheduled workflow: {schedule.workflow_id}")
        try:
            await self.workflow_engine.execute_workflow(
                schedule.workflow_id,
                schedule.context
            )
            schedule.last_run = datetime.now(pytz.UTC)
            schedule.run_count += 1
            self._calculate_next_run(schedule)
        except Exception as e:
            logger.error(f"Failed to execute schedule {schedule.id}: {e}")
            
    def _calculate_next_run(self, schedule: Schedule):
        """Calculate the next run time for a schedule."""
        if schedule.schedule_type == ScheduleType.CRON:
            cron = croniter.croniter(
                schedule.schedule_config["expression"],
                datetime.now(pytz.timezone(schedule.timezone))
            )
            schedule.next_run = cron.get_next(datetime)
        elif schedule.schedule_type == ScheduleType.INTERVAL:
            interval = timedelta(minutes=schedule.schedule_config["interval_minutes"])
            schedule.next_run = datetime.now(pytz.UTC) + interval
            
    def get_schedule_status(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a schedule.
        
        Args:
            schedule_id: Schedule ID
            
        Returns:
            Schedule status information
        """
        if schedule_id in self.schedules:
            schedule = self.schedules[schedule_id]
            return {
                "id": schedule.id,
                "workflow_id": schedule.workflow_id,
                "status": schedule.status.value,
                "last_run": schedule.last_run.isoformat() if schedule.last_run else None,
                "next_run": schedule.next_run.isoformat() if schedule.next_run else None,
                "run_count": schedule.run_count
            }
        return None