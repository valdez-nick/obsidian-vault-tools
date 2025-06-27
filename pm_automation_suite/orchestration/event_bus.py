"""
Event Bus

Provides inter-component communication through events:
- Publish/subscribe pattern implementation
- Event filtering and routing
- Async event handling
- Event persistence and replay
- Dead letter queue for failed events
"""

from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime
from enum import Enum
import asyncio
import logging
from dataclasses import dataclass, field
import json
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class Event:
    """
    Represents an event in the system.
    
    Attributes:
        id: Unique event identifier
        type: Event type/name
        source: Source component that generated the event
        data: Event payload
        timestamp: When the event was created
        priority: Event priority
        metadata: Additional event metadata
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    source: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "metadata": self.metadata
        }


class EventFilter:
    """
    Filter for selective event subscription.
    """
    
    def __init__(self, event_types: Optional[List[str]] = None,
                 sources: Optional[List[str]] = None,
                 min_priority: Optional[EventPriority] = None):
        """
        Initialize event filter.
        
        Args:
            event_types: List of event types to include
            sources: List of sources to include
            min_priority: Minimum priority level
        """
        self.event_types = set(event_types) if event_types else None
        self.sources = set(sources) if sources else None
        self.min_priority = min_priority
        
    def matches(self, event: Event) -> bool:
        """
        Check if an event matches this filter.
        
        Args:
            event: Event to check
            
        Returns:
            True if event matches filter criteria
        """
        if self.event_types and event.type not in self.event_types:
            return False
        if self.sources and event.source not in self.sources:
            return False
        if self.min_priority and event.priority.value < self.min_priority.value:
            return False
        return True


class EventBus:
    """
    Central event bus for component communication.
    """
    
    def __init__(self, persist_events: bool = False, 
                 event_store_path: Optional[str] = None):
        """
        Initialize the event bus.
        
        Args:
            persist_events: Whether to persist events to disk
            event_store_path: Path for event persistence
        """
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.filters: Dict[Callable, EventFilter] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.dead_letter_queue: List[Event] = []
        self.persist_events = persist_events
        self.event_store_path = event_store_path
        self._running = False
        self._processor_task = None
        
    def subscribe(self, handler: Callable, event_filter: Optional[EventFilter] = None):
        """
        Subscribe to events.
        
        Args:
            handler: Async function to handle events
            event_filter: Optional filter for events
        """
        # Subscribe to all events if no filter
        self.subscribers["*"].append(handler)
        if event_filter:
            self.filters[handler] = event_filter
        handler_name = getattr(handler, '__name__', str(handler))
        logger.info(f"Subscribed handler {handler_name}")
        
    def unsubscribe(self, handler: Callable):
        """
        Unsubscribe from events.
        
        Args:
            handler: Handler to unsubscribe
        """
        for event_type, handlers in self.subscribers.items():
            if handler in handlers:
                handlers.remove(handler)
        if handler in self.filters:
            del self.filters[handler]
        logger.info(f"Unsubscribed handler {handler.__name__}")
        
    async def publish(self, event: Event):
        """
        Publish an event to the bus.
        
        Args:
            event: Event to publish
        """
        logger.debug(f"Publishing event: {event.type} from {event.source}")
        await self.event_queue.put(event)
        
        if self.persist_events:
            await self._persist_event(event)
            
    async def publish_event(self, event_type: str, source: str, 
                           data: Dict[str, Any], priority: EventPriority = EventPriority.NORMAL):
        """
        Convenience method to publish an event.
        
        Args:
            event_type: Type of event
            source: Source component
            data: Event data
            priority: Event priority
        """
        event = Event(
            type=event_type,
            source=source,
            data=data,
            priority=priority
        )
        await self.publish(event)
        
    async def start(self):
        """Start the event bus processor."""
        logger.info("Starting event bus")
        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        
    async def stop(self):
        """Stop the event bus processor."""
        logger.info("Stopping event bus")
        self._running = False
        if self._processor_task:
            await self._processor_task
            
    async def _process_events(self):
        """Main event processing loop."""
        while self._running:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                await self._dispatch_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                
    async def _dispatch_event(self, event: Event):
        """
        Dispatch an event to all matching subscribers.
        
        Args:
            event: Event to dispatch
        """
        handlers = self._get_matching_handlers(event)
        
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Handler {handler.__name__} failed for event {event.id}: {e}")
                self.dead_letter_queue.append(event)
                
    def _get_matching_handlers(self, event: Event) -> List[Callable]:
        """
        Get all handlers that match an event.
        
        Args:
            event: Event to match
            
        Returns:
            List of matching handlers
        """
        handlers = []
        
        # Get all universal subscribers
        for handler in self.subscribers["*"]:
            # Check if handler has a filter
            if handler in self.filters:
                if self.filters[handler].matches(event):
                    handlers.append(handler)
            else:
                handlers.append(handler)
                
        return handlers
        
    async def _persist_event(self, event: Event):
        """Persist an event to disk."""
        if self.event_store_path:
            # Placeholder for event persistence
            logger.debug(f"Persisting event {event.id}")
            
    async def replay_events(self, start_time: datetime, 
                           end_time: Optional[datetime] = None,
                           event_filter: Optional[EventFilter] = None):
        """
        Replay historical events.
        
        Args:
            start_time: Start time for replay
            end_time: Optional end time
            event_filter: Optional filter for events
        """
        # Placeholder for event replay
        logger.info(f"Replaying events from {start_time}")
        
    def get_dead_letters(self, limit: int = 100) -> List[Event]:
        """
        Get events from dead letter queue.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of failed events
        """
        return self.dead_letter_queue[-limit:]
        
    def retry_dead_letter(self, event_id: str) -> bool:
        """
        Retry a dead letter event.
        
        Args:
            event_id: ID of event to retry
            
        Returns:
            True if event found and retried
        """
        for event in self.dead_letter_queue:
            if event.id == event_id:
                asyncio.create_task(self.publish(event))
                self.dead_letter_queue.remove(event)
                return True
        return False
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get event bus metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "queue_size": self.event_queue.qsize(),
            "dead_letter_count": len(self.dead_letter_queue),
            "subscriber_count": sum(len(handlers) for handlers in self.subscribers.values()),
            "filter_count": len(self.filters)
        }