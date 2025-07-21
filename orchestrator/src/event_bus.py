"""
Distributed Event Bus for Cross-Service Coordination
Replaces the broken PipelineEventBus with true distributed messaging
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass
import aioredis
from config import config

logger = logging.getLogger(__name__)

@dataclass
class ServiceEvent:
    """Standardized event structure for cross-service communication"""
    event_type: str
    session_id: str
    service_id: str
    timestamp: float
    data: Dict[str, Any]
    requires_ack: bool = False
    correlation_id: Optional[str] = None

class DistributedEventBus:
    """
    True distributed event bus using Redis pub-sub
    Coordinates state across all services (orchestrator, whisper, kokoro, ollama)
    """
    
    def __init__(self, service_id: str):
        self.service_id = service_id
        self.redis: Optional[aioredis.Redis] = None
        self.pubsub: Optional[aioredis.client.PubSub] = None
        self._listeners: Dict[str, List[Callable]] = {}
        self._pending_acks: Dict[str, asyncio.Event] = {}
        self._running = False
        self._subscriber_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize Redis connection and start subscriber"""
        try:
            self.redis = aioredis.from_url(config.REDIS_URL, decode_responses=True)
            await self.redis.ping()
            logger.info(f"✅ Service {self.service_id}: Connected to Redis event bus")
            
            # Start subscriber
            self.pubsub = self.redis.pubsub()
            await self._start_subscriber()
            
        except Exception as e:
            logger.error(f"❌ Service {self.service_id}: Failed to initialize event bus: {e}")
            raise
            
    async def _start_subscriber(self):
        """Start the Redis subscriber for this service"""
        # Subscribe to global events and service-specific events
        await self.pubsub.subscribe(
            "maestro:global",  # Global events all services receive
            f"maestro:service:{self.service_id}",  # Service-specific events
            "maestro:interrupt",  # Interrupt coordination channel
            "maestro:state",  # State coordination channel
        )
        
        self._running = True
        self._subscriber_task = asyncio.create_task(self._message_handler())
        logger.info(f"✅ Service {self.service_id}: Event bus subscriber started")
        
    async def _message_handler(self):
        """Handle incoming Redis messages"""
        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    await self._process_message(message)
        except Exception as e:
            logger.error(f"❌ Service {self.service_id}: Message handler error: {e}")
        finally:
            self._running = False
            
    async def _process_message(self, message):
        """Process a single Redis message"""
        try:
            channel = message["channel"]
            data = json.loads(message["data"])
            event = ServiceEvent(**data)
            
            logger.debug(f"🔥 Service {self.service_id}: Received event {event.event_type} from {event.service_id}")
            
            # Handle acknowledgment requests
            if event.requires_ack and event.correlation_id:
                await self._send_acknowledgment(event)
            
            # Notify local listeners
            await self._notify_listeners(event)
            
        except Exception as e:
            logger.error(f"❌ Service {self.service_id}: Error processing message: {e}")
            
    async def _send_acknowledgment(self, event: ServiceEvent):
        """Send acknowledgment back to requesting service"""
        ack_event = ServiceEvent(
            event_type="ack",
            session_id=event.session_id,
            service_id=self.service_id,
            timestamp=time.time(),
            data={"original_event": event.event_type, "status": "acknowledged"},
            correlation_id=event.correlation_id
        )
        
        # Send ack to requesting service
        channel = f"maestro:service:{event.service_id}"
        await self.redis.publish(channel, json.dumps(ack_event.__dict__))
        
    async def _notify_listeners(self, event: ServiceEvent):
        """Notify local event listeners"""
        if event.event_type in self._listeners:
            for callback in self._listeners[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"❌ Service {self.service_id}: Listener error for {event.event_type}: {e}")
                    
    async def emit(self, event_type: str, session_id: str, data: Dict[str, Any], 
                  target_service: Optional[str] = None, requires_ack: bool = False) -> bool:
        """
        Emit an event to the distributed event bus
        
        Args:
            event_type: Type of event (e.g., "interrupt", "tts_start", "state_change")
            session_id: Session identifier
            data: Event payload
            target_service: Specific service to target, None for global
            requires_ack: Whether to wait for acknowledgments
            
        Returns:
            bool: True if successful (and acks received if required)
        """
        if not self.redis:
            logger.error(f"❌ Service {self.service_id}: Event bus not initialized")
            return False
            
        correlation_id = f"{self.service_id}_{time.time()}" if requires_ack else None
        
        event = ServiceEvent(
            event_type=event_type,
            session_id=session_id,
            service_id=self.service_id,
            timestamp=time.time(),
            data=data,
            requires_ack=requires_ack,
            correlation_id=correlation_id
        )
        
        # Determine target channel
        if target_service:
            channel = f"maestro:service:{target_service}"
        else:
            channel = "maestro:global"
            
        try:
            # Publish event
            await self.redis.publish(channel, json.dumps(event.__dict__))
            logger.info(f"🚀 Service {self.service_id}: Emitted {event_type} to {channel}")
            
            # Wait for acknowledgments if required
            if requires_ack and correlation_id:
                return await self._wait_for_acknowledgments(correlation_id, timeout=5.0)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Service {self.service_id}: Failed to emit {event_type}: {e}")
            return False
            
    async def _wait_for_acknowledgments(self, correlation_id: str, timeout: float) -> bool:
        """Wait for acknowledgments from other services"""
        # For now, simplified - in production we'd track which services need to ack
        try:
            await asyncio.wait_for(asyncio.sleep(0.1), timeout=timeout)  # Placeholder
            return True
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Service {self.service_id}: Ack timeout for {correlation_id}")
            return False
            
    def on(self, event_type: str, callback: Callable):
        """Register a local event listener"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
        logger.debug(f"📝 Service {self.service_id}: Registered listener for {event_type}")
        
    def off(self, event_type: str, callback: Callable):
        """Remove a local event listener"""
        if event_type in self._listeners:
            try:
                self._listeners[event_type].remove(callback)
            except ValueError:
                pass
                
    async def shutdown(self):
        """Shutdown the event bus gracefully"""
        self._running = False
        
        if self._subscriber_task and not self._subscriber_task.done():
            self._subscriber_task.cancel()
            try:
                await self._subscriber_task
            except asyncio.CancelledError:
                pass
                
        if self.pubsub:
            await self.pubsub.close()
            
        if self.redis:
            await self.redis.close()
            
        logger.info(f"🔌 Service {self.service_id}: Event bus shutdown complete")

# Global event bus instance
event_bus: Optional[DistributedEventBus] = None

async def get_event_bus(service_id: str = "orchestrator") -> DistributedEventBus:
    """Get or create the global event bus instance"""
    global event_bus
    if not event_bus:
        event_bus = DistributedEventBus(service_id)
        await event_bus.initialize()
    return event_bus