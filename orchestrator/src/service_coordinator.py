"""
Service Coordinator - Manages state across all Maestro services
Implements the Daily.co pattern of coordinated interrupts and state management
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Set, List
from dataclasses import dataclass
from enum import Enum
from event_bus import get_event_bus, ServiceEvent, DistributedEventBus

logger = logging.getLogger(__name__)

class ServiceState(Enum):
    IDLE = "idle"
    LISTENING = "listening"      # EAR state
    TRANSCRIBING = "transcribing"  # EAR processing
    THINKING = "thinking"        # BRAIN processing  
    SPEAKING = "speaking"        # MOUTH active
    INTERRUPTED = "interrupted"  # Service halted
    READY = "ready"             # Confirmed ready for new input
    ERROR = "error"             # Service error state

class ServiceType(Enum):
    ORCHESTRATOR = "orchestrator"
    EAR = "ear"          # WhisperLive STT
    BRAIN = "brain"      # Ollama LLM
    MOUTH = "mouth"      # Kokoro TTS

@dataclass
class ServiceStatus:
    service_type: ServiceType
    state: ServiceState
    session_id: str
    last_update: float
    metadata: Dict[str, Any]

class ServiceCoordinator:
    """
    Coordinates state transitions across all Maestro services
    Ensures proper interrupt handling and state consistency
    """
    
    def __init__(self, service_id: str = "orchestrator"):
        self.service_id = service_id
        self.event_bus: Optional[DistributedEventBus] = None
        
        # Track service states per session
        self.service_states: Dict[str, Dict[ServiceType, ServiceStatus]] = {}
        
        # Track pending operations per session
        self.pending_interrupts: Dict[str, Set[ServiceType]] = {}
        self.pending_ready_confirmations: Dict[str, Set[ServiceType]] = {}
        
        # Expected services for full coordination
        self.required_services = {ServiceType.EAR, ServiceType.BRAIN, ServiceType.MOUTH}
        
    async def initialize(self):
        """Initialize service coordinator with event bus"""
        self.event_bus = await get_event_bus(self.service_id)
        
        # Register for coordination events
        self.event_bus.on("service_state_changed", self._handle_state_change)
        self.event_bus.on("interrupt_request", self._handle_interrupt_request)
        self.event_bus.on("service_ready", self._handle_service_ready)
        self.event_bus.on("ack", self._handle_acknowledgment)
        
        logger.info(f"✅ Service Coordinator initialized for {self.service_id}")
        
    async def _handle_state_change(self, event: ServiceEvent):
        """Handle service state change notifications"""
        session_id = event.session_id
        service_type = ServiceType(event.data.get("service_type"))
        new_state = ServiceState(event.data.get("state"))
        
        # Update our tracking
        if session_id not in self.service_states:
            self.service_states[session_id] = {}
            
        self.service_states[session_id][service_type] = ServiceStatus(
            service_type=service_type,
            state=new_state,
            session_id=session_id,
            last_update=event.timestamp,
            metadata=event.data.get("metadata", {})
        )
        
        logger.info(f"🔄 Session {session_id}: {service_type.value} → {new_state.value}")
        
        # Check for automatic state transitions
        await self._check_coordination_rules(session_id)
        
    async def _handle_interrupt_request(self, event: ServiceEvent):
        """Handle interrupt requests with full service coordination"""
        session_id = event.session_id
        logger.info(f"🛑 INTERRUPT REQUEST for session {session_id}")
        
        # Start coordinated interrupt
        await self._coordinate_interrupt(session_id, event.data.get("reason", "user_interrupt"))
        
    async def _handle_service_ready(self, event: ServiceEvent):
        """Handle service ready confirmations"""
        session_id = event.session_id
        service_type = ServiceType(event.data.get("service_type"))
        
        if session_id in self.pending_ready_confirmations:
            self.pending_ready_confirmations[session_id].discard(service_type)
            logger.info(f"✅ Session {session_id}: {service_type.value} confirmed READY")
            
            # Check if all services are ready
            if not self.pending_ready_confirmations[session_id]:
                logger.info(f"🎯 Session {session_id}: ALL SERVICES READY - can process new input")
                await self._emit_all_services_ready(session_id)
                
    async def _handle_acknowledgment(self, event: ServiceEvent):
        """Handle acknowledgments from services"""
        # Track acknowledgments for coordination
        logger.debug(f"✅ Ack received from {event.service_id}: {event.data}")
        
    async def _coordinate_interrupt(self, session_id: str, reason: str):
        """
        Coordinate interrupt across all services - Daily.co pattern
        1. Send halt signals to all services
        2. Wait for acknowledgments  
        3. Flush all buffers
        4. Wait for ready confirmations
        5. Signal coordinator ready for new input
        """
        logger.info(f"🚨 COORDINATING INTERRUPT for session {session_id}: {reason}")
        
        # Step 1: Send interrupt signal to all services
        self.pending_interrupts[session_id] = self.required_services.copy()
        
        interrupt_data = {
            "session_id": session_id,
            "reason": reason,
            "timestamp": time.time(),
            "requires_buffer_flush": True
        }
        
        # Send to each service type
        for service_type in self.required_services:
            await self.event_bus.emit(
                event_type="interrupt_signal",
                session_id=session_id,
                data={**interrupt_data, "target_service": service_type.value},
                target_service=service_type.value,
                requires_ack=True
            )
            
        # Step 2: Set timeout for interrupt coordination
        asyncio.create_task(self._interrupt_timeout(session_id, timeout=3.0))
        
    async def _interrupt_timeout(self, session_id: str, timeout: float):
        """Handle interrupt coordination timeout"""
        await asyncio.sleep(timeout)
        
        if session_id in self.pending_interrupts:
            remaining = self.pending_interrupts[session_id]
            if remaining:
                logger.warning(f"⏰ Session {session_id}: Interrupt timeout - missing acks from: {remaining}")
                # Force interrupt completion
                await self._force_interrupt_completion(session_id)
                
    async def _force_interrupt_completion(self, session_id: str):
        """Force interrupt completion even without all acks"""
        logger.warning(f"🔨 Session {session_id}: Forcing interrupt completion")
        
        # Clear pending interrupts
        if session_id in self.pending_interrupts:
            del self.pending_interrupts[session_id]
            
        # Request ready states
        await self._request_ready_confirmations(session_id)
        
    async def _request_ready_confirmations(self, session_id: str):
        """Request ready confirmations from all services"""
        logger.info(f"📋 Session {session_id}: Requesting ready confirmations")
        
        self.pending_ready_confirmations[session_id] = self.required_services.copy()
        
        ready_request = {
            "session_id": session_id,
            "timestamp": time.time(),
            "clear_buffers": True
        }
        
        # Request ready state from each service
        for service_type in self.required_services:
            await self.event_bus.emit(
                event_type="request_ready_state",
                session_id=session_id,
                data={**ready_request, "target_service": service_type.value},
                target_service=service_type.value
            )
            
        # Set timeout for ready confirmations
        asyncio.create_task(self._ready_timeout(session_id, timeout=2.0))
        
    async def _ready_timeout(self, session_id: str, timeout: float):
        """Handle ready confirmation timeout"""
        await asyncio.sleep(timeout)
        
        if session_id in self.pending_ready_confirmations:
            remaining = self.pending_ready_confirmations[session_id]
            if remaining:
                logger.warning(f"⏰ Session {session_id}: Ready timeout - assuming ready: {remaining}")
                # Force ready state
                del self.pending_ready_confirmations[session_id]
                await self._emit_all_services_ready(session_id)
                
    async def _emit_all_services_ready(self, session_id: str):
        """Emit that all services are ready for new input"""
        await self.event_bus.emit(
            event_type="all_services_ready", 
            session_id=session_id,
            data={
                "timestamp": time.time(),
                "message": "All services ready for new input"
            }
        )
        logger.info(f"🎯 Session {session_id}: Emitted ALL_SERVICES_READY")
        
    async def _check_coordination_rules(self, session_id: str):
        """Check if automatic coordination is needed based on state changes"""
        if session_id not in self.service_states:
            return
            
        states = self.service_states[session_id]
        
        # Example rule: If MOUTH starts speaking, notify all services
        mouth_status = states.get(ServiceType.MOUTH)
        if mouth_status and mouth_status.state == ServiceState.SPEAKING:
            await self.event_bus.emit(
                event_type="mouth_speaking",
                session_id=session_id,
                data={"started_at": mouth_status.last_update}
            )
            
    async def request_interrupt(self, session_id: str, reason: str = "user_interrupt"):
        """Public API to request session interrupt"""
        await self.event_bus.emit(
            event_type="interrupt_request",
            session_id=session_id,
            data={"reason": reason, "requester": self.service_id}
        )
        
    async def report_state_change(self, session_id: str, service_type: ServiceType, 
                                 new_state: ServiceState, metadata: Dict[str, Any] = None):
        """Public API to report service state changes"""
        await self.event_bus.emit(
            event_type="service_state_changed",
            session_id=session_id,
            data={
                "service_type": service_type.value,
                "state": new_state.value,
                "metadata": metadata or {}
            }
        )
        
    async def report_ready(self, session_id: str, service_type: ServiceType):
        """Public API to report service ready state"""
        await self.event_bus.emit(
            event_type="service_ready",
            session_id=session_id,
            data={"service_type": service_type.value}
        )
        
    def get_session_state(self, session_id: str) -> Dict[ServiceType, ServiceStatus]:
        """Get current state of all services for a session"""
        return self.service_states.get(session_id, {})
        
    async def shutdown(self):
        """Shutdown coordinator gracefully"""
        if self.event_bus:
            await self.event_bus.shutdown()
        logger.info(f"🔌 Service Coordinator shutdown complete")

# Global coordinator instance
coordinator: Optional[ServiceCoordinator] = None

async def get_coordinator(service_id: str = "orchestrator") -> ServiceCoordinator:
    """Get or create the global service coordinator"""
    global coordinator
    if not coordinator:
        coordinator = ServiceCoordinator(service_id)
        await coordinator.initialize()
    return coordinator