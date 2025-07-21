"""
Service Interface Standards
Defines the contract that all Maestro services must implement
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import time

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"  
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    status: HealthStatus
    timestamp: float
    details: Dict[str, Any]
    latency_ms: Optional[float] = None

@dataclass
class ServiceInfo:
    service_name: str
    service_type: str  # "ear", "brain", "mouth", "orchestrator"
    version: str
    capabilities: List[str]
    endpoints: Dict[str, str]

class ServiceInterface(ABC):
    """
    Abstract interface that all Maestro services must implement
    Provides standardized REST endpoints and event bus integration
    """
    
    @abstractmethod
    async def health_check(self) -> HealthCheck:
        """Return current health status"""
        pass
        
    @abstractmethod
    async def get_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status for a specific session"""
        pass
        
    @abstractmethod
    async def interrupt(self, session_id: str, reason: str = "user_interrupt") -> bool:
        """Handle interrupt request - must flush buffers and halt processing"""
        pass
        
    @abstractmethod
    async def reset(self, session_id: str) -> bool:
        """Reset to clean state and confirm ready for new input"""  
        pass
        
    @abstractmethod
    async def get_service_info(self) -> ServiceInfo:
        """Return service information and capabilities"""
        pass

# REST Endpoint Standards for All Services
SERVICE_ENDPOINTS = {
    "health": {
        "path": "/health",
        "method": "GET",
        "description": "Health check endpoint",
        "response": "HealthCheck object"
    },
    "status": {
        "path": "/status/{session_id}",
        "method": "GET", 
        "description": "Get session status",
        "response": "Session status object"
    },
    "interrupt": {
        "path": "/interrupt/{session_id}",
        "method": "POST",
        "description": "Interrupt session processing",
        "body": {"reason": "string"},
        "response": "Success boolean"
    },
    "reset": {
        "path": "/reset/{session_id}", 
        "method": "POST",
        "description": "Reset session to clean state",
        "response": "Success boolean"
    },
    "info": {
        "path": "/info",
        "method": "GET",
        "description": "Service information",
        "response": "ServiceInfo object"
    }
}

# Event Bus Message Standards
EVENT_TYPES = {
    # State Management
    "service_state_changed": {
        "description": "Service reports state change",
        "data": {
            "service_type": "ear|brain|mouth|orchestrator",
            "state": "idle|listening|transcribing|thinking|speaking|interrupted|ready|error",
            "metadata": "dict"
        }
    },
    "service_ready": {
        "description": "Service confirms ready for new input",
        "data": {
            "service_type": "string",
            "buffers_cleared": "boolean"
        }
    },
    
    # Interrupt Coordination
    "interrupt_request": {
        "description": "Request coordinated interrupt",
        "data": {
            "reason": "string",
            "requester": "string"
        }
    },
    "interrupt_signal": {
        "description": "Signal to specific service to interrupt",
        "data": {
            "reason": "string",
            "requires_buffer_flush": "boolean",
            "target_service": "string"
        }
    },
    "all_services_ready": {
        "description": "All services ready for new input",
        "data": {
            "message": "string"
        }
    },
    
    # Processing Flow
    "audio_received": {
        "description": "Audio data received by EAR",
        "data": {
            "audio_level": "float",
            "duration_ms": "int"
        }
    },
    "transcription_complete": {
        "description": "EAR completed transcription",
        "data": {
            "text": "string",
            "confidence": "float"
        }
    },
    "brain_thinking": {
        "description": "BRAIN started processing",
        "data": {
            "input_text": "string",
            "model": "string"
        }
    },
    "brain_response_ready": {
        "description": "BRAIN has response ready",
        "data": {
            "response_text": "string",
            "processing_time_ms": "int"
        }
    },
    "mouth_speaking": {
        "description": "MOUTH started speaking",
        "data": {
            "text": "string",
            "voice": "string",
            "estimated_duration_ms": "int"
        }
    },
    "mouth_finished": {
        "description": "MOUTH finished speaking",
        "data": {
            "actual_duration_ms": "int"
        }
    }
}

# Service-Specific Interface Extensions

class EarInterface(ServiceInterface):
    """Interface for STT services (WhisperLive)"""
    
    @abstractmethod
    async def start_listening(self, session_id: str) -> bool:
        """Start listening for audio input"""
        pass
        
    @abstractmethod 
    async def stop_listening(self, session_id: str) -> bool:
        """Stop listening and flush audio buffers"""
        pass
        
    @abstractmethod
    async def get_transcription_status(self, session_id: str) -> Dict[str, Any]:
        """Get current transcription status"""
        pass

class BrainInterface(ServiceInterface):
    """Interface for LLM services (Ollama)"""
    
    @abstractmethod
    async def process_text(self, session_id: str, text: str, 
                          context: List[Dict[str, str]]) -> str:
        """Process text and return response"""
        pass
        
    @abstractmethod
    async def stream_response(self, session_id: str, text: str,
                             context: List[Dict[str, str]]):
        """Stream LLM response tokens"""
        pass
        
    @abstractmethod
    async def stop_generation(self, session_id: str) -> bool:
        """Stop current text generation"""
        pass

class MouthInterface(ServiceInterface):
    """Interface for TTS services (Kokoro)"""
    
    @abstractmethod
    async def speak(self, session_id: str, text: str, voice: str) -> bytes:
        """Generate speech audio from text"""
        pass
        
    @abstractmethod
    async def stop_speaking(self, session_id: str) -> bool:
        """Stop current speech generation"""
        pass
        
    @abstractmethod
    async def get_speaking_status(self, session_id: str) -> Dict[str, Any]:
        """Get current speaking status"""
        pass

# Helper functions for service validation

def validate_service_health(health: HealthCheck) -> bool:
    """Validate health check response"""
    return (
        health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED] and
        health.timestamp > 0 and
        isinstance(health.details, dict)
    )

def validate_service_info(info: ServiceInfo) -> bool:
    """Validate service info response"""
    return (
        bool(info.service_name) and
        info.service_type in ["ear", "brain", "mouth", "orchestrator"] and
        bool(info.version) and
        isinstance(info.capabilities, list) and
        isinstance(info.endpoints, dict)
    )

# Service Registration Helper
def create_service_info(name: str, service_type: str, version: str = "1.0.0") -> ServiceInfo:
    """Create standard ServiceInfo object"""
    capabilities = []
    if service_type == "ear":
        capabilities = ["transcription", "voice_activity_detection", "real_time_streaming"]
    elif service_type == "brain": 
        capabilities = ["text_generation", "streaming_response", "context_management"]
    elif service_type == "mouth":
        capabilities = ["speech_synthesis", "voice_selection", "audio_streaming"]
    elif service_type == "orchestrator":
        capabilities = ["session_management", "service_coordination", "interrupt_handling"]
        
    return ServiceInfo(
        service_name=name,
        service_type=service_type,
        version=version,
        capabilities=capabilities,
        endpoints=SERVICE_ENDPOINTS
    )