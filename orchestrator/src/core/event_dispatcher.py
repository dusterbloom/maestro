import asyncio
import logging
from fastapi import WebSocket
from typing import Dict, List, Any, Literal

logger = logging.getLogger(__name__)

# Define structured event types for type hinting and clarity
EventType = Literal[
    "session.ready",
    "session.error",
    "audio.start",
    "audio.chunk",
    "audio.end",
    "transcript.partial",
    "transcript.final",
    "response.audio.start",
    "response.audio.chunk",
    "response.audio.end",
    "speaker.identified",
]

class Event:
    """A structured class for events."""
    def __init__(self, type: EventType, data: Dict[str, Any] = None):
        self.type = type
        self.data = data or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "data": self.data}

class EventDispatcher:
    """Handles sending events to connected clients."""
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[session_id] = websocket
        logger.info(f"WebSocket connected for session: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.connections:
            del self.connections[session_id]
            logger.info(f"WebSocket disconnected for session: {session_id}")

    async def dispatch_event(self, session_id: str, event: Event):
        if session_id in self.connections:
            websocket = self.connections[session_id]
            try:
                # Check if WebSocket is still connected before sending
                if websocket.client_state.name != "CONNECTED":
                    logger.warning(f"WebSocket for session {session_id} is not connected (state: {websocket.client_state.name}), removing connection")
                    self.disconnect(session_id)
                    return
                
                logger.info(f"Dispatching event type '{event.type}' to session {session_id}")
                await websocket.send_json(event.to_dict())
                logger.info(f"Successfully dispatched event type '{event.type}' to session {session_id}")
            except Exception as e:
                logger.error(f"Failed to send event to {session_id}: {e}")
                # Handle connection error, maybe disconnect
                self.disconnect(session_id)

    async def broadcast_event(self, event: Event):
        for session_id in list(self.connections.keys()):
            await self.dispatch_event(session_id, event)