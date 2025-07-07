import logging
from fastapi import WebSocket
from .state_machine import StateMachine, ConnectionState
from .event_dispatcher import EventDispatcher, Event

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, state_machine: StateMachine, event_dispatcher: EventDispatcher):
        self.state_machine = state_machine
        self.event_dispatcher = event_dispatcher

    async def handle_connect(self, session_id: str, websocket: WebSocket):
        """Handles a new client connection."""
        session = self.state_machine.get_or_create_session(session_id)
        session.transition_connection(ConnectionState.CONNECTING)
        
        await self.event_dispatcher.connect(session_id, websocket)
        
        session.transition_connection(ConnectionState.CONNECTED)
        
        # Send a session ready event to the client
        ready_event = Event(type="session.ready", data={"session_id": session_id, "status": "Connected and ready."})
        await self.event_dispatcher.dispatch_event(session_id, ready_event)
        
        session.transition_connection(ConnectionState.READY)

    def handle_disconnect(self, session_id: str):
        """Handles a client disconnection."""
        session = self.state_machine.get_session(session_id)
        if session:
            session.transition_connection(ConnectionState.DISCONNECTED)
            self.state_machine.remove_session(session_id)
        
        self.event_dispatcher.disconnect(session_id)
        logger.info(f"Session {session_id} has been fully cleaned up.")

    async def handle_event(self, session_id: str, event_data: dict):
        """Handles an incoming event from a client."""
        session = self.state_machine.get_session(session_id)
        if not session:
            logger.warning(f"Received event for unknown session: {session_id}")
            return

        event_type = event_data.get("type")
        
        # Simple echo for now to validate connection
        # In the next phase, this will route to appropriate services
        logger.info(f"Processing event '{event_type}' for session {session_id}")
        
        response_event = Event(
            type="session.ready", # Echoing back a known event type for now
            data={"status": f"Received your event of type '{event_type}'"}
        )
        await self.event_dispatcher.dispatch_event(session_id, response_event)