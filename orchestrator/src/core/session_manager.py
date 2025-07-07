
from fastapi import WebSocket
from .state_machine import StateMachine
from .event_dispatcher import EventDispatcher

class SessionManager:
    def __init__(self, state_machine: StateMachine, event_dispatcher: EventDispatcher):
        self.state_machine = state_machine
        self.event_dispatcher = event_dispatcher

    async def handle_connect(self, session_id: str, websocket: WebSocket):
        session = self.state_machine.get_or_create_session(session_id)
        await self.event_dispatcher.connect(session_id, websocket)
        # ... more logic for connection

    def handle_disconnect(self, session_id: str):
        self.state_machine.remove_session(session_id)
        self.event_dispatcher.disconnect(session_id)
        # ... more logic for disconnection
