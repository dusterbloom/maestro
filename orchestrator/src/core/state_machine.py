
from enum import Enum
from typing import Dict, Any

class SessionState(Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"

class SpeakerState(Enum):
    UNKNOWN = "unknown"
    IDENTIFYING = "identifying"
    RECOGNIZED = "recognized"
    NEW = "new"

class ConversationState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state: SessionState = SessionState.CONNECTING
        self.speaker_state: SpeakerState = SpeakerState.UNKNOWN
        self.speaker_id: str | None = None
        self.conversation_history: list = []
        self.active_request_id: str | None = None

    def transition_to(self, new_state: SessionState):
        # Add logic for valid state transitions
        self.state = new_state

class StateMachine:
    def __init__(self):
        self._sessions: Dict[str, ConversationState] = {}

    def get_or_create_session(self, session_id: str) -> ConversationState:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationState(session_id)
        return self._sessions[session_id]

    def get_session(self, session_id: str) -> ConversationState | None:
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]
