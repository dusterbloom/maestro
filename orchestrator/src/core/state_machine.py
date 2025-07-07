from enum import Enum
from typing import Dict, Any, List
import time

class ConnectionState(Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    READY = "ready"
    DISCONNECTED = "disconnected"
    ERROR = "error"

class SpeakerStateStatus(Enum):
    UNKNOWN = "unknown"
    IDENTIFYING = "identifying"
    RECOGNIZED = "recognized"
    NEW = "new"

class AudioStateStatus(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    PLAYING = "playing"

class Session:
    """Represents the complete state of a single client session."""
    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.connection_state: ConnectionState = ConnectionState.CONNECTING
        self.speaker_state: SpeakerStateStatus = SpeakerStateStatus.UNKNOWN
        self.audio_state: AudioStateStatus = AudioStateStatus.IDLE
        
        self.speaker_id: str | None = None
        self.speaker_name: str = "Guest"
        self.is_new_speaker: bool = True
        
        self.conversation_history: List[Dict[str, str]] = []
        self.active_request_id: str | None = None
        self.last_activity_ts: float = time.time()

    def transition_connection(self, new_state: ConnectionState):
        # In a real implementation, add validation logic here
        print(f"Session {self.session_id}: Connection state -> {new_state.value}")
        self.connection_state = new_state
        self.last_activity_ts = time.time()

    def transition_audio(self, new_state: AudioStateStatus):
        print(f"Session {self.session_id}: Audio state -> {new_state.value}")
        self.audio_state = new_state
        self.last_activity_ts = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the session state to a dictionary."""
        return {
            "session_id": self.session_id,
            "connection_state": self.connection_state.value,
            "speaker_state": self.speaker_state.value,
            "audio_state": self.audio_state.value,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "last_activity_ts": self.last_activity_ts,
        }

class StateMachine:
    """Manages the state of all active sessions."""
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def get_or_create_session(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            print(f"Creating new session: {session_id}")
            self._sessions[session_id] = Session(session_id)
        return self._sessions[session_id]

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str):
        if session_id in self._sessions:
            print(f"Removing session: {session_id}")
            del self._sessions[session_id]