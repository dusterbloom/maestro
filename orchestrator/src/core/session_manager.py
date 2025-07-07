import asyncio
import logging
import base64
from fastapi import WebSocket
from .state_machine import StateMachine, ConnectionState, AudioStateStatus, Session
from .event_dispatcher import EventDispatcher, Event
from ..services.stt_service import STTService
from ..services.speaker_service import SpeakerService
from ..services.conversation_service import ConversationService
from ..services.tts_service import TTSService
from ..services.voice_service import VoiceService
from ..services.memory_service import MemoryService
from ..utils.deduplicator import RequestDeduplicator

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, state_machine: StateMachine, event_dispatcher: EventDispatcher):
        self.state_machine = state_machine
        self.event_dispatcher = event_dispatcher
        
        # Initialize singleton/shared services
        self.deduplicator = RequestDeduplicator()
        self.voice_service = VoiceService()
        self.memory_service = MemoryService()
        self.speaker_service = SpeakerService(self.voice_service, self.memory_service, self.deduplicator)
        self.conversation_service = ConversationService()
        self.tts_service = TTSService()
        
        # Session-specific services
        self.stt_services: dict[str, STTService] = {}

    async def handle_connect(self, session_id: str, websocket: WebSocket):
        session = self.state_machine.get_or_create_session(session_id)
        session.transition_connection(ConnectionState.CONNECTING)
        
        await self.event_dispatcher.connect(session_id, websocket)
        
        # Initialize services that require a running event loop
        await self.memory_service.initialize_chroma_client()

        stt_service = STTService(session_id, event_callback=lambda event: asyncio.create_task(self.handle_event(session_id, event)))
        await stt_service.connect()
        self.stt_services[session_id] = stt_service
        
        session.transition_connection(ConnectionState.CONNECTED)
        await self.event_dispatcher.dispatch_event(session_id, Event(type="session.ready", data={"session_id": session_id}))
        session.transition_connection(ConnectionState.READY)

    def handle_disconnect(self, session_id: str):
        session = self.state_machine.get_session(session_id)
        if session:
            session.transition_connection(ConnectionState.DISCONNECTED)
            self.state_machine.remove_session(session_id)
        
        self.event_dispatcher.disconnect(session_id)
        
        if session_id in self.stt_services:
            asyncio.create_task(self.stt_services[session_id].close())
            del self.stt_services[session_id]
            
        logger.info(f"Session {session_id} cleaned up.")

    async def handle_event(self, session_id: str, event_data: dict):
        session = self.state_machine.get_session(session_id)
        if not session:
            return

        event_type = event_data.get("type")
        data = event_data.get("data", {})
        
        if event_type == "audio.chunk":
            audio_bytes = base64.b64decode(data.get("audio_chunk", ""))
            await asyncio.gather(
                self.stt_services[session_id].send_audio(audio_bytes),
                self.speaker_service.process(audio_bytes, {"session": session})
            )
        
        elif event_type == "transcript.final":
            await self.handle_transcript(session, data.get("transcript", ""))

    async def handle_transcript(self, session: Session, transcript: str):
        if not transcript.strip():
            return

        session.transition_audio(AudioStateStatus.PROCESSING)
        
        llm_result = await self.conversation_service.process(transcript, {"session": session})
        
        if not llm_result.success:
            await self.event_dispatcher.dispatch_event(session.session_id, Event(type="session.error", data={"message": llm_result.error}))
            session.transition_audio(AudioStateStatus.IDLE)
            return

        response_text = llm_result.data
        tts_result = await self.tts_service.process(response_text)
        
        if not tts_result.success:
            await self.event_dispatcher.dispatch_event(session.session_id, Event(type="session.error", data={"message": tts_result.error}))
            session.transition_audio(AudioStateStatus.IDLE)
            return

        session.transition_audio(AudioStateStatus.PLAYING)
        await self.event_dispatcher.dispatch_event(session.session_id, Event(
            type="response.audio.chunk",
            data={"audio_chunk": base64.b64encode(tts_result.data).decode('utf-8')}
        ))
        session.transition_audio(AudioStateStatus.IDLE)