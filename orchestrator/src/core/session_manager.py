import asyncio
import logging
import base64
import time
from fastapi import WebSocket
from core.state_machine import StateMachine, ConnectionState, AudioStateStatus, Session
from core.event_dispatcher import EventDispatcher, Event
from services.stt_service import STTService
from services.speaker_service import SpeakerService
from services.conversation_service import ConversationService
from services.tts_service import TTSService
from services.voice_service import VoiceService
from services.memory_service import MemoryService
from utils.deduplicator import RequestDeduplicator

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
            logger.info(f"Received transcript.final event for session {session_id}")
            transcript = data.get("transcript", "")
            # Forward transcript to frontend for display
            await self.event_dispatcher.dispatch_event(session_id, Event(
                type="transcript.final",
                data={"transcript": transcript}
            ))
            await self.handle_transcript(session, transcript)

    async def handle_transcript(self, session: Session, transcript: str):
        start_time = time.time()
        logger.info(f"Starting transcript processing for session {session.session_id}: '{transcript}' at {start_time}")
        
        if not transcript.strip():
            logger.warning(f"Empty transcript received for session {session.session_id}")
            return

        session.transition_audio(AudioStateStatus.PROCESSING)
        
        # LLM Processing
        llm_start = time.time()
        logger.info(f"Starting LLM processing for session {session.session_id}")
        llm_result = await self.conversation_service.process(transcript, {"session": session})
        llm_duration = time.time() - llm_start
        logger.info(f"LLM processing completed in {llm_duration:.3f}s for session {session.session_id}")
        
        if not llm_result.success:
            logger.error(f"LLM processing failed for session {session.session_id}: {llm_result.error}")
            await self.event_dispatcher.dispatch_event(session.session_id, Event(type="session.error", data={"message": llm_result.error}))
            session.transition_audio(AudioStateStatus.IDLE)
            return

        response_text = llm_result.data
        logger.info(f"LLM response for session {session.session_id}: '{response_text}'")
        
        # TTS Processing
        tts_start = time.time()
        logger.info(f"Starting TTS processing for session {session.session_id}")
        tts_result = await self.tts_service.process(response_text)
        tts_duration = time.time() - tts_start
        logger.info(f"TTS processing completed in {tts_duration:.3f}s for session {session.session_id}")
        
        if not tts_result.success:
            logger.error(f"TTS processing failed for session {session.session_id}: {tts_result.error}")
            await self.event_dispatcher.dispatch_event(session.session_id, Event(type="session.error", data={"message": tts_result.error}))
            session.transition_audio(AudioStateStatus.IDLE)
            return

        session.transition_audio(AudioStateStatus.PLAYING)
        audio_chunk_size = len(tts_result.data)
        
        # Audio Dispatch
        dispatch_start = time.time()
        logger.info(f"Dispatching audio chunk of {audio_chunk_size} bytes to session {session.session_id}")
        await self.event_dispatcher.dispatch_event(session.session_id, Event(
            type="response.audio.chunk",
            data={"audio_chunk": base64.b64encode(tts_result.data).decode('utf-8')}
        ))
        dispatch_duration = time.time() - dispatch_start
        
        total_duration = time.time() - start_time
        logger.info(f"Complete pipeline for session {session.session_id}: Total={total_duration:.3f}s (LLM={llm_duration:.3f}s, TTS={tts_duration:.3f}s, Dispatch={dispatch_duration:.3f}s)")
        session.transition_audio(AudioStateStatus.IDLE)