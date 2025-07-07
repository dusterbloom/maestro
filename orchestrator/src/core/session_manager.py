
import asyncio
import logging
from fastapi import WebSocket
from orchestrator.src.core.state_machine import StateMachine, ConnectionState, AudioStateStatus
from orchestrator.src.core.event_dispatcher import EventDispatcher, Event
from orchestrator.src.services.stt_service import STTService
from orchestrator.src.services.speaker_service import SpeakerService
from orchestrator.src.services.conversation_service import ConversationService
from orchestrator.src.services.tts_service import TTSService
from orchestrator.src.services.voice_service import VoiceService
from orchestrator.src.utils.deduplicator import RequestDeduplicator

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, state_machine: StateMachine, event_dispatcher: EventDispatcher):
        self.state_machine = state_machine
        self.event_dispatcher = event_dispatcher
        
        # Initialize services that will be used by all sessions
        self.deduplicator = RequestDeduplicator()
        self.voice_service = VoiceService() # Assuming it has its own config
        self.speaker_service = SpeakerService(self.voice_service, self.deduplicator)
        self.conversation_service = ConversationService()
        self.tts_service = TTSService()
        
        # Session-specific services
        self.stt_services: dict[str, STTService] = {}

    async def handle_connect(self, session_id: str, websocket: WebSocket):
        """Handles a new client connection and sets up session-specific services."""
        session = self.state_machine.get_or_create_session(session_id)
        session.transition_connection(ConnectionState.CONNECTING)
        
        await self.event_dispatcher.connect(session_id, websocket)
        
        # Create and connect the STT service for this session
        stt_service = STTService(session_id, event_callback=lambda event: asyncio.create_task(self.handle_event(session_id, event)))
        await stt_service.connect()
        self.stt_services[session_id] = stt_service
        
        session.transition_connection(ConnectionState.CONNECTED)
        ready_event = Event(type="session.ready", data={"session_id": session_id})
        await self.event_dispatcher.dispatch_event(session_id, ready_event)
        session.transition_connection(ConnectionState.READY)

    def handle_disconnect(self, session_id: str):
        """Handles a client disconnection and cleans up resources."""
        session = self.state_machine.get_session(session_id)
        if session:
            session.transition_connection(ConnectionState.DISCONNECTED)
            self.state_machine.remove_session(session_id)
        
        self.event_dispatcher.disconnect(session_id)
        
        if session_id in self.stt_services:
            asyncio.create_task(self.stt_services[session_id].close())
            del self.stt_services[session_id]
            
        logger.info(f"Session {session_id} has been fully cleaned up.")

    async def handle_event(self, session_id: str, event_data: dict):
        """The main event router for the orchestrator."""
        session = self.state_machine.get_session(session_id)
        if not session:
            logger.warning(f"Received event for unknown session: {session_id}")
            return

        event_type = event_data.get("type")
        data = event_data.get("data", {})
        
        logger.info(f"Handling event '{event_type}' for session {session_id}")

        if event_type == "audio.chunk":
            audio_chunk = data.get("audio_chunk") # Assuming data is sent as base64 string
            import base64
            audio_bytes = base64.b64decode(audio_chunk)
            
            # Forward to STT and Speaker services in parallel
            await asyncio.gather(
                self.stt_services[session_id].send_audio(audio_bytes),
                self.speaker_service.process(audio_bytes, {"session": session})
            )
        
        elif event_type == "transcript.final":
            transcript = data.get("transcript")
            session.transition_audio(AudioStateStatus.PROCESSING)
            
            # 1. Get conversation response from LLM
            llm_result = await self.conversation_service.process(transcript, {"session": session})
            
            if llm_result.success:
                response_text = llm_result.data
                session.conversation_history.append({"user": transcript, "assistant": response_text})
                
                # 2. Synthesize audio from the response
                tts_result = await self.tts_service.process(response_text)
                
                if tts_result.success:
                    session.transition_audio(AudioStateStatus.PLAYING)
                    # Send audio back to client
                    await self.event_dispatcher.dispatch_event(session_id, Event(
                        type="response.audio.chunk",
                        data={"audio_chunk": base64.b64encode(tts_result.data).decode('utf-8')}
                    ))
                    session.transition_audio(AudioStateStatus.IDLE)
                else:
                    await self.event_dispatcher.dispatch_event(session_id, Event(type="session.error", data={"message": tts_result.error}))
            else:
                await self.event_dispatcher.dispatch_event(session_id, Event(type="session.error", data={"message": llm_result.error}))
        
        # Add other event handlers here...
        else:
            logger.warning(f"Unhandled event type: {event_type}")
