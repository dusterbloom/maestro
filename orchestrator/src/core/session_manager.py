import asyncio
import logging
import base64
import time
import ollama
from fastapi import WebSocket
from config import config
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
            # Schedule session removal after a short grace period
            async def delayed_cleanup():
                await asyncio.sleep(5)  # 5 second grace period
                # Double-check if session still exists and is disconnected
                s = self.state_machine.get_session(session_id)
                if s and s.connection_state == ConnectionState.DISCONNECTED:
                    self.state_machine.remove_session(session_id)
                    self.event_dispatcher.disconnect(session_id)
                    if session_id in self.stt_services:
                        await self.stt_services[session_id].close()
                        del self.stt_services[session_id]
                    logger.info(f"Session {session_id} cleaned up after grace period.")
                else:
                    logger.info(f"Session {session_id} was reconnected before cleanup.")
            asyncio.create_task(delayed_cleanup())
        else:
            self.event_dispatcher.disconnect(session_id)
            if session_id in self.stt_services:
                asyncio.create_task(self.stt_services[session_id].close())
                del self.stt_services[session_id]
            logger.info(f"Session {session_id} cleaned up (no active session).")

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
            logger.info(f"Received transcript.final event for session {session_id} | event_data: {event_data}")
            transcript = data.get("transcript", "")
            completed = data.get("completed", None)
            logger.info(f"Transcript content: '{transcript}' | completed: {completed}")
            # Forward transcript to frontend for display
            await self.event_dispatcher.dispatch_event(session_id, Event(
                type="transcript.final",
                data={"transcript": transcript, "completed": completed}
            ))
            await self.handle_transcript(session, transcript)

    async def handle_transcript(self, session: Session, transcript: str):
        start_time = time.time()
        logger.info(f"Starting transcript processing for session {session.session_id}: '{transcript}' at {start_time}")
        
        if not transcript.strip():
            logger.warning(f"Empty transcript received for session {session.session_id}")
            return

        # 🔧 CRITICAL FIX: Use RequestDeduplicator to prevent duplicate transcript processing
        # This ensures we only process each unique transcript once, preventing voice floods
        transcript_key = f"{session.session_id}:{transcript.strip()}"
        logger.info(f"Deduplication key: {transcript_key}")

        async def process_transcript():
            logger.info(f"Launching new pipeline for session {session.session_id} with transcript: '{transcript}'")
            return await self._process_transcript_internal(session, transcript.strip(), start_time)
            
        # Use deduplicator - if same transcript is already processing, join that task instead of creating new one
        result = await self.deduplicator.process_or_join(transcript_key, process_transcript())
        logger.info(f"Transcript processing complete for session {session.session_id} | deduplication key: {transcript_key}")
        return result
    
    async def _process_transcript_internal(self, session: Session, transcript: str, start_time: float):

        logger.info(f"[Pipeline] START for session {session.session_id} | transcript: '{transcript}' | audio_state: {session.audio_state}")
        session.transition_audio(AudioStateStatus.PROCESSING)
        
        # Start LLM and TTS in parallel for ultra-low latency
        llm_start = time.time()
        logger.info(f"Starting streaming LLM processing for session {session.session_id}")
        
        # Create streaming LLM task
        async def stream_llm_to_tts():
            """Stream LLM tokens directly to TTS as they arrive"""
            client = ollama.AsyncClient(host=self.conversation_service.ollama_url)
            response = await client.generate(
                model=config.LLM_MODEL,
                prompt=self.conversation_service._build_prompt(transcript, session.conversation_history,
                                                             self.conversation_service._get_agentic_greeting(session)),
                stream=True,
                options={
                    "num_predict": config.LLM_MAX_TOKENS,
                    "temperature": config.LLM_TEMPERATURE,
                }
            )
            
            # Stream tokens and build sentences for immediate TTS
            sentence_buffer = ""
            full_response = ""
            sentence_endings = ['.', '!', '?', '\n']
            
            async for chunk in response:
                if 'response' in chunk:
                    token = chunk['response']
                    sentence_buffer += token
                    full_response += token
                    
                    # Check for sentence completion
                    if any(ending in token for ending in sentence_endings) and len(sentence_buffer.strip()) > 10:
                        # Send complete sentence to TTS immediately
                        logger.info(f"🚀 Streaming sentence to TTS: '{sentence_buffer.strip()}'")
                        async for audio_chunk in self.tts_service.process(sentence_buffer.strip()):
                            await self.event_dispatcher.dispatch_event(session.session_id, Event(
                                type="response.audio.chunk",
                                data={"audio_chunk": base64.b64encode(audio_chunk).decode('utf-8')}
                            ))
                        # Signal end of audio for this sentence
                        await self.event_dispatcher.dispatch_event(session.session_id, Event(
                            type="response.audio.end",
                            data={}
                        ))
                        sentence_buffer = ""  # Reset for next sentence
            
            # Handle any remaining text
            if sentence_buffer.strip():
                logger.info(f"🚀 Final sentence to TTS: '{sentence_buffer.strip()}'")
                async for audio_chunk in self.tts_service.process(sentence_buffer.strip()):
                    await self.event_dispatcher.dispatch_event(session.session_id, Event(
                        type="response.audio.chunk",
                        data={"audio_chunk": base64.b64encode(audio_chunk).decode('utf-8')}
                    ))
                await self.event_dispatcher.dispatch_event(session.session_id, Event(
                    type="response.audio.end",
                    data={}
                ))
            
            # Update conversation history
            session.conversation_history.append({"role": "user", "content": transcript})
            session.conversation_history.append({"role": "assistant", "content": full_response})
            
            return full_response
        
        try:
            logger.info(f"[Pipeline] Attempting to transition to PLAYING for session {session.session_id} | audio_state: {session.audio_state}")
            session.transition_audio(AudioStateStatus.PLAYING)
            full_response = await stream_llm_to_tts()
            total_duration = time.time() - start_time
            logger.info(f"🚀 ULTRA-LOW LATENCY pipeline completed in {total_duration:.3f}s for session {session.session_id}")
            logger.info(f"LLM response for session {session.session_id}: '{full_response}'")
            
        except Exception as e:
            logger.error(f"Streaming pipeline failed for session {session.session_id}: {e}")
            await self.event_dispatcher.dispatch_event(session.session_id, Event(type="session.error", data={"message": str(e)}))
        
        logger.info(f"[Pipeline] END for session {session.session_id} | audio_state: {session.audio_state}")
        session.transition_audio(AudioStateStatus.IDLE)