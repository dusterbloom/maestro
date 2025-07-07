import asyncio
import logging
import numpy as np
from orchestrator.src.core.state_machine import Session, SpeakerStateStatus
from orchestrator.src.services.base_service import BaseService, ServiceResult
from orchestrator.src.services.voice_service import VoiceService, AudioBufferManager
from orchestrator.src.services.memory_service import MemoryService
from orchestrator.src.utils.deduplicator import RequestDeduplicator
from orchestrator.src.config import config

logger = logging.getLogger(__name__)

class SpeakerService(BaseService):
    """
    Stateful service to manage speaker identification for each session.
    - Buffers audio per session.
    - Uses VoiceService to get embeddings.
    - Uses MemoryService to persist/retrieve speaker profiles.
    - Deduplicates identification requests to prevent redundant processing.
    """
    def __init__(self, voice_service: VoiceService, memory_service: MemoryService, deduplicator: RequestDeduplicator):
        self.voice_service = voice_service
        self.memory_service = memory_service
        self.deduplicator = deduplicator
        self.session_audio_buffers: dict[str, AudioBufferManager] = {}

    async def process(self, audio_chunk: bytes, context: dict) -> ServiceResult:
        """Accumulates audio and triggers identification when enough data is present."""
        session: Session = context.get("session")
        if not session:
            return ServiceResult(success=False, error="No session context provided.")

        if session.speaker_state == SpeakerStateStatus.RECOGNIZED:
            return ServiceResult(success=True, data={"status": "already_recognized"})

        if session.session_id not in self.session_audio_buffers:
            self.session_audio_buffers[session.session_id] = AudioBufferManager()
        
        buffer_manager = self.session_audio_buffers[session.session_id]
        
        float_array = np.frombuffer(audio_chunk, dtype=np.float32)
        buffer_ready = buffer_manager.add_audio_chunk(float_array)

        if buffer_ready:
            logger.info(f"SpeakerService: Buffer ready for session {session.session_id}.")
            request_id = f"speaker_id_{session.session_id}"
            
            # This task will only be executed if another one with the same ID isn't already running.
            asyncio.create_task(
                self.deduplicator.process_or_join(
                    request_id,
                    self._identify_speaker(session, buffer_manager)
                )
            )
            return ServiceResult(success=True, data={"status": "identification_triggered"})
        
        return ServiceResult(success=True, data={"status": "accumulating"})

    async def _identify_speaker(self, session: Session, buffer_manager: AudioBufferManager):
        """The actual identification logic, wrapped by the deduplicator."""
        session.speaker_state = SpeakerStateStatus.IDENTIFYING
        
        wav_bytes = buffer_manager.get_buffer_as_wav(apply_vad=True)
        if not wav_bytes:
            logger.error(f"Could not get WAV bytes from buffer for session {session.session_id}")
            session.speaker_state = SpeakerStateStatus.UNKNOWN
            return

        embedding = await self.voice_service.get_embedding(wav_bytes)
        if not embedding:
            session.speaker_state = SpeakerStateStatus.UNKNOWN
            return

        # Query ChromaDB for the closest speaker
        results = await self.memory_service.collection.query(query_embeddings=[embedding], n_results=1)
        
        if results and results["ids"][0]:
            user_id = results["ids"][0][0]
            distance = results["distances"][0][0]
            
            if distance < config.SPEAKER_SIMILARITY_THRESHOLD:
                profile = await self.memory_service.get_speaker_profile(user_id)
                session.speaker_state = SpeakerStateStatus.RECOGNIZED
                session.speaker_id = user_id
                session.speaker_name = profile.get("name", "Friend")
                session.is_new_speaker = False
                logger.info(f"Speaker for session {session.session_id} identified as {session.speaker_name} (ID: {user_id})")
            else:
                # No close match found, register as a new speaker
                await self._register_new_speaker(session, embedding)
        else:
            # No speakers in DB, register this one
            await self._register_new_speaker(session, embedding)

        # Clean up the buffer for this session
        if session.session_id in self.session_audio_buffers:
            del self.session_audio_buffers[session.session_id]

    async def _register_new_speaker(self, session: Session, embedding: list[float]):
        user_id = await self.memory_service.create_speaker_profile(embedding)
        session.speaker_state = SpeakerStateStatus.RECOGNIZED
        session.speaker_id = user_id
        session.speaker_name = f"Speaker {self.memory_service.collection.count()}" # Temporary name
        session.is_new_speaker = True
        logger.info(f"Registered new speaker {session.speaker_name} (ID: {user_id}) for session {session.session_id}")

    async def health_check(self) -> bool:
        return True

    def get_metrics(self) -> dict:
        return {"active_buffers": len(self.session_audio_buffers)}