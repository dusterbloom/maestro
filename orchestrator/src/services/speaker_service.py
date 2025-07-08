import asyncio
import logging
import numpy as np
from core.state_machine import Session, SpeakerStateStatus
from services.base_service import BaseService, ServiceResult
from utils.audio_buffer import AudioBufferManager
from services.voice_service import VoiceService
from services.memory_service import MemoryService
from utils.deduplicator import RequestDeduplicator
from config import config

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
            async def dedup_task():
                return await self.deduplicator.process_or_join(
                    request_id,
                    self._identify_speaker(session, buffer_manager)
                )
            asyncio.create_task(dedup_task())
            return ServiceResult(success=True, data={"status": "identification_triggered"})
        
        return ServiceResult(success=True, data={"status": "accumulating"})

    async def _identify_speaker(self, session: Session, buffer_manager: AudioBufferManager):
        """The actual identification logic, wrapped by the deduplicator."""
        session.speaker_state = SpeakerStateStatus.IDENTIFYING
        logger.info(f"🎤 Starting speaker identification for session {session.session_id}")
        
        try:
            wav_bytes = buffer_manager.get_buffer_as_wav(apply_vad=True)
            if not wav_bytes:
                logger.warning(f"🎤 VAD rejected audio for session {session.session_id} - proceeding with unknown speaker")
                session.speaker_state = SpeakerStateStatus.UNKNOWN
                session.speaker_name = "Guest"
                self._cleanup_buffer(session.session_id)
                return

            logger.info(f"🎤 Processing {len(wav_bytes)} bytes of audio for speaker identification")
            
            # Get embedding with error handling
            try:
                embedding = await self.voice_service.get_embedding(wav_bytes)
                if not embedding:
                    logger.error(f"🎤 Failed to get embedding for session {session.session_id}")
                    session.speaker_state = SpeakerStateStatus.UNKNOWN
                    session.speaker_name = "Guest"
                    return
                    
                logger.info(f"🎤 Successfully extracted embedding of length {len(embedding)} for session {session.session_id}")
            except Exception as embedding_error:
                logger.error(f"🎤 Embedding extraction failed for session {session.session_id}: {embedding_error}")
                session.speaker_state = SpeakerStateStatus.UNKNOWN
                session.speaker_name = "Guest"
                return

            # Query ChromaDB with error handling
            try:
                if not self.memory_service.collection:
                    logger.error(f"🎤 ChromaDB collection not available for session {session.session_id}")
                    await self._register_new_speaker(session, embedding)
                    return
                    
                results = self.memory_service.collection.query(query_embeddings=[embedding], n_results=1)
                logger.info(f"🎤 ChromaDB query returned {len(results.get('ids', [[]]))} results for session {session.session_id}")
                
                if results and results["ids"][0]:
                    user_id = results["ids"][0][0]
                    distance = results["distances"][0][0]
                    logger.info(f"🎤 Found potential match: user_id={user_id}, distance={distance:.4f}, threshold={config.SPEAKER_SIMILARITY_THRESHOLD}")
                    
                    if distance < config.SPEAKER_SIMILARITY_THRESHOLD:
                        try:
                            profile = await self.memory_service.get_speaker_profile(user_id)
                            session.speaker_state = SpeakerStateStatus.RECOGNIZED
                            session.speaker_id = user_id
                            session.speaker_name = profile.get("name", "Friend")
                            session.is_new_speaker = False
                            logger.info(f"🎤 Speaker for session {session.session_id} identified as {session.speaker_name} (ID: {user_id})")
                        except Exception as profile_error:
                            logger.error(f"🎤 Failed to get speaker profile for {user_id}: {profile_error}")
                            await self._register_new_speaker(session, embedding)
                    else:
                        logger.info(f"🎤 Distance {distance:.4f} exceeds threshold {config.SPEAKER_SIMILARITY_THRESHOLD}, registering new speaker")
                        await self._register_new_speaker(session, embedding)
                else:
                    logger.info(f"🎤 No speakers in DB, registering first speaker for session {session.session_id}")
                    await self._register_new_speaker(session, embedding)
                    
            except Exception as chromadb_error:
                logger.error(f"🎤 ChromaDB query failed for session {session.session_id}: {chromadb_error}")
                # Fall back to registering as new speaker
                await self._register_new_speaker(session, embedding)
                
        except Exception as identification_error:
            logger.error(f"🎤 Speaker identification failed completely for session {session.session_id}: {identification_error}")
            session.speaker_state = SpeakerStateStatus.UNKNOWN
            session.speaker_name = "Guest"
        finally:
            self._cleanup_buffer(session.session_id)

    def _cleanup_buffer(self, session_id: str):
        """Clean up audio buffer for a session"""
        if session_id in self.session_audio_buffers:
            del self.session_audio_buffers[session_id]
            logger.info(f"🎤 Cleaned up audio buffer for session {session_id}")

    async def _register_new_speaker(self, session: Session, embedding: list[float]):
        """Register a new speaker with comprehensive error handling"""
        try:
            user_id = await self.memory_service.create_speaker_profile(embedding)
            session.speaker_state = SpeakerStateStatus.RECOGNIZED
            session.speaker_id = user_id
            session.speaker_name = f"Speaker {self.memory_service.collection.count()}"
            session.is_new_speaker = True
            logger.info(f"🎤 Registered new speaker {session.speaker_name} (ID: {user_id}) for session {session.session_id}")
        except Exception as register_error:
            logger.error(f"🎤 Failed to register new speaker for session {session.session_id}: {register_error}")
            session.speaker_state = SpeakerStateStatus.UNKNOWN
            session.speaker_name = "Guest"

    async def health_check(self) -> bool:
        return True

    def get_metrics(self) -> dict:
        return {"active_buffers": len(self.session_audio_buffers)}