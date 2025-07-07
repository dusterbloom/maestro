
import asyncio
import logging
from orchestrator.src.core.state_machine import Session, SpeakerStateStatus
from orchestrator.src.services.base_service import BaseService, ServiceResult
from orchestrator.src.services.voice_service import VoiceService
from orchestrator.src.utils.deduplicator import RequestDeduplicator
from orchestrator.src.services.voice_service import AudioBufferManager

logger = logging.getLogger(__name__)

class SpeakerService(BaseService):
    def __init__(self, voice_service: VoiceService, deduplicator: RequestDeduplicator):
        self.voice_service = voice_service
        self.deduplicator = deduplicator
        # Each session will have its own audio buffer manager
        self.session_audio_buffers: dict[str, AudioBufferManager] = {}

    async def process(self, audio_chunk: bytes, context: dict) -> ServiceResult:
        """Accumulates audio and triggers identification when enough data is present."""
        session: Session = context.get("session")
        if not session:
            return ServiceResult(success=False, error="No session context provided.")

        # If speaker is already known, do nothing.
        if session.speaker_state == SpeakerStateStatus.RECOGNIZED:
            return ServiceResult(success=True, data={"status": "already_recognized"})

        # Get or create buffer for the session
        if session.session_id not in self.session_audio_buffers:
            self.session_audio_buffers[session.session_id] = AudioBufferManager()
        
        buffer_manager = self.session_audio_buffers[session.session_id]
        
        import numpy as np
        float_array = np.frombuffer(audio_chunk, dtype=np.float32)
        buffer_ready = buffer_manager.add_audio_chunk(float_array)

        if buffer_ready:
            logger.info(f"SpeakerService: Buffer ready for session {session.session_id}. Triggering identification.")
            # Use the deduplicator to ensure only one identification runs at a time
            request_id = f"speaker_id_{session.session_id}"
            identification_task = self._identify_speaker(session, buffer_manager)
            
            # Fire and forget, the result will be handled by an event
            asyncio.create_task(
                self.deduplicator.process_or_join(request_id, identification_task)
            )
            return ServiceResult(success=True, data={"status": "identification_triggered"})
        
        return ServiceResult(success=True, data={"status": "accumulating"})

    async def _identify_speaker(self, session: Session, buffer_manager: AudioBufferManager):
        """The actual identification logic, wrapped by the deduplicator."""
        session.speaker_state = SpeakerStateStatus.IDENTIFYING
        
        wav_bytes = buffer_manager.get_buffer_as_wav(apply_vad=False)
        if not wav_bytes:
            logger.error(f"Could not get WAV bytes from buffer for session {session.session_id}")
            session.speaker_state = SpeakerStateStatus.UNKNOWN
            return

        result = await self.voice_service.identify_or_register(wav_bytes, session.session_id)
        
        if result.get("status") in ["identified", "registered"]:
            session.speaker_state = SpeakerStateStatus.RECOGNIZED
            session.speaker_id = result.get("user_id")
            session.speaker_name = result.get("name")
            logger.info(f"Speaker for session {session.session_id} identified as {session.speaker_name}")
            # In a real implementation, we would dispatch a "speaker.identified" event here
        else:
            session.speaker_state = SpeakerStateStatus.UNKNOWN
            logger.warning(f"Speaker identification failed for session {session.session_id}")

    async def health_check(self) -> bool:
        return True

    def get_metrics(self) -> dict:
        return {"active_buffers": len(self.session_audio_buffers)}
