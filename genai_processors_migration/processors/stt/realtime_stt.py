import logging
import asyncio
import websockets
import json
from typing import AsyncIterator
from genai_processors import Processor
from genai_processors.content_api import AudioPart, TextPart, ImagePart

from RealtimeSTT import AudioToTextRecorder
import numpy as np

logger = logging.getLogger(__name__)

class RealtimeSTTProcessor(Processor):
    def __init__(self, local=True, model='base', vad_enabled=True, **kwargs):
        super().__init__(**kwargs)
        self.local = local
        self.model_name = model
        self.vad_enabled = vad_enabled
        self.recorder = None
        self._setup_recorder()
    
    def _setup_recorder(self):
        """Initialize the RealtimeSTT recorder with proper configuration."""
        try:
            # Remove invalid parameters and use only valid ones
            recorder_config = {
                'spinner': False,
                'model': self.model_name,
                'language': 'en',
                'silero_sensitivity': 0.4,
                'webrtc_sensitivity': 3,
                'post_speech_silence_duration': 0.4,
                'min_length_of_recording': 0.5,
                'min_gap_between_recordings': 0,
                'enable_realtime_transcription': True,
                'realtime_processing_pause': 0.2,
                'realtime_model_type': 'tiny',
                'on_realtime_transcription_update': self._on_realtime_transcription,
                'on_realtime_transcription_stabilized': self._on_realtime_transcription_stabilized,
            }
            
            self.recorder = AudioToTextRecorder(**recorder_config)
            logger.info("RealtimeSTT recorder initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RealtimeSTT: {e}")
            raise
    
    def _on_realtime_transcription(self, text):
        """Handle real-time transcription updates."""
        logger.debug(f"Real-time transcription: {text}")
    
    def _on_realtime_transcription_stabilized(self, text):
        """Handle stabilized transcription."""
        logger.debug(f"Stabilized transcription: {text}")
    
    async def process(self, audio_data: bytes) -> AsyncIterator[str]:
        """Process audio data and yield transcriptions."""
        if not self.recorder:
            logger.error("Recorder not initialized")
            return
        
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Feed audio to recorder
            self.recorder.feed_audio(audio_array)
            
            # Get transcription
            text = self.recorder.text()
            if text and text.strip():
                yield text.strip()
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            yield f"Error: {str(e)}"

class RealtimeSTTPart(ProcessorPart):
    def __init__(self, data, metadata=None):
        super().__init__(data, metadata)
        self.processor = RealtimeSTTProcessor(**(data or {}))
    
    async def process(self, data: bytes) -> AsyncIterator[str]:
        """Process audio data and yield transcriptions."""
        async for result in self.processor.process(data):
            yield result