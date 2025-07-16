"""Speech-to-text processor using official genai-processors patterns."""

import logging
import os
from typing import AsyncIterable

from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import speech_to_text

logger = logging.getLogger(__name__)


class SpeechToTextProcessor(processor.Processor):
    """Speech-to-text processor using genai-processors built-in STT."""
    
    def __init__(
        self, 
        project_id: str = None,
        language_code: str = "en-US",
        with_interim_results: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.project_id = project_id or os.environ.get('GOOGLE_PROJECT_ID')
        self.language_code = language_code
        self.with_interim_results = with_interim_results
        
        if not self.project_id:
            raise ValueError("project_id is required for SpeechToTextProcessor")
    
    async def __call__(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Convert audio to text using Google Speech-to-Text."""
        try:
            # Use genai-processors built-in speech-to-text
            stt_processor = speech_to_text.SpeechToText(
                project_id=self.project_id,
                with_interim_results=self.with_interim_results,
                language_code=self.language_code
            )
            
            async for part in stt_processor(content):
                yield part
                
        except Exception as e:
            logger.error(f"Speech-to-text error: {e}")
            # Pass through original content on error
            async for part in content:
                yield part


class LocalSTTProcessor(processor.Processor):
    """Local speech-to-text processor fallback."""
    
    def __init__(
        self, 
        model_size: str = "base",
        language: str = "en",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_size = model_size
        self.language = language
        self._recorder = None
        self._setup_recorder()
    
    def _setup_recorder(self):
        """Setup local STT recorder."""
        try:
            from RealtimeSTT import AudioToTextRecorder
            
            config = {
                'spinner': False,
                'model': self.model_size,
                'language': self.language,
                'silero_sensitivity': 0.4,
                'webrtc_sensitivity': 3,
                'post_speech_silence_duration': 0.4,
                'min_length_of_recording': 0.5,
                'min_gap_between_recordings': 0,
                'enable_realtime_transcription': True,
                'realtime_processing_pause': 0.2,
                'realtime_model_type': 'tiny',
            }
            
            self._recorder = AudioToTextRecorder(**config)
            logger.info("Local STT recorder initialized")
            
        except ImportError:
            logger.warning("RealtimeSTT not available, local STT disabled")
            self._recorder = None
        except Exception as e:
            logger.error(f"Error initializing local STT: {e}")
            self._recorder = None
    
    async def call(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Convert audio to text using local STT."""
        if not self._recorder:
            logger.warning("Local STT not available, passing through")
            async for part in content:
                yield part
            return
        
        async for part in content:
            if content_api.is_audio(part.mimetype):
                try:
                    text = await self._process_audio_to_text(part)
                    if text and text.strip():
                        yield content_api.ProcessorPart(
                            data=text.strip(),
                            mimetype="text/plain",
                            metadata={
                                **part.metadata,
                                "source": "local_stt",
                                "model": self.model_size,
                                "language": self.language
                            }
                        )
                except Exception as e:
                    logger.error(f"Local STT processing error: {e}")
            else:
                yield part
    
    async def _process_audio_to_text(self, part: content_api.ProcessorPart) -> str:
        """Process audio part to text."""
        try:
            import numpy as np
            
            # Extract audio data
            if hasattr(part, 'data') and part.data is not None:
                audio_data = part.data
            else:
                return ""
            
            # Convert to numpy array
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            else:
                audio_array = np.array(audio_data, dtype=np.int16)
            
            # Feed to recorder and get transcription
            self._recorder.feed_audio(audio_array)
            text = self._recorder.text()
            
            return text if text else ""
            
        except Exception as e:
            logger.error(f"Error processing audio to text: {e}")
            return ""


class STTRouterProcessor(processor.Processor):
    """Router that tries Google STT first, falls back to local."""
    
    def __init__(
        self,
        project_id: str = None,
        prefer_cloud: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.project_id = project_id
        self.prefer_cloud = prefer_cloud
        
        # Initialize processors
        if self.project_id and self.prefer_cloud:
            try:
                self.primary_stt = SpeechToTextProcessor(
                    project_id=self.project_id,
                    **kwargs
                )
                self.fallback_stt = LocalSTTProcessor(**kwargs)
                logger.info("STT router: Google STT primary, local fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize Google STT: {e}")
                self.primary_stt = LocalSTTProcessor(**kwargs)
                self.fallback_stt = None
                logger.info("STT router: Local STT only")
        else:
            self.primary_stt = LocalSTTProcessor(**kwargs)
            self.fallback_stt = None
            logger.info("STT router: Local STT only")
    
    async def __call__(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Route STT processing with fallback."""
        try:
            async for part in self.primary_stt(content):
                yield part
        except Exception as e:
            logger.error(f"Primary STT failed: {e}")
            if self.fallback_stt:
                logger.info("Switching to fallback STT")
                async for part in self.fallback_stt(content):
                    yield part
            else:
                # Pass through on complete failure
                async for part in content:
                    yield part