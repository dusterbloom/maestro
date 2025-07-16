"""Text-to-speech processor using official genai-processors patterns."""

import logging
import os
from typing import AsyncIterable, Optional

from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import text_to_speech, audio_io

logger = logging.getLogger(__name__)


class TextToSpeechProcessor(processor.Processor):
    """Text-to-speech processor using genai-processors built-in TTS."""
    
    def __init__(
        self,
        project_id: str = None,
        language_code: str = "en-US",
        voice_name: str = None,
        speaking_rate: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.project_id = project_id or os.environ.get('GOOGLE_PROJECT_ID')
        self.language_code = language_code
        self.voice_name = voice_name
        self.speaking_rate = speaking_rate
        
        if not self.project_id:
            raise ValueError("project_id is required for TextToSpeechProcessor")
    
    async def __call__(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Convert text to speech using Google Text-to-Speech."""
        try:
            # Use genai-processors built-in text-to-speech
            tts_processor = text_to_speech.TextToSpeech(
                project_id=self.project_id,
                language_code=self.language_code,
                voice_name=self.voice_name,
                speaking_rate=self.speaking_rate
            )
            
            async for part in tts_processor(content):
                yield part
                
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            # Pass through original content on error
            async for part in content:
                yield part


class KokoroTTSProcessor(processor.Processor):
    """Kokoro TTS processor as fallback."""
    
    def __init__(
        self,
        service_url: str = "http://localhost:8880",
        voice: str = "af_bella",
        speed: float = 1.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.service_url = service_url
        self.voice = voice
        self.speed = speed
    
    async def __call__(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Convert text to speech using Kokoro TTS."""
        async for part in content:
            if content_api.is_text(part.mimetype):
                try:
                    text = part.data if hasattr(part, 'data') else str(part)
                    if text and text.strip():
                        audio_data = await self._synthesize_speech(text.strip())
                        if audio_data:
                            yield content_api.ProcessorPart(
                                data=audio_data,
                                mimetype="audio/wav",
                                metadata={
                                    **part.metadata,
                                    "source": "kokoro_tts",
                                    "voice": self.voice,
                                    "speed": self.speed
                                }
                            )
                except Exception as e:
                    logger.error(f"Kokoro TTS processing error: {e}")
            else:
                # Pass through non-text parts
                yield part
    
    async def _synthesize_speech(self, text: str) -> bytes:
        """Synthesize speech using Kokoro TTS API."""
        try:
            import aiohttp
            
            url = f"{self.service_url}/v1/audio/speech"
            
            payload = {
                "model": "kokoro",
                "input": text,
                "voice": self.voice,
                "response_format": "wav",
                "speed": self.speed
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        logger.error(f"Kokoro TTS API error: {response.status}")
                        return b""
                        
        except Exception as e:
            logger.error(f"Error calling Kokoro TTS: {e}")
            return b""


class TTSWithAudioOutput(processor.Processor):
    """TTS processor that includes audio output."""
    
    def __init__(
        self,
        tts_config: dict = None,
        use_audio_output: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tts_config = tts_config or {}
        self.use_audio_output = use_audio_output
        self._pya = None
        
        # Setup TTS processor
        try:
            self.tts_processor = TextToSpeechProcessor(**self.tts_config)
        except Exception as e:
            logger.warning(f"Google TTS not available: {e}")
            self.tts_processor = KokoroTTSProcessor(**self.tts_config)
        
        # Setup audio output if requested
        if self.use_audio_output:
            try:
                import pyaudio
                self._pya = pyaudio.PyAudio()
                self.audio_output = audio_io.PyAudioOut(self._pya)
            except Exception as e:
                logger.warning(f"Audio output not available: {e}")
                self.audio_output = None
        else:
            self.audio_output = None
    
    async def __call__(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Convert text to speech and optionally play audio."""
        # Create TTS pipeline
        if self.audio_output:
            # TTS + Audio Output
            pipeline = self.tts_processor + self.audio_output
        else:
            # TTS only
            pipeline = self.tts_processor
        
        try:
            async for part in pipeline(content):
                yield part
        finally:
            if self._pya:
                self._pya.terminate()


class TTSRouterProcessor(processor.Processor):
    """Router that tries Google TTS first, falls back to Kokoro."""
    
    def __init__(
        self,
        project_id: str = None,
        prefer_google: bool = True,
        kokoro_url: str = "http://localhost:8880",
        voice: str = "af_bella",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.project_id = project_id
        self.prefer_google = prefer_google
        
        # Initialize processors based on preference and availability
        if self.project_id and self.prefer_google:
            try:
                self.primary_tts = TextToSpeechProcessor(
                    project_id=self.project_id,
                    **kwargs
                )
                self.fallback_tts = KokoroTTSProcessor(
                    service_url=kokoro_url,
                    voice=voice,
                    **kwargs
                )
                logger.info("TTS router: Google TTS primary, Kokoro fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize Google TTS: {e}")
                self.primary_tts = KokoroTTSProcessor(
                    service_url=kokoro_url,
                    voice=voice,
                    **kwargs
                )
                self.fallback_tts = None
                logger.info("TTS router: Kokoro only")
        else:
            self.primary_tts = KokoroTTSProcessor(
                service_url=kokoro_url,
                voice=voice,
                **kwargs
            )
            self.fallback_tts = None
            logger.info("TTS router: Kokoro only")
    
    async def __call__(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Route TTS processing with fallback."""
        try:
            async for part in self.primary_tts(content):
                yield part
        except Exception as e:
            logger.error(f"Primary TTS failed: {e}")
            if self.fallback_tts:
                logger.info("Switching to fallback TTS")
                async for part in self.fallback_tts(content):
                    yield part
            else:
                # Pass through on complete failure
                async for part in content:
                    yield part


# Filter processor for text parts only
@processor.create_filter
def text_for_tts_filter(part: content_api.ProcessorPart) -> bool:
    """Filter to pass only text parts to TTS."""
    return content_api.is_text(part.mimetype) and hasattr(part, 'data') and part.data


class TextToSpeechPipeline(processor.Processor):
    """Complete text-to-speech pipeline with filtering and audio output."""
    
    def __init__(
        self,
        tts_config: dict = None,
        with_audio_output: bool = True,
        with_rate_limiting: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tts_config = tts_config or {}
        self.with_audio_output = with_audio_output
        self.with_rate_limiting = with_rate_limiting
        
        # Build pipeline components
        components = []
        
        # Add text filter
        components.append(text_for_tts_filter)
        
        # Add TTS processor
        components.append(TTSRouterProcessor(**self.tts_config))
        
        # Add rate limiting if requested
        if self.with_rate_limiting:
            from genai_processors.core import rate_limit_audio
            components.append(rate_limit_audio.RateLimitAudio(
                sample_rate=24000,
                delay_other_parts=True
            ))
        
        # Add audio output if requested
        if self.with_audio_output:
            try:
                import pyaudio
                pya = pyaudio.PyAudio()
                components.append(audio_io.PyAudioOut(pya))
            except Exception as e:
                logger.warning(f"Audio output not available: {e}")
        
        # Chain components together
        self.pipeline = components[0]
        for component in components[1:]:
            self.pipeline = self.pipeline + component
    
    async def __call__(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Process through complete TTS pipeline."""
        async for part in self.pipeline(content):
            yield part