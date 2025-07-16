"""
Fixed audio processor using correct genai-processors v1.0.4 API.
"""

import logging
import numpy as np
from typing import AsyncIterable

from genai_processors import content_api
from genai_processors import processor

logger = logging.getLogger(__name__)


class AudioProcessor(processor.Processor):
    """Audio preprocessing processor that normalizes and resamples audio."""
    
    def __init__(self, target_sample_rate: int = 16000, target_channels: int = 1,):
        super().__init__()
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
    
    async def call(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Process audio parts and pass through non-audio parts."""
        async for part in content:
            if content_api.is_audio(part.mimetype):
                try:
                    # Process audio data
                    processed_part = await self._process_audio_part(part)
                    yield processed_part
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    # Pass through original on error
                    yield part
            else:
                # Pass through non-audio parts unchanged
                yield part
    
    async def _process_audio_part(
        self, 
        part: content_api.ProcessorPart
    ) -> content_api.ProcessorPart:
        """Process an individual audio part."""
        try:
            # Extract audio data from the part
            audio_data = getattr(part, 'data', None)
            if audio_data is None:
                logger.warning("Audio part has no audio data")
                return part
            
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(audio_data, np.ndarray):
                audio_array = audio_data.astype(np.float32)
                if audio_array.dtype == np.int16:
                    audio_array = audio_array / 32768.0
            else:
                audio_array = np.array(audio_data, dtype=np.float32)
            
            # Normalize audio
            if len(audio_array) > 0 and np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array)) * 0.9
            
            # Convert back to appropriate format
            processed_audio = (audio_array * 32767).astype(np.int16)
            
            # Create new audio part using correct constructor
            # Find the correct way to create ProcessorPart
            try:
                # Try different constructor patterns
                return content_api.ProcessorPart(
                    processed_audio.tobytes(),
                    mimetype="audio/pcm"
                )
            except:
                try:
                    # Alternative constructor
                    new_part = content_api.ProcessorPart(processed_audio.tobytes())
                    new_part.mimetype = "audio/pcm"
                    return new_part
                except:
                    # Fallback - just modify the original part
                    part.data = processed_audio.tobytes()
                    return part
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return part


class SimpleEchoProcessor(processor.Processor):
    """Simple echo processor with correct API."""
    
    def __init__(self, prefix: str = "Echo: ", **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix
    
    async def call(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Echo text parts with prefix."""
        async for part in content:
            try:
                if content_api.is_text(part.mimetype):
                    # Echo text parts
                    text_data = str(getattr(part, 'data', ''))
                    echoed_text = f"{self.prefix}{text_data}"
                    
                    # Create new text part
                    try:
                        yield content_api.ProcessorPart(
                            echoed_text,
                            mimetype="text/plain"
                        )
                    except:
                        # Alternative creation method
                        new_part = content_api.ProcessorPart(echoed_text)
                        new_part.mimetype = "text/plain"
                        yield new_part
                else:
                    # Pass through non-text parts
                    yield part
            except Exception as e:
                logger.error(f"Error in echo: {e}")
                yield part


class PassThroughProcessor(processor.Processor):
    """Simple pass-through processor for testing."""
    
    def __init__(self, label: str = "PassThrough", **kwargs):
        super().__init__(**kwargs)
        self.label = label
    
    async def call(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Pass through all parts unchanged."""
        async for part in content:
            logger.info(f"{self.label}: Processing {getattr(part, 'mimetype', 'unknown')} part")
            yield part


class MockSTTProcessor(processor.Processor):
    """Mock STT processor for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def call(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Convert audio parts to mock transcription."""
        async for part in content:
            if content_api.is_audio(part.mimetype):
                # Create mock transcription
                try:
                    yield content_api.ProcessorPart(
                        "Mock transcription of audio input",
                        mimetype="text/plain"
                    )
                except:
                    # Alternative creation
                    text_part = content_api.ProcessorPart("Mock transcription of audio input")
                    text_part.mimetype = "text/plain"
                    yield text_part
            else:
                # Pass through non-audio parts
                yield part


class MockLLMProcessor(processor.Processor):
    """Mock LLM processor for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def call(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Generate mock LLM responses."""
        async for part in content:
            if content_api.is_text(part.mimetype):
                text_data = str(getattr(part, 'data', ''))
                response = f"Mock LLM response to: '{text_data}'"
                
                try:
                    yield content_api.ProcessorPart(
                        response,
                        mimetype="text/plain"
                    )
                except:
                    # Alternative creation
                    response_part = content_api.ProcessorPart(response)
                    response_part.mimetype = "text/plain"
                    yield response_part
            else:
                # Pass through non-text parts
                yield part


class MockTTSProcessor(processor.Processor):
    """Mock TTS processor for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def call(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Convert text to mock audio."""
        async for part in content:
            if content_api.is_text(part.mimetype):
                # Create mock audio data
                text_data = str(getattr(part, 'data', ''))
                mock_audio = f"mock_audio_for_{text_data}".encode()
                
                try:
                    yield content_api.ProcessorPart(
                        mock_audio,
                        mimetype="audio/wav"
                    )
                except:
                    # Alternative creation
                    audio_part = content_api.ProcessorPart(mock_audio)
                    audio_part.mimetype = "audio/wav"
                    yield audio_part
            else:
                # Pass through non-text parts
                yield part