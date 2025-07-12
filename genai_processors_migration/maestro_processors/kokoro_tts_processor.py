"""
KokoroTTSProcessor - GenAI Processors integration for Kokoro TTS service

This processor integrates with the existing Kokoro TTS API while providing
sentence-level audio generation for ultra-low latency. It maintains all
existing voice settings and audio quality.
"""

import asyncio
import logging
import httpx
import time
from typing import AsyncIterable, Dict, Optional, Set

from genai_processors import content_api
from genai_processors import processor

from .config import config, VoiceMetadata


logger = logging.getLogger(__name__)


class KokoroTTSProcessor(processor.Processor):
    """
    Converts sentence-level text to audio using Kokoro TTS service.
    
    Features:
    - Sentence-level processing for immediate audio generation
    - Maintains existing Kokoro HTTP API integration
    - Preserves voice settings and audio quality
    - Supports interruption and cancellation
    - Connection pooling for performance optimization
    - Rich metadata for audio ProcessorParts
    """
    
    def __init__(
        self,
        tts_url: str = None,
        voice: str = None,
        speed: float = None,
        volume: float = None,
        session_id: str = "default",
        timeout: float = None,
        use_connection_pool: bool = True,
        max_concurrent_requests: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # TTS configuration
        self.tts_url = tts_url or config.TTS_URL
        self.voice = voice or config.TTS_VOICE
        self.speed = speed or config.TTS_SPEED
        self.volume = volume or config.TTS_VOLUME
        self.session_id = session_id
        self.timeout = timeout or config.TTS_TIMEOUT
        
        # Performance configuration
        self.use_connection_pool = use_connection_pool
        self.max_concurrent_requests = max_concurrent_requests
        
        # Processing state
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.active_requests: Set[str] = set()
        self.interrupted = False
        self.sequence_number = 0
        
        # Performance tracking
        self.tts_requests = 0
        self.total_processing_time = 0.0
        self.average_latency = 0.0
        
        # HTTP client with connection pooling
        self.http_client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        
        logger.info(f"KokoroTTSProcessor initialized for session {session_id}")
        logger.info(f"TTS URL: {self.tts_url}, Voice: {self.voice}")
    
    async def call(self, content: AsyncIterable[content_api.ProcessorPart]) -> AsyncIterable[content_api.ProcessorPart]:
        """
        Process LLM text input and generate audio output.
        
        Args:
            content: Stream of text ProcessorParts from LLM
            
        Yields:
            ProcessorParts containing audio data with metadata
        """
        try:
            # Initialize HTTP client
            await self._ensure_http_client()
            
            # Create concurrent processing tasks
            input_task = asyncio.create_task(self._process_text_input(content))
            output_task = asyncio.create_task(self._process_audio_output())
            
            try:
                # Yield audio parts as they become available
                async for audio_part in output_task:
                    yield audio_part
            finally:
                # Cleanup tasks
                if not input_task.done():
                    input_task.cancel()
                if not output_task.done():
                    output_task.cancel()
                    
        except Exception as e:
            logger.error(f"Error in KokoroTTSProcessor: {e}")
        finally:
            await self._cleanup_client()
    
    async def _ensure_http_client(self):
        """Ensure HTTP client is initialized with proper configuration."""
        async with self._client_lock:
            if self.http_client is None:
                # Configure connection limits for performance
                limits = httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=20,
                    keepalive_expiry=30
                )
                
                timeout_config = httpx.Timeout(
                    connect=5.0,
                    read=self.timeout,
                    write=5.0,
                    pool=5.0
                )
                
                self.http_client = httpx.AsyncClient(
                    limits=limits,
                    timeout=timeout_config,
                    follow_redirects=True
                )
                
                logger.info(f"HTTP client initialized for session {self.session_id}")
    
    async def _process_text_input(self, content: AsyncIterable[content_api.ProcessorPart]):
        """Process incoming text ProcessorParts and queue TTS requests."""
        try:
            async for part in content:
                # Only process LLM text content
                if (part.metadata and 
                    part.metadata.get("content_type") == VoiceMetadata.LLM_TEXT and
                    part.metadata.get("is_complete", False)):
                    
                    text_content = part.content if isinstance(part.content, str) else str(part.content)
                    
                    # Check for interruption
                    if self.interrupted:
                        logger.info(f"Session {self.session_id}: TTS processing interrupted")
                        self.interrupted = False
                        break
                    
                    # Queue text for TTS processing
                    await self.processing_queue.put({
                        "text": text_content,
                        "metadata": part.metadata,
                        "sequence": self.sequence_number
                    })
                    
                    self.sequence_number += 1
                    logger.debug(f"Session {self.session_id}: Queued TTS for sequence {self.sequence_number}: {text_content}")
                    
        except Exception as e:
            logger.error(f"Error processing text input: {e}")
    
    async def _process_audio_output(self) -> AsyncIterable[content_api.ProcessorPart]:
        """Process TTS requests and yield audio ProcessorParts."""
        try:
            # Create semaphore to limit concurrent TTS requests
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            processing_tasks: Set[asyncio.Task] = set()
            
            while True:
                try:
                    # Get next TTS request (with timeout to allow cleanup)
                    try:
                        tts_request = await asyncio.wait_for(
                            self.processing_queue.get(),
                            timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        # Check if we should continue processing
                        if self.interrupted or (not processing_tasks and self.processing_queue.empty()):
                            break
                        continue
                    
                    # Create TTS processing task
                    task = asyncio.create_task(
                        self._generate_tts_audio(tts_request, semaphore)
                    )
                    processing_tasks.add(task)
                    
                    # Check for completed tasks
                    if processing_tasks:
                        done_tasks, processing_tasks = await asyncio.wait(
                            processing_tasks,
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=0.001  # Non-blocking check
                        )
                        
                        # Yield results from completed tasks
                        for task in done_tasks:
                            try:
                                audio_part = await task
                                if audio_part:
                                    yield audio_part
                            except Exception as e:
                                logger.error(f"Error in TTS task: {e}")
                    
                    # Check for interruption
                    if self.interrupted:
                        # Cancel all pending tasks
                        for task in processing_tasks:
                            task.cancel()
                        break
                        
                except Exception as e:
                    logger.error(f"Error in audio output processing: {e}")
                    break
            
            # Wait for remaining tasks to complete or cancel them
            if processing_tasks:
                for task in processing_tasks:
                    if not task.done():
                        task.cancel()
                
                # Yield any remaining results
                for task in processing_tasks:
                    try:
                        audio_part = await task
                        if audio_part:
                            yield audio_part
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error completing TTS task: {e}")
                        
        except Exception as e:
            logger.error(f"Error in audio output processing: {e}")
    
    async def _generate_tts_audio(
        self, 
        tts_request: Dict, 
        semaphore: asyncio.Semaphore
    ) -> Optional[content_api.ProcessorPart]:
        """Generate audio for a single TTS request."""
        async with semaphore:
            tts_start_time = time.time()
            request_id = f"{self.session_id}_{tts_request['sequence']}"
            
            try:
                self.active_requests.add(request_id)
                
                text = tts_request["text"]
                source_metadata = tts_request["metadata"]
                sequence = tts_request["sequence"]
                
                logger.debug(f"Session {self.session_id}: Generating TTS for sequence {sequence}")
                
                # Prepare TTS request
                tts_payload = {
                    "model": "kokoro",
                    "voice": self.voice,
                    "input": text,
                    "response_format": "wav",  # Ensure WAV format for compatibility
                    "speed": self.speed
                }
                
                # Make TTS request
                response = await self.http_client.post(
                    f"{self.tts_url}/v1/audio/speech",
                    json=tts_payload,
                    headers={"Content-Type": "application/json"}
                )
                
                response.raise_for_status()
                
                # Get audio data
                audio_data = response.content
                tts_latency = time.time() - tts_start_time
                
                # Update performance metrics
                self.tts_requests += 1
                self.total_processing_time += tts_latency
                self.average_latency = self.total_processing_time / self.tts_requests
                
                logger.info(f"Session {self.session_id}: TTS sequence {sequence} completed in {tts_latency*1000:.2f}ms")
                
                # Create audio ProcessorPart with rich metadata
                audio_part = content_api.ProcessorPart(
                    audio_data,
                    mimetype="audio/wav",
                    metadata={
                        "session_id": self.session_id,
                        "content_type": VoiceMetadata.TTS_AUDIO,
                        "stage": VoiceMetadata.STAGE_TTS,
                        "status": VoiceMetadata.STATUS_COMPLETE,
                        "timestamp": time.time(),
                        "sequence_number": sequence,
                        "is_complete": True,
                        
                        # Audio properties
                        "sample_rate": 16000,  # Kokoro default
                        "channels": 1,
                        "format": "wav",
                        "duration_estimate": len(text) * 0.1,  # Rough estimate
                        
                        # TTS configuration
                        "voice": self.voice,
                        "speed": self.speed,
                        "volume": self.volume,
                        
                        # Performance metrics
                        "tts_latency_ms": tts_latency * 1000,
                        "text_length": len(text),
                        "audio_size_bytes": len(audio_data),
                        
                        # Source information
                        "source_text": text,
                        "source_llm_sequence": source_metadata.get("sequence_number"),
                        "source_llm_latency": source_metadata.get("llm_sentence_latency_ms"),
                        "source_timestamp": source_metadata.get("timestamp"),
                        
                        # Pipeline metrics
                        "pipeline_stage": "tts_complete",
                        "ready_for_playback": True
                    }
                )
                
                return audio_part
                
            except httpx.TimeoutException:
                logger.error(f"TTS timeout for sequence {sequence} (session {self.session_id})")
                return None
                
            except httpx.HTTPStatusError as e:
                logger.error(f"TTS HTTP error for sequence {sequence}: {e.response.status_code}")
                return None
                
            except Exception as e:
                logger.error(f"TTS generation error for sequence {sequence}: {e}")
                return None
                
            finally:
                self.active_requests.discard(request_id)
    
    async def interrupt(self):
        """Interrupt current TTS processing for barge-in functionality."""
        logger.info(f"KokoroTTSProcessor interrupted for session {self.session_id}")
        self.interrupted = True
        
        # Clear processing queue
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Reset sequence number
        self.sequence_number = 0
        
        # Note: Active HTTP requests will complete naturally due to timeout
        logger.info(f"Session {self.session_id}: TTS queue cleared, {len(self.active_requests)} requests still active")
    
    async def _cleanup_client(self):
        """Clean up HTTP client and resources."""
        async with self._client_lock:
            if self.http_client:
                try:
                    await self.http_client.aclose()
                    logger.info(f"HTTP client closed for session {self.session_id}")
                except Exception as e:
                    logger.warning(f"Error closing HTTP client: {e}")
                finally:
                    self.http_client = None
    
    def get_metrics(self) -> Dict:
        """Get performance metrics for monitoring."""
        return {
            "session_id": self.session_id,
            "voice": self.voice,
            "speed": self.speed,
            "volume": self.volume,
            
            # Processing state
            "sequence_number": self.sequence_number,
            "queue_size": self.processing_queue.qsize(),
            "active_requests": len(self.active_requests),
            "interrupted": self.interrupted,
            
            # Performance metrics
            "total_requests": self.tts_requests,
            "total_processing_time": self.total_processing_time,
            "average_latency_ms": self.average_latency * 1000,
            
            # Configuration
            "tts_url": self.tts_url,
            "timeout": self.timeout,
            "max_concurrent_requests": self.max_concurrent_requests,
            "use_connection_pool": self.use_connection_pool,
            
            # Client state
            "http_client_active": self.http_client is not None
        }
    
    async def health_check(self) -> bool:
        """Check if TTS service is healthy."""
        try:
            await self._ensure_http_client()
            
            # Simple health check request
            response = await self.http_client.get(
                f"{self.tts_url}/health",
                timeout=5.0
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"TTS health check failed: {e}")
            return False
    
    def get_voice_options(self) -> List[str]:
        """Get available voice options (static list for Kokoro)."""
        # Common Kokoro voices - this could be made dynamic with an API call
        return [
            "af_bella", "af_sarah", "af_nicole", "af_sky",
            "am_adam", "am_michael", "am_jacob", "am_mason"
        ]
    
    async def set_voice_settings(self, voice: str = None, speed: float = None, volume: float = None):
        """Update voice settings for the session."""
        if voice is not None:
            self.voice = voice
            logger.info(f"Session {self.session_id}: Voice changed to {voice}")
        
        if speed is not None:
            self.speed = max(0.25, min(4.0, speed))  # Reasonable bounds
            logger.info(f"Session {self.session_id}: Speed changed to {self.speed}")
        
        if volume is not None:
            self.volume = max(0.0, min(2.0, volume))  # Reasonable bounds
            logger.info(f"Session {self.session_id}: Volume changed to {self.volume}")
