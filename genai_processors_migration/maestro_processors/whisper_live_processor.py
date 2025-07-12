"""
WhisperLiveProcessor - GenAI Processors integration for WhisperLive STT service

This processor maintains the existing WhisperLive WebSocket integration while
adapting it to work within the GenAI Processors framework. It preserves all
performance characteristics and VAD functionality.
"""

import asyncio
import json
import logging
import time
import websockets
import websockets.exceptions
from typing import AsyncIterable, Dict, Optional, Set
from urllib.parse import urlparse

from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import streams

from .config import config, AudioConfig, VoiceMetadata


logger = logging.getLogger(__name__)


class WhisperLiveProcessor(processor.Processor):
    """
    Integrates existing WhisperLive WebSocket service with GenAI Processors framework.
    
    Features:
    - Maintains existing WebSocket connection to WhisperLive
    - Preserves VAD (Voice Activity Detection) settings  
    - Converts audio ProcessorParts to WhisperLive binary format
    - Maps WhisperLive responses to ProcessorParts with rich metadata
    - Handles connection management and error recovery
    """
    
    def __init__(
        self,
        whisper_host: str = None,
        whisper_port: int = None,
        session_id: str = "default",
        language: str = "en",
        model: str = None,
        use_vad: bool = None,
        no_speech_threshold: float = None,
        max_retries: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Connection configuration
        self.whisper_host = whisper_host or config.WHISPER_URL.replace("http://", "").replace("https://", "").split(":")[0]
        self.whisper_port = whisper_port or (int(config.WHISPER_URL.split(":")[-1]) if ":" in config.WHISPER_URL else 9090)
        self.session_id = session_id
        
        # WhisperLive configuration
        self.language = language
        self.model = model or config.STT_MODEL
        self.use_vad = use_vad if use_vad is not None else config.VAD_ENABLED
        self.no_speech_threshold = no_speech_threshold or config.NO_SPEECH_THRESHOLD
        self.max_retries = max_retries
        
        # Connection state
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False
        self.connection_retries = 0
        
        # Processing state
        self.message_handler_task: Optional[asyncio.Task] = None
        self.transcript_queue: asyncio.Queue = asyncio.Queue()
        self.current_transcript = ""
        
        # Performance metrics
        self.stt_start_time: Optional[float] = None
        self.processing_count = 0
        
        logger.info(f"WhisperLiveProcessor initialized for session {session_id}")
        logger.info(f"Target: {self.whisper_host}:{self.whisper_port}")
    
    async def call(self, content: AsyncIterable[content_api.ProcessorPart]) -> AsyncIterable[content_api.ProcessorPart]:
        """
        Process audio input through WhisperLive and yield transcript ProcessorParts.
        
        Args:
            content: Stream of audio ProcessorParts
            
        Yields:
            ProcessorParts containing transcription results with metadata
        """
        # Ensure WhisperLive connection is established
        if not await self._ensure_connection():
            logger.error(f"Failed to establish WhisperLive connection for session {self.session_id}")
            return
        
        # Start the message handler if not already running
        if not self.message_handler_task or self.message_handler_task.done():
            self.message_handler_task = asyncio.create_task(self._handle_whisper_messages())
        
        # Process audio input concurrently with transcript output
        audio_task = asyncio.create_task(self._process_audio_input(content))
        transcript_task = asyncio.create_task(self._process_transcript_output())
        
        try:
            # Yield transcripts as they become available
            async for transcript_part in transcript_task:
                yield transcript_part
        finally:
            # Cleanup tasks
            if not audio_task.done():
                audio_task.cancel()
            if not transcript_task.done():
                transcript_task.cancel()
                
            await self._cleanup_connection()
    
    async def _ensure_connection(self) -> bool:
        """Ensure WhisperLive WebSocket connection is active."""
        if self.connected and self._is_websocket_connected():
            return True
        
        return await self._connect_to_whisper()
    
    async def _connect_to_whisper(self) -> bool:
        """Connect to WhisperLive with correct protocol matching the actual implementation."""
        for attempt in range(self.max_retries):
            try:
                self.connection_retries = attempt + 1
                
                # Create WebSocket URL
                whisper_ws_url = f"ws://{self.whisper_host}:{self.whisper_port}"
                logger.info(f"Attempt {attempt + 1}: Connecting to WhisperLive at: {whisper_ws_url}")
                
                # Simple connection - no extra headers needed for WhisperLive
                self.websocket = await websockets.connect(
                    whisper_ws_url,
                    # Disable ping/pong - WhisperLive doesn't use it
                    ping_interval=None,
                    ping_timeout=None,
                    open_timeout=10,
                    close_timeout=5
                )
                
                logger.info(f"WebSocket connected to WhisperLive for session {self.session_id}")
                
                # Send WhisperLive configuration as first message (required per protocol)
                config_message = {
                    "uid": self.session_id,
                    "language": self.language,
                    "task": "transcribe",
                    "model": self.model,
                    "use_vad": self.use_vad,
                    "max_clients": 4,
                    "max_connection_time": 3600,
                    "send_last_n_segments": 10,
                    "no_speech_thresh": self.no_speech_threshold,
                    "clip_audio": False,
                    "same_output_threshold": 10
                }
                
                logger.info(f"Sending config to WhisperLive: {config_message}")
                await self.websocket.send(json.dumps(config_message))
                
                # WhisperLive starts processing immediately after config
                self.connected = True
                logger.info(f"✅ Session {self.session_id}: WhisperLive connected and configured")
                
                return True
                
            except websockets.exceptions.InvalidUpgrade as e:
                logger.error(f"❌ WebSocket handshake failed for session {self.session_id} (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"❌ All attempts failed. Check if WhisperLive is running at ws://{self.whisper_host}:{self.whisper_port}")
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            except ConnectionRefusedError as e:
                logger.error(f"❌ Connection refused for session {self.session_id} (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"❌ WhisperLive not accessible at ws://{self.whisper_host}:{self.whisper_port}")
                else:
                    await asyncio.sleep(2 ** attempt)
                    
            except Exception as e:
                logger.error(f"❌ Connection error for session {self.session_id} (attempt {attempt + 1}): {type(e).__name__}: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"❌ Failed to connect after {self.max_retries} attempts")
                else:
                    await asyncio.sleep(2 ** attempt)
                    
        return False
    
    def _is_websocket_connected(self) -> bool:
        """Safely check if WebSocket connection is still active."""
        if not self.websocket:
            return False
        try:
            if hasattr(self.websocket, 'closed'):
                return not self.websocket.closed
            elif hasattr(self.websocket, 'state'):
                return self.websocket.state.name == 'OPEN'
            elif hasattr(self.websocket, 'close_code'):
                return self.websocket.close_code is None
            else:
                return True
        except Exception:
            return False
    
    async def _process_audio_input(self, content: AsyncIterable[content_api.ProcessorPart]):
        """Process incoming audio ProcessorParts and send to WhisperLive."""
        try:
            async for part in content:
                if not self.connected or not self._is_websocket_connected():
                    logger.warning(f"WhisperLive connection lost for session {self.session_id}")
                    if not await self._ensure_connection():
                        break
                
                # Extract audio data from ProcessorPart
                if part.mime_type and part.mime_type.startswith("audio/"):
                    audio_data = self._extract_audio_data(part)
                    if audio_data:
                        await self._send_audio_to_whisper(audio_data)
                        
                        # Mark STT processing start time for latency tracking
                        if self.stt_start_time is None:
                            self.stt_start_time = time.time()
                            
        except Exception as e:
            logger.error(f"Error processing audio input: {e}")
    
    def _extract_audio_data(self, part: content_api.ProcessorPart) -> Optional[bytes]:
        """Extract audio data from ProcessorPart in format expected by WhisperLive."""
        try:
            # Handle different audio formats
            if hasattr(part, 'data') and isinstance(part.data, bytes):
                return part.data
            elif hasattr(part, 'content') and isinstance(part.content, bytes):
                return part.content
            elif hasattr(part, 'audio_bytes'):
                return part.audio_bytes
            else:
                logger.warning(f"Unsupported audio format in ProcessorPart: {type(part)}")
                return None
        except Exception as e:
            logger.error(f"Error extracting audio data: {e}")
            return None
    
    async def _send_audio_to_whisper(self, audio_data: bytes) -> bool:
        """Send audio data to WhisperLive WebSocket."""
        try:
            await self.websocket.send(audio_data)
            return True
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"WhisperLive connection closed while sending audio for session {self.session_id}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to send audio to WhisperLive: {e}")
            return False
    
    async def _handle_whisper_messages(self):
        """Handle incoming messages from WhisperLive WebSocket."""
        try:
            while self.websocket and self._is_websocket_connected():
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=30.0
                    )
                    
                    try:
                        data = json.loads(message)
                        
                        # WhisperLive sends transcription data with "uid" and "segments"
                        if "uid" in data and data["uid"] == self.session_id:
                            if "segments" in data:
                                await self._process_transcript_segments(data["segments"])
                            elif "message" in data:
                                # Handle status messages
                                if data["message"] == "DISCONNECT":
                                    logger.info(f"WhisperLive disconnected session {self.session_id}")
                                    break
                                elif "status" in data:
                                    logger.info(f"WhisperLive status for {self.session_id}: {data}")
                        else:
                            logger.debug(f"Unknown message from WhisperLive: {data}")
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON message from WhisperLive: {message}")
                        
                except websockets.exceptions.ConnectionClosed as e:
                    logger.info(f"WhisperLive connection closed normally for session {self.session_id}: {e}")
                    break
                except asyncio.TimeoutError:
                    logger.debug(f"No message from WhisperLive for 30 seconds (session {self.session_id})")
                    if not self._is_websocket_connected():
                        break
                    continue
                    
                except Exception as e:
                    logger.error(f"Error processing WhisperLive message: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"WhisperLive handler error for session {self.session_id}: {e}")
        finally:
            self.connected = False
    
    async def _process_transcript_segments(self, segments):
        """Process transcript segments and queue them for output."""
        try:
            for segment in segments:
                if segment.get("completed") and segment.get("text"):
                    text = segment["text"].strip()
                    if text:
                        # Calculate STT latency
                        stt_latency = time.time() - self.stt_start_time if self.stt_start_time else 0
                        
                        # Create ProcessorPart with rich metadata
                        transcript_part = content_api.ProcessorPart(
                            content=text,
                            mime_type="text/plain",
                            metadata={
                                "session_id": self.session_id,
                                "content_type": VoiceMetadata.TRANSCRIPT,
                                "stage": VoiceMetadata.STAGE_STT,
                                "status": VoiceMetadata.STATUS_COMPLETE,
                                "timestamp": time.time(),
                                "sequence_number": self.processing_count,
                                "language": self.language,
                                "confidence": segment.get("confidence", 1.0),
                                "is_complete": True,
                                "stt_latency_ms": stt_latency * 1000,
                                "whisper_segment": segment  # Original segment data
                            }
                        )
                        
                        await self.transcript_queue.put(transcript_part)
                        self.processing_count += 1
                        
                        # Reset STT timer for next processing cycle
                        self.stt_start_time = None
                        
                        logger.debug(f"Session {self.session_id}: Transcribed: {text}")
                        
                elif segment.get("text"):
                    # Handle partial/live transcripts
                    text = segment["text"].strip()
                    if text != self.current_transcript:
                        self.current_transcript = text
                        
                        # Create live transcript ProcessorPart
                        live_part = content_api.ProcessorPart(
                            content=text,
                            mime_type="text/plain",
                            metadata={
                                "session_id": self.session_id,
                                "content_type": "live_transcript",
                                "stage": VoiceMetadata.STAGE_STT,
                                "status": VoiceMetadata.STATUS_PROCESSING,
                                "timestamp": time.time(),
                                "language": self.language,
                                "is_complete": False,
                                "whisper_segment": segment
                            }
                        )
                        
                        await self.transcript_queue.put(live_part)
                        
        except Exception as e:
            logger.error(f"Error processing transcript segments: {e}")
    
    async def _process_transcript_output(self) -> AsyncIterable[content_api.ProcessorPart]:
        """Yield transcript ProcessorParts as they become available."""
        try:
            while True:
                try:
                    # Wait for transcript with timeout
                    transcript_part = await asyncio.wait_for(
                        self.transcript_queue.get(),
                        timeout=config.STREAM_TIMEOUT
                    )
                    yield transcript_part
                    
                except asyncio.TimeoutError:
                    # Check if connection is still active
                    if not self.connected:
                        break
                    continue
                    
                except Exception as e:
                    logger.error(f"Error yielding transcript: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Error in transcript output processing: {e}")
    
    async def _cleanup_connection(self):
        """Clean up WhisperLive connection and resources."""
        try:
            # Cancel message handler
            if self.message_handler_task and not self.message_handler_task.done():
                self.message_handler_task.cancel()
                try:
                    await self.message_handler_task
                except asyncio.CancelledError:
                    pass
            
            # Close WebSocket connection
            if self.websocket:
                try:
                    await self.websocket.close(code=1000, reason="Session cleanup")
                except Exception as e:
                    logger.warning(f"Error closing WhisperLive connection: {e}")
            
            self.connected = False
            self.websocket = None
            
            logger.info(f"WhisperLiveProcessor cleanup completed for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error during WhisperLive cleanup: {e}")
    
    async def interrupt(self):
        """Interrupt current processing (for barge-in functionality)."""
        logger.info(f"WhisperLiveProcessor interrupted for session {self.session_id}")
        
        # Clear transcript queue to stop processing current audio
        while not self.transcript_queue.empty():
            try:
                self.transcript_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Reset processing state
        self.current_transcript = ""
        self.stt_start_time = None
    
    def get_metrics(self) -> Dict:
        """Get performance metrics for monitoring."""
        return {
            "session_id": self.session_id,
            "connected": self.connected,
            "connection_retries": self.connection_retries,
            "processing_count": self.processing_count,
            "queue_size": self.transcript_queue.qsize(),
            "current_transcript": self.current_transcript,
            "whisper_host": self.whisper_host,
            "whisper_port": self.whisper_port
        }
