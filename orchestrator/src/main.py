import asyncio
import json
import logging
import os
import time
import base64
from typing import Dict, Set, Optional, Any
from urllib.parse import urlparse
import websockets
import websockets.exceptions
import httpx
import ollama
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Stream Orchestrator", version="2.0.0")

class StreamSession:
    """
    Represents a single voice conversation session with all its components
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        
        # WebSocket connections
        self.frontend_ws: Optional[WebSocket] = None
        self.whisper_ws: Optional[websockets.WebSocketServerProtocol] = None
        
        # Processing state
        self.is_recording = False
        self.is_processing = False
        self.current_transcript = ""
        self.processing_text: Optional[str] = None
        self.conversation_history = []
        self.last_interrupted_text: Optional[str] = None
        self.last_processed_text: Optional[str] = None
        
        # TTS state for interruption and sequencing
        self.tts_active = False
        self.tts_abort_event = asyncio.Event()
        self.tts_queue = []  # Queue for pending TTS sentences
        self.tts_sequence_number = 0  # Current sequence number for TTS
        self.tts_processing_lock = asyncio.Lock()  # Ensures sequential TTS processing
        self.tts_task: Optional[asyncio.Task] = None  # Task for processing TTS queue
        self.whisper_message_handler_task: Optional[asyncio.Task] = None # Task for handling whisper messages
        
        # Connection state
        self.whisper_connected = False
        self.connection_retries = 0
        self.max_retries = 3
        
        # Metrics
        self.total_requests = 0
        self.last_activity = time.time()
        self.stt_end_time: Optional[float] = None
        self.metrics = {
            "llm_first_token_latency": 0,
            "llm_total_latency": 0,
            "tts_latency": 0,
            "total_pipeline_latency": 0
        }
        
    def update_activity(self):
        self.last_activity = time.time()
        
    def can_be_cleaned_up(self, max_idle_time: int = 3600) -> bool:
        return (time.time() - self.last_activity) > max_idle_time

class VoiceStreamOrchestrator:
    """
    Central coordinator for all voice processing streams
    """
    def __init__(self):
        self.sessions: Dict[str, StreamSession] = {}
        # Robust URL parsing for WhisperLive
        logger.info(f"Parsing WHISPER_URL: '{config.WHISPER_URL}'")
        logger.info(f"Raw WHISPER_URL env var: '{os.getenv('WHISPER_URL', 'NOT_SET')}'")
        
        try:
            # Handle various URL formats
            whisper_url = config.WHISPER_URL
            if not whisper_url.startswith(('http://', 'https://', 'ws://', 'wss://')):
                whisper_url = f'http://{whisper_url}'
            
            parsed = urlparse(whisper_url)
            self.whisper_host = parsed.hostname or 'whisper-live'
            self.whisper_port = parsed.port or 9090
            
            # Fallback if parsing didn't work as expected
            if self.whisper_host == 'ws' or not self.whisper_host:
                logger.warning(f"Unexpected hostname '{self.whisper_host}', using fallback parsing")
                # Extract host:port from URL manually
                clean_url = config.WHISPER_URL.replace('http://', '').replace('https://', '').replace('ws://', '').replace('wss://', '')
                if ':' in clean_url:
                    self.whisper_host = clean_url.split(':')[0]
                    self.whisper_port = int(clean_url.split(':')[1])
                else:
                    self.whisper_host = clean_url or 'whisper-live'
                    self.whisper_port = 9090
                    
            logger.info(f"✅ Final WhisperLive target: {self.whisper_host}:{self.whisper_port}")
            
        except Exception as e:
            logger.error(f"URL parsing failed: {e}")
            # Ultimate fallback
            self.whisper_host = 'whisper-live'
            self.whisper_port = 9090
            logger.info(f"✅ Using fallback WhisperLive target: {self.whisper_host}:{self.whisper_port}")
            
        self.ollama_url = config.OLLAMA_URL
        self.tts_url = config.TTS_URL
        
        # Start background tasks
        asyncio.create_task(self._session_cleanup_task())
        asyncio.create_task(self._connection_health_monitor())
        
    def _is_websocket_connected(self, ws) -> bool:
        """Safely check if a WebSocket connection is still active"""
        if not ws:
            return False
        try:
            # Try to check connection state - different websocket libraries have different attributes
            if hasattr(ws, 'closed'):
                return not ws.closed
            elif hasattr(ws, 'state'):
                # For some websocket libraries, check state
                return ws.state.name == 'OPEN'
            elif hasattr(ws, 'close_code'):
                # If close_code is None, connection is still open
                return ws.close_code is None
            else:
                # Fallback: assume connected if object exists
                return True
        except Exception:
            return False
        
    async def create_session(self, session_id: str) -> StreamSession:
        """Create a new voice session with improved WhisperLive connection"""
        if session_id in self.sessions:
            await self.cleanup_session(session_id)
            
        session = StreamSession(session_id)
        
        # Establish WhisperLive connection with proper headers
        success = await self._connect_to_whisper(session)
        if not success:
            raise ConnectionError(f"Failed to connect to WhisperLive after {session.max_retries} attempts")
            
        self.sessions[session_id] = session
        return session
        
    async def _connect_to_whisper(self, session: StreamSession) -> bool:
        """Connect to WhisperLive with correct protocol matching the actual WhisperLive implementation"""
        for attempt in range(session.max_retries):
            try:
                session.connection_retries = attempt + 1
                
                # Create WebSocket URL
                whisper_ws_url = f"ws://{self.whisper_host}:{self.whisper_port}"
                logger.info(f"Attempt {attempt + 1}: Connecting to WhisperLive at: {whisper_ws_url}")
                
                # Simple connection - no extra headers needed for WhisperLive
                session.whisper_ws = await websockets.connect(
                    whisper_ws_url,
                    # Disable ping/pong - WhisperLive doesn't use it
                    ping_interval=None,
                    ping_timeout=None,
                    open_timeout=10,
                    close_timeout=5
                )
                
                logger.info(f"WebSocket connected to WhisperLive for session {session.session_id}")
                
                # Send WhisperLive configuration as first message (required per protocol)
                config_message = {
                    "uid": session.session_id,
                    "language": "en", 
                    "task": "transcribe",
                    "model": config.STT_MODEL,
                    "use_vad": config.VAD_ENABLED,
                    "max_clients": 4,
                    "max_connection_time": 3600,
                    "send_last_n_segments": 10,
                    "no_speech_thresh": config.NO_SPEECH_THRESHOLD,
                    "clip_audio": False,
                    "same_output_threshold": 10
                }
                
                logger.info(f"Sending config to WhisperLive: {config_message}")
                await session.whisper_ws.send(json.dumps(config_message))
                
                # WhisperLive starts processing immediately after config - no need to wait for SERVER_READY
                session.whisper_connected = True
                logger.info(f"✅ Session {session.session_id}: WhisperLive connected and configured")
                
                # Start message handler
                session.whisper_message_handler_task = asyncio.create_task(self._handle_whisper_messages(session))
                
                return True
                
            except websockets.exceptions.InvalidUpgrade as e:
                logger.error(f"❌ WebSocket handshake failed for session {session.session_id} (attempt {attempt + 1}): {e}")
                if attempt == session.max_retries - 1:
                    logger.error(f"❌ All attempts failed. Check if WhisperLive is running at ws://{self.whisper_host}:{self.whisper_port}")
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            except ConnectionRefusedError as e:
                logger.error(f"❌ Connection refused for session {session.session_id} (attempt {attempt + 1}): {e}")
                if attempt == session.max_retries - 1:
                    logger.error(f"❌ WhisperLive not accessible at ws://{self.whisper_host}:{self.whisper_port}")
                else:
                    await asyncio.sleep(2 ** attempt)
                    
            except Exception as e:
                logger.error(f"❌ Connection error for session {session.session_id} (attempt {attempt + 1}): {type(e).__name__}: {e}")
                if attempt == session.max_retries - 1:
                    logger.error(f"❌ Failed to connect after {session.max_retries} attempts")
                else:
                    await asyncio.sleep(2 ** attempt)
                    
        return False
        
    async def cleanup_session(self, session_id: str):
        """Clean up a session and its connections"""
        if session_id not in self.sessions:
            return
            
        session = self.sessions[session_id]
        
        # Close WhisperLive connection gracefully
        if session.whisper_ws:
            try:
                # Send close frame first
                await session.whisper_ws.close(code=1000, reason="Session cleanup")
            except Exception as e:
                logger.warning(f"Error closing WhisperLive connection: {e}")

        # Cancel any running whisper message handler task
        if session.whisper_message_handler_task and not session.whisper_message_handler_task.done():
            logger.info(f"Session {session_id}: Cancelling Whisper message handler task during cleanup")
            session.whisper_message_handler_task.cancel()
            try:
                await session.whisper_message_handler_task
            except asyncio.CancelledError:
                pass
                
        # Signal any ongoing TTS to stop
        session.tts_abort_event.set()
        
        # Cancel any running TTS task
        if hasattr(session, 'tts_task') and session.tts_task and not session.tts_task.done():
            logger.info(f"Session {session_id}: Cancelling TTS task during cleanup")
            session.tts_task.cancel()
            try:
                await session.tts_task
            except asyncio.CancelledError:
                pass
                
        # Close all active HTTP clients
        if hasattr(session, 'active_http_clients'):
            for client in list(session.active_http_clients):
                try:
                    await client.aclose()
                except Exception as e:
                    logger.warning(f"Session {session_id}: Error closing HTTP client during cleanup: {e}")
            session.active_http_clients.clear()
        
        del self.sessions[session_id]
        logger.info(f"Session {session_id} cleaned up")
        
    async def _handle_whisper_messages(self, session: StreamSession):
        """Handle incoming messages from WhisperLive with correct protocol"""
        try:
            while session.whisper_ws and self._is_websocket_connected(session.whisper_ws):
                try:
                    message = await asyncio.wait_for(
                        session.whisper_ws.recv(), 
                        timeout=30.0
                    )
                    
                    try:
                        data = json.loads(message)
                        
                        # WhisperLive sends transcription data with "uid" and "segments"
                        if "uid" in data and data["uid"] == session.session_id:
                            if "segments" in data:
                                await self._process_transcript_segments(session, data["segments"])
                            elif "message" in data:
                                # Handle status messages
                                if data["message"] == "DISCONNECT":
                                    logger.info(f"WhisperLive disconnected session {session.session_id}")
                                    break
                                elif "status" in data:
                                    # Handle WAIT, ERROR, WARNING status messages
                                    logger.info(f"WhisperLive status for {session.session_id}: {data}")
                        else:
                            logger.debug(f"Unknown message from WhisperLive: {data}")
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON message from WhisperLive: {message}")
                        
                except websockets.exceptions.ConnectionClosed as e:
                    logger.info(f"WhisperLive connection closed normally for session {session.session_id}: {e}")
                    break
                except asyncio.TimeoutError:
                    logger.debug(f"No message from WhisperLive for 30 seconds (session {session.session_id})")
                    # Check if connection is still alive
                    if not self._is_websocket_connected(session.whisper_ws):
                        break
                    continue
                    
                except Exception as e:
                    logger.error(f"Error processing WhisperLive message: {e}")
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WhisperLive connection closed for session {session.session_id}")
            session.whisper_connected = False
        except Exception as e:
            logger.error(f"WhisperLive handler error for session {session.session_id}: {e}")
            session.whisper_connected = False
            
    async def _connection_health_monitor(self):
        """Background task to monitor and reconnect failed WhisperLive connections"""
        while True:
            try:
                for session_id, session in list(self.sessions.items()):
                    if not session.whisper_connected or not self._is_websocket_connected(session.whisper_ws):
                        logger.warning(f"Detected failed WhisperLive connection for session {session_id}")
                        
                        # Attempt to reconnect
                        if session.connection_retries < session.max_retries:
                            logger.info(f"Attempting to reconnect session {session_id}")
                            success = await self._connect_to_whisper(session)
                            if success:
                                logger.info(f"Successfully reconnected session {session_id}")
                            else:
                                logger.error(f"Failed to reconnect session {session_id}")
                        else:
                            logger.error(f"Max retries exceeded for session {session_id}, cleaning up")
                            await self.cleanup_session(session_id)
                            
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)
                
    async def send_audio_to_whisper(self, session_id: str, audio_data: bytes):
        """Forward audio data to WhisperLive with connection validation"""
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        
        # Check connection health
        if not session.whisper_connected or not session.whisper_ws or not self._is_websocket_connected(session.whisper_ws):
            logger.warning(f"WhisperLive not connected for session {session_id}, attempting reconnect")
            success = await self._connect_to_whisper(session)
            if not success:
                return False
                
        try:
            await session.whisper_ws.send(audio_data)
            session.update_activity()
            return True
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"WhisperLive connection closed while sending audio for session {session_id}")
            session.whisper_connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to send audio to WhisperLive: {e}")
            return False

    async def _process_transcript_segments(self, session: StreamSession, segments):
        """Process transcript segments and trigger LLM+TTS when sentence is complete"""
        session.update_activity()
        
        # Find completed segments
        completed_texts = []
        current_transcript_parts = []
        
        for segment in segments:
            if segment.get("completed") and segment.get("text"):
                text = segment["text"].strip()
                if text and text not in [s.get("text", "") for s in session.conversation_history[-5:]]:
                    completed_texts.append(text)
            else:
                # Incomplete segment for live transcript display
                if segment.get("text"):
                    current_transcript_parts.append(segment["text"])
        
        # Send live transcript to frontend
        current_transcript = " ".join(current_transcript_parts).strip()
        if current_transcript != session.current_transcript:
            session.current_transcript = current_transcript
            await self._send_to_frontend(session, {
                "type": "live_transcript",
                "text": current_transcript
            })
        
        # Process completed sentences
        for completed_text in completed_texts:
            # If this text was the one that was interrupted, ignore it
            if session.last_interrupted_text and completed_text.strip() == session.last_interrupted_text.strip():
                logger.info(f"Session {session.session_id}: Ignoring previously interrupted text: {completed_text}")
                session.last_interrupted_text = None  # Clear after ignoring once
                continue

            if self._is_sentence_complete(completed_text):
                session.stt_end_time = time.time()
                logger.info(f"Session {session.session_id}: Processing complete sentence: {completed_text}")
                await self._process_complete_sentence(session, completed_text)
                
    def _is_sentence_complete(self, text: str) -> bool:
        """Simple sentence completion check"""
        if len(text.split()) < 3:
            return False
        return text.strip().endswith(('.', '!', '?'))
        
    async def _process_complete_sentence(self, session: StreamSession, text: str):
        """Process a complete sentence through LLM and TTS pipeline with proper interruption"""
        if session.is_processing:
            logger.info(f"Session {session.session_id}: Already processing, skipping: {text}")
            return
            
        session.is_processing = True
        session.processing_text = text  # Store the text being processed
        session.total_requests += 1
        
        try:
            # Clear any previous TTS abort signal and reset sequence
            session.tts_abort_event.clear()
            session.tts_sequence_number = 0
            session.tts_queue.clear()
            
            # Notify frontend that processing started
            await self._send_to_frontend(session, {
                "type": "processing_started",
                "text": text
            })
            
            # Generate LLM response with streaming
            full_response = ""
            sentence_buffer = ""
            
            llm_start_time = time.time()
            
            # Start LLM streaming
            async for token in self._stream_llm_response(session, text, session.conversation_history, llm_start_time):
                # Check for interruption FIRST before processing any token
                if session.tts_abort_event.is_set():
                    logger.info(f"🛑 Session {session.session_id}: LLM streaming interrupted")
                    return  # Exit immediately without storing anything
                    
                full_response += token
                sentence_buffer += token
                
                # Check for sentence boundary
                if self._has_sentence_boundary(sentence_buffer):
                    sentence = sentence_buffer.strip()
                    if sentence:
                        # Queue sentence for sequential TTS processing
                        await self._queue_sentence_for_tts(session, sentence)
                        sentence_buffer = ""
            
            # Process any remaining buffer only if not interrupted
            if sentence_buffer.strip() and not session.tts_abort_event.is_set():
                await self._queue_sentence_for_tts(session, sentence_buffer.strip())
            
            # Store conversation only if not interrupted
            if full_response and not session.tts_abort_event.is_set():
                session.conversation_history.append({
                    "user": text,
                    "assistant": full_response,
                    "timestamp": time.time()
                })
                session.last_processed_text = text  # Track the last successfully processed text
                
                # Keep history reasonable
                if len(session.conversation_history) > 10:
                    session.conversation_history = session.conversation_history[-10:]
                    
        except Exception as e:
            logger.error(f"Error processing sentence for session {session.session_id}: {e}")
            await self._send_to_frontend(session, {
                "type": "error",
                "message": "Failed to process request"
            })
        finally:
            session.is_processing = False
            session.processing_text = None # Clear the processing text
            # Reset the abort event when processing ends to prevent infinite loops
            if session.tts_abort_event.is_set():
                session.tts_abort_event.clear()
                logger.info(f"Session {session.session_id}: Cleared abort event after interruption")
            else:
                # Only send processing_complete if not interrupted
                await self._send_to_frontend(session, {
                    "type": "processing_complete"
                })
    
    async def _queue_sentence_for_tts(self, session: StreamSession, sentence: str):
        """Queue a sentence for sequential TTS processing"""
        if session.tts_abort_event.is_set():
            logger.info(f"🛑 Session {session.session_id}: TTS queuing skipped due to interruption")
            return
            
        session.tts_sequence_number += 1
        sequence_number = session.tts_sequence_number
        
        logger.info(f"Session {session.session_id}: Queuing sentence {sequence_number} for TTS: {sentence[:50]}...")
        
        # Add to queue
        session.tts_queue.append({
            "sequence": sequence_number,
            "text": sentence,
            "queued_at": time.time()
        })
        
        # Start processing the queue (this will handle sequential processing)
        # Only create a new task if there isn't one already running
        if not session.tts_task or session.tts_task.done():
            session.tts_task = asyncio.create_task(self._process_tts_queue(session))
    
    async def _process_tts_queue(self, session: StreamSession):
        """Process TTS queue sequentially to prevent voice avalanche"""
        try:
            async with session.tts_processing_lock:
                while session.tts_queue and not session.tts_abort_event.is_set():
                    # Check for interruption before processing each item
                    if session.tts_abort_event.is_set():
                        logger.info(f"🛑 Session {session.session_id}: TTS queue processing aborted due to interruption")
                        break
                        
                    # Get next sentence from queue
                    next_item = session.tts_queue.pop(0)
                    sequence = next_item["sequence"]
                    sentence = next_item["text"]
                    
                    logger.info(f"Session {session.session_id}: Processing TTS for sequence {sequence}")
                    
                    try:
                        session.tts_active = True
                        
                        # Create client and track it for cancellation
                        client = httpx.AsyncClient(timeout=config.TTS_TIMEOUT)
                        if not hasattr(session, 'active_http_clients'):
                            session.active_http_clients = set()
                        session.active_http_clients.add(client)
                        
                        try:
                            # Check for interruption before making request
                            if session.tts_abort_event.is_set():
                                logger.info(f"🛑 Session {session.session_id}: TTS aborted before HTTP request for sequence {sequence}")
                                break
                            
                            tts_start_time = time.time()
                            # Generate TTS audio with cancellation support
                            response = await client.post(
                                f"{config.TTS_URL}/v1/audio/speech",
                                json={
                                    "model": "kokoro",
                                    "input": sentence,
                                    "voice": config.TTS_VOICE,
                                    "response_format": "wav",
                                    "stream": False,
                                    "speed": config.TTS_SPEED,
                                    "volume_multiplier": config.TTS_VOLUME
                                }
                            )
                            
                            # Check for interruption after request completes
                            if session.tts_abort_event.is_set():
                                logger.info(f"🛑 Session {session.session_id}: TTS aborted after HTTP request for sequence {sequence}")
                                break
                                
                            if response.status_code == 200:
                                tts_end_time = time.time()
                                latency = tts_end_time - tts_start_time
                                logger.info(f"PERF: Session {session.session_id}: TTS generation for sequence {sequence} took {latency:.4f}s")
                                session.metrics['tts_latency'] += latency

                                audio_data = response.content
                                
                                # Final check before sending to frontend
                                if not session.tts_abort_event.is_set():
                                    # Stream to frontend
                                    await self._send_to_frontend(session, {
                                        "type": "sentence_audio",
                                        "sequence": sequence,
                                        "text": sentence,
                                        "audio_data": base64.b64encode(audio_data).decode(),
                                        "size_bytes": len(audio_data)
                                    })
                                    
                                    if sequence == 1 and session.stt_end_time:
                                        total_latency = time.time() - session.stt_end_time
                                        logger.info(f"PERF: Session {session.session_id}: Total pipeline latency (to first TTS audio): {total_latency:.4f}s")
                                        session.metrics['total_pipeline_latency'] = total_latency

                                    logger.info(f"Session {session.session_id}: Streamed sentence {sequence}")
                                else:
                                    logger.info(f"🛑 Session {session.session_id}: TTS aborted before streaming sequence {sequence}")
                                    break
                            else:
                                logger.warning(f"Session {session.session_id}: TTS failed for sequence {sequence}, status: {response.status_code}")
                                
                        finally:
                            # Clean up the client
                            session.active_http_clients.discard(client)
                            await client.aclose()
                            
                    except asyncio.CancelledError:
                        logger.info(f"🛑 Session {session.session_id}: TTS task cancelled for sequence {sequence}")
                        break
                    except Exception as e:
                        logger.error(f"TTS error for session {session.session_id}, sequence {sequence}: {e}")
                    finally:
                        session.tts_active = False
                    
                    # Small delay to prevent overwhelming the system while maintaining low latency
                    # Also check for interruption during delay
                    try:
                        await asyncio.wait_for(asyncio.sleep(0.1), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
                    
                logger.info(f"Session {session.session_id}: TTS queue processing complete")
                
        except asyncio.CancelledError:
            logger.info(f"🛑 Session {session.session_id}: TTS queue processing task cancelled")
            raise
            
    def _has_sentence_boundary(self, text: str) -> bool:
        """Check if text contains a sentence boundary"""
        import re
        # Check for sentence-ending punctuation at the end of text
        pattern = r'[.!?]\s*$'
        return bool(re.search(pattern, text.strip()))
        
    async def _stream_llm_response(self, session: StreamSession, text: str, history: list, llm_start_time: float):
        """Stream LLM tokens using ollama"""
        # Build context from history
        context_parts = []
        for exchange in history[-3:]:  # Last 3 exchanges
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant']}")
        
        context_parts.append(f"User: {text}")
        context_parts.append("Assistant:")
        prompt = "\n".join(context_parts)
        
        try:
            client = ollama.AsyncClient(host=self.ollama_url)
            stream = await client.generate(
                model=config.LLM_MODEL,
                prompt=prompt,
                stream=True,
                options={
                    "num_predict": config.LLM_MAX_TOKENS,
                    "temperature": config.LLM_TEMPERATURE,
                    "top_p": 0.8,
                    "num_ctx": 2048,
                }
            )
            
            first_token_received = False
            async for chunk in stream:
                if not first_token_received:
                    first_token_time = time.time()
                    latency = first_token_time - llm_start_time
                    logger.info(f"PERF: Session {session.session_id}: LLM time to first token: {latency:.4f}s")
                    session.metrics['llm_first_token_latency'] = latency
                    first_token_received = True

                if chunk.get('response'):
                    yield chunk['response']
                if chunk.get('done'):
                    llm_end_time = time.time()
                    total_latency = llm_end_time - llm_start_time
                    logger.info(f"PERF: Session {session.session_id}: LLM total response time: {total_latency:.4f}s")
                    session.metrics['llm_total_latency'] = total_latency
                    break
                    
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield "I'm sorry, I couldn't process your request right now."
            
    async def _process_sentence_tts(self, session: StreamSession, sentence: str, sequence: int):
        """Process a single sentence through TTS and stream to frontend"""
        if session.tts_abort_event.is_set():
            logger.info(f"🛑 Session {session.session_id}: TTS skipped due to interruption")
            return
            
        try:
            session.tts_active = True
            
            # Generate TTS audio
            async with httpx.AsyncClient(timeout=config.TTS_TIMEOUT) as client:
                response = await client.post(
                    f"{config.TTS_URL}/v1/audio/speech",
                    json={
                        "model": "kokoro",
                        "input": sentence,
                        "voice": config.TTS_VOICE,
                        "response_format": "wav",
                        "stream": False,
                        "speed": config.TTS_SPEED,
                        "volume_multiplier": config.TTS_VOLUME
                    }
                )
                
                if response.status_code == 200 and not session.tts_abort_event.is_set():
                    audio_data = response.content
                    
                    # Stream to frontend
                    await self._send_to_frontend(session, {
                        "type": "sentence_audio",
                        "sequence": sequence,
                        "text": sentence,
                        "audio_data": base64.b64encode(audio_data).decode(),
                        "size_bytes": len(audio_data)
                    })
                    
                    logger.info(f"Session {session.session_id}: Streamed sentence {sequence}")
                    
        except Exception as e:
            logger.error(f"TTS error for session {session.session_id}: {e}")
        finally:
            session.tts_active = False

    async def _send_to_frontend(self, session: StreamSession, message: dict):
        """Send message to frontend WebSocket"""
        if session.frontend_ws:
            try:
                await session.frontend_ws.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send to frontend: {e}")
                
    async def interrupt_session(self, session_id: str) -> bool:
        """Interrupt TTS and processing for a session without dropping the WhisperLive connection."""
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        
        # 1. Signal abort to all async operations
        session.tts_abort_event.set()
        
        # 2. Cancel any running TTS task
        if session.tts_task and not session.tts_task.done():
            session.tts_task.cancel()
            try:
                await session.tts_task
            except asyncio.CancelledError:
                pass # Expected

        # 3. Reset processing state
        session.is_processing = False
        session.tts_active = False
        
        # 4. Clear the TTS queue to prevent pending sentences from playing
        session.tts_queue.clear()
        session.tts_sequence_number = 0
        
        # 5. Send a reset message to WhisperLive to clear its internal buffer
        if session.whisper_ws and self._is_websocket_connected(session.whisper_ws):
            try:
                # This message tells WhisperLive to reset the client's audio buffer
                await session.whisper_ws.send(json.dumps({"uid": session.session_id, "message": "CLIENT_DISCONNECT"}))
                logger.info(f"Session {session_id}: Sent reset signal to WhisperLive.")
            except Exception as e:
                logger.error(f"Session {session_id}: Failed to send reset signal to WhisperLive: {e}")

        # 6. Notify the frontend
        await self._send_to_frontend(session, {
            "type": "interrupted"
        })
        
        logger.info(f"Session {session_id}: Interrupted TTS and processing. WhisperLive connection remains open.")
        return True
        
    async def _session_cleanup_task(self):
        """Background task to clean up inactive sessions"""
        while True:
            try:
                current_time = time.time()
                sessions_to_cleanup = [
                    sid for sid, session in self.sessions.items()
                    if session.can_be_cleaned_up()
                ]
                
                for session_id in sessions_to_cleanup:
                    await self.cleanup_session(session_id)
                    
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(60)

# Initialize global orchestrator
orchestrator = VoiceStreamOrchestrator()


# WebSocket endpoint for frontend connections
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Single WebSocket endpoint for all voice interactions"""
    await websocket.accept()
    logger.info(f"Frontend WebSocket connected for session {session_id}")
    
    try:
        # Create or get session
        session = await orchestrator.create_session(session_id)
        session.frontend_ws = websocket
        
        # Send ready signal
        await websocket.send_text(json.dumps({
            "type": "ready",
            "session_id": session_id
        }))
        
        # Handle incoming frontend messages
        while True:
            # Correct pattern for FastAPI WebSocket - use receive() 
            message = await websocket.receive()
            
            try:
                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        # Audio data - forward to WhisperLive
                        await orchestrator.send_audio_to_whisper(session_id, message["bytes"])
                        
                    elif "text" in message:
                        data = json.loads(message["text"])
                        
                        if data.get("type") == "interrupt":
                            await orchestrator.interrupt_session(session_id)
                            
                        elif data.get("type") == "end_audio":
                            # Forward end signal to WhisperLive
                            if session.whisper_ws:
                                await session.whisper_ws.send("END_OF_AUDIO")
                                
                elif message["type"] == "websocket.disconnect":
                    logger.info(f"Frontend WebSocket disconnect message received for session {session_id}")
                    break
                    
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from frontend: {message}")
            except Exception as e:
                logger.error(f"Error handling frontend message: {e}")
                
    except WebSocketDisconnect:
        logger.info(f"Frontend WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        if session_id in orchestrator.sessions:
            orchestrator.sessions[session_id].frontend_ws = None
        await orchestrator.cleanup_session(session_id)

# Health check endpoint
@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "active_sessions": len(orchestrator.sessions),
        "timestamp": time.time()
    })

# Debug endpoint
@app.get("/debug/sessions")
async def debug_sessions():
    return {
        "sessions": {
            sid: {
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "is_processing": session.is_processing,
                "total_requests": session.total_requests,
                "has_frontend": session.frontend_ws is not None,
                "has_whisper": session.whisper_ws is not None,
                "metrics": session.metrics
            }
            for sid, session in orchestrator.sessions.items()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)