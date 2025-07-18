import asyncio
import json
import logging
import os
import time
import base64
import traceback
import sys
from typing import Dict, Set, Optional, Any, Callable
from urllib.parse import urlparse
import websockets
import websockets.exceptions
import httpx
import ollama
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import config
from plugins import PluginManager, MemoryPlugin, SpeakerPlugin
import hashlib

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Global exception handler for unhandled exceptions
def global_exception_handler(exctype, value, tb):
    """Handle uncaught exceptions with detailed logging"""
    logger.critical("UNCAUGHT EXCEPTION", exc_info=(exctype, value, tb))
    logger.critical(f"Exception type: {exctype.__name__}")
    logger.critical(f"Exception value: {value}")
    logger.critical(f"Traceback: {''.join(traceback.format_tb(tb))}")
    
# Set the global exception handler
sys.excepthook = global_exception_handler

# Handle async exceptions
def handle_async_exception(loop, context):
    """Handle exceptions in async tasks"""
    logger.critical(f"ASYNC EXCEPTION: {context}")
    if 'exception' in context:
        logger.critical(f"Exception: {context['exception']}")
        logger.critical(f"Exception type: {type(context['exception']).__name__}")
        
# Set async exception handler (will be set on the event loop when it's created)
# asyncio.set_exception_handler(handle_async_exception) - This doesn't exist, need to set on loop

class PipelineEventBus:
    """
    Fire-and-forget event bus for ultra-fast pipeline processing
    """
    def __init__(self):
        self._listeners: Dict[str, list[Callable]] = {}
        self._logger = logging.getLogger(__name__ + ".PipelineEventBus")
        
    def on(self, event_type: str, callback: Callable):
        """Register an event listener"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
        
    def off(self, event_type: str, callback: Callable):
        """Remove an event listener"""
        if event_type in self._listeners:
            try:
                self._listeners[event_type].remove(callback)
            except ValueError:
                pass
                
    async def emit(self, event_type: str, data: Any = None):
        """Fire-and-forget event emission"""
        if event_type not in self._listeners:
            return
            
        # Create tasks for all listeners without waiting
        tasks = []
        for callback in self._listeners[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    task = asyncio.create_task(callback(data))
                else:
                    # Handle sync callbacks
                    task = asyncio.create_task(asyncio.to_thread(callback, data))
                tasks.append(task)
            except Exception as e:
                self._logger.error(f"Error creating task for event {event_type}: {e}")
                
        # Fire and forget - don't wait for completion
        if tasks:
            # Log completion but don't wait
            asyncio.create_task(self._log_completion(event_type, tasks))
            
    async def _log_completion(self, event_type: str, tasks: list):
        """Log completion of event processing without blocking"""
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                self._logger.warning(f"Event {event_type}: {len(errors)}/{len(tasks)} listeners failed")
        except Exception as e:
            self._logger.error(f"Error in event completion logging: {e}")

app = FastAPI(title="Voice Stream Orchestrator", version="2.0.0")

# Configure CORS for WebSocket connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up async exception handler on startup
@app.on_event("startup")
async def startup_event():
    """Set up async exception handler when the app starts"""
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_async_exception)
    logger.info("Async exception handler set up")
    logger.info("Voice Stream Orchestrator starting up...")

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
        
        # Event bus for ultra-fast processing
        self.event_bus = PipelineEventBus()
        
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

    async def process_ultra_fast(self, text: str) -> None:
        """Ultra-fast processing using event bus for fire-and-forget execution"""
        try:
            # Emit event for immediate processing without waiting
            await self.event_bus.emit("ultra_fast_process", {
                "session_id": self.session_id,
                "text": text,
                "timestamp": time.time()
            })
        except Exception as e:
            logger.error(f"Error in ultra-fast processing for session {self.session_id}: {e}")

class VoiceStreamOrchestrator:
    """
    Central coordinator for all voice processing streams
    """
    def __init__(self):
        self.sessions: Dict[str, StreamSession] = {}
        self.event_bus = PipelineEventBus()
        self.plugin_manager = PluginManager()
        
        # Initialize plugins
        self._initialize_plugins()
        
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
        
    def _initialize_plugins(self) -> None:
        """Initialize and register plugins."""
        try:
            # Register memory plugin
            if config.MEMORY_PLUGIN_ENABLED:
                from .plugins.base_plugin import PluginConfig
                memory_config = PluginConfig(
                    enabled=True,
                    priority=1,
                    max_workers=config.PLUGIN_MAX_WORKERS,
                    timeout=config.PLUGIN_TIMEOUT
                )
                memory_plugin = MemoryPlugin(memory_config)
                self.plugin_manager.register_plugin(memory_plugin)
                logger.info("Memory plugin registered")
            
            # Register speaker plugin
            if config.SPEAKER_PLUGIN_ENABLED:
                from .plugins.base_plugin import PluginConfig
                speaker_config = PluginConfig(
                    enabled=True,
                    priority=2,
                    max_workers=config.PLUGIN_MAX_WORKERS,
                    timeout=config.PLUGIN_TIMEOUT
                )
                speaker_plugin = SpeakerPlugin(speaker_config)
                self.plugin_manager.register_plugin(speaker_plugin)
                logger.info("Speaker plugin registered")
            
            # Start plugin manager
            asyncio.create_task(self.plugin_manager.start_all())
            logger.info("Plugins initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing plugins: {e}")
    
    async def cleanup_plugins(self) -> None:
        """Cleanup all plugins."""
        try:
            await self.plugin_manager.stop_all()
            logger.info("Plugins cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up plugins: {e}")
        
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
        logger.debug(f"Creating new session: {session_id}")
        
        if session_id in self.sessions:
            logger.debug(f"Session {session_id} already exists, cleaning up old session")
            await self.cleanup_session(session_id)
            
        logger.debug(f"Creating StreamSession object for {session_id}")
        session = StreamSession(session_id)
        logger.debug(f"StreamSession created for {session_id}")
        
        # Establish WhisperLive connection with proper headers
        logger.debug(f"Establishing WhisperLive connection for session {session_id}")
        success = await self._connect_to_whisper(session)
        if not success:
            logger.error(f"Failed to establish WhisperLive connection for session {session_id}")
            raise ConnectionError(f"Failed to connect to WhisperLive after {session.max_retries} attempts")
            
        logger.debug(f"WhisperLive connection established for session {session_id}")
        self.sessions[session_id] = session
        logger.debug(f"Session {session_id} added to orchestrator sessions")
        return session
        
    async def _connect_to_whisper(self, session: StreamSession) -> bool:
        """Connect to WhisperLive with correct protocol matching the actual WhisperLive implementation"""
        logger.debug(f"Starting WhisperLive connection for session {session.session_id}")
        
        for attempt in range(session.max_retries):
            try:
                session.connection_retries = attempt + 1
                
                # Create WebSocket URL
                whisper_ws_url = f"ws://{self.whisper_host}:{self.whisper_port}"
                logger.info(f"Attempt {attempt + 1}: Connecting to WhisperLive at: {whisper_ws_url}")
                logger.debug(f"Session {session.session_id}: Creating WebSocket connection...")
                
                # Simple connection - using the working pattern
                session.whisper_ws = await websockets.connect(whisper_ws_url)
                
                logger.info(f"✅ WebSocket connected to WhisperLive for session {session.session_id}")
                logger.debug(f"Session {session.session_id}: WebSocket state: {session.whisper_ws.state}")
                
                # CRITICAL: Send WhisperLive configuration immediately as first message
                # WhisperLive expects this exact format and will timeout if not received quickly
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
                
                logger.info(f"📤 Sending config to WhisperLive: {config_message}")
                logger.debug(f"Session {session.session_id}: Config message JSON: {json.dumps(config_message)}")
                
                # Send immediately without delay to avoid handshake timeout
                await session.whisper_ws.send(json.dumps(config_message))
                logger.info(f"✅ Configuration sent to WhisperLive successfully")
                logger.debug(f"Session {session.session_id}: Config sent, WebSocket state: {session.whisper_ws.state}")
                
                # WhisperLive initializes the client immediately after receiving config
                session.whisper_connected = True
                logger.info(f"✅ Session {session.session_id}: WhisperLive connected and configured")
                
                # Start message handler to process WhisperLive responses
                logger.debug(f"Session {session.session_id}: Starting WhisperLive message handler task")
                session.whisper_message_handler_task = asyncio.create_task(self._handle_whisper_messages(session))
                logger.debug(f"Session {session.session_id}: Message handler task started")
                
                return True
                
            except (websockets.exceptions.InvalidUpgrade, websockets.exceptions.InvalidHandshake) as e:
                logger.error(f"❌ WebSocket handshake failed for session {session.session_id} (attempt {attempt + 1}): {e}")
                logger.debug(f"Session {session.session_id}: Handshake failure details: {type(e).__name__}: {str(e)}")
                if attempt == session.max_retries - 1:
                    logger.error(f"❌ All handshake attempts failed. Check if WhisperLive is running correctly at ws://{self.whisper_host}:{self.whisper_port}")
                else:
                    sleep_time = 1 + attempt
                    logger.debug(f"Session {session.session_id}: Sleeping {sleep_time}s before retry")
                    await asyncio.sleep(sleep_time)  # Progressive backoff
                    
            except (ConnectionRefusedError, OSError) as e:
                logger.error(f"❌ Connection refused for session {session.session_id} (attempt {attempt + 1}): {e}")
                logger.debug(f"Session {session.session_id}: Connection refused details: {type(e).__name__}: {str(e)}")
                if attempt == session.max_retries - 1:
                    logger.error(f"❌ WhisperLive not accessible at ws://{self.whisper_host}:{self.whisper_port}")
                else:
                    sleep_time = 2 + attempt
                    logger.debug(f"Session {session.session_id}: Sleeping {sleep_time}s before retry")
                    await asyncio.sleep(sleep_time)
                    
            except asyncio.TimeoutError as e:
                logger.error(f"❌ Connection timeout for session {session.session_id} (attempt {attempt + 1}): {e}")
                logger.debug(f"Session {session.session_id}: Timeout details: {type(e).__name__}: {str(e)}")
                if attempt == session.max_retries - 1:
                    logger.error(f"❌ All timeout attempts failed for WhisperLive at ws://{self.whisper_host}:{self.whisper_port}")
                else:
                    sleep_time = 1 + attempt
                    logger.debug(f"Session {session.session_id}: Sleeping {sleep_time}s before retry")
                    await asyncio.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"❌ Connection error for session {session.session_id} (attempt {attempt + 1}): {type(e).__name__}: {e}")
                logger.debug(f"Session {session.session_id}: Exception details: {type(e).__name__}: {str(e)}")
                logger.debug(f"Session {session.session_id}: Exception traceback:", exc_info=True)
                if attempt == session.max_retries - 1:
                    logger.error(f"❌ Failed to connect after {session.max_retries} attempts")
                else:
                    sleep_time = 2 + attempt
                    logger.debug(f"Session {session.session_id}: Sleeping {sleep_time}s before retry")
                    await asyncio.sleep(sleep_time)
                    
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
        logger.debug(f"Session {session.session_id}: Starting WhisperLive message handler")
        
        try:
            while session.whisper_ws and self._is_websocket_connected(session.whisper_ws):
                try:
                    logger.debug(f"Session {session.session_id}: Waiting for WhisperLive message...")
                    message = await asyncio.wait_for(
                        session.whisper_ws.recv(), 
                        timeout=30.0
                    )
                    logger.debug(f"Session {session.session_id}: Received message from WhisperLive: {message[:100]}...")
                    
                    try:
                        data = json.loads(message)
                        logger.debug(f"Session {session.session_id}: Parsed JSON message: {data}")
                        
                        # WhisperLive sends transcription data with "uid" and "segments"
                        if "uid" in data and data["uid"] == session.session_id:
                            if "segments" in data:
                                logger.info(f"📝 Received transcript segments from WhisperLive: {data['segments']}")
                                with open("/tmp/whisper_transcripts.log", "a") as f:
                                    f.write(json.dumps(data['segments']) + "\n")
                                logger.debug(f"Session {session.session_id}: Processing transcript segments")
                                if session.frontend_ws:                                                       
                                  await session.frontend_ws.send_text(json.dumps({                          
                                    "type": "segments",                              
                                    "segments": data["segments"]                     
                                    }))  
                                await self._process_transcript_segments(session, data["segments"])
                            elif "message" in data:
                                # Handle status messages
                                logger.debug(f"Session {session.session_id}: Processing status message: {data['message']}")
                                if data["message"] == "DISCONNECT":
                                    logger.info(f"WhisperLive disconnected session {session.session_id}")
                                    break
                                elif data["message"] == "SERVER_READY":
                                    logger.info(f"WhisperLive server ready for session {session.session_id}")
                                else:
                                    logger.debug(f"Session {session.session_id}: Unknown message type: {data['message']}")
                            elif "status" in data:
                                # Handle WAIT, ERROR, WARNING status messages
                                logger.info(f"WhisperLive status for {session.session_id}: {data}")
                        else:
                            logger.debug(f"Session {session.session_id}: Message for different session or unknown format: {data}")
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Session {session.session_id}: Non-JSON message from WhisperLive: {message}")
                        logger.debug(f"Session {session.session_id}: JSON decode error: {e}")
                        
                except websockets.exceptions.ConnectionClosed as e:
                    logger.info(f"WhisperLive connection closed normally for session {session.session_id}: {e}")
                    logger.debug(f"Session {session.session_id}: Connection closed details: {type(e).__name__}: {str(e)}")
                    break
                except asyncio.TimeoutError:
                    logger.debug(f"No message from WhisperLive for 30 seconds (session {session.session_id})")
                    # Check if connection is still alive
                    if not self._is_websocket_connected(session.whisper_ws):
                        logger.debug(f"Session {session.session_id}: WhisperLive connection is no longer alive")
                        break
                    continue
                    
                except Exception as e:
                    logger.error(f"Error processing WhisperLive message for session {session.session_id}: {e}")
                    logger.debug(f"Session {session.session_id}: Message processing error details:", exc_info=True)
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
            logger.warning(f"❌ Session {session_id} not found")
            return False
            
        session = self.sessions[session_id]
        
        # Check connection health
        if not session.whisper_connected or not session.whisper_ws or not self._is_websocket_connected(session.whisper_ws):
            logger.warning(f"🔄 WhisperLive not connected for session {session_id}, attempting reconnect")
            success = await self._connect_to_whisper(session)
            if not success:
                logger.error(f"❌ Failed to reconnect WhisperLive for session {session_id}")
                return False
                
        try:
            # DETAILED LOGGING FOR AUDIO DATA FORMAT DEBUGGING
            logger.info(f"📤 Sending {len(audio_data)} bytes of audio to WhisperLive for session {session_id}")
            logger.debug(f"🔍 Audio data type: {type(audio_data)}")
            # logger.debug(f"🔍 Audio data repr: {repr(audio_data)}")
            logger.debug(f"🔍 Audio data length: {len(audio_data)}")
            
            if len(audio_data) >= 16:
                # logger.debug(f"🔍 First 16 bytes (hex): {audio_data[:16].hex()}")
                
                # Try to interpret as Float32Array
                import struct
                try:
                    first_samples = struct.unpack('<ffff', audio_data[:16])
                    # logger.debug(f"🔍 First 4 samples as Float32: {first_samples}")
                except struct.error as e:
                    logger.debug(f"🔍 Cannot interpret as Float32: {e}")
                
                # Try to interpret as Int16Array
                try:
                    first_samples_int16 = struct.unpack('<hhhhhhhh', audio_data[:16])
                    # logger.debug(f"🔍 First 8 samples as Int16: {first_samples_int16}")
                except struct.error as e:
                    logger.debug(f"🔍 Cannot interpret as Int16: {e}")
                # --- ADDED: Log audio stats for debugging ---
                import numpy as np
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                # logger.info(f"🔊 Audio stats: min={audio_array.min()}, max={audio_array.max()}, mean={audio_array.mean()}, std={audio_array.std()}")
                # --- END ADDED ---            
            # Check if audio_data is actually bytes vs string
            if isinstance(audio_data, str):
                logger.error(f"❌ CRITICAL: Audio data is a STRING, not bytes! This will cause the WhisperLive error!")
                logger.error(f"❌ String content: {audio_data[:100]}...")
                # Convert string to bytes if needed
                logger.warning(f"⚠️ Converting string to bytes as emergency fix")
                audio_data = audio_data.encode('utf-8')
            elif isinstance(audio_data, bytes):
                logger.debug(f"✅ Audio data is bytes type as expected")
            else:
                logger.error(f"❌ CRITICAL: Audio data is neither string nor bytes: {type(audio_data)}")
                
            await session.whisper_ws.send(audio_data)
            session.update_activity()
            logger.info(f"✅ Successfully sent audio to WhisperLive for session {session_id}")
            
            # Emit plugin events for audio data
            import numpy as np
            # Audio data is now Float32Array format (not int16)
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            await self.plugin_manager.emit_event("audio_data", {
                "session_id": session_id,
                "audio": audio_array.tolist(),
                "timestamp": time.time(),
                "user_id": session_id
            })
            
            return True
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"❌ WhisperLive connection closed while sending audio for session {session_id}")
            session.whisper_connected = False
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send audio to WhisperLive: {e}")
            logger.debug(f"❌ Exception details:", exc_info=True)
            return False

    async def _process_transcript_segments(self, session: StreamSession, segments):
        """Process transcript segments and trigger LLM+TTS when sentence is complete"""
        session.update_activity()
        logger.info(f"🎯 Processing transcript segments for session {session.session_id}: {segments}")
        
        # Find completed segments
        completed_texts = []
        current_transcript_parts = []
        
        for segment in segments:
            if segment.get("completed") and segment.get("text"):
                text = segment["text"].strip()
                if text and text not in [s.get("text", "") for s in session.conversation_history[-5:]]:
                    completed_texts.append(text)
                    logger.info(f"✅ Completed text found: {text}")
            else:
                # Incomplete segment for live transcript display
                if segment.get("text"):
                    current_transcript_parts.append(segment["text"])
        
        # Send live transcript to frontend
        current_transcript = " ".join(current_transcript_parts).strip()
        logger.info(f"📝 Current live transcript: {current_transcript}")
        
        if current_transcript != session.current_transcript:
            session.current_transcript = current_transcript
            await self._send_to_frontend(session, {
                "type": "live_transcript",
                "text": current_transcript
            })
            
            # Emit plugin events for live transcript
            await self.plugin_manager.emit_event("live_transcript", {
                "session_id": session.session_id,
                "text": current_transcript,
                "timestamp": time.time()
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
                
                # Emit plugin events for completed transcription
                await self.plugin_manager.emit_event("transcription_complete", {
                    "session_id": session.session_id,
                    "text": completed_text,
                    "timestamp": time.time(),
                    "user_id": session.session_id  # Use session_id as user_id for now
                })
                
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
                logger.info(f"📤 Sending to frontend: {message}")
                await session.frontend_ws.send_text(json.dumps(message))
                logger.info(f"✅ Successfully sent to frontend: {message.get('type', 'unknown')}")
            except Exception as e:
                logger.error(f"❌ Failed to send to frontend: {e}")
                
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
    logger.info(f"🎯 Frontend WebSocket connected for session {session_id}")
    
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
            # logger.info(f"📥 Received frontend message: {message}")
            
            try:
                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        # Audio data - forward to WhisperLive
                        audio_bytes = message["bytes"]
                        logger.info(f"🎤 Received {len(audio_bytes)} bytes of audio data from frontend")
                        
                        # DETAILED LOGGING FOR AUDIO DATA FORMAT DEBUGGING
                        logger.debug(f"🔍 WebSocket message type: {message['type']}")
                        logger.debug(f"🔍 WebSocket message keys: {list(message.keys())}")
                        # logger.debug(f"🔍 Raw audio_bytes type: {type(audio_bytes)}")
                        # logger.debug(f"🔍 Raw audio_bytes repr: {repr(audio_bytes)}")
                        
                        # Log audio data format for debugging
                        if len(audio_bytes) >= 4:
                            # Check if it's float32 by looking at the first few bytes
                            import struct
                            try:
                                first_sample = struct.unpack('<f', audio_bytes[:4])[0]
                                # logger.debug(f"📊 Audio data format check - first sample: {first_sample}")
                            except:
                                logger.debug(f"📊 Audio data format - raw bytes: {audio_bytes[:16].hex()}")
                        
                        success = await orchestrator.send_audio_to_whisper(session_id, audio_bytes)
                        if not success:
                            logger.error(f"❌ Failed to send audio to WhisperLive")
                         
                    elif "text" in message:
                        data = json.loads(message["text"])
                        logger.info(f"💬 Received text message: {data}")
                        
                        if data.get("type") == "interrupt":
                            logger.info(f"🛑 Interrupt requested for session {session_id}")
                            await orchestrator.interrupt_session(session_id)
                            
                        elif data.get("type") == "end_audio":
                            logger.info(f"🔚 End of audio signal received for session {session_id}")
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
# Ultra-fast WebSocket endpoint for voice processing
@app.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    """Ultra-fast WebSocket endpoint for voice processing with event bus"""
    logger.debug(f"New WebSocket connection attempt to /ws/voice")
    await websocket.accept()
    session_id = f"voice_{int(time.time() * 1000)}"
    logger.info(f"Ultra-fast WebSocket connected for session {session_id}")
    logger.debug(f"Session {session_id}: WebSocket accepted, client info: {websocket.client}")
    
    try:
        # Create session with event bus integration
        logger.debug(f"Session {session_id}: Creating orchestrator session...")
        session = await orchestrator.create_session(session_id)
        logger.debug(f"Session {session_id}: Orchestrator session created successfully")
        session.frontend_ws = websocket
        logger.debug(f"Session {session_id}: Frontend WebSocket assigned to session")
        
        # Register ultra-fast processing handler
        async def handle_ultra_fast_process(event_data):
            """Handle ultra-fast processing events"""
            try:
                text = event_data["text"]
                event_session_id = event_data["session_id"]
                
                # Skip if not for this session
                if event_session_id != session.session_id:
                    return
                    
                # Process immediately without waiting
                await session._process_complete_sentence(session, text)
                
            except Exception as e:
                logger.error(f"Error in ultra-fast processing handler: {e}")
        
        # Register the handler
        session.event_bus.on("ultra_fast_process", handle_ultra_fast_process)
        
        # Send ready signal
        await websocket.send_text(json.dumps({
            "type": "ready",
            "session_id": session_id,
            "mode": "ultra_fast"
        }))
        
        # Handle incoming messages
        while True:
            message = await websocket.receive()
            
            try:
                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        # Audio data - forward to WhisperLive
                        audio_bytes = message["bytes"]
                        logger.info(f"🎤 [Ultra-fast] Received {len(audio_bytes)} bytes of audio data from frontend")
                        
                        # DETAILED LOGGING FOR AUDIO DATA FORMAT DEBUGGING
                        logger.debug(f"🔍 WebSocket message type: {message['type']}")
                        logger.debug(f"🔍 WebSocket message keys: {list(message.keys())}")
                        # logger.debug(f"🔍 Raw audio_bytes type: {type(audio_bytes)}")
                        # logger.debug(f"🔍 Raw audio_bytes repr: {repr(audio_bytes)}")
                        
                        if len(audio_bytes) >= 16:
                            logger.debug(f"🔍 First 16 bytes (hex): {audio_bytes[:16].hex()}")
                        
                        await orchestrator.send_audio_to_whisper(session_id, audio_bytes)
                        
                    elif "text" in message:
                        data = json.loads(message["text"])
                        
                        if data.get("type") == "interrupt":
                            await orchestrator.interrupt_session(session_id)
                            
                        elif data.get("type") == "end_audio":
                            # Forward end signal to WhisperLive
                            if session.whisper_ws:
                                await session.whisper_ws.send("END_OF_AUDIO")
                                
                        elif data.get("type") == "ultra_fast_text":
                            # Direct text processing for ultra-fast mode
                            text = data.get("text", "").strip()
                            if text:
                                await session.process_ultra_fast(text)
                                
                elif message["type"] == "websocket.disconnect":
                    logger.info(f"Ultra-fast WebSocket disconnect for session {session_id}")
                    break
                    
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from ultra-fast client: {message}")
            except Exception as e:
                logger.error(f"Error handling ultra-fast message: {e}")
                
    except WebSocketDisconnect:
        logger.info(f"Ultra-fast WebSocket disconnected for session {session_id}")
        logger.debug(f"Session {session_id}: WebSocket disconnect details")
    except Exception as e:
        logger.error(f"Ultra-fast WebSocket error for session {session_id}: {e}")
        logger.debug(f"Session {session_id}: WebSocket exception details:", exc_info=True)
    finally:
        logger.debug(f"Session {session_id}: Cleaning up in WebSocket finally block")
        if session_id in orchestrator.sessions:
            # Clean up event handlers
            session = orchestrator.sessions[session_id]
            # Find and remove the handler by creating a closure
            orchestrator.sessions[session_id].frontend_ws = None
            logger.debug(f"Session {session_id}: Frontend WebSocket cleared")
        logger.debug(f"Session {session_id}: Calling cleanup_session")
        await orchestrator.cleanup_session(session_id)
        logger.debug(f"Session {session_id}: Cleanup completed")

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