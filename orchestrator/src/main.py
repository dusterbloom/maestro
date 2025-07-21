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
from functools import partial # <-- ADD THIS IMPORT

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
from event_bus import DistributedEventBus, ServiceEvent
from service_coordinator import ServiceCoordinator, ServiceState, ServiceType
try:
    from plugins.interrupt_plugin import InterruptPlugin
    INTERRUPT_PLUGIN_AVAILABLE = True
except ImportError as e:
    InterruptPlugin = None
    INTERRUPT_PLUGIN_AVAILABLE = False
    print(f"WARNING: InterruptPlugin not available: {e}")
import hashlib

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
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

# PipelineEventBus removed - replaced with DistributedEventBus

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
    """Set up async exception handler and initialize distributed systems when the app starts"""
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_async_exception)
    logger.info("Async exception handler set up")
    logger.info("Voice Stream Orchestrator starting up...")
    
    # Initialize distributed systems
    await orchestrator.initialize_distributed_systems()
    logger.info("Distributed systems initialized on startup")

class SegmentCache:
    """Event-driven segment deduplication cache for streaming transcription"""
    
    def __init__(self, ttl_seconds: int = 30):
        self.cache: Dict[str, float] = {}
        self.ttl_seconds = ttl_seconds
    
    def get_segment_hash(self, text: str, timestamp: float) -> str:
        """Generate hash for segment based on text content only"""
        content = text.strip().lower()
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_duplicate(self, text: str, timestamp: float) -> bool:
        """Check if segment is duplicate using content-based hashing"""
        segment_hash = self.get_segment_hash(text, timestamp)
        
        # Clean expired entries
        self._cleanup_expired()
        
        if segment_hash in self.cache:
            logger.debug(f"Duplicate segment detected: {text[:50]}...")
            return True
        
        # Add to cache
        self.cache[segment_hash] = timestamp
        return False
    
    def has_segment(self, text: str, timestamp: float) -> bool:
        """Check if segment was already processed"""
        self._cleanup_expired()
        segment_hash = self.get_segment_hash(text, timestamp)
        return segment_hash in self.cache
    
    def add_segment(self, text: str, timestamp: float):
        """Add segment to cache"""
        segment_hash = self.get_segment_hash(text, timestamp)
        self.cache[segment_hash] = timestamp
    
    def _cleanup_expired(self):
        """Remove expired segments"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.cache.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def clear(self):
        """Clear all segments"""
        self.cache.clear()

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
        # self.is_paused = False # New state
        # self.resume_timer_task: Optional[asyncio.Task] = None # To hold the resume timer


        # Distributed event bus for cross-service coordination
        self.event_bus: Optional[DistributedEventBus] = None
        self.service_coordinator: Optional[ServiceCoordinator] = None
        
        # Event-driven segment deduplication
        self.segment_cache = SegmentCache()
        
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
        """Ultra-fast processing using distributed event bus for fire-and-forget execution"""
        try:
            if not self.event_bus:
                logger.warning(f"Session {self.session_id}: Event bus not initialized, skipping ultra-fast processing")
                return
                
            # Create service event for immediate processing
            event = ServiceEvent(
                event_type="ultra_fast_process",
                session_id=self.session_id,
                service_id="orchestrator",
                timestamp=time.time(),
                data={"text": text}
            )
            await self.event_bus.publish("maestro:global", event)
        except Exception as e:
            logger.error(f"Error in ultra-fast processing for session {self.session_id}: {e}")

class VoiceStreamOrchestrator:
    """
    Central coordinator for all voice processing streams
    """
    def __init__(self):
        self.sessions: Dict[str, StreamSession] = {}
        self.event_bus: Optional[DistributedEventBus] = None
        self.service_coordinator: Optional[ServiceCoordinator] = None
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
                from plugins.base_plugin import PluginConfig
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
                from plugins.base_plugin import PluginConfig
                speaker_config = PluginConfig(
                    enabled=True,
                    priority=2,
                    max_workers=config.PLUGIN_MAX_WORKERS,
                    timeout=config.PLUGIN_TIMEOUT
                )
                speaker_plugin = SpeakerPlugin(speaker_config)
                self.plugin_manager.register_plugin(speaker_plugin)
                logger.info("Speaker plugin registered")
            
            # Register interrupt plugin (always enabled for voice activity detection)
            if INTERRUPT_PLUGIN_AVAILABLE:
                from plugins.base_plugin import PluginConfig
                interrupt_config = PluginConfig(
                    enabled=True,
                    priority=0,  # Highest priority for interrupt detection
                    max_workers=config.PLUGIN_MAX_WORKERS,
                    timeout=config.PLUGIN_TIMEOUT
                )
                interrupt_plugin = InterruptPlugin(interrupt_config)
                interrupt_plugin.orchestrator = self  # Pass orchestrator reference
                self.plugin_manager.register_plugin(interrupt_plugin)
                logger.info(f"✅ Interrupt plugin registered with orchestrator reference: {id(self)}")
                logger.info(f"🔍 [INTERRUPT_DEBUG] InterruptPlugin.orchestrator = {id(interrupt_plugin.orchestrator)}")
            else:
                logger.warning("InterruptPlugin not available - automatic interruption disabled")
            
            
            # Start plugin manager
            asyncio.create_task(self.plugin_manager.start_all())
            logger.info("Plugins initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing plugins: {e}")
    
    async def initialize_distributed_systems(self):
        """Initialize distributed event bus and service coordination"""
        try:
            # Initialize distributed event bus
            self.event_bus = DistributedEventBus("orchestrator")
            await self.event_bus.initialize()
            
            # Initialize service coordinator
            self.service_coordinator = ServiceCoordinator("orchestrator")
            await self.service_coordinator.initialize(self.event_bus)
            
            logger.info("✅ Distributed systems initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize distributed systems: {e}")
            raise
    
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
        
        # Share event bus and service coordinator with session
        session.event_bus = self.event_bus
        session.service_coordinator = self.service_coordinator
        logger.debug(f"Event bus and service coordinator assigned to session {session_id}")
    

        # Establish WhisperLive connection with proper headers
        logger.debug(f"Establishing WhisperLive connection for session {session_id}")
        success = await self._connect_to_whisper(session)
        if not success:
            logger.error(f"Failed to establish WhisperLive connection for session {session_id}")
            raise ConnectionError(f"Failed to connect to WhisperLive after {session.max_retries} attempts")
            
        logger.debug(f"WhisperLive connection established for session {session_id}")
        
        # Connect InterruptPlugin to session's event bus for real-time interrupt detection
        if INTERRUPT_PLUGIN_AVAILABLE:
            interrupt_plugin = None
            for plugin in self.plugin_manager.plugins.values():
                if isinstance(plugin, InterruptPlugin):
                    interrupt_plugin = plugin
                    break
            
            if interrupt_plugin and session.event_bus:
                # Register InterruptPlugin event handlers with distributed event bus
                await session.event_bus.subscribe(f"maestro:session:{session_id}", interrupt_plugin._handle_audio_monitor)
                await session.event_bus.subscribe(f"maestro:session:{session_id}", interrupt_plugin._handle_voice_during_tts)
                await session.event_bus.subscribe(f"maestro:interrupt", interrupt_plugin._handle_interrupted)
                await session.event_bus.subscribe(f"maestro:session:{session_id}", interrupt_plugin._handle_processing_complete)
                logger.info(f"🛑 InterruptPlugin connected to session {session_id} distributed event bus")
            else:
                logger.warning(f"⚠️ InterruptPlugin not found or event bus not initialized for session {session_id}")
        else:
            logger.warning(f"⚠️ InterruptPlugin not available for session {session_id}")
        
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
                    "max_clients": config.WHISPER_MAX_CLIENTS,
                    "max_connection_time": config.WHISPER_MAX_CONNECTION_TIME,
                    "send_last_n_segments": config.WHISPER_SEND_LAST_N_SEGMENTS,
                    "no_speech_thresh": config.NO_SPEECH_THRESHOLD,
                    "clip_audio": config.WHISPER_CLIP_AUDIO,
                    "same_output_threshold": config.WHISPER_SAME_OUTPUT_THRESHOLD
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
        logger.info(f"📨 [INTERRUPT_DEBUG] _handle_whisper_messages STARTED for session {session.session_id}")
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
                                logger.info(f"📝 [INTERRUPT_DEBUG] Received transcript segments from WhisperLive: {data['segments']}")
                                logger.info(f"📝 [INTERRUPT_DEBUG] Session state when segments received: is_processing={session.is_processing}, tts_active={session.tts_active}")
                                with open("/tmp/whisper_transcripts.log", "a") as f:
                                    f.write(json.dumps(data['segments']) + "\n")
                                logger.debug(f"Session {session.session_id}: Processing transcript segments")
                                if session.frontend_ws:                                                       
                                  await session.frontend_ws.send_text(json.dumps({                          
                                    "type": "segments",                              
                                    "segments": data["segments"]                     
                                    }))  
                                logger.info(f"📝 [INTERRUPT_DEBUG] About to call _process_transcript_segments")
                                await self._process_transcript_segments(session, data["segments"])
                                logger.info(f"📝 [INTERRUPT_DEBUG] _process_transcript_segments completed")
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
        logger.debug(f"🔥 [AUDIO_DEBUG] send_audio_to_whisper CALLED: session={session_id}, data_len={len(audio_data)}")
        
        if session_id not in self.sessions:
            logger.warning(f"❌ Session {session_id} not found")
            return False
            
        session = self.sessions[session_id]
        logger.debug(f"🔥 [AUDIO_DEBUG] Session found: tts_active={session.tts_active}, is_processing={session.is_processing}")
        
        # Check connection health
        if not session.whisper_connected or not session.whisper_ws or not self._is_websocket_connected(session.whisper_ws):
            logger.warning(f"🔄 WhisperLive not connected for session {session_id}, attempting reconnect")
            success = await self._connect_to_whisper(session)
            if not success:
                logger.error(f"❌ Failed to reconnect WhisperLive for session {session_id}")
                return False
                
        try:
            # DETAILED LOGGING FOR AUDIO DATA FORMAT DEBUGGING
            # logger.info(f"📤 Sending {len(audio_data)} bytes of audio to WhisperLive for session {session_id}")
            # logger.debug(f"🔍 Audio data type: {type(audio_data)}")
            # logger.debug(f"🔍 Audio data repr: {repr(audio_data)}")
            # logger.debug(f"🔍 Audio data length: {len(audio_data)}")
            
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
            # logger.info(f"✅ Successfully sent audio to WhisperLive for session {session_id}")
            
            # Emit plugin events for audio data
            logger.debug(f"🔥 [AUDIO_DEBUG] Starting audio processing for session {session_id}")
            try:
                import numpy as np
                
                # CRITICAL BINARY DEBUG: Log raw WebSocket data for comparison with terminal client
                logger.info(f"🔥 [BINARY_DEBUG] RAW WEBSOCKET DATA: {len(audio_data)} bytes, type={type(audio_data)}")
                if len(audio_data) >= 16:
                    logger.info(f"🔥 [BINARY_DEBUG] First 16 bytes (hex): {audio_data[:16].hex()}")
                    # Try to interpret as Float32 like terminal client does
                    import struct
                    try:
                        first_samples = struct.unpack('<ffff', audio_data[:16])
                        logger.info(f"🔥 [BINARY_DEBUG] First 4 samples as Float32: {first_samples}")
                    except struct.error as e:
                        logger.error(f"🔥 [BINARY_DEBUG] Cannot unpack as Float32: {e}")
                
                # Audio data is now Float32Array format (not int16)
                logger.debug(f"🔥 [AUDIO_DEBUG] Creating numpy array from {len(audio_data)} bytes")
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                logger.debug(f"🔥 [AUDIO_DEBUG] Numpy array created: shape={audio_array.shape}, dtype={audio_array.dtype}")
                
                # CRITICAL DEBUG: Log numpy array stats for comparison
                logger.info(f"🔥 [BINARY_DEBUG] NUMPY STATS: min={audio_array.min():.6f}, max={audio_array.max():.6f}, mean={audio_array.mean():.6f}, std={audio_array.std():.6f}")
                
                current_time = time.time()
                
                # Calculate audio level for monitoring
                audio_level = np.abs(audio_array).mean()
                voice_threshold = 0.015  # Match interrupt plugin threshold for consistent detection
                voice_detected = audio_level > voice_threshold
                
                logger.info(f"🔥 [AUDIO_DEBUG] CALCULATED: audio_level={audio_level:.6f}, voice_detected={voice_detected}, threshold={voice_threshold}, tts_active={session.tts_active}")
                
                # CRITICAL DEBUG: Log when audio stops being processed
                if audio_level < 0.000001:
                    logger.error(f"🚨 [CRITICAL] AUDIO LEVEL IS ZERO! session={session_id}, tts_active={session.tts_active}, is_processing={session.is_processing}")
                elif audio_level > 0.01:
                    logger.info(f"🎤 [CRITICAL] STRONG AUDIO DETECTED! level={audio_level:.6f}, tts_active={session.tts_active}")
                
            except Exception as e:
                logger.error(f"🔥 [AUDIO_DEBUG] ERROR in audio processing: {e}")
                logger.error(f"🔥 [AUDIO_DEBUG] Audio data type: {type(audio_data)}, length: {len(audio_data)}")
                if len(audio_data) > 0:
                    logger.error(f"🔥 [AUDIO_DEBUG] First 16 bytes: {audio_data[:16].hex()}")
                # Use fallback values
                audio_level = 0.0
                voice_detected = False
                current_time = time.time()
            
            # Emit standard audio_data event
            await self.plugin_manager.emit_event("audio_data", {
                "session_id": session_id,
                "audio": audio_array.tolist(),
                "timestamp": current_time,
                "user_id": session_id
            })
            
            # Emit continuous audio monitoring event via session event bus (fire-and-forget)
            if session.event_bus:
                event = ServiceEvent(
                    event_type="audio_monitor",
                    session_id=session_id,
                    service_id="orchestrator",
                    timestamp=current_time,
                    data={
                        "audio_level": float(audio_level),
                        "voice_detected": voice_detected,
                        "is_processing": session.is_processing,
                        "tts_active": session.tts_active
                    }
                )
                await session.event_bus.publish(f"maestro:session:{session_id}", event)
            
            # If voice detected during TTS, emit potential interrupt event
            logger.debug(f"🔥 [AUDIO_DEBUG] Checking interrupt conditions: voice_detected={voice_detected}, tts_active={session.tts_active}")
            if voice_detected and session.tts_active:
                logger.info(f"🗣️ [INTERRUPT_DEBUG] Voice detected during TTS for session {session_id}, audio level: {audio_level}")
                logger.info(f"🗣️ [INTERRUPT_DEBUG] Session state: is_processing={session.is_processing}, tts_active={session.tts_active}")
                logger.info(f"🔥 [AUDIO_DEBUG] EMITTING voice_during_tts event NOW!")
                if session.event_bus:
                    event = ServiceEvent(
                        event_type="voice_during_tts",
                        session_id=session_id,
                        service_id="orchestrator",
                        timestamp=current_time,
                        data={
                            "audio_level": float(audio_level)
                        }
                    )
                    await session.event_bus.publish(f"maestro:session:{session_id}", event)
                logger.info(f"🗣️ [INTERRUPT_DEBUG] voice_during_tts event emitted")
            elif voice_detected:
                logger.info(f"🔥 [AUDIO_DEBUG] Voice detected but NOT during TTS: tts_active={session.tts_active}")
            elif session.tts_active:
                logger.debug(f"🔥 [AUDIO_DEBUG] TTS active but no voice: level={audio_level:.6f} <= {voice_threshold}")
            
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
        logger.info(f"🎯 [INTERRUPT_DEBUG] Processing transcript segments for session {session.session_id}: {segments}")
        logger.info(f"🔍 [INTERRUPT_DEBUG] Session state: is_processing={session.is_processing}, tts_active={session.tts_active}, processing_text='{session.processing_text}'")
        
        # Find completed segments with event-driven deduplication
        completed_texts = []
        current_transcript_parts = []
        current_time = time.time()
        
        for segment in segments:
            if segment.get("completed") and segment.get("text"):
                text = segment["text"].strip()
                if text:
                    # Event-driven deduplication - check if duplicate
                    if not session.segment_cache.is_duplicate(text, current_time):
                        completed_texts.append(text)
                        logger.info(f"✅ [INTERRUPT_DEBUG] New completed text found: {text}")
                        
                        # Emit segment processed event via distributed event bus (fire-and-forget)
                        if session.event_bus:
                            event = ServiceEvent(
                                event_type="segment_processed",
                                session_id=session.session_id,
                                service_id="orchestrator",
                                timestamp=current_time,
                                data={
                                    "text": text,
                                    "segment_hash": session.segment_cache.get_segment_hash(text, current_time)
                                }
                            )
                            await session.event_bus.publish(f"maestro:session:{session.session_id}", event)
                    else:
                        logger.debug(f"⚠️ [INTERRUPT_DEBUG] Duplicate segment ignored: {text}")
            else:
                # Incomplete segment for live transcript display
                if segment.get("text"):
                    current_transcript_parts.append(segment["text"])
        
        # Send live transcript to frontend
        current_transcript = " ".join(current_transcript_parts).strip()
        logger.info(f"📝 [INTERRUPT_DEBUG] Current live transcript: {current_transcript}")
        
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
                "timestamp": current_time
            })
        
        # Process completed sentences
        logger.info(f"🔍 [INTERRUPT_DEBUG] Found {len(completed_texts)} completed texts to process")
        for completed_text in completed_texts:
            # This logic is now much simpler
            if self._is_sentence_complete(completed_text):
                session.stt_end_time = time.time()
                logger.info(f"📋 Session {session.session_id}: Processing complete sentence: {completed_text}")
                
                await self.plugin_manager.emit_event("transcription_complete", {
                    "session_id": session.session_id,
                    "text": completed_text,
                    "timestamp": time.time(),
                    "user_id": session.session_id
                })
                
                await self._process_complete_sentence(session, completed_text)
            else:
                logger.debug(f"Sentence not complete, skipping: {completed_text}")


                        
            # ALLOW transcript processing during TTS for interrupt detection
            # Previously blocked, but now needed for human-natural interruption behavior
            if session.tts_active:
                logger.info(f"🎤 [INTERRUPT_DEBUG] Session {session.session_id}: Processing transcript during TTS for interrupt: {completed_text}")
                # Continue processing - don't block

            if self._is_sentence_complete(completed_text):
                session.stt_end_time = time.time()
                logger.info(f"📋 [INTERRUPT_DEBUG] Session {session.session_id}: Processing complete sentence: {completed_text}")
                
                # Emit plugin events for completed transcription
                await self.plugin_manager.emit_event("transcription_complete", {
                    "session_id": session.session_id,
                    "text": completed_text,
                    "timestamp": current_time,
                    "user_id": session.session_id  # Use session_id as user_id for now
                })
                
                await self._process_complete_sentence(session, completed_text)
            else:
                logger.info(f"📋 [INTERRUPT_DEBUG] Sentence not complete, skipping: {completed_text}")
                
    def _is_sentence_complete(self, text: str) -> bool:
        """Simple sentence completion check"""
        if len(text.split()) < 3:
            return False
        return text.strip().endswith(('.', '!', '?'))
        
    async def _process_complete_sentence(self, session: StreamSession, text: str):
        """Process a complete sentence through LLM and TTS pipeline with proper interruption"""
        logger.info(f"🚀 [INTERRUPT_DEBUG] _process_complete_sentence CALLED for session {session.session_id}")
        logger.info(f"🚀 [INTERRUPT_DEBUG] Input text: '{text}'")
        logger.info(f"🚀 [INTERRUPT_DEBUG] Session state: is_processing={session.is_processing}, tts_active={session.tts_active}")
        
        # IMMEDIATE INTERRUPT: Only interrupt for REAL new speech, not trailing silence
        if session.is_processing or session.tts_active:
            # Check if this might be trailing audio from the same utterance
            current_time = time.time()
            time_since_last_processing = current_time - getattr(session, 'last_processing_start', 0)
            time_since_tts_start = current_time - getattr(session, 'tts_start_time', 0)
            
            # Protect against trailing audio for 5 seconds after processing/TTS starts
            min_time_since_start = min(time_since_last_processing, time_since_tts_start) if hasattr(session, 'tts_start_time') else time_since_last_processing
            
            if min_time_since_start < 5.0:
                logger.info(f"🎤 [INTERRUPT_DEBUG] Ignoring potential trailing audio (only {min_time_since_start:.2f}s since processing/TTS started): '{text}'")
                return
            
            logger.info(f"🛑 [IMMEDIATE_INTERRUPT] New user input detected during processing/TTS - ABORTING EVERYTHING")
            logger.info(f"🛑 [IMMEDIATE_INTERRUPT] Previous: processing={session.is_processing}, tts_active={session.tts_active}")
            await self.interrupt_session(session.session_id)
            logger.info(f"🛑 [IMMEDIATE_INTERRUPT] Pipeline cleared, now processing new input: '{text}'")
            
        logger.info(f"🚀 [INTERRUPT_DEBUG] Setting session.is_processing=True")
        session.is_processing = True
        session.processing_text = text  # Store the text being processed
        session.last_processing_start = time.time()  # Track when processing started
        session.total_requests += 1
        
        # Report state transition to service coordinator
        if session.service_coordinator:
            await session.service_coordinator.report_state(
                session_id=session.session_id,
                service_type=ServiceType.ORCHESTRATOR,
                state=ServiceState.THINKING,
                metadata={"text": text, "request_count": session.total_requests}
            )
        logger.info(f"🚀 [INTERRUPT_DEBUG] Session state updated: is_processing={session.is_processing}, processing_text='{session.processing_text}'")
        
        try:
            # Clear any previous TTS abort signal and reset sequence
            session.tts_abort_event.clear()
            session.tts_sequence_number = 0
            session.tts_queue.clear()
            
            # Emit state transition event via distributed event bus (fire-and-forget)
            if session.event_bus:
                event = ServiceEvent(
                    event_type="state_transition",
                    session_id=session.session_id,
                    service_id="orchestrator",
                    timestamp=time.time(),
                    data={
                        "from_state": "idle",
                        "to_state": "processing",
                        "text": text
                    }
                )
                await session.event_bus.publish(f"maestro:session:{session.session_id}", event)
            
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
            
            # Emit state transition event
            current_time = time.time()
            if session.tts_abort_event.is_set():
                # Interrupted processing
                session.tts_abort_event.clear()
                logger.info(f"Session {session.session_id}: Cleared abort event after interruption")
                
                if session.event_bus:
                    event = ServiceEvent(
                        event_type="state_transition",
                        session_id=session.session_id,
                        service_id="orchestrator",
                        timestamp=current_time,
                        data={
                            "from_state": "processing",
                            "to_state": "interrupted",
                            "text": text
                        }
                    )
                    await session.event_bus.publish(f"maestro:session:{session.session_id}", event)
            else:
                # Completed processing normally
                if session.event_bus:
                    event = ServiceEvent(
                        event_type="state_transition",
                        session_id=session.session_id,
                        service_id="orchestrator",
                        timestamp=current_time,
                        data={
                            "from_state": "processing",
                            "to_state": "idle",
                            "text": text
                        }
                    )
                    await session.event_bus.publish(f"maestro:session:{session.session_id}", event)
                
                # Report state transition to service coordinator
                if session.service_coordinator:
                    await session.service_coordinator.report_state(
                        session_id=session.session_id,
                        service_type=ServiceType.ORCHESTRATOR,
                        state=ServiceState.IDLE,
                        metadata={"text": text, "completed": True}
                    )
                
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
                        session.tts_start_time = time.time()  # Track when TTS started for trailing audio protection
                        logger.info(f"🔍 [INTERRUPT_DEBUG] 🔊 TTS ACTIVATED for session {session.session_id}, sequence {sequence}")
                        
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
                        logger.info(f"🔍 [INTERRUPT_DEBUG] 🔇 TTS DEACTIVATED for session {session.session_id}, sequence {sequence}")
                    
                    # Small delay to prevent overwhelming the system while maintaining low latency
                    # Also check for interruption during delay
                    try:
                        await asyncio.wait_for(asyncio.sleep(0.1), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
                    
                # logger.info(f"Session {session.session_id}: TTS queue processing complete")
                
                # Clear segment cache when TTS completes to allow new conversation turns
                session.segment_cache.clear()
                
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
            session.tts_start_time = time.time()
            logger.info(f"🔍 [INTERRUPT_DEBUG] 🔊 TTS ACTIVATED (direct) for session {session.session_id}")
            
            # Report TTS state to service coordinator
            if session.service_coordinator:
                await session.service_coordinator.report_state(
                    session_id=session.session_id,
                    service_type=ServiceType.MOUTH,
                    state=ServiceState.SPEAKING,
                    metadata={"sentence": sentence, "sequence": sequence}
                )
            
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
            logger.info(f"🔍 [INTERRUPT_DEBUG] 🔇 TTS DEACTIVATED (direct) for session {session.session_id}")
            
            # Report TTS completion to service coordinator
            if session.service_coordinator:
                await session.service_coordinator.report_state(
                    session_id=session.session_id,
                    service_type=ServiceType.MOUTH,
                    state=ServiceState.IDLE,
                    metadata={"sentence": sentence, "sequence": sequence, "completed": True}
                )

    async def _send_to_frontend(self, session: StreamSession, message: dict):
        """Send message to frontend WebSocket"""
        if session.frontend_ws:
            try:
                # logger.info(f"📤 Sending to frontend: {message}")
                await session.frontend_ws.send_text(json.dumps(message))
                # logger.info(f"✅ Successfully sent to frontend: {message.get('type', 'unknown')}")
            except Exception as e:
                logger.error(f"❌ Failed to send to frontend: {e}")
                
    async def interrupt_session(self, session_id: str) -> bool:
        """Interrupt TTS and processing for a session using coordinated workflow."""
        logger.info(f"🛑 [INTERRUPT_DEBUG] INTERRUPT REQUEST RECEIVED for session {session_id}")
        
        if session_id not in self.sessions:
            logger.error(f"🛑 [INTERRUPT_DEBUG] Session {session_id} not found in sessions")
            return False
            
        session = self.sessions[session_id]
        
        # Use coordinated interrupt workflow if service coordinator is available
        if session.service_coordinator:
            logger.info(f"🛑 [INTERRUPT_DEBUG] Using coordinated interrupt workflow")
            try:
                await session.service_coordinator.trigger_interrupt(session_id, "user_interrupt")
                logger.info(f"🛑 [INTERRUPT_DEBUG] Coordinated interrupt initiated")
            except Exception as e:
                logger.error(f"🛑 [INTERRUPT_DEBUG] Coordinated interrupt failed, falling back to local interrupt: {e}")
        else:
            logger.warning(f"🛑 [INTERRUPT_DEBUG] No service coordinator, using local interrupt only")
        
        # Log current session state before interruption
        logger.info(f"🛑 [INTERRUPT_DEBUG] Session state BEFORE interrupt:")
        logger.info(f"🛑 [INTERRUPT_DEBUG]   - is_processing: {session.is_processing}")
        logger.info(f"🛑 [INTERRUPT_DEBUG]   - tts_active: {session.tts_active}")
        logger.info(f"🛑 [INTERRUPT_DEBUG]   - processing_text: '{session.processing_text}'")
        logger.info(f"🛑 [INTERRUPT_DEBUG]   - tts_queue length: {len(session.tts_queue)}")
        logger.info(f"🛑 [INTERRUPT_DEBUG]   - tts_abort_event.is_set(): {session.tts_abort_event.is_set()}")
        
        # 1. Signal abort to all async operations
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 1: Setting tts_abort_event")
        session.tts_abort_event.set()
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 1: tts_abort_event set to {session.tts_abort_event.is_set()}")
        
        # 2. Cancel any running TTS task
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 2: Checking TTS task cancellation")
        if session.tts_task and not session.tts_task.done():
            logger.info(f"🛑 [INTERRUPT_DEBUG] Step 2: Cancelling active TTS task")
            session.tts_task.cancel()
            try:
                await session.tts_task
            except asyncio.CancelledError:
                logger.info(f"🛑 [INTERRUPT_DEBUG] Step 2: TTS task cancelled successfully")
                pass # Expected
        else:
            logger.info(f"🛑 [INTERRUPT_DEBUG] Step 2: No active TTS task to cancel")
                
        # 3. Reset processing state
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 3: Resetting processing state")
        was_processing = session.is_processing
        was_tts_active = session.tts_active
        session.is_processing = False
        session.tts_active = False
        logger.info(f"🔍 [INTERRUPT_DEBUG] 🔇 TTS DEACTIVATED (interrupt) for session {session_id}")
        session.processing_text = None  # Clear the processing text on interrupt
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 3: State reset - was_processing={was_processing}, was_tts_active={was_tts_active}")
        
        # 4. Clear the TTS queue to prevent pending sentences from playing
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 4: Clearing TTS queue (had {len(session.tts_queue)} items)")
        session.tts_queue.clear()
        session.tts_sequence_number = 0
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 4: TTS queue cleared, sequence reset to 0")
        
        # 4.5. Cancel any active HTTP requests to TTS service
        if hasattr(session, 'active_http_clients'):
            active_clients = list(session.active_http_clients)
            logger.info(f"🛑 [INTERRUPT_DEBUG] Step 4.5: Cancelling {len(active_clients)} active HTTP clients")
            for client in active_clients:
                try:
                    await client.aclose()
                    session.active_http_clients.discard(client)
                    logger.info(f"🛑 [INTERRUPT_DEBUG] Step 4.5: HTTP client cancelled successfully")
                except Exception as e:
                    logger.warning(f"🛑 [INTERRUPT_DEBUG] Step 4.5: Error closing HTTP client during interrupt: {e}")
            logger.info(f"🛑 [INTERRUPT_DEBUG] Step 4.5: All active HTTP clients cancelled")
        else:
            logger.info(f"🛑 [INTERRUPT_DEBUG] Step 4.5: No active HTTP clients to cancel")
        
        # 5. Clear segment cache to prevent duplicate processing of interrupted text
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 5: Clearing segment cache")
        session.segment_cache.clear()
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 5: Segment cache cleared")
        
        # 6. Emit interrupt event via session event bus (fire-and-forget)
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 6: Emitting interrupt events")
        current_time = time.time()
        if session.event_bus:
            event = ServiceEvent(
                event_type="interrupt_triggered",
                session_id=session_id,
                service_id="orchestrator",
                timestamp=current_time,
                data={
                    "was_processing": was_processing,
                    "was_tts_active": was_tts_active
                }
            )
            await session.event_bus.publish(f"maestro:interrupt", event)
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 6: Interrupt event emitted via session event bus")
        
        # 7. Keep WhisperLive connection open - do NOT disconnect during interrupt
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 7: Keeping WhisperLive connection open for continuous speech processing")
        # REMOVED: CLIENT_DISCONNECT message that was closing WhisperLive connection
        # This allows interrupted speech to continue being processed immediately

        # 8. Emit plugin events for interrupt
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 8: Emitting plugin interrupt events")
        await self.plugin_manager.emit_event("interrupted", {
            "session_id": session_id,
            "was_processing": was_processing,
            "was_tts_active": was_tts_active,
            "timestamp": current_time
        })
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 8: Plugin interrupt events emitted")

        # 9. Notify the frontend
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 9: Notifying frontend")
        await self._send_to_frontend(session, {
            "type": "interrupted"
        })
        logger.info(f"🛑 [INTERRUPT_DEBUG] Step 9: Frontend notification sent")
        
        # Log final session state after interruption
        logger.info(f"🛑 [INTERRUPT_DEBUG] Session state AFTER interrupt:")
        logger.info(f"🛑 [INTERRUPT_DEBUG]   - is_processing: {session.is_processing}")
        logger.info(f"🛑 [INTERRUPT_DEBUG]   - tts_active: {session.tts_active}")
        logger.info(f"🛑 [INTERRUPT_DEBUG]   - processing_text: '{session.processing_text}'")
        logger.info(f"🛑 [INTERRUPT_DEBUG]   - tts_queue length: {len(session.tts_queue)}")
        logger.info(f"🛑 [INTERRUPT_DEBUG]   - tts_abort_event.is_set(): {session.tts_abort_event.is_set()}")
        
        logger.info(f"🛑 [INTERRUPT_DEBUG] INTERRUPT COMPLETE for session {session_id}")
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


    # async def _schedule_resume(self, session: StreamSession, delay: float):
    #     """If no further user input, resume TTS after a delay."""
    #     try:
    #         await asyncio.sleep(delay)
    #         # Only resume if we are still in a paused state
    #         if session.is_paused:
    #             logger.info(f"User did not commit to interrupt. Resuming TTS for session {session.session_id}")
    #             session.is_paused = False
    #             await self._send_to_frontend(session, {"type": "resume_tts"})
    #     except asyncio.CancelledError:
    #         # This is expected if the user commits to the interrupt
    #         logger.info(f"Resume timer cancelled for session {session.session_id}")

    # # ADD THIS NEW METHOD
    # async def handle_soft_interrupt(self, session: StreamSession, event_data: Dict[str, Any]):

    #     """Handles the initial detection of voice during TTS."""
    #     # Don't do anything if TTS isn't active or if we're already paused
    #     if not session.tts_active or session.is_paused:
    #         return

    #     logger.info(f"Pausing TTS for session {session.session_id} to listen to user.")
    #     session.is_paused = True

    #     # 1. Tell the frontend to pause its audio playback
    #     await self._send_to_frontend(session, {"type": "pause_tts"})

    #     # 2. Cancel any pre-existing resume timer
    #     if session.resume_timer_task and not session.resume_timer_task.done():
    #         session.resume_timer_task.cancel()

    #     # 3. Start a new timer. If the user doesn't say a complete sentence
    #     #    within this time, we will automatically resume.
    #     session.resume_timer_task = asyncio.create_task(
    #         self._schedule_resume(session, delay=2.0)  # 2-second listening window
    #     )

# Initialize global orchestrator
orchestrator = VoiceStreamOrchestrator()


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
        async def handle_ultra_fast_process(event: ServiceEvent):
            """Handle ultra-fast processing events"""
            try:
                text = event.data["text"]
                event_session_id = event.session_id
                
                # Skip if not for this session
                if event_session_id != session.session_id:
                    return
                    
                # Process immediately without waiting
                await session._process_complete_sentence(session, text)
                
            except Exception as e:
                logger.error(f"Error in ultra-fast processing handler: {e}")
        
        # Register the handler
        if session.event_bus:
            await session.event_bus.subscribe("maestro:global", handle_ultra_fast_process)
        
        
        # Send ready signal
        await websocket.send_text(json.dumps({
            "type": "ready",
            "session_id": session_id,
            "mode": "ultra_fast"
        }))
        
        # Report initial listening state to service coordinator
        if session.service_coordinator:
            await session.service_coordinator.report_state(
                session_id=session_id,
                service_type=ServiceType.ORCHESTRATOR,
                state=ServiceState.LISTENING,
                metadata={"ready": True, "mode": "ultra_fast"}
            )
        
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
                        
                        logger.debug(f"🔥 [AUDIO_DEBUG] WebSocket calling send_audio_to_whisper for session {session_id}")
                        result = await orchestrator.send_audio_to_whisper(session_id, audio_bytes)
                        logger.debug(f"🔥 [AUDIO_DEBUG] send_audio_to_whisper returned: {result}")
                        
                    elif "text" in message:
                        data = json.loads(message["text"])
                        
                        if data.get("type") == "interrupt":
                            await orchestrator.interrupt_session(session_id)
                            
                        elif data.get("type") == "end_audio":
                            logger.info(f"🔚 End of audio signal received for session {session_id}")
                           
                        # Forward end signal to WhisperLive as before
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