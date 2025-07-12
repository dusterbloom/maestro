"""
Maestro GenAI Processors Application
Main orchestrator using Google's GenAI Processors framework

This application replaces the custom FastAPI orchestrator while maintaining
all existing functionality and performance characteristics.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from genai_processors import content_api, streams
from genai_processors.core import audio_io

from maestro_processors import (
    config,
    WhisperLiveProcessor,
    OllamaStreamProcessor,
    KokoroTTSProcessor,
    VoiceActivityProcessor,
    SessionManagerProcessor,
    VoiceMetadata
)


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app for compatibility with existing frontend
app = FastAPI(title="Maestro GenAI Processors Orchestrator", version="2.0.0")

# Global session management
active_sessions: Dict[str, 'VoiceSession'] = {}
session_cleanup_task: Optional[asyncio.Task] = None


class VoiceSession:
    """Represents a complete voice conversation session with GenAI Processors chain."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_activity = time.time()
        
        # Processor instances
        self.whisper_processor: Optional[WhisperLiveProcessor] = None
        self.ollama_processor: Optional[OllamaStreamProcessor] = None
        self.kokoro_processor: Optional[KokoroTTSProcessor] = None
        self.vad_processor: Optional[VoiceActivityProcessor] = None
        self.session_processor: Optional[SessionManagerProcessor] = None
        
        # WebSocket connection to frontend
        self.frontend_ws: Optional[WebSocket] = None
        
        # Processing state
        self.processor_chain: Optional[object] = None
        self.processing_task: Optional[asyncio.Task] = None
        self.is_active = True
        
        logger.info(f"VoiceSession created: {session_id}")
    
    async def initialize_processors(self):
        """Initialize all processors for the session."""
        try:
            # Create processor instances
            self.session_processor = SessionManagerProcessor(
                session_id=self.session_id,
                max_history_length=config.MAX_CONCURRENT_SESSIONS,
                enable_memory=config.MEMORY_ENABLED
            )
            
            self.whisper_processor = WhisperLiveProcessor(
                session_id=self.session_id,
                model=config.STT_MODEL,
                use_vad=config.VAD_ENABLED,
                no_speech_threshold=config.NO_SPEECH_THRESHOLD
            )
            
            self.ollama_processor = OllamaStreamProcessor(
                session_id=self.session_id,
                model=config.LLM_MODEL,
                host=config.OLLAMA_URL,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE
            )
            
            self.kokoro_processor = KokoroTTSProcessor(
                session_id=self.session_id,
                tts_url=config.TTS_URL,
                voice=config.TTS_VOICE,
                speed=config.TTS_SPEED,
                volume=config.TTS_VOLUME
            )
            
            # VAD processor with interrupt callback
            self.vad_processor = VoiceActivityProcessor(
                session_id=self.session_id,
                energy_threshold=config.VAD_ENERGY_THRESHOLD,
                dynamic_threshold=config.VAD_DYNAMIC_THRESHOLD,
                interrupt_callback=self.handle_interruption
            )
            
            logger.info(f"Session {self.session_id}: All processors initialized")
            
        except Exception as e:
            logger.error(f"Error initializing processors for session {self.session_id}: {e}")
            raise
    
    async def create_processor_chain(self):
        """Create the GenAI Processors chain for voice conversation."""
        try:
            # Basic processor chain: STT -> Session -> LLM -> TTS
            # Note: In actual GenAI Processors, we would use the + operator to chain
            # For now, we'll simulate this with manual chaining
            
            self.processor_chain = ProcessorChain([
                self.whisper_processor,
                self.session_processor,
                self.ollama_processor,
                self.kokoro_processor
            ])
            
            # Add VAD monitoring as parallel processor
            if config.ENABLE_VAD_MONITORING:
                self.processor_chain.add_parallel_processor(self.vad_processor)
            
            logger.info(f"Session {self.session_id}: Processor chain created")
            
        except Exception as e:
            logger.error(f"Error creating processor chain: {e}")
            raise
    
    async def start_processing(self, audio_stream):
        """Start the processor chain with audio input."""
        try:
            if not self.processor_chain:
                await self.create_processor_chain()
            
            # Process audio through the chain
            self.processing_task = asyncio.create_task(
                self._process_audio_stream(audio_stream)
            )
            
            logger.info(f"Session {self.session_id}: Processing started")
            
        except Exception as e:
            logger.error(f"Error starting processing: {e}")
            raise
    
    async def _process_audio_stream(self, audio_stream):
        """Process audio through the processor chain."""
        try:
            async for output_part in self.processor_chain.process(audio_stream):
                # Send results to frontend
                await self.send_to_frontend(output_part)
                
                # Update activity
                self.last_activity = time.time()
                
        except Exception as e:
            logger.error(f"Error in audio stream processing: {e}")
    
    async def send_to_frontend(self, part: content_api.ProcessorPart):
        """Send ProcessorPart results to frontend WebSocket."""
        try:
            if not self.frontend_ws:
                return
            
            # Convert ProcessorPart to frontend message format
            message = self._convert_part_to_message(part)
            if message:
                await self.frontend_ws.send_json(message)
                
        except Exception as e:
            logger.error(f"Error sending to frontend: {e}")
    
    def _convert_part_to_message(self, part: content_api.ProcessorPart) -> Optional[Dict]:
        """Convert ProcessorPart to frontend WebSocket message."""
        try:
            if not part.metadata:
                return None
            
            content_type = part.metadata.get("content_type")
            stage = part.metadata.get("stage")
            
            # Live transcript
            if content_type == "live_transcript":
                return {
                    "type": "live_transcript",
                    "text": str(part.content),
                    "session_id": self.session_id
                }
            
            # Completed transcript
            elif content_type == VoiceMetadata.TRANSCRIPT and part.metadata.get("is_complete"):
                return {
                    "type": "transcript_complete",
                    "text": str(part.content),
                    "session_id": self.session_id,
                    "confidence": part.metadata.get("confidence", 1.0),
                    "stt_latency_ms": part.metadata.get("stt_latency_ms", 0)
                }
            
            # LLM text response
            elif content_type == VoiceMetadata.LLM_TEXT and part.metadata.get("is_complete"):
                return {
                    "type": "llm_text",
                    "text": str(part.content),
                    "session_id": self.session_id,
                    "sequence": part.metadata.get("sequence_number", 0),
                    "is_final": part.metadata.get("is_final", False),
                    "llm_latency_ms": part.metadata.get("llm_sentence_latency_ms", 0)
                }
            
            # TTS audio
            elif content_type == VoiceMetadata.TTS_AUDIO and part.metadata.get("is_complete"):
                # Convert audio to base64 for WebSocket transmission
                import base64
                audio_b64 = base64.b64encode(part.content).decode() if isinstance(part.content, bytes) else ""
                
                return {
                    "type": "tts_audio",
                    "audio_data": audio_b64,
                    "session_id": self.session_id,
                    "sequence": part.metadata.get("sequence_number", 0),
                    "format": "wav",
                    "tts_latency_ms": part.metadata.get("tts_latency_ms", 0)
                }
            
            # Processing status updates
            elif part.metadata.get("status") in ["processing", "complete", "error"]:
                return {
                    "type": "processing_status",
                    "status": part.metadata.get("status"),
                    "stage": stage,
                    "session_id": self.session_id,
                    "timestamp": part.metadata.get("timestamp", time.time())
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error converting part to message: {e}")
            return None
    
    async def handle_interruption(self):
        """Handle voice activity interruption (barge-in)."""
        try:
            logger.info(f"Session {self.session_id}: Handling interruption")
            
            # Interrupt all processors
            if self.whisper_processor:
                await self.whisper_processor.interrupt()
            if self.ollama_processor:
                await self.ollama_processor.interrupt()
            if self.kokoro_processor:
                await self.kokoro_processor.interrupt()
            if self.session_processor:
                await self.session_processor.interrupt_session()
            
            # Notify frontend
            if self.frontend_ws:
                await self.frontend_ws.send_json({
                    "type": "interruption",
                    "session_id": self.session_id,
                    "timestamp": time.time()
                })
            
            logger.info(f"Session {self.session_id}: Interruption handled")
            
        except Exception as e:
            logger.error(f"Error handling interruption: {e}")
    
    async def send_audio_data(self, audio_data: bytes) -> bool:
        """Send audio data to the processor chain."""
        try:
            if not self.whisper_processor:
                return False
            
            # Create audio ProcessorPart
            audio_part = content_api.ProcessorPart(
                content=audio_data,
                mime_type="audio/wav",
                metadata={
                    "session_id": self.session_id,
                    "content_type": VoiceMetadata.AUDIO_INPUT,
                    "timestamp": time.time(),
                    "sample_rate": config.AUDIO_SAMPLE_RATE,
                    "channels": 1
                }
            )
            
            # Process through WhisperLive processor
            # Note: This is simplified - in real implementation we'd use the full chain
            async for result in self.whisper_processor.call(streams.stream_content([audio_part])):
                await self.send_to_frontend(result)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending audio data: {e}")
            return False
    
    async def cleanup(self):
        """Clean up session resources."""
        try:
            logger.info(f"Cleaning up session {self.session_id}")
            
            # Cancel processing task
            if self.processing_task and not self.processing_task.done():
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup processors
            if self.whisper_processor:
                await self.whisper_processor._cleanup_connection()
            if self.kokoro_processor:
                await self.kokoro_processor._cleanup_client()
            if self.session_processor:
                await self.session_processor.cleanup_session()
            
            self.is_active = False
            logger.info(f"Session {self.session_id} cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
    
    def get_metrics(self) -> Dict:
        """Get comprehensive session metrics."""
        metrics = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "is_active": self.is_active,
            "session_duration": time.time() - self.created_at
        }
        
        # Add processor metrics
        if self.whisper_processor:
            metrics["whisper"] = self.whisper_processor.get_metrics()
        if self.ollama_processor:
            metrics["ollama"] = self.ollama_processor.get_metrics()
        if self.kokoro_processor:
            metrics["kokoro"] = self.kokoro_processor.get_metrics()
        if self.vad_processor:
            metrics["vad"] = self.vad_processor.get_metrics()
        if self.session_processor:
            metrics["session"] = self.session_processor.get_metrics()
        
        return metrics


class ProcessorChain:
    """Simplified processor chain implementation for demonstration."""
    
    def __init__(self, processors):
        self.processors = processors
        self.parallel_processors = []
    
    def add_parallel_processor(self, processor):
        self.parallel_processors.append(processor)
    
    async def process(self, input_stream):
        """Process input through the chain."""
        current_stream = input_stream
        
        # Process through each processor in sequence
        for processor in self.processors:
            current_stream = processor.call(current_stream)
        
        # Yield results
        async for result in current_stream:
            yield result


# FastAPI WebSocket endpoint for frontend compatibility
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice conversation - compatible with existing frontend."""
    await websocket.accept()
    session_id = f"session_{int(time.time() * 1000)}"
    
    try:
        # Create and initialize session
        session = VoiceSession(session_id)
        session.frontend_ws = websocket
        await session.initialize_processors()
        
        active_sessions[session_id] = session
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": time.time()
        })
        
        logger.info(f"WebSocket connection established for session {session_id}")
        
        # Handle incoming messages
        while True:
            try:
                # Receive data from frontend
                data = await websocket.receive()
                
                if "bytes" in data:
                    # Audio data
                    audio_data = data["bytes"]
                    success = await session.send_audio_data(audio_data)
                    if not success:
                        logger.warning(f"Failed to process audio for session {session_id}")
                
                elif "text" in data:
                    # Text message (control commands)
                    import json
                    try:
                        message = json.loads(data["text"])
                        await handle_control_message(session, message)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON message: {data['text']}")
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session {session_id}")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                break
                
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint: {e}")
    finally:
        # Cleanup session
        if session_id in active_sessions:
            await active_sessions[session_id].cleanup()
            del active_sessions[session_id]


async def handle_control_message(session: VoiceSession, message: Dict):
    """Handle control messages from frontend."""
    try:
        msg_type = message.get("type")
        
        if msg_type == "interrupt":
            await session.handle_interruption()
        elif msg_type == "get_metrics":
            metrics = session.get_metrics()
            await session.frontend_ws.send_json({
                "type": "metrics_response",
                "metrics": metrics
            })
        elif msg_type == "set_voice":
            voice = message.get("voice")
            if voice and session.kokoro_processor:
                await session.kokoro_processor.set_voice_settings(voice=voice)
        
    except Exception as e:
        logger.error(f"Error handling control message: {e}")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return JSONResponse({
        "status": "healthy",
        "timestamp": time.time(),
        "active_sessions": len(active_sessions),
        "version": "2.0.0 (GenAI Processors)"
    })


# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    metrics = {
        "timestamp": time.time(),
        "active_sessions": len(active_sessions),
        "total_sessions": len(active_sessions),  # Simplified
        "configuration": {
            "whisper_url": config.WHISPER_URL,
            "ollama_url": config.OLLAMA_URL,
            "tts_url": config.TTS_URL,
            "memory_enabled": config.MEMORY_ENABLED,
            "vad_enabled": config.VAD_ENABLED
        }
    }
    
    # Add individual session metrics
    session_metrics = {}
    for session_id, session in active_sessions.items():
        session_metrics[session_id] = session.get_metrics()
    
    metrics["sessions"] = session_metrics
    
    return JSONResponse(metrics)


# Session cleanup task
async def cleanup_inactive_sessions():
    """Background task to clean up inactive sessions."""
    while True:
        try:
            current_time = time.time()
            sessions_to_remove = []
            
            for session_id, session in active_sessions.items():
                # Check if session is expired (1 hour of inactivity)
                if current_time - session.last_activity > 3600:
                    sessions_to_remove.append(session_id)
            
            # Clean up expired sessions
            for session_id in sessions_to_remove:
                logger.info(f"Cleaning up expired session: {session_id}")
                await active_sessions[session_id].cleanup()
                del active_sessions[session_id]
            
            # Sleep for cleanup interval
            await asyncio.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            logger.error(f"Error in session cleanup task: {e}")
            await asyncio.sleep(60)  # Short sleep on error


@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    global session_cleanup_task
    
    logger.info("Starting Maestro GenAI Processors Orchestrator")
    logger.info(f"Configuration: STT={config.STT_MODEL}, LLM={config.LLM_MODEL}, TTS={config.TTS_VOICE}")
    
    # Start background cleanup task
    session_cleanup_task = asyncio.create_task(cleanup_inactive_sessions())


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info("Shutting down Maestro GenAI Processors Orchestrator")
    
    # Cancel cleanup task
    if session_cleanup_task:
        session_cleanup_task.cancel()
    
    # Clean up all active sessions
    for session_id, session in list(active_sessions.items()):
        await session.cleanup()
    
    active_sessions.clear()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=config.LOG_LEVEL.lower(),
        access_log=True
    )
