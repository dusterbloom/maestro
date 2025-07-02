import asyncio
import json
import logging
import os
import time
from typing import Optional
import numpy as np
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Orchestrator", version="1.0.0")

class VoiceOrchestrator:
    def __init__(self):
        self.whisper_url = os.getenv("WHISPER_URL", "http://whisper-live:9090")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
        self.tts_url = os.getenv("TTS_URL", "http://kokoro:8880/v1")
        self.memory_enabled = os.getenv("MEMORY_ENABLED", "false").lower() == "true"
        self.amem_url = os.getenv("AMEM_URL", "http://a-mem:8001")
        
        # WebSocket connection to WhisperLive
        self.whisper_ws = None
        
    async def connect_to_whisper(self):
        """Connect to WhisperLive WebSocket service"""
        try:
            # Convert HTTP URL to WebSocket URL
            ws_url = self.whisper_url.replace("http://", "ws://")
            self.whisper_ws = await websockets.connect(ws_url)
            logger.info("Connected to WhisperLive")
        except Exception as e:
            logger.error(f"Failed to connect to WhisperLive: {e}")
            self.whisper_ws = None
    
    async def disconnect_from_whisper(self):
        """Disconnect from WhisperLive WebSocket"""
        if self.whisper_ws:
            await self.whisper_ws.close()
            self.whisper_ws = None
    
    async def process_audio(self, audio_data: bytes, session_id: str = "default") -> bytes:
        """Process audio through the complete pipeline"""
        start_time = time.time()
        
        try:
            # 1. Transcribe audio using WhisperLive
            transcript = await self.transcribe(audio_data)
            logger.info(f"Transcript: {transcript}")
            
            if not transcript.strip():
                return b""  # Return empty audio for silence
            
            # 2. Retrieve context if memory enabled
            context = ""
            if self.memory_enabled:
                try:
                    context = await self.retrieve_context(transcript, session_id)
                except Exception as e:
                    logger.warning(f"Memory retrieval failed: {e}")
            
            # 3. Generate response using Ollama
            response = await self.generate_response(transcript, context)
            logger.info(f"LLM Response: {response}")
            
            # 4. Store interaction if memory enabled
            if self.memory_enabled:
                try:
                    await self.store_interaction(transcript, response, session_id)
                except Exception as e:
                    logger.warning(f"Memory storage failed: {e}")
            
            # 5. Synthesize speech using Kokoro
            audio_response = await self.synthesize(response)
            
            total_time = (time.time() - start_time) * 1000
            logger.info(f"Total pipeline latency: {total_time:.2f}ms")
            
            return audio_response
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return b""  # Return empty audio on error
    
    async def transcribe(self, audio_data: bytes) -> str:
        """Convert audio to text using WhisperLive"""
        if not self.whisper_ws:
            await self.connect_to_whisper()
            
        if not self.whisper_ws:
            logger.error("WhisperLive not available")
            return ""
        
        try:
            # Convert audio bytes to numpy float32 array (WhisperLive format)
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            
            # Send to WhisperLive WebSocket (binary frame)
            await self.whisper_ws.send(audio_np.tobytes())
            
            # Receive JSON response with transcription segments
            response = await self.whisper_ws.recv()
            data = json.loads(response)
            
            # Extract text from segments: {"uid": client_id, "segments": [...]}
            segments = data.get("segments", [])
            return " ".join([seg.get("text", "") for seg in segments])
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            await self.disconnect_from_whisper()
            return ""
    
    async def generate_response(self, text: str, context: str = "") -> str:
        """Generate response using Ollama with streaming"""
        prompt = f"{context}\n\nUser: {text}\nAssistant:" if context else f"User: {text}\nAssistant:"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": os.getenv("LLM_MODEL", "gemma3n:latest"),
                        "prompt": prompt,
                        "stream": True  # Enable real-time streaming
                    }
                )
                response.raise_for_status()
                
                # Handle streaming NDJSON response
                full_response = ""
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        if chunk.get("response"):
                            full_response += chunk["response"]
                            # Optionally yield partial responses for real-time streaming
                        if chunk.get("done"):
                            break
                return full_response.strip()
                
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "I'm sorry, I couldn't process your request right now."
    
    async def synthesize(self, text: str) -> bytes:
        """Convert text to speech using Kokoro TTS"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.tts_url}/v1/audio/speech",
                    json={
                        "input": text,
                        "voice": os.getenv("TTS_VOICE", "af_bella"),
                        "response_format": "wav"
                    }
                )
                response.raise_for_status()
                return response.content
                
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return b""
    
    async def retrieve_context(self, query: str, session_id: str) -> str:
        """Retrieve conversation context from A-MEM"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.amem_url}/retrieve",
                    json={"query": query, "session_id": session_id, "k": 5}
                )
                response.raise_for_status()
                return response.json().get("context", "")
        except Exception as e:
            logger.error(f"Context retrieval error: {e}")
            return ""
    
    async def store_interaction(self, user_input: str, ai_response: str, session_id: str):
        """Store conversation in A-MEM"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{self.amem_url}/store",
                    json={
                        "user_input": user_input,
                        "ai_response": ai_response,
                        "session_id": session_id
                    }
                )
        except Exception as e:
            logger.error(f"Context storage error: {e}")

# Initialize orchestrator
orchestrator = VoiceOrchestrator()

@app.get("/health")
async def health():
    """Health check endpoint"""
    return JSONResponse({
        "status": "ok", 
        "memory_enabled": orchestrator.memory_enabled,
        "timestamp": time.time()
    })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time voice processing"""
    await websocket.accept()
    session_id = websocket.headers.get("x-session-id", f"session_{int(time.time())}")
    
    logger.info(f"WebSocket connection established for session: {session_id}")
    
    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive_bytes()
            
            if data == b"END_OF_AUDIO":
                logger.info("End of audio signal received")
                break
            
            # Process audio through pipeline
            response_audio = await orchestrator.process_audio(data, session_id)
            
            # Send audio response back to client
            if response_audio:
                await websocket.send_bytes(response_audio)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        # Clean up WhisperLive connection
        await orchestrator.disconnect_from_whisper()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Voice Orchestrator starting up...")
    logger.info(f"Memory enabled: {orchestrator.memory_enabled}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Voice Orchestrator shutting down...")
    await orchestrator.disconnect_from_whisper()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)