import asyncio
import base64
import json
import logging
import os
import time
import re
from typing import Optional
import httpx
import ollama
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from config import config
from services.voice_service import VoiceService
from services.memory_service import MemoryService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Orchestrator", version="2.0.0")

class VoiceOrchestrator:
    def __init__(self):
        self.voice_service = VoiceService()
        self.memory_service = MemoryService()
        self.ollama_client = ollama.AsyncClient(host=config.OLLAMA_URL)
        self.tts_client = httpx.AsyncClient(base_url=config.TTS_URL, timeout=config.TTS_TIMEOUT)
        self.active_connections = {}
        self.session_states = {}
        self.whisper_sessions = {}  # session_id -> whisper websocket

    async def handle_connection(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_states[session_id] = {"user_id": None, "status": "unidentified"}
        logger.info(f"Session {session_id} connected.")
        try:
            while True:
                data = await websocket.receive_json()
                event = data.get("event")
                if event == "audio_stream":
                    await self.handle_audio_stream(session_id, data.get("audio_data"))
                elif event == "speaker.claim":
                    await self.handle_speaker_claim(session_id, data.get("user_id"), data.get("new_name"))
        except WebSocketDisconnect:
            # Cleanup WhisperLive WebSocket connection
            if session_id in self.whisper_sessions:
                try:
                    await self.whisper_sessions[session_id].close()
                except:
                    pass
                del self.whisper_sessions[session_id]
            
            del self.active_connections[session_id]
            del self.session_states[session_id]
            logger.info(f"Session {session_id} disconnected.")

    async def get_whisper_websocket(self, session_id: str):
        """Get or create WhisperLive WebSocket connection for session"""
        if session_id not in self.whisper_sessions:
            try:
                # WhisperLive WebSocket URL - simple and direct
                whisper_ws_url = "ws://whisper-live:9090"
                
                # Connect to WhisperLive WebSocket
                websocket = await websockets.connect(whisper_ws_url, timeout=10)
                
                # Send required configuration as first message (per WhisperLive protocol)
                config_message = {
                    "uid": session_id,
                    "language": "en", 
                    "task": "transcribe",
                    "model": "tiny",
                    "use_vad": True
                }
                
                await websocket.send(json.dumps(config_message))
                logger.info(f"WhisperLive WebSocket connected for session {session_id}")
                
                # Wait for SERVER_READY confirmation
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    logger.info(f"WhisperLive response: {response}")
                except asyncio.TimeoutError:
                    logger.warning("No SERVER_READY response from WhisperLive")
                
                self.whisper_sessions[session_id] = websocket
                return websocket
                
            except Exception as e:
                logger.error(f"Failed to connect to WhisperLive: {e}")
                return None
        
        return self.whisper_sessions[session_id]

    async def transcribe_audio_websocket(self, session_id: str, audio_data: bytes) -> str:
        """Send audio to WhisperLive via WebSocket and return transcript"""
        try:
            websocket = await self.get_whisper_websocket(session_id)
            if not websocket:
                return ""
            
            # Send audio data directly to WhisperLive
            # For now, try sending the WAV data as-is to see what happens
            await websocket.send(audio_data)
            logger.info(f"Sent {len(audio_data)} bytes to WhisperLive for session {session_id}")
            
            # Listen for transcript response with longer timeout
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                logger.info(f"WhisperLive response: {response}")
                
                if isinstance(response, str):
                    message = json.loads(response)
                    
                    # Check for segments in response
                    if message.get("segments"):
                        transcript = ""
                        for segment in message["segments"]:
                            if segment.get("text"):
                                transcript += segment["text"] + " "
                        if transcript.strip():
                            logger.info(f"Extracted transcript: {transcript.strip()}")
                            return transcript.strip()
                    
                    # Check for other transcript fields
                    if message.get("text"):
                        logger.info(f"Found text field: {message['text']}")
                        return message["text"].strip()
                        
                return ""
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for WhisperLive response for session {session_id}")
                return ""
                
        except Exception as e:
            logger.error(f"WhisperLive transcription error: {e}")
            # Remove broken connection
            if session_id in self.whisper_sessions:
                try:
                    await self.whisper_sessions[session_id].close()
                except:
                    pass
                del self.whisper_sessions[session_id]
            return ""


    async def synthesize_speech(self, text: str) -> bytes:
        try:
            response = await self.tts_client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": text,
                    "voice": config.TTS_VOICE,
                    "response_format": "wav",
                    "stream": False,
                    "speed": config.TTS_SPEED,
                    "volume_multiplier": config.TTS_VOLUME
                }
            )
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return b""

    async def handle_audio_stream(self, session_id: str, audio_base64: str):
        audio_data = base64.b64decode(audio_base64)
        
        # 1. Transcribe audio
        transcript = await self.transcribe_audio_websocket(session_id, audio_data)
        if not transcript:
            logger.warning(f"No transcript generated for session {session_id}")
            return
        await self.send_event(session_id, "transcript.update", {"text": transcript})

        # 2. Speaker Identification
        embedding = await self.voice_service.get_embedding(audio_data)
        if not embedding:
            logger.warning(f"No embedding generated for session {session_id}")
            return

        current_user_id = self.session_states[session_id]["user_id"]
        profile = None

        if current_user_id:
            # User already identified in this session
            profile = await self.memory_service.get_speaker_profile(current_user_id)
        else:
            # Try to find speaker by embedding
            found_user_id = await self.memory_service.find_speaker_by_embedding(embedding)
            if found_user_id:
                profile = await self.memory_service.get_speaker_profile(found_user_id)
                self.session_states[session_id]["user_id"] = found_user_id
                self.session_states[session_id]["status"] = "identified"
                await self.send_event(session_id, "speaker.identified", {"user_id": found_user_id, "name": profile.get("name", "Unknown")})
            else:
                # New speaker, create profile
                new_user_id = await self.memory_service.create_speaker_profile(embedding)
                profile = await self.memory_service.get_speaker_profile(new_user_id)
                self.session_states[session_id]["user_id"] = new_user_id
                self.session_states[session_id]["status"] = "pending_naming"
                await self.send_event(session_id, "speaker.identified", {"user_id": new_user_id, "name": profile.get("name"), "status": "unclaimed"})
                await self.initiate_naming_conversation(session_id, new_user_id, profile.get("name"))

        # 3. LLM Processing and Response
        llm_response_text = ""
        if self.session_states[session_id]["status"] == "pending_naming":
            # LLM extracts name from transcript
            name_extraction_prompt = f"The user said: '{transcript}'. They are responding to a request for their name. Extract ONLY their name from this response. Respond with just the name, and nothing else."
            extracted_name = await self.generate_llm_response(name_extraction_prompt)
            if extracted_name and len(extracted_name.split()) <= 3: # Basic validation for name
                await self.memory_service.update_speaker_name(self.session_states[session_id]["user_id"], extracted_name)
                self.session_states[session_id]["status"] = "identified"
                profile = await self.memory_service.get_speaker_profile(self.session_states[session_id]["user_id"])
                await self.send_event(session_id, "speaker.renamed", {"user_id": profile["user_id"], "new_name": profile["name"]})
                llm_response_text = await self.generate_llm_response(f"Confirm to the user, whose name is {profile['name']}, that you will remember their name.")
            else:
                llm_response_text = await self.generate_llm_response(f"I couldn't quite get your name from '{transcript}'. Could you please say it again?")
        else:
            # General conversation
            llm_response_text = await self.generate_llm_response(transcript)

        # 4. Synthesize and send audio response
        if llm_response_text:
            audio_response = await self.synthesize_speech(llm_response_text)
            if audio_response:
                await self.send_event(session_id, "assistant.speak", {"text": llm_response_text, "audio_data": base64.b64encode(audio_response).decode()})

    async def initiate_naming_conversation(self, session_id: str, user_id: str, temp_name: str):
        prompt = f"A new user, currently named '{temp_name}', has joined. Please ask them what you should call them. Be friendly and direct."
        response = await self.generate_llm_response(prompt)
        # Send only text, audio will be generated by handle_audio_stream after transcription
        await self.send_event(session_id, "assistant.speak", {"text": response})

    async def handle_speaker_claim(self, session_id: str, user_id: str, new_name: str):
        await self.memory_service.update_speaker_name(user_id, new_name)
        self.session_states[session_id]["status"] = "identified"
        profile = await self.memory_service.get_speaker_profile(user_id)
        await self.send_event(session_id, "speaker.renamed", {"user_id": user_id, "new_name": new_name})
        prompt = f"The user has chosen the name '{new_name}'. Please confirm that you will remember it."
        response = await self.generate_llm_response(prompt)
        await self.send_event(session_id, "assistant.speak", {"text": response})

    async def generate_llm_response(self, prompt: str) -> str:
        try:
            response = await self.ollama_client.generate(
                model=config.LLM_MODEL,
                prompt=prompt,
                stream=False
            )
            return response.get("response", "").strip()
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "I'm sorry, I'm having trouble communicating right now."

    async def send_event(self, session_id: str, event: str, data: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json({"event": event, "data": data})

    async def send_error(self, session_id: str, message: str):
        await self.send_event(session_id, "error", {"message": message})

orchestrator = VoiceOrchestrator()

@app.websocket("/ws/v1/voice/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await orchestrator.handle_connection(websocket, session_id)

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})

@app.on_event("startup")
async def startup_event():
    logger.info("Voice Orchestrator v2.0 starting up...")
    await orchestrator.memory_service.initialize_chroma_client()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)