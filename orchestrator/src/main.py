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
        self.active_connections = {}

    async def handle_connection(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        try:
            while True:
                data = await websocket.receive_json()
                event = data.get("event")
                if event == "audio_stream":
                    await self.handle_audio_stream(session_id, data.get("audio_data"))
                elif event == "speaker.claim":
                    await self.handle_speaker_claim(session_id, data.get("user_id"), data.get("new_name"))
        except WebSocketDisconnect:
            del self.active_connections[session_id]
            logger.info(f"Session {session_id} disconnected.")

    async def handle_audio_stream(self, session_id: str, audio_base64: str):
        audio_data = base64.b64decode(audio_base64)
        embedding = await self.voice_service.get_embedding(audio_data)

        if not embedding:
            await self.send_error(session_id, "Failed to create voice embedding.")
            return

        user_id = await self.memory_service.find_speaker_by_embedding(embedding)

        if user_id:
            profile = await self.memory_service.get_speaker_profile(user_id)
            await self.send_event(session_id, "speaker.identified", {"user_id": user_id, "name": profile.get("name", "Unknown")})
        else:
            user_id = await self.memory_service.create_speaker_profile(embedding)
            profile = await self.memory_service.get_speaker_profile(user_id)
            await self.send_event(session_id, "speaker.identified", {"user_id": user_id, "name": profile.get("name"), "status": "unclaimed"})
            await self.initiate_naming_conversation(session_id, user_id, profile.get("name"))

    async def initiate_naming_conversation(self, session_id: str, user_id: str, temp_name: str):
        prompt = f"A new user, currently named '{temp_name}', has joined. Please ask them what you should call them. Be friendly and direct."
        response = await self.generate_llm_response(prompt)
        await self.send_event(session_id, "assistant.speak", {"text": response})

    async def handle_speaker_claim(self, session_id: str, user_id: str, new_name: str):
        await self.memory_service.update_speaker_name(user_id, new_name)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
