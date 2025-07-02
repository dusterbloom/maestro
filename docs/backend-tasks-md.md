# Backend Implementation Tasks

## BACKEND-001: Orchestrator Service Implementation

### Objective
Create the minimal FastAPI orchestrator that routes between WhisperLive, Ollama, and Kokoro.

### Files to Create

#### orchestrator/src/main.py
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import httpx
import os
import asyncio
from typing import Optional

app = FastAPI(title="Voice Orchestrator", version="1.0.0")

class VoiceOrchestrator:
    def __init__(self):
        self.whisper_url = os.getenv("WHISPER_URL", "http://whisper-live:9090")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
        self.tts_url = os.getenv("TTS_URL", "http://kokoro:8880/v1")
        self.memory_enabled = os.getenv("MEMORY_ENABLED", "false").lower() == "true"
        self.amem_url = os.getenv("AMEM_URL", "http://a-mem:8001")
        
    async def process_audio(self, audio_data: bytes, session_id: str = "default") -> bytes:
        # 1. Transcribe audio
        transcript = await self.transcribe(audio_data)
        
        # 2. Retrieve context if memory enabled
        context = ""
        if self.memory_enabled:
            try:
                context = await self.retrieve_context(transcript, session_id)
            except:
                pass  # Graceful degradation
        
        # 3. Generate response
        response = await self.generate_response(transcript, context)
        
        # 4. Store interaction if memory enabled
        if self.memory_enabled:
            try:
                await self.store_interaction(transcript, response, session_id)
            except:
                pass  # Graceful degradation
        
        # 5. Synthesize speech
        audio_response = await self.synthesize(response)
        
        return audio_response
    
    async def transcribe(self, audio: bytes) -> str:
        # TODO: Implement WhisperLive WebSocket connection
        # For now, return mock response
        return "Hello, how can I help you?"
    
    async def generate_response(self, text: str, context: str = "") -> str:
        prompt = f"{context}\n\nUser: {text}\nAssistant:" if context else text
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": os.getenv("LLM_MODEL", "gemma2:2b"),
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
    
    async def synthesize(self, text: str) -> bytes:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.tts_url}/audio/speech",
                json={
                    "model": "kokoro",
                    "voice": os.getenv("TTS_VOICE", "af_bella"),
                    "input": text
                }
            )
            response.raise_for_status()
            return response.content
    
    async def retrieve_context(self, query: str, session_id: str) -> str:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{self.amem_url}/retrieve",
                json={"query": query, "session_id": session_id, "k": 5}
            )
            response.raise_for_status()
            return response.json().get("context", "")
    
    async def store_interaction(self, user_input: str, ai_response: str, session_id: str):
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{self.amem_url}/store",
                json={
                    "user_input": user_input,
                    "ai_response": ai_response,
                    "session_id": session_id
                }
            )

orchestrator = VoiceOrchestrator()

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "memory_enabled": orchestrator.memory_enabled})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.headers.get("x-session-id", "default")
    
    try:
        while True:
            data = await websocket.receive_bytes()
            response = await orchestrator.process_audio(data, session_id)
            await websocket.send_bytes(response)
    except WebSocketDisconnect:
        pass
```

#### orchestrator/requirements.txt
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
httpx==0.26.0
websockets==12.0
python-multipart==0.0.6
```

#### orchestrator/Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Acceptance Criteria
1. Health endpoint returns 200 OK
2. WebSocket accepts connections
3. Ollama integration works
4. TTS integration works
5. Graceful error handling

---

## BACKEND-002: Memory Integration

### Objective
Add optional memory support with A-MEM, Redis, and ChromaDB.

### Implementation Notes

The memory integration is already included in BACKEND-001's main.py. Additional work includes:

1. **Testing memory fallback**
   - Verify system works when MEMORY_ENABLED=false
   - Verify graceful degradation when memory services are down

2. **Add connection pooling for Redis** (if needed in future)
   ```python
   import redis.asyncio as redis
   
   class RedisPool:
       def __init__(self):
           self.pool = redis.ConnectionPool.from_url(
               os.getenv("REDIS_URL", "redis://redis:6379"),
               max_connections=10
           )
   ```

3. **Add retry logic for memory operations**
   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential
   
   @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
   async def retrieve_context_with_retry(self, query: str, session_id: str) -> str:
       return await self.retrieve_context(query, session_id)
   ```

### Testing Commands
```bash
# Test without memory
MEMORY_ENABLED=false docker-compose up orchestrator

# Test with memory
MEMORY_ENABLED=true docker-compose up

# Test memory service failure
docker-compose stop a-mem
# Verify orchestrator still works
```

### Deliverables
1. Memory integration tested
2. Fallback behavior verified
3. Connection pooling (if needed)
4. Documentation updated