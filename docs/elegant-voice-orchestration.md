# Elegant Voice Assistant Orchestration

## Philosophy: Compose, Don't Build

Instead of creating custom integrations, we orchestrate existing, battle-tested Docker containers with a focus on:
- **Zero custom code** for core components
- **Configuration over coding**
- **Standard protocols** (OpenAI API, Wyoming)
- **Plug-and-play** architecture

## Architecture Overview

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Frontend - Next.js UI
  voice-ui:
    image: ghcr.io/your-org/voice-ui:latest
    build:
      context: ./ui
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_WS_URL=ws://orchestrator:8000/ws
    depends_on:
      - orchestrator

  # Orchestrator - Minimal FastAPI WebSocket Router
  orchestrator:
    image: ghcr.io/your-org/voice-orchestrator:latest
    build:
      context: ./orchestrator
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - WHISPER_URL=http://whisper-live:9090
      - OLLAMA_URL=http://host.docker.internal:11434
      - TTS_URL=http://kokoro:8880/v1
    depends_on:
      - whisper-live
      - kokoro

  # STT - WhisperLive with CUDA
  whisper-live:
    image: collabora/whisperlive:latest-gpu
    ports:
      - "9090:9090"
    volumes:
      - ./models/whisper:/models
    environment:
      - MODEL_SIZE=tiny
      - VAD_ENABLED=true
      - DEVICE=cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # TTS - Kokoro FastAPI
  kokoro:
    image: ghcr.io/remsky/kokoro-fastapi-gpu:v0.2.1
    ports:
      - "8880:8880"
    environment:
      - DEFAULT_VOICE=af_bella
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Memory Components
  a-mem:
    image: agiresearch/a-mem:latest
    ports:
      - "8001:8001"
    environment:
      - EMBEDDING_MODEL=ollama
      - OLLAMA_URL=http://host.docker.internal:11434
      - EMBEDDING_MODEL_NAME=nomic-embed-text
      - REDIS_URL=redis://redis:6379
      - VECTOR_DB_URL=http://chromadb:8002
    depends_on:
      - redis
      - chromadb

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data
    command: redis-server --appendonly yes

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8002:8000"
    volumes:
      - ./data/chromadb:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - PERSIST_DIRECTORY=/chroma/chroma
      - ANONYMIZED_TELEMETRY=FALSE

  # Alternative: Qdrant instead of ChromaDB
  # qdrant:
  #   image: qdrant/qdrant:latest
  #   ports:
  #     - "6333:6333"
  #   volumes:
  #     - ./data/qdrant:/qdrant/storage

  # Alternative: Wyoming Protocol Stack
  # wyoming-whisper:
  #   image: rhasspy/wyoming-whisper:latest
  #   ports:
  #     - "10300:10300"
  #   volumes:
  #     - ./whisper-data:/data
  #   command: --model tiny --language en --device cuda
  #   runtime: nvidia

networks:
  voice-net:
    driver: bridge
```

## Minimal Orchestrator Design

The only custom code needed is a lightweight orchestrator (< 200 lines):

```python
# orchestrator/main.py
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import httpx
import asyncio
import json
import os
from typing import Optional

app = FastAPI()

class VoiceOrchestrator:
    def __init__(self):
        self.whisper_url = os.getenv("WHISPER_URL")
        self.ollama_url = os.getenv("OLLAMA_URL")
        self.tts_url = os.getenv("TTS_URL")
        
        # Memory components (optional)
        self.memory_enabled = os.getenv("MEMORY_ENABLED", "false").lower() == "true"
        self.amem_url = os.getenv("AMEM_URL")
        self.redis_url = os.getenv("REDIS_URL")
        
    async def process_audio(self, audio_data: bytes, session_id: str) -> bytes:
        # 1. Send to WhisperLive
        transcript = await self.transcribe(audio_data)
        
        # 2. Memory lookup (if enabled)
        context = ""
        if self.memory_enabled:
            context = await self.retrieve_context(transcript, session_id)
        
        # 3. Send to Ollama with context
        response = await self.generate_response(transcript, context)
        
        # 4. Store in memory (if enabled)
        if self.memory_enabled:
            await self.store_interaction(transcript, response, session_id)
        
        # 5. Send to Kokoro
        audio_response = await self.synthesize(response)
        
        return audio_response
    
    async def transcribe(self, audio: bytes) -> str:
        # Use WhisperLive WebSocket API
        pass
    
    async def retrieve_context(self, query: str, session_id: str) -> str:
        # Query A-MEM for relevant context
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.amem_url}/retrieve",
                json={"query": query, "session_id": session_id, "k": 5}
            )
            return resp.json().get("context", "")
    
    async def store_interaction(self, user_input: str, ai_response: str, session_id: str):
        # Store conversation in A-MEM
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.amem_url}/store",
                json={
                    "user_input": user_input,
                    "ai_response": ai_response,
                    "session_id": session_id
                }
            )
    
    async def generate_response(self, text: str, context: str = "") -> str:
        # Use Ollama HTTP API
        prompt = f"{context}\n\nUser: {text}\nAssistant:" if context else text
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.ollama_url}/api/generate",
                json={"model": "gemma3n:latest", "prompt": prompt, "stream": False}
            )
            return resp.json()["response"]
    
    async def synthesize(self, text: str) -> bytes:
        # Use Kokoro OpenAI-compatible API
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.tts_url}/audio/speech",
                json={"model": "kokoro", "voice": "af_bella", "input": text}
            )
            return resp.content

orchestrator = VoiceOrchestrator()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.headers.get("x-session-id", "default")
    
    try:
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            
            # Process through pipeline
            response = await orchestrator.process_audio(data, session_id)
            
            # Send back audio
            await websocket.send_bytes(response)
            
    except WebSocketDisconnect:
        pass
```

## Configuration Management

```yaml
# config/voice-assistant.yml
pipeline:
  stt:
    provider: whisperlive  # or wyoming-whisper
    model: tiny
    language: en
    vad_enabled: true
    
  llm:
    provider: ollama
    model: gemma3n:latest
    max_tokens: 150
    temperature: 0.7
    
  tts:
    provider: kokoro  # or piper
    voice: af_bella
    speed: 1.0
    
performance:
  chunk_size_ms: 320
  target_latency_ms: 500
  gpu_memory_fraction: 0.8
```

## Deployment Variations

### 1. Minimal (CPU Only)
```bash
docker-compose -f docker-compose.cpu.yml up
```

### 2. Standard (Single GPU)
```bash
docker-compose up
```

### 3. Distributed (Multi-GPU)
```bash
docker-compose -f docker-compose.distributed.yml up
```

### 4. Wyoming Protocol (Home Assistant)
```bash
docker-compose -f docker-compose.wyoming.yml up
```

## Benefits of This Approach

1. **Maintainability**: Each component updates independently
2. **Flexibility**: Swap providers via environment variables
3. **Scalability**: Add more instances behind a load balancer
4. **Debugging**: Each service has its own logs and metrics
5. **Community**: Leverage existing ecosystems

## Quick Start (5 minutes)

```bash
# Clone the orchestration repo
git clone https://github.com/your-org/voice-orchestrator
cd voice-orchestrator

# Set your configuration
cp .env.example .env
# Edit .env to set OLLAMA_HOST if using external Ollama

# Pull and start everything
docker-compose pull
docker-compose up -d

# Check health
curl http://localhost:8000/health

# Open UI
open http://localhost:3000
```

## Monitoring Dashboard

```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    
  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    volumes:
      - ./monitoring/dashboards:/var/lib/grafana/dashboards
```

Pre-built dashboard shows:
- End-to-end latency per component
- GPU utilization
- Request throughput
- Error rates

## Configuration Examples

### .env.example
```bash
# Core Services
WHISPER_URL=http://whisper-live:9090
OLLAMA_URL=http://host.docker.internal:11434
TTS_URL=http://kokoro:8880/v1

# Memory Components (Optional)
MEMORY_ENABLED=false
AMEM_URL=http://a-mem:8001
REDIS_URL=redis://redis:6379

# Model Configuration
EMBEDDING_MODEL=nomic-embed-text
STT_MODEL=tiny
LLM_MODEL=gemma3n:latest
TTS_VOICE=af_bella

# Performance
CHUNK_SIZE_MS=320
TARGET_LATENCY_MS=500
```

### Production Deployment with Memory

```bash
# Enable memory components
export MEMORY_ENABLED=true

# Start all services including memory
docker-compose up -d

# Scale redis for production
docker-compose up -d --scale redis=3

# Monitor memory usage
docker-compose logs -f a-mem redis chromadb
```

## Architecture with Memory Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│ Orchestrator│────▶│ WhisperLive │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                    ┌──────▼──────┐
                    │   A-MEM     │
                    │  (Optional) │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   ┌─────────┐      ┌─────────┐      ┌─────────┐
   │  Redis  │      │ ChromaDB│      │ Ollama  │
   │ (Cache) │      │(Vectors)│      │  (LLM)  │
   └─────────┘      └─────────┘      └─────────┘
```

## Memory-Aware Conversation Example

```python
# When MEMORY_ENABLED=true, the orchestrator:
# 1. Retrieves relevant past interactions
# 2. Includes context in LLM prompt
# 3. Stores new interactions for future use

# Example context injection:
"""
Previous context:
- User asked about weather yesterday
- User prefers metric units
- User is in Milan

User: What should I wear today?
Assistant: [Contextual response considering location and preferences]
"""
```

## Scaling Considerations

### Memory Components Performance
- **Redis**: Sub-millisecond lookups, handles 100k+ ops/sec
- **ChromaDB**: ~50ms for similarity search on 1M vectors
- **A-MEM**: Adds ~100ms to pipeline when enabled

### Resource Requirements
```yaml
# docker-compose.prod.yml override
services:
  redis:
    deploy:
      resources:
        limits:
          memory: 2G
          
  chromadb:
    deploy:
      resources:
        limits:
          memory: 4G
          
  a-mem:
    deploy:
      resources:
        limits:
          memory: 1G
```