# Voice Orchestrator - Ultra-Low-Latency Voice Assistant

A production-ready voice assistant achieving <500ms end-to-end latency by orchestrating best-in-class Docker containers.

## Architecture

```
WhisperLive (STT) → Orchestrator → Ollama (LLM) → Kokoro (TTS)
                         ↓
                    A-MEM (Memory)
                    Redis + ChromaDB
```

## Quick Start

```bash
# 1. Ensure Ollama is running
ollama pull gemma3n:latest
ollama pull nomic-embed-text

# 2. Clone and configure
git clone https://github.com/your-org/voice-orchestrator
cd voice-orchestrator
cp .env.example .env

# 3. Start everything
docker-compose up -d

# 4. Open UI
open http://localhost:3000
```

## Features

- **<500ms latency** - Optimized streaming pipeline
- **100% Docker** - No custom ML code
- **Memory support** - Optional conversation memory
- **GPU accelerated** - CUDA support for all components
- **Multi-platform** - x86_64 and ARM64 support

## Components

| Component | Purpose | Docker Image |
|-----------|---------|--------------|
| WhisperLive | Speech-to-Text | `collabora/whisperlive:latest-gpu` |
| Ollama | Language Model | External service on port 11434 |
| Kokoro | Text-to-Speech | `ghcr.io/remsky/kokoro-fastapi-gpu:v0.2.1` |
| A-MEM | Agentic Memory | `agiresearch/a-mem:latest` |
| Redis | Cache | `redis:7-alpine` |
| ChromaDB | Vector Store | `chromadb/chroma:latest` |

## Configuration

All configuration via environment variables in `.env`:

```bash
# Core Services
WHISPER_URL=http://whisper-live:9090
OLLAMA_URL=http://host.docker.internal:11434
TTS_URL=http://kokoro:8880/v1

# Memory (Optional)
MEMORY_ENABLED=false
AMEM_URL=http://a-mem:8001
REDIS_URL=redis://redis:6379

# Models
STT_MODEL=tiny
LLM_MODEL=gemma3n:latest
TTS_VOICE=af_bella
```

## Deployment Options

```bash
# CPU only
docker-compose -f docker-compose.cpu.yml up

# With memory components
docker-compose -f docker-compose.yml -f docker-compose.memory.yml up

# Production with monitoring
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up
```

## Performance

Latency budget (measured on M2 Pro):
- Audio capture: 16ms
- STT (WhisperLive): 120ms  
- LLM (Ollama): 180ms
- TTS (Kokoro): 80ms
- Network overhead: 12ms
- **Total: 408ms**

## Development

See task guides:
- [Backend Tasks](backend-tasks.md)
- [Frontend Tasks](frontend-tasks.md)
- [DevOps Tasks](devops-tasks.md)
- [Testing Guide](testing-guide.md)

## License

Apache 2.0