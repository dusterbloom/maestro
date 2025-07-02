# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voice Orchestrator is an ultra-low-latency (<500ms) voice assistant that orchestrates Docker containers for speech-to-text, language models, and text-to-speech. The system achieves production-ready performance by orchestrating existing services rather than implementing custom ML code.

## Architecture

The system uses a streaming pipeline architecture:
```
WhisperLive (STT) → Orchestrator → Ollama (LLM) → Kokoro (TTS)
                         ↓
                    A-MEM (Memory)
                    Redis + ChromaDB
```

## Development Commands

### Docker Compose Deployment Options
```bash
# GPU-accelerated deployment (default)
docker-compose up -d

# CPU-only deployment
docker-compose -f docker-compose.cpu.yml up -d

# With memory components (Redis, ChromaDB, A-MEM)
docker-compose -f docker-compose.yml -f docker-compose.memory.yml up -d

# Quick start with prerequisites check
./scripts/quick-start.sh
```

### Testing
```bash
# End-to-end latency testing
python scripts/latency-test.py

# Health check all services
./scripts/health-check.sh

# Individual service health
curl http://localhost:8000/health    # Orchestrator
curl http://localhost:9090/health    # WhisperLive
curl http://localhost:8880/health    # Kokoro TTS
```

### Prerequisites
- Ollama running on port 11434 with models:
  ```bash
  ollama pull gemma2:2b
  ollama pull nomic-embed-text
  ```
- Docker with GPU support (NVIDIA runtime)
- Copy `.env.example` to `.env` and configure

## Key Components

### Orchestrator Service (`orchestrator/`)
- FastAPI backend with WebSocket support
- Routes audio between STT, LLM, and TTS services
- Handles memory integration (optional)
- Built with Python 3.11-slim container

### Frontend UI (`ui/`)
- Next.js 14 PWA with push-to-talk interface
- WebSocket connection for real-time audio streaming
- React 18+ with TypeScript
- Build commands: `npm run dev`, `npm run build`, `npm start`

### Memory Components (Optional)
- A-MEM: Agentic memory service
- Redis: Session caching
- ChromaDB: Vector storage for embeddings
- Enabled via `MEMORY_ENABLED=true` environment variable

## Performance Requirements

Target latency budget:
- Audio capture: 16ms
- STT (WhisperLive): 120ms  
- LLM (Ollama): 180ms
- TTS (Kokoro): 80ms
- Network overhead: 12ms
- **Total: 408ms** (< 500ms target)

## Configuration

All configuration via environment variables in `.env`:
- Core services: `WHISPER_URL`, `OLLAMA_URL`, `TTS_URL`
- Models: `STT_MODEL=tiny`, `LLM_MODEL=gemma2:2b`, `TTS_VOICE=af_bella`
- Memory: `MEMORY_ENABLED=false`, `AMEM_URL`, `REDIS_URL`
- Performance: `CHUNK_SIZE_MS=320`, `TARGET_LATENCY_MS=500`

## Project Structure

```
/docs/               # Implementation guides and architecture docs
/orchestrator/src/   # FastAPI backend service
/ui/                 # Next.js frontend application
/scripts/            # Deployment and testing scripts
/data/               # Persistent data volumes (Redis, ChromaDB)
```

The `/docs/` directory contains detailed implementation tasks for backend, frontend, DevOps, and testing components.

## Development Log

**Important:** Always maintain a log of changes, implementation decisions, and results in `DEVELOPMENT_LOG.md` for future Claude instances to reference. This ensures continuity and helps track what has been built, tested, and what issues were encountered.

### Log Management
- Create/update `DEVELOPMENT_LOG.md` after each significant change
- Document implementation decisions and rationale
- Record test results and performance measurements
- Note any blockers or issues encountered
- Track which components are complete vs. in-progress