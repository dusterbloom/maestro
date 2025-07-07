# CLAUDE.md

## CRITICAL TECHNICAL ACCURACY REQUIREMENTS

**INTEGRATION PROTOCOL:**
1. Search first, code second - use MCP Deep Graph to examine actual source code
2. Verify protocols from source, never assume formats/endpoints/WebSocket structures
3. If verification fails: "Cannot find verified implementation for [X], cannot provide reliable solution"
4. Debug current state before changes - examine logs, API responses, actual behavior
5. Fix root cause, no workarounds

**PROHIBITED:** Assuming protocols, typical patterns, educated guesses, new endpoints without verification, changes without understanding problems

**EXAMPLE:**
- ✅ "Using Deep Graph MCP to examine `collabora/WhisperLive` source code..."
- ❌ "WhisperLive typically uses WebSocket connections that expect..."

---

## Project Overview

Voice Orchestrator: Ultra-low-latency (<500ms) voice assistant orchestrating Docker containers for STT/LLM/TTS. Production-ready performance via service orchestration, not custom ML.

**Architecture:**
```
WhisperLive (STT) → Orchestrator → Ollama (LLM) → Kokoro (TTS)
                         ↓
                    A-MEM (Memory)
                    Redis + ChromaDB
```

## Development Commands

**Docker Compose:**
```bash
docker-compose up -d                                                    # GPU default
docker-compose -f docker-compose.cpu.yml up -d                        # CPU only
docker-compose -f docker-compose.yml -f docker-compose.memory.yml up -d # With memory
./scripts/quick-start.sh                                               # Quick start
```

**Testing:**
```bash
python scripts/latency-test.py         # E2E latency
./scripts/health-check.sh              # All services
curl http://localhost:8000/health      # Orchestrator
curl http://localhost:9090/health      # WhisperLive  
curl http://localhost:8880/health      # Kokoro TTS
```

**Prerequisites:**
- Ollama on port 11434: `ollama pull gemma3n:latest && ollama pull nomic-embed-text`
- Docker with GPU support (NVIDIA runtime)
- Copy `.env.example` to `.env`

## Key Components

- **Orchestrator:** FastAPI backend, WebSocket support, routes audio STT→LLM→TTS, memory integration
- **Frontend:** Next.js 14 PWA, push-to-talk, WebSocket streaming, React 18+/TypeScript
- **Memory:** A-MEM service, Redis caching, ChromaDB vectors, `MEMORY_ENABLED=true`

## Performance Requirements

Target latency (408ms < 500ms):
Audio capture: 16ms | STT: 120ms | LLM: 180ms | TTS: 80ms | Network: 12ms

## Configuration

Environment variables in `.env`:
- Services: `WHISPER_URL`, `OLLAMA_URL`, `TTS_URL`
- Models: `STT_MODEL=tiny`, `LLM_MODEL=gemma3n:latest`, `TTS_VOICE=af_bella`
- Memory: `MEMORY_ENABLED=false`, `AMEM_URL`, `REDIS_URL`
- Performance: `CHUNK_SIZE_MS=320`, `TARGET_LATENCY_MS=500`

## Project Structure

```
/docs/               # Implementation guides
/orchestrator/src/   # FastAPI backend
/ui/                 # Next.js frontend
/scripts/            # Deployment/testing
/data/               # Persistent volumes
```

## MCP Tools

**MANDATORY: Deep Graph MCP (examine before coding):**

Repositories: `collabora/WhisperLive`, `ollama/ollama`, `thewh1teagle/kokoro-onnx`, `remsky/Kokoro-FastAPI`

Commands:
- `folder-tree-structure`: Explore structure
- `nodes-semantic-search`: Search functionality
- `get-code`: Get implementation
- `find-direct-connections`: Analyze dependencies
- `docs-semantic-search`: Search docs
- `get-usage-dependency-links`: Impact analysis

**Vibe Git MCP:**
- `start_vibing`: Start auto-commit session
- `stop_vibing`: Complete with squashed commit/PR
- `vibe_status`: Check status
- `stash_and_vibe`: Stash and start fresh
- `commit_and_vibe`: Commit WIP and start new
- `vibe_from_here`: Continue from current state

**Additional:** Sequential Thinking MCP, MCP Resource Tools

## Technical Standards

**Integration:** Use MCP Deep Graph → verify protocols from source → test with real endpoints → document with source references

**Code Quality:** WebSocket implementations from verified protocols, audio formats from actual requirements, error handling for real failure modes, performance from measured latencies

**Testing:** Integration tests with actual Docker services, end-to-end latency measurements, WebSocket testing with actual services, audio pipeline with real STT/TTS

## Development Log

Maintain `DEVELOPMENT_LOG.md` with:
- Implementation decisions/rationale
- Source code references/protocol verifications
- Test results/performance measurements
- Blockers/issues
- Component completion status
- URLs/specific code references for integrations

Document per integration:
- Source repository/files examined
- Actual protocol/API details from source
- Assumptions made (why verification failed)
- Test results with real services
- Performance measurements/optimizations