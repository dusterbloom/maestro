# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL TECHNICAL ACCURACY REQUIREMENTS

**MANDATORY WORKFLOW FOR ALL TECHNICAL SOLUTIONS:**

1. **EVIDENCE-BASED RESPONSES ONLY**: Every technical implementation must be backed by actual source code, documentation, or verified examples from the real services being integrated
2. **SEARCH FIRST, CODE SECOND**: Before providing any solution, search for and examine actual implementations using MCP tools
3. **VERIFY PROTOCOLS**: For any API/WebSocket/protocol integration, find and examine the real implementation code
4. **SOURCE OR STOP**: If you cannot find authoritative sources, explicitly state limitations rather than making assumptions

**PROHIBITED APPROACHES:**
- Assuming WebSocket protocols without examining actual source code
- Providing "typical" FastAPI patterns without verifying target service requirements
- Making educated guesses about Docker service configurations
- Extrapolating from general documentation when service-specific implementation exists

**REQUIRED VERIFICATION PROCESS:**
Before implementing any integration:
1. Use **Deep Graph MCP** to examine actual source code of target services
2. Search for real protocol implementations, not generic patterns
3. Verify audio formats, WebSocket message structures, and API endpoints from source
4. Reference specific code examples and documentation URLs
5. If verification fails, state: "I cannot find verified implementation details for [X], so I cannot provide a reliable solution"

**EXAMPLE - WhisperLive Integration:**
- ✅ Correct: "Using Deep Graph MCP to examine `collabora/WhisperLive` source code..." → finds actual WebSocket protocol
- ❌ Incorrect: "WhisperLive typically uses WebSocket connections that expect..." [without source verification]

---

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
  ollama pull gemma3n:latest
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
- Models: `STT_MODEL=tiny`, `LLM_MODEL=gemma3n:latest`, `TTS_VOICE=af_bella`
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

## Available MCP Tools

### **MANDATORY: Codebase Analysis - Deep Graph MCP**
Use **Deepgraph** for understanding large codebases, analyzing dependencies, and mapping code relationships. **REQUIRED before any integration work** to examine actual implementations rather than making assumptions.

**Available Repositories (EXAMINE BEFORE CODING):**
- `collabora/WhisperLive` - Speech-to-text WebSocket service
- `ollama/ollama` - Language model API server  
- `thewh1teagle/kokoro-onnx` - Text-to-speech library
- `remsky/Kokoro-FastAPI` - Kokoro TTS web service wrapper

**Key Commands:**
- `mcp__Deep_Graph_MCP__folder-tree-structure`: Explore repository structure
- `mcp__Deep_Graph_MCP__nodes-semantic-search`: Search for functionality by description
- `mcp__Deep_Graph_MCP__get-code`: Get actual implementation code
- `mcp__Deep_Graph_MCP__find-direct-connections`: Analyze dependencies
- `mcp__Deep_Graph_MCP__docs-semantic-search`: Search documentation
- `mcp__Deep_Graph_MCP__get-usage-dependency-links`: Impact analysis

**MANDATORY WORKFLOW EXAMPLE:**
```
1. mcp__Deep_Graph_MCP__nodes-semantic-search: "WebSocket protocol WhisperLive"
2. mcp__Deep_Graph_MCP__get-code: [examine actual implementation]
3. Implement based on verified protocol, not assumptions
```

### Git Workflow - Vibe Git MCP
Advanced git workflow management with auto-commit functionality for clean development history.

**Key Commands:**
- `mcp__vibe-git__start_vibing`: Start auto-committing session (call first before code changes)
- `mcp__vibe-git__stop_vibing`: Complete session with squashed commit and PR creation
- `mcp__vibe-git__vibe_status`: Check current session status
- `mcp__vibe-git__stash_and_vibe`: Stash changes and start fresh session
- `mcp__vibe-git__commit_and_vibe`: Commit work-in-progress and start new session
- `mcp__vibe-git__vibe_from_here`: Continue vibing from current state

### Additional MCP Tools
- **Sequential Thinking MCP**: For complex problem-solving and multi-step analysis (highly recommended)
- **MCP Resource Tools**: List and read resources from configured servers

## TECHNICAL IMPLEMENTATION STANDARDS

### Integration Requirements
**Before implementing any service integration:**

1. **Use MCP Deep Graph to examine target service source code**
2. **Verify actual WebSocket/API protocols from source**
3. **Test with real service endpoints, not mock implementations**
4. **Document verified protocol details with source code references**

### Code Quality Standards
- All WebSocket implementations must be based on verified protocols from target services
- Audio format conversions must match actual service requirements (verified from source)
- Error handling must account for real service failure modes (not generic assumptions)
- Performance optimizations must be based on measured latencies with real services

### Testing Requirements
- Integration tests must use actual Docker services, not mocks
- Latency measurements must be end-to-end with real services
- WebSocket connection handling must be tested with actual target services
- Audio pipeline testing must use real STT/TTS services

## Development Log

**Important:** Always maintain a log of changes, implementation decisions, and results in `DEVELOPMENT_LOG.md` for future Claude instances to reference. This ensures continuity and helps track what has been built, tested, and what issues were encountered.

### Log Management
- Create/update `DEVELOPMENT_LOG.md` after each significant change
- Document implementation decisions and rationale
- **Record actual source code references and protocol verifications**
- Record test results and performance measurements
- Note any blockers or issues encountered
- Track which components are complete vs. in-progress
- **Include URLs and specific code references for all integrations**

### Verification Documentation Required
For each integration, document:
- Source repository and specific files examined
- Actual protocol/API details found in source code
- Any assumptions that had to be made (and why verification failed)
- Test results with real services
- Performance measurements and optimizations applied

