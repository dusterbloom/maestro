# GEMINI.md

This file provides guidance to Gemini when working with code in this repository. Adhering to these instructions is crucial for effective and accurate assistance.

## Guiding Principles

**MANDATORY WORKFLOW FOR ALL TECHNICAL TASKS:**

1.  **Understand First, Act Second**: Before writing or changing code, thoroughly understand the existing codebase, conventions, and requirements. Use `glob`, `search_file_content`, and `read_file` to explore the project and gather context.
2.  **Evidence-Based Implementation**: All technical solutions must be based on evidence found within the project's source code, its dependencies, or official documentation.
3.  **Verify, Don't Assume**: For any integration (API, WebSocket, etc.), find and examine the actual implementation code in the target service or library. Never assume formats, endpoints, or protocols. If you cannot find authoritative sources, explicitly state what is missing.
4.  **Incremental Changes & Verification**: Make small, incremental changes. After each change, use `run_shell_command` to execute relevant tests (e.g., `./scripts/run-all-tests.sh`), linters, or build commands to verify the change's integrity.
5.  **Root Cause Analysis**: When debugging, use tools to examine logs, check API responses, and understand the current state before attempting a fix. Address the root cause directly rather than creating workarounds.

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

## Prerequisites
- Ollama running on port 11434 with models:
  ```bash
  ollama pull llama3.2:latest
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

## Gemini's Toolkit & Workflow

Your primary tools for interacting with this repository are:

-   **`read_file` / `read_many_files`**: To read files for context before making changes.
-   **`search_file_content`**: To find specific code, API usage, or configuration patterns.
-   **`glob`**: To discover files and understand the project structure.
-   **`run_shell_command`**: To execute build scripts, tests, and other CLI operations. **Always** prefer the scripts in `/scripts/` for common tasks.
-   **`replace` / `write_file`**: For modifying or creating files. Always use `read_file` first to get the exact context for `replace`.

### Example Debugging Workflow

**User Request**: "The frontend is failing to connect to the orchestrator's WebSocket."

**Gemini's Workflow**:

1.  **Formulate a Hypothesis**: The WebSocket URL, path, or protocol might be mismatched between the client and server.
2.  **Gather Evidence (Client-side)**: Use `read_file` on `ui/lib/websocket.ts` to see how the frontend initiates the WebSocket connection.
3.  **Gather Evidence (Server-side)**: Use `read_file` on `orchestrator/src/main.py` to see how the FastAPI backend defines the WebSocket endpoint.
4.  **Analyze and Compare**: Compare the URL, path, and any subprotocols used in the frontend with the endpoint defined in the backend.
5.  **Check Service Health**: Use `run_shell_command` with `docker ps` and `./scripts/health-check.sh` to ensure all services are running and healthy.
6.  **Propose a Fix**: Based on the evidence, propose a targeted change using `replace` to align the client and server code.
7.  **Verify**: After applying the fix, explain that the next step is to run the application and check the browser console and service logs to confirm the connection is established.

## Development Log

Maintain a log of changes, implementation decisions, and results in `DEVELOPMENT_LOG.md`. This ensures continuity and helps track what has been built, tested, and what issues were encountered. Before starting a new task, review this log to understand the latest state of development.
