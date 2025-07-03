# Agent Implementation Tasks

## Task Assignment Matrix

| Task ID | Agent | Priority | Dependencies | Est. Hours |
|---------|-------|----------|--------------|------------|
| DEVOPS-001 | DevOps | P0 | None | 2 |
| DEVOPS-002 | DevOps | P0 | DEVOPS-001 | 1 |
| BACKEND-001 | Backend | P1 | DEVOPS-001 | 3 |
| BACKEND-002 | Backend | P1 | BACKEND-001 | 2 |
| FRONTEND-001 | Frontend | P2 | BACKEND-001 | 4 |
| DEVOPS-003 | DevOps | P2 | All above | 2 |
| TEST-001 | Testing | P3 | All above | 3 |

## DevOps Agent Tasks

### DEVOPS-001: Create Base Repository Structure
**Objective**: Initialize project repository with proper structure
**Deliverables**:
```
voice-orchestrator/
├── docker-compose.yml
├── docker-compose.cpu.yml
├── docker-compose.memory.yml
├── .env.example
├── .gitignore
├── README.md
├── orchestrator/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
├── ui/
│   ├── Dockerfile
│   └── package.json
├── data/
│   ├── redis/.gitkeep
│   ├── chromadb/.gitkeep
│   └── models/.gitkeep
└── scripts/
    ├── health-check.sh
    └── quick-start.sh
```
**Acceptance Criteria**:
- Repository created on GitHub
- All directories present
- .gitignore excludes data/, .env, node_modules/

### DEVOPS-002: Write Docker Compose Files
**Objective**: Create all docker-compose variants
**Files to create**:
1. `docker-compose.yml` - Base configuration with GPU support
2. `docker-compose.cpu.yml` - CPU-only override
3. `docker-compose.memory.yml` - Memory components override

**Key content for docker-compose.yml**:
```yaml
version: '3.8'
services:
  orchestrator:
    build: ./orchestrator
    ports: ["8000:8000"]
    environment:
      - WHISPER_URL=http://whisper-live:9090
      - OLLAMA_URL=http://host.docker.internal:11434
      - TTS_URL=http://kokoro:8880/v1
    depends_on: [whisper-live, kokoro]

  whisper-live:
    image: collabora/whisperlive:latest-gpu
    # ... (as defined in architecture)

  kokoro:
    image: ghcr.io/remsky/kokoro-fastapi-gpu:v0.2.1
    # ... (as defined in architecture)

  voice-ui:
    build: ./ui
    # ... (as defined in architecture)
```

### DEVOPS-003: Create CI/CD Pipeline
**Objective**: GitHub Actions for multi-arch builds
**File**: `.github/workflows/build.yml`
**Requirements**:
- Build on push to main
- Multi-arch support (amd64, arm64)
- Push to GitHub Container Registry
- Run basic health checks

## Backend Agent Tasks

### BACKEND-001: Implement Orchestrator Service
**Objective**: Create the minimal FastAPI orchestrator
**File**: `orchestrator/src/main.py`
**Requirements**:
- Copy the orchestrator code from architecture exactly
- Add proper error handling only for service connectivity
- Implement health endpoint: `GET /health`
- Add OpenAPI documentation

**Additional files**:
- `orchestrator/requirements.txt`:
  ```
  fastapi==0.109.0
  uvicorn==0.27.0
  httpx==0.26.0
  websockets==12.0
  python-multipart==0.0.6
  ```
- `orchestrator/Dockerfile`:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY src/ .
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

### BACKEND-002: Add Memory Integration
**Objective**: Implement optional memory methods
**File**: `orchestrator/src/memory.py`
**Requirements**:
- Create MemoryManager class
- Implement retrieve_context() and store_interaction()
- Add Redis connection pooling
- Handle memory service unavailability gracefully

## Frontend Agent Tasks

### FRONTEND-001: Create Next.js Voice UI
**Objective**: Minimal voice interface with WebSocket
**Files**:
```
ui/
├── app/
│   ├── page.tsx          # Main voice interface
│   ├── layout.tsx        # Root layout
│   └── globals.css       # Tailwind styles
├── components/
│   ├── VoiceButton.tsx   # Push-to-talk button
│   ├── Waveform.tsx      # Audio visualization
│   └── StatusIndicator.tsx
├── lib/
│   ├── websocket.ts      # WebSocket client
│   └── audio.ts          # Web Audio API wrapper
└── public/
    └── manifest.json     # PWA manifest
```

**Key Requirements**:
- Push-to-talk interface
- Real-time waveform visualization
- Connection status indicator
- Automatic reconnection
- PWA support for mobile

**Core WebSocket implementation**:
```typescript
// lib/websocket.ts
export class VoiceWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  
  connect(url: string) {
    this.ws = new WebSocket(url);
    this.ws.binaryType = 'arraybuffer';
    
    this.ws.onopen = () => this.onConnect();
    this.ws.onmessage = (e) => this.onAudioReceived(e.data);
    this.ws.onclose = () => this.scheduleReconnect();
  }
  
  sendAudio(audioData: ArrayBuffer) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(audioData);
    }
  }
}
```

### FRONTEND-002: Add Configuration UI
**Objective**: Settings panel for model selection
**File**: `ui/components/SettingsPanel.tsx`
**Requirements**:
- Dropdown for STT model size
- Voice selection for TTS
- Memory enable/disable toggle
- Save to localStorage

## Testing Agent Tasks

### TEST-001: Create E2E Latency Test
**Objective**: Automated latency measurement
**File**: `scripts/latency-test.py`
**Requirements**:
- Send synthetic audio through full pipeline
- Measure each component's latency
- Output JSON report
- Fail if total > 500ms

**Test scenarios**:
1. Simple greeting ("Hello")
2. Complex query (30 words)
3. Memory-enabled conversation
4. Concurrent requests (10 users)

## Quick Start Script

### DEVOPS-004: Create One-Command Setup
**File**: `scripts/quick-start.sh`
```bash
#!/bin/bash
set -e

echo "🎙️ Voice Orchestrator Quick Start"

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker required"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose required"; exit 1; }

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null; then
    echo "⚠️  Ollama not detected. Please start Ollama first."
    exit 1
fi

# Pull required model if not present
if ! curl -s http://localhost:11434/api/tags | grep -q "gemma3n:latest"; then
    echo "📥 Pulling gemma3n:latest model..."
    ollama pull gemma3n:latest
fi

# Create .env from example
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file"
fi

# Start services
echo "🚀 Starting services..."
docker-compose pull
docker-compose up -d

# Wait for health
echo "⏳ Waiting for services..."
sleep 10

# Check health
if curl -s http://localhost:8000/health | grep -q "ok"; then
    echo "✅ Voice assistant ready!"
    echo "🌐 Open http://localhost:3000"
else
    echo "❌ Health check failed"
    docker-compose logs
    exit 1
fi
```

## Task Execution Order

1. **Week 1**: DevOps creates structure and compose files
2. **Week 2**: Backend implements orchestrator while Frontend starts UI
3. **Week 3**: Integration testing and memory components
4. **Week 4**: Documentation and deployment scripts

## Success Metrics

- All services start with `docker-compose up`
- E2E latency < 500ms on reference hardware
- Zero custom code in service containers
- Health checks pass for all components
- Works on Linux, macOS, and Windows (WSL2)