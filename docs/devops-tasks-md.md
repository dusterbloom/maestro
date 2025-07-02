# DevOps Implementation Tasks

## DEVOPS-001: Create Repository Structure

### Objective
Initialize project repository with the exact directory structure needed for the voice orchestrator.

### Commands to Execute
```bash
# Create directories
mkdir -p voice-orchestrator/{orchestrator/src,ui/{app,components,lib,public},data/{redis,chromadb,models},scripts}

# Create base files
cd voice-orchestrator
touch docker-compose.yml docker-compose.cpu.yml docker-compose.memory.yml
touch .env.example .gitignore README.md
touch orchestrator/{Dockerfile,requirements.txt}
touch ui/{Dockerfile,package.json,tsconfig.json,tailwind.config.ts}
touch scripts/{health-check.sh,quick-start.sh}

# Create .gitignore
cat > .gitignore << 'EOF'
# Environment
.env
.env.local

# Data directories
data/redis/*
data/chromadb/*
data/models/*
!data/*/.gitkeep

# Dependencies
node_modules/
__pycache__/
*.pyc
.pytest_cache/

# Build artifacts
.next/
dist/
build/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF

# Create .gitkeep files
touch data/{redis,chromadb,models}/.gitkeep
```

### Repository Structure Verification
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
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── public/
├── data/
│   ├── redis/.gitkeep
│   ├── chromadb/.gitkeep
│   └── models/.gitkeep
└── scripts/
    ├── health-check.sh
    └── quick-start.sh
```

---

## DEVOPS-002: Docker Compose Configuration Files

### docker-compose.yml
```yaml
version: '3.8'

services:
  orchestrator:
    build:
      context: ./orchestrator
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - WHISPER_URL=${WHISPER_URL:-http://whisper-live:9090}
      - OLLAMA_URL=${OLLAMA_URL:-http://host.docker.internal:11434}
      - TTS_URL=${TTS_URL:-http://kokoro:8880/v1}
      - MEMORY_ENABLED=${MEMORY_ENABLED:-false}
      - AMEM_URL=${AMEM_URL:-http://a-mem:8001}
      - REDIS_URL=${REDIS_URL:-redis://redis:6379}
      - LLM_MODEL=${LLM_MODEL:-gemma2:2b}
      - TTS_VOICE=${TTS_VOICE:-af_bella}
    depends_on:
      - whisper-live
      - kokoro
    restart: unless-stopped

  whisper-live:
    image: collabora/whisperlive:latest-gpu
    ports:
      - "9090:9090"
    volumes:
      - ./models/whisper:/models
    environment:
      - MODEL_SIZE=${STT_MODEL:-tiny}
      - VAD_ENABLED=true
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  kokoro:
    image: ghcr.io/remsky/kokoro-fastapi-gpu:v0.2.1
    ports:
      - "8880:8880"
    environment:
      - DEFAULT_VOICE=${TTS_VOICE:-af_bella}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  voice-ui:
    build:
      context: ./ui
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_WS_URL=${WS_URL:-ws://localhost:8000/ws}
    depends_on:
      - orchestrator
    restart: unless-stopped

networks:
  default:
    name: voice-net
```

### docker-compose.cpu.yml
```yaml
version: '3.8'

services:
  whisper-live:
    image: collabora/whisperlive:latest
    deploy:
      resources:
        reservations:
          devices: []

  kokoro:
    image: ghcr.io/remsky/kokoro-fastapi-cpu:latest
    deploy:
      resources:
        reservations:
          devices: []
```

### docker-compose.memory.yml
```yaml
version: '3.8'

services:
  orchestrator:
    environment:
      - MEMORY_ENABLED=true

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
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
    restart: unless-stopped

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
    restart: unless-stopped
```

### .env.example
```bash
# Core Services
WHISPER_URL=http://whisper-live:9090
OLLAMA_URL=http://host.docker.internal:11434
TTS_URL=http://kokoro:8880/v1
WS_URL=ws://localhost:8000/ws

# Memory Components (Optional)
MEMORY_ENABLED=false
AMEM_URL=http://a-mem:8001
REDIS_URL=redis://redis:6379

# Model Configuration
STT_MODEL=tiny
LLM_MODEL=gemma2:2b
TTS_VOICE=af_bella
EMBEDDING_MODEL_NAME=nomic-embed-text

# Performance
CHUNK_SIZE_MS=320
TARGET_LATENCY_MS=500

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
```

---

## DEVOPS-003: CI/CD Pipeline

### .github/workflows/build.yml
```yaml
name: Build and Publish

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    strategy:
      matrix:
        component: [orchestrator, ui]
        platform: [linux/amd64, linux/arm64]
    
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-${{ matrix.component }}
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: ./${{ matrix.component }}
          platforms: ${{ matrix.platform }}
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
  
  test:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Start services
        run: |
          docker-compose up -d orchestrator
          sleep 10
      
      - name: Health check
        run: |
          curl -f http://localhost:8000/health || exit 1
      
      - name: Cleanup
        if: always()
        run: docker-compose down
```

---

## DEVOPS-004: Quick Start Script

### scripts/quick-start.sh
```bash
#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🎙️  Voice Orchestrator Quick Start${NC}"
echo

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is required but not installed.${NC}"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is required but not installed.${NC}"
    exit 1
fi

# Check if Ollama is running
echo "Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo -e "${YELLOW}⚠️  Ollama not detected on port 11434.${NC}"
    echo "Please ensure Ollama is running: https://ollama.ai"
    echo "You can start it with: ollama serve"
    exit 1
fi

# Check for required models
echo "Checking required models..."
if ! curl -s http://localhost:11434/api/tags | grep -q "gemma2:2b"; then
    echo -e "${YELLOW}📥 Pulling gemma2:2b model...${NC}"
    ollama pull gemma2:2b
fi

if ! curl -s http://localhost:11434/api/tags | grep -q "nomic-embed-text"; then
    echo -e "${YELLOW}📥 Pulling nomic-embed-text model...${NC}"
    ollama pull nomic-embed-text
fi

# Create .env if not exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo -e "${GREEN}✅ Created .env file${NC}"
fi

# Detect GPU support
GPU_COMPOSE=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
else
    echo -e "${YELLOW}⚠️  No NVIDIA GPU detected, using CPU mode${NC}"
    GPU_COMPOSE="-f docker-compose.cpu.yml"
fi

# Ask about memory components
echo
read -p "Enable memory components (Redis, ChromaDB, A-MEM)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    GPU_COMPOSE="$GPU_COMPOSE -f docker-compose.memory.yml"
    echo -e "${GREEN}✅ Memory components enabled${NC}"
fi

# Start services
echo
echo -e "${GREEN}🚀 Starting services...${NC}"
docker-compose $GPU_COMPOSE pull
docker-compose $GPU_COMPOSE up -d

# Wait for services
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 10

# Health check
echo "Performing health check..."
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:8000/health | grep -q "ok"; then
        echo -e "${GREEN}✅ Orchestrator is healthy${NC}"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    sleep 1
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo -e "${RED}❌ Health check failed${NC}"
    echo "Showing logs:"
    docker-compose logs orchestrator
    exit 1
fi

# Success message
echo
echo -e "${GREEN}✅ Voice assistant is ready!${NC}"
echo
echo "🌐 Open http://localhost:3000 in your browser"
echo "📊 View logs: docker-compose logs -f"
echo "🛑 Stop: docker-compose down"
echo
echo "Enjoy your voice assistant! 🎉"
```

### scripts/health-check.sh
```bash
#!/bin/bash

# Simple health check script
SERVICES=("orchestrator:8000" "whisper-live:9090" "kokoro:8880" "voice-ui:3000")

echo "Checking service health..."

for SERVICE in "${SERVICES[@]}"; do
    IFS=':' read -r NAME PORT <<< "$SERVICE"
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health | grep -q "200"; then
        echo "✅ $NAME is healthy"
    else
        echo "❌ $NAME is not responding"
    fi
done
```

### Acceptance Criteria
1. All scripts are executable
2. Repository structure matches specification
3. Docker compose files are valid YAML
4. CI/CD pipeline builds multi-arch images
5. Quick start script handles all edge cases