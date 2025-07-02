#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✅ Ollama is running${NC}"
fi

# Check for required models
echo "Checking required models..."
MODELS_MISSING=false

if ! curl -s http://localhost:11434/api/tags | grep -q "gemma3n:latest"; then
    echo -e "${YELLOW}📥 Model gemma3n:latest not found${NC}"
    MODELS_MISSING=true
fi

if ! curl -s http://localhost:11434/api/tags | grep -q "nomic-embed-text"; then
    echo -e "${YELLOW}📥 Model nomic-embed-text not found${NC}"
    MODELS_MISSING=true
fi

if [ "$MODELS_MISSING" = true ]; then
    echo -e "${BLUE}Would you like to download the required models? This may take several minutes.${NC}"
    read -p "Download models now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}📥 Pulling gemma3n:latest model...${NC}"
        ollama pull gemma3n:latest
        echo -e "${YELLOW}📥 Pulling nomic-embed-text model...${NC}"
        ollama pull nomic-embed-text
    fi
fi

# Create .env if not exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo -e "${GREEN}✅ Created .env file${NC}"
else
    echo -e "${GREEN}✅ Using existing .env file${NC}"
fi

# Detect GPU support
GPU_COMPOSE=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
    GPU_COMPOSE=""
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

# Create data directories
echo "Creating data directories..."
mkdir -p data/{redis,chromadb,models/whisper}

# Start services
echo
echo -e "${GREEN}🚀 Starting services...${NC}"
echo "Command: docker-compose $GPU_COMPOSE up -d"

docker-compose $GPU_COMPOSE pull
docker-compose $GPU_COMPOSE up -d

# Wait for services
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 15

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
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo -e "${RED}❌ Health check failed after ${MAX_ATTEMPTS} attempts${NC}"
    echo "Showing logs:"
    docker-compose logs orchestrator
    echo
    echo "Try running: docker-compose logs to see what went wrong"
    exit 1
fi

# Check other services
echo "Checking other services..."

# Check UI
if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 | grep -q "200"; then
    echo -e "${GREEN}✅ Voice UI is healthy${NC}"
else
    echo -e "${YELLOW}⚠️  Voice UI might still be starting${NC}"
fi

# Success message
echo
echo -e "${GREEN}✅ Voice Orchestrator is ready!${NC}"
echo
echo -e "${BLUE}📱 Open your web browser and go to:${NC}"
echo -e "   ${GREEN}http://localhost:3000${NC}"
echo
echo -e "${BLUE}🔧 Management commands:${NC}"
echo "   📊 View logs: docker-compose logs -f"
echo "   🛑 Stop: docker-compose down"
echo "   🔄 Restart: docker-compose restart"
echo "   📈 Monitor: docker-compose ps"
echo
echo -e "${BLUE}🎤 Usage:${NC}"
echo "   1. Allow microphone access when prompted"
echo "   2. Wait for 'Ready' status"
echo "   3. Hold the button and speak"
echo "   4. Release when finished"
echo
echo "🎉 Enjoy your ultra-low-latency voice assistant!"