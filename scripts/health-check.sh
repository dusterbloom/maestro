#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🏥 Voice Orchestrator Health Check${NC}"
echo "======================================"

# Function to check service health
check_service() {
    local name=$1
    local url=$2
    local expected=$3
    
    echo -n "Checking $name... "
    
    if response=$(curl -s -w "%{http_code}" "$url" 2>/dev/null); then
        http_code="${response: -3}"
        if [ "$http_code" = "$expected" ]; then
            echo -e "${GREEN}✅ Healthy${NC}"
            return 0
        else
            echo -e "${RED}❌ Unhealthy (HTTP $http_code)${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Not responding${NC}"
        return 1
    fi
}

# Function to check service with JSON response
check_service_json() {
    local name=$1
    local url=$2
    local key=$3
    local expected=$4
    
    echo -n "Checking $name... "
    
    if response=$(curl -s "$url" 2>/dev/null); then
        if echo "$response" | grep -q "\"$key\".*\"$expected\""; then
            echo -e "${GREEN}✅ Healthy${NC}"
            return 0
        else
            echo -e "${RED}❌ Unhealthy${NC}"
            echo "   Response: $response"
            return 1
        fi
    else
        echo -e "${RED}❌ Not responding${NC}"
        return 1
    fi
}

echo "Core Services:"
echo "-------------"

# Check Orchestrator
check_service_json "Orchestrator" "http://localhost:8000/health" "status" "ok"
ORCHESTRATOR_OK=$?

# Check Voice UI
check_service "Voice UI" "http://localhost:3001/" "200"
UI_OK=$?

echo
echo "External Dependencies:"
echo "--------------------"

# Check Ollama
echo -n "Checking Ollama... "
if curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo -e "${GREEN}✅ Available${NC}"
    OLLAMA_OK=0
    
    # Check for required models
    echo -n "  - gemma3n:latest model... "
    if curl -s http://localhost:11434/api/tags | grep -q "gemma3n:latest"; then
        echo -e "${GREEN}✅ Available${NC}"
    else
        echo -e "${YELLOW}⚠️  Missing${NC}"
    fi
    
    echo -n "  - nomic-embed-text model... "
    if curl -s http://localhost:11434/api/tags | grep -q "nomic-embed-text"; then
        echo -e "${GREEN}✅ Available${NC}"
    else
        echo -e "${YELLOW}⚠️  Missing${NC}"
    fi
else
    echo -e "${RED}❌ Not available${NC}"
    OLLAMA_OK=1
fi

echo
echo "Voice Pipeline Services:"
echo "----------------------"

# Check WhisperLive
check_service "WhisperLive" "http://localhost:9090/health" "200"
WHISPER_OK=$?

# Check Kokoro TTS
check_service "Kokoro TTS" "http://localhost:8880/health" "200"
KOKORO_OK=$?

echo
echo "Optional Memory Services:"
echo "-----------------------"

# Check if memory is enabled
if docker-compose ps | grep -q "a-mem"; then
    check_service "A-MEM" "http://localhost:8001/health" "200"
    check_service "Redis" "http://localhost:6379/" "200"
    check_service "ChromaDB" "http://localhost:8002/api/v1/heartbeat" "200"
else
    echo -e "${YELLOW}ℹ️  Memory services not enabled${NC}"
fi

echo
echo "Docker Container Status:"
echo "----------------------"
docker-compose ps

echo
echo "Summary:"
echo "-------"

TOTAL_ISSUES=0

if [ $ORCHESTRATOR_OK -ne 0 ]; then
    echo -e "${RED}❌ Orchestrator service issues${NC}"
    ((TOTAL_ISSUES++))
fi

if [ $UI_OK -ne 0 ]; then
    echo -e "${RED}❌ Voice UI issues${NC}"
    ((TOTAL_ISSUES++))
fi

if [ $OLLAMA_OK -ne 0 ]; then
    echo -e "${RED}❌ Ollama not available${NC}"
    ((TOTAL_ISSUES++))
fi

if [ ${WHISPER_OK:-1} -ne 0 ]; then
    echo -e "${YELLOW}⚠️  WhisperLive issues (may still be starting)${NC}"
fi

if [ ${KOKORO_OK:-1} -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Kokoro TTS issues (may still be starting)${NC}"
fi

if [ $TOTAL_ISSUES -eq 0 ]; then
    echo -e "${GREEN}✅ All critical services are healthy!${NC}"
    echo -e "${BLUE}🌐 Access the Voice UI at: http://localhost:3000${NC}"
    exit 0
else
    echo -e "${RED}❌ Found $TOTAL_ISSUES critical issues${NC}"
    echo
    echo "Troubleshooting steps:"
    echo "1. Check logs: docker-compose logs [service-name]"
    echo "2. Restart services: docker-compose restart"
    echo "3. Full restart: docker-compose down && docker-compose up -d"
    exit 1
fi