# Docker Compose Configuration Files

This document contains all Docker Compose configurations for the Voice Orchestrator project.

## Base Configuration: docker-compose.yml

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
      - LLM_MODEL=${LLM_MODEL:-gemma3n:latest}
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

## CPU-Only Override: docker-compose.cpu.yml

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

## Memory Components: docker-compose.memory.yml

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

## Monitoring Stack: docker-compose.monitoring.yml

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
```

## Wyoming Protocol Stack: docker-compose.wyoming.yml

```yaml
version: '3.8'

services:
  wyoming-whisper:
    image: rhasspy/wyoming-whisper:latest
    ports:
      - "10300:10300"
    volumes:
      - ./whisper-data:/data
    command: --model ${STT_MODEL:-tiny} --language en --device cuda
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  wyoming-piper:
    image: rhasspy/wyoming-piper:latest
    ports:
      - "10200:10200"
    volumes:
      - ./piper-data:/data
    command: --voice en_US-lessac-medium
    restart: unless-stopped
```

## Production Configuration: docker-compose.prod.yml

```yaml
version: '3.8'

services:
  orchestrator:
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  whisper-live:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  kokoro:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  voice-ui:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 512M

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - orchestrator
      - voice-ui
    restart: unless-stopped
```

## Usage Examples

### Development (GPU)
```bash
docker-compose up
```

### Development (CPU)
```bash
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up
```

### With Memory Components
```bash
docker-compose -f docker-compose.yml -f docker-compose.memory.yml up
```

### Production with Monitoring
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.monitoring.yml up -d
```

### Wyoming Protocol Stack
```bash
docker-compose -f docker-compose.wyoming.yml up
```

## Environment Variables Reference

Create a `.env` file based on `.env.example`:

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
LLM_MODEL=gemma3n:latest
TTS_VOICE=af_bella
EMBEDDING_MODEL_NAME=nomic-embed-text

# Performance
CHUNK_SIZE_MS=320
TARGET_LATENCY_MS=500

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
```