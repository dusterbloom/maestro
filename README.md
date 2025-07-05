# Voice Orchestrator 🎙️

Ultra-low-latency voice assistant achieving **<500ms end-to-end latency** by orchestrating best-in-class Docker containers.

![Architecture](https://img.shields.io/badge/Architecture-Microservices-blue) ![Latency](https://img.shields.io/badge/Latency-<500ms-green) ![Docker](https://img.shields.io/badge/Deployment-Docker-blue) ![GPU](https://img.shields.io/badge/GPU-Accelerated-orange)

## 🚀 Quick Start

```bash
# 1. Ensure Ollama is running with required models
ollama pull gemma3n:latest
ollama pull nomic-embed-text

# 2. One-command setup
./scripts/quick-start.sh

# 3. Open browser
open http://localhost:3000
```

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│   Browser   │◄──►│ Orchestrator │◄──►│   Ollama    │◄──►│   Memory    │
│  Next.js    │    │   FastAPI    │    │  (External) │    │   A-MEM     │
│ Push-to-Talk│    │  WebSocket   │    │             │    │ Redis+Chroma│
└─────────────┘    └──────┬───────┘    └─────────────┘    └─────────────┘
                          │
                ┌─────────┼─────────┐
                ▼                   ▼
        ┌─────────────┐    ┌─────────────┐
        │ WhisperLive │    │   Kokoro    │
        │     STT     │    │     TTS     │
        │  (Docker)   │    │  (Docker)   │
        └─────────────┘    └─────────────┘
```

## ⚡ Performance

**Target Latency Budget:**
- Audio capture: 16ms
- STT (WhisperLive): 120ms  
- LLM (Ollama): 180ms
- TTS (Kokoro): 80ms
- Network overhead: 12ms
- **Total: 408ms** (< 500ms target ✅)

## 🎤 Voice Interruption (Professional Barge-in)

**Real-time voice interruption system** inspired by professional voice applications:

- **⚡ Immediate Response**: < 100ms interruption latency when user speaks
- **🚫 Cascading Prevention**: Blocks new TTS while audio is playing  
- **🎯 Smart Detection**: Voice activity detection during TTS playback AND queue gaps
- **🔄 Seamless Transition**: Instant switch from TTS to recording mode
- **🧠 Professional Grade**: Based on WhisperLive VAD with 0.1 threshold

**How it works:**
1. User speaks while TTS is playing → Voice activity detected (< 100ms)
2. All audio instantly stopped + queue cleared → Recording starts immediately  
3. Transcription processed → New response generated → Seamless conversation

## 🛠️ Components

| Service | Purpose | Technology | Port |
|---------|---------|------------|------|
| **Orchestrator** | Audio pipeline coordination | FastAPI + WebSocket | 8000 |
| **Voice UI** | Push-to-talk interface | Next.js 14 PWA | 3000 |
| **WhisperLive** | Speech-to-text | Docker + CUDA | 9090 |
| **Kokoro** | Text-to-speech | Docker + CUDA | 8880 |
| **Ollama** | Language model | External service | 11434 |
| **A-MEM** | Conversation memory | Docker (optional) | 8001 |
| **Redis** | Session cache | Docker (optional) | 6379 |
| **ChromaDB** | Vector storage | Docker (optional) | 8002 |

## 📦 Deployment Options

### GPU Accelerated (Recommended)
```bash
docker-compose up -d
```

### CPU Only
```bash
docker-compose -f docker-compose.cpu.yml up -d
```

### With Memory Components
```bash
docker-compose -f docker-compose.yml -f docker-compose.memory.yml up -d
```

## 🔧 Configuration

Copy `.env.example` to `.env` and customize:

```bash
# Core Services
WHISPER_URL=http://whisper-live:9090
OLLAMA_URL=http://host.docker.internal:11434
TTS_URL=http://kokoro:8880/v1

# Memory (Optional)
MEMORY_ENABLED=false

# Models
STT_MODEL=tiny
LLM_MODEL=gemma3n:latest
TTS_VOICE=af_bella

# Performance
TARGET_LATENCY_MS=500
```

## 🧪 Testing

### Health Check
```bash
./scripts/health-check.sh
```

### Latency Benchmarking
```bash
python scripts/latency-test.py
```

### Example Output
```
📊 LATENCY TEST RESULTS
========================
📈 Tests: 20/20 successful (100.0%)
⏱️  Average latency: 387.2ms
📊 Median latency: 385.1ms
🎯 95th percentile: 423.7ms
✅ TARGET MET! (P95: 423.7ms)
```

## 📱 Usage

1. **Allow microphone access** when prompted
2. **Wait for "Ready" status** indicator
3. **Hold the blue button** and speak clearly
4. **Release when finished** speaking
5. **Listen for AI response** through speakers

## 🔍 Monitoring

### View Logs
```bash
docker-compose logs -f [service-name]
```

### Service Status
```bash
docker-compose ps
```

### Resource Usage
```bash
docker stats
```

## 🏃‍♂️ Development

### Prerequisites
- Docker with GPU support (NVIDIA runtime)
- Ollama running locally with required models
- Node.js 18+ (for UI development)

### Local Development
```bash
# Backend development
cd orchestrator
pip install -r requirements.txt
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Frontend development
cd ui
npm install
npm run dev
```

## 🔧 Troubleshooting

### Common Issues

**"Cannot connect to Ollama"**
```bash
# Ensure Ollama is running
ollama serve

# Check models are installed
ollama list
```

**"Microphone not working"**
- Grant microphone permissions in browser
- Use HTTPS in production (required for audio)
- Check browser compatibility

**"Services won't start"**
```bash
# Check Docker status
docker-compose ps

# View service logs
docker-compose logs [service-name]

# Restart everything
docker-compose down && docker-compose up -d
```

**"High latency"**
- Ensure GPU drivers are installed
- Check if using CPU fallback mode
- Monitor system resources with `docker stats`

## 📊 Performance Optimization

### GPU Acceleration
- Ensure NVIDIA drivers and runtime are installed
- Check GPU utilization: `nvidia-smi`
- Use GPU-optimized service images

### Memory Usage
- WhisperLive: ~2GB VRAM
- Kokoro: ~1GB VRAM  
- Ollama: Varies by model size

### Network Optimization
- Deploy on same host for minimal latency
- Use SSD storage for model loading
- Consider RAM disk for temporary files

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/awesome-feature`
3. **Commit changes**: `git commit -m 'Add awesome feature'`
4. **Push to branch**: `git push origin feature/awesome-feature`
5. **Open Pull Request**

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[WhisperLive](https://github.com/collabora/WhisperLive)** - Real-time speech transcription
- **[Ollama](https://github.com/ollama/ollama)** - Local language model serving
- **[Kokoro](https://github.com/remsky/Kokoro-FastAPI)** - High-quality text-to-speech
- **[A-MEM](https://github.com/AGI-Research/A-MEM)** - Agentic memory system

---

Built with ❤️ for ultra-low-latency voice AI experiences